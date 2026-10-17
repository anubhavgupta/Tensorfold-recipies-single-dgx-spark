// Dory's elementwise kernels: the embedding, RoPE with nGPT's q/k normalisation and the KV store, the gated attention
// output, the scaled SwiGLU, the slerp residual, the recurrent loop's input, the logit scale, and the top candidates of
// each row (the last from libbolt's bolt_kernels.cu). justnorm(x) = x * rsqrt(sum(x^2) + 1e-6), in fp32.

#include "dory_common.h"

#include <cuda_bf16.h>

namespace {

constexpr float EPS = 1e-6f;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Sum over a block of 256 threads; every thread gets the total. `sh` holds 8 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) t += sh[i];
    return t;
}

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t f2bf(float v) {
    __nv_bfloat16 b = __float2bfloat16_rn(v);
    return *reinterpret_cast<uint16_t*>(&b);
}

__global__ void __launch_bounds__(256) embed_k(const uint16_t* __restrict__ table, const Ctl* __restrict__ ctl, float* __restrict__ x) {
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    const uint16_t* row = table + (size_t)ctl->tokens[t] * HID;
    for (int i = threadIdx.x; i < HID; i += 256) x[(size_t)t * HID + i] = bf2f(row[i]);
}

// One row a block; warp w takes heads w, w + 8, ... of the 16 query and 4 key heads.
__global__ void __launch_bounds__(256) qk_prep_k(const Ctl* __restrict__ ctl, float* __restrict__ q, const float* __restrict__ k,
                                                 const float* __restrict__ v, const float* __restrict__ sqk2,
                                                 const float* __restrict__ inv_freq, KvView kv) {
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = ctl->pos[t];
    const long long seq = ctl->seg_seq[ctl->rseg[t]];
    const long long slot = kv.ring ? (pos & (kv.ring - 1)) : pos;
    const size_t crow = (size_t)(seq * kv.stride + slot) * NKV * HD;
    const float fpos = (float)pos;
    float cs[4], sn[4];
#pragma unroll
    for (int j = 0; j < 4; j++) sincosf(fpos * inv_freq[lane + 32 * j], &sn[j], &cs[j]);
    for (int h = warp; h < NH + NKV; h += 8) {
        const bool isq = h < NH;
        const float* src = isq ? q + (size_t)t * QDIM + h * HD : k + (size_t)t * KVDIM + (h - NH) * HD;
        float r1[4], r2[4], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = lane + 32 * j;
            const float x1 = src[i], x2 = src[i + HD / 2];
            r1[j] = x1 * cs[j] - x2 * sn[j];
            r2[j] = x2 * cs[j] + x1 * sn[j];
            ss += r1[j] * r1[j] + r2[j] * r2[j];
        }
        const float inv = rsqrtf(warp_sum(ss) + EPS);
        if (isq) {
            float* dst = q + (size_t)t * QDIM + h * HD;
            const float* s2 = sqk2 + h * HD;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int i = lane + 32 * j;
                dst[i] = s2[i] * (r1[j] * inv);
                dst[i + HD / 2] = s2[i + HD / 2] * (r2[j] * inv);
            }
        } else {
            uint16_t* dst = kv.k + crow + (h - NH) * HD;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int i = lane + 32 * j;
                dst[i] = f2bf(r1[j] * inv);
                dst[i + HD / 2] = f2bf(r2[j] * inv);
            }
        }
    }
    for (int i = threadIdx.x; i < KVDIM; i += 256) kv.v[crow + i] = f2bf(v[(size_t)t * KVDIM + i]);
}

__global__ void __launch_bounds__(256) gate_norm_k(const Ctl* __restrict__ ctl, const float* __restrict__ o, const float* __restrict__ g,
                                                   const float* __restrict__ sg2, float* __restrict__ out) {
    __shared__ float sh[8];
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    constexpr int PER = QDIM / 256;
    float v[PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = threadIdx.x + 256 * j;
        const float gv = g[(size_t)t * QDIM + i] * sg2[i];
        v[j] = o[(size_t)t * QDIM + i] / (1.f + __expf(-gv));
        ss += v[j] * v[j];
    }
    const float inv = rsqrtf(block_sum(ss, sh) + EPS);
#pragma unroll
    for (int j = 0; j < PER; j++) out[(size_t)t * QDIM + threadIdx.x + 256 * j] = v[j] * inv;
}

__global__ void __launch_bounds__(256) mlp_act_k(const Ctl* __restrict__ ctl, const float* __restrict__ g, const float* __restrict__ u,
                                                 const float* __restrict__ sug, const float* __restrict__ suu, float* __restrict__ out) {
    __shared__ float sh[8];
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    constexpr int PER = FFN / 256;
    float v[PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = threadIdx.x + 256 * j;
        const float a = g[(size_t)t * FFN + i] * sug[i];
        const float b = u[(size_t)t * FFN + i] * suu[i];
        v[j] = a / (1.f + __expf(-a)) * b;
        ss += v[j] * v[j];
    }
    const float inv = rsqrtf(block_sum(ss, sh) + EPS);
#pragma unroll
    for (int j = 0; j < PER; j++) out[(size_t)t * FFN + threadIdx.x + 256 * j] = v[j] * inv;
}

__global__ void __launch_bounds__(256) slerp_k(const Ctl* __restrict__ ctl, float* __restrict__ x, const float* __restrict__ r,
                                               const float* __restrict__ alpha, int post) {
    __shared__ float sh[8];
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    constexpr int PER = HID / 256;
    float a[PER], b[PER], sa = 0.f, sb = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = threadIdx.x + 256 * j;
        a[j] = x[(size_t)t * HID + i];
        b[j] = r[(size_t)t * HID + i];
        sa += a[j] * a[j];
        sb += b[j] * b[j];
    }
    const float ia = rsqrtf(block_sum(sa, sh) + EPS);
    const float ib = rsqrtf(block_sum(sb, sh) + EPS);
    float sy = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const float an = a[j] * ia, bn = b[j] * ib;
        a[j] = an + alpha[threadIdx.x + 256 * j] * (bn - an);
        sy += a[j] * a[j];
    }
    float iy = rsqrtf(block_sum(sy, sh) + EPS);
    if (post) {
        float s2 = 0.f;
#pragma unroll
        for (int j = 0; j < PER; j++) { a[j] *= iy; s2 += a[j] * a[j]; }
        iy = rsqrtf(block_sum(s2, sh) + EPS);
    }
#pragma unroll
    for (int j = 0; j < PER; j++) x[(size_t)t * HID + threadIdx.x + 256 * j] = a[j] * iy;
}

__global__ void __launch_bounds__(256) loop_in_k(const Ctl* __restrict__ ctl, float* __restrict__ x, const float* __restrict__ inp) {
    __shared__ float sh[8];
    const int t = blockIdx.x;
    if (t >= ctl->nrows) return;
    constexpr int PER = HID / 256;
    float a[PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const size_t i = (size_t)t * HID + threadIdx.x + 256 * j;
        a[j] = x[i] + inp[i];
        ss += a[j] * a[j];
    }
    const float inv = rsqrtf(block_sum(ss, sh) + EPS);
#pragma unroll
    for (int j = 0; j < PER; j++) x[(size_t)t * HID + threadIdx.x + 256 * j] = a[j] * inv;
}

__global__ void __launch_bounds__(256) logit_scale_k(float* __restrict__ logits, const float* __restrict__ scale) {
    const size_t r = blockIdx.y;
    const int i = blockIdx.x * 1024 + threadIdx.x;
#pragma unroll
    for (int j = 0; j < 4; j++) logits[r * VOCAB + i + 256 * j] *= scale[i + 256 * j];
}

}  // namespace

void k_embed(cudaStream_t s, const uint16_t* table, const Ctl* ctl, float* x, int T) { embed_k<<<T, 256, 0, s>>>(table, ctl, x); }
void k_qk_prep(cudaStream_t s, const Ctl* ctl, float* q, const float* k, const float* v, const float* sqk2,
               const float* inv_freq, const KvView& kv, int T) {
    qk_prep_k<<<T, 256, 0, s>>>(ctl, q, k, v, sqk2, inv_freq, kv);
}
void k_gate_norm(cudaStream_t s, const Ctl* ctl, const float* o, const float* g, const float* sg2, float* out, int T) {
    gate_norm_k<<<T, 256, 0, s>>>(ctl, o, g, sg2, out);
}
void k_mlp_act(cudaStream_t s, const Ctl* ctl, const float* g, const float* u, const float* sug, const float* suu, float* out, int T) {
    mlp_act_k<<<T, 256, 0, s>>>(ctl, g, u, sug, suu, out);
}
void k_slerp(cudaStream_t s, const Ctl* ctl, float* x, const float* r, const float* alpha, int post, int T) {
    slerp_k<<<T, 256, 0, s>>>(ctl, x, r, alpha, post);
}
void k_loop_in(cudaStream_t s, const Ctl* ctl, float* x, const float* inp, int T) { loop_in_k<<<T, 256, 0, s>>>(ctl, x, inp); }
void k_logit_scale(cudaStream_t s, float* logits, const float* scale, int rows) {
    logit_scale_k<<<dim3(VOCAB / 1024, rows), 256, 0, s>>>(logits, scale);
}

// ---------------------------------------------------------------------------------------------------------
// Row-wise top candidates of the logits. Stage one: 128 blocks per row, each selects its slice's best
// `topk`; stage two merges them. Ties go to the lower id, so the order is total.
// ---------------------------------------------------------------------------------------------------------

namespace {
constexpr int CAND_BLOCKS = 128;

static __device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void __launch_bounds__(256) cand1_k(const float* __restrict__ logits, int topk, float* __restrict__ sv, int* __restrict__ si) {
    const int t = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)t * VOCAB + (size_t)b * 1024;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = row[threadIdx.x + 256 * j];
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)t * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
}

__global__ void __launch_bounds__(256) cand2_k(int topk, const float* __restrict__ sv, const int* __restrict__ si,
                                                float* __restrict__ vals, int* __restrict__ ids) {
    const int t = blockIdx.x;
    // CAND_BLOCKS * topk candidates, 32 per thread at most
    const int total = CAND_BLOCKS * topk;
    float v[32];
    int id[32];
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int k = threadIdx.x + 256 * j;
        if (k < total) {
            int blk = k / topk, r = k % topk;
            size_t o = ((size_t)t * CAND_BLOCKS + blk) * CAND + r;
            v[j] = sv[o];
            id[j] = si[o];
        } else {
            v[j] = -3.4e38f;
            id[j] = 0x7fffffff;
        }
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bi = 0x7fffffff, bj = 0;
#pragma unroll
        for (int j = 0; j < 32; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; bj = j; }
        float lv = bv; int li = bi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            vals[(size_t)t * CAND + r] = fv;
            ids[(size_t)t * CAND + r] = fi;
        }
        // retire the winner from the thread that holds it
        if (li == fi && lv == fv) {
#pragma unroll
            for (int j = 0; j < 32; j++) if (j == bj) { v[j] = -3.4e38f; id[j] = 0x7fffffff; }
        }
    }
}

// The same for chosen rows among only the tokens a structured-output mask allows (bit t of a row's `words` u32s): stage one
// of row sel[i] reads its mask from masks + i * words, a disallowed token counting as -infinity.
__global__ void __launch_bounds__(256) cand1m_k(const float* __restrict__ logits, const int* __restrict__ sel, const uint32_t* __restrict__ masks,
                                                 int words, int topk, float* __restrict__ sv, int* __restrict__ si) {
    const int i = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)sel[i] * VOCAB + (size_t)b * 1024;
    const uint32_t* mk = masks + (size_t)i * words;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int tok = b * 1024 + threadIdx.x + 256 * j;
        const bool ok = (tok >> 5) < words && ((mk[tok >> 5] >> (tok & 31)) & 1u);
        v[j] = ok ? row[threadIdx.x + 256 * j] : -3.4e38f;
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)i * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
}

}  // namespace

void k_candidates_masked(cudaStream_t s, const float* logits, const int* sel, const uint32_t* masks, int words, int n, int topk,
                         float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    cand1m_k<<<dim3(CAND_BLOCKS, n), 256, 0, s>>>(logits, sel, masks, words, topk, sv, si);
    cand2_k<<<n, 256, 0, s>>>(topk, sv, si, vals, ids);
}

void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    cand1_k<<<dim3(CAND_BLOCKS, T), 256, 0, s>>>(logits, topk, sv, si);
    cand2_k<<<T, 256, 0, s>>>(topk, sv, si, vals, ids);
}

