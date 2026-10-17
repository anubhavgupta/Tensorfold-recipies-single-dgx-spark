// Attention on tensor cores for Dory: 16 query heads over 4 KV heads, head dimension 256, full or sliding-window layers.
//
// A warp takes four query rows of one segment with the four query heads of one KV head as the 16 rows of an m16 tile
// (tile row 4 * qrow + head). S = Q K^T and O += P V are m16n8k16 bf16 mmas, the softmax is online in registers and the
// S accumulators become the A fragments of P V directly. K and V tiles of 32 keys are staged row-major ([key][dim], as
// stored in the cache) and read with ldmatrix (.trans for V). Keys are split across CTAs; a combine kernel merges the
// partials in split order.
//
// Verify windows (multi): one CTA per segment, its keys cut into chunks at absolute multiples of a size that depends on
// the sequence's own length (win_chunk, SWA_CHUNK), so a row's bits do not depend on the other streams of the block or on
// the width of its window. Masked keys contribute exact zeros.

#include "dory_common.h"

#include <cuda_bf16.h>

namespace {

constexpr int KS = HD + 8;                 // staged row stride in halves: 528 bytes, conflict-free ldmatrix
constexpr int PART_ROW = HD + 2;           // per tile row of a partial: m, l, o[HD]
constexpr int PART = 64 * PART_ROW;        // per CTA: up to four warps of 16 tile rows

__device__ __forceinline__ void mma_bf16f(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_bf(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int NW>
__global__ void __launch_bounds__(NW * 32) attn_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q, KvView kv, int window,
                                                      float* __restrict__ part, int splits, int multi) {
    const int sp = blockIdx.x, kvh = blockIdx.y, blk = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    int sgi, row0, nrow;
    if (multi) {
        if (blk >= ctl->nseg) return;
        sgi = blk;
        row0 = ctl->seg_row0[sgi];
        nrow = ctl->seg_row0[sgi + 1] - row0;
    } else {
        sgi = 0;
        row0 = blk * 4 * NW;
        nrow = min(ctl->nrows - row0, 4 * NW);
        if (nrow <= 0) return;
    }
    const long long seq = ctl->seg_seq[sgi];
    const int base0 = ctl->seg_base[sgi];
    const int minpos = ctl->pos[row0], maxpos = ctl->pos[row0 + nrow - 1];
    const int hi = maxpos + 1;
    const int lo = window ? max(0, minpos - window + 1) : 0;
    int p0, p1;
    if (multi) {
        if (window) {
            p0 = (lo / SWA_CHUNK + sp) * SWA_CHUNK;
            p1 = min(hi, p0 + SWA_CHUNK);
        } else {
            const int chunk = win_chunk(base0);
            p0 = sp * chunk;
            p1 = min(hi, p0 + chunk);
        }
    } else {
        const int lo0 = lo & ~31;
        const int chunk = ((hi - lo0 + splits - 1) / splits + 31) & ~31;
        p0 = lo0 + sp * chunk;
        p1 = min(hi, p0 + chunk);
    }
    const int kstart = max(p0, lo & ~31);

    __shared__ __align__(16) uint16_t Ks[32 * KS];
    __shared__ __align__(16) uint16_t Vs[32 * KS];

    // this lane's two tile rows: gi -> (qrow gi/4, head gi%4), gi + 8 -> (qrow 2 + gi/4, head gi%4)
    const int qa_row = warp * 4 + (gi >> 2), qb_row = qa_row + 2, hg = gi & 3;
    const bool va = qa_row < nrow, vb = qb_row < nrow;
    const int pa = va ? ctl->pos[row0 + qa_row] : -1, pb = vb ? ctl->pos[row0 + qb_row] : -1;
    const bool live = warp * 4 < nrow;

    uint32_t qf[16][4];
    {
        const float scale = rsqrtf((float)HD);
        const float* ra = q + (size_t)(row0 + (va ? qa_row : 0)) * QDIM + (kvh * 4 + hg) * HD + 2 * ti;
        const float* rb = q + (size_t)(row0 + (vb ? qb_row : 0)) * QDIM + (kvh * 4 + hg) * HD + 2 * ti;
        const float sa = va ? scale : 0.f, sb = vb ? scale : 0.f;
#pragma unroll
        for (int ks = 0; ks < 16; ks++) {
            float2 a = *reinterpret_cast<const float2*>(ra + ks * 16), c = *reinterpret_cast<const float2*>(ra + ks * 16 + 8);
            float2 b = *reinterpret_cast<const float2*>(rb + ks * 16), d = *reinterpret_cast<const float2*>(rb + ks * 16 + 8);
            qf[ks][0] = pack_bf(a.x * sa, a.y * sa);
            qf[ks][1] = pack_bf(b.x * sb, b.y * sb);
            qf[ks][2] = pack_bf(c.x * sa, c.y * sa);
            qf[ks][3] = pack_bf(d.x * sb, d.y * sb);
        }
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    const int ring = kv.ring;
    for (int base = kstart; base < p1; base += 32) {
        __syncthreads();
        for (int i = tid; i < 32 * 32; i += NW * 32) {
            const int key = i >> 5, c8 = (i & 31) * 8;
            const int p = base + key;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (p < p1) {
                const long long slot = ring ? (p & (ring - 1)) : p;
                const size_t off = ((size_t)(seq * kv.stride + slot) * NKV + kvh) * HD + c8;
                kk = __ldg(reinterpret_cast<const uint4*>(kv.k + off));
                vv = __ldg(reinterpret_cast<const uint4*>(kv.v + off));
            }
            *reinterpret_cast<uint4*>(&Ks[key * KS + c8]) = kk;
            *reinterpret_cast<uint4*>(&Vs[key * KS + c8]) = vv;
        }
        __syncthreads();
        if (!live) continue;
        // S = Q K^T for 32 keys: four n8 tiles; ldmatrix x4 gives two key tiles' b0, b1 for one k16 step
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
        const int lm = lane >> 3, lr = lane & 7;
#pragma unroll
        for (int ks = 0; ks < 16; ks++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                // matrices: (keys np*16 + 0..7, dims ks*16 + 0..7), (same keys, dims +8), (keys +8, dims), (keys +8, dims +8)
                const uint16_t* p = &Ks[(np * 16 + (lm >> 1) * 8 + lr) * KS + ks * 16 + (lm & 1) * 8];
                uint32_t b0, b1, b2, b3;
                ldsm4(b0, b1, b2, b3, p);
                mma_bf16f(s[2 * np], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
                mma_bf16f(s[2 * np + 1], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b2, b3);
            }
        }
        float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int key = base + nt * 8 + 2 * ti + (c & 1);
                const int P = c < 2 ? pa : pb;
                const bool ok = key < p1 && key <= P && (window == 0 || key > P - window);
                if (!ok) s[nt][c] = -1e30f;
                if (c < 2) tm0 = fmaxf(tm0, s[nt][c]); else tm1 = fmaxf(tm1, s[nt][c]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float sc0 = __expf(m0 - n0), sc1 = __expf(m1 - n1);
        m0 = n0; m1 = n1;
        l0 *= sc0; l1 *= sc1;
#pragma unroll
        for (int i = 0; i < 32; i++) { o[i][0] *= sc0; o[i][1] *= sc0; o[i][2] *= sc1; o[i][3] *= sc1; }
        uint32_t pf[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const float p00 = s[nt][0] > -1e29f ? __expf(s[nt][0] - n0) : 0.f, p01 = s[nt][1] > -1e29f ? __expf(s[nt][1] - n0) : 0.f;
            const float p10 = s[nt][2] > -1e29f ? __expf(s[nt][2] - n1) : 0.f, p11 = s[nt][3] > -1e29f ? __expf(s[nt][3] - n1) : 0.f;
            l0 += p00 + p01; l1 += p10 + p11;
            const int j = nt >> 1, h2 = nt & 1;
            pf[j][h2 * 2 + 0] = pack_bf(p00, p01);
            pf[j][h2 * 2 + 1] = pack_bf(p10, p11);
        }
        // O += P V: 32 n8 tiles of the head dimension; ldmatrix x4.trans gives b0, b1 of two dim tiles
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int d = 0; d < 32; d += 2) {
                const uint16_t* p = &Vs[(j * 16 + (lm & 1) * 8 + lr) * KS + (d + (lm >> 1)) * 8];
                uint32_t b0, b1, b2, b3;
                ldsm4t(b0, b1, b2, b3, p);
                mma_bf16f(o[d], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                mma_bf16f(o[d + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
            }
    }
    if (!live) return;
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float* out = part + ((size_t)(blk * NKV + kvh) * splits + sp) * PART + (size_t)warp * 16 * PART_ROW;
    float* ra = out + (size_t)gi * PART_ROW;
    float* rb = out + (size_t)(gi + 8) * PART_ROW;
    if (ti == 0) { ra[0] = m0; ra[1] = l0; rb[0] = m1; rb[1] = l1; }
#pragma unroll
    for (int d = 0; d < 32; d++) {
        ra[2 + d * 8 + 2 * ti] = o[d][0]; ra[2 + d * 8 + 2 * ti + 1] = o[d][1];
        rb[2 + d * 8 + 2 * ti] = o[d][2]; rb[2 + d * 8 + 2 * ti + 1] = o[d][3];
    }
}

// out[t][h * HD + d] from the partials of row t, merged in split order.
__global__ void __launch_bounds__(HD) attn_comb_k(const Ctl* __restrict__ ctl, const float* __restrict__ part, float* __restrict__ out,
                                                  int splits, int multi, int rows_per_blk) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (t >= ctl->nrows) { out[(size_t)t * QDIM + h * HD + d] = 0.f; return; }
    int blk, local;
    if (multi) {
        blk = ctl->rseg[t];
        local = t - ctl->seg_row0[blk];
    } else {
        blk = t / rows_per_blk;
        local = t % rows_per_blk;
    }
    const int kvh = h >> 2, hg = h & 3;
    const int rr = (local & 3) * 4 + hg;   // tile row 4 * qrow + head within the warp
    const float* b = part + ((size_t)(blk * NKV + kvh) * splits) * PART + (size_t)((local >> 2) * 16 + rr) * PART_ROW;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, b[(size_t)s * PART]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = b + (size_t)s * PART;
        const float w = __expf(p[0] - M);
        L += w * p[1];
        O += w * p[2 + d];
    }
    out[(size_t)t * QDIM + h * HD + d] = O / L;
}

}  // namespace

// CTAs per KV head and query block for a prompt chunk: enough of them to fill the GPU when the chunk is short and the
// context long.
int attn_prompt_splits(int T, int ctxlen) {
    const int blocks = (T + 15) / 16 * NKV;
    int s = 1;
    while (blocks * s < 96 && s < 32 && (ctxlen / (s * 2)) >= 256) s *= 2;
    return s;
}

size_t attn_part_floats(int T, int multi, int nseg, int splits) {
    const int nblk = multi ? nseg : (T + 15) / 16;
    return (size_t)nblk * NKV * splits * PART;
}

void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, int window, float* part, float* out,
            int T, int multi, int nseg, int splits) {
    if (multi) {
        // one CTA per segment: a window of up to VERIFY_MAX rows is two warps, a single row one
        // four warps stage the keys whatever the rows: a window of up to VERIFY_MAX rows computes on two, a single row on one
        attn_part_k<4><<<dim3(splits, NKV, nseg), 128, 0, s>>>(ctl, q, kv, window, part, splits, 1);
        attn_comb_k<<<dim3(T, NH), HD, 0, s>>>(ctl, part, out, splits, 1, 0);
    } else {
        attn_part_k<4><<<dim3(splits, NKV, (T + 15) / 16), 128, 0, s>>>(ctl, q, kv, window, part, splits, 0);
        attn_comb_k<<<dim3(T, NH), HD, 0, s>>>(ctl, part, out, splits, 0, 16);
    }
}
