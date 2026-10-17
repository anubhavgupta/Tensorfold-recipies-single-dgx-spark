// Tensor-core GEMMs for packed weights, Y[T][N] = X[T][K] * W^T with fp32 activations (from libbolt's bolt_mma.cu).
//
// A warp owns tiles of 16 weight rows and walks K in steps of 64. The weights are repacked at load time so that lane
// (g, t) = (lane / 4, lane % 4) fetches the A fragments of mma.m16n8k16 with one 16-byte load:
//   a0 = (row g,   k 2t, 2t+1)   a1 = (row g+8, k 2t, 2t+1)
//   a2 = (row g,   k 2t+8, 2t+9) a3 = (row g+8, k 2t+8, 2t+9)
// Activations are the B operand (K x 8 tokens), converted from fp32 as they are loaded (fp16 for NVFP4: Dory's GEMM
// inputs are unit vectors or embeddings, well inside its range; bf16 for the bf16 head). An e2m1 value times its e4m3
// block scale is exact in fp16, so the block scales are folded into the A fragments and the fp32 tensor scale is applied
// to the fp32 accumulators: the only rounding beyond the weights is the activation conversion.
//
// Byte layouts (rt = row tile, lane = g * 4 + t):
//   NVFP4  q  [rt][k/64][lane][4 x u32]; word j is k16 step j. Nibble r (bits 4r..4r+3) is the first element of a_r and
//          nibble 4+r (bits 16+4r..) its second. sc [rt][k/64][g][8 bytes]; byte 2j+h is the e4m3 scale of row g+8h,
//          k group 4s+j.
//   bf16   w  [rt][k/16][lane][4 x u32] = a0..a3.

#include "dory_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdlib>

namespace {

__device__ __forceinline__ void mma_f16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16f(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool BF>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    if constexpr (BF) {
        __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    } else {
        a = fminf(fmaxf(a, -65504.f), 65504.f);
        b = fminf(fmaxf(b, -65504.f), 65504.f);
        __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    }
}

__device__ __forceinline__ uint32_t hmul2u(uint32_t a, __half2 s) {
    __half2 v = __hmul2(*reinterpret_cast<__half2*>(&a), s);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float e4m3f(unsigned b) {
    __nv_fp8_e4m3 f;
    f.__x = (unsigned char)b;
    return (float)f;
}

// ------------------------------------------------------------------------------------------------------
// Policies: how one lane loads and uses a 64-wide k step of one 16-row tile.
// ------------------------------------------------------------------------------------------------------

struct PNV4 {
    using W = TNV4;
    struct Raw { uint4 q; uint2 sc; };
    static constexpr bool kBF = false;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ float scale(const W& w) { return w.s2; }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int g) {
        size_t S = w.k >> 6;
        Raw r;
        r.q = __ldg(w.q + ((size_t)tile * S + s) * 32 + lane);
        r.sc = __ldg(w.sc + ((size_t)tile * S + s) * 8 + g);
        return r;
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4]) {
        const uint32_t words[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
        const uint32_t scb[2] = {r.sc.x, r.sc.y};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t wd = words[j];
            // e2m1 magnitudes 0..6 as fp16 high bytes (0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46), the sign bit moved in
            const uint32_t lo = __byte_perm(0x3E3C3800u, 0x46444240u, wd & 0x7777u);
            const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, (wd >> 16) & 0x7777u);
            uint32_t a[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t v = __byte_perm(lo, hi, q | (q << 4) | ((4 + q) << 8) | ((4 + q) << 12));
                a[q] = (v & 0xFF00FF00u) | ((wd << (12 - 4 * q)) & 0x80008000u);
            }
            const float sg = e4m3f((scb[(2 * j) >> 2] >> (8 * ((2 * j) & 3))) & 0xff);
            const float sg8 = e4m3f((scb[(2 * j + 1) >> 2] >> (8 * ((2 * j + 1) & 3))) & 0xff);
            const __half2 h0 = __float2half2_rn(sg), h1 = __float2half2_rn(sg8);
            const uint32_t m0 = hmul2u(a[0], h0), m1 = hmul2u(a[1], h1), m2 = hmul2u(a[2], h0), m3 = hmul2u(a[3], h1);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma_f16(acc[nt], m0, m1, m2, m3, b[j][nt][0], b[j][nt][1]);
        }
    }
};

struct PBF {
    using W = TBF;
    struct Raw { uint4 a[4]; };
    static constexpr bool kBF = true;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ float scale(const W&) { return 1.f; }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int) {
        size_t K16 = w.k >> 4;
        Raw r;
#pragma unroll
        for (int j = 0; j < 4; j++) r.a[j] = __ldg(w.w + ((size_t)tile * K16 + 4 * s + j) * 32 + lane);
        return r;
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4]) {
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma_bf16f(acc[nt], r.a[j].x, r.a[j].y, r.a[j].z, r.a[j].w, b[j][nt][0], b[j][nt][1]);
    }
};

// ------------------------------------------------------------------------------------------------------
// Decode and verify rows: a block is four warps splitting K for RT row tiles; NT is how many 8-token tiles the rows
// need. Each output element sums its K steps in the same order (warp-strided steps, then warps 0..3) whatever NT is, so
// a row's bits do not depend on the other rows of the block.
// ------------------------------------------------------------------------------------------------------

template <class P, int NT, int RT>
__device__ __forceinline__ void mma_body(const typename P::W& w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                         int Tn, int tile0) {
    __shared__ float red[4][RT][NT][4][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ntiles = P::tiles(w), steps = P::steps(w);
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }

    typename P::Raw cur[RT], nxt[RT];
    int s = warp;
    if (s < steps) {
#pragma unroll
        for (int r = 0; r < RT; r++) cur[r] = P::load(w, min(tile0 + r, ntiles - 1), s, lane, g);
    }
    for (; s < steps; s += 4) {
        const int sn = s + 4;
        if (sn < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) nxt[r] = P::load(w, min(tile0 + r, ntiles - 1), sn, lane, g);
        }
        uint32_t b[4][NT][2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const int tok = nt * 8 + g;
                if (tok < Tn) {
                    const float* p = x + (size_t)tok * ldx + s * 64 + 16 * j + 2 * t;
                    float2 v0 = __ldg(reinterpret_cast<const float2*>(p));
                    float2 v1 = __ldg(reinterpret_cast<const float2*>(p + 8));
                    b[j][nt][0] = pack2<P::kBF>(v0.x, v0.y);
                    b[j][nt][1] = pack2<P::kBF>(v1.x, v1.y);
                } else {
                    b[j][nt][0] = 0u;
                    b[j][nt][1] = 0u;
                }
            }
#pragma unroll
        for (int r = 0; r < RT; r++) P::template step<NT>(cur[r], b, acc[r]);
        if (sn < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) red[warp][r][n][q][lane] = acc[r][n][q];
    __syncthreads();
    const float s2 = P::scale(w);
    for (int e = threadIdx.x; e < RT * NT * 4 * 32; e += blockDim.x) {
        const int ln = e & 31, q = (e >> 5) & 3, n = (e >> 7) % NT, r = (e >> 7) / NT;
        float v = red[0][r][n][q][ln] + red[1][r][n][q][ln] + red[2][r][n][q][ln] + red[3][r][n][q][ln];
        const int lg = ln >> 2, lt = ln & 3;
        const int row = (tile0 + r) * 16 + lg + ((q >> 1) ? 8 : 0);
        const int tok = n * 8 + 2 * lt + (q & 1);
        if (row < w.n && tok < Tn) y[(size_t)tok * ldy + row] = v * s2;
    }
}

// Prompt chunks: the CTA owns 4 warps x RT row tiles and a group of 8*NT tokens. Each warp walks all of K for its own
// rows; the group's activations are converted and staged in shared memory 64 k at a time.
template <class P, int NT, int RT>
__device__ __forceinline__ void mma_big_body(const typename P::W& w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                             int Tn, int tile0, int g0) {
    constexpr int TOK = NT * 8;
    constexpr int STR = 72;   // halves per staged token row: 64 + 8 keeps fragment loads conflict-free
    __shared__ __align__(16) uint16_t xs[2][TOK * STR];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    tile0 += warp * RT;
    const int ntiles = P::tiles(w), steps = P::steps(w);
    const int tn = min(Tn - g0, TOK);
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }

    auto stage = [&](int buf, int s) {
        for (int i = threadIdx.x; i < TOK * 16; i += 128) {
            const int tok = i >> 4, k4 = (i & 15) * 4;
            float4 v = tok < tn ? __ldg(reinterpret_cast<const float4*>(x + (size_t)(g0 + tok) * ldx + s * 64 + k4)) : make_float4(0.f, 0.f, 0.f, 0.f);
            uint2 pk = make_uint2(pack2<P::kBF>(v.x, v.y), pack2<P::kBF>(v.z, v.w));
            *reinterpret_cast<uint2*>(&xs[buf][tok * STR + k4]) = pk;
        }
    };
    typename P::Raw cur[RT], nxt[RT];
    if (steps > 0) {
#pragma unroll
        for (int r = 0; r < RT; r++) cur[r] = P::load(w, min(tile0 + r, ntiles - 1), 0, lane, g);
    }
    stage(0, 0);
    for (int s = 0; s < steps; s++) {
        __syncthreads();
        const int buf = s & 1;
        if (s + 1 < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) nxt[r] = P::load(w, min(tile0 + r, ntiles - 1), s + 1, lane, g);
            stage(buf ^ 1, s + 1);
        }
        uint32_t b[4][NT][2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint16_t* p = &xs[buf][(nt * 8 + g) * STR + 16 * j + 2 * t];
                b[j][nt][0] = *reinterpret_cast<const uint32_t*>(p);
                b[j][nt][1] = *reinterpret_cast<const uint32_t*>(p + 8);
            }
#pragma unroll
        for (int r = 0; r < RT; r++) P::template step<NT>(cur[r], b, acc[r]);
        if (s + 1 < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) cur[r] = nxt[r];
        }
    }
    const float s2 = P::scale(w);
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int row = (tile0 + r) * 16 + g + ((q >> 1) ? 8 : 0);
                const int tok = n * 8 + 2 * t + (q & 1);
                if (row < w.n && tok < tn) y[(size_t)(g0 + tok) * ldy + row] = acc[r][n][q] * s2;
            }
}

// A launch over several matrices: blockIdx.y walks the row groups of matrix 0, then 1, ... (`per` row tiles a group).
template <class W>
struct Multi { W w[4]; float* y[4]; int ldy[4]; int count; };

template <class W>
__device__ __forceinline__ int pick(const Multi<W>& m, int per, int& tile0) {
    int gy = blockIdx.y, which = 0;
    while (which + 1 < m.count) {
        const int groups = (((m.w[which].n + 15) >> 4) + per - 1) / per;
        if (gy < groups) break;
        gy -= groups;
        which++;
    }
    tile0 = gy * per;
    return which;
}

template <class P, int NT, int RT>
__global__ void __launch_bounds__(128) mma_k(Multi<typename P::W> m, const float* __restrict__ x, int ldx, int Tn) {
    int tile0;
    const int which = pick(m, RT, tile0);
    const int g0 = blockIdx.x * NT * 8;
    mma_body<P, NT, RT>(m.w[which], x + (size_t)g0 * ldx, ldx, m.y[which] + (size_t)g0 * m.ldy[which], m.ldy[which], min(Tn - g0, NT * 8), tile0);
}

template <class P, int NT, int RT>
__global__ void __launch_bounds__(128) mma_big_k(Multi<typename P::W> m, const float* __restrict__ x, int ldx, int Tn) {
    int tile0;
    const int which = pick(m, 4 * RT, tile0);
    mma_big_body<P, NT, RT>(m.w[which], x, ldx, m.y[which], m.ldy[which], Tn, tile0, blockIdx.x * NT * 8);
}

static bool g_decode_rows = false;
static const bool g_rt2 = getenv("DORY_RT2") != nullptr;

template <class P>
void launch_mma(cudaStream_t s, const Multi<typename P::W>& m, const float* x, int ldx, int T) {
    constexpr int RT = 2, BR = 2;
    int rg = 0, rg2 = 0;
    for (int i = 0; i < m.count; i++) {
        const int tiles = (m.w[i].n + 15) >> 4;
        rg += (tiles + RT - 1) / RT;
        rg2 += (tiles + 4 * BR - 1) / (4 * BR);
    }
    // narrow matrices: one row tile a warp group, so the GPU has enough CTAs (and loads in flight) for a few rows
    int rg1 = 0;
    for (int i = 0; i < m.count; i++) rg1 += (m.w[i].n + 15) >> 4;
    const bool narrow = rg < 4 * 48 && !g_rt2;
    if (T <= 8 && narrow) mma_k<P, 1, 1><<<dim3(1, rg1), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 16 && narrow) mma_k<P, 2, 1><<<dim3(1, rg1), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 8) mma_k<P, 1, RT><<<dim3(1, rg), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 16) mma_k<P, 2, RT><<<dim3(1, rg), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 32) mma_k<P, 4, RT><<<dim3(1, rg), 128, 0, s>>>(m, x, ldx, T);
    else if (g_decode_rows) mma_k<P, 8, RT><<<dim3((T + 63) / 64, rg), 128, 0, s>>>(m, x, ldx, T);
    else if (T < 64) mma_big_k<P, 4, BR><<<dim3((T + 31) / 32, rg2), 128, 0, s>>>(m, x, ldx, T);
    else mma_big_k<P, 8, BR><<<dim3((T + 63) / 64, rg2), 128, 0, s>>>(m, x, ldx, T);
}

}  // namespace

void mma_decode_rows(bool on) { g_decode_rows = on; }

void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T) {
    Multi<TNV4> m{};
    m.w[0] = w; m.y[0] = y; m.ldy[0] = ldy; m.count = 1;
    launch_mma<PNV4>(s, m, x, ldx, T);
}
void mma_nv4_multi(cudaStream_t s, const NV4Multi& mm, const float* x, int ldx, int T) {
    Multi<TNV4> m{};
    for (int i = 0; i < mm.count; i++) { m.w[i] = mm.w[i]; m.y[i] = mm.y[i]; m.ldy[i] = mm.ldy[i]; }
    m.count = mm.count;
    launch_mma<PNV4>(s, m, x, ldx, T);
}
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T) {
    Multi<TBF> m{};
    m.w[0] = w; m.y[0] = y; m.ldy[0] = ldy; m.count = 1;
    launch_mma<PBF>(s, m, x, ldx, T);
}

// ------------------------------------------------------------------------------------------------------
// Repacking from the checkpoint layouts.
// ------------------------------------------------------------------------------------------------------

size_t tile_bytes_nv4(int n, int k) { return (size_t)((n + 15) / 16) * (k / 64) * 32 * 16; }
size_t tile_scale_bytes_nv4(int n, int k) { return (size_t)((n + 15) / 16) * (k / 64) * 64; }
size_t tile_bytes_bf16(int n, int k) { return (size_t)((n + 15) / 16) * (k / 16) * 32 * 16; }

namespace {

__global__ void tile_nv4_k(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sc, uint32_t* __restrict__ qt,
                           uint8_t* __restrict__ sct, int n, int k, int n16, int S) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * S * 32) return;
    const int lane = i & 31;
    const size_t sidx = i >> 5;
    const int s = sidx % S, rt = sidx / S;
    const int g = lane >> 2, t = lane & 3;
    for (int j = 0; j < 4; j++) {
        uint32_t word = 0;
        for (int r = 0; r < 4; r++) {
            const int row = rt * 16 + g + 8 * (r & 1);
            const int kk = s * 64 + 16 * j + 8 * (r >> 1) + 2 * t;
            uint32_t lo = 0, hi = 0;
            if (row < n) {
                uint8_t byte = q[(size_t)row * (k / 2) + kk / 2];  // kk even: low nibble is kk, high nibble kk+1
                lo = byte & 15;
                hi = byte >> 4;
            }
            word |= (lo << (4 * r)) | (hi << (16 + 4 * r));
        }
        qt[(sidx * 32 + lane) * 4 + j] = word;
    }
    if (t == 0) {
        for (int j = 0; j < 4; j++)
            for (int h = 0; h < 2; h++) {
                const int row = rt * 16 + g + 8 * h;
                sct[(sidx * 8 + g) * 8 + 2 * j + h] = row < n ? sc[(size_t)row * (k / 16) + 4 * s + j] : 0;
            }
    }
}

__global__ void tile_bf16_k(const uint16_t* __restrict__ w, uint32_t* __restrict__ wt, int n, int k, int n16, int K16) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * K16 * 32) return;
    const int lane = i & 31;
    const size_t sidx = i >> 5;
    const int s = sidx % K16, rt = sidx / K16;
    const int g = lane >> 2, t = lane & 3;
    for (int r = 0; r < 4; r++) {
        const int row = rt * 16 + g + 8 * (r & 1);
        const int kk = 16 * s + 8 * (r >> 1) + 2 * t;
        uint32_t v = 0;
        if (row < n) v = (uint32_t)w[(size_t)row * k + kk] | ((uint32_t)w[(size_t)row * k + kk + 1] << 16);
        wt[(sidx * 32 + lane) * 4 + r] = v;
    }
}

}  // namespace

void tile_nv4(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k) {
    int n16 = (n + 15) / 16, S = k / 64;
    size_t th = (size_t)n16 * S * 32;
    tile_nv4_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(q, sc, (uint32_t*)qt, (uint8_t*)sct, n, k, n16, S);
}
void tile_bf16(cudaStream_t s, const uint16_t* w, void* wt, int n, int k) {
    int n16 = (n + 15) / 16, K16 = k / 16;
    size_t th = (size_t)n16 * K16 * 32;
    tile_bf16_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(w, (uint32_t*)wt, n, k, n16, K16);
}
