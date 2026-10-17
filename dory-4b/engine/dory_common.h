// Shared declarations for libdory, NVIDIA Nemotron-Labs Dory1.0's forward pass on TensorFold: packed weight views, the
// per-launch control block and the kernel launchers. Everything here is C++ and private to libdory.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TMAX = 1024;      // most rows one block forward carries (prompt chunks)
constexpr int VERIFY_MAX = 8;   // most rows one sequence's verify window carries
constexpr int SEG_MAX = 16;     // most sequences in one block
constexpr int BATCH_MAX = SEG_MAX * VERIFY_MAX;  // most rows of a block that carries several sequences' windows
constexpr int LOGIT_ROWS = BATCH_MAX;            // rows whose logits are kept (verify windows and the last prompt row)
constexpr int VOCAB = 131072;
constexpr int CAND = 64;        // sampling candidates kept per row

// Model shape (checked against config.json at load).
constexpr int HID = 2560;
constexpr int NH = 16;          // query heads
constexpr int NKV = 4;          // key/value heads
constexpr int HD = 256;         // head dimension
constexpr int QDIM = NH * HD;   // 4096
constexpr int KVDIM = NKV * HD; // 1024
constexpr int FFN = 7168;
constexpr int SWA = 1024;       // sliding window: a row sees itself and the 1023 keys before it
constexpr int RING = 2048;      // sliding layers keep a ring of this many rows (>= SWA - 1 + TMAX)

// Keys one CTA of a verify window's full attention covers: 128, doubled until 32 CTAs reach the end of the sequence and its
// window. It depends on the sequence's own length alone, and chunks start at multiples of it, so a stream's rounding is the
// same alone, among other streams, or in a wider window.
__host__ __device__ inline int win_chunk(int base0) {
    int k = 128;
    while (k * 32 < base0 + VERIFY_MAX) k *= 2;
    return k;
}
constexpr int SWA_CHUNK = 128;  // the same for sliding layers: chunks of 128 keys at absolute multiples
constexpr int SWA_SPLITS = 16;  // enough of them for SWA - 1 + VERIFY_MAX keys

// Tile-major weights for the tensor-core GEMMs: rows come in tiles of 16, laid out so that each lane of a warp reads the A
// fragments of m16n8k16 with one 16-byte load (dory_mma.cu).
struct TNV4 { const uint4* q; const uint2* sc; float s2; int n, k; };   // ModelOpt NVFP4
struct TBF  { const uint4* w; int n, k; };                              // bf16

// Up to four matrices over the same input in one launch, each with its own output.
struct NV4Multi { TNV4 w[4]; float* y[4]; int ldy[4]; int count; };

// Device-visible values that change from launch to launch, so a captured graph can be replayed. A block is a list of
// segments: consecutive rows of one sequence (a prompt chunk is one segment, a shared verify round has one window per
// sequence). Row r belongs to segment rseg[r] and sits at absolute position pos[r] of its sequence.
struct Ctl {
    int tokens[TMAX];
    int pos[TMAX];
    int rseg[TMAX];
    int nrows;
    int nseg;
    int seg_row0[SEG_MAX + 1];  // first row of each segment; [nseg] == nrows
    int seg_base[SEG_MAX];      // the sequence's cache length before the segment
    int seg_seq[SEG_MAX];       // which sequence's cache the segment uses
};

// One attention layer's cache: K and V rows [seq][stride][NKV][HD] bf16. A full layer stores position p at row p; a sliding
// layer (ring > 0) at row p % ring.
struct KvView { uint16_t* k; uint16_t* v; long long stride; int ring; };

// GEMMs: x is [T][ldx] fp32, y is [T][ldy] fp32.
void mma_decode_rows(bool on);   // the GEMMs enqueued next are decode or verify rows (summation order independent of T)
void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_nv4_multi(cudaStream_t s, const NV4Multi& m, const float* x, int ldx, int T);
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T);

size_t tile_bytes_nv4(int n, int k);
size_t tile_scale_bytes_nv4(int n, int k);
size_t tile_bytes_bf16(int n, int k);
void tile_nv4(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k);
void tile_bf16(cudaStream_t s, const uint16_t* w, void* wt, int n, int k);

// Attention over the cache (dory_attn.cu). q is [T][QDIM] fp32 (normed and scaled), out [T][QDIM] fp32.
// multi: every segment is a verify window (<= VERIFY_MAX rows), one CTA group per segment; nseg is the launch bound.
// Otherwise one prompt segment. window: 0 for full attention, else the sliding window. splits: CTAs along the keys.
void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, int window, float* part, float* out,
            int T, int multi, int nseg, int splits);
size_t attn_part_floats(int T, int multi, int nseg, int splits);
int attn_prompt_splits(int T, int ctxlen);

// Elementwise building blocks (dory_kernels.cu); every kernel stops at ctl->nrows (padding rows of a graph are skipped).
void k_embed(cudaStream_t s, const uint16_t* table, const Ctl* ctl, float* x, int T);
// RoPE (neox, full head) on q and k, per-head justnorm, q *= sqk2; K and V go to the cache at each row's position.
void k_qk_prep(cudaStream_t s, const Ctl* ctl, float* q, const float* k, const float* v, const float* sqk2,
               const float* inv_freq, const KvView& kv, int T);
// out = justnorm(o * sigmoid(g * sg2)) over QDIM.
void k_gate_norm(cudaStream_t s, const Ctl* ctl, const float* o, const float* g, const float* sg2, float* out, int T);
// out = justnorm(silu(g * sug) * (u * suu)) over FFN.
void k_mlp_act(cudaStream_t s, const Ctl* ctl, const float* g, const float* u, const float* sug, const float* suu, float* out, int T);
// x = justnorm(justnorm(x) + alpha * (justnorm(r) - justnorm(x))), then (post) justnorm once more.
void k_slerp(cudaStream_t s, const Ctl* ctl, float* x, const float* r, const float* alpha, int post, int T);
// x = justnorm(x + inp)
void k_loop_in(cudaStream_t s, const Ctl* ctl, float* x, const float* inp, int T);
// logits[r][v] *= scale[v] for the first `rows` rows
void k_logit_scale(cudaStream_t s, float* logits, const float* scale, int rows);
// Row-wise top candidates of the logits: vals/ids are [T][CAND] sorted descending (ties to the lower id).
void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* scratch_v, int* scratch_i);
void k_candidates_masked(cudaStream_t s, const float* logits, const int* sel, const uint32_t* masks, int words, int n, int topk,
                         float* vals, int* ids, float* scratch_v, int* scratch_i);
