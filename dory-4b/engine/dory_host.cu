// libdory: NVIDIA Nemotron-Labs Dory1.0 (looped nGPT, NVFP4) on one GPU, the forward pass behind TensorFold's Dory lane
// backend (dory_native.zig). Checkpoint loading, the block forward and the C ABI.
//
// The model: 72 physical sublayers (attention or SwiGLU MLP, each with a slerp residual on the unit sphere) in three
// groups, input (16), recurrent (40) and output (16); the recurrent group runs DORY_LOOPS times (config default 2), its
// input re-injected each loop: h = justnorm(state + input). Attention layers are sliding-window (1024) or full, GQA
// 16/4, head dimension 256, RoPE theta 1e4 / 1e7. Recurrent attention keeps a KV cache per loop (per_loop, the reference
// mode) or one shared by every loop (DORY_KV_MODE=last_loop: each loop overwrites the current rows, so earlier rows hold
// the final loop's keys).
//
// Weights stay packed on the GPU (NVFP4 matrices tile-repacked for the tensor-core GEMMs, bf16 embeddings and head). A
// block of up to TMAX rows is one launch sequence, captured into a CUDA graph when its shape repeats. Several sequences
// (DORY_SEQS) keep their own caches; one block can carry a segment of rows from each, so a shared decode round is one
// forward. There is no recurrent state: rolling a verify window back is setting the sequence's length.

#include "dory_common.h"

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "dory: %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); } } while (0)

namespace {

constexpr int MAX_LOOPS = 4;
constexpr int PROMPT_SPLITS_MAX = 32;

struct Phys {
    int type = 0;   // 0 mlp, 1 sliding attention, 2 full attention
    // attention
    NV4Multi qkvg{};
    TNV4 o{};
    float *sqk2 = nullptr, *sg2 = nullptr, *inv_freq = nullptr;
    // mlp
    NV4Multi gu{};
    TNV4 down{};
    float *sug = nullptr, *suu = nullptr;
    float* alpha = nullptr;   // |alpha * ngpt_alpha_init / initializer_range|
};

// A layer as the forward runs it: its physical weights, its KV cache (attention), where it sits in the loops.
struct Virt {
    int phys;
    int kv = -1;          // index into Dory::kvs
    int loop_begin = 0;   // the recurrent input re-injection runs before it
    int loop_end = 0;     // the loop's closing justnorm runs after it
    int save_input = 0;   // the input group ends before it: keep x as the loop input
};

struct KvLayer { int full = 0; uint16_t *k = nullptr, *v = nullptr; };

struct Info { std::string dtype; std::vector<long> shape; size_t off = 0, len = 0; };

struct Store {
    const uint8_t* base = nullptr;
    size_t size = 0;
    size_t data0 = 0;
    std::unordered_map<std::string, Info> tensors;
};

std::string slurp(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(n, 0);
    if (fread(s.data(), 1, n, f) != (size_t)n) s.clear();
    fclose(f);
    return s;
}

bool parse_header(const uint8_t* file, size_t n, std::unordered_map<std::string, Info>& out, size_t& data0) {
    if (n < 8) return false;
    uint64_t hlen;
    memcpy(&hlen, file, 8);
    if (8 + hlen > n) return false;
    data0 = 8 + hlen;
    std::string h((const char*)file + 8, hlen);
    size_t i = 0;
    while (true) {
        size_t k = h.find("\"dtype\"", i);
        if (k == std::string::npos) break;
        size_t name_end = h.rfind('"', h.rfind('{', k) - 1);
        size_t name_start = h.rfind('"', name_end - 1);
        if (name_start == std::string::npos) break;
        std::string name = h.substr(name_start + 1, name_end - name_start - 1);
        Info info;
        size_t dq = h.find('"', h.find(':', k) + 1);
        size_t dq2 = h.find('"', dq + 1);
        info.dtype = h.substr(dq + 1, dq2 - dq - 1);
        size_t sb = h.find('[', h.find("\"shape\"", k));
        size_t se = h.find(']', sb);
        std::string sh = h.substr(sb + 1, se - sb - 1);
        long v = 0; bool any = false;
        for (char c : sh) {
            if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
            else if (any) { info.shape.push_back(v); v = 0; any = false; }
        }
        if (any) info.shape.push_back(v);
        size_t ob = h.find("\"data_offsets\"", k);
        size_t br = h.find('[', ob);
        unsigned long long a = 0, b = 0;
        sscanf(h.c_str() + br, "[%llu, %llu]", &a, &b);
        if (b < a) sscanf(h.c_str() + br, "[%llu,%llu]", &a, &b);
        info.off = a;
        info.len = b - a;
        out[name] = info;
        i = br;
    }
    return true;
}

Store* open_store(const char* dir) {
    std::string path = std::string(dir) + "/model.safetensors";
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "dory: cannot open %s (one model.safetensors file is read)\n", path.c_str()); return nullptr; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return nullptr;
    madvise(m, st.st_size, MADV_WILLNEED);
    auto* s = new Store();
    s->base = (const uint8_t*)m;
    s->size = st.st_size;
    if (!parse_header(s->base, s->size, s->tensors, s->data0)) { fprintf(stderr, "dory: bad safetensors header\n"); return nullptr; }
    fprintf(stderr, "dory: %zu tensors\n", s->tensors.size());
    return s;
}

bool lookup(Store* s, const std::string& name, Info& info, const uint8_t*& data) {
    auto it = s->tensors.find(name);
    if (it == s->tensors.end()) return false;
    info = it->second;
    data = s->base + s->data0 + info.off;
    return true;
}

std::vector<float> host_f32(Store* s, const std::string& name, bool& ok) {
    Info info; const uint8_t* data;
    std::vector<float> h;
    if (!lookup(s, name, info, data)) { fprintf(stderr, "dory: missing %s\n", name.c_str()); ok = false; return h; }
    size_t n = 1;
    for (long d : info.shape) n *= (size_t)d;
    h.resize(n);
    if (info.dtype == "F32") memcpy(h.data(), data, n * 4);
    else if (info.dtype == "BF16") {
        for (size_t i = 0; i < n; i++) {
            uint16_t v; memcpy(&v, data + i * 2, 2);
            uint32_t u = (uint32_t)v << 16;
            memcpy(&h[i], &u, 4);
        }
    } else { fprintf(stderr, "dory: %s dtype %s\n", name.c_str(), info.dtype.c_str()); ok = false; }
    return h;
}

float* dev_floats(const std::vector<float>& h) {
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

void* dev_copy(const void* src, size_t n) {
    void* d = nullptr;
    CK(cudaMalloc(&d, n));
    CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
    return d;
}

TNV4 load_nv4(Store* s, const std::string& name, int n, int k, bool& ok) {
    TNV4 m{};
    Info w, sc, g; const uint8_t *wd, *sd, *gd;
    if (!lookup(s, name + ".weight", w, wd) || w.dtype != "U8" || w.shape.size() != 2 || w.shape[0] != n || w.shape[1] * 2 != k ||
        !lookup(s, name + ".weight_scale", sc, sd) || sc.dtype != "F8_E4M3" || sc.shape.size() != 2 || sc.shape[1] * 16 != k ||
        !lookup(s, name + ".weight_scale_2", g, gd) || g.dtype != "F32") {
        fprintf(stderr, "dory: %s is not an NVFP4 [%d x %d] tensor\n", name.c_str(), n, k);
        ok = false;
        return m;
    }
    m.n = n;
    m.k = k;
    memcpy(&m.s2, gd, 4);
    void* rq = dev_copy(wd, w.len);
    void* rs = dev_copy(sd, sc.len);
    void *qt, *sct;
    CK(cudaMalloc(&qt, tile_bytes_nv4(n, k)));
    CK(cudaMalloc(&sct, tile_scale_bytes_nv4(n, k)));
    tile_nv4(0, (const uint8_t*)rq, (const uint8_t*)rs, qt, sct, n, k);
    CK(cudaDeviceSynchronize());
    cudaFree(rq); cudaFree(rs);
    m.q = (const uint4*)qt; m.sc = (const uint2*)sct;
    return m;
}

TBF load_bf16(Store* s, const std::string& name, bool& ok) {
    TBF m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name, w, data) || w.dtype != "BF16" || w.shape.size() != 2) {
        fprintf(stderr, "dory: %s is not a bf16 matrix\n", name.c_str());
        ok = false;
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1];
    void* raw = dev_copy(data, w.len);
    void* wt;
    CK(cudaMalloc(&wt, tile_bytes_bf16(m.n, m.k)));
    tile_bf16(0, (const uint16_t*)raw, wt, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    m.w = (const uint4*)wt;
    return m;
}

// ---- config.json, read by hand: numbers and the layer_types list ----

double cfg_num(const std::string& cfg, const char* key, double dflt) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = cfg.find(k);
    if (p == std::string::npos) return dflt;
    p = cfg.find(':', p);
    while (p < cfg.size() && (cfg[p] == ':' || cfg[p] == ' ' || cfg[p] == '\n' || cfg[p] == '\t')) p++;
    if (cfg.compare(p, 4, "null") == 0) return dflt;
    return atof(cfg.c_str() + p);
}

std::vector<std::string> cfg_layer_types(const std::string& cfg) {
    std::vector<std::string> t;
    size_t p = cfg.find("\"layer_types\"");
    if (p == std::string::npos) return t;
    size_t a = cfg.find('[', p), b = cfg.find(']', a);
    size_t i = a;
    while (true) {
        size_t q1 = cfg.find('"', i + 1);
        if (q1 == std::string::npos || q1 > b) break;
        size_t q2 = cfg.find('"', q1 + 1);
        t.push_back(cfg.substr(q1 + 1, q2 - q1 - 1));
        i = q2;
    }
    return t;
}

// rope theta of one layer type in rope_parameters
double cfg_theta(const std::string& cfg, const char* type, double dflt) {
    size_t p = cfg.find("\"rope_parameters\"");
    if (p == std::string::npos) return dflt;
    p = cfg.find(std::string("\"") + type + "\"", p);
    if (p == std::string::npos) return dflt;
    size_t e = cfg.find('}', p);
    size_t t = cfg.find("\"rope_theta\"", p);
    if (t == std::string::npos || t > e) return dflt;
    return atof(cfg.c_str() + cfg.find(':', t) + 1);
}

}  // namespace

// Rows of the full layers' caches, [seq][kv_stride tokens][NKV][HD] bf16 per layer for K and V. The address range is
// reserved at the full window for every sequence, but physical memory is mapped a granule (2 MiB, 1024 tokens) at a time
// as a sequence grows, so every stream can have the native 262,144-token window while the memory held follows what the
// streams have actually used. If the driver cannot do that, everything is allocated up front.
struct KvPool {
    bool lazy = false;
    size_t gran = 0;
    int tokens_per_gran = 0;
    int gran_max = 0;
    size_t stride_bytes = 0;
    std::vector<CUdeviceptr> base;
    std::vector<CUmemGenericAllocationHandle> handle;   // [buffer][seq][granule]
    std::vector<int> mapped;
    CUmemAllocationProp prop{};
    size_t reserve_free = 0;
};

struct Dory {
    int context = 32768;
    int loops = 2;
    bool last_loop = false;
    bool kv_pin = false;
    int kv_row = KVDIM * 2;   // bytes of one token's K (or V) in one layer
    int kv_stride = 32768;    // tokens between sequences in the full layers' buffers
    KvPool kv;
    int n_in = 16, n_rec = 40, n_out = 16;
    uint16_t* embed = nullptr;
    TBF head{};
    float* lscale = nullptr;
    std::vector<Phys> phys;
    std::vector<Virt> virt;
    std::vector<KvLayer> kvs;
    int n_full = 0, n_swa = 0;

    float *x = nullptr, *inp = nullptr, *r = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr;
    float *ao = nullptr, *an = nullptr, *mg = nullptr, *mu = nullptr, *ma = nullptr, *part = nullptr;
    float* logits = nullptr;
    float *cvals = nullptr, *csv = nullptr;
    int *cids = nullptr, *csi = nullptr;
    std::vector<float> cvals_h;
    std::vector<int> cids_h;
    int* msel = nullptr;
    uint32_t* mmask = nullptr;
    float* mvals = nullptr;
    int* mids = nullptr;

    Ctl* ctl_d = nullptr;
    Ctl* ctl_h = nullptr;
    cudaStream_t s0 = nullptr;
    std::unordered_map<long long, cudaGraphExec_t> graphs;
    std::unordered_map<long long, int> uses;

    struct Seq {
        bool used = false;
        int len = 0;
        int last_rows = 0, last_len = 0, last_record = 0;
    };
    int nseq = 1;
    std::vector<Seq> seqs;
    int last_cand_k = 0;
    double cost_ms[VERIFY_MAX + 1] = {};
    std::vector<std::pair<int, double>> shared_ms;
    bool prof = false;   // DORY_PROF: eager launches with an event after every stage, totals by dory_prof_report
    std::vector<cudaEvent_t> prof_ev;
    std::vector<const char*> prof_name;
    std::unordered_map<std::string, std::pair<double, int>> prof_acc;
};

namespace {

uint64_t kv_token_bytes(const Dory* b) { return (uint64_t)b->n_full * 2 * b->kv_row; }
uint64_t swa_seq_bytes(const Dory* b) { return (uint64_t)b->n_swa * 2 * RING * b->kv_row; }

bool kv_open(Dory* b, int nbuf) {
    KvPool& k = b->kv;
    if (getenv("DORY_KV_EAGER") || nbuf < 1) return false;
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return false;
    int vmm = 0;
    if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !vmm) return false;
    k.prop = {};
    k.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    k.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    k.prop.location.id = dev;
    if (cuMemGetAllocationGranularity(&k.gran, &k.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || k.gran % b->kv_row) return false;
    k.tokens_per_gran = (int)(k.gran / b->kv_row);
    b->kv_stride = (b->context + k.tokens_per_gran - 1) / k.tokens_per_gran * k.tokens_per_gran;
    k.gran_max = b->kv_stride / k.tokens_per_gran;
    k.stride_bytes = (size_t)b->kv_stride * b->kv_row;
    k.base.assign(nbuf, 0);
    for (int i = 0; i < nbuf; i++) {
        if (cuMemAddressReserve(&k.base[i], (size_t)b->nseq * k.stride_bytes, k.gran, 0, 0) != CUDA_SUCCESS) {
            for (int j = 0; j < i; j++) cuMemAddressFree(k.base[j], (size_t)b->nseq * k.stride_bytes);
            k.base.clear();
            return false;
        }
    }
    k.handle.assign((size_t)nbuf * b->nseq * k.gran_max, 0);
    k.mapped.assign(b->nseq, 0);
    const char* r = getenv("DORY_KV_RESERVE_GIB");
    k.reserve_free = (size_t)((r ? atof(r) : 3.0) * 1073741824.0);
    k.lazy = true;
    return true;
}

void kv_unmap_granule(Dory* b, int seq, int g, int nbufs) {
    KvPool& k = b->kv;
    for (int i = 0; i < nbufs; i++) {
        CUmemGenericAllocationHandle& h = k.handle[((size_t)i * b->nseq + seq) * k.gran_max + g];
        if (!h) continue;
        cuMemUnmap(k.base[i] + (size_t)seq * k.stride_bytes + (size_t)g * k.gran, k.gran);
        cuMemRelease(h);
        h = 0;
    }
}

// Map physical memory under the first `tokens` rows of the sequence. False when the driver or the free memory refuses.
bool kv_ensure(Dory* b, int seq, int tokens) {
    KvPool& k = b->kv;
    if (!k.lazy) return true;
    const int need = (tokens + k.tokens_per_gran - 1) / k.tokens_per_gran;
    if (need > k.gran_max) return false;
    if (k.mapped[seq] >= need) return true;
    cudaFree(0);
    const int nbuf = (int)k.base.size();
    CUmemAccessDesc acc = {};
    acc.location = k.prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    while (k.mapped[seq] < need) {
        const int g = k.mapped[seq];
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if (fr < k.reserve_free + (size_t)nbuf * k.gran) {
            fprintf(stderr, "dory: no memory to grow sequence %d past %d tokens (%.1f GiB free)\n", seq, g * k.tokens_per_gran, fr / 1073741824.0);
            return false;
        }
        int done = 0;
        for (; done < nbuf; done++) {
            CUmemGenericAllocationHandle h;
            if (cuMemCreate(&h, k.gran, &k.prop, 0) != CUDA_SUCCESS) break;
            CUdeviceptr addr = k.base[done] + (size_t)seq * k.stride_bytes + (size_t)g * k.gran;
            if (cuMemMap(addr, k.gran, 0, h, 0) != CUDA_SUCCESS) { cuMemRelease(h); break; }
            if (cuMemSetAccess(addr, k.gran, &acc, 1) != CUDA_SUCCESS) { cuMemUnmap(addr, k.gran); cuMemRelease(h); break; }
            k.handle[((size_t)done * b->nseq + seq) * k.gran_max + g] = h;
            cuMemsetD8Async(addr, 0, k.gran, b->s0);
        }
        if (done < nbuf) {
            kv_unmap_granule(b, seq, g, done);
            return false;
        }
        k.mapped[seq]++;
    }
    return true;
}

void kv_trim(Dory* b, int seq) {
    KvPool& k = b->kv;
    if (!k.lazy || b->kv_pin) return;
    cudaFree(0);
    CK(cudaStreamSynchronize(b->s0));
    const int nbuf = (int)k.base.size();
    while (k.mapped[seq] > 1) {
        kv_unmap_granule(b, seq, k.mapped[seq] - 1, nbuf);
        k.mapped[seq]--;
    }
}

KvView view_of(const Dory* b, const KvLayer& l) {
    KvView v;
    v.k = l.k;
    v.v = l.v;
    v.stride = l.full ? b->kv_stride : RING;
    v.ring = l.full ? 0 : RING;
    return v;
}

void mark(Dory* b, const char* name) {
    if (!b->prof) return;
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    CK(cudaEventRecord(e, b->s0));
    b->prof_ev.push_back(e);
    b->prof_name.push_back(name);
}

void prof_collect(Dory* b) {
    for (size_t i = 1; i < b->prof_ev.size(); i++) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, b->prof_ev[i - 1], b->prof_ev[i]));
        auto& a = b->prof_acc[b->prof_name[i]];
        a.first += ms;
        a.second++;
    }
    for (auto e : b->prof_ev) cudaEventDestroy(e);
    b->prof_ev.clear();
    b->prof_name.clear();
}

void enqueue_block(Dory* b, int T, int record, int cand_k, int last_only, int nseg, int split_log, int psplits) {
    mma_decode_rows(record != 0);
    cudaStream_t s = b->s0;
    const Ctl* ctl = b->ctl_d;
    mark(b, "start");
    k_embed(s, b->embed, ctl, b->x, T);
    mark(b, "embed");
    for (const Virt& vl : b->virt) {
        if (vl.save_input) CK(cudaMemcpyAsync(b->inp, b->x, (size_t)T * HID * 4, cudaMemcpyDeviceToDevice, s));
        if (vl.loop_begin) k_loop_in(s, ctl, b->x, b->inp, T);
        const Phys& p = b->phys[vl.phys];
        if (p.type == 0) {
            mma_nv4_multi(s, p.gu, b->x, HID, T);
            mark(b, "mlp.gate_up");
            k_mlp_act(s, ctl, b->mg, b->mu, p.sug, p.suu, b->ma, T);
            mark(b, "mlp.act");
            mma_nv4(s, p.down, b->ma, FFN, b->r, HID, T);
            mark(b, "mlp.down");
        } else {
            mma_nv4_multi(s, p.qkvg, b->x, HID, T);
            mark(b, "attn.qkvg");
            const KvView kv = view_of(b, b->kvs[vl.kv]);
            k_qk_prep(s, ctl, b->q, b->k, b->v, p.sqk2, p.inv_freq, kv, T);
            mark(b, "attn.qk_prep");
            const int window = p.type == 1 ? SWA : 0;
            if (record) k_attn(s, ctl, b->q, kv, window, b->part, b->ao, T, 1, nseg, window ? SWA_SPLITS : (1 << split_log));
            else k_attn(s, ctl, b->q, kv, window, b->part, b->ao, T, 0, 1, window ? 1 : psplits);
            mark(b, window ? "attn.core_swa" : "attn.core_full");
            k_gate_norm(s, ctl, b->ao, b->g, p.sg2, b->an, T);
            mark(b, "attn.gate_norm");
            mma_nv4(s, p.o, b->an, QDIM, b->r, HID, T);
            mark(b, "attn.o_proj");
        }
        k_slerp(s, ctl, b->x, b->r, p.alpha, vl.loop_end, T);
        mark(b, "slerp");
    }
    if (cand_k > 0) {
        const int rows = last_only ? 1 : T;
        const float* hx = last_only ? b->x + (size_t)(T - 1) * HID : b->x;
        mma_bf16(s, b->head, hx, HID, b->logits, VOCAB, rows);
        mark(b, "lm_head");
        k_logit_scale(s, b->logits, b->lscale, rows);
        k_candidates(s, b->logits, rows, cand_k, b->cvals, b->cids, b->csv, b->csi);
        mark(b, "candidates");
    }
}

int bucket_of(int nseg) { return nseg <= 1 ? 0 : nseg <= 2 ? 1 : nseg <= 4 ? 2 : nseg <= 8 ? 3 : 4; }
constexpr int BUCKET_SEGS[5] = {1, 2, 4, 8, 16};

long long graph_key(int T, int record, int cand_k, int last_only, int bucket, int split_log, int psplits) {
    return ((((((long long)T * 2 + record) * 2 + last_only) * 5 + bucket) * 6 + split_log) * 64 + psplits) * 128 + cand_k;
}

struct SegIn { int seq, rows; const int* tokens; };

// CTAs a verify window's full attention launches per stream and KV head, a power of two: the most any sequence needs at
// its own win_chunk (the kernel gives each sequence the same key chunks whatever this is, so it changes no rounding).
int split_log_of(const SegIn* sg, int nseg, const Dory* b) {
    int need = 1, lg = 0;
    for (int i = 0; i < nseg; i++) {
        const int base0 = b->seqs[sg[i].seq].len, chunk = win_chunk(base0);
        need = std::max(need, (base0 + sg[i].rows + chunk - 1) / chunk);
    }
    while ((1 << lg) < need && lg < 5) lg++;
    return lg;
}

// Run one block: a segment of rows for each of nseg distinct sequences. A record block (verify windows, up to VERIFY_MAX
// rows a segment) can be cut back per sequence by keep; any other block is a single segment. Returns 0 on success.
int run_blocks(Dory* b, const SegIn* sg, int nseg, int record, int cand_k, int last_only = 0) {
    if (nseg < 1 || nseg > SEG_MAX || (!record && nseg != 1)) return -1;
    int T = 0;
    for (int i = 0; i < nseg; i++) {
        if (sg[i].seq < 0 || sg[i].seq >= b->nseq || sg[i].rows < 1) return -1;
        for (int j = 0; j < i; j++) if (sg[j].seq == sg[i].seq) return -1;
        if (record && sg[i].rows > VERIFY_MAX) return -1;
        if (b->seqs[sg[i].seq].len + sg[i].rows > b->context) return -2;
        if (!kv_ensure(b, sg[i].seq, b->seqs[sg[i].seq].len + sg[i].rows)) return -4;
        T += sg[i].rows;
    }
    if (T > TMAX || (record && T > BATCH_MAX)) return -1;
    if (cand_k > 0 && !last_only && T > LOGIT_ROWS) return -1;
    Ctl* h = b->ctl_h;
    int row = 0;
    for (int i = 0; i < nseg; i++) {
        Dory::Seq& q = b->seqs[sg[i].seq];
        h->seg_row0[i] = row;
        h->seg_base[i] = q.len;
        h->seg_seq[i] = sg[i].seq;
        for (int t = 0; t < sg[i].rows; t++) {
            h->tokens[row + t] = sg[i].tokens[t];
            h->pos[row + t] = q.len + t;
            h->rseg[row + t] = i;
        }
        row += sg[i].rows;
    }
    h->seg_row0[nseg] = row;
    h->nrows = T;
    h->nseg = nseg;
    CK(cudaMemcpyAsync(b->ctl_d, h, sizeof(Ctl), cudaMemcpyHostToDevice, b->s0));
    const int bucket = bucket_of(nseg);
    const int split_log = record ? split_log_of(sg, nseg, b) : 0;
    const int psplits = record ? 0 : attn_prompt_splits(T, b->seqs[sg[0].seq].len + T);
    // Shared blocks come in many sizes: round the rows up so a handful of captured graphs serve them all. The extra rows
    // touch no sequence's state (the kernels stop at ctl->nrows).
    const int real_rows = T;
    if (record && (nseg > 1 || T > VERIFY_MAX)) T = T <= 64 ? (T + 7) / 8 * 8 : (T + 15) / 16 * 16;
    const long long key = graph_key(T, record, cand_k, last_only, bucket, split_log, psplits);
    auto it = b->graphs.find(key);
    static const bool eager = getenv("DORY_NO_GRAPHS") != nullptr;
    if (eager || b->prof) {
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log, psplits);
    } else if (it != b->graphs.end()) {
        CK(cudaGraphLaunch(it->second, b->s0));
    } else if (++b->uses[key] >= 2) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(b->s0, cudaStreamCaptureModeGlobal));
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log, psplits);
        CK(cudaStreamEndCapture(b->s0, &g));
        cudaGraphExec_t exec;
        CK(cudaGraphInstantiate(&exec, g, 0));
        cudaGraphDestroy(g);
        b->graphs[key] = exec;
        CK(cudaGraphLaunch(exec, b->s0));
    } else {
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log, psplits);
    }
    if (cand_k > 0) {
        const int rows = last_only ? 1 : real_rows;
        CK(cudaMemcpyAsync(b->cvals_h.data(), b->cvals, (size_t)rows * CAND * 4, cudaMemcpyDeviceToHost, b->s0));
        CK(cudaMemcpyAsync(b->cids_h.data(), b->cids, (size_t)rows * CAND * 4, cudaMemcpyDeviceToHost, b->s0));
    }
    CK(cudaStreamSynchronize(b->s0));
    if (b->prof) prof_collect(b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "dory: forward T=%d: %s\n", T, cudaGetErrorString(e));
        return -3;
    }
    b->last_cand_k = cand_k;
    for (int i = 0; i < nseg; i++) {
        Dory::Seq& q = b->seqs[sg[i].seq];
        q.last_rows = sg[i].rows;
        q.last_len = q.len;
        q.last_record = record;
        q.len += sg[i].rows;
    }
    return 0;
}

int run_block(Dory* b, int seq, const int* tokens, int T, int record, int cand_k, int last_only = 0) {
    SegIn s{seq, T, tokens};
    return run_blocks(b, &s, 1, record, cand_k, last_only);
}

}  // namespace

static void dory_calibrate(Dory* b);

extern "C" Dory* dory_open(const char* dir, int context) {
    CK(cudaFree(0));
    fprintf(stderr, "dory: loading %s\n", dir);
    const std::string cfg = slurp(std::string(dir) + "/config.json");
    if (cfg.find("\"dory\"") == std::string::npos) { fprintf(stderr, "dory: %s/config.json is not a Dory config\n", dir); return nullptr; }
    const auto types = cfg_layer_types(cfg);
    Dory* b = new Dory();
    b->n_in = (int)cfg_num(cfg, "n_input_layers", 0);
    b->n_rec = (int)cfg_num(cfg, "n_recurrent_layers", 0);
    b->n_out = (int)cfg_num(cfg, "n_output_layers", 0);
    b->loops = (int)cfg_num(cfg, "n_recurrent_loops", 1);
    if (const char* e = getenv("DORY_LOOPS")) b->loops = atoi(e);
    if (const char* e = getenv("DORY_KV_MODE")) b->last_loop = !strcmp(e, "last_loop");
    const double std_init = cfg_num(cfg, "initializer_range", 0.0177);
    const double alpha_init = cfg_num(cfg, "ngpt_alpha_init", 0.05);
    const double gate_init = cfg_num(cfg, "attention_output_gate_logit_scale_init", std::sqrt((double)HID));
    const int nl = (int)types.size();
    if ((int)cfg_num(cfg, "hidden_size", 0) != HID || (int)cfg_num(cfg, "num_attention_heads", 0) != NH ||
        (int)cfg_num(cfg, "num_key_value_heads", 0) != NKV || (int)cfg_num(cfg, "head_dim", 0) != HD ||
        (int)cfg_num(cfg, "intermediate_size", 0) != FFN || (int)cfg_num(cfg, "vocab_size", 0) != VOCAB ||
        (int)cfg_num(cfg, "sliding_window", 0) != SWA || nl != (int)cfg_num(cfg, "num_hidden_layers", -1) ||
        b->n_in + b->n_rec + b->n_out != nl || b->n_in < 1 || b->n_rec < 1 || b->loops < 1 || b->loops > MAX_LOOPS) {
        fprintf(stderr, "dory: this build serves Dory1.0-4B's shape (hidden %d, %d/%d heads of %d, ffn %d, window %d, 1-%d loops); config.json differs\n",
                HID, NH, NKV, HD, FFN, SWA, MAX_LOOPS);
        return nullptr;
    }
    Store* store = open_store(dir);
    if (!store) return nullptr;
    b->context = context > 0 ? context : 32768;
    if (b->context > 262144) b->context = 262144;
    {
        const char* ns = getenv("DORY_SEQS");
        b->nseq = ns ? atoi(ns) : 1;
        if (b->nseq < 1) b->nseq = 1;
        if (b->nseq > SEG_MAX) b->nseq = SEG_MAX;
        b->seqs.assign(b->nseq, Dory::Seq());
    }
    CK(cudaStreamCreateWithFlags(&b->s0, cudaStreamNonBlocking));
    bool ok = true;
    {
        Info ei; const uint8_t* ed;
        if (!lookup(store, "backbone.embeddings.weight", ei, ed) || ei.dtype != "BF16" || ei.len != (size_t)VOCAB * HID * 2) {
            fprintf(stderr, "dory: no bf16 backbone.embeddings.weight [%d x %d]\n", VOCAB, HID);
            return nullptr;
        }
        b->embed = (uint16_t*)dev_copy(ed, ei.len);
    }
    b->head = load_bf16(store, "lm_head.weight", ok);
    {
        auto ls = host_f32(store, "lm_head.logit_scale", ok);
        for (float& f : ls) f = (float)(f / std_init);
        if (ok) b->lscale = dev_floats(ls);
    }
    const double theta_swa = cfg_theta(cfg, "sliding_attention", 10000.0), theta_full = cfg_theta(cfg, "full_attention", 10000000.0);
    float* inv_swa = nullptr;
    float* inv_full = nullptr;
    {
        std::vector<float> a(HD / 2), c(HD / 2);
        for (int i = 0; i < HD / 2; i++) {
            const float e = (float)(2 * i) / (float)HD;
            a[i] = (float)(1.0 / std::pow((double)(float)theta_swa, (double)e));
            c[i] = (float)(1.0 / std::pow((double)(float)theta_full, (double)e));
        }
        inv_swa = dev_floats(a);
        inv_full = dev_floats(c);
    }
    const double sqk_f = std::sqrt((double)HD) / std_init, sg_f = std::sqrt(gate_init) / std_init, al_f = alpha_init / std_init, suv_f = std::sqrt((double)HID);
    b->phys.resize(nl);
    CK(cudaMalloc(&b->x, (size_t)TMAX * HID * 4));
    CK(cudaMalloc(&b->q, (size_t)TMAX * QDIM * 4));
    CK(cudaMalloc(&b->k, (size_t)TMAX * KVDIM * 4));
    CK(cudaMalloc(&b->v, (size_t)TMAX * KVDIM * 4));
    CK(cudaMalloc(&b->g, (size_t)TMAX * QDIM * 4));
    CK(cudaMalloc(&b->mg, (size_t)TMAX * FFN * 4));
    CK(cudaMalloc(&b->mu, (size_t)TMAX * FFN * 4));
    for (int li = 0; li < nl && ok; li++) {
        Phys& p = b->phys[li];
        const std::string pre = "backbone.layers." + std::to_string(li);
        auto vec = [&](const std::string& n, double f, bool square, bool absval) {
            auto h = host_f32(store, n, ok);
            for (float& x : h) {
                double v = x * f;
                if (square) v = v * v;
                if (absval) v = std::fabs(v);
                x = (float)v;
            }
            return ok ? dev_floats(h) : nullptr;
        };
        if (types[li] == "mlp") {
            p.type = 0;
            p.gu.w[0] = load_nv4(store, pre + ".mixer.gate_proj", FFN, HID, ok);
            p.gu.w[1] = load_nv4(store, pre + ".mixer.up_proj", FFN, HID, ok);
            p.gu.y[0] = b->mg; p.gu.y[1] = b->mu;
            p.gu.ldy[0] = FFN; p.gu.ldy[1] = FFN;
            p.gu.count = 2;
            p.down = load_nv4(store, pre + ".mixer.down_proj", HID, FFN, ok);
            p.sug = vec(pre + ".mixer.suv_gate", suv_f, false, false);
            p.suu = vec(pre + ".mixer.suv_up", suv_f, false, false);
            p.alpha = vec(pre + ".mlp_alpha", al_f, false, true);
        } else if (types[li] == "sliding_attention" || types[li] == "full_attention") {
            p.type = types[li] == "full_attention" ? 2 : 1;
            p.qkvg.w[0] = load_nv4(store, pre + ".mixer.q_proj", QDIM, HID, ok);
            p.qkvg.w[1] = load_nv4(store, pre + ".mixer.k_proj", KVDIM, HID, ok);
            p.qkvg.w[2] = load_nv4(store, pre + ".mixer.v_proj", KVDIM, HID, ok);
            p.qkvg.w[3] = load_nv4(store, pre + ".mixer.g_proj", QDIM, HID, ok);
            p.qkvg.y[0] = b->q; p.qkvg.y[1] = b->k; p.qkvg.y[2] = b->v; p.qkvg.y[3] = b->g;
            p.qkvg.ldy[0] = QDIM; p.qkvg.ldy[1] = KVDIM; p.qkvg.ldy[2] = KVDIM; p.qkvg.ldy[3] = QDIM;
            p.qkvg.count = 4;
            p.o = load_nv4(store, pre + ".mixer.o_proj", HID, QDIM, ok);
            p.sqk2 = vec(pre + ".mixer.sqk", sqk_f, true, false);
            p.sg2 = vec(pre + ".mixer.s_gate", sg_f, true, false);
            p.alpha = vec(pre + ".attn_alpha", al_f, false, true);
            p.inv_freq = p.type == 2 ? inv_full : inv_swa;
        } else {
            fprintf(stderr, "dory: layer %d has type %s\n", li, types[li].c_str());
            ok = false;
        }
    }
    munmap((void*)store->base, store->size);
    delete store;
    if (!ok) { fprintf(stderr, "dory: the checkpoint does not match a Dory NVFP4 export\n"); return nullptr; }

    // the layers in the order the forward runs them, and the KV cache each attention layer uses
    {
        std::unordered_map<long long, int> kv_of;   // (phys, loop) -> cache
        auto add = [&](int p, int loop, bool rec) {
            Virt v{};
            v.phys = p;
            if (b->phys[p].type != 0) {
                const long long key = (long long)p * 8 + ((rec && !b->last_loop) ? loop : 0);
                auto it = kv_of.find(key);
                if (it == kv_of.end()) {
                    KvLayer l;
                    l.full = b->phys[p].type == 2;
                    b->kvs.push_back(l);
                    it = kv_of.emplace(key, (int)b->kvs.size() - 1).first;
                }
                v.kv = it->second;
            }
            b->virt.push_back(v);
        };
        for (int p = 0; p < b->n_in; p++) add(p, 0, false);
        for (int lp = 0; lp < b->loops; lp++) {
            const size_t first = b->virt.size();
            for (int p = b->n_in; p < b->n_in + b->n_rec; p++) add(p, lp, true);
            b->virt[first].loop_begin = 1;
            if (lp == 0) b->virt[first].save_input = 1;
            b->virt.back().loop_end = 1;
        }
        for (int p = b->n_in + b->n_rec; p < nl; p++) add(p, 0, false);
        for (auto& l : b->kvs) (l.full ? b->n_full : b->n_swa)++;
    }
    {
        // full layers: lazily mapped address ranges; sliding layers: a ring a sequence, allocated now
        b->kv_stride = b->context;
        kv_open(b, 2 * b->n_full);
        int fi = 0;
        for (auto& l : b->kvs) {
            if (l.full) {
                if (b->kv.lazy) {
                    l.k = (uint16_t*)(uintptr_t)b->kv.base[2 * fi];
                    l.v = (uint16_t*)(uintptr_t)b->kv.base[2 * fi + 1];
                } else {
                    const size_t bytes = (size_t)b->nseq * b->context * b->kv_row;
                    CK(cudaMalloc(&l.k, bytes));
                    CK(cudaMalloc(&l.v, bytes));
                    CK(cudaMemset(l.k, 0, bytes));
                    CK(cudaMemset(l.v, 0, bytes));
                }
                fi++;
            } else {
                const size_t bytes = (size_t)b->nseq * RING * b->kv_row;
                CK(cudaMalloc(&l.k, bytes));
                CK(cudaMalloc(&l.v, bytes));
                CK(cudaMemset(l.k, 0, bytes));
                CK(cudaMemset(l.v, 0, bytes));
            }
        }
        if (b->kv.lazy) for (int q = 0; q < b->nseq; q++) if (!kv_ensure(b, q, 1)) { fprintf(stderr, "dory: cannot map the first KV granule\n"); return nullptr; }
    }
    auto zalloc = [&](float** p, size_t floats) { CK(cudaMalloc(p, floats * 4)); CK(cudaMemset(*p, 0, floats * 4)); };
    CK(cudaMemset(b->x, 0, (size_t)TMAX * HID * 4));
    zalloc(&b->inp, (size_t)TMAX * HID);
    zalloc(&b->r, (size_t)TMAX * HID);
    zalloc(&b->ao, (size_t)TMAX * QDIM);
    zalloc(&b->an, (size_t)TMAX * QDIM);
    zalloc(&b->ma, (size_t)TMAX * FFN);
    zalloc(&b->part, std::max({attn_part_floats(TMAX, 0, 1, 1), attn_part_floats(16, 0, 1, PROMPT_SPLITS_MAX) * 8,
                               attn_part_floats(0, 1, SEG_MAX, 32), attn_part_floats(0, 1, SEG_MAX, SWA_SPLITS)}));
    zalloc(&b->logits, (size_t)LOGIT_ROWS * VOCAB);
    zalloc(&b->cvals, (size_t)LOGIT_ROWS * CAND);
    zalloc(&b->csv, (size_t)LOGIT_ROWS * 128 * CAND);
    CK(cudaMalloc(&b->cids, (size_t)LOGIT_ROWS * CAND * 4));
    CK(cudaMalloc(&b->csi, (size_t)LOGIT_ROWS * 128 * CAND * 4));
    CK(cudaMalloc(&b->msel, (size_t)LOGIT_ROWS * 4));
    CK(cudaMalloc(&b->mmask, (size_t)LOGIT_ROWS * (VOCAB / 32) * 4));
    CK(cudaMalloc(&b->mvals, (size_t)LOGIT_ROWS * CAND * 4));
    CK(cudaMalloc(&b->mids, (size_t)LOGIT_ROWS * CAND * 4));
    b->cvals_h.assign((size_t)LOGIT_ROWS * CAND, 0.f);
    b->cids_h.assign((size_t)LOGIT_ROWS * CAND, 0);
    CK(cudaMalloc(&b->ctl_d, sizeof(Ctl)));
    CK(cudaMallocHost(&b->ctl_h, sizeof(Ctl)));
    memset(b->ctl_h, 0, sizeof(Ctl));
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError() != cudaSuccess) return nullptr;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "dory: weights resident; %d layers run (%d input, %d recurrent x %d loops, %d output), %d KV caches (%d full, %d sliding, %s), %d sequence%s of context %d, %.1f GiB free\n",
            (int)b->virt.size(), b->n_in, b->n_rec, b->loops, b->n_out, (int)b->kvs.size(), b->n_full, b->n_swa,
            b->last_loop ? "last_loop" : "per_loop", b->nseq, b->nseq == 1 ? "" : "s", b->context, fr / 1073741824.0);
    if (b->kv.lazy && getenv("DORY_KV_PIN") && atoi(getenv("DORY_KV_PIN")) != 0) {
        for (int q = 0; q < b->nseq; q++) {
            if (!kv_ensure(b, q, b->context)) {
                fprintf(stderr, "dory: cannot pin %d sequences of %d tokens of KV cache (mapped %d); lower PARALLEL or CONTEXT\n", b->nseq, b->context, q);
                return nullptr;
            }
        }
        CK(cudaStreamSynchronize(b->s0));
        b->kv_pin = true;
    }
    b->prof = getenv("DORY_PROF") != nullptr;
    if (!getenv("DORY_NOCAL")) dory_calibrate(b);
    return b;
}

extern "C" void dory_prof_report(Dory* b) {
    std::vector<std::pair<std::string, std::pair<double, int>>> v(b->prof_acc.begin(), b->prof_acc.end());
    std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
    double total = 0;
    for (auto& e : v) total += e.second.first;
    fprintf(stderr, "dory prof: total %.2f ms over the profiled blocks\n", total);
    for (auto& e : v) fprintf(stderr, "  %-16s %9.2f ms %5.1f%%  %6d calls  %.1f us each\n", e.first.c_str(), e.second.first, 100 * e.second.first / total, e.second.second, 1000 * e.second.first / e.second.second);
    b->prof_acc.clear();
}

// The C ABI version dory_native.zig checks at load: bump it with any change to the calls below.
extern "C" int dory_abi() { return 1; }

extern "C" int dory_seqs(Dory* b) { return b->nseq; }
extern "C" int dory_context(Dory* b) { return b->context; }
extern "C" int dory_loops(Dory* b) { return b->loops; }

// Device bytes one more sequence holds at this context: its sliding rings, and the full layers' rows (one granule when
// they are mapped as the sequence grows).
extern "C" uint64_t dory_seq_bytes(Dory* b) {
    return swa_seq_bytes(b) + (uint64_t)(b->kv.lazy && !b->kv_pin ? b->kv.tokens_per_gran : b->context) * kv_token_bytes(b);
}

// No drafter yet: proposals are empty.
extern "C" int dory_propose_seq(Dory*, int, int, int*) { return 0; }
extern "C" void dory_propose_multi(Dory*, int n, const int*, const int*, int*, int* got) { for (int i = 0; i < n; i++) got[i] = 0; }
extern "C" int dory_has_draft(Dory*) { return 0; }
extern "C" double dory_cost(Dory* b, int width) {
    if (width < 1 || width > VERIFY_MAX) return 0;
    return b->cost_ms[width];
}
extern "C" double dory_draft_multi_cost(Dory*) { return 0; }
extern "C" int dory_shared_costs(Dory* b, int* rows, double* ms, int cap) {
    int n = 0;
    for (auto& e : b->shared_ms) {
        if (n >= cap) break;
        rows[n] = e.first;
        ms[n] = e.second;
        n++;
    }
    return n;
}

extern "C" void dory_free(Dory* b) { (void)b; /* the process owns the model until it exits */ }

extern "C" void dory_reset_seq(Dory* b, int seq) {
    Dory::Seq& q = b->seqs[seq];
    const bool used = q.used;
    q = Dory::Seq();
    q.used = used;
}

extern "C" int dory_seq_alloc(Dory* b) {
    for (int i = 0; i < b->nseq; i++) {
        if (b->seqs[i].used) continue;
        dory_reset_seq(b, i);
        b->seqs[i].used = true;
        return i;
    }
    return -1;
}
extern "C" void dory_seq_release(Dory* b, int seq) {
    if (seq < 0 || seq >= b->nseq) return;
    b->seqs[seq].used = false;
    kv_trim(b, seq);
}

extern "C" int dory_len_seq(Dory* b, int seq) { return b->seqs[seq].len; }
extern "C" void dory_prompt_end(Dory*, int, int) {}

extern "C" int dory_forward_multi(Dory* b, int nseg, const int* seqs, const int* rows, const int* tokens, int cand_k) {
    SegIn sg[SEG_MAX];
    if (nseg < 1 || nseg > SEG_MAX) return -1;
    int at = 0;
    for (int i = 0; i < nseg; i++) {
        sg[i] = {seqs[i], rows[i], tokens + at};
        at += rows[i];
    }
    return run_blocks(b, sg, nseg, 1, cand_k);
}

extern "C" int dory_mask_rows(Dory* b, int n, const int* rows, const uint32_t* masks, int words, int k) {
    if (n < 1 || n > LOGIT_ROWS || words < 1 || words > VOCAB / 32 || k < 1) return -1;
    if (k > CAND) k = CAND;
    for (int i = 0; i < n; i++) if (rows[i] < 0 || rows[i] >= LOGIT_ROWS) return -2;
    CK(cudaMemcpyAsync(b->msel, rows, (size_t)n * 4, cudaMemcpyHostToDevice, b->s0));
    CK(cudaMemcpyAsync(b->mmask, masks, (size_t)n * words * 4, cudaMemcpyHostToDevice, b->s0));
    k_candidates_masked(b->s0, b->logits, b->msel, b->mmask, words, n, k, b->mvals, b->mids, b->csv, b->csi);
    std::vector<float> v((size_t)n * CAND);
    std::vector<int> id((size_t)n * CAND);
    CK(cudaMemcpyAsync(v.data(), b->mvals, v.size() * 4, cudaMemcpyDeviceToHost, b->s0));
    CK(cudaMemcpyAsync(id.data(), b->mids, id.size() * 4, cudaMemcpyDeviceToHost, b->s0));
    CK(cudaStreamSynchronize(b->s0));
    for (int i = 0; i < n; i++) {
        memcpy(&b->cvals_h[(size_t)rows[i] * CAND], &v[(size_t)i * CAND], (size_t)k * 4);
        memcpy(&b->cids_h[(size_t)rows[i] * CAND], &id[(size_t)i * CAND], (size_t)k * 4);
    }
    return 0;
}

extern "C" void dory_candidates(Dory* b, int row, int k, float* vals, int* ids) {
    if (k > CAND) k = CAND;
    memcpy(vals, &b->cvals_h[(size_t)row * CAND], (size_t)k * 4);
    memcpy(ids, &b->cids_h[(size_t)row * CAND], (size_t)k * 4);
}

// Keep the first n rows of the sequence's last record block (idempotent for the same n).
extern "C" void dory_keep_seq(Dory* b, int seq, int n) {
    Dory::Seq& q = b->seqs[seq];
    if (!q.last_record) return;
    if (n < 0) n = 0;
    if (n > q.last_rows) n = q.last_rows;
    q.last_rows = n;
    q.len = q.last_len + n;
}

// The prompt: chunks of TMAX rows. Afterwards row 0 of dory_candidates is the last prompt row.
extern "C" int dory_prefill_seq(Dory* b, int seq, const int* tokens, int n, int cand_k) {
    int done = 0;
    while (done < n) {
        int t = n - done < TMAX ? n - done : TMAX;
        int last = done + t == n;
        int rc = run_block(b, seq, tokens + done, t, 0, last ? cand_k : 0, 1);
        if (rc) return rc;
        done += t;
    }
    return 0;
}

extern "C" int dory_forward_seq(Dory* b, int seq, const int* tokens, int T, int record, int cand_k) { return run_block(b, seq, tokens, T, record, cand_k); }

// Test hooks: the logits of a row of the last forward (scaled, as sampled), and every row's (cand_k > 0, not last_only).
extern "C" void dory_logits_row(Dory* b, int row, float* out) {
    CK(cudaMemcpy(out, b->logits + (size_t)row * VOCAB, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
}
extern "C" int dory_prefill_all(Dory* b, int seq, const int* tokens, int n) {
    // one block, logits of every row (n <= LOGIT_ROWS): for the reference comparison
    return run_block(b, seq, tokens, n, 0, 1, 0);
}

static void dory_calibrate(Dory* b) {
    auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    int toks[BATCH_MAX];
    for (int i = 0; i < BATCH_MAX; i++) toks[i] = 1000 + (int)((i * 2654435761u + 12345u) % 90000u);
    for (int w = 1; w <= VERIFY_MAX; w++) {
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now();
            if (run_block(b, 0, toks, w, 1, 20)) break;
            double t1 = now();
            dory_keep_seq(b, 0, 0);
            if (rep >= 2 && t1 - t0 < best) best = t1 - t0;
        }
        b->cost_ms[w] = best;
    }
    if (b->nseq > 1) {
        for (int R : {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128}) {
            int nseg = R < b->nseq ? R : b->nseq;
            if (R > nseg * VERIFY_MAX) continue;
            int seqs[SEG_MAX], rows[SEG_MAX];
            for (int i = 0; i < nseg; i++) { seqs[i] = i; rows[i] = R / nseg + (i < R % nseg ? 1 : 0); }
            double best = 1e30;
            for (int rep = 0; rep < 5; rep++) {
                double t0 = now();
                if (dory_forward_multi(b, nseg, seqs, rows, toks, 64)) break;
                double t1 = now();
                for (int i = 0; i < nseg; i++) dory_keep_seq(b, seqs[i], 0);
                if (rep >= 2 && t1 - t0 < best) best = t1 - t0;
            }
            if (best < 1e29) b->shared_ms.push_back({R, best});
        }
    }
    for (int i = 0; i < b->nseq; i++) dory_reset_seq(b, i);
    fprintf(stderr, "dory: window ms");
    for (int w = 1; w <= VERIFY_MAX; w++) fprintf(stderr, " %d:%.2f", w, b->cost_ms[w]);
    if (!b->shared_ms.empty()) {
        fprintf(stderr, "; shared rows");
        for (auto& e : b->shared_ms) fprintf(stderr, " %d:%.2f", e.first, e.second);
    }
    fprintf(stderr, "\n");
}

// ---- prompt-cache snapshots: the caches after `at` prompt tokens, copied out and back ----

struct Snap {
    int at = 0, swa_from = 0;
    uint8_t* kv = nullptr;   // full layers: K rows [0, at), V rows; then sliding layers: K, V rows of positions [swa_from, at)
    size_t bytes = 0;
};

namespace {
int swa_from_of(int at) { return at > SWA - 1 ? at - (SWA - 1) : 0; }
}

extern "C" uint64_t dory_snap_bytes(Dory* b, int at) {
    return (uint64_t)at * kv_token_bytes(b) + (uint64_t)(at - swa_from_of(at)) * b->n_swa * 2 * b->kv_row;
}

namespace {
// Copy sliding-ring rows of positions [from, to) between a ring and a flat buffer (both ways).
void ring_copy(Dory* b, uint16_t* ring, int seq, uint8_t* flat, int from, int to, bool save) {
    int p = from;
    uint8_t* f = flat;
    while (p < to) {
        const int slot = p & (RING - 1);
        const int n = std::min(to - p, RING - slot);
        uint8_t* r = (uint8_t*)ring + ((size_t)seq * RING + slot) * b->kv_row;
        const size_t bytes = (size_t)n * b->kv_row;
        if (save) CK(cudaMemcpyAsync(f, r, bytes, cudaMemcpyDeviceToDevice, b->s0));
        else CK(cudaMemcpyAsync(r, f, bytes, cudaMemcpyDeviceToDevice, b->s0));
        f += bytes;
        p += n;
    }
}
}

extern "C" void* dory_snap_save_seq(Dory* b, int seq, int at) {
    Dory::Seq& q = b->seqs[seq];
    if (at != q.len || at <= 0) return nullptr;
    Snap* sn = new Snap();
    sn->at = at;
    sn->swa_from = swa_from_of(at);
    sn->bytes = dory_snap_bytes(b, at);
    if (cudaMalloc(&sn->kv, sn->bytes) != cudaSuccess) { cudaGetLastError(); delete sn; return nullptr; }
    uint8_t* dst = sn->kv;
    const size_t full = (size_t)at * b->kv_row, swa = (size_t)(at - sn->swa_from) * b->kv_row;
    const size_t row0 = (size_t)seq * b->kv_stride * b->kv_row;
    for (auto& l : b->kvs) {
        if (l.full) {
            CK(cudaMemcpyAsync(dst, (uint8_t*)l.k + row0, full, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync(dst + full, (uint8_t*)l.v + row0, full, cudaMemcpyDeviceToDevice, b->s0));
            dst += 2 * full;
        } else {
            ring_copy(b, l.k, seq, dst, sn->swa_from, at, true);
            ring_copy(b, l.v, seq, dst + swa, sn->swa_from, at, true);
            dst += 2 * swa;
        }
    }
    CK(cudaStreamSynchronize(b->s0));
    return sn;
}

extern "C" int dory_snap_restore_seq(Dory* b, int seq, void* handle) {
    Snap* sn = (Snap*)handle;
    if (!sn || sn->at > b->context) return -1;
    if (!kv_ensure(b, seq, sn->at)) return -1;
    Dory::Seq& q = b->seqs[seq];
    const int at = sn->at;
    const uint8_t* src = sn->kv;
    const size_t full = (size_t)at * b->kv_row, swa = (size_t)(at - sn->swa_from) * b->kv_row;
    const size_t row0 = (size_t)seq * b->kv_stride * b->kv_row;
    for (auto& l : b->kvs) {
        if (l.full) {
            CK(cudaMemcpyAsync((uint8_t*)l.k + row0, src, full, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync((uint8_t*)l.v + row0, src + full, full, cudaMemcpyDeviceToDevice, b->s0));
            src += 2 * full;
        } else {
            ring_copy(b, l.k, seq, (uint8_t*)src, sn->swa_from, at, false);
            ring_copy(b, l.v, seq, (uint8_t*)src + swa, sn->swa_from, at, false);
            src += 2 * swa;
        }
    }
    CK(cudaStreamSynchronize(b->s0));
    q.len = at;
    q.last_rows = 0;
    q.last_record = 0;
    return 0;
}

extern "C" void dory_snap_drop(Dory*, void* handle) {
    Snap* sn = (Snap*)handle;
    if (!sn) return;
    cudaFree(sn->kv);
    delete sn;
}
