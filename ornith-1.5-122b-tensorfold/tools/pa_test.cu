// Standalone check of tf_q35_attn_prompt against the stock tf_q35_attn_partial + attn_merge on one layer's cache.
// Build: nvcc -O3 -arch=sm_121 -I<kernels/cuda> pa_test.cu -o pa_test ; run: ./pa_test [depth...]
#include "qwen35.cu"
#include "qwen35_prompt_attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float frand(unsigned& s) { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 32768.0f - 1.0f; }

int main(int argc, char** argv) {
    const int ROWS = atoi(getenv("PROWS") ? getenv("PROWS") : "128"), QH = 32, KVH = 2;
    std::vector<int> depths;
    for (int i = 1; i < argc; i++) depths.push_back(atoi(argv[i]));
    if (depths.empty()) depths = {0, 1000, 8000, 32000, 100000};
    Geo geo{3072, 16, 64, QH, KVH, 36};
    struct V { const char* name; void (*f)(Geo, const __nv_bfloat16*, const Seg*, const int*, const Ancestor*, int, const int*, float, int, float*, float*, float*); int rows, smem; };
    V vs[] = {{"prompt", tf_q35_attn_prompt, 8, tf_q35_pa::smem_bytes<8, 16, 2>()}};
    for (auto& v : vs) CK(cudaFuncSetAttribute(v.f, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem));
    for (int D : depths) {
        const int cap = D + ROWS;
        const size_t kvn = (size_t)2 * KVH * cap * HD;
        std::vector<__nv_bfloat16> hkv(kvn), hq((size_t)ROWS * QH * HD);
        unsigned s = 12345u + D;
        // keys with some structure so softmax is not flat: a shared direction plus noise
        for (size_t i = 0; i < kvn; i++) hkv[i] = __float2bfloat16(frand(s) * 2.0f);
        for (size_t i = 0; i < hq.size(); i++) hq[i] = __float2bfloat16(frand(s) * 2.0f);
        __nv_bfloat16 *dkv, *dq, *y0, *y1;
        CK(cudaMalloc(&dkv, kvn * 2)); CK(cudaMalloc(&dq, hq.size() * 2));
        CK(cudaMalloc(&y0, hq.size() * 2)); CK(cudaMalloc(&y1, hq.size() * 2));
        CK(cudaMemcpy(dkv, hkv.data(), kvn * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
        Seg sg{}; sg.kv = (unsigned long long)dkv; sg.cap = cap; sg.len = D; sg.row0 = 0; sg.rows = ROWS;
        Seg* dseg; CK(cudaMalloc(&dseg, sizeof(Seg))); CK(cudaMemcpy(dseg, &sg, sizeof(Seg), cudaMemcpyHostToDevice));
        std::vector<int> rs(ROWS, 0); std::vector<Ancestor> an(ROWS);
        for (int j = 0; j < ROWS; j++) an[j] = Ancestor{0ull, (unsigned)j, 1u};
        int* drs; Ancestor* dan;
        CK(cudaMalloc(&drs, ROWS * 4)); CK(cudaMalloc(&dan, ROWS * sizeof(Ancestor)));
        CK(cudaMemcpy(drs, rs.data(), ROWS * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dan, an.data(), ROWS * sizeof(Ancestor), cudaMemcpyHostToDevice));
        float *pm, *pl, *po;
        const size_t slots = (size_t)ROWS * QH * MAXCH;
        CK(cudaMalloc(&pm, slots * 4)); CK(cudaMalloc(&pl, slots * 4)); CK(cudaMalloc(&po, slots * HD * 4));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const float scale = 1.0f / 16.0f;
        auto stock = [&]() {
            tf_q35_attn_partial<<<dim3(MAXCH, KVH, ROWS), 32 * (QH / KVH)>>>(geo, dq, dseg, drs, dan, 0, nullptr, scale, pm, pl, po);
            tf_q35_attn_merge<<<dim3(QH, ROWS), 32>>>(geo, pm, pl, po, dseg, drs, dan, nullptr, y0);
        };
        const int splits = atoi(getenv("SPLITS") ? getenv("SPLITS") : "4");
        float ts;
        stock(); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for (int i = 0; i < 3; i++) stock(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ts, e0, e1);
        for (auto& v : vs) {
        float tm;
        auto mine = [&]() {
            v.f<<<dim3((ROWS + v.rows - 1) / v.rows, KVH, splits), v.rows * 32, v.smem>>>(geo, dq, dseg, drs, dan, 0, nullptr, scale, ROWS, pm, pl, po);
            tf_q35_attn_prompt_merge<<<dim3(QH, ROWS), 32>>>(geo, pm, pl, po, splits, nullptr, y1);
        };
        mine(); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for (int i = 0; i < 3; i++) mine(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&tm, e0, e1);
        CK(cudaGetLastError());
        std::vector<__nv_bfloat16> a(hq.size()), b(hq.size());
        CK(cudaMemcpy(a.data(), y0, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y1, b.size() * 2, cudaMemcpyDeviceToHost));
        double mxd = 0, sd = 0, sa = 0; size_t bad = 0;
        for (size_t i = 0; i < a.size(); i++) {
            const double x = __bfloat162float(a[i]), z = __bfloat162float(b[i]);
            if (!std::isfinite(z)) bad++;
            mxd = fmax(mxd, fabs(x - z)); sd += fabs(x - z); sa += fabs(x);
        }
        printf("%-7s depth %6d: stock %8.3f ms  prompt %7.3f ms  (x%.1f)  max|d| %.4g  mean|d|/mean|y| %.3g  nonfinite %zu\n",
               v.name, D, ts / 3, tm / 3, ts / tm, mxd, sd / a.size() / (sa / a.size()), sa / a.size() > 0 ? bad : bad);
        }
        cudaFree(dkv); cudaFree(dq); cudaFree(y0); cudaFree(y1); cudaFree(dseg); cudaFree(drs); cudaFree(dan);
        cudaFree(pm); cudaFree(pl); cudaFree(po);
    }
    return 0;
}
