# flash-4 Long-Term / Quality-Sensitive Testing TODOs

Items moved here carry a potential quality tradeoff and require extended long-context (120K–248K needle, coding, math) evaluation before ever being considered for default use.

- [ ] **Asymmetric KV Cache (`K=MXFP8` / `V=Q6` or `V=NVFP4`, ExLlamaV3 Pattern)**
  - **Rationale:** Keys enter `exp(Q·K^T)` softmax (precision-sensitive), while Values are averaged linearly (more tolerant of 4–6 bits).
  - **Goal:** Keep `K` in MXFP8 (`E4M3 + E8M0/32`) and test `V` at 6-bit (`~4.4 GiB/262K`, 9 streams, `<0.001` PPL) or 4-bit NVFP4 (`~3.8 GiB/262K`, 10–11 streams, ~25% fewer 248K `attn_core` reads).
  - **Validation gate:** Must pass 120K & 248K needle retrieval, multi-turn coding/math benchmarks, and 5/5 `draft_on == draft_off` exactness.

- [ ] **INT8 (7-bit Mantissa, Group-32/64) Hyper-Connections Weights (`W_down` + `W_up`)**
  - **Rationale:** 96 HCs read `1.27 GB` of BF16 weights on every decode round. Earlier 8-bit MXFP8 HC (`E4M3`, 3-bit mantissa) saved **-4.5 ms/round (+8–18% N=1 decode)**, proving bandwidth is the lever, but 3-bit mantissa across 96 residual-mixing steps broke 120K needle retrieval.
  - **Goal:** Prototype symmetric INT8 (7-bit mantissa = 16× finer precision than E4M3) with per-channel or group-32 FP32 scales, cutting `1.27 GB → 0.64 GB` (est. `-3.0 to -4.0 ms/round` / **+8–10% N=1 decode**).
  - **Validation gate:** **Must pass 120K & 248K needle retrieval (`FALCON-9472-OMEGA`)**, 5/5 greedy & T=0.7 `draft_on == draft_off` exactness, and zero prefill regression.
