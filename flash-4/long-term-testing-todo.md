# flash-4 Long-Term / Quality-Sensitive Testing TODOs

Items moved here carry a potential quality tradeoff and require extended long-context (120K–248K needle, coding, math) evaluation before ever being considered for default use.

- [ ] **Asymmetric KV Cache (`K=MXFP8` / `V=Q6` or `V=NVFP4`, ExLlamaV3 Pattern)**
  - **Rationale:** Keys enter `exp(Q·K^T)` softmax (precision-sensitive), while Values are averaged linearly (more tolerant of 4–6 bits).
  - **Goal:** Keep `K` in MXFP8 (`E4M3 + E8M0/32`) and test `V` at 6-bit (`~4.4 GiB/262K`, 9 streams, `<0.001` PPL) or 4-bit NVFP4 (`~3.8 GiB/262K`, 10–11 streams, ~25% fewer 248K `attn_core` reads).
  - **Validation gate:** Must pass 120K & 248K needle retrieval, multi-turn coding/math benchmarks, and 5/5 `draft_on == draft_off` exactness.
