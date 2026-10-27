# Full context, 2026-10-27: bench.py long 4 250000 256 (greedy, thinking off), PREFILL_SHARE=0.85
# long: 4 simultaneous ~250000-token distinct prompts, 256 new tokens each, thinking off
req 0: prompt 245037 completion 256 ttft 318.5s decode 4.7 tok/s
req 1: prompt 245037 completion 256 ttft 614.3s decode 5.1 tok/s
req 2: prompt 245037 completion 256 ttft 919.2s decode 5.1 tok/s
req 3: prompt 245037 completion 256 ttft 1133.0s decode 51.8 tok/s
all 4 done in 1138.0s
