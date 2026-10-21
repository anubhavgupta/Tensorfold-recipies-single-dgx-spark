# decode set=code new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 72.4 | 74.0 | 0.17 | 7.01 |
| 2 | 77.0 | 42.3 | 0.25 | 5.82 |
| 4 | 96.1 | 28.9 | 0.43 | 4.72 |
# decode set=mixed new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 41.6 | 42.0 | 0.17 | 3.74 |
| 2 | 52.1 | 26.4 | 0.25 | 3.62 |
| 4 | 69.9 | 19.0 | 0.43 | 3.12 |
# decode set=mixed new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 41.6 | 42.1 | 0.17 | 3.85 |
| 2 | 51.3 | 26.8 | 0.25 | 3.78 |
| 4 | 74.5 | 20.5 | 0.42 | 3.52 |
# decode set=code new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 42.6 | 43.1 | 0.17 | 3.82 |
| 4 | 94.4 | 25.6 | 0.43 | 4.40 |
# prefill: one request at a time, unique prompt, max_tokens=1, thinking off
| prompt tokens | TTFT s | prefill tok/s |
|---:|---:|---:|
| 1997 | 3.34 | 597 |
| 7877 | 15.85 | 497 |
| 31397 | 116.00 | 271 |
| 98037 | 918.79 | 107 |
# interleave: 1 decoding stream (thinking off) + a 100000-token prompt sent 5 s later
long prompt 98037 tokens: TTFT 898.8s (109 tok/s)
decoding stream during that prefill: ~32 tokens = 0.0 tok/s
decoding stream: 3000 tokens in 950.4s (whole stream 3.2 tok/s)
# long: 1 simultaneous ~32000-token distinct prompts, 512 new tokens each, thinking off
req 0: prompt 31397 completion 512 ttft 120.2s decode 30.0 tok/s
all 1 done in 137.2s
