# decode set=code new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 48.4 | 49.1 | 0.17 | 3.71 |
| 2 | 55.6 | 32.5 | 0.29 | 3.25 |
| 4 | 93.3 | 24.6 | 0.32 | 3.74 |
# decode set=mixed new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 39.4 | 39.9 | 0.17 | 3.07 |
| 2 | 50.1 | 26.7 | 0.25 | 2.81 |
| 4 | 61.4 | 17.6 | 0.30 | 2.70 |
# decode set=mixed new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 33.8 | 34.1 | 0.17 | 2.60 |
| 2 | 51.5 | 27.0 | 0.24 | 2.87 |
| 4 | 64.8 | 17.5 | 0.30 | 2.75 |
# decode set=code new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 45.7 | 46.3 | 0.17 | 3.56 |
| 4 | 78.5 | 20.2 | 0.31 | 3.11 |
# prefill: one request at a time, unique prompt, max_tokens=1, thinking off
| prompt tokens | TTFT s | prefill tok/s |
|---:|---:|---:|
| 1997 | 0.72 | 2762 |
| 7877 | 2.43 | 3243 |
| 31397 | 12.26 | 2561 |
| 98037 | 63.87 | 1535 |
# interleave: 1 decoding stream (thinking off) + a 100000-token prompt sent 5 s later
long prompt 98037 tokens: TTFT 78.3s (1252 tok/s)
decoding stream during that prefill: ~807 tokens = 10.3 tok/s
decoding stream: 3000 tokens in 119.7s (whole stream 25.1 tok/s)
# long: 4 simultaneous ~250000-token distinct prompts, 256 new tokens each, thinking off
req 0: prompt 245037 completion 256 ttft 302.8s decode 3.4 tok/s
req 1: prompt 245036 completion 256 ttft 622.5s decode 12.3 tok/s
req 2: prompt 245037 completion 256 ttft 930.1s decode 12.9 tok/s
req 3: prompt 245037 completion 256 ttft 1235.1s decode 39.0 tok/s
all 4 done in 1241.7s
