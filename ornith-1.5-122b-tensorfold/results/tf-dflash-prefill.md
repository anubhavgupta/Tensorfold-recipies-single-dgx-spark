exactness draft=true 2b0363c5bcced210a951cb1bc80f2bf3
exactness draft=false 2b0363c5bcced210a951cb1bc80f2bf3
ALL PASS (0 failures)
# prefill: one request at a time, unique prompt, max_tokens=1, thinking off
| prompt tokens | TTFT s | prefill tok/s |
|---:|---:|---:|
| 1996 | 1.36 | 1466 |
| 7877 | 5.17 | 1525 |
| 31397 | 22.75 | 1380 |
| 98037 | 92.25 | 1063 |
# decode set=code new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 65.2 | 66.6 | 0.18 | 6.40 |
| 2 | 79.6 | 41.7 | 0.26 | 6.02 |
| 4 | 102.0 | 29.8 | 0.44 | 5.15 |
# decode set=mixed new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 38.5 | 38.9 | 0.17 | 3.51 |
| 2 | 49.4 | 25.4 | 0.26 | 3.57 |
| 4 | 70.3 | 20.0 | 0.43 | 3.33 |
# decode set=mixed new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 42.0 | 42.5 | 0.17 | 3.94 |
| 4 | 71.4 | 20.5 | 0.43 | 3.54 |
# interleave: 1 decoding stream (thinking off) + a 100000-token prompt sent 5 s later
long prompt 98035 tokens: TTFT 92.1s (1065 tok/s)
decoding stream during that prefill: ~14 tokens = 0.1 tok/s
decoding stream: 3000 tokens in 138.9s (whole stream 21.6 tok/s)
# needle: 211986 prompt tokens in 289s (735 tok/s incl. reply), answer 'violet-harbor-2700-quartz', expected violet-harbor-2700-quartz
# long: 4 simultaneous ~250000-token distinct prompts, 256 new tokens each, thinking off
req 0: prompt 245036 completion 256 ttft 357.9s decode 0.2 tok/s
req 1: prompt 245037 completion 256 ttft 703.9s decode 0.3 tok/s
req 2: prompt 245037 completion 256 ttft 1044.0s decode 0.5 tok/s
req 3: prompt 245037 completion 256 ttft 1385.5s decode 1.7 tok/s
all 4 done in 1533.0s
# long: 1 simultaneous ~250000-token distinct prompts, 512 new tokens each, thinking off
req 0: prompt 245037 completion 512 ttft 348.6s decode 12.5 tok/s
all 1 done in 389.6s
# long: 2 simultaneous ~120000-token distinct prompts, 512 new tokens each, thinking off
req 0: prompt 117637 completion 512 ttft 119.5s decode 2.5 tok/s
req 1: prompt 117637 completion 512 ttft 239.2s decode 6.0 tok/s
all 2 done in 324.8s
