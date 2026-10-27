# Prefill and interleave, 2026-10-27, same image, TENSORFOLD_PREFILL_SHARE=0.85
# prefill: one request at a time, unique prompt, max_tokens=1, thinking off
| prompt tokens | TTFT s | prefill tok/s |
|---:|---:|---:|
| 1997 | 0.85 | 2362 |
| 7877 | 3.16 | 2494 |
| 31397 | 14.57 | 2156 |
| 98037 | 67.93 | 1443 |

# interleave: 1 decoding stream (thinking off) + a 100000-token prompt sent 5 s later
long prompt 98037 tokens: TTFT 95.1s (1031 tok/s)
decoding stream during that prefill: ~1017 tokens = 10.7 tok/s
decoding stream: 3000 tokens in 125.2s (whole stream 24.0 tok/s)
