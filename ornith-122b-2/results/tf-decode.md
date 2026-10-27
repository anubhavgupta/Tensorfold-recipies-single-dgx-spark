# Decode, 2026-10-27, tensorfold-ornith122-2:zig-5904439 (patches 0001-0005), PARALLEL=4, CONTEXT=262144, KV bf16
# decode set=code new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 63.4 | 64.6 | 0.17 | 5.28 |
| 2 | 72.6 | 38.5 | 0.25 | 5.36 |
| 4 | 90.4 | 25.5 | 0.42 | 4.95 |

# decode set=mixed new=512 temp=0.0 thinking=False distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 49.8 | 50.5 | 0.17 | 3.58 |
| 2 | 52.2 | 26.9 | 0.24 | 3.70 |
| 4 | 68.6 | 19.2 | 0.41 | 3.74 |

# decode set=mixed new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 46.0 | 46.6 | 0.16 | 3.63 |
| 2 | 48.2 | 24.4 | 0.24 | 3.49 |
| 4 | 64.1 | 17.2 | 0.41 | 3.62 |

# decode set=code new=512 temp=0.6 thinking=True distinct prompts, ignore_eos
| N | total tok/s | per-stream tok/s | TTFT s | accept len |
|---:|---:|---:|---:|---:|
| 1 | 45.5 | 46.1 | 0.16 | 3.41 |
| 2 | 54.1 | 27.5 | 0.24 | 4.02 |
| 4 | 76.6 | 19.8 | 0.41 | 4.19 |

# seq set=code new=512 temp=0.0 thinking=False: one stream, each prompt in turn, ignore_eos
per prompt tok/s/accept: 60.4/4.45 57.3/4.00 67.5/4.70 58.8/3.91 63.7/4.53 73.8/5.89 55.0/3.63 79.7/6.48
aggregate: 63.6 tok/s, accept len 4.54
# seq set=mixed new=512 temp=0.0 thinking=False: one stream, each prompt in turn, ignore_eos
per prompt tok/s/accept: 48.1/3.14 43.2/2.81 59.2/4.13 42.0/2.68 57.4/3.74 56.0/3.71 82.3/7.53 63.7/4.79
aggregate: 54.1 tok/s, accept len 3.69
