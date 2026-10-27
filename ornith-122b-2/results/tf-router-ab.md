# Router A/B, 2026-10-27: the recipe's server restarted with TF_QWEN_ROUTER=tile, then the default (warp), back to back.
# bench.py seq --set code / --set mixed (one stream, 8 prompts, 512 tokens, greedy, thinking off), then decode --set code 4 / --set mixed 4.
## router tile
aggregate: 60.1 tok/s, accept len 4.73
aggregate: 54.2 tok/s, accept len 4.00
| 4 | 83.2 | 25.1 | 0.42 | 4.75 |
| 4 | 66.6 | 19.0 | 0.41 | 3.64 |
## router warp
aggregate: 63.8 tok/s, accept len 4.73
aggregate: 54.6 tok/s, accept len 3.69
| 4 | 93.1 | 27.7 | 0.42 | 5.15 |
| 4 | 67.6 | 19.0 | 0.42 | 3.66 |
