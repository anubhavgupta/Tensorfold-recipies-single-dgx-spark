# tfmon

Zero-dependency Node.js dashboard for the TensorFold engine. Polls
`GET /health` (default `http://localhost:8888/health`, 1 Hz) and renders an
htop-style view: throughput sparkline, decode/spec stats, cache gauge,
stream utilization, lifetime counters. Requires Node ≥ 18 (uses global
`fetch`).

## Usage

```sh
node tfmon.js                 # TUI (alternate screen, raw terminal)
node tfmon.js --cli           # plain text: one table (TTY) or one line (pipe) per poll
node tfmon.js --url http://host:port/health --interval 500 --model my-engine
```

| Flag            | Meaning                                        |
|-----------------|------------------------------------------------|
| `--url URL`     | health endpoint (default `http://localhost:8888/health`) |
| `--interval MS` | poll period, ≥ 50 ms (default 1000)            |
| `--theme NAME`  | `dark` \| `light` (default `dark`)             |
| `--model NAME`  | header label override                          |
| `--cli`         | plain-text mode, no TUI                        |
| `--help`        | usage                                         |

TUI keys: `q` quit · `h` help · `g` toggle sparkline · `t` cycle theme.
If the engine goes silent for 3× the poll interval, a red
`engine unreachable — last ok Ns ago` banner is shown (the last known
layout stays on screen, dimmed — never blank).

## Install (optional)

```sh
cd dashboard && npm link     # exposes `tfmon`
```

Or just `node dashboard/tfmon.js` from anywhere — there are no
dependencies and no build step.
