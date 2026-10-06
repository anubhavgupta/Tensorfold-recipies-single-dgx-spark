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
| `--price FILE`  | pricing JSON, $/1M tokens (default: `pricing.json` next to `tfmon.js`) |
| `--cli`         | plain-text mode, no TUI                        |
| `--help`        | usage                                         |

TUI keys: `q` quit · `h` help · `g` toggle sparkline · `t` cycle theme.
If the engine goes silent for 3× the poll interval, a red
`engine unreachable — last ok Ns ago` banner is shown (the last known
layout stays on screen, dimmed — never blank).

## Cost tracking

Cost is derived from the cumulative `/health` token counters:
window cost = (Δprompt − Δcached) × input + Δcached × cache +
Δcompletion × output. The Cache panel shows `spend/sec` (window
cost normalized to 1 s, so it is a true rate at any poll interval);
the footer lifetime line shows the total `cost` since engine start,
and `--cli` prints both.
Pricing is read from a JSON file (`--price FILE`, default
`pricing.json` next to `tfmon.js`):

```json
{
  "model": "Qwen3.8-27B",
  "currency": "$",
  "input_per_mtok": 2.0,
  "output_per_mtok": 10.0,
  "cache_read_per_mtok": 0.2
}
```

Values are per 1M tokens: `input` = uncached prompt tokens,
`cache_read` = cached prompt tokens (a.k.a. cache-hit tokens),
`output` = completion tokens. `model` (optional) is shown in the
header; `currency` (optional, default `$`) is the symbol rendered
before cost values (e.g. `"€"` or `"₹"`). Missing file or
missing/invalid keys fall back to the built-in defaults
($2 / $10 / $0.20 per 1M tokens).

## Install (optional)

```sh
cd dashboard && npm link     # exposes `tfmon`
```

Or just `node dashboard/tfmon.js` from anywhere — there are no
dependencies and no build step.
