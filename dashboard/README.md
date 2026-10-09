# tfmon

Zero-dependency Node.js dashboard for the TensorFold engine. Polls
`GET /health` (default `http://localhost:8888/health`, 1 Hz) and renders an
htop-style view: throughput sparkline, decode/spec stats, cache gauge,
stream utilization, lifetime counters. Below the three panels are
full-width `Streams`, `Lifetime` and `Cost` boxes, then the key hints.
Requires Node ≥ 18 (uses global `fetch`).

## Layout

```
┌ TensorFold: Qwen3.8-27B ───────────────────────────────────────────OK · idle┐
│ ┌─  Throughput ────────┐ ┌─  Spec Decode ───────┐ ┌─  Cache ───────────────┐│
│ │             ▁▁▁▁▁▁▁  │ │  Acc   0% ░░░░░░░░░░ │ │  Hit   0.0% ░░░░░░░░░░ ││
│ │  Decode 0.0 tok/s    │ │  0.0 tok/round       │ │  All  97.3% ██████████ ││
│ │  Prefill 0.0 tok/s   │ │  Round 0 ms          │ │  Spend/sec ₹0.000000   ││
│ │  Avg in 59.3K tok/.. │ │  Stream util 0%      │ │  9.76M / 10.02M tok    ││
│ └──────────────────────┘ └──────────────────────┘ └────────────────────────┘│
│ ┌─  Streams ──────────────────────────────────────────────────────────────┐ │
│ │ ░░░░░░░░░ 0/9   Decoding 0   Prefilling 0   Req 0.0/s   Running 0       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─  Lifetime ─────────────────────────────────────────────────────────────┐ │
│ │ Requests 169.0   Prompt 10.02M   Completion 88.4K   Ctx 262144          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─  Cost ─────────────────────────────────────────────────────────────────┐ │
│ │ Total ₹5785.13   Session ₹323.79   Earlier ₹5461.34                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│  q quit · h help · g graphs · t theme                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

`Hit` and `All` are the same gauge over two horizons: `Hit` = Δcached ÷
Δprompt for the last poll window (jumpy, that is the point), `All` = cached
÷ prompt over the engine's whole lifetime — it barely moves and tells you
what the prefix cache has actually been worth since the server booted.

An idle window measures 0 tok/s, which looks like a stalled engine, so
`Decode` and `Prefill` keep showing the last rate they actually measured for
up to 60 s afterwards — dimmed and marked `*` — before falling back to 0.0.

Colours follow the Lifetime box: `Decode` is green like Completion, `Prefill`
cyan like Prompt, labels and units are dim. The percent gauges (`Hit`, `All`,
`Acc`) right-align their number to the width of `100.0`, so the bar keeps one
start column and one length instead of sliding and resizing every time a
digit appears.

Labels inside the boxes are capitalised (`Decode`, `Spend/sec`, `Stream util`).
The `--cli` single-line fields stay lowercase (`decode`, `cache`, `cost`) so
they remain easy to grep, and the key hints keep the literal keys
(`q quit · h help`).

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
| `--theme NAME`  | `auto` \| `dark` \| `light` (default `auto`, see Themes) |
| `--model NAME`  | header label override                          |
| `--price FILE`  | pricing JSON, $/1M tokens (default: `pricing.json` next to `tfmon.js`) |
| `--state FILE`  | cost accumulator JSON (default: `cost.json` next to `tfmon.js`) |
| `--cli`         | plain-text mode, no TUI                        |
| `--help`        | usage                                         |

TUI keys: `q` quit · `h` help · `g` toggle sparkline · `t` cycle theme.
If the engine goes silent for 3× the poll interval, a red
`engine unreachable — last ok Ns ago` banner is shown (the last known
layout stays on screen, dimmed — never blank).

Piping `--cli` output is safe: `tfmon` exits quietly (after saving the cost
state) as soon as the reader closes, e.g. `tfmon --cli | head -20`.

### Themes

`dark` is for a dark terminal background, `light` paints near-black text for
a light one — on the wrong background either one is unreadable, so the
default is `auto`: before the first frame tfmon asks the terminal for its
background colour (OSC 11 — answered by xterm, GNOME Terminal, kitty, alacritty,
iTerm, Windows Terminal, …), computes its luminance and picks the theme.
No answer within 250 ms means `dark`; keystrokes typed while waiting are
not lost. `--theme dark|light` skips the question. `t` still switches by
hand — in `light` the key hints say so, since that theme only makes sense
on a light background. `--cli` never probes (its stdin is not in raw mode)
and starts in `dark`.

## Cost tracking

Cost is derived from the cumulative `/health` token counters:
window cost = (Δprompt − Δcached) × input + Δcached × cache +
Δcompletion × output. The Cache panel shows `Spend/sec` (window
cost normalized to 1 s, so it is a true rate at any poll interval).

The `Cost` box above the key hints (and `--cli`) prints the
**running total** cost, which persists across tfmon runs in a small state
file (`--state FILE`, default `cost.json` next to `tfmon.js`, shape
`{ cost, engineCost, ts }`):

```
│ ┌─  Cost ─────────────────────────────────────────────────────────────────┐ │
│ │ Total ₹5594.77   Session ₹133.43   Earlier ₹5461.34                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
```

- `Total` — running total: `Earlier + Session`;
- `Session` — the cost of the **live engine process** only, recomputed from
  its lifetime counters, so it restarts at ₹0 with the server (`-` before
  the first successful poll);
- `Earlier` — what previous engine sessions accrued (₹0 until an engine
  session has ended — it is folded in when a restart is detected);
- the `@ …` price basis (per 1M tokens, from `--price FILE`) is appended
  only when it fits whole — on narrower terminals it is left out rather
  than truncated mid-rate.

The running total covers:

- cost accrued **while tfmon was down** — on the first sample after a
  start it is picked up from the engine's lifetime counters;
- **engine restarts** — when the lifetime counters reset below the saved
  value, the finished session is folded into `Earlier` and the new
  session accumulates on top of it.

If pricing or currency changes between runs, the carried total keeps the
old price basis — delete the state file to reset the total.
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
