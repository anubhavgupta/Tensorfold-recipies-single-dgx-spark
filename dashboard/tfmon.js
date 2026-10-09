#!/usr/bin/env node
// tfmon — TensorFold engine dashboard for Node.js (zero dependencies).
//
// Polls GET /health (default http://localhost:8888/health, 1 Hz) and renders
// an htop-style dashboard: throughput sparkline, decode/spec stats, cache
// gauge, stream utilization, lifetime counters, and a running cost total
// persisted across runs in cost.json. Single-server scope.
//
// Usage:
//   tfmon [--url URL] [--interval MS] [--theme dark|light] [--model NAME] [--cli] [--help]
//
// Keys (TUI mode): q quit · h help · g graphs · t theme

import process from 'node:process';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_PRICE_FILE = join(dirname(fileURLToPath(import.meta.url)), 'pricing.json');
const DEFAULT_STATE_FILE = join(dirname(fileURLToPath(import.meta.url)), 'cost.json');

// ---------------------------------------------------------------- config

const DEFAULTS = {
  url: 'http://localhost:8888/health',
  intervalMs: 1000,
  cli: false,
  theme: 'dark',
  model: '',
  price: DEFAULT_PRICE_FILE,
  state: DEFAULT_STATE_FILE,
};

const HELP = `tfmon — TensorFold engine dashboard

  tfmon [options]

Options:
  --url URL        health endpoint        (default: ${DEFAULTS.url})
  --interval MS    poll period, ms        (default: 1000)
  --theme NAME     dark|light             (default: dark)
  --model NAME     label override for the header
  --price FILE     pricing JSON, $/1M tokens (default: pricing.json next to tfmon.js)
  --state FILE     cost accumulator JSON, persists across runs (default: cost.json next to tfmon.js)
  --cli            plain-text mode: print one table per poll, no TUI
  --help           show this help

Keys (TUI mode):  q quit · h help · g graphs · t theme
`;

function parseArgs(argv) {
  const cfg = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    switch (a) {
      case '--url': cfg.url = val(); break;
      case '--interval': {
        const ms = Number(val());
        if (!Number.isFinite(ms) || ms < 50) throw new Error('--interval must be ms >= 50');
        cfg.intervalMs = Math.round(ms);
        break;
      }
      case '--theme': {
        const v = val();
        if (v !== 'dark' && v !== 'light') throw new Error('--theme must be dark|light');
        cfg.theme = v;
        break;
      }
      case '--model': cfg.model = val(); break;
      case '--price': cfg.price = val(); break;
      case '--state': cfg.state = val(); break;
      case '--cli': cfg.cli = true; break;
      case '--help':
        process.stdout.write(HELP);
        process.exit(0);
        break;
      default:
        throw new Error(`unknown flag: ${a} (see --help)`);
    }
  }
  return cfg;
}

// ---------------------------------------------------------------- themes

const THEMES = {
  dark: { fg: 252, dim: 241, accent: 45, ok: 46, warn: 214, err: 196, graph: 51 },
  light: { fg: 234, dim: 247, accent: 21, ok: 28, warn: 166, err: 160, graph: 21 },
};
const R = '\x1b[0m';
const paint = (s, code) => (code == null ? s : `\x1b[38;5;${code}m${s}${R}`);

// visible length, ignoring ANSI codes
const vlen = (s) => s.replace(/\x1b\[[0-9;]*m/g, '').length;
const pad = (s, w) => s + ' '.repeat(Math.max(0, w - vlen(s)));
// truncate to w visible chars, never splitting an ANSI escape
function trunc(s, w) {
  if (vlen(s) <= w) return s;
  let vis = 0, i = 0;
  while (i < s.length && vis < w - 2) {
    if (s[i] === '\x1b' && s[i + 1] === '[') {
      const m = /^\x1b\[[0-9;?]*[a-zA-Z]/.exec(s.slice(i));
      if (m) { i += m[0].length; continue; }
    }
    vis++; i++;
  }
  return s.slice(0, i) + '..';
}

// ------------------------------------------------------------- derivation

const EPS = 1e-9;
const clamp01 = (x) => Math.min(1, Math.max(0, x));

function fmt(n, digits = 1) {
  if (!Number.isFinite(n)) return '-';
  const a = Math.abs(n);
  if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (a >= 1e4) return (n / 1e3).toFixed(1) + 'K';
  return n.toFixed(digits);
}

// USD per 1M tokens, formatted with the configured currency symbol
function fmtUsd(n) {
  if (!Number.isFinite(n)) return '-';
  return `${P.currency}${n.toFixed(6)}`;
}

// cost totals for the cost row: 6 decimals only make sense for tiny rates
function fmtCost(n) {
  if (!Number.isFinite(n)) return '-';
  const a = Math.abs(n);
  if (a >= 1e6) return `${P.currency}${fmt(n)}`;
  return `${P.currency}${a >= 1 ? n.toFixed(2) : n.toFixed(4)}`;
}

// Window diff of two samples { t, h }. Lifetime/util always from cur.
function diff(prev, cur) {
  const out = {
    hasWindow: false, dt: 0,
    outTps: 0, prefillTps: 0, cacheHit: 0,
    accRate: 0, tokPerRound: 0, roundMs: 0,
    reqRate: 0, util: 0, avgCtx: 0, avgOut: 0, winCost: 0, spentPerSec: 0,
  };
  const ch = cur.h;
  if (ch.requestsTotal > 0) {
    out.avgCtx = ch.promptTotal / ch.requestsTotal;
    out.avgOut = ch.completionTotal / ch.requestsTotal;
  }
  if (ch.streams.max > 0) {
    out.util = clamp01((ch.streams.decoding + ch.streams.prefilling) / ch.streams.max);
  }
  if (!prev || !prev.t) return out;
  const dt = cur.t - prev.t;
  if (!(dt > 0)) return out;
  const ph = prev.h;
  const d = (a, b) => Math.max(0, b - a); // counters are monotonic; be defensive
  const dComp = d(ph.completionTotal, ch.completionTotal);
  const dPrompt = d(ph.promptTotal, ch.promptTotal);
  const dCached = d(ph.cachedTotal, ch.cachedTotal);
  const dPrefillS = d(ph.prefillS, ch.prefillS);
  const dDecodeS = d(ph.decodeS, ch.decodeS);
  const dRounds = d(ph.rounds, ch.rounds);
  const dDrafted = d(ph.drafted, ch.drafted);
  const dAccepted = d(ph.accepted, ch.accepted);
  const dReq = d(ph.requestsTotal, ch.requestsTotal);
  out.hasWindow = true;
  out.dt = dt;
  out.outTps = dComp / dt;
  out.prefillTps = Math.max(0, dPrompt - dCached) / Math.max(EPS, dPrefillS);
  out.cacheHit = clamp01(dCached / Math.max(EPS, dPrompt));
  out.accRate = dDrafted > 0 ? clamp01(dAccepted / dDrafted) : 0;
  out.tokPerRound = dAccepted / Math.max(EPS, dRounds);
  out.roundMs = dRounds > 0 ? (dDecodeS / dRounds) * 1000 : 0;
  out.reqRate = dReq / dt;
  out.winCost = costOf(dPrompt - dCached, dCached, dComp);
  out.spentPerSec = dt > 0 ? out.winCost / dt : 0;
  return out;
}

// pricing: USD per 1M tokens { input (uncached prompt), output, cache (cached prompt) }
function loadPricing(path) {
  const defs = { input: 2, output: 10, cache: 0.2, model: '', currency: '$' };
  try {
    const j = JSON.parse(readFileSync(path, 'utf8'));
    const num = (v) => (Number.isFinite(+v) ? +v : null);
    const out = { ...defs };
    out.currency = typeof j.currency === 'string' && j.currency ? j.currency : '$';
    const a = num(j.input_per_mtok);
    if (a != null) out.input = a;
    const b = num(j.output_per_mtok);
    if (b != null) out.output = b;
    const c = num(j.cache_read_per_mtok ?? j.cache_per_mtok);
    if (c != null) out.cache = c;
    out.model = typeof j.model === 'string' ? j.model : '';
    return out;
  } catch (e) {
    process.stderr.write(`tfmon: pricing file ${path}: ${e.message} — using built-in defaults ($2 / $10 / $0.20 per 1M tok)\n`);
    return defs;
  }
}
const costOf = (uncachedIn, cachedIn, outTok) =>
  (Math.max(0, uncachedIn) * P.input + Math.max(0, cachedIn) * P.cache + Math.max(0, outTok) * P.output) / 1e6;
// engine-lifetime cost from a health sample
const lifetimeCost = (h) => costOf(h.promptTotal - h.cachedTotal, h.cachedTotal, h.completionTotal);

// ------------------------------------------- persisted cost accumulator
// cost.json holds { cost, engineCost, ts }. `cost` is the running total as of
// the last save (inclusive of the engine session then running); `engineCost`
// is that engine session's lifetime cost at save time. On the first sample of
// a run the total resolves against the engine's lifetime cost: if it grew,
// cost accrued while tfmon was down is added; if the engine restarted
// (lifetime counters reset below the saved value), the new session's cost is
// added on top of the previous total.
function loadCostState(path) {
  const defs = { cost: 0, engineCost: 0 };
  try {
    const j = JSON.parse(readFileSync(path, 'utf8'));
    const num = (v) => (Number.isFinite(+v) && +v >= 0 ? +v : 0);
    return { cost: num(j.cost), engineCost: num(j.engineCost) };
  } catch {
    return defs;
  }
}
let lastStateSave = 0;
function saveCostState(force = false) {
  const now = Date.now() / 1000;
  if (!force && now - lastStateSave < 10) return;
  lastStateSave = now;
  try {
    writeFileSync(cfg.state, JSON.stringify({
      cost: state.costAccrual,
      engineCost: state.lastEngineCost,
      model: cfg.model || P.model || '',
      ts: now,
    }, null, 2) + '\n');
  } catch { /* state file is best-effort; never crash the dashboard */ }
}

// ------------------------------------------------------------------ health

function parseHealth(json) {
  if (typeof json !== 'object' || json === null || !('ok' in json)) return null;
  const s = json.streams || {};
  const num = (v, dflt = 0) => (Number.isFinite(+v) ? +v : dflt);
  return {
    ok: !!json.ok,
    busy: !!json.busy,
    requestsRunning: num(json.requests_running),
    requestsTotal: num(json.requests_total),
    promptTotal: num(json.prompt_tokens_total),
    completionTotal: num(json.completion_tokens_total),
    prefillS: num(json.prefill_seconds_total),
    decodeS: num(json.decode_seconds_total),
    cachedTotal: num(json.cached_tokens_total),
    rounds: num(json.rounds_total),
    drafted: num(json.drafted_total),
    accepted: num(json.accepted_total),
    streams: {
      decoding: num(s.decoding),
      prefilling: num(s.prefilling),
      max: num(s.max, 8),
    },
    ctx: num(json.context_length),
  };
}

// ------------------------------------------------------------------ poller

let cfg;
try {
  cfg = parseArgs(process.argv.slice(2));
} catch (e) {
  process.stderr.write(`tfmon: ${e.message}\n\n${HELP}`);
  process.exit(2);
}

const P = loadPricing(cfg.price); // $/1M tokens: input (uncached prompt), output, cache

const state = {
  sample: null,        // { t: seconds, h }
  have: false,         // at least one successful sample
  error: '',           // last failure message
  lastSuccessT: 0,     // same clock as sample.t
  derived: null,       // last window diff
  histOut: [],         // capped ring, back = newest
  histPrefill: [],
  carried: 0,          // cost before the current engine session (loaded from state)
  lastEngineCost: 0,   // engine-lifetime cost at the last successful poll
  costAccrual: 0,      // carried + current session = running total
};
const HIST_MAX = 120;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// restore the running cost total from the previous run
const savedCostState = loadCostState(cfg.state);
state.carried = Math.max(0, savedCostState.cost - savedCostState.engineCost);
state.lastEngineCost = savedCostState.engineCost;
state.costAccrual = savedCostState.cost; // running total as of the last save

async function pollOnce() {
  const t0 = Date.now() / 1000;
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 3000);
    const res = await fetch(cfg.url, { signal: ac.signal, headers: { connection: 'close' } });
    clearTimeout(timer);
    if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
    let json;
    try {
      json = JSON.parse(await res.text());
    } catch {
      throw new Error('not JSON');
    }
    const h = parseHealth(json);
    if (!h) throw new Error('not a /health response (no "ok" field)');
    if (state.sample) {
      const d = diff(state.sample, { t: t0, h });
      state.derived = d;
      if (d.hasWindow) {
        state.histOut.push(d.outTps);
        state.histPrefill.push(d.prefillTps);
        if (state.histOut.length > HIST_MAX) state.histOut.shift();
        if (state.histPrefill.length > HIST_MAX) state.histPrefill.shift();
      }
    }
    state.sample = { t: t0, h };
    state.lastSuccessT = t0;
    state.error = '';
    const lNow = lifetimeCost(h);
    const reset = lNow < state.lastEngineCost; // engine restarted: lifetime counters reset
    if (!state.have) {
      // first sample this run: resolve the saved state against the current engine session
      state.carried = reset ? savedCostState.cost : Math.max(0, savedCostState.cost - savedCostState.engineCost);
    } else if (reset) {
      state.carried += state.lastEngineCost; // fold the finished engine session into the carried cost
    }
    state.lastEngineCost = lNow;
    state.costAccrual = state.carried + lNow;
    state.have = true;
    saveCostState();
  } catch (e) {
    state.error = e.name === 'AbortError' ? 'timeout after 3s'
      : (e?.cause?.message && e.cause.message !== e.message ? `${e.message}: ${e.cause.message}` : String(e?.message || e));
  }
}

// --------------------------------------------------------------- ui state

let uiStop = false;
let tuiMode = false;
let themeIdx = cfg.theme === 'light' ? 1 : 0;
let showGraphs = true;
let showHelp = false;
const theme = () => THEMES[themeIdx === 1 ? 'light' : 'dark'];

const BLOCKS = '▁▂▃▄▅▆▇█';
function sparkline(values, width, code) {
  const t = theme();
  if (!values.length) return paint(pad('', width), t.dim);
  const slice = values.slice(-width);
  const max = Math.max(EPS, ...slice);
  let s = '';
  for (const v of slice) s += BLOCKS[Math.min(7, Math.floor((v / max) * 8))];
  s = ' '.repeat(Math.max(0, width - slice.length)) + s;
  return paint(trunc(s, width), code);
}

function bar(frac, width) {
  frac = clamp01(frac);
  const filled = Math.round(frac * width);
  const s = '█'.repeat(filled) + '░'.repeat(width - filled);
  return paint(s, frac < 0.5 ? theme().err : frac < 0.8 ? theme().warn : theme().ok);
}

function box(title, lines, w) {
  const inner = w - 2;
  const out = [`┌─ ${title} ` + '─'.repeat(Math.max(0, inner - 3 - vlen(title))) + '┐'];
  for (const ln of lines) out.push(`│ ${pad(trunc(ln, inner - 2), inner - 2)} │`);
  out.push(`└${'─'.repeat(inner)}┘`);
  return out;
}

function frame() {
  const t = theme();
  const cols = process.stdout.columns || 80;
  const W = Math.min(Math.max(cols - 1, 62), 100);

  const h = state.sample?.h;
  const d = state.derived;
  const now = Date.now() / 1000;

  // ---- header + status / unreachable banner
  const iv = cfg.intervalMs / 1000;
  let status;
  let banner = '';
  const age = now - state.lastSuccessT; // huge while never succeeded
  if (state.error) {
    status = paint('STALE', t.err);
    banner = state.have
      ? `engine unreachable - last ok ${Math.round(age)}s ago`
      : `engine unreachable - ${state.error}`;
  } else if (!state.have) {
    status = paint('connecting...', t.dim);
  } else if (age > 3 * iv) {
    status = paint('STALE', t.err);
    banner = `engine unreachable - last ok ${Math.round(age)}s ago`;
  } else {
    status = h.busy ? paint('OK · busy', t.warn) : paint('OK · idle', t.ok);
  }
  let host = cfg.url;
  try { host = new URL(cfg.url).host; } catch { /* keep raw url */ }
  const title = cfg.model || P.model ? `TensorFold: ${cfg.model || P.model}` : `TensorFold @ ${host}`;
  const mid = '─'.repeat(Math.max(1, W - 2 - vlen(` ${title} `) - vlen(status)));
  const head = `┌${paint(` ${title} `, t.accent)}${paint(mid, t.dim)}${status}┐`;

  const lines = [head];
  if (banner) lines.push(`┤ ${paint(pad(trunc(banner, W - 4), W - 4), t.err)} ├`);

  // ---- three panels
  const pw = Math.floor((W - 5) / 3);
  const pwLast = W - 5 - pw * 2;
  const noData = paint(' waiting for data...', t.dim);
  const v = (s) => (s != null ? s : noData);

  const bTp = box(' Throughput', [
    v(state.histOut.length
      ? (showGraphs ? sparkline(state.histOut, Math.max(4, pw - 5), t.graph) : paint(' (graphs off - g)', t.dim))
      : noData),
    v(d && paint(` decode ${d.outTps.toFixed(1)} tok/s`, t.fg)),
    v(d && paint(` prefill ${fmt(d.prefillTps)} tok/s`, t.fg)),
    v(d && paint(` avg in ${fmt(d.avgCtx)} tok/req`, t.dim)),
  ], pw);

  const accPre = d ? ` acc ${(d.accRate * 100).toFixed(0)}% ` : '';
  const bSp = box(' Spec Decode', [
    v(d && paint(accPre, t.fg) + bar(d.accRate, Math.max(4, (pw - 4) - vlen(accPre)))),
    v(d && paint(` ${d.tokPerRound.toFixed(1)} tok/round`, t.fg)),
    v(d && paint(` round ${d.roundMs.toFixed(0)} ms`, t.fg)),
    v(d && paint(` stream util ${(d.util * 100).toFixed(0)}%`, t.dim)),
  ], pw);

  const lifeCost = h ? lifetimeCost(h) : 0;
  const hitPre = d ? ` hit ${(d.cacheHit * 100).toFixed(1)}% ` : '';
  const bCa = box(' Cache', [
    v(d && paint(hitPre, t.fg) + bar(d.cacheHit, Math.max(4, (pwLast - 4) - vlen(hitPre)))),
    v(d && paint(' spend/sec ', t.fg) + paint(fmtUsd(d.spentPerSec), t.warn)),
    v(h && paint(` ${fmt(h.cachedTotal)} / `, t.dim) + paint(fmt(h.promptTotal), t.accent) + paint(' tok', t.dim)),
  ], pwLast);

  lines.push('│ ' + bTp[0] + ' ' + bSp[0] + ' ' + bCa[0] + pad('', Math.max(0, W - 5 - vlen(bTp[0]) - vlen(bSp[0]) - vlen(bCa[0]))) + '│');
  lines.push('│ ' + bTp[1] + ' ' + bSp[1] + ' ' + bCa[1] + pad('', Math.max(0, W - 5 - vlen(bTp[1]) - vlen(bSp[1]) - vlen(bCa[1]))) + '│');
  lines.push('│ ' + bTp[2] + ' ' + bSp[2] + ' ' + bCa[2] + pad('', Math.max(0, W - 5 - vlen(bTp[2]) - vlen(bSp[2]) - vlen(bCa[2]))) + '│');
  const n = Math.max(bTp.length, bSp.length, bCa.length);
  for (let i = 3; i < n; i++) {
    const a = bTp[i] ?? pad('', pw);
    const b = bSp[i] ?? pad('', pw);
    const c = bCa[i] ?? pad('', pwLast);
    lines.push('│ ' + a + ' ' + b + ' ' + c + pad('', Math.max(0, W - 5 - vlen(a) - vlen(b) - vlen(c))) + '│');
  }

  // ---- footer: Streams / Lifetime / Cost boxes, then the key hints
  const s = h?.streams ?? { decoding: 0, prefilling: 0, max: 8 };
  const dec = Math.min(s.max, s.decoding);
  const pre = Math.min(Math.max(0, s.max - dec), s.prefilling);
  const free = Math.max(0, s.max - dec - pre);
  const sBar = paint('▓'.repeat(dec), t.ok) + paint('▒'.repeat(pre), t.accent) + '░'.repeat(free);
  // streams box: the bar is the live picture, the labels spell out the counts
  const streamsBox = box(' Streams', [
    sBar + paint(` ${dec + pre}/${s.max}`, t.fg)
    + paint('   Decoding ', t.dim) + paint(String(s.decoding), t.ok)
    + paint('   Prefilling ', t.dim) + paint(String(s.prefilling), t.accent)
    + (d ? paint('   Req ', t.dim) + paint(`${d.reqRate.toFixed(1)}/s`, t.fg) : '')
    + paint('   Running ', t.dim) + paint(String(h?.requestsRunning ?? 0), t.fg),
  ], W - 4);
  const cum = state.costAccrual;
  const life = h
    ? paint('Requests ', t.dim) + paint(fmt(h.requestsTotal), t.fg)
      + paint('   Prompt ', t.dim) + paint(fmt(h.promptTotal), t.accent)
      + paint('   Completion ', t.dim) + paint(fmt(h.completionTotal), t.ok)
      + paint('   Ctx ', t.dim) + paint(String(h.ctx), t.fg)
    : paint('- (no data yet)', t.dim);
  const lifeBox = box(' Lifetime', [life], W - 4);
  // cost box: running total, the live engine session's share, and what earlier sessions carried
  const rt = (v) => String(+v.toFixed(4)); // 192 / 19.2 / 0.2 without trailing zeros
  const costBase = paint('Total ', t.fg) + paint(fmtCost(cum), t.warn)
    + paint('   Session ', t.dim) + paint(h ? fmtCost(lifeCost) : '-', t.fg)
    + paint('   Earlier ', t.dim) + paint(fmtCost(state.carried), t.dim);
  // rates only when they fit whole - never truncate a price in half
  const rates = `   @ ${P.currency}${rt(P.input)} in/${P.currency}${rt(P.cache)} cache/${P.currency}${rt(P.output)} out /M`;
  // box(' Cost', .., W - 4) lines are W - 4 visible chars wide -> W with the frame's "│ .. │"
  const costBox = box(' Cost', [costBase + (W - 8 - vlen(costBase) >= vlen(rates) ? paint(rates, t.dim) : '')], W - 4);
  const keys = showHelp
    ? paint(' q quit · h hide help · g toggle graphs · t theme — rates are per poll window; cost is a running total (persisted)', t.dim)
    : paint(' q quit · h help · g graphs · t theme', t.dim);

  const dimOn = age > 3 * iv; // includes never-succeeded (lastSuccessT = 0)
  const dimmed = (l) => (dimOn ? paint(l, t.dim) : l);
  const foot = (s) => dimmed('│ ' + pad(trunc(s, W - 4), W - 4) + ' │');
  for (const l of streamsBox) lines.push(dimmed('│ ' + l + ' │'));
  for (const l of lifeBox) lines.push(dimmed('│ ' + l + ' │'));
  for (const l of costBox) lines.push(dimmed('│ ' + l + ' │'));
  lines.push(foot(keys));
  lines.push(`└${'─'.repeat(W - 2)}┘`);
  return lines.join('\n');
}

// ------------------------------------------------------------------ modes

let onPoll = null; // called after each successful/failed poll

async function pollLoop() {
  while (!uiStop) {
    const t0 = Date.now();
    await pollOnce();
    if (onPoll) onPoll();
    await sleep(Math.max(0, cfg.intervalMs - (Date.now() - t0)));
  }
}

function cliLine() {
  const h = state.sample?.h;
  const d = state.derived;
  const now = Date.now() / 1000;
  let st;
  if (!state.have) st = `CONNECT${state.error ? ` (${state.error})` : ''}`;
  else if (now - state.lastSuccessT > 3 * (cfg.intervalMs / 1000))
    st = `STALE ${state.error || 'no response'} (last ok ${Math.round(now - state.lastSuccessT)}s ago)`;
  else st = h.busy ? 'BUSY' : 'IDLE';
  const s = h?.streams;
  return [
    new Date().toTimeString().slice(0, 8),
    st,
    `decode ${d ? d.outTps.toFixed(1) : '-'} t/s`,
    `prefill ${d ? fmt(d.prefillTps) : '-'} t/s`,
    `cache ${d ? (d.cacheHit * 100).toFixed(1) + '%' : '-'}`,
    `acc ${d ? (d.accRate * 100).toFixed(0) + '%' : '-'}`,
    `round ${d ? d.roundMs.toFixed(0) + ' ms' : '-'}`,
    `streams ${s ? `${Math.min(s.max, s.decoding + s.prefilling)}/${s.max}` : '-'} (dec ${s?.decoding ?? 0} pre ${s?.prefilling ?? 0})`,
    `req/s ${d ? d.reqRate.toFixed(2) : '-'}`,
    `cost ${h || state.costAccrual > 0 ? fmtCost(state.costAccrual) + (state.carried > 0 && h ? ` (session ${fmtCost(lifetimeCost(h))})` : '') + (d ? ` (spend/sec ${fmtUsd(d.spentPerSec)})` : '') : '-'}`,
    `lifetime ${h ? `req ${fmt(h.requestsTotal)} prompt ${fmt(h.promptTotal)} comp ${fmt(h.completionTotal)}` : '-'}`,
  ].join('  ');
}

function restoreTerm() {
  try {
    if (tuiMode) {
      process.stdout.write('\x1b[?25h\x1b[0m\x1b[?1049l');
      if (process.stdin.isTTY) process.stdin.setRawMode(false);
    }
  } catch { /* terminal already gone */ }
}

function quit() {
  if (uiStop) return;
  uiStop = true;
  saveCostState(true);
  restoreTerm();
  process.exit(0);
}

process.on('exit', () => saveCostState(true));

// the reader is gone (`tfmon --cli | head`): stop quietly instead of crashing on EPIPE
process.stdout.on('error', (e) => {
  if (e.code === 'EPIPE') { quit(); return; } // quit() saves state, restores the term, exits 0
  try {
    process.stderr.write(`tfmon: stdout: ${e.message}\n`);
  } catch { /* stderr gone too */ }
  restoreTerm();
  process.exit(1);
});

function runTui() {
  tuiMode = true;
  const out = process.stdout;
  process.on('SIGINT', quit);
  process.on('SIGTERM', quit);
  process.on('exit', restoreTerm);
  out.write('\x1b[?1049h\x1b[2J\x1b[?25l');
  if (process.stdin.isTTY) {
    process.stdin.setRawMode(true);
    process.stdin.setEncoding('utf8');
    process.stdin.resume();
    process.stdin.on('data', (k) => {
      if (k === 'q' || k === '\x03') quit();
      else if (k === 'h') showHelp = !showHelp;
      else if (k === 'g') showGraphs = !showGraphs;
      else if (k === 't') themeIdx = 1 - themeIdx;
    });
    process.stdin.on('end', quit);
  }
  let lastDraw = 0;
  setInterval(() => {
    if (uiStop || Date.now() - lastDraw < 100) return; // 10 Hz redraw cap
    lastDraw = Date.now();
    out.write('\x1b[H' + frame().split('\n').map((l) => l + '\x1b[K').join('\n') + '\x1b[K');
  }, 50);
}

function runCli() {
  process.on('SIGINT', quit);
  process.on('SIGTERM', quit);
  const isTTY = process.stdout.isTTY;
  onPoll = () => {
    if (uiStop) return;
    if (isTTY) process.stdout.write('\x1b[2J\x1b[H' + frame() + '\n');
    else process.stdout.write(cliLine() + '\n');
  };
}

// -------------------------------------------------------------------- main

if (cfg.cli || !process.stdout.isTTY) runCli();
else runTui();
pollLoop(); // background poller; keeps the process alive
