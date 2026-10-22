#!/usr/bin/env python3
"""Benchmarks for the Ornith-1.5-122B SGLang recipe (stdlib only).

  bench.py decode [--set code|mixed] [--new 512] [--temp 0] [--think] N [N ...]
      N simultaneous streaming chat requests per run, distinct prompt prefixes (no prefix-cache hits), ignore_eos so
      every request emits exactly --new tokens. Prints total tok/s (all output tokens / wall), mean per-stream decode
      tok/s (after the first token), mean TTFT and the speculative accept length (tokens per verify step, from the
      server's Prometheus counters).
  bench.py seq [--set code|mixed] [--new 512] [--temp 0] [--think]
      One stream: every prompt of the set in turn (fixed text, so greedy runs repeat exactly). Prints per-prompt and
      aggregate decode tok/s (all tokens / all decode time) and accept length; steadier than decode 1 (one prompt).
  bench.py prefill [--new 1] L [L ...]
      One request at a time with a unique ~L-token prompt; prefill tok/s = server prompt_tokens / TTFT.
  bench.py long N TOKENS NEW
      N simultaneous ~TOKENS-token prompts (each distinct), NEW output tokens each: max concurrency at long context.
  bench.py interleave TOKENS NEW
      One stream decodes NEW tokens while a TOKENS-token prompt prefills: how much a long prefill stalls other streams.
Env: URL (http://127.0.0.1:8888), MODEL (Ornith-1.5-122B).
"""
import argparse, http.client, json, os, random, re, statistics, threading, time, urllib.parse, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "Ornith-1.5-122B")

CODE = [
    "Write a Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
    "Write a Go HTTP server with graceful shutdown, structured logging, rate limiting middleware and tests.",
    "Write a TypeScript React component for a sortable, filterable data table with pagination and hooks.",
    "Write a Rust command-line tool that recursively finds duplicate files by content hash, with clap and rayon.",
    "Write a C++17 thread pool with a work-stealing queue, futures for results, and a usage example.",
    "Write PostgreSQL SQL for an e-commerce schema (users, orders, items, payments) with indexes and 5 report queries.",
    "Write a robust Bash script that backs up directories with rsync, rotates old backups and logs failures.",
    "Write a Java Spring Boot REST controller, service and JPA repository for a todo app with validation.",
]
MIXED = [
    "Explain in detail how the immune system distinguishes self from non-self.",
    "Write an essay on the causes and consequences of the industrial revolution.",
    "Write a polite but firm email to a landlord about a heating failure that has lasted two weeks.",
    "Compare PostgreSQL and MongoDB for a startup's analytics workload: strengths, weaknesses, costs.",
    "Write a Python function that parses ISO-8601 durations, with tests.",
    "Write a Go program that computes word frequencies across files concurrently.",
    "Solve step by step: find all real x with x^4 - 5x^2 + 4 = 0, and verify each root.",
    "Prove by induction that the sum of the first n odd numbers is n^2, explaining each step.",
]


def metrics():
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^((?:sglang|tensorfold):[A-Za-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def stream(messages, max_tokens, temp, think, ignore_eos=True, res=None):
    u = urllib.parse.urlparse(URL)
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": temp, "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": think}}
    if temp == 0:
        body["top_k"] = 1
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]}")
    first = None; usage = None; text = []; stamps = []
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                now = time.time()
                if first is None:
                    first = now
                text.append(piece); stamps.append((now, len(piece)))
    t1 = time.time()
    out = dict(t0=t0, first=first or t1, end=t1, usage=usage or {}, text="".join(text), stamps=stamps)
    if res is not None:
        res.append(out)
    return out


def run_parallel(jobs):
    res, th = [], []
    for args in jobs:
        t = threading.Thread(target=stream, args=tuple(args) + (True, res)); t.start(); th.append(t)
    for t in th:
        t.join()
    return res


def spec_delta(m0, m1):
    """Accept length = output tokens / verify rounds over the run, from the server's counters (SGLang or TensorFold;
    TensorFold's rounds are per stream, as its per-request log line's rounds= are)."""
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    toks = d.get("sglang:generation_tokens_total") or d.get("tensorfold:generation_tokens_total")
    steps = d.get("sglang:spec_verify_calls_total") or d.get("sglang:spec_verify_ct_total") \
        or d.get("tensorfold:decode_rounds_total")
    return toks / steps if toks and steps else None


def cmd_decode(a):
    prompts = CODE if a.set == "code" else MIXED
    print(f"# decode set={a.set} new={a.new} temp={a.temp} thinking={a.think} distinct prompts, ignore_eos")
    print("| N | total tok/s | per-stream tok/s | TTFT s | accept len |\n|---:|---:|---:|---:|---:|")
    for n in a.N:
        tag = random.randrange(10**9)
        jobs = [([{"role": "user", "content": f"[run {tag} req {i}] {prompts[i % len(prompts)]}"}], a.new, a.temp, a.think)
                for i in range(n)]
        m0 = metrics(); t0 = time.time(); res = run_parallel(jobs); wall = time.time() - t0; m1 = metrics()
        toks = sum(r["usage"].get("completion_tokens", 0) for r in res)
        per = statistics.mean((r["usage"].get("completion_tokens", 1) - 1) / max(r["end"] - r["first"], 1e-6) for r in res)
        ttft = statistics.mean(r["first"] - r["t0"] for r in res)
        acc = spec_delta(m0, m1)
        print(f"| {n} | {toks / wall:.1f} | {per:.1f} | {ttft:.2f} | {f'{acc:.2f}' if acc else '-'} |", flush=True)


def cmd_seq(a):
    prompts = CODE if a.set == "code" else MIXED
    print(f"# seq set={a.set} new={a.new} temp={a.temp} thinking={a.think}: one stream, each prompt in turn, ignore_eos")
    toks = secs = 0; m0 = metrics(); rows = []
    for i, q in enumerate(prompts):
        m = metrics()
        r = stream([{"role": "user", "content": q}], a.new, a.temp, a.think)
        n = r["usage"].get("completion_tokens", 1) - 1; dt = max(r["end"] - r["first"], 1e-6)
        toks += n; secs += dt; acc = spec_delta(m, metrics())
        rows.append(f"{n / dt:.1f}" + (f"/{acc:.2f}" if acc else ""))
    acc = spec_delta(m0, metrics())
    print("per prompt tok/s/accept: " + " ".join(rows))
    print(f"aggregate: {toks / secs:.1f} tok/s, accept len {acc:.2f}" if acc else f"aggregate: {toks / secs:.1f} tok/s")


WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point home "
         "water room mother area money story fact month lot right study book eye job word business issue side kind "
         "head house service friend father power hour game line end member law car city community name president").split()


def long_prompt(n_tokens, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(n_tokens * 0.98))]   # ~1 token per common word with its space
    return f"[doc {seed}] " + " ".join(words) + "\n\nIn one sentence, which word appears most often above?"


def cmd_prefill(a):
    print(f"# prefill: one request at a time, unique prompt, max_tokens={a.new}, thinking off")
    print("| prompt tokens | TTFT s | prefill tok/s |\n|---:|---:|---:|")
    for L in a.L:
        r = stream([{"role": "user", "content": long_prompt(L, random.randrange(10**9))}], a.new, 0, False)
        pt = r["usage"].get("prompt_tokens", 0); ttft = r["first"] - r["t0"]
        print(f"| {pt} | {ttft:.2f} | {pt / ttft:.0f} |", flush=True)


def cmd_long(a):
    print(f"# long: {a.N} simultaneous ~{a.TOKENS}-token distinct prompts, {a.NEW} new tokens each, thinking off")
    jobs = [([{"role": "user", "content": long_prompt(a.TOKENS, random.randrange(10**9))}], a.NEW, 0, False)
            for _ in range(a.N)]
    t0 = time.time(); res = run_parallel(jobs); wall = time.time() - t0
    for i, r in enumerate(sorted(res, key=lambda r: r["first"])):
        u = r["usage"]; dec = (u.get("completion_tokens", 1) - 1) / max(r["end"] - r["first"], 1e-6)
        print(f"req {i}: prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')} "
              f"ttft {r['first'] - r['t0']:.1f}s decode {dec:.1f} tok/s")
    print(f"all {len(res)} done in {wall:.1f}s")


def cmd_interleave(a):
    """One stream decodes while a long prompt prefills: decode speed during the prefill, and the prefill's TTFT."""
    print(f"# interleave: 1 decoding stream (thinking off) + a {a.TOKENS}-token prompt sent 5 s later")
    res = []
    t = threading.Thread(target=stream, args=([{"role": "user", "content": f"[{random.randrange(10**9)}] " + CODE[0]}],
                                              a.NEW, 0, False, True, res))
    t.start(); time.sleep(5)
    t1 = time.time()
    r = stream([{"role": "user", "content": long_prompt(a.TOKENS, random.randrange(10**9))}], 1, 0, False)
    t2 = r["first"]
    t.join()
    d = res[0]
    chars = sum(n for _, n in d["stamps"]) or 1
    win = sum(n for ts, n in d["stamps"] if t1 <= ts <= t2)
    win_tok = d["usage"].get("completion_tokens", 0) * win / chars   # tokens apportioned by text length
    print(f"long prompt {r['usage'].get('prompt_tokens')} tokens: TTFT {t2 - t1:.1f}s "
          f"({r['usage'].get('prompt_tokens', 0) / (t2 - t1):.0f} tok/s)")
    print(f"decoding stream during that prefill: ~{win_tok:.0f} tokens = {win_tok / (t2 - t1):.1f} tok/s")
    print(f"decoding stream: {d['usage'].get('completion_tokens')} tokens in {d['end'] - d['first']:.1f}s "
          f"(whole stream {(d['usage'].get('completion_tokens', 1) - 1) / (d['end'] - d['first']):.1f} tok/s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("decode"); d.add_argument("N", type=int, nargs="+"); d.add_argument("--set", default="code")
    d.add_argument("--new", type=int, default=512); d.add_argument("--temp", type=float, default=0.0)
    d.add_argument("--think", action="store_true")
    q = sp.add_parser("seq"); q.add_argument("--set", default="code"); q.add_argument("--new", type=int, default=512)
    q.add_argument("--temp", type=float, default=0.0); q.add_argument("--think", action="store_true")
    f = sp.add_parser("prefill"); f.add_argument("L", type=int, nargs="+"); f.add_argument("--new", type=int, default=1)
    l = sp.add_parser("long"); l.add_argument("N", type=int); l.add_argument("TOKENS", type=int); l.add_argument("NEW", type=int)
    i = sp.add_parser("interleave"); i.add_argument("TOKENS", type=int); i.add_argument("NEW", type=int)
    a = p.parse_args()
    {"decode": cmd_decode, "seq": cmd_seq, "prefill": cmd_prefill, "long": cmd_long, "interleave": cmd_interleave}[a.cmd](a)


if __name__ == "__main__":
    main()
