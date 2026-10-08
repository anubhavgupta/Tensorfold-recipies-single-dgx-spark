#!/usr/bin/env python3
"""Engine-neutral benchmark over an OpenAI-compatible /v1 endpoint (TensorFold or TabbyAPI).

  suite.py content <tag> <Ns e.g. 1,2,4,9> [gen=512]     per content class (code, prose, devops, json): decode tok/s, N concurrent streams
  suite.py prefill <tag> <token counts e.g. 4096,32768>   one stream, unique prompt, gen=1: prefill tok/s = prompt_tokens / TTFT
  suite.py full    <tag> <N> [prompt_tokens=250000] [gen=256]   N streams at once, each with a unique long prompt

BASE_URL (default http://127.0.0.1:8899/v1), API_KEY optional. Results go to results/<mode>.<tag>.json.
"""
import json, os, random, sys, threading, time, urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1").rstrip("/")
KEY = os.environ.get("API_KEY")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONTENT = {
    "code": [
        "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints and a usage example.",
        "Implement an LRU cache class in Python with get and put in O(1), plus unit tests using pytest.",
        "Write a Go program that reads a CSV from stdin and prints the per-column mean of numeric columns.",
        "Write a Rust function that merges overlapping intervals, with tests and an explanation of its complexity.",
    ],
    "prose": [
        "Write a 400-word short story about a lighthouse keeper who discovers the light has been answering ships on its own.",
        "Write an essay of about 400 words on why the printing press changed European society.",
        "Describe in vivid detail a market in a coastal town at dawn, in about 400 words.",
        "Write a reflective 400-word letter from a retired astronaut to a child who wants to travel to Mars.",
    ],
    "devops": [
        "Explain how to deploy a three-replica web app on Kubernetes with a rolling update and a readiness probe, then give the full YAML.",
        "Write a docker-compose.yml for postgres, redis and a Python API with healthchecks, and explain each service.",
        "Write a GitHub Actions workflow that builds a Go service, runs tests, and pushes a Docker image on tags, with explanation.",
        "Give a Terraform module for an AWS VPC with two public and two private subnets, and describe its variables.",
    ],
    "json": [
        "Generate a JSON array of 12 fictional employees with id, name, department, salary, skills (array) and manager id.",
        "Produce an OpenAPI 3.0 JSON spec for a small todo-list API with CRUD endpoints.",
        "Output a JSON object describing a periodic-table-style dataset of 15 elements with symbol, mass, group and state.",
        "Create a JSON config for a CI pipeline with 5 stages, each with commands, env and retry policy.",
    ],
}


def req(path, payload=None):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return urllib.request.Request(BASE + path, json.dumps(payload).encode() if payload else None, h)


def model():
    with urllib.request.urlopen(req("/models"), timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def stream(path, body, out):
    t0 = time.perf_counter(); tf = None; usage = None; err = None
    try:
        with urllib.request.urlopen(req(path, body), timeout=7200) as resp:
            for raw in resp:
                if not raw.startswith(b"data: "):
                    continue
                c = raw[6:].strip()
                if c == b"[DONE]":
                    break
                d = json.loads(c)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch and tf is None:
                    x = ch[0]
                    piece = x.get("text") or (x.get("delta") or {}).get("content") or (x.get("delta") or {}).get("reasoning_content")
                    if piece:
                        tf = time.perf_counter()
    except Exception as e:  # recorded, not raised, so one failed stream does not hide the others
        err = repr(e)
    tl = time.perf_counter()
    u = usage or {}
    out.append({"t0": t0, "tf": tf or tl, "tl": tl, "gen": u.get("completion_tokens", 0),
                "prompt": u.get("prompt_tokens", 0), "err": err})


def run_streams(path, bodies):
    out, th = [], []
    for b in bodies:
        t = threading.Thread(target=stream, args=(path, b, out)); t.start(); th.append(t)
    for t in th:
        t.join()
    return out


def words():
    ws = [w.strip().lower() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 <= len(w.strip()) <= 9]
    return ws


def long_prompt(ws, rng, ntok, cpt):
    # Random common-ish words: no repetition for n-gram/MTP drafting to exploit, and unique per stream.
    chars, parts, n = int(ntok * cpt), [], 0
    while n < chars:
        s = " ".join(rng.choice(ws) for _ in range(12)).capitalize() + ". "
        parts.append(s); n += len(s)
    return "".join(parts)[:chars] + "\n\nSummarize the text above in one sentence."


def calibrate(ws, m):
    rng = random.Random(1)
    p = long_prompt(ws, rng, 4000, 4.0)
    out = run_streams("/completions", [{"model": m, "prompt": p, "max_tokens": 1, "temperature": 0, "stream": True,
                                       "stream_options": {"include_usage": True}}])
    return len(p) / max(1, out[0]["prompt"])


def summarize(rows):
    ok = [r for r in rows if not r["err"] and r["gen"] > 1]
    if not ok:
        return {"ok": 0, "errors": [r["err"] for r in rows if r["err"]]}
    per = [round((r["gen"] - 1) / max(r["tl"] - r["tf"], 1e-6), 1) for r in ok]
    tot = sum(r["gen"] for r in ok)
    window = max(r["tl"] for r in ok) - min(r["tf"] for r in ok)
    sw = min(r["tl"] for r in ok) - max(r["tf"] for r in ok)
    steady = None
    if sw > 0:
        steady = round(sum(r["gen"] * sw / max(r["tl"] - r["tf"], 1e-6) for r in ok) / sw, 1)
    return {"ok": len(ok), "of": len(rows), "per_stream": per, "per_stream_mean": round(sum(per) / len(per), 1),
            "aggregate_window": round(tot / window, 1), "aggregate_steady": steady,
            "errors": [r["err"] for r in rows if r["err"]]}


def save(mode, tag, data):
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, f"{mode}.{tag}.json"), "w"), indent=1)


def main():
    mode, tag = sys.argv[1], sys.argv[2]
    m = model()
    res = {"model": m, "base": BASE}
    if mode == "content":
        Ns = [int(x) for x in sys.argv[3].split(",")]
        gen = int(sys.argv[4]) if len(sys.argv) > 4 else 512
        for cls, prompts in [(c,v) for c,v in CONTENT.items() if c in os.environ.get('CLS','code,prose,devops,json').split(',')]:
            for N in Ns:
                # two rounds; round 0 also warms the engine, so report the better-known second one
                for r in range(2):
                    bodies = [{"model": m, "messages": [{"role": "user", "content": prompts[(i + r) % len(prompts)] + f" (variant {r}-{i})"}],
                               "max_tokens": gen, "temperature": 0, "top_k": 1, "stream": True,
                               "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
                              for i in range(N)]
                    s = summarize(run_streams("/chat/completions", bodies))
                print(f"[{tag}] {cls} N={N}: {s}", flush=True)
                res[f"{cls}/N={N}"] = s
                save(mode, tag, res)
    elif mode == "prefill":
        ws = words(); cpt = calibrate(ws, m); res["chars_per_token"] = round(cpt, 3)
        rng = random.Random(7)
        for n in [int(x) for x in sys.argv[3].split(",")]:
            p = long_prompt(ws, rng, n, cpt)
            rows = run_streams("/completions", [{"model": m, "prompt": p, "max_tokens": 1, "temperature": 0, "stream": True,
                                                 "stream_options": {"include_usage": True}}])
            r = rows[0]
            ttft = r["tf"] - r["t0"]
            res[str(n)] = {"prompt_tokens": r["prompt"], "ttft_s": round(ttft, 2),
                           "prefill_tok_s": round(r["prompt"] / ttft, 1) if not r["err"] else None, "err": r["err"]}
            print(f"[{tag}] prefill {res[str(n)]}", flush=True)
            save(mode, tag, res)
    elif mode == "full":
        N = int(sys.argv[3]); ptok = int(sys.argv[4]) if len(sys.argv) > 4 else 250000
        gen = int(sys.argv[5]) if len(sys.argv) > 5 else 256
        ws = words(); cpt = calibrate(ws, m); res["chars_per_token"] = round(cpt, 3)
        rng = random.Random(100 + N)
        bodies = [{"model": m, "prompt": long_prompt(ws, rng, ptok, cpt), "max_tokens": gen, "temperature": 0, "top_k": 1,
                   "stream": True, "stream_options": {"include_usage": True}} for _ in range(N)]
        t0 = time.perf_counter()
        rows = run_streams("/completions", bodies)
        wall = time.perf_counter() - t0
        ok = [r for r in rows if not r["err"]]
        s = summarize(rows)
        pt = sum(r["prompt"] for r in ok)
        last_first = max((r["tf"] for r in ok), default=t0) - t0
        s.update({"N": N, "prompt_tokens_each": [r["prompt"] for r in rows], "wall_s": round(wall, 1),
                  "ttft_s": [round(r["tf"] - r["t0"], 1) for r in rows],
                  "aggregate_prefill_tok_s": round(pt / last_first, 1) if ok else None})
        res.update(s)
        print(f"[{tag}] full N={N}: {json.dumps(s)}", flush=True)
        save(mode, f"{tag}-N{N}", res)
    else:
        sys.exit(__doc__)


main()
