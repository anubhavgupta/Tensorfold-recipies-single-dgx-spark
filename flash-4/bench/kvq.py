#!/usr/bin/env python3
"""KV-cache quality probe: greedy replies to fixed prompts, plus needle retrieval in a long context.

  kvq.py run TAG [NEEDLE_TOKENS]   # writes results/kvq.TAG.json (replies, needle answers)
  kvq.py cmp BASE_TAG TAG          # per-prompt common-prefix share against BASE_TAG, needle results side by side

Needles: a short fact hidden at 10 %, 50 % and 90 % of NEEDLE_TOKENS (default 100000) of filler; the question asks for it.
"""
import json, os, random, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a short docstring and two tests.",
    "Explain how a B-tree keeps itself balanced during insertion. Be precise.",
    "Write a bash script that rotates logs older than 7 days and compresses them, handling spaces in names.",
    "What are the trade-offs between optimistic and pessimistic locking in a relational database?",
    "Write a SQL query for the top 3 customers by revenue per month, using window functions.",
    "Summarize the causes of the 1929 stock market crash in five bullet points.",
    "Write a Go function that reads a CSV from stdin and prints the column sums.",
    "Derive the closed form of the sum of the first n cubes and prove it by induction.",
    "Draft a polite email asking a vendor to extend an invoice deadline by two weeks.",
    "Write a Rust struct for a ring buffer with push, pop and iterator, no unsafe.",
    "Explain the difference between TCP slow start and congestion avoidance.",
    "Write a Dockerfile for a small Flask app using a multi-stage build and a non-root user.",
]


def post(body, timeout=3600):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(content, max_tokens):
    r = post({"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
              "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"]["content"] or "", r["usage"]["prompt_tokens"]


def filler(rng, n_tokens):
    words = ("alpha beta gamma delta river stone window lantern copper meadow harbor engine velvet orbit pencil canyon marble thunder "
             "violet anchor ribbon summit forest compass paper candle ladder mirror saddle quartz willow thistle bridge ember").split()
    out, chars = [], 0
    while chars < n_tokens * 5.9:
        s = " ".join(rng.choice(words) for _ in range(rng.randint(8, 16))).capitalize() + ". "
        out.append(s)
        chars += len(s)
    return out


def needle(n_tokens, depth, seed):
    rng = random.Random(seed)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    parts = filler(rng, n_tokens)
    parts.insert(int(len(parts) * depth), f"IMPORTANT NOTE: the vault passphrase is {code}. ")
    q = "\n\nWhat is the vault passphrase mentioned in the text above? Reply with the passphrase only."
    return "".join(parts) + q, code


def run(tag, ntok):
    res = {"replies": [], "needles": []}
    for p in PROMPTS:
        t = time.time()
        text, _ = ask(p, 300)
        res["replies"].append(text)
        print(f"[{tag}] reply {len(res['replies'])}/{len(PROMPTS)} {time.time() - t:.1f}s", flush=True)
    for i, depth in enumerate((0.1, 0.5, 0.9)):
        prompt, code = needle(ntok, depth, 7 + i)
        t = time.time()
        text, pt = ask(prompt, 40)
        ok = code in text
        res["needles"].append({"depth": depth, "tokens": pt, "code": code, "answer": text.strip()[:80], "ok": ok})
        print(f"[{tag}] needle depth {depth}: {pt} tokens ok={ok} ({time.time() - t:.0f}s) {text.strip()[:40]!r}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"kvq.{tag}.json"), "w"), indent=1)


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cmp(base, tag):
    a = json.load(open(os.path.join(OUT, f"kvq.{base}.json")))
    b = json.load(open(os.path.join(OUT, f"kvq.{tag}.json")))
    same = 0
    shares = []
    for i, (x, y) in enumerate(zip(a["replies"], b["replies"])):
        n = prefix(x, y)
        shares.append(n / max(len(x), 1))
        same += x == y
        print(f"prompt {i:2d}: identical={x == y} common prefix {n} of {len(x)} chars ({shares[-1]:.0%})")
    print(f"identical replies {same}/{len(shares)}, mean common-prefix share {sum(shares) / len(shares):.0%}")
    for x, y in zip(a["needles"], b["needles"]):
        print(f"needle depth {x['depth']}: {base} ok={x['ok']}  {tag} ok={y['ok']} ({y['tokens']} tokens)")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100000)
    else:
        cmp(sys.argv[2], sys.argv[3])
