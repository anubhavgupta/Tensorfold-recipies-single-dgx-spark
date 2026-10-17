"""The repo README's throughput test: N clients send the same prompt at once, greedy, thinking off, 200-token replies.

python3 same.py [N ...]  (default 1 2 4 8); prints aggregate decode tok/s (completion tokens / slowest decode time)
and aggregate wall tok/s (completion tokens / wall time, prefill included) for prose and code."""
import json, sys, threading, time, urllib.request

URL = "http://localhost:8888/v1/chat/completions"
PROMPTS = {"prose": "Explain in detail how the Roman Republic transformed into the Empire, covering the key figures, "
                    "battles and institutions.",
           "code": "Write a complete Python implementation of a red-black tree with insert, delete and in-order "
                   "iteration, with docstrings and a test suite."}


def call(prompt, n_tok):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": n_tok, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            if first is None and any((c.get("delta") or {}).get("content") for c in j.get("choices", [])):
                first = time.time()
    end = time.time()
    return usage["completion_tokens"], first, end, t0


def run(kind, n, n_tok=200):
    res = [None] * n
    th = [threading.Thread(target=lambda i=i: res.__setitem__(i, call(PROMPTS[kind], n_tok))) for i in range(n)]
    t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    toks = sum(r[0] for r in res)
    decode = max(r[2] for r in res) - min(r[1] for r in res)
    print(f"{kind:5s} N={n}: decode {toks / decode:6.1f} tok/s  wall {toks / (time.time() - t0):6.1f} tok/s", flush=True)


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    call(PROMPTS["prose"], 16)
    for kind in PROMPTS:
        for n in ns:
            run(kind, n)
