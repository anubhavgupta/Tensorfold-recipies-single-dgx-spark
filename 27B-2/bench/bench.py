"""Decode / prefill / concurrency benchmark for the 27B-2 server.

  python3 bench.py decode
  python3 bench.py prefill 2000 8000 32000 64000
  python3 bench.py conc N TOKENS [MAX_NEW]     N parallel streams, each with a distinct TOKENS-token prompt
"""
import json, sys, time, random, threading, urllib.request

URL = "http://localhost:8888/v1/chat/completions"


def call(messages, max_tokens, think=False, temperature=None):
    body = {"model": "Qwen3.8-27B", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": think}}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = ""; err = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                j = json.loads(line[5:])
                if "error" in j:
                    err = j["error"]
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices", []):
                    d = c["delta"]
                    piece = d.get("content") or d.get("reasoning_content") or ""
                    if piece and first is None:
                        first = time.time()
                    text += piece
    except Exception as e:
        err = repr(e)
    t1 = time.time()
    n = (usage or {}).get("completion_tokens", 0)
    return {"ttft": (first or t1) - t0, "total": t1 - t0, "usage": usage, "text": text, "err": err,
            "decode_tps": (n - 1) / max(t1 - (first or t0), 1e-6) if n > 1 else 0.0}


def filler(tokens, seed):
    rnd = random.Random(seed)
    words = ["alpha", "river", "stone", "market", "engine", "silver", "garden", "window", "forest", "bridge", "castle",
             "planet", "violin", "harbor", "meadow", "lantern", "compass", "thunder", "orchard", "canyon"]
    return " ".join(f"{rnd.choice(words)}{rnd.randint(0, 9999)}" for _ in range(tokens))


def prompt_of(tokens, seed, tail="Say OK."):
    return filler(int(tokens / 3.1), seed) + "\n\n" + tail


def decode_tests():
    prompts = {
        "code": "Write a complete Python implementation of a red-black tree with insert, delete and in-order iteration, with docstrings and a test suite.",
        "chat": "Explain in detail how the Roman Republic transformed into the Empire, covering the key figures, battles and institutions.",
    }
    for k, p in prompts.items():
        for temp in (0.0, 1.0):
            r = call([{"role": "user", "content": p}], 700, temperature=temp)
            print(f"decode {k} T={temp}: {r['decode_tps']:.1f} tok/s ({r['usage']['completion_tokens']} tok, ttft {r['ttft']:.2f}s)", flush=True)


def conc(n, tokens, new):
    res = [None] * n
    def run(i):
        res[i] = call([{"role": "user", "content": prompt_of(tokens, time.time() + i, "Ignore the text above and write a very long, detailed essay (at least 3000 words) on the Roman Empire.")}], new)
    t0 = time.time()
    th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    for i, r in enumerate(res):
        u = r["usage"] or {}
        print(f"  stream {i}: prompt {u.get('prompt_tokens')} ttft {r['ttft']:.1f}s total {r['total']:.1f}s decode {r['decode_tps']:.1f} tok/s err={r['err']}", flush=True)
    print(f"{n} x {tokens}: wall {time.time() - t0:.1f}s, ok {sum(1 for r in res if not r['err'] and r['usage'])}/{n}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "decode":
        decode_tests()
    elif mode == "prefill":
        for t in [int(x) for x in sys.argv[2:]]:
            r = call([{"role": "user", "content": prompt_of(t, random.random())}], 4)
            pt = r["usage"]["prompt_tokens"]
            print(f"prefill {pt} tok: {r['ttft']:.1f}s = {pt / r['ttft']:.0f} tok/s err={r['err']}", flush=True)
    elif mode == "conc":
        conc(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 64)
