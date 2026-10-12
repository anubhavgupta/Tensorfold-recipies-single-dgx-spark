"""Aggregate decode throughput at N concurrent streams: python3 tput.py [--code] [N ...] (default 1 2 4 8).

Essays by default; --code asks for Python implementations (more of each draft is accepted)."""
import sys, threading, time
from bench import call

ESSAYS = ["the Roman Empire", "red-black trees in Python", "photosynthesis", "the French Revolution",
          "TCP congestion control", "the history of jazz", "a Rust HTTP server", "black holes",
          "the Silk Road", "garbage collection in the JVM", "the Renaissance", "a Go key-value store",
          "plate tectonics", "the Cold War", "transformers in ML", "the Ottoman Empire"]
CODE = ["a red-black tree", "an LRU cache", "a JSON parser", "a trie", "a thread pool", "a B-tree", "an HTTP router",
        "a priority queue", "a skip list", "a regex engine", "a SQL tokenizer", "a rate limiter", "a Markdown parser",
        "an event bus", "a bloom filter", "a consistent-hash ring"]


def prompt(i, code):
    if code:
        return f"Write a complete implementation of {CODE[i % len(CODE)]} in Python with tests."
    return f"Write a long, detailed essay on {ESSAYS[i % len(ESSAYS)]}."


def run(n, new=800, code=False):
    res = [None] * n
    def one(i):
        res[i] = call([{"role": "user", "content": prompt(i, code)}], new, temperature=1.0)
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    toks = sum((r["usage"] or {}).get("completion_tokens", 0) for r in res)
    per = [r["decode_tps"] for r in res]
    errs = sum(1 for r in res if r["err"])
    print(f"{'code' if code else 'essay'} N={n:2d}: total {toks} tok in {wall:.1f}s = {toks / wall:6.1f} tok/s aggregate | "
          f"per-stream mean {sum(per) / n:5.1f} (min {min(per):.1f}, max {max(per):.1f}) | errors {errs}", flush=True)


if __name__ == "__main__":
    code = "--code" in sys.argv
    for n in [int(x) for x in sys.argv[1:] if x != "--code"] or [1, 2, 4, 8]:
        run(n, code=code)
