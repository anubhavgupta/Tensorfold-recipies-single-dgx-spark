"""Aggregate decode throughput at N concurrent streams: python3 tput.py [N ...] (default 1 2 4 8)."""
import sys, threading, time
from bench import call

TOPICS = ["the Roman Empire", "red-black trees in Python", "photosynthesis", "the French Revolution",
          "TCP congestion control", "the history of jazz", "a Rust HTTP server", "black holes",
          "the Silk Road", "garbage collection in the JVM", "the Renaissance", "a Go key-value store",
          "plate tectonics", "the Cold War", "transformers in ML", "the Ottoman Empire"]

def run(n, new=800):
    res = [None] * n
    def one(i):
        res[i] = call([{"role": "user", "content": f"Write a long, detailed essay on {TOPICS[i % len(TOPICS)]}."}],
                      new, temperature=1.0)
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    toks = sum((r["usage"] or {}).get("completion_tokens", 0) for r in res)
    per = [r["decode_tps"] for r in res]
    errs = sum(1 for r in res if r["err"])
    print(f"N={n:2d}: total {toks} tok in {wall:.1f}s = {toks / wall:6.1f} tok/s aggregate | "
          f"per-stream mean {sum(per) / n:5.1f} (min {min(per):.1f}, max {max(per):.1f}) | errors {errs}", flush=True)

if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]:
        run(n)
