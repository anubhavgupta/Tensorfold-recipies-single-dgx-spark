#!/usr/bin/env python3
"""Greedy replies with drafts on and off must be byte-identical (speculative decoding is exact): 4 prompts, 4 at once, plus a long one."""
import concurrent.futures as cf, json, os, sys, urllib.request
BASE = os.environ.get("BASE", "http://127.0.0.1:8888/v1")
def ask(prompt, draft, n=300):
    body = {"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "draft": draft}
    r = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))["choices"][0]["message"]["content"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvq
prompts = kvq.PROMPTS[:4] + [kvq.needle(20000, 0.5, 3)[0]]
bad = 0
with cf.ThreadPoolExecutor(5) as ex:
    on = list(ex.map(lambda p: ask(p, True), prompts))
    off = list(ex.map(lambda p: ask(p, False), prompts))
for i, (a, b) in enumerate(zip(on, off)):
    print(f"prompt {i}: drafts on == off: {a == b}")
    bad += a != b
print("EXACT" if not bad else f"MISMATCH {bad}")
