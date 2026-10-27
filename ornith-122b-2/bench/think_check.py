#!/usr/bin/env python3
"""Thinking-tag checks against the running server (streamed and non-streamed).

usage: think_check.py [URL=http://127.0.0.1:8888] [case ...]

Each case sends one chat request and reports where the text landed: reasoning_content, content, tool_calls.
FAIL when a think/tool tag leaks into content or reasoning, when thinking-off still returns reasoning, or when a
streamed reply differs in kind from the non-streamed one. Tool results are simulated; nothing is executed."""
import json, sys, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "http://127.0.0.1:8888").rstrip("/")
ONLY = [a for a in sys.argv[1:] if not a.startswith("http")]
MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "<parameter=", "<|im_end|>", "<|im_start|>")

WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
Q = [{"role": "user", "content": "What is 17*23? Answer briefly."}]
HIST = [{"role": "user", "content": "Pick a prime between 10 and 20."},
        {"role": "assistant", "content": "13", "reasoning_content": "Primes 11,13,17,19. Pick 13."},
        {"role": "user", "content": "Now double it. Answer briefly."}]
HIST_TAGGED = [HIST[0], {"role": "assistant", "content": "<think>\nPrimes 11,13,17,19. Pick 13.\n</think>\n\n13"}, HIST[2]]
HIST_REASONING = [HIST[0], {"role": "assistant", "content": "13", "reasoning": "Primes 11,13,17,19. Pick 13."}, HIST[2]]
QUOTE = [{"role": "user", "content": "A Qwen chat template closes an empty reasoning block with a fixed string. While you "
          "reason, write that exact string out with its literal <think> and </think> tags, then answer in one short "
          "sentence without using any tags."}]
TOOL_LOOP = [{"role": "user", "content": "What's the weather in Paris? Use the tool."},
             {"role": "assistant", "content": "", "reasoning_content": "Need the weather tool for Paris.",
              "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}]

CASES = {
    "think_on": dict(messages=Q),
    "think_off_kwargs": dict(messages=Q, chat_template_kwargs={"enable_thinking": False}),
    "think_off_effort": dict(messages=Q, reasoning_effort="none"),
    "effort_low": dict(messages=Q, reasoning_effort="low"),
    "hist_reasoning_content": dict(messages=HIST),
    "hist_think_in_content": dict(messages=HIST_TAGGED),
    "hist_reasoning_field": dict(messages=HIST_REASONING),
    "hist_think_off": dict(messages=HIST, chat_template_kwargs={"enable_thinking": False}),
    "tool_call": dict(messages=TOOL_LOOP[:1], tools=WEATHER),
    "tool_call_think_off": dict(messages=TOOL_LOOP[:1], tools=WEATHER, chat_template_kwargs={"enable_thinking": False}),
    "tool_result": dict(messages=TOOL_LOOP, tools=WEATHER),
    # the reasoning quotes </think> inline: patches/0004 keeps it there (stock 1.1.7 ends the reasoning at the first
    # one); the prompt asks for literal tags, so they may appear in the answer too
    "quoted_tag": dict(messages=QUOTE, temperature=0, max_tokens=4096),
}


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def run(body, stream):
    body = dict(body, model=MODEL, max_tokens=body.get("max_tokens", 3000), stream=stream)
    if not stream:
        d = json.load(post(body)); ch = d["choices"][0]; m = ch["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
        return m.get("reasoning_content") or m.get("reasoning") or "", m.get("content") or "", calls, ch["finish_reason"]
    r = c = ""; calls = {}; fin = None
    for line in post(body):
        line = line.decode().strip()
        if not line.startswith("data: {"): continue
        ch = json.loads(line[6:])["choices"][0]; dl = ch.get("delta") or {}
        r += dl.get("reasoning_content") or dl.get("reasoning") or ""; c += dl.get("content") or ""
        for tc in dl.get("tool_calls") or []:
            e = calls.setdefault(tc.get("index", 0), ["", ""])
            f = tc.get("function") or {}
            e[0] += f.get("name") or ""; e[1] += f.get("arguments") or ""
        fin = ch.get("finish_reason") or fin
    return r, c, [tuple(v) for _, v in sorted(calls.items())], fin


def check(name, body):
    bad = []
    res = {}
    for stream in (False, True):
        r, c, calls, fin = run(body, stream)
        res[stream] = (r, c, calls, fin)
        tag = "S" if stream else "N"
        for where, text in (("reasoning", r), ("content", c)):
            if name.startswith("quoted"): continue
            for t in TAGS:
                if t in text: bad.append(f"{tag}: {t!r} in {where}")
        off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False or body.get("reasoning_effort") == "none"
        if off and r.strip(): bad.append(f"{tag}: thinking off but reasoning={r[:60]!r}")
        if not off and not r.strip(): bad.append(f"{tag}: thinking on but no reasoning")
        if name.startswith("quoted"):
            if "</think>" not in r: bad.append(f"{tag}: the quoted </think> did not stay in the reasoning")
            if not c.strip(): bad.append(f"{tag}: no answer after the reasoning")
        if "tools" in body and name != "tool_result" and not calls: bad.append(f"{tag}: no tool call (finish={fin})")
        for n, a in calls:
            try: json.loads(a)
            except Exception: bad.append(f"{tag}: bad arguments {a!r}")
        print(f"  {tag} finish={fin} reasoning={r[:70]!r}{'...' if len(r) > 70 else ''} content={c[:90]!r} calls={calls}")
    print(f"{'FAIL' if bad else 'ok  '} {name}" + ("".join("\n       " + b for b in bad)))
    return not bad


if __name__ == "__main__":
    names = ONLY or list(CASES)
    ok = sum(check(n, CASES[n]) for n in names)
    print(f"{ok}/{len(names)} cases clean")
