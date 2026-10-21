#!/usr/bin/env python3
"""Sampled tool-call check for the dsv41 server (stdlib): N chat requests asking for two get_weather calls at the
server's default sampling. A reply is clean when it has exactly the calls Paris and Tokyo, each argument object
parsing to plain strings ({"city": "Paris"} or with "unit": "celsius"), nothing quoted twice, no placeholder keys.
usage: tool_ab.py [N=8] [URL=http://127.0.0.1:8888]"""
import json, sys, urllib.request
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8888") + "/v1/chat/completions"
tool = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
clean = 0
for i in range(n):
    body = {"model": "dsv41", "messages": [{"role": "user", "content": "What is the weather in Paris and in Tokyo right now? Use celsius."}],
            "tools": [tool], "max_tokens": 2000}
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    calls = d["choices"][0]["message"].get("tool_calls") or []
    try:
        args = [json.loads(c["function"]["arguments"]) for c in calls]
        ok = (sorted(a.get("city") for a in args) == ["Paris", "Tokyo"]
              and all(set(a) <= {"city", "unit"} and a.get("unit", "celsius") == "celsius" for a in args))
    except Exception:  # noqa: BLE001
        ok, args = False, [c["function"]["arguments"] for c in calls]
    clean += ok
    print(f"{'clean' if ok else 'BAD  '} tokens={d['usage']['completion_tokens']} {json.dumps(args)[:160]}", flush=True)
print(f"{clean}/{n} clean")
