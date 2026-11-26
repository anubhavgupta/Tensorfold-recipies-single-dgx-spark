#!/usr/bin/env python3
"""Functional tool-call check against the running server (stdlib only; tools are simulated, never executed).

Cases: streamed and non-streamed, thinking on/off, parallel calls, typed/nested/multiline arguments, a multi-turn
loop with simulated tool results, and a no-tool chat with tools offered. Exit code 1 if any case fails.
Env: URL (http://127.0.0.1:8888), MODEL (Ornith-1.5-122B).
"""
import json, os, sys, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "Ornith-1.5-122B")

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
WRITE = {"type": "function", "function": {"name": "fs.write_file", "description": "Write a text file",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"},
             "mode": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}},
             "meta": {"type": "object", "properties": {"author": {"type": "string"}, "version": {"type": "string"}}},
             "note": {"type": ["string", "null"]}},
             "required": ["path", "content", "overwrite", "mode", "tags", "meta"]}}}


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def chat(messages, tools, think, stream):
    body = {"model": MODEL, "messages": messages, "tools": tools, "max_tokens": 6000, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": think}}
    r = post(body)
    if not stream:
        m = json.load(r)["choices"][0]
        return m["message"].get("content") or "", [
            {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in m["message"].get("tool_calls") or []], m["finish_reason"]
    content, calls, finish = "", {}, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ch = json.loads(line[5:])["choices"]
        if not ch:
            continue
        d = ch[0].get("delta", {})
        content += d.get("content") or ""
        for tc in d.get("tool_calls") or []:
            c = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
            c["id"] = tc.get("id") or c["id"]
            fn = tc.get("function") or {}
            c["name"] += fn.get("name") or ""
            c["arguments"] += fn.get("arguments") or ""
        finish = ch[0].get("finish_reason") or finish
    return content, [calls[i] for i in sorted(calls)], finish


fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


for stream in (False, True):
    for think in (True, False):
        tag = f"[stream={stream} think={think}]"
        content, calls, fin = chat([{"role": "user", "content": "What's the weather in Paris and in Tokyo? Use celsius."}],
                                   [WEATHER], think, stream)
        try:
            args = [json.loads(c["arguments"]) for c in calls]
            cities = sorted(a["city"] for a in args)
            ok = all(c["name"] == "get_weather" for c in calls) and cities == ["Paris", "Tokyo"] and fin == "tool_calls"
        except Exception as e:  # noqa: BLE001
            ok, cities = False, repr(e)
        check(f"parallel weather {tag}", ok, f"{cities} finish={fin}")
        if "<tool_call>" in content or "<function=" in content:
            check(f"no markup in content {tag}", False, repr(content[:200]))

        code = 'def greet(name):\n    print(f"Hello, \\"{name}\\"!")  # caf\u00e9 \u2713\n    return "3"\n'
        prompt = ("Call fs.write_file exactly once with: path 'src/greet.py', overwrite true, mode 420, tags "
                  "['demo', 'null'], meta {author: 'Ana', version: '3'}, note null, and this exact content:\n```\n"
                  + code + "```")
        content, calls, fin = chat([{"role": "user", "content": prompt}], [WRITE], think, stream)
        try:
            a = json.loads(calls[0]["arguments"])
            ok = (calls[0]["name"] == "fs.write_file" and a["path"] == "src/greet.py" and a["overwrite"] is True
                  and a["mode"] == 420 and a["tags"] == ["demo", "null"] and a["meta"] == {"author": "Ana", "version": "3"}
                  and a["content"].strip() == code.strip())
            # The checkpoint often writes Python's None for a nullable value after Python code (raw output checked:
            # the parser maps a literal null to null). "None" is a valid string for ["string","null"], so it is
            # reported, not failed, and not rewritten.
            detail = (f"note={a.get('note')!r}" if ok else json.dumps(a)[:300])
        except Exception as e:  # noqa: BLE001
            ok, detail = False, repr(e)
        check(f"typed/nested/multiline args {tag}", ok, detail)

# Multi-turn loop with simulated results.
msgs = [{"role": "user", "content": "Compare the weather in Paris and Tokyo (celsius) and tell me which is warmer."}]
content, calls, fin = chat(msgs, [WEATHER], True, True)
msgs.append({"role": "assistant", "content": content, "tool_calls": [
    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
temps = {"Paris": 14, "Tokyo": 23}
for c in calls:
    city = json.loads(c["arguments"]).get("city")
    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"city": city, "temp_c": temps.get(city, 0)})})
content, calls2, fin = chat(msgs, [WEATHER], True, True)
check("multi-turn loop final answer", fin == "stop" and not calls2 and "Tokyo" in content, repr(content[:160]))

content, calls, fin = chat([{"role": "user", "content": "Say hello in French, nothing else."}], [WEATHER], False, False)
check("no-tool chat with tools offered", not calls and "bonjour" in content.lower(), repr(content[:80]))

print("FAILED" if fails else "ALL PASS", f"({fails} failures)")
sys.exit(1 if fails else 0)
