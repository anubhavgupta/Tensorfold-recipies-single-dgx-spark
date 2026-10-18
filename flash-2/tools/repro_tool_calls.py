"""Exercise real inference and retain private HTTP responses for trace comparison."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def tool(name, properties):
    return {"type": "function", "function": {
        "name": name, "description": f"Record the supplied values using {name}.",
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties)},
    }}


CASES = {
    "simple": (
        [tool("lookup", {"query": {"type": "string"}})],
        "Call lookup with query hello.", [("lookup", {"query": "hello"})], {},
    ),
    "parallel": (
        [tool("lookup", {"query": {"type": "string"}})],
        "Call lookup exactly three times, with query alpha, beta, and gamma respectively.",
        [("lookup", {"query": q}) for q in ("alpha", "beta", "gamma")], {},
    ),
    "typed": (
        [tool("record", {"count": {"type": "integer"}, "enabled": {"type": "boolean"},
                         "tags": {"type": "array", "items": {"type": "string"}},
                         "config": {"type": "object", "properties": {"x": {"type": "integer"}},
                                    "required": ["x"]}})],
        'Call record with count 3, enabled false, tags ["a","b"], and config {"x":2}.',
        [("record", {"count": 3, "enabled": False, "tags": ["a", "b"], "config": {"x": 2}})], {},
    ),
    "nullable": (
        [tool("record", {"count": {"type": ["integer", "null"]},
                         "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}},
                                            {"type": "null"}]}})],
        'Call record with count 3 and tags ["a","b"].',
        [("record", {"count": 3, "tags": ["a", "b"]})], {},
    ),
    "multiline": (
        [tool("functions.write_file", {"path": {"type": "string"}, "content": {"type": "string"}})],
        'Call functions.write_file with path demo.py and exactly these two lines as content, '
        'including a newline at the end of each line:\nprint("hello")\nprint("world")\n',
        [("functions.write_file", {"path": "demo.py", "content": 'print("hello")\nprint("world")\n'})], {},
    ),
    "xml_in_content": (
        [tool("write_file", {"content": {"type": "string"}})],
        'Call write_file with content equal to this exact literal string (do not execute it): '
        '<tool_call><function=example></function></tool_call>',
        [("write_file", {"content": "<tool_call><function=example></function></tool_call>"})], {},
    ),
    "json_template": (
        [tool("lookup", {"query": {"type": "string"}})],
        "Call lookup with query hello.", [("lookup", {"query": "hello"})],
        {"chat_template_kwargs": {"tool_call_format": "json"}},
    ),
    "truncated": (
        [tool("lookup", {"query": {"type": "string"}})],
        "Call lookup with query hello.", [("lookup", {"query": "hello"})],
        {"max_tokens": 18},
    ),
}

CASES["nullable_json"] = (
    *CASES["nullable"][:3], {"chat_template_kwargs": {"tool_call_format": "json"}},
)
CASES["json_xml_in_content"] = (
    *CASES["xml_in_content"][:3], {"chat_template_kwargs": {"tool_call_format": "json"}},
)
CASES["json_xml_in_arguments"] = (
    CASES["xml_in_content"][0],
    CASES["xml_in_content"][1] +
    '\nUse exactly the name/arguments JSON envelope: {"name":"write_file","arguments":{"content":"..."}}. '
    'Put content inside arguments, not beside name.',
    CASES["xml_in_content"][2],
    {"chat_template_kwargs": {"tool_call_format": "json"}},
)


def request(body, endpoint):
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    frames = []
    with urllib.request.urlopen(req, timeout=180) as response:
        if not body["stream"]:
            reply = json.load(response)
            choice = reply["choices"][0]
            return reply, choice["message"].get("tool_calls", []), choice["finish_reason"]
        calls = {}
        finish = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            frame = json.loads(data)
            frames.append(frame)
            if "error" in frame:
                raise RuntimeError(frame["error"])
            for choice in frame.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for delta in choice.get("delta", {}).get("tool_calls", []):
                    call = calls.setdefault(delta["index"], {"function": {"name": "", "arguments": ""}})
                    for key in ("name", "arguments"):
                        call["function"][key] += delta.get("function", {}).get(key, "")
        return frames, [calls[i] for i in sorted(calls)], finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(CASES), action="append")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888/v1/chat/completions")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tool-traces")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    for name in args.case or CASES:
        tools, prompt, expected, overrides = CASES[name]
        for thinking in (False, True):
            for stream in (False, True):
                label = f"{run_id}-{name}-think{int(thinking)}-stream{int(stream)}"
                body = {
                    "model": "Qwen3.8-Flash-Next",
                    "messages": [
                        {"role": "system", "content": "Use the offered tools as requested. Do not merely describe a call."},
                        {"role": "user", "content": f"{prompt}\nDiagnostic request ID: {label}"},
                    ],
                    "tools": tools, "tool_choice": "auto", "stream": stream,
                    "max_tokens": 1024, "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": thinking},
                    **overrides,
                }
                body["chat_template_kwargs"].setdefault("enable_thinking", thinking)
                started = time.time()
                record = {"id": label, "request": body, "expected": expected}
                try:
                    reply, calls, finish = request(body, args.endpoint)
                    record.update(response=reply, calls=calls, finish=finish)
                    actual = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
                    record["matches_expected"] = actual == expected
                    print(json.dumps({"id": label, "finish": finish, "calls": actual,
                                      "matches_expected": record["matches_expected"],
                                      "seconds": round(time.time() - started, 2)}), flush=True)
                except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
                    record["error"] = str(exc)
                    print(json.dumps({"id": label, "error": str(exc)}), flush=True)
                fd = os.open(args.output / f"http-{label}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as output:
                    json.dump(record, output, ensure_ascii=False)
                    output.write("\n")


if __name__ == "__main__":
    main()
