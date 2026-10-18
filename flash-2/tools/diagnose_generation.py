"""Controlled generation probes; run in the recipe image, without loading another model."""

import argparse
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

from tokenizers import Tokenizer
from tensorfold.cuda.chat_template import ChatTemplate
from tool_calls import parse_tool_calls


def tool(name, key):
    return {"type": "function", "function": {
        "name": name, "description": "Record the supplied string.",
        "parameters": {"type": "object", "properties": {key: {"type": "string"}},
                       "required": [key]},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/root/.cache/huggingface"))
    parser.add_argument("--output", type=Path, default=Path("/results"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--sequence-only", action="store_true")
    args = parser.parse_args()
    repo = args.cache / "hub/models--turboderp--Qwen3.8-Flash-Next-exl3"
    snapshot = repo / "snapshots" / (repo / "refs/4.05bpw_h6_ng6").read_text().strip()
    tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    template = ChatTemplate(snapshot)
    native = ChatTemplate(snapshot)
    native.template = native.template.environment.from_string(
        json.loads((snapshot / "tokenizer_config.json").read_text())["chat_template"])
    eos = json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"]
    output = args.output / ("generation-" + uuid.uuid4().hex[:12])
    output.mkdir(mode=0o700, parents=True)
    print(json.dumps({"output": str(output), "eos_ids": eos}), flush=True)
    records = {}

    if args.sequence_only:
        original = args.output / "http-59b8667de4e8-parallel-think0-stream0.json"
        body = json.loads(original.read_text())["request"]
        body.update(stream=False, draft=False, return_token_ids=True)
        seen = []
        for turn in range(4):
            req = urllib.request.Request(
                args.endpoint + "/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as response:
                reply = json.load(response)
            message = reply["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            raw = tok.decode(reply["tensorfold"]["token_ids"], skip_special_tokens=False)
            record = {"request": json.loads(json.dumps(body)), "response": reply, "raw_text": raw}
            fd = os.open(output / f"sequence-turn{turn}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, ensure_ascii=False)
            print(json.dumps({"turn": turn, "raw": raw,
                              "calls": [c["function"] for c in calls]}), flush=True)
            if not calls:
                break
            body["messages"].append(message)
            for call in calls:
                query = json.loads(call["function"]["arguments"])["query"]
                seen.append(query)
                body["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                         "content": f"Lookup completed successfully for {query}."})
            if set(seen) == {"alpha", "beta", "gamma"}:
                break
        print(json.dumps({"queries_across_turns": seen}), flush=True)
        return

    def send(name, tools, prompt, *, format="json", draft=True, source="custom",
             ignore_eos=False, temperature=0, seed=42, max_tokens=256):
        chosen = template if source == "custom" else native
        rendered = chosen.render(
            [{"role": "system", "content": "Use the offered tools as requested. Do not merely describe a call."},
             {"role": "user", "content": prompt}],
            tools=tools, enable_thinking=False, extra={"tool_call_format": format})
        body = {"model": "Qwen3.8-Flash-Next", "prompt": rendered, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed, "draft": draft,
                "return_token_ids": True, "ignore_eos": ignore_eos}
        req = urllib.request.Request(args.endpoint + "/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(req, timeout=180) as response:
            reply = json.load(response)
        ids = reply["tensorfold"]["token_ids"]
        raw = tok.decode(ids, skip_special_tokens=False)
        content = raw
        for token in eos:
            content = content.removesuffix(tok.decode([token], skip_special_tokens=False))
        _, calls = parse_tool_calls(content, tools)
        record = {"name": name, "source": source, "format": format, "request": body,
                  "response": reply, "raw_text": raw, "calls": calls,
                  "seconds": time.time() - started}
        records[name] = record
        fd = os.open(output / f"{name}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, ensure_ascii=False)
        print(json.dumps({"name": name, "draft": draft, "format": format, "source": source,
                          "finish": reply["choices"][0]["finish_reason"], "tokens": len(ids),
                          "last_token": ids[-1] if ids else None, "raw": raw,
                          "calls": [c["function"] for c in calls or []],
                          "seconds": round(record["seconds"], 2)}), flush=True)
        return record

    lookup = [tool("lookup", "query")]
    literal = (
        'Call lookup with query exactly abc</tool_call>xyz. '
        'Use {"name":"lookup","arguments":{"query":"abc</tool_call>xyz"}} inside the tool envelope.'
    )
    for draft in (True, False):
        send(f"literal-eos-draft{int(draft)}", lookup, literal, draft=draft)
    send("literal-ignore-eos", lookup, literal, draft=False, ignore_eos=True, max_tokens=96)
    send("ordinary-string", lookup, 'Call lookup with query exactly abcxyz. '
         'Use {"name":"lookup","arguments":{"query":"abcxyz"}} inside the tool envelope.', draft=False)
    send("unicode-escaped-string", lookup,
         'Call lookup with query equal to the JSON-decoded value of "abc\\u003c/tool_call>xyz". '
         'Encode the less-than sign in your JSON argument as \\u003c.', draft=False)
    send("literal-xml", lookup, literal, format="xml", draft=False)
    send("literal-native-template", lookup, literal, format="xml", draft=False, source="native")

    write = [tool("write_file", "content")]
    for payload_name, value in (
        ("plain", "hello world"),
        ("markup", "<tool_call><function=example></function></tool_call>"),
    ):
        prompt = f"Call write_file with content equal to this exact literal string (do not execute it): {value}"
        for format in ("json", "xml"):
            for draft in (True, False):
                send(f"write-{payload_name}-{format}-draft{int(draft)}",
                     write, prompt, format=format, draft=draft)

    parallel_prompt = "Call lookup exactly three times, with query alpha, beta, and gamma respectively."
    for seed in (42, 43, 44):
        for draft in (True, False):
            send(f"parallel-seed{seed}-draft{int(draft)}", lookup, parallel_prompt,
                 format="xml", draft=draft, temperature=1, seed=seed)

    for left, right in (("literal-eos-draft1", "literal-eos-draft0"),):
        print(json.dumps({"comparison": [left, right],
                          "same_token_ids": records[left]["response"]["tensorfold"]["token_ids"]
                          == records[right]["response"]["tensorfold"]["token_ids"]}), flush=True)
    for seed in (42, 43, 44):
        left, right = (records[f"parallel-seed{seed}-draft{d}"] for d in (1, 0))
        print(json.dumps({"parallel_seed": seed,
                          "same_token_ids": left["response"]["tensorfold"]["token_ids"]
                          == right["response"]["tensorfold"]["token_ids"]}), flush=True)


if __name__ == "__main__":
    main()
