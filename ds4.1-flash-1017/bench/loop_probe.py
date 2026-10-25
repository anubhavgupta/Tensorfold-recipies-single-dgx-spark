#!/usr/bin/env python3
"""Repetition-loop probe: sends pi-style chat requests and scores the reply for loops.

pi (and similar agent clients) stream, send no temperature/top_p (the server defaults apply)
and pass thinking through chat_template_kwargs. This sends the same shape, N samples per
prompt, and reports per sample: finish reason, reasoning/content tokens, time, the most
repeated sentence (count) and the distinct share of word 6-grams in the reply's tail.
LOOP = distinct < 0.5 or one sentence >= 6 times; STUCK = hit the cap still thinking.

  python3 bench/loop_probe.py                       # all prompts, 3 samples, server defaults
  python3 bench/loop_probe.py -n 5 -p story --set repetition_penalty=1.05
  python3 bench/loop_probe.py --effort high --max-tokens 8192
"""
import argparse, collections, json, os, re, time, urllib.request

PORT = os.environ.get("API_PORT", "8888")
MODEL = os.environ.get("SERVED_MODEL_NAME", "dsv41")

SYSTEM = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
          "You help users by reading files, executing commands, editing code, and writing new files.")
PROMPTS = {
    "story": "Create a long story about elephants in india",
    "todo": "Create a todo app as a single html file. Reply with the full file.",
    "essay": "Write a detailed 2000-word explanation of how TCP congestion control works, "
             "covering slow start, AIMD, fast retransmit, CUBIC and BBR.",
}


def score(text):
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n", text) if len(s.strip()) > 25]
    top = collections.Counter(sents).most_common(1)
    words = text.split()[-1500:]
    grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    return (top[0][1] if top else 0), distinct


def run(prompt, args):
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True},
            "max_completion_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": not args.no_think}}
    if args.effort and not args.no_think:
        body["chat_template_kwargs"]["reasoning_effort"] = args.effort
    for kv in args.set:
        k, v = kv.split("=", 1)
        body[k] = json.loads(v)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    reasoning, content, finish, usage = [], [], None, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                finish = c.get("finish_reason") or finish
    return "".join(reasoning), "".join(content), finish, usage, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("-p", "--prompts", default=",".join(PROMPTS))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--set", action="append", default=[], help="extra request field, KEY=JSON")
    ap.add_argument("--save", help="append raw samples (jsonl) here")
    args = ap.parse_args()
    loops = total = 0
    for name in args.prompts.split(","):
        for i in range(args.n):
            rs, ct, fin, use, dt = run(PROMPTS[name], args)
            rep_r, dis_r = score(rs)
            rep_c, dis_c = score(ct)
            # LOOP: verbatim repetition; STUCK: hit the cap without leaving thinking.
            loop = min(dis_r, dis_c) < 0.5 or max(rep_r, rep_c) >= 6
            stuck = fin == "length" and not ct
            loops += loop or stuck
            total += 1
            print(f"{name:6s} #{i} finish={fin:6s} tokens={use.get('completion_tokens')} "
                  f"reasoning_chars={len(rs)} content_chars={len(ct)} {dt:5.0f}s "
                  f"think(rep={rep_r},distinct={dis_r:.2f}) reply(rep={rep_c},distinct={dis_c:.2f})"
                  f"{'  LOOP' if loop else ''}{'  STUCK' if stuck else ''}", flush=True)
            if args.save:
                with open(args.save, "a") as f:
                    f.write(json.dumps({"prompt": name, "set": args.set, "effort": args.effort,
                                        "finish": fin, "usage": use, "reasoning": rs, "content": ct}) + "\n")
    print(f"loops {loops}/{total}  ({' '.join(args.set) or 'server defaults'})")


if __name__ == "__main__":
    main()
