#!/usr/bin/env python3
"""Teacher-forced quality check: mean negative log-likelihood of fixed texts under the served model (stdlib only).

  nll.py score OUT.json [FILE ...]    score each file (default: a built-in set) and save every token's logprob
  nll.py compare A.json B.json        mean NLL of each, their difference, and per-token agreement

Works against this recipe's TensorFold server (/v1/completions with vLLM's prompt_logprobs) and the SGLang recipe
(ornith-1.5-122B: /generate with return_logprob), so one checkpoint's two implementations can be compared on the same
token ids. Env: URL (http://127.0.0.1:8888), MODEL (Ornith-1.5-122B), MAX_CHARS (12000 per text).
"""
import json, math, os, sys, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "Ornith-1.5-122B")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "12000"))
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.join(HERE, "..", "..", p) for p in ("README.md", "readme-detailed.md", "ornith-1.5-122B/bench/bench.py", "tensorfold.sh")]


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def engine():
    try:
        info = json.load(urllib.request.urlopen(URL + "/get_model_info", timeout=10))
        return "sglang" if info else "tensorfold"
    except Exception:
        return "tensorfold"


def score_text(text, kind):
    """(token ids, logprob of each token given the ones before it; None for the first)."""
    if kind == "sglang":
        r = post("/generate", {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                              "return_logprob": True, "logprob_start_len": 0})
        rows = r["meta_info"]["input_token_logprobs"]  # [logprob, id, text]
        return [row[1] for row in rows], [row[0] for row in rows]
    r = post("/v1/completions", {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    ids, lps = [], []
    for entry in r["choices"][0]["prompt_logprobs"]:
        if entry is None:
            ids.append(None); lps.append(None); continue
        tid, v = next(iter(entry.items()))
        ids.append(int(tid)); lps.append(v["logprob"])
    return ids, lps


def cmd_score(out, files):
    kind = engine()
    res = {"engine": kind, "texts": []}
    for f in files or DEFAULT:
        text = open(f, encoding="utf-8", errors="replace").read()[:MAX_CHARS]
        ids, lps = score_text(text, kind)
        vals = [x for x in lps if x is not None]
        nll = -sum(vals) / len(vals)
        print(f"{os.path.basename(f)}: {len(ids)} tokens, mean NLL {nll:.4f} (ppl {math.exp(nll):.3f})")
        res["texts"].append({"file": os.path.basename(f), "ids": ids, "logprobs": lps})
    json.dump(res, open(out, "w"))
    print(f"[{kind}] saved {out}")


def cmd_compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    tot = {"a": 0.0, "b": 0.0, "n": 0, "absdiff": 0.0}
    for ta, tb in zip(A["texts"], B["texts"]):
        # the first token has no logprob in either; SGLang's ids may lack TensorFold's leading null
        la = [x for x in ta["logprobs"] if x is not None]
        lb = [x for x in tb["logprobs"] if x is not None]
        n = min(len(la), len(lb))
        if len(la) != len(lb):
            print(f"  {ta['file']}: token counts differ ({len(la)} vs {len(lb)}); comparing the first {n}")
        sa, sb = -sum(la[:n]) / n, -sum(lb[:n]) / n
        d = sum(abs(x - y) for x, y in zip(la[:n], lb[:n])) / n
        print(f"{ta['file']}: NLL {A['engine']} {sa:.4f}  {B['engine']} {sb:.4f}  diff {sa - sb:+.4f}  mean |dlogp| {d:.4f}  ({n} tokens)")
        tot["a"] += sa * n; tot["b"] += sb * n; tot["n"] += n; tot["absdiff"] += d * n
    n = tot["n"]
    print(f"ALL: NLL {A['engine']} {tot['a'] / n:.4f}  {B['engine']} {tot['b'] / n:.4f}  diff {(tot['a'] - tot['b']) / n:+.4f}  mean |dlogp| {tot['absdiff'] / n:.4f}  ({n} tokens)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "score":
        cmd_score(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
