"""Compare the server's prompt_tokens with Hugging Face apply_chat_template on the model's chat_template.jinja."""
import glob, json, os, sys, urllib.request
from transformers import AutoTokenizer

URL = "http://127.0.0.1:8888/v1/chat/completions"
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-122B-NVFP4/snapshots/*/"))[0]
tok = AutoTokenizer.from_pretrained(snap)
W = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
U = {"role": "user", "content": "Pick a prime between 10 and 20."}
A = {"role": "assistant", "content": "13", "reasoning_content": "Primes 11,13,17,19. Pick 13."}
U2 = {"role": "user", "content": "Double it."}
CALL = {"role": "assistant", "content": "", "reasoning_content": "Use the tool.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 18}"}
cases = {
    "user": ([{"role": "user", "content": "Hi"}], None, {}),
    "system+user": ([{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}], None, {}),
    "think_off": ([{"role": "user", "content": "Hi"}], None, {"enable_thinking": False}),
    "history_reasoning": ([U, A, U2], None, {}),
    "history_think_in_content": ([U, {"role": "assistant", "content": "<think>\nPrimes. Pick 13.\n</think>\n\n13"}, U2], None, {}),
    "tools": ([{"role": "user", "content": "Weather in Paris?"}], W, {}),
    "tool_loop": ([{"role": "user", "content": "Weather in Paris?"}, CALL, TOOL], W, {}),
}
bad = 0
for name, (msgs, tools, kw) in cases.items():
    ref = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True, **kw)
    ref = ref["input_ids"] if isinstance(ref, dict) or hasattr(ref, "keys") else ref
    sm = [dict(m, tool_calls=[dict(c, function=dict(c["function"], arguments=json.dumps(c["function"]["arguments"]))) for c in m["tool_calls"]]) if m.get("tool_calls") else m for m in msgs]
    body = {"model": "x", "messages": sm, "max_tokens": 1, "chat_template_kwargs": kw}
    if tools: body["tools"] = tools
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})))
    n = r["usage"]["prompt_tokens"]
    ok = n == len(ref); bad += not ok
    print(f"{'ok  ' if ok else 'DIFF'} {name}: server {n}, HF template {len(ref)}")
    if name == "tool_loop": print("   tail:", repr(tok.decode(ref[-40:])))
sys.exit(1 if bad else 0)
