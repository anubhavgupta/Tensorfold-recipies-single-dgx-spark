#!/usr/bin/env python3
"""A client that leaves during a long prefill must free the server: send a ~LEN-token prompt, close the socket after HOLD
seconds, then time a one-line request. A server that prefills the abandoned prompt anyway answers it only after that prefill.
Env: BASE (http://127.0.0.1:8888), LEN (60000), HOLD (3). Prints the small request's time to its first token."""
import http.client, json, os, random, socket, time

HOST = os.environ.get("BASE", "http://127.0.0.1:8888").split("//")[1]
LEN, HOLD = int(os.environ.get("LEN", 60000)), float(os.environ.get("HOLD", 3))
random.seed(7)
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


def body(text, stream=True, max_tokens=8):
    return json.dumps({"model": "Qwen3.8-Flash-Next", "stream": stream, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}})


long_text = " ".join(random.choice(words) + str(random.randrange(10**6)) for _ in range(LEN // 3)) + "\nSay OK."
h, p = HOST.split(":")
s = socket.create_connection((h, int(p)))
payload = body(long_text).encode()
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
t0 = time.time()
time.sleep(HOLD)
s.close()
print(f"closed the long request after {HOLD}s")
c = http.client.HTTPConnection(h, int(p), timeout=600)
t1 = time.time()
c.request("POST", "/v1/chat/completions", body("Say hi.", stream=False), {"Content-Type": "application/json"})
r = json.loads(c.getresponse().read())
print(f"small request answered after {time.time() - t1:.1f}s: {r['choices'][0]['message']['content']!r}")

# resending the abandoned prompt resumes from a kept state when the server kept one at the cancel
if os.environ.get("RESEND", "1") == "1":
    c = http.client.HTTPConnection(h, int(p), timeout=900)
    t2 = time.time()
    c.request("POST", "/v1/chat/completions", body(long_text, stream=False), {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    if "usage" not in r:
        raise SystemExit(f"resend failed: {json.dumps(r)[:300]}")
    u = r["usage"]
    print(f"the long prompt resent: {u['prompt_tokens']} tokens, {u['prompt_tokens_details']['cached_tokens']} cached, {time.time() - t2:.1f}s")
