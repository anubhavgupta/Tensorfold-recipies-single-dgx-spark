#!/usr/bin/env python3
"""Logging proxy in front of the TensorFold server: finds why prompts miss the prefix cache.

A cache hit needs the new prompt to start with exactly the tokens of a stored one. For every
/v1/chat/completions request this proxy compares tools + messages with the last HISTORY requests,
picks the most similar one, and reports where the two first differ (segment, offset, a snippet).
The server's own `cached=` for that request is in `docker logs`; match them by time or order.

Usage: python3 prefix_proxy.py [--listen 8889] [--upstream http://127.0.0.1:8888]
Logs:  stdout and <log-dir>/requests.jsonl (default ./proxy-logs); request bodies in <log-dir>/bodies/
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

HISTORY = 20
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "host", "content-length", "content-encoding"}
lock = threading.Lock()
recent = []          # (id, segments)
counter = [0]


def segments(body):
    """Prompt order of the Qwen template: tools (in the system block), then each message."""
    segs = []
    if body.get("tools"):
        segs.append(("tools", json.dumps(body["tools"], sort_keys=True, ensure_ascii=False)))
    for i, m in enumerate(body.get("messages") or []):
        segs.append((f"msg[{i}:{m.get('role')}]", json.dumps(m, sort_keys=True, ensure_ascii=False)))
    return segs


def compare(a, b):
    """(common chars, segment index where they differ or None, offset in it) for segment lists a, b."""
    common = 0
    for i, ((na, sa), (nb, sb)) in enumerate(zip(a, b)):
        if sa == sb:
            common += len(sa)
            continue
        n = 0
        for x, y in zip(sa, sb):
            if x != y:
                break
            n += 1
        return common + n, i, n
    return common, None, 0


def analyse(rid, body):
    segs = segments(body)
    total = sum(len(s) for _, s in segs)
    best = None
    with lock:
        for oid, osegs in recent:
            common, idx, off = compare(osegs, segs)
            if best is None or common > best[1]:
                best = (oid, common, idx, off, osegs)
        recent.append((rid, segs))
        del recent[:-HISTORY]
    rep = {"id": rid, "chars": total, "segments": len(segs), "model": body.get("model"),
           "stream": bool(body.get("stream")), "tools": len(body.get("tools") or [])}
    if best is None:
        rep["verdict"] = "first request"
        return rep
    oid, common, idx, off, osegs = best
    rep["closest"] = oid
    rep["common_chars"] = common
    if idx is None:
        rep["verdict"] = ("extends the earlier request (cacheable)" if len(segs) >= len(osegs)
                          else "is a prefix of the earlier request")
        rep["old_segments"] = len(osegs)
        return rep
    name_new, new_s = segs[idx]
    name_old, old_s = osegs[idx] if idx < len(osegs) else ("", "")
    rep["verdict"] = f"DIFFERS from request {oid} at {name_new} offset {off} ({common}/{total} chars shared)"
    rep["old_total_chars"] = sum(len(s) for _, s in osegs)
    lo = max(0, off - 60)
    rep["old_snippet"] = old_s[lo:off + 120]
    rep["new_snippet"] = new_s[lo:off + 120]
    return rep


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ""
    log_dir = ""

    def log_message(self, *a):
        pass

    def _forward(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""
        if self.command == "POST" and self.path.split("?")[0].endswith("/chat/completions"):
            try:
                body = json.loads(data)
                with lock:
                    counter[0] += 1
                    rid = counter[0]
                os.makedirs(os.path.join(self.log_dir, "bodies"), exist_ok=True)
                with open(os.path.join(self.log_dir, "bodies", f"{rid:05d}.json"), "wb") as f:
                    f.write(data)
                rep = analyse(rid, body)
                rep["time"] = time.strftime("%H:%M:%S")
                print(json.dumps(rep, ensure_ascii=False), flush=True)
                with open(os.path.join(self.log_dir, "requests.jsonl"), "a") as f:
                    f.write(json.dumps(rep, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - logging must never break forwarding
                print(f"[proxy] analysis failed: {type(exc).__name__}: {exc}", flush=True)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            r = requests.request(self.command, self.upstream + self.path, data=data, headers=headers,
                                 stream=True, timeout=(10, 3600))
        except Exception as exc:  # noqa: BLE001
            msg = json.dumps({"error": f"proxy: {exc}"}).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(r.status_code)
        for k, v in r.headers.items():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in r.raw.stream(8192, decode_content=True):
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            r.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = _forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8889)
    ap.add_argument("--upstream", default="http://127.0.0.1:8888")
    ap.add_argument("--log-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy-logs"))
    a = ap.parse_args()
    Handler.upstream = a.upstream.rstrip("/")
    Handler.log_dir = a.log_dir
    os.makedirs(a.log_dir, exist_ok=True)
    print(f"[proxy] 0.0.0.0:{a.listen} -> {Handler.upstream}, logs in {a.log_dir}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
