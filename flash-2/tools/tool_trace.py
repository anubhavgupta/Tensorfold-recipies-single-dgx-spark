"""Opt-in, local-only capture of CUDA tool requests without changing parsing."""

import copy
import functools
import json
import logging
import os
import time
import uuid
from pathlib import Path


def capture_tool_calls(run):
    @functools.wraps(run)
    def traced(self, body, chat, emit, **kwargs):
        directory = os.environ.get("TENSORFOLD_TOOL_TRACE_DIR")
        if not directory or not chat or not body.get("tools"):
            return run(self, body, chat, emit, **kwargs)

        trace = {
            "version": 1,
            "id": uuid.uuid4().hex,
            "started_at": time.time(),
            "request": copy.deepcopy(body),
            "events": [],
            "raw_text": None,
            "result": None,
        }
        prepared = kwargs.get("prepared")
        if prepared is not None:
            trace["effective"] = {
                "thinking": prepared.thinking,
                "max_tokens": prepared.max_tokens,
                "stop": list(prepared.stop),
                "tools": prepared.tools,
            }

        def capture(delta):
            event = {"elapsed": time.time() - trace["started_at"], "delta": copy.deepcopy(delta)}
            trace["events"].append(event)
            accepted = emit(delta)
            event["accepted"] = accepted
            return accepted

        try:
            result = run(self, {**body, "return_token_ids": True}, chat, capture, **kwargs)
            trace["result"] = copy.deepcopy(result)
            ids = result["stats"]["token_ids"]
            trace["raw_text"] = self.tok.decode(ids, skip_special_tokens=False)
            # Token IDs are requested internally, not added to the client's response.
            if not body.get("return_token_ids"):
                result["stats"].pop("token_ids")
            return result
        except Exception as exc:
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            trace["finished_at"] = time.time()
            try:
                root = Path(directory)
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
                path = root / f"{trace['id']}.json"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as output:
                    json.dump(trace, output, ensure_ascii=False)
                    output.write("\n")
                logging.getLogger(__name__).warning("Tool trace saved: %s", path)
            except (OSError, TypeError, ValueError):
                logging.getLogger(__name__).exception("Could not save tool trace %s", trace["id"])

    return traced
