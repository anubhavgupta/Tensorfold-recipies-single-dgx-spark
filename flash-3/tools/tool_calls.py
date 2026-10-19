"""Boundary-aware Qwen tool calls, shared by final parsing and streaming."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from tensorfold.server.tools import parse_glm_tool_call_block
from tensorfold.tool_parameters import _python_literal, parameter_schemas


_OPEN = re.compile(r"<tool_call>", re.I)
_JSON_MARK = re.compile(r'["\\]|</tool_call>', re.I)
_XML_MARK = re.compile(r"<parameter=[^>\s]+>|</tool_call>", re.I)
_FUNCTION = re.compile(r"<function=([^>\s]+)>", re.I)
_FUNCTION_END = re.compile(r"</function>\s*$", re.I)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>", re.I)
_PARAMETER_END = re.compile(r"</parameter>", re.I)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Envelope:
    start: int
    end: int | None
    body: str


def envelopes(text: str):
    """Locate outer envelopes without interpreting delimiters inside values."""
    pos = 0
    while match := _OPEN.search(text, pos):
        start = match.end()
        i = start
        while i < len(text) and text[i].isspace():
            i += 1
        is_json = text[i:i + 1] in ("{", "[")
        quoted = False
        marks = _JSON_MARK if is_json else _XML_MARK
        complete = False
        while mark := marks.search(text, i):
            token = mark.group()
            i = mark.end()
            if is_json:
                if token == "\\" and quoted:
                    i += 1
                    continue
                if token == '"':
                    quoted = not quoted
                    continue
                if quoted or token == "\\":
                    continue
            elif token.lower().startswith("<parameter="):
                end = _PARAMETER_END.search(text, i)
                if end is None:
                    break
                i = end.end()
                continue
            yield Envelope(match.start(), mark.end(), text[start:mark.start()].strip())
            pos = mark.end()
            complete = True
            break
        if not complete:
            yield Envelope(match.start(), None, text[start:])
            return


def hide_tool_calls(text: str, *, finished: bool) -> str:
    out = []
    pos = 0
    for envelope in envelopes(text):
        out.append(text[pos:envelope.start])
        if envelope.end is None:
            return "".join(out)
        pos = envelope.end
    tail = text[pos:]
    if not finished:
        for count in range(min(len("<tool_call>") - 1, len(tail)), 0, -1):
            if tail.lower().endswith("<tool_call>"[:count]):
                tail = tail[:-count]
                break
    return "".join(out) + tail


def _types(schema: dict[str, Any]) -> set[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        kinds = {kind}
    elif isinstance(kind, list):
        kinds = {item for item in kind if isinstance(item, str)}
    else:
        kinds = set()
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    kinds.update(_types(branch))
    return kinds


def _decode(value: str, schema: dict[str, Any]) -> Any:
    kinds = _types(schema)
    if not kinds or kinds == {"string"}:
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        try:
            parsed = _python_literal(value)
        except ValueError:
            if "string" in kinds:
                return value
            raise ValueError("typed parameter is not a complete JSON/Python value") from None
    matches = {
        "array": isinstance(parsed, list),
        "object": isinstance(parsed, dict),
        "boolean": isinstance(parsed, bool),
        "integer": type(parsed) is int,
        "number": type(parsed) in (int, float),
        "null": parsed is None,
    }
    if any(matches.get(kind, False) for kind in kinds):
        json.dumps(parsed, allow_nan=False)
        return parsed
    if "string" in kinds:
        return value
    raise ValueError("parameter value does not match its declared types")


def _xml(block: str, schemas: dict[str, dict[str, Any]]):
    function = _FUNCTION.match(block)
    if function is None:
        return None
    name = function.group(1)
    props = schemas.get(name.lower(), {})
    args = {}
    pos = function.end()
    while True:
        while pos < len(block) and block[pos].isspace():
            pos += 1
        if _FUNCTION_END.match(block, pos):
            return name, args
        parameter = _PARAMETER.match(block, pos)
        if parameter is None:
            raise ValueError("missing parameter or function closing tag")
        end = _PARAMETER_END.search(block, parameter.end())
        if end is None:
            raise ValueError("missing parameter closing tag")
        key = parameter.group(1)
        if key in args:
            raise ValueError("duplicate parameter")
        value = block[parameter.end():end.start()]
        value = value.removeprefix("\n").removesuffix("\n")
        args[key] = _decode(value, props.get(key, {}))
        pos = end.end()


def _parse(text, tools, *, max_calls=None, warn=True):
    if not tools:
        return text, None
    known = {}
    for tool in tools:
        function = tool.get("function") or tool
        known[str(function.get("name", "")).lower()] = function
    schemas = parameter_schemas(tools)
    calls, residue = [], []
    pos = 0
    for envelope in envelopes(text):
        residue.append(text[pos:envelope.start])
        if envelope.end is None:
            if warn:
                _LOG.warning("Incomplete tool envelope; no call emitted for it")
            residue.append(text[envelope.start:])
            pos = len(text)
            break
        pos = envelope.end
        if max_calls is not None and len(calls) >= max_calls:
            continue
        try:
            if envelope.body.startswith(("{", "[")):
                payload = json.loads(envelope.body)
                if not isinstance(payload, dict):
                    raise ValueError("tool payload must be an object")
                function = payload.get("function") or payload
                if not isinstance(function, dict):
                    raise ValueError("function must be an object")
                name = function.get("name")
                args = function.get("arguments", function.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
            else:
                parsed = _xml(envelope.body, schemas)
                if parsed is None:
                    parsed = parse_glm_tool_call_block(envelope.body, tools, complete=True)
                if parsed is None:
                    raise ValueError("unrecognized tool payload")
                name, args = parsed
            spec = known.get(str(name).lower())
            if spec is None:
                raise ValueError("tool name was not offered")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            required = (spec.get("parameters") or {}).get("required") or []
            if any(key not in args for key in required):
                raise ValueError("required tool arguments are missing")
            arguments = json.dumps(args, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as exc:
            if warn:
                _LOG.warning("Invalid tool call; not emitted: %s", exc)
            residue.append(text[envelope.start:envelope.end])
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": spec["name"], "arguments": arguments}})
    residue.append(text[pos:])
    return "".join(residue).strip(), calls or None


def parse_tool_calls(text, tools, *, max_calls=None):
    return _parse(text, tools, max_calls=max_calls)


class ToolCallStreamer:
    """Emit each complete call atomically using the final parser's decisions."""

    def __init__(self, tools):
        self.tools = tools
        self.index = -1
        self.streamed = False
        self.pos = 0

    def feed(self, text):
        out = []
        completed = self.pos
        for envelope in envelopes(text[self.pos:]):
            if envelope.end is None:
                break
            end = self.pos + envelope.end
            completed = end
            block = text[self.pos + envelope.start:end]
            _, calls = _parse(block, self.tools, warn=False)
            if calls:
                self.index += 1
                self.streamed = True
                out.append({"tool_calls": [{"index": self.index, **calls[0]}]})
        self.pos = completed
        return out
