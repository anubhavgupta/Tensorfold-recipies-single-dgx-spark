"""CPU-only stock characterization, parser-fix, launcher and capture regressions."""

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tensorfold.cuda.reply_text import parse_tool_calls
from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.cuda.tool_trace import capture_tool_calls
from tensorfold.cuda import tool_calls as fixed
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.text import split_thinking


TOOLS = [{"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "limit": {"type": "integer"}}},
}}]
XML = (
    "<tool_call>\n<function=lookup>\n<parameter=query>\nhello\n</parameter>\n"
    "<parameter=limit>\n2\n</parameter>\n</function>\n</tool_call>"
)
JSON_CALL = '<tool_call>{"name":"lookup","arguments":{"query":"hello","limit":2}}</tool_call>'


def deltas(text):
    stream = ToolCallStreamer(TOOLS)
    events = []
    for end in range(1, len(text) + 1):
        events.extend(stream.feed(text[:end]))
    return events


class ParserCases(unittest.TestCase):
    def test_supported_formats(self):
        for text in (XML, JSON_CALL):
            with self.subTest(text=text):
                content, calls = parse_tool_calls(text, TOOLS)
                self.assertEqual(content, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                                 {"query": "hello", "limit": 2})

    def test_xml_stream_arguments_match_final(self):
        fragments = [c["function"].get("arguments", "")
                     for event in deltas(XML) for c in event["tool_calls"]]
        self.assertEqual(json.loads("".join(fragments)), {"query": "hello", "limit": 2})

    def test_json_falls_back_to_final_parser(self):
        self.assertEqual(deltas(JSON_CALL), [])
        self.assertIsNotNone(parse_tool_calls(JSON_CALL, TOOLS)[1])

    def test_missing_closers_expose_stream_final_disagreement(self):
        for text in (XML.removesuffix("</tool_call>"), XML.replace("</function>", "")):
            with self.subTest(text=text):
                self.assertTrue(deltas(text))
                self.assertIsNone(parse_tool_calls(text, TOOLS)[1])

    def test_unrecognized_envelopes(self):
        for text in (
            JSON_CALL.removeprefix("<tool_call>").removesuffix("</tool_call>"),
            XML.replace("<tool_call>", "").replace("</tool_call>", ""),
            XML.replace("<tool_call>", "<x:tool_call>").replace("</tool_call>", "</x:tool_call>"),
            XML.replace("lookup", "unknown"),
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_tool_calls(text, TOOLS)[1])

    def test_thinking_and_parallel_policy(self):
        for text in ("<think>plan</think>\n" + XML, "<think>plan\n" + XML):
            answer = split_thinking(text, finished=True)[1]
            self.assertEqual(len(parse_tool_calls(answer, TOOLS)[1]), 1)
        self.assertEqual(len(parse_tool_calls(XML + XML, TOOLS)[1]), 2)
        self.assertEqual(len(parse_tool_calls(XML + XML, TOOLS, max_calls=1)[1]), 1)

    def test_union_schema_values_are_incorrectly_left_as_strings(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {
                "query": {"anyOf": [{"type": "array"}, {"type": "null"}]},
                "limit": {"type": ["integer", "null"]},
            }},
        }}]
        text = XML.replace("hello", '["a","b"]')
        args = json.loads(parse_tool_calls(text, tools)[1][0]["function"]["arguments"])
        self.assertEqual(args, {"query": '["a","b"]', "limit": "2"})
        streamer = ToolCallStreamer(tools)
        events = streamer.feed(text)
        fragments = [c["function"].get("arguments", "") for e in events for c in e["tool_calls"]]
        self.assertEqual(json.loads("".join(fragments)), args)

    def test_literal_tool_close_corrupts_final_xml_arguments(self):
        literal = "<tool_call><function=example></function></tool_call>"
        text = XML.replace("hello", literal)
        args = json.loads(parse_tool_calls(text, TOOLS)[1][0]["function"]["arguments"])
        self.assertEqual(args, {})
        fragments = [c["function"].get("arguments", "")
                     for event in deltas(text) for c in event["tool_calls"]]
        self.assertEqual(json.loads("".join(fragments)), {"query": literal, "limit": 2})

    def test_literal_tool_close_makes_json_call_disappear(self):
        text = JSON_CALL.replace("hello", "</tool_call>")
        self.assertIsNone(parse_tool_calls(text, TOOLS)[1])
        self.assertEqual(deltas(text), [])


class FixedParserCases(unittest.TestCase):
    def stream(self, text):
        streamer = fixed.ToolCallStreamer(TOOLS)
        events = []
        for end in range(1, len(text) + 1):
            events.extend(streamer.feed(text[:end]))
        return [call for event in events for call in event["tool_calls"]]

    def test_complete_xml_and_json_calls_stream_identically(self):
        for text in (XML, JSON_CALL, XML + JSON_CALL + XML):
            with self.subTest(text=text):
                calls = fixed.parse_tool_calls(text, TOOLS)[1]
                streamed = self.stream(text)
                self.assertEqual([c["function"] for c in calls], [c["function"] for c in streamed])
                self.assertEqual([c["index"] for c in streamed], list(range(len(calls))))

    def test_literal_delimiters_preserve_arguments_and_do_not_leak(self):
        literal = '<tool_call><function=example></function></tool_call>'
        for text in (XML.replace("hello", literal), JSON_CALL.replace("hello", literal)):
            with self.subTest(text=text):
                args = json.loads(fixed.parse_tool_calls(text, TOOLS)[1][0]["function"]["arguments"])
                self.assertEqual(args, {"query": literal, "limit": 2})
                self.assertEqual(self.stream(text)[0]["function"]["arguments"],
                                 fixed.parse_tool_calls(text, TOOLS)[1][0]["function"]["arguments"])
                for end in range(1, len(text) + 1):
                    self.assertEqual(fixed.hide_tool_calls(text[:end], finished=False), "")
                self.assertEqual(fixed.hide_tool_calls("before " + text + " after", finished=True),
                                 "before  after")

    def test_json_strings_with_escaped_quotes_and_backslashes(self):
        values = ['a"</tool_call>b', 'a\\"</tool_call>b', "\\\\</tool_call>",
                  "h\u00e9llo\n</tool_call>", "</function><parameter=query>literal"]
        for value in values:
            text = "<tool_call>" + json.dumps({"name": "lookup", "arguments": {"query": value}}) + "</tool_call>"
            with self.subTest(value=value):
                calls = fixed.parse_tool_calls(text, TOOLS)[1]
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"query": value})
                self.assertEqual(len(self.stream(text)), 1)

    def test_truncated_calls_never_emit(self):
        for text in (XML, JSON_CALL):
            for end in range(len(text)):
                with self.subTest(end=end, text=text):
                    self.assertEqual(self.stream(text[:end]), [])
        for text in (XML.removesuffix("</tool_call>"), XML.replace("</function>", "")):
            with self.assertLogs(fixed.__name__, level="WARNING"):
                self.assertIsNone(fixed.parse_tool_calls(text, TOOLS)[1])

    def test_invalid_call_does_not_shift_stream_indices(self):
        text = XML.replace("lookup", "unknown") + JSON_CALL + XML
        with self.assertLogs(fixed.__name__, level="WARNING"):
            final = fixed.parse_tool_calls(text, TOOLS)[1]
        streamed = self.stream(text)
        self.assertEqual(len(final), 2)
        self.assertEqual([c["function"] for c in final], [c["function"] for c in streamed])
        self.assertEqual([c["index"] for c in streamed], [0, 1])

    def test_union_schemas_and_string_preservation(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"properties": {
                "query": {"anyOf": [{"type": "array"}, {"type": "null"}]},
                "limit": {"type": ["integer", "null"]},
            }},
        }}]
        for query, limit, expected in (
            ('["a","b"]', "2", {"query": ["a", "b"], "limit": 2}),
            ("null", "null", {"query": None, "limit": None}),
        ):
            text = XML.replace("hello", query).replace("\n2\n", f"\n{limit}\n")
            result = fixed.parse_tool_calls(text, tools)[1]
            self.assertEqual(json.loads(result[0]["function"]["arguments"]), expected)
        for value in ("3", "null", '["a"]', '"quoted"', " trailing \n"):
            text = XML.replace("hello", value)
            args = json.loads(fixed.parse_tool_calls(text, TOOLS)[1][0]["function"]["arguments"])
            self.assertEqual(args["query"], value)

    def test_bad_typed_and_missing_required_arguments_are_rejected(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"required": ["query"], "properties": {
                "query": {"type": ["array", "null"]}, "limit": {"type": "integer"},
            }},
        }}]
        for text in (XML, XML.replace("hello", '["a"'), XML.replace("hello", '["a"]').replace("\n2\n", "\ntrue\n")):
            with self.assertLogs(fixed.__name__, level="WARNING"):
                self.assertIsNone(fixed.parse_tool_calls(text, tools)[1])
        for block in ('{"name":"lookup","arguments":{}}',
                      '{"name":"lookup","arguments":[]}', '{"name":"lookup","arguments":{"query":NaN}}'):
            with self.assertLogs(fixed.__name__, level="WARNING"):
                self.assertIsNone(fixed.parse_tool_calls(f"<tool_call>{block}</tool_call>", tools)[1])

    def test_parallel_limit_and_case_insensitive_tags(self):
        text = XML.replace("tool_call", "TOOL_CALL").replace("parameter", "PARAMETER")
        self.assertEqual(len(fixed.parse_tool_calls(text + text, TOOLS, max_calls=1)[1]), 1)
        self.assertEqual(len(self.stream(text + text)), 2)
        self.assertEqual(fixed.parse_tool_calls("plain", [])[0], "plain")


class LauncherCases(unittest.TestCase):
    def launch(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "flash-2"
            recipe.mkdir()
            shutil.copy2(Path(__file__).resolve().parents[1] / "start.sh", recipe / "start.sh")
            wrapper = root / "tensorfold.sh"
            wrapper.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "print(json.dumps({'args':sys.argv[1:],'trace':os.getenv('TENSORFOLD_TOOL_TRACE_DIR')}))\n"
            )
            wrapper.chmod(0o755)
            binary = root / "bin"
            binary.mkdir()
            docker = binary / "docker"
            docker.write_text("#!/bin/sh\nexit 0\n")
            docker.chmod(0o755)
            env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
                   "ENV_FILE": str(root / "absent"), "MODEL_DIR": "/models/test",
                   "TOOL_PARSER_FIX": "1", "TOOL_TRACE": "0", "PATCHES": "0",
                   "TF_VERSION": "0.6.5", "VISION": "0", "FOREGROUND": "0",
                   "TOOL_TRACE_DIR": str(root / "captures"), **overrides}
            env.pop("TENSORFOLD_TOOL_TRACE_DIR", None)
            result = subprocess.run(["bash", str(recipe / "start.sh")], env=env,
                                    text=True, capture_output=True)
            replies = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            return result, replies[0] if replies else None

    def test_default_fix_and_optional_trace_share_one_patch_set(self):
        for trace in ("0", "1"):
            result, reply = self.launch(TOOL_TRACE=trace)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(reply["args"].count("--tf-patches"), 1)
            self.assertTrue(reply["args"][reply["args"].index("--tf-patches") + 1].endswith("/flash-2/patches"))
            self.assertTrue(any("tool_calls.py:" in arg for arg in reply["args"]))
            self.assertEqual(reply["trace"], "/tool-traces" if trace == "1" else None)

    def test_stock_parser_and_trace_only_rollbacks(self):
        for trace in ("0", "1"):
            result, reply = self.launch(TOOL_PARSER_FIX="0", TOOL_TRACE=trace)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any("tool_calls.py:" in arg for arg in reply["args"]))
            self.assertEqual(reply["args"].count("--tf-patches"), int(trace))
            if trace == "1":
                self.assertTrue(reply["args"][reply["args"].index("--tf-patches") + 1].endswith("/tools/patches"))

    def test_unsupported_build_fails_explicitly(self):
        for overrides in ({"TF_VERSION": "0.6.6"}, {"PATCHES": "1"}):
            result, reply = self.launch(**overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(reply)
            self.assertIn("TOOL_PARSER_FIX=0", result.stderr)
        result, reply = self.launch(TOOL_PARSER_FIX="0", TF_VERSION="0.6.6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(reply)


class TraceCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "captures"
        env = patch.dict(os.environ, {"TENSORFOLD_TOOL_TRACE_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def read_capture(self):
        paths = list(self.root.glob("*.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(paths[0].stat().st_mode), 0o600)
        return json.loads(paths[0].read_text())

    def run_server(self, text, body, traced=True):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return "".join(chr(i) for i in ids)

            def token_to_id(self, token):
                return None

        class Engine:
            concurrent = True
            eos = ()

            def generate(self, ids, count, sampling, feed):
                feed([ord(c) for c in text])
                return {}

        app = App.__new__(App)
        app.tok, app.engine = Tokenizer(), Engine()
        prepared = PreparedRequest(prompt=[1], max_tokens=512, tools=TOOLS, thinking=False, sampling=None)
        events = []
        with patch("tensorfold.cuda.server.health.of", return_value=MagicMock()), \
             patch("tensorfold.cuda.server.print_done"):
            run = App.run if traced else App.run.__wrapped__
            result = run(app, body, True, lambda d: events.append(d) or True, prepared=prepared)
        return result, events

    def test_capture_real_server_run_preserves_result_and_request(self):
        body = {"tools": TOOLS, "messages": [{"role": "user", "content": "lookup hello"}]}
        result, events = self.run_server(XML, body)
        trace = self.read_capture()
        self.assertEqual(trace["raw_text"], XML)
        self.assertEqual(trace["request"], body)
        self.assertNotIn("return_token_ids", body)
        self.assertNotIn("token_ids", result["stats"])
        self.assertEqual(trace["result"]["stats"]["token_ids"], [ord(c) for c in XML])
        self.assertEqual(trace["effective"]["max_tokens"], 512)
        self.assertEqual([e["delta"] for e in trace["events"]], events)
        self.assertEqual(result["finish"], "tool_calls")
        self.assertEqual(len(result["calls"]), 1)

    def test_capture_preserves_results_across_failure_modes(self):
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items() if k != "id"}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value

        for text in (XML, JSON_CALL, XML.removesuffix("</tool_call>")):
            for parallel in (False, True):
                for stream in (False, True):
                    with self.subTest(text=text, parallel=parallel, stream=stream):
                        body = {"tools": TOOLS, "parallel_tool_calls": parallel, "stream": stream}
                        stock, stock_events = self.run_server(text, body, traced=False)
                        result, events = self.run_server(text, body)
                        self.assertEqual(stable(result), stable(stock))
                        self.assertEqual(stable(events), stable(stock_events))
        paths = list(self.root.glob("*.json"))
        self.assertEqual(len(paths), 12)
        incomplete = [json.loads(p.read_text()) for p in paths
                      if json.loads(p.read_text())["raw_text"] == XML.removesuffix("</tool_call>")]
        self.assertTrue(all(t["result"]["calls"] is None for t in incomplete))
        self.assertFalse(any(any(e["delta"].get("tool_calls") for e in t["events"]) for t in incomplete))

    def test_disabled_and_non_tool_requests_are_untouched(self):
        run = MagicMock(return_value={"ok": True})
        wrapped = capture_tool_calls(run)
        for directory, body, chat in (("", {"tools": TOOLS}, True),
                                      (str(self.root), {}, True),
                                      (str(self.root), {"tools": TOOLS}, False)):
            with patch.dict(os.environ, {"TENSORFOLD_TOOL_TRACE_DIR": directory}):
                wrapped(None, body, chat, None)
                self.assertIs(run.call_args.args[1], body)
        self.assertFalse(self.root.exists())

    def test_errors_are_recorded_and_reraised(self):
        def fail(self, body, chat, emit):
            emit({"content": "partial"})
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            capture_tool_calls(fail)(None, {"tools": TOOLS}, True, lambda d: True)
        trace = self.read_capture()
        self.assertEqual(trace["error"]["type"], "RuntimeError")
        self.assertIsNone(trace["raw_text"])
        self.assertEqual(len(trace["events"]), 1)

    def test_write_failure_is_visible_without_changing_response(self):
        def run(self, body, chat, emit):
            return {"stats": {"token_ids": [65]}}

        app = SimpleNamespace(tok=SimpleNamespace(decode=lambda ids, **kw: "A"))
        with patch("tensorfold.cuda.tool_trace.os.open", side_effect=OSError("disk full")), \
             self.assertLogs("tensorfold.cuda.tool_trace", level="ERROR") as logs:
            result = capture_tool_calls(run)(app, {"tools": TOOLS}, True, lambda d: True)
        self.assertIn("Could not save tool trace", logs.output[0])
        self.assertEqual(result, {"stats": {}})

    def test_explicit_token_ids_are_preserved(self):
        def run(self, body, chat, emit):
            return {"stats": {"token_ids": [65]}}

        app = SimpleNamespace(tok=SimpleNamespace(decode=lambda ids, **kw: "A"))
        result = capture_tool_calls(run)(
            app, {"tools": TOOLS, "return_token_ids": True}, True, lambda d: True)
        self.assertEqual(result["stats"]["token_ids"], [65])


if __name__ == "__main__":
    unittest.main()
