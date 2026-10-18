# EXL3 prompt-speed overlay for TensorFold 0.6.5

## Tool-call parser fix (enabled by default)

`TOOL_PARSER_FIX=1` applies `patches/0001-tool-parser.patch` and mounts
`tools/tool_calls.py`. It requires TensorFold 0.6.5 and `PATCHES=0`, independently
of `OVERLAY`. Set `TOOL_PARSER_FIX=0` to restore stock parsing (also required when
selecting a different TensorFold version or the legacy `PATCHES=1`).

The CUDA server uses one parser for final results and streamed calls. Envelope
boundaries respect JSON quoting/escapes and XML parameter values, so a literal
`</tool_call>` inside an argument does not end the outer call. XML decoding handles
nullable type arrays and `anyOf`/`oneOf` primitive types, preserving declared
string values. This is type decoding, not full JSON Schema validation or
grammar-constrained generation.

**Intentional streaming change:** a tool event contains the complete argument
JSON and is emitted only after its whole envelope closes. Reasoning and ordinary
text still stream normally. `parallel_tool_calls=false` still limits the result
to one call. Incomplete envelopes, malformed typed values, missing required
arguments, duplicate XML parameters and unknown tool names are logged and not
emitted. Missing JSON brackets are not guessed or repaired. Genuine model
omissions or premature EOS remain model-generation failures.

## Tool-call diagnostics (opt-in)

From the `flash-2` directory, run `TOOL_TRACE=1 ./start.sh` and reproduce the
missed call through the client's usual endpoint. Stop an existing recipe server
with `./end.sh` before restarting. This requires TensorFold 0.6.5 and `PATCHES=0`,
works with either `OVERLAY=0` or `1`. Tracing itself changes neither parsing nor
sampling; the independent parser fix above is enabled by default.
Restart without `TOOL_TRACE=1` to disable it.

One JSON file per tool-enabled chat request is written to `../tool-traces/`
(relative to this README), or the host's `TOOL_TRACE_DIR`. Files contain the
request, effective prepared tool/thinking/stop settings, exact generated token
IDs and decoded text (including thinking and end tokens), emitted internal
tool/content events, and final parser result. These are server-side events,
not a packet capture of the client's received SSE stream. Failed/cancelled
requests record the error and events so far; raw tokens are unavailable if
generation does not return.

**Sensitive:** captures include full conversation history and tool arguments,
possibly credentials or private media URLs. No HTTP headers are captured.
The directory is mode 0700 and files mode 0600, ignored by Git at the default
location. Container-created files may be owned by root. Custom locations must
also be kept out of Git. There is no automatic rotation: enable only while
reproducing, then disable and remove specific capture files when no longer needed.

Compare `raw_text`, `events[].delta.tool_calls`, `result.calls`, `result.finish`
and `result.calls_streamed`. A complete call in raw text but absent from final
calls indicates parsing; emitted calls but absent final calls indicate a
stream/final disagreement. Neither means the client executed a tool.
`finish: "length"` suggests truncation, not permission to repair or execute
incomplete arguments.

Run the CPU-only regression cases in the installed image from `flash-2`:

```bash
docker run --rm --network none --entrypoint bash \
  -v "$PWD:/recipe:ro" tensorfold:v0.6.5 -c \
  'cd /usr/local/lib/python3.12/dist-packages &&
   patch -p0 < /recipe/patches/0001-tool-parser.patch &&
   cp /recipe/tools/tool_trace.py /recipe/tools/tool_calls.py tensorfold/cuda/ &&
   PYTHONDONTWRITEBYTECODE=1 python3 /recipe/tools/test_tool_calls.py'
```

The suite characterizes the stock defects and verifies the fixed parser,
including every partial prefix, escaped JSON strings, unions, plain string
preservation, mixed XML/JSON calls, parallel indices and trace transparency.

### Live reproduction results (2026-10-18)

`tools/repro_tool_calls.py` sends synthetic requests to the actual model and saves
the HTTP responses beside the server captures (files prefixed `http-`). It does
not execute any generated tool. Run from `flash-2` while tracing is enabled:

```bash
python3 tools/repro_tool_calls.py
# Narrow reproduction:
python3 tools/repro_tool_calls.py --case nullable --case xml_in_content --case json_xml_in_arguments
```

The initial investigation made 44 requests: the eight base cases with thinking
on/off and streaming on/off, plus JSON-mode nullable, literal markup and explicit
JSON-envelope follow-ups. Recipe sampling defaults were retained, seed 42 was
requested, and the normal token limit was 1024 (18 for the deliberate truncation
case). Request IDs differ, so different requests do not establish identical
generation across modes. Raw server captures identify whether each failure was
already present in generation.

| Reproduction | Observed result | Cause |
|---|---|---|
| Nullable/union schemas, XML format | 4/4 responses returned integer/array arguments as strings | `tool_parameters.typed_parameter` only recognizes a single string-valued `type`; type arrays and `anyOf` are treated as untyped text |
| Literal tool markup inside an XML string argument | All 4 final parser results had `{}`; streamed HTTP responses retained the correct argument | `cuda/reply_text.py` uses a non-greedy regex ending at the first `</tool_call>`, even inside a parameter value |
| Correct JSON envelope with literal tool markup, thinking on | 2/2 complete raw calls became no calls, with HTTP `finish_reason: "stop"` | The same regex terminates inside a JSON string, causing JSON decoding to fail |
| Same explicit JSON-envelope request, thinking off | 2/2 generations ended inside the string before completing JSON | Model emitted EOS early; distinct from the parser dropping a complete call |
| Nullable/union schemas, JSON format | 4/4 returned correctly typed arguments | JSON parsing bypasses the deficient XML parameter type selection |
| Deliberate 18-token limit | Final parser returned no calls; one stream exposed incomplete argument JSON | Truncated generation plus independent streaming/final parsers; do not execute incomplete calls |

Simple, ordinary typed, multiline/dotted-name and simple JSON calls succeeded.
One parallel-call request generated only its first requested call; the raw output
also contained only that call, so that instance was not a parser loss. A later
multi-turn reproduction confirmed that it completed alpha, beta and gamma
sequentially after receiving each tool result, rather than omitting the calls.
The first JSON literal-markup follow-up generated a flat `name/content` object
instead of `name/arguments`; the explicit-envelope follow-up above separates
that model formatting mistake from the proven parser failure.

Evidence is retained privately in `tool-traces/`: run prefixes `59b8667de4e8`
(32 base requests), `ca95d64a662f` (8 JSON follow-ups), and `bf668d96ab7c`
(4 explicit-envelope requests). The complete JSON calls dropped by the parser
are server captures `769ab40086f84045a05945cb0c129811.json` and
`a734f64e52c847f3b1acc4e90fb4ad1c.json`.

These reproduce server defects, not the user's unspecified failing client
workload. The default parser fix above addresses these defects; changing to
JSON formatting alone does not solve literal-delimiter failures.

Replaying the 44 saved raw generations through the fixed parser recovered all
four union-type cases, all four literal-markup XML arguments, and both complete
JSON calls previously dropped. Final and streamed decisions agreed for all 44;
all four deliberately truncated generations emitted no calls or partial argument
events. Invalid/incomplete model output remained rejected.

The patched server's subsequent 44-request HTTP matrix (`e82f925c6bdd`) returned
correct union values in all four XML cases and preserved literal markup in all
four XML cases, for both thinking/streaming settings. Replaying the exact
original requests for the two complete JSON calls also recovered both calls
over HTTP, including the streamed response. Single-call policy and ordinary
non-tool chat were checked separately in streamed and non-streamed modes.
The model still sometimes generated the wrong JSON envelope or premature EOS,
and the low-budget cases still ended at the token limit; those outputs were
rejected without partial tool events rather than guessed into executable calls.

### Remaining generation behavior: controlled diagnosis

`tools/diagnose_generation.py` renders identical prompts and compares raw
completion token IDs with MTP on/off, EOS stopping on/off, literal/escaped
control-tag text, and the custom/native cached templates. It uses the existing
server, not a second model instance. With the server running, from `flash-2`:

```bash
docker run --rm --network host --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD/tools:/tests:ro" \
  -v "$PWD/tool-traces:/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  tensorfold:v0.6.5 /tests/diagnose_generation.py
```

The 21 controlled requests are retained in
`tool-traces/generation-c0b88c189f7c/`. All prompts and results are synthetic;
runtime tracing remains disabled.

| Observation | Controlled result and interpretation |
|---|---|
| EOS halfway through a literal-marker argument | The target emitted actual EOS ID 248046 after 17 tokens with a 256-token budget. MTP on/off produced identical token IDs. `</tool_call>` in user data tokenizes as control ID 248059; describing that same value using a JSON Unicode escape avoided early EOS. This isolates sensitivity to control-tag content, not a misplaced parser stop or an MTP-only failure. |
| Ignoring EOS | Continuing to 96 tokens produced synthetic user/assistant `Continue` turns and repeated incomplete calls, not a repaired argument. Disabling EOS is not a safe fix. |
| Flattened JSON arguments | Ordinary `content: "hello world"` generated a correct JSON envelope; literal nested tool markup generated flat `name/content` JSON. Both behaviors were identical with MTP on/off. XML formatting preserved the markup correctly. This is generated format adherence, not serialization changing keys. |
| Custom versus native template | Cached `tokenizer_config.json` is XML-only, whereas the preferred `chat_template.jinja` is the custom `qwen3.8-froggeric-v22.5` template with an added JSON mode. The native-template literal probe completed a call but lost the marker and suffix, so merely switching templates is not a demonstrated fix. |
| One versus three calls | Seeds 42/43 generated three calls; seed 44 generated one, identically with MTP on/off. Resuming the exact original one-call conversation yielded alpha, beta, gamma on successive assistant turns. The first-turn-only assertion in the original matrix was stricter than a valid sequential agent workflow. |
| Missing structural guarantees | `tool_choice: auto` does not install a tool argument grammar. The required-call gate only forces the opener/name and then stops constraining; it does not enforce schema completeness or prevent EOS inside an argument. Parsing after generation cannot provide those guarantees. |

The sequential replay is in `tool-traces/generation-8d890cd2361c/` (three
assistant turns with simulated successful tool results). To repeat it against
the retained original synthetic HTTP fixture, add `--sequence-only` to the
command above.

The deliberate 18-token truncation case is an expected request-budget limit,
not evidence that the recipe's default 32768-token reply limit is broken.
Thinking and answer tokens share that reply budget.

These experiments exclude MTP as the cause of the tested failures and identify
their protocol/content triggers. They do not distinguish checkpoint training
effects from EXL3 quantization effects; that would require the same prompts
against a reference checkpoint/backend or another quantization. No claim is
made that quantization cannot affect generation quality.

## Prompt-speed overlay

Three files from `tensorfold/cuda/exl3/` of the 0.6.5 image, mounted read-only over the originals by `start.sh`
(`OVERLAY=0` turns it off). The output is bit-identical to stock (greedy text compared; every row's sum keeps the same
order). Measured on `turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, cold 8K prompt: 842 -> 1064 tok/s.

1. `experts_grouped.cuh`: the grouped expert kernel decoded each expert's weights once per 16-row tile, about five times
   per 2048-row chunk. With 64+ rows in the window one program now takes up to 5 row tiles (`TENSORFOLD_MTL`: 4, 5, 6 or 8)
   and decodes each weight tile once for all of them. Decode (few rows) uses the old kernel.
2. `experts.cu`: `group_kernel` (routing groups) was one block scanning every pick per expert; it now uses a shared-memory
   histogram and ballot-ordered member lists (1.4 s -> 0.15 s in the 8K profile).
3. `experts.py`: extension name `_v2`, so the JIT cache rebuilds.

The first start compiles the kernels (a few extra minutes). `patch.diff` is the change against stock.

## Prefix cache for image requests (`overlay/cache`)

Stock 0.6.5 never keeps or reuses a prompt prefix when the request has an image, so every turn of an image chat
re-prefills the whole history. `multi.py`, `multi_fill.py`, `prefixes.py` and `state.py` (mounted over the image's copies;
`VISION_CACHE=0` turns it off, `cache.diff` is the change) now:

- key kept prefixes by the prompt with each image's placeholder run replaced by a token derived from that image's
  content hash, so the same image matches and a different one cannot;
- drop the "image stream" guards on kept prompt points and reuse.

On a hit the images are still encoded (the full rotary positions are needed), but only the uncached rows are prefilled.
Tested on `Qwen3.8-Flash-Next` with a 1.7K-token image prompt: first 0 cached, repeat 1767, next turn 1767, a different
image after the same system text 1551 (the text before the image only); greedy output identical cached and uncached.

### Elastic kept snapshots

Stock keeps at most 8 prompt snapshots (`KEEP = 8`) and counts snapshots, not conversations: each conversation leaves one per message start plus one at its prompt end, so 8 covers about 4 conversations. With 9 parallel chats, 5 of 9 follow-ups re-prefilled cold. The overlay keeps 8 as a floor and lets up to `TENSORFOLD_KEEP_MAX` (256) be held while memory allows. Extra snapshots (about 110 MiB each) are shed oldest-first when a stream needs to grow (`_evict_kept`) or when free memory falls under `TENSORFOLD_KEEP_HEADROOM_GIB` (4). Only a snapshot whose slot holds another one is shed; slot-level eviction is unchanged. Measured: 9 parallel ~15K-token chats, follow-ups 9 of 9 warm (1 s each; stock 5 of 9 cold, 71 s); then 6 x 100K streams over the same server finished with no errors.

## Larger media limits (`limits/`, `VISION_LIMITS=1`)

Ported from the Qwen3.8-27B TensorFold patch. Stock 0.6.5 limits -> overlay:

| Limit | Stock | Overlay |
|---|---|---|
| Request body (`server/request_body.py`) | 32 MiB | 192 MiB, set with `TENSORFOLD_MAX_BODY_MIB` |
| Video, each / total (`vision/videos.py`) | 16 / 20 MiB | 64 / 192 MiB |
| Videos per request | 2 | 4 |
| Images, total bytes / pixels (`vision/images.py`) | 20 MiB / 32 MP | 64 MiB / 400 MP |
| Images, total decode timeout | 30 s | 60 s |

Still 50 images (`--vision-max-images`) per request. Tested: a 23 MB video (13K tokens, 23 s) and 50 images of 1080p (15.7K tokens, 24 s); the 51st is refused. `limits.diff` holds the diff.

## n-gram table pinning (`ngram/`, `TENSORFOLD_NGRAM_LOCK_GIB`)

Stock TensorFold `mlock`s the 39 GiB `ngram_embedding.safetensors` (27 GiB of it) into RAM at startup, which pins that memory and leaves ~5 GiB of `MemAvailable` for KV caches. The overlay caps the pin: `0` (default) leaves the tables as reclaimable page cache, `N` pins at most N GiB, `stock` keeps the original behaviour (the overlay is not mounted). `ngram.diff` holds the diff.

Measured on this Spark (same prompts, three restarts):

| Lock | `MemAvailable` after start | Decode | Cold 35K prefill | 6 x 80K concurrent |
|---|---|---|---|---|
| stock (all pinned) | 7.5 GiB | 47.0 tok/s | 33.4 s | 501 s, no errors |
| 16 GiB | 29.5 GiB | 47.0 tok/s | 33.1 s | 512 s, no errors |
| 0 | 45.0 GiB | 46.9 tok/s | 33.5 s | 493 s, no errors |

A 6 x 100K hold test (700 new tokens each) also passed with no errors in both stock (618 s) and 0 (678 s).
