# pi extensions for TensorFold servers

## `tf-context-clamp.ts`

pi sizes a request's reply budget as `contextWindow - estimate - 4096`, with the prompt estimated at
characters / 3.5. Code and shell-heavy prompts run nearer 2-3.2 characters per token, so a large request that has no
usage anchor (a compaction summary) asks for more than the window holds. TensorFold (and vLLM) check prompt + `max_tokens`
before starting and refuse with 400 `context_length_exceeded`, so compaction fails:

```
Auto-compaction failed: Summarization failed: 400: ... the rendered prompt has 162424 tokens and requests 110592 reply tokens ...
```

Upstream reports: earendil-works/pi#9101, earendil-works/pi#9295, earendil-works/pi#11161.

The extension wraps the `openai-completions` stream of the listed providers. A request whose worst case
(JSON characters / 2 + reply budget) could exceed the window is sent to the server's `/v1/tokenize` first, which renders
the same messages, tools and `chat_template_kwargs` as the chat route, and its reply budget is lowered to
`max_model_len - prompt - 16`. Smaller requests are not checked. A prompt that alone fills the window is left unchanged,
so pi still sees the overflow error and compacts. If `/tokenize` fails, it falls back to characters / 2.5 with an
8192-token margin. It wraps the provider because pi builds summary requests without `before_provider_request`.

Install (a symlink, so edits here apply on pi's next start):

```bash
mkdir -p ~/.pi/agent/extensions
ln -sf "$PWD/pi-extensions/tf-context-clamp.ts" ~/.pi/agent/extensions/tf-context-clamp.ts
```

| Env | Default | |
|---|---|---|
| `PI_TF_CLAMP_PROVIDERS` | `dgx-spark-sg-lang` | comma-separated pi provider names (from `models.json`) to wrap |
| `PI_TF_CLAMP` | on | `0` disables it |
| `PI_TF_CLAMP_LOG` | unset | file for one line per checked request (token counts only, no prompt text) |
| `PI_TF_CLAMP_CHECK_ALL` | unset | `1` tokenizes every request, to confirm the hook runs |

Verified with pi 1.4.0 against flash-3 (TensorFold 0.6.6): a crafted session whose summary prompt rendered to 190,799
tokens failed `compact` without the extension (pi asked for 120,000 reply tokens); with it the budget became 71,329, the
server accepted it and the compaction was recorded. `/tokenize` for a 195K-token prompt took 0.38 s. Normal turns and
small compactions are unchanged.
