/**
 * tf-context-clamp: keep pi's reply budget inside a local server's context window.
 *
 * pi sizes `max_tokens` as `contextWindow - estimate - 4096`, where the estimate is characters / 3.5. On code-heavy
 * prompts the real count is ~10% higher (Qwen: ~3.2 chars/token), so a large request (typically a compaction summary,
 * which has no usage anchor) asks for more than the window holds and TensorFold/vLLM refuse it with 400
 * context_length_exceeded (earendil-works/pi#9101, #9295, #11161).
 *
 * For the listed providers this wraps the openai-completions stream. Before each request whose worst-case size
 * (JSON characters / 2) plus max_tokens could exceed the window, it asks the server's `/tokenize` (vLLM-compatible;
 * TensorFold and vLLM render the same messages, tools and chat_template_kwargs as the chat route) for the exact prompt
 * size and lowers max_tokens to `max_model_len - prompt - MARGIN`. A prompt that alone fills the window is left
 * unchanged, so the server's overflow error still reaches pi's compaction recovery. When /tokenize fails it falls back
 * to a conservative chars / 2.5 estimate with an 8192-token margin.
 *
 * It wraps the provider rather than using `before_provider_request`, because pi builds summary requests without that
 * hook. Env: PI_TF_CLAMP_PROVIDERS (comma list, default dgx-spark-sg-lang), PI_TF_CLAMP=0 (off),
 * PI_TF_CLAMP_LOG=<file> (one line per checked request; token counts only, no prompt content),
 * PI_TF_CLAMP_CHECK_ALL=1 (tokenize every request, for verifying the hook).
 */

import { appendFileSync } from "node:fs";
import { getApiProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MARGIN = 16;                       // tokens kept free beyond the exact count
const FALLBACK_CHARS_PER_TOKEN = 2.5;    // used only when /tokenize is unreachable (code/shell text reaches ~2.7)
const FALLBACK_MARGIN = 8192;
const PESSIMISTIC_CHARS_PER_TOKEN = 2;   // a payload this small with this reply cannot overflow: no check
const TOKENIZE_TIMEOUT_MS = 60_000;

type Payload = Record<string, any>;
type Tokenized = { count: number; maxModelLen?: number };
type Tokenize = (baseUrl: string, params: Payload, apiKey?: string, signal?: AbortSignal) => Promise<Tokenized>;

function log(line: string): void {
  const file = process.env.PI_TF_CLAMP_LOG;
  if (!file) return;
  try {
    appendFileSync(file, `${new Date().toISOString()} ${line}\n`);
  } catch {
    // logging must never break a request
  }
}

function payloadChars(params: Payload): number {
  return JSON.stringify(params.messages ?? []).length + JSON.stringify(params.tools ?? []).length;
}

export const tokenize: Tokenize = async (baseUrl, params, apiKey, signal) => {
  const body: Payload = { model: params.model, messages: params.messages };
  for (const k of ["tools", "chat_template_kwargs"]) if (params[k] !== undefined) body[k] = params[k];
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (apiKey && apiKey !== "none") headers.authorization = `Bearer ${apiKey}`;
  const timeout = AbortSignal.timeout(TOKENIZE_TIMEOUT_MS);
  const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/tokenize`, {
    method: "POST", headers, body: JSON.stringify(body),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!res.ok) throw new Error(`/tokenize ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const out = (await res.json()) as { count?: number; tokens?: unknown[]; max_model_len?: number };
  const count = typeof out.count === "number" ? out.count : Array.isArray(out.tokens) ? out.tokens.length : NaN;
  if (!Number.isFinite(count)) throw new Error("/tokenize returned no count");
  return { count, maxModelLen: typeof out.max_model_len === "number" ? out.max_model_len : undefined };
};

/** Fit the payload's reply budget to the window (edits `params` in place and returns it). */
export async function clampPayload(params: Payload, model: { baseUrl: string; contextWindow: number; id: string },
                                   apiKey?: string, signal?: AbortSignal,
                                   tokenizeFn: Tokenize = tokenize): Promise<Payload> {
  const field = params.max_completion_tokens !== undefined ? "max_completion_tokens" : "max_tokens";
  const asked = params[field];
  if (typeof asked !== "number" || asked <= 0) return params;
  const window = model.contextWindow;
  const chars = payloadChars(params);
  if (process.env.PI_TF_CLAMP_CHECK_ALL !== "1" && window > 0 &&
      Math.ceil(chars / PESSIMISTIC_CHARS_PER_TOKEN) + asked + MARGIN <= window) return params;
  let room: number;
  let how: string;
  try {
    const t = await tokenizeFn(model.baseUrl, params, apiKey, signal);
    const limit = t.maxModelLen ?? window;
    if (t.count >= limit) {
      log(`${model.id} prompt=${t.count} >= window=${limit}: unchanged (the server reports the overflow)`);
      return params;
    }
    room = limit - t.count - MARGIN;
    how = `prompt=${t.count} window=${limit}`;
  } catch (err) {
    if (signal?.aborted) throw err;
    room = window - Math.ceil(chars / FALLBACK_CHARS_PER_TOKEN) - FALLBACK_MARGIN;
    how = `fallback chars=${chars} window=${window} (${err instanceof Error ? err.message : String(err)})`;
    if (room < 1024) {
      log(`${model.id} ${how}: unchanged`);
      return params;
    }
  }
  if (room >= asked) {
    log(`${model.id} ${how} ${field}=${asked}: fits`);
    return params;
  }
  const next = Math.max(1, room);
  log(`${model.id} ${how} ${field}=${asked} -> ${next}`);
  params[field] = next;
  return params;
}

export default function (pi: ExtensionAPI) {
  if (process.env.PI_TF_CLAMP === "0") return;
  const api = getApiProvider("openai-completions");
  if (!api) return;
  const providers = (process.env.PI_TF_CLAMP_PROVIDERS ?? "dgx-spark-sg-lang")
    .split(",").map((s) => s.trim()).filter(Boolean);
  for (const provider of providers) {
    pi.registerProvider(provider, {
      api: "openai-completions",
      streamSimple: (model: any, context: any, options: any) => {
        const outer = options?.onPayload;
        return api.streamSimple(model, context, {
          ...options,
          onPayload: async (params: Payload, m: any) => {
            const clamped = await clampPayload(params, m ?? model, options?.apiKey, options?.signal);
            const next = outer ? await outer(clamped, m) : undefined;
            return next !== undefined ? next : clamped;
          },
        });
      },
    });
  }
}
