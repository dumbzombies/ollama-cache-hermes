# ollama-cache-hermes

Ollama prefix-cache maximizer for [Hermes](https://hermes-agent.nousresearch.com), as a
model-provider plugin.

Ollama's prefix cache is automatic and client-invisible: it keys on the exact bytes of
system prompt + tool schemas + message prefix. Hermes has no per-request body hook, so this
plugin owns the transport: it runs an OpenAI-compatible server on loopback *inside the
Hermes process* and points the provider's client at it. Every request is rewritten there
before it reaches ollama, and every response is piped back byte for byte.

Nothing to deploy: no proxy, no Worker, no daemon, no port to open. Stdlib only.

Sibling repos: [ollama-cache-pi](https://github.com/dumbzombies/ollama-cache-pi) (the omp/pi
extension, which carries the same rule set natively) and
[ollama-cache-opencode](https://github.com/dumbzombies/ollama-cache-opencode) (the same proxy
for opencode).

## Install

One command, native to Hermes:

```bash
hermes plugins install dumbzombies/ollama-cache-hermes
```

A local checkout, when your Hermes build only accepts git sources:

```bash
rm -rf "$HERMES_HOME/plugins/model-providers/ollama-cache"
cp -r /path/to/ollama-cache-hermes "$HERMES_HOME/plugins/model-providers/ollama-cache"
# a plugin imported from a fresh checkout must be readable by the agent user
chown -R hermes:hermes "$HERMES_HOME/plugins/model-providers/ollama-cache"
```

`$HERMES_HOME` defaults to `~/.hermes`; in the official Docker image it is `/opt/data`.
Remote host and Docker are the same command on the target:

```bash
ssh host 'hermes plugins install dumbzombies/ollama-cache-hermes'
docker exec hermes hermes plugins install dumbzombies/ollama-cache-hermes
```

Hermes' install-time security scan will warn that a third-party source is unreviewed;
approve it if you trust the source.

Check the install:

```bash
hermes plugins list | grep ollama-cache
hermes plugins doctor "$HERMES_HOME/plugins/model-providers/ollama-cache"
```

A `kind: model-provider` plugin lives under `plugins/model-providers/<name>/`, and `doctor`
resolves it by path (not by name). Remove it with `hermes plugins remove ollama-cache`.

## First run

```bash
export OLLAMA_API_KEY=ollama-local  # dummy: Hermes needs a credential to select the provider
hermes model                        # picker: ollama-cache/<id> rows with names
hermes -m qwen3-coder:30b "reply OK"
/ollama-usage                       # local counters + account block
```

## Models

The list is discovered from the server, not hardcoded — `/api/tags` + `/api/show` probed
once per 5 minutes and served as annotated picker rows:

```
qwen3-coder:30b  · Qwen3 Coder 30B (Ollama) · 64k ctx, reasoning
gpt-oss:20b      · GPT-OSS 20B (Ollama)     · 128k ctx
```

Context comes from `num_ctx` / `*context_length`; the note carries `reasoning` when the
model reports the `thinking` capability. Embedding and rerank tags are skipped. The result is
mirrored to `~/.cache/ollama-cache/models.json` so a cold start is named before the server
answers.

## What it does

1. **Reminder stripping.** `<system-reminder>` blocks carrying a date (`Today: YYYY-MM-DD`),
   `current working directory`, or `cwd:` are cut from the system message and from user
   messages. Fence-aware: a pasted sample inside a ``` block is content, not markup.
2. **Volatile lines.** In the system message only, standalone `Today:` / directory / `date:` /
   `session-id:` / `context left` / `This session may have run before.` lines are dropped and
   the text is right-trimmed. User text outside a reminder block and all assistant messages
   are never rewritten.
3. **Fixed tool order.** `tools` is sorted by `function.name ?? name` — ollama hashes tools
   in order, so a reordered schema list is a full cache miss.
4. **Append-only assert.** The prefix (`model` + system text + tools + messages, each element
   `json.dumps`ed and joined with `\u0000`, no enclosing array) is compared with the previous
   request's. A non-append prints `[ollama-cache] prefix drift at byte N` when
   `OLLAMA_CACHE_DEBUG=1`. Diagnostic only: the request still goes out.
5. **Live catalog + metering.** `/api/tags` + `/api/show` are probed once per 5 minutes,
   mirrored, and served as annotated picker rows. Requests and token counts are published to
   `~/.cache/ollama-cache/hermes-server.json`.

No `<environment>` item is injected: Hermes owns its own environment block, and a second
source would only add bytes.

## Endpoints on loopback

| route | purpose |
|---|---|
| `GET /healthz` | liveness + `service: "ollama-cache-hermes"` (port-reuse contract) |
| `GET /v1/models` | ollama catalog mapped to the OpenAI list shape |
| `GET /api/models` | full rows (display name, context, reasoning) |
| `POST /v1/chat/completions` | rewrite → forward → byte passthrough |
| `GET /api/state` | meter + account |

Port contract: `OLLAMA_HERMES_PORT` (default `11501`); on `EADDRINUSE` the plugin walks
+1…+5, then reuses a live `ollama-cache-hermes` (or `ollama-cache-opencode`) sibling it finds
on 11501/11500. It never crashes `register`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | ollama server (`OLLAMA_HOST` also read) |
| `OLLAMA_API_KEY` | — | any non-empty value (e.g. `ollama-local`); Hermes only selects an `api_key` provider when a credential resolves, and a local ollama needs none |
| `OLLAMA_HERMES_PORT` | `11501` | loopback port |
| `OLLAMA_CLOUD_KEY` | — | ollama.com account key for `fetch_account_usage` + `account:` lines |
| `OLLAMA_CLOUD_URL` | `https://ollama.com` | account API host (mirror or test double) |
| `OLLAMA_CACHE_STATE_DIR` | `~/.cache/ollama-cache` | state + model mirror location |
| `OLLAMA_CACHE_DEBUG` | — | `1` logs prefix drift and probe failures to stderr |

The account key is env-only: there is no `/login` store for ollama-cloud in Hermes. Without
`OLLAMA_CLOUD_KEY`, `fetch_account_usage` returns `None` (Hermes shows no quota) and the
report prints `account: no OLLAMA_CLOUD_KEY set`; local counters still work.

## Troubleshooting

No models in the picker: `OLLAMA_API_KEY` is unset — Hermes only offers an `api_key` provider
when a credential resolves, and a local ollama needs no real key. Set any non-empty value.
Provider missing entirely: the plugin failed to load; `hermes plugins doctor <path>` names the
error.
No names, only raw ids: the probe could not reach `OLLAMA_BASE_URL`; a cold start with the
server down serves the mirror or raw ids. Start ollama and restart Hermes.
`/ollama-usage` says no account key: `OLLAMA_CLOUD_KEY` is env-only here; local counters
still work without it.
A new tag is missing: the probe cache is 5 minutes; delete `~/.cache/ollama-cache/models.json`.

## Tests

```bash
python3 test/plugin_check.py    # no network, no key, no Hermes install: exit 0
```

41 checks against a local fake upstream: the rewrite (system/user reminders stripped, fenced
samples preserved, tools sorted, assistant untouched), byte-identical non-stream and stream
passthrough, metering into the state file, catalog mapping, `/api/state`, account parsing,
`discover_models` labels and the `ollama-usage` report.

## Limits

- One append-only baseline per process: interleaved conversations are reported as drift. It
  is a diagnostic, so nothing breaks.
- The model mirror is a cache: delete `~/.cache/ollama-cache/models.json` to force a probe.
- Costs are zero for local ollama; the report counts tokens rather than money.

## License

MIT
