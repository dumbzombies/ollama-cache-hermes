# State — ollama-cache-hermes

## Goal
Give Hermes the same ollama prefix-cache benefit: a `kind: model-provider` plugin
that runs an OpenAI-compatible rewriting proxy inside the Hermes process and
points the provider's client at it.

## Current (v1.0.0, verified 2026-11-27)

Files: `plugin.yaml` (manifest_version 1, kind `model-provider`, MIT, tags
`[model-provider, ollama, local, prefix-cache]`), `__init__.py` (stdlib only),
`test/plugin_check.py`, `README.md`, `LICENSE`, `.gitignore`.

`__init__.py`:
- **Registers at import**: `PROFILE = _build_profile()` → `register_provider(...)`.
  `register(ctx)` only adds the `/ollama-usage` command
  (`ctx.register_command(name, handler, description)`, positional) and returns
  the profile.
- No network and no hard-fail bind at import — The doctor imports the module with
  sockets patched to fail. `_Server.start()` wraps `_start()` and degrades to "no
  server" instead of raising; `fallback_models` is deliberately unset (the
  catalog is discovered lazily by `discover_models()`).
- `_Server`: `ThreadingHTTPServer` daemon thread on
  `127.0.0.1:${OLLAMA_HERMES_PORT ?? 11501}`; on bind failure walks +1…+5, then
  reuses a live sibling answering `/healthz` with `ollama-cache-hermes` (or
  `ollama-cache-opencode`) on 11501/11500, then prints a warning and continues.
- Routes: `/healthz`, `/v1/models`, `/api/models`, `POST /v1/chat/completions`,
  `/api/state`. Same rewrite rules as the opencode proxy (fence-aware volatile
  reminder strip, volatile system lines, right-trim system text, `tools` sorted
  by `function.name ?? name`, assistant untouched, no `<environment>` injection),
  same `\u0000`-joined append-only assert with `prefix drift at byte N` on stderr
  under `OLLAMA_CACHE_DEBUG=1`.
- Metering: `requests`/`errors`/token counts from JSON bodies and from the last
  SSE chunk carrying `usage`, published atomically to
  `~/.cache/ollama-cache/hermes-server.json` (override:
  `OLLAMA_CACHE_STATE_DIR`).
- `OllamaCacheProfile(ProviderProfile)`: `name="ollama-cache"`, alias
  `ollama-local`, `base_url` → the bound loopback `/v1`, `env_vars =
  ("OLLAMA_API_KEY", "OLLAMA_CLOUD_KEY")`, `create_client()` returns an
  `openai.OpenAI` pointed at the proxy with `api_key="local"`,
  `discover_models()` maps the catalog to `{id, label, note}`,
  `fetch_account_usage()` reads ollama.com when `OLLAMA_CLOUD_KEY` is set.
- `format_usage(port)` renders the pi-style report (`server …`, `requests …`,
  token line, `account:` lines or `account: no OLLAMA_CLOUD_KEY set`).

## Verification (real Docker host: `ssh hostinger`, container `hermes`, HERMES_HOME=/opt/data)
- `python3 test/plugin_check.py` → all checks pass (rewrite, fenced sample,
  stable reminder, tools sorted, assistant untouched, byte-identical non-stream
  and SSE passthrough, metering, catalog mapping, `/api/state`, account parsing,
  `discover_models` labels/notes, `fetch_account_usage`).
- Installed at `/opt/data/plugins/model-providers/ollama-cache` (owned by
  `hermes`); `hermes plugins list --plain` → `enabled user 1.0.0 ollama-cache`;
  `hermes plugins doctor /opt/data/plugins/model-providers/ollama-cache` → OK,
  `provider(s): ollama-cache`.
- Real turn: `OLLAMA_API_KEY=ollama-local hermes -z "reply OK"` with
  `model.provider=ollama-cache`, `model.default=qwen3-coder:30b` →
  `--usage-file` reports model `qwen3-coder:30b`, provider `ollama-cache`; the
  stub ollama logged exactly one `POST /v1/chat/completions`; the state file
  recorded the request and tokens.
- The host's `config.yaml` was restored to `provider: mistral-vibe`,
  `default: deepseek-v4-1-flash-1017` afterwards; the test stub was killed.

## Blockers
None.

## Next actions
- Set `OLLAMA_CLOUD_KEY` in the container to populate the `account:` lines.
- No live ollama in the container: `/api/*` was verified against a Python stub.
  Point `OLLAMA_BASE_URL` at a reachable server before relying on discovery.
- Publish when ready (nothing published).
