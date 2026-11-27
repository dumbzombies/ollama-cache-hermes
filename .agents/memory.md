# Memory — ollama-cache-hermes

## 2026-11-27 — initial port

- **`kind: model-provider` plugins must call `register_provider(...)` at import,
  not inside `register(ctx)`.** The plugin doctor imports the module directly and
  checks the provider registry; a profile registered from `register(ctx)` is
  invisible and the doctor reports
  `model-provider plugin registered no ProviderProfile at import`. The fix is
  `PROFILE = _build_profile()` at module level, with `register(ctx)` reduced to
  command registration.
- **The doctor imports the module with sockets patched to fail** (`socket.
  create_connection` and `socket.socket.connect`/`connect_ex` raise). Two
  consequences, both initially fatal:
  1. any network probe at import (e.g. building `fallback_models` from the
     catalog) aborts registration → leave `fallback_models` unset and discover
     lazily;
  2. a bind error at import must not propagate → `_Server.start()` wraps
     `_start()` in a broad try/except and degrades to "no server".
  Symptom to recognize: doctor says "registered no ProviderProfile" but a manual
  `providers._import_plugin_dir(...)` in a plain interpreter registers fine.
- **Install path and ownership.** A `kind: model-provider` plugin lives under
  `$HERMES_HOME/plugins/model-providers/<name>/` (NOT `$HERMES_HOME/plugins/
  <name>/`, which is for `register(ctx)` plugins). On the official Docker image
  `HERMES_HOME=/opt/data`. A directory extracted from a tarball is owned by
  root, and the scanner then reports `unreadable plugin directory` — `chown -R
  hermes:hermes` the directory.
- **`hermes plugins doctor` resolves by path or installed plugin id**, not by
  directory name for a model-provider: `doctor ollama-cache` returns "was not
  found", while `doctor /opt/data/plugins/model-providers/ollama-cache` works.
- **Hermes gates an `api_key` provider on a resolvable credential**, even when
  the upstream needs none. Without one, the turn silently walks the fallback
  chain (`agent.log`: `Primary provider auth failed (No usable credentials found
  for provider 'ollama-cache'…)`) and the `--usage-file` shows the fallback
  provider — the plugin looks "installed but unused". `OLLAMA_API_KEY=<any
  non-placeholder value>` satisfies the gate; the plugin still forwards no auth
  header to ollama. The registry row's `api_key_env_vars` comes from the
  profile's `env_vars`, split into key vars vs a URL var.
- **Command registration signature** is
  `ctx.register_command(name, handler, description)` — positional, in
  `hermes_cli/plugins_context.py`. The keyword shape (`args_hint=`,
  `argument_mode=`) exists but `description=` is positional-adjacent; keep it
  positional.
- **Provider auth resolution for debugging:** `resolve_runtime_provider(requested=
  ..., target_model=...)` (no `provider=` kwarg). Calling it after
  `PluginManager().discover_and_load()` shows exactly which gate fails.
- **A non-SSE stub makes Hermes retry in a loop** exactly like opencode — the
  fixture must stream `data: {...}\n\n` events plus `data: [DONE]` and a final
  chunk carrying `finish_reason` and `usage`.
- **Chunked SSE passthrough in pure stdlib:** set
  `Transfer-Encoding: chunked` and write `f"{len(chunk):x}\r\n" + chunk + b"\r\n"`
  per read, flushing each time, then a terminating `0\r\n\r\n`. Buffer bytes and
  scan only complete `\n\n`-delimited events for `usage` — a split mid-JSON
  otherwise parses into garbage.
- **`displayName` copied verbatim from `ollama-cache-pi`** (BRAND table +
  tokenizer + version fold) so all three hosts render identical model names:
  `qwen3-coder:30b` → `Qwen3 Coder 30B (Ollama)`.
- **Env-only account key.** There is no `/login` store for ollama-cloud in
  Hermes, so `fetch_account_usage()` returns `None` without `OLLAMA_CLOUD_KEY`
  and the report prints `account: no OLLAMA_CLOUD_KEY set`.
- **`OLLAMA_CACHE_STATE_DIR`** exists as a test seam: the checker writes the
  state file into a temp dir so a real `~/.cache/ollama-cache` is never touched.
- **Restoring the host after testing:** `hermes config set model.provider …` /
  `model.default …` is the supported way to change the model section — editing
  `config.yaml` text directly did not take effect in this image.

## 2026-11-27 — published; picker notes aligned with mistral-vibe-hermes

- **Published:** `dumbzombies/ollama-cache-hermes`, public, default branch
  `main` (renamed from `master` before the first push; the README's
  `hermes plugins install dumbzombies/ollama-cache-hermes` and the `ssh`/`docker`
  variants all rely on that default).
- **Picker note was the visible divergence from the sibling trio.** It read
  `65536 ctx` (raw count, no name) while `mistral-vibe-hermes` renders
  `GLM 5.3 Flash (Vibe) · 1M ctx, 128k out, reasoning, vision`. Now:
  `tokens_label()` (decimal units first, then binary — the exact heuristic the
  vibe sibling uses), `model_specs()`, and
  `model_note() -> "<name> · <ctx> ctx[, reasoning]"`. The test asserts the
  literal string, not a substring, so a regression to raw counts fails.
- **Copy the sibling's label logic rather than re-deriving it.** The
  decimal-before-binary order matters: `131072` is `128k` via 1024 and `1048576`
  is `1M` via 1048576, so trying decimal first is what makes both read
  naturally. Getting this "simpler" (one division) yields `131.1k`.
- **`GPT OSS` normalization now uses a word boundary** (`re.sub(r"\bGPT OSS\b",
  …)`) to match the TS ports. The previous plain `str.replace` would also have
  hit `GPT OSSX`; no real tag does, but the three ports should not disagree on
  how a name is rendered.
- **README now matches the family skeleton** (Install / First run / Models /
  What it does / Endpoints / Configuration / Troubleshooting / Tests / Limits /
  License). It previously opened with "What it does" and had no Troubleshooting,
  Tests claim, or License section.
- **The container install predates these changes** — `/opt/data/plugins/.../ollama-cache`
  must be reinstalled from the repo to pick up readable notes.

## 2026-11-27 (2) — sync hygiene: do not record a HEAD you are about to move

- `state.md` had `HEAD <sha>`, but writing that file is a commit, so the recorded
  sha was always one behind. Now: **"code as of `<sha>` — later commits here only
  touch this bank"**, which stays true.
- No source changed; `python3 test/plugin_check.py` 41/41.
