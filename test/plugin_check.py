"""Plugin checks: no network, no key, no Hermes install.

Stubs the `providers` / `agent.account_usage` modules Hermes supplies, points
`OLLAMA_BASE_URL` at a local fake ollama, and drives the plugin's own loopback
server. Exits non-zero on any failed assertion.

    python3 test/plugin_check.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "__init__.py"

failures: list[str] = []
seen: list[dict] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("PASS " if condition else "FAIL ") + label + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(label)


# ── fake ollama upstream ─────────────────────────────────────────────


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args, **_kwargs):
        return

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/tags"):
            self._send(200, json.dumps({"models": [{"name": "qwen3-coder:30b"}, {"name": "nomic-embed-text:latest"}]}).encode())
            return
        if self.path.startswith("/api/balance"):
            self._send(200, json.dumps({"included": {"balance_usd": 57.53, "allowance_usd": 60, "period": {"until": "2026-12-22T00:00:00Z"}}, "purchased": {"balance_usd": 40}}).encode())
            return
        if self.path.startswith("/api/usage"):
            self._send(200, json.dumps({"totals": {"request_count": 41, "usage_usd": 0.2894, "input_tokens": 233000, "cached_input_tokens": 121160, "output_tokens": 366000}}).encode())
            return
        self._send(404, b"{}")

    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        request = json.loads(raw or b"{}")
        if self.path.startswith("/api/show"):
            self._send(200, json.dumps({"capabilities": ["completion"], "model_info": {"llama.context_length": 65536}}).encode())
            return
        seen.append(request)
        if request.get("stream"):
            events = [
                {"choices": [{"index": 0, "delta": {"content": "he"}}]},
                {"choices": [{"index": 0, "delta": {"content": "y"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            ]
            body = ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()
            self._send(200, body, "text/event-stream")
            return
        payload = {
            "id": "s",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 40}},
        }
        self._send(200, json.dumps(payload).encode())


def load_plugin() -> types.ModuleType:
    providers = types.ModuleType("providers")
    base = types.ModuleType("providers.base")

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def create_client(self, **kwargs):
            return None

        def fetch_account_usage(self, **kwargs):
            return None

    base.ProviderProfile = ProviderProfile
    providers.base = base
    providers.register_provider = lambda profile: None
    sys.modules["providers"] = providers
    sys.modules["providers.base"] = base

    account = types.ModuleType("agent")
    account_usage = types.ModuleType("agent.account_usage")

    class AccountUsageWindow:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AccountUsageSnapshot:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    account_usage.AccountUsageWindow = AccountUsageWindow
    account_usage.AccountUsageSnapshot = AccountUsageSnapshot
    account.account_usage = account_usage
    sys.modules["agent"] = account
    sys.modules["agent.account_usage"] = account_usage

    spec = importlib.util.spec_from_file_location("ollama_cache_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> int:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{upstream.server_address[1]}"

    scratch = tempfile.mkdtemp(prefix="ollama-cache-hermes-")
    os.environ.update(
        {
            "OLLAMA_BASE_URL": base,
            "OLLAMA_CACHE_STATE_DIR": scratch,
            "OLLAMA_HERMES_PORT": "11601",
            "OLLAMA_CACHE_DEBUG": "1",
        }
    )
    os.environ.pop("OLLAMA_CLOUD_KEY", None)
    os.environ["OLLAMA_CLOUD_URL"] = base

    module = load_plugin()

    # -- pure helpers --
    reminder = "<system-reminder>\nToday: 2026-11-26; current working directory: '/tmp/x'.\n</system-reminder>"
    text, changed = module.sanitize_text(f"{reminder}\nhello", False)
    check("system reminder stripped", changed and "2026-11-26" not in text and "hello" in text, text)
    fenced = f"```\n{reminder}\n```"
    check("fenced sample preserved", module.sanitize_text(fenced, False) == (fenced, False))
    stable = "<system-reminder>Always run tests.</system-reminder>"
    check("stable reminder kept", module.sanitize_text(stable, False) == (stable, False))
    lines = "You are a coder.\nToday: 2026-11-26\ncwd: /tmp/x\nkeep"
    check("volatile lines dropped with line rules", module.sanitize_text(lines, True)[0] == "You are a coder.\nkeep")
    check("volatile lines kept in user text", module.sanitize_text(lines, False)[0] == lines)

    tools = [{"type": "function", "function": {"name": "z"}}, {"type": "function", "function": {"name": "a"}}]
    check("tools sorted", [module.tool_name(t) for t in module.sort_tools(tools)] == ["a", "z"])
    check("sorted tools returned as-is", module.sort_tools(tools[:1]) is not None)

    check("displayName matches the pi names", module.display_name("qwen3-coder:30b") == "Qwen3 Coder 30B (Ollama)", module.display_name("qwen3-coder:30b"))
    check("displayName version fold", module.display_name("deepseek-v4.1-flash-1017") == "DeepSeek V4.1 Flash (Ollama)", module.display_name("deepseek-v4.1-flash-1017"))
    check("displayName gpt-oss", module.display_name("gpt-oss:20b") == "GPT-OSS 20B (Ollama)", module.display_name("gpt-oss:20b"))

    # -- append-only assert --
    module._baseline = None
    check("first request establishes baseline", module.assert_append_only("a\u0000b", "m"))
    check("append passes", module.assert_append_only("a\u0000b\u0000c", "m"))
    check("mutation reported", not module.assert_append_only("a\u0000X", "m"))

    # -- rewrite of a full body --
    body = {
        "model": "qwen3-coder:30b",
        "messages": [
            {"role": "system", "content": f"You are a coder.\nToday: 2026-11-26\n{reminder}\n"},
            {"role": "user", "content": f"{reminder}\nhello"},
            {"role": "assistant", "content": f"  {reminder} keep  "},
        ],
        "tools": tools,
    }
    rewritten = module.rewrite_request(json.loads(json.dumps(body)))
    check("system text trimmed and stripped", rewritten["messages"][0]["content"] == "You are a coder.", repr(rewritten["messages"][0]["content"]))
    check("user reminder stripped", rewritten["messages"][1]["content"] == "hello", repr(rewritten["messages"][1]["content"]))
    check("assistant untouched", rewritten["messages"][2]["content"] == f"  {reminder} keep  ")
    check("tools sorted in body", [module.tool_name(t) for t in rewritten["tools"]] == ["a", "z"])

    # -- the server --
    class Ctx:
        def __init__(self):
            self.commands = {}

        def register_command(self, name, handler, description=""):
            self.commands[name] = handler

    ctx = Ctx()
    profile = module.register(ctx)
    port = module._Server.port
    root = f"http://127.0.0.1:{port}"
    check("register returns a profile", profile is not None and profile.name == "ollama-cache")
    check("register_command wired", "ollama-usage" in ctx.commands, str(ctx.commands.keys()))

    with urllib.request.urlopen(f"{root}/healthz", timeout=5) as response:
        health = json.loads(response.read().decode())
    check("healthz identifies the service", health.get("service") == "ollama-cache-hermes", str(health))
    check("healthz reports the upstream", health.get("upstream") == base, str(health.get("upstream")))

    with urllib.request.urlopen(f"{root}/v1/models", timeout=5) as response:
        catalog = json.loads(response.read().decode())
    ids = [row["id"] for row in catalog.get("data", [])]
    check("catalog maps /api/tags", ids == ["qwen3-coder:30b"], str(ids))

    def post(payload: dict) -> tuple[int, bytes]:
        request = urllib.request.Request(
            f"{root}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()

    seen.clear()
    status, raw = post(json.loads(json.dumps(body)))
    check("non-stream status 200", status == 200, str(status))
    check("non-stream body byte-identical", json.loads(raw)["usage"]["prompt_tokens"] == 100)
    forwarded = seen[-1]
    check("upstream saw the stripped system text", forwarded["messages"][0]["content"] == "You are a coder.", repr(forwarded["messages"][0]["content"]))
    check("upstream saw sorted tools", [module.tool_name(t) for t in forwarded["tools"]] == ["a", "z"])

    state_file = pathlib.Path(scratch) / "hermes-server.json"
    state = json.loads(state_file.read_text())
    check("state file published", state.get("service") == "ollama-cache-hermes", str(state)[:120])
    check("state file counts the request", state.get("requests", 0) >= 1, str(state.get("requests")))
    check("state file meters tokens", state.get("input_tokens", 0) >= 100, str(state.get("input_tokens")))

    seen.clear()
    status, raw = post({"model": "qwen3-coder:30b", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    check("stream status 200", status == 200, str(status))
    check("stream body passes through", raw.decode().rstrip().endswith("data: [DONE]"), raw[-40:].decode(errors="replace"))
    check("stream usage metered", json.loads(state_file.read_text()).get("output_tokens", 0) >= 3)

    # -- account block --
    os.environ["OLLAMA_CLOUD_KEY"] = "test-key"
    module._cloud_cache.update({"at": 0.0, "info": {}})
    info = module.cloud_info()
    check("account plan parsed", (info.get("plan") or {}).get("balance") == 57.53, str(info))
    check("account total summed", info.get("total") == 97.53, str(info.get("total")))
    check("account 24h usage parsed", (info.get("usage") or {}).get("requests") == 41, str(info.get("usage")))

    report = module.format_usage(port)
    check("report names the server", "server " + base in report, report.splitlines()[1] if len(report.splitlines()) > 1 else report)
    check("report has account lines", "account: plan" in report and "account: 24h" in report, report)
    with urllib.request.urlopen(f"{root}/api/state", timeout=5) as response:
        state_api = json.loads(response.read().decode())
    check("/api/state carries the account", state_api.get("hasKey") is True and (state_api.get("cloud") or {}).get("total") == 97.53, str(state_api)[:160])

    # -- profile hooks --
    rows = profile.discover_models()
    check("discover_models labels", rows and rows[0]["label"] == "Qwen3 Coder 30B (Ollama)", str(rows)[:120])
    check("discover_models note", rows and "65536" in rows[0]["note"], str(rows)[:120])
    usage = profile.fetch_account_usage()
    check("fetch_account_usage returns a window", usage is not None and usage.windows, str(usage))

    upstream.shutdown()
    shutil.rmtree(scratch, ignore_errors=True)
    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
