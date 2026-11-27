"""ollama-cache provider for Hermes Agent — self-contained.

Hermes has no per-request body hook, so the plugin owns the transport: it runs an
OpenAI-compatible server inside the Hermes process (a daemon thread on loopback)
and points the provider's client at it. Every request is rewritten there before
it reaches ollama:

  * date/cwd `<system-reminder>` blocks are cut from the system message and from
    user messages, fence-aware (a pasted sample inside a ``` block is content);
  * standalone volatile lines in the SYSTEM message are dropped (`Today:`,
    directory/cwd, `date:`/`time:`, session/request id, `context left`);
  * `tools` is sorted by `function.name ?? name` — ollama hashes tools in order;
  * the message prefix is compared with the previous request's and a non-append
    is logged on stderr (diagnostic only, never a rewrite).

No `<environment>` item is injected: Hermes owns its own environment block, and a
second source would only add bytes. Responses are never modified — they are
piped back byte for byte.

Env:
  OLLAMA_BASE_URL       ollama server (default http://127.0.0.1:11434)
  OLLAMA_HERMES_PORT    loopback port (default 11501; +1..+5, then reuse)
  OLLAMA_CLOUD_KEY      ollama.com account key for the account block
  OLLAMA_CLOUD_URL      account API host (default https://ollama.com)
  OLLAMA_CACHE_DEBUG    1 => prefix-drift and probe diagnostics on stderr
  OLLAMA_CACHE_STATE_DIR  state dir (default ~/.cache/ollama-cache)

Stdlib only: no daemon, no external proxy, no Worker.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

VERSION = "1.0.0"
SERVICE = "ollama-cache-hermes"
DEFAULT_BASE = "http://127.0.0.1:11434"
DEFAULT_PORT = 11501
MODELS_TTL_S = 5 * 60
CLOUD_TTL_S = 60
CLOUD_TIMEOUT_S = 4
CLOUD_ATTEMPTS = 3
RETRY_STATUS = {401, 403, 429, 500, 502, 503, 504}
MAX_BODY_BYTES = 32 * 1024 * 1024
NOT_CHAT = re.compile(r"(?:^|[^a-z])(?:embed|rerank)(?:$|[^a-z])", re.IGNORECASE)


# ── config ───────────────────────────────────────────────────────────


def ollama_base() -> str:
    raw = (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "").strip()
    return (raw or DEFAULT_BASE).rstrip("/")


def state_dir() -> Path:
    override = os.environ.get("OLLAMA_CACHE_STATE_DIR")
    if override:
        return Path(override)
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "ollama-cache"


def debug(message: str) -> None:
    if os.environ.get("OLLAMA_CACHE_DEBUG") == "1":
        print(f"[ollama-cache] {message}", file=sys.stderr)


# ── text sanitation ──────────────────────────────────────────────────

REMINDER_OPEN = "<system-reminder>"
REMINDER_CLOSE = "</system-reminder>"
FENCE_LINE = re.compile(r"^\s*(```|~~~)")
REMINDER_NEEDLE = re.compile(r"<system-reminder>|current working directory|Today: \d{4}-\d{2}-\d{2}")

VOLATILE_LINE = (
    re.compile(r"^\s*Today:\s*\d{4}-\d{2}-\d{2}\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:current )?(?:working )?directory\s*[:=]\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*cwd\s*[:=]\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*(?:current )?(?:date|time)\s*[:=]\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*(?:session|request)[-_ ]?id\s*[:=]\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*(?:context (?:left|window|used))\b.*$", re.IGNORECASE),
    re.compile(r"^\s*This session may have run before\..*$", re.IGNORECASE),
)
VOLATILE_GATE = re.compile("|".join(r.pattern for r in VOLATILE_LINE), re.IGNORECASE | re.MULTILINE)


def reminder_is_volatile(block: str) -> bool:
    """A reminder block is volatile when it carries a clock or working directory."""
    return bool(
        re.search(r"current working directory", block, re.IGNORECASE)
        or re.search(r"Today:\s*\d{4}-\d{2}-\d{2}", block)
        or re.search(r"\bcwd\s*[:=]", block, re.IGNORECASE)
    )


def sanitize_text(text: str, line_rules: bool) -> tuple[str, bool]:
    """Remove volatile reminder blocks; `line_rules` also drops volatile whole lines.

    Fenced code is copied verbatim: a pasted `<system-reminder>` sample inside a
    ``` block is content, not an injection point.
    """
    if not REMINDER_NEEDLE.search(text) and not (line_rules and VOLATILE_GATE.search(text)):
        return text, False
    out: list[str] = []
    changed = False
    fenced = False
    in_reminder = False
    block: list[str] = []

    def flush() -> None:
        nonlocal changed, block
        if reminder_is_volatile("\n".join(block)):
            changed = True
        else:
            out.extend(block)
        block = []

    for line in text.split("\n"):
        if not in_reminder and FENCE_LINE.match(line):
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        if in_reminder:
            block.append(line)
            if REMINDER_CLOSE in line:
                in_reminder = False
                flush()
            continue
        open_at = line.find(REMINDER_OPEN)
        if open_at != -1:
            before = line[:open_at]
            if before.strip():
                out.append(before)
            close_at = line.find(REMINDER_CLOSE, open_at)
            if close_at != -1:
                body = line[open_at : close_at + len(REMINDER_CLOSE)]
                if reminder_is_volatile(body):
                    changed = True
                else:
                    out.append(body)
                after = line[close_at + len(REMINDER_CLOSE) :]
                if after.strip():
                    out.append(after)
            else:
                in_reminder = True
                block.append(line[open_at:])
            continue
        if line_rules and any(rule.match(line) for rule in VOLATILE_LINE):
            # Dropped outright: a fixed-width marker would still come and go with
            # the line it replaced, which is the instability being removed.
            changed = True
            continue
        out.append(line)
    if in_reminder:
        flush()
    return ("\n".join(out), True) if changed else (text, False)


def _rewrite_content(content: Any, line_rules: bool) -> tuple[Any, bool]:
    if isinstance(content, str):
        text, changed = sanitize_text(content, line_rules)
        # System text loses its trailing whitespace so equal prompts really are
        # equal byte for byte; user text is never trimmed.
        if line_rules:
            trimmed = text.rstrip()
            changed = changed or trimmed != content
            text = trimmed
        return text, changed
    if not isinstance(content, list):
        return content, False
    changed = False
    out: list[Any] = []
    for part in content:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            out.append(part)
            continue
        if part.get("type") not in ("text", "input_text"):
            out.append(part)
            continue
        text, part_changed = sanitize_text(part["text"], line_rules)
        if line_rules:
            trimmed = text.rstrip()
            part_changed = part_changed or trimmed != part["text"]
            text = trimmed
        if not part_changed:
            out.append(part)
            continue
        changed = True
        if text.strip():
            out.append({**part, "text": text})
    return out, changed


def sanitize_messages(messages: list[Any]) -> list[Any]:
    """Strip volatile bytes from system/developer and user messages.

    Assistant messages are never touched: their bytes are history the model
    already produced.
    """
    changed = False
    out: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        role = message.get("role")
        line_rules = role in ("system", "developer")
        if not line_rules and role != "user":
            out.append(message)
            continue
        content, touched = _rewrite_content(message.get("content"), line_rules)
        if not touched:
            out.append(message)
            continue
        changed = True
        out.append({**message, "content": content})
    return out if changed else messages


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    if isinstance(tool.get("name"), str):
        return str(tool["name"])
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return str(function["name"])
    return ""


def sort_tools(tools: list[Any]) -> list[Any]:
    """Stable name order; the original list when it is already sorted."""
    names = [tool_name(tool) for tool in tools]
    if names == sorted(names):
        return tools
    return [tool for _, tool in sorted(zip(names, tools), key=lambda pair: pair[0])]


# ── append-only assert ───────────────────────────────────────────────


def first_diff_offset(previous: str, current: str) -> int:
    for index in range(min(len(previous), len(current))):
        if previous[index] != current[index]:
            return index
    return min(len(previous), len(current))


_baseline: str | None = None
_baseline_lock = threading.Lock()


def assert_append_only(prefix: str, model: str) -> bool:
    """Compare this request prefix with the previous one's.

    Items are joined without an enclosing array: a closed `[…]` is never a
    prefix of its own growth, which would false-positive every turn.
    """
    global _baseline
    with _baseline_lock:
        previous, _baseline = _baseline, prefix
    if previous is None or prefix.startswith(previous):
        return True
    offset = first_diff_offset(previous, prefix)
    debug(
        f"prefix drift at byte {offset} (prev {len(previous)}B, cur {len(prefix)}B, model {model or '?'}): "
        + json.dumps(prefix[max(0, offset - 40) : offset + 60])
    )
    return False


def canonical_prefix(body: dict[str, Any]) -> str:
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
    rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    head = json.dumps([body.get("model") or "", (system or {}).get("content") or "", body.get("tools") or []])
    return head + "\u0000" + "\u0000".join(json.dumps(message) for message in rest)


def rewrite_request(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    if isinstance(messages, list):
        body["messages"] = sanitize_messages(messages)
    tools = body.get("tools")
    if isinstance(tools, list) and len(tools) > 1:
        body["tools"] = sort_tools(tools)
    return body


# ── models ───────────────────────────────────────────────────────────

BRAND = {
    "gpt": "GPT", "oss": "OSS", "glm": "GLM", "qwen": "Qwen", "llama": "Llama",
    "gemma": "Gemma", "phi": "Phi", "mistral": "Mistral", "ministral": "Ministral",
    "magistral": "Magistral", "devstral": "Devstral", "codestral": "Codestral",
    "deepseek": "DeepSeek", "qwq": "QwQ", "qwen2": "Qwen", "nomic": "Nomic",
    "snowflake": "Snowflake", "command": "Command", "cogito": "Cogito", "yi": "Yi",
    "kimi": "Kimi", "minimax": "MiniMax", "nemotron": "Nemotron", "lfm": "LFM",
    "smollm": "SmolLM", "smol": "Smol", "tinyllama": "TinyLlama", "starcoder": "StarCoder",
    "codegemma": "CodeGemma", "granite": "Granite", "olmo": "OLMo", "falcon": "Falcon",
    "vicuna": "Vicuna", "openchat": "OpenChat", "aya": "Aya", "exaone": "EXAONE",
    "glm4": "GLM", "anthropic": "Anthropic",
}
TOKEN_RE = re.compile(r"[a-z]+[0-9]*(?:\.[0-9]+)*|[0-9]+(?:\.[0-9]+)*[a-z]*", re.IGNORECASE)
TAG_SKIP = re.compile(r"^(latest|f16|f32|fp16|fp32|bf16)$", re.IGNORECASE)


def tokens_label(tokens: int) -> str:
    """`131072` -> `128k`; `262144` -> `256k`; `1048576` -> `1M`.

    Decimal units are exact for published figures (128000), binary ones for real
    windows (1048576), so decimal is tried first and binary after. Mirrors
    mistral-vibe-hermes so both pickers read the same way.
    """
    for unit, size in (("M", 1000000), ("k", 1000), ("M", 1048576), ("k", 1024)):
        if tokens >= size and tokens % size == 0:
            return f"{tokens // size}{unit}"
    return str(tokens)


def model_specs(model: dict[str, Any]) -> str:
    """Capabilities line for a picker note, from what /api/show actually reported."""
    parts = [f"{tokens_label(int(model.get('contextWindow') or 32768))} ctx"]
    if model.get("reasoning"):
        parts.append("reasoning")
    return ", ".join(parts)


def model_note(model: dict[str, Any]) -> str:
    """The dim text beside a picker row: `Qwen3 Coder 30B (Ollama) · 64k ctx`.

    Hermes renders `id · note` and offers no other label slot, so the friendly name
    lives here; the ollama id stays untouched.
    """
    return f"{model.get('name') or model['id']} · {model_specs(model)}"



def display_name(model_id: str) -> str:
    """Human name in the style of the other two repos.

    `deepseek-v4.1-flash-1017` -> `DeepSeek V4.1 Flash (Ollama)`.
    """
    suffix = " (Ollama)"
    base = model_id.rsplit("/", 1)[-1]
    name, _, tag = base.partition(":")
    parts: list[str] = []
    for chunk in re.split(r"[-_\s]+", name):
        if chunk:
            parts.extend(TOKEN_RE.findall(chunk))
    if tag and not TAG_SKIP.match(tag):
        parts.extend(TOKEN_RE.findall(tag))
    while len(parts) > 1 and re.fullmatch(r"\d{3,}", parts[-1]):
        parts.pop()
    tokens: list[str] = []
    for part in parts:
        if tokens and re.fullmatch(r"\d+(?:\.\d+)*", tokens[-1]) and re.fullmatch(r"\d+", part):
            tokens[-1] = f"{tokens[-1]}.{part}"
        else:
            tokens.append(part)
    words: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in BRAND:
            words.append(BRAND[lower])
        elif token[:1].isdigit():
            words.append(token.upper())
        elif token[:1].islower():
            words.append(token[:1].upper() + token[1:])
        else:
            words.append(token)
    joined = re.sub(r"\bGPT OSS\b", "GPT-OSS", " ".join(words))
    return f"{joined}{suffix}" if joined else model_id


class _Catalog:
    """Cached `/api/tags` + `/api/show` catalog, mirrored to disk."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.models: list[dict[str, Any]] = []
        self.at = 0.0

    def mirror_path(self) -> Path:
        return state_dir() / "models.json"

    def read_mirror(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.mirror_path().read_text())
        except (OSError, ValueError) as error:
            debug(f"model mirror unreadable: {error}")
            return []
        models = payload.get("models") if isinstance(payload, dict) else None
        return models if isinstance(models, list) else []

    def write_mirror(self, models: list[dict[str, Any]]) -> None:
        try:
            path = self.mirror_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"version": 1, "at": int(time.time() * 1000), "models": models}, indent=2) + "\n")
            os.replace(tmp, path)
        except OSError as error:
            debug(f"model mirror not written: {error}")

    def fetch(self) -> list[dict[str, Any]]:
        base = ollama_base()
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as error:
            debug(f"catalog probe failed: {error}")
            return []
        entries = payload.get("models") if isinstance(payload, dict) else None
        ids = [
            str(entry.get("name") or entry.get("model"))
            for entry in (entries or [])
            if isinstance(entry, dict) and (entry.get("name") or entry.get("model")) and not NOT_CHAT.search(str(entry.get("name") or entry.get("model")))
        ]
        models: list[dict[str, Any]] = []
        for model_id in ids:
            show: dict[str, Any] = {}
            try:
                request = urllib.request.Request(
                    f"{base}/api/show",
                    data=json.dumps({"model": model_id}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    show = json.loads(response.read().decode())
            except (urllib.error.URLError, OSError, ValueError) as error:
                debug(f"show {model_id} failed: {error}")
            capabilities = [c.lower() for c in (show.get("capabilities") or []) if isinstance(c, str)]
            models.append(
                {
                    "id": model_id,
                    "name": display_name(model_id),
                    "reasoning": "thinking" in capabilities,
                    "contextWindow": context_from_show(show) or 32768,
                    "maxTokens": 32768,
                }
            )
        return models

    def get(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.models and time.time() - self.at < MODELS_TTL_S:
                return self.models
            models = self.fetch()
            if models:
                self.models, self.at = models, time.time()
                self.write_mirror(models)
            return self.models or self.read_mirror()


CATALOG = _Catalog()


def context_from_show(show: dict[str, Any]) -> int | None:
    parameters = show.get("parameters")
    if isinstance(parameters, str):
        match = re.search(r"(?:^|\n)\s*num_ctx\s+(\d+)", parameters)
        if match:
            return int(match.group(1))
    for key, value in (show.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


# ── metering + account ───────────────────────────────────────────────

_stats_lock = threading.Lock()
_stats = {
    "startedAt": int(time.time() * 1000),
    "requests": 0,
    "errors": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_tokens": 0,
}


def record_usage(usage: Any) -> None:
    if not isinstance(usage, dict):
        return

    def number(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) and value == value else 0

    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    with _stats_lock:
        _stats["input_tokens"] += number(usage.get("prompt_tokens") or usage.get("input_tokens"))
        _stats["output_tokens"] += number(usage.get("completion_tokens") or usage.get("output_tokens"))
        _stats["cached_tokens"] += number(details.get("cached_tokens") or usage.get("cached_tokens"))


def publish_state(port: int) -> None:
    try:
        directory = state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with _stats_lock:
            payload = dict(_stats)
        payload.update(
            {
                "service": SERVICE,
                "version": VERSION,
                "port": port,
                "pid": os.getpid(),
                "upstream": ollama_base(),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        tmp = directory / f"hermes-server.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, directory / "hermes-server.json")
    except OSError as error:
        debug(f"state file not written: {error}")


def read_cloud_info(balance: Any, usage: Any) -> dict[str, Any]:
    """Parse `/api/balance` + `/api/usage` into the report shape."""
    out: dict[str, Any] = {}
    root = balance if isinstance(balance, dict) else {}
    included = root.get("included") if isinstance(root.get("included"), dict) else None
    if included:
        period = included.get("period") if isinstance(included.get("period"), dict) else {}
        out["plan"] = {
            "balance": included.get("balance_usd"),
            "allowance": included.get("allowance_usd"),
            "resetsAt": period.get("until"),
        }
    purchased = (root.get("purchased") or {}).get("balance_usd") if isinstance(root.get("purchased"), dict) else None
    if isinstance(purchased, (int, float)):
        out["purchased"] = purchased
    plan_balance = out.get("plan", {}).get("balance")
    if isinstance(plan_balance, (int, float)) or isinstance(purchased, (int, float)):
        out["total"] = (plan_balance or 0) + (purchased or 0)
    totals = (usage or {}).get("totals") if isinstance(usage, dict) else None
    if isinstance(totals, dict):
        out["usage"] = {
            "requests": totals.get("request_count"),
            "usd": totals.get("usage_usd"),
            "input": totals.get("input_tokens"),
            "cached": totals.get("cached_input_tokens"),
            "output": totals.get("output_tokens"),
        }
    return out


_cloud_lock = threading.Lock()
_cloud_cache: dict[str, Any] = {"at": 0.0, "info": {}}


def cloud_info(key: str | None = None) -> dict[str, Any]:
    key = key or (os.environ.get("OLLAMA_CLOUD_KEY") or "").strip() or None
    if not key:
        return {}
    with _cloud_lock:
        if time.time() - float(_cloud_cache["at"]) < CLOUD_TTL_S:
            return dict(_cloud_cache["info"])
    host = (os.environ.get("OLLAMA_CLOUD_URL") or "https://ollama.com").rstrip("/")

    def read(path: str) -> tuple[Any, int | None]:
        status: int | None = None
        for attempt in range(1, CLOUD_ATTEMPTS + 1):
            request = urllib.request.Request(f"{host}{path}", headers={"Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(request, timeout=CLOUD_TIMEOUT_S) as response:
                    return json.loads(response.read().decode()), None
            except urllib.error.HTTPError as error:
                status = error.code
                if error.code not in RETRY_STATUS or attempt == CLOUD_ATTEMPTS:
                    return None, status
                wait = min(int(error.headers.get("Retry-After") or 1), 5)
                time.sleep(wait)
            except (urllib.error.URLError, OSError, ValueError) as error:
                debug(f"account {path} unreachable: {error}")
                return None, None
        return None, status

    balance, balance_status = read("/api/balance")
    usage, usage_status = read("/api/usage?range=24h")
    info = read_cloud_info(balance, usage)
    if balance is None and usage is None:
        if balance_status == 429 or usage_status == 429:
            info["error"] = "account rate limited (HTTP 429) — try again in a minute"
            with _cloud_lock:
                _cloud_cache.update({"at": 0.0, "info": {}})
            return info
        status = balance_status or usage_status
        info["error"] = (
            f"account HTTP {status} (key rejected or endpoint down)"
            if status is not None
            else "account endpoint unreachable or key rejected"
        )
    with _cloud_lock:
        _cloud_cache.update({"at": time.time(), "info": dict(info)})
    return info


def format_usage(port: int) -> str:
    """The `/ollama-usage` report, in the pi extension's line format."""
    with _stats_lock:
        stats = dict(_stats)
    models = CATALOG.get()
    info = cloud_info()
    prompt = stats["input_tokens"] + stats["cached_tokens"]
    hit = (100.0 * stats["cached_tokens"] / prompt) if prompt else 0.0
    lines = [
        "Ollama usage",
        f"server {ollama_base()} · proxy port {port} · v{VERSION}",
        f"requests {stats['requests']} · errors {stats['errors']} · catalog {len(models)} models",
    ]
    if prompt:
        lines.append(
            f"prompt {stats['input_tokens']} tok · cached {stats['cached_tokens']} ({hit:.1f}%) · "
            f"output {stats['output_tokens']} tok"
        )
    if not (os.environ.get("OLLAMA_CLOUD_KEY") or "").strip():
        lines.append("account: no OLLAMA_CLOUD_KEY set")
    else:
        plan = info.get("plan") or {}
        if plan.get("balance") is not None:
            reset = (plan.get("resetsAt") or "")[:10]
            lines.append(
                f"account: plan ${plan.get('balance')} left of ${plan.get('allowance')}"
                + (f" · resets {reset}" if reset else "")
            )
        if info.get("purchased") is not None:
            lines.append(f"account: credits ${info['purchased']} purchased")
        if info.get("total") is not None:
            lines.append(f"account: total ${info['total']} available")
        usage = info.get("usage") or {}
        if usage:
            lines.append(
                f"account: 24h {usage.get('requests') or 0} req · ${usage.get('usd') or 0} · "
                f"{usage.get('input') or 0} in · {usage.get('output') or 0} out"
            )
        if info.get("error"):
            lines.append(f"account: {info['error']}")
    if stats["requests"] == 0:
        lines.append("no requests yet this session — chat once, then run this again")
    return "\n".join(lines)


# ── loopback server ──────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"{SERVICE}/{VERSION}"

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?")[0].rstrip("/")
        port = self.server.server_address[1]
        if path in ("/healthz", "/health"):
            self._json(200, {"ok": True, "service": SERVICE, "version": VERSION, "port": port, "upstream": ollama_base()})
            return
        if path in ("/v1/models", "/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": model["id"], "object": "model", "created": 0, "owned_by": "ollama"}
                        for model in CATALOG.get()
                    ],
                },
            )
            return
        if path == "/api/models":
            self._json(200, {"models": CATALOG.get()})
            return
        if path == "/api/state":
            with _stats_lock:
                stats = dict(_stats)
            self._json(
                200,
                {
                    "ok": True,
                    "service": SERVICE,
                    "version": VERSION,
                    "port": port,
                    "upstream": ollama_base(),
                    **stats,
                    "hasKey": bool((os.environ.get("OLLAMA_CLOUD_KEY") or "").strip()),
                    "cloud": cloud_info(),
                    "models": [model["id"] for model in CATALOG.get()],
                },
            )
            return
        self._json(404, {"error": {"message": f"unknown route GET {path}"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"unknown route POST {path}"}})
            return
        body = self._read_body()
        if body is None:
            self.close_connection = True
            self._json(413, {"error": {"message": "request body too large"}})
            return
        rewrite_request(body)
        assert_append_only(canonical_prefix(body), str(body.get("model") or ""))
        with _stats_lock:
            _stats["requests"] += 1

        request = urllib.request.Request(
            f"{ollama_base()}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            with _stats_lock:
                _stats["errors"] += 1
            detail = error.read().decode("utf-8", "replace")[:2048]
            self._json(error.code, {"error": {"message": detail or str(error.reason), "type": "upstream_error"}})
            return
        except (urllib.error.URLError, OSError) as error:
            with _stats_lock:
                _stats["errors"] += 1
            self._json(502, {"error": {"message": f"could not reach ollama ({ollama_base()}): {error}"}})
            return

        content_type = response.headers.get("Content-Type") or "application/json"
        if not body.get("stream"):
            with response:
                raw = response.read()
            try:
                record_usage((json.loads(raw) or {}).get("usage"))
            except (ValueError, AttributeError):
                pass
            publish_state(self.server.server_address[1])
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        # Stream: the bytes go out untouched, but each complete SSE event is
        # scanned so the final `usage` chunk can be metered.
        self.send_response(200)
        self.send_header("Content-Type", content_type if "event-stream" in content_type else "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        carry = b""
        try:
            with response:
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    carry += chunk
                    while b"\n\n" in carry:
                        event, _, carry = carry.partition(b"\n\n")
                        for line in event.split(b"\n"):
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload or payload == b"[DONE]":
                                continue
                            try:
                                record_usage((json.loads(payload) or {}).get("usage"))
                            except (ValueError, AttributeError):
                                pass
                        publish_state(self.server.server_address[1])
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            publish_state(self.server.server_address[1])
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server:
    """Loopback server shared by every profile instance in this process."""

    lock = threading.Lock()
    httpd: ThreadingHTTPServer | None = None
    port: int = 0

    @classmethod
    def start(cls) -> int:
        """Bind once per process; never raise.

        Hermes' plugin doctor imports this module with sockets patched to fail,
        so a bind error must degrade to "no server" instead of aborting the
        import — a module that raises registers no ProviderProfile at all.
        """
        try:
            return cls._start()
        except Exception as error:  # noqa: BLE001 - import must always succeed
            debug(f"loopback server unavailable: {error}")
            cls.port = cls.port or int(os.environ.get("OLLAMA_HERMES_PORT") or DEFAULT_PORT)
            return cls.port

    @classmethod
    def _start(cls) -> int:
        with cls.lock:
            if cls.httpd is not None:
                return cls.port
            wanted = int(os.environ.get("OLLAMA_HERMES_PORT") or DEFAULT_PORT)
            httpd: ThreadingHTTPServer | None = None
            for offset in range(6):
                try:
                    httpd = ThreadingHTTPServer(("127.0.0.1", wanted + offset), _Handler)
                    break
                except OSError as error:
                    debug(f"port {wanted + offset} unavailable: {error}")
            if httpd is None:
                # Reuse a live sibling when one is already serving our endpoints
                # (a second Hermes process, or the opencode plugin's proxy).
                for candidate in (wanted, DEFAULT_PORT, 11500):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{candidate}/healthz", timeout=2) as response:
                            payload = json.loads(response.read().decode())
                        if payload.get("service") in (SERVICE, "ollama-cache-opencode"):
                            cls.port = candidate
                            print(f"[ollama-cache] reusing {payload.get('service')} on {candidate}", file=sys.stderr)
                            return candidate
                    except (urllib.error.URLError, OSError, ValueError):
                        continue
                print("[ollama-cache] no loopback port available; the provider will fail to reach ollama", file=sys.stderr)
                cls.port = wanted
                return wanted
            httpd.daemon_threads = True
            threading.Thread(target=httpd.serve_forever, name="ollama-cache", daemon=True).start()
            cls.httpd = httpd
            cls.port = httpd.server_address[1]
            publish_state(cls.port)
            debug(f"loopback on {cls.port}")
            return cls.port

    @classmethod
    def url(cls) -> str:
        return f"http://127.0.0.1:{cls.start()}/v1"


# ── profile ──────────────────────────────────────────────────────────


class OllamaCacheProfile(ProviderProfile):
    """Local ollama behind the rewriting proxy."""

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        """Picker rows: the ollama id stays the id, the note carries the name."""
        return [
            {"id": model["id"], "label": model.get("name") or model["id"], "note": model_note(model)}
            for model in CATALOG.get()
        ]

    def create_client(self, **client_kwargs: Any) -> Any:
        """Point the shared OpenAI client at this process's own loopback server."""
        try:
            import openai
        except ImportError:
            return None
        kwargs = {
            key: value
            for key, value in client_kwargs.items()
            if key in {"api_key", "timeout", "max_retries", "http_client", "default_headers"}
        }
        # ollama needs no credential; the OpenAI client requires a non-empty one.
        kwargs["api_key"] = "local"
        kwargs["base_url"] = _Server.url()
        return openai.OpenAI(**kwargs)

    def fetch_account_usage(self, *, api_key: str | None = None, base_url: str | None = None):
        """ollama.com account state, when a key is configured."""
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

        key = api_key or (os.environ.get("OLLAMA_CLOUD_KEY") or "").strip()
        if not key:
            return None
        info = cloud_info(key)
        plan = info.get("plan") or {}
        usage = info.get("usage") or {}
        if plan.get("balance") is None and not usage:
            return None
        allowance = plan.get("allowance") or 0
        balance = plan.get("balance") or 0
        spent_percent = round(100 * (1 - balance / allowance), 2) if allowance else None
        window = AccountUsageWindow(
            label="ollama.com 24h",
            used_percent=spent_percent,
            reset_at=None,
            detail=(
                f"${balance:.2f} of ${allowance:.2f} left"
                if allowance
                else f"{usage.get('requests') or 0} requests, ${usage.get('usd') or 0}"
            ),
        )
        return AccountUsageSnapshot(
            provider=self.name,
            source="ollama_com",
            fetched_at=datetime.now(timezone.utc),
            windows=(window,),
        )


def _build_profile() -> OllamaCacheProfile:
    """Register the profile with the local server already bound.

    Registration must happen at import: Hermes' plugin doctor and the provider
    registry both read `providers.register_provider` during module import, and a
    profile registered later is invisible to them.
    """
    port = _Server.start()
    profile = OllamaCacheProfile(
        name="ollama-cache",
        aliases=("ollama-local",),
        display_name="Ollama (prefix cache)",
        description="Local ollama behind a prompt-stabilizing proxy (prefix cache hits)",
        signup_url="https://ollama.com",
        # Hermes selects an api_key provider only when a credential resolves, and
        # a local ollama needs none — so OLLAMA_API_KEY exists purely to satisfy
        # that gate (any non-placeholder value works; ollama ignores it).
        env_vars=("OLLAMA_API_KEY", "OLLAMA_CLOUD_KEY"),
        base_url=f"http://127.0.0.1:{port}/v1",
        auth_type="api_key",
        api_mode="chat_completions",
        # No import-time probe: the catalog is discovered lazily by
        # discover_models(), and a socket here breaks hosts (Hermes' plugin
        # doctor denies network during import) that only want to inspect us.
    )
    register_provider(profile)
    return profile


PROFILE = _build_profile()


def register(ctx: Any) -> None:
    """Hermes plugin entry point: the profile is already registered at import."""
    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        register_command(
            "ollama-usage",
            lambda *args, **kwargs: print(format_usage(_Server.port)),
            "Ollama local + cloud account usage",
        )
    return PROFILE
