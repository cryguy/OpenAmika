"""Custom chat endpoints: any server that speaks the OpenAI chat-completions API (LM Studio, vLLM, llama.cpp, Groq,
DeepSeek…) as a transport for the four Build F roles.

``config/custom_llm.json`` (machine-local, like ollama.json) lists the endpoints: an id, a display name, the base URL,
the JSON mode its server accepts for the memory writer, how its server is told how much to think (a switch and an
effort per role), and its models, read from ``GET {base}/models`` or added by hand for a server without that route.
f_roles generates one option per endpoint, chat model and role ("<role>:custom:<endpoint>:<model>"), zero-priced in
the ledger: a local server charges nothing, and a paid one's price is not known here, so its cost is not tracked.

An endpoint's key is optional and lives only in ``brain/.env`` as ``CUSTOM_LLM_KEY_<ID>``. It is read when a call is
made and sent only to that endpoint's own URL, never along a redirect. Receipts, metering and the drain are the
OpenRouter client's; this module changes the request body, the usage and price checks, and strips the
``<think>…</think>`` blocks that a server without a reasoning parser streams inside the answer.
"""
import copy
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from f_memory import _write
from f_ollama import ANSWER_TOKENS, THINKING_ROLES
from f_openrouter import (FACT_SCHEMA, MAX_OUTPUT_TOKENS, OpenRouterClient, OpenRouterError, ROLES, _body_size, _merge,
                          normalize_usage, provider_detail, response_detail)
from f_usage import _label

CUSTOM = "custom"
CONFIG_NAME = "custom_llm.json"
JSON_MODES = ("json_object", "json_schema", "none")
# How a server is told how much to think; the OpenAI API has no field every server reads. "none" sends nothing and
# leaves it to the server. Each role has its own effort (None: nothing sent for it); chat_template_kwargs knows only
# on and off, so there "none" is off and any other effort on. By default only THINKING_ROLES think, as on Ollama:
# elsewhere thinking delays her voice and spends the memory writer's answer cap.
REASONING_SWITCHES = ("none", "reasoning_effort", "chat_template_kwargs")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORTS = {role: "low" if role in THINKING_ROLES else "none" for role in ANSWER_TOKENS}   # role order
KINDS = ("chat", "embedding")
SOURCES = ("detected", "manual")
KEY_PREFIX = "CUSTOM_LLM_KEY_"
MAX_NAME = 60
MAX_MODEL_NAME = 200
# Where OpenAI-compatible servers put a model's context window in their /models rows: OpenRouter-style catalogs,
# Groq, vLLM. Other servers leave it out, and then it is unknown.
CONTEXT_FIELDS = ("context_length", "context_window", "max_model_len")

_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?")
_KEY_RE = re.compile(r"[A-Za-z0-9._~+/=:@!$%^&*,;?|-]{1,512}")
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>\s*|\Z)", re.S)
# One writer at a time for config/custom_llm.json and brain/.env: two saves at once must not lose one.
_LOCK = threading.Lock()


def custom_base_url(url):
    """http(s)://host[:port][/path], without credentials, query or fragment; a pasted …/chat/completions is cut off."""
    text = str(url or "").strip()
    if not text:
        raise ValueError("Enter the server's address, e.g. http://127.0.0.1:1234/v1")
    if re.search(r"\s", text):
        raise ValueError("The address cannot contain spaces.")
    parts = urlsplit(text)
    try:
        parts.port
    except ValueError:
        raise ValueError("The port in that address must be a number between 0 and 65535.") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("The address must start with http:// or https:// and name a host, e.g. http://127.0.0.1:1234/v1")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Leave the user name and password out of the address; save a key for the endpoint instead.")
    if parts.query or parts.fragment:
        raise ValueError("Leave the ?… and #… parts out of the address.")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")].rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


def key_name(endpoint_id):
    """The brain/.env variable that holds an endpoint's key."""
    return KEY_PREFIX + endpoint_id.upper().replace("-", "_")


def endpoint_slug(name, taken=()):
    """A new endpoint id from its display name, unique among `taken` (lmstudio, lmstudio-2, …)."""
    base = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:32].strip("-") or "endpoint"
    slug, number = base, 1
    while slug in taken:
        number += 1
        suffix = f"-{number}"
        slug = base[:32 - len(suffix)].strip("-") + suffix
    return slug


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _model_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("name"), str):
        return None
    name = row["name"].strip()
    if not name or len(name) > MAX_MODEL_NAME:
        return None
    return {"name": name, "context_length": row["context_length"] if _positive_int(row.get("context_length")) else None,
            "kind": row["kind"] if row.get("kind") in KINDS else "chat",
            "source": row["source"] if row.get("source") in SOURCES else "manual"}


def _efforts(value):
    """role → effort or None (nothing sent), each missing or odd role at its default."""
    value = value if isinstance(value, dict) else {}
    return {role: value[role] if role in value and (value[role] is None or value[role] in EFFORTS) else default
            for role, default in DEFAULT_EFFORTS.items()}


def _endpoint_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not _ID_RE.fullmatch(row["id"]):
        return None
    name = row["name"].strip() if isinstance(row.get("name"), str) else ""
    if not 1 <= len(name) <= MAX_NAME:
        return None
    try:
        base = custom_base_url(row.get("base_url"))
    except ValueError:
        return None
    models, seen = [], set()
    for model in (_model_row(r) for r in (row["models"] if isinstance(row.get("models"), list) else [])):
        if model and model["name"] not in seen:
            seen.add(model["name"])
            models.append(model)
    return {"id": row["id"], "name": name, "base_url": base,
            "json_mode": row["json_mode"] if row.get("json_mode") in JSON_MODES else "json_object",
            "reasoning_switch": row["reasoning_switch"] if row.get("reasoning_switch") in REASONING_SWITCHES else "none",
            "reasoning_effort": _efforts(row.get("reasoning_effort")),
            "models": models, "detected_at": row["detected_at"] if isinstance(row.get("detected_at"), str) else None}


def load_custom_config(root):
    """config/custom_llm.json as the role manager uses it: no endpoints when the file is missing or odd, odd rows left
    out."""
    path = Path(root) / "config" / CONFIG_NAME
    doc = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            doc = {}
    rows = doc.get("endpoints") if isinstance(doc, dict) else None
    endpoints, seen = [], set()
    for endpoint in (_endpoint_row(r) for r in (rows if isinstance(rows, list) else [])):
        if endpoint and endpoint["id"] not in seen:
            seen.add(endpoint["id"])
            endpoints.append(endpoint)
    return {"endpoints": endpoints}


def save_custom_config(root, doc):
    _write(Path(root) / "config" / CONFIG_NAME, doc)


def _context_length(row):
    return next((row[key] for key in CONTEXT_FIELDS if _positive_int(row.get(key))), None)


def discover(http, base_url, key=None, *, timeout=(3, 10)):
    """The models an endpoint serves (``GET {base}/models``, OpenAI shape), sorted by name. Read-only; raises
    OpenRouterError when the server answers with something else, the transport's error when it does not answer."""
    base = custom_base_url(base_url)
    headers = {"Authorization": "Bearer " + key} if key else {}
    response = http.get(base + "/models", headers=headers, timeout=timeout, allow_redirects=False)
    try:
        status = getattr(response, "status_code", 200)
        if status in (401, 403):
            raise OpenRouterError(f"The server refused the key (HTTP {status}): " + (
                "check the key saved for this endpoint." if key else "it needs one; save a key for this endpoint."))
        if 300 <= status < 400:
            target = (getattr(response, "headers", None) or {}).get("Location") or "another address"
            raise OpenRouterError(f"The server redirects to {target}: use that address instead.")
        if status != 200:
            raise OpenRouterError(f"The server answered HTTP {status} to GET {base}/models")
        try:
            payload = response.json()
        except ValueError:
            raise OpenRouterError(f"GET {base}/models did not answer with JSON") from None
    finally:
        try:
            response.close()
        except Exception:
            pass
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise OpenRouterError(f"GET {base}/models did not answer with an OpenAI model list ({{\"data\": […]}})")
    models = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        kind = "embedding" if "embed" in row["id"].lower() else "chat"
        model = _model_row({"name": row["id"], "context_length": _context_length(row), "kind": kind, "source": "detected"})
        if model:
            models[model["name"]] = model
    return sorted(models.values(), key=lambda m: m["name"])


def merge_detected(models, found, keep=()):
    """An endpoint's model list after a detect: what the server lists now, plus the rows it does not list that were
    added by hand or that a role uses (`keep`; a server may list only the models it has loaded, and dropping one would
    move that role to the fallback preset). A row the server lists now becomes detected."""
    names = {m["name"] for m in found}
    kept = [m for m in models if m["name"] not in names and (m["source"] == "manual" or m["name"] in keep)]
    return sorted(list(found) + kept, key=lambda m: m["name"])


def custom_role_config(custom, endpoint_id, model_name, role, *, max_tokens=None):
    """Per-role parameters for a chat model of a custom endpoint (the shape the role manager and the client read)."""
    role = "memory_write" if role == "memory_extraction" else role
    endpoint = next((e for e in custom.get("endpoints") or [] if e["id"] == endpoint_id), None)
    if role not in ROLES or endpoint is None:
        raise ValueError("Unknown role or custom endpoint")
    row = next((m for m in endpoint["models"] if m["name"] == model_name and m["kind"] == "chat"), None)
    if row is None:
        raise ValueError(f"{model_name} is not a chat model of {endpoint['name']} (detect or add it first).")
    answer = ANSWER_TOKENS[role] if max_tokens is None else max_tokens
    if isinstance(answer, bool) or not isinstance(answer, int) or not 1 <= answer <= MAX_OUTPUT_TOKENS:
        raise ValueError("max_tokens must be an integer between 1 and 8192")
    if row["context_length"] and answer >= row["context_length"]:
        raise ValueError(f"{model_name}'s context window ({row['context_length']} tokens) is too small for the "
                         f"{role} role's answer ({answer} tokens).")
    model_id = endpoint_id + "/" + model_name
    try:
        _label(model_id)
    except ValueError:
        raise ValueError(f"The usage ledger cannot record the model name {model_name}: it takes letters, digits, "
                         f"spaces and . _ - : / @ + only, up to {159 - len(endpoint_id)} characters.") from None
    switch = endpoint["reasoning_switch"]
    effort = endpoint["reasoning_effort"][role] if switch != "none" else None
    return {"transport": CUSTOM, "model_id": model_id, "api_model": model_name, "canonical_slug": model_name,
            "provider_tag": CUSTOM, "provider_name": endpoint["name"], "endpoint_id": endpoint_id,
            "base_url": endpoint["base_url"], "json_mode": endpoint["json_mode"], "quantization": None,
            "role": role, "max_tokens": answer,
            "reasoning": None if effort is None else {"switch": switch, "effort": effort},
            "reasoning_switch": switch, "reasoning_effort": effort,
            "reasoning_enabled": None if effort is None else effort != "none",
            "mandatory_reasoning": False,
            "max_price": {"prompt": 0.0, "completion": 0.0, "request": 0.0}, "reservation_input_rate": 0.0,
            "endpoint_context_tokens": row["context_length"], "endpoint_max_completion_tokens": None,
            "supported_parameters": [], "structured_outputs_advertised": endpoint["json_mode"] == "json_schema"}


def make_custom_body(config, messages, *, stream=True, json_schema=None):
    """OpenAI chat-completions request: the answer cap, the endpoint's thinking switch and, for the memory writer, its
    JSON mode."""
    if config.get("transport") != CUSTOM or config.get("role") not in ROLES or not config.get("api_model"):
        raise ValueError("Use a validated custom role config")
    if not isinstance(messages, list) or not messages:
        raise ValueError("A nonempty message list is required")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Only system, user, and assistant text messages are supported")
        if not isinstance(message.get("content"), str):
            raise ValueError("This transport accepts text-only messages")
    body = {"model": config["api_model"], "messages": copy.deepcopy(messages), "stream": bool(stream),
            "max_tokens": config["max_tokens"]}
    if stream:
        body["stream_options"] = {"include_usage": True}
    effort = config.get("reasoning_effort")
    if effort is not None and config["reasoning_switch"] == "reasoning_effort":
        body["reasoning_effort"] = effort
    elif effort is not None and config["reasoning_switch"] == "chat_template_kwargs":
        body["chat_template_kwargs"] = {"enable_thinking": effort != "none"}
    mode = config.get("json_mode") if (config["role"] == "memory_write" or json_schema is not None) else "none"
    if mode == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "amika_memory_facts", "strict": True,
            "schema": copy.deepcopy(FACT_SCHEMA if json_schema is None else json_schema)}}
    elif mode == "json_object" and any("json" in message["content"].lower() for message in messages):
        # OpenAI, and servers that copy its checks, refuse JSON object mode when no message mentions JSON.
        body["response_format"] = {"type": "json_object"}
    _body_size(body, config)
    return body


def strip_think(text):
    """`text` without its <think>…</think> blocks (an unclosed one runs to the end) and the blank space after each."""
    return _THINK_BLOCK.sub("", text)


def _partial_tag(text, tag):
    """How many characters at the end of `text` could be the start of `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class _ThinkFilter:
    """strip_think for a stream: text arrives in chunks and a tag can be split across two of them."""

    def __init__(self):
        self.inside, self.held, self.trim = False, "", False

    def feed(self, text):
        text, self.held, out = self.held + text, "", []
        while text:
            if self.inside:
                end = text.find(_THINK_CLOSE)
                if end < 0:
                    size = _partial_tag(text, _THINK_CLOSE)
                    self.held = text[len(text) - size:] if size else ""
                    break
                text, self.inside, self.trim = text[end + len(_THINK_CLOSE):], False, True
                continue
            if self.trim:
                text = text.lstrip()
                if not text:
                    break
                self.trim = False
            start = text.find(_THINK_OPEN)
            if start >= 0:
                out.append(text[:start])
                text, self.inside = text[start + len(_THINK_OPEN):], True
                continue
            size = _partial_tag(text, _THINK_OPEN)
            out.append(text[:len(text) - size])
            self.held = text[len(text) - size:] if size else ""
            break
        return "".join(out)

    def flush(self):
        """The end of the answer: held text outside a block was answer text; an unclosed block is dropped."""
        text = "" if self.inside else self.held
        self.inside, self.held, self.trim = False, "", False
        return text


class CustomClient(OpenRouterClient):
    """An OpenAI-compatible endpoint with the OpenRouter client's receipts, metering and drain. Cost not tracked."""
    transport = CUSTOM

    def __init__(self, http, api_key, before_call, after_call, *, base_url, name, timeout=(10, 600), event_sink=None,
                 drainer=None):
        # The parent insists on a key; a local server usually has none, and then no Authorization header is sent.
        super().__init__(http, api_key or "none", before_call, after_call, timeout=timeout, event_sink=event_sink,
                         drainer=drainer)
        self._key = api_key or None
        self.provider_name = name
        self.chat_url = custom_base_url(base_url) + "/chat/completions"
        self._hidden = {}       # per call (receipt): the stream's think filter, read by the drain thread too

    def _chat_body(self, config, messages, *, stream, json_schema=None):
        return make_custom_body(config, messages, stream=stream, json_schema=json_schema)

    def _reservation(self, config, body):
        return 0.00001        # a positive reservation keeps the budget gate happy; the price is zero

    def _prepare(self, config, body, metadata, url):
        handle, receipt = super()._prepare(config, body, metadata, url)
        receipt["request_id"] = "custom-" + uuid.uuid4().hex[:12]     # until the server's own id arrives
        receipt["actual_provider"] = CUSTOM
        self._hidden[id(receipt)] = _ThinkFilter()
        return handle, receipt

    def _settle(self, handle, receipt, started, done, cancelled, error, **kwargs):
        self._hidden.pop(id(receipt), None)
        receipt["content"] = strip_think(receipt["content"])     # complete() reads the whole answer at once
        super()._settle(handle, receipt, started, done, cancelled, error, **kwargs)

    def _usage(self, receipt, embedding):
        raw = receipt.get("raw_usage") or {}
        reported = raw.get("prompt_tokens") is not None or raw.get("completion_tokens") is not None
        if reported:
            usage = normalize_usage(dict(raw, cost=0.0), embedding=embedding)
        else:
            # Some servers never report usage, even when asked to. That must not fail the turn: the counts read 0
            # and the warning says why.
            usage = normalize_usage({"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}, embedding=embedding)
            usage["warnings"].append("usage_not_reported")
        usage["cost_source"] = "not_tracked"
        return usage

    def _post(self, url, body, stream):
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        return self.http.post(url, headers=headers, json=body, stream=stream, timeout=self.timeout,
                              allow_redirects=False)

    def _raise_for_status(self, response):
        """The parent's, without OpenRouter's advice (credits, the Setup page's key) in what the server said."""
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool) and status >= 400:
            detail = response_detail(self.provider_name, response, hints=None)
            try:
                response.raise_for_status()
            except Exception as exc:
                try:
                    exc.provider_detail = detail
                except Exception:
                    pass
                raise
        response.raise_for_status()

    def _event(self, config, receipt, data):
        # No model or provider check: local servers report an alias or a file path as the model, and the provider is
        # the endpoint itself. A mismatch issue would leave the receipt "unknown" and fail the turn.
        if not isinstance(data, dict):
            raise OpenRouterError("Provider returned a non-object event")
        if data.get("id"):
            receipt["request_id"] = str(data["id"])
        if data.get("model"):
            receipt["actual_model"] = str(data["model"])
        if isinstance(data.get("usage"), dict):
            _merge(receipt["raw_usage"], data["usage"])
        if data.get("error"):
            error = OpenRouterError(self.provider_name + " returned an error event")
            error.provider_detail = provider_detail(self.provider_name, None, data["error"], hints=None)
            raise error
        if self.event_sink:
            self.event_sink(copy.deepcopy(data))

    def _stream_line(self, config, receipt, line, started):
        """One SSE line: None = skip, True = [DONE], else the answer text with its <think> blocks taken out."""
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line or not line.startswith("data:"):
            return None
        hidden = self._hidden[id(receipt)]
        raw = line[5:].strip()
        if raw == "[DONE]":
            receipt["content"] += hidden.flush()
            return True
        data = json.loads(raw)
        if receipt["timing"]["first_event_s"] is None:
            receipt["timing"]["first_event_s"] = time.perf_counter() - started
        self._event(config, receipt, data)
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        text = (choice.get("delta") or {}).get("content")
        if text is not None and not isinstance(text, str):
            raise OpenRouterError("Provider returned non-text content")
        text = hidden.feed(text or "")
        if choice.get("finish_reason") is not None:
            receipt["finish_reason"] = choice["finish_reason"]
            receipt["truncated"] = choice["finish_reason"] in {"length", "max_tokens"}
            text += hidden.flush()
        if text:
            if receipt["timing"]["first_content_s"] is None:
                receipt["timing"]["first_content_s"] = time.perf_counter() - started
            receipt["content"] += text
        return text

    def embed(self, *args, **kwargs):
        raise ValueError("The custom transport serves chat roles only")


def set_env_key(path, name, value):
    """Write (value) or remove (None) the one NAME=value line of a .env file, atomically, and apply it to this
    process at once: every call reads its key from the environment when it is made."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    out, written = [], False
    for line in lines:
        if line.strip().startswith(name + "="):
            if value is not None and not written:
                out.append(f"{name}={value}")
                written = True
            continue
        out.append(line)
    if value is not None and not written:
        out.append(f"{name}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(out) + "\n" if out else "")
    os.replace(temporary, path)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def key_state(endpoint_id):
    """What the page may know about an endpoint's key: whether there is one, and its last four characters."""
    key = os.environ.get(key_name(endpoint_id)) or ""
    # A short key's last four characters would be most of it.
    return {"present": bool(key), "hint": "…" + key[-4:] if len(key) >= 12 else None}


# ------------------------------------------------------------------ brain routes
def install_custom(g):
    """GET /api/build_f/custom (the endpoints, their models and keys, up or down), POST …/custom/endpoint (add,
    change or remove one, and its key), …/custom/detect (read its model list), …/custom/model (add or remove a model
    by hand) and …/custom/use (one model for all four roles, or a role → model and effort map). Installed by f_bridge
    after the role manager."""
    from f_roles import ROLES as ROLE_ORDER     # here, not at the top: f_roles imports this module
    app, request, jsonify = g["app"], g["request"], g["jsonify"]
    root = Path(g["__file__"]).resolve().parent.parent
    env_path = root / "brain" / ".env"

    def action_request(action):
        if request.headers.get("Origin") or request.headers.get("X-Build-F-Action") != action:
            raise PermissionError("Use the OpenAmika page for this action.")
        value = request.get_json(silent=True) if request.is_json else None
        if not isinstance(value, dict):
            raise ValueError("A JSON object is required.")
        return value

    def flag(value, name):
        on = value.get(name, False)
        if not isinstance(on, bool):
            raise ValueError(f"{name} must be true or false.")
        return on

    def find(custom, endpoint_id):
        endpoint = next((e for e in custom["endpoints"] if e["id"] == endpoint_id), None)
        if endpoint is None:
            raise ValueError("There is no such endpoint; reload the page.")
        return endpoint

    def selected(manager):
        """role → the custom config it uses now."""
        if manager is None:
            return {}
        with manager.lock:
            chosen = {role: manager.options[role][manager.selections[role]] for role in ROLE_ORDER}
        return {role: item["config"] for role, item in chosen.items() if item.get("transport") == CUSTOM}

    def roles_using(endpoint_id, model=None):
        return [role for role, config in selected(g.get("F_ROLES")).items()
                if config["endpoint_id"] == endpoint_id and (model is None or config["api_model"] == model)]

    def in_use(manager):
        return {role: {"endpoint": config["endpoint_id"], "model": config["api_model"]}
                for role, config in selected(manager).items()}

    def view(endpoint, probe):
        out = dict(endpoint, models=copy.deepcopy(endpoint["models"]), key=key_state(endpoint["id"]),
                   reachable=None, error=None)
        if probe:
            try:
                discover(g["http_requests"], endpoint["base_url"], os.environ.get(key_name(endpoint["id"])),
                         timeout=(2, 3))
                out["reachable"] = True
            except OpenRouterError as error:
                out["reachable"], out["error"] = False, str(error)[:160]     # it answered: a refused key, a redirect…
            except Exception:
                out["reachable"] = False        # nothing answered; the page says so with the address
        return out

    def status(probe=True):
        manager = g.get("F_ROLES")
        custom = manager.custom if manager is not None else load_custom_config(root)
        endpoints = custom["endpoints"]
        if probe and len(endpoints) > 1:
            # Side by side: one server that does not answer must not make the page wait for the others in turn.
            with ThreadPoolExecutor(max_workers=min(8, len(endpoints))) as pool:
                views = list(pool.map(lambda e: view(e, True), endpoints))
        else:
            views = [view(e, probe) for e in endpoints]
        return {"ok": True, "endpoints": views, "in_use": in_use(manager)}

    def save(doc):
        save_custom_config(root, doc)
        manager = g.get("F_ROLES")
        if manager is not None:
            manager.reload_custom()

    @app.route("/api/build_f/custom", methods=["GET"])
    def custom_get():
        return jsonify(status())

    @app.route("/api/build_f/custom/endpoint", methods=["POST"])
    def custom_endpoint():
        try:
            value = action_request("custom-endpoint")
            remove, clear_key = flag(value, "remove"), flag(value, "clear_key")
            key = value.get("key")
            if key is not None and not isinstance(key, str):
                raise ValueError("The key must be text.")
            key = (key or "").strip()
            if key and not _KEY_RE.fullmatch(key):
                raise ValueError("That key has spaces, quotes or other characters an API key does not have.")
            if key and clear_key:
                raise ValueError("Save a new key or clear the old one, not both.")
            with _LOCK:
                doc = load_custom_config(root)
                endpoint = find(doc, value["id"]) if value.get("id") is not None else None
                if remove:
                    if endpoint is None:
                        raise ValueError("Name the endpoint to remove.")
                    roles = roles_using(endpoint["id"])
                    if roles:
                        raise ValueError(f"{endpoint['name']} is in use: pick another model for {', '.join(roles)} first.")
                    doc["endpoints"].remove(endpoint)
                    save(doc)
                    set_env_key(env_path, key_name(endpoint["id"]), None)
                    return jsonify(status(probe=False))
                if endpoint is None:
                    endpoint = {"id": endpoint_slug(value.get("name") or "", {e["id"] for e in doc["endpoints"]}),
                                "name": "", "base_url": None, "json_mode": "json_object", "reasoning_switch": "none",
                                "reasoning_effort": dict(DEFAULT_EFFORTS), "models": [], "detected_at": None}
                    doc["endpoints"].append(endpoint)
                name = value.get("name", endpoint["name"])
                if not isinstance(name, str) or not 1 <= len(name.strip()) <= MAX_NAME:
                    raise ValueError(f"Give the endpoint a name of 1 to {MAX_NAME} characters.")
                mode = value.get("json_mode", endpoint["json_mode"])
                if mode not in JSON_MODES:
                    raise ValueError("json_mode must be json_object, json_schema or none.")
                switch = value.get("reasoning_switch", endpoint["reasoning_switch"])
                if switch not in REASONING_SWITCHES:
                    raise ValueError("reasoning_switch must be none, reasoning_effort or chat_template_kwargs.")
                efforts = value.get("reasoning_effort", {})
                valid = isinstance(efforts, dict) and all(
                    role in DEFAULT_EFFORTS and (effort is None or effort in EFFORTS) for role, effort in efforts.items())
                if not valid:
                    raise ValueError("reasoning_effort maps surface, deep, reflection or memory_write to one of "
                                     + ", ".join(EFFORTS) + ", or to null (nothing sent).")
                base_url = custom_base_url(value.get("base_url", endpoint["base_url"]))
                endpoint.update(name=name.strip(), base_url=base_url, json_mode=mode, reasoning_switch=switch,
                                reasoning_effort=dict(endpoint["reasoning_effort"], **efforts))
                save(doc)
                if key or clear_key:
                    set_env_key(env_path, key_name(endpoint["id"]), key or None)
            return jsonify(status(probe=False))
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.route("/api/build_f/custom/detect", methods=["POST"])
    def custom_detect():
        try:
            value = action_request("custom-detect")
            endpoint = find(load_custom_config(root), value.get("id"))
            try:
                found = discover(g["http_requests"], endpoint["base_url"], os.environ.get(key_name(endpoint["id"])))
            except OpenRouterError as error:
                return jsonify({"ok": False, "error": f"{endpoint['name']} ({endpoint['base_url']}): {error}"}), 400
            except Exception as error:
                return jsonify({"ok": False, "error": f"Nothing is answering at {endpoint['base_url']} — is the "
                                                      f"server running? · {str(error)[:120]}"}), 503
            with _LOCK:
                doc = load_custom_config(root)
                current = find(doc, endpoint["id"])
                in_use_here = {config["api_model"] for config in selected(g.get("F_ROLES")).values()
                               if config["endpoint_id"] == endpoint["id"]}
                current.update(models=merge_detected(current["models"], found, keep=in_use_here),
                               detected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                save(doc)
            out = status(probe=False)
            for row in out["endpoints"]:
                if row["id"] == endpoint["id"]:
                    row["reachable"] = True
            return jsonify(out)
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.route("/api/build_f/custom/model", methods=["POST"])
    def custom_model():
        try:
            value = action_request("custom-model")
            name = value.get("name")
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= MAX_MODEL_NAME:
                raise ValueError(f"Enter the model's id as the server knows it (1 to {MAX_MODEL_NAME} characters).")
            name = name.strip()
            remove = flag(value, "remove")
            with _LOCK:
                doc = load_custom_config(root)
                endpoint = find(doc, value.get("id"))
                row = next((m for m in endpoint["models"] if m["name"] == name), None)
                if remove:
                    if row is None:
                        raise ValueError(f"{endpoint['name']} does not list {name}.")
                    roles = roles_using(endpoint["id"], name)
                    if roles:
                        raise ValueError(f"{name} is in use: pick another model for {', '.join(roles)} first.")
                    endpoint["models"].remove(row)
                elif row is None:
                    try:
                        _label(endpoint["id"] + "/" + name)
                    except ValueError:
                        raise ValueError(f"The usage ledger cannot record that model name: it takes letters, digits, "
                                         f"spaces and . _ - : / @ + only, up to {159 - len(endpoint['id'])} "
                                         f"characters.") from None
                    endpoint["models"].append({"name": name, "context_length": None, "kind": "chat", "source": "manual"})
                    endpoint["models"].sort(key=lambda m: m["name"])
                save(doc)
            return jsonify(status(probe=False))
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.route("/api/build_f/custom/use", methods=["POST"])
    def custom_use():
        manager = g.get("F_ROLES")
        if manager is None:
            return jsonify({"ok": False, "error": "Model choices are not installed yet."}), 503
        try:
            value = action_request("custom-use")
            roles = value.get("roles") if isinstance(value.get("roles"), dict) else {}
            if set(roles) - set(ROLE_ORDER):
                raise ValueError("roles maps " + ", ".join(ROLE_ORDER) + " to an endpoint and a model.")
            # A top-level endpoint and model fill every role that roles does not name; without them, a role that roles
            # does not name keeps what it runs now (a page that shows OpenRouter or Ollama roles leaves them alone).
            everyone = "endpoint" in value or "model" in value
            if not everyone and not roles:
                raise ValueError("Pick an endpoint and one of its models (one for all four roles, or one per role).")
            # force: after Unlock (f_controls) — work still in flight keeps its model, new calls take this one
            force = flag(value, "force")
            picks, efforts = {}, {}
            for role in ROLE_ORDER:
                if role not in roles and not everyone:
                    continue
                pick = roles.get(role, value)
                endpoint, model = (pick.get("endpoint"), pick.get("model")) if isinstance(pick, dict) else (None, None)
                if not isinstance(endpoint, str) or not isinstance(model, str) or not model.strip():
                    raise ValueError("Pick an endpoint and one of its models (one for all four roles, or one per role).")
                key = f"{role}:custom:{endpoint}:{model.strip()}"
                if key not in manager.options[role]:
                    try:
                        custom_role_config(manager.custom, endpoint, model.strip(), role)
                        reason = f"{model} is not offered for the {role} role."
                    except ValueError as error:
                        reason = str(error)
                    raise ValueError(reason)
                picks[role] = key
                # how hard the role thinks on that endpoint: kept with the endpoint, like the efforts …/endpoint saves
                if "effort" in pick:
                    if pick["effort"] is not None and pick["effort"] not in EFFORTS:
                        raise ValueError("effort is one of " + ", ".join(EFFORTS) + ", or null (nothing sent).")
                    efforts.setdefault(endpoint, {})[role] = pick["effort"]
            # The switch first: a refused one (busy) changes nothing, efforts included. The lock keeps another switch
            # from landing between reading the roles this one keeps and setting them.
            with manager.lock:
                selections = dict(manager.selections, **picks)
                result = manager.set(selections, force=True) if force else manager.set(selections)
            if efforts:
                with _LOCK:
                    doc = load_custom_config(root)
                    for endpoint_id, by_role in efforts.items():
                        endpoint = find(doc, endpoint_id)
                        endpoint["reasoning_effort"] = dict(endpoint["reasoning_effort"], **by_role)
                    save(doc)
            out = status(probe=False)
            return jsonify({"ok": True, "selections": result["selections"], "transport": result["transport"],
                            "in_use": out["in_use"], "endpoints": out["endpoints"]})
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        except RuntimeError as error:
            reasons = list(getattr(error, "reasons", None) or [])
            message = ("Model selection is busy: " + "; ".join(reasons) + ". Wait a moment, or press Unlock (she stops "
                       "what she is doing).") if reasons else "Model selection is busy. Wait for all speech and memory work to finish."
            return jsonify({"ok": False, "error": message, "code": "model_busy", "busy": reasons}), 409

    print("[custom] OpenAI-compatible endpoints installed")
