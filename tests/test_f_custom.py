"""Custom chat endpoints: any OpenAI-compatible server as a role transport (f_custom + the role manager's options)."""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))
from f_custom import (CUSTOM, CustomClient, _ThinkFilter, custom_base_url, custom_role_config, discover,   # noqa: E402
                      endpoint_slug, install_custom, key_name, load_custom_config, make_custom_body,
                      save_custom_config, strip_think)
from f_memory import MemoryRuntime                                   # noqa: E402
from f_ollama import ANSWER_TOKENS                                   # noqa: E402
from f_openrouter import FACT_SCHEMA, OpenRouterError                # noqa: E402
from f_roles import ROLES, NATIVE_KEYS, install_roles              # noqa: E402
from f_usage import UsageMeter                                      # noqa: E402

KEY = "sk-local-abcdefgh1234"
ENDPOINT = {"id": "lmstudio", "name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1", "json_mode": "json_object",
            "models": [{"name": "qwen3-32b", "context_length": 32768, "kind": "chat", "source": "detected"},
                       {"name": "text-embedding-nomic-embed-text-v1.5", "context_length": None, "kind": "embedding",
                        "source": "detected"},
                       {"name": "tiny-model", "context_length": 2048, "kind": "chat", "source": "manual"}],
            "detected_at": "2026-11-18T10:00:00+00:00"}


class Response:
    def __init__(self, lines=None, data=None, status=200, headers=None):
        self.status_code, self._lines, self._data, self.encoding = status, lines or [], data, None
        self.headers, self.closed = headers or {}, False
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http " + str(self.status_code))
    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)
    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data
    def close(self):
        self.closed = True


class HTTP:
    def __init__(self):
        self.responses, self.calls = [], []
    def _next(self, url, kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    def post(self, url, **kwargs):
        return self._next(url, kwargs)
    def get(self, url, **kwargs):
        return self._next(url, kwargs)


def chunk(text=None, finish=None, model="qwen3-32b"):
    return {"id": "chatcmpl-7", "model": model,
            "choices": [{"index": 0, "delta": {} if text is None else {"content": text}, "finish_reason": finish}]}


def sse(*events):
    return ["data: " + json.dumps(event) for event in events] + ["", "data: [DONE]"]


USAGE = {"id": "chatcmpl-7", "model": "qwen3-32b", "choices": [],
         "usage": {"prompt_tokens": 812, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 512}}}


class AddressAndNameTests(unittest.TestCase):
    def test_base_url_is_normalised(self):
        self.assertEqual(custom_base_url(" http://127.0.0.1:1234/v1/ "), "http://127.0.0.1:1234/v1")
        self.assertEqual(custom_base_url("https://api.groq.com/openai/v1/chat/completions"), "https://api.groq.com/openai/v1")
        self.assertEqual(custom_base_url("http://box:8000"), "http://box:8000")
        self.assertEqual(custom_base_url("http://[::1]:8080/compatible-mode/v1"), "http://[::1]:8080/compatible-mode/v1")

    def test_base_url_refusals_say_what_to_change(self):
        for url, words in (("", "address"), ("ftp://box/v1", "http://"), ("127.0.0.1:1234/v1", "http://"),
                           ("http://user:pw@box/v1", "password"), ("http://box/v1?x=1", "?"), ("http://box/v1#a", "#"),
                           ("http://box:99999/v1", "port"), ("http://box /v1", "spaces"), ("http:///v1", "host")):
            with self.assertRaises(ValueError, msg=url) as caught:
                custom_base_url(url)
            self.assertIn(words, str(caught.exception), url)

    def test_ids_and_key_names(self):
        self.assertEqual(endpoint_slug("LM Studio"), "lm-studio")
        self.assertEqual(endpoint_slug("LM Studio", {"lm-studio", "lm-studio-2"}), "lm-studio-3")
        self.assertEqual(endpoint_slug("本地"), "endpoint")
        long = endpoint_slug("x" * 40, {"x" * 32})
        self.assertEqual((len(long), long[-2:]), (32, "-2"))
        self.assertNotIn(":", endpoint_slug("a:b"))
        self.assertEqual(key_name("lm-studio-2"), "CUSTOM_LLM_KEY_LM_STUDIO_2")


class ConfigTests(unittest.TestCase):
    def test_missing_odd_and_partly_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_custom_config(root), {"endpoints": []})
            (root / "config").mkdir()
            (root / "config" / "custom_llm.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(load_custom_config(root), {"endpoints": []})
            (root / "config" / "custom_llm.json").write_text('{"endpoints": 3}', encoding="utf-8")
            self.assertEqual(load_custom_config(root), {"endpoints": []})
            rows = [dict(ENDPOINT, base_url="http://127.0.0.1:1234/v1/", json_mode="xml",
                         models=ENDPOINT["models"] + [{"name": ""}, {"name": "x" * 201}, {"name": "qwen3-32b"}, "odd",
                                                      {"name": "hand-added", "kind": "odd"}]),
                    dict(ENDPOINT, name="Duplicate id"), dict(ENDPOINT, id="Bad:Id"), dict(ENDPOINT, id="bad-url", base_url="box"),
                    dict(ENDPOINT, id="long", name="n" * 61), dict(ENDPOINT, id="no-models", models={"a": 1}), "odd"]
            save_custom_config(root, {"endpoints": rows})
            doc = load_custom_config(root)
            self.assertEqual([e["id"] for e in doc["endpoints"]], ["lmstudio", "no-models"])
            first = doc["endpoints"][0]
            self.assertEqual((first["base_url"], first["json_mode"]), ("http://127.0.0.1:1234/v1", "json_object"))
            self.assertEqual([m["name"] for m in first["models"]],
                             ["qwen3-32b", "text-embedding-nomic-embed-text-v1.5", "tiny-model", "hand-added"])
            self.assertEqual(first["models"][-1], {"name": "hand-added", "context_length": None, "kind": "chat", "source": "manual"})
            self.assertEqual(doc["endpoints"][1]["models"], [])


class DiscoveryTests(unittest.TestCase):
    def test_reads_ids_context_and_kind(self):
        http = HTTP()
        http.responses = [Response(data={"object": "list", "data": [
            {"id": "zeta", "max_model_len": 8192}, {"id": "gamma", "context_length": 32768},
            {"id": "beta", "context_length": True, "context_window": 131072}, {"id": "Nomic-Embed-Text"},
            {"id": "alpha", "meta": {"n_ctx_train": 40960}, "loaded_context_length": 4096, "type": "embeddings"},
            {"id": "  "}, {"object": "model"}, "odd"]})]
        found = discover(http, "http://127.0.0.1:1234/v1/")
        # only the standard fields count: a server-specific context figure or type is not read
        self.assertEqual([(m["name"], m["context_length"], m["kind"]) for m in found],
                         [("Nomic-Embed-Text", None, "embedding"), ("alpha", None, "chat"), ("beta", 131072, "chat"),
                          ("gamma", 32768, "chat"), ("zeta", 8192, "chat")])
        self.assertTrue(all(m["source"] == "detected" for m in found))
        url, kwargs = http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:1234/v1/models")
        self.assertEqual(kwargs["headers"], {})
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertTrue(http.calls and http.responses == [])

    def test_key_header_and_refusals(self):
        http = HTTP()
        http.responses = [Response(data={"data": []})]
        discover(http, "https://api.example.com/v1", KEY)
        self.assertEqual(http.calls[0][1]["headers"], {"Authorization": "Bearer " + KEY})
        for response, words in ((Response(status=401), "refused the key"), (Response(status=403), "refused the key"),
                                (Response(status=500), "HTTP 500"), (Response(data=ValueError("no json")), "JSON"),
                                (Response(data={"models": []}), "OpenAI model list"),
                                (Response(status=301, headers={"Location": "https://api.example.com/v1/models"}),
                                 "https://api.example.com/v1/models")):
            http.responses = [response]
            with self.assertRaises(OpenRouterError) as caught:
                discover(http, "http://api.example.com/v1")
            self.assertIn(words, str(caught.exception))
            self.assertTrue(response.closed)


class BodyTests(unittest.TestCase):
    def setUp(self):
        self.custom = {"endpoints": [dict(ENDPOINT)]}

    def test_role_config_and_caps(self):
        cfg = custom_role_config(self.custom, "lmstudio", "qwen3-32b", "deep")
        self.assertEqual((cfg["transport"], cfg["model_id"], cfg["api_model"], cfg["provider_name"], cfg["endpoint_id"]),
                         (CUSTOM, "lmstudio/qwen3-32b", "qwen3-32b", "LM Studio", "lmstudio"))
        self.assertEqual((cfg["max_price"]["completion"], cfg["endpoint_context_tokens"], cfg["reasoning"]), (0.0, 32768, None))
        for role in ROLES:
            self.assertEqual(custom_role_config(self.custom, "lmstudio", "qwen3-32b", role)["max_tokens"], ANSWER_TOKENS[role])
        with self.assertRaises(ValueError):
            custom_role_config(self.custom, "lmstudio", "text-embedding-nomic-embed-text-v1.5", "deep")
        with self.assertRaises(ValueError):
            custom_role_config(self.custom, "nope", "qwen3-32b", "deep")
        with self.assertRaisesRegex(ValueError, "context window"):
            custom_role_config(self.custom, "lmstudio", "tiny-model", "memory_write")
        self.assertEqual(custom_role_config(self.custom, "lmstudio", "tiny-model", "deep")["max_tokens"], 1500)
        odd = {"endpoints": [dict(ENDPOINT, models=[{"name": "Qwen (Q4)", "context_length": None, "kind": "chat",
                                                     "source": "manual"}])]}
        with self.assertRaisesRegex(ValueError, "usage ledger"):
            custom_role_config(odd, "lmstudio", "Qwen (Q4)", "deep")

    def test_body_per_json_mode(self):
        asks = [{"role": "system", "content": "Reply in JSON."}, {"role": "user", "content": "u"}]
        plain = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        deep = custom_role_config(self.custom, "lmstudio", "qwen3-32b", "deep")
        body = make_custom_body(deep, asks)
        self.assertEqual(body, {"model": "qwen3-32b", "messages": asks, "stream": True, "max_tokens": 1500,
                                "stream_options": {"include_usage": True}})
        writer = custom_role_config(self.custom, "lmstudio", "qwen3-32b", "memory_write")
        self.assertEqual(make_custom_body(writer, asks, stream=False)["response_format"], {"type": "json_object"})
        self.assertNotIn("stream_options", make_custom_body(writer, asks, stream=False))
        self.assertNotIn("response_format", make_custom_body(writer, plain))
        schema = dict(self.custom["endpoints"][0], json_mode="json_schema")
        writer = custom_role_config({"endpoints": [schema]}, "lmstudio", "qwen3-32b", "memory_write")
        self.assertTrue(writer["structured_outputs_advertised"])
        rf = make_custom_body(writer, plain)["response_format"]
        self.assertEqual((rf["type"], rf["json_schema"]["name"], rf["json_schema"]["strict"], rf["json_schema"]["schema"]),
                         ("json_schema", "amika_memory_facts", True, FACT_SCHEMA))
        none = dict(self.custom["endpoints"][0], json_mode="none")
        writer = custom_role_config({"endpoints": [none]}, "lmstudio", "qwen3-32b", "memory_write")
        self.assertNotIn("response_format", make_custom_body(writer, asks))
        with self.assertRaises(ValueError):
            make_custom_body(deep, [{"role": "tool", "content": "x"}])


class ThinkTests(unittest.TestCase):
    def test_strip_think(self):
        self.assertEqual(strip_think("<think>a\nb</think>\n\n{\"facts\": []}"), "{\"facts\": []}")
        self.assertEqual(strip_think("Hi <think>x</think>  there <think>never closed"), "Hi there ")
        self.assertEqual(strip_think("no tags < here"), "no tags < here")

    def test_the_stream_filter_matches_strip_think_however_the_text_is_cut(self):
        for text in ("<think>a</think>\n\nHello", "Hi <think>x</think>  there <think>never closed", "a < b <th <think>q</think>c",
                     "x<think>y</thi", "</think>stray", "<<think>>a</think> b<", "<think>a</think><think>b</think> c"):
            for i in range(len(text) + 1):
                for j in range(i, len(text) + 1):
                    hidden = _ThinkFilter()
                    out = hidden.feed(text[:i]) + hidden.feed(text[i:j]) + hidden.feed(text[j:]) + hidden.flush()
                    self.assertEqual(out, strip_think(text), (text, i, j))


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.custom = {"endpoints": [dict(ENDPOINT)]}
        self.http = HTTP()
        self.handles, self.receipts, self.events = [], [], []
        self.client = self.make(None)

    def make(self, key):
        return CustomClient(self.http, key, lambda request: self.handles.append(request) or "h1",
                            lambda handle, receipt: self.receipts.append((handle, receipt)),
                            base_url="http://127.0.0.1:1234/v1/", name="LM Studio", event_sink=self.events.append)

    def run_stream(self, client, *events, role="deep"):
        cfg = custom_role_config(self.custom, "lmstudio", "qwen3-32b", role)
        self.http.responses.append(Response(lines=sse(*events)))
        return list(client.stream(cfg, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                  metadata={"phase": role, "turn_id": "t1"}))

    def test_think_blocks_split_across_chunks_are_never_yielded(self):
        pieces = self.run_stream(self.client, chunk("<thi"), chunk("nk>Let me think: reply: no"), chunk(" really.</th"),
                                 chunk("ink>\n\n"), chunk("[casual_happy]\"Hel"), chunk("lo.\" <thi"), chunk("nk>more"),
                                 chunk("</think> Bye"), chunk(finish="stop"), USAGE)
        self.assertEqual("".join(pieces), "[casual_happy]\"Hello.\" Bye")
        self.assertFalse([p for p in pieces if "think" in p or "<" in p])
        handle, receipt = self.receipts[0]
        self.assertEqual((receipt["content"], receipt["status"], receipt["cost_usd"]), ("[casual_happy]\"Hello.\" Bye", "complete", 0.0))
        self.assertEqual(receipt["usage"]["tokens"]["cache_read_tokens"], 512)
        self.assertEqual(receipt["usage"]["cost_source"], "not_tracked")
        self.assertEqual((receipt["request_id"], receipt["actual_provider"]), ("chatcmpl-7", CUSTOM))
        self.assertEqual(self.handles[0]["estimated_max_usd"], 0.00001)
        self.assertIn("<thi", "".join(json.dumps(e) for e in self.events))      # the audit sink keeps the raw events
        url, kwargs = self.http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:1234/v1/chat/completions")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertIs(kwargs["allow_redirects"], False)

    def test_unclosed_block_and_a_held_partial_tag_at_the_end(self):
        self.assertEqual(self.run_stream(self.client, chunk("<think>still going"), chunk(finish="length"), USAGE), [])
        self.assertEqual((self.receipts[0][1]["content"], self.receipts[0][1]["truncated"]), ("", True))
        self.assertEqual("".join(self.run_stream(self.client, chunk("2 <"), chunk(finish="stop"), USAGE)), "2 <")
        self.assertEqual(self.receipts[1][1]["content"], "2 <")

    def test_a_stream_without_usage_completes_with_a_warning(self):
        text = "".join(self.run_stream(self.client, chunk("reply: hi"), chunk(finish="stop")))
        receipt = self.receipts[0][1]
        self.assertEqual((text, receipt["status"], receipt["cost_usd"]), ("reply: hi", "complete", 0.0))
        self.assertIn("usage_not_reported", receipt["warnings"])
        self.assertEqual(receipt["issues"], [])

    def test_a_model_alias_is_recorded_not_refused(self):
        alias = "/models/Qwen3-32B-Q4_K_M.gguf"
        self.run_stream(self.client, chunk("ok", model=alias), chunk(finish="stop", model=alias), dict(USAGE, model=alias))
        receipt = self.receipts[0][1]
        self.assertEqual((receipt["status"], receipt["actual_model"], receipt["issues"]), ("complete", alias, []))

    def test_the_key_is_sent_only_when_set(self):
        self.run_stream(self.make(KEY), chunk("ok"), chunk(finish="stop"), USAGE)
        self.assertEqual(self.http.calls[0][1]["headers"]["Authorization"], "Bearer " + KEY)

    def test_errors_carry_the_servers_words_without_openrouter_advice(self):
        cfg = custom_role_config(self.custom, "lmstudio", "qwen3-32b", "surface")
        self.http.responses = [Response(status=401, data={"error": {"message": "Invalid API key"}})]
        with self.assertRaises(RuntimeError) as caught:
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}], metadata={"phase": "surface"}))
        self.assertEqual(caught.exception.provider_detail, "LM Studio (HTTP 401): Invalid API key")
        self.http.responses = [Response(lines=sse({"error": {"message": "model not found", "code": 404}}))]
        with self.assertRaises(OpenRouterError) as caught:
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}], metadata={"phase": "surface"}))
        self.assertEqual(caught.exception.provider_detail, "LM Studio (HTTP 404): model not found")
        self.assertEqual([r["status"] for _, r in self.receipts], ["unknown", "unknown"])

    def test_complete_returns_clean_json_after_a_think_block(self):
        cfg = custom_role_config(self.custom, "lmstudio", "qwen3-32b", "memory_write")
        self.http.responses = [Response(data={"id": "c1", "model": "qwen3-32b", "choices": [
            {"message": {"role": "assistant", "content": "<think>which facts?</think>\n{\"facts\": []}"},
             "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 12}})]
        result = self.client.complete(cfg, [{"role": "system", "content": "reply in JSON"}, {"role": "user", "content": "u"}],
                                      metadata={"phase": "memory_write"})
        self.assertEqual(result["text"], "{\"facts\": []}")
        self.assertEqual(self.receipts[0][1]["content"], "{\"facts\": []}")
        self.assertEqual(self.http.calls[0][1]["json"]["response_format"], {"type": "json_object"})
        with self.assertRaises(ValueError):
            self.client.embed({}, ["x"])


class Request:
    def __init__(self):
        self.headers, self.body, self.is_json = {}, None, True
    def get_json(self, silent=False):
        return self.body


class App:
    def __init__(self):
        self.routes = {}
    def route(self, path, methods=("GET",)):
        def register(fn):
            for method in methods:
                self.routes[(path, method)] = fn
            return fn
        return register


class RoleManagerTests(unittest.TestCase):
    """The role manager offers one option per chat model and role; the routes add, detect, pick and remove them."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        (self.root / "brain").mkdir()
        (self.root / "brain" / ".env").write_text("OPENROUTER_API_KEY=sk-or-x\n", encoding="utf-8")
        shutil.copyfile(ROOT / "config/openrouter_models.json", self.root / "config/openrouter_models.json")
        document = {"version": 1, "options": {}, "selections": {}, "presets": [
            {"name": "cloud", "provider": "openrouter", "selections": {role: "or:" + role for role in ROLES}}], "fallback_preset": "cloud"}
        for role in ROLES:
            document["options"][role] = [
                {"key": "native:" + NATIVE_KEYS[role], "native_key": NATIVE_KEYS[role], "name": "Native"},
                {"key": "or:" + role, "model_id": "deepseek/deepseek-v4-flash-0731", "name": "Reviewed " + role}]
            document["selections"][role] = "or:" + role
        (self.root / "config/llm_roles.json").write_text(json.dumps(document), encoding="utf-8")
        save_custom_config(self.root, {"endpoints": [ENDPOINT]})
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop(key_name("lmstudio"), None)
        self.memory = MemoryRuntime(self.root / "memory_f/now", lambda events, chunk: {"native": True},
                                    lambda texts, task: [[1.] + [0.] * 767 for t in texts])
        self.meter = UsageMeter(self.root / "memory_f/usage.sqlite", {"models": {}}, budget_usd=1)
        self.local = threading.local()
        self.request = Request()
        self.g = {"AVAILABLE_MODELS": {key: {"name": key, "provider": "native", "model": key} for key in NATIVE_KEYS.values()},
                  "_ws_active_lock": threading.RLock(), "_ws_active": {}, "_thinking_inflight": 0,
                  "http_requests": HTTP(), "OPENROUTER_KEY": "fake-key",
                  "_pick_stream": lambda p, c, **kw: iter(["native"]),
                  "_openai_messages": lambda p, **kw: [{"role": "system", "content": p.get("system", "")},
                                                      {"role": "user", "content": p["user_content"]}],
                  "app": App(), "request": self.request, "jsonify": lambda value: value,
                  "__file__": str(self.root / "brain" / "webui.py")}
        @contextlib.contextmanager
        def scope(phase, turn_id="", **kwargs):
            yield
        self.manager = install_roles(self.g, self.root, self.memory, self.meter, scope, self.local, {})
        install_custom(self.g)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def call(self, path, action=None, body=None, origin=None):
        self.request.headers = {"X-Build-F-Action": action} if action else {}
        if origin:
            self.request.headers["Origin"] = origin
        self.request.body = body
        result = self.g["app"].routes[(path, "POST" if action or body is not None else "GET")]()
        return result if isinstance(result, tuple) else (result, 200)

    def test_options_pricing_reload_and_dispatch(self):
        for role in ROLES:
            item = self.manager.options[role][f"{role}:custom:lmstudio:qwen3-32b"]
            self.assertEqual((item["transport"], item["name"]), (CUSTOM, "qwen3-32b (LM Studio)"))
            self.assertIn("127.0.0.1:1234", item["description"])
            self.assertFalse([k for k in self.manager.options[role] if "embed" in k])
        # a context window too small for the answer: offered only where the answer fits
        self.assertIn("deep:custom:lmstudio:tiny-model", self.manager.options["deep"])
        self.assertNotIn("memory_write:custom:lmstudio:tiny-model", self.manager.options["memory_write"])
        self.assertEqual(self.meter.pricing["models"]["custom:lmstudio/qwen3-32b"],
                         {"input": 0.0, "output": 0.0, "cache_read": 0.0})
        self.assertEqual(self.g["AVAILABLE_MODELS"]["deep:custom:lmstudio:qwen3-32b"]["provider"], CUSTOM)
        result = self.manager.set({role: f"{role}:custom:lmstudio:qwen3-32b" for role in ROLES})
        self.assertEqual((result["transport"], self.manager.transport_of()), (CUSTOM, CUSTOM))
        self.assertEqual(result["custom"], {"endpoints": [{"id": "lmstudio", "name": "LM Studio",
                                                           "base_url": "http://127.0.0.1:1234/v1", "json_mode": "json_object",
                                                           "models": ["qwen3-32b", "tiny-model"]}]})
        # a surface turn goes through _pick_stream to the endpoint, with the key read from the environment now
        os.environ[key_name("lmstudio")] = KEY
        http = self.g["http_requests"]
        http.responses = [Response(lines=sse(chunk("<think>hm</think>reply: hi"), chunk(finish="stop"), USAGE))]
        parts = {"system": "s", "user_content": "u", "_f_phase": "surface"}
        text = "".join(self.g["_pick_stream"](parts, self.g["AVAILABLE_MODELS"]["surface:custom:lmstudio:qwen3-32b"]))
        self.assertEqual(text, "reply: hi")
        self.assertEqual(http.calls[-1][0], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(http.calls[-1][1]["headers"]["Authorization"], "Bearer " + KEY)
        self.assertEqual(len(list((self.root / "memory_f" / "custom_receipts").glob("*.json"))), 1)
        with self.meter.read_connection() as conn:
            row = conn.execute("SELECT provider, model, cost_usd, pricing_status FROM requests").fetchone()
        self.assertEqual(tuple(row), ("custom", "lmstudio/qwen3-32b", 0.0, "priced"))
        # the model leaves the server: the next detect drops the option and every role falls back to the preset
        save_custom_config(self.root, {"endpoints": [dict(ENDPOINT, models=ENDPOINT["models"][1:])]})
        result = self.manager.reload_custom()
        self.assertNotIn("deep:custom:lmstudio:qwen3-32b", self.manager.options["deep"])
        self.assertNotIn("deep:custom:lmstudio:qwen3-32b", self.g["AVAILABLE_MODELS"])
        self.assertEqual(result["selections"], {role: "or:" + role for role in ROLES})
        self.assertEqual(json.loads((self.root / "config/llm_roles.json").read_text())["selections"]["deep"], "or:deep")

    def test_actions_need_the_page(self):
        result, status = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"name": "x", "base_url": "http://x"},
                                   origin="https://evil.example")
        self.assertEqual((status, result["error"]), (403, "Use the OpenAmika page for this action."))
        result, status = self.call("/api/build_f/custom/use", "custom-detect", {"endpoint": "lmstudio", "model": "qwen3-32b"})
        self.assertEqual(status, 403)
        self.assertEqual(self.manager.selections["deep"], "or:deep")

    def test_endpoint_key_is_saved_applied_and_never_echoed(self):
        result, status = self.call("/api/build_f/custom/endpoint", "custom-endpoint",
                                   {"name": " Groq ", "base_url": "https://api.groq.com/openai/v1/chat/completions", "key": KEY})
        self.assertEqual(status, 200, result)
        self.assertNotIn(KEY, json.dumps(result))
        groq = next(e for e in result["endpoints"] if e["id"] == "groq")
        self.assertEqual((groq["name"], groq["base_url"], groq["key"]),
                         ("Groq", "https://api.groq.com/openai/v1", {"present": True, "hint": "…1234"}))
        self.assertEqual(os.environ["CUSTOM_LLM_KEY_GROQ"], KEY)
        env = (self.root / "brain" / ".env").read_text(encoding="utf-8")
        self.assertEqual(env, f"OPENROUTER_API_KEY=sk-or-x\nCUSTOM_LLM_KEY_GROQ={KEY}\n")
        self.assertNotIn(KEY, (self.root / "config" / "custom_llm.json").read_text(encoding="utf-8"))
        # the same name again gets its own id; a change without a key leaves the key alone
        result, _ = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"name": "Groq", "base_url": "https://x.example/v1"})
        self.assertEqual([e["id"] for e in result["endpoints"]], ["lmstudio", "groq", "groq-2"])
        result, _ = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"id": "groq", "json_mode": "json_schema"})
        groq = next(e for e in result["endpoints"] if e["id"] == "groq")
        self.assertEqual((groq["json_mode"], groq["key"]["present"]), ("json_schema", True))
        # a new key replaces the line; clear_key removes it
        self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"id": "groq", "key": "gsk_" + "b" * 20})
        self.assertEqual((self.root / "brain" / ".env").read_text(encoding="utf-8").count("CUSTOM_LLM_KEY_GROQ="), 1)
        result, _ = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"id": "groq", "clear_key": True})
        self.assertEqual(next(e for e in result["endpoints"] if e["id"] == "groq")["key"], {"present": False, "hint": None})
        self.assertNotIn("CUSTOM_LLM_KEY_GROQ", os.environ)
        self.assertEqual((self.root / "brain" / ".env").read_text(encoding="utf-8"), "OPENROUTER_API_KEY=sk-or-x\n")
        for body, words in (({"id": "groq", "key": "two words"}, "characters"), ({"id": "nope", "name": "x"}, "no such"),
                            ({"name": "x", "base_url": "box:1234"}, "http://"), ({"name": "", "base_url": "http://x"}, "name"),
                            ({"id": "groq", "json_mode": "xml"}, "json_mode"), ({"id": "groq", "key": KEY, "clear_key": True}, "not both")):
            result, status = self.call("/api/build_f/custom/endpoint", "custom-endpoint", body)
            self.assertEqual(status, 400, body)
            self.assertIn(words, result["error"])

    def test_use_and_removal_while_in_use(self):
        result, status = self.call("/api/build_f/custom/use", "custom-use", {"endpoint": "lmstudio", "model": "qwen3-32b",
                                   "roles": {"surface": {"endpoint": "lmstudio", "model": "tiny-model"}}})
        self.assertEqual(status, 200, result)
        self.assertEqual(result["selections"]["surface"], "surface:custom:lmstudio:tiny-model")
        self.assertEqual(result["selections"]["deep"], "deep:custom:lmstudio:qwen3-32b")
        self.assertEqual(result["in_use"]["memory_write"], {"endpoint": "lmstudio", "model": "qwen3-32b"})
        result, status = self.call("/api/build_f/custom/use", "custom-use", {"endpoint": "lmstudio", "model": "tiny-model"})
        self.assertEqual(status, 400)
        self.assertIn("context window", result["error"])
        result, status = self.call("/api/build_f/custom/use", "custom-use", {"endpoint": "lmstudio", "model": "nope"})
        self.assertEqual(status, 400)
        result, status = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"id": "lmstudio", "remove": True})
        self.assertEqual(status, 400)
        self.assertIn("pick another model for surface, deep, reflection, memory_write first", result["error"])
        result, status = self.call("/api/build_f/custom/model", "custom-model", {"id": "lmstudio", "name": "qwen3-32b", "remove": True})
        self.assertEqual(status, 400)
        self.assertIn("deep, reflection, memory_write", result["error"])
        # a server that lists only what it has loaded: a detect keeps the models her roles use
        self.g["http_requests"].responses = [Response(data={"data": [{"id": "llama-3.3-70b"}]})]
        result, status = self.call("/api/build_f/custom/detect", "custom-detect", {"id": "lmstudio"})
        self.assertEqual(status, 200, result)
        self.assertEqual([m["name"] for m in result["endpoints"][0]["models"]],
                         ["llama-3.3-70b", "qwen3-32b", "tiny-model"])
        self.assertEqual(self.manager.selections["deep"], "deep:custom:lmstudio:qwen3-32b")
        # back on the preset, the endpoint and its key go
        self.manager.set({role: "or:" + role for role in ROLES})
        os.environ[key_name("lmstudio")] = KEY
        (self.root / "brain" / ".env").write_text(f"CUSTOM_LLM_KEY_LMSTUDIO={KEY}\n", encoding="utf-8")
        result, status = self.call("/api/build_f/custom/endpoint", "custom-endpoint", {"id": "lmstudio", "remove": True})
        self.assertEqual((status, result["endpoints"]), (200, []))
        self.assertNotIn(key_name("lmstudio"), os.environ)
        self.assertEqual((self.root / "brain" / ".env").read_text(encoding="utf-8"), "")
        self.assertFalse([k for k in self.manager.options["deep"] if ":custom:" in k])

    def test_detect_keeps_hand_added_models_and_says_why_it_failed(self):
        http = self.g["http_requests"]
        http.responses = [Response(data={"data": [{"id": "qwen3-32b", "max_model_len": 40960}, {"id": "tiny-model"},
                                                  {"id": "llama-3.3-70b"}]})]
        result, status = self.call("/api/build_f/custom/detect", "custom-detect", {"id": "lmstudio"})
        self.assertEqual(status, 200, result)
        rows = {m["name"]: m for m in result["endpoints"][0]["models"]}
        self.assertEqual(sorted(rows), ["llama-3.3-70b", "qwen3-32b", "tiny-model"])
        self.assertEqual((rows["tiny-model"]["source"], rows["qwen3-32b"]["context_length"]), ("detected", 40960))
        self.assertTrue(result["endpoints"][0]["reachable"])
        self.assertIn("deep:custom:lmstudio:llama-3.3-70b", self.manager.options["deep"])
        result, status = self.call("/api/build_f/custom/model", "custom-model", {"id": "lmstudio", "name": "my-finetune"})
        self.assertEqual(status, 200, result)
        self.assertIn("deep:custom:lmstudio:my-finetune", self.manager.options["deep"])
        result, status = self.call("/api/build_f/custom/model", "custom-model", {"id": "lmstudio", "name": "Qwen (Q4)"})
        self.assertEqual(status, 400)
        self.assertIn("usage ledger", result["error"])
        http.responses = [Response(data={"data": [{"id": "qwen3-32b"}]})]
        result, _ = self.call("/api/build_f/custom/detect", "custom-detect", {"id": "lmstudio"})
        self.assertEqual([m["name"] for m in result["endpoints"][0]["models"]], ["my-finetune", "qwen3-32b"])
        http.responses = [ConnectionError("refused")]
        result, status = self.call("/api/build_f/custom/detect", "custom-detect", {"id": "lmstudio"})
        self.assertEqual(status, 503)
        self.assertIn("http://127.0.0.1:1234/v1", result["error"])
        http.responses = [Response(status=401)]
        result, status = self.call("/api/build_f/custom/detect", "custom-detect", {"id": "lmstudio"})
        self.assertEqual(status, 400)
        self.assertIn("refused the key", result["error"])
        # the page's status probe: up or down, never an error page
        http.responses = [ConnectionError("refused")]
        result, status = self.call("/api/build_f/custom")
        self.assertEqual((status, result["endpoints"][0]["reachable"], result["endpoints"][0]["error"]), (200, False, None))
        self.assertEqual(http.calls[-1][1]["timeout"], (2, 3))
        http.responses = [Response(status=401)]
        result, _ = self.call("/api/build_f/custom")
        self.assertFalse(result["endpoints"][0]["reachable"])
        self.assertIn("refused the key", result["endpoints"][0]["error"])


if __name__ == "__main__":
    unittest.main()
