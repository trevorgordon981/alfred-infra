#!/usr/bin/env python3
import importlib.util
import io
import os
import stat
import tempfile
import threading
import unittest
from types import SimpleNamespace
from email.message import Message
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("m3_serve_batched_staged", HERE / "m3_serve_batched.py")
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def headers(*pairs):
    msg = Message()
    for key, value in pairs:
        msg[key] = value
    return msg


class ContentLengthTests(unittest.TestCase):
    def assert_body_error(self, status, hdrs, body=b"", limit=1024):
        with self.assertRaises(SERVER.RequestBodyError) as caught:
            SERVER._read_json_request(hdrs, io.BytesIO(body), max_bytes=limit)
        self.assertEqual(status, caught.exception.status)
        return caught.exception

    def test_valid_object(self):
        body = b'{"messages": []}'
        got = SERVER._read_json_request(
            headers(("Content-Length", str(len(body)))), io.BytesIO(body), max_bytes=1024)
        self.assertEqual({"messages": []}, got)

    def test_missing_content_length_is_411(self):
        self.assert_body_error(411, headers())

    def test_non_numeric_negative_and_duplicate_lengths_are_400(self):
        for hdrs in (
            headers(("Content-Length", "12x")),
            headers(("Content-Length", "-1")),
            headers(("Content-Length", "2"), ("Content-Length", "2")),
        ):
            with self.subTest(values=hdrs.get_all("Content-Length")):
                self.assert_body_error(400, hdrs)

    def test_transfer_encoding_is_rejected(self):
        self.assert_body_error(
            400, headers(("Content-Length", "2"), ("Transfer-Encoding", "chunked")), b"{}")

    def test_oversize_is_413_before_body_read(self):
        stream = mock.Mock()
        with self.assertRaises(SERVER.RequestBodyError) as caught:
            SERVER._read_json_request(headers(("Content-Length", "1025")), stream, max_bytes=1024)
        err = caught.exception
        self.assertEqual(413, err.status)
        self.assertEqual("payload_too_large", err.error_type)
        stream.read.assert_not_called()

    def test_leading_zero_length_is_canonicalized(self):
        self.assertEqual(2, SERVER._parse_content_length(headers(("Content-Length", "0002")), 1024))

    def test_malformed_empty_array_and_truncated_bodies_are_400(self):
        cases = [
            (b"{bad", 4),
            (b"", 0),
            (b"[]", 2),
            (b"{}", 3),
        ]
        for body, claimed in cases:
            with self.subTest(body=body, claimed=claimed):
                self.assert_body_error(
                    400, headers(("Content-Length", str(claimed))), body=body, limit=1024)


class PriorityTests(unittest.TestCase):
    TOKEN = "priority-owner-token-0123456789abcdef"

    def test_default_and_unparseable_priority_are_one(self):
        self.assertEqual(1, SERVER._request_priority(headers()))
        self.assertEqual(1, SERVER._request_priority(headers(("X-M3-Priority", "urgent"))))

    def test_urgent_without_token_file_is_downgraded_but_bench_yield_is_honored(self):
        with mock.patch.object(SERVER, "PRIORITY_TOKEN_FILE", ""):
            self.assertEqual(1, SERVER._request_priority(headers(("X-M3-Priority", "0"))))
            self.assertEqual(2, SERVER._request_priority(headers(("X-M3-Priority", "2"))))

    def test_secure_token_file_authenticates_nondefault_priority(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priority.token"
            path.write_text(self.TOKEN + "\n", encoding="utf-8")
            path.chmod(0o600)
            with mock.patch.object(SERVER, "PRIORITY_TOKEN_FILE", str(path)):
                hdrs = headers(
                    ("X-M3-Priority", "0"),
                    ("X-M3-Priority-Token", self.TOKEN),
                )
                self.assertEqual(0, SERVER._request_priority(hdrs))
                wrong = headers(
                    ("X-M3-Priority", "0"),
                    ("X-M3-Priority-Token", self.TOKEN + "x"),
                )
                self.assertEqual(1, SERVER._request_priority(wrong))

    def test_group_readable_token_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priority.token"
            path.write_text(self.TOKEN, encoding="utf-8")
            path.chmod(0o640)
            with mock.patch.object(SERVER, "PRIORITY_TOKEN_FILE", str(path)):
                hdrs = headers(
                    ("X-M3-Priority", "0"),
                    ("X-M3-Priority-Token", self.TOKEN),
                )
                self.assertEqual(1, SERVER._request_priority(hdrs))


class SaveToTests(unittest.TestCase):
    def test_save_to_is_disabled_by_default(self):
        with mock.patch.object(SERVER, "ALLOW_SAVE_TO", False):
            self.assertEqual((False, "save_to disabled"), SERVER._save_text_create_only("x.txt", "x"))

    def test_opt_in_creates_owner_only_file_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            save_dir = Path(td) / "saves"
            with mock.patch.object(SERVER, "ALLOW_SAVE_TO", True), \
                    mock.patch.object(SERVER, "SAVE_DIR", str(save_dir)):
                self.assertEqual((True, None), SERVER._save_text_create_only("answer.txt", "first"))
                target = save_dir / "answer.txt"
                self.assertEqual("first", target.read_text(encoding="utf-8"))
                self.assertEqual(0, stat.S_IMODE(target.stat().st_mode) & 0o077)
                ok, error = SERVER._save_text_create_only("answer.txt", "second")
                self.assertFalse(ok)
                self.assertIn("already exists", error)
                self.assertEqual("first", target.read_text(encoding="utf-8"))

    def test_path_escape_and_non_owner_only_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            save_dir = Path(td) / "saves"
            save_dir.mkdir(mode=0o700)
            with mock.patch.object(SERVER, "ALLOW_SAVE_TO", True), \
                    mock.patch.object(SERVER, "SAVE_DIR", str(save_dir)):
                ok, _ = SERVER._save_text_create_only("../escape.txt", "no")
                self.assertFalse(ok)
                self.assertFalse((Path(td) / "escape.txt").exists())
                save_dir.chmod(0o755)
                ok, error = SERVER._save_text_create_only("answer.txt", "no")
                self.assertFalse(ok)
                self.assertIn("owner-only", error)


class OpenAIValidationTests(unittest.TestCase):
    def test_valid_fields_and_defaults(self):
        self.assertEqual(([], 4096, 0.0), SERVER._validate_openai_fields({"messages": []}))
        req = {"messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 0, "temperature": 1.25}
        self.assertEqual((req["messages"], 0, 1.25), SERVER._validate_openai_fields(req))

    def test_messages_must_be_list_of_objects(self):
        for messages in (None, "hello", ["hello"], [None], {}):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                SERVER._validate_openai_fields({"messages": messages})

    def test_max_tokens_must_be_finite_integer_in_configured_range(self):
        bad = (True, "10", 1.5, -1, float("inf"), float("nan"),
               SERVER.M3_MAX_TOKENS + 1, 10 ** 10000)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SERVER._validate_openai_fields({"messages": [], "max_tokens": value})

    def test_temperature_must_be_finite_numeric_in_configured_range(self):
        bad = (True, "0.5", -0.1, float("inf"), float("nan"),
               SERVER.M3_MAX_TEMPERATURE + 0.1, 10 ** 10000)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SERVER._validate_openai_fields({"messages": [], "temperature": value})


class RouteTests(unittest.TestCase):
    def _handler(self, path):
        handler = object.__new__(SERVER.H)
        handler.path = path
        handler.headers = headers(("Content-Length", "2"))
        handler.rfile = io.BytesIO(b"{}")
        handler.close_connection = False
        handler._send = mock.Mock()
        return handler

    def test_unknown_routes_are_404_even_while_loading(self):
        handler = self._handler("/v1/models")
        with mock.patch.object(SERVER, "READY", False):
            handler.do_POST()
        self.assertEqual(404, handler._send.call_args.args[0])

    def test_only_exact_generation_routes_pass_route_gate(self):
        for path in ("/v1/chat/completions/", "/v1/other", "/legacy"):
            with self.subTest(path=path):
                handler = self._handler(path)
                handler.do_POST()
                self.assertEqual(404, handler._send.call_args.args[0])

        for path in ("/", "/v1/chat/completions", "/v1/chat/completions?trace=1"):
            with self.subTest(path=path):
                handler = self._handler(path)
                with mock.patch.object(SERVER, "READY", False):
                    handler.do_POST()
                self.assertEqual(503, handler._send.call_args.args[0])

    def test_malformed_openai_fields_return_400_before_generation(self):
        bad_requests = (
            {"messages": "not-a-list"},
            {"messages": ["not-an-object"]},
            {"messages": [], "max_tokens": "900"},
            {"messages": [], "temperature": float("nan")},
        )
        for req in bad_requests:
            with self.subTest(req=req):
                body = __import__("json").dumps(req).encode()
                handler = object.__new__(SERVER.H)
                handler.path = "/v1/chat/completions"
                handler.headers = headers(("Content-Length", str(len(body))))
                handler.rfile = io.BytesIO(body)
                handler.close_connection = False
                handler._send = mock.Mock()
                with mock.patch.object(SERVER, "READY", True), \
                        mock.patch.object(SERVER, "submit_and_wait") as generate_call:
                    handler.do_POST()
                self.assertEqual(400, handler._send.call_args.args[0])
                generate_call.assert_not_called()


class RedactionTests(unittest.TestCase):
    def test_openai_internal_exception_is_not_returned(self):
        handler = object.__new__(SERVER.H)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = io.BytesIO()
        secret = "SECRET /Users/service/private-model-path"
        req = {"messages": [], "max_tokens": 1, "temperature": 0.0}
        with mock.patch.object(SERVER, "submit_and_wait", side_effect=RuntimeError(secret)), \
                mock.patch.object(SERVER, "_log_exc") as logged:
            handler._openai(req, 1)
        response = handler.wfile.getvalue().decode()
        self.assertNotIn(secret, response)
        self.assertIn("internal server error", response)
        logged.assert_called_once_with("_openai")


class StructuredOutputTests(unittest.TestCase):
    VALID = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}

    def test_absent_response_format_preserves_unconstrained_behavior(self):
        with mock.patch.object(SERVER, "_STRUCTURED_OK", False):
            self.assertIsNone(SERVER._build_json_lp(None, "disabled", explicit=False))

    def test_explicit_malformed_response_formats_are_400_class_errors(self):
        malformed = (
            None,
            "json_object",
            {},
            {"type": "text"},
            {"type": "json_schema"},
            {"type": "json_schema", "json_schema": []},
            {"type": "json_schema", "json_schema": {"schema": {}}},
        )
        for response_format in malformed:
            with self.subTest(response_format=response_format), self.assertRaises(ValueError):
                SERVER._build_json_lp(response_format, "disabled", explicit=True)

    def test_explicit_valid_format_fails_closed_when_runtime_unavailable(self):
        with mock.patch.object(SERVER, "_STRUCTURED_OK", False), \
                mock.patch.object(SERVER, "_log_error") as logged:
            with self.assertRaises(SERVER.StructuredDecodingUnavailable):
                SERVER._build_json_lp(self.VALID, "disabled", explicit=True)
        logged.assert_called_once()

    def test_processor_build_failure_is_logged_and_fails_closed(self):
        with mock.patch.object(SERVER, "_STRUCTURED_OK", True), \
                mock.patch.object(SERVER, "PROC", SimpleNamespace(tokenizer=object())), \
                mock.patch.object(SERVER, "build_json_schema_logits_processor",
                                  side_effect=RuntimeError("private runtime detail")), \
                mock.patch.object(SERVER, "_log_exc") as logged:
            with self.assertRaises(SERVER.StructuredDecodingUnavailable):
                SERVER._build_json_lp(self.VALID, "disabled", explicit=True)
        logged.assert_called_once_with("_build_json_lp")

    def test_invalid_schema_reported_by_builder_is_400_class_error(self):
        with mock.patch.object(SERVER, "_STRUCTURED_OK", True), \
                mock.patch.object(SERVER, "PROC", SimpleNamespace(tokenizer=object())), \
                mock.patch.object(SERVER, "build_json_schema_logits_processor",
                                  side_effect=ValueError("schema detail")), \
                mock.patch.object(SERVER, "_log_exc") as logged:
            with self.assertRaises(ValueError):
                SERVER._build_json_lp(self.VALID, "disabled", explicit=True)
        logged.assert_called_once_with("_build_json_lp invalid schema")

    def test_valid_processor_is_returned(self):
        base = object()
        wrapped = object()
        with mock.patch.object(SERVER, "_STRUCTURED_OK", True), \
                mock.patch.object(SERVER, "PROC", SimpleNamespace(tokenizer=object())), \
                mock.patch.object(SERVER, "build_json_schema_logits_processor", return_value=base), \
                mock.patch.object(SERVER, "ThinkingAwareLogitsProcessor", return_value=wrapped):
            self.assertEqual([wrapped], SERVER._build_json_lp(self.VALID, "enabled", explicit=True))

    def test_handler_returns_503_and_never_enqueues_when_structured_unavailable(self):
        req = {"messages": [], "response_format": self.VALID}
        body = __import__("json").dumps(req).encode()
        handler = object.__new__(SERVER.H)
        handler.path = "/v1/chat/completions"
        handler.headers = headers(("Content-Length", str(len(body))))
        handler.rfile = io.BytesIO(body)
        handler.close_connection = False
        handler._send = mock.Mock()
        with mock.patch.object(SERVER, "READY", True), \
                mock.patch.object(SERVER, "_STRUCTURED_OK", False), \
                mock.patch.object(SERVER, "_log_error"), \
                mock.patch.object(SERVER, "submit_and_wait") as generate_call:
            handler.do_POST()
        self.assertEqual(503, handler._send.call_args.args[0])
        self.assertIn(b"structured decoding unavailable", handler._send.call_args.args[1])
        generate_call.assert_not_called()


class StreamingRaceTests(unittest.TestCase):
    def test_empty_stream_fallback_text_precedes_done_exactly_once(self):
        stream_q = SERVER._queue.Queue()
        fallback = SimpleNamespace(text="fallback answer", prompt_tokens=3,
                                   generation_tokens=2, finish_reason="stop")
        with mock.patch.object(SERVER, "stream_generate", return_value=iter(())), \
                mock.patch.object(SERVER, "generate", return_value=fallback) as blocking_generate:
            result = SERVER._gen("prompt", 8, 0.0, False, stream_q=stream_q)
        self.assertIs(fallback, result)
        self.assertEqual([("tok", "fallback answer"), ("done", None)],
                         [stream_q.get_nowait(), stream_q.get_nowait()])
        self.assertTrue(stream_q.empty())
        blocking_generate.assert_called_once()

    def test_structured_empty_stream_fails_closed_without_unconstrained_fallback(self):
        stream_q = SERVER._queue.Queue()
        with mock.patch.object(SERVER, "stream_generate", return_value=iter(())), \
                mock.patch.object(SERVER, "generate") as blocking_generate:
            with self.assertRaises(RuntimeError):
                SERVER._gen("prompt", 8, 0.0, False,
                            logits_processors=[object()], stream_q=stream_q)
        self.assertEqual(("done", None), stream_q.get_nowait())
        self.assertTrue(stream_q.empty())
        blocking_generate.assert_not_called()

    def test_stream_worker_error_emits_generic_error_not_success(self):
        sq_job = SimpleNamespace(ev=threading.Event(), result=None,
                                 error=RuntimeError("SECRET internal path"), cancelled=False)
        sq_job.ev.set()

        def enqueue(*args, **kwargs):
            kwargs["stream_q"].put(("done", None))
            return sq_job

        handler = object.__new__(SERVER.H)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = io.BytesIO()
        with mock.patch.object(SERVER, "_enqueue", side_effect=enqueue):
            handler._stream_tokens({"model": "m"}, [], 8, 0.0, "disabled", [], [], None, 1)
        response = handler.wfile.getvalue().decode()
        self.assertIn('"error"', response)
        self.assertIn("internal server error", response)
        self.assertNotIn("SECRET", response)
        self.assertNotIn("finish_reason", response)
        self.assertTrue(response.endswith("data: [DONE]\n\n"))

    def test_pre_stream_worker_error_without_sentinel_is_detected(self):
        job = SimpleNamespace(ev=threading.Event(), result=None,
                              error=RuntimeError("SECRET before stream"), cancelled=False)
        job.ev.set()
        handler = object.__new__(SERVER.H)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = io.BytesIO()
        with mock.patch.object(SERVER, "_enqueue", return_value=job):
            handler._stream_tokens({"model": "m"}, [], 8, 0.0, "disabled", [], [], None, 1)
        response = handler.wfile.getvalue().decode()
        self.assertIn("internal server error", response)
        self.assertNotIn("SECRET", response)
        self.assertNotIn("finish_reason", response)


class ImportSafetyTests(unittest.TestCase):
    def test_import_did_not_start_server(self):
        self.assertTrue(callable(SERVER.main))
        self.assertFalse(hasattr(SERVER, "httpd"))


if __name__ == "__main__":
    unittest.main()
