#!/usr/bin/env python3
"""Security tests for the authenticated network boundary in front of M3."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _support import load_proxy_module


PROXY = load_proxy_module()
TOKEN = "owner-token-0123456789abcdef"


def headers(*pairs):
    message = Message()
    for key, value in pairs:
        message[key] = value
    return message


class FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


def make_config(token_file, **overrides):
    values = {
        "upstream": "http://127.0.0.1:8082",
        "token_file": Path(token_file),
        "max_body_bytes": 1024,
        "upstream_timeout_seconds": 10,
        "header_timeout_seconds": 2,
        "body_timeout_seconds": 3,
        "client_write_timeout_seconds": 4,
    }
    values.update(overrides)
    return PROXY.ProxyConfig(**values)


def make_handler(config, request_headers=None):
    handler = object.__new__(PROXY.M3LanProxyHandler)
    handler.server = SimpleNamespace(proxy_config=config)
    handler.headers = request_headers or headers()
    handler.connection = FakeSocket()
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.command = "POST"
    handler.path = "/v1/chat/completions"
    handler.close_connection = False
    handler._response_started = False
    handler.send_response = mock.Mock()
    handler.send_header = mock.Mock()
    handler.end_headers = mock.Mock()
    return handler


class TokenAndAuthenticationTests(unittest.TestCase):
    def test_token_file_must_be_regular_owner_only_ascii(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "token"
            token_file.write_text(TOKEN + "\n", encoding="ascii")
            token_file.chmod(0o600)
            self.assertEqual(TOKEN, PROXY.read_owner_only_token(token_file))
            token_file.chmod(0o640)
            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(token_file)
            token_file.chmod(0o600)
            token_file.write_text("bad token with spaces", encoding="ascii")
            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(token_file)

    def test_post_requires_exact_bearer_token(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "token"
            token_file.write_text(TOKEN, encoding="ascii")
            token_file.chmod(0o600)
            config = make_config(token_file)

            authorized = make_handler(
                config, headers(("Authorization", f"Bearer {TOKEN}"))
            )
            authorized._send_json_error = mock.Mock()
            self.assertTrue(authorized._authorize_post())
            authorized._send_json_error.assert_not_called()

            for value in (None, f"Basic {TOKEN}", f"Bearer {TOKEN}x", "Bearer"):
                with self.subTest(value=value):
                    request_headers = headers()
                    if value is not None:
                        request_headers["Authorization"] = value
                    rejected = make_handler(config, request_headers)
                    rejected._send_json_error = mock.Mock()
                    self.assertFalse(rejected._authorize_post())
                    self.assertEqual(
                        HTTPStatus.UNAUTHORIZED,
                        rejected._send_json_error.call_args.args[0],
                    )

    def test_insecure_or_missing_token_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "missing"
            handler = make_handler(
                make_config(token_file),
                headers(("Authorization", f"Bearer {TOKEN}")),
            )
            handler._send_json_error = mock.Mock()
            self.assertFalse(handler._authorize_post())
            self.assertEqual(
                HTTPStatus.SERVICE_UNAVAILABLE,
                handler._send_json_error.call_args.args[0],
            )

    def test_auth_and_priority_are_stripped_but_fixed_profile_pair_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = make_handler(
                make_config(Path(directory) / "token"),
                headers(
                    ("Authorization", f"Bearer {TOKEN}"),
                    ("X-M3-Priority", "0"),
                    ("X-M3-Priority-Token", "private-priority-token"),
                    ("X-M3-LoRA-Profile", "trader"),
                    ("X-M3-LoRA-Profile-Token", "private-profile-token"),
                    ("Content-Type", "application/json"),
                    ("Accept", "text/event-stream"),
                ),
            )
            forwarded = handler._forwarded_request_headers()
        self.assertEqual(
            {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-M3-LoRA-Profile": "trader",
                "X-M3-LoRA-Profile-Token": "private-profile-token",
            },
            forwarded,
        )

    def test_profile_headers_cannot_be_duplicated_or_connection_nominated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "token")
            cases = (
                headers(
                    ("X-M3-LoRA-Profile", "general"),
                    ("X-M3-LoRA-Profile", "trader"),
                ),
                headers(
                    ("Connection", "X-M3-LoRA-Profile"),
                    ("X-M3-LoRA-Profile", "trader"),
                ),
                headers(
                    ("Connection", "X-M3-LoRA-Profile-Token"),
                    ("X-M3-LoRA-Profile-Token", "private-profile-token"),
                ),
            )
            for request_headers in cases:
                with self.subTest(headers=list(request_headers.items())):
                    request = make_handler(config, request_headers)
                    request._send_json_error = mock.Mock()
                    self.assertIsNone(request._forwarded_request_headers())
                    self.assertEqual(
                        HTTPStatus.BAD_REQUEST,
                        request._send_json_error.call_args.args[0],
                    )


class FramingAndConfigurationTests(unittest.TestCase):
    def test_upstream_is_loopback_only_and_has_no_credentials_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "token"
            for upstream in (
                "http://192.168.1.2:8082",
                "http://user:pass@127.0.0.1:8082",
                "http://127.0.0.1:8082/private",
                "file:///tmp/socket",
            ):
                with self.subTest(upstream=upstream), self.assertRaises(ValueError):
                    make_config(token_file, upstream=upstream)

    def test_content_length_is_required_unique_bounded_and_unchunked(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "token", max_body_bytes=8)
            cases = (
                (headers(), HTTPStatus.LENGTH_REQUIRED),
                (
                    headers(("Content-Length", "1"), ("Content-Length", "1")),
                    HTTPStatus.BAD_REQUEST,
                ),
                (headers(("Content-Length", "-1")), HTTPStatus.BAD_REQUEST),
                (headers(("Content-Length", "9")), HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
                (
                    headers(
                        ("Content-Length", "2"), ("Transfer-Encoding", "chunked")
                    ),
                    HTTPStatus.BAD_REQUEST,
                ),
            )
            for request_headers, expected in cases:
                with self.subTest(headers=request_headers):
                    handler = make_handler(config, request_headers)
                    with self.assertRaises(PROXY.RequestLengthError) as caught:
                        handler._parse_content_length()
                    self.assertEqual(expected, caught.exception.status)
            handler = make_handler(config, headers(("Content-Length", "8")))
            self.assertEqual(8, handler._parse_content_length())

    def test_routes_are_positive_allowlists(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = make_handler(make_config(Path(directory) / "token"))
            handler._send_json_error = mock.Mock()
            for path in ("/", "/health", "/v1/models"):
                handler.path = path
                self.assertEqual(path, handler._target(PROXY.GET_PATHS))
            for path in ("/health?full=1", "/health-secret", "//health", "/metrics"):
                handler.path = path
                self.assertIsNone(handler._target(PROXY.GET_PATHS))
            handler.path = "/v1/chat/completions"
            self.assertEqual(handler.path, handler._target(PROXY.POST_PATHS))

    def test_public_listener_uses_authenticated_proxy_while_upstream_stays_loopback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "token")
            sentinel = object()
            with mock.patch.object(
                PROXY, "M3LanProxyServer", return_value=sentinel
            ) as server_class:
                self.assertIs(sentinel, PROXY.build_server(config))
            server_class.assert_called_once_with(("0.0.0.0", PROXY.DEFAULT_PORT), config)
            self.assertEqual("127.0.0.1", PROXY.urlsplit(config.upstream).hostname)

    def test_main_validates_token_before_opening_network_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "missing")
            with mock.patch.object(
                PROXY.ProxyConfig, "from_environment", return_value=config
            ), mock.patch.object(
                PROXY,
                "read_owner_only_token",
                side_effect=PROXY.TokenFileError("unavailable"),
            ), mock.patch.object(PROXY, "build_server") as build:
                with self.assertRaises(PROXY.TokenFileError):
                    PROXY.main()
            build.assert_not_called()


class HealthAndRedactionTests(unittest.TestCase):
    def test_lan_health_redacts_runtime_paths_receipts_and_nonce(self):
        class FakeResponse:
            def __init__(self, payload):
                self.status = 200
                self.headers = headers(("Content-Length", str(len(payload))))
                self._stream = io.BytesIO(payload)

            def read(self, size=-1):
                return self._stream.read(size)

            def close(self):
                return None

        detailed = json.dumps(
            {
                "ready": True,
                "model_id": "minimax-m3",
                "model_realpath": "/Users/private/model",
                "startup_nonce": "secret-nonce",
                "runtime_receipt": {"private": "detail"},
            }
        ).encode("utf-8")
        response = FakeResponse(detailed)

        class FakeConnection:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, *_args, **_kwargs):
                return None

            def getresponse(self):
                return response

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            handler = make_handler(make_config(Path(directory) / "token"))
            handler.command = "GET"
            with mock.patch.object(PROXY.http.client, "HTTPConnection", FakeConnection):
                handler._open_redacted_health()
        public = json.loads(handler.wfile.getvalue())
        self.assertEqual({"ready": True, "model": "minimax-m3"}, public)
        self.assertNotIn(b"private", handler.wfile.getvalue())
        self.assertNotIn(b"nonce", handler.wfile.getvalue())

    def test_health_is_read_only_and_does_not_require_bearer_token(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = make_handler(make_config(Path(directory) / "missing"), headers())
            handler.path = "/health"
            handler.command = "GET"
            handler._open_redacted_health = mock.Mock()
            handler._authorize_post = mock.Mock()
            handler.do_GET()
            handler._open_redacted_health.assert_called_once_with()
            handler._authorize_post.assert_not_called()

    def test_error_and_access_logging_never_echo_caller_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = make_handler(make_config(Path(directory) / "token"))
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                handler.log_message("Authorization: %s", TOKEN)
            self.assertEqual("", captured.getvalue())
            handler._send_json_error = mock.Mock()
            handler.send_error(500, f"SECRET {TOKEN} /Users/private/path")
            args = handler._send_json_error.call_args.args
            self.assertEqual(500, args[0])
            self.assertEqual("internal server error", args[1])
            self.assertNotIn(TOKEN, repr(args))


if __name__ == "__main__":
    unittest.main()
