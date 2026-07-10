#!/usr/bin/env python3
"""Offline security and compatibility tests for the staged M3 LAN proxy."""

from __future__ import annotations

import base64
import http.client
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m3_lan_proxy_staged", HERE.parent / "m3_lan_proxy.py"
)
PROXY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROXY
assert SPEC.loader is not None
SPEC.loader.exec_module(PROXY)

MIB = 1024 * 1024
TEST_TOKEN = "offline-test-token-not-a-real-secret-0123456789"


def message(*pairs: tuple[str, str]) -> HTTPMessage:
    result = HTTPMessage()
    for name, value in pairs:
        result.add_header(name, value)
    return result


def parse_raw_response(raw: bytes) -> tuple[int, dict[str, list[str]], bytes]:
    head, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        raise AssertionError(f"response has no header terminator: {raw[:200]!r}")
    lines = head.split(b"\r\n")
    status = int(lines[0].split(b" ", 2)[1])
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        name, value = line.split(b":", 1)
        headers.setdefault(name.decode("ascii").lower(), []).append(
            value.strip().decode("latin-1")
        )
    return status, headers, body


def decode_chunked(data: bytes) -> bytes:
    decoded = bytearray()
    cursor = 0
    while True:
        line_end = data.find(b"\r\n", cursor)
        if line_end < 0:
            raise AssertionError("truncated chunk size")
        size = int(data[cursor:line_end].split(b";", 1)[0], 16)
        cursor = line_end + 2
        if size == 0:
            if data[cursor:cursor + 2] != b"\r\n":
                raise AssertionError("invalid final chunk")
            return bytes(decoded)
        end = cursor + size
        if len(data) < end + 2 or data[end:end + 2] != b"\r\n":
            raise AssertionError("truncated chunk data")
        decoded.extend(data[cursor:end])
        cursor = end + 2


class ConfigAndTokenTests(unittest.TestCase):
    def test_default_cap_is_twenty_mib_and_all_timeouts_are_bounded(self):
        self.assertEqual(20 * MIB, PROXY.DEFAULT_MAX_BODY_BYTES)
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token"
            config = PROXY.ProxyConfig("http://127.0.0.1:8082", token)
            self.assertEqual(20 * MIB, config.max_body_bytes)
            for field in (
                "upstream_timeout_seconds",
                "header_timeout_seconds",
                "body_timeout_seconds",
                "client_write_timeout_seconds",
            ):
                self.assertGreater(getattr(config, field), 0)

    def test_upstream_must_be_loopback_without_credentials_or_url_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token"
            invalid = (
                "http://user:pass@127.0.0.1:8082",
                "http://127.0.0.1:8082/path",
                "http://127.0.0.1:8082/?query=yes",
                "http://192.0.2.20:8082",
                "file:///tmp/socket",
                "http://127.0.0.1:99999",
            )
            for upstream in invalid:
                with self.subTest(upstream=upstream), self.assertRaises(ValueError):
                    PROXY.ProxyConfig(upstream, token)

    def test_token_file_must_be_owner_only_regular_ascii(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "token"
            token.write_text(TEST_TOKEN + "\n", encoding="ascii")
            token.chmod(0o600)
            self.assertEqual(TEST_TOKEN, PROXY.read_owner_only_token(token))

            token.chmod(0o640)
            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(token)
            token.chmod(0o600)

            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(root)

            link = root / "link"
            link.symlink_to(token)
            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(link)

            token.write_bytes(b"short\nsecret")
            with self.assertRaises(PROXY.TokenFileError):
                PROXY.read_owner_only_token(token)

    def test_upstream_framing_rejects_ambiguous_or_noncanonical_values(self):
        self.assertEqual(
            ("chunked", None),
            PROXY._upstream_transfer_mode(message(("Transfer-Encoding", "chunked"))),
        )
        self.assertEqual(
            ("length", 12),
            PROXY._upstream_transfer_mode(message(("Content-Length", "12"))),
        )
        invalid = (
            message(("Transfer-Encoding", "chunked"), ("Content-Length", "12")),
            message(("Transfer-Encoding", "gzip, chunked")),
            message(("Transfer-Encoding", "chunked,")),
            message(("Content-Length", "01")),
            message(("Content-Length", "12"), ("Content-Length", "12")),
        )
        for headers in invalid:
            with self.subTest(headers=list(headers.items())), self.assertRaises(
                PROXY.UpstreamFramingError
            ):
                PROXY._upstream_transfer_mode(headers)


class _OfflineUpstream(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _OfflineUpstreamHandler)
        self.records: list[dict[str, object]] = []
        self.records_lock = threading.Lock()
        self.first_chunk_sent = threading.Event()
        self.release_second_chunk = threading.Event()

    def record(self, item: dict[str, object]) -> None:
        with self.records_lock:
            self.records.append(item)


class _OfflineUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args) -> None:
        del args

    def _record(self, body: bytes = b"") -> None:
        names = {name.lower() for name in self.headers.keys()}
        self.server.record(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path,
                "headers": {
                    name: self.headers.get_all(name, []) for name in sorted(names)
                },
                "body_length": len(body),
                "body_prefix": body[:64],
                "body_suffix": body[-64:],
            }
        )

    def _normal_response(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "X-Upstream-Hop, X-Upstream-Second")
        self.send_header("X-Upstream-Hop", "must-not-pass")
        self.send_header("X-Upstream-Second", "must-not-pass")
        self.send_header("Keep-Alive", "timeout=100")
        self.send_header("X-End-To-End", "kept")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._record()
        accept = self.headers.get("Accept", "")
        if accept == "application/x-test-close":
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        if accept == "application/x-test-redirect":
            self.send_response(302)
            self.send_header("Location", "http://192.0.2.88/never-follow")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if accept == "application/x-test-chunked":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "X-Upstream-Hop")
            self.send_header("X-Upstream-Hop", "must-not-pass")
            self.end_headers()
            self.wfile.write(b"5\r\nalpha\r\n")
            self.wfile.flush()
            self.server.first_chunk_sent.set()  # type: ignore[attr-defined]
            self.server.release_second_chunk.wait(3)  # type: ignore[attr-defined]
            self.wfile.write(b"4\r\nbeta\r\n0\r\n\r\n")
            self.wfile.flush()
            return
        if self.path == "/health":
            payload = json.dumps({
                "ready": True, "model_id": "minimax-m3",
                "model_realpath": "/private/models/secret",
                "runtime_receipt": {"device": "secret-device"},
            }).encode("utf-8")
        else:
            payload = json.dumps({"ready": True, "path": self.path}).encode("utf-8")
        self._normal_response(payload)

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self._record(body)
        self._normal_response(json.dumps({"received": len(body)}).encode("ascii"))


class ProxyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.token_file = Path(cls.tempdir.name) / "bearer-token"
        cls.token_file.write_text(TEST_TOKEN + "\n", encoding="ascii")
        cls.token_file.chmod(0o600)

        cls.upstream = _OfflineUpstream()
        cls.upstream_thread = threading.Thread(
            target=cls.upstream.serve_forever, daemon=True
        )
        cls.upstream_thread.start()

        upstream_host, upstream_port = cls.upstream.server_address
        config = PROXY.ProxyConfig(
            upstream=f"http://{upstream_host}:{upstream_port}",
            token_file=cls.token_file,
            upstream_timeout_seconds=2.0,
            header_timeout_seconds=0.20,
            body_timeout_seconds=0.50,
            client_write_timeout_seconds=1.0,
        )
        cls.proxy = PROXY.build_server(config, bind="127.0.0.1", port=0)
        cls.proxy_thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.proxy_thread.start()
        cls.proxy_host, cls.proxy_port = cls.proxy.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.upstream.release_second_chunk.set()
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.proxy_thread.join(2)
        cls.upstream_thread.join(2)
        cls.tempdir.cleanup()

    def setUp(self) -> None:
        self.token_file.chmod(0o600)
        self.token_file.write_text(TEST_TOKEN + "\n", encoding="ascii")
        self.token_file.chmod(0o600)
        with self.upstream.records_lock:
            self.upstream.records.clear()
        self.upstream.first_chunk_sent.clear()
        self.upstream.release_second_chunk.clear()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        connection = http.client.HTTPConnection(self.proxy_host, self.proxy_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def _raw_exchange(self, request: bytes, *, shutdown_write: bool = True) -> bytes:
        client = socket.create_connection((self.proxy_host, self.proxy_port), timeout=3)
        client.settimeout(3)
        try:
            client.sendall(request)
            if shutdown_write:
                client.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(64 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            client.close()

    def _raw_post(self, headers: bytes, body: bytes = b"") -> bytes:
        return self._raw_exchange(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            + f"Host: {self.proxy_host}:{self.proxy_port}\r\n".encode("ascii")
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode("ascii")
            + headers
            + b"\r\n"
            + body
        )

    def _last_record(self) -> dict[str, object]:
        with self.upstream.records_lock:
            self.assertTrue(self.upstream.records)
            return self.upstream.records[-1]

    def test_preserves_readiness_discovery_and_authenticated_chat_routes(self):
        for path in ("/", "/health", "/v1/models"):
            with self.subTest(path=path):
                status, _headers, body = self._request("GET", path)
                self.assertEqual(200, status)
                value = json.loads(body)
                if path == "/health":
                    self.assertEqual({"ready": True, "model": "minimax-m3"}, value)
                    self.assertNotIn(b"private", body)
                    self.assertNotIn(b"secret-device", body)
                else:
                    self.assertEqual(path, value["path"])

        body = b'{"messages":[{"role":"user","content":"hello"}]}'
        status, _headers, response_body = self._request(
            "POST",
            "/v1/chat/completions",
            body,
            {
                "Authorization": f"Bearer {TEST_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual(len(body), json.loads(response_body)["received"])
        self.assertEqual("/v1/chat/completions", self._last_record()["path"])

        status, _headers, _body = self._request(
            "POST", "/", b"{}", {"Authorization": f"Bearer {TEST_TOKEN}"}
        )
        self.assertEqual(404, status, "legacy POST / must remain disabled")

    def test_authentication_failures_are_generic_and_never_forwarded(self):
        for authorization in (None, "Bearer wrong-token-value-000000", "Basic abc"):
            headers = {} if authorization is None else {"Authorization": authorization}
            with self.subTest(authorization=authorization):
                status, response_headers, body = self._request(
                    "POST", "/v1/chat/completions", b"{}", headers
                )
                self.assertEqual(401, status)
                self.assertIn(
                    "Bearer", dict((name.lower(), value) for name, value in response_headers)["www-authenticate"]
                )
                self.assertNotIn(TEST_TOKEN.encode(), body)

        duplicate = self._raw_exchange(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            + f"Host: {self.proxy_host}:{self.proxy_port}\r\n".encode()
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Content-Length: 2\r\n\r\n{}"
        )
        self.assertEqual(401, parse_raw_response(duplicate)[0])
        with self.upstream.records_lock:
            self.assertEqual([], self.upstream.records)

        self.token_file.chmod(0o640)
        status, _headers, body = self._request(
            "POST",
            "/v1/chat/completions",
            b"{}",
            {"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        self.assertEqual(503, status)
        self.assertNotIn(str(self.token_file).encode(), body)
        self.assertNotIn(TEST_TOKEN.encode(), body)

    def test_request_framing_matrix_is_fail_closed(self):
        cases = (
            (b"", b"", 411),
            (b"Content-Length: 2\r\nContent-Length: 2\r\n", b"{}", 400),
            (
                b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
                b"{}",
                400,
            ),
            (b"Transfer-Encoding: chunked\r\n", b"2\r\n{}\r\n0\r\n\r\n", 400),
            (b"Content-Length: 02\r\n", b"{}", 400),
            (f"Content-Length: {20 * MIB + 1}\r\n".encode(), b"", 413),
            (b"Content-Length: 3\r\n", b"{}", 400),
        )
        for headers, body, expected in cases:
            with self.subTest(headers=headers):
                response = self._raw_post(headers, body)
                self.assertEqual(expected, parse_raw_response(response)[0])

        expect = self._raw_post(b"Content-Length: 2\r\nExpect: 100-continue\r\n")
        self.assertEqual(417, parse_raw_response(expect)[0])
        with self.upstream.records_lock:
            self.assertEqual([], self.upstream.records)

    def test_body_read_has_an_absolute_timeout(self):
        client = socket.create_connection((self.proxy_host, self.proxy_port), timeout=2)
        client.settimeout(2)
        try:
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                + f"Host: {self.proxy_host}:{self.proxy_port}\r\n".encode()
                + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
                + b"Content-Length: 10\r\n\r\n{"
            )
            time.sleep(0.65)
            raw = bytearray()
            while True:
                try:
                    chunk = client.recv(8192)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                raw.extend(chunk)
        finally:
            client.close()
        self.assertEqual(408, parse_raw_response(bytes(raw))[0])

    def test_slowloris_trickle_cannot_extend_the_header_deadline(self):
        client = socket.create_connection((self.proxy_host, self.proxy_port), timeout=2)
        client.settimeout(2)
        started = time.monotonic()
        try:
            client.sendall(
                b"GET /v1/models HTTP/1.1\r\n"
                + f"Host: {self.proxy_host}:{self.proxy_port}\r\n".encode()
                + b"X-Slow:"
            )
            for _ in range(8):
                time.sleep(0.05)
                try:
                    client.sendall(b"x")
                except OSError:
                    break
            raw = bytearray()
            while True:
                try:
                    chunk = client.recv(8192)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                raw.extend(chunk)
        finally:
            client.close()
        elapsed = time.monotonic() - started
        self.assertEqual(408, parse_raw_response(bytes(raw))[0])
        self.assertLess(elapsed, 1.0)

    def test_strict_targets_methods_hosts_and_header_lines(self):
        absolute = self._raw_exchange(
            b"GET http://user:pass@127.0.0.1/health HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n\r\n"
        )
        self.assertEqual(400, parse_raw_response(absolute)[0])

        for target in ("/health?debug=1", "/v1/models/"):
            status, _headers, _body = self._request("GET", target)
            self.assertEqual(404, status)

        method = self._raw_exchange(
            b"PATCH /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        )
        status, headers, _body = parse_raw_response(method)
        self.assertEqual(405, status)
        self.assertEqual(["GET, POST"], headers["allow"])

        malformed = (
            b"GET /health HTTP/1.1\r\nHost: one\r\nHost: two\r\n\r\n",
            b"GET /health HTTP/1.1\r\nHost: user@host\r\n\r\n",
            b"GET /health HTTP/1.1\r\nHost: local\r\n folded: bad\r\n\r\n",
            b"GET /health HTTP/1.1\nHost: local\n\n",
            b"GET /health HTTP/1.1\r\nHost : local\r\n\r\n",
        )
        for request in malformed:
            with self.subTest(request=request):
                self.assertEqual(400, parse_raw_response(self._raw_exchange(request))[0])

    def test_request_and_response_hop_headers_are_removed_dynamically(self):
        status, response_headers, _body = self._request(
            "POST",
            "/v1/chat/completions",
            b"{}",
            {
                "Authorization": f"Bearer {TEST_TOKEN}",
                "Connection": "X-Caller-Hop, Content-Type",
                "X-Caller-Hop": "remove-me",
                "Keep-Alive": "timeout=999",
                "Proxy-Authorization": "Basic remove-me",
                "Priority": "u=0",
                "X-M3-Priority": "0",
                "Content-Type": "application/x-must-be-nominated-away",
                "Accept": "application/json",
            },
        )
        self.assertEqual(200, status)
        record_headers = self._last_record()["headers"]
        assert isinstance(record_headers, dict)
        for removed in (
            "authorization",
            "x-caller-hop",
            "keep-alive",
            "proxy-authorization",
            "priority",
            "x-m3-priority",
            "content-type",
        ):
            self.assertNotIn(removed, record_headers)
        self.assertEqual(["application/json"], record_headers["accept"])
        self.assertNotIn("x-caller-hop", ",".join(record_headers.get("connection", [])).lower())

        grouped: dict[str, list[str]] = {}
        for name, value in response_headers:
            grouped.setdefault(name.lower(), []).append(value)
        for removed in ("x-upstream-hop", "x-upstream-second", "keep-alive"):
            self.assertNotIn(removed, grouped)
        self.assertEqual(["kept"], grouped["x-end-to-end"])
        for exactly_once in ("server", "date", "content-length", "connection"):
            self.assertEqual(1, len(grouped[exactly_once]), exactly_once)
        self.assertEqual(["close"], grouped["connection"])

    def test_twelve_mib_binary_base64_fixture_fits_under_twenty_mib_cap(self):
        encoded = base64.b64encode(b"\xa5" * (12 * MIB))
        body = b'{"image_base64":"' + encoded + b'"}'
        self.assertGreater(len(body), 16 * MIB)
        self.assertLess(len(body), 20 * MIB)
        status, _headers, response_body = self._request(
            "POST",
            "/v1/chat/completions",
            body,
            {
                "Authorization": f"Bearer {TEST_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual(len(body), json.loads(response_body)["received"])
        self.assertEqual(len(body), self._last_record()["body_length"])

    def test_chunked_upstream_is_dechunked_streamed_and_rechunked_once(self):
        client = socket.create_connection((self.proxy_host, self.proxy_port), timeout=2)
        client.settimeout(2)
        try:
            client.sendall(
                b"GET /v1/models HTTP/1.1\r\n"
                + f"Host: {self.proxy_host}:{self.proxy_port}\r\n".encode()
                + b"Accept: application/x-test-chunked\r\n\r\n"
            )
            raw = bytearray()
            while b"\r\n\r\n" not in raw:
                raw.extend(client.recv(8192))
            self.assertTrue(self.upstream.first_chunk_sent.wait(1))
            deadline = time.monotonic() + 1
            while b"5\r\nalpha\r\n" not in raw and time.monotonic() < deadline:
                raw.extend(client.recv(8192))
            self.assertIn(b"5\r\nalpha\r\n", raw)
            self.upstream.release_second_chunk.set()
            while True:
                chunk = client.recv(8192)
                if not chunk:
                    break
                raw.extend(chunk)
        finally:
            self.upstream.release_second_chunk.set()
            client.close()

        status, headers, wire_body = parse_raw_response(bytes(raw))
        self.assertEqual(200, status)
        self.assertEqual(["chunked"], headers["transfer-encoding"])
        self.assertNotIn("content-length", headers)
        self.assertNotIn("x-upstream-hop", headers)
        self.assertEqual(b"alphabeta", decode_chunked(wire_body))
        self.assertNotIn(b"\r\n5\r\nalpha", decode_chunked(wire_body))

    def test_upstream_failures_are_generic_and_redirects_are_not_followed(self):
        status, _headers, body = self._request(
            "GET", "/health", headers={"Accept": "application/x-test-close"}
        )
        self.assertEqual(502, status)
        self.assertEqual("upstream unavailable", json.loads(body)["error"]["message"])
        self.assertNotIn(str(self.upstream.server_address).encode(), body)

        status, headers, _body = self._request(
            "GET", "/v1/models", headers={"Accept": "application/x-test-redirect"}
        )
        self.assertEqual(302, status)
        self.assertIn(
            ("Location", "http://192.0.2.88/never-follow"),
            headers,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
