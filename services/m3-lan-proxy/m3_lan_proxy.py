#!/usr/bin/env python3
"""Authenticated LAN proxy for the loopback-only M3 serving engine.

The live proxy's useful contract is intentionally small:

* listen on ``0.0.0.0:8096``;
* expose redacted health and forward model discovery GETs to ``127.0.0.1:8082``;
* forward OpenAI-compatible chat-completion POSTs; and
* stream the upstream response without buffering it in memory.

POST authentication terminates here.  The proxy credential is never forwarded
to M3, and arbitrary caller headers (including queue/priority controls) are not
trusted or relayed.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import re
import socket
import stat
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit


DEFAULT_UPSTREAM = "http://127.0.0.1:8082"
DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 8096
DEFAULT_TOKEN_FILE = "~/.config/m3-lan-proxy/bearer-token"
DEFAULT_MAX_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 1800.0
DEFAULT_HEADER_TIMEOUT_SECONDS = 10.0
DEFAULT_BODY_TIMEOUT_SECONDS = 120.0
DEFAULT_CLIENT_WRITE_TIMEOUT_SECONDS = 30.0
MAX_TOKEN_BYTES = 4096
STREAM_CHUNK_BYTES = 64 * 1024
MAX_REQUEST_LINE_BYTES = 8192
MAX_HEADER_LINE_BYTES = 8192
MAX_HEADER_BYTES = 64 * 1024
MAX_HEADER_COUNT = 100
MAX_PUBLIC_HEALTH_BYTES = 64 * 1024

GET_PATHS = frozenset({"/", "/health", "/v1/models"})
POST_PATHS = frozenset({"/v1/chat/completions"})

# urllib/http.client dechunks a chunked upstream response.  The proxy therefore
# generates fresh downstream chunk framing instead of copying the original
# Transfer-Encoding header onto an already-dechunked byte stream.
STATIC_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "keep-alive",
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# These are documented explicitly because priority injection was the dangerous
# behavior in the original open LAN proxy.  Request forwarding uses a positive
# allowlist, so these and every unlisted variant are dropped.
CALLER_PRIORITY_HEADERS = frozenset(
    {
        "priority",
        "x-priority",
        "x-request-priority",
        "x-queue-priority",
        "x-m3-priority",
    }
)

FORWARDED_REQUEST_HEADERS = ("Content-Type", "Accept")
TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/=-]{16,4096}\Z")
CONTENT_LENGTH_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
HTTP_TOKEN_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
HEADER_NAME_RE = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")


class TokenFileError(RuntimeError):
    """The proxy token file is absent, insecure, or malformed."""


class RequestLengthError(ValueError):
    """A request body has invalid or unsupported framing."""

    def __init__(self, status: int, public_message: str) -> None:
        super().__init__(public_message)
        self.status = status
        self.public_message = public_message


class UpstreamFramingError(RuntimeError):
    """The trusted upstream returned framing the proxy cannot relay safely."""


class HeaderSyntaxError(ValueError):
    """A request or upstream Connection header is syntactically invalid."""


@dataclass(frozen=True)
class ProxyConfig:
    upstream: str
    token_file: Path
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    upstream_timeout_seconds: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    header_timeout_seconds: float = DEFAULT_HEADER_TIMEOUT_SECONDS
    body_timeout_seconds: float = DEFAULT_BODY_TIMEOUT_SECONDS
    client_write_timeout_seconds: float = DEFAULT_CLIENT_WRITE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        parsed = urlsplit(self.upstream)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("upstream must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("upstream URL credentials are forbidden")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("upstream must be loopback-only")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("upstream port is invalid") from exc
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("upstream must not contain a path, query, or fragment")
        if not self.token_file.is_absolute():
            raise ValueError("token_file must be absolute")
        if not 1 <= self.max_body_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_body_bytes must be between 1 byte and 64 MiB")
        _validate_timeout(
            "upstream timeout", self.upstream_timeout_seconds, maximum=3600
        )
        _validate_timeout("header timeout", self.header_timeout_seconds, maximum=300)
        _validate_timeout("body timeout", self.body_timeout_seconds, maximum=3600)
        _validate_timeout(
            "client write timeout", self.client_write_timeout_seconds, maximum=300
        )

    @classmethod
    def from_environment(cls) -> "ProxyConfig":
        upstream = os.environ.get("M3_LAN_PROXY_UPSTREAM", DEFAULT_UPSTREAM).rstrip("/")
        token_file = Path(
            os.environ.get("M3_LAN_PROXY_TOKEN_FILE", DEFAULT_TOKEN_FILE)
        ).expanduser()
        max_body = _strict_environment_int(
            "M3_LAN_PROXY_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES
        )
        timeout = _strict_environment_float(
            "M3_LAN_PROXY_UPSTREAM_TIMEOUT_SECONDS",
            DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
        )
        header_timeout = _strict_environment_float(
            "M3_LAN_PROXY_HEADER_TIMEOUT_SECONDS",
            DEFAULT_HEADER_TIMEOUT_SECONDS,
        )
        body_timeout = _strict_environment_float(
            "M3_LAN_PROXY_BODY_TIMEOUT_SECONDS",
            DEFAULT_BODY_TIMEOUT_SECONDS,
        )
        client_write_timeout = _strict_environment_float(
            "M3_LAN_PROXY_CLIENT_WRITE_TIMEOUT_SECONDS",
            DEFAULT_CLIENT_WRITE_TIMEOUT_SECONDS,
        )
        return cls(
            upstream=upstream,
            token_file=token_file,
            max_body_bytes=max_body,
            upstream_timeout_seconds=timeout,
            header_timeout_seconds=header_timeout,
            body_timeout_seconds=body_timeout,
            client_write_timeout_seconds=client_write_timeout,
        )


def _strict_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not CONTENT_LENGTH_RE.fullmatch(raw):
        raise ValueError(f"{name} must be an unsigned decimal integer")
    return int(raw)


def _strict_environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not value or value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_timeout(name: str, value: float, *, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= maximum
        or value != value
        or value in {float("inf"), float("-inf")}
    ):
        raise ValueError(f"{name} must be finite and between 0 and {maximum} seconds")


def read_owner_only_token(path: Path) -> str:
    """Read one bearer token without following a symlink or trusting its path.

    The descriptor is opened first and then validated with ``fstat`` so a path
    replacement cannot race a separate ``stat``/``open`` sequence.
    """

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise TokenFileError("token file unavailable") from exc

    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise TokenFileError("token file is not regular")
        if info.st_uid != os.geteuid():
            raise TokenFileError("token file has the wrong owner")
        if stat.S_IMODE(info.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
            raise TokenFileError("token file permissions are too broad")
        raw = os.read(fd, MAX_TOKEN_BYTES + 1)
    except OSError as exc:
        raise TokenFileError("token file unavailable") from exc
    finally:
        os.close(fd)

    if len(raw) > MAX_TOKEN_BYTES:
        raise TokenFileError("token file is too large")
    # One conventional trailing newline is accepted; embedded newlines, NULs,
    # surrounding spaces, and non-ASCII lookalikes are not.
    raw = raw.removesuffix(b"\n").removesuffix(b"\r")
    if b"\n" in raw or b"\r" in raw or b"\x00" in raw:
        raise TokenFileError("token file must contain one line")
    try:
        token = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TokenFileError("token must be ASCII") from exc
    if not TOKEN_RE.fullmatch(token):
        raise TokenFileError("token is malformed")
    return token


def _comma_tokens(headers, name: str) -> frozenset[str]:
    """Parse a comma-separated HTTP token header without accepting empty items."""

    result: set[str] = set()
    for value in headers.get_all(name, []):
        pieces = value.split(",")
        if not pieces:
            raise HeaderSyntaxError(f"invalid {name} header")
        for piece in pieces:
            token = piece.strip(" \t")
            if not token or not HTTP_TOKEN_RE.fullmatch(token):
                raise HeaderSyntaxError(f"invalid {name} header")
            result.add(token.lower())
    return frozenset(result)


def _upstream_transfer_mode(headers) -> tuple[str, int | None]:
    """Return ``(mode, length)`` after validating upstream framing."""

    transfer_values = headers.get_all("Transfer-Encoding", [])
    length_values = headers.get_all("Content-Length", [])
    if transfer_values and length_values:
        raise UpstreamFramingError("ambiguous upstream framing")

    transfer_tokens: list[str] = []
    for value in transfer_values:
        pieces = value.split(",")
        for piece in pieces:
            token = piece.strip(" \t").lower()
            if not token or not HTTP_TOKEN_RE.fullmatch(token):
                raise UpstreamFramingError("invalid upstream transfer encoding")
            transfer_tokens.append(token)
    if transfer_tokens:
        if transfer_tokens != ["chunked"]:
            raise UpstreamFramingError("unsupported upstream transfer encoding")
        return "chunked", None

    if not length_values:
        return "close", None
    if len(length_values) != 1:
        raise UpstreamFramingError("conflicting upstream content lengths")
    raw = length_values[0]
    if not CONTENT_LENGTH_RE.fullmatch(raw) or len(raw) > 20:
        raise UpstreamFramingError("invalid upstream content length")
    return "length", int(raw)


class _SocketReadDeadline:
    """Enforce an absolute read deadline, including byte-at-a-time slowloris IO."""

    def __init__(self, connection: socket.socket, seconds: float) -> None:
        self.connection = connection
        self.seconds = seconds
        self.expired = threading.Event()
        self._lock = threading.Lock()
        self._active = False
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> "_SocketReadDeadline":
        with self._lock:
            self._active = True
        self.connection.settimeout(self.seconds)
        self._timer.start()
        return self

    def _expire(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self.expired.set()
        try:
            self.connection.shutdown(socket.SHUT_RD)
        except OSError:
            pass

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        with self._lock:
            self._active = False
        self._timer.cancel()


def _read_available(stream: BinaryIO, size: int) -> bytes:
    read1 = getattr(stream, "read1", None)
    if read1 is not None:
        return read1(size)
    return stream.read(size)


class M3LanProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "m3-lan-proxy"
    sys_version = ""

    @property
    def config(self) -> ProxyConfig:
        return self.server.proxy_config  # type: ignore[attr-defined]

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, _format: str, *args) -> None:
        # launchd owns operational logging.  Never risk echoing authorization
        # data or exception strings through BaseHTTPRequestHandler logging.
        del args

    def send_response_only(self, code: int, message=None) -> None:
        self._response_started = True
        super().send_response_only(code, message)

    def send_error(self, code: int, message=None, explain=None) -> None:
        del message, explain
        try:
            phrase = HTTPStatus(code).phrase.lower()
        except ValueError:
            phrase = "request rejected"
        self._send_json_error(code, phrase)

    def _send_json_error(
        self,
        status: int,
        message: str,
        *,
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if getattr(self, "_response_started", False):
            self.close_connection = True
            return
        payload = json.dumps(
            {"error": {"message": message, "type": "proxy_error"}},
            separators=(",", ":"),
        ).encode("utf-8")
        self.close_connection = True
        try:
            self.connection.settimeout(self.config.client_write_timeout_seconds)
        except (AttributeError, OSError):
            pass
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass

    def handle_one_request(self) -> None:
        """Read exactly one strict HTTP request, then close the connection."""

        self.close_connection = True
        self.command = ""
        self.requestline = ""
        self.request_version = "HTTP/1.1"
        self._response_started = False
        try:
            deadline = _SocketReadDeadline(
                self.connection, self.config.header_timeout_seconds
            )
            with deadline:
                try:
                    self.raw_requestline = self.rfile.readline(
                        MAX_REQUEST_LINE_BYTES + 1
                    )
                except (TimeoutError, OSError):
                    self._send_json_error(
                        HTTPStatus.REQUEST_TIMEOUT, "request headers timed out"
                    )
                    return
                if deadline.expired.is_set():
                    self._send_json_error(
                        HTTPStatus.REQUEST_TIMEOUT, "request headers timed out"
                    )
                    return
                if not self.raw_requestline:
                    return
                if len(self.raw_requestline) > MAX_REQUEST_LINE_BYTES:
                    self._send_json_error(
                        HTTPStatus.REQUEST_URI_TOO_LONG, "request target is too long"
                    )
                    return
                if not self._parse_request_strict(deadline):
                    return

            self.connection.settimeout(self.config.client_write_timeout_seconds)
            if self.command == "GET":
                self.do_GET()
            elif self.command == "POST":
                self.do_POST()
            else:
                self._method_not_allowed()
            try:
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            self.close_connection = True
        except Exception:
            # Never serialize exception text, paths, URLs, or credentials.
            if not self._response_started:
                self._send_json_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "internal proxy error"
                )
            self.close_connection = True

    def _parse_request_strict(self, deadline: _SocketReadDeadline) -> bool:
        raw = self.raw_requestline
        if not raw.endswith(b"\r\n"):
            self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed request line")
            return False
        try:
            requestline = raw[:-2].decode("ascii")
        except UnicodeDecodeError:
            self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed request line")
            return False
        parts = requestline.split(" ")
        if len(parts) != 3 or any(not part for part in parts):
            self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed request line")
            return False

        method, target, version = parts
        self.requestline = requestline
        self.command = method
        if not HTTP_TOKEN_RE.fullmatch(method):
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid method")
            return False
        if version not in {"HTTP/1.0", "HTTP/1.1"}:
            self._send_json_error(
                HTTPStatus.HTTP_VERSION_NOT_SUPPORTED,
                "HTTP version is not supported",
            )
            return False
        self.request_version = version
        if (
            not target.startswith("/")
            or target.startswith("//")
            or "#" in target
            or any(ord(character) <= 32 or ord(character) == 127 for character in target)
        ):
            # This rejects authority-form, absolute-form (including URLs with
            # userinfo), and asterisk-form request targets.
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid request target")
            return False
        self.path = target

        parsed_headers = HTTPMessage()
        total_bytes = 0
        count = 0
        while True:
            try:
                line = self.rfile.readline(MAX_HEADER_LINE_BYTES + 1)
            except (TimeoutError, OSError):
                self._send_json_error(
                    HTTPStatus.REQUEST_TIMEOUT, "request headers timed out"
                )
                return False
            if deadline.expired.is_set():
                self._send_json_error(
                    HTTPStatus.REQUEST_TIMEOUT, "request headers timed out"
                )
                return False
            if not line:
                self._send_json_error(HTTPStatus.BAD_REQUEST, "incomplete headers")
                return False
            if len(line) > MAX_HEADER_LINE_BYTES:
                self._send_json_error(
                    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
                    "request headers are too large",
                )
                return False
            total_bytes += len(line)
            if total_bytes > MAX_HEADER_BYTES:
                self._send_json_error(
                    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
                    "request headers are too large",
                )
                return False
            if line == b"\r\n":
                break
            if not line.endswith(b"\r\n") or line[:1] in {b" ", b"\t"}:
                self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed header")
                return False
            count += 1
            if count > MAX_HEADER_COUNT:
                self._send_json_error(
                    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
                    "too many request headers",
                )
                return False
            name, separator, value = line[:-2].partition(b":")
            if not separator or not HEADER_NAME_RE.fullmatch(name):
                self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed header")
                return False
            if any(byte < 32 and byte != 9 or byte == 127 for byte in value):
                self._send_json_error(HTTPStatus.BAD_REQUEST, "malformed header")
                return False
            parsed_headers.add_header(
                name.decode("ascii"), value.strip(b" \t").decode("latin-1")
            )
        self.headers = parsed_headers

        hosts = self.headers.get_all("Host", [])
        if (version == "HTTP/1.1" and len(hosts) != 1) or len(hosts) > 1:
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid Host header")
            return False
        if hosts and not self._valid_host_header(hosts[0]):
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid Host header")
            return False
        if self.headers.get_all("Expect", []):
            self._send_json_error(
                HTTPStatus.EXPECTATION_FAILED, "Expect is not supported"
            )
            return False
        try:
            _comma_tokens(self.headers, "Connection")
        except HeaderSyntaxError:
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid Connection header")
            return False
        return True

    @staticmethod
    def _valid_host_header(value: str) -> bool:
        if (
            not value
            or value != value.strip(" \t")
            or any(character in value for character in "/?#@")
            or any(ord(character) <= 32 or ord(character) >= 127 for character in value)
        ):
            return False
        try:
            parsed = urlsplit("//" + value)
            port = parsed.port
        except ValueError:
            return False
        del port
        return (
            parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )

    def _target(self, allowed_paths: frozenset[str]) -> str | None:
        if self.path not in allowed_paths:
            self._send_json_error(HTTPStatus.NOT_FOUND, "endpoint not found")
            return None
        return self.path

    def _authorize_post(self) -> bool:
        try:
            expected = read_owner_only_token(self.config.token_file)
        except TokenFileError:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service authentication unavailable",
            )
            return False

        values = self.headers.get_all("Authorization", [])
        supplied = ""
        if len(values) == 1:
            scheme, separator, credential = values[0].partition(" ")
            if (
                separator
                and scheme.lower() == "bearer"
                and TOKEN_RE.fullmatch(credential)
            ):
                supplied = credential
        if not supplied or not hmac.compare_digest(supplied, expected):
            self._send_json_error(
                HTTPStatus.UNAUTHORIZED,
                "authentication required",
                extra_headers=(("WWW-Authenticate", 'Bearer realm="m3-lan-proxy"'),),
            )
            return False
        return True

    def _parse_content_length(self) -> int:
        if self.headers.get_all("Transfer-Encoding", []):
            raise RequestLengthError(
                HTTPStatus.BAD_REQUEST, "transfer encoding is not accepted"
            )
        values = self.headers.get_all("Content-Length", [])
        if not values:
            raise RequestLengthError(
                HTTPStatus.LENGTH_REQUIRED, "content length is required"
            )
        if len(values) != 1:
            raise RequestLengthError(
                HTTPStatus.BAD_REQUEST, "content length must appear once"
            )
        raw = values[0]
        if len(raw) > 20 or not CONTENT_LENGTH_RE.fullmatch(raw):
            raise RequestLengthError(
                HTTPStatus.BAD_REQUEST, "content length is invalid"
            )
        length = int(raw)
        if length > self.config.max_body_bytes:
            raise RequestLengthError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large"
            )
        return length

    def _read_post_body(self) -> bytes | None:
        try:
            length = self._parse_content_length()
        except RequestLengthError as exc:
            self._send_json_error(exc.status, exc.public_message)
            return None
        deadline = _SocketReadDeadline(
            self.connection, self.config.body_timeout_seconds
        )
        try:
            with deadline:
                body = self.rfile.read(length)
        except (TimeoutError, OSError):
            self._send_json_error(HTTPStatus.REQUEST_TIMEOUT, "request body timed out")
            return None
        if deadline.expired.is_set():
            self._send_json_error(HTTPStatus.REQUEST_TIMEOUT, "request body timed out")
            return None
        if len(body) != length:
            self._send_json_error(HTTPStatus.BAD_REQUEST, "request body is incomplete")
            return None
        return body

    def _validate_bodyless_get(self) -> bool:
        if self.headers.get_all("Transfer-Encoding", []):
            self._send_json_error(
                HTTPStatus.BAD_REQUEST, "GET requests cannot have a body"
            )
            return False
        values = self.headers.get_all("Content-Length", [])
        if not values:
            return True
        if (
            len(values) != 1
            or not CONTENT_LENGTH_RE.fullmatch(values[0])
            or values[0] != "0"
        ):
            self._send_json_error(
                HTTPStatus.BAD_REQUEST, "GET requests cannot have a body"
            )
            return False
        return True

    def _forwarded_request_headers(self) -> dict[str, str] | None:
        try:
            nominated = _comma_tokens(self.headers, "Connection")
        except HeaderSyntaxError:
            self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid Connection header")
            return None
        forwarded: dict[str, str] = {}
        for name in FORWARDED_REQUEST_HEADERS:
            lowered = name.lower()
            if lowered in STATIC_HOP_BY_HOP_HEADERS or lowered in nominated:
                continue
            values = self.headers.get_all(name, [])
            if not values:
                continue
            if lowered == "content-type" and len(values) != 1:
                self._send_json_error(HTTPStatus.BAD_REQUEST, "invalid Content-Type header")
                return None
            forwarded[name] = ", ".join(values)
        return forwarded

    def _open_upstream(self, method: str, target: str, body: bytes | None) -> None:
        # Positive allowlist.  Authorization authenticates to this proxy and is
        # deliberately not sent to M3.  Priority and hop-by-hop caller headers
        # are likewise absent.
        headers = self._forwarded_request_headers()
        if headers is None:
            return
        parsed = urlsplit(self.config.upstream)
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(
            parsed.hostname,
            parsed.port,
            timeout=self.config.upstream_timeout_seconds,
        )
        try:
            connection.request(method, target, body=body, headers=headers)
            upstream = connection.getresponse()
        except Exception:
            connection.close()
            self._send_json_error(HTTPStatus.BAD_GATEWAY, "upstream unavailable")
            return
        try:
            self._relay_upstream(upstream)
        finally:
            connection.close()

    def _open_redacted_health(self) -> None:
        """Probe detailed loopback health but return only LAN-safe readiness."""
        headers = self._forwarded_request_headers()
        if headers is None:
            return
        parsed = urlsplit(self.config.upstream)
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(
            parsed.hostname, parsed.port,
            timeout=self.config.upstream_timeout_seconds,
        )
        upstream = None
        try:
            connection.request("GET", "/health", headers=headers)
            upstream = connection.getresponse()
            status = int(getattr(upstream, "status", None) or upstream.code)
            mode, length = _upstream_transfer_mode(upstream.headers)
            if status != HTTPStatus.OK or length is not None \
                    and length > MAX_PUBLIC_HEALTH_BYTES:
                raise UpstreamFramingError("health response is not usable")
            raw = upstream.read(MAX_PUBLIC_HEALTH_BYTES + 1)
            if len(raw) > MAX_PUBLIC_HEALTH_BYTES:
                raise UpstreamFramingError("health response is too large")
            if (mode == "length" and len(raw) != length) or upstream.read(1):
                raise UpstreamFramingError("health response framing is invalid")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or type(value.get("ready")) is not bool:
                raise UpstreamFramingError("health response schema is invalid")
            public = {"ready": value["ready"]}
            model = value.get("model_id")
            if isinstance(model, str) and model and len(model) <= 200:
                public["model"] = model
            payload = json.dumps(public, separators=(",", ":")).encode("utf-8")
            self.close_connection = True
            self.send_response(HTTPStatus.OK if value["ready"] else HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (AttributeError, HeaderSyntaxError, OSError, TypeError, ValueError,
                UnicodeDecodeError, json.JSONDecodeError, UpstreamFramingError):
            self._send_json_error(HTTPStatus.BAD_GATEWAY, "upstream unavailable")
        finally:
            try:
                if upstream is not None:
                    upstream.close()
            except (AttributeError, OSError):
                pass
            connection.close()

    def _relay_upstream(self, upstream) -> None:
        try:
            try:
                mode, length = _upstream_transfer_mode(upstream.headers)
                nominated = _comma_tokens(upstream.headers, "Connection")
                status = int(getattr(upstream, "status", None) or upstream.code)
                if not 200 <= status <= 599:
                    raise UpstreamFramingError("invalid upstream status")
                excluded = STATIC_HOP_BY_HOP_HEADERS | nominated | {"server", "date"}
                response_headers: list[tuple[str, str]] = []
                for name, value in upstream.headers.items():
                    if (
                        not HTTP_TOKEN_RE.fullmatch(name)
                        or "\r" in value
                        or "\n" in value
                        or any(
                            ord(character) < 32 and character != "\t"
                            or ord(character) == 127
                            for character in value
                        )
                    ):
                        raise UpstreamFramingError("invalid upstream header")
                    if name.lower() not in excluded:
                        response_headers.append((name, value))
            except (
                AttributeError,
                HeaderSyntaxError,
                TypeError,
                ValueError,
                UpstreamFramingError,
            ):
                upstream.close()
                self._send_json_error(
                    HTTPStatus.BAD_GATEWAY, "invalid upstream response"
                )
                return

            self.close_connection = True
            self.connection.settimeout(self.config.client_write_timeout_seconds)
            self.send_response(status)
            for name, value in response_headers:
                self.send_header(name, value)
            if mode == "length":
                assert length is not None
                self.send_header("Content-Length", str(length))
            elif mode == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            complete = False
            try:
                if mode == "length":
                    remaining = length or 0
                    while remaining:
                        chunk = _read_available(
                            upstream, min(STREAM_CHUNK_BYTES, remaining)
                        )
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    complete = remaining == 0
                else:
                    while True:
                        chunk = _read_available(upstream, STREAM_CHUNK_BYTES)
                        if not chunk:
                            complete = True
                            break
                        if mode == "chunked":
                            self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        else:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                if mode == "chunked" and complete:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                # Headers may already be on the wire; closing is the only valid
                # response to either an upstream truncation or client departure.
                pass
        finally:
            upstream.close()

    def do_GET(self) -> None:
        target = self._target(GET_PATHS)
        if target is None or not self._validate_bodyless_get():
            return
        if target == "/health":
            self._open_redacted_health()
        else:
            self._open_upstream("GET", target, None)

    def do_POST(self) -> None:
        target = self._target(POST_PATHS)
        if target is None or not self._authorize_post():
            return
        body = self._read_post_body()
        if body is not None:
            self._open_upstream("POST", target, body)

    def _method_not_allowed(self) -> None:
        self._send_json_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method not allowed",
            extra_headers=(("Allow", "GET, POST"),),
        )


class M3LanProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: ProxyConfig) -> None:
        self.proxy_config = config
        super().__init__(address, M3LanProxyHandler)


def build_server(
    config: ProxyConfig,
    *,
    bind: str = DEFAULT_BIND,
    port: int = DEFAULT_PORT,
) -> M3LanProxyServer:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return M3LanProxyServer((bind, port), config)


def main() -> None:
    config = ProxyConfig.from_environment()
    # Fail closed before opening a LAN socket.  Requests re-read the file so a
    # secure atomic token rotation takes effect without restarting the proxy.
    read_owner_only_token(config.token_file)
    port = _strict_environment_int("M3_LAN_PROXY_PORT", DEFAULT_PORT)
    server = build_server(config, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
