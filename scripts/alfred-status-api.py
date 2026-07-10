#!/usr/bin/env python3
"""alfred-status-api: read-only HTTP API on bat-studio for Alfred (Modal sandbox).

Exposes the bat-studio-local CLIs as JSON-over-HTTPS endpoints. Tailscale
Funnel then makes this reachable from Alfred's public-internet container.

Bind to localhost only; Funnel forwards to it. Bearer-token auth via
ALFRED_STATUS_API_TOKEN env (so a leaked URL isn't open-internet abuse).
"""
from __future__ import annotations

import hmac
import json
import os
import stat
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Mapping

PORT = int(os.environ.get("ALFRED_STATUS_API_PORT", "8888"))
MIN_TOKEN_BYTES = 32
MAX_TOKEN_BYTES = 512
MAX_CLI_JSON_BYTES = 1024 * 1024


def _validated_token(value: str) -> str:
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PermissionError("status API token must be ASCII") from exc
    if not MIN_TOKEN_BYTES <= len(raw) <= MAX_TOKEN_BYTES \
            or any(byte <= 0x20 or byte >= 0x7F for byte in raw):
        raise PermissionError(
            f"status API token must be {MIN_TOKEN_BYTES}-{MAX_TOKEN_BYTES} "
            "printable non-whitespace ASCII bytes")
    return value

def load_token(
    environ: Mapping[str, str] | None = None,
    token_path: Path | None = None,
) -> str:
    """Load an API token with explicit env-first precedence.

    The previous conditional expression accidentally discarded a valid env
    token whenever the fallback file was absent. Token loading is kept out of
    module import so tests/tools never read the operator's credential file.
    """
    environ = os.environ if environ is None else environ
    env_token = environ.get("ALFRED_STATUS_API_TOKEN", "").strip()
    if env_token:
        return _validated_token(env_token)

    if token_path is None:
        configured = environ.get("ALFRED_STATUS_API_TOKEN_FILE", "").strip()
        token_path = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".alfred-status-api-token"
        )
    try:
        token_lstat = token_path.lstat()
    except FileNotFoundError:
        return ""
    if stat.S_ISLNK(token_lstat.st_mode) or not stat.S_ISREG(token_lstat.st_mode):
        raise PermissionError(f"token path is not a regular file: {token_path}")
    if stat.S_IMODE(token_lstat.st_mode) != 0o600:
        raise PermissionError(f"token file must be mode 0600: {token_path}")
    if hasattr(os, "getuid") and token_lstat.st_uid != os.getuid():
        raise PermissionError(f"token file must be owned by the service user: {token_path}")
    fd = os.open(token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (token_lstat.st_dev, token_lstat.st_ino):
            raise PermissionError("token file changed while opening")
        raw = os.read(fd, MAX_TOKEN_BYTES + 2)
        closed = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
                opened.st_ctime_ns) != (closed.st_dev, closed.st_ino,
                closed.st_size, closed.st_mtime_ns, closed.st_ctime_ns):
            raise PermissionError("token file changed while reading")
    finally:
        os.close(fd)
    try:
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PermissionError("token file must contain ASCII") from exc
    return _validated_token(token)


# Populated once, in main(). An empty token always means deny, never dev mode.
TOKEN = ""

HERMES_BIN = Path.home() / ".hermes" / "bin"

# Endpoint → CLI command map. Each CLI must support --json output.
ROUTES = {
    "/pretrain-status": [str(HERMES_BIN / "pretrain-status"), "--json"],
    "/spark-health": [str(HERMES_BIN / "spark-health"), "--json"],
    "/mining-summary": [str(HERMES_BIN / "mining-summary"), "--json"],
    "/forgejo-mirror-audit": [str(HERMES_BIN / "forgejo-mirror-audit"), "--json"],
    "/bitaxe-monitor": [str(HERMES_BIN / "bitaxe-monitor"), "--json"],
    "/nas-storage-report": [str(HERMES_BIN / "nas-storage-report"), "--json"],
    # Routes below accept query-string args appended via QUERY_PARAM_MAP.
    "/runbook": [str(HERMES_BIN / "runbook"), "--json"],
    "/rma-evidence": [str(HERMES_BIN / "rma-evidence"), "--json"],
    "/alfred-incident": [str(HERMES_BIN / "alfred-incident"), "--json"],
}

# For routes that take args, allow specific query-string keys to be appended
# to the CLI. Keys not in this list are silently ignored (defense-in-depth).
QUERY_PARAM_MAP = {
    "/runbook": ["name"],          # /runbook?name=03-nfs-dropped
    "/rma-evidence": ["host"],     # /rma-evidence?host=node1
    "/alfred-incident": ["host"],  # /alfred-incident?host=node1
}


class Handler(BaseHTTPRequestHandler):
    server_version = "alfred-status-api"
    sys_version = ""

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if not TOKEN:
            return False
        header = self.headers.get("Authorization", "")
        scheme, separator, candidate = header.partition(" ")
        if scheme.lower() != "bearer" or not separator or not candidate:
            return False
        return hmac.compare_digest(candidate, TOKEN)

    def do_GET(self) -> None:  # noqa: N802
        # Split off query string so /runbook?name=X resolves to the /runbook route.
        parsed = urllib.parse.urlparse(self.path)
        route_path = parsed.path
        query = urllib.parse.parse_qs(parsed.query) if parsed.query else {}

        # Authenticate every route, including the endpoint index. This service
        # is reachable through Funnel; there is no unauthenticated dev mode.
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return

        if route_path == "/":
            self._json(200, {
                "endpoints": sorted(ROUTES.keys()),
                "auth": "Bearer token required",
            })
            return

        if route_path not in ROUTES:
            self._json(404, {"error": "not found"})
            return

        cmd = list(ROUTES[route_path])
        # Append whitelisted query-string args as positional CLI args.
        for key in QUERY_PARAM_MAP.get(route_path, []):
            val = query.get(key, [None])[0]
            if val:
                # Allow simple alphanumeric + a few safe punct chars only.
                if all(c.isalnum() or c in "-_." for c in val):
                    cmd.append(val)
        try:
            # 180s tolerates the slow du in nas-storage-report; others return in seconds.
            child_env = {
                key: value for key, value in os.environ.items()
                if key not in {"ALFRED_STATUS_API_TOKEN", "ALFRED_STATUS_API_TOKEN_FILE"}
            }
            r = subprocess.run(
                cmd, capture_output=True, text=False, timeout=180,
                env=child_env, check=False)
            if r.returncode != 0:
                self._json(502, {"error": "status command failed"})
                return
            if not r.stdout or len(r.stdout) > MAX_CLI_JSON_BYTES:
                self._json(502, {"error": "status command returned invalid data"})
                return
            try:
                payload = json.loads(
                    r.stdout.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant: {value}")),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._json(502, {"error": "status command returned invalid data"})
                return
            if not isinstance(payload, (dict, list)):
                self._json(502, {"error": "status command returned invalid data"})
                return
            body = json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "status command timed out"})
        except OSError:
            # Missing or temporarily unavailable local tools are an upstream
            # failure, not an excuse to leak host paths or exception text.
            self._json(502, {"error": "status command unavailable"})

    def log_message(self, fmt, *args):  # quieter
        return


def main() -> None:
    global TOKEN
    try:
        TOKEN = load_token()
    except (OSError, PermissionError) as exc:
        print(f"alfred-status-api: refusing to start: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not TOKEN:
        print(
            "alfred-status-api: refusing to start without a bearer token",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"alfred-status-api listening on 127.0.0.1:{PORT}", flush=True)
    print("  auth: Bearer token", flush=True)
    print(f"  endpoints: {sorted(ROUTES.keys())}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
