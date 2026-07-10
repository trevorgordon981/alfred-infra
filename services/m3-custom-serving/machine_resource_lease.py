#!/usr/bin/env python3
"""Shared fail-closed machine resource lease used by train/build/eval/serve."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import stat
import time


SCHEMA = "alfred-machine-resource-lease.v1"
TOKEN_RE = re.compile(r"[0-9a-f]{64}")


class LeaseError(RuntimeError):
    pass


def _raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _path(value):
    if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
        raise LeaseError("lease path must be normalized and absolute")
    path = Path(value); parent = path.parent
    if not parent.is_dir() or parent.is_symlink() or os.path.realpath(parent) != str(parent):
        raise LeaseError("lease parent must be a canonical directory")
    info = parent.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise LeaseError("lease parent must be owner-controlled")
    return path


def _token(value):
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise LeaseError("lease token is invalid")
    return value


def _load(path):
    if path.is_symlink() or not path.is_dir():
        raise LeaseError("lease directory is invalid")
    info = path.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise LeaseError("lease directory is not owner-only")
    owner = path / "owner.json"
    if owner.is_symlink() or not owner.is_file():
        raise LeaseError("lease owner receipt is missing")
    raw = owner.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise LeaseError("lease owner receipt is invalid") from exc
    expected = {"schema", "token", "pid", "purpose", "created_unix"}
    if not isinstance(value, dict) or set(value) != expected or value.get("schema") != SCHEMA \
            or _raw(value) != raw:
        raise LeaseError("lease owner receipt is malformed")
    _token(value.get("token"))
    return value


def acquire(path_text, purpose, token=None):
    path = _path(path_text)
    if not isinstance(purpose, str) or not purpose or len(purpose) > 200:
        raise LeaseError("lease purpose is invalid")
    token = _token(token) if token is not None else secrets.token_hex(32)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as exc:
        raise LeaseError("machine resource lease is held") from exc
    owner = path / "owner.json"; fd = None
    try:
        value = {"schema": SCHEMA, "token": token, "pid": os.getpid(),
                 "purpose": purpose, "created_unix": int(time.time())}
        fd = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_raw(value)); stream.flush()
        os.fsync(fd); os.close(fd); fd = None
    except BaseException:
        if fd is not None:
            os.close(fd)
        owner.unlink(missing_ok=True)
        path.rmdir()
        raise
    return token


def assert_owned(path_text, token):
    value = _load(_path(path_text)); expected = _token(token)
    if not secrets.compare_digest(value["token"], expected):
        raise LeaseError("machine resource lease token does not match")
    return value


def release(path_text, token):
    path = _path(path_text); assert_owned(str(path), token)
    (path / "owner.json").unlink(); path.rmdir()
