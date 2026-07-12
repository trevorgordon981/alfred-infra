#!/usr/bin/env python3
"""V2 machine lease API for direct serving owners or supervised participants.

The custom server owns the lease in-process when launchd starts it directly.
When a pipeline supervisor starts it, the server instead proves that it is in
the supervisor's bound child session.  Token possession without the required
PID/session identity is never sufficient.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
import sys


SCHEMA = "alfred-machine-resource-lease.v2"
SUCCESSOR_SCHEMA = "alfred-machine-resource-successor.v3"
SUCCESSOR_NAME = "successor.json"
TOKEN_RE = re.compile(r"[0-9a-f]{64}")
STARTUP_NONCE_RE = re.compile(r"[0-9a-f]{32}")


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


def _pid(value, field, nullable=False):
    if nullable and value is None:
        return None
    if type(value) is not int or value < 2:
        raise LeaseError("%s is invalid" % field)
    return value


def _process_group_id(value, field):
    if type(value) is not int or value < 1:
        raise LeaseError("%s is invalid" % field)
    return value


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _lease_guard(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LeaseError("cannot open lease coordination guard") from exc
    try:
        opened = os.fstat(fd); current = os.lstat(path)
        if not stat.S_ISDIR(opened.st_mode) \
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise LeaseError("lease directory changed during coordination")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError as exc:
        raise LeaseError("lease coordination failed") from exc
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _process_birth_identity(pid):
    pid = _pid(pid, "process birth PID")
    if sys.platform == "darwin":
        import ctypes

        class ProcBSDInfo(ctypes.Structure):
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32), ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pidinfo
            function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_int]
            function.restype = ctypes.c_int
            info = ProcBSDInfo()
            count = function(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        except (AttributeError, OSError) as exc:
            raise LeaseError("cannot query Darwin process birth identity") from exc
        if count != ctypes.sizeof(info) or info.pbi_pid != pid \
                or info.pbi_start_tvsec <= 0:
            raise LeaseError("process disappeared during Darwin birth proof")
        return "darwin:%d:%d" % (info.pbi_start_tvsec, info.pbi_start_tvusec)
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
            close = stat_text.rfind(")"); fields = stat_text[close + 2:].split()
            start_ticks = int(fields[19])
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii").strip().lower()
        except (OSError, ValueError, IndexError) as exc:
            raise LeaseError("cannot query Linux process birth identity") from exc
        if close < 0 or start_ticks <= 0 \
                or re.fullmatch(r"[0-9a-f-]{36}", boot_id) is None:
            raise LeaseError("Linux process birth identity is malformed")
        return "linux:%s:%d" % (boot_id, start_ticks)
    raise LeaseError("process birth identity is unsupported on this platform")


def _load(path):
    if path.is_symlink() or not path.is_dir():
        raise LeaseError("lease directory is invalid")
    info = path.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise LeaseError("lease directory is not owner-only")
    owner = path / "owner.json"
    if owner.is_symlink() or not owner.is_file():
        raise LeaseError("lease owner receipt is missing")
    owner_info = owner.stat()
    if owner_info.st_uid != os.geteuid() \
            or stat.S_IMODE(owner_info.st_mode) & 0o077:
        raise LeaseError("lease owner receipt is not owner-only")
    raw = owner.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise LeaseError("lease owner receipt is invalid") from exc
    expected = {"schema", "token", "owner_pid", "child_pid", "purpose",
                "created_unix", "bound_unix"}
    if not isinstance(value, dict) or set(value) != expected or value.get("schema") != SCHEMA \
            or _raw(value) != raw:
        raise LeaseError("lease owner receipt is malformed")
    _token(value.get("token"))
    _pid(value.get("owner_pid"), "lease owner PID")
    child_pid = _pid(value.get("child_pid"), "lease child PID", nullable=True)
    if not isinstance(value.get("purpose"), str) or not value["purpose"] \
            or len(value["purpose"]) > 200:
        raise LeaseError("lease purpose is invalid")
    if type(value.get("created_unix")) is not int or value["created_unix"] <= 0:
        raise LeaseError("lease creation time is invalid")
    bound_unix = value.get("bound_unix")
    if (child_pid is None) != (bound_unix is None) \
            or (bound_unix is not None
                and (type(bound_unix) is not int
                     or bound_unix < value["created_unix"])):
        raise LeaseError("lease child binding time is invalid")
    return value


def _load_successor(path):
    successor = path / SUCCESSOR_NAME
    if not os.path.lexists(successor):
        return None
    if successor.is_symlink() or not successor.is_file():
        raise LeaseError("lease successor receipt is invalid")
    info = successor.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise LeaseError("lease successor receipt is not owner-only")
    raw = successor.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise LeaseError("lease successor receipt is invalid") from exc
    expected = {"schema", "original_owner_pid", "child_pid", "successor_pid",
                "successor_sid", "successor_pgid", "successor_birth",
                "state", "startup_nonce", "result_code", "created_unix",
                "committed_unix", "withdrawn_unix"}
    if not isinstance(value, dict) or set(value) != expected \
            or value.get("schema") != SUCCESSOR_SCHEMA or _raw(value) != raw:
        raise LeaseError("lease successor receipt is malformed")
    for field in ("original_owner_pid", "child_pid", "successor_pid"):
        _pid(value.get(field), field)
    _process_group_id(value.get("successor_sid"), "successor_sid")
    _process_group_id(value.get("successor_pgid"), "successor_pgid")
    if not isinstance(value.get("successor_birth"), str) \
            or len(value["successor_birth"]) > 128 \
            or re.fullmatch(r"[A-Za-z0-9:._-]+", value["successor_birth"]) is None:
        raise LeaseError("successor birth identity is invalid")
    if type(value.get("created_unix")) is not int or value["created_unix"] <= 0:
        raise LeaseError("lease successor creation time is invalid")
    state = value.get("state"); nonce = value.get("startup_nonce")
    result_code = value.get("result_code")
    committed = value.get("committed_unix"); withdrawn = value.get("withdrawn_unix")
    if state == "bound":
        valid = nonce is None and result_code is None \
            and committed is None and withdrawn is None
    elif state == "committed":
        valid = isinstance(nonce, str) \
            and STARTUP_NONCE_RE.fullmatch(nonce) is not None \
            and type(result_code) is int and 0 <= result_code <= 255 \
            and type(committed) is int and committed >= value["created_unix"] \
            and withdrawn is None
    elif state == "withdrawn":
        valid = nonce is None and result_code is None and committed is None \
            and type(withdrawn) is int and withdrawn >= value["created_unix"]
    else:
        valid = False
    if not valid:
        raise LeaseError("lease successor state is invalid")
    return value


def _successor_relation(owner, successor):
    if owner["child_pid"] is not None \
            and successor["original_owner_pid"] == owner["owner_pid"] \
            and successor["child_pid"] == owner["child_pid"]:
        return "pending"
    if owner["child_pid"] is None \
            and owner["owner_pid"] == successor["successor_pid"] \
            and successor["state"] == "committed":
        return "transferred-residue"
    raise LeaseError("lease successor receipt does not match owner binding")


def _assert_successor_identity(successor):
    pid = successor["successor_pid"]
    if not _pid_alive(pid):
        raise LeaseError("successor PID is not alive")
    try:
        if os.getsid(pid) != successor["successor_sid"] \
                or os.getpgid(pid) != successor["successor_pgid"] \
                or _process_birth_identity(pid) != successor["successor_birth"]:
            raise LeaseError("successor PID identity changed")
    except ProcessLookupError as exc:
        raise LeaseError("successor exited during identity proof") from exc


def _replace_successor(path, value):
    temp = path / (".successor.%s.tmp" % secrets.token_hex(16))
    fd = None
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_raw(value)); stream.flush()
        os.fsync(fd); os.close(fd); fd = None
        os.replace(temp, path / SUCCESSOR_NAME)
        dfd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def await_successor(path_text, timeout_seconds=180.0):
    """Wait behind the lock until this exact launchd PID is bound as successor."""
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) \
            or not 0 < timeout_seconds <= 300:
        raise LeaseError("successor wait must be between 0 and 300 seconds")
    path = _path(path_text)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = _load(path)
        successor = _load_successor(path)
        if successor is not None and successor["successor_pid"] == os.getpid():
            if _successor_relation(value, successor) != "pending":
                raise LeaseError("successor receipt does not match the lease owner")
            if successor["state"] == "withdrawn":
                raise LeaseError("this successor has been withdrawn")
            child_pid = value["child_pid"]
            if child_pid is None or not _pid_alive(value["owner_pid"]) \
                    or not _pid_alive(child_pid):
                raise LeaseError("successor owner or child is no longer alive")
            _assert_successor_identity(successor)
            return value
        time.sleep(0.05)
    raise LeaseError("this server PID was not bound as the lease successor")


def _assert_entries(path, allowed):
    if set(os.listdir(path)) != allowed:
        raise LeaseError("lease directory contains unexpected entries")


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
        value = {"schema": SCHEMA, "token": token,
                 "owner_pid": os.getpid(), "child_pid": None,
                 "purpose": purpose, "created_unix": int(time.time()),
                 "bound_unix": None}
        fd = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_raw(value)); stream.flush()
        os.fsync(fd); os.close(fd); fd = None
        dfd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
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
    if value["owner_pid"] != os.getpid():
        raise LeaseError("caller is not the recorded lease owner")
    return value


def assert_participant(path_text, token):
    value = _load(_path(path_text)); expected = _token(token)
    if not secrets.compare_digest(value["token"], expected):
        raise LeaseError("machine resource lease token does not match")
    child_pid = value["child_pid"]
    if child_pid is None:
        raise LeaseError("lease child has not been bound")
    if not _pid_alive(value["owner_pid"]) or not _pid_alive(child_pid):
        raise LeaseError("lease owner or child is no longer alive")
    try:
        if os.getsid(0) != child_pid or os.getpgid(0) != child_pid:
            raise LeaseError("caller is outside the bound lease child session")
    except ProcessLookupError as exc:
        raise LeaseError("caller process identity disappeared") from exc
    return value


def release(path_text, token):
    path = _path(path_text)
    with _lease_guard(path):
        value = assert_owned(str(path), token)
        child_pid = value["child_pid"]
        if child_pid is not None and (_pid_alive(child_pid) or _group_alive(child_pid)):
            raise LeaseError("cannot release while the bound child group is alive")
        successor = _load_successor(path)
        if successor is not None:
            relation = _successor_relation(value, successor)
            if relation == "transferred-residue":
                _assert_successor_identity(successor)
                try:
                    (path / SUCCESSOR_NAME).unlink()
                except FileNotFoundError:
                    pass
            elif _pid_alive(successor["successor_pid"]):
                raise LeaseError("cannot release while a bound successor is alive")
            else:
                (path / SUCCESSOR_NAME).unlink()
        _assert_entries(path, {"owner.json"})
        (path / "owner.json").unlink(); path.rmdir()


def recover_dead_owner(path_text):
    """Remove only a canonical lease whose complete recorded lifecycle is dead."""
    if not path_text or not os.path.isabs(path_text) \
            or os.path.normpath(path_text) != path_text:
        raise LeaseError("lease path must be normalized and absolute")
    if not os.path.lexists(path_text):
        return False
    path = _path(path_text)
    with _lease_guard(path):
        value = _load(path); live = []
        if _pid_alive(value["owner_pid"]):
            live.append("owner_pid=%d" % value["owner_pid"])
        child_pid = value["child_pid"]
        if child_pid is not None:
            if _pid_alive(child_pid):
                live.append("child_pid=%d" % child_pid)
            if _group_alive(child_pid):
                live.append("child_pgid=%d" % child_pid)
        successor = _load_successor(path)
        _assert_entries(path, {"owner.json"} if successor is None
                        else {"owner.json", SUCCESSOR_NAME})
        if successor is not None:
            _successor_relation(value, successor)
            if _pid_alive(successor["successor_pid"]):
                live.append("successor_pid=%d" % successor["successor_pid"])
        if live:
            raise LeaseError(
                "recorded lease process is still alive (%s)" % ", ".join(live))
        if successor is not None:
            (path / SUCCESSOR_NAME).unlink()
        (path / "owner.json").unlink(); path.rmdir()
        return True


def assert_pending_successor(path_text, token, startup_nonce):
    """Prove this exact process is a live pending successor, never just a token holder."""
    path = _path(path_text); expected = _token(token)
    value = _load(path)
    if not secrets.compare_digest(value["token"], expected):
        raise LeaseError("machine resource lease token does not match")
    successor = _load_successor(path)
    if successor is None or _successor_relation(value, successor) != "pending" \
            or successor["successor_pid"] != os.getpid():
        raise LeaseError("caller is not the pending lease successor")
    if not _pid_alive(value["owner_pid"]):
        raise LeaseError("pending successor owner is no longer alive")
    if successor["state"] == "withdrawn":
        raise LeaseError("pending successor is already withdrawn")
    if successor["state"] == "committed" and (
            not isinstance(startup_nonce, str)
            or not secrets.compare_digest(successor["startup_nonce"], startup_nonce)):
        raise LeaseError("pending successor startup nonce does not match")
    _assert_successor_identity(successor)
    return value, successor


def retain_or_withdraw_successor(path_text, token, startup_nonce):
    """Atomically retain after transfer or withdraw before transfer.

    The directory flock prevents the supervisor from transferring ownership
    between the ownership check and the withdrawal write.
    """
    path = _path(path_text)
    with _lease_guard(path):
        value = _load(path); expected = _token(token)
        if not secrets.compare_digest(value["token"], expected):
            raise LeaseError("machine resource lease token does not match")
        if value["owner_pid"] == os.getpid():
            if value["child_pid"] is not None:
                raise LeaseError("transferred successor unexpectedly has a child")
            successor = _load_successor(path)
            if successor is not None:
                if _successor_relation(value, successor) != "transferred-residue":
                    raise LeaseError("transferred successor receipt is inconsistent")
                _assert_successor_identity(successor)
                if successor["state"] == "committed" and (
                        not isinstance(startup_nonce, str)
                        or not secrets.compare_digest(
                            successor["startup_nonce"], startup_nonce)):
                    raise LeaseError("transferred successor startup nonce does not match")
            _assert_entries(
                path, {"owner.json"} if successor is None
                else {"owner.json", SUCCESSOR_NAME})
            # Never delete a direct/transferred owner while this resource
            # process is alive. The next authorized launcher recovers only
            # after kernel death proof.
            return "retained"

        _owner, successor = assert_pending_successor(
            str(path), token, startup_nonce)
        withdrawn = dict(successor)
        withdrawn["state"] = "withdrawn"
        withdrawn["startup_nonce"] = None
        withdrawn["result_code"] = None
        withdrawn["committed_unix"] = None
        withdrawn["withdrawn_unix"] = max(
            int(time.time()), successor["created_unix"])
        _replace_successor(path, withdrawn)
        return "withdrawn"
