#!/usr/bin/env python3
import importlib.util
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
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

    @staticmethod
    def _enqueue(priority):
        return SERVER._enqueue([], 1, 0.0, False, None, priority=priority)

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

    def test_default_backlog_cannot_consume_reserved_urgent_capacity(self):
        pending = SERVER._queue.PriorityQueue(maxsize=3)
        slots = threading.BoundedSemaphore(3)
        with mock.patch.object(SERVER, "JOBQ", pending), \
                mock.patch.object(SERVER, "_QUEUE_SLOTS", slots), \
                mock.patch.object(SERVER, "MAX_PENDING", 2), \
                mock.patch.object(SERVER, "PRIORITY0_RESERVED", 1), \
                mock.patch.object(SERVER, "MAX_TOTAL_PENDING", 3):
            self._enqueue(1)
            self._enqueue(1)
            with self.assertRaises(SERVER.Busy):
                self._enqueue(1)
            self._enqueue(0)
            self.assertEqual(3, pending.qsize())
            with self.assertRaises(SERVER.Busy):
                self._enqueue(0)

    def test_concurrent_urgent_overload_never_exceeds_hard_cap(self):
        pending = SERVER._queue.PriorityQueue(maxsize=4)
        slots = threading.BoundedSemaphore(4)
        barrier = threading.Barrier(16)
        accepted = []
        rejected = []
        outcome_lock = threading.Lock()

        def submit():
            barrier.wait()
            try:
                self._enqueue(0)
                outcome = accepted
            except SERVER.Busy:
                outcome = rejected
            with outcome_lock:
                outcome.append(1)

        with mock.patch.object(SERVER, "JOBQ", pending), \
                mock.patch.object(SERVER, "_QUEUE_SLOTS", slots), \
                mock.patch.object(SERVER, "MAX_PENDING", 2), \
                mock.patch.object(SERVER, "PRIORITY0_RESERVED", 2), \
                mock.patch.object(SERVER, "MAX_TOTAL_PENDING", 4):
            threads = [threading.Thread(target=submit) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(4, len(accepted))
            self.assertEqual(12, len(rejected))
            self.assertEqual(4, pending.qsize())


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


class RuntimePackageTreeTests(unittest.TestCase):
    PACKAGES = ("mlx", "mlx_lm", "mlx_vlm")

    def _packages(self, root):
        roots = {}
        modules = {}
        for package_name in self.PACKAGES:
            package_root = Path(root) / package_name
            package_root.mkdir()
            package_file = package_root / "__init__.py"
            package_file.write_text("PACKAGE = %r\n" % package_name, encoding="utf-8")
            (package_root / "model.py").write_text(
                "MODEL = %r\n" % package_name, encoding="utf-8")
            (package_root / "native.dylib").write_bytes(
                ("native-" + package_name).encode("ascii"))
            roots[package_name] = package_root
            modules[package_name] = SimpleNamespace(__file__=str(package_file))
        return roots, modules

    def _digests(self):
        return {name: SERVER._runtime_package_tree_sha256(name)
                for name in self.PACKAGES}

    def test_all_package_fingerprints_bind_source_and_native_artifacts(self):
        tamper_cases = (("mlx", "native.dylib", b"changed-native"),
                        ("mlx_lm", "model.py", b"MODEL = 'changed'\n"),
                        ("mlx_vlm", "native.dylib", b"changed-vlm-native"))
        for changed_name, relative_path, replacement in tamper_cases:
            with self.subTest(package=changed_name, path=relative_path), \
                    tempfile.TemporaryDirectory(dir=HERE) as td:
                roots, modules = self._packages(td)
                with mock.patch.dict(sys.modules, modules):
                    before = self._digests()
                    self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", item)
                                        for item in before.values()))
                    (roots[changed_name] / relative_path).write_bytes(replacement)
                    after = self._digests()
                self.assertNotEqual(before[changed_name], after[changed_name])
                for stable_name in set(self.PACKAGES) - {changed_name}:
                    self.assertEqual(before[stable_name], after[stable_name])

    def test_package_tree_binds_bytecode_cache(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            roots, modules = self._packages(td)
            with mock.patch.dict(sys.modules, modules):
                first = SERVER._runtime_package_tree_sha256("mlx_vlm")
                cache = roots["mlx_vlm"] / "__pycache__"
                cache.mkdir()
                (cache / "ignored.pyc").write_bytes(b"cache")
                self.assertNotEqual(first, SERVER._runtime_package_tree_sha256("mlx_vlm"))

    def test_package_tree_traversal_error_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            _roots, modules = self._packages(td)

            def denied_walk(_root, *, topdown, followlinks, onerror):
                self.assertTrue(topdown)
                self.assertFalse(followlinks)
                onerror(PermissionError("unreadable native subtree"))
                yield from ()

            with mock.patch.dict(sys.modules, modules), \
                    mock.patch.object(SERVER.os, "walk", new=denied_walk), \
                    self.assertRaises(SERVER.StartupIdentityError):
                SERVER._runtime_package_tree_sha256("mlx")

    def test_loaded_runtime_must_equal_preimport_snapshot(self):
        trees = {"mlx": "1" * 64, "mlx_lm": "2" * 64,
                 "mlx_vlm": "3" * 64}
        batch = {"path": "/tmp/m3_batch_core.py", "sha256": "4" * 64}
        loaded = SimpleNamespace(__file__=batch["path"])
        with mock.patch.dict(sys.modules, {"m3_batch_core": loaded}), \
                mock.patch.object(SERVER, "_runtime_package_tree_sha256",
                                  side_effect=lambda name: trees[name]), \
                mock.patch.object(SERVER, "_runtime_file_identity", return_value=batch), \
                mock.patch.object(SERVER, "_PREIMPORT_PACKAGE_TREES", dict(trees)), \
                mock.patch.object(SERVER, "_PREIMPORT_BATCH_CORE_IDENTITY", dict(batch)):
            self.assertEqual((trees, batch), SERVER._verified_runtime_code_identity())
        changed = dict(trees, mlx="9" * 64)
        with mock.patch.dict(sys.modules, {"m3_batch_core": loaded}), \
                mock.patch.object(SERVER, "_runtime_package_tree_sha256",
                                  side_effect=lambda name: trees[name]), \
                mock.patch.object(SERVER, "_runtime_file_identity", return_value=batch), \
                mock.patch.object(SERVER, "_PREIMPORT_PACKAGE_TREES", changed), \
                mock.patch.object(SERVER, "_PREIMPORT_BATCH_CORE_IDENTITY", dict(batch)), \
                self.assertRaises(SERVER.StartupIdentityError):
            SERVER._verified_runtime_code_identity()

    def test_each_package_rejects_symlink_files_and_directories(self):
        for package_name in self.PACKAGES:
            for kind in ("file", "directory"):
                with self.subTest(package=package_name, kind=kind), \
                        tempfile.TemporaryDirectory(dir=HERE) as td:
                    roots, modules = self._packages(td)
                    target = Path(td) / ("target.py" if kind == "file" else "target-dir")
                    if kind == "file":
                        target.write_text("outside = True\n", encoding="utf-8")
                        (roots[package_name] / "linked.py").symlink_to(target)
                    else:
                        target.mkdir()
                        (roots[package_name] / "linked-dir").symlink_to(
                            target, target_is_directory=True)
                    with mock.patch.dict(sys.modules, modules), \
                            self.assertRaises(SERVER.StartupIdentityError):
                        SERVER._runtime_package_tree_sha256(package_name)

    def test_package_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            roots, modules = self._packages(td)
            package_root = roots["mlx"]
            real_root = Path(td) / "real-mlx"
            package_root.rename(real_root)
            package_root.symlink_to(real_root, target_is_directory=True)
            with mock.patch.dict(sys.modules, modules), \
                    self.assertRaises(SERVER.StartupIdentityError):
                SERVER._runtime_package_tree_sha256("mlx")


class StartupIdentityTests(unittest.TestCase):
    def _manifest(self, root, candidate, value=None):
        path = Path(root) / "artifact.json"
        body = value if value is not None else {
            "schema": 3, "kind": "pipeline-artifact", "artifact_type": "model-build",
            "artifact_id": "0123456789abcdef0123456789abcdef", "created_unix": 1,
            "label": "M3", "version": "v1", "candidate": {"path": str(candidate)},
            "base": {}, "adapter": {}, "evidence": [], "metadata": {},
        }
        raw = (json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True) + "\n").encode("ascii")
        path.write_bytes(raw)
        return path, raw

    def test_malformed_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; model.mkdir()
            manifest = Path(td) / "artifact.json"
            for raw in (b"{bad", b"[]", b'{"candidate": {}}'):
                with self.subTest(raw=raw):
                    manifest.write_bytes(raw)
                    with self.assertRaises(SERVER.StartupIdentityError):
                        SERVER._build_startup_identity(str(model), str(manifest))

    def test_wrong_candidate_path_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; other = Path(td) / "other"
            model.mkdir(); other.mkdir()
            manifest, _ = self._manifest(td, other)
            with self.assertRaisesRegex(SERVER.StartupIdentityError,
                                        "does not match M3_MODEL_DIR"):
                SERVER._build_startup_identity(str(model), str(manifest))

    def test_bad_configured_manifest_aborts_module_startup(self):
        loader = (
            "import importlib.util,sys; "
            "s=importlib.util.spec_from_file_location('identity_fail',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
        )
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; other = Path(td) / "other"
            model.mkdir(); other.mkdir()
            malformed = Path(td) / "malformed.json"
            malformed.write_text("{bad", encoding="utf-8")
            wrong, _ = self._manifest(td, other)
            for manifest in (malformed, wrong):
                with self.subTest(manifest=manifest.name):
                    env = os.environ.copy()
                    env.update({"M3_MODEL_DIR": str(model),
                                "M3_ARTIFACT_MANIFEST": str(manifest),
                                "M3_APC": "0", "M3_PREFIX_CACHE": "0"})
                    proc = subprocess.run(
                        [sys.executable, "-c", loader, str(HERE / "m3_serve_batched.py")],
                        env=env, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(0, proc.returncode)
                    self.assertIn("StartupIdentityError", proc.stderr)

    def test_valid_manifest_binds_raw_digest_and_stable_fields(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; model.mkdir()
            manifest, raw = self._manifest(td, model)
            identity = SERVER._build_startup_identity(
                str(model), str(manifest), pid=123, started_unix=456,
                startup_nonce="0123456789abcdef0123456789abcdef")
            self.assertEqual("0123456789abcdef0123456789abcdef", identity["startup_nonce"])
            self.assertEqual(123, identity["pid"])
            self.assertEqual(456, identity["started_unix"])
            self.assertEqual(str(model.resolve()), identity["model_realpath"])
            self.assertEqual("0123456789abcdef0123456789abcdef", identity["artifact_id"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),
                             identity["artifact_manifest_sha256"])

    def test_direction_artifact_schemas_are_supported_but_strict(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; model.mkdir()
            base = {
                "kind": "pipeline-artifact", "artifact_type": "m3-abliteration-build",
                "artifact_id": "0123456789abcdef0123456789abcdef", "created_unix": 1,
                "label": "M3", "version": "abl-v1", "candidate": {"path": str(model)},
                "base": {}, "direction_bundle": {}, "recipe": {},
                "direction_binding": {}, "evidence": [], "metadata": {},
            }
            for schema in (4, 5):
                with self.subTest(schema=schema):
                    value = dict(base, schema=schema)
                    if schema == 5:
                        value["input_binding"] = {}
                    manifest, raw = self._manifest(td, model, value)
                    identity = SERVER._build_startup_identity(
                        str(model), str(manifest), pid=123, started_unix=456,
                        startup_nonce="0123456789abcdef0123456789abcdef")
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                     identity["artifact_manifest_sha256"])
            value = dict(base, schema=5, input_binding={})
            value["artifact_type"] = "model-build"
            manifest.write_bytes((json.dumps(
                value, sort_keys=True, separators=(",", ":")) + "\n").encode())
            with self.assertRaises(SERVER.StartupIdentityError):
                SERVER._build_startup_identity(str(model), str(manifest))

    def test_unconfigured_manifest_preserves_identity_with_null_artifact(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            model = Path(td) / "model"; model.mkdir()
            identity = SERVER._build_startup_identity(
                str(model), pid=2, started_unix=2,
                startup_nonce="0123456789abcdef0123456789abcdef")
            self.assertIsNone(identity["artifact_id"])
            self.assertIsNone(identity["artifact_manifest_sha256"])

    def test_runtime_receipt_is_create_only_and_health_bound(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root = Path(td); model = root / "model"; model.mkdir()
            runtime = root / "runtime"; runtime.mkdir(mode=0o700)
            manifest, _raw = self._manifest(td, model)
            identity = SERVER._build_startup_identity(
                str(model), str(manifest), pid=os.getpid(), started_unix=456,
                startup_nonce="0123456789abcdef0123456789abcdef")
            old = SERVER.SERVER_IDENTITY
            old_runtime_out = SERVER._RUNTIME_RECEIPT_OUT
            old_loaded = (SERVER.MODEL, SERVER.PROC, SERVER.CFG)
            runtime_digests = {"mlx": "1" * 64, "mlx_lm": "2" * 64,
                               "mlx_vlm": "3" * 64}
            batch_identity = {"path": str(HERE / "m3_batch_core.py"),
                              "sha256": "4" * 64}
            try:
                SERVER.SERVER_IDENTITY = identity
                SERVER._RUNTIME_RECEIPT_OUT = None
                SERVER.MODEL, SERVER.PROC, SERVER.CFG = object(), object(), object()
                smoke = {"schema": "m3-readiness-smoke.v1", "startup_nonce":
                         "0123456789abcdef0123456789abcdef",
                         "artifact_id": "0123456789abcdef0123456789abcdef",
                         "tests": [{"name": "mock"}], "created_unix": 1}
                with mock.patch.object(
                        SERVER, "_runtime_package_tree_sha256",
                        side_effect=lambda package: runtime_digests[package]), \
                        mock.patch.object(SERVER, "_runtime_file_identity",
                                          side_effect=lambda _path, label: batch_identity
                                          if label == "m3_batch_core"
                                          else {"sha256": "5" * 64}), \
                        mock.patch.object(SERVER, "_PREIMPORT_PACKAGE_TREES",
                                          dict(runtime_digests)), \
                        mock.patch.object(SERVER, "_PREIMPORT_BATCH_CORE_IDENTITY",
                                          dict(batch_identity)), \
                        mock.patch.dict(sys.modules, {"m3_batch_core": SimpleNamespace(
                            __file__=batch_identity["path"])}), \
                        mock.patch.object(SERVER, "_ARTIFACT_MANIFEST", str(manifest)), \
                        mock.patch.dict(os.environ,
                                        {"M3_RUNTIME_RECEIPT_DIR": str(runtime)}, clear=False):
                    SERVER._initialize_runtime_identity(smoke)
                self.assertRegex(identity["runtime_receipt_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(identity["runtime_contract_sha256"], r"^[0-9a-f]{64}$")
                receipts = list(runtime.glob("runtime.M3.*.json"))
                self.assertEqual(1, len(receipts))
                self.assertEqual(identity["runtime_receipt_sha256"],
                                 hashlib.sha256(receipts[0].read_bytes()).hexdigest())
                receipt = json.loads(receipts[0].read_text())
                self.assertEqual(receipt, identity["runtime_receipt"])
                self.assertEqual(runtime_digests["mlx"],
                                 receipt["contract"]["mlx_tree_sha256"])
                self.assertEqual(runtime_digests["mlx_lm"],
                                 receipt["contract"]["mlx_lm_tree_sha256"])
                self.assertEqual(runtime_digests["mlx_vlm"],
                                 receipt["contract"]["mlx_vlm_tree_sha256"])
                self.assertEqual(SERVER.MAX_TOTAL_PENDING,
                                 receipt["contract"]["max_total_pending"])
                self.assertEqual(SERVER.PRIORITY0_RESERVED,
                                 receipt["contract"]["priority0_reserved"])
                self.assertEqual(smoke, receipt["readiness_smoke"])
                self.assertEqual(identity["readiness_smoke_sha256"],
                                 receipt["readiness_smoke_sha256"])
                contract_raw = (json.dumps(receipt["contract"], sort_keys=True,
                                           separators=(",", ":"), ensure_ascii=True,
                                           allow_nan=False) + "\n").encode("ascii")
                self.assertEqual(receipt["contract_sha256"],
                                 hashlib.sha256(contract_raw).hexdigest())
                with mock.patch.object(SERVER, "_ARTIFACT_MANIFEST", str(manifest)), \
                        mock.patch.dict(os.environ,
                                        {"M3_RUNTIME_RECEIPT_DIR": str(runtime)}, clear=False), \
                        self.assertRaises(SERVER.StartupIdentityError):
                    SERVER._initialize_runtime_identity(smoke)
            finally:
                SERVER.SERVER_IDENTITY = old
                SERVER._RUNTIME_RECEIPT_OUT = old_runtime_out
                SERVER.MODEL, SERVER.PROC, SERVER.CFG = old_loaded

    def test_prepare_readiness_requires_artifact_manifest(self):
        with mock.patch.object(SERVER, "_ARTIFACT_MANIFEST", None), \
                self.assertRaisesRegex(SERVER.StartupIdentityError, "required"):
            SERVER._prepare_runtime_identity()


class ReadinessSmokeTests(unittest.TestCase):
    def test_all_protocol_smokes_are_required_and_receipted(self):
        outputs = [
            SimpleNamespace(text="READY", prompt_tokens=3, generation_tokens=1),
            SimpleNamespace(text='{"ready":true}', prompt_tokens=5, generation_tokens=4),
            SimpleNamespace(text='<tool_call><invoke name="health_probe"></invoke></tool_call>',
                            prompt_tokens=7, generation_tokens=5),
        ]

        def fake_run(_messages, _mt, _temp, _think, _tools, **kwargs):
            if kwargs.get("stream_q") is not None:
                out = SimpleNamespace(text="STREAM_READY", prompt_tokens=3,
                                      generation_tokens=1)
                kwargs["stream_q"].put(("tok", out.text))
                kwargs["stream_q"].put(("done", None))
                return out
            return outputs.pop(0)

        identity = dict(SERVER.SERVER_IDENTITY)
        identity.update({"startup_nonce": "a" * 32, "artifact_id": "b" * 32})
        with mock.patch.object(SERVER, "SERVER_IDENTITY", identity), \
                mock.patch.object(SERVER, "_run", side_effect=fake_run), \
                mock.patch.object(SERVER, "_build_json_lp", return_value=[object()]):
            receipt = SERVER._run_readiness_smoke()
        self.assertEqual("m3-readiness-smoke.v1", receipt["schema"])
        self.assertEqual(["plain-json-envelope", "structured-json", "tool-format", "true-stream"],
                         [test["name"] for test in receipt["tests"]])

    def test_any_failed_smoke_blocks_readiness(self):
        with mock.patch.object(SERVER, "_run",
                               return_value=SimpleNamespace(text="", prompt_tokens=0,
                                                            generation_tokens=0)), \
                self.assertRaises(SERVER.StartupIdentityError):
            SERVER._run_readiness_smoke()


class MachineResourceLeaseTests(unittest.TestCase):
    _manifest = StartupIdentityTests._manifest

    def test_server_lease_blocks_competing_owner_and_releases_cleanly(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            lock = Path(td) / "machine.lock"
            old_token, old_owned = SERVER._MACHINE_RESOURCE_TOKEN, SERVER._MACHINE_RESOURCE_OWNED
            try:
                with mock.patch.dict(os.environ, {"M3_MACHINE_RESOURCE_LOCK": str(lock)}, clear=False):
                    path = SERVER._acquire_machine_resource()
                    self.assertTrue(lock.is_dir())
                    with self.assertRaises(SERVER.StartupIdentityError):
                        SERVER._acquire_machine_resource()
                    SERVER._release_machine_resource(path)
                    self.assertFalse(lock.exists())
            finally:
                SERVER._MACHINE_RESOURCE_TOKEN = old_token
                SERVER._MACHINE_RESOURCE_OWNED = old_owned

    def test_runtime_receipt_parent_must_be_owner_controlled(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root = Path(td); model = root / "model"; model.mkdir()
            unsafe = root / "unsafe"; unsafe.mkdir(mode=0o777); unsafe.chmod(0o777)
            manifest, _raw = self._manifest(td, model)
            identity = SERVER._build_startup_identity(
                str(model), str(manifest), pid=os.getpid(), started_unix=456,
                startup_nonce="0123456789abcdef0123456789abcdef")
            old = SERVER.SERVER_IDENTITY
            try:
                SERVER.SERVER_IDENTITY = identity
                with mock.patch.object(SERVER, "_ARTIFACT_MANIFEST", str(manifest)), \
                        mock.patch.dict(os.environ,
                                        {"M3_RUNTIME_RECEIPT": str(unsafe / "r.json")},
                                        clear=False), \
                        self.assertRaisesRegex(SERVER.StartupIdentityError,
                                                "owner-controlled"):
                    SERVER._runtime_receipt_output()
            finally:
                SERVER.SERVER_IDENTITY = old

    def test_health_and_models_preserve_contract_and_stable_identity(self):
        identity_keys = tuple(SERVER.SERVER_IDENTITY)
        with mock.patch.object(SERVER, "READY", False):
            loading = SERVER._health_payload()
        with mock.patch.object(SERVER, "READY", True):
            ready = SERVER._health_payload()
        self.assertFalse(loading["ready"])
        self.assertTrue(ready["ready"])
        self.assertEqual("list", ready["object"])
        self.assertEqual(SERVER.MODEL_ID, ready["data"][0]["id"])
        for key in identity_keys:
            self.assertEqual(loading[key], ready[key])

        for route in ("/health", "/v1/models"):
            with self.subTest(route=route):
                handler = object.__new__(SERVER.H)
                handler.path = route
                handler._send = mock.Mock()
                with mock.patch.object(SERVER, "READY", True):
                    handler.do_GET()
                status, body = handler._send.call_args.args[:2]
                payload = json.loads(body)
                self.assertEqual(200, status)
                for key, value in SERVER.SERVER_IDENTITY.items():
                    self.assertEqual(value, payload[key])


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
