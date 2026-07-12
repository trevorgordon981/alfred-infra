#!/usr/bin/env python3
"""CPU-only tests for fixed, authenticated request-scoped LoRA profiles."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _support import load_server_module


SERVER = load_server_module()
TOKEN = "fixed-profile-owner-token-0123456789abcdef"


def headers(*pairs):
    result = Message()
    for key, value in pairs:
        result[key] = value
    return result


def handler(body=b"{}", *header_pairs):
    result = object.__new__(SERVER.H)
    result.path = "/v1/chat/completions"
    result.headers = headers(
        ("Content-Length", str(len(body))), *header_pairs
    )
    result.rfile = io.BytesIO(body)
    result.wfile = io.BytesIO()
    result.close_connection = False
    result._send = mock.Mock()
    return result


class ReceiptFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir(mode=0o700)
        self.adapter_config = self.adapter / "adapter_config.json"
        self.adapter_weights = self.adapter / "adapters.safetensors"
        self.adapter_config.write_bytes(b'{"rank":8}\n')
        self.adapter_weights.write_bytes(b"unit-test-adapter-weights")
        self.adapter_config.chmod(0o600)
        self.adapter_weights.chmod(0o600)
        self.profile_config = self.root / "profiles.json"
        self.token = self.root / "profile.token"
        self.token.write_text(TOKEN + "\n", encoding="ascii")
        self.token.chmod(0o600)
        self.write_profile_config()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def profile_value(self):
        return {
            "schema": "m3-lora-profiles.v1",
            "adapter": {
                "path": str(self.adapter),
                "config_sha256": self.sha256(self.adapter_config),
                "weights_sha256": self.sha256(self.adapter_weights),
                "weights_bytes": self.adapter_weights.stat().st_size,
            },
            "default_profile": "general",
            "profiles": {"general": 0.0, "trader": 1.0},
        }

    def write_profile_config(self, value=None, *, canonical=True):
        value = self.profile_value() if value is None else value
        if canonical:
            raw = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        else:
            raw = json.dumps(value, indent=2) + "\n"
        self.profile_config.write_text(raw, encoding="ascii")
        self.profile_config.chmod(0o600)


class FixedProfileReceiptTests(ReceiptFixture):
    def test_receipt_binds_exact_adapter_bytes_and_fixed_profiles(self):
        loaded = SERVER._load_lora_profile_config(str(self.profile_config))
        self.assertEqual(str(self.adapter), loaded["adapter"]["path"])
        self.assertEqual(
            {"general": 0.0, "trader": 1.0}, loaded["profiles"]
        )
        with mock.patch.object(SERVER, "LORA_PROFILES_ENABLED", True), mock.patch.object(
            SERVER, "LORA_PROFILE_CONFIG", loaded
        ), mock.patch.object(
            SERVER, "LORA_ADAPTER_DIR", str(self.adapter)
        ), mock.patch.object(
            SERVER, "SERVER_IDENTITY", {}
        ):
            identity = SERVER._verify_lora_adapter_binding()
        self.assertEqual(self.sha256(self.adapter_weights), identity["weights_sha256"])

    def test_receipt_rejects_noncanonical_or_nonfixed_profiles_and_tampering(self):
        value = self.profile_value()
        self.write_profile_config(value, canonical=False)
        with self.assertRaises(SERVER.StartupIdentityError):
            SERVER._load_lora_profile_config(str(self.profile_config))

        value["profiles"]["trader"] = 0.5
        self.write_profile_config(value)
        with self.assertRaises(SERVER.StartupIdentityError):
            SERVER._load_lora_profile_config(str(self.profile_config))

        self.write_profile_config()
        loaded = SERVER._load_lora_profile_config(str(self.profile_config))
        self.profile_config.write_text("{}\n", encoding="ascii")
        with mock.patch.object(SERVER, "LORA_PROFILES_ENABLED", True), mock.patch.object(
            SERVER, "LORA_PROFILE_CONFIG", loaded
        ), mock.patch.object(SERVER, "LORA_ADAPTER_DIR", str(self.adapter)):
            with self.assertRaises(SERVER.StartupIdentityError):
                SERVER._verify_lora_adapter_binding()

        self.write_profile_config()
        loaded = SERVER._load_lora_profile_config(str(self.profile_config))
        self.adapter_weights.write_bytes(b"tampered")
        with mock.patch.object(SERVER, "LORA_PROFILES_ENABLED", True), mock.patch.object(
            SERVER, "LORA_PROFILE_CONFIG", loaded
        ), mock.patch.object(SERVER, "LORA_ADAPTER_DIR", str(self.adapter)):
            with self.assertRaises(SERVER.StartupIdentityError):
                SERVER._verify_lora_adapter_binding()


class RequestProfileTests(ReceiptFixture):
    def profile_patches(self):
        return mock.patch.multiple(
            SERVER,
            LORA_PROFILES_ENABLED=True,
            LORA_DEFAULT_PROFILE="general",
            LORA_PROFILE_TOKEN_FILE=str(self.token),
        )

    def test_missing_header_is_general_and_every_explicit_profile_is_authenticated(self):
        with self.profile_patches():
            self.assertEqual("general", SERVER._request_lora_profile(headers()))
            for profile in ("general", "trader"):
                self.assertEqual(
                    profile,
                    SERVER._request_lora_profile(
                        headers(
                            ("X-M3-LoRA-Profile", profile),
                            ("X-M3-LoRA-Profile-Token", TOKEN),
                        )
                    ),
                )
            for hdrs in (
                headers(("X-M3-LoRA-Profile", "trader")),
                headers(
                    ("X-M3-LoRA-Profile", "trader"),
                    ("X-M3-LoRA-Profile-Token", "wrong"),
                ),
                headers(
                    ("X-M3-LoRA-Profile", "arbitrary"),
                    ("X-M3-LoRA-Profile-Token", TOKEN),
                ),
                headers(("X-M3-LoRA-Profile-Token", TOKEN)),
                headers(
                    ("X-M3-LoRA-Profile", "general"),
                    ("X-M3-LoRA-Profile", "trader"),
                    ("X-M3-LoRA-Profile-Token", TOKEN),
                ),
            ):
                with self.subTest(headers=list(hdrs.items())), self.assertRaises(
                    SERVER.ProfileAuthorizationError
                ):
                    SERVER._request_lora_profile(hdrs)

    def test_disabled_mode_rejects_explicit_profile_and_keeps_fused_default(self):
        with mock.patch.object(SERVER, "LORA_PROFILES_ENABLED", False), mock.patch.object(
            SERVER, "LORA_DEFAULT_PROFILE", "fused"
        ):
            self.assertEqual("fused", SERVER._request_lora_profile(headers()))
            with self.assertRaises(SERVER.ProfileAuthorizationError):
                SERVER._request_lora_profile(
                    headers(
                        ("X-M3-LoRA-Profile", "general"),
                        ("X-M3-LoRA-Profile-Token", TOKEN),
                    )
                )

    def test_unauthorized_http_request_is_403_and_never_enqueues(self):
        body = json.dumps({"messages": []}).encode()
        request = handler(body, ("X-M3-LoRA-Profile", "trader"))
        with self.profile_patches(), mock.patch.object(
            SERVER, "READY", True
        ), mock.patch.object(SERVER, "submit_and_wait") as generation:
            request.do_POST()
        self.assertEqual(403, request._send.call_args.args[0])
        generation.assert_not_called()

    def test_body_cannot_override_header_gate(self):
        body = json.dumps({"messages": [], "m3_profile": "trader"}).encode()
        request = handler(body)
        with self.profile_patches(), mock.patch.object(
            SERVER, "READY", True
        ), mock.patch.object(SERVER, "submit_and_wait") as generation:
            request.do_POST()
        self.assertEqual(400, request._send.call_args.args[0])
        generation.assert_not_called()


class ScaleAndIsolationTests(unittest.TestCase):
    def profile_patches(self, layers):
        return mock.patch.multiple(
            SERVER,
            LORA_PROFILES_ENABLED=True,
            LORA_DEFAULT_PROFILE="general",
            _LORA_LAYERS=layers,
        )

    def test_scale_is_changed_only_under_generation_lock_and_restored_on_failure(self):
        layer = SimpleNamespace(scale=0.0, lora_a=object(), lora_b=object())
        fake_mx = SimpleNamespace(synchronize=mock.Mock())
        layers = [("layer", layer, 2.5)]
        with self.profile_patches(layers), mock.patch.object(SERVER, "mx", fake_mx):
            with self.assertRaises(RuntimeError):
                with SERVER._lora_profile_scope("trader"):
                    pass
            with self.assertRaisesRegex(ValueError, "synthetic"):
                with SERVER.GEN_LOCK:
                    with SERVER._lora_profile_scope("trader"):
                        self.assertEqual(2.5, layer.scale)
                        raise ValueError("synthetic")
            self.assertEqual(0.0, layer.scale)
            fake_mx.synchronize.assert_called_once_with()

    def test_apc_tenants_are_profile_specific(self):
        chunk = SimpleNamespace(
            text="ok", prompt_tokens=1, generation_tokens=1, finish_reason="stop"
        )
        fake_mx = SimpleNamespace(synchronize=mock.Mock())
        tenants = {"general": "tenant-general", "trader": "tenant-trader"}
        observed = []

        def stream(*_args, **kwargs):
            observed.append(kwargs["apc_tenant"])
            return iter([chunk])

        with mock.patch.multiple(
            SERVER,
            LORA_PROFILES_ENABLED=True,
            LORA_DEFAULT_PROFILE="general",
            LORA_PROFILE_CACHE_TENANTS=tenants,
            APC=object(),
        ), mock.patch.object(SERVER, "stream_generate", side_effect=stream), mock.patch.object(
            SERVER, "mx", fake_mx
        ):
            SERVER._gen("prompt", 2, 0.0, True, profile="general")
            SERVER._gen("prompt", 2, 0.0, True, profile="trader")
        self.assertEqual(["tenant-general", "tenant-trader"], observed)

    def test_different_profiles_can_never_share_one_batch(self):
        with mock.patch.multiple(
            SERVER, LORA_PROFILES_ENABLED=True, LORA_DEFAULT_PROFILE="general"
        ):
            general = SERVER._Job([], 1, 0.0, "disabled", None, None, profile="general")
            trader = SERVER._Job([], 1, 0.0, "disabled", None, None, profile="trader")
            with self.assertRaisesRegex(RuntimeError, "different LoRA profiles"):
                SERVER._do_batched([general, trader], [[], []])


class LoaderContractTests(unittest.TestCase):
    def test_loader_passes_explicit_trust_and_bound_adapter(self):
        model = SimpleNamespace()
        processor = SimpleNamespace()
        config = {"eos_token_id": 7}
        with mock.patch.multiple(
            SERVER,
            LORA_PROFILES_ENABLED=True,
            LORA_ADAPTER_DIR="/fixture/adapter",
        ), mock.patch.object(
            SERVER, "load", return_value=(model, processor)
        ) as load_model, mock.patch.object(
            SERVER, "load_config", return_value=config
        ) as load_config, mock.patch.object(
            SERVER, "_reprove_lora_adapter_stability"
        ), mock.patch.object(
            SERVER, "_initialize_lora_layers"
        ), mock.patch.object(
            SERVER, "_run_readiness_smoke", return_value={"schema": "m3-readiness-smoke.v1"}
        ), mock.patch.object(SERVER, "_initialize_runtime_identity"):
            SERVER._load()
        load_model.assert_called_once_with(
            SERVER.MD, trust_remote_code=True, adapter_path="/fixture/adapter"
        )
        load_config.assert_called_once_with(SERVER.MD, trust_remote_code=True)


if __name__ == "__main__":
    unittest.main()
