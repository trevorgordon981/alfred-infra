#!/usr/bin/env python3
"""Dependency-free regression tests for backup and status-API invariants."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def load_status_api():
    path = SCRIPTS / "alfred-status-api.py"
    spec = importlib.util.spec_from_file_location("alfred_status_api_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StatusApiAuthTests(unittest.TestCase):
    TOKEN = "synthetic-status-token-0123456789abcdef"

    @classmethod
    def setUpClass(cls):
        cls.api = load_status_api()

    def test_env_token_wins_when_fallback_file_is_missing(self):
        missing = Path(tempfile.gettempdir()) / "definitely-not-a-real-api-token"
        self.assertEqual(
            self.TOKEN,
            self.api.load_token(
                {"ALFRED_STATUS_API_TOKEN": self.TOKEN}, missing),
        )

    def test_no_token_means_denied(self):
        self.api.TOKEN = ""
        handler = object.__new__(self.api.Handler)
        handler.headers = {}
        self.assertFalse(handler._authed())

    def test_constant_time_bearer_contract(self):
        self.api.TOKEN = self.TOKEN
        handler = object.__new__(self.api.Handler)
        handler.headers = {"Authorization": f"Bearer {self.TOKEN}"}
        self.assertTrue(handler._authed())
        handler.headers = {"Authorization": "Bearer wrong-token"}
        self.assertFalse(handler._authed())

    def test_secure_fallback_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text(self.TOKEN + "\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertEqual(self.TOKEN, self.api.load_token({}, path))

    def test_group_readable_fallback_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text(self.TOKEN + "\n", encoding="utf-8")
            path.chmod(0o640)
            with self.assertRaises(PermissionError):
                self.api.load_token({}, path)

    def test_symlink_and_weak_tokens_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real-token"
            target.write_text(self.TOKEN + "\n", encoding="utf-8")
            target.chmod(0o600)
            linked = root / "linked-token"
            linked.symlink_to(target)
            with self.assertRaises(PermissionError):
                self.api.load_token({}, linked)
            with self.assertRaises(PermissionError):
                self.api.load_token({"ALFRED_STATUS_API_TOKEN": "too-short"})

    def test_public_errors_are_redacted_and_child_does_not_inherit_token(self):
        source = (SCRIPTS / "alfred-status-api.py").read_text(encoding="utf-8")
        self.assertNotIn('"stderr"', source)
        self.assertIn('"ALFRED_STATUS_API_TOKEN_FILE"', source)
        self.assertIn("MAX_CLI_JSON_BYTES", source)
        self.assertIn("parse_constant=", source)
        self.assertIn("except OSError:", source)


class BackupScriptInvariantTests(unittest.TestCase):
    def test_config_backup_is_fail_closed_and_omits_live_env(self):
        text = (SCRIPTS / "config-backup.sh").read_text(encoding="utf-8")
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("validate_origin", text)
        self.assertIn('git -C "$R" push --quiet', text)
        self.assertIn('"$HOME/exitmgr-app/README.md"', text)
        self.assertIn('"$HOME/m3_serve_batched.py"', text)
        self.assertIn('"$HOME/m3_lan_proxy.py"', text)
        self.assertIn('"$HOME/longcall-manager/"', text)
        self.assertIn('ai.alfred.m3-prod.plist', text)
        self.assertNotIn('"$HOME/.hermes/.env"', text)
        for line in text.splitlines():
            if line.lstrip().startswith("rsync "):
                self.assertNotIn("2>/dev/null", line)

    def test_ssh_origin_is_allowed_but_http_userinfo_is_rejected(self):
        text = (SCRIPTS / "config-backup.sh").read_text(encoding="utf-8")
        self.assertIn("https://*|ssh://*|git@*:*)", text)
        self.assertIn("http://*@*|https://*@*)", text)
        self.assertNotIn("*://*@*)", text)

    def test_k3s_secret_material_is_encrypted_or_omitted(self):
        text = (SCRIPTS / "k3s-backup.sh").read_text(encoding="utf-8")
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("umask 077", text)
        self.assertIn("k3s-state.db.$TS.age", text)
        self.assertIn("sealedsecrets-controller-keys.$TS.yaml.age", text)
        self.assertIn("argocd-resources.no-secrets.$TS.yaml", text)
        self.assertNotIn(
            "applications.argoproj.io,appprojects.argoproj.io,secrets", text)

    def test_backup_heartbeat_only_advances_after_success(self):
        text = (SCRIPTS / "backup-guard.sh").read_text(encoding="utf-8")
        success = text.index('if (( rc == 0 )); then')
        heartbeat = text.index('"$HB_DIR/$LABEL.ok"')
        failure = text.index('"$HB_DIR/$LABEL.fail"')
        self.assertLess(success, heartbeat)
        self.assertLess(heartbeat, failure)
        self.assertIn('exit "$rc"', text)

    def test_gitignore_blocks_secret_and_backup_outputs(self):
        lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        for required in (".env", ".env.*", "*.key", "*.pem", "*.age",
                         "k3s-backups/", "backup-heartbeats/"):
            self.assertIn(required, lines)


if __name__ == "__main__":
    unittest.main()
