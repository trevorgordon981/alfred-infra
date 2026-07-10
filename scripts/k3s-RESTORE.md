# k3s backup restore runbook

These backups deliberately separate portable, non-secret resources from
encrypted recovery material. Treat every decrypted file as a live credential.

## Before the first backup

1. Install `age` on bat-studio.
2. Generate an age identity on a separate, offline recovery system. Keep at
   least two protected copies of that identity; do not place it on bat-studio,
   the NAS, Forgejo, or the k3s cluster.
3. Put only the corresponding public recipient(s), one per line, at
   `~/.config/backup/k3s-age-recipients.txt` on bat-studio. The path can instead
   be supplied through `K3S_BACKUP_AGE_RECIPIENTS`.
4. Run a restore drill on an isolated host. A backup is not considered healthy
   until the age files decrypt and the SQLite integrity check passes.

The backup job fails closed if `age`, the recipient file, or this runbook is
missing. It never falls back to a plaintext copy.

## Restore order

Use an isolated recovery machine with `umask 077` and sufficient encrypted
temporary storage. Never paste decrypted data into a terminal, ticket, or log.

1. Decrypt `k3s-state.db.<timestamp>.age` with the offline age identity.
2. Run `sqlite3 <decrypted-file> 'PRAGMA integrity_check;'` and require `ok`.
3. Install the same k3s version and configuration used by the failed control
   plane. Stop k3s before replacing its SQLite database. Preserve the failed
   host's database separately, install the verified replacement at
   `/var/lib/rancher/k3s/server/db/state.db` with root ownership and mode 0600,
   then restart k3s. Do not overwrite a running datastore.
4. Once the API server is healthy, decrypt
   `sealedsecrets-controller-keys.<timestamp>.yaml.age` and apply it to the
   cluster before starting/restarting the sealed-secrets controller. Confirm
   the key Secret exists, restart the controller, and verify it can unseal a
   known test SealedSecret.
5. Apply `sealedsecrets.<timestamp>.yaml`, then
   `argocd-resources.no-secrets.<timestamp>.yaml` after their CRDs/controllers
   are installed.
6. Delete decrypted working files as soon as the drill/restore is complete.
   On SSDs and snapshots, ordinary deletion is not guaranteed secure; use an
   encrypted ephemeral volume and destroy its key.

## Legacy plaintext cleanup (mandatory once this version is deployed)

Older generations named `k3s-state.db.*`, `argocd-state.*.yaml`, and
`sealedsecrets-controller-keys.*.yaml` contain Kubernetes credentials in
plaintext/base64 form. Locate them on bat-studio and the NAS without printing
their contents. Encrypt any generation that must be retained with the public
age recipient, verify decryption on the offline recovery system, and then remove
the plaintext originals and any snapshots/copies that contain them.

Rotate credentials represented in those old backups if either storage location
was accessible beyond the intended administrator account. Base64-encoded Secret
fields are plaintext for incident-response purposes.

## Restore-drill record

2026-07-09: all 24 migrated legacy age envelopes and both current encrypted
artifacts decrypted successfully using the off-Studio identity. The newest
`state.db` passed `PRAGMA integrity_check` (`ok`), the controller-key manifest
contained the expected Sealed Secrets key label, and the NAS copies of the
current database/key envelopes matched Studio byte-for-byte by SHA-256. All 21
local and 24 NAS plaintext legacy files were then removed. The subsequent live
backup completed through `backup-guard` with a healthy `0600` heartbeat.

Outstanding physical DR step: place a second protected copy of the decrypting
identity in the offline password/recovery vault. The working off-Studio copy is
mode `0600`; no identity is stored on Studio, the NAS, Forgejo, or Kubernetes.
