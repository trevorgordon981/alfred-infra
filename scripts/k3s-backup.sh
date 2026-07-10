#!/usr/bin/env bash
# Secret-safe k3s disaster-recovery backup.
#
# The cluster database and sealed-secrets controller keys contain plaintext
# secret material.  They are streamed directly into age encryption and are
# never written to local or NAS storage in plaintext.  Argo CD Secret objects
# are omitted; SealedSecret CRs are the portable source of truth.
set -Eeuo pipefail
umask 077

readonly DEST="${K3S_BACKUP_DEST:-$HOME/k3s-backups}"
readonly LOG="${K3S_BACKUP_LOG:-$HOME/Library/Logs/k3s-backup.log}"
readonly TS="$(date -u +%Y%m%dT%H%M%SZ)"
readonly K="${KUBECTL_BIN:-/opt/homebrew/bin/kubectl}"
readonly AGE="${AGE_BIN:-/opt/homebrew/bin/age}"
readonly RSYNC="${RSYNC_BIN:-/opt/homebrew/bin/rsync}"
readonly RECIPIENTS="${K3S_BACKUP_AGE_RECIPIENTS:-$HOME/.config/backup/k3s-age-recipients.txt}"
readonly SPARK="${K3S_BACKUP_SPARK:-node3@100.107.197.68}"
readonly NAS="${K3S_BACKUP_NAS:-trevorbg@192.168.68.62}"
readonly NAS_DEST="${K3S_BACKUP_NAS_DEST:-/volume1/Backups/k3s}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly RESTORE_RUNBOOK="$SCRIPT_DIR/k3s-RESTORE.md"
readonly -a SSH=(ssh -o ConnectTimeout=15 -o BatchMode=yes)

mkdir -p "$DEST" "$(dirname -- "$LOG")"
chmod 700 "$DEST"
find "$DEST" -maxdepth 1 -type f -exec chmod 600 {} +
touch "$LOG"
chmod 600 "$LOG"

log() {
    echo "[$(date '+%F %T')] $*" >>"$LOG"
}

fail=0
tmp_files=()

cleanup() {
    local file
    for file in "${tmp_files[@]}"; do
        [[ -n "$file" ]] && rm -f -- "$file"
    done
}
trap cleanup EXIT

require_executable() {
    [[ -x "$1" ]] || {
        log "ERROR: required executable is unavailable: $1"
        exit 1
    }
}

prune_local() {
    local prefix=$1 suffix=$2 keep=$3 listing file index=0
    local -a files
    shopt -s nullglob
    files=("$DEST"/"$prefix".*"$suffix")
    shopt -u nullglob
    ((${#files[@]} <= keep)) && return 0
    listing=$(/bin/ls -1t -- "${files[@]}") || return 1
    while IFS= read -r file; do
        ((index += 1))
        ((index <= keep)) || rm -f -- "$file"
    done <<<"$listing"
}

log "=== backup $TS ==="

# Refuse to declare the backup set healthy while known plaintext generations
# remain. Names are checked without reading or printing file contents.
legacy_local=$(find "$DEST" -maxdepth 1 -type f \
    \( \( -name 'k3s-state.db.*' ! -name '*.age' \) \
       -o -name 'argocd-state.*.yaml' \
       -o \( -name 'sealedsecrets-controller-keys.*.yaml' ! -name '*.age' \) \) \
    -print -quit)
if [[ -n "$legacy_local" ]]; then
    log "ERROR: legacy plaintext Kubernetes backups remain locally; follow k3s-RESTORE.md"
    fail=1
fi
unset legacy_local

require_executable "$K"
require_executable "$AGE"
require_executable "$RSYNC"
[[ -s "$RECIPIENTS" ]] || {
    log "ERROR: age recipient file is missing/empty: $RECIPIENTS"
    exit 1
}
[[ -r "$RESTORE_RUNBOOK" ]] || {
    log "ERROR: restore runbook is missing: $RESTORE_RUNBOOK"
    exit 1
}

# 1. Atomically copy the k3s SQLite database on the control plane and stream it
#    straight into age.  There is no raw-cat fallback: an inconsistent database
#    is worse than an honest failed backup.
state_out="$DEST/k3s-state.db.$TS.age"
state_tmp="$state_out.tmp.$$"
tmp_files+=("$state_tmp")
remote_dump='command -v sqlite3 >/dev/null 2>&1 || exit 127; t=$(sudo mktemp /tmp/k3s-state.db.XXXXXX) || exit 1; sudo sqlite3 /var/lib/rancher/k3s/server/db/state.db ".backup $t" >/dev/null; rc=$?; if [ "$rc" -eq 0 ]; then sudo cat "$t"; rc=$?; fi; sudo rm -f "$t"; exit "$rc"'
if "${SSH[@]}" "$SPARK" "$remote_dump" 2>>"$LOG" \
    | "$AGE" -R "$RECIPIENTS" -o "$state_tmp" 2>>"$LOG" \
    && [[ -s "$state_tmp" ]]; then
    chmod 600 "$state_tmp"
    mv -f -- "$state_tmp" "$state_out"
    log "encrypted state.db copied ($(du -h "$state_out" | cut -f1))"
else
    log "ERROR: encrypted state.db backup failed/empty"
    rm -f -- "$state_tmp"
    fail=1
fi

# 2. Back up only non-Secret Argo CD resources.  The old dump included Secret
#    objects (base64 is encoding, not encryption) and must be remediated using
#    the legacy-cleanup section of the restore runbook.
argocd_out="$DEST/argocd-resources.no-secrets.$TS.yaml"
argocd_tmp="$argocd_out.tmp.$$"
tmp_files+=("$argocd_tmp")
if "$K" -n argocd get \
    applications.argoproj.io,appprojects.argoproj.io,configmaps \
    -o yaml >"$argocd_tmp" 2>>"$LOG" \
    && [[ -s "$argocd_tmp" ]]; then
    chmod 600 "$argocd_tmp"
    mv -f -- "$argocd_tmp" "$argocd_out"
    log "Argo CD non-Secret resources dumped"
else
    log "ERROR: Argo CD non-Secret resource dump failed/empty"
    rm -f -- "$argocd_tmp"
    fail=1
fi

# 3. SealedSecret CRs contain ciphertext and are safe to store, but retain
#    private file modes because names/metadata can still be sensitive.
sealed_out="$DEST/sealedsecrets.$TS.yaml"
sealed_tmp="$sealed_out.tmp.$$"
tmp_files+=("$sealed_tmp")
if "$K" get sealedsecrets -A -o yaml >"$sealed_tmp" 2>>"$LOG" \
    && [[ -s "$sealed_tmp" ]]; then
    chmod 600 "$sealed_tmp"
    mv -f -- "$sealed_tmp" "$sealed_out"
    log "SealedSecret CRs dumped"
else
    log "ERROR: SealedSecret CR dump failed/empty"
    rm -f -- "$sealed_tmp"
    fail=1
fi

# 4. The controller's decryption keys are required to recover the SealedSecret
#    CRs.  Verify a key exists, then stream the YAML directly into age.
keys_out="$DEST/sealedsecrets-controller-keys.$TS.yaml.age"
keys_tmp="$keys_out.tmp.$$"
tmp_files+=("$keys_tmp")
if key_names=$("$K" -n kube-system get secret \
    -l sealedsecrets.bitnami.com/sealed-secrets-key -o name 2>>"$LOG") \
    && [[ -n "$key_names" ]]; then
    unset key_names
    if "$K" -n kube-system get secret \
        -l sealedsecrets.bitnami.com/sealed-secrets-key -o yaml 2>>"$LOG" \
        | "$AGE" -R "$RECIPIENTS" -o "$keys_tmp" 2>>"$LOG" \
        && [[ -s "$keys_tmp" ]]; then
        chmod 600 "$keys_tmp"
        mv -f -- "$keys_tmp" "$keys_out"
        log "sealed-secrets controller keys encrypted"
    else
        log "ERROR: sealed-secrets controller key encryption failed/empty"
        rm -f -- "$keys_tmp"
        fail=1
    fi
else
    unset key_names
    log "ERROR: no sealed-secrets controller key was found"
    fail=1
fi

# Keep seven local generations of each safe artifact.  Plaintext legacy names
# are intentionally not matched or deleted automatically; follow the runbook so
# irreplaceable recovery data is encrypted before removal.
prune_local k3s-state.db .age 7 || fail=1
prune_local argocd-resources.no-secrets .yaml 7 || fail=1
prune_local sealedsecrets .yaml 7 || fail=1
prune_local sealedsecrets-controller-keys .yaml.age 7 || fail=1

# 5. Push only this run's encrypted/sanitized artifacts off-box.  Every remote
#    file is forced to mode 0600 and the backup directory to 0700.
if "${SSH[@]}" "$NAS" "umask 077; mkdir -p '$NAS_DEST' && chmod 700 '$NAS_DEST'" \
    2>>"$LOG"; then
    nas_legacy_rc=0
    "${SSH[@]}" "$NAS" \
        "legacy=\$(find '$NAS_DEST' -maxdepth 1 -type f \
            \\( \\( -name 'k3s-state.db.*' ! -name '*.age' \\) \
               -o -name 'argocd-state.*.yaml' \
               -o \\( -name 'sealedsecrets-controller-keys.*.yaml' ! -name '*.age' \\) \\) \
            -print -quit) || exit 43; [ -z \"\$legacy\" ] || exit 42" \
        2>>"$LOG" || nas_legacy_rc=$?
    if (( nas_legacy_rc == 42 )); then
        log "ERROR: legacy plaintext Kubernetes backups remain on NAS; follow RESTORE.md"
        fail=1
    elif (( nas_legacy_rc != 0 )); then
        log "ERROR: could not verify NAS for legacy plaintext backups"
        fail=1
    fi

    for file in "$state_out" "$argocd_out" "$sealed_out" "$keys_out"; do
        [[ -f "$file" ]] || continue
        if "$RSYNC" -a --chmod=F600 \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
            "$file" "$NAS:$NAS_DEST/" 2>>"$LOG"; then
            log "NAS <- $(basename -- "$file")"
        else
            log "ERROR: NAS push failed for $(basename -- "$file")"
            fail=1
        fi
    done

    if "$RSYNC" -a --chmod=F600 \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
        "$RESTORE_RUNBOOK" "$NAS:$NAS_DEST/RESTORE.md" 2>>"$LOG"; then
        log "NAS <- RESTORE.md"
    else
        log "ERROR: NAS push failed for RESTORE.md"
        fail=1
    fi

    if "${SSH[@]}" "$NAS" \
        "cd '$NAS_DEST' && for p in 'k3s-state.db.*.age' 'argocd-resources.no-secrets.*.yaml' 'sealedsecrets.*.yaml' 'sealedsecrets-controller-keys.*.yaml.age'; do ls -1t \$p 2>/dev/null | tail -n +15 | xargs -r rm -f; done && chmod 700 . && find . -maxdepth 1 -type f -exec chmod 600 {} +" \
        2>>"$LOG"; then
        log "NAS retention and permissions applied"
    else
        log "ERROR: NAS retention/permission update failed"
        fail=1
    fi
else
    log "ERROR: NAS unreachable; local encrypted copies kept, NAS push skipped"
    fail=1
fi

sizes=$(du -sh "$DEST"/*"$TS"* 2>/dev/null | tr '\n' ' ' || true)
log "sizes: ${sizes:-none}"
if (( fail != 0 )); then
    log "=== COMPLETED WITH ERRORS (fail=$fail) ==="
    exit 1
fi
log "=== OK ==="
exit 0
