#!/usr/bin/env bash
# Daily secret-free snapshot of bat-studio configuration -> private Forgejo.
#
# Live credentials are deliberately excluded.  A private Git server is not a
# secrets manager: repository clones, object databases, logs, and future
# mirrors all expand the blast radius of a credential committed here.
#
# Set NOPUSH=1 to commit without pushing.
set -Eeuo pipefail
umask 077

readonly R="$HOME/config-repo"
readonly -a EX=(
    --exclude='.git'
    --include='.env.example'
    --include='.env.sample'
    --include='.env.template'
    --exclude='.env'
    --exclude='.env.*'
    --exclude='*.safetensors'
    --exclude='*.bin'
    --exclude='*.gguf'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.bak*'
    --exclude='*.log'
    --exclude='*venv*'
    --exclude='.cache'
    --exclude='.pytest_cache'
    --exclude='results'
    --exclude='substack'
    --exclude='runs'
    --exclude='*.lock'
    --exclude='.DS_Store'
    --exclude='KILL_SWITCH'
)

die() {
    echo "[config-backup] ERROR: $*" >&2
    exit 1
}

on_error() {
    local rc=$?
    echo "[config-backup] ERROR: command failed at line ${BASH_LINENO[0]} (rc=$rc)" >&2
    exit "$rc"
}
trap on_error ERR

command -v rsync >/dev/null 2>&1 || die "rsync is required"
command -v git >/dev/null 2>&1 || die "git is required"
[[ -d "$R/.git" ]] || die "$R is not an initialized Git repository"

validate_origin() {
    local remote_url remote_urls ssl_verify
    remote_urls=$(
        git -C "$R" remote get-url --all origin
        git -C "$R" remote get-url --push --all origin
    ) || die "origin remote is missing"

    # Never print the URL: a legacy URL may itself contain the leaked token.
    while IFS= read -r remote_url; do
        [[ -n "$remote_url" ]] || continue
        case "$remote_url" in
            http://*)
                die "origin uses plaintext HTTP; replace it with SSH or verified HTTPS"
                ;;
            https://*|ssh://*|git@*:*)
                ;;
            *)
                die "origin must use SSH or verified HTTPS"
                ;;
        esac
        # URI userinfo is forbidden for HTTP(S), where it commonly embeds a
        # password/token. ``ssh://git@host/...`` is normal SSH syntax and the
        # username is not a bearer credential.
        case "$remote_url" in
            http://*@*|https://*@*)
                die "HTTP(S) origin contains userinfo/credentials; use SSH or a credential helper"
                ;;
        esac
    done <<<"$remote_urls"

    ssl_verify=$(git -C "$R" config --bool --get http.sslVerify || true)
    [[ "$ssl_verify" != false ]] || die "Git TLS certificate verification is disabled"

    # Extra HTTP headers are a common place for persistent bearer credentials.
    if git -C "$R" config --local --name-only --get-regexp '^http\..*\.extraheader$' \
        >/dev/null 2>&1; then
        die "repository config contains an HTTP authorization header; remove and rotate it"
    fi
}

remove_live_env_files() {
    # rsync excludes do not delete an already-present excluded destination, so
    # explicitly remove live env files before staging.  Keep documentation-only
    # templates, which must not contain real credentials.
    find "$R" -path "$R/.git" -prune -o -type f -name '.env*' \
        ! -name '.env.example' ! -name '.env.sample' ! -name '.env.template' \
        -exec rm -f -- {} +
}

assert_no_live_env_on_disk() {
    local remaining
    remaining=$(find "$R" -path "$R/.git" -prune -o -type f -name '.env*' \
        ! -name '.env.example' ! -name '.env.sample' ! -name '.env.template' \
        -print -quit)
    [[ -z "$remaining" ]] || die "a live .env file remains in the backup worktree"
}

assert_no_tracked_live_env() {
    local path base
    while IFS= read -r -d '' path; do
        base=${path##*/}
        case "$base" in
            .env.example|.env.sample|.env.template)
                ;;
            .env|.env.*)
                die "a live .env path remains tracked; refusing to commit"
                ;;
        esac
    done < <(git -C "$R" ls-files -z)
}

validate_origin

# Existing object files in this backup repo previously had group-readable
# modes.  Restrict both the existing object database and everything Git creates
# during this run.
chmod -R go-rwx "$R/.git"

mkdir -p "$R/hermes" "$R/exitmgr-app" "$R/gordon-gauntlet" \
    "$R/serving" "$R/launch-agents" "$R/longcall-manager" \
    "$R/pipeline-lifecycle"
rsync -a --delete "${EX[@]}" "$HOME/.hermes/skills" "$R/hermes/"
rsync -a "${EX[@]}" "$HOME/.hermes/SOUL.md" "$HOME/.hermes/config.yaml" "$R/hermes/"
rsync -a --delete "${EX[@]}" "$HOME/exitmgr-app/exitmgr" "$HOME/exitmgr-app/tests" "$R/exitmgr-app/"
rsync -a --delete "${EX[@]}" "$HOME/exitmgr-app/data" "$R/exitmgr-app/"
rsync -a "${EX[@]}" "$HOME"/exitmgr-app/*.py "$HOME/exitmgr-app/config.yaml" \
    "$HOME/exitmgr-app/README.md" "$R/exitmgr-app/"
rsync -a "${EX[@]}" "$HOME/m3_serve.py" "$R/"
rsync -a "${EX[@]}" "$HOME/m3_serve_batched.py" "$HOME/m3_batch_core.py" \
    "$HOME/m3_lan_proxy.py" "$HOME/machine_resource_lease.py" "$R/serving/"
rsync -a "${EX[@]}" \
    "$HOME/pipeline-automation/build_gate_promote_abliterated_m3.sh" \
    "$HOME/pipeline-automation/eval_and_score.sh" \
    "$HOME/pipeline-automation/evaluate_m3_supervised.sh" \
    "$HOME/pipeline-automation/fuse_promote_m3.sh" \
    "$HOME/pipeline-automation/launch_bound_training.sh" \
    "$HOME/pipeline-automation/lib_pipeline.sh" \
    "$HOME/pipeline-automation/m3v2_done_watch.sh" \
    "$HOME/pipeline-automation/machine_resource_lease.py" \
    "$HOME/pipeline-automation/onready_v1.sh" \
    "$HOME/pipeline-automation/prepare_next_corpus.py" \
    "$HOME/pipeline-automation/training_run_controller.py" \
    "$R/pipeline-lifecycle/"
rsync -a "${EX[@]}" \
    "$HOME/Library/LaunchAgents/ai.alfred.m3-prod.plist" \
    "$HOME/Library/LaunchAgents/ai.alfred.m3-lan-proxy.plist" \
    "$HOME/Library/LaunchAgents/ai.alfred.status-api.plist" \
    "$HOME/Library/LaunchAgents/ai.alfred.config-backup.plist" \
    "$HOME/Library/LaunchAgents/ai.alfred.k3s-backup.plist" \
    "$R/launch-agents/"
rsync -a --delete "${EX[@]}" "$HOME/longcall-manager/" "$R/longcall-manager/"
rsync -a --delete "${EX[@]}" "$HOME/scripts" "$R/"
rsync -a --delete "${EX[@]}" "$HOME/gordon-gauntlet/batteries" "$R/gordon-gauntlet/"
rsync -a "${EX[@]}" "$HOME"/gordon-gauntlet/*.py "$HOME"/gordon-gauntlet/*.sh \
    "$HOME/gordon-gauntlet/README.md" "$R/gordon-gauntlet/"

remove_live_env_files
assert_no_live_env_on_disk

git -C "$R" add -A
assert_no_tracked_live_env
git -C "$R" diff --cached --check

committed=0
if git -C "$R" diff --cached --quiet; then
    echo "[config-backup] no config changes"
else
    git -C "$R" \
        -c user.email='alfred@batcloud.local' \
        -c user.name='bat-studio config backup' \
        commit -q -m "config snapshot $(date '+%F %T %Z')"
    committed=1
fi

if [[ ${NOPUSH:-0} == 1 ]]; then
    if (( committed )); then
        echo "[config-backup] committed (NOPUSH=1, not pushed)"
    else
        echo "[config-backup] NOPUSH=1, not pushed"
    fi
    exit 0
fi

# Do this even when there was no new commit: a prior failed push must be
# retried, not converted into a false-success heartbeat on the next run.
git -C "$R" push --quiet origin main
chmod -R go-rwx "$R/.git"
echo "[config-backup] snapshot committed and/or pending commits pushed"
