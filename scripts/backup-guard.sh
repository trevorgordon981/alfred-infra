#!/usr/bin/env bash
# backup-guard.sh <label> <command...>
# Preserve the wrapped job's exit status, record a heartbeat only on genuine
# success, and alert on failure. Backup commands must never mask their own
# errors; this wrapper treats their status as authoritative.
set -Eeuo pipefail
umask 077

LABEL=${1:-}
shift || true
if [[ -z "$LABEL" || $# -eq 0 ]]; then
    echo "usage: backup-guard.sh <label> <command...>" >&2
    exit 2
fi
if [[ ! "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "backup-guard: label may contain only letters, digits, dot, dash, and underscore" >&2
    exit 2
fi

ENV_FILE=${BACKUP_GUARD_ENV_FILE:-/Users/alfredpennyworth/.hermes/.env}
if [[ -r "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090 -- deployment-specific credential file
    source "$ENV_FILE"
fi

SLACK_CHANNEL=${SLACK_CHANNEL:-C0ADZGK58PL} # #error-logs
HB_DIR=${BACKUP_HEARTBEAT_DIR:-/Users/alfredpennyworth/.local/var/backup-heartbeats}
mkdir -p "$HB_DIR"
chmod 700 "$HB_DIR"

OUT=$(mktemp -t bkguard.XXXXXX)
chmod 600 "$OUT"
cleanup() { rm -f -- "$OUT"; }
trap cleanup EXIT

START=$(date -Iseconds)
if "$@" >"$OUT" 2>&1; then
    rc=0
else
    rc=$?
fi
END=$(date -Iseconds)

if (( rc == 0 )); then
    printf '%s rc=0\n' "$END" >"$HB_DIR/$LABEL.ok"
    chmod 600 "$HB_DIR/$LABEL.ok"
    rm -f -- "$HB_DIR/$LABEL.fail"
else
    printf '%s rc=%d\n' "$END" "$rc" >"$HB_DIR/$LABEL.fail"
    chmod 600 "$HB_DIR/$LABEL.fail"
    rm -f -- "$HB_DIR/$LABEL.ok"
    TAILOUT=$(tail -20 "$OUT")
    if [[ -n ${SLACK_BOT_TOKEN:-} ]]; then
        text=":rotating_light: *backup job FAILED*: \`$LABEL\` rc=$rc on bat-studio ($(date '+%Y-%m-%d %H:%M %Z'))"$'\n'"started $START"$'\n'"\`\`\`"$'\n'"$TAILOUT"$'\n'"\`\`\`"
        payload=$(SLACK_CHANNEL="$SLACK_CHANNEL" TEXT="$text" python3 -c \
            'import json,os;print(json.dumps({"channel":os.environ["SLACK_CHANNEL"],"text":os.environ["TEXT"]}))')
        curl -fsS -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
            -H "Content-Type: application/json; charset=utf-8" \
            --data "$payload" >/dev/null 2>&1 || true
    fi
fi

# Preserve the wrapped job's output for the launchd log, but always return the
# wrapped status rather than the status of cat/cleanup/notification work.
cat "$OUT" || true
exit "$rc"
