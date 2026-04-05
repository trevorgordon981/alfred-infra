#!/usr/bin/env bash
# alfred-backup.sh — Cold backup of critical files to both GPU nodes (Sparks)
# Run from the Mac Studio. Cron daily or manual.
# Usage: ./alfred-backup.sh [--dry-run]
#
# Requires env vars:
#   SPARK_A_HOST   — e.g. "user@<spark-a-tailscale-ip>"  (RAG node)
#   SPARK_B_HOST   — e.g. "user@<spark-b-tailscale-ip>"  (Voice node)

set -euo pipefail

SPARK_A_HOST="${SPARK_A_HOST:?set SPARK_A_HOST, e.g. user@100.x.y.z}"
SPARK_B_HOST="${SPARK_B_HOST:?set SPARK_B_HOST, e.g. user@100.x.y.z}"

# --- Configuration ---
BACKUP_DIR="${BACKUP_DIR:-alfred-backups}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$HOME/alfred-backup.log"

TARGETS=("$SPARK_A_HOST" "$SPARK_B_HOST")

DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run" && echo "[DRY RUN MODE]"

# --- What to back up ---
# Add/remove paths as your stack evolves
BACKUP_SOURCES=(
    # Agent identity / config
    "$HOME/.openclaw/openclaw.json"
    "$HOME/.openclaw/workspace/AGENTS.md"
    "$HOME/.openclaw/workspace/SOUL.md"

    # OTel stack configs
    "$HOME/alfred-otel/"

    # mlx-vlm / proxy configs
    "$HOME/blockops-proxy.py"
)

RSYNC_OPTS=(
    -avz
    --delete
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.git'
    --exclude='node_modules'
    --exclude='*.egg-info'
    --exclude='venv/'
    --exclude='.venv/'
    --exclude='*.log'
    $DRY_RUN
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# --- Step 1: Pull Spark data to Mac Studio staging area ---
STAGING="$HOME/${BACKUP_DIR}/staging"
mkdir -p "$STAGING"

log "=== Backup Started ($TIMESTAMP) ==="

log "Pulling RAG data from Spark A..."
rsync "${RSYNC_OPTS[@]}" \
    --exclude='*.lance' \
    --exclude='training_data/*.jsonl' \
    "${SPARK_A_HOST}:~/alfred-rag/" "$STAGING/alfred-rag/" 2>&1 | tail -1 | tee -a "$LOG_FILE"

log "Pulling voice pipeline from Spark B..."
rsync "${RSYNC_OPTS[@]}" \
    "${SPARK_B_HOST}:~/alfred-voice/" "$STAGING/alfred-voice/" 2>&1 | tail -1 | tee -a "$LOG_FILE"

# --- Step 2: Collect Mac Studio local files ---
for src in "${BACKUP_SOURCES[@]}"; do
    if [[ -e "$src" ]]; then
        # Preserve directory structure relative to $HOME
        rel_path="${src#$HOME/}"
        dest_dir="$STAGING/mac-studio/$(dirname "$rel_path")"
        mkdir -p "$dest_dir"
        rsync "${RSYNC_OPTS[@]}" "$src" "$dest_dir/" 2>&1 | tail -1 | tee -a "$LOG_FILE"
    else
        log "SKIP (not found): $src"
    fi
done

# --- Step 3: Push everything to both Sparks ---
for target in "${TARGETS[@]}"; do
    log "Pushing backup to ${target}..."
    ssh "$target" "mkdir -p ~/${BACKUP_DIR}" 2>/dev/null || true
    rsync "${RSYNC_OPTS[@]}" \
        "$STAGING/" "${target}:~/${BACKUP_DIR}/" 2>&1 | tail -1 | tee -a "$LOG_FILE"
    log "Done: ${target}"
done

# --- Step 4: Write manifest ---
MANIFEST="$STAGING/MANIFEST.txt"
echo "Backup Manifest — $TIMESTAMP" > "$MANIFEST"
echo "---" >> "$MANIFEST"
find "$STAGING" -type f -not -name "MANIFEST.txt" | wc -l | xargs -I{} echo "Total files: {}" >> "$MANIFEST"
du -sh "$STAGING" | awk '{print "Total size: " $1}' >> "$MANIFEST"
echo "---" >> "$MANIFEST"
find "$STAGING" -type f -not -name "MANIFEST.txt" -exec stat -f '%Sm %z %N' -t '%Y-%m-%d %H:%M' {} \; | sort -r | head -20 >> "$MANIFEST"

# Push manifest
for target in "${TARGETS[@]}"; do
    scp -q "$MANIFEST" "${target}:~/${BACKUP_DIR}/MANIFEST.txt" 2>/dev/null
done

log "=== Backup Complete ==="
log "Staged at: $STAGING"
log "Pushed to: ${TARGETS[*]}"
log "Manifest: $MANIFEST"
