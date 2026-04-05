#!/usr/bin/env bash
# check-bindings.sh — Audit what's listening on 0.0.0.0 vs Tailscale-only
# Run FROM the Mac Studio.
#
# Requires env vars (or set defaults below):
#   SPARK_A_HOST   — e.g. "node1@<spark-a-tailscale-ip>"
#   SPARK_B_HOST   — e.g. "node2@<spark-b-tailscale-ip>"
#   SPARK_A_IP     — Tailscale IP of Spark A
#   SPARK_B_IP     — Tailscale IP of Spark B
#   STUDIO_IP      — Tailscale IP of the Mac Studio

set -euo pipefail

SPARK_A_HOST="${SPARK_A_HOST:?set SPARK_A_HOST, e.g. user@100.x.y.z}"
SPARK_B_HOST="${SPARK_B_HOST:?set SPARK_B_HOST, e.g. user@100.x.y.z}"
SPARK_A_IP="${SPARK_A_IP:?set SPARK_A_IP to the Tailscale IP of Spark A}"
SPARK_B_IP="${SPARK_B_IP:?set SPARK_B_IP to the Tailscale IP of Spark B}"
STUDIO_IP="${STUDIO_IP:?set STUDIO_IP to the Tailscale IP of the Mac Studio}"

check_spark() {
    local label=$1
    local host=$2
    echo ""
    echo "=== $label ==="
    ssh "$host" << 'REMOTE'
echo "Listening on 0.0.0.0 (exposed to LAN):"
ss -tlnp 2>/dev/null | grep -E '0\.0\.0\.0:(8080|8082|8090|9000|9001|4317|9090|3000)\b' || echo "  (none)"
echo ""
echo "Listening on 127.0.0.1 or Tailscale only (safe):"
ss -tlnp 2>/dev/null | grep -E '(127\.0\.0\.1|100\.):.*(8080|8082|8090|9000|9001|4317|9090|3000)\b' || echo "  (none)"
echo ""
echo "All listeners on service ports:"
ss -tlnp 2>/dev/null | grep -E ':(8080|8082|8090|9000|9001|4317|9090|3000)\b' || echo "  (none)"
REMOTE
}

check_local_mac() {
    echo ""
    echo "=== Mac Studio (local) ==="
    echo "All listeners on service ports:"
    lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null | grep -E ':(8080|8082|8090|9000|9001|4317|9090|3000) ' || echo "  (none — are services running?)"
    echo ""
    echo "Bound to *: (exposed to LAN):"
    lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null | grep -E ':(8080|8082|8090|9000|9001|4317|9090|3000) ' | grep '\*:' || echo "  (none)"
    echo ""
    echo "Bound to 127.0.0.1 or Tailscale (safe):"
    lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null | grep -E ':(8080|8082|8090|9000|9001|4317|9090|3000) ' | grep -E '(127\.0\.0\.1|100\.)' || echo "  (none)"
}

echo "Service Binding Audit"
echo "====================="

check_spark "Spark A (RAG :9000)" "$SPARK_A_HOST"
check_spark "Spark B (Voice :9001)" "$SPARK_B_HOST"
check_local_mac

echo ""
echo "=== Recommendations ==="
echo "If any service shows 0.0.0.0 or *:, fix by binding to Tailscale IP:"
echo ""
echo "  RAG server (Spark A):    uvicorn ... --host ${SPARK_A_IP}"
echo "  Voice server (Spark B):  uvicorn ... --host ${SPARK_B_IP}"
echo "  mlx-vlm (Mac Studio):    bind proxy to ${STUDIO_IP}"
