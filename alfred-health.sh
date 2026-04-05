#!/bin/bash
# Alfred Infrastructure Health Check
# Checks all services across a 3-machine cluster (Mac Studio + 2 NVIDIA Sparks).
#
# Requires env vars:
#   SPARK_A_HOST   — e.g. "user@<spark-a-tailscale-ip>"
#   SPARK_B_HOST   — e.g. "user@<spark-b-tailscale-ip>"
#   SPARK_A_IP     — Tailscale IP of Spark A (RAG node)
#   SPARK_B_IP     — Tailscale IP of Spark B (Voice node)

SPARK_A_HOST="${SPARK_A_HOST:?set SPARK_A_HOST, e.g. user@100.x.y.z}"
SPARK_B_HOST="${SPARK_B_HOST:?set SPARK_B_HOST, e.g. user@100.x.y.z}"
SPARK_A_IP="${SPARK_A_IP:?set SPARK_A_IP to the Tailscale IP of Spark A}"
SPARK_B_IP="${SPARK_B_IP:?set SPARK_B_IP to the Tailscale IP of Spark B}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

PASS="${GREEN}PASS${NC}"
FAIL="${RED}FAIL${NC}"

check_http() {
    local name="$1" url="$2" timeout="${3:-3}"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" 2>/dev/null)
    if [[ "$code" =~ ^(200|301|302|307|308)$ ]]; then
        printf "  %-28s %b  (HTTP %s)\n" "$name" "$PASS" "$code"
    else
        printf "  %-28s %b  (HTTP %s)\n" "$name" "$FAIL" "${code:-timeout}"
        return 1
    fi
}

check_ssh() {
    local name="$1" host="$2"
    if ssh -o ConnectTimeout=3 -o BatchMode=yes "$host" 'echo ok' &>/dev/null; then
        printf "  %-28s %b\n" "$name" "$PASS"
    else
        printf "  %-28s %b\n" "$name" "$FAIL"
        return 1
    fi
}

failures=0

echo ""
printf "${BOLD}%-30s %-10s %s${NC}\n" "SERVICE" "STATUS" ""
echo "  ─────────────────────────────────────────────"

printf "\n${BOLD}  Studio (Mac Studio, Apple Silicon)${NC}\n"
check_http "mlx-vlm"           "http://localhost:8082/health"    || ((failures++))
check_http "proxy"             "http://localhost:8080/health"    || ((failures++))
check_http "metrics-proxy"     "http://localhost:8090/health"    || ((failures++))
check_http "prometheus"        "http://localhost:9090/-/healthy"  || ((failures++))
check_http "grafana"           "http://localhost:3000/api/health" || ((failures++))
check_http "node-exporter"     "http://localhost:9100/metrics"    || ((failures++))

printf "\n${BOLD}  Spark A (RAG)${NC}\n"
check_ssh  "ssh"               "$SPARK_A_HOST"                          || ((failures++))
check_http "rag-server"        "http://${SPARK_A_IP}:9000/health"       || ((failures++))
check_http "node-exporter"     "http://${SPARK_A_IP}:9100/metrics"      || ((failures++))
check_http "dcgm-exporter"     "http://${SPARK_A_IP}:9400/metrics"      || ((failures++))

printf "\n${BOLD}  Spark B (Voice)${NC}\n"
check_ssh  "ssh"               "$SPARK_B_HOST"                          || ((failures++))
check_http "voice-server"      "http://${SPARK_B_IP}:9001/health"       || ((failures++))
check_http "node-exporter"     "http://${SPARK_B_IP}:9100/metrics"      || ((failures++))
check_http "dcgm-exporter"     "http://${SPARK_B_IP}:9400/metrics"      || ((failures++))

echo ""
echo "  ─────────────────────────────────────────────"
if [ "$failures" -eq 0 ]; then
    printf "  ${GREEN}${BOLD}All services healthy${NC}\n"
else
    printf "  ${RED}${BOLD}%d service(s) down${NC}\n" "$failures"
fi
echo ""

exit "$failures"
