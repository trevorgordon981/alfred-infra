#!/usr/bin/env bash
# setup-monitoring.sh — Install system metrics exporters across the cluster
# Run sections on each respective machine. Ports:
#   node_exporter:     :9100 (all machines)
#   dcgm-exporter:     :9400 (Sparks only, NVIDIA GPU metrics)
#
# After running, add scrape targets to Prometheus on the Mac Studio.
#
# Requires env vars (or set defaults below):
#   SPARK_A_HOST   — e.g. "user@<spark-a-tailscale-ip>"
#   SPARK_B_HOST   — e.g. "user@<spark-b-tailscale-ip>"
#   STUDIO_HOST    — e.g. "user@<studio-tailscale-ip>"

set -euo pipefail

SPARK_A_HOST="${SPARK_A_HOST:?set SPARK_A_HOST, e.g. user@100.x.y.z}"
SPARK_B_HOST="${SPARK_B_HOST:?set SPARK_B_HOST, e.g. user@100.x.y.z}"
STUDIO_HOST="${STUDIO_HOST:?set STUDIO_HOST, e.g. user@100.x.y.z}"

NODE_EXPORTER_VERSION="1.8.2"

# =============================================================================
# SPARK SETUP (run this section on each Spark, or use the remote installer below)
# =============================================================================
install_spark_exporters() {
    local HOST=$1
    echo "=== Installing exporters on $HOST ==="

    ssh "$HOST" bash -s << 'REMOTE_SCRIPT'
set -euo pipefail

# --- node_exporter ---
if ! command -v node_exporter &>/dev/null && [ ! -f /usr/local/bin/node_exporter ]; then
    echo "Installing node_exporter..."
    cd /tmp
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" ]]; then
        NE_ARCH="linux-arm64"
    else
        NE_ARCH="linux-amd64"
    fi
    curl -sLO "https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.${NE_ARCH}.tar.gz"
    tar xzf "node_exporter-1.8.2.${NE_ARCH}.tar.gz"
    sudo cp "node_exporter-1.8.2.${NE_ARCH}/node_exporter" /usr/local/bin/
    rm -rf "node_exporter-1.8.2.${NE_ARCH}"*
    echo "node_exporter installed."
else
    echo "node_exporter already installed."
fi

# --- systemd service for node_exporter ---
if [ ! -f /etc/systemd/system/node_exporter.service ]; then
    sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'SVC'
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/node_exporter \
    --web.listen-address=:9100 \
    --collector.filesystem.mount-points-exclude="^/(sys|proc|dev|host|etc)($$|/)"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVC
    sudo systemctl daemon-reload
    sudo systemctl enable --now node_exporter
    echo "node_exporter service started."
else
    sudo systemctl restart node_exporter
    echo "node_exporter service restarted."
fi

# --- DCGM exporter for NVIDIA GPU metrics ---
if ! docker ps --format '{{.Names}}' | grep -q dcgm-exporter 2>/dev/null; then
    echo "Starting dcgm-exporter container..."
    # Check if docker is available
    if command -v docker &>/dev/null; then
        docker run -d --restart=always \
            --name dcgm-exporter \
            --gpus all \
            -p 9400:9400 \
            nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.1-ubuntu22.04 || {
            echo "DCGM container failed. Falling back to nvidia-smi text collector..."
            # Fallback: cron-based nvidia-smi metrics for node_exporter textfile collector
            sudo mkdir -p /var/lib/node_exporter/textfile_collector
            sudo tee /usr/local/bin/nvidia-smi-collector.sh > /dev/null << 'NVSCRIPT'
#!/bin/bash
OUTPUT="/var/lib/node_exporter/textfile_collector/gpu.prom"
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
    --format=csv,noheader,nounits | while IFS=', ' read -r idx temp util mem_used mem_total power clk; do
    echo "nvidia_gpu_temperature_celsius{gpu=\"$idx\"} $temp"
    echo "nvidia_gpu_utilization_percent{gpu=\"$idx\"} $util"
    echo "nvidia_gpu_memory_used_bytes{gpu=\"$idx\"} $((mem_used * 1048576))"
    echo "nvidia_gpu_memory_total_bytes{gpu=\"$idx\"} $((mem_total * 1048576))"
    echo "nvidia_gpu_power_draw_watts{gpu=\"$idx\"} $power"
    echo "nvidia_gpu_clock_mhz{gpu=\"$idx\"} $clk"
done > "$OUTPUT.tmp" && mv "$OUTPUT.tmp" "$OUTPUT"
NVSCRIPT
            sudo chmod +x /usr/local/bin/nvidia-smi-collector.sh
            (crontab -l 2>/dev/null; echo "* * * * * /usr/local/bin/nvidia-smi-collector.sh") | crontab -
            /usr/local/bin/nvidia-smi-collector.sh
            # Restart node_exporter with textfile collector
            sudo sed -i 's|ExecStart=.*|ExecStart=/usr/local/bin/node_exporter --web.listen-address=:9100 --collector.textfile.directory=/var/lib/node_exporter/textfile_collector --collector.filesystem.mount-points-exclude="^/(sys\\|proc\\|dev\\|host\\|etc)($$\\|/)"|' /etc/systemd/system/node_exporter.service
            sudo systemctl daemon-reload
            sudo systemctl restart node_exporter
            echo "Fallback nvidia-smi textfile collector configured."
        }
    else
        echo "WARNING: Docker not available. Skipping DCGM exporter."
    fi
else
    echo "dcgm-exporter already running."
fi

echo "=== Spark exporter setup complete ==="
REMOTE_SCRIPT
}

# =============================================================================
# MAC STUDIO SETUP (macOS — no NVIDIA GPU metrics needed, Apple Silicon)
# =============================================================================
install_mac_studio_exporter() {
    echo "=== Installing node_exporter on Mac Studio ==="

    ssh "$STUDIO_HOST" bash -s << 'REMOTE_SCRIPT'
set -euo pipefail

# Use Homebrew on macOS
if ! command -v node_exporter &>/dev/null; then
    if command -v brew &>/dev/null; then
        brew install node_exporter
    else
        echo "ERROR: Homebrew not found. Install manually."
        exit 1
    fi
fi

# Create launchd plist for node_exporter
PLIST="$HOME/Library/LaunchAgents/com.prometheus.node_exporter.plist"
if [ ! -f "$PLIST" ]; then
    NE_PATH=$(which node_exporter)
    cat > "$PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.prometheus.node_exporter</string>
    <key>ProgramArguments</key>
    <array>
        <string>${NE_PATH}</string>
        <string>--web.listen-address=:9100</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/node_exporter.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/node_exporter.err</string>
</dict>
</plist>
PLISTEOF
    launchctl load "$PLIST"
    echo "node_exporter launchd service started."
else
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "node_exporter launchd service restarted."
fi

# Apple Silicon GPU/thermal metrics via powermetrics textfile
# (requires sudo — optional, skip if you don't want to grant it)
echo ""
echo "NOTE: For Apple Silicon GPU metrics, you can optionally set up a"
echo "cron job running 'sudo powermetrics' to export thermal/GPU data."
echo "This requires passwordless sudo for powermetrics. Skipping for now."

echo "=== Mac Studio exporter setup complete ==="
REMOTE_SCRIPT
}

# =============================================================================
# MAIN
# =============================================================================
echo "Infrastructure Monitoring Setup"
echo "================================"
echo ""
echo "This will install Prometheus exporters on all 3 machines."
echo "Prerequisite: Passwordless SSH from this machine to all targets."
echo ""
read -p "Install on Spark A? (y/n) " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] && install_spark_exporters "$SPARK_A_HOST"

read -p "Install on Spark B? (y/n) " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] && install_spark_exporters "$SPARK_B_HOST"

read -p "Install on Mac Studio? (y/n) " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] && install_mac_studio_exporter

echo ""
echo "=== Next Steps ==="
echo "Add scrape targets to your Prometheus config (e.g. ~/alfred-otel/prometheus.yml):"
echo "See the prometheus-scrape-config.yml file for the snippet to add."
