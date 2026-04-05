# alfred-infra

AI-infrastructure hardening kit for multi-machine local-LLM clusters: system and
GPU monitoring (node_exporter, DCGM, Prometheus), a ready-to-import Grafana
dashboard, cold backups across nodes, network-binding audits, and a context-
window benchmark for the Mac Studio inference box.

It assumes a common small-cluster topology:

- **Mac Studio** (Apple Silicon) — serving mlx-vlm / proxy stack, Prometheus +
  Grafana.
- **Spark A** (NVIDIA GPU box, e.g. DGX Spark) — RAG workload.
- **Spark B** (NVIDIA GPU box, e.g. DGX Spark) — voice pipeline.
- Nodes reach each other over **Tailscale** (100.x IPs).

Adapt the role labels (`rag`, `voice`) to whatever workloads your GPU nodes run.

## Files

| File | Purpose |
|------|---------|
| `alfred-backup.sh` | Cold backup. Pulls RAG + voice workloads from Sparks, collects Mac Studio configs, pushes everything to both nodes. |
| `alfred-health.sh` | HTTP + SSH health checks across the whole cluster. |
| `setup-monitoring.sh` | Installs `node_exporter` (all machines) + DCGM GPU exporter (Sparks). Interactive, machine-by-machine. |
| `check-bindings.sh` | Audits which services are on 0.0.0.0 vs Tailscale-only. Shows fix commands. |
| `prometheus-scrape-config.yml` | Snippet to add to your `prometheus.yml` for new scrape targets. |
| `alfred-infrastructure-dashboard.json` | Grafana dashboard: CPU, memory, disk, network, GPU temp/util/memory/power, uptime, status. |
| `context-bench.py` | Context-window benchmark for mlx-vlm (TTFT, throughput, page-out pressure across 4K–128K). |
| `context-bench-results.json` | Sample benchmark output for reference. |

## Configure before use

All shell scripts read host/IP settings from environment variables so nothing is
hard-coded. Set these (e.g. in your shell profile) before running anything:

```bash
# Tailscale hostnames / SSH targets (user@host)
export STUDIO_HOST="you@100.x.y.z"         # Mac Studio
export SPARK_A_HOST="user@100.x.y.z"       # Spark A — RAG node
export SPARK_B_HOST="user@100.x.y.z"       # Spark B — Voice node

# Raw Tailscale IPs (used in binding audits and health checks)
export STUDIO_IP="100.x.y.z"
export SPARK_A_IP="100.x.y.z"
export SPARK_B_IP="100.x.y.z"
```

The `prometheus-scrape-config.yml` file uses `<SPARK_A_IP>` / `<SPARK_B_IP>`
placeholders — substitute your own Tailscale IPs before pasting it into
Prometheus.

The Grafana dashboard expects Prometheus series labelled
`machine="spark-a"` / `machine="spark-b"` / `machine="mac-studio"`. If you
change those labels in the scrape config, update the dashboard queries to
match.

## Architecture

```
                    ┌──────────────────────────┐
                    │  Mac Studio (inference)  │
                    │  - mlx-vlm / proxy       │
                    │  - Prometheus + Grafana  │
                    │  - node_exporter         │
                    └────────────┬─────────────┘
                                 │ Tailscale
                    ┌────────────┴─────────────┐
                    │                          │
          ┌─────────▼─────────┐    ┌───────────▼─────────┐
          │ Spark A (RAG)     │    │ Spark B (Voice)     │
          │ - node_exporter   │    │ - node_exporter     │
          │ - dcgm-exporter   │    │ - dcgm-exporter     │
          │ - RAG server      │    │ - Voice server      │
          └───────────────────┘    └─────────────────────┘
```

## Deployment order

1. **Audit bindings first** — `./check-bindings.sh` (read-only, safe to run
   now). Fix any service listening on `0.0.0.0` before going further.
2. **Install exporters** — `./setup-monitoring.sh` (installs node_exporter +
   DCGM on each machine over SSH).
3. **Add Prometheus targets** — paste `prometheus-scrape-config.yml` into your
   `prometheus.yml`, replace `<SPARK_A_IP>`/`<SPARK_B_IP>`, restart Prometheus.
4. **Import Grafana dashboard** — Grafana → Dashboards → Import → Upload
   `alfred-infrastructure-dashboard.json`.
5. **Set up backups** — `chmod +x alfred-backup.sh`, test with
   `./alfred-backup.sh --dry-run`, then cron:
   ```
   # On Mac Studio crontab:
   0 3 * * * /path/to/alfred-backup.sh >> /tmp/alfred-backup-cron.log 2>&1
   ```
6. **Benchmark your context window** — `python3 context-bench.py` (from the
   Mac Studio, pointed at your mlx-vlm proxy) to find the largest context
   that runs without swap pressure.

## Notes

- DCGM exporter uses Docker. If Docker isn't on your GPU boxes, the installer
  falls back to an `nvidia-smi` cron that writes textfile metrics for
  node_exporter.
- The Grafana dashboard queries both DCGM and `nvidia-smi` metric names so it
  works with either exporter.
- Mac Studio GPU metrics (Apple Silicon) aren't covered by node_exporter. The
  dashboard focuses on CPU/memory/disk/network for the Mac Studio.
- Backup excludes `.lance` index files and training JSONL (large, regenerable).
  Add them back in the script if you want full corpus snapshots.
- The Tailscale IPs in the examples (`100.x.y.z`) are placeholders — replace
  them with your own Tailscale network's IPs per-install.

## Contributing

Issues and PRs welcome. This is a personal hardening kit first and a public
reference second, so expect opinionated defaults. If you adapt it for a
different topology (more nodes, different roles, no Tailscale), a PR with
those variants would be appreciated.

## Related projects

Part of a self-hosted LLM operations toolkit:

- [blockops-proxy](https://github.com/trevorgordon981/blockops-proxy) — tool-call-translating proxy for local LLM serving (monitored by this kit)
- [llm-otel-proxy](https://github.com/trevorgordon981/llm-otel-proxy) — OTel metrics proxy whose Prometheus output this kit's dashboards visualize
- [context-bench](https://github.com/trevorgordon981/context-bench) — context-window benchmark used from this kit to characterize new model deployments
- [alfred-rag](https://github.com/trevorgordon981/alfred-rag) — hybrid RAG stack (example workload running on this infrastructure)

## License

MIT — see [LICENSE](LICENSE).
