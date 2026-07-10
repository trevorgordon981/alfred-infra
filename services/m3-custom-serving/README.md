# M3 custom Python serving engine

Dependency-light OpenAI-compatible HTTP serving for MiniMax M3 on Apple
Silicon. This is the custom Python/MLX engine used by Alfred; it is not vMLX and
does not install or launch a second server implementation.

The process binds to `127.0.0.1` only. Put authentication/TLS in a separately
managed loopback-to-LAN proxy when remote access is required.

## Guarantees

- `POST /v1/chat/completions` plus the retained local legacy `POST /` contract.
- `GET /health`, `/v1/models`, and `/metrics`; health includes semantic
  `ready:false` while the model is loading even though the socket is already up.
- One serialized model worker with bounded priority queue and batching for short
  compatible requests.
- Optional keyed automatic prefix cache (`M3_APC=1`); the old shared prefix
  cache remains off by default because it can cross-contaminate conversations.
- Explicit JSON-schema output fails closed when the structured decoder is
  absent or broken; it never silently falls back to unconstrained generation.
- Canonical `Content-Length`, no request chunking, 20 MiB body cap, 12 MiB image
  cap, bounded token/temperature inputs, generic client errors, and no internal
  path/exception disclosure.
- Priority 0 requires an owner-only token file. Priority 2 (bench/yield) remains
  available without privilege; unknown priorities become normal priority 1.
- Arbitrary `save_to` is disabled by default. The opt-in path is confined to an
  owner-only directory and uses create-only owner-only files.
- Evaluation/promotion startup can bind a canonical pipeline artifact/reference
  receipt. Health then exposes a create-only runtime receipt containing the exact
  custom-Python shim, batch core, complete loaded `mlx_vlm` package tree, MLX
  versions, realized device, and cache/generation contract; readiness is not
  published until that receipt is durable.

## Runtime

Use Python 3.12 on an Apple Silicon Mac with the pinned MLX stack:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-apple-silicon.txt

M3_MODEL_DIR=~/models/MiniMax-M3-6bit-v1-hotpath-abl \
M3_PORT=8082 \
M3_APC=1 \
M3_APC_BLOCKS=4096 \
M3_ARTIFACT_MANIFEST=/absolute/path/to/artifact-or-reference.json \
M3_RUNTIME_RECEIPT_DIR=/absolute/owner-only/runtime-receipts \
PYTHONPATH="$PWD:$PWD/vendor/mlx-vlm" \
python m3_serve_batched.py
```

The pinned `mlx-vlm` fork provides MiniMax M3 support, the batch KV-cache API,
and structured decoding used here. Keep the server and fork revisions locked
together; upgrading either requires the security suite plus real model smoke.

Important environment controls:

| Variable | Default | Purpose |
|---|---:|---|
| `M3_MODEL_DIR` | `~/models/MiniMax-M3-8bit` | Model directory |
| `M3_PORT` | `8085` | Loopback port |
| `M3_MAX_BODY_BYTES` | `20971520` | Request-body cap |
| `M3_MAX_IMG_BYTES` | `12582912` | Decoded image cap |
| `M3_MAX_TOKENS` | `16384` | Per-request output cap |
| `M3_MAX_PENDING` | `8` | Queued request cap |
| `M3_PRIORITY_TOKEN_FILE` | unset | Owner-only priority credential |
| `M3_ARTIFACT_MANIFEST` | unset | Canonical artifact/reference receipt that must bind `M3_MODEL_DIR` |
| `M3_RUNTIME_RECEIPT_DIR` | unset | Owner-only directory for unique create-only runtime receipts (required with a manifest) |
| `M3_RUNTIME_RECEIPT` | unset | Optional explicit fresh receipt path for one-shot launches |
| `M3_ALLOW_SAVE_TO` | `0` | Enable confined create-only saves |
| `M3_SAVE_DIR` | `~/m3_saves` | Confined save root |

## Verification

The CI suite deliberately imports the server without MLX installed to prove
that import has no side effect and to test request framing, bounds, routes,
priority authentication, save confinement, error redaction, streaming failure
ordering, and structured-output fail-closed behavior:

```bash
python -m unittest -v tests/test_m3_serve_batched_security.py
```

A production deployment additionally requires a real-model structured JSON
request and streaming smoke. Do not restart a resident model for source-only
test or publication work.
