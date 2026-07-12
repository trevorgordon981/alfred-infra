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
  custom-Python shim, batch core, complete loaded `mlx`, `mlx_lm`, and `mlx_vlm`
  package trees (including native artifacts), MLX
  versions, realized device, and cache/generation contract; readiness is not
  published until that receipt is durable.
- Runtime package and batch-core identities are taken before import and reproved
  after model load. Traversal errors fail startup, bytecode is included in the
  tree identity, and imports use an isolated no-write bytecode cache so the
  receipt cannot describe different code from the code resident in memory.
- Readiness is fail-closed: an artifact/reference manifest is mandatory, and
  representative plain, constrained-JSON, native tool-call, and true-stream
  generations must all succeed. Their prompt/output digests and token counts
  are embedded in and cryptographically bound by the create-only runtime
  receipt. A failed warmup never becomes `ready:true`.
- Serving holds the same owner-token machine resource lease as training,
  full-model builds, and evaluation. Pipeline children re-prove an inherited
  token; standalone serving owns the lease for its full lifetime. It never
  expires by age.

## Runtime

Use Python 3.12 on an Apple Silicon Mac with the pinned MLX stack:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-apple-silicon.txt

M3_MODEL_DIR=~/models/MiniMax-M3-6bit-v1-hotpath-abl \
M3_ARTIFACT_MANIFEST=~/.local/share/m3-serving/production-artifact.json \
M3_PORT=8082 \
M3_APC=1 \
M3_APC_BLOCKS=4096 \
M3_ARTIFACT_MANIFEST=/absolute/path/to/artifact-or-reference.json \
M3_RUNTIME_RECEIPT_DIR=/absolute/owner-only/runtime-receipts \
PYTHONPATH="$PWD:$PWD/vendor/mlx-vlm" \
python m3_serve_batched.py
```

`M3_ARTIFACT_MANIFEST` is mandatory. It must be a canonical, immutable
pipeline artifact/reference receipt whose bound model path exactly equals
`M3_MODEL_DIR`; an empty value, a placeholder, or a receipt for another model
fails before the server binds. The old manifest-free launch example is invalid
and must not be used for production or benchmarking.

The pinned `mlx-vlm` fork provides MiniMax M3 support, the batch KV-cache API,
and structured decoding used here. Keep the server and fork revisions locked
together; upgrading either requires the security suite plus real model smoke.

Important environment controls:

| Variable | Default | Purpose |
|---|---:|---|
| `M3_MODEL_DIR` | `~/models/MiniMax-M3-8bit` | Model directory |
| `M3_ARTIFACT_MANIFEST` | none; required | Canonical receipt binding the exact model artifact |
| `M3_PORT` | `8085` | Loopback port |
| `M3_MAX_BODY_BYTES` | `20971520` | Request-body cap |
| `M3_MAX_IMG_BYTES` | `12582912` | Decoded image cap |
| `M3_MAX_TOKENS` | `16384` | Per-request output cap |
| `M3_MAX_PENDING` | `8` | Ordinary/default queued-request ceiling |
| `M3_PRIORITY0_RESERVED` | `2` | Additional hard-bounded slots reserved for authenticated priority-0 work |
| `M3_PRIORITY_TOKEN_FILE` | unset | Owner-only priority credential |
| `M3_ARTIFACT_MANIFEST` | unset | Canonical artifact/reference receipt that must bind `M3_MODEL_DIR` |
| `M3_RUNTIME_RECEIPT_DIR` | unset | Owner-only directory for unique create-only runtime receipts (required with a manifest) |
| `M3_RUNTIME_RECEIPT` | unset | Optional explicit fresh receipt path for one-shot launches |
| `M3_MACHINE_RESOURCE_LOCK` | `~/.local/state/alfred-machine-resource.lock` | Shared train/build/eval/serve lease |
| `M3_ALLOW_SAVE_TO` | `0` | Enable confined create-only saves |
| `M3_SAVE_DIR` | `~/m3_saves` | Confined save root |
| `M3_LORA_PROFILES` | `0` | Enable the receipt-bound experimental `general=0` / `trader=1` adapter profiles |
| `M3_LORA_ADAPTER_DIR` | unset | Canonical unfused adapter directory used only when profile mode is enabled |
| `M3_LORA_PROFILE_CONFIG` | unset | Canonical receipt binding the exact adapter and fixed profiles |
| `M3_LORA_PROFILE_TOKEN_FILE` | unset | Owner-only token required for every explicit profile request |

## Receipt-bound adapter profiles

Profile mode is an opt-in evaluation and rollback facility; it is not the
default production architecture. `M3_LORA_PROFILES=0` preserves the existing
fused model behavior exactly. When enabled, the server accepts only two fixed
strengths: `general=0` and `trader=1`. Arbitrary strengths, body-selected
profiles, unknown names, and multiple adapters fail closed.

Every explicit profile request requires both `X-M3-LoRA-Profile` and
`X-M3-LoRA-Profile-Token`. The separate LAN proxy forwards only this exact
pair in addition to its existing request-header allowlist; it never forwards
the proxy bearer credential. Missing, duplicated, connection-nominated, or
incorrect profile controls are rejected before work reaches the model queue.

The profile receipt binds the adapter directory, `adapter_config.json`,
`adapters.safetensors`, their SHA-256 digests and sizes, the `general` default,
and exactly `{"general":0.0,"trader":1.0}`. Startup rechecks those files
before and after model load and loads the adapter with explicit
`trust_remote_code=True`. Profile scale changes are serialized under the
generation lock, MLX is synchronized before restoring `general=0`, batches
never mix profiles, and automatic-prefix-cache tenants include the profile
receipt and strength.

Do not enable this mode merely because an adapter has lower training loss. A
candidate still needs the same frozen general-reasoning, coding, safety,
tool-use, portfolio, latency, and memory gates required for a fused promotion.

## Verification

The CI suite deliberately imports the server without MLX installed to prove
that import has no side effect and to test request framing, bounds, routes,
priority authentication, save confinement, error redaction, streaming failure
ordering, and structured-output fail-closed behavior:

```bash
python -m unittest discover -v -s tests -p 'test_*.py'
```

A production deployment still requires supervised real-model validation. The
server itself now runs structured JSON, tool-format, and streaming smokes before
readiness and receipts their outcomes. Do not restart a resident model for
source-only test or publication work.
