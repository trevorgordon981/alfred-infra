# M3 LAN proxy

A small authenticated HTTP proxy for exposing a loopback-only OpenAI-compatible model server to a trusted LAN.

Security properties:

- The upstream must be loopback-only and may not contain URL credentials.
- `POST /v1/chat/completions` requires a bearer token loaded from an owner-only regular file.
- Readiness and model discovery are limited to `GET /`, `GET /health`, and `GET /v1/models`;
  LAN health is reduced to `ready` and `model` so runtime paths, devices, and receipts stay local.
- Request framing, body size, methods, paths, and timeouts are fail-closed.
- Caller authorization, priority, and hop-by-hop headers are never forwarded upstream.
- Upstream hop-by-hop headers are stripped, and chunked responses are safely reframed while streaming.

Legacy `POST /` is intentionally unsupported.

## Run

Create a random token outside the repository, make it owner-only, and point clients at the proxy:

```bash
install -d -m 700 "$HOME/.config/m3-lan-proxy"
python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$HOME/.config/m3-lan-proxy/bearer-token"
chmod 600 "$HOME/.config/m3-lan-proxy/bearer-token"
python3 m3_lan_proxy.py
```

The defaults are `0.0.0.0:8096`, upstream `127.0.0.1:8082`, and a 20 MiB request cap. Environment variables beginning with `M3_LAN_PROXY_` configure ports, paths, limits, and bounded timeouts.

Configure every client to send:

```text
Authorization: Bearer <token>
```

## Test

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The tests are offline and use only synthetic credentials and loopback servers.
