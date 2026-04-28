# DreamOn diffusion inference server

FastAPI server wrapping `Dream-org/DreamOn-v0-7B` for code-completion serving. Designed to run on a single GPU (RTX 5090), called only by `creor-backend`.

## Endpoints

- `POST /infill` — `{prefix, suffix, max_tokens, language?}` → `{completion, tokens, latency_ms, model, finish_reason}`. Bearer-auth.
- `GET /health` — liveness + VRAM stats. No auth.

## Local quickstart on Vast

Assumes the Vast box already ran `scripts/00_remote_setup.sh` so torch + DreamOn deps are present.

```bash
# 1. Mint a shared secret (used by both server and creor-backend)
openssl rand -hex 32 > /workspace/.diffusion_token

# 2. Start the server
cd /workspace/diffusion-llm/diffusion-experiments/server
bash run.sh
```

Test it locally on the box:

```bash
curl -s http://localhost:8000/health | jq .
curl -s -X POST http://localhost:8000/infill \
  -H "Authorization: Bearer $(cat /workspace/.diffusion_token)" \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' \
  | jq .
```

Expected: `completion` is something like `"a + b"`, `latency_ms` ≈ 150–400.

## Stable URL via Cloudflare Tunnel

Vast IPs change on restart. Put a stable Cloudflare-fronted URL in front:

```bash
# 1. Install cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb

# 2. Auth once with your Cloudflare account
cloudflared tunnel login    # opens a browser link; complete on your laptop

# 3. Create the tunnel and persist its credentials
mkdir -p /workspace/.cloudflared
cloudflared tunnel create creor-diffusion
mv ~/.cloudflared/*.json /workspace/.cloudflared/

# 4. Route a subdomain (you must own the zone in Cloudflare)
cloudflared tunnel route dns creor-diffusion diffusion.<your-domain>

# 5. Run the tunnel (point it at the local FastAPI)
cloudflared tunnel --credentials-file /workspace/.cloudflared/<tunnel-id>.json \
  run --url http://localhost:8000 creor-diffusion
```

Now `https://diffusion.<your-domain>` resolves to the local port 8000 regardless of Vast IP.

After the first setup, on instance restart you only need:

```bash
cloudflared tunnel --credentials-file /workspace/.cloudflared/<id>.json \
  run --url http://localhost:8000 creor-diffusion &
bash /workspace/diffusion-llm/diffusion-experiments/server/run.sh
```

(Both should be wrapped in tmux/systemd for unattended operation.)

## Wiring into creor-backend

Set the same secret on the Fly.io app:

```bash
flyctl secrets set --app creor-api \
  DIFFUSION_VAST_URL="https://diffusion.<your-domain>" \
  DIFFUSION_VAST_TOKEN="$(cat /workspace/.diffusion_token)" \
  DIFFUSION_TIMEOUT_MS=5000 \
  DIFFUSION_ENABLED=true
```

Then `POST https://creor-api.fly.dev/v1/diffusion/infill` will work end-to-end.

## Operational notes

- **Single worker**: `--workers 1` because the model is one in-process instance. FastAPI's asyncio handles HTTP concurrency; an internal lock serializes generate calls.
- **Cold start**: model load is ~10s. First request after restart will land slow; subsequent are 150–400ms p50.
- **VRAM**: ~15.3 GB resident. Single concurrent request peaks at ~18 GB. Stay below batch=2 — DreamOn's variable-length expansion breaks at batch>1.
- **Queue cap**: 8 in-flight requests max. Past that returns 503 with `Retry-After: 1` so the backend can back off cleanly.

## Acceptance test (full chain)

```bash
# From your Mac, against deployed creor-backend
curl -s -X POST https://creor-api.fly.dev/v1/diffusion/infill \
  -H "Authorization: Bearer crk_<your-key>" \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' \
  | jq .
```
