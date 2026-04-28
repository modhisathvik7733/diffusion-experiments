# Phase 1 — From-scratch setup runbook

End-to-end commands to recreate the diffusion-LM autocomplete backend
(Vast.ai inference server + Cloudflare Tunnel + creor-backend proxy).
Every step has a verification check. Every gotcha we hit during the
original setup is documented in the inline notes and the troubleshooting
section at the bottom.

**Estimated time from a fresh box: ~90 min.** Most is waiting for builds.

## Prerequisites

Before starting:

- [ ] **Vast.ai account** with payment configured. Need ~$0.40/hr GPU.
- [ ] **Cloudflare-managed domain** (we used `creor.ai`). For testing
  only you can substitute Cloudflare's `*.trycloudflare.com` quick
  tunnels — see Troubleshooting §F.
- [ ] **Fly.io account** with `creor-api` app already provisioned and
  `flyctl` installed locally.
- [ ] **GitHub repo access**: write to `creor-labs/creor-backend` and
  the diffusion-experiments repo.
- [ ] **HuggingFace account + token** with read access to
  `Dream-org/DreamOn-v0-7B` (Apache 2.0, public — login optional but
  recommended for download speed via `hf-transfer`).
- [ ] **macOS / Linux dev box** with `gh`, `flyctl`, `ssh-keygen`, `curl`,
  `jq`, `openssl`.

---

## Step 1 — Vast.ai instance

### 1.1 Pick the right hardware

Filter on Vast for:

- **GPU**: RTX 5090 (32 GB), or A6000 (48 GB), or any GPU with **≥ 24 GB VRAM**.
- **Disk**: ≥ 60 GB.
- **Image**: `vastai/pytorch_cuda-12.8.1-auto` (or any image with PyTorch ≥ 2.7 and CUDA 12.8).
- **Network**: ≥ 100 Mbps down (model downloads are 16 GB).

> **⚠️ Gotcha:** Smaller GPUs (24 GB) work but leave very little headroom.
> 16 GB GPUs can't run DreamOn-v0-7B in bf16 — would need INT8.

### 1.2 Add your SSH key

On your Mac:

```bash
[ -f ~/.ssh/id_ed25519.pub ] || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub | pbcopy   # then paste into Vast.ai SSH key dialog
```

Wait ~30 s for Vast to push the key to the new instance.

### 1.3 Add a host alias

In `~/.ssh/config` on your Mac:

```ssh-config
Host vast-diffusion
    HostName <ip-from-vast>
    Port <port-from-vast>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Test:

```bash
ssh vast-diffusion 'hostname && nvidia-smi --query-gpu=name --format=csv,noheader'
# expected: hostname + "NVIDIA GeForce RTX 5090"
```

### 1.4 Bootstrap the workspace

On the remote (`ssh vast-diffusion`):

```bash
mkdir -p /workspace/diffusion-llm && cd /workspace/diffusion-llm

# System deps
apt-get update -y
apt-get install -y git git-lfs curl tmux htop build-essential
git lfs install

# Repos
git clone --depth 1 https://github.com/ML-GSAI/LLaDA.git
git clone --depth 1 https://github.com/pengzhangzhi/Open-dLLM.git
git clone --depth 1 https://github.com/ZHZisZZ/dllm.git
git clone --depth 1 https://github.com/DreamLM/Dream.git
git clone --depth 1 https://github.com/DreamLM/DreamOn.git
git clone https://github.com/<your-fork>/diffusion-experiments.git

ls
# expected: Dream  DreamOn  LLaDA  Open-dLLM  diffusion-experiments  dllm
```

### 1.5 Install Python deps for the server

```bash
cd /workspace/diffusion-llm/diffusion-experiments
bash scripts/00_remote_setup.sh
# takes ~3-5 min
```

The script:

1. Reinstalls `torch + torchvision` from `cu128` channel — required because the next steps will try to downgrade torch and we want to defend against that.
2. Installs runtime deps with `datasets>=3.0,<4.0` (pinned for `lm_eval` compatibility).
3. Installs Open-dLLM **with `--no-deps`** so its `requirements.txt` (which has `flash-attn>=2.6.3` and `transformers<=4.49.0`) doesn't downgrade your environment.
4. Verifies `torch.cuda.is_available()` and `cuda capability >= 9` (Blackwell).

**Verify:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expected: 2.11.0+cu128 NVIDIA GeForce RTX 5090 (12, 0)
```

> **⚠️ Gotchas in this step (already handled by the script):**
> - `pyext` is broken on Python 3.12 — excluded from deps.
> - Open-dLLM's `requirements.txt` pins `transformers<=4.49.0` and pulls `flash-attn>=2.6.3` (no Blackwell wheel). The `--no-deps` flag avoids both.
> - `datasets 4.x` breaks `lm_eval`. Pin to 3.x.

---

## Step 2 — Diffusion server

### 2.1 Mint the shared secret

This token authenticates `creor-backend → Vast`. Persist it under `/workspace/`
so it survives container restarts.

```bash
[ -f /workspace/.diffusion_token ] || openssl rand -hex 32 > /workspace/.diffusion_token
echo "secret head: $(cat /workspace/.diffusion_token | head -c 16)..."
```

### 2.2 Start the FastAPI server

```bash
cd /workspace/diffusion-llm/diffusion-experiments
nohup bash server/run.sh > /workspace/diffusion-server.log 2>&1 &
echo "PID: $!"
sleep 25
```

> **⚠️ Gotcha:** the FastAPI server takes ~10–15 s on first launch (model
> download + load). Don't curl `/health` immediately — wait at least 20 s.

**Verify:**

```bash
curl -s http://localhost:8000/health | jq .ok
# expected: true
```

If `false` or curl returns nothing:

```bash
tail -40 /workspace/diffusion-server.log
```

Common failure modes — see **Troubleshooting** below.

### 2.3 Smoke-test inference

```bash
TOKEN=$(cat /workspace/.diffusion_token)
curl -s -X POST http://localhost:8000/infill \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' | jq .
# expected completion: " a + b" (give or take whitespace)
```

> **⚠️ Gotcha (fixed in current code):** Earlier the server used `n_mask =
> max_tokens` which broke DreamOn's `<|expand|>` variable-length mechanism.
> Output ended up sliced wrong by the post-processor (see commit `f441f4f`).
> If you see `"ition.addition(2, 3"` instead of `"(2, 3"` for the f-string
> test, this regression is back — `inference.py` should use `initial_masks=4`
> with `max_new_tokens=cap_tokens`.

---

## Step 3 — Cloudflare Tunnel (stable URL)

Vast IPs change on instance restart, so we put a Cloudflare-fronted DNS name in
front. Once-only setup; persists across Vast restarts as long as
`/workspace/.cloudflared/` is preserved.

### 3.1 Install cloudflared on the Vast box

```bash
which cloudflared || {
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i cloudflared-linux-amd64.deb
}
cloudflared --version
```

### 3.2 Authenticate cloudflared with your Cloudflare account

```bash
cloudflared tunnel login
```

Copy the printed URL → paste into your **laptop browser** → log in to
Cloudflare → select your zone (e.g. `creor.ai`) → "Authorize". Vast terminal
continues automatically. This writes `~/.cloudflared/cert.pem`.

### 3.3 Persist credentials to /workspace

```bash
mkdir -p /workspace/.cloudflared
mv ~/.cloudflared/cert.pem /workspace/.cloudflared/
ln -sf /workspace/.cloudflared/cert.pem ~/.cloudflared/cert.pem
```

### 3.4 Create the named tunnel

```bash
cloudflared tunnel create creor-diffusion
# output line: "Tunnel credentials written to /root/.cloudflared/<UUID>.json"

mv ~/.cloudflared/*.json /workspace/.cloudflared/
ln -sf /workspace/.cloudflared/*.json ~/.cloudflared/
ls /workspace/.cloudflared/
# expected: cert.pem  <UUID>.json
```

### 3.5 Route DNS

```bash
cloudflared tunnel route dns creor-diffusion diffusion.<your-domain>
```

Auto-creates a CNAME record under your zone. Confirm in Cloudflare's DNS
dashboard: there should be a new `diffusion` CNAME with target
`<UUID>.cfargotunnel.com` and the orange-cloud (proxied) icon.

### 3.6 Run the tunnel

```bash
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1
nohup cloudflared tunnel \
  --credentials-file /workspace/.cloudflared/$(ls /workspace/.cloudflared/ | grep '\.json$' | head -1) \
  run --url http://localhost:8000 creor-diffusion \
  > /workspace/cloudflared.log 2>&1 &
sleep 5
tail -10 /workspace/cloudflared.log
# expected: "Registered tunnel connection ... protocol=quic"
```

### 3.7 Verify from your Mac

```bash
curl -s https://diffusion.<your-domain>/health | jq .ok
# expected: true

TOKEN=$(ssh vast-diffusion 'cat /workspace/.diffusion_token')
curl -s -X POST https://diffusion.<your-domain>/infill \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' | jq .
```

If both succeed, the entire Vast → Cloudflare → world chain is live.

---

## Step 4 — creor-backend integration

### 4.1 Verify the diffusion route exists

The route was added in commit `4931198` of `creor-labs/creor-backend`. Confirm:

```bash
cd ~/Creative_Work/creor-backend
git log --oneline --all | grep -E "diffusion proxy|fly-deploy" | head -5
ls src/routes/diffusion.ts && echo "route present"
```

If `diffusion.ts` is missing on a fresh clone, that means you're working on a
branch that pre-dates the integration. Cherry-pick or rebase from `main`.

### 4.2 Set Fly secrets

```bash
TOKEN=$(ssh vast-diffusion 'cat /workspace/.diffusion_token')
echo "token head: ${TOKEN:0:8}..."

flyctl secrets set --app creor-api \
  DIFFUSION_VAST_URL="https://diffusion.<your-domain>" \
  DIFFUSION_VAST_TOKEN="$TOKEN" \
  DIFFUSION_TIMEOUT_MS=5000 \
  DIFFUSION_ENABLED=true
```

> **⚠️ Gotcha:** `flyctl secrets set` triggers an automatic redeploy of the
> *previous* image. It does NOT rebuild from source. If your deployed image
> doesn't already have the diffusion route, you must run a real
> `flyctl deploy` (next step).

### 4.3 First-time Fly deploy (or via CI)

#### Option A — push triggers CI deploy (recommended)

We added `.github/workflows/fly-deploy.yml` (commit `6fc70e6`). Any push to
`main` runs `flyctl deploy --remote-only`. Requires:

```bash
# One-time: set the Fly token as a GitHub secret
flyctl auth token | gh secret set FLY_API_TOKEN --repo creor-labs/creor-backend
```

Then any push to `main` deploys automatically. Watch:

```bash
gh run watch --repo creor-labs/creor-backend \
  $(gh run list --repo creor-labs/creor-backend --workflow "Deploy to Fly" --limit 1 --json databaseId --jq '.[0].databaseId')
```

#### Option B — manual flyctl deploy

```bash
cd ~/Creative_Work/creor-backend
flyctl deploy --app creor-api --remote-only
```

If "Waiting for depot builder" hangs > 2 min, fall back to:

```bash
flyctl deploy --app creor-api --local-only   # needs Docker Desktop running
```

### 4.4 Verify the deployment

```bash
# (No auth on health uses gateway-key middleware — pass any creor key)
CREOR_KEY="crk_..."   # any valid creor API key

curl -s https://creor-api.fly.dev/v1/diffusion/health \
  -H "Authorization: Bearer $CREOR_KEY" | jq .ok
# expected: true (with model info)

curl -s -X POST https://creor-api.fly.dev/v1/diffusion/infill \
  -H "Authorization: Bearer $CREOR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' | jq .
# expected: { "completion": "...", "tokens": 4, "latency_ms": 250-600, "request_id": "..." }
```

Round-trip latency should land **250–600 ms p50** depending on geography.

> **⚠️ Gotcha:** if `/v1/diffusion/health` returns `{ "error": "Not found" }`,
> the deployed Fly image doesn't have the route — `flyctl secrets set` didn't
> rebuild from source. Run `flyctl deploy --app creor-api --remote-only`.

---

## Step 5 — Operational notes

### 5.1 Restart procedures

#### Vast container restart (e.g. instance stopped + restarted)

`/workspace/` survives container restarts. Servers don't auto-start. To
restore the chain:

```bash
ssh vast-diffusion
cd /workspace/diffusion-llm/diffusion-experiments

# 1. Restart FastAPI
nohup bash server/run.sh > /workspace/diffusion-server.log 2>&1 &
sleep 20

# 2. Restart Cloudflare Tunnel
nohup cloudflared tunnel \
  --credentials-file /workspace/.cloudflared/$(ls /workspace/.cloudflared/*.json | head -1 | xargs basename) \
  run --url http://localhost:8000 creor-diffusion \
  > /workspace/cloudflared.log 2>&1 &
sleep 5

# 3. Verify
curl -s http://localhost:8000/health | jq .ok          # local
curl -s https://diffusion.<your-domain>/health | jq .ok   # via tunnel
```

For unattended operation, wrap the two commands in a tmux session that you
can detach from:

```bash
tmux new -s diffusion
# inside tmux:
bash server/run.sh   # in pane 1
# Ctrl-B " (split horizontally)
cloudflared tunnel ...   # in pane 2
# Ctrl-B D to detach
```

#### Vast instance destroyed (volume gone)

Steps 1.4 (clone repos) → 2.1 (mint NEW token) → 3.4 (create tunnel) again.
The Cloudflare Tunnel UUID will be different; old DNS entry needs to be
deleted from the Cloudflare dashboard. Update Fly secrets:

```bash
NEW_TOKEN=$(ssh vast-diffusion 'cat /workspace/.diffusion_token')
flyctl secrets set --app creor-api DIFFUSION_VAST_TOKEN="$NEW_TOKEN"
```

The DNS hostname (`diffusion.<your-domain>`) can be reused by routing the new
tunnel to it — Cloudflare lets you reassign.

### 5.2 Secret rotation

Rotate `DIFFUSION_VAST_TOKEN` when needed:

```bash
# On Vast:
openssl rand -hex 32 > /workspace/.diffusion_token
# Restart server to pick up new token
pkill -f "uvicorn app:app"; sleep 2; bash server/run.sh &

# On Mac:
flyctl secrets set --app creor-api \
  DIFFUSION_VAST_TOKEN="$(ssh vast-diffusion 'cat /workspace/.diffusion_token')"
# Fly auto-redeploys with the new secret
```

### 5.3 Monitoring

```bash
# Server logs
ssh vast-diffusion 'tail -f /workspace/diffusion-server.log'

# Tunnel logs
ssh vast-diffusion 'tail -f /workspace/cloudflared.log'

# Backend logs
flyctl logs --app creor-api | grep '\[diffusion\]'

# GPU utilization
ssh vast-diffusion 'nvidia-smi -l 2'
```

---

## Troubleshooting (gotchas we hit)

### A. `pip install` fails on `pyext` (Python 3.12)

Symptom: `AttributeError: module 'inspect' has no attribute 'getargspec'`
during `pip install`.

Fix: `pyext` is dead. The `00_remote_setup.sh` already excludes it. If you're
manually installing deps, skip `pyext`.

### B. `bun.lock: not found` in CI

Symptom: Fly Docker build fails at `COPY package.json bun.lock ./`.

Fix: `bun.lock` was in `.gitignore`. Remove that line and commit the
lockfile (commit `82c69e7`).

### C. `Cannot find package 'dodopayments'` in Fly logs

Symptom: app boots, immediately crashes with module-not-found error,
restart-loops 10× then exits.

Fix: `package.json` on `main` was missing the `dodopayments` dep that
`src/lib/dodopayments.ts` imports. Add it (commit `9b49626`).

### D. Health endpoint returns `{"error":"Missing API key"}`

Symptom: `/v1/diffusion/health` rejects unauthenticated requests.

Cause: app-level `CREOR_SUPABASE_ANON_KEY` middleware in `app.ts:101-125`
shields all paths except a small exempt list, and our health path isn't on
that list.

Workaround: pass any valid `crk_` key in `Authorization: Bearer`.

Permanent fix (when ready): add `/v1/diffusion/health` to `EXEMPT_PREFIXES`
in `app.ts`.

### E. `flyctl secrets set` doesn't deploy new code

Symptom: secrets set successfully, but `/v1/diffusion/*` still 404s.

Cause: `flyctl secrets set` redeploys the *existing image* with new env
vars. It does NOT rebuild from source.

Fix: run `flyctl deploy --app creor-api --remote-only` (or push to `main`
to trigger CI).

### F. No Cloudflare-managed domain

Use a quick tunnel instead of named tunnel:

```bash
nohup cloudflared tunnel --url http://localhost:8000 \
  > /workspace/cloudflared.log 2>&1 &
sleep 8
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /workspace/cloudflared.log | head -1
# example: https://elephant-violet-river-xyz.trycloudflare.com
```

Use that URL as `DIFFUSION_VAST_URL`. Caveat: every restart of `cloudflared`
gives a new URL — you'll need to re-run `flyctl secrets set` each time. Fine
for v1 alpha, not for production.

### G. Build hangs at "Waiting for depot builder"

Fly's depot builder fleet sometimes stalls. Two fallbacks:

```bash
flyctl deploy --app creor-api --remote-only   # uses older Fly remote builder
flyctl deploy --app creor-api --local-only    # builds on your Mac via Docker Desktop
```

### H. Server boots but `/infill` returns garbage like `"ition.addition(2, 3"`

Cause: `inference.py` regression — initial mask count tied to `max_tokens`,
disabling DreamOn's variable-length expansion.

Fix: ensure `inference.py` has:

```python
initial_masks = 4
cap_tokens = max(initial_masks, min(int(max_tokens), 128))
input_ids, prompt_len = _encode(prefix, suffix, initial_masks)
out = _model.diffusion_generate(input_ids, max_new_tokens=cap_tokens, ...)
```

NOT `n_mask = max_tokens` for both initial and cap.

### I. Three GitHub workflows run on every push (Supabase + Fly + PR Tests)

This is by design. Each `.yml` file in `.github/workflows/` triggers
independently. The Fly one is the only one that affects diffusion.

Watch only the Fly run:

```bash
gh run watch --repo creor-labs/creor-backend \
  $(gh run list --repo creor-labs/creor-backend --workflow "Deploy to Fly" --limit 1 --json databaseId --jq '.[0].databaseId')
```

---

## Acceptance summary

After Phase 1, all of these pass:

```bash
# 1. Vast server health (on the box)
ssh vast-diffusion 'curl -s http://localhost:8000/health | jq .ok'   # → true

# 2. Cloudflare-fronted health (anywhere)
curl -s https://diffusion.<your-domain>/health | jq .ok   # → true

# 3. Backend health (with auth)
curl -s https://creor-api.fly.dev/v1/diffusion/health \
  -H "Authorization: Bearer crk_..." | jq .ok   # → true

# 4. End-to-end infill
curl -s -X POST https://creor-api.fly.dev/v1/diffusion/infill \
  -H "Authorization: Bearer crk_..." \
  -H "Content-Type: application/json" \
  -d '{"prefix":"def add(a, b):\n    return ","suffix":"\n","max_tokens":16}' \
  | jq .completion   # → " a + b" (or similar valid completion)

# 5. Fly logs show our request
flyctl logs --app creor-api | grep '\[diffusion\]' | tail -3
# expected line: [diffusion] ws=... prefix=Nc suffix=Nc tokens_out=N latency_ms=...

# 6. Supabase usage table has the row
# (in Supabase SQL editor)
# SELECT request_id, model, output_tokens FROM usage
# WHERE model = 'diffusion/dreamon-v0-7b'
# ORDER BY time_created DESC LIMIT 5;
```

---

## What's next (Phase 2)

This runbook only covers Phase 1: server + backend route. Phase 2 wires this
chain into the IDE so users see ghost-text autocomplete:

1. Add `POST /inference/infill` to `opencode/packages/opencode/src/server/routes/inference.ts` (proxies to `creor-backend` with the user's `crk_` key).
2. Add `engineClient.infill()` method in `vscode-fork/src/vs/workbench/contrib/creor/common/engine-client.ts`.
3. Register an `InlineCompletionItemProvider` in `vscode-fork/src/vs/workbench/contrib/creor/browser/inlineCompletion/`.
4. Add `creor.inlineCompletion.enabled` setting.

That's a separate plan and PR. See `binary-meandering-parasol.md` planning notes.
