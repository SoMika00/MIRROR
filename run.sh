#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose not available (need Docker Compose v2)"
  exit 1
fi

echo "[1/4] Building images..."
docker compose build

echo "[2/4] Starting stack..."
docker compose up -d

# Resolve app container ID (robust even if container_name is not honored)
APP_CID="$(docker compose ps -q mirror | tail -n 1)"
if [ -z "${APP_CID}" ]; then
  echo "ERROR: could not resolve app container id for service 'mirror'"
  docker compose ps || true
  exit 1
fi

# Wait until app is responding
echo "[3/4] Waiting for mirror-app to be ready..."
for i in $(seq 1 60); do
  if docker exec "${APP_CID}" python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/models/status', timeout=2).read(); print('OK')" >/dev/null 2>&1; then
    echo "mirror-app is ready"
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "ERROR: mirror-app did not become ready in time"
    docker compose logs mirror --tail 80 || true
    exit 1
  fi
done

echo "[4/4] Ensuring models are present (download missing only)..."

docker exec "${APP_CID}" python3 - <<'PY'
import json
import time
import urllib.request

BASE = "http://localhost:5000/api/models"

def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))

def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

registry = get_json(f"{BASE}/llm/registry")["models"]
default = [m for m in registry if m.get("default")]
target = default[0]["id"] if default else None
missing = [target] if target and not next((m for m in registry if m.get("id") == target and m.get("downloaded")), None) else []
present = [m["id"] for m in registry if m.get("downloaded")]

print(f"Found {len(registry)} models in registry")
print(f"Already present: {len(present)}")
print(f"Target default model: {target}")
print(f"Missing default: {len(missing)}")

if not missing:
    print("Default model already present. Skipping downloads.")

for mid in missing:
    print(f"\n=== {mid} ===")
    # Start download
    try:
        resp = post_json(f"{BASE}/llm/download", {"model_id": mid})
    except Exception as e:
        print(f"Download start failed for {mid}: {e}")
        raise

    if resp.get("message") == "Already downloaded":
        print("Already downloaded")
        continue
    if resp.get("message") == "Download already in progress":
        print("Download already in progress")

    # Poll progress
    last_pct = -1
    while True:
        prog = get_json(f"{BASE}/llm/download-progress/{mid}")
        status = prog.get("status")
        if status == "done":
            print("Done")
            break
        if status == "error":
            raise RuntimeError(f"Download error for {mid}: {prog.get('error')}")
        pct = int(round(float(prog.get("progress", 0.0)) * 100))
        if pct != last_pct:
            print(f"Progress: {pct}%")
            last_pct = pct
        time.sleep(2)

print("\nDone. Missing models downloaded.")
PY

echo "All set. Stack is up and models are downloaded in ./models"

echo ""
echo "========================================"
echo "Open the site"
echo "========================================"

echo "Local (direct to app container):"
echo "  http://localhost:5000"
echo ""
echo "Local (via Caddy):"
echo "  http://localhost"
echo "  https://localhost"
echo ""

if [ -f "${PROJECT_DIR}/Caddyfile" ]; then
  echo "Configured public hostnames (from Caddyfile):"
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    /\{/ {
      h=$1
      gsub("\r","",h)
      if (h ~ /^https?:\/\//) { print "  " h }
      else if (h ~ /^[A-Za-z0-9.-]+$/) { print "  https://" h }
    }
  ' "${PROJECT_DIR}/Caddyfile" | sort -u
  echo ""
  echo "If your DNS isn't pointing yet, use the IP/localhost URLs above."
else
  echo "Caddyfile not found. (No public domain URL to display.)"
fi
