#!/usr/bin/env bash
# Update .env to point API/vLLM to artifacts from a pipeline run
# - Updates FAISS_INDEX_PATH, STORE_PATH
# - Sets either ADAPTER_PATH (LoRA) or MODEL_NAME (merged)
# - Optionally restarts the stack
#
# Usage:
#   ./scripts/switch_artifacts.sh [--run-dir PATH] [--mode lora|merged] [--no-restart]
# Examples:
#   ./scripts/switch_artifacts.sh                                   # auto-pick latest complete_*
#   ./scripts/switch_artifacts.sh --run-dir /mnt/data/.../complete_dev1_YYYYMMDD_HHMMSS
#   ./scripts/switch_artifacts.sh --mode merged                      # use merged model if present

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

RUN_DIR=""
MODE="lora"           # default: LoRA adapter
RESTART=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
    --no-restart)
      RESTART=0; shift;;
    -h|--help)
      sed -n '1,30p' "$0" | sed -n '1,20p'
      exit 0;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# Find latest complete_* run if not provided
if [[ -z "$RUN_DIR" ]]; then
  # Prefer /mnt/data path; fallback to repo outputs
  if compgen -G "/mnt/data/employment-act/outputs/complete_*" > /dev/null; then
    RUN_DIR=$(ls -dt /mnt/data/employment-act/outputs/complete_* | head -1)
  elif compgen -G "outputs/complete_*" > /dev/null; then
    RUN_DIR=$(ls -dt outputs/complete_* | head -1)
  else
    echo "No complete_* runs found. Provide --run-dir explicitly." >&2
    exit 1
  fi
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run dir not found: $RUN_DIR" >&2
  exit 1
fi

FAISS="$RUN_DIR/data/indices/faiss.index"
STORE="$RUN_DIR/data/indices/store.pkl"

if [[ ! -f "$FAISS" || ! -f "$STORE" ]]; then
  echo "Missing indices in run dir. Expected: $FAISS and $STORE" >&2
  exit 1
fi

# Determine model artifact based on mode
ADAPTER=""
MERGED=""
if [[ "$MODE" == "lora" ]]; then
  # pick newest lora_sft dir under sft/
  if compgen -G "$RUN_DIR/sft/sft_*/lora_sft" > /dev/null; then
    ADAPTER=$(ls -dt "$RUN_DIR"/sft/sft_*/lora_sft | head -1)
  fi
  if [[ -z "$ADAPTER" || ! -d "$ADAPTER" ]]; then
    echo "LoRA adapter dir not found under $RUN_DIR/sft. Use --mode merged if you merged weights." >&2
    exit 1
  fi
else
  # merged model path
  if [[ -d "$RUN_DIR/merged_model" ]]; then
    MERGED="$RUN_DIR/merged_model"
  else
    echo "Merged model dir not found at $RUN_DIR/merged_model. Use --mode lora or merge weights first." >&2
    exit 1
  fi
fi

ENV_FILE=".env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo ".env not found in repo root ($PROJECT_ROOT)." >&2
  exit 1
fi

# Backup .env
STAMP=$(date +%Y%m%d_%H%M%S)
cp "$ENV_FILE" ".env.bak.$STAMP"

echo "Updating .env → $RUN_DIR"

# Safe in-place edits: replace if present; else append
update_kv() {
  local key="$1"; shift
  local val="$1"; shift
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i -E "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '%s\n' "${key}=${val}" >> "$ENV_FILE"
  fi
}

update_kv FAISS_INDEX_PATH "$FAISS"
update_kv STORE_PATH "$STORE"

if [[ -n "$ADAPTER" ]]; then
  update_kv ADAPTER_PATH "$ADAPTER"
  # Keep MODEL_NAME as base; leave as-is
else
  update_kv MODEL_NAME "$MERGED"
  # Clear adapter if set
  if grep -qE '^ADAPTER_PATH=' "$ENV_FILE"; then
    sed -i -E 's|^ADAPTER_PATH=.*|ADAPTER_PATH=|' "$ENV_FILE"
  fi
fi

echo "✓ .env updated. Backup at .env.bak.$STAMP"

if [[ "$RESTART" -eq 1 ]]; then
  echo "Restarting stack (vLLM + API)..."
  VLLM_WAIT_SECS=${VLLM_WAIT_SECS:-1200} ./scripts/run_stack_docker.sh up
  echo "Done. Check health:"
  echo "  curl -sf http://localhost:${API_PORT:-8018}/health && echo API OK"
fi

