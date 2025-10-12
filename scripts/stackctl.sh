#!/usr/bin/env bash
# Unified control for serving and artifacts
#
# Usage:
#   ./scripts/stackctl.sh up [--mode lora|merged] [--run-dir PATH] [--no-switch] [--wait SECS]
#   ./scripts/stackctl.sh down
#   ./scripts/stackctl.sh restart [--mode ...] [--run-dir ...] [--wait SECS]
#   ./scripts/stackctl.sh status | logs
#   ./scripts/stackctl.sh switch [--mode lora|merged] [--run-dir PATH] [--no-restart]
#   ./scripts/stackctl.sh check      # Docker + GPU checks
#   ./scripts/stackctl.sh health     # Curl vLLM/API health endpoints
#
# Examples:
#   ./scripts/stackctl.sh up                               # switch to latest run then start
#   ./scripts/stackctl.sh up --mode merged                 # use merged model if present
#   ./scripts/stackctl.sh up --run-dir /path/to/complete_* # target a specific run
#   ./scripts/stackctl.sh restart                          # cycle both services
#   ./scripts/stackctl.sh switch --no-restart              # only update .env

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

cmd=${1:-help}
shift || true

MODE="lora"
RUN_DIR=""
DO_SWITCH=1
WAIT_SECS=""
RESTART=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --run-dir) RUN_DIR="$2"; shift 2;;
    --no-switch) DO_SWITCH=0; shift;;
    --no-restart) RESTART=0; shift;;
    --wait) WAIT_SECS="$2"; shift 2;;
    *) break;;
  esac
done

source_env() {
  if [[ -f .env ]]; then
    set +u; set -a; source .env; set +a; set -u
  fi
}

case "$cmd" in
  up)
    if [[ $DO_SWITCH -eq 1 ]]; then
      args=( )
      [[ -n "$RUN_DIR" ]] && args+=(--run-dir "$RUN_DIR")
      [[ -n "$MODE" ]] && args+=(--mode "$MODE")
      ./scripts/switch_artifacts.sh "${args[@]}" --no-restart
    fi
    if [[ -n "$WAIT_SECS" ]]; then
      VLLM_WAIT_SECS="$WAIT_SECS" ./scripts/run_stack_docker.sh up
    else
      ./scripts/run_stack_docker.sh up
    fi
    ;;
  restart)
    ./scripts/run_stack_docker.sh down || true
    "$0" up "$@"
    ;;
  down)
    ./scripts/run_stack_docker.sh down
    ;;
  status)
    ./scripts/run_stack_docker.sh status
    ;;
  logs)
    ./scripts/run_stack_docker.sh logs
    ;;
  switch)
    args=( )
    [[ -n "$RUN_DIR" ]] && args+=(--run-dir "$RUN_DIR")
    [[ -n "$MODE" ]] && args+=(--mode "$MODE")
    [[ $RESTART -eq 0 ]] && args+=(--no-restart)
    ./scripts/switch_artifacts.sh "${args[@]}"
    ;;
  check)
    ./scripts/check_gpu_docker.sh
    ;;
  health)
    source_env
    VPORT=${VLLM_PORT:-8019}
    APORT=${API_PORT:-8018}
    echo "vLLM: http://localhost:$VPORT/health"
    curl -sf "http://localhost:$VPORT/health" && echo "vLLM OK" || echo "vLLM UNHEALTHY"
    echo "API:  http://localhost:$APORT/health"
    curl -sf "http://localhost:$APORT/health" && echo "API OK" || echo "API UNHEALTHY"
    ;;
  help|--help|-h|*)
    sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac

