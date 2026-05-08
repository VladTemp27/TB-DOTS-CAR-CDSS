#!/usr/bin/env bash
# dev.sh — start MedGemma backend + React frontend together
# Usage: ./dev.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
WEBAPP="$ROOT/web-app"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[dev]${RESET} $*"; }
ok()   { echo -e "${GREEN}[dev]${RESET} $*"; }
warn() { echo -e "${YELLOW}[dev]${RESET} $*"; }
err()  { echo -e "${RED}[dev]${RESET} $*"; }

# ── cleanup on exit ──────────────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  log "Shutting down..."
  [[ -n "$BACKEND_PID" ]]  && kill "$BACKEND_PID"  2>/dev/null && log "Backend stopped"
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null && log "Frontend stopped"
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── preflight checks ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}TB-DOTS-CAR-CDSS Dev Environment${RESET}"
echo "────────────────────────────────────"

# Python
if ! command -v python3 &>/dev/null; then
  err "python3 not found. Install Python 3.10+."
  exit 1
fi

# Node
if ! command -v node &>/dev/null; then
  err "node not found. Install Node.js 18+."
  exit 1
fi

# GGUF model
MODEL="$ROOT/models/medgemma-1.5-4b-it-IQ4_XS.gguf"
if [[ ! -f "$MODEL" ]]; then
  warn "Model file not found: $MODEL"
  warn "MedGemma backend will start but show errors until the model is present."
fi

# ── install backend deps if needed ───────────────────────────────────────────
if ! python3 -c "import llama_cpp" &>/dev/null 2>&1; then
  log "Installing backend dependencies (first run may take a few minutes)..."
  pip install -r "$BACKEND/requirements.txt" --quiet
  ok "Backend dependencies installed"
fi

# ── install frontend deps if needed ──────────────────────────────────────────
if [[ ! -d "$WEBAPP/node_modules" ]]; then
  log "Installing frontend dependencies..."
  npm install --prefix "$WEBAPP" --silent
  ok "Frontend dependencies installed"
fi

# ── start backend ─────────────────────────────────────────────────────────────
log "Starting MedGemma backend on port $BACKEND_PORT..."
cd "$BACKEND"
uvicorn main:app --port "$BACKEND_PORT" --log-level warning &
BACKEND_PID=$!

# wait for backend to become healthy (model load can take 10-30s)
log "Waiting for model to load..."
ATTEMPTS=0
MAX_ATTEMPTS=60  # 60 × 2s = 2 minutes max
until curl -sf "http://localhost:$BACKEND_PORT/api/health" | grep -q '"ready"'; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "Backend process died unexpectedly. Check uvicorn output above."
    exit 1
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  if [[ $ATTEMPTS -ge $MAX_ATTEMPTS ]]; then
    warn "Backend health check timed out after 2 minutes."
    warn "The app will still work — MedGemma card will show retry button."
    break
  fi
  printf "."
  sleep 2
done
echo ""
ok "MedGemma backend ready  →  http://localhost:$BACKEND_PORT"

# ── start frontend ────────────────────────────────────────────────────────────
log "Starting React frontend on port $FRONTEND_PORT..."
cd "$WEBAPP"
npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

# give Vite a moment to print its URL
sleep 2
echo ""
echo -e "${BOLD}────────────────────────────────────${RESET}"
ok "App running  →  http://localhost:$FRONTEND_PORT"
echo -e "${BOLD}────────────────────────────────────${RESET}"
echo ""
log "Press Ctrl+C to stop both servers"
echo ""

# ── keep alive ────────────────────────────────────────────────────────────────
wait
