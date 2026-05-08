#!/usr/bin/env bash
# dev.sh — start MedGemma backend + React frontend with a live dashboard
# Usage: ./dev.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
WEBAPP="$ROOT/web-app"
LOGS="$ROOT/logs"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── colors / ANSI ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[dev]${RESET} $*"; }
ok()   { echo -e "${GREEN}[dev]${RESET} $*"; }
warn() { echo -e "${YELLOW}[dev]${RESET} $*"; }
err()  { echo -e "${RED}[dev]${RESET} $*"; }

# ── log files ─────────────────────────────────────────────────────────────────
mkdir -p "$LOGS"
BACKEND_LOG="$LOGS/backend.log"
FRONTEND_LOG="$LOGS/frontend.log"
: > "$BACKEND_LOG"   # truncate on each run
: > "$FRONTEND_LOG"

# ── cleanup on exit ───────────────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  tput cnorm 2>/dev/null || true   # restore cursor
  echo ""
  log "Shutting down..."
  [[ -n "$BACKEND_PID" ]]  && kill "$BACKEND_PID"  2>/dev/null; log "Backend stopped"
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null; log "Frontend stopped"
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── preflight checks ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}TB-DOTS-CAR-CDSS Dev Environment${RESET}"
echo "────────────────────────────────────"

if ! command -v python3 &>/dev/null; then
  err "python3 not found. Install Python 3.10+."
  exit 1
fi
if ! command -v node &>/dev/null; then
  err "node not found. Install Node.js 18+."
  exit 1
fi

MODEL="$ROOT/models/medgemma-1.5-4b-it-IQ4_XS.gguf"
if [[ ! -f "$MODEL" ]]; then
  warn "Model file not found: $MODEL"
  warn "Backend will start but MedGemma card will show an error."
fi

# ── install deps if needed ────────────────────────────────────────────────────
if ! python3 -c "import llama_cpp" &>/dev/null 2>&1; then
  log "Installing backend dependencies..."
  pip install -r "$BACKEND/requirements.txt" --quiet
  ok "Backend dependencies installed"
fi
if [[ ! -d "$WEBAPP/node_modules" ]]; then
  log "Installing frontend dependencies..."
  npm install --prefix "$WEBAPP" --silent
  ok "Frontend dependencies installed"
fi

# ── start processes (output → log files) ─────────────────────────────────────
log "Starting MedGemma backend on :$BACKEND_PORT  (log → logs/backend.log)"
cd "$BACKEND"
uvicorn main:app --port "$BACKEND_PORT" --log-level info >> "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

log "Starting React frontend on :$FRONTEND_PORT  (log → logs/frontend.log)"
cd "$WEBAPP"
npm run dev -- --port "$FRONTEND_PORT" >> "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

# ── wait for backend health ───────────────────────────────────────────────────
log "Waiting for model to load..."
ATTEMPTS=0
MAX_ATTEMPTS=60
until curl -sf "http://localhost:$BACKEND_PORT/api/health" | grep -q '"ready"\|"loading"\|"error"'; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "Backend process died. Check logs/backend.log"
    exit 1
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  [[ $ATTEMPTS -ge $MAX_ATTEMPTS ]] && { warn "Health check timed out."; break; }
  printf "."
  sleep 2
done
echo ""

# ── helper: draw dashboard ────────────────────────────────────────────────────
draw_dashboard() {
  local W
  W=$(tput cols 2>/dev/null || echo 80)
  local DIVIDER
  DIVIDER=$(printf '─%.0s' $(seq 1 "$W"))

  # ── fetch stats ──
  local raw=""
  raw=$(curl -sf "http://localhost:$BACKEND_PORT/api/stats" 2>/dev/null || echo "")

  local be_status="unreachable"
  local ram_mb=0
  local inference_active="false"
  local requests_served=0
  local requests_active=0
  local last_patient="—"
  local last_tokens=0
  local last_elapsed=0
  local load_time=0

  if [[ -n "$raw" ]]; then
    be_status=$(echo "$raw"       | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
    ram_mb=$(echo "$raw"          | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ram_mb',0))" 2>/dev/null || echo 0)
    inference_active=$(echo "$raw"| python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('inference_active',False)).lower())" 2>/dev/null || echo false)
    requests_served=$(echo "$raw" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('requests_served',0))" 2>/dev/null || echo 0)
    requests_active=$(echo "$raw" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('requests_active',0))" 2>/dev/null || echo 0)
    last_patient=$(echo "$raw"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('last_patient') or '—')" 2>/dev/null || echo "—")
    last_tokens=$(echo "$raw"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('last_tokens',0))" 2>/dev/null || echo 0)
    last_elapsed=$(echo "$raw"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('last_elapsed_s',0))" 2>/dev/null || echo 0)
    load_time=$(echo "$raw"       | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_load_time_s',0))" 2>/dev/null || echo 0)
  fi

  # ── backend proc RAM (RSS) via ps ──
  local proc_ram="—"
  if kill -0 "$BACKEND_PID" 2>/dev/null; then
    local rss_kb
    rss_kb=$(ps -o rss= -p "$BACKEND_PID" 2>/dev/null | tr -d ' ' || echo 0)
    proc_ram="$(( rss_kb / 1024 )) MB"
  fi

  # ── status badges ──
  local be_badge
  case "$be_status" in
    ready)       be_badge="${GREEN}● ready${RESET}" ;;
    loading)     be_badge="${YELLOW}◌ loading${RESET}" ;;
    error)       be_badge="${RED}✖ error${RESET}" ;;
    unreachable) be_badge="${RED}✖ unreachable${RESET}" ;;
    *)           be_badge="${DIM}? ${be_status}${RESET}" ;;
  esac

  local fe_badge
  if kill -0 "$FRONTEND_PID" 2>/dev/null; then
    fe_badge="${GREEN}● running${RESET}"
  else
    fe_badge="${RED}✖ stopped${RESET}"
  fi

  local infer_badge
  if [[ "$inference_active" == "true" ]]; then
    infer_badge="${MAGENTA}⬤ GENERATING${RESET}"
  else
    infer_badge="${DIM}○ idle${RESET}"
  fi

  # ── last request tok/s ──
  local tok_per_s="—"
  if [[ "$last_elapsed" != "0" && "$last_elapsed" != "0.0" && "$last_tokens" -gt 0 ]]; then
    tok_per_s=$(python3 -c "print(f'{$last_tokens/$last_elapsed:.1f}')" 2>/dev/null || echo "?")
  fi

  # ── recent log lines (last 5) ──
  local be_tail=""
  be_tail=$(tail -n 5 "$BACKEND_LOG" 2>/dev/null | sed 's/^/  /' || true)

  # ── draw ──
  clear
  echo -e "${BOLD}  TB-DOTS-CAR-CDSS  ·  Dev Dashboard${RESET}  ${DIM}$(date '+%H:%M:%S')${RESET}"
  echo -e "  ${DIM}$DIVIDER${RESET}"

  echo ""
  echo -e "  ${BOLD}${CYAN}SERVICES${RESET}"
  printf "  %-22s %b\n" "MedGemma Backend" "$be_badge"
  printf "  %-22s %b\n" "React Frontend"   "$fe_badge"

  echo ""
  echo -e "  ${BOLD}${CYAN}AI BACKEND${RESET}"
  printf "  %-22s %b\n"  "Inference"       "$infer_badge"
  printf "  %-22s %s\n"  "Process RAM"     "$proc_ram"
  printf "  %-22s %s MB\n" "App-reported RAM" "$ram_mb"
  printf "  %-22s %s s\n"  "Model load time"  "$load_time"

  echo ""
  echo -e "  ${BOLD}${CYAN}REQUESTS${RESET}"
  printf "  %-22s %s\n"  "Total served"    "$requests_served"
  printf "  %-22s %s\n"  "Active"          "$requests_active"
  printf "  %-22s %s\n"  "Last patient"    "$last_patient"
  printf "  %-22s %s tokens in %s s (%s tok/s)\n" \
    "Last generation" "$last_tokens" "$last_elapsed" "$tok_per_s"

  echo ""
  echo -e "  ${BOLD}${CYAN}RECENT BACKEND LOGS${RESET}"
  if [[ -n "$be_tail" ]]; then
    echo -e "${DIM}${be_tail}${RESET}"
  else
    echo -e "  ${DIM}(no log output yet)${RESET}"
  fi

  echo ""
  echo -e "  ${DIM}$DIVIDER${RESET}"
  echo -e "  ${DIM}Backend → http://localhost:$BACKEND_PORT   Frontend → http://localhost:$FRONTEND_PORT${RESET}"
  echo -e "  ${DIM}Full logs: logs/backend.log  |  logs/frontend.log  |  Ctrl+C to stop${RESET}"
}

# ── dashboard loop ────────────────────────────────────────────────────────────
tput civis 2>/dev/null || true   # hide cursor during loop

while true; do
  draw_dashboard
  sleep 3
done
