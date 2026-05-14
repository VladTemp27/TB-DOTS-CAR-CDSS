#!/usr/bin/env bash
# dev.sh — start backend API + React frontend with a live dashboard
# Usage: ./dev.sh [--from-source]
#   --from-source  compile llama-cpp-python from C++ source instead of using the
#                  prebuilt Metal/CUDA wheel (slower; only needed for unsupported
#                  Python versions or custom CUDA architectures)

set -Eeuo pipefail

# ── constants ─────────────────────────────────────────────────────────────────
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BACKEND="$ROOT/backend"
readonly WEBAPP="$ROOT/web-app"
readonly BACKEND_PORT=8000
readonly FRONTEND_PORT=5173
readonly VENV="$ROOT/.venv"
readonly PYTHON="$VENV/bin/python"
readonly PIP="$VENV/bin/pip"
readonly UVICORN="$VENV/bin/uvicorn"

# Pinned prebuilt-wheel versions (update when abetlen/llama-cpp-python releases)
readonly LLAMA_VER_METAL="0.3.23"   # Metal index — latest May 2026
readonly LLAMA_VER_CU124="0.3.22"   # CUDA 12.4 — driver >= 550
readonly LLAMA_VER_CU121="0.3.23"   # CUDA 12.1 — driver >= 530 (fallback)

FROM_SOURCE=false
[[ "${1:-}" == "--from-source" ]] && FROM_SOURCE=true

# ── colors / ANSI ─────────────────────────────────────────────────────────────
is_tty() { [[ -t 1 ]]; }

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[dev]${RESET} $*"; }
ok()   { echo -e "${GREEN}[dev]${RESET} $*"; }
warn() { echo -e "${YELLOW}[dev]${RESET} $*"; }
err()  { echo -e "${RED}[dev]${RESET} $*" >&2; }

# ── log rotation ──────────────────────────────────────────────────────────────
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
_TS="$(date +%Y%m%d-%H%M%S)"
BACKEND_LOG="$LOG_DIR/backend-${_TS}.log"
FRONTEND_LOG="$LOG_DIR/frontend-${_TS}.log"
find "$LOG_DIR" \( -name 'backend-*.log' -o -name 'frontend-*.log' \) \
  -mtime +7 -delete 2>/dev/null || true

# ── process tracking ──────────────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""
CLEANED=0

# ── cleanup (EXIT trap — fires on every exit path) ────────────────────────────
cleanup() {
  [[ $CLEANED -eq 1 ]] && return
  CLEANED=1
  is_tty && { tput cnorm 2>/dev/null || true; }
  echo ""
  log "Shutting down..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    pkill -P "$BACKEND_PID" 2>/dev/null || true
    log "Backend stopped (PID $BACKEND_PID)"
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    pkill -P "$FRONTEND_PID" 2>/dev/null || true
    log "Frontend stopped (PID $FRONTEND_PID)"
  fi
}
trap cleanup EXIT

# ── preflight ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}TB-DOTS-CAR-CDSS Dev Environment${RESET}"
echo "────────────────────────────────────"

if [[ ! -x "$PYTHON" ]]; then
  log "Virtualenv not found at $VENV"
  log "Run: python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

# Prebuilt wheels exist only for cp310/cp311/cp312 — gate hard
if ! "$PYTHON" -c \
  'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
  PYVER="$("$PYTHON" -V 2>&1)"
  err "llama-cpp-python prebuilt wheels require Python 3.10–3.12. Detected: $PYVER"
  err "Recreate venv:  python3.12 -m venv .venv"
  exit 1
fi

for _cmd in node curl; do
  command -v "$_cmd" &>/dev/null || { err "$_cmd not found."; exit 1; }
done

if [[ "$(uname -s)" == "Darwin" ]] && ! xcode-select -p &>/dev/null; then
  err "Xcode Command Line Tools missing (required for any C++ source builds)."
  err "Run: xcode-select --install"
  exit 1
fi

# ── model detection ───────────────────────────────────────────────────────────
MODEL="$ROOT/models/medgemma-1.5-4b-it-IQ4_XS.gguf"
MODEL_PRESENT=false
if [[ -f "$MODEL" ]]; then
  MODEL_PRESENT=true
else
  warn "Model file not found: $MODEL"
  warn "Backend will start without MedGemma inference (llama-cpp-python skipped)."
fi

# ── base backend deps (everything except llama-cpp-python) ────────────────────
if ! "$PYTHON" -c "import fastapi,uvicorn,sqlalchemy,alembic" &>/dev/null 2>&1; then
  log "Installing backend dependencies..."
  _TMP=$(mktemp -t devsh.XXXXXXXX)
  grep -v 'llama-cpp-python' "$BACKEND/requirements.txt" > "$_TMP"
  "$PIP" install -r "$_TMP" --quiet
  rm -f "$_TMP"
  ok "Backend dependencies installed"
fi

# ── NVIDIA library path (Linux only, derived dynamically) ─────────────────────
if [[ "$(uname -s)" == "Linux" ]]; then
  _PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  _EXTRA=""
  for _pkg in cuda_runtime cublas cuda_nvrtc; do
    _D="$VENV/lib/python${_PYVER}/site-packages/nvidia/$_pkg/lib"
    [[ -d "$_D" ]] && _EXTRA="${_EXTRA}${_D}:"
  done
  _PYLIB=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || true)
  [[ -n "$_PYLIB" && -d "$_PYLIB" ]] && _EXTRA="${_EXTRA}${_PYLIB}:"
  [[ -n "$_EXTRA" ]] && \
    export LD_LIBRARY_PATH="${_EXTRA%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ── llama-cpp-python ──────────────────────────────────────────────────────────
# Default: prebuilt wheel (seconds). Opt-in: --from-source (minutes).
if [[ "$MODEL_PRESENT" == "true" ]] && \
   ! "$PYTHON" -c "import llama_cpp" &>/dev/null 2>&1; then

  if [[ "$FROM_SOURCE" == "true" ]]; then
    # Source build — GGML_NATIVE=OFF eliminates the i8mm CMake-probe hang
    _LLAMA_LOG=$(mktemp -t devsh.XXXXXXXX)
    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
      log "Source build: Metal + GGML_NATIVE=OFF (avoids i8mm probe hang)..."
      CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=ON \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_APPLE_SILICON_PROCESSOR=arm64" \
        "$PIP" install "llama-cpp-python>=0.3.0" \
          --no-binary llama-cpp-python --no-cache-dir -v \
          >"$_LLAMA_LOG" 2>&1 &
    else
      log "Source build: CUDA + GGML_NATIVE=OFF..."
      CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=all-major \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF" \
      FORCE_CMAKE=1 \
        "$PIP" install "llama-cpp-python>=0.3.0" \
          --no-binary llama-cpp-python --no-cache-dir -v \
          >"$_LLAMA_LOG" 2>&1 &
    fi
    _LLAMA_PID=$!
    _SPIN=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    _T=0; _STALL=0; _PREV=""; _WARNED=false
    while kill -0 "$_LLAMA_PID" 2>/dev/null; do
      _LAST=$(tail -1 "$_LLAMA_LOG" 2>/dev/null \
        | tr -d '\000-\037\177' \
        | sed 's/\x1b\[[0-9;]*[mGKJH]//g' \
        | cut -c1-60)
      printf "\r  ${YELLOW}%s${RESET} ${DIM}%ds${RESET}  %-62s" \
        "${_SPIN[$((_T % 10))]}" "$_T" "$_LAST"
      if [[ "$_LAST" == "$_PREV" ]]; then
        _STALL=$((_STALL+1))
      else
        _STALL=0; _PREV="$_LAST"; _WARNED=false
      fi
      if [[ $_STALL -ge 120 && "$_WARNED" == "false" ]]; then
        printf "\r%${COLUMNS:-80}s\r" ""
        warn "Build stalled 2 min on same line — CMake probe hang?"
        warn "Live log: tail -f $_LLAMA_LOG"
        _WARNED=true
      fi
      if [[ $_T -ge 900 ]]; then
        printf "\r%${COLUMNS:-80}s\r" ""
        err "Build timed out (900s) — killing. Last 20 lines:"
        kill "$_LLAMA_PID" 2>/dev/null
        tail -20 "$_LLAMA_LOG" | sed 's/^/  /'
        rm -f "$_LLAMA_LOG"; exit 1
      fi
      sleep 1; _T=$((_T+1))
    done
    wait "$_LLAMA_PID"; _EXIT=$?
    printf "\r%${COLUMNS:-80}s\r" ""
    if [[ $_EXIT -eq 0 ]]; then
      ok "llama-cpp-python compiled from source in ${_T}s"
    else
      err "Source build failed (${_T}s). Last 30 lines:"
      tail -30 "$_LLAMA_LOG" | sed 's/^/  /'
      rm -f "$_LLAMA_LOG"; exit 1
    fi
    rm -f "$_LLAMA_LOG"

  else
    # Prebuilt wheel — try each version in order, auto-fallback to source on all failures
    log "Installing llama-cpp-python prebuilt wheel..."
    _PIP_ERR=$(mktemp -t devsh.XXXXXXXX)
    _WHEEL_OK=false

    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
      # Try pinned versions in descending order (server-side corruption can affect any release)
      for _VER in "$LLAMA_VER_METAL" "0.3.22" "0.3.21"; do
        log "Trying Metal wheel v${_VER}..."
        if "$PIP" install --prefer-binary --no-cache-dir --quiet \
            --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/metal" \
            "llama-cpp-python==${_VER}" 2>"$_PIP_ERR"; then
          _WHEEL_OK=true; break
        fi
        warn "v${_VER} failed: $(tail -1 "$_PIP_ERR" | tr -d '\n')"
      done
    elif [[ "$(uname -s)" == "Linux" ]]; then
      if "$PIP" install --prefer-binary --no-cache-dir --quiet \
          --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu121" \
          "llama-cpp-python==${LLAMA_VER_CU121}" 2>"$_PIP_ERR"; then
        _WHEEL_OK=true
      elif "$PIP" install --prefer-binary --no-cache-dir --quiet \
          --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu124" \
          "llama-cpp-python==${LLAMA_VER_CU124}" 2>"$_PIP_ERR"; then
        _WHEEL_OK=true
      fi
    else
      "$PIP" install "llama-cpp-python>=0.3.0" --quiet 2>"$_PIP_ERR" && _WHEEL_OK=true
    fi

    rm -f "$_PIP_ERR"

    if [[ "$_WHEEL_OK" == "false" ]]; then
      warn "All prebuilt wheels failed (likely server-side corruption). Falling back to source build."
      warn "Using GGML_NATIVE=OFF to skip the i8mm CMake probe hang (~3 min on M1 Air)..."
      _LLAMA_LOG=$(mktemp -t devsh.XXXXXXXX)
      CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=ON \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_APPLE_SILICON_PROCESSOR=arm64" \
        "$PIP" install "llama-cpp-python>=0.3.0" \
          --no-binary llama-cpp-python --no-cache-dir -v \
          >"$_LLAMA_LOG" 2>&1 &
      _LLAMA_PID=$!
      _SPIN=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
      _T=0
      while kill -0 "$_LLAMA_PID" 2>/dev/null; do
        _LAST=$(tail -1 "$_LLAMA_LOG" 2>/dev/null \
          | tr -d '\000-\037\177' | sed 's/\x1b\[[0-9;]*[mGKJH]//g' | cut -c1-60)
        printf "\r  ${YELLOW}%s${RESET} ${DIM}%ds${RESET}  %-62s" \
          "${_SPIN[$((_T % 10))]}" "$_T" "$_LAST"
        sleep 1; _T=$((_T+1))
      done
      wait "$_LLAMA_PID"; _SRC_EXIT=$?
      printf "\r%${COLUMNS:-80}s\r" ""
      if [[ $_SRC_EXIT -ne 0 ]]; then
        err "Source build also failed. Last 20 lines:"
        tail -20 "$_LLAMA_LOG" | sed 's/^/  /'
        rm -f "$_LLAMA_LOG"; exit 1
      fi
      rm -f "$_LLAMA_LOG"
    fi

    ok "llama-cpp-python installed"
  fi
fi

# ── frontend deps ─────────────────────────────────────────────────────────────
if [[ ! -d "$WEBAPP/node_modules" ]] \
   || [[ "$WEBAPP/package.json"      -nt "$WEBAPP/node_modules" ]] \
   || [[ "$WEBAPP/package-lock.json" -nt "$WEBAPP/node_modules" ]]; then
  log "Installing frontend dependencies..."
  npm install --prefix "$WEBAPP" --silent
  ok "Frontend dependencies installed"
fi

# ── database: migrate + seed ─────────────────────────────────────────────────
log "Running database migrations..."
PYTHONPATH="$ROOT" "$PYTHON" -c "
from pathlib import Path
from alembic import command
from alembic.config import Config
backend = Path('$BACKEND')
cfg = Config(str(backend / 'alembic.ini'))
cfg.set_main_option('script_location', str(backend / 'alembic'))
command.upgrade(cfg, 'head')
"
ok "Migrations up to date"

_DB_EMPTY=false
if PYTHONPATH="$ROOT" "$PYTHON" -c "
import sys
from sqlalchemy import select
from backend.db import session_factory
from backend.models import Patient as PatientRow
db = session_factory()()
try:
    existing = db.scalar(select(PatientRow.id).limit(1))
    sys.exit(0 if existing is None else 1)
finally:
    db.close()
" 2>/dev/null; then
  _DB_EMPTY=true
fi

if [[ "$_DB_EMPTY" == "true" ]]; then
  log "Database is empty — seeding demo data..."
  (cd "$ROOT" && "$PYTHON" -m backend.seed_demo --include-xrays)
  ok "Demo data seeded"
else
  ok "Database already has data — skipping seed"
fi

# ── start services ────────────────────────────────────────────────────────────
log "Starting backend API on :$BACKEND_PORT  (log → logs/backend-${_TS}.log)"
"$UVICORN" backend.main:app --port "$BACKEND_PORT" --log-level info \
  >> "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

log "Starting React frontend on :$FRONTEND_PORT  (log → logs/frontend-${_TS}.log)"
(cd "$WEBAPP" && npm run dev -- --port "$FRONTEND_PORT") \
  >> "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

# ── wait for backend health ───────────────────────────────────────────────────
if [[ "$MODEL_PRESENT" == "true" ]]; then
  log "Waiting for model to load..."
else
  log "Waiting for backend to start..."
fi
_ATTEMPTS=0; _MAX=60
while true; do
  _body=$(curl -s --max-time 2 "http://localhost:$BACKEND_PORT/api/health" 2>/dev/null \
    || echo '{}')
  if "$PYTHON" -c "
import sys, json
d = json.loads(sys.stdin.read())
sys.exit(0 if d.get('status') in ('ready','loading','error') else 1)
" <<< "$_body" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${BACKEND_PID:-}" 2>/dev/null; then
    err "Backend process died — check logs/backend-${_TS}.log"; exit 1
  fi
  _ATTEMPTS=$((_ATTEMPTS+1))
  if [[ $_ATTEMPTS -ge $_MAX ]]; then warn "Health check timed out."; break; fi
  printf "."; sleep 2
done
echo ""

# ── dashboard: parse all stats in one Python call ────────────────────────────
parse_stats() {
  # reads JSON from stdin, prints 9 tab-separated fields
  "$PYTHON" -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    fields = [
        d.get('status', '?'),
        str(d.get('ram_mb', 0)),
        str(d.get('inference_active', False)).lower(),
        str(d.get('requests_served', 0)),
        str(d.get('requests_active', 0)),
        d.get('last_patient') or '—',
        str(d.get('last_tokens', 0)),
        str(d.get('last_elapsed_s', 0)),
        str(d.get('model_load_time_s', 0)),
    ]
    print('\t'.join(fields))
except Exception:
    print('?\t0\tfalse\t0\t0\t—\t0\t0\t0')
" 2>/dev/null || echo "?	0	false	0	0	—	0	0	0"
}

draw_dashboard() {
  local W; W=$(tput cols 2>/dev/null || echo 80)
  local DIVIDER; DIVIDER=$(printf '─%.0s' $(seq 1 "$W"))

  # single curl + single Python parse (replaces 8 separate subprocesses)
  local raw; raw=$(curl -s --max-time 2 \
    "http://localhost:$BACKEND_PORT/api/stats" 2>/dev/null || echo '{}')
  local be_status ram_mb inference_active requests_served requests_active \
        last_patient last_tokens last_elapsed load_time
  IFS=$'\t' read -r be_status ram_mb inference_active requests_served \
    requests_active last_patient last_tokens last_elapsed load_time \
    < <(echo "$raw" | parse_stats) || true

  # process RAM via ps
  local proc_ram="—"
  if kill -0 "${BACKEND_PID:-}" 2>/dev/null; then
    local rss_kb
    rss_kb=$(ps -o rss= -p "$BACKEND_PID" 2>/dev/null | tr -d ' ' || echo 0)
    proc_ram="$(( ${rss_kb:-0} / 1024 )) MB"
  fi

  # status badges
  local be_badge
  case "${be_status:-}" in
    ready)       be_badge="${GREEN}● ready${RESET}" ;;
    loading)     be_badge="${YELLOW}◌ loading${RESET}" ;;
    error)       be_badge="${RED}✖ error${RESET}" ;;
    unreachable) be_badge="${RED}✖ unreachable${RESET}" ;;
    *)           be_badge="${DIM}? ${be_status}${RESET}" ;;
  esac

  local fe_badge
  kill -0 "${FRONTEND_PID:-}" 2>/dev/null \
    && fe_badge="${GREEN}● running${RESET}" \
    || fe_badge="${RED}✖ stopped${RESET}"

  local infer_badge
  [[ "${inference_active:-false}" == "true" ]] \
    && infer_badge="${MAGENTA}⬤ GENERATING${RESET}" \
    || infer_badge="${DIM}○ idle${RESET}"

  local tok_per_s="—"
  if [[ "${last_elapsed:-0}" != "0" && "${last_elapsed:-0}" != "0.0" \
        && "${last_tokens:-0}" -gt 0 ]] 2>/dev/null; then
    tok_per_s=$("$PYTHON" -c \
      "print(f'{${last_tokens}/${last_elapsed}:.1f}')" 2>/dev/null || echo "?")
  fi

  local be_tail=""
  be_tail=$(tail -n 5 "$BACKEND_LOG" 2>/dev/null | sed 's/^/  /' || true)

  # Detect actual bound ports from process logs (Vite may auto-increment if configured port is busy)
  local actual_be_port actual_fe_port
  actual_be_port=$(grep -m1 'Uvicorn running on' "$BACKEND_LOG" 2>/dev/null \
    | grep -oE ':[0-9]+' | tail -1 | tr -d ':')
  [[ -z "$actual_be_port" ]] && actual_be_port="$BACKEND_PORT"
  actual_fe_port=$(grep -m1 'localhost:' "$FRONTEND_LOG" 2>/dev/null \
    | grep -oE 'localhost:[0-9]+' | grep -oE '[0-9]+$')
  [[ -z "$actual_fe_port" ]] && actual_fe_port="$FRONTEND_PORT"

  local be_port_str fe_port_str
  be_port_str="${DIM}:${actual_be_port}${RESET}"
  [[ "$actual_be_port" != "$BACKEND_PORT" ]] && \
    be_port_str="${YELLOW}:${actual_be_port}${RESET} ${DIM}(cfg :${BACKEND_PORT})${RESET}"
  fe_port_str="${DIM}:${actual_fe_port}${RESET}"
  [[ "$actual_fe_port" != "$FRONTEND_PORT" ]] && \
    fe_port_str="${YELLOW}:${actual_fe_port}${RESET} ${DIM}(cfg :${FRONTEND_PORT})${RESET}"

  clear
  echo -e "${BOLD}  TB-DOTS-CAR-CDSS  ·  Dev Dashboard${RESET}  ${DIM}$(date '+%H:%M:%S')${RESET}"
  echo -e "  ${DIM}$DIVIDER${RESET}"

  echo ""
  echo -e "  ${BOLD}${CYAN}SERVICES${RESET}"
  printf "  %-22s %b  %b\n" "Backend API"    "$be_badge" "$be_port_str"
  printf "  %-22s %b  %b\n" "React Frontend" "$fe_badge" "$fe_port_str"

  echo ""
  echo -e "  ${BOLD}${CYAN}AI BACKEND${RESET}"
  printf "  %-22s %b\n"    "Inference"        "$infer_badge"
  printf "  %-22s %s\n"    "Process RAM"      "$proc_ram"
  printf "  %-22s %s MB\n" "App-reported RAM" "${ram_mb:-0}"
  printf "  %-22s %s s\n"  "Model load time"  "${load_time:-0}"

  echo ""
  echo -e "  ${BOLD}${CYAN}REQUESTS${RESET}"
  printf "  %-22s %s\n" "Total served"    "${requests_served:-0}"
  printf "  %-22s %s\n" "Active"          "${requests_active:-0}"
  printf "  %-22s %s\n" "Last patient"    "${last_patient:-—}"
  printf "  %-22s %s tokens in %s s (%s tok/s)\n" \
    "Last generation" "${last_tokens:-0}" "${last_elapsed:-0}" "$tok_per_s"

  echo ""
  echo -e "  ${BOLD}${CYAN}RECENT BACKEND LOGS${RESET}"
  if [[ -n "$be_tail" ]]; then
    echo -e "${DIM}${be_tail}${RESET}"
  else
    echo -e "  ${DIM}(no log output yet)${RESET}"
  fi

  echo ""
  echo -e "  ${DIM}$DIVIDER${RESET}"
  echo -e "  ${DIM}Backend → http://localhost:${actual_be_port}   Frontend → http://localhost:${actual_fe_port}${RESET}"
  echo -e "  ${DIM}logs/backend-${_TS}.log  |  logs/frontend-${_TS}.log  |  Ctrl+C to stop${RESET}"
}

# ── dashboard loop ────────────────────────────────────────────────────────────
is_tty && { tput civis 2>/dev/null || true; }   # hide cursor during loop

while true; do
  draw_dashboard
  sleep 3
  if ! kill -0 "${BACKEND_PID:-}" 2>/dev/null \
     && ! kill -0 "${FRONTEND_PID:-}" 2>/dev/null; then
    err "Both services have stopped. Check logs/backend-${_TS}.log and logs/frontend-${_TS}.log"
    break
  fi
done
