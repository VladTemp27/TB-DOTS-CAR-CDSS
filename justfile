# justfile — human entry point for TB-DOTS-CAR-CDSS
# Install: brew install just   |   https://just.systems
# Usage:   just [recipe]

set shell := ["bash", "-Eeuo", "pipefail", "-c"]

root     := justfile_directory()
venv     := root / ".venv"
python   := venv / "bin/python"
pip      := venv / "bin/pip"
webapp   := root / "web-app"

# Pinned wheel versions (sync with dev.sh when updating)
llama_metal  := "0.3.23"
llama_cu124  := "0.3.22"
llama_cu121  := "0.3.23"

# ── default: show available recipes ──────────────────────────────────────────
default:
    @just --list

# ── dev: start everything (delegates to process-compose or dev.py) ───────────
dev:
    #!/usr/bin/env bash
    if command -v process-compose &>/dev/null; then
        exec process-compose up
    else
        echo "[just] process-compose not found — falling back to dev.py"
        exec "{{python}}" "{{root}}/dev.py"
    fi

# ── install: create venv + install all deps ───────────────────────────────────
install: _venv _base-deps install-llm _frontend-deps

_venv:
    #!/usr/bin/env bash
    if [[ ! -x "{{python}}" ]]; then
        python3.12 -m venv "{{venv}}"
        echo "[just] virtualenv created at {{venv}}"
    fi

_base-deps:
    #!/usr/bin/env bash
    if ! "{{python}}" -c "import fastapi,uvicorn,sqlalchemy,alembic" &>/dev/null 2>&1; then
        TMP=$(mktemp)
        grep -v 'llama-cpp-python' "{{root}}/backend/requirements.txt" > "$TMP"
        "{{pip}}" install -r "$TMP" --quiet
        rm -f "$TMP"
        echo "[just] base backend deps installed"
    fi

_frontend-deps:
    #!/usr/bin/env bash
    if [[ ! -d "{{webapp}}/node_modules" ]] \
       || [[ "{{webapp}}/package.json"      -nt "{{webapp}}/node_modules" ]] \
       || [[ "{{webapp}}/package-lock.json" -nt "{{webapp}}/node_modules" ]]; then
        npm install --prefix "{{webapp}}" --silent
        echo "[just] frontend deps installed"
    fi

# ── install-llm: install llama-cpp-python prebuilt wheel ─────────────────────
# Pass from-source=true to compile from C++ source instead
install-llm from-source="false":
    #!/usr/bin/env bash
    if "{{python}}" -c "import llama_cpp" &>/dev/null 2>&1; then
        echo "[just] llama-cpp-python already installed — skipping"
        exit 0
    fi
    if [[ "{{from-source}}" == "true" ]]; then
        echo "[just] Source build: GGML_NATIVE=OFF (avoids i8mm CMake-probe hang)"
        if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
            CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=ON \
              -DCMAKE_OSX_ARCHITECTURES=arm64 \
              -DCMAKE_APPLE_SILICON_PROCESSOR=arm64" \
              "{{pip}}" install "llama-cpp-python>=0.3.0" \
                --no-binary llama-cpp-python --no-cache-dir
        else
            CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF \
              -DCMAKE_CUDA_ARCHITECTURES=all-major \
              -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF" \
            FORCE_CMAKE=1 \
              "{{pip}}" install "llama-cpp-python>=0.3.0" \
                --no-binary llama-cpp-python --no-cache-dir
        fi
    elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo "[just] Installing Metal prebuilt wheel v{{llama_metal}}..."
        "{{pip}}" install --only-binary=:all: \
          --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/metal" \
          "llama-cpp-python=={{llama_metal}}" \
          || { echo "[just] ERR: wheel not found — run: just install-llm from-source=true"; exit 1; }
    elif [[ "$(uname -s)" == "Linux" ]]; then
        echo "[just] Installing CUDA prebuilt wheel (cu124 v{{llama_cu124}})..."
        "{{pip}}" install --only-binary=:all: \
          --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu124" \
          "llama-cpp-python=={{llama_cu124}}" 2>/dev/null \
        || {
          echo "[just] cu124 failed — trying cu121 fallback v{{llama_cu121}}..."
          "{{pip}}" install --only-binary=:all: \
            --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu121" \
            "llama-cpp-python=={{llama_cu121}}" \
          || { echo "[just] ERR: no wheel matched — run: just install-llm from-source=true"; exit 1; }
        }
    else
        "{{pip}}" install "llama-cpp-python>=0.3.0"
    fi
    echo "[just] llama-cpp-python installed"

# ── clean: remove venv, node_modules, logs ────────────────────────────────────
clean:
    rm -rf "{{venv}}" "{{webapp}}/node_modules" "{{root}}/logs"
    echo "[just] cleaned"

# ── lint: run shellcheck on dev.sh ────────────────────────────────────────────
lint:
    shellcheck "{{root}}/dev.sh"
