#!/usr/bin/env python3
# dev.py — start backend API + React frontend with a live TUI dashboard

import argparse
import datetime
import glob
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
BACKEND = ROOT / "backend"
WEBAPP = ROOT / "web-app"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
VENV = ROOT / ".venv"

# Pinned prebuilt-wheel versions (update when abetlen/llama-cpp-python releases)
LLAMA_VER_METAL = "0.3.23"
LLAMA_VER_CU124 = "0.3.22"
LLAMA_VER_CU121 = "0.3.23"

# ── re-exec inside virtual environment ───────────────────────────────────────
if sys.platform == "win32":
    VENV_PYTHON = VENV / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV / "bin" / "python"

if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    # Re-execute the script using the virtual environment python interpreter
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# ── install rich if missing ──────────────────────────────────────────────────
try:
    import rich
except ImportError:
    print("Installing 'rich' library for TUI dashboard...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "--quiet"])
        import rich
    except Exception as e:
        print(f"Failed to install 'rich': {e}", file=sys.stderr)
        sys.exit(1)

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

console = Console()

# ── keyboard input reader (macOS/Linux) ───────────────────────────────────────
class RawTerminal:
    def __enter__(self):
        if not sys.stdin.isatty():
            self.old_settings = None
            return self
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, type, value, traceback):
        if self.old_settings:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if not self.old_settings:
            return None
        # Check if stdin is ready for reading
        r, _, _ = select.select([sys.stdin], [], [], 0.01)
        if r:
            try:
                # Use os.read to bypass Python's stdin stream buffering
                return os.read(self.fd, 1).decode("utf-8", errors="ignore")
            except Exception:
                return None
        return None

# ── preflight checks ──────────────────────────────────────────────────────────
def preflight(args):
    console.print(f"\n[bold]TB-DOTS-CAR-CDSS Dev Environment[/bold]")
    console.print("────────────────────────────────────")

    if not VENV_PYTHON.exists():
        console.print(f"[bold red][dev][/bold red] Virtualenv not found at {VENV}")
        console.print("Run: python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt")
        sys.exit(1)

    # Prebuilt wheels exist only for cp310/cp311/cp312
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        console.print(f"[bold red][dev][/bold red] llama-cpp-python prebuilt wheels require Python 3.10–3.12. Detected: {py_ver}", file=sys.stderr)
        console.print("Recreate venv:  python3.12 -m venv .venv", file=sys.stderr)
        sys.exit(1)

    for cmd in ["node", "curl"]:
        if not shutil.which(cmd):
            console.print(f"[bold red][dev][/bold red] {cmd} not found.", file=sys.stderr)
            sys.exit(1)

    if sys.platform == "darwin":
        res = subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            console.print("[bold red][dev][/bold red] Xcode Command Line Tools missing (required for C++ source builds).", file=sys.stderr)
            console.print("Run: xcode-select --install", file=sys.stderr)
            sys.exit(1)

    # model detection
    model_file = ROOT / "models" / "medgemma-1.5-4b-it-IQ4_XS.gguf"
    model_present = model_file.is_file()
    if not model_present:
        console.print(f"[yellow][dev][/yellow] Warning: Model file not found: {model_file}")
        console.print("[yellow][dev][/yellow] Backend will start without MedGemma inference (llama-cpp-python skipped).")

    # install base backend deps
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import alembic
    except ImportError:
        console.print("[bold cyan][dev][/bold cyan] Installing backend dependencies...")
        req_file = BACKEND / "requirements.txt"
        if req_file.is_file():
            import tempfile
            with req_file.open("r") as f:
                lines = f.readlines()
            filtered = [line for line in lines if "llama-cpp-python" not in line]
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
                tmp.writelines(filtered)
                tmp_name = tmp.name
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", tmp_name, "--quiet"])
            finally:
                try:
                    os.unlink(tmp_name)
                except Exception:
                    pass
            console.print("[bold green][dev][/bold green] Backend dependencies installed.")

    # install llama-cpp-python
    install_llama_cpp(model_present, args.from_source)

    # install frontend deps
    webapp_node_modules = WEBAPP / "node_modules"
    webapp_pkg_json = WEBAPP / "package.json"
    webapp_pkg_lock = WEBAPP / "package-lock.json"

    needs_npm = False
    if not webapp_node_modules.is_dir():
        needs_npm = True
    else:
        mtime = webapp_node_modules.stat().st_mtime
        if webapp_pkg_json.is_file() and webapp_pkg_json.stat().st_mtime > mtime:
            needs_npm = True
        elif webapp_pkg_lock.is_file() and webapp_pkg_lock.stat().st_mtime > mtime:
            needs_npm = True

    if needs_npm:
        console.print("[bold cyan][dev][/bold cyan] Installing frontend dependencies...")
        subprocess.check_call(["npm", "install", "--prefix", str(WEBAPP), "--silent"])
        console.print("[bold green][dev][/bold green] Frontend dependencies installed.")

    # database migrations
    console.print("[bold cyan][dev][/bold cyan] Running database migrations...")
    sys.path.insert(0, str(ROOT))
    from alembic import command
    from alembic.config import Config
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    command.upgrade(cfg, "head")
    console.print("[bold green][dev][/bold green] Migrations up to date.")

    # database seed check
    from sqlalchemy import select
    from backend.db import session_factory
    from backend.models import Patient as PatientRow
    db = session_factory()()
    db_empty = False
    try:
        existing = db.scalar(select(PatientRow.id).limit(1))
        db_empty = existing is None
    except Exception as e:
        console.print(f"[yellow][dev][/yellow] Warning checking database content: {e}")
    finally:
        db.close()

    if db_empty:
        console.print("[bold cyan][dev][/bold cyan] Database is empty — seeding demo data...")
        subprocess.check_call([sys.executable, "-m", "backend.seed_demo", "--include-xrays"], cwd=str(ROOT))
        console.print("[bold green][dev][/bold green] Demo data seeded.")
    else:
        console.print("[bold green][dev][/bold green] Database already has data — skipping seed.")

    return model_present

def install_llama_cpp(model_present, from_source):
    if not model_present:
        return

    try:
        import llama_cpp
        return
    except ImportError:
        pass

    if from_source:
        console.print("[bold cyan][dev][/bold cyan] Source build: GGML_NATIVE=OFF (avoids i8mm CMake-probe hang)...")
        env = os.environ.copy()
        if sys.platform == "darwin" and platform_is_arm64():
            env["CMAKE_ARGS"] = "-DGGML_NATIVE=OFF -DGGML_METAL=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64"
        else:
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=all-major -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"
            env["FORCE_CMAKE"] = "1"
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0",
            "--no-binary", "llama-cpp-python", "--no-cache-dir"
        ], env=env)
        console.print("[bold green][dev][/bold green] llama-cpp-python compiled and installed from source.")
    else:
        console.print("[bold cyan][dev][/bold cyan] Installing llama-cpp-python prebuilt wheel...")
        success = False
        if sys.platform == "darwin" and platform_is_arm64():
            for ver in [LLAMA_VER_METAL, "0.3.22", "0.3.21"]:
                try:
                    console.print(f"[bold cyan][dev][/bold cyan] Trying Metal wheel v{ver}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--prefer-binary", "--no-cache-dir", "--quiet",
                        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/metal",
                        f"llama-cpp-python=={ver}"
                    ])
                    success = True
                    break
                except subprocess.CalledProcessError:
                    console.print(f"[yellow][dev][/yellow] Metal wheel v{ver} failed.")
        elif sys.platform.startswith("linux"):
            for ver, index in [(LLAMA_VER_CU121, "cu121"), (LLAMA_VER_CU124, "cu124")]:
                try:
                    console.print(f"[bold cyan][dev][/bold cyan] Trying CUDA {index} wheel v{ver}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--prefer-binary", "--no-cache-dir", "--quiet",
                        "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{index}",
                        f"llama-cpp-python=={ver}"
                    ])
                    success = True
                    break
                except subprocess.CalledProcessError:
                    console.print(f"[yellow][dev][/yellow] CUDA {index} wheel v{ver} failed.")
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0", "--quiet"])
                success = True
            except subprocess.CalledProcessError:
                pass

        if not success:
            console.print("[yellow][dev][/yellow] All prebuilt wheels failed. Falling back to source build (~3 mins)...")
            env = os.environ.copy()
            if sys.platform == "darwin" and platform_is_arm64():
                env["CMAKE_ARGS"] = "-DGGML_NATIVE=OFF -DGGML_METAL=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64"
            else:
                env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=all-major -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"
                env["FORCE_CMAKE"] = "1"
            
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0",
                "--no-binary", "llama-cpp-python", "--no-cache-dir"
            ], env=env)
            console.print("[bold green][dev][/bold green] llama-cpp-python compiled and installed from source.")
        else:
            console.print("[bold green][dev][/bold green] llama-cpp-python installed.")

def platform_is_arm64():
    import platform
    return platform.machine() == "arm64"

# ── tail log helper ───────────────────────────────────────────────────────────
def tail_file(filepath, num_lines=12):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()
            return lines[-num_lines:]
    except Exception:
        return []

def get_formatted_logs(filepath, num_lines=12):
    lines = tail_file(filepath, num_lines)
    text = Text()
    for line in lines:
        line_text = Text(line)
        # Highlight log levels
        if " [INFO] " in line or "INFO:" in line:
            line_text.highlight_regex(r"(\[INFO\]|INFO:)", "green")
        elif " [WARNING] " in line or "WARNING:" in line:
            line_text.highlight_regex(r"(\[WARNING\]|WARNING:)", "yellow")
        elif " [ERROR] " in line or "ERROR:" in line or "CRITICAL:" in line:
            line_text.highlight_regex(r"(\[ERROR\]|ERROR:|CRITICAL:)", "bold red")
        elif " [DEBUG] " in line or "DEBUG:" in line:
            line_text.highlight_regex(r"(\[DEBUG\]|DEBUG:)", "dim cyan")
        
        # Dim timestamps (HH:MM:SS or YYYY-MM-DD HH:MM:SS)
        line_text.highlight_regex(r"^\d{2}:\d{2}:\d{2}", "dim white")
        line_text.highlight_regex(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", "dim white")
        text.append(line_text)
    return text

# ── stats polling ─────────────────────────────────────────────────────────────
def fetch_stats(port):
    url = f"http://localhost:{port}/api/stats"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=1.5) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except Exception:
        return {"status": "unreachable"}

def get_process_ram(pid):
    if not pid:
        return "—"
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        rss_kb = int(out.decode("utf-8").strip())
        return f"{rss_kb // 1024} MB"
    except Exception:
        return "—"

def detect_actual_ports(backend_log, frontend_log, default_be=8000, default_fe=5173):
    be_port = default_be
    fe_port = default_fe
    
    if os.path.exists(backend_log):
        try:
            with open(backend_log, "r", errors="replace") as f:
                for line in f:
                    if "Uvicorn running on" in line:
                        match = re.search(r":(\d+)", line)
                        if match:
                            be_port = int(match.group(1))
                            break
        except Exception:
            pass

    if os.path.exists(frontend_log):
        try:
            with open(frontend_log, "r", errors="replace") as f:
                for line in f:
                    if "localhost:" in line or "Local:" in line:
                        match = re.search(r"localhost:(\d+)", line)
                        if not match:
                            match = re.search(r"Local:\s+http://localhost:(\d+)", line)
                        if match:
                            fe_port = int(match.group(1))
                            break
        except Exception:
            pass
            
    return be_port, fe_port

# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Start backend API + React frontend with a live TUI dashboard")
    parser.add_argument("--from-source", action="store_true", help="Compile llama-cpp-python from C++ source")
    args = parser.parse_args()

    model_present = preflight(args)

    # ── setup logs ────────────────────────────────────────────────────────────
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backend_log = log_dir / f"backend-{ts}.log"
    frontend_log = log_dir / f"frontend-{ts}.log"

    # clean old logs (7 days)
    now_time = time.time()
    for f in glob.glob(str(log_dir / "backend-*.log")) + glob.glob(str(log_dir / "frontend-*.log")):
        if os.path.exists(f) and os.stat(f).st_mtime < now_time - 7 * 86400:
            try:
                os.remove(f)
            except Exception:
                pass

    # ── environment preparation ───────────────────────────────────────────────
    env = os.environ.copy()
    if sys.platform.startswith("linux"):
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        extra_paths = []
        for pkg in ["cuda_runtime", "cublas", "cuda_nvrtc"]:
            pkg_path = VENV / "lib" / f"python{py_ver}" / "site-packages" / "nvidia" / pkg / "lib"
            if pkg_path.is_dir():
                extra_paths.append(str(pkg_path))
        import sysconfig
        py_lib = sysconfig.get_config_var("LIBDIR")
        if py_lib and os.path.isdir(py_lib):
            extra_paths.append(py_lib)
        if extra_paths:
            env["LD_LIBRARY_PATH"] = ":".join(extra_paths) + (":" + env.get("LD_LIBRARY_PATH", "") if env.get("LD_LIBRARY_PATH") else "")

    # ── spawn processes ───────────────────────────────────────────────────────
    console.print(f"[bold cyan][dev][/bold cyan] Starting backend API on :{BACKEND_PORT} (log → logs/backend-{ts}.log)")
    be_file = open(backend_log, "w")
    
    # Use process group creation so we can cleanly terminate the tree on exit
    popen_kwargs = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid

    be_proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "backend.main:app", "--port", str(BACKEND_PORT), "--log-level", "info"],
        stdout=be_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
        **popen_kwargs
    )

    console.print(f"[bold cyan][dev][/bold cyan] Starting React frontend on :{FRONTEND_PORT} (log → logs/frontend-{ts}.log)")
    fe_file = open(frontend_log, "w")
    fe_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(FRONTEND_PORT)],
        stdout=fe_file,
        stderr=subprocess.STDOUT,
        cwd=str(WEBAPP),
        **popen_kwargs
    )

    # ── wait for backend health check ─────────────────────────────────────────
    if model_present:
        console.print("[bold cyan][dev][/bold cyan] Waiting for model to load...")
    else:
        console.print("[bold cyan][dev][/bold cyan] Waiting for backend to start...")

    health_url = f"http://localhost:{BACKEND_PORT}/api/health"
    start_time = time.time()
    backend_ready = False
    while time.time() - start_time < 90:
        if be_proc.poll() is not None:
            console.print("[bold red][dev][/bold red] Backend process died during startup. Check backend log.")
            break
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                if data.get("status") in ["ready", "loading", "error"]:
                    backend_ready = True
                    break
        except Exception:
            pass
        console.print(".", end="", flush=True)
        time.sleep(2)
    console.print()

    # ── prepare cleanup handler ───────────────────────────────────────────────
    cleaned = False
    def cleanup(signum=None, frame=None):
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        
        # Restore cursor if hidden
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

        console.print("\n[bold cyan][dev][/bold cyan] Shutting down services...")
        for proc, name in [(be_proc, "Backend"), (fe_proc, "Frontend")]:
            if proc and proc.poll() is None:
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                    proc.wait(timeout=3)
                    console.print(f"[bold green][dev][/bold green] {name} stopped successfully.")
                except Exception:
                    try:
                        if sys.platform != "win32":
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                        proc.wait(timeout=1)
                        console.print(f"[yellow][dev][/yellow] {name} killed.")
                    except Exception:
                        pass
        
        try:
            be_file.close()
            fe_file.close()
        except Exception:
            pass
            
        sys.exit(0)

    # Register exit hooks
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── build layout ──────────────────────────────────────────────────────────
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2)
    )
    layout["left"].split(
        Layout(name="services", size=9),
        Layout(name="ai_backend", size=10),
        Layout(name="system")
    )
    layout["right"].split(
        Layout(name="requests", size=9),
        Layout(name="logs")
    )

    # hide cursor during dashboard loop
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    # ── dashboard loop ────────────────────────────────────────────────────────
    try:
        with Live(layout, screen=True, auto_refresh=False) as live, RawTerminal() as term:
            loop_start = time.time()
            last_poll_time = 0.0
            
            # Initial stats/render values
            stats = {"status": "unreachable"}
            be_ram = "—"
            fe_ram = "—"
            act_be_port = BACKEND_PORT
            act_fe_port = FRONTEND_PORT
            
            while True:
                # 1. Read keyboard inputs (checked at high speed: every ~10ms)
                key = term.get_key()
                if key in ['q', 'Q', '\x03']: # 'q', 'Q', or Ctrl+C
                    break
                
                # Check process statuses
                be_running = be_proc.poll() is None
                fe_running = fe_proc.poll() is None
                
                if not be_running and not fe_running:
                    # Both stopped, break out of live rendering to print full error
                    break

                now = time.time()
                # 2. Update panel state and refresh screen at a controlled pace (every 0.5s)
                if now - last_poll_time >= 0.5:
                    last_poll_time = now

                    # Get actual ports (Vite might change its port)
                    act_be_port, act_fe_port = detect_actual_ports(backend_log, frontend_log, BACKEND_PORT, FRONTEND_PORT)
                    
                    # Fetch stats
                    stats = fetch_stats(act_be_port) if be_running else {"status": "unreachable"}
                    
                    # Process RAM
                    be_ram = get_process_ram(be_proc.pid) if be_running else "—"
                    fe_ram = get_process_ram(fe_proc.pid) if fe_running else "—"

                    # ── RENDER COMPONENT: Header ──────────────────────────────────
                    uptime = str(datetime.timedelta(seconds=int(time.time() - loop_start)))
                    header_text = Text.assemble(
                        (" TB-DOTS-CAR-CDSS ", "bold white on blue"),
                        "  ·  Dev Dashboard  ·  ",
                        (datetime.datetime.now().strftime("%H:%M:%S"), "bold cyan"),
                        f"  ·  Uptime: {uptime}  ·  Press ",
                        ("q", "bold magenta"),
                        " to quit"
                    )
                    layout["header"].update(Panel(header_text, border_style="blue"))

                    # ── RENDER COMPONENT: Services ────────────────────────────────
                    services_table = Table.grid(expand=True)
                    services_table.add_column(style="bold cyan")
                    services_table.add_column()
                    
                    be_status = stats.get("status", "unreachable")
                    if be_status == "ready":
                        be_badge = "[bold green]● ready[/bold green]"
                    elif be_status == "loading":
                        be_badge = "[bold yellow]◌ loading[/bold yellow]"
                    elif be_status == "error":
                        be_badge = "[bold red]✖ error[/bold red]"
                    else:
                        be_badge = "[bold red]✖ unreachable[/bold red]"
                    
                    fe_badge = "[bold green]● running[/bold green]" if fe_running else "[bold red]✖ stopped[/bold red]"
                    
                    be_port_str = f":{act_be_port}"
                    if act_be_port != BACKEND_PORT:
                        be_port_str += f" (cfg :{BACKEND_PORT})"
                    fe_port_str = f":{act_fe_port}"
                    if act_fe_port != FRONTEND_PORT:
                        fe_port_str += f" (cfg :{FRONTEND_PORT})"

                    services_table.add_row("Backend API", f"{be_badge}  [dim]{be_port_str}[/dim]")
                    services_table.add_row("  PID / Log", f"[dim]{be_proc.pid if be_running else '—'} / logs/{backend_log.name}[/dim]")
                    services_table.add_row("React Webapp", f"{fe_badge}  [dim]{fe_port_str}[/dim]")
                    services_table.add_row("  PID / Log", f"[dim]{fe_proc.pid if fe_running else '—'} / logs/{frontend_log.name}[/dim]")
                    
                    layout["services"].update(Panel(services_table, title="[bold white]Services[/bold white]", border_style="cyan"))

                    # ── RENDER COMPONENT: AI Backend ──────────────────────────────
                    ai_table = Table.grid(expand=True)
                    ai_table.add_column(style="bold cyan")
                    ai_table.add_column()
                    
                    inf_active = stats.get("inference_active", False)
                    inf_badge = "[bold magenta]⬤ GENERATING[/bold magenta]" if inf_active else "[dim]○ idle[/dim]"
                    
                    rep_ram = f"{stats.get('ram_mb', 0)} MB"
                    load_time = f"{stats.get('model_load_time_s', 0)} s"
                    backend_type = stats.get("backend", "—").upper()

                    ai_table.add_row("Backend Type", backend_type)
                    ai_table.add_row("Inference State", inf_badge)
                    ai_table.add_row("App Reported RAM", rep_ram)
                    ai_table.add_row("Model Load Time", load_time)
                    
                    layout["ai_backend"].update(Panel(ai_table, title="[bold white]AI Backend[/bold white]", border_style="cyan"))

                    # ── RENDER COMPONENT: System ──────────────────────────────────
                    sys_table = Table.grid(expand=True)
                    sys_table.add_column(style="bold cyan")
                    sys_table.add_column()
                    
                    sys_table.add_row("Host OS", sys.platform.capitalize())
                    sys_table.add_row("Backend RAM", be_ram)
                    sys_table.add_row("Frontend RAM", fe_ram)
                    
                    layout["system"].update(Panel(sys_table, title="[bold white]System Stats[/bold white]", border_style="cyan"))

                    # ── RENDER COMPONENT: Requests ────────────────────────────────
                    req_table = Table.grid(expand=True)
                    req_table.add_column(style="bold cyan")
                    req_table.add_column()
                    
                    served = stats.get("requests_served", 0)
                    active_reqs = stats.get("requests_active", 0)
                    last_patient = stats.get("last_patient") or "—"
                    last_tok = stats.get("last_tokens", 0)
                    last_el = stats.get("last_elapsed_s", 0.0)
                    
                    t_per_s = "—"
                    if last_el > 0 and last_tok > 0:
                        t_per_s = f"{last_tok / last_el:.1f}"

                    req_table.add_row("Total Served", str(served))
                    req_table.add_row("Active Requests", str(active_reqs))
                    req_table.add_row("Last Patient", last_patient)
                    req_table.add_row("Last Generation", f"{last_tok} tokens in {last_el:.1f}s ({t_per_s} tok/s)")
                    
                    layout["requests"].update(Panel(req_table, title="[bold white]Request Statistics[/bold white]", border_style="cyan"))

                    # ── RENDER COMPONENT: Recent Backend Logs ─────────────────────
                    term_height = live.console.height
                    log_lines = max(5, term_height - 18)
                    logs_text = get_formatted_logs(backend_log, log_lines)
                    layout["logs"].update(Panel(logs_text, title="[bold white]Recent Backend Logs[/bold white]", border_style="cyan"))

                    # Refresh live output
                    live.refresh()
                
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

    # If we broke out of the loop and processes died
    console.print("\n[bold red][dev][/bold red] Both services have stopped.")
    console.print(f"Check backend logs: logs/backend-{ts}.log")
    console.print(f"Check frontend logs: logs/frontend-{ts}.log")
    cleanup()

if __name__ == "__main__":
    main()
