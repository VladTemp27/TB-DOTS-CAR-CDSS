from pathlib import Path
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "medgemma-1.5-4b-it-IQ4_XS.gguf")
N_CTX = 2048
N_THREADS = 4
N_GPU_LAYERS = 0      # CPU-only, no CUDA/Metal dependency
TEMPERATURE = 0.3     # Low for clinical accuracy
MAX_TOKENS = 512
TOP_P = 0.9
REPEAT_PENALTY = 1.1
