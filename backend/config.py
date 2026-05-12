import os
import platform
import sys
from pathlib import Path


def is_apple_silicon() -> bool:
    override = os.environ.get("TB_INFERENCE_BACKEND", "").lower()
    if override == "metal":
        return True
    if override == "cuda":
        return False
    return sys.platform == "darwin" and platform.processor() == "arm"


APPLE_SILICON = is_apple_silicon()

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "medgemma-1.5-4b-it-IQ4_XS.gguf")
N_CTX      = 2048
N_BATCH    = 512
N_GPU_LAYERS = -1

if APPLE_SILICON:
    N_THREADS = N_THREADS_BATCH = max(4, (os.cpu_count() or 4) // 2)
    FLASH_ATTN = False      # causes errors on several Metal llama-cpp-python builds
    TYPE_K = TYPE_V = None  # KV quant unreliable on Metal
else:
    N_THREADS = N_THREADS_BATCH = 6
    FLASH_ATTN = True
    TYPE_K = TYPE_V = 8

TEMPERATURE    = 0.3
MAX_TOKENS     = 1536
TOP_P          = 0.9
REPEAT_PENALTY = 1.1
