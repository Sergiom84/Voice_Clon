from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = APP_ROOT / "outputs"
JOBS_DIR = OUTPUTS_DIR / "jobs"
PREPARED_REFERENCES_DIR = OUTPUTS_DIR / "prepared_references"
TRAINING_DATA_DIR = APP_ROOT / "training_data"

DEFAULT_OUTPUT_SAMPLE_RATE = 24_000
DEFAULT_FRAGMENT_PAUSE_MS = 250
DEFAULT_REFERENCE_EXCERPT_SECONDS = 12.0
DEFAULT_REFERENCE_EXCERPT_COUNT = 1
MAX_FRAGMENT_CHARS = 240
MAX_SYNTHESIS_CHARS = 300
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 30.0
DEFAULT_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SYNTHESIS_SEED = 42
MAX_RESYNTHESIS_ATTEMPTS = 3

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860


def ensure_app_directories() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
