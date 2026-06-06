"""Project-wide configuration: paths, constants, schemas."""
import os
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Raw (private) corpus location. Override via MVPS_DATA_RAW env var; defaults
# to a project-local data/raw directory so the repo is runnable without the
# non-redistributable raw event logs.
DATA_RAW = Path(os.environ.get("MVPS_DATA_RAW", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_EXT = PROJECT_ROOT / "data" / "external"
MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "results"
LOGS = PROJECT_ROOT / "logs"

for p in (DATA_PROC, DATA_INTERIM, DATA_EXT, MODELS, RESULTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# === Cohort definition ===
COHORTS = {
    "Kor2024-1": {
        "label": "spring2024",
        "raw_dir": DATA_RAW / "Kor2024-1",
        "file_suffix": "",  # no suffix
        "period": ("2024-03-01", "2024-08-31"),
    },
    "Kor2024-2": {
        "label": "fall2024",
        "raw_dir": DATA_RAW / "Kor2024-2",
        "file_suffix": "2",  # *2.csv
        "period": ("2024-09-01", "2024-12-31"),
    },
    "Kor2025-1": {
        "label": "spring2025",
        "raw_dir": DATA_RAW / "Kor2025-1",
        "file_suffix": "",
        "period": ("2025-03-01", "2025-08-31"),
    },
}
COHORT_LABEL_TO_ID = {c["label"]: i for i, c in enumerate(COHORTS.values())}
N_COHORTS = len(COHORTS)

# === Event vocabulary v1 ===
# 20 tokens. Submission status + execution error types + LLM + engagement.
VOCAB = [
    # 0-3: Submission outcomes
    "submit_pass",       # status == 2
    "submit_fail",       # status == 0
    "submit_error",      # status == 4
    "submit_partial",    # status == 1 (rare)
    # 4-15: Execution events
    "exec_clean",        # error_name empty
    "exec_SyntaxError",
    "exec_NameError",
    "exec_IndentationError",
    "exec_TypeError",
    "exec_KeyboardInterrupt",
    "exec_ValueError",
    "exec_AttributeError",
    "exec_KeyError",
    "exec_FileNotFoundError",
    "exec_IndexError",
    "exec_OtherError",   # 나머지 묶음
    # 16: LLM intervention
    "llm_help_request",
    # 17-19: Engagement
    "problem_view",
    "problem_start",
    "session_break",     # 30 min+ gap
]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}

# Mapping from execution error_name to vocab token
KNOWN_ERROR_TYPES = {
    "SyntaxError", "NameError", "IndentationError", "TypeError",
    "KeyboardInterrupt", "ValueError", "AttributeError", "KeyError",
    "FileNotFoundError", "IndexError",
}

def error_name_to_token(err: str) -> str:
    if err is None or (isinstance(err, float)) or err == "" or str(err).strip() == "":
        return "exec_clean"
    err = str(err).strip()
    if err in KNOWN_ERROR_TYPES:
        return f"exec_{err}"
    return "exec_OtherError"

def submission_status_to_token(status) -> str:
    try:
        s = int(status)
    except (ValueError, TypeError):
        return "submit_fail"
    if s == 2:
        return "submit_pass"
    if s == 0:
        return "submit_fail"
    if s == 4:
        return "submit_error"
    if s == 1:
        return "submit_partial"
    return "submit_fail"

# === Trajectory definition ===
SESSION_BREAK_THRESHOLD_MIN = 30  # 30+ min idle => session_break token
MAX_TRAJECTORY_LEN = 1024         # truncate from end (keep recent)
MIN_TRAJECTORY_LEN = 5            # filter students with too few events

# === Random seeds ===
SEED = 42
