"""
Central configuration for the Complexity Kink Research pipeline.
All paths, hyperparameters, and constants in one place.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

DEFAULT_SCORED_FILE = os.path.join(DATA_DIR, "final_results_scored.jsonl")
DEFAULT_ENRICHED_FILE = os.path.join(DATA_DIR, "iv_enriched_dataset.jsonl")
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "kappa_predictor_stage1.joblib")

# --- Stage 1 Hyperparameters ---
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
CV_FOLDS = 10

# --- Stage 2 Parameters ---
THRESHOLD_GRID_START = 2.0
THRESHOLD_GRID_END = 15.0
THRESHOLD_GRID_STEP = 0.5
HANSEN_BOOTSTRAP_ITERATIONS = 500
PLACEBO_ITERATIONS = 500
MIN_REGIME_SIZE = 100

# --- Feature columns used as instruments ---
IV_FEATURE_COLS = [
    'inst_tokens',
    'inst_if_count',
    'inst_loop_count',
    'inst_class_count',
    'inst_func_count',
    'inst_logic_count',
    'inst_total_structural',
    'inst_avg_word_len',
]

# --- Control variables for Stage 2 ---
CONTROL_COLS = ['e_norm', 'm_mem_jaccard']
