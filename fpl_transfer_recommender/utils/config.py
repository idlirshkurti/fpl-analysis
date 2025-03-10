"""
Configuration settings for FPL Transfer Recommender.

This module contains constants and configuration variables used throughout the application.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data file paths
PLAYER_DATA_FILE = DATA_DIR / "player_data.csv"
TEAM_DATA_FILE = DATA_DIR / "team_data.csv"
FIXTURE_DATA_FILE = DATA_DIR / "fixture_data.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.pkl"
FEATURES_DATA_FILE = DATA_DIR / "features_data.csv"

# API settings
API_CACHE_TTL = 3600  # Cache API data for 1 hour (in seconds)

# Model settings
DEFAULT_MODEL_TYPE = "random_forest"
DEFAULT_MODEL_FILE = MODELS_DIR / "fpl_points_predictor.pkl"
MODEL_TRAINING_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42

# Feature engineering settings
FUTURE_GAMEWEEKS = 5  # Number of future gameweeks to consider
FORM_WINDOW_SIZE = 5  # Number of gameweeks to consider for form calculation

# Recommendation settings
DEFAULT_NUM_TRANSFERS = 1
DEFAULT_EXTRA_BUDGET = 0.0
MAX_PLAYER_COST = 15.0  # Maximum player cost in millions

# FPL team constraints
TEAM_SIZE = 15
GOALKEEPER_COUNT = 2
DEFENDER_COUNT = 5
MIDFIELDER_COUNT = 5
FORWARD_COUNT = 3
TEAM_BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3

# Position mapping
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

