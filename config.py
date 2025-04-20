"""
Configuration settings for the Movie Recommender System with Quantum Computing.

This module contains all the configuration parameters used throughout the project,
including data paths, model parameters, and analysis settings.
"""

import os
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).parent

# Project paths
PROJECT_ROOT = BASE_DIR
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, SRC_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data paths
DATA_PATHS = {
    # Path to the ratings data file
    "ratings": DATA_DIR / "u.data",
    # Path to the users data file
    "users": DATA_DIR / "u.user",
    # Path to the movies data file
    "movies": DATA_DIR / "u.item",
    # Path to the genres data file
    "genres": DATA_DIR / "u.genre"
}

# Data preprocessing settings
PREPROCESSING = {
    # Minimum number of ratings required for a user to be included
    "min_user_ratings": 20,
    # Minimum number of ratings required for a movie to be included
    "min_movie_ratings": 10,
    # Number of most recent ratings to consider for each user
    "recent_ratings_window": 50,
    # Time decay factor for rating weights (higher = more weight on recent ratings)
    "time_decay_factor": 0.1
}

# Graph construction settings
GRAPH_SETTINGS = {
    # Weight threshold for creating edges between users
    "weight_threshold": 0.5,
    # Maximum number of neighbors for each user
    "max_neighbors": 10,
    # Number of features to extract from the graph
    "n_features": 32
}

# Topological analysis settings
TOPOLOGY_SETTINGS = {
    # Maximum dimension for persistent homology
    "max_dim": 2,
    # Maximum filtration value for the Rips complex
    "max_filtration": 1.0,
    # Number of points in the persistence landscape
    "n_points": 100,
    # Number of dimensions to consider in the landscape
    "n_dimensions": 3
}

# Outlier detection settings
OUTLIER_SETTINGS = {
    # Number of standard deviations for outlier threshold
    "n_std": 2.5,
    # Minimum number of ratings required for outlier analysis
    "min_ratings": 30,
    # Window size for moving average in rating pattern analysis
    "window_size": 5,
    # Threshold for rating pattern deviation
    "pattern_threshold": 0.3
}

# Fairness analysis settings
FAIRNESS_SETTINGS = {
    # Age groups for fairness analysis
    "age_groups": {
        "young": (0, 25),
        "middle": (26, 45),
        "old": (46, 100)
    },
    # Minimum number of ratings per age group
    "min_ratings_per_group": 100,
    # Number of bins for rating distribution
    "n_bins": 5
}

# Quantum model settings
QUANTUM_SETTINGS = {
    # Number of qubits in the quantum circuit
    "n_qubits": 2,
    # Size of the hidden layer
    "hidden_size": 32,
    # Learning rate for the optimizer
    "learning_rate": 0.01,
    # Number of training epochs
    "n_epochs": 5,
    # Batch size for training
    "batch_size": 16
}

# Privacy settings
PRIVACY_SETTINGS = {
    "epsilon": 1.0,  # Privacy budget
    "delta": 1e-5    # Privacy parameter
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "time_decay_alpha": 0.95,
    "n_bootstraps": 100,
    "confidence_level": 0.95
}

# Visualization settings
VISUALIZATION = {
    # Figure size for plots
    "figsize": (12, 8),
    # Font size for plot titles
    "title_fontsize": 16,
    # Font size for axis labels
    "label_fontsize": 14,
    # Font size for tick labels
    "tick_fontsize": 12,
    # Color palette for plots
    "palette": "viridis",
    # Style for plots
    "style": "seaborn"
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "recommender.log" 