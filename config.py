"""
Configuration settings for the Movie Recommender System with Quantum Computing.

This module contains all the configuration parameters used throughout the project,
including data paths, model parameters, and analysis settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Model settings optimized for better performance
MODEL_SETTINGS = {
    "batch_size": 128,  # Increased batch size for better GPU utilization
    "learning_rate": 0.001,  # Increased learning rate for faster convergence
    "weight_decay": 1e-4,  # Increased regularization
    "num_epochs": 50,  # More epochs for better convergence
    "early_stopping_patience": 5,  # Increased patience
    "validation_split": 0.2,
    "num_workers": 4,  # More workers for data loading
    "embedding_dim": 64,  # Larger embedding size
    "hidden_dim": 128,  # Larger hidden layer size
    "dropout": 0.3,  # Increased dropout for better regularization
}

# Feature processing settings
FEATURE_SETTINGS = {
    "min_ratings_per_user": 5,
    "min_ratings_per_movie": 5,
    "time_decay_factor": 0.95,
    "genre_threshold": 0.1,
    "max_features": 500,  # Reduced for CPU memory constraints
    "chunk_size": 5000,  # Process data in smaller chunks
}

# Multi-objective optimization settings
OPTIMIZATION_SETTINGS = {
    "accuracy_weight": 0.4,
    "diversity_weight": 0.3,
    "fairness_weight": 0.2,
    "novelty_weight": 0.1,
    "population_size": 20,  # Reduced for CPU
    "num_generations": 10,  # Reduced for CPU
    "batch_size": 32,  # Reduced for CPU
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "top_k": 10,
    "min_support": 0.01,
    "confidence_threshold": 0.5,
    "max_clusters": 3,  # Reduced for CPU
}

# Memory management settings
MEMORY_SETTINGS = {
    "max_memory_usage": 0.7,  # 70% of available RAM
    "chunk_size": 5000,  # Process data in smaller chunks
    "cache_size": 500,  # Reduced cache size
}

# Environment settings optimized for performance
ENVIRONMENT_SETTINGS = {
    "device": "cpu",
    "num_threads": 8,  # Increased thread count
    "use_mkl": True,
    "use_openmp": True,
    "torch_num_threads": 8,
    "torch_num_interop_threads": 8,
}

# Dataset settings
DATASET_SETTINGS = {
    "name": "ml-100k",
    "min_rating": 1,
    "max_rating": 5,
    "rating_scale": 5,
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
}

# Path settings
PATH_SETTINGS = {
    "raw_data": DATA_DIR / "raw",
    "processed_data": DATA_DIR / "processed",
    "model_checkpoints": MODELS_DIR / "checkpoints",
    "results": RESULTS_DIR,
    "visualizations": RESULTS_DIR / "visualizations",
}

# Create all necessary subdirectories
for path in PATH_SETTINGS.values():
    path.mkdir(parents=True, exist_ok=True)

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