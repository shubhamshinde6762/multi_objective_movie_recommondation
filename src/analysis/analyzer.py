"""
Analysis module for the Movie Recommender System.

This module contains classes for performing various analyses on the movie rating data:
- Topological analysis of movie genres
- Outlier detection in user ratings
- Fairness analysis across different user groups
"""

import numpy as np
import pandas as pd
import gudhi as gd
from gudhi.representations import Landscape
import ot
from scipy.stats import hypergeom
import networkx as nx
import torch
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from pathlib import Path
import seaborn as sns
from config import ANALYSIS_SETTINGS, TOPOLOGY_SETTINGS, OUTLIER_SETTINGS, FAIRNESS_SETTINGS, VISUALIZATION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopologicalAnalyzer:
    """
    Analyzes the topological structure of movie genres using persistent homology.
    
    This class performs topological data analysis on the movie genre space to identify
    persistent features and patterns in genre relationships.
    """
    
    def __init__(self):
        self.persistence = None
        self.landscape = None

    def analyze_genres(self, movies: pd.DataFrame) -> Tuple[Any, Any]:
        """
        Perform topological analysis of movie genres.
        
        Args:
            movies (pd.DataFrame): DataFrame containing movie information including genres
            
        Returns:
            tuple: (persistence diagrams, persistence landscapes)
        """
        logger.info("Performing topological analysis of genres...")
        
        # Extract genre vectors
        genre_columns = [col for col in movies.columns if col.startswith('genre_')]
        genre_vectors = movies[genre_columns].values
        
        # Create Rips complex
        rips_complex = gd.RipsComplex(points=genre_vectors, max_edge_length=TOPOLOGY_SETTINGS["max_filtration"])
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=TOPOLOGY_SETTINGS["max_dim"])
        
        # Compute persistence
        self.persistence = simplex_tree.persistence()
        
        # Compute persistence landscape
        self.landscape = Landscape(
            num_landscapes=TOPOLOGY_SETTINGS["n_dimensions"],
            resolution=TOPOLOGY_SETTINGS["n_points"]
        )
        
        # Convert persistence to numpy array for landscape computation
        persistence_array = np.array([[p[1][0], p[1][1]] for p in self.persistence if p[1][1] != float('inf')])
        if len(persistence_array) > 0:
            self.landscape.fit([persistence_array])
        else:
            self.landscape = None
        
        logger.info("Topological analysis completed")
        return self.persistence, self.landscape

    def save_analysis(self, output_dir: Path) -> None:
        """
        Save topological analysis results and visualizations.
        
        Args:
            output_dir (Path): Directory to save the analysis results
        """
        logger.info("Saving topological analysis results...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.persistence:
            # Plot persistence diagram
            plt.figure(figsize=VISUALIZATION["figsize"])
            gd.plot_persistence_diagram(self.persistence)
            plt.title("Persistence Diagram of Movie Genres", 
                     fontsize=VISUALIZATION["title_fontsize"])
            plt.xlabel("Birth Time", fontsize=VISUALIZATION["label_fontsize"])
            plt.ylabel("Death Time", fontsize=VISUALIZATION["label_fontsize"])
            plt.tick_params(labelsize=VISUALIZATION["tick_fontsize"])
            plt.savefig(output_dir / "persistence_diagram.png")
            plt.close()
        
        if self.landscape is not None:
            # Plot persistence landscape
            plt.figure(figsize=VISUALIZATION["figsize"])
            persistence_array = np.array([[p[1][0], p[1][1]] for p in self.persistence if p[1][1] != float('inf')])
            if len(persistence_array) > 0:
                landscape_data = self.landscape.transform([persistence_array])[0]
                plt.plot(landscape_data)
                plt.title("Persistence Landscape of Movie Genres",
                         fontsize=VISUALIZATION["title_fontsize"])
                plt.xlabel("Filtration Parameter", fontsize=VISUALIZATION["label_fontsize"])
                plt.ylabel("Landscape Value", fontsize=VISUALIZATION["label_fontsize"])
                plt.tick_params(labelsize=VISUALIZATION["tick_fontsize"])
                plt.savefig(output_dir / "persistence_landscape.png")
            plt.close()
        
        logger.info("Analysis results saved")

class OutlierDetector:
    """
    Detects outliers in user rating patterns using statistical and topological methods.
    
    This class identifies users whose rating patterns deviate significantly from
    the norm, using both statistical methods and pattern analysis.
    """
    
    def __init__(self):
        self.outlier_scores = None

    def detect_outliers(self, ratings: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers in user ratings.
        
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings
            users (pd.DataFrame): DataFrame containing user information
            
        Returns:
            pd.DataFrame: DataFrame containing outlier scores for each user
        """
        logger.info("Detecting outliers...")
        
        # Calculate rating statistics
        user_stats = ratings.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count']
        }).reset_index()
        user_stats.columns = ['user_id', 'mean_rating', 'std_rating', 'rating_count']
        
        # Filter users with sufficient ratings
        user_stats = user_stats[user_stats['rating_count'] >= OUTLIER_SETTINGS["min_ratings"]]
        
        # Calculate outlier scores
        user_stats['z_score'] = (user_stats['mean_rating'] - user_stats['mean_rating'].mean()) / user_stats['mean_rating'].std()
        user_stats['pattern_score'] = self._calculate_pattern_score(ratings, user_stats['user_id'])
        
        # Combine scores
        user_stats['outlier_score'] = (
            np.abs(user_stats['z_score']) * 0.5 +
            user_stats['pattern_score'] * 0.5
        )
        
        # Store outlier scores
        self.outlier_scores = user_stats
        
        logger.info("Outlier detection completed")
        return self.outlier_scores
    
    def _calculate_pattern_score(self, ratings: pd.DataFrame, user_ids: pd.Series) -> pd.Series:
        """
        Calculate pattern-based outlier scores.
        
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings
            user_ids (pd.Series): Series of user IDs to analyze
            
        Returns:
            pd.Series: Pattern-based outlier scores
        """
        pattern_scores = []
        
        for user_id in user_ids:
            user_ratings = ratings[ratings['user_id'] == user_id].sort_values('timestamp')
            
            # Calculate moving average
            window = OUTLIER_SETTINGS["window_size"]
            ma = user_ratings['rating'].rolling(window=window, min_periods=1).mean()
            
            # Calculate pattern deviation
            pattern_deviation = np.mean(np.abs(user_ratings['rating'] - ma))
            pattern_scores.append(pattern_deviation)
        
        return pd.Series(pattern_scores, index=user_ids)
    
    def save_outliers(self, output_dir: Path) -> None:
        """
        Save outlier detection results and visualizations.
        
        Args:
            output_dir (Path): Directory to save the results
        """
        logger.info("Saving outlier detection results...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.outlier_scores is not None:
            # Save scores
            self.outlier_scores.to_csv(output_dir / "outlier_scores.csv", index=False)
            
            # Plot outlier score distribution
            plt.figure(figsize=VISUALIZATION["figsize"])
            sns.histplot(data=self.outlier_scores, x='outlier_score', bins=30)
            plt.title("Distribution of User Outlier Scores",
                     fontsize=VISUALIZATION["title_fontsize"])
            plt.xlabel("Outlier Score", fontsize=VISUALIZATION["label_fontsize"])
            plt.ylabel("Number of Users", fontsize=VISUALIZATION["label_fontsize"])
            plt.tick_params(labelsize=VISUALIZATION["tick_fontsize"])
            
            plt.savefig(output_dir / "outlier_distribution.png")
            plt.close()
        
        logger.info("Outlier results saved")

class FairnessAnalyzer:
    """
    Analyzes fairness in movie ratings across different user groups.
    
    This class examines whether there are systematic biases in ratings based on
    user demographics, particularly age groups.
    """
    
    def __init__(self):
        self.fairness_metrics = None

    def analyze_fairness(self, ratings: pd.DataFrame, users: pd.DataFrame) -> dict:
        """
        Analyze fairness in ratings across different age groups.
        
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings
            users (pd.DataFrame): DataFrame containing user information
            
        Returns:
            dict: Dictionary containing fairness metrics
        """
        logger.info("Analyzing fairness...")
        
        # Merge ratings with user information
        data = pd.merge(ratings, users, on='user_id')
        
        # Calculate average ratings by age group
        age_groups = FAIRNESS_SETTINGS["age_groups"]
        self.fairness_metrics = {}
        
        for group_name, (min_age, max_age) in age_groups.items():
            group_data = data[
                (data['age'] >= min_age) & 
                (data['age'] <= max_age)
            ]
            
            if len(group_data) >= FAIRNESS_SETTINGS["min_ratings_per_group"]:
                # Calculate rating distribution
                rating_dist = group_data['rating'].value_counts().sort_index()
                
                # Calculate Wasserstein distance from uniform distribution
                uniform_dist = np.ones(len(rating_dist)) / len(rating_dist)
                wasserstein_dist = ot.emd2(
                    rating_dist.values / rating_dist.sum(),
                    uniform_dist,
                    ot.dist(rating_dist.index.values.reshape(-1, 1))
                )
                
                self.fairness_metrics[group_name] = {
                    'mean_rating': group_data['rating'].mean(),
                    'std_rating': group_data['rating'].std(),
                    'wasserstein_distance': wasserstein_dist
                }
        
        logger.info("Fairness analysis completed")
        return self.fairness_metrics
    
    def save_fairness(self, output_dir: Path) -> None:
        """
        Save fairness analysis results and visualizations.
        
        Args:
            output_dir (Path): Directory to save the results
        """
        logger.info("Saving fairness analysis results...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.fairness_metrics is not None:
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame(self.fairness_metrics).T
            
            # Save metrics
            metrics_df.to_csv(output_dir / "fairness_metrics.csv")
            
            # Plot fairness metrics
            plt.figure(figsize=VISUALIZATION["figsize"])
            
            # Plot mean ratings
            plt.subplot(2, 1, 1)
            sns.barplot(x=metrics_df.index, y='mean_rating', data=metrics_df)
            plt.title("Average Ratings by Age Group",
                     fontsize=VISUALIZATION["title_fontsize"])
            plt.xlabel("Age Group", fontsize=VISUALIZATION["label_fontsize"])
            plt.ylabel("Mean Rating", fontsize=VISUALIZATION["label_fontsize"])
            plt.tick_params(labelsize=VISUALIZATION["tick_fontsize"])
            
            # Plot Wasserstein distances
            plt.subplot(2, 1, 2)
            sns.barplot(x=metrics_df.index, y='wasserstein_distance', data=metrics_df)
            plt.title("Rating Distribution Distance from Uniform by Age Group",
                     fontsize=VISUALIZATION["title_fontsize"])
            plt.xlabel("Age Group", fontsize=VISUALIZATION["label_fontsize"])
            plt.ylabel("Wasserstein Distance", fontsize=VISUALIZATION["label_fontsize"])
            plt.tick_params(labelsize=VISUALIZATION["tick_fontsize"])
            
            plt.tight_layout()
            plt.savefig(output_dir / "fairness_metrics.png")
            plt.close()
        
        logger.info("Fairness results saved") 