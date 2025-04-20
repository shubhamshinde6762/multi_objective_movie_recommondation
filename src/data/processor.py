"""
Data processing module for the Movie Recommender System.

This module handles loading, preprocessing, and feature engineering of the movie rating data.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import torch
from typing import Tuple, Dict, Any
from config import DATA_PATHS, PREPROCESSING, GRAPH_SETTINGS

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering.
    
    This class manages the entire data pipeline from raw data to processed features,
    including graph construction and feature extraction.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.ratings = None
        self.users = None
        self.movies = None
        self.graph = None
        self.edge_index = None
        self.edge_attr = None
        self.fft_values = None
    
    def load_data(self) -> None:
        """
        Load the raw data files.
        
        This method loads the ratings, users, and movies data from the specified paths.
        """
        logger.info("Loading data files...")
        
        try:
            # Load ratings data
            self.ratings = pd.read_csv(
                DATA_PATHS["ratings"],
                sep='\t',
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            # Load users data
            self.users = pd.read_csv(
                DATA_PATHS["users"],
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            
            # Load movies data
            self.movies = pd.read_csv(
                DATA_PATHS["movies"],
                sep='|',
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date',
                      'imdb_url'] + [f'genre_{i}' for i in range(19)]
            )
            
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _calculate_time_decay(self, timestamps, decay_factor=0.1):
        """
        Calculate time decay factor for each rating based on timestamp.
        
        Args:
            timestamps (pd.Series): Series of timestamps
            decay_factor (float): Decay factor for time weighting
            
        Returns:
            np.ndarray: Array of time decay factors
        """
        max_timestamp = timestamps.max()
        time_diff = max_timestamp - timestamps
        time_decay = np.exp(-decay_factor * time_diff.dt.total_seconds() / (24 * 60 * 60))  # Convert to days
        return time_decay
    
    def preprocess_data(self) -> None:
        """
        Preprocess the loaded data.
        
        This method performs data cleaning, feature engineering, and graph construction.
        """
        logger.info("Preprocessing data...")
        
        try:
            # Convert timestamps to datetime
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
            
            # Calculate time decay
            self.ratings['time_decay'] = self._calculate_time_decay(self.ratings['timestamp'])
            
            # Filter users and movies with sufficient ratings
            user_counts = self.ratings['user_id'].value_counts()
            movie_counts = self.ratings['movie_id'].value_counts()
            
            valid_users = user_counts[user_counts >= PREPROCESSING["min_user_ratings"]].index
            valid_movies = movie_counts[movie_counts >= PREPROCESSING["min_movie_ratings"]].index
            
            self.ratings = self.ratings[
                (self.ratings['user_id'].isin(valid_users)) &
                (self.ratings['movie_id'].isin(valid_movies))
            ]
            
            # Create user-movie graph
            self._create_graph()
            
            # Process categorical features
            self._process_categorical_features()
            
            logger.info("Data preprocessing completed")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _create_graph(self) -> None:
        """
        Create a user-movie bipartite graph.
        
        This method constructs a graph where users and movies are nodes,
        and ratings are edge weights.
        """
        logger.info("Creating user-movie graph...")
        
        try:
            # Create bipartite graph
            self.graph = nx.Graph()
            
            # Add nodes
            self.graph.add_nodes_from(self.ratings['user_id'].unique(), bipartite=0)
            self.graph.add_nodes_from(self.ratings['movie_id'].unique(), bipartite=1)
            
            # Add edges with weights
            for _, row in self.ratings.iterrows():
                self.graph.add_edge(
                    row['user_id'],
                    row['movie_id'],
                    weight=row['rating']
                )
            
            # Convert to PyTorch Geometric format
            edge_index = np.array(list(self.graph.edges())).T
            edge_attr = np.array([self.graph[u][v]['weight'] for u, v in self.graph.edges()])
            
            self.edge_index = torch.tensor(edge_index, dtype=torch.long)
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            logger.info("Graph created successfully")
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise
    
    def _process_categorical_features(self) -> None:
        """
        Process categorical features into numerical representations.
        
        This method converts categorical features like genres into numerical
        representations suitable for the model.
        """
        logger.info("Processing categorical features...")
        
        try:
            # Process genre features
            genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
            genre_matrix = self.movies[genre_cols].values
            
            # Compute FFT of genre vectors
            self.fft_values = np.fft.fft(genre_matrix, axis=1)
            
            logger.info("Categorical features processed")
        except Exception as e:
            logger.error(f"Error processing categorical features: {str(e)}")
            raise
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        nx.Graph, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Get the processed data.
        
        Returns:
            Tuple containing:
            - ratings DataFrame
            - users DataFrame
            - movies DataFrame
            - graph NetworkX object
            - edge_index tensor
            - edge_attr tensor
            - fft_values array
        """
        return (self.ratings, self.users, self.movies,
                self.graph, self.edge_index, self.edge_attr, self.fft_values)
    
    def save_processed_data(self, output_dir: Path) -> None:
        """
        Save the processed data to disk.
        
        Args:
            output_dir (Path): Directory to save the processed data
        """
        logger.info("Saving processed data...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrames
            self.ratings.to_csv(output_dir / "ratings.csv", index=False)
            self.users.to_csv(output_dir / "users.csv", index=False)
            self.movies.to_csv(output_dir / "movies.csv", index=False)
            
            # Save graph
            nx.write_gexf(self.graph, output_dir / "graph.gexf")
            
            # Save tensors
            torch.save(self.edge_index, output_dir / "edge_index.pt")
            torch.save(self.edge_attr, output_dir / "edge_attr.pt")
            
            # Save FFT values
            np.save(output_dir / "fft_values.npy", self.fft_values)
            
            logger.info("Processed data saved successfully")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise 