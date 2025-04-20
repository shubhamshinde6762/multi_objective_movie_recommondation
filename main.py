import logging
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    QUANTUM_SETTINGS, RANDOM_SEED
)
from src.data.processor import DataProcessor
from src.models.quantum_model import QuantumModel
from src.analysis.analyzer import (
    TopologicalAnalyzer,
    OutlierDetector,
    FairnessAnalyzer
)

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Set random seeds
        torch.manual_seed(RANDOM_SEED)
        
        # Initialize components
        data_processor = DataProcessor()
        topological_analyzer = TopologicalAnalyzer()
        outlier_detector = OutlierDetector()
        fairness_analyzer = FairnessAnalyzer()
        
        # Pipeline steps
        logger.info("1. Loading and preprocessing data...")
        data_processor.load_data()
        data_processor.preprocess_data()
        ratings, users, movies, graph, edge_index, edge_attr, fft_values = data_processor.get_processed_data()
        data_processor.save_processed_data(RESULTS_DIR / "processed_data")
        
        logger.info("2. Performing topological analysis...")
        persistence, landscape = topological_analyzer.analyze_genres(movies)
        topological_analyzer.save_analysis(RESULTS_DIR / "topology")
        
        logger.info("3. Detecting outliers...")
        outlier_scores = outlier_detector.detect_outliers(ratings, users)
        outlier_detector.save_outliers(RESULTS_DIR / "outliers")
        
        logger.info("4. Analyzing fairness...")
        fairness_scores = fairness_analyzer.analyze_fairness(ratings, users)
        fairness_analyzer.save_fairness(RESULTS_DIR / "fairness")
        
        logger.info("5. Training quantum model...")
        X = torch.tensor(ratings[['rating', 'time_decay']].values, dtype=torch.float32)
        y = torch.tensor(ratings['rating'].values, dtype=torch.float32).view(-1, 1)
        
        model = QuantumModel(
            input_size=X.shape[1],
            hidden_size=QUANTUM_SETTINGS["hidden_size"],
            num_qubits=QUANTUM_SETTINGS["n_qubits"]
        )
        optimizer = optim.Adam(model.parameters(), lr=QUANTUM_SETTINGS["learning_rate"])
        
        # Training loop
        n_epochs = QUANTUM_SETTINGS["n_epochs"]
        batch_size = QUANTUM_SETTINGS["batch_size"]
        n_batches = len(X) // batch_size
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                loss = model.train_step(X_batch, y_batch, optimizer)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        model.save(MODELS_DIR / "quantum_model.pt")
        logger.info("Pipeline completed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 