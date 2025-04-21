import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import logging
import numpy as np

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    MODEL_SETTINGS, RANDOM_SEED, ENVIRONMENT_SETTINGS
)
from src.data.processor import DataProcessor
from src.models.neural_network import NeuralNetwork
from src.visualization.visualizer import Visualizer
from src.evaluation.evaluator import Evaluator
from src.analysis.analyzer import (
    TopologicalAnalyzer,
    OutlierDetector,
    FairnessAnalyzer
)

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

def main():
    try:
        # Set CPU-specific settings
        torch.set_num_threads(ENVIRONMENT_SETTINGS["torch_num_threads"])
        torch.set_num_interop_threads(ENVIRONMENT_SETTINGS["torch_num_interop_threads"])
        torch.set_float32_matmul_precision('medium')
        
        # Set random seeds
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        # Initialize components
        data_processor = DataProcessor()
        visualizer = Visualizer(RESULTS_DIR / "visualizations")
        
        print("Starting data processing...")
        data_processor.load_data()
        data_processor.preprocess_data()
        ratings, users, movies, graph, edge_index, edge_attr, fft_values = data_processor.get_processed_data()
        
        # Use only essential features
        X = torch.tensor(ratings[['rating']].values, dtype=torch.float32)
        y = torch.tensor(ratings['rating'].values, dtype=torch.float32).view(-1, 1)
        
        # Split data into train and validation
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print("Starting model training...")
        # Create a more powerful model
        model = NeuralNetwork(
            input_size=X.shape[1],
            hidden_size=MODEL_SETTINGS["hidden_dim"],
            output_size=1
        )
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=MODEL_SETTINGS["learning_rate"],
            weight_decay=MODEL_SETTINGS["weight_decay"]
        )
        
        # Training parameters
        n_epochs = MODEL_SETTINGS["num_epochs"]
        batch_size = MODEL_SETTINGS["batch_size"]
        patience = MODEL_SETTINGS["early_stopping_patience"]
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Lists to store training history
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Training loop with early stopping and learning rate monitoring
        for epoch in tqdm(range(n_epochs), desc="Training"):
            # Training
            model.train()
            train_loss = 0.0
            n_batches = len(X_train) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = torch.nn.functional.mse_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = torch.nn.functional.mse_loss(val_outputs, y_val).item()
            
            avg_train_loss = train_loss / n_batches
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check with improved monitoring
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model.save(MODELS_DIR / "model.pt")
                print(f"\nNew best model saved at epoch {epoch + 1} with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Print detailed progress every epoch
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Patience Counter: {patience_counter}/{patience}")
            print("-" * 50)
        
        print("Training completed")
        
        # Generate comprehensive visualizations
        print("\nGenerating visualizations...")
        
        # 1. Training history
        visualizer.plot_training_history(train_losses, val_losses, learning_rates)
        
        # 2. Prediction analysis
        with torch.no_grad():
            y_pred = model(X_val).numpy()
            errors = y_pred - y_val.numpy()
        visualizer.plot_prediction_analysis(y_val.numpy(), y_pred, errors)
        
        # 3. Feature importance (simple version for this model)
        feature_names = ['rating']
        importance_scores = [1.0]  # Since we only have one feature
        visualizer.plot_feature_importance(feature_names, importance_scores)
        
        # 4. Distribution analysis
        visualizer.plot_distribution_analysis(y_val.numpy(), y_pred)
        
        # 5. Correlation analysis
        visualizer.plot_correlation_analysis(X_val.numpy(), y_val.numpy())
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluator = Evaluator(model, RESULTS_DIR / "evaluation")
        eval_results = evaluator.evaluate(X_val, y_val)
        evaluator.print_metrics(eval_results['metrics'])
        
        # 6. Performance metrics visualization
        visualizer.plot_performance_metrics(eval_results)
        
        print("\nPipeline completed successfully")
        print("\nResults saved in:")
        print(f"- Model: {MODELS_DIR}/model.pt")
        print(f"- Evaluation metrics: {RESULTS_DIR}/evaluation/metrics.json")
        print(f"- Detailed results: {RESULTS_DIR}/evaluation/detailed_results.csv")
        print(f"- Visualizations: {RESULTS_DIR}/visualizations/")
        print("\nVisualization files:")
        print("1. Training History:")
        print("   - 01_training_history.png")
        print("2. Prediction Analysis:")
        print("   - 02_prediction_analysis.png")
        print("3. Feature Analysis:")
        print("   - 03_feature_importance.png")
        print("4. Distribution Analysis:")
        print("   - 04_distribution_analysis.png")
        print("5. Correlation Analysis:")
        print("   - 05_correlation_analysis.png")
        print("6. Performance Metrics:")
        print("   - 06_performance_metrics.png")
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 