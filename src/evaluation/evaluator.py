import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error,
    median_absolute_error, mean_squared_log_error
)
import torch
import json
from pathlib import Path
import pandas as pd
from scipy import stats
import logging

class Evaluator:
    def __init__(self, model, save_dir, batch_size=1000):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.error_stats = {}
        self.pred_stats = {}
        self.error_percentiles = {}
        self.accuracy_ranges = {}
        
    def _batch_predict(self, X):
        """Make predictions in batches to handle large datasets"""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                # Convert numpy array to tensor if needed
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                batch_pred = self.model(batch).numpy()
                predictions.append(batch_pred)
        return np.vstack(predictions)
        
    def evaluate(self, X, y_true):
        """Comprehensive model evaluation with large-scale support"""
        # Convert to numpy if tensors
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
            
        # Make predictions in batches
        y_pred = self._batch_predict(X)
        
        # Ensure arrays are flattened
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Calculate errors
        errors = y_pred - y_true
        
        # Calculate basic metrics
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'max_error': float(max_error(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
            'medae': float(median_absolute_error(y_true, y_pred)),
            'msle': float(mean_squared_log_error(y_true, y_pred)),
            'rmsle': float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
        }
        
        # Calculate error statistics
        self.error_stats = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'median': float(np.median(errors)),
            'skewness': float(stats.skew(errors)),
            'kurtosis': float(stats.kurtosis(errors))
        }
        
        # Calculate prediction statistics
        self.pred_stats = {
            'true_mean': float(np.mean(y_true)),
            'true_std': float(np.std(y_true)),
            'pred_mean': float(np.mean(y_pred)),
            'pred_std': float(np.std(y_pred)),
            'true_median': float(np.median(y_true)),
            'pred_median': float(np.median(y_pred))
        }
        
        # Calculate error percentiles
        self.error_percentiles = {
            'p10': float(np.percentile(errors, 10)),
            'p25': float(np.percentile(errors, 25)),
            'p50': float(np.percentile(errors, 50)),
            'p75': float(np.percentile(errors, 75)),
            'p90': float(np.percentile(errors, 90))
        }
        
        # Calculate accuracy ranges
        self.accuracy_ranges = {
            'within_0.5': float(np.mean(np.abs(errors) <= 0.5)),
            'within_1.0': float(np.mean(np.abs(errors) <= 1.0)),
            'within_1.5': float(np.mean(np.abs(errors) <= 1.5))
        }
        
        results = {
            'metrics': metrics,
            'error_stats': self.error_stats,
            'pred_stats': self.pred_stats,
            'error_percentiles': self.error_percentiles,
            'accuracy_ranges': self.accuracy_ranges
        }
        
        # Save results
        self._save_results(results, y_true, y_pred, errors)
        
        return results
    
    def _save_results(self, results, y_true, y_pred, errors):
        """Save evaluation results to files"""
        # Save metrics to JSON
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'true_values': y_true,
            'predicted_values': y_pred,
            'errors': errors,
            'absolute_errors': np.abs(errors),
            'squared_errors': errors ** 2,
            'percentage_errors': (errors / y_true) * 100
        })
        results_df.to_csv(self.save_dir / 'detailed_results.csv', index=False)
        
        # Save summary statistics
        summary_df = pd.DataFrame({
            'Metric': list(results['metrics'].keys()),
            'Value': list(results['metrics'].values())
        })
        summary_df.to_csv(self.save_dir / 'summary_statistics.csv', index=False)
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in a formatted way"""
        print("\nModel Evaluation Results")
        print("=" * 50)
        
        print("\n1. Basic Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.upper():15s}: {value:.6f}")
            
        print("\n2. Error Statistics:")
        print("-" * 30)
        for stat, value in self.error_stats.items():
            print(f"{stat.capitalize():15s}: {value:.6f}")
            
        print("\n3. Prediction Statistics:")
        print("-" * 30)
        for stat, value in self.pred_stats.items():
            print(f"{stat.capitalize():15s}: {value:.6f}")
            
        print("\n4. Error Percentiles:")
        print("-" * 30)
        for percentile, value in self.error_percentiles.items():
            print(f"{percentile.upper():15s}: {value:.6f}")
            
        print("\n5. Accuracy Ranges:")
        print("-" * 30)
        for range_name, value in self.accuracy_ranges.items():
            print(f"{range_name.capitalize():15s}: {value*100:.2f}%") 