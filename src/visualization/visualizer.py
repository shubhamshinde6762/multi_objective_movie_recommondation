import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_history(self, train_losses, val_losses, learning_rates=None):
        """Plot training and validation loss history with learning rate"""
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('1.1 Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate if available
        if learning_rates:
            plt.subplot(1, 2, 2)
            plt.plot(learning_rates, color='green')
            plt.title('1.2 Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '01_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_prediction_analysis(self, y_true, y_pred, errors):
        """Plot comprehensive prediction analysis"""
        plt.figure(figsize=(15, 10))
        
        # 1. True vs Predicted Values
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title('1. True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        # 2. Error Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('2. Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.grid(True)
        
        # 3. Error vs True Values
        plt.subplot(2, 2, 3)
        plt.scatter(y_true, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('3. Error vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Error')
        plt.grid(True)
        
        # 4. Error vs Predicted Values
        plt.subplot(2, 2, 4)
        plt.scatter(y_pred, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('4. Error vs Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '02_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self, feature_names, importance_scores):
        """Plot feature importance analysis"""
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('3.1 Feature Importance Analysis')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '03_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_distribution_analysis(self, ratings, predictions):
        """Plot distribution analysis of ratings and predictions"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(ratings, kde=True, label='True Ratings')
        plt.title('4.1 True Ratings Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        sns.histplot(predictions, kde=True, label='Predicted Ratings')
        plt.title('4.2 Predicted Ratings Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '04_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_analysis(self, features, target):
        """Plot correlation analysis between features and target"""
        plt.figure(figsize=(12, 6))
        
        # Convert numpy arrays to pandas Series
        if isinstance(features, np.ndarray):
            features = pd.Series(features.flatten())
        if isinstance(target, np.ndarray):
            target = pd.Series(target.flatten())
            
        # Calculate correlation
        correlation = features.corr(target)
        
        # Create a bar plot
        plt.bar(['Correlation'], [correlation])
        plt.title('5.1 Feature-Target Correlation')
        plt.ylabel('Correlation Coefficient')
        plt.ylim(-1, 1)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '05_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_metrics(self, results):
        """Plot comprehensive performance metrics"""
        plt.figure(figsize=(15, 10))
        
        # 1. Basic Metrics
        plt.subplot(2, 2, 1)
        basic_metrics = {
            'MSE': results['metrics']['mse'],
            'RMSE': results['metrics']['rmse'],
            'MAE': results['metrics']['mae'],
            'R²': results['metrics']['r2'],
            'Explained Variance': results['metrics']['explained_variance'],
            'Max Error': results['metrics']['max_error'],
            'MAPE': results['metrics']['mape'],
            'MedAE': results['metrics']['medae'],
            'MSLE': results['metrics']['msle'],
            'RMSLE': results['metrics']['rmsle']
        }
        plt.bar(range(len(basic_metrics)), list(basic_metrics.values()))
        plt.title('1. Basic Performance Metrics')
        plt.xticks(range(len(basic_metrics)), list(basic_metrics.keys()), rotation=45)
        plt.ylabel('Value')
        plt.grid(True)
        
        # 2. Error Statistics
        plt.subplot(2, 2, 2)
        error_stats = results['error_stats']
        plt.bar(range(len(error_stats)), list(error_stats.values()))
        plt.title('2. Error Statistics')
        plt.xticks(range(len(error_stats)), list(error_stats.keys()), rotation=45)
        plt.ylabel('Value')
        plt.grid(True)
        
        # 3. Error Percentiles
        plt.subplot(2, 2, 3)
        error_percentiles = results['error_percentiles']
        plt.bar(range(len(error_percentiles)), list(error_percentiles.values()))
        plt.title('3. Error Percentiles')
        plt.xticks(range(len(error_percentiles)), list(error_percentiles.keys()), rotation=45)
        plt.ylabel('Value')
        plt.grid(True)
        
        # 4. Accuracy Ranges
        plt.subplot(2, 2, 4)
        accuracy_ranges = results['accuracy_ranges']
        plt.bar(range(len(accuracy_ranges)), [v * 100 for v in accuracy_ranges.values()])
        plt.title('4. Accuracy Ranges')
        plt.xticks(range(len(accuracy_ranges)), list(accuracy_ranges.keys()), rotation=45)
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / '06_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close() 