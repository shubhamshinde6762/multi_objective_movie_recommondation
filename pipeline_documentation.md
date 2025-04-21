# Quantum-Enhanced Recommendation System Pipeline Documentation

## 🔹 **1. Initialization and Configuration**

**File**: `main.py`  
**Input**: None  
**Output**: Initialized config, seed, logs, directory structure  

**Sequence of Execution**:
1. Load `config.py`  
   - Parses paths, hyperparameters, toggles  
2. Initialize logging  
   - Log file at `logs/init.log`  
3. Set global seeds  
   - Ensures reproducibility (e.g., NumPy, PyTorch)  
4. Create directory structure  
   - `results/`, `models/`, `data/processed/`, `results/visualizations/`

**Outcome**:
- Configuration object ready
- Logging and reproducibility ensured
- Directory skeleton prepared for all following steps

---

## 🔹 **2. Data Loading and Preprocessing**

**File**: `src/data/processor.py`  
**Input**:  
- `data/raw/u.data`  
- `data/raw/u.item`  
- `data/raw/u.user`  
- `data/raw/u.genre`

**Output**:  
- `data/processed/ratings.pt`  
- `data/processed/movies.pt`  
- `data/processed/users.pt`  
- `data/processed/features.pt`

**Sequence**:
1. **load_data()**  
   - Read raw files into DataFrames  
   - Handle missing values, format timestamps  

2. **preprocess_data()**  
   - Normalize rating scale  
   - One-hot encode genres  
   - Encode demographics  

3. **create_features()**  
   - **Time Decay (α = 0.95)**  
   - **Genre vectors** (19-D binary)  
   - **User Preference Vectors** (mean rating by genre)  
   - **Quantum embeddings** via parameterized circuits  

4. **save_processed_data()**  
   - Save all DataFrames and tensors as `.pt` files  

**Outcome**:
- Cleaned, normalized data  
- Rich feature set with classical and quantum attributes  
- Ready for graph creation

---

## 🔹 **3. Graph Creation**

**File**: `src/data/graph.py`  
**Input**:  
- `ratings.pt`, `users.pt`, `movies.pt`  

**Output**:  
- `graph.pt` (heterogeneous graph object)  
- `embeddings.pt` (graph-level node and edge embeddings)

**Sequence**:
1. **create_graph()**  
   - Nodes: Users and Movies  
   - Edges: Ratings (with weights)  
   - Node features: demographics, genres, time decay  

2. **generate_embeddings()**  
   - Use GNN or manual embedding layer  
   - Generate:  
     - **User embeddings** (demographic + historical + graph)  
     - **Movie embeddings** (genre + structure)  
   - Save both as tensors  

**Outcome**:
- Bipartite graph encoding rating behavior  
- Node and edge embeddings available for model input

---

## 🔹 **4. Model Training**

**File**: `src/models/quantum_model.py`  
**Input**:  
- `features.pt`, `graph.pt`, `embeddings.pt`  

**Output**:  
- `models/quantum_model.pt`  
- `results/training_history.csv`

**Sequence**:
1. **initialize_model()**  
   - Create hybrid model:  
     - Classical NN layers  
     - Quantum circuit with 2 qubits  
     - Feature fusion layer  

2. **train()**  
   - Batch training on rating prediction  
   - Use MSE or RMSE loss  
   - Save intermediate checkpoints and logs  

3. **validate()**  
   - Periodic evaluation on validation set  
   - Compute loss, R², MAE, RMSE  
   - Log to CSV  

**Outcome**:
- Fully trained hybrid quantum-classical model  
- Complete training logs and metrics

---

## 🔹 **5. Evaluation**

**File**: `src/evaluation/evaluator.py`  
**Input**:  
- `quantum_model.pt`, `features.pt`, `graph.pt`

**Output**:  
- `metrics.csv`, `predictions.pt`, `errors.pt`  
  (All saved in `results/evaluation/`)

**Sequence**:
1. **load_model()**  
   - Load trained model in eval mode  

2. **evaluate()**  
   - Predict on test set  
   - Metrics:  
     - MSE, RMSE, MAE, R²  
     - RMSLE, Explained Variance, MSLE  

3. **analyze_errors()**  
   - Save residuals and error analysis  
   - Compute skew, kurtosis, error percentiles  

4. **save_results()**  
   - Store everything in organized files  

**Outcome**:
- Full metric breakdown  
- Predicted vs actual comparisons  
- In-depth error analysis

---

## 🔹 **6. Visualization**

**File**: `src/visualization/visualizer.py`  
**Input**:  
- `metrics.csv`, `predictions.pt`, `errors.pt`

**Output**:  
- `visualizations/*.png`  
  (6+ plots saved in `results/visualizations/`)

**Sequence**:
1. **plot_training_history()**  
   - Loss, accuracy curves  

2. **plot_predictions()**  
   - Scatter: true vs predicted  
   - Residuals and error heatmaps  

3. **plot_feature_importance()**  
   - Feature contributions from SHAP/weights  

4. **plot_distributions()**  
   - Rating distributions, genre histograms  

5. **plot_metrics()**  
   - Visual comparison of different metrics  

**Outcome**:
- Intuitive plots for performance, fairness, distribution  
- Useful for analysis and final report

---

## 🔹 **7. Analysis**

**File**: `src/analysis/analyzer.py`  
**Input**:  
- `metrics.csv`, `predictions.pt`, `errors.pt`

**Output**:  
- Topology: `results/analysis/topology/`  
- Outliers: `results/analysis/outliers/`  
- Fairness: `results/analysis/fairness/`

**Sequence**:
1. **analyze_topology()**  
   - Generate persistence diagrams  
   - Landscape and barcode plots  

2. **detect_outliers()**  
   - Identify users/movies with high errors  
   - Save plots and statistics  

3. **analyze_fairness()**  
   - Check gender/age bias in predictions  
   - Demographic parity, equal opportunity  

**Outcome**:
- Topological Data Analysis (TDA) reports  
- Fairness and ethical auditing  
- Outlier identification

---

## 🔹 **8. Final Report Generation**

**File**: `src/reporting/reporter.py`  
**Input**:  
- All metrics, plots, and logs  

**Output**:  
- `results/final_report.md`  
- `results/summary.csv`

**Sequence**:
1. **generate_report()**  
   - Combine metrics, errors, and visualizations  
   - Summarize model architecture and config  

2. **save_report()**  
   - Output a human-readable `.md` file  
   - Save tabular summary of results  

**Outcome**:
- Ready-to-share final report  
- Contains all experiment results and visuals  
- Includes insights on fairness, accuracy, and recommendations

---

## 🔹 Execution Modes

To execute the entire pipeline or individual steps:

```bash
# Run complete pipeline
python main.py --step all

# Run specific steps
python main.py --step preprocess
python main.py --step train
python main.py --step evaluate
python main.py --step visualize
python main.py --step analyze
python main.py --step report
```

---

## Summary Table of Artifacts

| Stage         | Output Directory                  | Key Files Generated                                |
|---------------|-----------------------------------|----------------------------------------------------|
| Initialization| `logs/`, `results/`               | `init.log`                                         |
| Preprocessing | `data/processed/`                 | `ratings.pt`, `features.pt`, `movies.pt`, etc.     |
| Graph         | `data/processed/`                 | `graph.pt`, `embeddings.pt`                        |
| Training      | `models/`, `results/`             | `quantum_model.pt`, `training_history.csv`         |
| Evaluation    | `results/evaluation/`             | `metrics.csv`, `predictions.pt`, `errors.pt`       |
| Visualization | `results/visualizations/`         | Multiple `.png` plots                              |
| Analysis      | `results/analysis/`               | TDA plots, fairness reports, outlier summaries     |
| Report        | `results/`                        | `final_report.md`, `summary.csv`                   | 