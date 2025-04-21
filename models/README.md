# Models Directory Documentation

## 📁 Directory Structure

```
models/
├── neural_network.py    # Classical neural network model
├── quantum_model.py     # Quantum-enhanced model
├── hybrid_model.py      # Combined classical-quantum model
└── checkpoints/         # Saved model states
    ├── epoch_1.pt
    ├── epoch_5.pt
    └── best_model.pt
```

## 🤖 Model Architectures

### 1. Neural Network Model
```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

**Configuration**:
- Input size: 32 (feature dimension)
- Hidden size: 128
- Output size: 1 (rating prediction)
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-4

### 2. Quantum Model
```python
class QuantumModel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits)
        self.classical_net = nn.Linear(n_qubits, 1)
        
    def forward(self, x):
        quantum_state = self.quantum_circuit(x)
        return self.classical_net(quantum_state)
```

**Configuration**:
- Number of qubits: 2
- Number of layers: 3
- Quantum gates: RX, RY, CZ
- Classical post-processing: Linear layer

### 3. Hybrid Model
```python
class HybridModel(nn.Module):
    def __init__(self, neural_net, quantum_net):
        super().__init__()
        self.neural_net = neural_net
        self.quantum_net = quantum_net
        self.fusion = nn.Linear(2, 1)
        
    def forward(self, x):
        classical_out = self.neural_net(x)
        quantum_out = self.quantum_net(x)
        combined = torch.cat([classical_out, quantum_out], dim=1)
        return self.fusion(combined)
```

## 🏋️ Training Process

### Training Configuration
```python
training_config = {
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'validation_split': 0.2
}
```

### Loss Functions
1. **Mean Squared Error (MSE)**
   ```python
   criterion = nn.MSELoss()
   ```

2. **Custom Hybrid Loss**
   ```python
   def hybrid_loss(pred, target, alpha=0.5):
       mse = nn.MSELoss()(pred, target)
       quantum_loss = quantum_circuit_loss(pred)
       return alpha * mse + (1 - alpha) * quantum_loss
   ```

## 📊 Model Comparison

| Feature | Neural Network | Quantum Model | Hybrid Model |
|---------|---------------|---------------|--------------|
| Speed   | Fast          | Slower        | Medium       |
| Memory  | Low           | High          | Medium       |
| Accuracy| High          | Comparable    | Best         |
| Scalability| Excellent | Limited       | Good         |
| Training Time| Short    | Long          | Medium       |

## 📝 Usage

### Training
```python
from src.models.neural_network import NeuralNetwork
from src.models.quantum_model import QuantumModel
from src.models.hybrid_model import HybridModel

# Initialize models
neural_net = NeuralNetwork(32, 128, 1)
quantum_net = QuantumModel(2, 3)
hybrid_model = HybridModel(neural_net, quantum_net)

# Train model
trainer = ModelTrainer(hybrid_model)
trainer.train(train_data, val_data)
```

### Inference
```python
# Load saved model
model = torch.load('models/checkpoints/best_model.pt')
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(test_data)
```

## 🔍 Model Evaluation

### Metrics
1. **Accuracy Metrics**
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - R² Score

2. **Quantum Metrics**
   - Quantum Circuit Depth
   - Gate Count
   - Quantum Volume

3. **Hybrid Metrics**
   - Classical Component Contribution
   - Quantum Component Contribution
   - Fusion Effectiveness

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Model Implementation Code](src/models/) 