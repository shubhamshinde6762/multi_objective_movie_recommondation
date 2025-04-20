"""
Quantum Model for Movie Recommendations.

This module implements a quantum neural network for movie rating predictions,
using parameterized quantum circuits and classical neural networks.
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from config import QUANTUM_SETTINGS
from qiskit.utils import QuantumInstance

logger = logging.getLogger(__name__)

class QuantumLayer(nn.Module):
    """
    Quantum layer implementing a parameterized quantum circuit.
    
    This layer uses a quantum circuit with parameterized rotations and entangling gates
    to process input data in a quantum-enhanced way.
    """
    
    def __init__(self, num_qubits: int, quantum_instance: QuantumInstance):
        """
        Initialize the quantum layer.
        
        Args:
            num_qubits (int): Number of qubits in the quantum circuit
            quantum_instance (QuantumInstance): Quantum instance for quantum circuit execution
        """
        super().__init__()
        self.n_qubits = num_qubits
        self.quantum_instance = quantum_instance
        
        # Create quantum circuit
        circuit = QuantumCircuit(num_qubits)
        
        # Add weight parameters
        weight_params = []
        for qubit in range(self.n_qubits):
            param = Parameter(f'weight_{qubit}')
            weight_params.append(param)
            circuit.rx(param, qubit)
        
        # Add measurement
        circuit.measure_all()
        self.circuit = circuit
        self.weight_params = weight_params
        
        # Initialize weights
        self.weights = nn.Parameter(torch.randn(len(weight_params)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Convert input to numpy array
            x_i = x[i].detach().cpu().numpy().astype(np.float32)
            
            # Bind parameters
            parameter_binds = dict(zip(self.weight_params, x_i))
            bound_circuit = self.circuit.bind_parameters(parameter_binds)
            
            # Execute circuit
            job = self.quantum_instance.execute(bound_circuit)
            result = job.get_counts()
            
            # Convert result to probabilities
            probs = np.zeros(2**self.n_qubits)
            total_shots = sum(result.values())
            for bitstring, count in result.items():
                idx = int(bitstring, 2)
                probs[idx] = count / total_shots
            
            outputs.append(torch.tensor(probs, dtype=torch.float32))
        
        return torch.stack(outputs)

class QuantumModel(nn.Module):
    """
    Quantum-enhanced neural network for movie rating predictions.
    
    This model combines quantum and classical neural networks to predict
    movie ratings based on user preferences and movie features.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_qubits: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits
        
        # Initialize quantum instance
        self.quantum_instance = QuantumInstance(
            Aer.get_backend('aer_simulator'),
            shots=1024
        )
        
        # Create quantum layer
        self.quantum_layer = QuantumLayer(num_qubits, self.quantum_instance)
        
        # Neural network layers
        self.pre_quantum = nn.Linear(input_size, num_qubits)
        self.post_quantum = nn.Linear(2**num_qubits, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is float32
        x = x.to(torch.float32)
        
        # Pre-quantum processing
        x = self.pre_quantum(x)
        
        # Quantum processing
        quantum_out = self.quantum_layer(x)
        
        # Post-quantum processing
        x = self.post_quantum(quantum_out)
        x = self.activation(x)
        x = self.output(x)
        
        return x
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform a single training step.
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters
            
        Returns:
            float: Loss value
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = self.forward(x)
        
        # Calculate loss
        loss = nn.MSELoss()(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        return loss.item()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the trained model.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        self.eval()
        with torch.no_grad():
            predictions = self(x)
        return predictions
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_qubits': self.num_qubits,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'QuantumModel':
        """Load model state"""
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_qubits=checkpoint['num_qubits']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model 