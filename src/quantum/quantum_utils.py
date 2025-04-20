"""
Utility functions for quantum computing operations.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_parameterized_circuit(num_qubits: int, num_params: int) -> QuantumCircuit:
    """
    Create a parameterized quantum circuit.
    
    Args:
        num_qubits: Number of qubits in the circuit
        num_params: Number of parameters in the circuit
        
    Returns:
        QuantumCircuit: Parameterized quantum circuit
    """
    circuit = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', num_params)
    
    # Add parameterized rotations
    for i in range(num_qubits):
        circuit.ry(params[i], i)
        
    # Add entangling gates
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
        
    return circuit

def encode_data(data: np.ndarray, num_qubits: int) -> np.ndarray:
    """
    Encode classical data into quantum state amplitudes.
    
    Args:
        data: Classical data to encode
        num_qubits: Number of qubits to use for encoding
        
    Returns:
        np.ndarray: Encoded quantum state
    """
    # Normalize data
    data = data / np.linalg.norm(data)
    
    # Pad or truncate to match number of qubits
    target_size = 2**num_qubits
    if len(data) < target_size:
        data = np.pad(data, (0, target_size - len(data)))
    else:
        data = data[:target_size]
        
    return data

def measure_circuit(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Measure a quantum circuit and return the results.
    
    Args:
        circuit: Quantum circuit to measure
        shots: Number of measurement shots
        
    Returns:
        dict: Measurement results
    """
    # Add measurement gates
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    
    return measured_circuit 