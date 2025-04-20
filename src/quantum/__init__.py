"""
Quantum computing module for hybrid quantum-classical machine learning.
"""

from .quantum_interface import QuantumBackend
from .quantum_utils import (
    create_parameterized_circuit,
    encode_data,
    measure_circuit
)

__all__ = [
    'QuantumBackend',
    'create_parameterized_circuit',
    'encode_data',
    'measure_circuit'
] 