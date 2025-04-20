"""
Interface for quantum computing backends.
"""

from typing import Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

class QuantumBackend:
    """
    Interface for quantum computing backends.
    """
    
    def __init__(self, backend_name: str = "ibmq_qasm_simulator"):
        """
        Initialize the quantum backend.
        
        Args:
            backend_name: Name of the backend to use
        """
        self.backend_name = backend_name
        self.service = None
        self.backend = None
        self.session = None
        
    def connect(self, api_token: Optional[str] = None):
        """
        Connect to the quantum backend.
        
        Args:
            api_token: IBM Quantum API token
        """
        if api_token:
            self.service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
        else:
            self.service = QiskitRuntimeService()
            
        self.backend = self.service.get_backend(self.backend_name)
        
    def start_session(self):
        """
        Start a quantum computing session.
        """
        if not self.backend:
            raise ValueError("Not connected to backend")
            
        self.session = Session(backend=self.backend)
        
    def end_session(self):
        """
        End the quantum computing session.
        """
        if self.session:
            self.session.close()
            self.session = None
            
    def run_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Run a quantum circuit on the backend.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of measurement shots
            
        Returns:
            Dict[str, Any]: Results from running the circuit
        """
        if not self.session:
            raise ValueError("No active session")
            
        sampler = Sampler(session=self.session)
        result = sampler.run(circuit, shots=shots).result()
        
        return result.quasi_dists[0] 