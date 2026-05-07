"""
Angle encoding for quantum circuits.

Encodes classical data into quantum states using rotation gates.
"""

import numpy as np
from qiskit import QuantumCircuit


class AngleEncoding:
    """
    Encode classical features into quantum states using Ry rotations.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit
    entangle : bool, default=False
        If True, add CX gates in a chain pattern (0→1→2→...)
        
    Examples
    --------
    >>> encoder = AngleEncoding(n_qubits=4)
    >>> x = np.array([0.1, 0.5, 1.2, 2.0])
    >>> qc = encoder.build_circuit(x)
    >>> print(qc)
    """
    
    def __init__(self, n_qubits: int, entangle: bool = False):
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        
        self.n_qubits = n_qubits
        self.entangle = entangle
    
    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build quantum circuit encoding the input data.
        
        Parameters
        ----------
        x : np.ndarray
            Input feature vector. If len(x) > n_qubits, features wrap around
            using modulo indexing.
            
        Returns
        -------
        qc : QuantumCircuit
            Quantum circuit with Ry rotations (and optional entanglement)
        """
        x = np.asarray(x).flatten()
        
        if len(x) == 0:
            raise ValueError("Input array x cannot be empty")
        
        # Create circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply Ry rotations
        for i in range(len(x)):
            qubit_idx = i % self.n_qubits
            qc.ry(x[i], qubit_idx)
        
        # Optional entanglement layer
        if self.entangle:
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AngleEncoding")
    print("=" * 60)
    
    # Test 1: Basic encoding
    print("\n1. Basic angle encoding (4 qubits, 4 features)")
    encoder = AngleEncoding(n_qubits=4)
    x = np.array([0.5, 1.0, 1.5, 2.0])
    qc = encoder.build_circuit(x)
    print(qc)
    print(f"   Circuit depth: {qc.depth()}")
    print(f"   Number of gates: {len(qc.data)}")
    
    # Test 2: With entanglement
    print("\n2. Angle encoding with entanglement")
    encoder_ent = AngleEncoding(n_qubits=4, entangle=True)
    qc_ent = encoder_ent.build_circuit(x)
    print(qc_ent)
    print(f"   Circuit depth: {qc_ent.depth()}")
    print(f"   Number of gates: {len(qc_ent.data)}")
    
    # Test 3: Wrap around (more features than qubits)
    print("\n3. Wrap around: 8 features on 4 qubits")
    x_long = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qc_wrap = encoder.build_circuit(x_long)
    print(qc_wrap)
    print(f"   Input length: {len(x_long)}")
    print(f"   Qubits: {encoder.n_qubits}")
    print(f"   Each qubit gets 2 rotations (wrap around)")
    
    # Test 4: Fewer features than qubits
    print("\n4. Fewer features than qubits: 2 features on 4 qubits")
    x_short = np.array([1.0, 2.0])
    qc_short = encoder.build_circuit(x_short)
    print(qc_short)
    print(f"   Only first 2 qubits are rotated")
    
    # Test 5: Statevector simulation (optional, if qiskit_aer available)
    print("\n5. Statevector example (3 qubits)")
    try:
        from qiskit_aer import AerSimulator
        
        encoder3 = AngleEncoding(n_qubits=3)
        x3 = np.array([np.pi/4, np.pi/2, np.pi])
        qc3 = encoder3.build_circuit(x3)
        
        # Save statevector
        qc3.save_statevector()
        
        # Run simulation
        simulator = AerSimulator(method='statevector')
        result = simulator.run(qc3).result()
        statevector = result.get_statevector()
        
        print(f"   Input angles: {x3}")
        print(f"   Statevector shape: {statevector.data.shape}")
        probs = np.abs(statevector.data) ** 2
        print(f"   Sum of probabilities = {probs.sum():.6f} (should be 1.0)")
        
    except ImportError:
        print("   (Skipping - qiskit_aer not available)")
    
    # Test 6: Error handling
    print("\n6. Error handling")
    try:
        bad_encoder = AngleEncoding(n_qubits=0)
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    try:
        encoder.build_circuit(np.array([]))
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)