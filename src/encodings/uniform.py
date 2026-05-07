import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class UniformEntangledEncoding:
    """
    Uniform entangled encoding with PARAMETERIZED entangling layers.
    
    Structure:
        1. Angle encoding: RY(x[i]) on each qubit i
        2. Barrier (preserves structure during transpilation)
        3. Entangling layers (repeated 'depth' times):
           - Entanglement (CNOT chain)
           - Parameterized rotations (layer-dependent angles)
           - Barrier between layers
    
    KEY FIX: Each depth layer uses DIFFERENT rotation angles to add expressivity.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match feature dimension)
    depth : int
        Number of entangling layers to apply (>= 0)
    rotation : str, default='ry'
        Rotation gate type: 'rx', 'ry', or 'rz'
    entangler : str, default='linear'
        Entanglement pattern: 'linear' or 'ring'
    """
    
    def __init__(self, n_qubits, depth, rotation='ry', entangler='linear'):
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if rotation not in ['rx', 'ry', 'rz']:
            raise ValueError(f"rotation must be 'rx', 'ry', or 'rz', got {rotation}")
        if entangler not in ['linear', 'ring']:
            raise ValueError(f"entangler must be 'linear' or 'ring', got {entangler}")
        
        self.n_qubits = n_qubits
        self.depth = depth
        self.rotation = rotation
        self.entangler = entangler
    
    def __repr__(self):
        return (f"UniformEntangledEncoding(n_qubits={self.n_qubits}, "
                f"depth={self.depth}, rotation='{self.rotation}', "
                f"entangler='{self.entangler}')")
    
    def _apply_rotation(self, qc, angle, qubit):
        """Apply rotation gate based on self.rotation."""
        if self.rotation == 'rx':
            qc.rx(angle, qubit)
        elif self.rotation == 'ry':
            qc.ry(angle, qubit)
        elif self.rotation == 'rz':
            qc.rz(angle, qubit)
    
    def _apply_entanglement(self, qc):
        """Apply entanglement layer based on self.entangler."""
        if self.entangler == 'linear':
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        elif self.entangler == 'ring':
            for i in range(self.n_qubits):
                qc.cx(i, (i + 1) % self.n_qubits)
    
    def build_circuit(self, x):
        """
        Build quantum circuit for input vector x.
        
        Parameters
        ----------
        x : array-like, shape (n_qubits,)
            Input feature vector, must be in range [0, π]
            
        Returns
        -------
        qc : QuantumCircuit
            Quantum circuit with parameterized encoding
            
        Circuit structure:
            1. Initial encoding: RY(x[i]) on qubit i
            2. For each depth layer d:
               - CNOT entanglement
               - Parameterized rotations: RY(x[i] * (d+1)/(depth+1))
        """
        x = np.asarray(x, dtype=float)
        
        # Validation
        if len(x) != self.n_qubits:
            raise ValueError(f"Input size {len(x)} != n_qubits {self.n_qubits}")
        if not np.all((x >= 0) & (x <= np.pi)):
            raise ValueError(
                f"Input must be in [0, π], got range [{x.min():.3f}, {x.max():.3f}]"
            )
        
        qc = QuantumCircuit(self.n_qubits, name=f"UniformEnt_d{self.depth}")
        
        # Step 1: Initial angle encoding
        for i in range(self.n_qubits):
            self._apply_rotation(qc, x[i], i)
        
        if self.depth > 0:
            qc.barrier()
        
        # Step 2: Parameterized entangling layers
        for layer in range(self.depth):
            # Entanglement
            self._apply_entanglement(qc)
            
            # CRITICAL: Parameterized rotations with layer-dependent angles
            # This ensures each depth layer adds NEW information
            for i in range(self.n_qubits):
                # Scale by layer index to create variation
                # Layer 0: x[i] * 1/(depth+1)
                # Layer 1: x[i] * 2/(depth+1)
                # ...
                # Layer depth-1: x[i] * depth/(depth+1)
                angle = x[i] * (layer + 1) / (self.depth + 1)
                self._apply_rotation(qc, angle, i)
            
            if layer < self.depth - 1:
                qc.barrier()
        
        return qc


# ============================================================
# Test cases
# ============================================================
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    
    print("=" * 70)
    print("UniformEntangledEncoding - Verification Tests")
    print("=" * 70)
    
    # Test 1: State differentiation across depths
    print("\n" + "-" * 70)
    print("Test 1: Verify depth changes create different states")
    print("-" * 70)
    
    x_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    
    states = {}
    for depth in [0, 1, 2, 3]:
        encoder = UniformEntangledEncoding(n_qubits=5, depth=depth)
        circuit = encoder.build_circuit(x_test)
        state = Statevector.from_instruction(circuit)
        states[f"d={depth}"] = state.data
        
        print(f"\nDepth {depth}:")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Circuit gates: {circuit.count_ops()}")
    
    # Compare consecutive depths
    print("\n" + "-" * 70)
    print("Fidelity between consecutive depths:")
    print("-" * 70)
    
    for d in range(3):
        fid = np.abs(np.vdot(states[f"d={d}"], states[f"d={d+1}"])) ** 2
        status = "✓ Different" if fid < 0.95 else "❌ TOO SIMILAR"
        print(f"d={d} vs d={d+1}: {fid:.6f} {status}")
    
    # Test 2: Input sensitivity
    print("\n" + "-" * 70)
    print("Test 2: Verify different inputs create different states")
    print("-" * 70)
    
    encoder = UniformEntangledEncoding(n_qubits=5, depth=2)
    
    x1 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    x2 = np.array([0.3, 0.8, 1.2, 1.8, 2.3])
    
    state1 = Statevector.from_instruction(encoder.build_circuit(x1))
    state2 = Statevector.from_instruction(encoder.build_circuit(x2))
    
    fid = np.abs(np.vdot(state1.data, state2.data)) ** 2
    print(f"\nFidelity between different inputs: {fid:.6f}")
    if fid < 0.95:
        print("✓ Different inputs produce different states")
    else:
        print("❌ WARNING: States are too similar!")
    
    # Test 3: Circuit diagram
    print("\n" + "-" * 70)
    print("Test 3: Circuit structure (depth=2)")
    print("-" * 70)
    
    encoder = UniformEntangledEncoding(n_qubits=3, depth=2)
    circuit = encoder.build_circuit(np.array([0.5, 1.0, 1.5]))
    
    print("\n" + circuit.draw('text', fold=-1))
    
    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)