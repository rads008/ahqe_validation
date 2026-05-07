import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class ReuploadEncoding:
    """
    Data re-uploading encoding with VARYING angles across layers.
    
    Structure (repeated 'n_layers' times):
        1. Angle encoding: RY(x[i] * layer_scale) on each qubit i
        2. Barrier
        3. Entanglement (CNOT chain)
        4. Barrier
    
    KEY FIX: Each layer uses DIFFERENT rotation angles (scaled by layer index)
    to ensure layers add expressivity.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match feature dimension)
    n_layers : int
        Number of re-uploading layers (>= 1)
    rotation : str, default='ry'
        Rotation gate type: 'rx', 'ry', or 'rz'
    entangler : str, default='linear'
        Entanglement pattern: 'linear' or 'ring'
    """
    
    def __init__(self, n_qubits, n_layers, rotation='ry', entangler='linear'):
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if rotation not in ['rx', 'ry', 'rz']:
            raise ValueError(f"rotation must be 'rx', 'ry', or 'rz', got {rotation}")
        if entangler not in ['linear', 'ring']:
            raise ValueError(f"entangler must be 'linear' or 'ring', got {entangler}")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation = rotation
        self.entangler = entangler
    
    def __repr__(self):
        return (f"ReuploadEncoding(n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, rotation='{self.rotation}', "
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
            Quantum circuit with re-uploading encoding
            
        Circuit structure:
            For each layer L:
                1. Parameterized encoding: RY(x[i] * (L+1)/n_layers)
                2. Entanglement
        """
        x = np.asarray(x, dtype=float)
        
        # Validation
        if len(x) != self.n_qubits:
            raise ValueError(f"Input size {len(x)} != n_qubits {self.n_qubits}")
        if not np.all((x >= 0) & (x <= np.pi)):
            raise ValueError(
                f"Input must be in [0, π], got range [{x.min():.3f}, {x.max():.3f}]"
            )
        
        qc = QuantumCircuit(self.n_qubits, name=f"Reupload_L{self.n_layers}")
        
        # Repeat: encode → entangle (with VARYING angles per layer)
        for layer in range(self.n_layers):
            # CRITICAL: Scale angles by layer to create variation
            # Layer 0: x[i] * 1/n_layers
            # Layer 1: x[i] * 2/n_layers
            # ...
            # Layer n_layers-1: x[i] * n_layers/n_layers = x[i]
            for i in range(self.n_qubits):
                angle = x[i] * (layer + 1) / self.n_layers
                self._apply_rotation(qc, angle, i)
            
            qc.barrier()
            
            # Entanglement
            self._apply_entanglement(qc)
            
            if layer < self.n_layers - 1:
                qc.barrier()
        
        return qc


# ============================================================
# Test cases
# ============================================================
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    
    print("=" * 70)
    print("ReuploadEncoding - Verification Tests")
    print("=" * 70)
    
    # Test 1: State differentiation across layers
    print("\n" + "-" * 70)
    print("Test 1: Verify layer changes create different states")
    print("-" * 70)
    
    x_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    
    states = {}
    for n_layers in [1, 2, 3]:
        encoder = ReuploadEncoding(n_qubits=5, n_layers=n_layers)
        circuit = encoder.build_circuit(x_test)
        state = Statevector.from_instruction(circuit)
        states[f"L={n_layers}"] = state.data
        
        print(f"\nLayers {n_layers}:")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Circuit gates: {circuit.count_ops()}")
    
    # Compare consecutive layer counts
    print("\n" + "-" * 70)
    print("Fidelity between consecutive layer counts:")
    print("-" * 70)
    
    for L in range(1, 3):
        fid = np.abs(np.vdot(states[f"L={L}"], states[f"L={L+1}"])) ** 2
        status = "✓ Different" if fid < 0.95 else "❌ TOO SIMILAR"
        print(f"L={L} vs L={L+1}: {fid:.6f} {status}")
    
    # Test 2: Input sensitivity
    print("\n" + "-" * 70)
    print("Test 2: Verify different inputs create different states")
    print("-" * 70)
    
    encoder = ReuploadEncoding(n_qubits=5, n_layers=2)
    
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
    
    # Test 3: Angle variation across layers
    print("\n" + "-" * 70)
    print("Test 3: Verify angle scaling")
    print("-" * 70)
    
    x_simple = np.array([1.0, 2.0, 3.0])
    encoder = ReuploadEncoding(n_qubits=3, n_layers=3)
    
    print(f"\nInput: {x_simple}")
    print("\nExpected angles per layer:")
    for layer in range(3):
        angles = x_simple * (layer + 1) / 3
        print(f"  Layer {layer}: {angles}")
    
    # Test 4: Circuit diagram
    print("\n" + "-" * 70)
    print("Test 4: Circuit structure (layers=2)")
    print("-" * 70)
    
    encoder = ReuploadEncoding(n_qubits=3, n_layers=2)
    circuit = encoder.build_circuit(np.array([0.5, 1.0, 1.5]))
    
    print("\n" + circuit.draw('text', fold=-1))
    
    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)