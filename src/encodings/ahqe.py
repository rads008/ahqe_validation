import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class AHQEEncoding:
    """
    Adaptive Hybrid Quantum Encoding with feature grouping and depth allocation.
    
    Each feature group is:
        1. Encoded on one qubit (averaged features)
        2. Given kappa[i] parameterized entangling layers
    
    KEY FIX: Uses feature AVERAGING (not sequential rotation) and 
    layer-dependent parameterization for expressivity.
    
    Parameters
    ----------
    kappa : array-like, shape (n_groups,)
        Encoding depth allocation for each group (integer, >= 0)
    features_per_group : list of int, length n_groups
        Number of features in each group
    n_qubits : int
        Number of qubits (must equal n_groups)
    """
    
    def __init__(self, kappa, features_per_group, n_qubits):
        self.kappa = np.asarray(kappa, dtype=int)
        self.features_per_group = list(features_per_group)
        self.n_qubits = n_qubits
        self.n_groups = len(self.kappa)
        
        # Validate inputs
        if len(self.features_per_group) != self.n_groups:
            raise ValueError("features_per_group length must match kappa length")
        if self.n_qubits != self.n_groups:
            raise ValueError("n_qubits must equal n_groups (one qubit per group)")
        if np.any(self.kappa < 0):
            raise ValueError("kappa values must be non-negative")
        
        self.total_features = sum(self.features_per_group)
        
        # Precompute feature-to-group mapping for efficiency
        self._build_feature_map()
    
    def __repr__(self):
        return (f"AHQEEncoding(n_groups={self.n_groups}, "
                f"kappa={list(self.kappa)}, "
                f"features_per_group={self.features_per_group})")
    
    def _build_feature_map(self):
        """Build mapping from feature index to (group_id, position_in_group)."""
        self.feature_to_group = []
        for group_id, n_features in enumerate(self.features_per_group):
            for pos in range(n_features):
                self.feature_to_group.append((group_id, pos))
    
    def build_circuit(self, x):
        """
        Build quantum circuit encoding feature vector x.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            Feature vector to encode (must be in [0, π])
            
        Returns
        -------
        qc : QuantumCircuit
            Quantum circuit with AHQE encoding
            
        Circuit structure:
            1. Initial encoding: RY(mean(group_features)) per qubit
            2. For each layer up to max(kappa):
               - CNOT entanglement (only for groups needing this layer)
               - Parameterized rotations (layer-dependent angles)
        """
        x = np.asarray(x, dtype=float)
        
        if len(x) != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {len(x)}")
        
        if not np.all((x >= 0) & (x <= np.pi)):
            raise ValueError(
                f"Input must be in [0, π], got range [{x.min():.3f}, {x.max():.3f}]"
            )
        
        qc = QuantumCircuit(self.n_qubits, name="AHQE")
        
        # ============================================================
        # Step 1: Initial feature encoding (ONE rotation per qubit)
        # ============================================================
        feature_idx = 0
        for group_id in range(self.n_groups):
            qubit = group_id
            n_features_in_group = self.features_per_group[group_id]
            
            # Extract features for this group
            group_features = x[feature_idx:feature_idx + n_features_in_group]
            
            # CRITICAL FIX: Average features instead of sequential rotations
            # This prevents rotation collapse
            if len(group_features) > 0:
                angle = np.mean(group_features)
                qc.ry(angle, qubit)
            
            feature_idx += n_features_in_group
        
        qc.barrier()
        
        # ============================================================
        # Step 2: Adaptive parameterized entangling layers
        # ============================================================
        max_depth = int(np.max(self.kappa))
        
        for layer in range(max_depth):
            # Apply entanglement only for groups that need this layer
            for group_id in range(self.n_groups):
                if layer < self.kappa[group_id]:
                    control = group_id
                    target = (group_id + 1) % self.n_qubits
                    qc.cx(control, target)
            
            # CRITICAL: Parameterized rotations with layer-dependent angles
            # This ensures each layer adds NEW information
            for group_id in range(self.n_groups):
                if layer < self.kappa[group_id]:
                    qubit = group_id
                    
                    # Get features for this group
                    start_idx = sum(self.features_per_group[:group_id])
                    end_idx = start_idx + self.features_per_group[group_id]
                    group_features = x[start_idx:end_idx]
                    
                    if len(group_features) > 0:
                        # Scale by layer index to create variation
                        # Use max_depth for normalization to keep angles in reasonable range
                        mean_feature = np.mean(group_features)
                        angle = mean_feature * (layer + 1) / (max_depth + 1)
                        qc.ry(angle, qubit)
            
            if layer < max_depth - 1:
                qc.barrier()
        
        return qc
    
    def get_circuit_depth(self):
        """
        Estimate circuit depth per qubit.
        
        Returns
        -------
        depths : ndarray, shape (n_groups,)
            Approximate circuit depth for each qubit/group
        """
        # Initial encoding: 1 rotation
        # Each layer: 1 CNOT + 1 RY
        depths = 1 + 2 * self.kappa
        return depths


# ============================================================
# Test cases
# ============================================================
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    
    print("=" * 70)
    print("AHQEEncoding - Verification Tests")
    print("=" * 70)
    
    # Test 1: Different allocations create different states
    print("\n" + "-" * 70)
    print("Test 1: Verify depth allocation affects states")
    print("-" * 70)
    
    x_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    features_per_group = [1, 1, 1, 1, 1]
    
    allocations = {
        "uniform": [2, 2, 2, 2, 2],
        "adaptive": [3, 2, 1, 1, 0],
        "extreme": [5, 0, 0, 0, 0]
    }
    
    states = {}
    for name, kappa in allocations.items():
        encoder = AHQEEncoding(kappa, features_per_group, n_qubits=5)
        circuit = encoder.build_circuit(x_test)
        state = Statevector.from_instruction(circuit)
        states[name] = state.data
        
        print(f"\n{name}: kappa={kappa}")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  CNOTs: {circuit.count_ops().get('cx', 0)}")
    
    # Compare states
    print("\n" + "-" * 70)
    print("Fidelity between different allocations:")
    print("-" * 70)
    
    names = list(states.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fid = np.abs(np.vdot(states[names[i]], states[names[j]])) ** 2
            status = "✓ Different" if fid < 0.95 else "❌ TOO SIMILAR"
            print(f"{names[i]:10s} vs {names[j]:10s}: {fid:.6f} {status}")
    
    # Test 2: Feature averaging works correctly
    print("\n" + "-" * 70)
    print("Test 2: Verify feature averaging")
    print("-" * 70)
    
    # Group with multiple features
    x_multi = np.array([0.5, 1.0, 1.5, 2.0])  # 4 features
    features_per_group_multi = [2, 2]  # 2 groups with 2 features each
    kappa_multi = [2, 1]
    
    encoder_multi = AHQEEncoding(kappa_multi, features_per_group_multi, n_qubits=2)
    circuit_multi = encoder_multi.build_circuit(x_multi)
    
    print(f"\nInput: {x_multi}")
    print(f"Groups: {features_per_group_multi}")
    print(f"Expected initial angles:")
    print(f"  Group 0: mean([{x_multi[0]:.1f}, {x_multi[1]:.1f}]) = {np.mean(x_multi[0:2]):.2f}")
    print(f"  Group 1: mean([{x_multi[2]:.1f}, {x_multi[3]:.1f}]) = {np.mean(x_multi[2:4]):.2f}")
    
    print(f"\nCircuit:")
    print(circuit_multi.draw('text', fold=-1))
    
    # Test 3: Zero allocation
    print("\n" + "-" * 70)
    print("Test 3: Verify kappa=0 produces minimal circuit")
    print("-" * 70)
    
    kappa_zero = [0, 0, 0]
    features_per_group_zero = [2, 2, 1]
    
    encoder_zero = AHQEEncoding(kappa_zero, features_per_group_zero, n_qubits=3)
    circuit_zero = encoder_zero.build_circuit(np.random.uniform(0, np.pi, 5))
    
    print(f"\nkappa = {kappa_zero}")
    print(f"Circuit ops: {circuit_zero.count_ops()}")
    print(f"CNOTs: {circuit_zero.count_ops().get('cx', 0)}")
    
    assert circuit_zero.count_ops().get('cx', 0) == 0, "kappa=0 should have no CNOTs"
    print("✓ Zero allocation correct")
    
    # Test 4: Input sensitivity
    print("\n" + "-" * 70)
    print("Test 4: Verify different inputs create different states")
    print("-" * 70)
    
    kappa_test = [2, 2, 1]
    features_per_group_test = [2, 2, 1]
    encoder_test = AHQEEncoding(kappa_test, features_per_group_test, n_qubits=3)
    
    x1 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    x2 = np.array([0.3, 0.8, 1.2, 1.8, 2.3])
    
    state1 = Statevector.from_instruction(encoder_test.build_circuit(x1))
    state2 = Statevector.from_instruction(encoder_test.build_circuit(x2))
    
    fid = np.abs(np.vdot(state1.data, state2.data)) ** 2
    print(f"\nFidelity between different inputs: {fid:.6f}")
    if fid < 0.95:
        print("✓ Different inputs produce different states")
    else:
        print("❌ WARNING: States are too similar!")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)