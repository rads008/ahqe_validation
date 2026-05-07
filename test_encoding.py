def test_encoding_differentiation():
    """
    Test that different encodings produce DIFFERENT quantum states.
    """
    print("\n" + "=" * 70)
    print("ENCODING DIFFERENTIATION TEST")
    print("=" * 70)
    
    from qiskit.quantum_info import Statevector
    
    # Test input
    x_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    
    # Build encoders
    encoders = {
        "Uniform(d=1)": UniformEntangledEncoding(n_qubits=5, depth=1),
        "Uniform(d=2)": UniformEntangledEncoding(n_qubits=5, depth=2),
        "Uniform(d=3)": UniformEntangledEncoding(n_qubits=5, depth=3),
        "Reupload(L=1)": ReuploadEncoding(n_qubits=5, n_layers=1),
        "Reupload(L=2)": ReuploadEncoding(n_qubits=5, n_layers=2),
    }
    
    # Compute states
    states = {}
    for name, encoder in encoders.items():
        circuit = encoder.build_circuit(x_test)
        state = Statevector.from_instruction(circuit)
        states[name] = state.data
        print(f"\n{name}:")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  State norm: {np.linalg.norm(state.data):.6f}")
    
    # Compute pairwise fidelities
    print("\n" + "-" * 70)
    print("PAIRWISE STATE FIDELITIES")
    print("-" * 70)
    print("(Should be < 0.99 for different encodings)\n")
    
    names = list(states.keys())
    fidelity_matrix = np.zeros((len(names), len(names)))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i <= j:
                fid = np.abs(np.vdot(states[name1], states[name2])) ** 2
                fidelity_matrix[i, j] = fid
                fidelity_matrix[j, i] = fid
                
                if i != j:
                    status = "❌ TOO SIMILAR" if fid > 0.99 else "✓ Different"
                    print(f"{name1:20s} vs {name2:20s}: {fid:.6f} {status}")
    
    # Check depth scaling
    print("\n" + "-" * 70)
    print("DEPTH SCALING CHECK")
    print("-" * 70)
    
    uniform_depths = ["Uniform(d=1)", "Uniform(d=2)", "Uniform(d=3)"]
    for i in range(len(uniform_depths) - 1):
        fid = fidelity_matrix[names.index(uniform_depths[i]), 
                             names.index(uniform_depths[i+1])]
        print(f"{uniform_depths[i]} vs {uniform_depths[i+1]}: {fid:.6f}")
        if fid > 0.99:
            print(f"  ❌ PROBLEM: Increasing depth doesn't change state!")
        else:
            print(f"  ✓ Depth adds variation")
    
    return fidelity_matrix


# Run this in your main experiment
if __name__ == "__main__":
    test_encoding_differentiation()