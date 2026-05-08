"""
Qiskit Aer Noise Utility for AHQE Experiments
Compatible with Qiskit 1.0+

This module provides noise models based on real IBM quantum backends
(FakeVigo or FakeManila) for use with AerSimulator.
"""

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeSantiagoV2
from qiskit_aer import AerSimulator


def get_noise_model(backend_name='FakeSantiago'):
    """
    Create a noise model from a fake backend for AHQE experiments.
    
    Parameters
    ----------
    backend_name : str, optional
        Name of the fake backend to use ('FakeSantiago' or 'FakeManila')
        Default: 'FakeSantiago' (5 qubits, similar to deprecated FakeVigo)
    
    Returns
    -------
    tuple
        A tuple containing:
        - noise_model: NoiseModel object for AerSimulator
        - coupling_map: CouplingMap defining qubit connectivity
        - basis_gates: list of basis gate names
    
    Raises
    ------
    ValueError
        If backend_name is not 'FakeSantiago' or 'FakeManila'
    
    Examples
    --------
    >>> noise_model, coupling_map, basis_gates = get_noise_model('FakeSantiago')
    >>> simulator = AerSimulator(noise_model=noise_model,
    ...                          coupling_map=coupling_map,
    ...                          basis_gates=basis_gates)
    """
    # Select the fake backend (V2 versions for Qiskit 1.0+)
    if backend_name == 'FakeSantiago':
        fake_backend = FakeSantiagoV2()
    elif backend_name == 'FakeManila':
        fake_backend = FakeManilaV2()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. "
                        "Use 'FakeSantiago' or 'FakeManila'")
    
    # Extract noise model from the fake backend
    noise_model = NoiseModel.from_backend(fake_backend)
    
    # Get coupling map and basis gates
    coupling_map = fake_backend.coupling_map
    basis_gates = noise_model.basis_gates
    
    return noise_model, coupling_map, basis_gates


def create_noisy_simulator(backend_name='FakeSantiago'):
    """
    Create a pre-configured AerSimulator with realistic IBM noise.
    Uses density matrix simulation for proper noisy mixed-state evolution.
    """

    noise_model, coupling_map, basis_gates = get_noise_model(backend_name)

    # Keep only physical gate operations compatible with density_matrix
    supported_basis_gates = [
        gate for gate in basis_gates
        if gate in ['cx', 'id', 'rz', 'sx', 'x']
    ]

    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=supported_basis_gates
    )

    return simulator

# Test block
if __name__ == "__main__":
    print("=" * 60)
    print("Qiskit Aer Noise Utility - Test Block")
    print("=" * 60)
    
    # Test with FakeSantiago
    print("\n[1] Testing with FakeSantiago backend:")
    print("-" * 60)
    noise_model_santiago, coupling_map_santiago, basis_gates_santiago = get_noise_model('FakeSantiago')
    
    print(f"✓ Noise model created: {noise_model_santiago}")
    print(f"✓ Number of qubits: {coupling_map_santiago.size()}")
    print(f"✓ Coupling map: {coupling_map_santiago}")
    print(f"✓ Basis gates: {basis_gates_santiago}")
    print(f"✓ Number of noise instructions: {len(noise_model_santiago.noise_instructions)}")
    
    # Test with FakeManila
    print("\n[2] Testing with FakeManila backend:")
    print("-" * 60)
    noise_model_manila, coupling_map_manila, basis_gates_manila = get_noise_model('FakeManila')
    
    print(f"✓ Noise model created: {noise_model_manila}")
    print(f"✓ Number of qubits: {coupling_map_manila.size()}")
    print(f"✓ Coupling map: {coupling_map_manila}")
    print(f"✓ Basis gates: {basis_gates_manila}")
    print(f"✓ Number of noise instructions: {len(noise_model_manila.noise_instructions)}")
    
    # Test AerSimulator creation
    print("\n[3] Testing AerSimulator creation:")
    print("-" * 60)
    simulator_santiago = create_noisy_simulator('FakeSantiago')
    print(f"✓ AerSimulator created with FakeSantiago noise")
    print(f"✓ Simulator options: {simulator_santiago.options}")
    
    simulator_manila = create_noisy_simulator('FakeManila')
    print(f"✓ AerSimulator created with FakeManila noise")
    
    # Quick circuit test
    print("\n[4] Running a simple circuit test:")
    print("-" * 60)
    from qiskit import QuantumCircuit, transpile
    
    # Create a simple Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    # Transpile for the noisy backend
    transpiled_qc = transpile(qc, simulator_santiago)
    
    # Run simulation
    job = simulator_santiago.run(transpiled_qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print(f"✓ Circuit executed successfully")
    print(f"✓ Measurement results: {counts}")
    print(f"✓ Total shots: {sum(counts.values())}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nUsage example:")
    print("  from qiskit_noise_utility import get_noise_model")
    print("  noise_model, coupling_map, basis_gates = get_noise_model('FakeSantiago')")
    print("  # Use with your AHQE experiments")