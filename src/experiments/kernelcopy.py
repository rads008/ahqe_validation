"""
Quantum Kernel Computation Module – Optimized with Batching for Noisy Backends
Supports both noiseless (statevector) and noisy (density matrix) kernel computation.
Compatible with Qiskit 1.0+
"""

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit import transpile
from qiskit_aer import AerSimulator


def compute_quantum_kernel(X1, X2=None, encoder=None, verbose=False,
                          use_noise=False, noise_backend='FakeSantiago',
                          shots=None, backend=None):
    """
    Compute quantum kernel matrix using fidelity.

    Parameters
    ----------
    X1 : array-like, shape (n_samples_1, n_features)
    X2 : array-like, shape (n_samples_2, n_features), optional
    encoder : Encoder instance
    verbose : bool
    use_noise : bool (ignored if backend is provided)
    noise_backend : str (ignored if backend is provided)
    shots : int (ignored if backend is provided)
    backend : AerSimulator, optional
        If provided, use this backend (can be noisy). Overrides use_noise & noise_backend.

    Returns
    -------
    K : ndarray, shape (n_samples_1, n_samples_2)
    """
    if encoder is None:
        raise ValueError("encoder must be provided")

    X1 = np.asarray(X1)
    if X2 is None:
        X2 = X1
        is_symmetric = True
    else:
        X2 = np.asarray(X2)
        is_symmetric = False

    n1, n2 = len(X1), len(X2)

    if verbose:
        if backend is not None:
            print(f"[Quantum Kernel] Computing {n1}×{n2} matrix using provided backend...")
        else:
            mode = "noisy (density matrix)" if use_noise else "noiseless (statevector)"
            print(f"[Quantum Kernel] Computing {n1}×{n2} matrix ({mode})...")
            if use_noise:
                print(f"[Quantum Kernel] Using noise model: {noise_backend}")

    if backend is not None:
        K = _compute_kernel_with_backend_batched(X1, X2, encoder, is_symmetric, backend, verbose)
    elif use_noise:
        K = _compute_noisy_kernel(X1, X2, encoder, is_symmetric, noise_backend, shots, verbose)
    else:
        K = _compute_noiseless_kernel(X1, X2, encoder, is_symmetric, verbose)

    if verbose:
        print("✔ Kernel computed")
        print(f"  Kernel stats: min={K.min():.6f}, max={K.max():.6f}, mean={K.mean():.6f}")

    return K


def _compute_kernel_with_backend_batched(X1, X2, encoder, is_symmetric, backend, verbose):
    """
    Compute kernel using a provided backend with batched density-matrix simulation.
    """

    n1 = len(X1)
    n2 = len(X2)

    if verbose:
        print("[Kernel] Building circuits...")

    circuits1 = [encoder.build_circuit(x) for x in X1]

    if is_symmetric:
        circuits2 = circuits1
    else:
        circuits2 = [encoder.build_circuit(x) for x in X2]

    if verbose:
        print("[Kernel] Transpiling circuits...")

    transpiled1 = transpile(circuits1, backend, optimization_level=1)

    if is_symmetric:
        transpiled2 = transpiled1
    else:
        transpiled2 = transpile(circuits2, backend, optimization_level=1)

    #
    # IMPORTANT:
    # Never modify transpiled circuits in-place.
    #
    transpiled1 = [c.copy() for c in transpiled1]
    for circ in transpiled1:
        circ.save_density_matrix()

    if is_symmetric:
        transpiled2 = transpiled1
    else:
        transpiled2 = [c.copy() for c in transpiled2]
        for circ in transpiled2:
            circ.save_density_matrix()

    if verbose:
        print("[Kernel] Running batches...")

    batch_size = 50

    def run_batches(circuit_list, label):
        density_matrices = []

        for start in range(0, len(circuit_list), batch_size):

            batch = circuit_list[start:start + batch_size]

            #
            # shots are unnecessary for density-matrix simulation
            #
            job = backend.run(batch)

            result = job.result()

            for k in range(len(batch)):
                dm = result.data(k)["density_matrix"]

                # Diagnostic
                if verbose:
                    tr = np.trace(dm)
                    sf = state_fidelity(dm, dm)
                    print(
                        f"{label}[{start+k}] "
                        f"trace={tr:.6f} "
                        f"self_fidelity={sf:.6f}"
                    )

                density_matrices.append(dm)

        return density_matrices

    all_dms1 = run_batches(transpiled1, "set1")

    if is_symmetric:
        all_dms2 = all_dms1
    else:
        all_dms2 = run_batches(transpiled2, "set2")

    if verbose:
        print("[Kernel] Computing fidelities...")

    K = np.zeros((n1, n2))

    if is_symmetric:

        for i in range(n1):
            for j in range(i, n2):

                fid = state_fidelity(all_dms1[i], all_dms2[j])
                fid = float(np.clip(fid, 0.0, 1.0))

                K[i, j] = fid

                if i != j:
                    K[j, i] = fid

    else:

        for i in range(n1):
            for j in range(n2):

                fid = state_fidelity(all_dms1[i], all_dms2[j])
                K[i, j] = float(np.clip(fid, 0.0, 1.0))

    return K

def _compute_noiseless_kernel(X1, X2, encoder, is_symmetric, verbose):
    """Compute kernel using exact statevector simulation."""
    n1 = len(X1)
    n2 = len(X2)

    if verbose:
        print("[Quantum Kernel] Precomputing statevectors...")

    states1 = [
        Statevector.from_instruction(encoder.build_circuit(x)).data
        for x in X1
    ]

    if is_symmetric:
        states2 = states1
    else:
        states2 = [
            Statevector.from_instruction(encoder.build_circuit(x)).data
            for x in X2
        ]

    if verbose:
        print(f"[Quantum Kernel] Computing {n1}×{n2} fidelities...")

    K = np.zeros((n1, n2))

    if is_symmetric:
        for i in range(n1):
            for j in range(i, n2):
                overlap = np.vdot(states1[i], states2[j])
                val = np.abs(overlap) ** 2
                val = min(val, 1.0)
                K[i, j] = val
                if i != j:
                    K[j, i] = val
    else:
        for i in range(n1):
            for j in range(n2):
                overlap = np.vdot(states1[i], states2[j])
                val = np.abs(overlap) ** 2
                K[i, j] = min(val, 1.0)

    return K


def _compute_noisy_kernel(X1, X2, encoder, is_symmetric, noise_backend, shots, verbose):
    """Compute kernel using noisy density matrix simulation (original implementation)."""
    n1 = len(X1)
    n2 = len(X2)

    try:
        from qiskit_noise_utility import create_noisy_simulator
    except ImportError:
        raise ImportError(
            "qiskit_noise_utility not found. "
            "Make sure qiskit_noise_utility.py is in your Python path."
        )

    if verbose:
        print(f"[Quantum Kernel] Creating noisy simulator ({noise_backend})...")

    simulator = create_noisy_simulator(noise_backend)
    simulator.set_options(method='density_matrix')

    if shots is not None:
        simulator.set_options(shots=shots)

    if verbose:
        print("[Quantum Kernel] Building and transpiling circuits...")

    circuits1 = [encoder.build_circuit(x) for x in X1]
    if is_symmetric:
        circuits2 = circuits1
    else:
        circuits2 = [encoder.build_circuit(x) for x in X2]

    transpiled1 = transpile(circuits1, simulator, optimization_level=1)
    if is_symmetric:
        transpiled2 = transpiled1
    else:
        transpiled2 = transpile(circuits2, simulator, optimization_level=1)

    if verbose:
        print("[Quantum Kernel] Simulating circuits to get density matrices...")

    density_matrices1 = []
    for i, circuit in enumerate(transpiled1):
        if verbose and (i + 1) % max(1, n1 // 10) == 0:
            print(f"  Progress: {i + 1}/{n1} circuits (set 1)")
        circuit_with_save = circuit.copy()
        circuit_with_save.save_density_matrix()
        job = simulator.run(circuit_with_save, shots=1)
        result = job.result()
        density_matrix = result.data()['density_matrix']
        density_matrices1.append(density_matrix)

    if is_symmetric:
        density_matrices2 = density_matrices1
    else:
        density_matrices2 = []
        for i, circuit in enumerate(transpiled2):
            if verbose and (i + 1) % max(1, n2 // 10) == 0:
                print(f"  Progress: {i + 1}/{n2} circuits (set 2)")
            circuit_with_save = circuit.copy()
            circuit_with_save.save_density_matrix()
            job = simulator.run(circuit_with_save, shots=1)
            result = job.result()
            density_matrix = result.data()['density_matrix']
            density_matrices2.append(density_matrix)

    if verbose:
        print(f"[Quantum Kernel] Computing {n1}×{n2} fidelities...")

    K = np.zeros((n1, n2))

    if is_symmetric:
        for i in range(n1):
            for j in range(i, n2):
                fid = state_fidelity(density_matrices1[i], density_matrices2[j])
                fid = min(max(fid, 0.0), 1.0)
                K[i, j] = fid
                if i != j:
                    K[j, i] = fid
    else:
        for i in range(n1):
            for j in range(n2):
                fid = state_fidelity(density_matrices1[i], density_matrices2[j])
                K[i, j] = min(max(fid, 0.0), 1.0)

    return K


def compute_kernel_alignment(K1, K2):
    """Compute centered kernel alignment between two kernel matrices."""
    K1 = np.asarray(K1)
    K2 = np.asarray(K2)

    if K1.shape != K2.shape:
        raise ValueError(f"Kernel matrices must have same shape: {K1.shape} vs {K2.shape}")

    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1_c = H @ K1 @ H
    K2_c = H @ K2 @ H

    numerator = np.sum(K1_c * K2_c)
    denominator = np.sqrt(np.sum(K1_c * K1_c) * np.sum(K2_c * K2_c))

    if denominator == 0:
        return 0.0

    alignment = numerator / denominator
    alignment = min(max(alignment, 0.0), 1.0)
    return alignment


def validate_kernel_matrix(K, tol=1e-6, verbose=False):
    """Validate kernel matrix properties."""
    K = np.asarray(K)
    issues = []

    if K.min() < -tol or K.max() > 1.0 + tol:
        issues.append(f"Values outside [0,1]: min={K.min():.6f}, max={K.max():.6f}")

    if K.shape[0] == K.shape[1]:
        diag = np.diag(K)
        if not np.allclose(diag, 1.0, atol=tol):
            diag_min, diag_max = diag.min(), diag.max()
            issues.append(f"Diagonal not all 1: min={diag_min:.6f}, max={diag_max:.6f}")

        if not np.allclose(K, K.T, atol=tol):
            max_asym = np.abs(K - K.T).max()
            issues.append(f"Not symmetric: max asymmetry={max_asym:.6e}")

        try:
            eigvals = np.linalg.eigvalsh(K)
            if eigvals.min() < -tol:
                issues.append(f"Not PSD: min eigenvalue={eigvals.min():.6e}")
        except np.linalg.LinAlgError:
            issues.append("Could not compute eigenvalues")

    is_valid = len(issues) == 0

    if verbose:
        if is_valid:
            print("✔ Kernel matrix validation passed")
            print(f"  Shape: {K.shape}")
            print(f"  Range: [{K.min():.6f}, {K.max():.6f}]")
            if K.shape[0] == K.shape[1]:
                print(f"  Diagonal: [{np.diag(K).min():.6f}, {np.diag(K).max():.6f}]")
        else:
            print("✗ Kernel matrix validation failed")
            for issue in issues:
                print(f"  - {issue}")

    return is_valid, issues


def compare_noiseless_vs_noisy(X, encoder, noise_backend='FakeSantiago', verbose=True):
    """Compare noiseless vs noisy kernels."""
    if verbose:
        print("=" * 60)
        print("Comparing Noiseless vs Noisy Quantum Kernels")
        print("=" * 60)

    if verbose:
        print("\n[1] Computing noiseless kernel...")
    K_noiseless = compute_quantum_kernel(X, encoder=encoder, use_noise=False, verbose=verbose)

    if verbose:
        print(f"\n[2] Computing noisy kernel ({noise_backend})...")
    K_noisy = compute_quantum_kernel(X, encoder=encoder, use_noise=True,
                                    noise_backend=noise_backend, verbose=verbose)

    alignment = compute_kernel_alignment(K_noiseless, K_noisy)

    if verbose:
        print(f"\n[3] Kernel Alignment: {alignment:.6f}")
        print("\nNoiseless kernel stats:")
        print(f"  Mean: {K_noiseless.mean():.6f}")
        print(f"  Std:  {K_noiseless.std():.6f}")
        print(f"  Range: [{K_noiseless.min():.6f}, {K_noiseless.max():.6f}]")

        print("\nNoisy kernel stats:")
        print(f"  Mean: {K_noisy.mean():.6f}")
        print(f"  Std:  {K_noisy.std():.6f}")
        print(f"  Range: [{K_noisy.min():.6f}, {K_noisy.max():.6f}]")

        diff = np.abs(K_noiseless - K_noisy)
        print("\nAbsolute differences:")
        print(f"  Mean: {diff.mean():.6f}")
        print(f"  Max:  {diff.max():.6f}")
        print("=" * 60)

    return K_noiseless, K_noisy, alignment