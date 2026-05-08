"""
Quantum Kernel Computation Module
Supports both noiseless (statevector) and noisy (density matrix) kernel computation.

Compatible with Qiskit 1.0+
"""

import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit import transpile
from qiskit_aer import AerSimulator


def compute_quantum_kernel(X1, X2=None, encoder=None, verbose=False, 
                          use_noise=False, noise_backend='FakeSantiago',
                          shots=None):
    """
    Compute quantum kernel matrix using fidelity:
        K(x, x') = |<ψ(x') | ψ(x)>|^2  (noiseless)
        K(x, x') = F(ρ(x), ρ(x'))      (noisy, via density matrices)
    
    Parameters
    ----------
    X1 : array-like, shape (n_samples_1, n_features)
        First set of samples
    X2 : array-like, shape (n_samples_2, n_features), optional
        Second set of samples. If None, computes K(X1, X1)
    encoder : Encoder instance
        Must have build_circuit(x) method that returns a QuantumCircuit
    verbose : bool, default=False
        Print progress information
    use_noise : bool, default=False
        If True, use noisy simulation with density matrices
        If False, use exact statevector computation (default)
    noise_backend : str, default='FakeSantiago'
        Backend for noise model ('FakeSantiago' or 'FakeManila')
        Only used when use_noise=True
    shots : int, optional
        Number of shots for noisy simulation
        If None, uses density matrix method (exact under noise model)
        Only relevant when use_noise=True
        
    Returns
    -------
    K : ndarray, shape (n_samples_1, n_samples_2)
        Quantum kernel matrix with values in [0, 1]
        
    Notes
    -----
    Noiseless mode (use_noise=False):
        - Uses Statevector.from_instruction()
        - Computes exact fidelity |<ψ(x')|ψ(x)>|^2
        - Fast and exact
        
    Noisy mode (use_noise=True):
        - Uses AerSimulator with density_matrix method
        - Applies realistic noise model from fake backends
        - Transpiles circuits before execution
        - Computes fidelity between density matrices
        - More realistic but slower
        
    Examples
    --------
    Noiseless kernel:
    >>> K = compute_quantum_kernel(X_train, encoder=my_encoder)
    
    Noisy kernel with FakeSantiago:
    >>> K = compute_quantum_kernel(X_train, encoder=my_encoder, 
    ...                           use_noise=True, 
    ...                           noise_backend='FakeSantiago')
    
    Noisy kernel with custom backend:
    >>> K = compute_quantum_kernel(X_train, X_test, encoder=my_encoder,
    ...                           use_noise=True,
    ...                           noise_backend='FakeManila')
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
    
    n1 = len(X1)
    n2 = len(X2)
    
    if verbose:
        mode = "noisy (density matrix)" if use_noise else "noiseless (statevector)"
        print(f"[Quantum Kernel] Computing {n1} x {n2} matrix ({mode})...")
        if use_noise:
            print(f"[Quantum Kernel] Using noise model: {noise_backend}")
    
    if use_noise:
        K = _compute_noisy_kernel(X1, X2, encoder, is_symmetric, 
                                 noise_backend, shots, verbose)
    else:
        K = _compute_noiseless_kernel(X1, X2, encoder, is_symmetric, verbose)
    
    if verbose:
        print("✔ Kernel computed")
        print(f"  Kernel stats: min={K.min():.6f}, max={K.max():.6f}, mean={K.mean():.6f}")
    
    return K


def _compute_noiseless_kernel(X1, X2, encoder, is_symmetric, verbose):
    """
    Compute kernel using exact statevector simulation.
    
    This is the original implementation using Statevector.from_instruction().
    """
    n1 = len(X1)
    n2 = len(X2)
    
    if verbose:
        print("[Quantum Kernel] Precomputing statevectors...")
    
    # Precompute statevectors as numpy arrays
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
        print(f"[Quantum Kernel] Computing {n1} x {n2} fidelities...")
    
    # Kernel computation
    K = np.zeros((n1, n2))
    
    if is_symmetric:
        # Exploit symmetry for efficiency
        for i in range(n1):
            for j in range(i, n2):
                overlap = np.vdot(states1[i], states2[j])
                val = np.abs(overlap) ** 2
                val = min(val, 1.0)  # numerical safety
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


def _compute_noisy_kernel(X1, X2, encoder, is_symmetric, noise_backend, 
                         shots, verbose):
    """
    Compute kernel using noisy density matrix simulation.
    
    Uses AerSimulator with density_matrix method and realistic noise models.
    """
    n1 = len(X1)
    n2 = len(X2)
    
    # Import noise utility
    try:
        from qiskit_noise_utility import create_noisy_simulator
    except ImportError:
        raise ImportError(
            "qiskit_noise_utility not found. "
            "Make sure qiskit_noise_utility.py is in your Python path."
        )
    
    # Create noisy simulator with density matrix method
    if verbose:
        print(f"[Quantum Kernel] Creating noisy simulator ({noise_backend})...")
    
    simulator = create_noisy_simulator(noise_backend)
    # Override method to use density_matrix
    simulator.set_options(method='density_matrix')
    
    if shots is not None:
        simulator.set_options(shots=shots)
    
    if verbose:
        print("[Quantum Kernel] Building and transpiling circuits...")
    
    # Build circuits for all samples
    circuits1 = [encoder.build_circuit(x) for x in X1]
    if is_symmetric:
        circuits2 = circuits1
    else:
        circuits2 = [encoder.build_circuit(x) for x in X2]
    
    # Transpile all circuits
    transpiled1 = transpile(circuits1, simulator, optimization_level=1)
    if is_symmetric:
        transpiled2 = transpiled1
    else:
        transpiled2 = transpile(circuits2, simulator, optimization_level=1)
    
    if verbose:
        print("[Quantum Kernel] Simulating circuits to get density matrices...")
    
    # Run simulations to get density matrices
    density_matrices1 = []
    for i, circuit in enumerate(transpiled1):
        if verbose and (i + 1) % max(1, n1 // 10) == 0:
            print(f"  Progress: {i + 1}/{n1} circuits (set 1)")
        
        # Save density matrix instruction
        circuit_with_save = circuit.copy()
        circuit_with_save.save_density_matrix()
        
        # Run simulation
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
        print(f"[Quantum Kernel] Computing {n1} x {n2} fidelities from density matrices...")
    
    # Compute kernel matrix using density matrix fidelity
    K = np.zeros((n1, n2))
    
    if is_symmetric:
        # Exploit symmetry
        for i in range(n1):
            for j in range(i, n2):
                # Fidelity between density matrices
                fid = state_fidelity(density_matrices1[i], density_matrices2[j])
                fid = min(max(fid, 0.0), 1.0)  # numerical stability
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
    """
    Compute kernel alignment between two kernel matrices.
    
    Measures similarity between kernels using centered kernel alignment:
        CKA(K1, K2) = <K1_c, K2_c>_F / (||K1_c||_F * ||K2_c||_F)
    
    where K_c is the centered kernel matrix.
    
    Parameters
    ----------
    K1 : ndarray, shape (n, n)
        First kernel matrix
    K2 : ndarray, shape (n, n)
        Second kernel matrix
        
    Returns
    -------
    alignment : float
        Kernel alignment value in [0, 1]
        
    Examples
    --------
    >>> K_noiseless = compute_quantum_kernel(X, encoder=enc, use_noise=False)
    >>> K_noisy = compute_quantum_kernel(X, encoder=enc, use_noise=True)
    >>> alignment = compute_kernel_alignment(K_noiseless, K_noisy)
    >>> print(f"Alignment: {alignment:.4f}")
    """
    K1 = np.asarray(K1)
    K2 = np.asarray(K2)
    
    if K1.shape != K2.shape:
        raise ValueError(f"Kernel matrices must have same shape: {K1.shape} vs {K2.shape}")
    
    n = K1.shape[0]
    
    # Center the kernels
    H = np.eye(n) - np.ones((n, n)) / n
    K1_c = H @ K1 @ H
    K2_c = H @ K2 @ H
    
    # Compute alignment
    numerator = np.sum(K1_c * K2_c)
    denominator = np.sqrt(np.sum(K1_c * K1_c) * np.sum(K2_c * K2_c))
    
    if denominator == 0:
        return 0.0
    
    alignment = numerator / denominator
    
    # Ensure in [0, 1]
    alignment = min(max(alignment, 0.0), 1.0)
    
    return alignment


def validate_kernel_matrix(K, tol=1e-6, verbose=False):
    """
    Validate that a kernel matrix satisfies expected properties.
    
    Checks:
    1. All values in [0, 1]
    2. Diagonal elements close to 1
    3. Symmetry (if square)
    4. Positive semi-definite
    
    Parameters
    ----------
    K : ndarray, shape (n, m) or (n, n)
        Kernel matrix to validate
    tol : float, default=1e-6
        Tolerance for numerical checks
    verbose : bool, default=False
        Print validation results
        
    Returns
    -------
    is_valid : bool
        True if all checks pass
    issues : list of str
        List of validation issues found (empty if valid)
        
    Examples
    --------
    >>> K = compute_quantum_kernel(X, encoder=enc)
    >>> is_valid, issues = validate_kernel_matrix(K, verbose=True)
    >>> if not is_valid:
    ...     print("Issues:", issues)
    """
    K = np.asarray(K)
    issues = []
    
    # Check 1: Values in [0, 1]
    if K.min() < -tol or K.max() > 1.0 + tol:
        issues.append(f"Values outside [0,1]: min={K.min():.6f}, max={K.max():.6f}")
    
    # Check 2: Diagonal elements (if square)
    if K.shape[0] == K.shape[1]:
        diag = np.diag(K)
        if not np.allclose(diag, 1.0, atol=tol):
            diag_min, diag_max = diag.min(), diag.max()
            issues.append(f"Diagonal not all 1: min={diag_min:.6f}, max={diag_max:.6f}")
        
        # Check 3: Symmetry
        if not np.allclose(K, K.T, atol=tol):
            max_asym = np.abs(K - K.T).max()
            issues.append(f"Not symmetric: max asymmetry={max_asym:.6e}")
        
        # Check 4: Positive semi-definite
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


# Convenience function for common use case
def compare_noiseless_vs_noisy(X, encoder, noise_backend='FakeSantiago', verbose=True):
    """
    Compute and compare noiseless vs noisy kernels for the same data.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input samples
    encoder : Encoder instance
        Quantum feature encoder
    noise_backend : str, default='FakeSantiago'
        Backend for noise model
    verbose : bool, default=True
        Print comparison statistics
        
    Returns
    -------
    K_noiseless : ndarray, shape (n, n)
        Noiseless kernel matrix
    K_noisy : ndarray, shape (n, n)
        Noisy kernel matrix
    alignment : float
        Kernel alignment between the two
        
    Examples
    --------
    >>> K_ideal, K_real, align = compare_noiseless_vs_noisy(
    ...     X_train, encoder=my_encoder, noise_backend='FakeManila'
    ... )
    >>> print(f"Kernel alignment: {align:.4f}")
    """
    if verbose:
        print("=" * 60)
        print("Comparing Noiseless vs Noisy Quantum Kernels")
        print("=" * 60)
    
    # Compute noiseless kernel
    if verbose:
        print("\n[1] Computing noiseless kernel...")
    K_noiseless = compute_quantum_kernel(X, encoder=encoder, use_noise=False, verbose=verbose)
    
    # Compute noisy kernel
    if verbose:
        print(f"\n[2] Computing noisy kernel ({noise_backend})...")
    K_noisy = compute_quantum_kernel(X, encoder=encoder, use_noise=True, 
                                    noise_backend=noise_backend, verbose=verbose)
    
    # Compute alignment
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
        
        # Difference analysis
        diff = np.abs(K_noiseless - K_noisy)
        print("\nAbsolute differences:")
        print(f"  Mean: {diff.mean():.6f}")
        print(f"  Max:  {diff.max():.6f}")
        
        print("=" * 60)
    
    return K_noiseless, K_noisy, alignment