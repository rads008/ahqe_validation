import numpy as np
from qiskit.quantum_info import Statevector


def compute_quantum_kernel(X1, X2=None, encoder=None, verbose=False):
    """
    Compute quantum kernel matrix using fidelity:
        K(x, x') = |<ψ(x') | ψ(x)>|^2
    
    Parameters
    ----------
    X1 : array-like, shape (n_samples_1, n_features)
        First set of samples
    X2 : array-like, shape (n_samples_2, n_features), optional
        Second set of samples. If None, computes K(X1, X1)
    encoder : Encoder instance
        Must have build_circuit(x) method
    verbose : bool
        Print progress
        
    Returns
    -------
    K : ndarray, shape (n_samples_1, n_samples_2)
        Quantum kernel matrix
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
        print(f"[Quantum Kernel] Computing {n1} x {n2} matrix...")
    
    # Step 1: Precompute statevectors as numpy arrays
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
    
    # Step 2: Kernel computation
    K = np.zeros((n1, n2))
    
    if is_symmetric:
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
    
    if verbose:
        print("✔ Kernel computed")
    
    return K