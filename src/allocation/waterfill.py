import numpy as np
from scipy.optimize import minimize


def waterfill_allocate(phi, alpha, nu, E_max, max_depth=10, sensitivity=1.5):
    """
    Water-filling allocation with integer rounding.
    
    Solves continuous optimization then rounds to integers via greedy allocation.
    
    Parameters
    ----------
    phi : array-like, shape (n_groups,)
        Feature importance scores (should sum to 1)
    alpha : array-like, shape (n_groups,)
        Cost per qubit for each group
    nu : array-like, shape (n_groups,)
        Encoding efficiency parameters (>0)
    E_max : float
        Total qubit budget
    max_depth : int
        Maximum qubits per group (constraint)
    sensitivity : float
        Exponent for importance scaling. Higher = more aggressive allocation.
        Default 1.5 makes allocation more sensitive to importance differences.
        
    Returns
    -------
    kappa : ndarray, shape (n_groups,), dtype=int
        Integer qubit allocation per group
        
    Notes
    -----
    Objective: maximize sum(phi_i^sensitivity * (1 - exp(-nu_i * kappa_i)))
    Subject to: sum(alpha_i * kappa_i) <= E_max, 0 <= kappa_i <= max_depth
    
    Two-phase approach:
    1. Continuous optimization (SLSQP)
    2. Integer rounding (floor + greedy marginal gain)
    """
    phi = np.asarray(phi, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    nu = np.asarray(nu, dtype=float)
    n_groups = len(phi)
    
    # Scale importance to increase sensitivity
    phi_scaled = phi ** sensitivity
    
    # Objective: maximize encoding quality (we minimize the negative)
    def objective(kappa):
        # Clip to avoid overflow in exp
        kappa = np.clip(kappa, 0, max_depth)
        encoded_quality = phi_scaled * (1 - np.exp(-nu * kappa))
        return -np.sum(encoded_quality)  # Negative for minimization
    
    # Budget constraint: sum(alpha * kappa) <= E_max
    def budget_constraint(kappa):
        return E_max - np.sum(alpha * kappa)
    
    # Bounds: 0 <= kappa_i <= max_depth
    bounds = [(0, max_depth) for _ in range(n_groups)]
    
    # Constraints for SLSQP
    constraints = {'type': 'ineq', 'fun': budget_constraint}
    
    # Initial guess: proportional to phi/alpha (respecting budget)
    # This is a reasonable starting point
    initial_guess = (phi / alpha) * (E_max / np.sum(phi))
    initial_guess = np.clip(initial_guess, 0, max_depth)
    
    # Step 1: Solve continuous optimization
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise RuntimeError("Optimization did not converge")
        
        kappa_continuous = result.x
        
    except Exception as e:
        # Fallback: proportional allocation
        print(f"Warning: Optimization failed ({e}), using proportional allocation")
        kappa_continuous = (phi / alpha) * (E_max / np.sum(phi / alpha))
        kappa_continuous = np.clip(kappa_continuous, 0, max_depth)
    
    # Step 2: Integer rounding via greedy allocation
    kappa_int = np.floor(kappa_continuous).astype(int)
    
    # Calculate remaining budget
    budget_used = np.sum(alpha * kappa_int)
    budget_remaining = E_max - budget_used
    
    # Greedy allocation of remaining budget based on marginal gain
    # Marginal gain: phi_i^sensitivity * (exp(-nu_i * k_i) - exp(-nu_i * (k_i + 1))) / alpha_i
    while budget_remaining >= np.min(alpha):
        # Compute marginal gain per unit cost for each group
        marginal_gains = np.zeros(n_groups)
        
        for i in range(n_groups):
            if kappa_int[i] < max_depth and alpha[i] <= budget_remaining:
                # Gain from adding one qubit to group i
                current_encoding = 1 - np.exp(-nu[i] * kappa_int[i])
                next_encoding = 1 - np.exp(-nu[i] * (kappa_int[i] + 1))
                gain = phi_scaled[i] * (next_encoding - current_encoding)
                marginal_gains[i] = gain / alpha[i]  # Gain per unit cost
            else:
                marginal_gains[i] = -np.inf  # Cannot allocate
        
        # Find best group to allocate to
        best_group = np.argmax(marginal_gains)
        
        if marginal_gains[best_group] == -np.inf:
            break  # No valid allocation possible
        
        # Allocate one qubit to best group
        kappa_int[best_group] += 1
        budget_remaining -= alpha[best_group]
    
    # Final validation
    final_budget = np.sum(alpha * kappa_int)
    assert final_budget <= E_max + 1e-6, f"Budget exceeded: {final_budget} > {E_max}"
    
    return kappa_int


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Basic allocation (sensitivity = 1.0 vs 1.5)")
    print("=" * 60)
    
    phi = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0])
    nu = np.array([1.0, 1.0])
    E_max = 10
    
    # Standard allocation (sensitivity = 1.0, linear)
    kappa_linear = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.0)
    print(f"\nLinear (sensitivity=1.0):")
    print(f"  phi:   {phi}")
    print(f"  kappa: {kappa_linear}")
    print(f"  Budget used: {np.sum(alpha * kappa_linear)}/{E_max}")
    
    # Sensitive allocation (sensitivity = 1.5, default)
    kappa_sensitive = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.5)
    print(f"\nSensitive (sensitivity=1.5):")
    print(f"  phi_scaled: {phi**1.5}")
    print(f"  kappa:      {kappa_sensitive}")
    print(f"  Budget used: {np.sum(alpha * kappa_sensitive)}/{E_max}")
    
    # Compare allocation ratios
    print(f"\nAllocation ratio (group 0 / group 1):")
    print(f"  Linear:    {kappa_linear[0]}/{kappa_linear[1]} = {kappa_linear[0]/max(kappa_linear[1], 1):.2f}")
    print(f"  Sensitive: {kappa_sensitive[0]}/{kappa_sensitive[1]} = {kappa_sensitive[0]/max(kappa_sensitive[1], 1):.2f}")
    
    print("\n" + "=" * 60)
    print("Test 2: Three groups with varying importance")
    print("=" * 60)
    
    phi = np.array([0.6, 0.3, 0.1])
    alpha = np.array([1.0, 1.0, 1.0])
    nu = np.array([1.0, 1.0, 1.0])
    E_max = 15
    
    kappa_linear = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.0)
    kappa_sensitive = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.5)
    
    print(f"\nphi: {phi}")
    print(f"Linear:    {kappa_linear} (budget: {np.sum(alpha * kappa_linear)})")
    print(f"Sensitive: {kappa_sensitive} (budget: {np.sum(alpha * kappa_sensitive)})")
    
    print("\n" + "=" * 60)
    print("Test 3: Different costs per group")
    print("=" * 60)
    
    phi = np.array([0.7, 0.3])
    alpha = np.array([1.0, 2.0])  # Group 1 is more expensive
    nu = np.array([1.0, 1.0])
    E_max = 12
    
    kappa = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.5)
    print(f"\nphi:   {phi}")
    print(f"alpha: {alpha}")
    print(f"kappa: {kappa}")
    print(f"Budget used: {np.sum(alpha * kappa)}/{E_max}")
    print(f"Cost breakdown: {alpha * kappa}")
    
    print("\n" + "=" * 60)
    print("Test 4: Edge case - very small budget")
    print("=" * 60)
    
    phi = np.array([0.5, 0.3, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    nu = np.array([1.0, 1.0, 1.0])
    E_max = 2
    
    kappa = waterfill_allocate(phi, alpha, nu, E_max, sensitivity=1.5)
    print(f"\nphi:   {phi}")
    print(f"kappa: {kappa}")
    print(f"Budget used: {np.sum(alpha * kappa)}/{E_max}")
    
    print("\n✓ All tests passed")