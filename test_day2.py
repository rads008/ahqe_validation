import numpy as np
from src.allocation.importance import estimate_importance
from src.allocation.waterfill import waterfill_allocate

# Dummy data
X = np.random.rand(100, 30)
y = np.random.randint(0, 2, 100)

phi = estimate_importance(X, y, n_groups=5)

alpha = np.ones(5)
nu = np.ones(5)

kappa = waterfill_allocate(phi, alpha, nu, E_max=15)

print("phi:", phi)
print("kappa:", kappa)
print("budget used:", np.sum(kappa))



print("\n--- Extreme Importance Test ---")

phi_extreme = np.array([0.9, 0.05, 0.03, 0.01, 0.01])
alpha = np.ones(5)
nu = np.ones(5)

kappa_extreme = waterfill_allocate(phi_extreme, alpha, nu, E_max=15)

print("phi_extreme:", phi_extreme)
print("kappa_extreme:", kappa_extreme)
print("budget used:", np.sum(kappa_extreme))