import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from src.encodings.ahqe import AHQEEncoding
from src.allocation.importance import estimate_importance
from src.allocation.waterfill import waterfill_allocate
from src.experiments.kernel import compute_quantum_kernel


# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Scale to [0, π]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = (X - X.min()) / (X.max() - X.min()) * np.pi

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)

# 4. Feature importance
phi = estimate_importance(X_train, y_train, n_groups=5)

# 5. AHQE allocation
alpha = np.ones(5)
nu = np.ones(5)
kappa_ahqe = waterfill_allocate(phi, alpha, nu, E_max=13)

print("\n=== AHQE Allocation ===")
print("phi:", phi)
print("kappa:", kappa_ahqe)

# ------------------ AHQE ------------------
encoder_ahqe = AHQEEncoding(
    kappa=kappa_ahqe,
    features_per_group=[6,6,6,6,6],
    n_qubits=5
)

print("\n[AHQE] Computing kernel...")
K_train_ahqe = compute_quantum_kernel(X_train, encoder_ahqe)
K_test_ahqe  = compute_quantum_kernel(X_test, encoder_ahqe, X_train)

svm_ahqe = SVC(kernel='precomputed')
svm_ahqe.fit(K_train_ahqe, y_train)
acc_ahqe = svm_ahqe.score(K_test_ahqe, y_test)

# ------------------ UNIFORM ------------------
kappa_uniform = np.array([3,3,3,3,3])

encoder_uniform = AHQEEncoding(
    kappa=kappa_uniform,
    features_per_group=[6,6,6,6,6],
    n_qubits=5
)

print("\n[Uniform] Computing kernel...")
K_train_u = compute_quantum_kernel(X_train, encoder_uniform)
K_test_u  = compute_quantum_kernel(X_test, encoder_uniform, X_train)

svm_u = SVC(kernel='precomputed')
svm_u.fit(K_train_u, y_train)
acc_uniform = svm_u.score(K_test_u, y_test)

# ------------------ RESULT ------------------
print("\n=== FINAL COMPARISON ===")
print(f"AHQE Accuracy:    {acc_ahqe:.4f}")
print(f"Uniform Accuracy: {acc_uniform:.4f}")
print(f"Difference:       {acc_ahqe - acc_uniform:+.4f}")