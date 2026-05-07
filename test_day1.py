import numpy as np
from src.utils.data import load_dataset
from src.encodings.angle import AngleEncoding
from src.experiments.kernel import compute_kernel_matrix
from src.experiments.evaluate import run_svm

# Load
X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

# Keep it small for speed
X_train, y_train = X_train[:40], y_train[:40]
X_test,  y_test  = X_test[:20],  y_test[:20]

# Encode
encoder = AngleEncoding(n_qubits=5)

# Kernels
K_train = compute_kernel_matrix(X_train, encoder)
K_test = compute_kernel_matrix(X_test, encoder, X_train)
# Sanity checks (worth it)
assert K_train.shape == (len(X_train), len(X_train))
assert K_test.shape  == (len(X_test),  len(X_train))
assert np.allclose(K_train, K_train.T, atol=1e-6)
assert np.allclose(np.diag(K_train), 1.0, atol=1e-6)

# Train + eval
acc = run_svm(K_train, y_train, K_test, y_test)
print("\n🔥 Final Accuracy:", acc)