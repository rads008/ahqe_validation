import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

from src.encodings.uniform import UniformEntangledEncoding
from src.experiments.kernel import compare_noiseless_vs_noisy


# ============================================================
# Load small dataset
# ============================================================

X, y = load_breast_cancer(return_X_y=True)

# Use only first 5 samples and first 5 features
X = X[:5, :5]

# Scale features
X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)

print("=" * 60)
print("Tiny Noisy Kernel Test")
print("=" * 60)

print(f"Dataset shape: {X.shape}")

# ============================================================
# Create encoder
# ============================================================

encoder = UniformEntangledEncoding(
    n_qubits=5,
    depth=1
)

print("\nEncoder created:")
print(encoder)

# ============================================================
# Compare kernels
# ============================================================

K_ideal, K_noisy, alignment = compare_noiseless_vs_noisy(
    X,
    encoder=encoder,
    noise_backend='FakeSantiago',
    verbose=True
)

# ============================================================
# Print results
# ============================================================

print("\nNoiseless Kernel:")
print(K_ideal)

print("\nNoisy Kernel:")
print(K_noisy)

print(f"\nKernel Alignment: {alignment:.6f}")

print("\nDiagonal (Noisy):")
print(np.diag(K_noisy))

print("\nDone.")
