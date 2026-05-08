import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# Import kernel function and encoder (adjust paths if needed)
from src.experiments.kernel import compute_quantum_kernel
from src.encodings.uniform import UniformEntangledEncoding

# 5 samples, 5 features, scale to (0, π - 1e-5] to pass strict validation
X, _ = make_classification(n_samples=5, n_features=5, random_state=42)
scaler = MinMaxScaler(feature_range=(0, np.pi - 1e-5))
X_scaled = scaler.fit_transform(X)

print("Scaled data range:", X_scaled.min(), X_scaled.max())

encoder = UniformEntangledEncoding(n_qubits=5, depth=1)

print("\n[1] Noiseless kernel...")
K_ideal = compute_quantum_kernel(X_scaled, encoder=encoder, use_noise=False, verbose=True)

print("\n[2] Noisy kernel (shots=1024, FakeSantiago)...")
K_noisy = compute_quantum_kernel(X_scaled, encoder=encoder, use_noise=True,
                                 noise_backend='FakeSantiago', shots=1024, verbose=True)

print("\nIdeal kernel mean:", K_ideal.mean())
print("Noisy kernel mean:", K_noisy.mean())