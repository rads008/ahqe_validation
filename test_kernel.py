from src.utils.data import load_dataset
from src.encodings.angle import AngleEncoding
from src.experiments.kernel import compute_kernel_matrix

X_train, _, y_train, _ = load_dataset("breast_cancer")

# Keep it small (IMPORTANT)
X_train = X_train[:20]

encoder = AngleEncoding(n_qubits=5)

K = compute_kernel_matrix(X_train, encoder)

print("Kernel shape:", K.shape)
print("Diagonal:", K.diagonal())