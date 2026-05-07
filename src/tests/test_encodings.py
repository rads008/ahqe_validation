import numpy as np
from qiskit.quantum_info import Operator
from src.encodings import UniformEntangledEncoding, ReuploadEncoding

rng = np.random.RandomState(42)
x = rng.uniform(0, np.pi, size=4)

enc1 = UniformEntangledEncoding(4, depth=2)
qc1 = enc1.build_circuit(x)

enc2 = ReuploadEncoding(4, n_layers=2)
qc2 = enc2.build_circuit(x)

print("Uniform depth:", qc1.depth())
print("Reupload depth:", qc2.depth())

assert Operator(qc1).is_unitary()
assert Operator(qc2).is_unitary()

print("✓ Basic tests passed")
