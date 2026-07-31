import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

from src.encodings import UniformEntangledEncoding, ReuploadEncoding, AHQEEncoding
from src.experiments.kernelcopy import compute_quantum_kernel
from src.allocation.importance import estimate_importance
from src.allocation.waterfill import waterfill_allocate


def count_cnots(qc):
    """Count number of CNOT (CX) gates in a quantum circuit."""
    return qc.count_ops().get('cx', 0)


def load_and_preprocess(seed=42):
    """Load and preprocess with MORE samples and HARDER task."""
    from sklearn.datasets import make_classification

    # Generate harder synthetic data
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=0.8,
        random_state=seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    X_min = X_train.min()
    X_max = X_train.max()
    X_train = (X_train - X_min) / (X_max - X_min) * np.pi
    X_test = (X_test - X_min) / (X_max - X_min) * np.pi
    X_train = np.clip(X_train, 0, np.pi)
    X_test = np.clip(X_test, 0, np.pi)
    return X_train, X_test, y_train, y_test


def evaluate_encoder(name, encoder, X_train, X_test, y_train, y_test,
                     param_value=None, diagnose=False, backend=None):
    """Evaluate encoder performance and circuit properties."""
    sample_circuit = encoder.build_circuit(X_train[0])
    depth = sample_circuit.depth()
    cnots = count_cnots(sample_circuit)

    if diagnose:
        print(f"\n[CIRCUIT CHECK: {name}]")
        circuit_0 = encoder.build_circuit(X_train[0])
        circuit_1 = encoder.build_circuit(X_train[1])
        from qiskit.quantum_info import Statevector
        state_0 = Statevector.from_instruction(circuit_0)
        state_1 = Statevector.from_instruction(circuit_1)
        fidelity = np.abs(np.vdot(state_0.data, state_1.data)) ** 2
        print(f"  Fidelity between sample 0 and 1: {fidelity:.6f}")
        if fidelity > 0.99:
            print("  ⚠️  WARNING: States are nearly identical!")
        else:
            print("  ✓ States are different")

    if backend is None:
        backend = AerSimulator()

    K_train = compute_quantum_kernel(X_train, encoder=encoder, backend=backend)
    if diagnose:
        diagnose_kernel(K_train, f"{name} (train)")

    model = SVC(kernel="precomputed", random_state=42)
    model.fit(K_train, y_train)

    K_test = compute_quantum_kernel(X_test, X2=X_train, encoder=encoder, backend=backend)
    if diagnose:
        diagnose_kernel(K_test, f"{name} (test)")

    acc = model.score(K_test, y_test)

    return {
        'name': name,
        'param': param_value,
        'accuracy': acc,
        'depth': depth,
        'cnots': cnots,
        'K_train': K_train if diagnose else None,
        'K_test': K_test if diagnose else None
    }


def run_experiment(seed=42, backend=None):
    X_train, X_test, y_train, y_test = load_and_preprocess(seed)

    print("=" * 70)
    print("AHQE Empirical Validation - Day 4")
    print("=" * 70)
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Feature range: [0, π]")

    # Diagnostic
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Kernel Variation Test")
    print("=" * 70)

    X_diag = X_train[:10]
    y_diag = y_train[:10]
    encoders_diag = {
        "Uniform(d=1)": UniformEntangledEncoding(n_qubits=8, depth=1),
        "Uniform(d=2)": UniformEntangledEncoding(n_qubits=8, depth=2),
        "Reupload(L=1)": ReuploadEncoding(n_qubits=8, n_layers=1),
    }
    kernels_diag = {}
    for enc_name, encoder in encoders_diag.items():
        print(f"\n--- {enc_name} ---")
        metrics = evaluate_encoder(
            enc_name, encoder, X_diag, X_diag, y_diag, y_diag,
            diagnose=True, backend=backend
        )
        kernels_diag[enc_name] = metrics['K_train']

    print("\n" + "=" * 70)
    print("PAIRWISE KERNEL COMPARISON")
    print("=" * 70)
    enc_names = list(kernels_diag.keys())
    for i in range(len(enc_names)):
        for j in range(i + 1, len(enc_names)):
            compare_kernels(
                kernels_diag[enc_names[i]],
                kernels_diag[enc_names[j]],
                enc_names[i],
                enc_names[j]
            )

    print("\n" + "=" * 70)
    print("END DIAGNOSTIC - Starting main experiment...")
    print("=" * 70)

    # Reload data
    X_train, X_test, y_train, y_test = load_and_preprocess(seed)

    print("=" * 70)
    print("AHQE Empirical Validation - Day 4")
    print("=" * 70)
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    results = []

    # BASELINE 1: Uniform
    print("\n" + "-" * 70)
    print("BASELINE 1: UniformEntangledEncoding - Depth Sweep")
    print("-" * 70)

    for depth in [1, 2, 3]:
        encoder = UniformEntangledEncoding(n_qubits=8, depth=depth)
        print(f"\nRunning Uniform (depth={depth})...")
        metrics = evaluate_encoder(
            "Uniform", encoder, X_train, X_test, y_train, y_test, depth,
            backend=backend
        )
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | Depth: {metrics['depth']} | CNOTs: {metrics['cnots']}")

    # BASELINE 2: Reupload
    print("\n" + "-" * 70)
    print("BASELINE 2: ReuploadEncoding - Layer Sweep")
    print("-" * 70)

    for layers in [1, 2, 3]:
        encoder = ReuploadEncoding(n_qubits=8, n_layers=layers)
        print(f"\nRunning Reupload (layers={layers})...")
        metrics = evaluate_encoder(
            "Reupload", encoder, X_train, X_test, y_train, y_test, layers,
            backend=backend
        )
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | Depth: {metrics['depth']} | CNOTs: {metrics['cnots']}")

    # AHQE
    print("\n" + "-" * 70)
    print("AHQE: Adaptive Hybrid Quantum Encoding")
    print("-" * 70)

    n_groups = 8
    phi, features_per_group = estimate_importance(X_train, y_train, n_groups)
    print(f"\nFeature importance (phi): {phi}")
    print(f"Features per group: {features_per_group}")

    alpha = np.ones(n_groups)
    nu = np.ones(n_groups)
    E_max = 10
    kappa = waterfill_allocate(phi=phi, alpha=alpha, nu=nu, E_max=E_max, sensitivity=1.5)
    print(f"\nDepth allocation (kappa): {kappa}")
    print(f"Budget used: {np.sum(alpha * kappa)}/{E_max}")

    encoder_ahqe = AHQEEncoding(
        kappa=kappa,
        features_per_group=features_per_group,
        n_qubits=n_groups
    )

    print(f"\nRunning AHQE...")
    metrics_ahqe = evaluate_encoder(
        "AHQE", encoder_ahqe, X_train, X_test, y_train, y_test,
        param_value="adaptive", backend=backend
    )
    results.append(metrics_ahqe)
    print(f"  Accuracy: {metrics_ahqe['accuracy']:.4f} | Depth: {metrics_ahqe['depth']} | CNOTs: {metrics_ahqe['cnots']}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Encoding':<12} | {'Param':<10} | {'Accuracy':<10} | {'Depth':<6} | {'CNOTs':<6}")
    print("-" * 70)
    for metrics in results:
        if metrics['name'] == 'Uniform':
            param_label = f"d={metrics['param']}"
        elif metrics['name'] == 'Reupload':
            param_label = f"L={metrics['param']}"
        else:
            param_label = str(metrics['param'])
        print(
            f"{metrics['name']:<12} | {param_label:<10} | "
            f"{metrics['accuracy']:<10.4f} | {metrics['depth']:<6} | {metrics['cnots']:<6}"
        )

    # Best configs
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    uniform_results = [r for r in results if r['name'] == 'Uniform']
    reupload_results = [r for r in results if r['name'] == 'Reupload']
    ahqe_results = [r for r in results if r['name'] == 'AHQE']

    best_uniform = max(uniform_results, key=lambda x: x['accuracy'])
    best_reupload = max(reupload_results, key=lambda x: x['accuracy'])
    best_ahqe = ahqe_results[0] if ahqe_results else None

    print(f"\nBest Uniform:  depth={best_uniform['param']}, accuracy={best_uniform['accuracy']:.4f}")
    print(f"Best Reupload: layers={best_reupload['param']}, accuracy={best_reupload['accuracy']:.4f}")
    if best_ahqe:
        print(f"AHQE:          {best_ahqe['param']}, accuracy={best_ahqe['accuracy']:.4f}")

    best_baseline = max([best_uniform, best_reupload], key=lambda x: x['accuracy'])
    if best_ahqe:
        improvement = (best_ahqe['accuracy'] - best_baseline['accuracy']) * 100
        print(f"\nAHQE improvement over best baseline: {improvement:+.2f}%")

    return results


def diagnose_kernel(K, name):
    print(f"\n[KERNEL DIAGNOSTIC: {name}]")
    print(f"  Shape: {K.shape}")
    print(f"  Diagonal (self-fidelity):")
    print(f"    Mean: {np.diag(K).mean():.6f}")
    print(f"    Std:  {np.diag(K).std():.6f}")
    print(f"    Min:  {np.diag(K).min():.6f}")
    print(f"    Max:  {np.diag(K).max():.6f}")

    n = K.shape[0]
    off_diag = K[np.triu_indices(n, k=1)]
    print(f"  Off-diagonal (cross-fidelity):")
    print(f"    Mean: {off_diag.mean():.6f}")
    print(f"    Std:  {off_diag.std():.6f}")
    print(f"    Min:  {off_diag.min():.6f}")
    print(f"    Max:  {off_diag.max():.6f}")

    unique_vals = len(np.unique(np.round(K, 6)))
    total_vals = K.shape[0] * K.shape[1]
    print(f"  Unique values: {unique_vals}/{total_vals}")

    if off_diag.std() < 0.01:
        print("  ⚠️  WARNING: Kernel collapsed (no variation)!")

    print("  Sample (top-left 5×5):")
    print(K[:5, :5])

    return {
        'diag_mean': np.diag(K).mean(),
        'diag_std': np.diag(K).std(),
        'off_diag_mean': off_diag.mean(),
        'off_diag_std': off_diag.std(),
        'unique_vals': unique_vals
    }


def compare_kernels(K1, K2, name1, name2):
    print(f"\n[COMPARING {name1} vs {name2}]")
    diff = np.linalg.norm(K1 - K2, 'fro')
    max_diff = np.max(np.abs(K1 - K2))
    print(f"  Frobenius norm difference: {diff:.6f}")
    print(f"  Max elementwise difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {np.mean(np.abs(K1 - K2)):.6f}")
    if np.allclose(K1, K2, atol=1e-6):
        print("  ⚠️  IDENTICAL KERNELS (within tolerance)!")
    else:
        print("  ✓ Kernels are different")
    return diff


def run_multiple_seeds(n_seeds=5, backend=None):
    all_results = []
    for seed in range(n_seeds):
        print(f"\n===== SEED {seed} =====")
        results = run_experiment(seed, backend=backend)
        for r in results:
            r_copy = r.copy()
            r_copy["seed"] = seed
            all_results.append(r_copy)
    return all_results


def summarize_results(all_results):
    df = pd.DataFrame(all_results)
    summary = df.groupby(["name", "param"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        depth=("depth", "mean"),
        cnots=("cnots", "mean")
    ).reset_index()
    print("\n===== MEAN ± STD RESULTS =====")
    print(summary)
    return summary


def save_results(all_results, summary):
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(all_results).to_csv("outputs/raw_results.csv", index=False)
    summary.to_csv("outputs/summary_results.csv", index=False)
    print("\nSaved results to outputs/")


def plot_accuracy_vs_depth(summary):
    plt.figure()
    for name in summary["name"].unique():
        subset = summary[summary["name"] == name]
        depths = subset["depth"]
        accs = subset["accuracy_mean"]
        plt.plot(depths, accs, marker='o', label=name)
    plt.xlabel("Circuit Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Depth (Mean across seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/accuracy_vs_depth.png", dpi=300)
    plt.show()


# ========================================================================
# MAIN
# ========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--noise', type=str, default=None, choices=[None, 'FakeVigo'],
                        help='Add noise model (FakeVigo)')
    args = parser.parse_args()

    if args.noise == "FakeVigo":
        print("🔥 RUNNING WITH CUSTOM DEPOLARIZING NOISE (simulating real hardware)")
        noise_model = NoiseModel()   
        error1 = depolarizing_error(0.001, 1)
        noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
        error2 = depolarizing_error(0.003, 2)
        noise_model.add_all_qubit_quantum_error(error2, ['cx'])
        backend = AerSimulator(noise_model=noise_model)
    else:
        print("✅ RUNNING NOISELESS (ideal simulator)")
        backend = AerSimulator()

    all_results = run_multiple_seeds(n_seeds=args.seeds, backend=backend)
    summary = summarize_results(all_results)
    save_results(all_results, summary)
    plot_accuracy_vs_depth(summary)