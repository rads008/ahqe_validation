<!--
═══════════════════════════════════════════════════════════════════════════
  AHQE: Adaptive Hybrid Quantum Encoding
  Resource-Efficient Feature Allocation for NISQ Devices

  ╔═════════════════════════════════════════════════════════════════════╗
  ║  Quantum Machine Learning — NISQ Computing — Resource Allocation   ║
  ║  Code: https://github.com/rads008/ahqe_validation                   ║
  ╚═════════════════════════════════════════════════════════════════════╝
═══════════════════════════════════════════════════════════════════════════
-->

<div align="center">

# 🧠 AHQE
## *Adaptive Hybrid Quantum Encoding*

### Resource-Efficient Feature Allocation for NISQ Devices

---

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg?style=for-the-badge&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-3DA639.svg?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/)
[![GitHub](https://img.shields.io/badge/GitHub-rads008/ahqe_validation-181717.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rads008/ahqe_validation)

---

**📄 Paper:** *Adaptive Hybrid Quantum Encoding: Resource-Efficient Feature Allocation for NISQ Devices*  
**📁 Code:** [https://github.com/rads008/ahqe_validation](https://github.com/rads008/ahqe_validation)  
**📫 Contact:** [gandhiradhika@cvs.du.ac.in](mailto:gandhiradhika@cvs.du.ac.in)

---

</div>

---

## 📋 Abstract

Uniform quantum encodings waste circuit depth on redundant features, leaving NISQ devices unable to execute expressive models within coherence windows. We propose **Adaptive Hybrid Quantum Encoding (AHQE)** , a framework that allocates depth based on feature importance via mutual-information estimation, water-filling allocation under a hardware-derived fidelity budget, and per-feature depth attribution.

Our evaluation demonstrates that AHQE achieves **$0.700 \pm 0.033$** accuracy compared to **$0.727 \pm 0.055$** for the best baseline, while using **54% shallower circuits** and **52% fewer CNOTs**. The difference is not statistically significant ($p = 0.086$), establishing that AHQE achieves **competitive performance with dramatically fewer quantum resources**.

---

## 📖 Table of Contents

- [Abstract](#-abstract)
- [Paper at a Glance](#-paper-at-a-glance)
- [Framework Overview](#-framework-overview)
- [Theoretical Foundation](#-theoretical-foundation)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Reproducing Results](#-reproducing-results)
- [Citation](#-citation)
- [Authors](#-authors)
- [License](#-license)

---

## 🏛️ Paper at a Glance

| **Aspect** | **Details** |
|------------|-------------|
| **Title** | Adaptive Hybrid Quantum Encoding: Resource-Efficient Feature Allocation for NISQ Devices |
| **Authors** | R. Gandhi, G. Vashishta |
| **Institution** | University of Delhi, College of Vocational Studies |
| **Domain** | Quantum Machine Learning, NISQ Computing, Quantum Resource Allocation |
| **Contribution** | Novel water-filling allocation framework for quantum data encoding |
| **Theoretical Guarantees** | Uniqueness, discretisation bounds, Lipschitz stability, circuit complexity bounds |
| **Validation** | Binary classification (120 samples, 8 features), 5 seeds, depolarising noise (1% CNOT) |

### 📊 Core Results

| Metric | AHQE | Best Baseline | Advantage |
|--------|------|---------------|-----------|
| Accuracy | `0.700 ± 0.033` | `0.727 ± 0.055` | Competitive (−2.7%) |
| Circuit Depth | **11** | **24** | **54% shallower** |
| CNOT Count | **10** | **21** | **52% fewer** |
| Variance | **0.033** | **0.055** | **40% more stable** |
| p-value | — | — | `0.086` (not significant) |

---

## 🔬 Framework Overview

### The Four Phases

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│   ╔═════════════════════════════════════════════════════════════════════╗ │
│   ║  PHASE I: Feature Importance Estimation                            ║ │
│   ║  ┌───────────────────────────────────────────────────────────────┐ ║ │
│   ║  │  • Partition N features into m subsets                        │ ║ │
│   ║  │  • Estimate mutual information I(x^(i); Y) via k-NN (k=3)   │ ║ │
│   ║  │  • φ_i = I(x^(i); Y) / Σ I(x^(j); Y)  [Normalised]          │ ║ │
│   ║  └───────────────────────────────────────────────────────────────┘ ║ │
│   ╚═════════════════════════════════════════════════════════════════════╝ │
│                                    ↓                                      │
│   ╔═════════════════════════════════════════════════════════════════════╗ │
│   ║  PHASE II: Water-Filling Allocation                                ║ │
│   ║  ┌───────────────────────────────────────────────────────────────┐ ║ │
│   ║  │  • Optimisation:                                              │ ║ │
│   ║  │    R(κ) = Σ φ_i (1 - e^{-ν_i κ_i})                           │ ║ │
│   ║  │    s.t. Σ α_i κ_i ≤ E_max                                    │ ║ │
│   ║  │  • KKT Solution:                                              │ ║ │
│   ║  │    κ_i* = max(0, 1/ν_i ln(φ_i ν_i / (λ α_i)))               │ ║ │
│   ║  │  • Budget constraint: λ determined by Σ α_i κ_i* = E_max    │ ║ │
│   ║  └───────────────────────────────────────────────────────────────┘ ║ │
│   ╚═════════════════════════════════════════════════════════════════════╝ │
│                                    ↓                                      │
│   ╔═════════════════════════════════════════════════════════════════════╗ │
│   ║  PHASE III: Circuit Construction                                   ║ │
│   ║  ┌───────────────────────────────────────────────────────────────┐ ║ │
│   ║  │  • Tensor product: U_AHQE(x) = ⊗_{i=1}^m E_{κ_i}(x^(i))     │ ║ │
│   ║  │  • Data re-uploading: E_κ = W ∘ D(x) ∘ E_{κ-1}              │ ║ │
│   ║  │  • D(x) = ⊗ R_y(x_j)  [Angle encoding]                       │ ║ │
│   ║  │  • W = nearest-neighbour CNOTs + parameterised R_y           │ ║ │
│   ║  └───────────────────────────────────────────────────────────────┘ ║ │
│   ╚═════════════════════════════════════════════════════════════════════╝ │
│                                    ↓                                      │
│   ╔═════════════════════════════════════════════════════════════════════╗ │
│   ║  PHASE IV: Depth Attribution (Interpretability)                    ║ │
│   ║  ┌───────────────────────────────────────────────────────────────┐ ║ │
│   ║  │  • Per-feature depth narrative:                               │ ║ │
│   ║  │    "Features 2, 4, 5, and 8 received 3, 2, 2, and 2 layers.  │ ║ │
│   ║  │     Features 1, 3, and 7 allocated zero depth."              │ ║ │
│   ║  │  • Forensic-ready: tamper-evident, interpretable records     │ ║ │
│   ║  └───────────────────────────────────────────────────────────────┘ ║ │
│   ╚═════════════════════════════════════════════════════════════════════╝ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Quantum Circuit Architecture

The AHQE encoding circuit is a **tensor product of independent subcircuits**:

$$U_{\text{AHQE}}(x) = \bigotimes_{i=1}^m \mathcal{E}_{\kappa_i}(x^{(i)})$$

For each feature subset, the subcircuit is constructed via **data re-uploading**:

$$\mathcal{E}_{\kappa}(x^{(i)}) = \mathcal{W} \circ \mathcal{D}(x^{(i)}) \circ \mathcal{E}_{\kappa-1}(x^{(i)})$$

where:
- $\mathcal{D}(x^{(i)}) = \bigotimes_{j=1}^{n_i} R_y(x_j^{(i)})$ — Angle encoding
- $\mathcal{W}$ — nearest-neighbour CNOTs + parameterised $R_y$ rotations
- $\kappa_i$ — number of entangling layers allocated to feature subset $i$

---

## 📐 Theoretical Foundation

AHQE provides the following **mathematical guarantees**:

### Proposition 1: Uniqueness
The continuous relaxation admits a **unique global maximiser**.

### Proposition 2: Discretisation Bound
$$R(\kappa^*) - R(\kappa) \leq \phi_{\max}(1 - e^{-\nu_{\max}/2}) + \frac{\phi_{\max} \nu_{\max}}{\alpha_{\min}(1 - e^{-\nu_{\min}})}$$

### Proposition 3: Lipschitz Stability
$$\|\kappa^* - \kappa'\|_1 \leq C_\phi \Delta_\phi + C_\nu \Delta_\nu + C_\alpha \Delta_\alpha$$

### Corollary 1: Scale-Invariance
The discretisation bound depends only on parameter ratios, not on $m$ or $E_{\max}$.

### Corollary 2: Circuit Complexity
$$G_{\text{CNOT}} \leq \frac{n_q E_{\max}}{\alpha_{\min}}, \quad D_{\text{critical}} \leq \frac{2E_{\max}}{\alpha_{\min}} + 1$$

---

## 📊 Key Results

### 1. Noiseless Results

| Encoding | Param | Depth | CNOTs | Accuracy |
|----------|-------|-------|-------|----------|
| Uniform | d=1 | 9 | 7 | 0.900 |
| Uniform | d=2 | 17 | 14 | 0.900 |
| Uniform | d=3 | 25 | 21 | 0.900 |
| Reupload | L=1 | 8 | 7 | 0.920 |
| Reupload | L=2 | 16 | 14 | 0.920 |
| Reupload | L=3 | 24 | 21 | 0.900 |
| **AHQE** | **adaptive** | **14** | **10** | **0.920** |

### 2. Noisy Results (5 Seeds, 1% CNOT Error)

| Encoding | Param | Depth | CNOTs | Accuracy (Mean ± Std) |
|----------|-------|-------|-------|----------------------|
| Uniform | d=1 | 9 | 7 | 0.720 ± 0.038 |
| Uniform | d=2 | 17 | 14 | 0.713 ± 0.065 |
| Uniform | d=3 | 25 | 21 | 0.720 ± 0.069 |
| Reupload | L=1 | 8 | 7 | 0.720 ± 0.038 |
| Reupload | L=2 | 16 | 14 | 0.720 ± 0.051 |
| Reupload | L=3 | 24 | 21 | 0.727 ± 0.055 |
| **AHQE** | **adaptive** | **11** | **10** | **0.700 ± 0.033** |

### 3. Allocation Analysis

| Feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|---|---|
| $\phi_i$ | 0.000 | 0.305 | 0.000 | 0.269 | 0.142 | 0.107 | 0.000 | 0.177 |
| $\kappa_i$ | 0 | 3 | 0 | 2 | 2 | 1 | 0 | 2 |

**Key Observations:**
- ✅ Zero-importance features ($\phi_i = 0$) receive zero depth
- ✅ Important features receive depth proportional to importance-to-noise ratio
- ✅ Budget fully utilised (10.0/10)
- ✅ Interpretable, forensically-attributable allocation

---

## 📂 Repository Structure

```
ahqe_validation/
│
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
│
├── 📁 src/                              # Source code
│   ├── 📁 allocation/                   # Allocation algorithms
│   │   ├── __init__.py
│   │   ├── importance.py                # Mutual information estimation
│   │   ├── waterfill.py                 # Water-filling allocation
│   │   └── plotallocation.py            # Generate allocation plots
│   │
│   ├── 📁 encodings/                    # Quantum encoding implementations
│   │   ├── __init__.py
│   │   └── encodings.py                 # Uniform, Reupload, AHQE encodings
│   │
│   ├── 📁 experiments/                  # Experiment runner
│   │   ├── __init__.py
│   │   ├── kernel.py                    # Original kernel computation
│   │   ├── kernelcopy.py                # Optimised batched kernel
│   │   ├── run_experiment.py            # Original experiment runner
│   │   └── run_experimentcopy.py        # Main experiment runner
│   │
│   └── 📁 utils/                        # Utility functions
│       └── __init__.py
│
├── 📁 scripts/                          # Plotting scripts
│   ├── plot_allocation.py               # Generate allocation visualisation
│   └── plot_noiseless.py                # Generate noiseless figure
│
├── 📁 outputs/                          # Experimental results (generated)
│   ├── raw_results.csv                  # Per-seed raw results
│   ├── summary_results.csv              # Aggregated statistics
│   └── accuracy_vs_depth.png            # Generated plot
│
└── 📁 figures/                          # Paper figures (generated)
    ├── Figure_1.png                     # Noisy accuracy vs depth
    ├── plot_accuracyVsDepth.png         # Noiseless accuracy vs depth
    └── allocation.png                   # Allocation visualisation
```

---

## 🚀 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Qiskit | 1.0+ |
| PennyLane | 0.35+ |
| NumPy | 1.24+ |
| SciPy | 1.10+ |
| Matplotlib | 3.7+ |
| Pandas | 2.0+ |
| scikit-learn | 1.3+ |

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/rads008/ahqe_validation.git
cd ahqe_validation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import qiskit; print(f'Qiskit {qiskit.__version__}')"
```

---

## 🎯 Usage

### Running the Main Experiment

```bash
# Noiseless simulation (statevector)
python -m src.experiments.run_experimentcopy --seeds 5

# Noisy simulation with depolarising noise (recommended for NISQ realism)
python -m src.experiments.run_experimentcopy --noise FakeVigo --seeds 5

# High-statistics run (10 seeds)
python -m src.experiments.run_experimentcopy --noise FakeVigo --seeds 10
```

### Generating Plots

```bash
# Generate allocation visualisation
python src/allocation/plotallocation.py

# Generate noiseless accuracy vs depth figure
python scripts/plot_noiseless.py
```

### Output Location

| File | Description |
|------|-------------|
| `outputs/raw_results.csv` | Per-seed experiment results |
| `outputs/summary_results.csv` | Aggregated statistics (mean ± std) |
| `outputs/accuracy_vs_depth.png` | Accuracy vs depth plot |

---

## 📈 Reproducing Results

To reproduce the exact results from the paper:

```bash
# Step 1: Run the experiment with the same configuration
python -m src.experiments.run_experimentcopy --noise FakeVigo --seeds 5

# Step 2: Generate the figures
python src/allocation/plotallocation.py
python scripts/plot_noiseless.py

# Step 3: Check outputs
cat outputs/summary_results.csv
```

### Expected Output

```
name,param,accuracy_mean,accuracy_std,depth,cnots
AHQE,adaptive,0.700,0.033,11.0,10.0
Reupload,1,0.720,0.038,8.0,7.0
Reupload,2,0.720,0.051,16.0,14.0
Reupload,3,0.727,0.055,24.0,21.0
Uniform,1,0.720,0.038,9.0,7.0
Uniform,2,0.713,0.065,17.0,14.0
Uniform,3,0.720,0.069,25.0,21.0
```

---

## 📝 Citation

If you use this code or find it useful in your research, please cite:

```bibtex
@article{gandhi2026ahqe,
  title={Adaptive Hybrid Quantum Encoding: Resource-Efficient Feature Allocation for {NISQ} Devices},
  author={Gandhi, Radhika and Vashishta, Geetika},
  year={2026},
  note={Source code available at \url{https://github.com/rads008/ahqe_validation}}
}
```

### BibTeX Entry (For Your Paper)

```bibtex
@misc{ahqe2026code,
  author = {Gandhi, Radhika and Vashishta, Geetika},
  title = {AHQE: Adaptive Hybrid Quantum Encoding — Source Code},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/rads008/ahqe_validation}}
}
```

---

## 👩‍💻 Authors

**Radhika Gandhi**  
*Department of Computer Science, College of Vocational Studies*  
*University of Delhi, New Delhi, India*  
📧 [gandhiradhika@cvs.du.ac.in](mailto:gandhiradhika@cvs.du.ac.in)  
🔗 [GitHub](https://github.com/rads008)

**Dr. Geetika Vashishta** (Advisor)  
*Assistant Professor, Department of Computer Science*  
*College of Vocational Studies, University of Delhi*  
📧 [geetikavashishta@cvs.du.ac.in](mailto:geetikavashishta@cvs.du.ac.in)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2026 Radhika Gandhi, Geetika Vashishta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Department of Computer Science**, University of Delhi
- **College of Vocational Studies**, University of Delhi
- **Qiskit** and **PennyLane** open-source communities

---

## 📧 Contact

| Purpose | Email |
|---------|-------|
| Research inquiries | gandhiradhika@cvs.du.ac.in |
| Advisor | geetikavashishta@cvs.du.ac.in |
| GitHub Issues | [https://github.com/rads008/ahqe_validation/issues](https://github.com/rads008/ahqe_validation/issues) |

---

<div align="center">

---

*"A circuit that cannot execute is worth less than a circuit that executes with slightly lower accuracy."*

— AHQE, 2026

---

[⬆ Back to Top](#-ahqe)

</div>
