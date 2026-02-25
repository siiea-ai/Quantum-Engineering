# Quantum Engineering - Interactive Notebooks

Interactive Jupyter notebooks for the [Quantum Engineering Curriculum](https://github.com/siiea-ai/Quantum-Engineering).

## Quick Start

```bash
# From the repo root:
./setup.sh              # One-time setup (auto-detects hardware)
source activate.sh      # Activate environment
jupyter lab             # Launch JupyterLab
```

## Structure

```
notebooks/
├── hardware_config.py              # Auto-generated hardware profile
├── year_0/
│   ├── month_01_calculus_I/        # Limits, derivatives, applications
│   ├── month_02_calculus_II/       # Integration, series, convergence
│   ├── month_03_multivariable_ode/ # Vector calculus, ODEs
│   ├── month_04_linear_algebra_I/  # Matrices, eigenvalues, SVD
│   ├── month_05_linear_algebra_II_complex/ # Complex numbers, advanced LA
│   ├── month_06_classical_mechanics/ # Lagrangian, Hamiltonian
│   ├── month_07_complex_analysis/  # Contour integration, residues
│   ├── month_08_electromagnetism/  # Maxwell's equations, EM waves
│   ├── month_09_functional_analysis/ # Hilbert spaces, operators
│   ├── month_10_scientific_computing/ # NumPy, SciPy, simulation
│   ├── month_11_group_theory/      # Symmetries, representations
│   └── month_12_capstone/          # Integration project
└── mlx_labs/                       # Apple Silicon-optimized labs
    ├── 01_mlx_quantum_basics.ipynb
    ├── 02_large_scale_simulation.ipynb
    └── 03_quantum_neural_network.ipynb
```

## Hardware Profiles

The setup script auto-detects your hardware and configures simulation limits:

| Machine | Memory | Max Qubits | Profile |
|---------|--------|------------|---------|
| Mac Studio (M2/M4 Ultra) | 512 GB | ~33 | `studio` |
| MacBook Pro (M4 Max) | 128 GB | ~30 | `pro` |
| Standard Mac | < 96 GB | ~25 | `standard` |

Notebooks automatically adapt to your hardware via `hardware_config.py`.

## Companion Curriculum

Each notebook corresponds to specific days in the main curriculum. References are included in every notebook header.
