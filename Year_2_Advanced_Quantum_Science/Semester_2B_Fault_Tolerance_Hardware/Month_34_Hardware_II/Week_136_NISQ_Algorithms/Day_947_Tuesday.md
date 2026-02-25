# Day 947: Variational Quantum Eigensolver (VQE)

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | VQE algorithm theory and ansatz design |
| Afternoon | 2.5 hours | Problem solving and gradient computation |
| Evening | 2 hours | Computational lab: VQE for H₂ molecule |

## Learning Objectives

By the end of today, you will be able to:

1. Explain the variational principle and its application to quantum ground state estimation
2. Design and implement parameterized quantum circuits (ansatzes) for VQE
3. Apply the parameter shift rule for computing quantum gradients
4. Decompose molecular Hamiltonians into Pauli strings
5. Implement complete VQE workflows for small molecules
6. Analyze VQE convergence and optimization landscapes

## Core Content

### 1. The Variational Principle

The variational principle is the theoretical foundation of VQE:

$$\boxed{E_0 \leq E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle}$$

where:
- $E_0$ is the true ground state energy
- $|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|0\rangle$ is the variational ansatz
- $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_p)$ are variational parameters

**Key insight:** Any trial wavefunction gives an energy **at or above** the ground state energy. By minimizing over $\boldsymbol{\theta}$, we approximate the ground state.

### 2. VQE Algorithm Structure

The VQE algorithm consists of a quantum-classical hybrid loop:

```
┌─────────────────────────────────────────────────────────────┐
│  CLASSICAL COMPUTER                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ Initialize  │───▶│   Optimize   │───▶│ Check         │   │
│  │ θ           │    │   θ          │    │ Convergence   │   │
│  └─────────────┘    └──────────────┘    └───────────────┘   │
│         │                  ▲                    │            │
│         ▼                  │                    ▼            │
│  ┌─────────────────────────┴────────────┐  ┌──────────┐     │
│  │         E(θ), ∇E(θ)                  │  │  Output  │     │
│  └──────────────────────────────────────┘  │  E_min   │     │
│         ▲                                   └──────────┘     │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│  QUANTUM PROCESSOR                                           │
│         │                                                    │
│  ┌──────┴──────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ Prepare     │───▶│  Measure     │───▶│ Estimate      │   │
│  │ |ψ(θ)⟩     │    │  ⟨H_i⟩       │    │ E(θ)          │   │
│  └─────────────┘    └──────────────┘    └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Algorithm Steps:**

1. **Initialize** parameters $\boldsymbol{\theta}$
2. **Prepare** the variational state $|\psi(\boldsymbol{\theta})\rangle$ on quantum hardware
3. **Measure** expectation values of Hamiltonian terms
4. **Compute** total energy $E(\boldsymbol{\theta})$
5. **Update** parameters using classical optimizer
6. **Repeat** until convergence

### 3. Ansatz Design

The ansatz (trial wavefunction circuit) is critical to VQE success.

#### 3.1 Hardware-Efficient Ansatz (HEA)

Designed to match native hardware operations:

$$U(\boldsymbol{\theta}) = \prod_{l=1}^{L} \left[\prod_{j=1}^{n} R_Y(\theta_{l,j}) \cdot \text{Entangling Layer}\right]$$

**Example (4 qubits, 2 layers):**

```
     ┌───────┐     ┌───────┐          ┌───────┐     ┌───────┐
q0: ─┤ RY(θ₁)├──■──┤ RY(θ₅)├───■──────┤ RY(θ₉)├──■──┤RY(θ₁₃)├
     └───────┘  │  └───────┘   │      └───────┘  │  └───────┘
     ┌───────┐┌─┴─┐┌───────┐   │      ┌───────┐┌─┴─┐┌───────┐
q1: ─┤ RY(θ₂)├┤ X ├┤ RY(θ₆)├──■──────┤RY(θ₁₀)├┤ X ├┤RY(θ₁₄)├
     └───────┘└───┘└───────┘  │      └───────┘└───┘└───────┘
     ┌───────┐     ┌───────┐┌─┴─┐    ┌───────┐     ┌───────┐
q2: ─┤ RY(θ₃)├──■──┤ RY(θ₇)├┤ X ├─■──┤RY(θ₁₁)├──■──┤RY(θ₁₅)├
     └───────┘  │  └───────┘└───┘ │  └───────┘  │  └───────┘
     ┌───────┐┌─┴─┐┌───────┐    ┌─┴─┐┌───────┐┌─┴─┐┌───────┐
q3: ─┤ RY(θ₄)├┤ X ├┤ RY(θ₈)├────┤ X ├┤RY(θ₁₂)├┤ X ├┤RY(θ₁₆)├
     └───────┘└───┘└───────┘    └───┘└───────┘└───┘└───────┘
```

#### 3.2 Unitary Coupled Cluster (UCC) Ansatz

Physics-inspired for molecular systems:

$$|\psi\rangle = e^{\hat{T} - \hat{T}^\dagger}|\phi_0\rangle$$

where $|\phi_0\rangle$ is the Hartree-Fock reference and:

$$\hat{T} = \hat{T}_1 + \hat{T}_2 + \ldots = \sum_{ia} t_i^a \hat{a}_a^\dagger \hat{a}_i + \sum_{ijab} t_{ij}^{ab} \hat{a}_a^\dagger \hat{a}_b^\dagger \hat{a}_j \hat{a}_i + \ldots$$

**UCCSD (singles and doubles):**
$$\boxed{U_{\text{UCCSD}}(\boldsymbol{\theta}) = e^{\sum_k \theta_k (\hat{\tau}_k - \hat{\tau}_k^\dagger)}}$$

### 4. Hamiltonian Decomposition

Molecular Hamiltonians must be expressed as sums of Pauli operators:

$$\hat{H} = \sum_i c_i \hat{P}_i$$

where each $\hat{P}_i$ is a tensor product of Pauli matrices:

$$\hat{P}_i \in \{I, X, Y, Z\}^{\otimes n}$$

**Jordan-Wigner Transformation:**
Maps fermionic operators to qubit operators:

$$\hat{a}_j^\dagger \rightarrow \frac{1}{2}(X_j - iY_j) \otimes Z_{j-1} \otimes Z_{j-2} \otimes \ldots \otimes Z_0$$

$$\hat{a}_j \rightarrow \frac{1}{2}(X_j + iY_j) \otimes Z_{j-1} \otimes Z_{j-2} \otimes \ldots \otimes Z_0$$

**Example: H₂ Hamiltonian (4 qubits, STO-3G basis):**

$$\hat{H}_{H_2} = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_2 + g_4 Z_3 + g_5 Z_0Z_1 + \ldots + g_{14} X_0Y_1Y_2X_3$$

with approximately 15 terms (can be reduced by symmetry).

### 5. Energy Estimation

For each Pauli term $\hat{P}_i$:

$$\langle\hat{P}_i\rangle = \langle\psi(\boldsymbol{\theta})|\hat{P}_i|\psi(\boldsymbol{\theta})\rangle$$

**Measurement procedure:**
1. Apply basis rotation to diagonalize $\hat{P}_i$
2. Measure in computational basis
3. Compute expectation from bitstring statistics

**Basis rotations:**
- $X \rightarrow Z$: Apply $H$ before measurement
- $Y \rightarrow Z$: Apply $S^\dagger H$ before measurement
- $Z \rightarrow Z$: No rotation needed

**Statistical error (shot noise):**

$$\boxed{\sigma_E = \sqrt{\sum_i |c_i|^2 \frac{1 - \langle\hat{P}_i\rangle^2}{N_{\text{shots}}}}}$$

### 6. Parameter Shift Rule

The gradient of expectation values for parameterized gates:

For a gate $R(\theta) = e^{-i\theta G/2}$ with generator $G$ having eigenvalues $\pm 1$:

$$\boxed{\frac{\partial}{\partial\theta}\langle\hat{O}\rangle_\theta = \frac{1}{2}\left(\langle\hat{O}\rangle_{\theta+\frac{\pi}{2}} - \langle\hat{O}\rangle_{\theta-\frac{\pi}{2}}\right)}$$

**Derivation:**

$$\langle\hat{O}\rangle_\theta = \langle 0|U^\dagger(\theta) \hat{O} U(\theta)|0\rangle$$

Using $e^{i\theta G/2} = \cos(\theta/2)I + i\sin(\theta/2)G$:

$$\frac{\partial}{\partial\theta}\langle\hat{O}\rangle = \frac{1}{2}\langle[\hat{O}, G]\rangle_\theta$$

The shift rule evaluates this without explicitly computing commutators.

### 7. Classical Optimization

**Gradient-free methods:**
- COBYLA: Constrained optimization by linear approximation
- Nelder-Mead: Simplex-based search
- SPSA: Simultaneous perturbation stochastic approximation

**Gradient-based methods:**
- L-BFGS-B: Quasi-Newton with limited memory
- Adam: Adaptive moment estimation (from ML)
- Natural gradient: Uses quantum Fisher information

**SPSA (practical for noisy gradients):**

$$\theta_{k+1} = \theta_k - a_k \frac{E(\theta_k + c_k\Delta) - E(\theta_k - c_k\Delta)}{2c_k\Delta}$$

where $\Delta$ is a random perturbation vector.

## Quantum Computing Applications

### Application: Quantum Chemistry

VQE enables simulation of molecular systems beyond classical capability:

| Molecule | Qubits (STO-3G) | Parameters (UCCSD) | Classical Alternative |
|----------|-----------------|--------------------|-----------------------|
| H₂ | 4 | 3 | Full CI |
| LiH | 12 | 40+ | CCSD(T) |
| H₂O | 14 | 100+ | DMRG |
| FeMoCo | 150+ | 10000+ | None feasible |

### Application: Materials Science

VQE can compute:
- Band structures of materials
- Magnetic ordering in spin systems
- Superconducting gaps

## Worked Examples

### Example 1: VQE Circuit Construction

**Problem:** Construct a 2-qubit hardware-efficient ansatz with RY rotations and CNOT entanglement.

**Solution:**

The ansatz structure:
$$U(\theta_1, \theta_2, \theta_3, \theta_4) = U_{\text{layer2}}U_{\text{layer1}}$$

Layer 1:
$$U_{\text{layer1}} = \text{CNOT}_{01} \cdot (R_Y(\theta_1) \otimes R_Y(\theta_2))$$

Layer 2:
$$U_{\text{layer2}} = \text{CNOT}_{01} \cdot (R_Y(\theta_3) \otimes R_Y(\theta_4))$$

**Circuit:**
```
     ┌─────────┐     ┌─────────┐
q0: ─┤ RY(θ₁) ├──■──┤ RY(θ₃) ├──■──
     └─────────┘  │  └─────────┘  │
     ┌─────────┐┌─┴─┐┌─────────┐┌─┴─┐
q1: ─┤ RY(θ₂) ├┤ X ├┤ RY(θ₄) ├┤ X ├
     └─────────┘└───┘└─────────┘└───┘
```

**Matrix form:**
$$U = \text{CNOT} \cdot (R_Y(\theta_3) \otimes R_Y(\theta_4)) \cdot \text{CNOT} \cdot (R_Y(\theta_1) \otimes R_Y(\theta_2))$$

### Example 2: Parameter Shift Gradient

**Problem:** Compute $\frac{\partial}{\partial\theta}\langle Z\rangle$ for the circuit $|0\rangle \xrightarrow{R_Y(\theta)} |\psi\rangle$.

**Solution:**

Step 1: State after rotation
$$|\psi(\theta)\rangle = R_Y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$$

Step 2: Expectation value
$$\langle Z\rangle_\theta = \cos^2(\theta/2) - \sin^2(\theta/2) = \cos\theta$$

Step 3: Apply parameter shift rule
$$\frac{\partial\langle Z\rangle}{\partial\theta} = \frac{1}{2}\left(\langle Z\rangle_{\theta+\pi/2} - \langle Z\rangle_{\theta-\pi/2}\right)$$

$$= \frac{1}{2}\left(\cos(\theta+\pi/2) - \cos(\theta-\pi/2)\right)$$

$$= \frac{1}{2}\left(-\sin\theta - \sin\theta\right) = -\sin\theta$$

Step 4: Verify analytically
$$\frac{d}{d\theta}\cos\theta = -\sin\theta \quad \checkmark$$

### Example 3: Hamiltonian Term Measurement

**Problem:** Measure $\langle X_0 Z_1\rangle$ for a 2-qubit state.

**Solution:**

Step 1: Diagonalization
- $X_0$ requires $H$ rotation on qubit 0
- $Z_1$ requires no rotation

Step 2: Modified circuit
```
     ┌─────────────┐┌───┐┌─┐
q0: ─┤ Ansatz     ├┤ H ├┤M├───
     └──────┬──────┘└───┘└╥┘
            │             ║
q1: ────────┴────────────╫─┤M├
                         ║ └╥┘
c:  ═════════════════════╩══╩═
```

Step 3: Measurement mapping
For outcome $|b_0 b_1\rangle$:
$$X_0 Z_1 |b_0 b_1\rangle \rightarrow (-1)^{b_0} \cdot (-1)^{b_1} |b_0 b_1\rangle$$

Expectation:
$$\langle X_0 Z_1\rangle = \sum_{b_0, b_1} (-1)^{b_0 + b_1} P(b_0, b_1)$$

$$= P(00) - P(01) - P(10) + P(11)$$

## Practice Problems

### Level 1: Direct Application

1. **Ansatz counting:** For a hardware-efficient ansatz with $n$ qubits and $L$ layers, with $R_Y$ and $R_Z$ rotations per qubit per layer, how many parameters are there?

2. **Pauli measurement:** What basis rotation is needed to measure $\langle Y_0 X_1\rangle$?

3. **Shot noise:** If measuring a Pauli operator with true expectation $\langle P\rangle = 0.5$, how many shots are needed for standard error $<0.01$?

### Level 2: Intermediate Analysis

4. **Gradient computation:** For the circuit $R_X(\theta_1)R_Y(\theta_2)|0\rangle$, compute the gradient of $\langle Z\rangle$ with respect to both parameters.

5. **Grouping optimization:** Given the Hamiltonian $H = 0.5 Z_0 + 0.3 Z_1 - 0.2 Z_0Z_1 + 0.1 X_0X_1$, how many distinct measurement circuits are needed?

6. **Energy precision:** Estimate the total number of circuit executions needed to achieve 1 mHa (millihartree) precision for an 8-term Hamiltonian with equal coefficients of magnitude 0.5 Ha.

### Level 3: Challenging Problems

7. **UCCSD analysis:** For the H₂ molecule in a minimal basis, derive the form of the UCCSD operator and count the number of variational parameters.

8. **Optimizer comparison:** Design an experiment to compare COBYLA, L-BFGS-B, and SPSA for VQE optimization on a noisy simulator. What metrics would you use?

9. **Symmetry exploitation:** The H₂ Hamiltonian has particle number and spin symmetry. How can these symmetries be used to reduce the number of qubits and parameters?

## Computational Lab: VQE for H₂

### Lab 1: Complete VQE Implementation

```python
"""
Day 947 Lab: VQE for Hydrogen Molecule
Complete implementation from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# Part 1: H₂ Hamiltonian at Different Bond Lengths
# ============================================================

def get_h2_hamiltonian(bond_length: float) -> SparsePauliOp:
    """
    Get H₂ Hamiltonian in qubit representation.
    Using pre-computed coefficients for STO-3G basis.
    """
    # Coefficients depend on bond length
    # These are for Jordan-Wigner mapping, 4 qubits
    # Simplified version with most significant terms

    if bond_length == 0.735:  # Near equilibrium
        coeffs = {
            'IIII': -0.8105,
            'IIIZ': 0.1721,
            'IIZI': -0.2257,
            'IZII': 0.1721,
            'ZIII': -0.2257,
            'IIZZ': 0.1209,
            'IZIZ': 0.1689,
            'IZZI': 0.0454,
            'ZIIZ': 0.0454,
            'ZIZI': 0.1689,
            'ZZII': 0.1209,
            'XXXX': 0.0454,
            'XXYY': 0.0454,
            'YYXX': 0.0454,
            'YYYY': 0.0454
        }
    else:
        # Linear interpolation for demonstration
        r = bond_length
        coeffs = {
            'IIII': -0.8105 + 0.1 * (r - 0.735),
            'IIIZ': 0.1721 - 0.05 * (r - 0.735),
            'IIZI': -0.2257 + 0.03 * (r - 0.735),
            'IZII': 0.1721 - 0.05 * (r - 0.735),
            'ZIII': -0.2257 + 0.03 * (r - 0.735),
            'IIZZ': 0.1209,
            'IZIZ': 0.1689,
            'IZZI': 0.0454,
            'ZIIZ': 0.0454,
            'ZIZI': 0.1689,
            'ZZII': 0.1209,
            'XXXX': 0.0454,
            'XXYY': 0.0454,
            'YYXX': 0.0454,
            'YYYY': 0.0454
        }

    pauli_list = [(pauli, coeff) for pauli, coeff in coeffs.items()]
    return SparsePauliOp.from_list(pauli_list)

# Test Hamiltonian
H = get_h2_hamiltonian(0.735)
print("H₂ Hamiltonian at equilibrium (r = 0.735 Å):")
print(f"Number of terms: {len(H)}")
print(f"Sample terms: {list(H.to_list())[:5]}")

# ============================================================
# Part 2: Hardware-Efficient Ansatz
# ============================================================

def hardware_efficient_ansatz(n_qubits: int, n_layers: int,
                               params: np.ndarray) -> QuantumCircuit:
    """
    Create hardware-efficient ansatz with RY rotations and CNOT entanglement.
    """
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1

        # Entangling layer (linear connectivity)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

    # Final rotation layer
    for qubit in range(n_qubits):
        qc.ry(params[param_idx], qubit)
        param_idx += 1

    return qc

def count_parameters(n_qubits: int, n_layers: int) -> int:
    """Count parameters in hardware-efficient ansatz."""
    return n_qubits * (n_layers + 1)

# Create example ansatz
n_qubits = 4
n_layers = 2
n_params = count_parameters(n_qubits, n_layers)
print(f"\nAnsatz with {n_qubits} qubits, {n_layers} layers:")
print(f"Number of parameters: {n_params}")

test_params = np.random.random(n_params) * 2 * np.pi
qc = hardware_efficient_ansatz(n_qubits, n_layers, test_params)
print(f"Circuit depth: {qc.depth()}")
print(qc.draw(output='text', fold=80))

# ============================================================
# Part 3: VQE Cost Function
# ============================================================

def vqe_cost(params: np.ndarray, hamiltonian: SparsePauliOp,
             n_qubits: int, n_layers: int, estimator: Estimator) -> float:
    """
    Compute VQE energy for given parameters.
    """
    # Create ansatz with current parameters
    qc = hardware_efficient_ansatz(n_qubits, n_layers, params)

    # Compute expectation value
    job = estimator.run([(qc, hamiltonian)])
    result = job.result()

    return float(result[0].data.evs)

# ============================================================
# Part 4: VQE Optimization
# ============================================================

def run_vqe(hamiltonian: SparsePauliOp, n_qubits: int, n_layers: int,
            max_iter: int = 100, seed: int = 42) -> dict:
    """
    Run complete VQE optimization.
    """
    np.random.seed(seed)

    # Initialize parameters
    n_params = count_parameters(n_qubits, n_layers)
    initial_params = np.random.random(n_params) * 0.1  # Small random init

    # Create estimator
    estimator = Estimator()

    # Track optimization history
    energy_history = []

    def callback_wrapper(params):
        energy = vqe_cost(params, hamiltonian, n_qubits, n_layers, estimator)
        energy_history.append(energy)
        return energy

    # Run optimization
    print(f"Starting VQE with {n_params} parameters...")
    print(f"Initial energy: {callback_wrapper(initial_params):.6f} Ha")

    result = minimize(
        callback_wrapper,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.5}
    )

    return {
        'optimal_params': result.x,
        'optimal_energy': result.fun,
        'history': energy_history,
        'n_iterations': len(energy_history),
        'success': result.success
    }

# Run VQE for H₂
print("\n" + "="*60)
print("Running VQE for H₂ molecule")
print("="*60)

H = get_h2_hamiltonian(0.735)
vqe_result = run_vqe(H, n_qubits=4, n_layers=2, max_iter=150)

print(f"\nVQE Results:")
print(f"  Final energy: {vqe_result['optimal_energy']:.6f} Ha")
print(f"  Iterations: {vqe_result['n_iterations']}")
print(f"  Exact ground state: -1.1373 Ha (FCI)")
print(f"  Error: {abs(vqe_result['optimal_energy'] - (-1.1373)):.6f} Ha")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(vqe_result['history'], 'b-', linewidth=2)
plt.axhline(y=-1.1373, color='r', linestyle='--', label='Exact (FCI)')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('VQE Convergence for H₂ Molecule', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vqe_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 5: Potential Energy Surface
# ============================================================

def compute_pes(bond_lengths: np.ndarray, n_layers: int = 2) -> dict:
    """
    Compute potential energy surface using VQE.
    """
    vqe_energies = []
    exact_energies = []

    # Approximate exact energies (FCI values for H₂)
    exact_data = {
        0.5: -0.9685, 0.6: -1.0551, 0.7: -1.1051, 0.735: -1.1373,
        0.8: -1.1188, 0.9: -1.1116, 1.0: -1.1015, 1.2: -1.0614,
        1.5: -0.9905, 2.0: -0.9183, 2.5: -0.8796
    }

    for r in bond_lengths:
        print(f"Computing VQE at r = {r:.2f} Å...")

        # Get Hamiltonian
        H = get_h2_hamiltonian(r)

        # Run VQE (fewer iterations for speed)
        result = run_vqe(H, n_qubits=4, n_layers=n_layers, max_iter=80, seed=42)
        vqe_energies.append(result['optimal_energy'])

        # Get exact value (interpolate if needed)
        closest_r = min(exact_data.keys(), key=lambda x: abs(x - r))
        exact_energies.append(exact_data[closest_r])

    return {
        'bond_lengths': bond_lengths,
        'vqe_energies': np.array(vqe_energies),
        'exact_energies': np.array(exact_energies)
    }

# Compute PES (reduced set for speed)
print("\n" + "="*60)
print("Computing H₂ Potential Energy Surface")
print("="*60)

bond_lengths = np.array([0.5, 0.735, 1.0, 1.5, 2.0])
pes = compute_pes(bond_lengths)

# Plot PES
plt.figure(figsize=(10, 6))
plt.plot(pes['bond_lengths'], pes['vqe_energies'], 'bo-',
         markersize=10, linewidth=2, label='VQE')
plt.plot(pes['bond_lengths'], pes['exact_energies'], 'r^--',
         markersize=10, linewidth=2, label='Exact (FCI)')
plt.xlabel('Bond Length (Å)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('H₂ Potential Energy Surface', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2_pes.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 6: Parameter Shift Gradient
# ============================================================

def parameter_shift_gradient(params: np.ndarray, hamiltonian: SparsePauliOp,
                             n_qubits: int, n_layers: int,
                             estimator: Estimator) -> np.ndarray:
    """
    Compute gradient using parameter shift rule.
    """
    gradient = np.zeros_like(params)
    shift = np.pi / 2

    for i in range(len(params)):
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += shift

        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= shift

        # Compute energies
        e_plus = vqe_cost(params_plus, hamiltonian, n_qubits, n_layers, estimator)
        e_minus = vqe_cost(params_minus, hamiltonian, n_qubits, n_layers, estimator)

        # Parameter shift gradient
        gradient[i] = (e_plus - e_minus) / 2

    return gradient

# Demonstrate gradient computation
print("\n" + "="*60)
print("Parameter Shift Gradient Demonstration")
print("="*60)

H = get_h2_hamiltonian(0.735)
estimator = Estimator()
test_params = np.random.random(count_parameters(4, 2)) * 0.5

print("Computing gradient at random point...")
gradient = parameter_shift_gradient(test_params, H, 4, 2, estimator)

print(f"Gradient shape: {gradient.shape}")
print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
print(f"Largest gradient component: {np.max(np.abs(gradient)):.6f}")

# Visualize gradient
plt.figure(figsize=(10, 4))
plt.bar(range(len(gradient)), gradient, color='steelblue', alpha=0.7)
plt.xlabel('Parameter Index', fontsize=12)
plt.ylabel('Gradient Value', fontsize=12)
plt.title('VQE Parameter Gradients', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('vqe_gradients.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

### Lab 2: VQE with PennyLane

```python
"""
Day 947 Lab Part 2: VQE with PennyLane
Alternative implementation with automatic differentiation
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Part 1: PennyLane H₂ VQE
# ============================================================

# Define H₂ Hamiltonian
symbols = ['H', 'H']
coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.735])  # Angstroms

# Build molecular Hamiltonian (requires openfermion)
# For simplicity, using manual construction
coeffs = [
    -0.8105, 0.1721, -0.2257, 0.1721, -0.2257,
    0.1209, 0.1689, 0.0454, 0.0454, 0.1689, 0.1209,
    0.0454, 0.0454, 0.0454, 0.0454
]
obs = [
    qml.Identity(0),
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliZ(2),
    qml.PauliZ(3),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliZ(2),
    qml.PauliZ(0) @ qml.PauliZ(3),
    qml.PauliZ(1) @ qml.PauliZ(2),
    qml.PauliZ(1) @ qml.PauliZ(3),
    qml.PauliZ(2) @ qml.PauliZ(3),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
    qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3)
]

H = qml.Hamiltonian(coeffs, obs)
print("H₂ Hamiltonian:")
print(f"Number of terms: {len(H.ops)}")

# ============================================================
# Part 2: Quantum Circuit in PennyLane
# ============================================================

n_qubits = 4
n_layers = 2
dev = qml.device('default.qubit', wires=n_qubits)

def ansatz(params, wires):
    """Hardware-efficient ansatz."""
    n_layers = (len(params) // len(wires)) - 1

    param_idx = 0
    for layer in range(n_layers):
        for w in wires:
            qml.RY(params[param_idx], wires=w)
            param_idx += 1
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])

    # Final rotation
    for w in wires:
        qml.RY(params[param_idx], wires=w)
        param_idx += 1

@qml.qnode(dev, interface='autograd')
def cost_fn(params):
    """VQE cost function."""
    ansatz(params, wires=range(n_qubits))
    return qml.expval(H)

# ============================================================
# Part 3: Gradient Descent Optimization
# ============================================================

# Initialize parameters
np.random.seed(42)
n_params = n_qubits * (n_layers + 1)
params = np.random.random(n_params) * 0.1

# Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# Training loop
n_steps = 100
energies = []

print("\nOptimizing VQE with gradient descent...")
for i in range(n_steps):
    params, energy = opt.step_and_cost(cost_fn, params)
    energies.append(energy)

    if (i + 1) % 20 == 0:
        print(f"Step {i+1}: Energy = {energy:.6f} Ha")

print(f"\nFinal energy: {energies[-1]:.6f} Ha")
print(f"Exact energy: -1.1373 Ha")
print(f"Error: {abs(energies[-1] - (-1.1373)):.6f} Ha")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(energies, 'b-', linewidth=2)
plt.axhline(y=-1.1373, color='r', linestyle='--', label='Exact')
plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('PennyLane VQE Convergence', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pennylane_vqe.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: Analyze Gradients
# ============================================================

# Compute gradient at final point
grad_fn = qml.grad(cost_fn)
final_gradient = grad_fn(params)

print("\nGradient Analysis at Optimum:")
print(f"Gradient norm: {np.linalg.norm(final_gradient):.6f}")
print(f"Max |grad|: {np.max(np.abs(final_gradient)):.6f}")

# Check gradient vanishing
if np.linalg.norm(final_gradient) < 0.01:
    print("Gradient is nearly zero - local minimum reached!")

# ============================================================
# Part 5: Different Optimizers Comparison
# ============================================================

def run_with_optimizer(opt, n_steps=80):
    """Run VQE with specified optimizer."""
    np.random.seed(42)
    params = np.random.random(n_params) * 0.1
    energies = []

    for _ in range(n_steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energies.append(energy)

    return energies

optimizers = {
    'GD (η=0.4)': qml.GradientDescentOptimizer(0.4),
    'Adam (η=0.1)': qml.AdamOptimizer(0.1),
    'Momentum (η=0.2)': qml.MomentumOptimizer(0.2)
}

plt.figure(figsize=(10, 6))
for name, opt in optimizers.items():
    energies = run_with_optimizer(opt)
    plt.plot(energies, linewidth=2, label=name)

plt.axhline(y=-1.1373, color='k', linestyle='--', alpha=0.5, label='Exact')
plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('VQE Optimizer Comparison', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Variational Principle | $E_0 \leq \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle$ |
| VQE Energy | $E(\boldsymbol{\theta}) = \sum_i c_i \langle\hat{P}_i\rangle$ |
| Parameter Shift | $\partial_\theta\langle\hat{O}\rangle = \frac{1}{2}(\langle\hat{O}\rangle_{\theta+\pi/2} - \langle\hat{O}\rangle_{\theta-\pi/2})$ |
| Shot Noise | $\sigma_E = \sqrt{\sum_i |c_i|^2(1 - \langle\hat{P}_i\rangle^2)/N}$ |
| HEA Parameters | $N_{\text{params}} = n \cdot (L + 1)$ for $n$ qubits, $L$ layers |

### Key Takeaways

1. **VQE uses the variational principle** to approximate ground state energies on quantum hardware.

2. **Ansatz design is critical** - hardware-efficient ansatzes match native gates, while UCC ansatzes capture physical structure.

3. **The parameter shift rule** enables exact gradient computation with only 2 circuit evaluations per parameter.

4. **Hamiltonian decomposition** into Pauli strings allows term-by-term measurement.

5. **Classical optimizers** complete the hybrid loop - gradient-free methods (COBYLA, SPSA) are often more robust to noise.

6. **Shot noise limits precision** - achieving chemical accuracy requires careful shot budgeting.

## Daily Checklist

- [ ] I understand the variational principle and its application to VQE
- [ ] I can design hardware-efficient and problem-inspired ansatzes
- [ ] I can apply the parameter shift rule for gradient computation
- [ ] I understand Hamiltonian decomposition into Pauli strings
- [ ] I implemented VQE for H₂ and analyzed convergence
- [ ] I can compare different classical optimizers for VQE
- [ ] I understand shot noise and its impact on energy estimation

## Preview of Day 948

Tomorrow we explore the **Quantum Approximate Optimization Algorithm (QAOA)** for combinatorial optimization:
- Formulating optimization problems as cost Hamiltonians
- Designing mixer Hamiltonians for constraint satisfaction
- Understanding depth scaling and approximation ratios
- Implementing QAOA for MaxCut on random graphs
- Analyzing the relationship between circuit depth and solution quality

QAOA represents VQE's cousin for discrete optimization, demonstrating how variational principles extend beyond quantum chemistry.
