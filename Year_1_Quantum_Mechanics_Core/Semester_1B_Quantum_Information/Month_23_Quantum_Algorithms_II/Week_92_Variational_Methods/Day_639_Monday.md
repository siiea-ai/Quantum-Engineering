# Day 639: VQE Basics

## Overview
**Day 639** | Week 92, Day 2 | Year 1, Month 23 | Variational Methods

Today we learn the Variational Quantum Eigensolver (VQE), the most well-known NISQ algorithm for finding ground state energies of quantum systems.

---

## Learning Objectives

1. Understand the variational principle
2. Decompose Hamiltonians into Pauli terms
3. Design ansatz circuits
4. Implement VQE optimization loop
5. Analyze VQE for simple molecules
6. Understand measurement strategies

---

## Core Content

### The Variational Principle

For any trial state $|\psi\rangle$:
$$\boxed{E_0 \leq \langle\psi|H|\psi\rangle}$$

Equality holds when $|\psi\rangle$ is the ground state.

**VQE Strategy:** Minimize $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$ over parameters $\theta$.

### Hamiltonian Decomposition

Any Hamiltonian can be written as:
$$H = \sum_i c_i P_i$$

where $P_i$ are Pauli strings (e.g., $X_1 Z_2 I_3$).

**Expectation value:**
$$\langle H \rangle = \sum_i c_i \langle P_i \rangle$$

Each $\langle P_i \rangle$ measured separately on quantum device.

### Ansatz Design

**Hardware-efficient ansatz:** Alternating single-qubit rotations and entangling gates.

```
|0⟩ ─Ry(θ₁)─●─Ry(θ₄)─●─
            │        │
|0⟩ ─Ry(θ₂)─X─Ry(θ₅)─┼─
                     │
|0⟩ ─Ry(θ₃)──────────X─
```

**Chemistry-inspired ansatz:** UCCSD (Unitary Coupled Cluster)
$$U = e^{T - T^\dagger}$$

where $T$ includes single and double excitations.

### VQE Algorithm

```
1. Initialize parameters θ randomly
2. Repeat until converged:
   a. Prepare |ψ(θ)⟩ on quantum device
   b. Measure ⟨P_i⟩ for all Pauli terms
   c. Compute E(θ) = Σ c_i ⟨P_i⟩
   d. Update θ using classical optimizer
3. Return E_min and optimal θ
```

### Example: H₂ Molecule

Simplest VQE application: Hydrogen molecule in minimal basis.

**Hamiltonian (STO-3G basis):**
$$H = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0 Z_1 + g_4 X_0 X_1 + g_5 Y_0 Y_1$$

**2-qubit ansatz:**
$$|\psi(\theta)\rangle = R_y(\theta)|01\rangle$$

**Energy landscape:** Has clear minimum at equilibrium bond length.

---

## Worked Examples

### Example 1: Simple VQE
For $H = Z$, find ground state using VQE with ansatz $R_y(\theta)|0\rangle$.

**Solution:**
$|\psi(\theta)\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$

$E(\theta) = \cos^2(\theta/2) - \sin^2(\theta/2) = \cos\theta$

Minimum at $\theta = \pi$: $E_{min} = -1$ ✓

Ground state: $|1\rangle$

### Example 2: Two-Qubit VQE
For $H = Z_0 + Z_1 + 0.5 X_0 X_1$, sketch the VQE approach.

**Solution:**
1. Use 2-qubit ansatz with entangling gate
2. Measure $\langle Z_0 \rangle$, $\langle Z_1 \rangle$, $\langle X_0 X_1 \rangle$ separately
3. Combine: $E = \langle Z_0 \rangle + \langle Z_1 \rangle + 0.5\langle X_0 X_1 \rangle$
4. Optimize over parameters

---

## Computational Lab

```python
"""Day 639: VQE Basics"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def ry_gate(theta):
    """Y-rotation gate."""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

def cnot():
    """CNOT gate."""
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])

def hardware_efficient_ansatz(params, n_qubits=2, depth=1):
    """Simple hardware-efficient ansatz."""
    n_params = n_qubits * (depth + 1)
    assert len(params) >= n_params

    # Initialize |00⟩
    state = np.zeros(2**n_qubits)
    state[0] = 1

    idx = 0
    for d in range(depth + 1):
        # Single-qubit rotations
        U = np.eye(1)
        for q in range(n_qubits):
            U = np.kron(U, ry_gate(params[idx]))
            idx += 1

        state = U @ state

        # Entangling layer (if not last layer)
        if d < depth and n_qubits >= 2:
            CNOT = cnot()
            state = CNOT @ state

    return state

def vqe_energy(params, H):
    """Compute VQE energy."""
    state = hardware_efficient_ansatz(params)
    return np.real(state.conj() @ H @ state)

def run_vqe(H, n_params=4, method='COBYLA'):
    """Run VQE optimization."""
    initial_params = np.random.randn(n_params) * 0.1

    result = minimize(
        lambda p: vqe_energy(p, H),
        initial_params,
        method=method
    )

    return result.fun, result.x

# Example: H = Z_0 + Z_1 + 0.5 X_0 X_1
H = np.kron(Z, I) + np.kron(I, Z) + 0.5 * np.kron(X, X)

# Find exact ground state
eigvals, eigvecs = np.linalg.eigh(H)
E_exact = eigvals[0]
print(f"Exact ground state energy: {E_exact:.6f}")

# Run VQE
E_vqe, params_opt = run_vqe(H)
print(f"VQE ground state energy: {E_vqe:.6f}")
print(f"Error: {abs(E_vqe - E_exact):.6f}")

# Energy landscape for 1-qubit case
theta = np.linspace(0, 2*np.pi, 100)
energies = [np.cos(t) for t in theta]

plt.figure(figsize=(10, 5))
plt.plot(theta, energies, 'b-', linewidth=2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('E(θ) = ⟨Z⟩', fontsize=12)
plt.title('VQE Energy Landscape for H = Z', fontsize=14)
plt.axhline(y=-1, color='red', linestyle='--', label='Ground state')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vqe_landscape.png', dpi=150)
plt.show()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Variational principle | $E_0 \leq \langle\psi(\theta)\|H\|\psi(\theta)\rangle$ |
| Hamiltonian decomposition | $H = \sum_i c_i P_i$ |
| Energy measurement | $E = \sum_i c_i \langle P_i \rangle$ |

### Key Takeaways

1. **VQE uses** variational principle to find ground states
2. **Hamiltonian** decomposed into Pauli terms
3. **Ansatz** is parameterized quantum circuit
4. **Classical optimizer** updates parameters
5. **Chemistry** is primary application
6. **Measurement** dominates runtime

---

## Daily Checklist

- [ ] I understand the variational principle
- [ ] I can decompose Hamiltonians into Paulis
- [ ] I know how to design ansatz circuits
- [ ] I can implement VQE loop
- [ ] I ran the computational lab

---

*Next: Day 640 — QAOA Formulation*
