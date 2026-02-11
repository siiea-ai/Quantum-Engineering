# Day 640: QAOA Formulation

## Overview
**Day 640** | Week 92, Day 3 | Year 1, Month 23 | Variational Methods

Today we study the Quantum Approximate Optimization Algorithm (QAOA), designed for combinatorial optimization problems on NISQ devices.

---

## Learning Objectives

1. Formulate combinatorial optimization as QAOA
2. Understand cost and mixer Hamiltonians
3. Construct the QAOA circuit
4. Analyze the p=1 case analytically
5. Apply QAOA to MaxCut
6. Understand the quantum-classical optimization loop

---

## Core Content

### QAOA Overview

**Goal:** Approximate solutions to combinatorial optimization problems.

**Problem:** Maximize $C(z)$ over bitstrings $z \in \{0,1\}^n$

**Quantum encoding:** Cost function as Hamiltonian
$$H_C = \sum_{\text{clauses}} C_{\text{clause}}(Z)$$

### The QAOA Ansatz

$$|\gamma, \beta\rangle = \prod_{p=1}^{P} U_M(\beta_p) U_C(\gamma_p) |+\rangle^{\otimes n}$$

where:
- **Cost unitary:** $U_C(\gamma) = e^{-i\gamma H_C}$
- **Mixer unitary:** $U_M(\beta) = e^{-i\beta H_M}$
- **Standard mixer:** $H_M = \sum_j X_j$

### MaxCut Problem

**Problem:** Partition graph vertices to maximize edges cut.

**Cost function:**
$$C(z) = \sum_{(i,j) \in E} \frac{1}{2}(1 - z_i z_j)$$

**Quantum Hamiltonian:**
$$H_C = \sum_{(i,j) \in E} \frac{1}{2}(I - Z_i Z_j)$$

### QAOA Circuit

```
|+⟩ ─[e^{-iγZ₁Z₂}]─[e^{-iβX}]─[e^{-iγ'Z₁Z₂}]─[e^{-iβ'X}]─ ... ─[Measure]
```

For p layers:
- 2p parameters: $(\gamma_1, \beta_1, ..., \gamma_p, \beta_p)$
- Circuit depth: $O(pm)$ where $m$ = number of edges

### p=1 QAOA Analysis

For MaxCut on triangle:

$$\langle C \rangle_{p=1} = \frac{3}{2} + \frac{1}{4}(\sin 4\beta \sin \gamma)(1 + \cos^2\gamma)$$

Maximum at $\gamma^* \approx 0.6155$, $\beta^* = \pi/8$

Approximation ratio: $\approx 0.6924$ (vs optimal 1)

### QAOA Performance

| Graph | p=1 ratio | p→∞ |
|-------|-----------|-----|
| Triangle | 0.69 | 1.0 |
| 3-regular | 0.69 | 1.0 |
| General | ≥0.5 | 1.0 |

As $p \to \infty$, QAOA approaches adiabatic algorithm (exact solution).

---

## Computational Lab

```python
"""Day 640: QAOA for MaxCut"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def maxcut_hamiltonian(edges, n_qubits):
    """Build MaxCut cost Hamiltonian."""
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    H = np.zeros((2**n_qubits, 2**n_qubits))

    for i, j in edges:
        # (1 - Z_i Z_j) / 2
        term = np.eye(2**n_qubits)
        for k in range(n_qubits):
            if k == i or k == j:
                term = np.kron(term if k == 0 else Z, Z if k == i else (Z if k == j else I)) if k > 0 else Z
        # Rebuild properly
        ZiZj = np.eye(1)
        for k in range(n_qubits):
            if k == i or k == j:
                ZiZj = np.kron(ZiZj, Z)
            else:
                ZiZj = np.kron(ZiZj, I)

        H += 0.5 * (np.eye(2**n_qubits) - ZiZj)

    return H

def mixer_hamiltonian(n_qubits):
    """Build mixer Hamiltonian sum of X."""
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)

    H = np.zeros((2**n_qubits, 2**n_qubits))

    for q in range(n_qubits):
        term = np.eye(1)
        for k in range(n_qubits):
            if k == q:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += term

    return H

def qaoa_circuit(params, H_C, H_M, p):
    """Simulate QAOA circuit."""
    n = int(np.log2(H_C.shape[0]))

    # Initial |+⟩^n state
    state = np.ones(2**n) / np.sqrt(2**n)

    for layer in range(p):
        gamma = params[2*layer]
        beta = params[2*layer + 1]

        # Cost unitary
        U_C = np.linalg.matrix_power(
            np.diag(np.exp(-1j * gamma * np.diag(H_C))), 1
        )
        state = U_C @ state

        # Mixer unitary (via matrix exponential)
        from scipy.linalg import expm
        U_M = expm(-1j * beta * H_M)
        state = U_M @ state

    return state

def qaoa_expectation(params, H_C, H_M, p):
    """QAOA expectation value of cost."""
    state = qaoa_circuit(params, H_C, H_M, p)
    return np.real(state.conj() @ H_C @ state)

# Triangle graph MaxCut
edges = [(0, 1), (1, 2), (0, 2)]
n = 3

H_C = maxcut_hamiltonian(edges, n)
H_M = mixer_hamiltonian(n)

# Optimize p=1 QAOA
p = 1
result = minimize(
    lambda x: -qaoa_expectation(x, H_C, H_M, p),
    np.random.randn(2*p) * 0.1,
    method='COBYLA'
)

print(f"Triangle MaxCut with p={p}:")
print(f"  Optimal params: γ={result.x[0]:.4f}, β={result.x[1]:.4f}")
print(f"  Expected cut: {-result.fun:.4f}")
print(f"  Optimal (exact): {1.5}")
print(f"  Approximation ratio: {-result.fun / 1.5:.4f}")

# Landscape visualization
gamma_range = np.linspace(0, np.pi, 50)
beta_range = np.linspace(0, np.pi/2, 50)
landscape = np.zeros((50, 50))

for i, g in enumerate(gamma_range):
    for j, b in enumerate(beta_range):
        landscape[j, i] = qaoa_expectation([g, b], H_C, H_M, 1)

plt.figure(figsize=(8, 6))
plt.imshow(landscape, extent=[0, np.pi, 0, np.pi/2], origin='lower',
           aspect='auto', cmap='viridis')
plt.colorbar(label='⟨C⟩')
plt.xlabel('γ', fontsize=12)
plt.ylabel('β', fontsize=12)
plt.title('QAOA p=1 Landscape for Triangle MaxCut', fontsize=14)
plt.savefig('qaoa_landscape.png', dpi=150)
plt.show()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| QAOA ansatz | $\|\gamma,\beta\rangle = \prod_p U_M(\beta_p)U_C(\gamma_p)\|+\rangle^n$ |
| Cost unitary | $U_C(\gamma) = e^{-i\gamma H_C}$ |
| Mixer unitary | $U_M(\beta) = e^{-i\beta \sum X}$ |
| MaxCut cost | $\sum_{ij} (1-Z_iZ_j)/2$ |

### Key Takeaways

1. **QAOA** solves combinatorial optimization
2. **Cost Hamiltonian** encodes objective function
3. **Mixer** explores solution space
4. **Approximation ratio** improves with p
5. **p=1** already gives non-trivial approximation

---

## Daily Checklist

- [ ] I understand QAOA structure
- [ ] I can formulate MaxCut as QAOA
- [ ] I know cost and mixer Hamiltonians
- [ ] I can analyze p=1 case
- [ ] I ran the computational lab

---

*Next: Day 641 — Parameterized Circuits*
