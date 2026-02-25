# Day 547: Von Neumann Entropy

## Overview
**Day 547** | Week 79, Day 1 | Year 1, Month 20 | Information-Theoretic Foundation

Today we study the von Neumann entropy—the quantum generalization of Shannon entropy that underlies all information-theoretic entanglement measures.

---

## Learning Objectives
1. Define von Neumann entropy for quantum states
2. Prove key properties (concavity, subadditivity)
3. Compute entropy for common states
4. Connect to classical Shannon entropy
5. Understand entropy in quantum information theory
6. Apply to entanglement quantification

---

## Core Content

### Von Neumann Entropy Definition

$$\boxed{S(\rho) = -\text{Tr}(\rho \log_2 \rho)}$$

In terms of eigenvalues $\{\lambda_i\}$ of ρ:
$$S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

Convention: $0 \log 0 = 0$

### Properties

**1. Non-negativity:** $S(\rho) \geq 0$

**2. Bounds:** $0 \leq S(\rho) \leq \log_2 d$
- $S = 0$ iff ρ is pure
- $S = \log_2 d$ iff $\rho = I/d$ (maximally mixed)

**3. Unitary invariance:** $S(U\rho U^\dagger) = S(\rho)$

**4. Concavity:** $S(\sum_i p_i \rho_i) \geq \sum_i p_i S(\rho_i)$

**5. Subadditivity:** $S(\rho_{AB}) \leq S(\rho_A) + S(\rho_B)$

**6. Strong subadditivity:** $S(\rho_{ABC}) + S(\rho_B) \leq S(\rho_{AB}) + S(\rho_{BC})$

### Connection to Classical Entropy

**Shannon entropy:** $H(X) = -\sum_i p_i \log_2 p_i$

Von Neumann entropy = Shannon entropy of eigenvalue spectrum

### Examples

**Pure state:** $S(|\psi\rangle\langle\psi|) = 0$

**Maximally mixed (qubit):** $S(I/2) = 1$ bit

**Bell state (global):** $S(|\Phi^+\rangle\langle\Phi^+|) = 0$

**Bell state (reduced):** $S(I/2) = 1$ bit

### Quantum Relative Entropy

$$S(\rho \| \sigma) = \text{Tr}(\rho \log_2 \rho) - \text{Tr}(\rho \log_2 \sigma)$$

**Properties:**
- $S(\rho \| \sigma) \geq 0$ (Klein's inequality)
- $S(\rho \| \sigma) = 0$ iff $\rho = \sigma$

### Mutual Information

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

**For product states:** $I(A:B) = 0$
**For entangled states:** $I(A:B) > 0$

---

## Worked Examples

### Example 1: Qubit Entropy
Compute $S(\rho)$ for $\rho = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.1 \end{pmatrix}$

**Solution:**
First find eigenvalues: $\lambda_{1,2} = \frac{1 \pm \sqrt{0.64 + 0.04}}{2} = \frac{1 \pm 0.825}{2}$

$\lambda_1 = 0.9125$, $\lambda_2 = 0.0875$

$S = -0.9125 \log_2(0.9125) - 0.0875 \log_2(0.0875)$
$= 0.120 + 0.314 = 0.434$ bits ∎

### Example 2: Werner State Entropy
Find $S(\rho_W)$ for Werner state $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$.

**Solution:**
Eigenvalues of Werner state:
- $\lambda_1 = (1+3p)/4$ (multiplicity 1 for singlet component)
- $\lambda_{2,3,4} = (1-p)/4$ (multiplicity 3)

$S(\rho_W) = -\frac{1+3p}{4}\log_2\frac{1+3p}{4} - 3\frac{1-p}{4}\log_2\frac{1-p}{4}$ ∎

### Example 3: Mutual Information
Compute $I(A:B)$ for a Bell state.

**Solution:**
For $|\Phi^+\rangle$:
- $S(\rho_{AB}) = 0$ (pure state)
- $S(\rho_A) = S(I/2) = 1$
- $S(\rho_B) = S(I/2) = 1$

$I(A:B) = 1 + 1 - 0 = 2$ bits ∎

---

## Computational Lab

```python
"""Day 547: Von Neumann Entropy"""
import numpy as np
from scipy.linalg import logm

def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def purity(rho):
    """γ = Tr(ρ²)"""
    return np.trace(rho @ rho).real

# Test states
print("=== Von Neumann Entropy Examples ===\n")

# Pure state
psi = np.array([1, 0])
rho_pure = np.outer(psi, psi.conj())
print(f"Pure |0⟩: S = {von_neumann_entropy(rho_pure):.4f}")

# Maximally mixed
rho_mixed = np.eye(2) / 2
print(f"Maximally mixed I/2: S = {von_neumann_entropy(rho_mixed):.4f}")

# Bell state (global)
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())
print(f"Bell state |Φ⁺⟩: S = {von_neumann_entropy(rho_bell):.4f}")

# Bell state (reduced)
rho_A = rho_bell.reshape(2,2,2,2).trace(axis1=1, axis2=3)
print(f"Bell reduced ρ_A: S = {von_neumann_entropy(rho_A):.4f}")

# Werner states
print("\n=== Werner State Entropy ===")
psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rho_W = p * rho_singlet + (1-p) * I4
    S = von_neumann_entropy(rho_W)
    print(f"p = {p:.2f}: S = {S:.4f}")

# Mutual information
print("\n=== Mutual Information ===")

def mutual_info(rho_AB, dim_A=2, dim_B=2):
    rho_A = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=1, axis2=3)
    rho_B = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=0, axis2=2)
    return von_neumann_entropy(rho_A) + von_neumann_entropy(rho_B) - von_neumann_entropy(rho_AB)

print(f"Bell state I(A:B) = {mutual_info(rho_bell):.4f}")

# Product state
prod = np.kron(np.outer([1,0], [1,0]), np.outer([1,0], [1,0]))
print(f"Product |00⟩ I(A:B) = {mutual_info(prod):.4f}")
```

---

## Summary

### Key Formulas

| Quantity | Formula | Range |
|----------|---------|-------|
| Von Neumann entropy | $S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i$ | $[0, \log_2 d]$ |
| Pure state | $S = 0$ | — |
| Max mixed | $S = \log_2 d$ | — |
| Mutual info | $I(A:B) = S_A + S_B - S_{AB}$ | $[0, 2\min(S_A, S_B)]$ |

### Key Takeaways
1. **Von Neumann entropy** measures mixedness of quantum states
2. **Pure states** have zero entropy
3. **Reduced states** of entangled pure states have non-zero entropy
4. **Mutual information** quantifies total correlations
5. **Properties:** non-negative, concave, subadditive

---

*Next: Day 548 — Entropy of Entanglement*
