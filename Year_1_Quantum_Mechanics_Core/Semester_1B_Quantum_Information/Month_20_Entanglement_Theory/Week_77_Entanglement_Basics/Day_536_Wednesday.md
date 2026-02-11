# Day 536: PPT Criterion Deep Dive

## Overview
**Day 536** | Week 77, Day 4 | Year 1, Month 20 | The Peres-Horodecki Criterion

Today we explore the Positive Partial Transpose (PPT) criterion—the most powerful computable test for entanglement in low dimensions.

---

## Learning Objectives
1. Define partial transpose operation precisely
2. State the Peres-Horodecki theorem
3. Apply PPT criterion to various states
4. Understand when PPT is sufficient vs necessary
5. Compute partial transpose for density matrices
6. Connect PPT to entanglement witnesses

---

## Core Content

### Partial Transpose

For a bipartite density matrix ρ_AB, the **partial transpose** with respect to B is:

$$\boxed{(\rho^{T_B})_{ij,kl} = \rho_{il,kj}}$$

In computational basis:
$$|ij\rangle\langle kl| \xrightarrow{T_B} |il\rangle\langle kj|$$

**Matrix block form:** If $\rho = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$ in 2×2 blocks:
$$\rho^{T_B} = \begin{pmatrix} A^T & C^T \\ B^T & D^T \end{pmatrix}$$

### Peres-Horodecki Theorem

**Theorem (Peres, 1996; Horodecki³, 1996):**

$$\rho \text{ separable} \Rightarrow \rho^{T_B} \geq 0 \text{ (positive semidefinite)}$$

**Contrapositive (detection criterion):**
$$\rho^{T_B} \text{ has negative eigenvalue} \Rightarrow \rho \text{ is entangled}$$

### When PPT is Sufficient

**Theorem (Horodecki³):** For $2 \times 2$ and $2 \times 3$ systems:
$$\rho \text{ separable} \Leftrightarrow \rho^{T_B} \geq 0$$

In these dimensions, PPT completely characterizes separability!

### When PPT Fails

For $d_A \times d_B \geq 3 \times 3$: There exist **PPT entangled states** (bound entangled).
$$\rho^{T_B} \geq 0 \text{ but } \rho \text{ entangled}$$

### PPT for Bell States

For $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

$$\rho_{\Phi^+} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Partial transpose:
$$\rho_{\Phi^+}^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Eigenvalues:** $\{1/2, 1/2, 1/2, -1/2\}$

Negative eigenvalue → **Entangled!**

### Werner State PPT Analysis

For $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$:

The partial transpose has minimum eigenvalue:
$$\lambda_{min} = \frac{1-3p}{4}$$

This is negative when $p > 1/3$.

**Transition point:** $p = 1/3$ marks the separable/entangled boundary.

### Negativity from PPT

The **negativity** quantifies PPT violation:
$$\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

where $\|M\|_1 = \text{Tr}\sqrt{M^\dagger M}$ is the trace norm.

For a matrix with eigenvalues $\lambda_i$:
$$\mathcal{N}(\rho) = \sum_{\lambda_i < 0} |\lambda_i|$$

---

## Worked Examples

### Example 1: PPT for Product State
Show that $|00\rangle\langle 00|$ is PPT.

**Solution:**
$$|00\rangle\langle 00| = |0\rangle\langle 0| \otimes |0\rangle\langle 0|$$

Partial transpose on B:
$$(|0\rangle\langle 0| \otimes |0\rangle\langle 0|)^{T_B} = |0\rangle\langle 0| \otimes (|0\rangle\langle 0|)^T = |0\rangle\langle 0| \otimes |0\rangle\langle 0|$$

Same matrix! Eigenvalues: $\{1, 0, 0, 0\}$ — all non-negative. **PPT!** ∎

### Example 2: PPT Threshold for Werner
Find the critical p for Werner state.

**Solution:**
The Werner state density matrix:
$$\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$$

where $|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$.

Matrix form:
$$\rho_W = \frac{1}{4}\begin{pmatrix} 1-p & 0 & 0 & 0 \\ 0 & 1+p & -2p & 0 \\ 0 & -2p & 1+p & 0 \\ 0 & 0 & 0 & 1-p \end{pmatrix}$$

Partial transpose:
$$\rho_W^{T_B} = \frac{1}{4}\begin{pmatrix} 1-p & 0 & 0 & -2p \\ 0 & 1+p & 0 & 0 \\ 0 & 0 & 1+p & 0 \\ -2p & 0 & 0 & 1-p \end{pmatrix}$$

Eigenvalues: $\frac{1+p}{4}$ (double), $\frac{1-p+2p}{4} = \frac{1+p}{4}$, $\frac{1-p-2p}{4} = \frac{1-3p}{4}$

Minimum eigenvalue: $\frac{1-3p}{4} < 0$ when $p > 1/3$. ∎

### Example 3: Isotropic State
The isotropic state: $\rho_F = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\frac{I-|\Phi^+\rangle\langle\Phi^+|}{3}$

Simplifies to: $\rho_F = \frac{4F-1}{3}|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{3}I$

PPT boundary: $F = 1/2$.

---

## Practice Problems

### Problem 1: PPT Calculation
Compute $\rho^{T_B}$ for $\rho = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$.

### Problem 2: Negativity
Calculate the negativity of $|\Phi^+\rangle$.

### Problem 3: Mixture
For $\rho = \alpha|\Phi^+\rangle\langle\Phi^+| + (1-\alpha)|00\rangle\langle 00|$, find the PPT threshold.

---

## Computational Lab

```python
"""Day 536: PPT Criterion Deep Dive"""
import numpy as np
from scipy.linalg import eigvalsh

def partial_transpose_B(rho, dim_A=2, dim_B=2):
    """
    Compute partial transpose over subsystem B.
    ρ_{ij,kl}^{T_B} = ρ_{il,kj}
    """
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    # Swap indices 1 and 3 (B's bra and ket)
    rho_TB = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_TB.reshape(dim_A * dim_B, dim_A * dim_B)

def is_ppt(rho, dim_A=2, dim_B=2, tol=1e-10):
    """Check if state is PPT (positive partial transpose)"""
    rho_TB = partial_transpose_B(rho, dim_A, dim_B)
    eigenvalues = eigvalsh(rho_TB)
    min_eig = np.min(eigenvalues)
    return min_eig >= -tol, min_eig

def negativity(rho, dim_A=2, dim_B=2):
    """
    Compute negativity: N(ρ) = (||ρ^TB||_1 - 1) / 2
    Equals sum of absolute values of negative eigenvalues.
    """
    rho_TB = partial_transpose_B(rho, dim_A, dim_B)
    eigenvalues = eigvalsh(rho_TB)
    return np.sum(np.abs(eigenvalues[eigenvalues < 0]))

def projector(psi):
    """Create projector |ψ⟩⟨ψ|"""
    return np.outer(psi, psi.conj())

# Bell states
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

print("=== PPT Analysis of Bell States ===\n")

for name, state in [("Φ⁺", phi_plus), ("Ψ⁻", psi_minus)]:
    rho = projector(state)
    rho_TB = partial_transpose_B(rho)

    print(f"|{name}⟩ density matrix:")
    print(np.round(rho.real, 4))

    print(f"\nPartial transpose:")
    print(np.round(rho_TB.real, 4))

    eigenvalues = eigvalsh(rho_TB)
    print(f"\nEigenvalues of ρ^TB: {eigenvalues}")

    is_ppt_flag, min_eig = is_ppt(rho)
    print(f"Is PPT: {is_ppt_flag}, min eigenvalue: {min_eig:.4f}")
    print(f"Negativity: {negativity(rho):.4f}")
    print("-" * 50)

# Werner state analysis
print("\n=== Werner State PPT Analysis ===")
print("ρ_W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4\n")

rho_psi = projector(psi_minus)
I4 = np.eye(4) / 4

p_values = np.linspace(0, 1, 21)
transitions = []

print(f"{'p':<6} {'min_eig':<12} {'negativity':<12} {'status'}")
print("-" * 45)

for p in p_values:
    rho_werner = p * rho_psi + (1-p) * I4
    is_ppt_flag, min_eig = is_ppt(rho_werner)
    neg = negativity(rho_werner)

    status = "separable" if is_ppt_flag else "ENTANGLED"
    print(f"{p:.2f}   {min_eig:+.6f}    {neg:.6f}     {status}")

# Find exact transition
print("\n=== PPT Transition Point ===")
from scipy.optimize import brentq

def min_eigenvalue(p):
    rho = p * rho_psi + (1-p) * I4
    return np.min(eigvalsh(partial_transpose_B(rho)))

p_critical = brentq(min_eigenvalue, 0.3, 0.4)
print(f"Critical p (PPT boundary): {p_critical:.6f}")
print(f"Theoretical value: 1/3 = {1/3:.6f}")

# Product state verification
print("\n=== Product State PPT Check ===")
zero_zero = np.array([1, 0, 0, 0], dtype=complex)
rho_product = projector(zero_zero)
is_ppt_flag, min_eig = is_ppt(rho_product)
print(f"|00⟩⟨00| is PPT: {is_ppt_flag}, min eigenvalue: {min_eig:.4f}")

# Classical mixture
rho_classical = 0.5 * projector(np.array([1,0,0,0])) + 0.5 * projector(np.array([0,0,0,1]))
is_ppt_flag, min_eig = is_ppt(rho_classical)
print(f"0.5|00⟩⟨00|+0.5|11⟩⟨11| is PPT: {is_ppt_flag}")
```

**Expected Output:**
```
=== PPT Analysis of Bell States ===

|Φ⁺⟩ density matrix:
[[0.5 0.  0.  0.5]
 [0.  0.  0.  0. ]
 [0.  0.  0.  0. ]
 [0.5 0.  0.  0.5]]

Partial transpose:
[[0.5 0.  0.  0. ]
 [0.  0.  0.5 0. ]
 [0.  0.5 0.  0. ]
 [0.  0.  0.  0.5]]

Eigenvalues of ρ^TB: [-0.5  0.5  0.5  0.5]
Is PPT: False, min eigenvalue: -0.5000
Negativity: 0.5000
--------------------------------------------------

=== Werner State PPT Analysis ===
p = 0.33: separable
p = 0.35: ENTANGLED

=== PPT Transition Point ===
Critical p (PPT boundary): 0.333333
Theoretical value: 1/3 = 0.333333
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Partial transpose | $(\rho^{T_B})_{ij,kl} = \rho_{il,kj}$ |
| PPT criterion | $\rho$ separable ⇒ $\rho^{T_B} \geq 0$ |
| Negativity | $\mathcal{N}(\rho) = \sum_{\lambda_i < 0} \|\lambda_i\|$ |
| Werner threshold | $p > 1/3$ entangled |

### Key Takeaways
1. **PPT** is necessary for separability in all dimensions
2. **PPT is sufficient** only for 2×2 and 2×3 systems
3. **Negativity** quantifies entanglement detected by PPT
4. **Bell states** have negativity 0.5 (maximal for qubits)
5. **Computational**: PPT requires only eigenvalue computation

---

## Daily Checklist

- [ ] I can compute partial transpose of any density matrix
- [ ] I understand the Peres-Horodecki theorem
- [ ] I can apply PPT criterion to detect entanglement
- [ ] I know when PPT is sufficient vs only necessary
- [ ] I can compute negativity from eigenvalues

---

*Next: Day 537 — Bound Entanglement*
