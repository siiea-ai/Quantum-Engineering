# Day 550: Negativity

## Overview
**Day 550** | Week 79, Day 4 | Year 1, Month 20 | Partial Transpose Entanglement Measure

Today we study negativity—an entanglement measure based on the partial transpose operation that can detect bound entanglement and works for arbitrary dimensional systems beyond the two-qubit case.

---

## Learning Objectives
1. Define partial transpose and compute it explicitly
2. Understand the PPT criterion for separability
3. Define negativity and logarithmic negativity
4. Compute negativity for common states
5. Connect negativity to distillability
6. Implement negativity calculations numerically

---

## Core Content

### Partial Transpose Operation

For a bipartite state $\rho_{AB}$, the **partial transpose** with respect to B is:

$$\boxed{\rho^{T_B} = (I_A \otimes T_B)(\rho_{AB})}$$

In matrix element notation:
$$\langle ij|\rho^{T_B}|kl\rangle = \langle il|\rho|kj\rangle$$

The indices of subsystem B are transposed while A remains unchanged.

### Partial Transpose Matrix

For a 2-qubit density matrix in computational basis:
$$\rho = \begin{pmatrix} \rho_{00,00} & \rho_{00,01} & \rho_{00,10} & \rho_{00,11} \\ \rho_{01,00} & \rho_{01,01} & \rho_{01,10} & \rho_{01,11} \\ \rho_{10,00} & \rho_{10,01} & \rho_{10,10} & \rho_{10,11} \\ \rho_{11,00} & \rho_{11,01} & \rho_{11,10} & \rho_{11,11} \end{pmatrix}$$

The partial transpose $\rho^{T_B}$ swaps B indices (second index in each pair):
$$\rho^{T_B} = \begin{pmatrix} \rho_{00,00} & \rho_{01,00} & \rho_{00,10} & \rho_{01,10} \\ \rho_{00,01} & \rho_{01,01} & \rho_{00,11} & \rho_{01,11} \\ \rho_{10,00} & \rho_{11,00} & \rho_{10,10} & \rho_{11,10} \\ \rho_{10,01} & \rho_{11,01} & \rho_{10,11} & \rho_{11,11} \end{pmatrix}$$

### PPT Criterion (Peres-Horodecki)

**Theorem (Peres, 1996):** If $\rho_{AB}$ is separable, then $\rho^{T_B} \geq 0$ (positive semidefinite).

$$\boxed{\rho \text{ separable} \Rightarrow \rho^{T_B} \geq 0 \text{ (PPT)}}$$

**Contrapositive:** If $\rho^{T_B}$ has negative eigenvalues, $\rho$ is entangled!

### PPT Sufficiency

**Theorem (Horodecki³, 1996):** For $2 \times 2$ and $2 \times 3$ systems:
$$\rho^{T_B} \geq 0 \iff \rho \text{ separable}$$

For higher dimensions: PPT is necessary but NOT sufficient for separability (bound entangled states exist).

### Negativity Definition

$$\boxed{\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2}}$$

where $\|A\|_1 = \text{Tr}\sqrt{A^\dagger A}$ is the **trace norm**.

**Equivalent formula:**
$$\mathcal{N}(\rho) = \sum_{\lambda_i < 0} |\lambda_i|$$

Sum of absolute values of negative eigenvalues of $\rho^{T_B}$.

### Trace Norm

For Hermitian matrix with eigenvalues $\{\lambda_i\}$:
$$\|A\|_1 = \sum_i |\lambda_i|$$

For valid density matrix $\rho$: $\|\rho\|_1 = 1$

For partial transpose: $\|\rho^{T_B}\|_1 = 1 + 2\mathcal{N}(\rho)$

### Logarithmic Negativity

$$\boxed{E_\mathcal{N}(\rho) = \log_2 \|\rho^{T_B}\|_1 = \log_2(1 + 2\mathcal{N})}$$

**Properties:**
- $E_\mathcal{N} \geq 0$ always
- $E_\mathcal{N} = 0$ for PPT states
- Upper bound on distillable entanglement: $E_D \leq E_\mathcal{N}$

### Negativity Properties

**1. Non-negative:** $\mathcal{N}(\rho) \geq 0$

**2. Zero for PPT states:** $\mathcal{N}(\rho) = 0 \iff \rho^{T_B} \geq 0$

**3. Convexity:** $\mathcal{N}(\sum_i p_i \rho_i) \leq \sum_i p_i \mathcal{N}(\rho_i)$

**4. LOCC monotone:** Cannot increase under LOCC

**5. Faithful for 2×2 and 2×3:** $\mathcal{N} = 0 \iff$ separable

### Bell State Negativity

For $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$\rho^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Eigenvalues of $\rho^{T_B}$: $\{1/2, 1/2, 1/2, -1/2\}$

$$\mathcal{N}(|\Phi^+\rangle) = |-1/2| = 0.5$$
$$E_\mathcal{N} = \log_2(1 + 2 \cdot 0.5) = \log_2 2 = 1 \text{ ebit}$$

### Werner State Negativity

For Werner state $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$:

$$\mathcal{N}(\rho_W) = \max\left(0, \frac{p - 1/2}{2}\right)$$

**PPT threshold:** $p = 1/2$

Compare with concurrence threshold $p = 1/3$: Different thresholds!

### Bound Entanglement and PPT

**PPT entangled states** (bound entangled):
- $\rho^{T_B} \geq 0$ but $\rho$ is entangled
- $\mathcal{N}(\rho) = 0$ but not separable
- Exist in dimensions $\geq 3 \times 3$
- Cannot be distilled to Bell pairs

**Example:** Horodecki 3×3 bound entangled state.

### Negativity vs Concurrence

| Property | Negativity | Concurrence |
|----------|------------|-------------|
| Dimensions | Any | 2 qubits only |
| Computation | SVD | Special formula |
| PPT bound states | Fails to detect | N/A |
| Distillability | Upper bounds E_D | Related to E_F |

### Realignment Criterion

Alternative to PPT: the **realignment** (or computable cross-norm) criterion.

For $\rho = \sum_{ijkl} \rho_{ij,kl} |i\rangle\langle j| \otimes |k\rangle\langle l|$:

Define realigned matrix: $R(\rho)_{ik,jl} = \rho_{ij,kl}$

If $\|R(\rho)\|_1 > 1$, state is entangled.

Can detect some PPT entangled states!

---

## Worked Examples

### Example 1: Computing Partial Transpose
Find $\rho^{T_B}$ for the Bell state $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$.

**Solution:**
$$\rho = |\Psi^+\rangle\langle\Psi^+| = \frac{1}{2}(|01\rangle + |10\rangle)(\langle 01| + \langle 10|)$$
$$= \frac{1}{2}(|01\rangle\langle 01| + |01\rangle\langle 10| + |10\rangle\langle 01| + |10\rangle\langle 10|)$$

In matrix form (basis ordering $|00\rangle, |01\rangle, |10\rangle, |11\rangle$):
$$\rho = \frac{1}{2}\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Applying partial transpose (swap B indices):
$$\rho^{T_B} = \frac{1}{2}\begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}$$

Eigenvalues: $\{1/2, 1/2, 1/2, -1/2\}$

$$\mathcal{N} = 1/2$$ ∎

### Example 2: Product State Negativity
Show that product state $|00\rangle$ has zero negativity.

**Solution:**
$$\rho = |00\rangle\langle 00| = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

For product states: $\rho^{T_B} = \rho$ (separable states are PPT).

Eigenvalues of $\rho^{T_B}$: $\{1, 0, 0, 0\}$

No negative eigenvalues → $\mathcal{N} = 0$ ∎

### Example 3: Isotropic State Negativity
Compute $\mathcal{N}$ for isotropic state $\rho_F = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{4}I$ with $F = 0.6$.

**Solution:**
The eigenvalues of $\rho_F^{T_B}$ for isotropic states are:
- $\lambda_1 = \frac{1+F}{4}$ (multiplicity 1)
- $\lambda_2 = \frac{1-F}{4}$ (multiplicity 2)
- $\lambda_3 = \frac{F}{2} - \frac{1}{4}$ (multiplicity 1)

For $F = 0.6$:
- $\lambda_1 = 0.4$
- $\lambda_2 = 0.1$ (×2)
- $\lambda_3 = 0.3 - 0.25 = 0.05$

All eigenvalues positive → PPT at $F = 0.6$? Let me recalculate...

Actually for isotropic state, the negative eigenvalue appears when $F > 1/3$:
$\lambda_3 = \frac{1-3F}{4}$ becomes negative.

For $F = 0.6$: $\lambda_3 = \frac{1-1.8}{4} = -0.2$

$$\mathcal{N} = |-0.2| = 0.2$$ ∎

---

## Practice Problems

### Problem 1: Partial Transpose Computation
Compute $\rho^{T_B}$ and find all eigenvalues for:
(a) $|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$
(b) The mixed state $\rho = 0.7|00\rangle\langle 00| + 0.3|\Phi^+\rangle\langle\Phi^+|$

### Problem 2: Negativity Bounds
Prove that for any two-qubit state: $0 \leq \mathcal{N} \leq 1/2$.

### Problem 3: Logarithmic Negativity
For a state with $\mathcal{N} = 0.3$:
(a) Compute $E_\mathcal{N}$
(b) What does this tell us about distillable entanglement?

### Problem 4: PPT vs Entanglement
The state $\rho = 0.4|\Phi^+\rangle\langle\Phi^+| + 0.6|00\rangle\langle00|$:
(a) Is it PPT?
(b) Is it entangled?
(c) What is its negativity?

### Problem 5: Higher Dimensions
For a qutrit-qutrit state ($3 \times 3$):
(a) What is the maximum possible negativity?
(b) Give an example of a maximally entangled state achieving this.

---

## Computational Lab

```python
"""Day 550: Negativity"""
import numpy as np
from scipy.linalg import sqrtm

def partial_transpose_B(rho, dim_A, dim_B):
    """
    Compute partial transpose with respect to subsystem B.
    ⟨ij|ρᵀᴮ|kl⟩ = ⟨il|ρ|kj⟩
    """
    # Reshape to (dim_A, dim_B, dim_A, dim_B)
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)

    # Transpose the B indices (axes 1 and 3)
    rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))

    # Reshape back to matrix
    return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)

def negativity(rho, dim_A=2, dim_B=2):
    """
    Compute negativity: N(ρ) = (||ρᵀᴮ||₁ - 1) / 2
    Equivalently: sum of absolute values of negative eigenvalues
    """
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho_pt)

    # Sum of negative eigenvalues (absolute value)
    neg = np.sum(np.abs(eigenvalues[eigenvalues < -1e-12]))

    return neg, eigenvalues

def trace_norm(A):
    """Compute trace norm ||A||₁ = Tr(√(A†A))"""
    return np.sum(np.abs(np.linalg.eigvalsh(A)))

def log_negativity(rho, dim_A=2, dim_B=2):
    """
    Logarithmic negativity: E_N = log₂||ρᵀᴮ||₁
    """
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)
    tn = trace_norm(rho_pt)
    return np.log2(tn)

def is_ppt(rho, dim_A=2, dim_B=2, tol=1e-10):
    """Check if state is PPT (positive partial transpose)"""
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.all(eigenvalues >= -tol)

# ============================================
# Test Cases
# ============================================
print("=" * 60)
print("NEGATIVITY CALCULATIONS")
print("=" * 60)

# 1. Bell states
print("\n1. Bell States")
bell_states = {
    'Φ⁺': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'Φ⁻': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    'Ψ⁺': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    'Ψ⁻': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

for name, psi in bell_states.items():
    rho = np.outer(psi, psi.conj())
    N, eigs = negativity(rho)
    E_N = log_negativity(rho)
    print(f"   |{name}⟩: N = {N:.4f}, E_N = {E_N:.4f}")
    print(f"        Eigenvalues of ρᵀᴮ: {np.sort(eigs)[::-1]}")

# 2. Product states (should be PPT with N=0)
print("\n2. Product States (PPT, N=0)")
product_states = {
    '|00⟩': np.array([1, 0, 0, 0], dtype=complex),
    '|++⟩': np.array([1, 1, 1, 1], dtype=complex) / 2,
}

for name, psi in product_states.items():
    rho = np.outer(psi, psi.conj())
    N, eigs = negativity(rho)
    ppt = is_ppt(rho)
    print(f"   {name}: N = {N:.4f}, PPT = {ppt}")

# 3. Werner states
print("\n3. Werner States: ρ_W = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

print("   p       N(num)    N(exact)   PPT?    Entangled?")
print("   " + "-" * 55)

for p in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
    rho_W = p * rho_singlet + (1-p) * I4
    N, _ = negativity(rho_W)
    N_exact = max(0, (p - 0.5) / 2)  # Formula for Werner state
    ppt = is_ppt(rho_W)
    # Werner states entangled iff p > 1/3 (concurrence threshold)
    entangled = "Yes" if p > 1/3 else "No"
    print(f"   {p:.2f}    {N:.4f}    {N_exact:.4f}     {str(ppt):5s}   {entangled}")

print("\n   Note: PPT threshold (p=0.5) differs from entanglement threshold (p=1/3)")
print("   For 1/3 < p < 1/2: Entangled but PPT! (Detectable by concurrence, not negativity)")

# 4. Partial transpose visualization
print("\n4. Partial Transpose of |Φ⁺⟩⟨Φ⁺|")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())

print("   Original ρ:")
print(np.round(rho_bell.real, 3))

rho_pt = partial_transpose_B(rho_bell, 2, 2)
print("\n   ρᵀᴮ (partial transpose):")
print(np.round(rho_pt.real, 3))

# 5. Mixed entangled state
print("\n5. Mixed State: 0.7|Φ⁺⟩⟨Φ⁺| + 0.3|00⟩⟨00|")
rho_00 = np.zeros((4, 4), dtype=complex)
rho_00[0, 0] = 1
rho_mixed = 0.7 * rho_bell + 0.3 * rho_00

N, eigs = negativity(rho_mixed)
E_N = log_negativity(rho_mixed)
ppt = is_ppt(rho_mixed)

print(f"   Negativity: N = {N:.4f}")
print(f"   Log-negativity: E_N = {E_N:.4f}")
print(f"   PPT: {ppt}")
print(f"   ρᵀᴮ eigenvalues: {np.sort(eigs)[::-1]}")

# 6. Higher dimensional systems (qutrits)
print("\n6. Qutrit Systems (3×3)")

# Maximally entangled qutrit state
psi_mes_3 = np.zeros(9, dtype=complex)
psi_mes_3[0] = psi_mes_3[4] = psi_mes_3[8] = 1/np.sqrt(3)
rho_mes_3 = np.outer(psi_mes_3, psi_mes_3.conj())

N_3, eigs_3 = negativity(rho_mes_3, 3, 3)
E_N_3 = log_negativity(rho_mes_3, 3, 3)

print(f"   Maximally entangled qutrit:")
print(f"   N = {N_3:.4f}")
print(f"   E_N = {E_N_3:.4f}")
print(f"   Max for qutrits: log₂(3) = {np.log2(3):.4f}")

# 7. Negativity vs concurrence comparison
print("\n7. Comparing Negativity and Concurrence Thresholds")
print("   State: ρ = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
print()
print("   | p value | Concurrence | Negativity | Entangled? |")
print("   |---------|-------------|------------|------------|")

for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
    rho_W = p * rho_singlet + (1-p) * I4
    N, _ = negativity(rho_W)
    C = max(0, (3*p - 1)/2)  # Concurrence formula
    entangled = "Yes" if C > 0 else "No"
    print(f"   |  {p:.2f}   |    {C:.3f}    |    {N:.3f}   |    {entangled}     |")

# 8. Plot negativity and log-negativity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Negativity vs Werner parameter
p_vals = np.linspace(0, 1, 100)
N_vals = []
E_N_vals = []

for p in p_vals:
    rho_W = p * rho_singlet + (1-p) * I4
    N, _ = negativity(rho_W)
    N_vals.append(N)
    E_N_vals.append(np.log2(1 + 2*N))

axes[0].plot(p_vals, N_vals, 'b-', linewidth=2, label='Negativity N')
axes[0].axvline(x=0.5, color='r', linestyle='--', label='PPT threshold (p=0.5)')
axes[0].axvline(x=1/3, color='g', linestyle='--', label='Entanglement threshold (p=1/3)')
axes[0].set_xlabel('p', fontsize=12)
axes[0].set_ylabel('Negativity', fontsize=12)
axes[0].set_title('Werner State Negativity', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Comparison with concurrence
C_vals = [max(0, (3*p - 1)/2) for p in p_vals]

axes[1].plot(p_vals, C_vals, 'b-', linewidth=2, label='Concurrence')
axes[1].plot(p_vals, [2*n for n in N_vals], 'r-', linewidth=2, label='2×Negativity')
axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=1/3, color='gray', linestyle=':', alpha=0.5)
axes[1].set_xlabel('p', fontsize=12)
axes[1].set_ylabel('Measure value', fontsize=12)
axes[1].set_title('Concurrence vs Negativity (Werner State)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('negativity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n8. Plots saved to 'negativity_analysis.png'")

# 9. Verify trace norm formula
print("\n9. Trace Norm Verification")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())
rho_pt = partial_transpose_B(rho_bell, 2, 2)

tn = trace_norm(rho_pt)
N, _ = negativity(rho_bell)

print(f"   ||ρᵀᴮ||₁ = {tn:.4f}")
print(f"   1 + 2N = {1 + 2*N:.4f}")
print(f"   Match: {np.isclose(tn, 1 + 2*N)}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula | Notes |
|----------|---------|-------|
| Partial transpose | $\langle ij\|\rho^{T_B}\|kl\rangle = \langle il\|\rho\|kj\rangle$ | Transpose B indices |
| Negativity | $\mathcal{N} = (\|\|\rho^{T_B}\|\|_1 - 1)/2$ | Sum of negative eigenvalues |
| Log-negativity | $E_\mathcal{N} = \log_2 \|\|\rho^{T_B}\|\|_1$ | Upper bounds $E_D$ |
| PPT criterion | $\rho$ separable $\Rightarrow \rho^{T_B} \geq 0$ | Necessary condition |
| Bell state | $\mathcal{N} = 0.5$, $E_\mathcal{N} = 1$ | Maximum for 2 qubits |

### Key Takeaways
1. **Partial transpose** provides computable entanglement criterion
2. **PPT** is necessary but not sufficient for separability (in high dimensions)
3. **Negativity = 0** does not guarantee separability (bound entanglement)
4. **Log-negativity** upper bounds distillable entanglement
5. **Different thresholds**: Werner state PPT at $p=0.5$, entangled at $p>1/3$
6. **Works in any dimension**, unlike concurrence

---

## Daily Checklist

- [ ] I can compute the partial transpose explicitly
- [ ] I understand the PPT criterion and its limitations
- [ ] I can calculate negativity from eigenvalues
- [ ] I understand the difference between negativity and concurrence
- [ ] I know what bound entanglement means
- [ ] I can interpret log-negativity as an upper bound on E_D

---

*Next: Day 551 — Entanglement of Formation*
