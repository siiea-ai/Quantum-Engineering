# Day 553: Week Review - Entanglement Measures

## Overview
**Day 553** | Week 79, Day 7 | Year 1, Month 20 | Comprehensive Review

Today we consolidate our understanding of entanglement measures, comparing their properties, use cases, and relationships. We work through comprehensive problems that integrate all the measures studied this week.

---

## Learning Objectives
1. Compare and contrast all entanglement measures
2. Understand when to use which measure
3. Master the mathematical relationships between measures
4. Apply multiple measures to the same state
5. Solve advanced problems integrating all concepts
6. Prepare for next week's entanglement dynamics

---

## Core Content Review

### Measures Overview

| Measure | Symbol | Applicability | Computability |
|---------|--------|---------------|---------------|
| Von Neumann Entropy | $S(\rho)$ | Any state | Exact (eigenvalues) |
| Entropy of Entanglement | $E$ | Pure bipartite | Exact (Schmidt) |
| Concurrence | $C$ | 2-qubit | Exact (Wootters) |
| Negativity | $\mathcal{N}$ | Any bipartite | Exact (partial transpose) |
| Log-negativity | $E_\mathcal{N}$ | Any bipartite | Exact |
| Entanglement of Formation | $E_F$ | Any | 2-qubit exact, else hard |
| Distillable Entanglement | $E_D$ | Any | Generally hard (asymptotic) |
| Entanglement Cost | $E_C$ | Any | Generally hard (asymptotic) |

### Key Formulas Summary

$$\boxed{\text{Entropy of Entanglement: } E(|\psi\rangle) = S(\rho_A) = -\sum_i \lambda_i \log_2 \lambda_i}$$

$$\boxed{\text{Concurrence: } C = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)}$$

$$\boxed{\text{Negativity: } \mathcal{N} = \frac{\|\rho^{T_B}\|_1 - 1}{2} = \sum_{\lambda_i < 0}|\lambda_i|}$$

$$\boxed{\text{E_F (2-qubit): } E_F = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)}$$

### Measure Hierarchy

For any bipartite state:
$$E_D \leq E_\mathcal{N} \leq E_F \leq E_C$$

For pure states:
$$E_D = E = E_\mathcal{N} = E_F = E_C$$

### When to Use Which Measure

| Situation | Recommended Measure | Reason |
|-----------|---------------------|--------|
| Pure bipartite state | Entropy of entanglement $E$ | Unique, meaningful |
| 2-qubit mixed state | Concurrence $C$ | Analytical formula |
| High-dimensional mixed | Negativity $\mathcal{N}$ | Computable |
| Check if distillable | Negativity (NPT test) | Necessary condition |
| Bound entanglement | Both $C$ and $\mathcal{N}$ | Different thresholds |
| Quantum protocols | $E_D$, $E_C$ | Operational meaning |

### Werner State Complete Analysis

For $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$:

| p range | Separable | PPT | Distillable | Bound Entangled |
|---------|-----------|-----|-------------|-----------------|
| $[0, 1/3]$ | Yes | Yes | No | No |
| $(1/3, 1/2]$ | No | Yes | No | **Yes** |
| $(1/2, 1]$ | No | No | Yes | No |

### Critical Thresholds

| Threshold | Value | Physical Meaning |
|-----------|-------|------------------|
| Werner separability | $p = 1/3$ | $C = 0$ threshold |
| Werner PPT | $p = 1/2$ | $\mathcal{N} = 0$ threshold |
| Isotropic separability | $F = 1/(d+1)$ | d-dimensional |
| Isotropic PPT | $F = 1/d$ | d-dimensional |

### Entanglement Detection Flowchart

```
Given state ρ:
│
├─ Is ρ pure? ──Yes──► E = S(ρ_A) gives complete answer
│
└─ No (mixed)
   │
   ├─ Is it 2-qubit? ──Yes──► Compute C (Wootters)
   │                           If C > 0: entangled, E_F from C
   │
   └─ No (higher dim)
      │
      ├─ Compute ρ^{T_B}
      │
      ├─ All eigenvalues ≥ 0? ──Yes──► PPT
      │   │                            May be separable or bound entangled
      │   │                            Use other tests (realignment, etc.)
      │   │
      │   └─ No (NPT) ──► Entangled AND distillable
      │                   N > 0, compute log-negativity
      │
      └─ Compute negativity N, log-negativity E_N
```

### Connections Between Measures

**1. Concurrence ↔ E_F:**
$$E_F = h\left(\frac{1+\sqrt{1-C^2}}{2}\right)$$

**2. Entropy ↔ Schmidt:**
$$E = H(\{\lambda_i\}) = -\sum_i \lambda_i \log_2 \lambda_i$$

**3. Concurrence ↔ Purity (pure states):**
$$C^2 = 2(1 - \text{Tr}(\rho_A^2))$$

**4. Negativity ↔ Trace norm:**
$$\mathcal{N} = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

**5. Log-negativity ↔ E_D:**
$$E_D \leq E_\mathcal{N} = \log_2\|\rho^{T_B}\|_1$$

---

## Comprehensive Problems

### Problem Set A: Pure States

**A1.** For $|\psi\rangle = \frac{1}{\sqrt{5}}|00\rangle + \frac{2}{\sqrt{5}}|11\rangle$:
(a) Find Schmidt coefficients
(b) Compute entropy of entanglement
(c) Compute concurrence
(d) Verify $C^2 = 2(1-\text{Tr}(\rho_A^2))$

**A2.** A three-qubit state $|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$:
(a) Compute $E$ for the A|BC bipartition
(b) Compute $E$ for the AB|C bipartition
(c) Is this state fully tripartite entangled?

**A3.** Compare the states:
- $|\psi_1\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$
- $|\psi_2\rangle = \cos\theta|01\rangle + \sin\theta|10\rangle$

Do they have the same entanglement? Prove it.

### Problem Set B: Mixed States - Two Qubits

**B1.** For $\rho = 0.4|\Phi^+\rangle\langle\Phi^+| + 0.3|\Psi^-\rangle\langle\Psi^-| + 0.3|00\rangle\langle00|$:
(a) Write the explicit density matrix
(b) Compute concurrence
(c) Compute negativity
(d) Compare $E_F$ and $E_\mathcal{N}$

**B2.** At what value of $p$ does the state $\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|\Psi^-\rangle\langle\Psi^-|$ have $C = 0$? (Hint: it's a pure state mixture!)

**B3.** For Werner state with $p = 0.6$:
(a) Verify the state is NPT
(b) Compute all applicable measures
(c) Estimate bounds on $E_D$

### Problem Set C: Higher Dimensions

**C1.** Qutrit maximally entangled state $|\Phi_3\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)$:
(a) Compute entropy of entanglement
(b) Compute negativity
(c) What is the maximum possible negativity for qutrits?

**C2.** For the isotropic state in $d=3$ dimensions:
$\rho_F = F|\Phi_3\rangle\langle\Phi_3| + \frac{1-F}{9}I_9$

Find the PPT threshold (value of F).

**C3.** Design a 3×3 state that is PPT but might be entangled. (Hint: Use the Horodecki construction)

### Problem Set D: Operational Measures

**D1.** You have 1000 copies of $|\psi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$:
(a) How many Bell pairs can you distill?
(b) How many Bell pairs would you need to create these 1000 copies?
(c) Is the process reversible?

**D2.** For Werner state $\rho_W$ with $p = 0.45$:
(a) Is the state entangled?
(b) Is it distillable?
(c) What type of entanglement does it have?
(d) Can you use this state for quantum teleportation?

**D3.** Explain the "entanglement gap" $E_C - E_D$ in terms of:
(a) Information loss
(b) Thermodynamic irreversibility
(c) Practical implications

---

## Solutions to Selected Problems

### Solution A1
(a) State is already in Schmidt form:
$\lambda_1 = 1/5 = 0.2$, $\lambda_2 = 4/5 = 0.8$

(b) $E = -0.2\log_2(0.2) - 0.8\log_2(0.8) = 0.464 + 0.258 = 0.722$ ebits

(c) $C = 2\sqrt{\lambda_1\lambda_2} = 2\sqrt{0.16} = 0.8$

(d) $\rho_A = \text{diag}(0.2, 0.8)$, $\text{Tr}(\rho_A^2) = 0.04 + 0.64 = 0.68$
$C^2 = 0.64$, $2(1-0.68) = 0.64$ ✓

### Solution B2
For $\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|\Psi^-\rangle\langle\Psi^-|$:

Both $|\Phi^+\rangle$ and $|\Psi^-\rangle$ are maximally entangled, so this is a mixture of Bell states.

For Bell diagonal states: $C = 2\max(\lambda_1, \lambda_2, \lambda_3, \lambda_4) - 1$

Here eigenvalues are $p$ and $1-p$ (with zeros). $C = 2\max(p, 1-p) - 1$

For $C = 0$: $\max(p, 1-p) = 1/2$, which requires $p = 1/2$.

But wait—at $p = 1/2$, both Bell states equally mixed: $C = 2(0.5) - 1 = 0$? Let me recalculate...

Actually for Bell diagonal, $C = 2(\lambda_{\max} - 1/2)$ when $\lambda_{\max} > 1/2$.

At $p = 0.5$: $\lambda_{\max} = 0.5$, so $C = 0$.

### Solution D2
Werner state with $p = 0.45$:

(a) Entangled? $C = \max(0, (3p-1)/2) = \max(0, 0.35/2) = 0.175 > 0$. **Yes, entangled.**

(b) Distillable? Check PPT: $\mathcal{N} = \max(0, (p-0.5)/2) = 0$ since $p < 0.5$. **PPT, not distillable.**

(c) Type: **Bound entangled** (entangled but PPT).

(d) Teleportation? Requires distillable entanglement. **Cannot be used directly for standard teleportation** (fidelity would be limited).

---

## Computational Lab: Comprehensive Analysis

```python
"""Day 553: Week Review - All Entanglement Measures"""
import numpy as np
from scipy.linalg import sqrtm

# ============================================
# Utility Functions
# ============================================

def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace_B(rho, dim_A, dim_B):
    """Trace out subsystem B"""
    return rho.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=1, axis2=3)

def entropy_of_entanglement(psi, dim_A=2, dim_B=2):
    """E for pure state"""
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace_B(rho, dim_A, dim_B)
    return von_neumann_entropy(rho_A)

def concurrence_pure(psi):
    """Concurrence for pure 2-qubit state"""
    c = psi.reshape(2, 2)
    return 2 * np.abs(c[0,0]*c[1,1] - c[0,1]*c[1,0])

def concurrence_mixed(rho):
    """Wootters concurrence for mixed 2-qubit state"""
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)
    rho_tilde = sigma_yy @ rho.conj() @ sigma_yy
    R = rho @ rho_tilde
    eigenvalues = np.linalg.eigvals(R)
    lambdas = np.sqrt(np.abs(eigenvalues.real))
    lambdas = np.sort(lambdas)[::-1]
    C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return max(0, C.real)

def negativity(rho, dim_A=2, dim_B=2):
    """Negativity from partial transpose"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1)).reshape(dim_A*dim_B, dim_A*dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < -1e-12]))

def log_negativity(rho, dim_A=2, dim_B=2):
    """Logarithmic negativity"""
    N = negativity(rho, dim_A, dim_B)
    return np.log2(1 + 2*N)

def entanglement_of_formation(C):
    """E_F from concurrence"""
    if C < 1e-10:
        return 0.0
    x = (1 + np.sqrt(max(0, 1 - C**2))) / 2
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def is_ppt(rho, dim_A=2, dim_B=2, tol=1e-10):
    """Check if state is PPT"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1)).reshape(dim_A*dim_B, dim_A*dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.all(eigenvalues >= -tol)

def purity(rho):
    """γ = Tr(ρ²)"""
    return np.trace(rho @ rho).real

def complete_analysis(rho, name, dim_A=2, dim_B=2, is_pure=False):
    """Perform complete entanglement analysis"""
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS: {name}")
    print('='*60)

    # Basic properties
    S = von_neumann_entropy(rho)
    gamma = purity(rho)
    print(f"\nBasic Properties:")
    print(f"  Von Neumann entropy S(ρ): {S:.4f}")
    print(f"  Purity Tr(ρ²): {gamma:.4f}")
    print(f"  Is pure state: {gamma > 0.999}")

    # Reduced state entropy
    rho_A = partial_trace_B(rho, dim_A, dim_B)
    S_A = von_neumann_entropy(rho_A)
    print(f"  S(ρ_A): {S_A:.4f}")

    if is_pure:
        print(f"\nPure State Measures:")
        print(f"  Entropy of Entanglement E: {S_A:.4f} ebits")

    # Two-qubit specific
    if dim_A == 2 and dim_B == 2:
        C = concurrence_mixed(rho)
        E_F = entanglement_of_formation(C)
        print(f"\nTwo-Qubit Measures:")
        print(f"  Concurrence C: {C:.4f}")
        print(f"  E_F (from C): {E_F:.4f} ebits")

    # General measures
    N = negativity(rho, dim_A, dim_B)
    E_N = log_negativity(rho, dim_A, dim_B)
    ppt = is_ppt(rho, dim_A, dim_B)

    print(f"\nGeneral Measures:")
    print(f"  Negativity N: {N:.4f}")
    print(f"  Log-negativity E_N: {E_N:.4f}")
    print(f"  Is PPT: {ppt}")

    # Diagnosis
    print(f"\nDiagnosis:")
    if dim_A == 2 and dim_B == 2:
        C = concurrence_mixed(rho)
        if C < 1e-10:
            print(f"  → State is SEPARABLE (C = 0)")
        elif ppt:
            print(f"  → State is BOUND ENTANGLED (C > 0, PPT)")
        else:
            print(f"  → State is DISTILLABLE (NPT)")
    else:
        if ppt and N < 1e-10:
            print(f"  → State might be separable (PPT)")
        elif ppt:
            print(f"  → State is PPT, might be bound entangled")
        else:
            print(f"  → State is DISTILLABLE (NPT)")

    return {'S': S, 'S_A': S_A, 'N': N, 'E_N': E_N, 'ppt': ppt}

# ============================================
# Test Suite
# ============================================
print("=" * 60)
print("WEEK 79 REVIEW: COMPREHENSIVE ANALYSIS")
print("=" * 60)

# 1. Bell state
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())
complete_analysis(rho_bell, "Bell State |Φ⁺⟩", is_pure=True)

# 2. Product state
psi_00 = np.array([1, 0, 0, 0], dtype=complex)
rho_00 = np.outer(psi_00, psi_00.conj())
complete_analysis(rho_00, "Product State |00⟩", is_pure=True)

# 3. Partially entangled pure state
psi_partial = np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)], dtype=complex)
rho_partial = np.outer(psi_partial, psi_partial.conj())
complete_analysis(rho_partial, "Partial: √0.8|00⟩ + √0.2|11⟩", is_pure=True)

# 4. Werner states at different p values
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4, dtype=complex) / 4

for p in [0.3, 0.45, 0.7]:
    rho_W = p * rho_singlet + (1-p) * np.eye(4, dtype=complex) / 4
    complete_analysis(rho_W, f"Werner State (p={p})")

# 5. Mixed entangled state
rho_mixed = 0.6 * rho_bell + 0.4 * rho_00
complete_analysis(rho_mixed, "0.6|Φ⁺⟩⟨Φ⁺| + 0.4|00⟩⟨00|")

# 6. Qutrit maximally entangled
print("\n" + "="*60)
print("QUTRIT ANALYSIS (3×3)")
print("="*60)

psi_mes_3 = np.zeros(9, dtype=complex)
psi_mes_3[0] = psi_mes_3[4] = psi_mes_3[8] = 1/np.sqrt(3)
rho_mes_3 = np.outer(psi_mes_3, psi_mes_3.conj())
complete_analysis(rho_mes_3, "Qutrit MES (|00⟩+|11⟩+|22⟩)/√3", dim_A=3, dim_B=3, is_pure=True)

# 7. Summary comparison table
print("\n" + "="*60)
print("SUMMARY COMPARISON TABLE")
print("="*60)

print("\n" + "-"*80)
print(f"{'State':<35} {'C':>8} {'N':>8} {'E_F':>8} {'E_N':>8} {'Type':>12}")
print("-"*80)

test_cases = [
    ("Bell |Φ⁺⟩", rho_bell),
    ("Product |00⟩", rho_00),
    ("Partial (0.8, 0.2)", rho_partial),
    ("Werner p=0.3", 0.3*rho_singlet + 0.7*I4),
    ("Werner p=0.45", 0.45*rho_singlet + 0.55*I4),
    ("Werner p=0.7", 0.7*rho_singlet + 0.3*I4),
    ("0.6 Bell + 0.4 product", rho_mixed),
]

for name, rho in test_cases:
    C = concurrence_mixed(rho)
    N = negativity(rho)
    E_F = entanglement_of_formation(C)
    E_N = log_negativity(rho)

    if C < 1e-10:
        state_type = "Separable"
    elif N < 1e-10:
        state_type = "Bound"
    else:
        state_type = "Distillable"

    print(f"{name:<35} {C:>8.4f} {N:>8.4f} {E_F:>8.4f} {E_N:>8.4f} {state_type:>12}")

print("-"*80)

# 8. Plot: All measures vs Werner parameter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

p_range = np.linspace(0, 1, 200)
C_vals, N_vals, EF_vals, EN_vals = [], [], [], []

for p in p_range:
    rho_W = p * rho_singlet + (1-p) * I4
    C = concurrence_mixed(rho_W)
    N = negativity(rho_W)
    C_vals.append(C)
    N_vals.append(N)
    EF_vals.append(entanglement_of_formation(C))
    EN_vals.append(log_negativity(rho_W))

# Plot 1: Concurrence and Negativity
axes[0,0].plot(p_range, C_vals, 'b-', linewidth=2, label='Concurrence C')
axes[0,0].plot(p_range, [2*n for n in N_vals], 'r--', linewidth=2, label='2×Negativity')
axes[0,0].axvline(x=1/3, color='g', linestyle=':', alpha=0.7, label='p=1/3 (C threshold)')
axes[0,0].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7, label='p=1/2 (N threshold)')
axes[0,0].set_xlabel('Werner parameter p', fontsize=11)
axes[0,0].set_ylabel('Value', fontsize=11)
axes[0,0].set_title('Concurrence vs Negativity', fontsize=12)
axes[0,0].legend(fontsize=9)
axes[0,0].grid(True, alpha=0.3)

# Plot 2: E_F and E_N
axes[0,1].plot(p_range, EF_vals, 'b-', linewidth=2, label='E_F (formation)')
axes[0,1].plot(p_range, EN_vals, 'r--', linewidth=2, label='E_N (log-negativity)')
axes[0,1].axvline(x=1/3, color='g', linestyle=':', alpha=0.7)
axes[0,1].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
axes[0,1].fill_between(p_range, EF_vals, EN_vals, where=[ef > en for ef, en in zip(EF_vals, EN_vals)],
                        alpha=0.3, color='purple', label='E_F - E_N gap')
axes[0,1].set_xlabel('Werner parameter p', fontsize=11)
axes[0,1].set_ylabel('Entanglement (ebits)', fontsize=11)
axes[0,1].set_title('Formation vs Log-Negativity', fontsize=12)
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, alpha=0.3)

# Plot 3: State classification regions
axes[1,0].axvspan(0, 1/3, alpha=0.3, color='green', label='Separable')
axes[1,0].axvspan(1/3, 0.5, alpha=0.3, color='orange', label='Bound Entangled')
axes[1,0].axvspan(0.5, 1, alpha=0.3, color='red', label='Distillable')
axes[1,0].plot(p_range, C_vals, 'k-', linewidth=2, label='Concurrence')
axes[1,0].set_xlabel('Werner parameter p', fontsize=11)
axes[1,0].set_ylabel('Concurrence', fontsize=11)
axes[1,0].set_title('Werner State Phase Diagram', fontsize=12)
axes[1,0].legend(fontsize=9)
axes[1,0].grid(True, alpha=0.3)

# Plot 4: E_F(C) function
C_range = np.linspace(0, 1, 100)
EF_from_C = [entanglement_of_formation(c) for c in C_range]

axes[1,1].plot(C_range, EF_from_C, 'b-', linewidth=2)
axes[1,1].plot(C_range, C_range, 'r--', linewidth=1, alpha=0.5, label='y = C')
axes[1,1].plot([0.5], [entanglement_of_formation(0.5)], 'go', markersize=10, label=f'C=0.5 → E_F={entanglement_of_formation(0.5):.3f}')
axes[1,1].set_xlabel('Concurrence C', fontsize=11)
axes[1,1].set_ylabel('E_F (ebits)', fontsize=11)
axes[1,1].set_title('Wootters Formula: E_F(C)', fontsize=12)
axes[1,1].legend(fontsize=9)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_xlim([0, 1])
axes[1,1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('week79_review.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlots saved to 'week79_review.png'")

# 9. Quick reference card
print("\n" + "="*60)
print("QUICK REFERENCE CARD")
print("="*60)
print("""
ENTANGLEMENT MEASURES CHEAT SHEET
=================================

PURE STATES (use entropy of entanglement):
  E = S(ρ_A) = -Σᵢ λᵢ log₂ λᵢ   (Schmidt coefficients)
  For 2 qubits: C = 2|ad - bc| for |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩

MIXED 2-QUBIT STATES (use concurrence):
  C = max(0, λ₁ - λ₂ - λ₃ - λ₄)  where λᵢ = √(eigenvalues of ρρ̃)
  E_F = h((1 + √(1-C²))/2)

GENERAL MIXED STATES (use negativity):
  N = (||ρᵀᴮ||₁ - 1)/2 = Σ_{λ<0} |λᵢ|
  E_N = log₂(1 + 2N)

KEY THRESHOLDS (Werner state):
  p = 1/3: Separability threshold (C = 0)
  p = 1/2: PPT threshold (N = 0)
  1/3 < p ≤ 1/2: Bound entangled region

HIERARCHY:
  E_D ≤ E_N ≤ E_F ≤ E_C
  Pure states: E_D = E_F = E_C = E

SEPARABILITY TESTS:
  C = 0 → separable (2 qubits)
  PPT + additional test → separable (higher dim)
  NPT → definitely entangled and distillable
""")

print("\n" + "=" * 60)
print("WEEK 79 REVIEW COMPLETE")
print("=" * 60)
```

---

## Summary

### Master Formula Sheet

| Pure States | Mixed States (2-qubit) | General Mixed |
|-------------|------------------------|---------------|
| $E = S(\rho_A)$ | $C = \max(0, \lambda_1-\lambda_2-\lambda_3-\lambda_4)$ | $\mathcal{N} = \sum_{\lambda_i<0}\|\lambda_i\|$ |
| $E = H(\{\lambda_i^{Schmidt}\})$ | $E_F = h((1+\sqrt{1-C^2})/2)$ | $E_\mathcal{N} = \log_2(1+2\mathcal{N})$ |

### Decision Tree Summary

1. **Is state pure?** → Use entropy of entanglement
2. **Is it 2-qubit?** → Use concurrence + Wootters formula
3. **Is it PPT?** → If yes, may be separable or bound entangled
4. **Is negativity > 0?** → If yes, state is distillable

### Week 79 Key Achievements

1. **Von Neumann entropy** - foundation of quantum information
2. **Entropy of entanglement** - unique pure state measure
3. **Concurrence** - computable 2-qubit measure
4. **Negativity** - detects NPT entanglement
5. **Entanglement of formation** - convex roof extension
6. **Operational measures** - asymptotic rates

---

## Weekly Checklist

- [ ] I can compute von Neumann entropy
- [ ] I understand entropy of entanglement and Schmidt decomposition
- [ ] I can apply the Wootters formula for concurrence
- [ ] I can compute negativity via partial transpose
- [ ] I understand the convex roof construction for E_F
- [ ] I know the measure hierarchy E_D ≤ E_N ≤ E_F ≤ E_C
- [ ] I can identify separable, bound entangled, and distillable states
- [ ] I can choose the appropriate measure for a given problem

---

*Next Week: Week 80 — Entanglement Dynamics and Decoherence*
