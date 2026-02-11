# Day 551: Entanglement of Formation

## Overview
**Day 551** | Week 79, Day 5 | Year 1, Month 20 | Mixed State Entanglement Measure

Today we study the entanglement of formation—the canonical extension of pure state entanglement to mixed states via the convex roof construction, representing the minimum entanglement needed to create a state.

---

## Learning Objectives
1. Define entanglement of formation via convex roof
2. Understand optimal pure state decompositions
3. Apply Wootters formula for two qubits
4. Compute E_F for common mixed states
5. Connect E_F to entanglement cost
6. Analyze additivity properties and open problems

---

## Core Content

### The Challenge of Mixed States

For pure states, entropy of entanglement is unique and well-defined:
$$E(|\psi\rangle) = S(\rho_A)$$

For mixed states $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$, different decompositions give different average entanglement!

### Convex Roof Extension

**Entanglement of Formation:**

$$\boxed{E_F(\rho) = \min_{\{p_i, |\psi_i\rangle\}} \sum_i p_i E(|\psi_i\rangle)}$$

Minimize over all pure state decompositions: $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$

### Properties of the Minimum

The decomposition achieving the minimum is called the **optimal decomposition**.

**Existence:** The minimum exists (compact set, continuous function).

**Non-uniqueness:** Multiple decompositions may achieve the same minimum.

### Physical Interpretation

$E_F(\rho)$ = minimum average entanglement to create $\rho$ via LOCC + shared entanglement

Starting from separable state + classical communication:
- Alice prepares $|\psi_i\rangle$ with probability $p_i$
- Requires $E(|\psi_i\rangle)$ ebits each time
- Average cost: $\sum_i p_i E(|\psi_i\rangle)$
- $E_F$ = minimum achievable average cost

### E_F Properties

**1. Non-negativity:** $E_F(\rho) \geq 0$

**2. Faithfulness:** $E_F(\rho) = 0 \iff \rho$ is separable

**3. Convexity:** $E_F(\sum_i p_i \rho_i) \leq \sum_i p_i E_F(\rho_i)$

**4. LOCC monotonicity:** Cannot increase under LOCC

**5. Reduces to entropy for pure states:** $E_F(|\psi\rangle\langle\psi|) = E(|\psi\rangle)$

### Wootters Formula (Two Qubits)

For any two-qubit state, $E_F$ has an **analytical formula**:

$$\boxed{E_F(\rho) = h\left(\frac{1 + \sqrt{1-C^2(\rho)}}{2}\right)}$$

where:
- $C(\rho)$ is the concurrence (Day 549)
- $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is binary entropy

### Derivation of Wootters Formula

**Step 1:** Show optimal decomposition uses states with equal entanglement.

**Step 2:** Relate entanglement of pure states to concurrence:
For pure state $|\psi\rangle$: $E(|\psi\rangle) = h\left(\frac{1+\sqrt{1-C^2}}{2}\right)$

**Step 3:** Prove concurrence satisfies convex roof:
$C(\rho) = \min \sum_i p_i C(|\psi_i\rangle)$

**Step 4:** Combine using monotonicity of $h$.

### Function Analysis

Define $f(C) = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$

**Properties of f:**
- $f(0) = 0$ (separable)
- $f(1) = 1$ (maximally entangled)
- $f$ is monotonically increasing
- $f$ is convex in $C$

| C | f(C) = E_F |
|---|------------|
| 0.0 | 0.000 |
| 0.2 | 0.043 |
| 0.4 | 0.163 |
| 0.6 | 0.345 |
| 0.8 | 0.581 |
| 1.0 | 1.000 |

### Werner State E_F

For Werner state $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$:

Concurrence: $C = \max(0, \frac{3p-1}{2})$

Entanglement of formation:
$$E_F(\rho_W) = \begin{cases} 0 & p \leq 1/3 \\ h\left(\frac{1+\sqrt{1-((3p-1)/2)^2}}{2}\right) & p > 1/3 \end{cases}$$

### Isotropic State E_F

For isotropic state $\rho_F$ with fidelity $F$:

$$E_F(\rho_F) = \begin{cases} 0 & F \leq 1/2 \\ h\left(\frac{1+\sqrt{1-((2F-1))^2}}{2}\right) & F > 1/2 \end{cases}$$

### Optimal Decomposition Example

**Bell diagonal state:**
$$\rho = \sum_{i=1}^{4} \lambda_i |\beta_i\rangle\langle\beta_i|$$

If $\lambda_1 \geq 1/2$ (one Bell state dominates), the optimal decomposition is NOT the Bell basis decomposition!

The optimal decomposition uses **correlated superpositions** of Bell states.

### E_F vs E_D

| Property | E_F (Formation) | E_D (Distillable) |
|----------|-----------------|-------------------|
| Definition | Min avg to create | Max extractable |
| Direction | Input entanglement | Output entanglement |
| Inequality | $E_F \geq E_D$ | Always |
| Equality | Pure states only | $E_F = E_D = E$ |

### Additivity Questions

**Conjecture (disproved):** $E_F(\rho \otimes \sigma) = E_F(\rho) + E_F(\sigma)$

**Hastings (2009):** Showed additivity fails for some states!

This has profound implications for quantum channel capacity.

### Regularized E_F

$$E_F^\infty(\rho) = \lim_{n \to \infty} \frac{E_F(\rho^{\otimes n})}{n}$$

**Key result:** $E_F^\infty(\rho) = E_C(\rho)$ (entanglement cost)

### Computing E_F for Higher Dimensions

For systems larger than 2×2:
- No analytical formula exists
- Must use numerical optimization
- Convex optimization over decompositions
- SDP relaxations provide bounds

---

## Worked Examples

### Example 1: E_F from Concurrence
Compute $E_F$ for a state with $C = 0.6$.

**Solution:**
Using Wootters formula:
$$E_F = h\left(\frac{1 + \sqrt{1-0.36}}{2}\right) = h\left(\frac{1 + 0.8}{2}\right) = h(0.9)$$

$$E_F = -0.9\log_2(0.9) - 0.1\log_2(0.1)$$
$$= 0.137 + 0.332 = 0.469 \text{ ebits}$$ ∎

### Example 2: Werner State E_F
Find $E_F(\rho_W)$ for $p = 0.7$.

**Solution:**
First, compute concurrence:
$$C = \max\left(0, \frac{3(0.7)-1}{2}\right) = \frac{1.1}{2} = 0.55$$

Then apply Wootters formula:
$$E_F = h\left(\frac{1 + \sqrt{1-0.3025}}{2}\right) = h\left(\frac{1 + 0.835}{2}\right) = h(0.9175)$$

$$E_F = -0.9175\log_2(0.9175) - 0.0825\log_2(0.0825)$$
$$= 0.109 + 0.292 = 0.401 \text{ ebits}$$ ∎

### Example 3: Separability Test via E_F
Show that $\rho = 0.3|\Phi^+\rangle\langle\Phi^+| + 0.7\frac{I}{4}$ is separable.

**Solution:**
Concurrence of isotropic state:
$$C = \max\left(0, 2F - 1\right)$$

where $F = \langle\Phi^+|\rho|\Phi^+\rangle = 0.3 + 0.7/4 = 0.475$.

Since $F = 0.475 < 1/2$: $C = 0$

Therefore $E_F = 0$, confirming separability. ∎

---

## Practice Problems

### Problem 1: E_F Computation
Compute $E_F$ for:
(a) A Bell state $|\Phi^+\rangle$
(b) A state with $C = 0.8$
(c) Werner state with $p = 0.5$

### Problem 2: Threshold Analysis
At what value of $p$ does the Werner state have $E_F = 0.5$ ebits?

### Problem 3: Decomposition Analysis
For the state $\rho = \frac{1}{2}|\Phi^+\rangle\langle\Phi^+| + \frac{1}{2}|00\rangle\langle00|$:
(a) Compute $E_F$ using Wootters formula
(b) Find the average entanglement using the given decomposition
(c) Why is the given decomposition not optimal?

### Problem 4: Inequality Verification
For any state, verify that $E_F \geq \mathcal{N}$ (negativity). Does equality ever hold?

### Problem 5: Convexity
Prove that for two separable states $\rho_1$, $\rho_2$:
$$E_F(p\rho_1 + (1-p)\rho_2) = 0$$

---

## Computational Lab

```python
"""Day 551: Entanglement of Formation"""
import numpy as np
from scipy.optimize import minimize_scalar

def binary_entropy(x):
    """h(x) = -x log₂(x) - (1-x) log₂(1-x)"""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def concurrence_to_EF(C):
    """
    Wootters formula: E_F = h((1 + √(1-C²))/2)
    """
    if C < 1e-10:
        return 0.0
    x = (1 + np.sqrt(max(0, 1 - C**2))) / 2
    return binary_entropy(x)

def concurrence_mixed(rho):
    """
    Compute concurrence for mixed two-qubit state.
    """
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)

    rho_tilde = sigma_yy @ rho.conj() @ sigma_yy
    R = rho @ rho_tilde

    eigenvalues = np.linalg.eigvals(R)
    lambdas = np.sqrt(np.abs(eigenvalues.real))
    lambdas = np.sort(lambdas)[::-1]

    C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return max(0, C.real)

def entanglement_of_formation(rho):
    """
    E_F for two-qubit state using Wootters formula.
    """
    C = concurrence_mixed(rho)
    return concurrence_to_EF(C)

# ============================================
# Test Cases
# ============================================
print("=" * 60)
print("ENTANGLEMENT OF FORMATION")
print("=" * 60)

# 1. E_F function analysis
print("\n1. E_F as Function of Concurrence")
print("   C        E_F (ebits)")
print("   " + "-" * 25)

C_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for C in C_values:
    E_F = concurrence_to_EF(C)
    print(f"   {C:.1f}      {E_F:.4f}")

# 2. Bell states
print("\n2. Bell States")
bell_states = {
    'Φ⁺': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'Ψ⁻': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

for name, psi in bell_states.items():
    rho = np.outer(psi, psi.conj())
    C = concurrence_mixed(rho)
    E_F = entanglement_of_formation(rho)
    print(f"   |{name}⟩: C = {C:.4f}, E_F = {E_F:.4f} ebits")

# 3. Werner states
print("\n3. Werner States: ρ_W = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

print("   p       C        E_F (ebits)   Separable?")
print("   " + "-" * 50)

for p in [0.0, 0.2, 1/3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    rho_W = p * rho_singlet + (1-p) * I4
    C = concurrence_mixed(rho_W)
    E_F = entanglement_of_formation(rho_W)
    sep = "Yes" if C < 1e-10 else "No"
    print(f"   {p:.3f}   {C:.4f}   {E_F:.4f}        {sep}")

# 4. Mixed entangled states
print("\n4. Mixed Entangled States")

# State 1: Bell + product
print("\n   State: 0.6|Φ⁺⟩⟨Φ⁺| + 0.4|00⟩⟨00|")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())
rho_00 = np.zeros((4, 4), dtype=complex)
rho_00[0, 0] = 1
rho_mixed = 0.6 * rho_bell + 0.4 * rho_00

C = concurrence_mixed(rho_mixed)
E_F = entanglement_of_formation(rho_mixed)
print(f"   Concurrence: {C:.4f}")
print(f"   E_F: {E_F:.4f} ebits")

# Naive estimate using decomposition
E_naive = 0.6 * 1.0 + 0.4 * 0.0  # 0.6 × E(Bell) + 0.4 × E(product)
print(f"   Naive (non-optimal) estimate: {E_naive:.4f} ebits")
print(f"   Wootters gives LOWER value → decomposition is not optimal!")

# 5. E_F vs negativity comparison
print("\n5. E_F vs Negativity Comparison")

def negativity(rho, dim_A=2, dim_B=2):
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1)).reshape(4, 4)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < -1e-12]))

print("   Werner state p | E_F     | N       | E_F ≥ N?")
print("   " + "-" * 50)

for p in [0.4, 0.5, 0.6, 0.8, 1.0]:
    rho_W = p * rho_singlet + (1-p) * I4
    E_F = entanglement_of_formation(rho_W)
    N = negativity(rho_W)
    check = "Yes" if E_F >= N - 1e-10 else "No"
    print(f"       {p:.1f}       | {E_F:.4f} | {N:.4f} | {check}")

# 6. Find Werner parameter for specific E_F
print("\n6. Finding Werner Parameter for E_F = 0.5")

def werner_EF(p):
    rho_W = p * rho_singlet + (1-p) * I4
    return entanglement_of_formation(rho_W)

# Binary search for p where E_F = 0.5
def find_p_for_EF(target_EF):
    for p in np.linspace(1/3, 1, 1000):
        if werner_EF(p) >= target_EF:
            return p
    return None

p_half = find_p_for_EF(0.5)
print(f"   E_F = 0.5 ebits at p ≈ {p_half:.4f}")
print(f"   Verification: E_F({p_half:.4f}) = {werner_EF(p_half):.4f}")

# 7. Convex roof illustration
print("\n7. Convex Roof Illustration")
print("   For ρ = 0.5|Φ⁺⟩⟨Φ⁺| + 0.5|00⟩⟨00|")

# Different decompositions
print("\n   Decomposition 1 (given):")
print("   {|Φ⁺⟩, |00⟩} with p = {0.5, 0.5}")
E_decomp1 = 0.5 * 1.0 + 0.5 * 0.0
print(f"   Average E = {E_decomp1:.4f} ebits")

# Optimal (Wootters)
rho_test = 0.5 * rho_bell + 0.5 * rho_00
E_F_opt = entanglement_of_formation(rho_test)
print(f"\n   Optimal (Wootters): E_F = {E_F_opt:.4f} ebits")
print(f"   Savings from optimization: {E_decomp1 - E_F_opt:.4f} ebits")

# 8. Plot E_F curves
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: E_F as function of concurrence
C_range = np.linspace(0, 1, 100)
EF_range = [concurrence_to_EF(c) for c in C_range]

axes[0].plot(C_range, EF_range, 'b-', linewidth=2)
axes[0].plot(C_range, C_range, 'r--', linewidth=1, label='y = C (linear)')
axes[0].set_xlabel('Concurrence C', fontsize=12)
axes[0].set_ylabel('Entanglement of Formation E_F', fontsize=12)
axes[0].set_title('Wootters Formula: E_F(C)', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.1])

# Plot 2: E_F for Werner state
p_range = np.linspace(0, 1, 100)
EF_werner = [werner_EF(p) for p in p_range]
C_werner = [max(0, (3*p-1)/2) for p in p_range]

axes[1].plot(p_range, EF_werner, 'b-', linewidth=2, label='E_F')
axes[1].plot(p_range, C_werner, 'r--', linewidth=2, label='Concurrence')
axes[1].axvline(x=1/3, color='g', linestyle=':', label='Entanglement threshold')
axes[1].set_xlabel('Werner parameter p', fontsize=12)
axes[1].set_ylabel('Value', fontsize=12)
axes[1].set_title('Werner State: E_F and C vs p', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('entanglement_of_formation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n8. Plots saved to 'entanglement_of_formation.png'")

# 9. E_F bounds and inequalities
print("\n9. E_F Bounds and Inequalities")
print("   For all states: E_D ≤ E_F")
print("   For pure states: E_D = E_F = E (entropy of entanglement)")
print("   For Werner (p=0.8):")

rho_W_08 = 0.8 * rho_singlet + 0.2 * I4
E_F_08 = entanglement_of_formation(rho_W_08)
N_08 = negativity(rho_W_08)
log_neg = np.log2(1 + 2*N_08)

print(f"      E_F = {E_F_08:.4f}")
print(f"      Log-negativity (upper bound on E_D) = {log_neg:.4f}")
print(f"      Gap indicates irreversibility in entanglement manipulation")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula | Notes |
|----------|---------|-------|
| E_F definition | $E_F = \min \sum_i p_i E(\|\psi_i\rangle)$ | Convex roof |
| Wootters (2 qubits) | $E_F = h((1+\sqrt{1-C^2})/2)$ | Analytical |
| Binary entropy | $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ | Used in formula |
| Werner state | $E_F = 0$ for $p \leq 1/3$ | Threshold |

### Key Takeaways
1. **E_F** extends pure state entanglement to mixed states via minimization
2. **Convex roof** captures minimum resources needed to create state
3. **Wootters formula** gives analytical answer for two qubits
4. **E_F ≥ E_D** always (formation requires at least as much as can be extracted)
5. **Optimal decomposition** is generally NOT obvious
6. **Additivity failure** has deep implications for quantum information

---

## Daily Checklist

- [ ] I can define entanglement of formation via convex roof
- [ ] I understand the physical meaning of E_F
- [ ] I can apply Wootters formula for two qubits
- [ ] I can compute E_F for Werner states
- [ ] I understand why naive decompositions are suboptimal
- [ ] I know the relationship E_F ≥ E_D

---

*Next: Day 552 — Operational Measures*
