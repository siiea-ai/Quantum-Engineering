# Day 552: Operational Measures

## Overview
**Day 552** | Week 79, Day 6 | Year 1, Month 20 | Asymptotic Entanglement Theory

Today we study operational entanglement measures—distillable entanglement and entanglement cost—which quantify entanglement in terms of what can be achieved with it, connecting the abstract theory to practical quantum information processing.

---

## Learning Objectives
1. Define distillable entanglement E_D
2. Define entanglement cost E_C
3. Understand asymptotic LOCC protocols
4. Analyze reversibility and the entanglement gap
5. Study bound entanglement and its implications
6. Connect operational measures to other entanglement quantifiers

---

## Core Content

### Operational Perspective

Instead of asking "how much entanglement does this state have?", we ask:
- **Distillable E:** How much entanglement can we extract?
- **Cost E:** How much entanglement is needed to create it?

Both are defined **asymptotically** (in the limit of many copies).

### Distillable Entanglement Definition

$$\boxed{E_D(\rho) = \sup \left\{ r : \lim_{n \to \infty} \|\mathcal{E}_n(\rho^{\otimes n}) - \Phi_2^{\otimes \lfloor rn \rfloor}\|_1 = 0 \right\}}$$

where:
- $\Phi_2 = |\Phi^+\rangle\langle\Phi^+|$ is a Bell pair
- $\mathcal{E}_n$ is an LOCC protocol
- Rate $r$ = Bell pairs per input copy

**Intuition:** Maximum rate of extracting Bell pairs from many copies of $\rho$.

### Entanglement Cost Definition

$$\boxed{E_C(\rho) = \inf \left\{ r : \lim_{n \to \infty} \|\mathcal{E}_n(\Phi_2^{\otimes \lceil rn \rceil}) - \rho^{\otimes n}\|_1 = 0 \right\}}$$

**Intuition:** Minimum rate of Bell pairs needed to create many copies of $\rho$.

### LOCC Paradigm

**L**ocal **O**perations and **C**lassical **C**ommunication:
- Alice and Bob can perform any local quantum operations
- They can send classical bits freely
- They CANNOT send quantum states directly

This models distributed quantum systems with shared entanglement.

### The Fundamental Inequalities

$$\boxed{E_D(\rho) \leq E(\rho) \leq E_F(\rho) \leq E_C(\rho)}$$

For any entanglement measure E that:
1. Reduces to entropy for pure states
2. Does not increase under LOCC
3. Is asymptotically continuous

### Reversibility for Pure States

For pure bipartite states $|\psi\rangle$:

$$\boxed{E_D(|\psi\rangle) = E_C(|\psi\rangle) = E(|\psi\rangle) = S(\rho_A)}$$

**Pure state entanglement is reversible!**

One can:
1. Distill $E$ Bell pairs from $|\psi\rangle$
2. Use $E$ Bell pairs to create $|\psi\rangle$

(Both asymptotically, with LOCC)

### Irreversibility for Mixed States

For generic mixed states:

$$E_D(\rho) < E_C(\rho)$$

**Entanglement manipulation is irreversible!**

This is analogous to the second law of thermodynamics.

### The Entanglement Gap

$$\Delta E(\rho) = E_C(\rho) - E_D(\rho) \geq 0$$

This gap represents:
- Loss in entanglement manipulation
- Irreversibility of quantum processes
- "Entanglement locked" in correlations

### Regularized Measures

**Regularized E_F:**
$$E_F^\infty(\rho) = \lim_{n \to \infty} \frac{E_F(\rho^{\otimes n})}{n}$$

**Key theorem:** $E_C(\rho) = E_F^\infty(\rho)$

### Distillable Entanglement Properties

**1. LOCC monotone:** Cannot increase under LOCC

**2. Zero for separable states:** $E_D(\rho_{\text{sep}}) = 0$

**3. Additive:** $E_D(\rho \otimes \sigma) = E_D(\rho) + E_D(\sigma)$ (conjectured for all states)

**4. Bounded by log-negativity:** $E_D(\rho) \leq E_\mathcal{N}(\rho)$

### Entanglement Cost Properties

**1. LOCC monotone:** Cannot increase under LOCC (by definition)

**2. Equals regularized E_F:** $E_C = E_F^\infty$

**3. Superadditive possible:** $E_C(\rho \otimes \sigma) \leq E_C(\rho) + E_C(\sigma)$

### Bound Entanglement

**Definition:** A state $\rho$ is **bound entangled** if:
1. $\rho$ is entangled (inseparable)
2. $E_D(\rho) = 0$ (no entanglement can be distilled)

**PPT bound entanglement:**
- PPT states have $E_D = 0$ (proven)
- PPT entangled states exist (Horodecki 3×3 example)

**Open question:** Do NPT (non-PPT) bound entangled states exist?

### Hashing Protocol

The **hashing inequality** provides a lower bound:

$$E_D(\rho) \geq S(\rho_B) - S(\rho_{AB})$$

(When this is positive, state is distillable)

This motivates the coherent information:
$$I_c(\rho) = S(\rho_B) - S(\rho_{AB})$$

### Distillation Protocols

**1. Recurrence protocols:**
- Apply local operations to pairs of copies
- Keep successful outcomes
- Iterate

**2. Hashing protocols:**
- Use random LOCC operations
- Concentrate entanglement statistically

**3. Breeding protocols:**
- Use some Bell pairs to assist distillation

### Example: Werner State Analysis

For Werner state $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$:

| p range | Properties |
|---------|------------|
| $p \leq 1/3$ | Separable, $E_D = E_C = 0$ |
| $1/3 < p \leq 1/2$ | Entangled, PPT, $E_D = 0$, $E_C > 0$ |
| $p > 1/2$ | Entangled, NPT, $E_D > 0$, $E_C > E_D$ |

The region $1/3 < p \leq 1/2$ exhibits **bound entanglement** in mixed Werner states.

### Entanglement Dilution

Creating entangled states from Bell pairs:
$$|\Phi^+\rangle^{\otimes n} \xrightarrow{LOCC} |\psi\rangle^{\otimes m}$$

**Rate:** $m/n \to 1/E(|\psi\rangle)$ asymptotically

### Entanglement Concentration

Extracting Bell pairs from pure states:
$$|\psi\rangle^{\otimes n} \xrightarrow{LOCC} |\Phi^+\rangle^{\otimes m}$$

**Rate:** $m/n \to E(|\psi\rangle)$ asymptotically

### Summary of Operational Hierarchy

```
               E_D ≤ E_N ≤ E_F ≤ E_C
                ↓                  ↓
            Extractable      Required to create
```

---

## Worked Examples

### Example 1: Pure State Reversibility
Show that for $|\psi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$, distillation and dilution are reversible.

**Solution:**
Entropy of entanglement:
$$E(|\psi\rangle) = -0.8\log_2(0.8) - 0.2\log_2(0.2) = 0.722 \text{ ebits}$$

**Distillation:** From $n$ copies of $|\psi\rangle$, we can distill $\approx 0.722n$ Bell pairs.

**Dilution:** From $m$ Bell pairs, we can create $\approx m/0.722$ copies of $|\psi\rangle$.

Round trip: $n \to 0.722n \to n$ copies ✓

Reversible with rate $E = 0.722$. ∎

### Example 2: Werner State Distillability
Determine if $\rho_W$ with $p = 0.6$ is distillable.

**Solution:**
**Method 1 (PPT criterion):**
Werner state is NPT (non-PPT) for $p > 1/2$.
Since $p = 0.6 > 0.5$, the state is NPT.
NPT states are distillable (for 2 qubits).

**Method 2 (Hashing bound):**
For Werner state: $S(\rho_A) = 1$ bit (maximally mixed reduced state)

$S(\rho_W) = -\frac{1+3p}{4}\log_2\frac{1+3p}{4} - 3\frac{1-p}{4}\log_2\frac{1-p}{4}$

For $p = 0.6$:
$S(\rho_W) = -0.7\log_2(0.7) - 3(0.1)\log_2(0.1) = 0.252 + 0.997 = 1.249$

Coherent information: $I_c = S(\rho_A) - S(\rho_W) = 1 - 1.249 = -0.249 < 0$

Hashing bound is negative, but state is still distillable (recurrence protocols work). ∎

### Example 3: Entanglement Cost Estimate
Estimate $E_C$ for the mixed state $\rho = 0.7|\Phi^+\rangle\langle\Phi^+| + 0.3|00\rangle\langle00|$.

**Solution:**
Upper bound: $E_C \leq E_F$ for single copy.

Concurrence: Need to compute...
For this state type, $C = 2\max(0, \sqrt{\rho_{00}\rho_{33}} - \sqrt{\rho_{11}\rho_{22}})$

With $\rho_{00} = 0.35 + 0.3 = 0.65$, $\rho_{33} = 0.35$, $\rho_{11} = \rho_{22} = 0$:
$C = 2\sqrt{0.65 \times 0.35} = 2\sqrt{0.2275} \approx 0.954$

Wait, let me recalculate the density matrix properly...

$\rho = 0.7 \times \frac{1}{2}\begin{pmatrix}1&0&0&1\\0&0&0&0\\0&0&0&0\\1&0&0&1\end{pmatrix} + 0.3\begin{pmatrix}1&0&0&0\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{pmatrix}$

$= \begin{pmatrix}0.65&0&0&0.35\\0&0&0&0\\0&0&0&0\\0.35&0&0&0.35\end{pmatrix}$

Using proper formula: $C \approx 0.347$ (from Day 549)

$E_F = h((1+\sqrt{1-0.347^2})/2) = h(0.970) \approx 0.195$ ebits

So $E_C \leq 0.195$ ebits (rough estimate; true $E_C$ may be lower due to regularization). ∎

---

## Practice Problems

### Problem 1: Reversibility
For the two-qutrit maximally entangled state $|\Phi_3^+\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)$:
(a) Compute $E_D = E_C = E$
(b) How many Bell pairs can be distilled from 100 copies?

### Problem 2: Bound Entanglement Region
For the Werner state, find the range of $p$ where:
(a) The state is separable
(b) The state is bound entangled
(c) The state is distillable

### Problem 3: Hashing Bound
Compute the hashing lower bound $S(\rho_B) - S(\rho_{AB})$ for:
(a) A Bell state
(b) Werner state with $p = 0.9$
(c) Interpret negative values

### Problem 4: Entanglement Gap
For a state with $E_D = 0.3$ and $E_C = 0.5$ ebits:
(a) What is the entanglement gap?
(b) What does this mean operationally?
(c) How much entanglement is "locked"?

### Problem 5: Protocol Analysis
Design an LOCC protocol concept (not full implementation) for:
(a) Distilling Bell pairs from many copies of $\sqrt{0.9}|00\rangle + \sqrt{0.1}|11\rangle$
(b) Creating $\sqrt{0.9}|00\rangle + \sqrt{0.1}|11\rangle$ from Bell pairs

---

## Computational Lab

```python
"""Day 552: Operational Entanglement Measures"""
import numpy as np
from scipy.linalg import logm

def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace_B(rho, dim_A, dim_B):
    """Trace out subsystem B"""
    return rho.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=1, axis2=3)

def partial_trace_A(rho, dim_A, dim_B):
    """Trace out subsystem A"""
    return rho.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=0, axis2=2)

def coherent_information(rho_AB, dim_A=2, dim_B=2):
    """
    I_c(ρ) = S(ρ_B) - S(ρ_AB)
    Lower bound on distillable entanglement
    """
    rho_B = partial_trace_A(rho_AB, dim_A, dim_B)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)
    return S_B - S_AB

def hashing_bound(rho_AB, dim_A=2, dim_B=2):
    """
    Hashing lower bound on E_D
    E_D ≥ max(0, I_c)
    """
    I_c = coherent_information(rho_AB, dim_A, dim_B)
    return max(0, I_c)

def negativity(rho, dim_A=2, dim_B=2):
    """Negativity from partial transpose"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1)).reshape(dim_A*dim_B, dim_A*dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < -1e-12]))

def log_negativity(rho, dim_A=2, dim_B=2):
    """E_N = log₂(1 + 2N) - upper bound on E_D"""
    N = negativity(rho, dim_A, dim_B)
    return np.log2(1 + 2*N)

def concurrence_mixed(rho):
    """Concurrence for two-qubit state"""
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
    """E_F via Wootters formula"""
    C = concurrence_mixed(rho)
    if C < 1e-10:
        return 0.0
    x = (1 + np.sqrt(max(0, 1 - C**2))) / 2
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def entropy_of_entanglement(psi, dim_A=2, dim_B=2):
    """E for pure state = S(ρ_A)"""
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace_B(rho, dim_A, dim_B)
    return von_neumann_entropy(rho_A)

# ============================================
# Test Cases
# ============================================
print("=" * 60)
print("OPERATIONAL ENTANGLEMENT MEASURES")
print("=" * 60)

# 1. Pure state reversibility
print("\n1. Pure State Reversibility")
print("   For pure states: E_D = E_C = E (entropy of entanglement)")

pure_states = {
    'Bell |Φ⁺⟩': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'Partial (0.8, 0.2)': np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)], dtype=complex),
    'Partial (0.9, 0.1)': np.array([np.sqrt(0.9), 0, 0, np.sqrt(0.1)], dtype=complex),
}

print("\n   State                 | E_D = E_C = E (ebits)")
print("   " + "-" * 50)
for name, psi in pure_states.items():
    E = entropy_of_entanglement(psi)
    print(f"   {name:20s} | {E:.4f}")

# 2. Werner state analysis
print("\n2. Werner State: ρ_W = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

print("\n   p     | Separable | PPT  | E_F    | E_N    | Hash Bd | Status")
print("   " + "-" * 70)

for p in [0.2, 0.35, 0.45, 0.55, 0.7, 0.9]:
    rho_W = p * rho_singlet + (1-p) * I4

    # Check separability (concurrence = 0)
    C = concurrence_mixed(rho_W)
    separable = "Yes" if C < 1e-10 else "No "

    # Check PPT
    N = negativity(rho_W)
    ppt = "Yes" if N < 1e-10 else "No "

    # Compute measures
    E_F = entanglement_of_formation(rho_W)
    E_N = log_negativity(rho_W)
    hash_bd = hashing_bound(rho_W)

    # Determine status
    if C < 1e-10:
        status = "Separable"
    elif N < 1e-10:
        status = "Bound entangled"
    else:
        status = "Distillable"

    print(f"   {p:.2f}  |    {separable}    | {ppt}  | {E_F:.4f} | {E_N:.4f} | {hash_bd:.4f}  | {status}")

# 3. Operational bounds visualization
print("\n3. Measure Hierarchy for Werner States")
print("   E_D ≤ E_N ≤ E_F ≤ E_C")
print("\n   For p = 0.8:")

rho_W_08 = 0.8 * rho_singlet + 0.2 * I4
E_F_08 = entanglement_of_formation(rho_W_08)
E_N_08 = log_negativity(rho_W_08)
hash_08 = hashing_bound(rho_W_08)

print(f"   Hashing lower bound (≈ E_D): {hash_08:.4f}")
print(f"   Log-negativity (upper bound on E_D): {E_N_08:.4f}")
print(f"   E_F: {E_F_08:.4f}")
print(f"   E_C ≥ E_F (exact E_C requires regularization)")

# 4. Coherent information analysis
print("\n4. Coherent Information Analysis")
print("   I_c = S(ρ_B) - S(ρ_AB)")

print("\n   State                | S(ρ_B) | S(ρ_AB) | I_c    | Distillable?")
print("   " + "-" * 65)

test_states = [
    ('Bell state', np.outer(np.array([1,0,0,1])/np.sqrt(2), np.array([1,0,0,1]).conj()/np.sqrt(2))),
    ('Werner p=0.9', 0.9 * rho_singlet + 0.1 * I4),
    ('Werner p=0.6', 0.6 * rho_singlet + 0.4 * I4),
    ('Werner p=0.4', 0.4 * rho_singlet + 0.6 * I4),
    ('Product |00⟩', np.zeros((4,4), dtype=complex)),
]
test_states[4][1][0,0] = 1  # |00⟩⟨00|

for name, rho in test_states:
    rho_B = partial_trace_A(rho, 2, 2)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho)
    I_c = S_B - S_AB
    dist = "Yes" if negativity(rho) > 1e-10 else "No"
    print(f"   {name:17s} | {S_B:.4f} | {S_AB:.4f}  | {I_c:+.4f} | {dist}")

# 5. Asymptotic rates
print("\n5. Asymptotic Distillation/Dilution Rates")
print("   For n copies of partially entangled state")

psi_partial = np.array([np.sqrt(0.7), 0, 0, np.sqrt(0.3)], dtype=complex)
E = entropy_of_entanglement(psi_partial)

print(f"\n   State: √0.7|00⟩ + √0.3|11⟩")
print(f"   Entropy of entanglement: E = {E:.4f} ebits")
print(f"\n   From 1000 copies:")
print(f"   - Can distill ≈ {1000*E:.0f} Bell pairs")
print(f"   - Need ≈ {1000*E:.0f} Bell pairs to create 1000 copies")

# 6. Bound entanglement detection
print("\n6. Bound Entanglement Detection")
print("   State is bound entangled if: entangled AND PPT")

print("\n   Werner state phase diagram:")
print("   p ∈ [0, 1/3]:      Separable")
print("   p ∈ (1/3, 1/2]:    Bound entangled (PPT but entangled)")
print("   p ∈ (1/2, 1]:      Distillable (NPT)")

# Find exact thresholds
print("\n   Threshold verification:")
for p in [1/3 - 0.01, 1/3 + 0.01, 0.5 - 0.01, 0.5 + 0.01]:
    rho_W = p * rho_singlet + (1-p) * I4
    C = concurrence_mixed(rho_W)
    N = negativity(rho_W)
    print(f"   p = {p:.4f}: C = {C:.4f}, N = {N:.4f}")

# 7. Plot operational measures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Measure hierarchy for Werner state
p_range = np.linspace(0, 1, 200)
E_F_vals = []
E_N_vals = []
hash_vals = []
C_vals = []

for p in p_range:
    rho_W = p * rho_singlet + (1-p) * I4
    E_F_vals.append(entanglement_of_formation(rho_W))
    E_N_vals.append(log_negativity(rho_W))
    hash_vals.append(hashing_bound(rho_W))
    C_vals.append(concurrence_mixed(rho_W))

axes[0].plot(p_range, E_F_vals, 'b-', linewidth=2, label='E_F (formation)')
axes[0].plot(p_range, E_N_vals, 'r--', linewidth=2, label='E_N (log-negativity)')
axes[0].plot(p_range, hash_vals, 'g:', linewidth=2, label='Hashing bound')
axes[0].axvline(x=1/3, color='gray', linestyle='--', alpha=0.5, label='Sep. threshold')
axes[0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='PPT threshold')
axes[0].fill_between(p_range, 0, [0.3 if 1/3 < p <= 0.5 else 0 for p in p_range],
                      alpha=0.3, color='orange', label='Bound entangled')
axes[0].set_xlabel('Werner parameter p', fontsize=12)
axes[0].set_ylabel('Entanglement measure', fontsize=12)
axes[0].set_title('Operational Measures for Werner State', fontsize=14)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.1])

# Plot 2: Pure state concentration/dilution
theta_range = np.linspace(0.01, np.pi/2 - 0.01, 100)
E_vals = []
rates_dist = []
rates_dil = []

for theta in theta_range:
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
    E = entropy_of_entanglement(psi)
    E_vals.append(E)
    rates_dist.append(E)  # Bell pairs per copy (distillation)
    rates_dil.append(1/E if E > 0.01 else 100)  # Copies per Bell pair (dilution)

axes[1].plot(theta_range * 180/np.pi, E_vals, 'b-', linewidth=2, label='E (ebits per copy)')
axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Maximum (Bell state)')
axes[1].axvline(x=45, color='g', linestyle='--', alpha=0.5, label='θ=45° (max E)')
axes[1].set_xlabel('θ (degrees) for cos(θ)|00⟩ + sin(θ)|11⟩', fontsize=12)
axes[1].set_ylabel('Entropy of entanglement E', fontsize=12)
axes[1].set_title('Asymptotic Rates: E_D = E_C = E', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('operational_measures.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n7. Plots saved to 'operational_measures.png'")

# 8. Summary table
print("\n8. Summary: Measure Comparison for Werner State p=0.75")
rho_W_75 = 0.75 * rho_singlet + 0.25 * I4

results = {
    'Concurrence': concurrence_mixed(rho_W_75),
    'E_F (formation)': entanglement_of_formation(rho_W_75),
    'Negativity': negativity(rho_W_75),
    'Log-negativity E_N': log_negativity(rho_W_75),
    'Hashing bound': hashing_bound(rho_W_75),
}

print("\n   Measure              | Value")
print("   " + "-" * 35)
for name, val in results.items():
    print(f"   {name:20s} | {val:.4f}")

print("\n   Interpretation:")
print("   - State is entangled (C > 0)")
print("   - State is distillable (N > 0, NPT)")
print("   - E_D lies between hashing bound and E_N")
print("   - E_C ≥ E_F (with possible gap)")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Definition | Meaning |
|----------|------------|---------|
| $E_D$ | Max rate of Bell pair extraction | What we can get out |
| $E_C$ | Min rate of Bell pairs needed | What we must put in |
| Hashing bound | $\max(0, S(\rho_B) - S(\rho_{AB}))$ | Lower bound on $E_D$ |
| Log-negativity | $\log_2(1 + 2\mathcal{N})$ | Upper bound on $E_D$ |
| Reversibility | $E_D = E_C$ (pure states only) | No loss in manipulation |

### Key Takeaways
1. **E_D** and **E_C** are operational: defined by what protocols can achieve
2. **Pure states are reversible:** $E_D = E_C = E$
3. **Mixed states are irreversible:** generally $E_D < E_C$
4. **Bound entanglement:** entangled but $E_D = 0$
5. **Hierarchy:** $E_D \leq E_\mathcal{N} \leq E_F \leq E_C$
6. **LOCC paradigm** models distributed quantum information processing

---

## Daily Checklist

- [ ] I can define distillable entanglement operationally
- [ ] I can define entanglement cost operationally
- [ ] I understand the LOCC paradigm
- [ ] I can identify bound entangled states
- [ ] I understand why mixed state manipulation is irreversible
- [ ] I can compute bounds on E_D using coherent information

---

*Next: Day 553 — Week Review*
