# Day 549: Concurrence

## Overview
**Day 549** | Week 79, Day 3 | Year 1, Month 20 | Two-Qubit Entanglement Measure

Today we study concurrence—the elegant entanglement measure for two-qubit systems introduced by Wootters, which provides an analytical formula for entanglement of formation and serves as a practical tool for quantifying mixed-state entanglement.

---

## Learning Objectives
1. Define concurrence for pure and mixed two-qubit states
2. Master the Wootters formula with spin-flip operator
3. Compute eigenvalues of $\rho\tilde{\rho}$
4. Relate concurrence to entanglement of formation
5. Analyze concurrence for Bell states and Werner states
6. Implement concurrence calculation numerically

---

## Core Content

### Concurrence for Pure States

For a two-qubit pure state $|\psi\rangle$:

$$\boxed{C(|\psi\rangle) = |\langle\psi|\tilde{\psi}\rangle|}$$

where $|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)|\psi^*\rangle$ is the **spin-flipped state**.

**Explicit formula:** For $|\psi\rangle = \sum_{i,j} c_{ij}|ij\rangle$:
$$C = 2|c_{00}c_{11} - c_{01}c_{10}|$$

### Spin-Flip Operator

The spin-flip operation involves:
$$\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

$$\sigma_y \otimes \sigma_y = \begin{pmatrix} 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \end{pmatrix}$$

The spin-flip of a state: $|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)|\psi^*\rangle$

### Concurrence Range

$$\boxed{0 \leq C \leq 1}$$

- **C = 0**: Separable (product) state
- **C = 1**: Maximally entangled (Bell state)
- **0 < C < 1**: Partially entangled

### Wootters Formula for Mixed States

For a general two-qubit mixed state $\rho$:

$$\boxed{C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)}$$

where $\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq \lambda_4 \geq 0$ are the **square roots** of eigenvalues of:

$$R = \rho \tilde{\rho}$$

with $\tilde{\rho} = (\sigma_y \otimes \sigma_y) \rho^* (\sigma_y \otimes \sigma_y)$.

### Constructing $\tilde{\rho}$

1. Take complex conjugate: $\rho \to \rho^*$
2. Apply spin-flip: $\tilde{\rho} = (\sigma_y \otimes \sigma_y) \rho^* (\sigma_y \otimes \sigma_y)$

**Key identity:** $(\sigma_y \otimes \sigma_y)^2 = I_4$

### Properties of Concurrence

**1. Invariance:** $C(U_A \otimes U_B \rho U_A^\dagger \otimes U_B^\dagger) = C(\rho)$

**2. Convexity:** $C(\sum_i p_i \rho_i) \leq \sum_i p_i C(\rho_i)$

**3. LOCC monotonicity:** Concurrence cannot increase under LOCC

**4. Connection to linear entropy:**
For pure states: $C^2 = 2(1 - \text{Tr}(\rho_A^2))$

### Bell State Concurrence

For any Bell state $|\beta\rangle$:

Using $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
- $c_{00} = c_{11} = 1/\sqrt{2}$, $c_{01} = c_{10} = 0$
- $C = 2|c_{00}c_{11} - c_{01}c_{10}| = 2 \cdot \frac{1}{2} = 1$

All Bell states have **C = 1** (maximally entangled).

### Product State Concurrence

For $|00\rangle$:
- $c_{00} = 1$, all others = 0
- $C = 2|1 \cdot 0 - 0 \cdot 0| = 0$

Product states have **C = 0**.

### Werner State Concurrence

For Werner state: $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$

$$\boxed{C(\rho_W) = \max\left(0, \frac{3p-1}{2}\right)}$$

**Threshold:** $C > 0$ only when $p > 1/3$

### Isotropic State Concurrence

For isotropic state: $\rho_F = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{3}(I - |\Phi^+\rangle\langle\Phi^+|)$

$$C(\rho_F) = \max\left(0, \frac{3F-1}{2}\right)$$

### Concurrence and Tangle

The **tangle** (or squared concurrence) is:
$$\tau = C^2$$

Useful for multipartite entanglement studies (Coffman-Kundu-Wootters inequality).

### Connection to Entanglement of Formation

$$E_F(\rho) = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$$

where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is binary entropy.

This remarkable formula allows analytical computation of $E_F$ for any two-qubit state!

### Derivation of Pure State Formula

For $|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$:

$$|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)(a^*|00\rangle + b^*|01\rangle + c^*|10\rangle + d^*|11\rangle)$$

Using $\sigma_y|0\rangle = i|1\rangle$ and $\sigma_y|1\rangle = -i|0\rangle$:

$$|\tilde{\psi}\rangle = d^*|00\rangle - c^*|01\rangle - b^*|10\rangle + a^*|11\rangle$$

$$\langle\psi|\tilde{\psi}\rangle = ad^* - bc^* - cb^* + da^* = 2(ad - bc)$$

$$C = |2(ad - bc)|$$

---

## Worked Examples

### Example 1: Concurrence of Pure State
Compute C for $|\psi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$.

**Solution:**
Coefficients: $c_{00} = \sqrt{0.8}$, $c_{11} = \sqrt{0.2}$, $c_{01} = c_{10} = 0$

$$C = 2|c_{00}c_{11} - c_{01}c_{10}| = 2\sqrt{0.8 \cdot 0.2} = 2\sqrt{0.16} = 0.8$$ ∎

### Example 2: Werner State Concurrence
Find the concurrence of $\rho_W$ with $p = 0.5$.

**Solution:**
Using the Werner state formula:
$$C(\rho_W) = \max\left(0, \frac{3(0.5)-1}{2}\right) = \max\left(0, \frac{0.5}{2}\right) = 0.25$$

The state is entangled with moderate concurrence. ∎

### Example 3: Wootters Formula Application
Compute C for the mixed state:
$$\rho = 0.6|\Phi^+\rangle\langle\Phi^+| + 0.4|00\rangle\langle00|$$

**Solution:**
First, construct the density matrix:
$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$|00\rangle\langle00| = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

$$\rho = \begin{pmatrix} 0.7 & 0 & 0 & 0.3 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0.3 & 0 & 0 & 0.3 \end{pmatrix}$$

Since $\rho$ is real: $\tilde{\rho} = (\sigma_y \otimes \sigma_y)\rho(\sigma_y \otimes \sigma_y)$

$$\tilde{\rho} = \begin{pmatrix} 0.3 & 0 & 0 & 0.3 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0.3 & 0 & 0 & 0.7 \end{pmatrix}$$

$$R = \rho\tilde{\rho} = \begin{pmatrix} 0.3 & 0 & 0 & 0.3 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0.18 & 0 & 0 & 0.3 \end{pmatrix}$$

Eigenvalues of R: $\lambda^2 \in \{0.48, 0.12, 0, 0\}$

Square roots: $\lambda_1 = \sqrt{0.48} \approx 0.693$, $\lambda_2 = \sqrt{0.12} \approx 0.346$

$$C = \max(0, 0.693 - 0.346 - 0 - 0) = 0.347$$ ∎

---

## Practice Problems

### Problem 1: Pure State Concurrence
Compute the concurrence for:
(a) $|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \sqrt{\frac{2}{3}}|11\rangle$
(b) $|\psi\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
(c) $|\psi\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$

### Problem 2: Entanglement Threshold
For what range of $p$ does the state $\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$ have non-zero concurrence?

### Problem 3: Concurrence Bounds
Prove that for any two-qubit pure state:
$$C^2 = 2(1 - \text{Tr}(\rho_A^2))$$
where $\rho_A$ is the reduced density matrix.

### Problem 4: X-State Concurrence
For the X-state (non-zero elements only on diagonal and anti-diagonal):
$$\rho_X = \begin{pmatrix} a & 0 & 0 & w \\ 0 & b & z & 0 \\ 0 & z^* & c & 0 \\ w^* & 0 & 0 & d \end{pmatrix}$$
Show that $C = 2\max(|w| - \sqrt{bc}, |z| - \sqrt{ad}, 0)$.

### Problem 5: Concurrence Dynamics
For the state $\rho(t) = e^{-\gamma t}|\Phi^+\rangle\langle\Phi^+| + (1-e^{-\gamma t})\frac{I}{4}$:
(a) Find C(t)
(b) At what time does entanglement vanish (entanglement sudden death)?

---

## Computational Lab

```python
"""Day 549: Concurrence"""
import numpy as np
from scipy.linalg import sqrtm

# Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def spin_flip_operator():
    """σ_y ⊗ σ_y matrix"""
    return np.kron(sigma_y, sigma_y)

def concurrence_pure(psi):
    """
    Concurrence for pure two-qubit state.
    C = 2|c₀₀c₁₁ - c₀₁c₁₀|
    """
    # Reshape to 2x2 matrix of coefficients
    c = psi.reshape(2, 2)
    return 2 * np.abs(c[0,0]*c[1,1] - c[0,1]*c[1,0])

def concurrence_pure_spinflip(psi):
    """
    Concurrence via spin-flip: C = |⟨ψ|ψ̃⟩|
    """
    sigma_yy = spin_flip_operator()
    psi_tilde = sigma_yy @ psi.conj()
    return np.abs(np.vdot(psi, psi_tilde))

def rho_tilde(rho):
    """
    Compute ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
    """
    sigma_yy = spin_flip_operator()
    return sigma_yy @ rho.conj() @ sigma_yy

def concurrence_mixed(rho):
    """
    Wootters formula for mixed state concurrence.
    C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    where λᵢ are sqrt of eigenvalues of ρρ̃ in decreasing order.
    """
    # Compute R = ρρ̃
    rho_t = rho_tilde(rho)
    R = rho @ rho_t

    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(R)

    # Take sqrt of absolute values (eigenvalues should be non-negative real)
    lambdas = np.sqrt(np.abs(eigenvalues.real))

    # Sort in decreasing order
    lambdas = np.sort(lambdas)[::-1]

    # Wootters formula
    C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return max(0, C), lambdas

def entanglement_of_formation_from_C(C):
    """
    E_F(ρ) = h((1 + √(1-C²))/2)
    """
    if C < 1e-10:
        return 0.0
    x = (1 + np.sqrt(1 - C**2)) / 2
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

# ============================================
# Test Cases
# ============================================
print("=" * 60)
print("CONCURRENCE CALCULATIONS")
print("=" * 60)

# 1. Bell states
print("\n1. Bell States (Maximally Entangled)")
bell_states = {
    'Φ⁺': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'Φ⁻': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    'Ψ⁺': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    'Ψ⁻': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

for name, psi in bell_states.items():
    C1 = concurrence_pure(psi)
    C2 = concurrence_pure_spinflip(psi)
    print(f"   |{name}⟩: C = {C1:.4f} (formula), {C2:.4f} (spin-flip)")

# 2. Product states
print("\n2. Product States (Separable)")
product_states = {
    '|00⟩': np.array([1, 0, 0, 0], dtype=complex),
    '|01⟩': np.array([0, 1, 0, 0], dtype=complex),
    '|++⟩': np.array([1, 1, 1, 1], dtype=complex) / 2,
}

for name, psi in product_states.items():
    C = concurrence_pure(psi)
    print(f"   {name}: C = {C:.4f}")

# 3. Partially entangled states
print("\n3. Partially Entangled States")
print("   √p|00⟩ + √(1-p)|11⟩")
print("   p       C")
print("   " + "-" * 20)

for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    if p == 0:
        psi = np.array([0, 0, 0, 1], dtype=complex)
    elif p == 1:
        psi = np.array([1, 0, 0, 0], dtype=complex)
    else:
        psi = np.array([np.sqrt(p), 0, 0, np.sqrt(1-p)], dtype=complex)
    C = concurrence_pure(psi)
    print(f"   {p:.2f}    {C:.4f}")

# 4. Werner states (mixed)
print("\n4. Werner States: ρ_W = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

print("   p       C(num)    C(exact)   Entangled?")
print("   " + "-" * 45)

for p in [0.0, 0.2, 1/3, 0.5, 0.7, 1.0]:
    rho_W = p * rho_singlet + (1-p) * I4
    C_num, lambdas = concurrence_mixed(rho_W)
    C_exact = max(0, (3*p - 1)/2)
    entangled = "Yes" if C_num > 1e-10 else "No"
    print(f"   {p:.2f}    {C_num:.4f}    {C_exact:.4f}      {entangled}")

# 5. Mixed state example
print("\n5. Mixed State: 0.6|Φ⁺⟩⟨Φ⁺| + 0.4|00⟩⟨00|")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())
rho_00 = np.zeros((4, 4), dtype=complex)
rho_00[0, 0] = 1

rho_mixed = 0.6 * rho_bell + 0.4 * rho_00
C, lambdas = concurrence_mixed(rho_mixed)
E_F = entanglement_of_formation_from_C(C)
print(f"   Concurrence: C = {C:.4f}")
print(f"   λ values: {lambdas}")
print(f"   Entanglement of Formation: E_F = {E_F:.4f} ebits")

# 6. Verify C² = 2(1 - Tr(ρ_A²)) for pure states
print("\n6. Verification: C² = 2(1 - Tr(ρ_A²))")
test_states = [
    ('Bell |Φ⁺⟩', np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)),
    ('Partial', np.array([np.sqrt(0.7), 0, 0, np.sqrt(0.3)], dtype=complex)),
    ('Product', np.array([1, 0, 0, 0], dtype=complex)),
]

for name, psi in test_states:
    C = concurrence_pure(psi)
    rho = np.outer(psi, psi.conj())
    rho_A = rho.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
    purity_A = np.trace(rho_A @ rho_A).real
    C_from_purity = np.sqrt(2 * (1 - purity_A))
    print(f"   {name}:")
    print(f"      C (direct) = {C:.4f}")
    print(f"      C (from purity) = {C_from_purity:.4f}")

# 7. C vs E_F curve
print("\n7. Concurrence vs Entanglement of Formation")
print("   C        E_F (ebits)")
print("   " + "-" * 25)

for C_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    E_F = entanglement_of_formation_from_C(C_val)
    print(f"   {C_val:.2f}     {E_F:.4f}")

# 8. Plot concurrence and E_F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Concurrence vs parameter p for partially entangled state
p_vals = np.linspace(0, 1, 100)
C_vals_pure = []
for p in p_vals:
    if p == 0 or p == 1:
        C_vals_pure.append(0)
    else:
        C_vals_pure.append(2 * np.sqrt(p * (1-p)))

axes[0].plot(p_vals, C_vals_pure, 'b-', linewidth=2)
axes[0].set_xlabel('p', fontsize=12)
axes[0].set_ylabel('Concurrence C', fontsize=12)
axes[0].set_title('Concurrence of √p|00⟩ + √(1-p)|11⟩', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=0.5, color='g', linestyle='--', alpha=0.5)

# Plot 2: E_F vs C
C_range = np.linspace(0.001, 0.999, 100)
E_F_vals = [entanglement_of_formation_from_C(c) for c in C_range]

axes[1].plot(C_range, E_F_vals, 'r-', linewidth=2)
axes[1].set_xlabel('Concurrence C', fontsize=12)
axes[1].set_ylabel('Entanglement of Formation E_F', fontsize=12)
axes[1].set_title('E_F vs Concurrence (Two Qubits)', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('concurrence_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n8. Plots saved to 'concurrence_analysis.png'")

# 9. Entanglement sudden death in Werner states
print("\n9. Werner State Entanglement Threshold")
print("   Werner state entangled iff p > 1/3")
print(f"   Threshold: p = {1/3:.4f}")
print(f"   At threshold, C = {max(0, (3*(1/3) - 1)/2):.4f}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula | Notes |
|----------|---------|-------|
| Pure state C | $C = 2\|c_{00}c_{11} - c_{01}c_{10}\|$ | Two-qubit |
| Spin-flip | $\|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)\|\psi^*\rangle$ | Complex conjugate + flip |
| Wootters formula | $C = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$ | Mixed states |
| Werner state | $C = \max(0, (3p-1)/2)$ | Threshold at p = 1/3 |
| C to E_F | $E_F = h((1+\sqrt{1-C^2})/2)$ | Analytical formula |

### Key Takeaways
1. **Concurrence** is a computable entanglement measure for two qubits
2. **C = 1** for maximally entangled states, **C = 0** for separable states
3. **Wootters formula** works for any mixed two-qubit state
4. **Spin-flip operation** is central to the definition
5. **Werner states** are entangled iff $p > 1/3$
6. **Direct connection** to entanglement of formation via $E_F(C)$

---

## Daily Checklist

- [ ] I can compute concurrence for pure two-qubit states
- [ ] I understand the spin-flip operation
- [ ] I can apply the Wootters formula for mixed states
- [ ] I know the Werner state entanglement threshold
- [ ] I can relate concurrence to entanglement of formation
- [ ] I understand concurrence as an LOCC monotone

---

*Next: Day 550 — Negativity*
