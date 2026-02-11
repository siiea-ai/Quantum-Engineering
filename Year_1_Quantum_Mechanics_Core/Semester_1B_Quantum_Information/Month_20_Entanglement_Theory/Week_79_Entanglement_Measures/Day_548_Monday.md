# Day 548: Entropy of Entanglement

## Overview
**Day 548** | Week 79, Day 2 | Year 1, Month 20 | Pure State Entanglement Measure

Today we study the entropy of entanglement—the unique measure of entanglement for pure bipartite states, directly connecting the Schmidt decomposition to von Neumann entropy.

---

## Learning Objectives
1. Define entropy of entanglement for pure states
2. Derive E from Schmidt decomposition
3. Prove E = S(ρ_A) = S(ρ_B) for pure states
4. Compute entropy of entanglement for common states
5. Understand maximally entangled states via entropy
6. Apply entropy of entanglement in quantum protocols

---

## Core Content

### Entropy of Entanglement Definition

For a pure bipartite state $|\psi\rangle_{AB}$:

$$\boxed{E(|\psi\rangle) = S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)}$$

where $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ is the reduced density matrix.

**Key property:** $E(|\psi\rangle) = S(\rho_A) = S(\rho_B)$ for pure states.

### Connection to Schmidt Decomposition

Any pure bipartite state can be written as:
$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \sqrt{\lambda_i} |a_i\rangle_A |b_i\rangle_B$$

where:
- $\{|a_i\rangle\}$ and $\{|b_i\rangle\}$ are orthonormal bases
- $\lambda_i \geq 0$ are Schmidt coefficients with $\sum_i \lambda_i = 1$
- $r$ is the Schmidt rank

**Reduced density matrices:**
$$\rho_A = \sum_i \lambda_i |a_i\rangle\langle a_i|, \quad \rho_B = \sum_i \lambda_i |b_i\rangle\langle b_i|$$

### Entropy of Entanglement Formula

$$\boxed{E(|\psi\rangle) = -\sum_{i=1}^{r} \lambda_i \log_2 \lambda_i = H(\{\lambda_i\})}$$

where $H(\{\lambda_i\})$ is the Shannon entropy of the Schmidt spectrum.

### Properties of Entropy of Entanglement

**1. Non-negativity:** $E(|\psi\rangle) \geq 0$

**2. Zero for product states:**
$$E(|\psi\rangle) = 0 \iff |\psi\rangle = |a\rangle \otimes |b\rangle$$

**3. Maximum value:**
$$E_{\max} = \log_2(\min(d_A, d_B))$$
achieved by maximally entangled states.

**4. Invariance under local unitaries:**
$$E((U_A \otimes U_B)|\psi\rangle) = E(|\psi\rangle)$$

**5. Continuity:** Small changes in state → small changes in E

### Maximally Entangled States

A state is **maximally entangled** when:
$$\lambda_i = \frac{1}{d} \quad \text{for all } i = 1, \ldots, d$$

where $d = \min(d_A, d_B)$.

**General form:**
$$|\Phi_d^+\rangle = \frac{1}{\sqrt{d}} \sum_{i=0}^{d-1} |ii\rangle$$

**Entropy of entanglement:** $E = \log_2 d$ ebits

### Two-Qubit Examples

**Product state** $|00\rangle$:
- Schmidt coefficients: $\lambda_1 = 1$
- $E = -1 \cdot \log_2(1) = 0$

**Bell state** $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
- Schmidt coefficients: $\lambda_1 = \lambda_2 = 1/2$
- $E = -2 \cdot \frac{1}{2} \log_2(\frac{1}{2}) = 1$ ebit

**Partially entangled state** $|\psi\rangle = \sqrt{p}|00\rangle + \sqrt{1-p}|11\rangle$:
- Schmidt coefficients: $\lambda_1 = p$, $\lambda_2 = 1-p$
- $E = -p\log_2 p - (1-p)\log_2(1-p) = h(p)$

where $h(p)$ is the **binary entropy function**.

### Binary Entropy Function

$$\boxed{h(p) = -p \log_2 p - (1-p) \log_2(1-p)}$$

**Properties:**
- $h(0) = h(1) = 0$
- $h(1/2) = 1$ (maximum)
- $h(p) = h(1-p)$ (symmetric)
- Concave function

### Schmidt Rank and Entanglement

| Schmidt Rank | Example | Entanglement |
|--------------|---------|--------------|
| 1 | Product state | None |
| 2 | Bell state | Up to 1 ebit |
| d | d-dimensional MES | Up to $\log_2 d$ ebits |

### Proof: S(ρ_A) = S(ρ_B)

For pure state $|\psi\rangle_{AB}$:

**Claim:** The reduced density matrices $\rho_A$ and $\rho_B$ have the same non-zero eigenvalues.

**Proof:**
From Schmidt decomposition: $|\psi\rangle = \sum_i \sqrt{\lambda_i}|a_i\rangle|b_i\rangle$

$\rho_A = \sum_i \lambda_i |a_i\rangle\langle a_i|$ has eigenvalues $\{\lambda_i\}$

$\rho_B = \sum_i \lambda_i |b_i\rangle\langle b_i|$ has eigenvalues $\{\lambda_i\}$

Since both have the same eigenvalue spectrum:
$$S(\rho_A) = -\sum_i \lambda_i \log_2 \lambda_i = S(\rho_B)$$ ∎

### Entanglement Monotonicity

For pure states undergoing LOCC (Local Operations and Classical Communication):
$$E(|\psi\rangle) \geq \sum_i p_i E(|\phi_i\rangle)$$

Entanglement cannot increase on average under LOCC.

### Physical Interpretation

**E = 1 ebit means:**
- One Bell pair worth of entanglement
- Can teleport one qubit
- Can transmit 2 classical bits via superdense coding

**E = n ebits means:**
- Equivalent to n Bell pairs for asymptotic protocols
- Resource for quantum communication

---

## Worked Examples

### Example 1: Three-Level Entangled State
Compute E for $|\psi\rangle = \frac{1}{\sqrt{6}}(|00\rangle + \sqrt{2}|11\rangle + \sqrt{3}|22\rangle)$.

**Solution:**
The state is already in Schmidt form with:
$$\lambda_1 = \frac{1}{6}, \quad \lambda_2 = \frac{2}{6} = \frac{1}{3}, \quad \lambda_3 = \frac{3}{6} = \frac{1}{2}$$

Verify normalization: $\frac{1}{6} + \frac{1}{3} + \frac{1}{2} = \frac{1+2+3}{6} = 1$ ✓

Entropy of entanglement:
$$E = -\frac{1}{6}\log_2\frac{1}{6} - \frac{1}{3}\log_2\frac{1}{3} - \frac{1}{2}\log_2\frac{1}{2}$$
$$= \frac{1}{6}\log_2 6 + \frac{1}{3}\log_2 3 + \frac{1}{2}\log_2 2$$
$$= \frac{1}{6}(2.585) + \frac{1}{3}(1.585) + \frac{1}{2}(1)$$
$$= 0.431 + 0.528 + 0.5 = 1.459 \text{ ebits}$$ ∎

### Example 2: Non-Schmidt Form State
Find E for $|\psi\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$.

**Solution:**
First, write as $|\psi\rangle = |+\rangle_A \otimes |+\rangle_B$ where $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$.

This is a **product state**!

Schmidt decomposition: $|\psi\rangle = 1 \cdot |+\rangle|+\rangle$

Only one Schmidt coefficient: $\lambda_1 = 1$

$$E = -1 \cdot \log_2(1) = 0$$ ∎

### Example 3: Entropy via Reduced Density Matrix
Compute E for $|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \sqrt{\frac{2}{3}}|11\rangle$ using $\rho_A$.

**Solution:**
$$|\psi\rangle\langle\psi| = \frac{1}{3}|00\rangle\langle 00| + \frac{\sqrt{2}}{3}|00\rangle\langle 11| + \frac{\sqrt{2}}{3}|11\rangle\langle 00| + \frac{2}{3}|11\rangle\langle 11|$$

Partial trace over B:
$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|) = \frac{1}{3}|0\rangle\langle 0| + \frac{2}{3}|1\rangle\langle 1|$$
$$= \begin{pmatrix} 1/3 & 0 \\ 0 & 2/3 \end{pmatrix}$$

Eigenvalues: $\lambda_1 = 1/3$, $\lambda_2 = 2/3$

$$E = S(\rho_A) = -\frac{1}{3}\log_2\frac{1}{3} - \frac{2}{3}\log_2\frac{2}{3}$$
$$= \frac{1}{3}\log_2 3 + \frac{2}{3}(\log_2 3 - 1)$$
$$= \log_2 3 - \frac{2}{3} = 1.585 - 0.667 = 0.918 \text{ ebits}$$ ∎

---

## Practice Problems

### Problem 1: Entropy Bounds
For a state $|\psi\rangle \in \mathbb{C}^3 \otimes \mathbb{C}^5$:
(a) What is the maximum possible entropy of entanglement?
(b) What Schmidt coefficients achieve this maximum?
(c) Write the maximally entangled state explicitly.

### Problem 2: Equal Entanglement
Find all two-qubit pure states with $E = 0.5$ ebits. Characterize the family of solutions.

### Problem 3: Entropy Ordering
Rank the following states by entropy of entanglement:
(a) $|\psi_1\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
(b) $|\psi_2\rangle = \frac{1}{2}|00\rangle + \frac{\sqrt{3}}{2}|11\rangle$
(c) $|\psi_3\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)$

### Problem 4: Product State Verification
Prove that $|\chi\rangle = \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$ is a product state by:
(a) Factoring explicitly
(b) Computing the entropy of entanglement

### Problem 5: GHZ State
For the three-qubit GHZ state $|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$:
(a) Compute E for the bipartition A|BC
(b) Compute E for the bipartition AB|C
(c) Compare and interpret

---

## Computational Lab

```python
"""Day 548: Entropy of Entanglement"""
import numpy as np
from scipy.linalg import svd, sqrtm

def schmidt_decomposition(state, dim_A, dim_B):
    """
    Compute Schmidt decomposition of bipartite pure state.
    Returns Schmidt coefficients and rank.
    """
    # Reshape state vector into matrix
    psi_matrix = state.reshape(dim_A, dim_B)

    # SVD gives Schmidt decomposition
    U, s, Vh = svd(psi_matrix)

    # Schmidt coefficients are singular values squared
    # (s are already the sqrt(lambda_i))
    schmidt_coeffs = s**2

    # Filter out numerical zeros
    schmidt_coeffs = schmidt_coeffs[schmidt_coeffs > 1e-12]

    return schmidt_coeffs, len(schmidt_coeffs)

def entropy_of_entanglement(state, dim_A, dim_B):
    """
    Compute entropy of entanglement E(|ψ⟩) = S(ρ_A).
    """
    schmidt_coeffs, rank = schmidt_decomposition(state, dim_A, dim_B)

    # E = -Σ λᵢ log₂ λᵢ
    entropy = -np.sum(schmidt_coeffs * np.log2(schmidt_coeffs))
    return entropy

def entropy_via_reduced_dm(state, dim_A, dim_B):
    """
    Compute entropy of entanglement via reduced density matrix.
    """
    # Form density matrix
    rho_AB = np.outer(state, state.conj())

    # Partial trace over B
    rho_A = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B).trace(axis1=1, axis2=3)

    # Von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy, rho_A

def binary_entropy(p):
    """Binary entropy function h(p)"""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

# ============================================
# Test Cases
# ============================================
print("=" * 60)
print("ENTROPY OF ENTANGLEMENT CALCULATIONS")
print("=" * 60)

# 1. Product state |00⟩
print("\n1. Product State |00⟩")
psi_00 = np.array([1, 0, 0, 0], dtype=complex)
E = entropy_of_entanglement(psi_00, 2, 2)
print(f"   E(|00⟩) = {E:.4f} ebits")
print(f"   Expected: 0.0000 ebits")

# 2. Bell state |Φ⁺⟩
print("\n2. Bell State |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
E = entropy_of_entanglement(phi_plus, 2, 2)
print(f"   E(|Φ⁺⟩) = {E:.4f} ebits")
print(f"   Expected: 1.0000 ebits")

# 3. Partially entangled state
print("\n3. Partially Entangled: √p|00⟩ + √(1-p)|11⟩")
print("   p       E (ebits)   h(p)")
print("   " + "-" * 35)
for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    if p == 0:
        psi = np.array([0, 0, 0, 1], dtype=complex)
    elif p == 1:
        psi = np.array([1, 0, 0, 0], dtype=complex)
    else:
        psi = np.array([np.sqrt(p), 0, 0, np.sqrt(1-p)], dtype=complex)
    E = entropy_of_entanglement(psi, 2, 2)
    h = binary_entropy(p)
    print(f"   {p:.2f}    {E:.4f}      {h:.4f}")

# 4. Three-level system
print("\n4. Three-Level System: (|00⟩ + √2|11⟩ + √3|22⟩)/√6")
psi_3level = np.zeros(9, dtype=complex)
psi_3level[0] = 1/np.sqrt(6)        # |00⟩
psi_3level[4] = np.sqrt(2/6)        # |11⟩
psi_3level[8] = np.sqrt(3/6)        # |22⟩
E = entropy_of_entanglement(psi_3level, 3, 3)
print(f"   E = {E:.4f} ebits")

# Verify with reduced DM
E_via_rho, rho_A = entropy_via_reduced_dm(psi_3level, 3, 3)
print(f"   Via ρ_A: E = {E_via_rho:.4f} ebits")
print(f"   ρ_A eigenvalues: {np.sort(np.linalg.eigvalsh(rho_A))[::-1]}")

# 5. Maximally entangled qutrit state
print("\n5. Maximally Entangled Qutrit: (|00⟩ + |11⟩ + |22⟩)/√3")
psi_mes = np.zeros(9, dtype=complex)
psi_mes[0] = psi_mes[4] = psi_mes[8] = 1/np.sqrt(3)
E = entropy_of_entanglement(psi_mes, 3, 3)
print(f"   E = {E:.4f} ebits")
print(f"   Maximum possible: log₂(3) = {np.log2(3):.4f} ebits")

# 6. Schmidt coefficients
print("\n6. Schmidt Coefficients Analysis")
states = {
    'Product |00⟩': np.array([1, 0, 0, 0], dtype=complex),
    'Bell |Φ⁺⟩': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'Partial (p=0.25)': np.array([0.5, 0, 0, np.sqrt(3)/2], dtype=complex),
}

for name, state in states.items():
    coeffs, rank = schmidt_decomposition(state, 2, 2)
    E = entropy_of_entanglement(state, 2, 2)
    print(f"   {name}:")
    print(f"      Schmidt coefficients: {coeffs}")
    print(f"      Schmidt rank: {rank}")
    print(f"      E = {E:.4f} ebits\n")

# 7. Entanglement entropy for different bipartitions
print("7. GHZ State Bipartitions")
# GHZ = (|000⟩ + |111⟩)/√2
ghz = np.zeros(8, dtype=complex)
ghz[0] = ghz[7] = 1/np.sqrt(2)

# A|BC bipartition (2 x 4)
E_A_BC = entropy_of_entanglement(ghz, 2, 4)
print(f"   E(A|BC) = {E_A_BC:.4f} ebits")

# AB|C bipartition (4 x 2)
E_AB_C = entropy_of_entanglement(ghz, 4, 2)
print(f"   E(AB|C) = {E_AB_C:.4f} ebits")

# 8. Verify S(ρ_A) = S(ρ_B)
print("\n8. Verification: S(ρ_A) = S(ρ_B)")
psi_test = np.array([0.3, 0.1, 0.4, np.sqrt(1-0.09-0.01-0.16)], dtype=complex)
psi_test = psi_test / np.linalg.norm(psi_test)

rho_AB = np.outer(psi_test, psi_test.conj())
rho_A = rho_AB.reshape(2,2,2,2).trace(axis1=1, axis2=3)
rho_B = rho_AB.reshape(2,2,2,2).trace(axis1=0, axis2=2)

def von_neumann_entropy(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return -np.sum(eigs * np.log2(eigs))

S_A = von_neumann_entropy(rho_A)
S_B = von_neumann_entropy(rho_B)
print(f"   S(ρ_A) = {S_A:.6f}")
print(f"   S(ρ_B) = {S_B:.6f}")
print(f"   Difference: {abs(S_A - S_B):.2e}")

# 9. Entanglement as a function of state parameter
print("\n9. Entanglement Curve for |ψ(θ)⟩ = cos(θ)|00⟩ + sin(θ)|11⟩")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

theta_vals = np.linspace(0, np.pi/2, 50)
E_vals = []

for theta in theta_vals:
    psi_theta = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
    E_vals.append(entropy_of_entanglement(psi_theta, 2, 2))

plt.figure(figsize=(10, 6))
plt.plot(theta_vals * 180/np.pi, E_vals, 'b-', linewidth=2)
plt.xlabel('θ (degrees)', fontsize=12)
plt.ylabel('E (ebits)', fontsize=12)
plt.title('Entropy of Entanglement vs State Parameter', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', label='Maximum (1 ebit)')
plt.axvline(x=45, color='g', linestyle='--', label='θ = 45° (Bell state)')
plt.legend()
plt.savefig('entropy_of_entanglement.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Plot saved to 'entropy_of_entanglement.png'")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula | Notes |
|----------|---------|-------|
| Entropy of Entanglement | $E = -\sum_i \lambda_i \log_2 \lambda_i$ | Schmidt coefficients |
| Via reduced state | $E = S(\rho_A) = S(\rho_B)$ | Pure states only |
| Binary entropy | $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$ | Two-level systems |
| Maximum E (d-level) | $E_{\max} = \log_2 d$ | Maximally entangled |

### Key Takeaways
1. **Entropy of entanglement** is THE unique measure for pure bipartite states
2. **Schmidt coefficients** completely determine entanglement content
3. **E = 0** if and only if state is a product state
4. **E = 1 ebit** for Bell states (maximally entangled qubits)
5. **Local unitaries** preserve entropy of entanglement
6. **Equal entropies:** $S(\rho_A) = S(\rho_B)$ always for pure states

---

## Daily Checklist

- [ ] I can define entropy of entanglement
- [ ] I understand the connection to Schmidt decomposition
- [ ] I can compute E from Schmidt coefficients
- [ ] I can compute E via reduced density matrix
- [ ] I understand what "maximally entangled" means
- [ ] I can use binary entropy for two-qubit states

---

*Next: Day 549 — Concurrence*
