# Day 666: Months 19-20 Review - Density Matrices and Entanglement

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Review Scope

**Month 19: Density Matrices and Mixed States (Days 505-532)**
- Week 73: Pure vs Mixed States
- Week 74: Density Matrix Properties
- Week 75: Partial Trace and Reduced States
- Week 76: Applications and Month Review

**Month 20: Entanglement Theory (Days 533-560)**
- Week 77: Bell States and EPR
- Week 78: Entanglement Measures
- Week 79: Entanglement in Practice
- Week 80: Month Review

---

## Core Concepts: Density Matrices

### 1. Definition and Properties

**Density operator:**
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

**Required properties:**
1. Hermitian: $\rho = \rho^\dagger$
2. Positive semi-definite: $\rho \geq 0$
3. Trace one: $\text{Tr}(\rho) = 1$

### 2. Pure vs Mixed States

| Property | Pure | Mixed |
|----------|------|-------|
| Definition | $\rho = \|\psi\rangle\langle\psi\|$ | $\rho = \sum_i p_i \|\psi_i\rangle\langle\psi_i\|$ |
| Purity | $\text{Tr}(\rho^2) = 1$ | $\text{Tr}(\rho^2) < 1$ |
| Entropy | $S(\rho) = 0$ | $S(\rho) > 0$ |
| Eigenvalues | One 1, rest 0 | Multiple non-zero |

### 3. Bloch Representation (Single Qubit)

$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$

where $\vec{r} = (r_x, r_y, r_z)$ with $|\vec{r}| \leq 1$.

- Pure states: $|\vec{r}| = 1$ (surface of Bloch sphere)
- Mixed states: $|\vec{r}| < 1$ (interior)
- Maximally mixed: $\vec{r} = 0$ (center)

### 4. Partial Trace

For bipartite system $\rho_{AB}$:
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)$$

**Key property:** If $\rho_{AB}$ is pure and entangled, then $\rho_A$ and $\rho_B$ are mixed.

---

## Core Concepts: Entanglement

### 5. Bell States

$$|\Phi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)$$
$$|\Psi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

**Properties:**
- Maximally entangled
- Orthonormal basis for 2-qubit Hilbert space
- Reduced states are maximally mixed: $\rho_A = \rho_B = I/2$

### 6. Entanglement Measures

**For pure bipartite states:**

$$\boxed{E(|\psi\rangle_{AB}) = S(\rho_A) = S(\rho_B)}$$

where $S(\rho) = -\text{Tr}(\rho \log_2 \rho)$ is von Neumann entropy.

**Concurrence (2 qubits):**
$$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$

where $\lambda_i$ are eigenvalues of $\sqrt{\sqrt{\rho}\tilde{\rho}\sqrt{\rho}}$ in decreasing order.

**Entanglement of formation:**
$$E_F(\rho) = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$$

where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$.

### 7. Separability Criteria

**Separable states:**
$$\rho_{AB} = \sum_i p_i \rho_A^{(i)} \otimes \rho_B^{(i)}$$

**PPT criterion (Peres-Horodecki):**
- If $\rho_{AB}^{T_B} \geq 0$, state may be separable
- For 2×2 and 2×3 systems: PPT ⟺ separable

### 8. Applications

**Quantum teleportation:** Use Bell state to transfer quantum state

**Superdense coding:** Send 2 classical bits using 1 qubit + entanglement

**CHSH inequality:** $|S| \leq 2$ classically, $|S| \leq 2\sqrt{2}$ quantum

---

## Integration with Later Topics

### Connection to Open Systems (Month 21)

When system interacts with environment:
$$\rho_{SE}(t) = U(t)\rho_S(0) \otimes \rho_E(0) U^\dagger(t)$$

Reduced system state:
$$\rho_S(t) = \text{Tr}_E[\rho_{SE}(t)]$$

**Key insight:** Entanglement with environment causes decoherence!

### Connection to Quantum Channels (Months 23-24)

The map $\rho_S(0) \to \rho_S(t)$ is a quantum channel:
$$\mathcal{E}(\rho) = \text{Tr}_E[U(\rho \otimes \rho_E)U^\dagger]$$

This is the **Stinespring dilation** connecting channels to unitary evolution.

---

## Practice Problems

### Problem 1: Purity Calculation
Calculate the purity of $\rho = \frac{1}{4}|0\rangle\langle 0| + \frac{3}{4}|1\rangle\langle 1|$.

**Solution:**
$$\rho^2 = \frac{1}{16}|0\rangle\langle 0| + \frac{9}{16}|1\rangle\langle 1|$$
$$\text{Tr}(\rho^2) = \frac{1}{16} + \frac{9}{16} = \frac{10}{16} = \frac{5}{8}$$

### Problem 2: Partial Trace
Find $\rho_A$ for $|\psi\rangle_{AB} = \frac{1}{\sqrt{3}}|00\rangle + \sqrt{\frac{2}{3}}|11\rangle$.

**Solution:**
$$\rho_{AB} = \frac{1}{3}|00\rangle\langle 00| + \frac{\sqrt{2}}{3}|00\rangle\langle 11| + \frac{\sqrt{2}}{3}|11\rangle\langle 00| + \frac{2}{3}|11\rangle\langle 11|$$

$$\rho_A = \text{Tr}_B(\rho_{AB}) = \frac{1}{3}|0\rangle\langle 0| + \frac{2}{3}|1\rangle\langle 1|$$

### Problem 3: Entanglement Entropy
Calculate the entanglement entropy for Problem 2.

**Solution:**
$$S(\rho_A) = -\frac{1}{3}\log_2\frac{1}{3} - \frac{2}{3}\log_2\frac{2}{3}$$
$$= \frac{1}{3}\log_2 3 + \frac{2}{3}\log_2\frac{3}{2}$$
$$= \frac{1}{3}\log_2 3 + \frac{2}{3}(\log_2 3 - 1) = \log_2 3 - \frac{2}{3} \approx 0.918$$

### Problem 4: Bell State Verification
Show that the reduced density matrix of $|\Phi^+\rangle$ is $I/2$.

**Solution:**
$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}(|00\rangle + |11\rangle)(\langle 00| + \langle 11|)$$
$$= \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

$$\rho_A = \text{Tr}_B = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

---

## Computational Lab

```python
"""Day 666: Months 19-20 Review - Density Matrices and Entanglement"""

import numpy as np
from scipy.linalg import sqrtm, logm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(A, B):
    return np.kron(A, B)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("Months 19-20 Review: Density Matrices and Entanglement")
print("=" * 60)

# ============================================
# Part 1: Density Matrix Properties
# ============================================
print("\nPART 1: Density Matrix Properties")
print("-" * 40)

# Create various density matrices
rho_pure = ket_0 @ ket_0.conj().T  # |0⟩⟨0|
rho_mixed = 0.25 * ket_0 @ ket_0.conj().T + 0.75 * ket_1 @ ket_1.conj().T
rho_max_mixed = I / 2

def analyze_density_matrix(rho, name):
    """Analyze properties of a density matrix."""
    purity = np.real(np.trace(rho @ rho))
    eigenvalues = np.real(np.linalg.eigvalsh(rho))

    # Von Neumann entropy
    eigs = eigenvalues[eigenvalues > 1e-10]
    entropy = -np.sum(eigs * np.log2(eigs))

    # Bloch vector (for single qubit)
    rx = np.real(np.trace(X @ rho))
    ry = np.real(np.trace(Y @ rho))
    rz = np.real(np.trace(Z @ rho))
    bloch_length = np.sqrt(rx**2 + ry**2 + rz**2)

    print(f"\n{name}:")
    print(f"  Purity: {purity:.4f}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Bloch vector: ({rx:.2f}, {ry:.2f}, {rz:.2f}), |r| = {bloch_length:.4f}")

analyze_density_matrix(rho_pure, "|0⟩⟨0|")
analyze_density_matrix(rho_mixed, "0.25|0⟩⟨0| + 0.75|1⟩⟨1|")
analyze_density_matrix(rho_max_mixed, "I/2")

# ============================================
# Part 2: Partial Trace
# ============================================
print("\n" + "=" * 60)
print("PART 2: Partial Trace and Entanglement")
print("-" * 40)

def partial_trace_B(rho_AB, dim_A=2, dim_B=2):
    """Compute partial trace over B."""
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    for j in range(dim_B):
        # Projector |j⟩⟨j| on B
        proj = np.zeros((dim_B, dim_B), dtype=complex)
        proj[j, j] = 1
        rho_A += np.trace(rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)[:, j, :, j].reshape(dim_A, dim_A))
    return rho_A

# Alternative cleaner implementation
def partial_trace_B_v2(rho_AB, dim_A=2, dim_B=2):
    """Partial trace over B using reshape."""
    reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(reshaped, axis1=1, axis2=3)

# Bell states
Phi_plus = (tensor(ket_0, ket_0) + tensor(ket_1, ket_1)) / np.sqrt(2)
Psi_minus = (tensor(ket_0, ket_1) - tensor(ket_1, ket_0)) / np.sqrt(2)

# Product state for comparison
product = tensor(ket_0, ket_1)

def analyze_bipartite(psi, name):
    """Analyze bipartite pure state."""
    rho_AB = psi @ psi.conj().T
    rho_A = partial_trace_B_v2(rho_AB)

    purity_A = np.real(np.trace(rho_A @ rho_A))
    eigs = np.real(np.linalg.eigvalsh(rho_A))
    eigs = eigs[eigs > 1e-10]
    entropy_A = -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0

    print(f"\n{name}:")
    print(f"  ρ_A eigenvalues: {np.real(np.linalg.eigvalsh(rho_A))}")
    print(f"  Purity of ρ_A: {purity_A:.4f}")
    print(f"  Entanglement entropy: {entropy_A:.4f} ebits")

analyze_bipartite(Phi_plus, "|Φ+⟩ (maximally entangled)")
analyze_bipartite(Psi_minus, "|Ψ-⟩ (maximally entangled)")
analyze_bipartite(product, "|01⟩ (product state)")

# Non-maximally entangled
alpha = 1/np.sqrt(3)
beta = np.sqrt(2/3)
non_max = alpha * tensor(ket_0, ket_0) + beta * tensor(ket_1, ket_1)
analyze_bipartite(non_max, f"α|00⟩ + β|11⟩ (α={alpha:.3f})")

# ============================================
# Part 3: Concurrence
# ============================================
print("\n" + "=" * 60)
print("PART 3: Concurrence for 2-Qubit States")
print("-" * 40)

def concurrence(rho):
    """Calculate concurrence for 2-qubit density matrix."""
    # Spin-flip matrix
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y_Y = tensor(sigma_y, sigma_y)

    # rho_tilde = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
    rho_tilde = Y_Y @ rho.conj() @ Y_Y

    # R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
    sqrt_rho = sqrtm(rho)
    R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)

    # Eigenvalues in decreasing order
    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]

    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    return C

# Test on various states
rho_phi_plus = Phi_plus @ Phi_plus.conj().T
rho_product = product @ product.conj().T
rho_non_max = non_max @ non_max.conj().T

print(f"\nConcurrence of |Φ+⟩: {concurrence(rho_phi_plus):.4f}")
print(f"Concurrence of |01⟩: {concurrence(rho_product):.4f}")
print(f"Concurrence of non-max entangled: {concurrence(rho_non_max):.4f}")

# Werner state: p|Φ+⟩⟨Φ+| + (1-p)I/4
print("\nWerner state concurrence:")
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    werner = p * rho_phi_plus + (1-p) * np.eye(4)/4
    C = concurrence(werner)
    print(f"  p = {p:.2f}: C = {C:.4f}")

print("\n" + "=" * 60)
print("Review Complete!")
```

---

## Summary

### Month 19 Key Points
1. Density matrices describe mixed states
2. Purity $\text{Tr}(\rho^2)$ distinguishes pure from mixed
3. Bloch sphere represents single-qubit states
4. Partial trace gives reduced density matrices

### Month 20 Key Points
1. Bell states are maximally entangled
2. Entanglement entropy quantifies pure-state entanglement
3. Concurrence works for mixed 2-qubit states
4. PPT criterion detects entanglement

### Connection to Later Months
- Open systems (Month 21): Environment causes mixing
- Channels (Months 23-24): Partial trace over environment
- Error correction: Entanglement protects information

---

## Preview: Day 667

Tomorrow: **Month 21 Review** - Open quantum systems, master equations, and decoherence!
