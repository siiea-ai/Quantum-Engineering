# Day 123: Composite Quantum Systems — Partial Trace and Reduced Density Matrices

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Partial Trace & Reduced States |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and compute the partial trace
2. Derive reduced density matrices from composite systems
3. Understand the connection between entanglement and mixed reduced states
4. Compute entanglement entropy
5. Apply partial trace to analyze quantum correlations
6. Distinguish separable from entangled states using reduced density matrices

---

## 📚 Required Reading

### Primary Text
- **Nielsen & Chuang, Section 2.4.3**: Reduced density operator

### Secondary
- **Preskill's Lecture Notes, Chapter 2**: Density matrices and entanglement
- **Wilde, "Quantum Information Theory," Chapter 3**: Partial trace

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Describing Subsystems

**Problem:** Given a composite system AB in state ρ_AB, how do we describe subsystem A alone?

**Classical:** If we don't know B, just ignore it (marginalize probability)
**Quantum:** We need the **partial trace** to "trace out" system B

### 2. The Trace Operation (Review)

For operator A on space V with orthonormal basis {|i⟩}:
$$\text{tr}(A) = \sum_i \langle i|A|i\rangle$$

**Key properties:**
- tr(AB) = tr(BA) (cyclic)
- tr(A + B) = tr(A) + tr(B) (linear)
- tr(cA) = c·tr(A)
- tr(A ⊗ B) = tr(A)·tr(B)

### 3. Definition of Partial Trace

**Partial trace over B:** For ρ_AB on ℋ_A ⊗ ℋ_B:

$$\boxed{\rho_A = \text{tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)}$$

where {|j⟩_B} is any orthonormal basis for ℋ_B.

**In components:** If ρ_AB = Σ ρ_{ik,jl} |i⟩⟨k|_A ⊗ |j⟩⟨l|_B, then:
$$(\rho_A)_{ik} = \sum_j \rho_{ij,kj}$$

**Intuition:** Sum over diagonal elements in the B index.

### 4. Partial Trace of Product States

For product state ρ_AB = ρ_A ⊗ ρ_B:
$$\text{tr}_B(\rho_A \otimes \rho_B) = \rho_A \cdot \text{tr}(\rho_B) = \rho_A$$

(since tr(ρ_B) = 1 for any density matrix)

### 5. Partial Trace of Pure Entangled States

**Example: Bell state** |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

Density matrix:
$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

**Partial trace over B:**
$$\rho_A = \text{tr}_B(|\Phi^+\rangle\langle\Phi^+|)$$

Using basis {|0⟩, |1⟩} for B:
$$\rho_A = \langle 0|_B |\Phi^+\rangle\langle\Phi^+| |0\rangle_B + \langle 1|_B |\Phi^+\rangle\langle\Phi^+| |1\rangle_B$$

Computing:
- ⟨0|_B |Φ⁺⟩ = |0⟩_A/√2
- ⟨1|_B |Φ⁺⟩ = |1⟩_A/√2

$$\rho_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{I}{2}$$

**Key result:** The reduced state of an entangled pure state is MIXED!

### 6. Criterion for Entanglement

**Theorem:** For a pure state |ψ⟩_AB:
- |ψ⟩ is a product state ⟺ ρ_A = tr_B(|ψ⟩⟨ψ|) is pure
- |ψ⟩ is entangled ⟺ ρ_A is mixed

**Purity of reduced state:**
$$\text{tr}(\rho_A^2) = 1 \iff \text{product state}$$
$$\text{tr}(\rho_A^2) < 1 \iff \text{entangled}$$

### 7. Entanglement Entropy

**Von Neumann entropy:**
$$S(\rho) = -\text{tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where λᵢ are eigenvalues of ρ.

**Entanglement entropy:** For pure state |ψ⟩_AB:
$$\boxed{E(|\psi\rangle) = S(\rho_A) = S(\rho_B)}$$

**Properties:**
- E = 0 ⟺ product state
- E = log₂(d) for maximally entangled state (d = min(dim_A, dim_B))
- E(|Φ⁺⟩) = 1 ebit (maximally entangled for 2 qubits)

### 8. Schmidt Decomposition Revisited

For |ψ⟩ = Σᵢ λᵢ |aᵢ⟩|bᵢ⟩ (Schmidt form):

$$\rho_A = \sum_i \lambda_i^2 |a_i\rangle\langle a_i|$$

**Connection:**
- Schmidt coefficients λᵢ = √(eigenvalues of ρ_A)
- E = -Σᵢ λᵢ² log₂(λᵢ²)

---

## 🔬 Quantum Mechanics Connection

### Quantum Correlations

**Entanglement creates correlations:**
For |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
- Measure A: get 0 or 1 with 50% each
- Measure B: get same result as A with 100% certainty!
- But neither A nor B alone has definite value

**No classical explanation:** These correlations violate Bell inequalities.

### Decoherence and Open Systems

**System + Environment:**
Total state: |ψ⟩_SE (pure)
System alone: ρ_S = tr_E(|ψ⟩⟨ψ|) (generally mixed!)

**Decoherence:** As system becomes entangled with environment, the reduced state ρ_S becomes increasingly mixed.

### Quantum Discord (Preview)

**Classical correlations:** Can be explained by shared randomness
**Quantum correlations:** Include entanglement AND quantum discord
**Discord:** Captures quantum correlations beyond entanglement

### Monogamy of Entanglement

**Principle:** A qubit maximally entangled with one system cannot be entangled with another.

For three qubits A, B, C:
$$E_{AB} + E_{AC} \leq E_{A(BC)}$$

This is why quantum cryptography is secure!

---

## ✏️ Worked Examples

### Example 1: Partial Trace Calculation

Compute ρ_A for |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3.

**Step 1:** Write density matrix
$$\rho_{AB} = |\psi\rangle\langle\psi| = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**Step 2:** Partial trace (sum 2×2 blocks on diagonal)
The matrix in block form (A index labels rows/cols of blocks, B index within blocks):

$$\rho_{AB} = \frac{1}{3}\begin{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} & \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix} \\ \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} & \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \end{pmatrix}$$

$$\rho_A = \text{tr}_B(\rho_{AB}) = \frac{1}{3}\begin{pmatrix} 1+1 & 1+0 \\ 1+0 & 1+0 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 3:** Check properties
- tr(ρ_A) = (2+1)/3 = 1 ✓
- Hermitian? Yes ✓
- Eigenvalues: λ = (3 ± √5)/6 ≈ 0.873, 0.127 (both positive ✓)

**Step 4:** Purity and entropy
- Purity: tr(ρ_A²) = (4+1+1+1)/9 = 7/9 ≈ 0.778 < 1 (mixed!)
- E ≈ 0.55 ebits

### Example 2: Verifying ρ_A = ρ_B for Bell State

For |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, show ρ_A = ρ_B = I/2.

**ρ_A (trace out B):**
Using Schmidt form |Φ⁺⟩ = (1/√2)|0⟩|0⟩ + (1/√2)|1⟩|1⟩:
$$\rho_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{I}{2}$$

**ρ_B (trace out A):**
By symmetry (or explicit calculation): ρ_B = I/2 ✓

**Entanglement entropy:**
$$E = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1 \text{ ebit}$$

### Example 3: Product State Check

For |ψ⟩ = |+⟩|0⟩ = (|00⟩ + |10⟩)/√2, compute ρ_A.

$$\rho_{AB} = |+\rangle\langle+| \otimes |0\rangle\langle 0|$$

**Partial trace:**
$$\rho_A = \text{tr}_B(|+\rangle\langle+| \otimes |0\rangle\langle 0|) = |+\rangle\langle+| \cdot \text{tr}(|0\rangle\langle 0|) = |+\rangle\langle+|$$

**Check:**
- ρ_A = |+⟩⟨+| is a pure state
- tr(ρ_A²) = 1
- E = 0 (no entanglement)

Consistent with |ψ⟩ being a product state!

### Example 4: Three-Qubit GHZ State

|GHZ⟩ = (|000⟩ + |111⟩)/√2

**Reduced state of qubit A:**
$$\rho_A = \text{tr}_{BC}(|GHZ\rangle\langle GHZ|) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

**Reduced state of qubits AB:**
$$\rho_{AB} = \text{tr}_C(|GHZ\rangle\langle GHZ|) = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$

This is a **classical mixture**, not an entangled state of AB!

**Key insight:** GHZ state has 3-way entanglement but no 2-way entanglement.

---

## 📝 Practice Problems

### Level 1: Basic Partial Trace
1. Compute ρ_B for |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.

2. Find ρ_A for the state ρ_AB = |01⟩⟨01|.

3. Verify tr(ρ_A) = 1 for any ρ_AB with tr(ρ_AB) = 1.

### Level 2: Entanglement Detection
4. Compute ρ_A for |ψ⟩ = (2|00⟩ + |11⟩)/√5. Is the state entangled?

5. Find the entanglement entropy of |ψ⟩ = (|00⟩ + |01⟩)/√2.

6. Show that |ψ⟩ = |+⟩|−⟩ has ρ_A = |+⟩⟨+| (pure).

### Level 3: Advanced
7. For the W state |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3, compute ρ_A.

8. Show that S(ρ_A) = S(ρ_B) for any pure state |ψ⟩_AB.

9. Compute ρ_AB for |GHZ⟩ and show it is separable (not entangled).

### Level 4: Theory
10. Prove: tr_B(A ⊗ B) = A·tr(B).

11. Show that partial trace preserves trace: tr(tr_B(ρ_AB)) = tr(ρ_AB).

12. Prove: For pure |ψ⟩_AB, the eigenvalues of ρ_A and ρ_B are identical.

---

## 💻 Evening Computational Lab

```python
import numpy as np
from scipy.linalg import logm

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Partial Trace Implementation
# ============================================

def partial_trace(rho, dims, trace_over):
    """
    Compute partial trace of density matrix.
    
    rho: density matrix of composite system
    dims: list of subsystem dimensions [dim_A, dim_B, ...]
    trace_over: index of subsystem to trace out (0-indexed)
    
    Returns: reduced density matrix
    """
    n_systems = len(dims)
    total_dim = np.prod(dims)
    
    # Reshape rho into tensor with indices for each subsystem
    # rho_{i1 i2 ... in, j1 j2 ... jn} -> tensor[i1, i2, ..., in, j1, j2, ..., jn]
    tensor_shape = dims + dims
    rho_tensor = rho.reshape(tensor_shape)
    
    # Trace over specified subsystem: contract indices trace_over and trace_over + n_systems
    result = np.trace(rho_tensor, axis1=trace_over, axis2=trace_over + n_systems)
    
    # Reshape back to matrix
    remaining_dims = [d for i, d in enumerate(dims) if i != trace_over]
    new_dim = np.prod(remaining_dims)
    
    return result.reshape(new_dim, new_dim)

def partial_trace_B(rho_AB, dim_A, dim_B):
    """Trace out system B (simpler implementation for bipartite)"""
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    
    for i in range(dim_A):
        for k in range(dim_A):
            for j in range(dim_B):
                # Sum over B basis: (rho_A)_{ik} = sum_j rho_{ij, kj}
                row = i * dim_B + j
                col = k * dim_B + j
                rho_A[i, k] += rho_AB[row, col]
    
    return rho_A

def partial_trace_A(rho_AB, dim_A, dim_B):
    """Trace out system A"""
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    
    for j in range(dim_B):
        for l in range(dim_B):
            for i in range(dim_A):
                # Sum over A basis
                row = i * dim_B + j
                col = i * dim_B + l
                rho_B[j, l] += rho_AB[row, col]
    
    return rho_B

# ============================================
# Entanglement Measures
# ============================================

def von_neumann_entropy(rho):
    """Compute S(ρ) = -tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def purity(rho):
    """Compute tr(ρ²)"""
    return np.real(np.trace(rho @ rho))

def entanglement_entropy(psi, dim_A, dim_B):
    """Compute entanglement entropy of pure state"""
    rho_AB = np.outer(psi, psi.conj())
    rho_A = partial_trace_B(rho_AB, dim_A, dim_B)
    return von_neumann_entropy(rho_A)

# ============================================
# Test with Bell States
# ============================================

print("=== Bell State Analysis ===")

# Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())

print(f"Bell state |Φ+⟩ = {phi_plus}")
print(f"\nDensity matrix ρ_AB:")
print(rho_bell)

# Partial traces
rho_A = partial_trace_B(rho_bell, 2, 2)
rho_B = partial_trace_A(rho_bell, 2, 2)

print(f"\nρ_A (trace out B):\n{rho_A}")
print(f"\nρ_B (trace out A):\n{rho_B}")

print(f"\nρ_A = ρ_B = I/2: {np.allclose(rho_A, np.eye(2)/2)}")
print(f"Purity of ρ_A: {purity(rho_A):.4f}")
print(f"Entanglement entropy: {entanglement_entropy(phi_plus, 2, 2):.4f} ebits")

# ============================================
# Product State Analysis
# ============================================

print("\n=== Product State Analysis ===")

# Product state |+0⟩ = |+⟩ ⊗ |0⟩
ket_plus = np.array([1, 1]) / np.sqrt(2)
ket_0 = np.array([1, 0])
product_state = np.kron(ket_plus, ket_0)

print(f"Product state |+0⟩ = {product_state}")

rho_prod = np.outer(product_state, product_state.conj())
rho_A_prod = partial_trace_B(rho_prod, 2, 2)

print(f"\nρ_A:\n{rho_A_prod}")
print(f"Expected |+⟩⟨+|:\n{np.outer(ket_plus, ket_plus.conj())}")
print(f"\nPurity of ρ_A: {purity(rho_A_prod):.4f} (should be 1)")
print(f"Entanglement entropy: {entanglement_entropy(product_state, 2, 2):.4f}")

# ============================================
# Partially Entangled State
# ============================================

print("\n=== Partially Entangled State ===")

# |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3
psi_partial = np.array([1, 1, 1, 0], dtype=complex) / np.sqrt(3)
rho_partial = np.outer(psi_partial, psi_partial.conj())

print(f"State: (|00⟩ + |01⟩ + |10⟩)/√3")

rho_A_partial = partial_trace_B(rho_partial, 2, 2)
print(f"\nρ_A:\n{rho_A_partial}")

eigenvalues = np.linalg.eigvalsh(rho_A_partial)
print(f"\nEigenvalues of ρ_A: {eigenvalues}")
print(f"Purity: {purity(rho_A_partial):.4f}")
print(f"Entanglement entropy: {entanglement_entropy(psi_partial, 2, 2):.4f} ebits")

# ============================================
# GHZ State (3 qubits)
# ============================================

print("\n=== GHZ State (3 qubits) ===")

# |GHZ⟩ = (|000⟩ + |111⟩)/√2
ghz = np.zeros(8, dtype=complex)
ghz[0] = ghz[7] = 1/np.sqrt(2)

rho_ghz = np.outer(ghz, ghz.conj())

# Trace out qubit C (last qubit)
rho_AB = partial_trace(rho_ghz, [2, 2, 2], 2)
print(f"ρ_AB (trace out C):\n{rho_AB}")

# Trace out qubits B and C
rho_A_ghz = partial_trace(rho_ghz, [2, 2, 2], 2)
rho_A_ghz = partial_trace(rho_A_ghz, [2, 2], 1)
print(f"\nρ_A (trace out B and C):\n{rho_A_ghz}")

# ============================================
# Comparison: Different Entanglement Types
# ============================================

print("\n=== Entanglement Comparison ===")

states = {
    "Product |00⟩": np.array([1, 0, 0, 0], dtype=complex),
    "Product |+0⟩": np.kron(ket_plus, ket_0),
    "Bell |Φ+⟩": phi_plus,
    "Partial ψ": psi_partial,
    "Product |++⟩": np.kron(ket_plus, ket_plus)
}

print(f"{'State':<20} {'Purity(ρ_A)':<15} {'Entropy (ebits)':<15} {'Entangled?'}")
print("-" * 65)

for name, psi in states.items():
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace_B(rho, 2, 2)
    p = purity(rho_A)
    e = von_neumann_entropy(rho_A)
    entangled = "No" if np.isclose(p, 1) else "YES"
    print(f"{name:<20} {p:<15.4f} {e:<15.4f} {entangled}")

# ============================================
# Visualize Entanglement
# ============================================

import matplotlib.pyplot as plt

# Sweep entanglement parameter
theta_vals = np.linspace(0, np.pi/2, 50)
entropies = []
purities = []

for theta in theta_vals:
    # |ψ(θ)⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
    psi_theta = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
    rho = np.outer(psi_theta, psi_theta.conj())
    rho_A = partial_trace_B(rho, 2, 2)
    
    entropies.append(von_neumann_entropy(rho_A))
    purities.append(purity(rho_A))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(theta_vals * 180/np.pi, entropies, 'b-', linewidth=2)
plt.xlabel('θ (degrees)')
plt.ylabel('Entanglement Entropy (ebits)')
plt.title('Entropy vs Mixing Parameter')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', label='Max (1 ebit)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(theta_vals * 180/np.pi, purities, 'r-', linewidth=2)
plt.xlabel('θ (degrees)')
plt.ylabel('Purity of ρ_A')
plt.title('Purity vs Mixing Parameter')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='b', linestyle='--', label='Min (maximally mixed)')
plt.legend()

plt.tight_layout()
plt.savefig('entanglement_sweep.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Understand the partial trace operation
- [ ] Compute reduced density matrices
- [ ] Connect entanglement to mixed reduced states
- [ ] Calculate entanglement entropy
- [ ] Distinguish GHZ vs W state entanglement
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 124: Density Matrices — Mixed States and Quantum Channels**
- Pure vs mixed states
- Density matrix formalism
- Quantum operations and channels
- Kraus representation
- Decoherence models

---

*"Entanglement is not just a peculiar feature of quantum mechanics — it's the resource that makes quantum computation and communication possible."*
— Quantum Information Saying
