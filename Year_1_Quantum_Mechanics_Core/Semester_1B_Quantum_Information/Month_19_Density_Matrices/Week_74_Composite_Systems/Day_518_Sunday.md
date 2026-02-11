# Day 518: Week 74 Review — Composite Systems

## Overview

**Day 518** | Week 74, Day 7 | Year 1, Month 19 | Integration and Consolidation

Today we synthesize all concepts from Week 74: tensor products, partial trace, reduced density matrices, Schmidt decomposition, entanglement detection, and purification.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Concept review and connections |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Comprehensive problem solving |
| Evening | 7:00 PM - 8:30 PM | 1.5 hrs | Self-assessment |

---

## Week 74 Concept Map

```
                  COMPOSITE QUANTUM SYSTEMS
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    TENSOR PRODUCTS   PARTIAL TRACE    SCHMIDT DECOMP
          │               │               │
    H_AB = H_A⊗H_B    ρ_A = Tr_B(ρ_AB)  |ψ⟩ = Σ√λᵢ|aᵢ⟩|bᵢ⟩
    dim = d_A × d_B   Local observables  Equal spectra
          │               │               │
          └───────────────┴───────┬───────┘
                                  │
                            ENTANGLEMENT
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        SEPARABILITY        DETECTION           PURIFICATION
              │                   │                   │
    ρ = Σpᵢρᵢ^A⊗ρᵢ^B        PPT criterion      |ψ⟩_AR → ρ_A
    vs entangled            Witnesses          Mixed = reduced
              │                   │                   │
              └───────────────────┴───────────────────┘
                                  │
                           APPLICATIONS
                                  │
           ┌─────────┬────────┬───────┬─────────┐
           │         │        │       │         │
        Quantum   Crypto-   Error  Channels   QIP
        Computing  graphy  Models           Theory
```

---

## Key Concepts Summary

### 1. Tensor Products (Day 512)

**Composite space:** ℋ_AB = ℋ_A ⊗ ℋ_B with dim = d_A × d_B

**Product states:** |ψ⟩_AB = |φ⟩_A ⊗ |χ⟩_B

**Operators:** (A⊗B)|i,j⟩ = (A|i⟩)⊗(B|j⟩)

**Key rule:** (A⊗B)(C⊗D) = (AC)⊗(BD)

### 2. Partial Trace (Day 513)

$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j \langle j|_B \rho_{AB} |j\rangle_B$$

**Key insight:** Partial trace gives the local description of a subsystem.

### 3. Reduced Density Matrices (Day 514)

- Contain all information about local measurements
- ⟨A⊗I⟩_ρAB = ⟨A⟩_ρA
- Entangled pure states have mixed reduced states

### 4. Schmidt Decomposition (Day 515)

$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \sqrt{\lambda_i} |a_i\rangle_A |b_i\rangle_B$$

- Schmidt rank r: r=1 ⟺ product state
- Equal spectra: eigenvalues of ρ_A and ρ_B are {λᵢ}
- Entanglement entropy: E = -Σλᵢ log λᵢ

### 5. Entanglement Detection (Day 516)

**PPT criterion:** ρ^(T_B) ≥ 0 for separable states

**Entanglement witness:** Observable W with Tr(Wρ_ent) < 0

**Werner state threshold:** Entangled for p > 1/3

### 6. Purification (Day 517)

Every ρ_A = Tr_R(|ψ⟩_AR⟨ψ|) for some pure |ψ⟩_AR

**Construction:** |ψ⟩ = Σᵢ √λᵢ |eᵢ⟩|i⟩

**Non-uniqueness:** Related by unitaries on R

---

## Master Formula Sheet

### Tensor Products
| Formula | Description |
|---------|-------------|
| dim(ℋ_AB) = d_A × d_B | Dimension |
| (A⊗B)(C⊗D) = (AC)⊗(BD) | Mixed product |
| Tr(A⊗B) = Tr(A)·Tr(B) | Trace factorization |

### Partial Trace
| Formula | Description |
|---------|-------------|
| ρ_A = Tr_B(ρ_AB) | Definition |
| Tr_B(\|ij⟩⟨kl\|) = \|i⟩⟨k\|δ_jl | Basis elements |
| ⟨A⊗I⟩ = Tr(ρ_A A) | Local expectation |

### Schmidt Decomposition
| Formula | Description |
|---------|-------------|
| \|ψ⟩ = Σ√λᵢ\|aᵢ⟩\|bᵢ⟩ | Decomposition |
| Σλᵢ = 1 | Normalization |
| E = -Σλᵢ log λᵢ | Entanglement entropy |
| spec(ρ_A) = spec(ρ_B) | Equal spectra |

### Entanglement Detection
| Criterion | Formula |
|-----------|---------|
| PPT | ρ^(T_B) ≥ 0 |
| Witness | Tr(Wρ) < 0 |
| Werner threshold | p > 1/3 |

### Purification
| Formula | Description |
|---------|-------------|
| \|ψ⟩_AR = Σ√λᵢ\|eᵢ⟩\|i⟩ | Construction |
| ρ_A = Tr_R(\|ψ⟩⟨ψ\|) | Recovery |
| dim(ℋ_R) ≥ rank(ρ) | Minimum dimension |

---

## Comprehensive Problem Set

### Part A: Tensor Products

**A1.** Compute (H⊗H)|00⟩ and express the result in the computational basis.

**A2.** Show that CNOT = |0⟩⟨0|⊗I + |1⟩⟨1|⊗X in matrix form.

**A3.** For a 3-qubit system, what is the dimension of the Hilbert space?

### Part B: Partial Trace

**B1.** Compute Tr_B(|GHZ⟩⟨GHZ|) where |GHZ⟩ = (|000⟩+|111⟩)/√2.

**B2.** Verify that Tr(Tr_B(ρ_AB)) = Tr(ρ_AB).

**B3.** For |ψ⟩ = (|00⟩+|01⟩+|10⟩)/√3, find ρ_A and ρ_B.

### Part C: Schmidt Decomposition

**C1.** Find the Schmidt decomposition of |ψ⟩ = (2|00⟩+|11⟩)/√5.

**C2.** Calculate the entanglement entropy of the state in C1.

**C3.** Show that |W⟩ = (|001⟩+|010⟩+|100⟩)/√3 has Schmidt rank 2 across any bipartition.

### Part D: Entanglement

**D1.** Apply PPT criterion to ρ = ½|Φ⁺⟩⟨Φ⁺| + ½I/4.

**D2.** Find an entanglement witness for |Ψ⁻⟩ = (|01⟩-|10⟩)/√2.

**D3.** Construct a purification of ρ = ½|+⟩⟨+| + ½|−⟩⟨−|.

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] I understand tensor product structure for composite systems
- [ ] I can explain why partial trace gives local descriptions
- [ ] I understand Schmidt decomposition and its significance
- [ ] I know how to detect entanglement using PPT
- [ ] I understand purification and its non-uniqueness

### Computational Skills
- [ ] I can compute tensor products of states and operators
- [ ] I can perform partial traces
- [ ] I can find Schmidt decompositions
- [ ] I can apply the PPT criterion
- [ ] I can construct purifications

---

## Computational Review Lab

```python
"""
Day 518: Week 74 Comprehensive Review
Complete composite systems toolkit
"""

import numpy as np

# === TOOLKIT ===

def tensor(A, B):
    return np.kron(A, B)

def partial_trace_B(rho_AB, dim_A, dim_B):
    rho = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=1, axis2=3)

def partial_trace_A(rho_AB, dim_A, dim_B):
    rho = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=0, axis2=2)

def schmidt_decomposition(psi, dim_A, dim_B):
    C = psi.reshape(dim_A, dim_B)
    U, S, Vh = np.linalg.svd(C, full_matrices=False)
    return S**2, U, Vh.conj()

def entanglement_entropy(lambdas):
    return -np.sum(lambdas * np.log2(lambdas + 1e-15))

def partial_transpose_B(rho, dim_A, dim_B):
    rho_r = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    return rho_r.transpose(0, 3, 2, 1).reshape(dim_A*dim_B, dim_A*dim_B)

def is_ppt(rho, dim_A, dim_B):
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)
    return np.min(np.linalg.eigvalsh(rho_pt)) >= -1e-10

def purify(rho):
    evals, evecs = np.linalg.eigh(rho)
    d = len(evals)
    psi = np.zeros(d * d, dtype=complex)
    for i in range(d):
        if evals[i] > 1e-10:
            for j in range(d):
                psi[j*d + i] = np.sqrt(evals[i]) * evecs[j, i]
    return psi

def density(psi):
    return np.outer(psi, psi.conj())

# Standard states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

ket_00 = np.kron(ket_0, ket_0)
ket_11 = np.kron(ket_1, ket_1)
phi_plus = (ket_00 + ket_11) / np.sqrt(2)

print("=" * 70)
print("WEEK 74 COMPREHENSIVE REVIEW")
print("=" * 70)

# Test all operations on Bell state
rho_bell = density(phi_plus)

print("\n--- Bell State |Φ⁺⟩ Analysis ---")
print(f"Global purity: {np.trace(rho_bell @ rho_bell).real:.4f}")

rho_A = partial_trace_B(rho_bell, 2, 2)
print(f"\nReduced state ρ_A:\n{rho_A}")
print(f"Local purity: {np.trace(rho_A @ rho_A).real:.4f}")

lambdas, _, _ = schmidt_decomposition(phi_plus, 2, 2)
print(f"\nSchmidt coefficients: {lambdas}")
print(f"Entanglement entropy: {entanglement_entropy(lambdas):.4f} bits")

print(f"\nPPT satisfied: {is_ppt(rho_bell, 2, 2)}")

psi_purified = purify(rho_A)
rho_recovered = partial_trace_B(density(psi_purified), 2, 2)
print(f"Purification of ρ_A recovers it: {np.allclose(rho_A, rho_recovered)}")

print("\n" + "=" * 70)
print("Week 74 Complete! Ready for Week 75: Generalized Measurements")
print("=" * 70)
```

---

## Looking Ahead: Week 75

Next week we study **generalized measurements** (POVMs):

- Beyond projective measurements
- Positive operator-valued measures
- Neumark's theorem
- Optimal state discrimination
- Measurement implementations

POVMs extend our measurement toolkit beyond von Neumann projective measurements.

---

## Key Takeaways from Week 74

1. **Tensor products** describe composite quantum systems with dimension d_A × d_B
2. **Partial trace** extracts local information from global states
3. **Schmidt decomposition** reveals entanglement structure of pure states
4. **PPT criterion** detects many entangled states
5. **Purification** shows every mixed state is part of a pure state
6. **Entanglement = local mixedness** for pure global states

---

**Week 74 Complete!**

You now understand the mathematical framework for composite quantum systems—essential for quantum information theory and multi-qubit quantum computing.

---

*Next: Week 75 — Generalized Measurements*
