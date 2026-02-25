# Day 343: Week 49 Review — Hilbert Space Formalism Synthesis

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive Review & Problem Solving |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Practice Exam & Self-Assessment |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Capstone Computational Lab |

**Total Study Time:** 7 hours

---

## Week 49 Summary

This week established the mathematical foundation of quantum mechanics:

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 337 | Complex Vector Spaces | Vector space axioms, superposition, why ℂ |
| 338 | Dirac Notation | Bras, kets, brackets, outer products |
| 339 | Operators | Linear operators, matrix representation, operator algebra |
| 340 | Hermitian & Unitary | A† = A for observables, U†U = I for evolution |
| 341 | Eigenvalue Problems | Spectral decomposition, diagonalization |
| 342 | Continuous Spectra | Position/momentum eigenstates, delta functions |

---

## Learning Objectives Achieved

By completing Week 49, you can now:

- [x] Define and work in complex Hilbert spaces
- [x] Use Dirac bra-ket notation fluently
- [x] Represent operators as matrices in any basis
- [x] Identify and work with Hermitian and unitary operators
- [x] Solve eigenvalue problems and construct spectral decompositions
- [x] Handle continuous spectra with generalized eigenvectors
- [x] Connect wave functions to abstract state vectors

---

## Core Concepts Integration

### The Hilbert Space Framework

```
                     HILBERT SPACE ℋ
                           │
            ┌──────────────┼──────────────┐
            │              │              │
         States        Operators      Inner Product
         |ψ⟩ ∈ ℋ        Â: ℋ→ℋ        ⟨φ|ψ⟩ ∈ ℂ
            │              │              │
            │         ┌────┴────┐         │
            │    Hermitian  Unitary       │
            │     Â†=Â      Û†Û=I        │
            │       │          │          │
            │  Observables  Evolution     │
            │       │          │          │
            └───────┴──────────┴──────────┘
                           │
                    MEASUREMENT
                    P(a) = |⟨a|ψ⟩|²
```

### Key Formula Summary

| Concept | Formula |
|---------|---------|
| **Inner product** | ⟨φ\|ψ⟩ = ⟨ψ\|φ⟩* |
| **Completeness** | Î = Σₙ\|n⟩⟨n\| = ∫\|x⟩⟨x\|dx |
| **Expansion** | \|ψ⟩ = Σₙ cₙ\|n⟩, cₙ = ⟨n\|ψ⟩ |
| **Adjoint** | ⟨φ\|Â†\|ψ⟩ = ⟨ψ\|Â\|φ⟩* |
| **Hermitian** | Â† = Â ⟹ eigenvalues real |
| **Unitary** | Û†Û = Î ⟹ \|λ\| = 1 |
| **Spectral** | Â = Σₐ a\|a⟩⟨a\| |
| **Wave function** | ψ(x) = ⟨x\|ψ⟩ |
| **Position-momentum** | ⟨x\|p⟩ = (2πℏ)^{-1/2} e^{ipx/ℏ} |

---

## Practice Exam

**Time: 90 minutes | Total: 100 points**

### Section A: Short Answer (30 points)

**A1.** (6 pts) Define a Hilbert space. What three properties distinguish it from a general vector space?

**A2.** (6 pts) Write down the completeness relation for an orthonormal discrete basis {|n⟩}. What is the analogous relation for position eigenstates?

**A3.** (6 pts) State the condition for an operator to be Hermitian. Why is this important physically?

**A4.** (6 pts) If |ψ⟩ = (3|0⟩ + 4i|1⟩)/5, what is ⟨ψ|? Verify ⟨ψ|ψ⟩ = 1.

**A5.** (6 pts) What is the physical significance of unitary operators in quantum mechanics?

---

### Section B: Calculations (40 points)

**B1.** (10 pts) Given operator Â with matrix:
$$A = \begin{pmatrix} 2 & 1+i \\ 1-i & 3 \end{pmatrix}$$

(a) Is Â Hermitian? Verify.
(b) Find the eigenvalues.
(c) Are the eigenvalues real? Explain why or why not.

**B2.** (10 pts) Consider the Hadamard operator:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

(a) Verify H is unitary.
(b) Find H|0⟩ and H|1⟩.
(c) Show H² = I.

**B3.** (10 pts) Let |ψ⟩ = α|+⟩ + β|-⟩ where |±⟩ = (|0⟩ ± |1⟩)/√2.

(a) Find |α|² + |β|² for |ψ⟩ to be normalized.
(b) Express |ψ⟩ in the {|0⟩, |1⟩} basis.
(c) Calculate ⟨0|ψ⟩ and ⟨1|ψ⟩.

**B4.** (10 pts) The momentum operator in position representation is p̂ = -iℏ d/dx.

(a) Show that p̂ is Hermitian using integration by parts (assume ψ → 0 at ±∞).
(b) Verify that ψₚ(x) = (2πℏ)^{-1/2} e^{ipx/ℏ} is an eigenfunction of p̂ with eigenvalue p.

---

### Section C: Conceptual (30 points)

**C1.** (10 pts) Explain why quantum mechanics requires complex numbers rather than real numbers. Give a specific physical example.

**C2.** (10 pts) The spectral theorem states that any Hermitian operator can be written as:
$$\hat{A} = \sum_a a|a⟩⟨a|$$

(a) What do |a⟩ represent?
(b) What does the sum become for continuous spectra?
(c) How does this connect to the measurement postulate?

**C3.** (10 pts) Compare and contrast:
(a) Discrete vs. continuous spectra
(b) Hermitian vs. unitary operators
(c) Bras vs. kets

---

## Solutions Outline

### Section A

**A1.** Hilbert space: complete inner product space over ℂ. Distinguished by: (1) inner product ⟨·|·⟩, (2) completeness (Cauchy sequences converge), (3) separability (countable basis).

**A2.** Discrete: Î = Σₙ|n⟩⟨n|. Continuous: Î = ∫|x⟩⟨x|dx.

**A3.** Â† = Â. Physical: eigenvalues (measurement outcomes) must be real.

**A4.** ⟨ψ| = (3⟨0| - 4i⟨1|)/5. Verify: |3|²/25 + |4i|²/25 = 9/25 + 16/25 = 1 ✓

**A5.** Unitary operators preserve norms (probability conservation) and represent time evolution and symmetry transformations.

### Section B

**B1.** (a) A† = A ✓ (check conjugate transpose). (b) det(A-λI) = 0 gives λ = (5±√5)/2 ≈ 3.62, 1.38. (c) Yes, real because Hermitian.

**B2.** (a) H†H = I ✓. (b) H|0⟩ = |+⟩, H|1⟩ = |-⟩. (c) H² = I ✓.

**B3.** (a) |α|² + |β|² = 1. (b) |ψ⟩ = [(α+β)|0⟩ + (α-β)|1⟩]/√2. (c) ⟨0|ψ⟩ = (α+β)/√2, ⟨1|ψ⟩ = (α-β)/√2.

**B4.** (a) Integration by parts shows ⟨φ|p̂ψ⟩ = ⟨p̂φ|ψ⟩. (b) p̂ψₚ = -iℏ(ip/ℏ)ψₚ = pψₚ ✓.

---

## Capstone Computational Lab

```python
"""
Day 343: Week 49 Capstone Lab
Comprehensive Hilbert Space Operations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

print("=" * 70)
print("WEEK 49 CAPSTONE: Hilbert Space Formalism")
print("=" * 70)

# =============================================================================
# Part 1: Complete Qubit State Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 1: Complete Qubit State Analysis")
print("=" * 70)

# Define computational basis
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Define a general qubit state
theta = np.pi / 3  # 60 degrees
phi = np.pi / 4    # 45 degrees

psi = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1
print(f"\n|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
print(f"θ = {np.degrees(theta):.1f}°, φ = {np.degrees(phi):.1f}°")
print(f"|ψ⟩ = {psi.flatten()}")

# Verify normalization
norm = np.vdot(psi, psi)
print(f"\n⟨ψ|ψ⟩ = {norm.real:.6f}")

# Expansion coefficients
c0 = np.vdot(ket_0, psi)
c1 = np.vdot(ket_1, psi)
print(f"\nExpansion: |ψ⟩ = c₀|0⟩ + c₁|1⟩")
print(f"c₀ = ⟨0|ψ⟩ = {c0:.4f}")
print(f"c₁ = ⟨1|ψ⟩ = {c1:.4f}")
print(f"|c₀|² + |c₁|² = {abs(c0)**2 + abs(c1)**2:.6f}")

# =============================================================================
# Part 2: Operator Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Complete Operator Analysis")
print("=" * 70)

# Pauli matrices
I = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = {'I': I, 'σx': sigma_x, 'σy': sigma_y, 'σz': sigma_z}

print("\nPauli Matrix Properties:")
for name, P in paulis.items():
    is_hermitian = np.allclose(P, P.conj().T)
    is_unitary = np.allclose(P @ P.conj().T, I)
    eigenvalues = np.linalg.eigvalsh(P) if is_hermitian else np.linalg.eigvals(P)
    print(f"{name}: Hermitian={is_hermitian}, Unitary={is_unitary}, eigenvalues={eigenvalues}")

# =============================================================================
# Part 3: Spectral Decomposition
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Spectral Decomposition")
print("=" * 70)

# General Hermitian matrix
H = np.array([[2, 1-1j], [1+1j, 3]], dtype=complex)
print("\nHermitian matrix H:")
print(H)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(H)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (columns):")
print(eigenvectors)

# Verify orthonormality
print(f"\nOrthonormality check:")
print(f"⟨v₁|v₁⟩ = {np.vdot(eigenvectors[:,0], eigenvectors[:,0]):.4f}")
print(f"⟨v₂|v₂⟩ = {np.vdot(eigenvectors[:,1], eigenvectors[:,1]):.4f}")
print(f"⟨v₁|v₂⟩ = {np.vdot(eigenvectors[:,0], eigenvectors[:,1]):.4f}")

# Reconstruct H from spectral decomposition
H_reconstructed = np.zeros_like(H)
for i, lam in enumerate(eigenvalues):
    v = eigenvectors[:, i:i+1]
    H_reconstructed += lam * (v @ v.conj().T)

print(f"\nSpectral reconstruction H = Σλᵢ|vᵢ⟩⟨vᵢ|:")
print(H_reconstructed)
print(f"Matches original: {np.allclose(H, H_reconstructed)}")

# =============================================================================
# Part 4: Unitary Time Evolution
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Unitary Time Evolution Simulation")
print("=" * 70)

# Hamiltonian (in units where ℏ=1)
omega = 1.0
H_evolve = omega * sigma_z / 2

# Time evolution operator U(t) = exp(-iHt)
def time_evolution(H, t):
    return linalg.expm(-1j * H * t)

# Evolve |+⟩ state
psi_0 = (ket_0 + ket_1) / np.sqrt(2)
times = np.linspace(0, 4*np.pi, 100)

# Track expectation values
exp_x = []
exp_y = []
exp_z = []

for t in times:
    U = time_evolution(H_evolve, t)
    psi_t = U @ psi_0

    exp_x.append((psi_t.conj().T @ sigma_x @ psi_t)[0,0].real)
    exp_y.append((psi_t.conj().T @ sigma_y @ psi_t)[0,0].real)
    exp_z.append((psi_t.conj().T @ sigma_z @ psi_t)[0,0].real)

# Plot Bloch vector evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time evolution of expectation values
ax1 = axes[0]
ax1.plot(times, exp_x, label='⟨σx⟩', linewidth=2)
ax1.plot(times, exp_y, label='⟨σy⟩', linewidth=2)
ax1.plot(times, exp_z, label='⟨σz⟩', linewidth=2)
ax1.set_xlabel('Time (ωt)', fontsize=12)
ax1.set_ylabel('Expectation Value', fontsize=12)
ax1.set_title('Spin Precession: |+⟩ under H = ωσz/2', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Bloch sphere projection (x-y plane)
ax2 = axes[1]
ax2.plot(exp_x, exp_y, 'b-', linewidth=2)
ax2.plot(exp_x[0], exp_y[0], 'go', markersize=10, label='Start')
ax2.plot(exp_x[-1], exp_y[-1], 'rs', markersize=10, label='End')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax2.add_patch(circle)
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect('equal')
ax2.set_xlabel('⟨σx⟩', fontsize=12)
ax2.set_ylabel('⟨σy⟩', fontsize=12)
ax2.set_title('Bloch Vector Trajectory (x-y projection)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week49_capstone.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'week49_capstone.png'")

# =============================================================================
# Part 5: Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Week 49 Knowledge Summary")
print("=" * 70)

summary = """
WEEK 49 COMPLETE - Hilbert Space Formalism Mastered!

Key Accomplishments:
✓ Complex vector spaces and superposition principle
✓ Dirac notation: bras, kets, brackets, outer products
✓ Linear operators and matrix representations
✓ Hermitian operators (observables) and their properties
✓ Unitary operators (evolution) and norm preservation
✓ Eigenvalue problems and spectral decomposition
✓ Continuous spectra and wave function interpretation

Ready for Week 50: Observables & Measurement!
"""
print(summary)

print("=" * 70)
print("CAPSTONE LAB COMPLETE!")
print("=" * 70)
```

---

## Self-Assessment Rubric

Rate yourself 1-5 on each skill:

| Skill | 1 (Struggling) | 3 (Competent) | 5 (Mastery) | Your Score |
|-------|----------------|---------------|-------------|------------|
| Vector space axioms | Cannot state | Can state and verify | Apply to new spaces | |
| Dirac notation | Confuse bras/kets | Use correctly | Fluent manipulation | |
| Operator matrices | Struggle with basis | Compute in given basis | Change bases easily | |
| Hermitian properties | Cannot identify | Verify and use | Prove theorems | |
| Unitary properties | Cannot identify | Verify and use | Prove theorems | |
| Eigenvalue problems | Cannot solve | Solve 2×2 | Solve and interpret | |
| Continuous spectra | Confused by δ(x) | Use completeness | Derive new results | |

**Target:** Average score ≥ 4.0 before proceeding to Week 50.

---

## Preparing for Week 50

Week 50 introduces the **measurement postulate**—the heart of quantum mechanics:

### Preview Topics:
1. **Measurement outcomes** as eigenvalues
2. **Born rule:** P(a) = |⟨a|ψ⟩|²
3. **State collapse** after measurement
4. **Expectation values** ⟨Â⟩ = ⟨ψ|Â|ψ⟩
5. **Compatible observables** and [Â,B̂] = 0

### Key Question for Next Week:
*When we measure a quantum system, what determines the outcome, and what happens to the state afterward?*

---

## Daily Checklist

- [ ] Complete practice exam (90 minutes, closed book)
- [ ] Score practice exam and identify weak areas
- [ ] Review any topics scoring below 70%
- [ ] Run capstone computational lab
- [ ] Update Week 49 summary notes
- [ ] Preview Week 50 material in Shankar Ch. 4
- [ ] Rest and consolidate learning

---

## Week 49 Complete!

**Congratulations!** You have completed the first week of Year 1. You now possess the mathematical language of quantum mechanics—Hilbert space formalism and Dirac notation.

**Progress:**
- Week 49: ✅ Complete (Days 337-343)
- Month 13: 7/28 days (25%)
- Year 1: 7/336 days (2%)

---

*"The formalism is complete. Now we ask: what does it mean to measure?"*

---

**Next:** [Week 50 README](../Week_50_Observables/README.md) — Observables & Measurement
