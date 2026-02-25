# Day 413: Week 59 Review — Addition of Angular Momenta

## Overview
**Day 413** | Year 1, Month 15, Week 59 | Synthesis

---

## Week 59 Summary

| Day | Topic | Key Result |
|-----|-------|------------|
| 407 | Two Angular Momenta | Ĵ = Ĵ₁ + Ĵ₂, tensor product space |
| 408 | Coupled vs Uncoupled | Two complete bases |
| 409 | Clebsch-Gordan | \|j,m⟩ = Σ CG \|j₁,m₁;j₂,m₂⟩ |
| 410 | Spin-Orbit | J = L + S, fine structure |
| 411 | Two Spin-1/2 | Singlet + Triplet, Bell states |
| 412 | Wigner 3j | Symmetric CG coefficients |

---

## Master Formula Sheet

### Triangle Rule
$$|j_1 - j_2| \leq j \leq j_1 + j_2$$

### Selection Rule
$$m = m_1 + m_2$$

### Dimension Conservation
$$(2j_1+1)(2j_2+1) = \sum_j (2j+1)$$

### Two Spin-1/2
$$\frac{1}{2} \otimes \frac{1}{2} = 0 \oplus 1$$

**Singlet:** |0,0⟩ = (|↑↓⟩ - |↓↑⟩)/√2

**Triplet:** |1,m⟩ for m = -1, 0, +1

### Spin-Orbit Coupling
$$\hat{\mathbf{L}}\cdot\hat{\mathbf{S}} = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

---

## Quantum Computing Connections

1. **Singlet state = Bell state** |Ψ⁻⟩
2. **Angular momentum conservation** → selection rules for gates
3. **CG coefficients** appear in multi-qubit state preparation

---

## Comprehensive Problem Set

1. For j₁ = 3/2, j₂ = 1, find all allowed j values and the total dimension.

2. Express |↑↓⟩ in terms of |1,0⟩ and |0,0⟩.

3. Calculate ⟨Ŝ₁·Ŝ₂⟩ for the triplet |1,0⟩ state.

4. Find the CG coefficient ⟨1,1; 1,-1|1,0⟩.

5. What is the spectroscopic notation for an electron with n=3, l=2, j=5/2?

---

## Week 59 Checklist

- [ ] I understand the triangle rule
- [ ] I can use coupled and uncoupled bases
- [ ] I know the two spin-1/2 decomposition
- [ ] I can calculate simple CG coefficients
- [ ] I understand spin-orbit coupling
- [ ] I see the connection to Bell states

---

**Next Week:** [Week_60_Rotations_Wigner/README.md](../Week_60_Rotations_Wigner/README.md) — Rotations & Wigner D-Matrices
