# Day 439: Good Quantum Numbers

## Overview
**Day 439** | Year 1, Month 16, Week 63 | Which Quantum Numbers Commute?

Today we identify which quantum numbers remain good when spin-orbit coupling is included.

---

## Learning Objectives

1. Define "good quantum numbers" via commutation
2. Show why l and s are no longer good
3. Demonstrate that j and m_j are good
4. Construct the coupled basis |n, l, j, m_j⟩
5. Apply Clebsch-Gordan coefficients

---

## Core Content

### Good Quantum Numbers

A quantum number is "good" if its operator commutes with H.

### Without Spin-Orbit

H₀ commutes with: L², L_z, S², S_z
Good quantum numbers: n, l, m_l, s, m_s

### With Spin-Orbit: H = H₀ + ξ(r)L·S

Now [H, L_z] ≠ 0 and [H, S_z] ≠ 0

But J = L + S still commutes:
- [H, J²] = 0
- [H, J_z] = 0

**New good quantum numbers:** n, l, j, m_j

### The Coupled Basis

$$|n, l, j, m_j\rangle = \sum_{m_l, m_s} \langle l, m_l; s, m_s | j, m_j\rangle |n, l, m_l\rangle |s, m_s\rangle$$

### For Spin-1/2

j = l + 1/2 or j = l - 1/2 (for l > 0)

Using CG coefficients:
$$|l, j=l+1/2, m_j\rangle = \sqrt{\frac{l+m_j+1/2}{2l+1}}|m_l=m_j-1/2\rangle|↑\rangle + \sqrt{\frac{l-m_j+1/2}{2l+1}}|m_l=m_j+1/2\rangle|↓\rangle$$

---

## Practice Problems

1. What are the good quantum numbers for the 2P states with spin-orbit?
2. Write |2P_{3/2}, m_j=3/2⟩ explicitly.
3. How many states exist with n=3, j=3/2?

---

## Summary

| Without SO | With SO |
|------------|---------|
| n, l, m_l, s, m_s | n, l, j, m_j |
| L_z, S_z conserved | J_z conserved |
| Uncoupled basis | Coupled basis |

---

**Next:** [Day_440_Saturday.md](Day_440_Saturday.md) — Spectroscopic Notation
