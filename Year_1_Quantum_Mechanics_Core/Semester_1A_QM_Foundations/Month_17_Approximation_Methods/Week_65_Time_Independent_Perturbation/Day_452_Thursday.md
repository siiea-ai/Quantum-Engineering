# Day 452: Degenerate Perturbation Theory II

## Overview
**Day 452** | Year 1, Month 17, Week 65 | Good Quantum Numbers and Selection

Today we explore how symmetry can simplify degenerate perturbation theory by identifying "good quantum numbers" that survive the perturbation.

---

## Learning Objectives

By the end of today, you will be able to:
1. Use symmetry to simplify degenerate PT
2. Identify good quantum numbers
3. Apply selection rules to reduce matrix size
4. Handle partial lifting of degeneracy
5. Use commutators to find good bases
6. Apply to realistic atomic physics problems

---

## Core Content

### Good Quantum Numbers from Symmetry

If [H', Â] = 0 for some operator Â:
- Eigenstates of Â are automatically "good" states
- H' is block diagonal in the Â basis
- Matrix simplifies significantly

### Finding Good Quantum Numbers

**Recipe:**
1. Find operators that commute with both H₀ and H'
2. Use their eigenvalues to label states
3. H' only connects states with same quantum numbers

### Example: Hydrogen Fine Structure

H₀ commutes with: L², S², L_z, S_z, J², J_z

H'_SO = ξL·S commutes with: L², S², J², J_z

**Good quantum numbers with H'_SO:** l, s, j, m_j
**Not good:** m_l, m_s

### Block Diagonalization

If A|α⟩ = a|α⟩, then:
$$\langle\alpha|H'|\alpha'\rangle = 0 \quad \text{unless } a = a'$$

The H' matrix breaks into blocks labeled by the good quantum number.

### Partially Lifted Degeneracy

Sometimes H' doesn't fully lift degeneracy:
- Original: g-fold degenerate
- After H': g_1 + g_2 + ... = g (sum of smaller degeneracies)

### Higher-Order Corrections

After finding "good" states, apply non-degenerate PT:
$$E^{(2)}_\alpha = \sum_{m \notin \text{deg}} \frac{|\langle m|H'|\alpha\rangle|^2}{E^{(0)} - E_m^{(0)}}$$

---

## Quantum Computing Connection

### Symmetry-Protected Qubits

Good quantum numbers enable:
- **Decoherence-free subspaces**
- **Error detection** via symmetry
- **Topological protection**

---

## Worked Example

**Problem:** n = 2 hydrogen with spin-orbit coupling.

Without spin-orbit: 8 degenerate states (2s, 2p × spin)

With H'_SO ∝ L·S:
- l = 0 (2s): L·S = 0, no splitting
- l = 1 (2p): j = 1/2 or j = 3/2

Good basis: |n, l, j, m_j⟩

Result: 2S₁/₂ (2 states), 2P₁/₂ (2 states), 2P₃/₂ (4 states)

---

## Practice Problems

1. What are the good quantum numbers for Zeeman effect?
2. For H' = λL_z, which basis is "good"?
3. Show that J_z remains good for any central potential + spin-orbit.

---

## Summary

| Concept | Application |
|---------|-------------|
| Good QN | [H', Â] = 0 → A eigenvalue is conserved |
| Block diagonal | Only same-A states couple |
| Selection rule | ⟨α|H'|α'⟩ = 0 if a ≠ a' |

---

**Next:** [Day_453_Friday.md](Day_453_Friday.md) — Fine Structure Application
