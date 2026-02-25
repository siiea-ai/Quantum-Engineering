# Day 455: Week 65 Review — Time-Independent Perturbation Theory

## Overview
**Day 455** | Year 1, Month 17, Week 65 | Comprehensive Review

Today we consolidate our understanding of time-independent perturbation theory before moving to the variational method.

---

## Week 65 Summary

### Day 449-450: Non-Degenerate Theory
- First-order energy: E^(1) = ⟨n|H'|n⟩
- Second-order energy: E^(2) = Σ|⟨m|H'|n⟩|²/(E_n - E_m)
- First-order state: |n^(1)⟩ = Σ ⟨m|H'|n⟩/(E_n - E_m)|m⟩

### Day 451-452: Degenerate Theory
- Secular equation: det(H' - E^(1)I) = 0
- Good quantum numbers from [H', Â] = 0
- Block diagonalization via symmetry

### Day 453: Fine Structure
- Combined relativistic + spin-orbit + Darwin
- E_FS = E_n(α²/n²)[n/(j+1/2) - 3/4]

### Day 454: Stark Effect
- Quadratic Stark: ΔE = -αE²/2 (ground state)
- Linear Stark: ΔE = ±pE (degenerate excited states)

---

## Master Formula Sheet

### Non-Degenerate Perturbation Theory

$$E_n^{(1)} = \langle n^{(0)} | H' | n^{(0)} \rangle$$

$$E_n^{(2)} = \sum_{m \neq n} \frac{|\langle m^{(0)} | H' | n^{(0)} \rangle|^2}{E_n^{(0)} - E_m^{(0)}}$$

$$|n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}}|m^{(0)}\rangle$$

### Degenerate Perturbation Theory

$$\det(H'_{ij} - E^{(1)}\delta_{ij}) = 0$$

$$H'_{ij} = \langle n_i^{(0)} | H' | n_j^{(0)} \rangle$$

---

## Week 65 Checklist

- [ ] I can derive first and second-order energy corrections
- [ ] I understand when degeneracy causes problems
- [ ] I can apply the secular equation
- [ ] I know how to identify good quantum numbers
- [ ] I can apply PT to hydrogen atom problems

---

## Preview: Week 66

Next week: **Variational Method**
- Variational principle: E[ψ] ≥ E₀
- Trial wavefunctions
- Helium atom ground state
- Hydrogen molecule ion

---

**Congratulations on completing Week 65!**

**Next:** [Week_66_Variational_Method](../Week_66_Variational_Method/README.md)
