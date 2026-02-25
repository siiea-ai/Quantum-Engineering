# Day 445: Strong-Field Zeeman Effect

## Overview
**Day 445** | Year 1, Month 16, Week 64 | The Paschen-Back Effect

Today we study atoms in strong magnetic fields where the field energy exceeds the spin-orbit coupling.

---

## Learning Objectives

1. Define the strong-field (Paschen-Back) regime
2. Understand the uncoupling of L and S
3. Calculate energy levels using m_l, m_s
4. Analyze the transition between regimes
5. Apply to high-field spectroscopy

---

## Core Content

### Strong-Field Condition

μ_B B >> ΔE_FS (fine structure splitting)

For hydrogen n=2: B >> 0.5 T (strong field)

### Paschen-Back Regime

L and S precess independently around B:
$$\hat{H}_B = \frac{\mu_B B}{\hbar}(\hat{L}_z + g_e\hat{S}_z)$$

**Good quantum numbers:** m_l, m_s (not j, m_j)

### Energy Levels

$$\boxed{E = E_n + \mu_B B(m_l + 2m_s)}$$

Plus fine structure correction (now a perturbation):
$$E_{\text{FS}} \approx \xi_{nl}\langle\mathbf{L}\cdot\mathbf{S}\rangle = \xi_{nl}\hbar^2 m_l m_s$$

### Transition: Weak → Strong

| Weak Field | Strong Field |
|------------|--------------|
| j, m_j good | m_l, m_s good |
| g_J varies | g ≈ 1, 2 |
| Anomalous pattern | Simpler triplet |

### Intermediate Field

Both SO and Zeeman comparable — must diagonalize full Hamiltonian numerically.

---

## Practice Problems

1. At what field does n=2 hydrogen enter Paschen-Back?
2. How many distinct energy levels for 2P in strong field?
3. Sketch the energy levels vs B from weak to strong field.

---

## Summary

| Regime | Condition | Good QN |
|--------|-----------|---------|
| Weak | μ_B B << ΔE_FS | j, m_j |
| Strong | μ_B B >> ΔE_FS | m_l, m_s |
| Intermediate | Comparable | None simple |

---

**Next:** [Day_446_Friday.md](Day_446_Friday.md) — Stark Effect
