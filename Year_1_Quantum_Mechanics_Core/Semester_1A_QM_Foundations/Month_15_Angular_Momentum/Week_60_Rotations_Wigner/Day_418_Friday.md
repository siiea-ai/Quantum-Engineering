# Day 418: Selection Rules

## Overview
**Day 418** | Year 1, Month 15, Week 60 | Which Transitions Are Allowed?

Selection rules determine which quantum transitions are allowed, arising from symmetry and the Wigner-Eckart theorem.

---

## Core Content

### Electric Dipole (E1) Selection Rules

The electric dipole operator has k = 1:

| Rule | Allowed Values |
|------|----------------|
| Δj | 0, ±1 (but j=0 → j'=0 forbidden) |
| Δm | 0, ±1 |
| Δl | ±1 (parity must change) |

### Magnetic Dipole (M1) Selection Rules

| Rule | Allowed Values |
|------|----------------|
| Δj | 0, ±1 |
| Δm | 0, ±1 |
| Δl | 0 (parity unchanged) |

### Electric Quadrupole (E2) Selection Rules

The quadrupole operator has k = 2:

| Rule | Allowed Values |
|------|----------------|
| Δj | 0, ±1, ±2 (but j=0→0, 1/2→1/2 forbidden) |
| Δm | 0, ±1, ±2 |
| Δl | 0, ±2 |

### Polarization and Δm

| Polarization | Δm | Geometry |
|--------------|-----|----------|
| π (z-polarized) | 0 | Light along x or y |
| σ⁺ (right circular) | +1 | Light along +z |
| σ⁻ (left circular) | -1 | Light along -z |

### Forbidden Transitions

- j = 0 → j' = 0 always forbidden (no direction to define)
- Δj = 0, Δm = 0 for j = 0 forbidden
- Parity-forbidden transitions (e.g., 1s → 2s for E1)

---

## Applications

### Hydrogen Spectrum

| Transition | Type | Selection Rule |
|------------|------|----------------|
| 2p → 1s | E1 allowed | Δl = -1 |
| 2s → 1s | E1 forbidden | Δl = 0 |
| 3d → 2p | E1 allowed | Δl = -1 |

### 21 cm Line

Hyperfine transition in hydrogen (F=1 → F=0): magnetic dipole allowed!

---

## Practice Problems
1. Is the transition 3d_{5/2} → 2p_{3/2} allowed for E1?
2. What Δm values allow σ⁺ absorption?
3. Why is 2s → 1s forbidden for E1 but allowed for two-photon?

---

**Next:** [Day_419_Saturday.md](Day_419_Saturday.md) — Tensor Operators
