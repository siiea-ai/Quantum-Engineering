# Day 453: Fine Structure — Perturbation Theory Application

## Overview
**Day 453** | Year 1, Month 17, Week 65 | Revisiting Hydrogen Fine Structure

Today we apply perturbation theory systematically to re-derive the hydrogen fine structure, demonstrating all the techniques in a physically important context.

---

## Learning Objectives

By the end of today, you will be able to:
1. Apply perturbation theory to hydrogen atom
2. Calculate relativistic, spin-orbit, and Darwin corrections
3. Use degenerate PT for the n = 2 shell
4. Verify the fine structure formula
5. Understand the role of good quantum numbers
6. Connect perturbation theory to exact results

---

## Core Content

### The Perturbations

$$H = H_0 + H'_{\text{rel}} + H'_{\text{SO}} + H'_D$$

### Relativistic Correction

$$H'_{\text{rel}} = -\frac{p^4}{8m_e^3 c^2}$$

For state |nlm⟩ (non-degenerate in l):
$$E^{(1)}_{\text{rel}} = -\frac{1}{2m_e c^2}\left[E_n^2 + 2E_n\langle V\rangle + \langle V^2\rangle\right]$$

### Spin-Orbit Correction

$$H'_{\text{SO}} = \frac{e^2}{2m_e^2 c^2 r^3}\mathbf{L}\cdot\mathbf{S}$$

Using good basis |n,l,j,m_j⟩:
$$E^{(1)}_{\text{SO}} = \frac{e^2\hbar^2}{4m_e^2 c^2}\langle\frac{1}{r^3}\rangle_{nl}[j(j+1) - l(l+1) - 3/4]$$

### Darwin Term

$$H'_D = \frac{\pi e^2\hbar^2}{2m_e^2 c^2}\delta^3(\mathbf{r})$$

Only affects l = 0:
$$E^{(1)}_D = \frac{\pi e^2\hbar^2}{2m_e^2 c^2}|\psi_{n00}(0)|^2$$

### Combined Result

All three combine to give the famous formula:
$$\boxed{E^{(1)}_{\text{FS}} = \frac{E_n\alpha^2}{n^2}\left(\frac{n}{j+1/2} - \frac{3}{4}\right)}$$

### Example: n = 2 Shell

| State | j | E_FS/(E₂α²/4) |
|-------|---|---------------|
| 2S₁/₂ | 1/2 | 5/4 |
| 2P₁/₂ | 1/2 | 5/4 |
| 2P₃/₂ | 3/2 | 1/4 |

Note: 2S₁/₂ and 2P₁/₂ remain degenerate (Lamb shift lifts this at higher order).

---

## Practice Problems

1. Calculate E^(1)_FS for the 3D states.
2. Show that the three corrections combine to depend only on j.
3. What is the splitting between 2P₃/₂ and 2P₁/₂ in eV?

---

## Summary

| Correction | Formula | Affects |
|------------|---------|---------|
| Relativistic | -p⁴/(8m³c²) | All states |
| Spin-orbit | ξ L·S | l > 0 |
| Darwin | δ³(r) contact | l = 0 only |

---

**Next:** [Day_454_Saturday.md](Day_454_Saturday.md) — Stark Effect
