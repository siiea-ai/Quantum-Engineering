# Day 438: Total Fine Structure

## Overview
**Day 438** | Year 1, Month 16, Week 63 | Combining All Corrections

Today we combine all three fine structure corrections to obtain the famous total fine structure formula.

---

## Learning Objectives

1. Combine relativistic, spin-orbit, and Darwin terms
2. Derive the unified fine structure formula
3. Calculate energy shifts for specific states
4. Understand the remarkable simplification
5. Compare theory with experiment

---

## Core Content

### The Three Corrections

1. **Relativistic kinetic:** E_rel = -E_n(α²/n²)[n/(l+1/2) - 3/4]
2. **Spin-orbit:** E_SO = E_n(α²/n²)[j(j+1)-l(l+1)-3/4]/[l(l+1/2)(l+1)] (l>0)
3. **Darwin:** E_D = E_n(α²/n²)[n/(l+1/2)] (l=0 only)

### The Remarkable Result

All three combine to give:

$$\boxed{E^{(1)}_{\text{FS}} = E_n \frac{\alpha^2}{n^2}\left(\frac{n}{j+1/2} - \frac{3}{4}\right)}$$

This depends on **j only**, not separately on l and s!

### Fine Structure Energy Levels

$$E_{nj} = E_n\left[1 + \frac{\alpha^2}{n^2}\left(\frac{n}{j+1/2} - \frac{3}{4}\right)\right]$$

### Example: n = 2 Shell

| State | j | E^{(1)}_FS/E_2 |
|-------|---|----------------|
| 2S_{1/2} | 1/2 | α²(2 - 3/4) = 5α²/4 |
| 2P_{1/2} | 1/2 | α²(2 - 3/4) = 5α²/4 |
| 2P_{3/2} | 3/2 | α²(1 - 3/4) = α²/4 |

**Note:** 2S_{1/2} and 2P_{1/2} have the **same energy** to this order!
This is the "accidental" degeneracy that the Lamb shift breaks.

### Numerical Value

For hydrogen:
$$\Delta E_{\text{FS}} \sim \alpha^2 E_n \approx \frac{E_n}{(137)^2}$$

For n=2: E_FS ~ 10⁻⁴ eV ~ 10 GHz (microwave)

---

## Quantum Computing Connection

Fine structure sets:
- **Qubit transition frequencies** in atomic qubits
- **Spectral line widths** for laser addressing
- **Selection rules** for state manipulation

---

## Practice Problems

1. Calculate E^{(1)}_FS for the 3P_{3/2} state.
2. What is the fine structure splitting Δ = E(2P_{3/2}) - E(2P_{1/2})?
3. Show that j = 1/2 states have larger (more negative) fine structure shifts.

---

## Summary

| Result | Formula |
|--------|---------|
| Total FS | E_n(α²/n²)[n/(j+1/2) - 3/4] |
| Depends on | n and j only |
| Order | α² ~ 5×10⁻⁵ |

---

**Next:** [Day_439_Friday.md](Day_439_Friday.md) — Good Quantum Numbers
