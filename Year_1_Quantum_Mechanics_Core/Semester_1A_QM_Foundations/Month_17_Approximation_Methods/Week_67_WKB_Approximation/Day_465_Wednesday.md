# Day 465: Connection Formulas

## Overview
**Day 465** | Year 1, Month 17, Week 67 | Matching at Turning Points

---

## Core Content

### The Problem at Turning Points

WKB fails where p(x) = 0. Need local solutions to connect allowed ↔ forbidden regions.

### Airy Function Solution

Near turning point x₀ where V(x) ≈ V(x₀) + V'(x₀)(x-x₀):

Exact local solutions are **Airy functions** Ai(z), Bi(z).

### Connection Formulas (Right Turning Point)

From forbidden (left) to allowed (right):

$$\frac{1}{\sqrt{\kappa}}e^{-\int\kappa\,dx} \longleftrightarrow \frac{2}{\sqrt{p}}\sin\left(\int p\,dx + \frac{\pi}{4}\right)$$

$$\frac{1}{\sqrt{\kappa}}e^{+\int\kappa\,dx} \longleftrightarrow \frac{1}{\sqrt{p}}\cos\left(\int p\,dx + \frac{\pi}{4}\right)$$

### The π/4 Phase Shift

Each turning point contributes π/4 to the phase — crucial for quantization!

---

**Next:** [Day_466_Thursday.md](Day_466_Thursday.md) — Bound State Quantization
