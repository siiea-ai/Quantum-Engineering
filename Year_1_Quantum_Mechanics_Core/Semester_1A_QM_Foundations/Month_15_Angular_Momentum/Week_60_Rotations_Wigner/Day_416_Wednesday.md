# Day 416: Wigner D-Matrices

## Overview
**Day 416** | Year 1, Month 15, Week 60 | Rotation Matrix Elements

Wigner D-matrices are the matrix elements of rotation operators in the angular momentum basis—fundamental to understanding how quantum states transform under rotations.

---

## Core Content

### Definition

$$D^j_{m'm}(\alpha, \beta, \gamma) = \langle j, m' | \hat{D}(\alpha, \beta, \gamma) | j, m\rangle$$

where D̂(α,β,γ) is the rotation operator.

### Factorization

$$\boxed{D^j_{m'm}(\alpha, \beta, \gamma) = e^{-im'\alpha} d^j_{m'm}(\beta) e^{-im\gamma}}$$

The **reduced rotation matrix** d^j_{m'm}(β) contains the non-trivial angular dependence.

### Small d-Matrix (Spin-1/2)

$$d^{1/2}(\beta) = \begin{pmatrix} \cos(\beta/2) & -\sin(\beta/2) \\ \sin(\beta/2) & \cos(\beta/2) \end{pmatrix}$$

### Small d-Matrix (Spin-1)

$$d^1(\beta) = \begin{pmatrix} \frac{1+\cos\beta}{2} & -\frac{\sin\beta}{\sqrt{2}} & \frac{1-\cos\beta}{2} \\ \frac{\sin\beta}{\sqrt{2}} & \cos\beta & -\frac{\sin\beta}{\sqrt{2}} \\ \frac{1-\cos\beta}{2} & \frac{\sin\beta}{\sqrt{2}} & \frac{1+\cos\beta}{2} \end{pmatrix}$$

### Orthogonality

$$\int_0^{2\pi}d\alpha \int_0^\pi \sin\beta\,d\beta \int_0^{2\pi}d\gamma\, D^{j*}_{m'n'}D^j_{mn} = \frac{8\pi^2}{2j+1}\delta_{m'm}\delta_{n'n}$$

### Transformation of States

Under rotation R, a state transforms as:
$$|j, m'\rangle = \sum_m D^j_{m'm}(R) |j, m\rangle$$

---

## Quantum Computing Connection

D-matrices for j=1/2 are exactly single-qubit gates:
$$D^{1/2}(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$$

---

## Practice Problems
1. Calculate D^{1/2}(0, π/2, 0).
2. Verify d^1_{00}(β) = cos(β).
3. Show that D^j_{m'm}(0, 0, 0) = δ_{m'm}.

---

**Next:** [Day_417_Thursday.md](Day_417_Thursday.md) — Wigner-Eckart Theorem
