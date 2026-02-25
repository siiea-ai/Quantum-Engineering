# Day 315: Mathematics Integration Exam

## Overview

**Month 12, Week 45, Day 7 — Sunday**

Today tests mastery of all Year 0 mathematics through a comprehensive integration exam. Problems require combining multiple mathematical domains.

## Exam Instructions

- Time: 4 hours
- Open notes permitted
- Show all work
- Use mathematical rigor

---

## Part A: Calculus and Analysis (25 points)

### Problem A1 (10 points)

Evaluate using contour integration:
$$I = \int_{-\infty}^{\infty} \frac{x\sin x}{x^2 + 1}dx$$

### Problem A2 (15 points)

For $f(r, \theta) = r^2 e^{-r}\cos\theta$:
(a) Find $\nabla f$ in spherical coordinates
(b) Evaluate $\int_0^{2\pi}\int_0^{\pi}\int_0^{\infty} f \, r^2\sin\theta \, dr\,d\theta\,d\phi$

---

## Part B: Linear Algebra (25 points)

### Problem B1 (12 points)

For the matrix:
$$A = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$$

(a) Find all eigenvalues
(b) Write the spectral decomposition
(c) Compute $e^{iAt}$

### Problem B2 (13 points)

Prove that for any Hermitian matrix $H$ and unitary matrix $U$:
(a) $UHU^\dagger$ is Hermitian
(b) $H$ and $UHU^\dagger$ have the same eigenvalues
(c) If $[H_1, H_2] = 0$, they share a common eigenbasis

---

## Part C: Differential Equations (25 points)

### Problem C1 (15 points)

Solve the eigenvalue problem:
$$-\frac{d^2\psi}{dx^2} + x^2\psi = E\psi$$

(a) Use series solution to find the first three eigenvalues
(b) Verify orthogonality of the first two eigenfunctions
(c) Explain the connection to the quantum harmonic oscillator

### Problem C2 (10 points)

Using Green's function method, solve:
$$y'' + y = \delta(x - \pi/2), \quad y(0) = y(\pi) = 0$$

---

## Part D: Group Theory (25 points)

### Problem D1 (15 points)

For SU(2):
(a) Write the commutation relations for generators $T_a = \sigma_a/2$
(b) Compute $e^{i\theta\sigma_z/2}$ explicitly
(c) Show this is a rotation on the Bloch sphere

### Problem D2 (10 points)

For the addition $j_1 = 1 \otimes j_2 = 1/2$:
(a) List all resulting $(j, m)$ states
(b) Express $|3/2, 1/2\rangle$ in the uncoupled basis
(c) Verify normalization

---

## Solutions Guide

### A1 Solution

Use $\text{Im}(\int \frac{ze^{iz}}{z^2+1}dz)$. Pole at $z = i$ in upper half-plane.

$$\text{Res}(z=i) = \frac{ie^{-1}}{2i} = \frac{e^{-1}}{2}$$

$$I = \text{Im}(2\pi i \cdot \frac{e^{-1}}{2}) = \frac{\pi}{e}$$

### B1 Solution

Characteristic polynomial: $(\lambda - 2)^3 - 2(\lambda - 2) = 0$

$\lambda = 2, 2 \pm \sqrt{2}$

### C1 Solution

$E_n = 2n + 1$ (in units where $\hbar\omega = 1$)

$\psi_n \propto H_n(x)e^{-x^2/2}$

### D2 Solution

$|3/2, 1/2\rangle = \sqrt{1/3}|1,1\rangle|1/2,-1/2\rangle + \sqrt{2/3}|1,0\rangle|1/2,1/2\rangle$

---

## Self-Assessment

Score yourself:
- 90-100: Ready for Year 1
- 80-89: Review weak areas
- 70-79: Additional practice needed
- Below 70: Revisit core material

---

## Preview: Week 46

Next week: **Physics Comprehensive Review** — classical mechanics synthesis.
