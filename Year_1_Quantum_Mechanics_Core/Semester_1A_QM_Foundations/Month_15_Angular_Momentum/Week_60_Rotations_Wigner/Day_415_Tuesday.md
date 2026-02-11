# Day 415: Euler Angles

## Overview
**Day 415** | Year 1, Month 15, Week 60 | Parameterizing Rotations

Euler angles provide a complete parameterization of 3D rotations, essential for describing angular momentum transformations and quantum gate decompositions.

---

## Core Content

### Euler Angle Convention (ZYZ)

Any rotation can be written as:
$$R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$$

**Parameters:**
- α ∈ [0, 2π): First rotation about z
- β ∈ [0, π]: Rotation about new y
- γ ∈ [0, 2π): Final rotation about new z

### Quantum Rotation Operator

$$\hat{D}(\alpha, \beta, \gamma) = e^{-i\alpha\hat{J}_z/\hbar} e^{-i\beta\hat{J}_y/\hbar} e^{-i\gamma\hat{J}_z/\hbar}$$

For spin-1/2:
$$D^{1/2}(\alpha, \beta, \gamma) = e^{-i\alpha\sigma_z/2} e^{-i\beta\sigma_y/2} e^{-i\gamma\sigma_z/2}$$

### Explicit Matrix (Spin-1/2)

$$D^{1/2}(\alpha, \beta, \gamma) = \begin{pmatrix} e^{-i(\alpha+\gamma)/2}\cos(\beta/2) & -e^{-i(\alpha-\gamma)/2}\sin(\beta/2) \\ e^{i(\alpha-\gamma)/2}\sin(\beta/2) & e^{i(\alpha+\gamma)/2}\cos(\beta/2) \end{pmatrix}$$

### Physical Interpretation

1. **α:** Rotate coordinate axes about z
2. **β:** Tilt the z-axis (colatitude)
3. **γ:** Rotate about the new z-axis

---

## Quantum Computing Connection

Every single-qubit gate can be decomposed:
$$U = e^{i\delta} R_z(\alpha) R_y(\beta) R_z(\gamma)$$

This is the **ZYZ decomposition** used by quantum compilers!

---

## Practice Problems
1. Write the Euler angles for a 90° rotation about x.
2. Show that β = 0 gives a pure z-rotation.
3. Find Euler angles for the Hadamard gate.

---

**Next:** [Day_416_Wednesday.md](Day_416_Wednesday.md) — Wigner D-Matrices
