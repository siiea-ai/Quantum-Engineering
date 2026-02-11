# Day 333: Simple Systems Preview

## Overview

**Month 12, Week 48, Day 4 — Thursday**

Today we preview the simple quantum systems you'll study in Year 1: particle in a box, harmonic oscillator, and hydrogen atom.

## Learning Objectives

1. See the Schrödinger equation in action
2. Understand energy quantization
3. Preview wavefunctions
4. Connect to Year 0 math

---

## 1. Particle in a Box

### The Problem

$$V(x) = \begin{cases} 0 & 0 < x < L \\ \infty & \text{otherwise} \end{cases}$$

### Solutions

$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)$$

$$E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$

### Year 0 Connection
- Eigenvalue problem for $-d^2/dx^2$
- Orthogonal basis (Fourier series)

---

## 2. Harmonic Oscillator

### The Problem

$$V(x) = \frac{1}{2}m\omega^2 x^2$$

### Solutions

$$\psi_n(x) = N_n H_n(\xi)e^{-\xi^2/2}$$

$$E_n = \hbar\omega\left(n + \frac{1}{2}\right)$$

### Year 0 Connection
- Hermite polynomials (special functions)
- Ladder operators (Lie algebra)
- Your capstone project!

---

## 3. Hydrogen Atom

### The Problem

$$V(r) = -\frac{e^2}{4\pi\epsilon_0 r}$$

### Solutions

$$\psi_{nlm}(r,\theta,\phi) = R_{nl}(r)Y_l^m(\theta,\phi)$$

$$E_n = -\frac{13.6\text{ eV}}{n^2}$$

### Year 0 Connection
- Separation of variables (PDEs)
- Spherical harmonics (group theory)
- Laguerre polynomials (special functions)
- Angular momentum (SU(2))

---

## Summary

### The Pattern

All these systems are **eigenvalue problems**:

$$\hat{H}\psi_n = E_n\psi_n$$

Year 0 has prepared you to solve them!

---

## Preview: Day 334

Tomorrow: **Interpretation and Philosophy** — what does quantum mechanics mean?
