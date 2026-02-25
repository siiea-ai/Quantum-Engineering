# Week 148: Tunneling and WKB — Review Guide

## Introduction

Quantum tunneling is a quintessential quantum phenomenon: particles penetrate barriers that are classically impenetrable. This week covers exact solutions for simple barriers and the powerful WKB (Wentzel-Kramers-Brillouin) semiclassical approximation. These techniques are essential for qualifying exams and appear in applications from alpha decay to scanning tunneling microscopy.

---

## 1. Step Potential

### 1.1 Setup

**Potential:**
$$V(x) = \begin{cases} 0 & x < 0 \\ V_0 & x > 0 \end{cases}$$

### 1.2 Case 1: $$E > V_0$$ (Above Barrier)

**Wave numbers:**
$$k_1 = \frac{\sqrt{2mE}}{\hbar}, \quad k_2 = \frac{\sqrt{2m(E-V_0)}}{\hbar}$$

**Wavefunctions:**
- Region I ($$x < 0$$): $$\psi_I = e^{ik_1x} + Re^{-ik_1x}$$ (incident + reflected)
- Region II ($$x > 0$$): $$\psi_{II} = Te^{ik_2x}$$ (transmitted)

**Matching at $$x = 0$$:**
$$1 + R = T, \quad ik_1(1 - R) = ik_2 T$$

**Coefficients:**
$$\boxed{R = \frac{k_1 - k_2}{k_1 + k_2}, \quad T = \frac{2k_1}{k_1 + k_2}}$$

**Reflection and Transmission Probabilities:**
$$\boxed{\mathcal{R} = |R|^2 = \left(\frac{k_1-k_2}{k_1+k_2}\right)^2, \quad \mathcal{T} = \frac{k_2}{k_1}|T|^2 = \frac{4k_1k_2}{(k_1+k_2)^2}}$$

Note: $$\mathcal{R} + \mathcal{T} = 1$$ (probability conservation).

### 1.3 Case 2: $$E < V_0$$ (Below Barrier)

**In Region II:** $$\kappa = \sqrt{2m(V_0-E)}/\hbar$$, wave decays: $$\psi_{II} = Ce^{-\kappa x}$$

**Result:** $$\mathcal{R} = 1$$, $$\mathcal{T} = 0$$ — total reflection

But the wavefunction penetrates a distance $$\sim 1/\kappa$$ into the barrier (evanescent wave).

---

## 2. Rectangular Barrier

### 2.1 Setup

**Potential:**
$$V(x) = \begin{cases} 0 & x < 0 \\ V_0 & 0 < x < a \\ 0 & x > a \end{cases}$$

### 2.2 Tunneling ($$E < V_0$$)

**Inside barrier:** $$\kappa = \sqrt{2m(V_0-E)}/\hbar$$

**Transmission Coefficient:**
$$\boxed{\mathcal{T} = \frac{1}{1 + \frac{V_0^2\sinh^2(\kappa a)}{4E(V_0-E)}}}$$

**Approximation for thick barrier** ($$\kappa a \gg 1$$):
$$\boxed{\mathcal{T} \approx 16\frac{E(V_0-E)}{V_0^2}e^{-2\kappa a}}$$

The dominant factor is $$e^{-2\kappa a}$$: exponential suppression.

### 2.3 Above Barrier ($$E > V_0$$)

**Inside barrier:** $$k' = \sqrt{2m(E-V_0)}/\hbar$$

**Transmission:**
$$\mathcal{T} = \frac{1}{1 + \frac{V_0^2\sin^2(k'a)}{4E(E-V_0)}}$$

**Resonances:** $$\mathcal{T} = 1$$ when $$k'a = n\pi$$ (transmission resonances).

### 2.4 Physical Applications

- Scanning tunneling microscope (STM)
- Tunnel diodes
- Nuclear fusion in stars
- Josephson junctions

---

## 3. The WKB Approximation

### 3.1 Motivation

The WKB method is a semiclassical approximation valid when the potential varies slowly compared to the de Broglie wavelength:
$$\left|\frac{d\lambda}{dx}\right| \ll 1$$

### 3.2 Derivation

Write $$\psi(x) = e^{iS(x)/\hbar}$$ and substitute into Schrödinger equation:
$$(S')^2 - i\hbar S'' = 2m(E - V(x))$$

Expand $$S = S_0 + (\hbar/i)S_1 + ...$$:

**Leading order:** $$(S_0')^2 = 2m(E-V) = p^2(x)$$
$$S_0 = \pm\int p(x)\,dx$$

**Next order:** $$S_1 = \frac{1}{2}\ln p(x)$$

### 3.3 WKB Wavefunctions

**Classically allowed region** ($$E > V$$):
$$\boxed{\psi(x) = \frac{A}{\sqrt{p(x)}}\exp\left[\frac{i}{\hbar}\int^x p(x')\,dx'\right] + \frac{B}{\sqrt{p(x)}}\exp\left[-\frac{i}{\hbar}\int^x p(x')\,dx'\right]}$$

**Classically forbidden region** ($$E < V$$):
$$\boxed{\psi(x) = \frac{C}{\sqrt{\kappa(x)}}\exp\left[-\frac{1}{\hbar}\int^x \kappa(x')\,dx'\right] + \frac{D}{\sqrt{\kappa(x)}}\exp\left[\frac{1}{\hbar}\int^x \kappa(x')\,dx'\right]}$$

where $$p(x) = \sqrt{2m(E-V(x))}$$ and $$\kappa(x) = \sqrt{2m(V(x)-E)}$$.

### 3.4 Validity Condition

WKB is valid when:
$$\boxed{\left|\frac{d\lambda}{dx}\right| = \left|\frac{d}{dx}\frac{h}{p}\right| = \frac{\hbar}{p^2}\left|\frac{dp}{dx}\right| \ll 1}$$

This fails at **classical turning points** where $$p = 0$$.

---

## 4. Connection Formulas

### 4.1 The Problem at Turning Points

At a turning point $$x = a$$ where $$E = V(a)$$:
- $$p(a) = 0$$, so WKB diverges
- Need to match solutions across turning point

### 4.2 Linear Approximation Near Turning Point

Near $$x = a$$: $$V(x) \approx V(a) + V'(a)(x-a) = E + V'(a)(x-a)$$

The Schrödinger equation becomes the **Airy equation**, with solutions in terms of Airy functions.

### 4.3 Connection Formulas

For a right-hand turning point (allowed region on left):

$$\frac{2}{\sqrt{p}}\cos\left[\frac{1}{\hbar}\int_a^x p\,dx' - \frac{\pi}{4}\right] \longleftrightarrow \frac{1}{\sqrt{\kappa}}e^{-\frac{1}{\hbar}\int_x^a \kappa\,dx'}$$

$$\frac{1}{\sqrt{p}}\sin\left[\frac{1}{\hbar}\int_a^x p\,dx' - \frac{\pi}{4}\right] \longleftrightarrow \frac{-1}{\sqrt{\kappa}}e^{\frac{1}{\hbar}\int_x^a \kappa\,dx'}$$

### 4.4 Mnemonic

"One to one, two to two" — the coefficient 2 goes with cosine and decaying exponential.

---

## 5. WKB Bound State Quantization

### 5.1 Setup

For a potential well with turning points at $$x_1$$ and $$x_2$$:
- Apply connection formulas at both turning points
- Require single-valued wavefunction

### 5.2 Bohr-Sommerfeld Quantization

$$\boxed{\oint p\,dx = \int_{x_1}^{x_2} p(x)\,dx = \pi\hbar\left(n + \frac{1}{2}\right)}$$

The $$+1/2$$ comes from the connection formula phases.

### 5.3 Examples

**Harmonic oscillator:** $$V = \frac{1}{2}m\omega^2x^2$$
$$\int_{-x_0}^{x_0}\sqrt{2m(E - \frac{1}{2}m\omega^2x^2)}\,dx = \pi\hbar(n + \frac{1}{2})$$
$$E_n = \hbar\omega(n + 1/2)$$ — exact result!

**Linear potential:** $$V = mgx$$ for $$x > 0$$
$$\int_0^{E/(mg)} \sqrt{2m(E - mgx)}\,dx = \pi\hbar(n + 1/2)$$

---

## 6. WKB Tunneling

### 6.1 Tunneling Probability

For a barrier between turning points $$x_1$$ and $$x_2$$:
$$\boxed{\mathcal{T} \approx e^{-2\gamma}, \quad \gamma = \frac{1}{\hbar}\int_{x_1}^{x_2}\kappa(x)\,dx}$$

This is the **Gamow factor**.

### 6.2 Alpha Decay

For alpha particle tunneling through Coulomb barrier:
$$V(r) = \frac{2Ze^2}{4\pi\epsilon_0 r} \quad \text{for } r > R$$

**Gamow factor:**
$$\gamma = \frac{2Ze^2}{4\pi\epsilon_0\hbar v}\left[\cos^{-1}\sqrt{\frac{R}{b}} - \sqrt{\frac{R}{b}\left(1-\frac{R}{b}\right)}\right]$$

where $$b = 2Ze^2/(4\pi\epsilon_0 E)$$ is the classical turning point.

**Decay rate:** $$\lambda = f \cdot e^{-2\gamma}$$, where $$f$$ is the "attempt frequency."

### 6.3 Geiger-Nuttall Law

$$\log_{10} t_{1/2} \propto Z/\sqrt{E}$$

Explains wide range of alpha decay half-lives (microseconds to billions of years).

---

## 7. Summary of Key Results

| Method | Formula | Application |
|--------|---------|-------------|
| Rectangular barrier | $$\mathcal{T} \sim e^{-2\kappa a}$$ | STM, tunnel diode |
| WKB wavefunction | $$\psi \propto p^{-1/2}e^{\pm i\int p\,dx/\hbar}$$ | Slowly varying potentials |
| Bound state quantization | $$\oint p\,dx = 2\pi\hbar(n+1/2)$$ | Energy levels |
| WKB tunneling | $$\mathcal{T} = e^{-2\gamma}$$ | Alpha decay |

---

## 8. Common Exam Problem Types

1. **Calculate transmission coefficient** for step/barrier
2. **Apply WKB** to find bound state energies
3. **Estimate tunneling rate** using Gamow factor
4. **Determine validity** of WKB approximation
5. **Matching problems** at interfaces
6. **Alpha decay lifetime** calculations

---

## 9. Month 37 Integration

This week synthesizes all Month 37 topics:

| Week | Topics | Connections |
|------|--------|-------------|
| 145 | Mathematical framework | Operators, eigenvalues used everywhere |
| 146 | Measurement & dynamics | Time evolution of tunneling particles |
| 147 | 1D systems | Wells, oscillators as reference systems |
| 148 | Tunneling & WKB | Combines all techniques |

---

*Review Guide for Week 148 — Tunneling and WKB*
*Month 37: QM Foundations Review I*
