# Week 147: One-Dimensional Systems — Review Guide

## Introduction

One-dimensional quantum systems are the foundation of quantum mechanics problem-solving. These exactly solvable models appear on every qualifying exam and provide intuition for more complex systems. This guide covers the essential 1D systems in detail.

---

## 1. The Infinite Square Well

### 1.1 Setup

**Potential:**
$$V(x) = \begin{cases} 0 & 0 < x < L \\ \infty & \text{otherwise} \end{cases}$$

**Boundary Conditions:**
$$\psi(0) = \psi(L) = 0$$

### 1.2 Solution

Inside the well, the Schrödinger equation becomes:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$

General solution: $$\psi(x) = A\sin(kx) + B\cos(kx)$$ where $$k = \sqrt{2mE}/\hbar$$

Applying $$\psi(0) = 0$$: $$B = 0$$

Applying $$\psi(L) = 0$$: $$kL = n\pi$$ for integer $$n$$

### 1.3 Energy Levels and Wavefunctions

$$\boxed{E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \quad n = 1, 2, 3, ...}$$

$$\boxed{\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)}$$

**Key Properties:**
- Ground state energy $$E_1 \neq 0$$ (zero-point energy)
- Energy grows as $$n^2$$
- $$n-1$$ nodes in $$\psi_n$$
- Parity: $$\psi_n$$ even/odd about center for odd/even $$n$$

### 1.4 Matrix Elements

$$\langle n|\hat{x}|m\rangle = \begin{cases} 0 & n-m \text{ even} \\ \frac{8Lnm}{\pi^2(n^2-m^2)^2} & n-m \text{ odd} \end{cases}$$

$$\langle n|\hat{x}|n\rangle = \frac{L}{2}$$

### 1.5 Momentum Representation

$$\tilde{\psi}_n(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_0^L \psi_n(x)e^{-ipx/\hbar}dx$$

---

## 2. The Finite Square Well

### 2.1 Setup

**Potential:**
$$V(x) = \begin{cases} 0 & |x| < a \\ V_0 & |x| > a \end{cases}$$

### 2.2 Bound States ($$E < V_0$$)

**Inside ($$|x| < a$$):**
$$\psi''= -k^2\psi, \quad k = \sqrt{2mE}/\hbar$$

**Outside ($$|x| > a$$):**
$$\psi'' = \kappa^2\psi, \quad \kappa = \sqrt{2m(V_0-E)}/\hbar$$

### 2.3 Matching Conditions

For continuity of $$\psi$$ and $$\psi'$$ at $$x = a$$:

**Even solutions:** $$k\tan(ka) = \kappa$$

**Odd solutions:** $$-k\cot(ka) = \kappa$$

### 2.4 Graphical Solution

Define dimensionless variables:
$$z = ka, \quad z_0 = \frac{a}{\hbar}\sqrt{2mV_0}$$

**Even states:** $$\tan z = \sqrt{(z_0/z)^2 - 1}$$

**Odd states:** $$-\cot z = \sqrt{(z_0/z)^2 - 1}$$

### 2.5 Number of Bound States

- Always at least one bound state (for any $$V_0 > 0$$)
- Number of bound states: $$N = \lfloor z_0/(\pi/2) \rfloor + 1$$
- Deep well ($$z_0 \gg 1$$): approaches infinite well result

### 2.6 Scattering States ($$E > V_0$$)

For $$E > V_0$$, states are continuous (not quantized):
- Reflection coefficient $$R$$
- Transmission coefficient $$T$$
- $$R + T = 1$$

---

## 3. Harmonic Oscillator — Analytic Method

### 3.1 Setup

$$V(x) = \frac{1}{2}m\omega^2x^2$$

Schrödinger equation:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2x^2\psi = E\psi$$

### 3.2 Dimensionless Form

Define $$\xi = \sqrt{m\omega/\hbar}\, x$$:
$$\frac{d^2\psi}{d\xi^2} = (\xi^2 - 2\epsilon)\psi, \quad \epsilon = E/(\hbar\omega)$$

### 3.3 Asymptotic Behavior

For large $$|\xi|$$: $$\psi \sim e^{-\xi^2/2}$$

Ansatz: $$\psi(\xi) = H(\xi)e^{-\xi^2/2}$$

### 3.4 Hermite Polynomials

$$H(\xi)$$ satisfies Hermite's equation:
$$H'' - 2\xi H' + 2nH = 0$$

**Hermite polynomials:**
- $$H_0(\xi) = 1$$
- $$H_1(\xi) = 2\xi$$
- $$H_2(\xi) = 4\xi^2 - 2$$
- $$H_3(\xi) = 8\xi^3 - 12\xi$$

**Recurrence:** $$H_{n+1} = 2\xi H_n - 2nH_{n-1}$$

### 3.5 Complete Solution

$$\boxed{E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, ...}$$

$$\boxed{\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}\frac{1}{\sqrt{2^n n!}}H_n(\xi)e^{-\xi^2/2}}$$

### 3.6 Ground State

$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/(2\hbar)}$$

This is a Gaussian — a minimum uncertainty state.

---

## 4. Harmonic Oscillator — Algebraic Method

### 4.1 Ladder Operators

**Lowering (annihilation) operator:**
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\hat{x} + \frac{i\hat{p}}{\sqrt{2m\omega\hbar}}$$

**Raising (creation) operator:**
$$\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\hat{x} - \frac{i\hat{p}}{\sqrt{2m\omega\hbar}}$$

### 4.2 Commutation Relations

$$[\hat{a}, \hat{a}^\dagger] = 1$$

### 4.3 Hamiltonian in Terms of Ladder Operators

$$\boxed{\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\left(\hat{n} + \frac{1}{2}\right)}$$

where $$\hat{n} = \hat{a}^\dagger\hat{a}$$ is the **number operator**.

### 4.4 Action on States

$$\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$$

$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

$$\hat{n}|n\rangle = n|n\rangle$$

### 4.5 Building the Spectrum

1. Ground state $$|0\rangle$$ satisfies $$\hat{a}|0\rangle = 0$$
2. Excited states: $$|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$$
3. Energies: $$E_n = \hbar\omega(n + 1/2)$$

### 4.6 Position and Momentum Operators

$$\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$$

$$\hat{p} = i\sqrt{\frac{m\omega\hbar}{2}}(\hat{a}^\dagger - \hat{a})$$

### 4.7 Matrix Elements

$$\langle n|\hat{x}|m\rangle = \sqrt{\frac{\hbar}{2m\omega}}(\sqrt{m}\delta_{n,m-1} + \sqrt{m+1}\delta_{n,m+1})$$

$$\langle n|\hat{x}|n\rangle = 0$$

$$\langle n|\hat{x}^2|n\rangle = \frac{\hbar}{m\omega}\left(n + \frac{1}{2}\right)$$

### 4.8 Coherent States

Eigenstates of $$\hat{a}$$:
$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$

$$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

**Properties:**
- Minimum uncertainty: $$\Delta x \Delta p = \hbar/2$$
- $$\langle\hat{x}\rangle$$ oscillates classically
- Poissonian photon number distribution

---

## 5. Delta Function Potential

### 5.1 Attractive Delta Potential

$$V(x) = -\alpha\delta(x), \quad \alpha > 0$$

**Discontinuity condition:** Integrating Schrödinger equation across $$x=0$$:
$$\psi'(0^+) - \psi'(0^-) = -\frac{2m\alpha}{\hbar^2}\psi(0)$$

### 5.2 Bound State

There is exactly **one bound state**:

$$\boxed{E_0 = -\frac{m\alpha^2}{2\hbar^2}}$$

$$\psi_0(x) = \frac{\sqrt{m\alpha}}{\hbar}e^{-m\alpha|x|/\hbar^2}$$

### 5.3 Scattering States

For $$E > 0$$, incident wave from left:
$$\psi = \begin{cases} e^{ikx} + Re^{-ikx} & x < 0 \\ Te^{ikx} & x > 0 \end{cases}$$

**Reflection and transmission:**
$$R = \frac{1}{1 + 2\hbar^2 E/(m\alpha^2)}, \quad T = \frac{1}{1 + m\alpha^2/(2\hbar^2 E)}$$

Note: $$R + T = 1$$

### 5.4 Repulsive Delta Potential

$$V(x) = +\alpha\delta(x)$$

- No bound states
- Scattering only (same $$R$$, $$T$$ formulas)

### 5.5 Double Delta Potential

$$V(x) = -\alpha[\delta(x-a) + \delta(x+a)]$$

- Two bound states (symmetric and antisymmetric)
- Model for diatomic molecule (H$$_2^+$$ analog)

---

## 6. Wave Packets

### 6.1 Free Particle

General solution:
$$\psi(x,t) = \frac{1}{\sqrt{2\pi\hbar}}\int \tilde{\psi}(p)e^{i(px - p^2t/(2m\hbar))/\hbar}dp$$

### 6.2 Gaussian Wave Packet

Initial state:
$$\psi(x,0) = \left(\frac{1}{2\pi\sigma_0^2}\right)^{1/4}e^{ip_0x/\hbar}e^{-x^2/(4\sigma_0^2)}$$

**Time evolution:**
$$|\psi(x,t)|^2 = \frac{1}{\sqrt{2\pi}\sigma(t)}\exp\left[-\frac{(x-v_gt)^2}{2\sigma(t)^2}\right]$$

where:
- Group velocity: $$v_g = p_0/m$$
- Spreading: $$\sigma(t) = \sigma_0\sqrt{1 + (t/\tau)^2}$$
- Time scale: $$\tau = 2m\sigma_0^2/\hbar$$

### 6.3 Group and Phase Velocity

**Phase velocity:** $$v_p = \omega/k = \hbar k/(2m) = p/(2m)$$

**Group velocity:** $$v_g = d\omega/dk = \hbar k/m = p/m$$

For free particle, $$v_g = 2v_p$$.

### 6.4 Spreading

- Wave packet spreads because different momentum components travel at different speeds
- $$\sigma(t) \to \hbar t/(2m\sigma_0)$$ for $$t \gg \tau$$
- Narrower initial packet spreads faster

---

## 7. Summary of Key Results

| System | Energy Levels | Key Feature |
|--------|---------------|-------------|
| Infinite well | $$E_n = n^2\pi^2\hbar^2/(2mL^2)$$ | $$n = 1, 2, 3, ...$$ |
| Harmonic oscillator | $$E_n = \hbar\omega(n + 1/2)$$ | $$n = 0, 1, 2, ...$$ |
| Delta potential | $$E_0 = -m\alpha^2/(2\hbar^2)$$ | One bound state |

---

## 8. Common Exam Problem Types

1. **Calculate energy levels** for given potential
2. **Normalize wavefunctions** and verify orthogonality
3. **Expand arbitrary state** in energy eigenbasis
4. **Calculate expectation values** using ladder operators
5. **Time evolution** of superposition states
6. **Finite well:** determine number of bound states
7. **Scattering:** calculate R and T

---

*Review Guide for Week 147 — One-Dimensional Systems*
*Month 37: QM Foundations Review I*
