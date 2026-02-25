# Week 154: Variational and WKB Methods - Comprehensive Review Guide

## Table of Contents
1. [The Variational Principle](#1-the-variational-principle)
2. [Constructing Trial Wavefunctions](#2-constructing-trial-wavefunctions)
3. [Classical Applications](#3-classical-applications)
4. [The WKB Approximation](#4-the-wkb-approximation)
5. [Connection Formulas](#5-connection-formulas)
6. [Tunneling and Decay](#6-tunneling-and-decay)
7. [Born-Oppenheimer Approximation](#7-born-oppenheimer-approximation)
8. [Advanced Topics](#8-advanced-topics)

---

## 1. The Variational Principle

### Fundamental Theorem

The variational principle is one of the most powerful tools in quantum mechanics. It states that for any trial wavefunction $|\tilde{\psi}\rangle$, normalized so that $\langle\tilde{\psi}|\tilde{\psi}\rangle = 1$:

$$\boxed{E_0 \leq \langle\tilde{\psi}|H|\tilde{\psi}\rangle}$$

The equality holds if and only if $|\tilde{\psi}\rangle$ is the exact ground state.

### Rigorous Proof

Let $\{|n\rangle\}$ be the complete set of energy eigenstates with eigenvalues $E_n$ where $E_0 \leq E_1 \leq E_2 \leq \cdots$.

Any trial state can be expanded:
$$|\tilde{\psi}\rangle = \sum_{n=0}^{\infty} c_n |n\rangle$$

with normalization $\sum_n |c_n|^2 = 1$.

The expectation value of energy:
$$\langle\tilde{\psi}|H|\tilde{\psi}\rangle = \sum_n |c_n|^2 E_n$$

Since $E_n \geq E_0$ for all $n$:
$$\sum_n |c_n|^2 E_n \geq E_0 \sum_n |c_n|^2 = E_0$$

Therefore: $\langle\tilde{\psi}|H|\tilde{\psi}\rangle \geq E_0$ $\square$

### The Rayleigh-Ritz Method

For unnormalized trial functions, use the Rayleigh quotient:

$$E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$

**Procedure:**
1. Choose $\psi(\alpha_1, \alpha_2, \ldots)$ with variational parameters
2. Compute $E(\alpha_1, \alpha_2, \ldots)$
3. Find stationary points: $\frac{\partial E}{\partial \alpha_i} = 0$
4. The minimum value is an upper bound on $E_0$

### Extension to Excited States

**Theorem:** If the trial function is orthogonal to the ground state, the variational principle gives an upper bound on the first excited state:

If $\langle\tilde{\psi}|\psi_0\rangle = 0$, then $E[\tilde{\psi}] \geq E_1$

This can be extended to higher excited states by ensuring orthogonality to all lower states.

---

## 2. Constructing Trial Wavefunctions

### Guidelines for Good Trial Functions

1. **Correct symmetry:** Match the symmetry of the Hamiltonian and expected ground state
2. **Proper boundary conditions:** Vanish at infinity, satisfy any constraints
3. **Cusp conditions:** For Coulomb potentials, include proper cusp behavior
4. **Physical intuition:** Incorporate known limiting behaviors
5. **Computational tractability:** Must be able to evaluate integrals

### Common Trial Function Forms

**Gaussian:**
$$\psi(r) = Ae^{-\alpha r^2}$$
- Easy to integrate analytically
- Missing cusp for Coulomb problems
- Good for harmonic oscillator-like systems

**Exponential (Hydrogenic):**
$$\psi(r) = Ae^{-\alpha r}$$
- Correct cusp condition for hydrogen
- Natural for atomic problems
- Easy radial integration

**Slater-type orbitals:**
$$\psi(r) = Ar^{n-1}e^{-\zeta r}Y_l^m(\theta,\phi)$$
- Physically motivated for atoms
- Harder to integrate for multi-center problems

**Jastrow factor (for correlation):**
$$\Psi = \Phi_0 \cdot \exp\left(-\sum_{i<j}u(r_{ij})\right)$$
- Captures electron-electron correlation
- Used in quantum Monte Carlo

### Linear Variational Method

Use a trial function as a linear combination:
$$|\tilde{\psi}\rangle = \sum_{i=1}^N c_i |\phi_i\rangle$$

Minimizing $E$ with respect to $c_i$ gives the generalized eigenvalue problem:
$$\sum_j (H_{ij} - ES_{ij})c_j = 0$$

where $H_{ij} = \langle\phi_i|H|\phi_j\rangle$ and $S_{ij} = \langle\phi_i|\phi_j\rangle$.

The N eigenvalues are upper bounds on the first N energy levels.

---

## 3. Classical Applications

### Hydrogen Ground State

**Trial function:** $\psi(r) = Ae^{-\alpha r}$

**Normalization:** $A = \sqrt{\alpha^3/\pi}$

**Kinetic energy:**
$$\langle T\rangle = \frac{\hbar^2\alpha^2}{2m}$$

**Potential energy:**
$$\langle V\rangle = -\frac{e^2\alpha}{4\pi\epsilon_0}$$

**Total energy:**
$$E(\alpha) = \frac{\hbar^2\alpha^2}{2m} - \frac{e^2\alpha}{4\pi\epsilon_0}$$

**Minimization:**
$$\frac{dE}{d\alpha} = \frac{\hbar^2\alpha}{m} - \frac{e^2}{4\pi\epsilon_0} = 0$$

$$\alpha_{\text{opt}} = \frac{me^2}{4\pi\epsilon_0\hbar^2} = \frac{1}{a_0}$$

**Result:** $E_0 = -13.6$ eV (exact!)

The exponential trial function happens to be the exact ground state wavefunction.

### Helium Ground State

**Trial function:**
$$\psi(\mathbf{r}_1, \mathbf{r}_2) = \frac{Z_{\text{eff}}^3}{\pi a_0^3}e^{-Z_{\text{eff}}(r_1 + r_2)/a_0}$$

**Energy functional:**
$$E(Z_{\text{eff}}) = 2Z_{\text{eff}}^2 E_1 - 4ZZ_{\text{eff}} E_1 + \frac{5Z_{\text{eff}}}{4}\frac{e^2}{4\pi\epsilon_0 a_0}$$

where $E_1 = -13.6$ eV.

**Minimization:**
$$\frac{dE}{dZ_{\text{eff}}} = 0 \Rightarrow Z_{\text{eff}} = Z - \frac{5}{16} = 2 - 0.3125 = \frac{27}{16} \approx 1.69$$

**Result:** $E_0 \approx -77.5$ eV (experimental: $-78.98$ eV, error $\approx 2\%$)

**Physical interpretation:** Each electron screens the nuclear charge from the other.

### Harmonic Oscillator with Gaussian Trial

**Trial function:** $\psi(x) = Ae^{-\alpha x^2}$

**For** $H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2$:

$$\langle T\rangle = \frac{\hbar^2\alpha}{2m}, \quad \langle V\rangle = \frac{m\omega^2}{8\alpha}$$

$$E(\alpha) = \frac{\hbar^2\alpha}{2m} + \frac{m\omega^2}{8\alpha}$$

Minimizing: $\alpha_{\text{opt}} = \frac{m\omega}{2\hbar}$

Result: $E_0 = \frac{\hbar\omega}{2}$ (exact!)

---

## 4. The WKB Approximation

### Derivation

Start with the time-independent Schrodinger equation:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

Write $\psi(x) = A(x)e^{iS(x)/\hbar}$ and expand in powers of $\hbar$:

**To zeroth order in $\hbar$:**
$$(S')^2 = 2m(E - V) = p^2(x)$$

This is the classical Hamilton-Jacobi equation! $S(x) = \pm\int p(x)dx$

**To first order in $\hbar$:**
$$A(x) = \frac{C}{\sqrt{|p(x)|}}$$

### WKB Wavefunction

**Classically allowed region** ($E > V$):
$$\boxed{\psi(x) = \frac{C}{\sqrt{p(x)}}\left[B_+ e^{i\int p\,dx/\hbar} + B_- e^{-i\int p\,dx/\hbar}\right]}$$

**Classically forbidden region** ($E < V$):
$$\boxed{\psi(x) = \frac{C}{\sqrt{\kappa(x)}}\left[D_+ e^{\int \kappa\,dx/\hbar} + D_- e^{-\int \kappa\,dx/\hbar}\right]}$$

where:
- $p(x) = \sqrt{2m(E - V(x))}$
- $\kappa(x) = \sqrt{2m(V(x) - E)}$

### Validity Condition

WKB is valid when the potential varies slowly on the scale of the de Broglie wavelength:

$$\left|\frac{d\lambda}{dx}\right| = \left|\frac{d}{dx}\frac{h}{p}\right| = \frac{h|p'|}{p^2} \ll 1$$

Equivalently: $\left|\frac{\hbar p'}{p^2}\right| \ll 1$

This breaks down at:
- Classical turning points ($p = 0$)
- Rapidly varying potentials

### Bohr-Sommerfeld Quantization

For a bound state between turning points $x_1$ and $x_2$:

$$\boxed{\oint p\,dx = \int_{x_1}^{x_2}p\,dx + \int_{x_2}^{x_1}p\,dx = 2\int_{x_1}^{x_2}p\,dx = \left(n + \frac{1}{2}\right)h}$$

The $1/2$ accounts for the phase shifts at turning points.

**Example - Harmonic oscillator:**
$$\int_{-x_0}^{x_0}\sqrt{2m(E - \frac{1}{2}m\omega^2 x^2)}\,dx = \left(n + \frac{1}{2}\right)h$$

Evaluating: $E_n = \left(n + \frac{1}{2}\right)\hbar\omega$ (exact!)

---

## 5. Connection Formulas

### The Problem

At classical turning points, $p(x) \to 0$, so the WKB amplitude $1/\sqrt{p} \to \infty$. We need to match solutions across these points.

### Airy Function Solution

Near a turning point at $x = x_0$ where $V(x) \approx E + V'(x_0)(x - x_0)$:

The exact solution involves Airy functions $\text{Ai}(z)$ and $\text{Bi}(z)$.

### Connection Formulas (Linear Turning Point)

**Case 1:** Allowed region on left, forbidden on right

At $x < x_0$ (allowed):
$$\psi \sim \frac{2}{\sqrt{p}}\sin\left(\frac{1}{\hbar}\int_x^{x_0}p\,dx' + \frac{\pi}{4}\right)$$

At $x > x_0$ (forbidden):
$$\psi \sim \frac{1}{\sqrt{\kappa}}e^{-\frac{1}{\hbar}\int_{x_0}^x\kappa\,dx'}$$

**Case 2:** Forbidden region on left, allowed on right

At $x < x_0$ (forbidden):
$$\psi \sim \frac{1}{\sqrt{\kappa}}e^{-\frac{1}{\hbar}\int_x^{x_0}\kappa\,dx'}$$

At $x > x_0$ (allowed):
$$\psi \sim \frac{2}{\sqrt{p}}\sin\left(\frac{1}{\hbar}\int_{x_0}^x p\,dx' + \frac{\pi}{4}\right)$$

### Phase Shift at Turning Point

Each turning point contributes a phase shift of $\pi/4$ in the WKB phase.

For a bound state between two turning points: total phase shift = $\pi/2$, leading to the $1/2$ in the quantization condition.

---

## 6. Tunneling and Decay

### Tunneling Through a Barrier

For a particle incident from the left on a barrier $V(x) > E$ between $x_1$ and $x_2$:

**Transmission coefficient:**
$$\boxed{T \approx \exp\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V(x) - E)}\,dx\right)}$$

This is the Gamow factor.

### Square Barrier (Exact)

For a rectangular barrier of height $V_0$ and width $a$:

$$T = \frac{1}{1 + \frac{V_0^2}{4E(V_0 - E)}\sinh^2(\kappa a)}$$

where $\kappa = \sqrt{2m(V_0 - E)}/\hbar$.

For thick barriers ($\kappa a \gg 1$):
$$T \approx 16\frac{E(V_0-E)}{V_0^2}e^{-2\kappa a}$$

### Alpha Decay

The nucleus can be modeled as a deep square well plus Coulomb barrier:

$$V(r) = \begin{cases}
-V_0 & r < R_{\text{nuc}} \\
\frac{2Ze^2}{4\pi\epsilon_0 r} & r > R_{\text{nuc}}
\end{cases}$$

**Decay rate:**
$$\Gamma = \frac{\hbar v}{2R_{\text{nuc}}} \cdot T$$

where $v$ is the alpha particle velocity inside the nucleus and $T$ is the tunneling probability through the Coulomb barrier.

**Geiger-Nuttall Law:**
$$\log t_{1/2} \propto Z/\sqrt{E_\alpha}$$

The WKB calculation reproduces this empirical law beautifully.

### Tunneling Time

The time for tunneling is approximately:
$$\tau \approx \frac{m \cdot (\text{barrier width})}{\hbar\kappa}$$

This is typically very short (femtoseconds for electrons).

---

## 7. Born-Oppenheimer Approximation

### Physical Motivation

In molecules:
- Electron mass $m_e \approx 10^{-3} \times$ nuclear mass
- Electrons move much faster than nuclei
- To electrons, nuclei appear stationary
- To nuclei, electrons appear as averaged charge cloud

### Mathematical Formulation

**Full Hamiltonian:**
$$H = T_{\text{nuc}} + T_{\text{elec}} + V_{\text{nn}} + V_{\text{en}} + V_{\text{ee}}$$

**Step 1:** Fix nuclear positions $\mathbf{R}$, solve electronic problem:
$$H_{\text{elec}}(\mathbf{R})\phi_n(\mathbf{r}; \mathbf{R}) = E_n^{\text{elec}}(\mathbf{R})\phi_n(\mathbf{r}; \mathbf{R})$$

**Step 2:** The electronic energy becomes the potential for nuclear motion:
$$V_{\text{eff}}(\mathbf{R}) = E_n^{\text{elec}}(\mathbf{R}) + V_{\text{nn}}(\mathbf{R})$$

**Step 3:** Solve nuclear Schrodinger equation:
$$[T_{\text{nuc}} + V_{\text{eff}}(\mathbf{R})]\chi(\mathbf{R}) = E\chi(\mathbf{R})$$

### Potential Energy Surfaces

The function $V_{\text{eff}}(\mathbf{R})$ defines a potential energy surface (PES) for nuclear motion.

Key features:
- **Equilibrium geometry:** Minimum of PES
- **Dissociation energy:** Depth of potential well
- **Vibrational frequencies:** Curvature at minimum
- **Reaction barriers:** Saddle points on PES

### Breaking the Born-Oppenheimer Approximation

The approximation fails when:
1. Electronic states are nearly degenerate (avoided crossings)
2. Nuclear velocities are high
3. Nonadiabatic transitions occur

These situations require treatment of **nonadiabatic dynamics**.

---

## 8. Advanced Topics

### Excited States via Variational Method

**Linear variational method:** Expand in a basis, solve generalized eigenvalue problem.

**Constraint method:** Require trial function to be orthogonal to known lower states.

### Systematic Improvements

**Hartree-Fock:** Variational principle with Slater determinant ansatz

**Configuration Interaction:** Linear combination of Slater determinants

**Variational Monte Carlo:** Sample high-dimensional integrals stochastically

### WKB for Radial Problems

For central potentials, use the effective potential:
$$V_{\text{eff}}(r) = V(r) + \frac{\ell(\ell+1)\hbar^2}{2mr^2}$$

The quantization condition becomes:
$$\int_{r_1}^{r_2}\sqrt{2m(E - V_{\text{eff}}(r))}\,dr = \left(n_r + \frac{1}{2}\right)\pi\hbar$$

### Double-Well Tunneling

For a symmetric double well, tunneling splits the would-be degenerate levels:

$$\Delta E = E_+ - E_- \approx \frac{\hbar\omega}{\pi}e^{-\gamma}$$

where $\gamma = \int_{x_1}^{x_2}\kappa\,dx/\hbar$ is the Gamow factor.

This leads to oscillation between wells with frequency $\omega_{\text{tunnel}} = \Delta E/\hbar$.

---

## Summary: Key Results

### Variational Principle
1. $E_0 \leq \langle\psi|H|\psi\rangle$ for any $|\psi\rangle$
2. Optimize trial parameters by $\partial E/\partial \alpha = 0$
3. Result is guaranteed upper bound

### WKB
1. Valid when $|d\lambda/dx| \ll 1$
2. Quantization: $\oint p\,dx = (n+1/2)h$
3. Tunneling: $T \approx e^{-2\int\kappa\,dx/\hbar}$

### Connection Formulas
1. Phase shift of $\pi/4$ at each turning point
2. Amplitude changes by factor of 2 across turning point
3. Use Airy functions for exact matching

---

## References

1. Griffiths, D.J. *Introduction to Quantum Mechanics*, Chapter 8
2. Shankar, R. *Principles of Quantum Mechanics*, Chapter 16
3. Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 5
4. Landau, L.D. & Lifshitz, E.M. *Quantum Mechanics*, Chapter 7
5. [Physics LibreTexts - WKB Approximation](https://phys.libretexts.org/)

---

**Word Count:** ~2400 words
