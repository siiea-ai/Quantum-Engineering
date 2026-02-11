# Week 154: Variational and WKB Methods - Problem Set

## Instructions

This problem set contains 27 problems at PhD qualifying exam level. Problems are organized by topic and difficulty:
- **Level A (1-9):** Fundamental concepts and direct applications
- **Level B (10-18):** Intermediate problems requiring synthesis
- **Level C (19-27):** Challenging problems at qualifying exam level

---

## Part I: Variational Method

### Problem 1: Variational Principle Proof

(a) Prove the variational principle: for any normalized trial function $|\psi\rangle$, show that $\langle\psi|H|\psi\rangle \geq E_0$.

(b) Under what condition does equality hold?

(c) Extend the proof to show that if $\langle\psi|\psi_0\rangle = 0$, then $\langle\psi|H|\psi\rangle \geq E_1$.

---

### Problem 2: Hydrogen Ground State

Use the trial wavefunction $\psi(r) = Ae^{-\alpha r}$ for the hydrogen atom.

(a) Determine the normalization constant $A$ in terms of $\alpha$.

(b) Calculate $\langle T\rangle$ and $\langle V\rangle$ as functions of $\alpha$.

(c) Find the optimal $\alpha$ and the corresponding energy estimate.

(d) Compare to the exact ground state energy.

---

### Problem 3: Gaussian Trial for Hydrogen

Use a Gaussian trial function $\psi(r) = Ae^{-\alpha r^2}$ for hydrogen.

(a) Calculate the energy $E(\alpha)$.

(b) Find the optimal $\alpha$ and the best energy estimate.

(c) Why is the Gaussian result worse than the exponential? What physical feature is missing?

---

### Problem 4: 1D Harmonic Oscillator

For the harmonic oscillator $H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2$:

(a) Use the trial function $\psi(x) = A(a^2 - x^2)$ for $|x| < a$, zero otherwise. Find $E(a)$.

(b) Minimize to find the best estimate for $E_0$.

(c) Compare to the exact result $\hbar\omega/2$.

---

### Problem 5: Linear Potential

A particle is confined to $x > 0$ with potential $V(x) = \beta x$ (linear potential, $\beta > 0$).

(a) Use the trial function $\psi(x) = Axe^{-\alpha x}$. Calculate $E(\alpha)$.

(b) Find the optimal $\alpha$ and the ground state energy estimate.

(c) Express your answer in terms of the natural energy scale $(\hbar^2\beta^2/2m)^{1/3}$.

---

### Problem 6: Delta Function Potential

For $V(x) = -\alpha\delta(x)$ in 1D:

(a) Calculate the exact bound state energy.

(b) Use the trial function $\psi(x) = Ae^{-\beta|x|}$ and find $E(\beta)$.

(c) Show that the variational result gives the exact answer. Why?

---

### Problem 7: Helium Variational Calculation

For helium with trial function $\psi = \frac{Z_{\text{eff}}^3}{\pi a_0^3}e^{-Z_{\text{eff}}(r_1+r_2)/a_0}$:

(a) Show that $\langle T\rangle = Z_{\text{eff}}^2 \cdot 2 \times 13.6$ eV.

(b) Show that $\langle V_{en}\rangle = -4 Z_{\text{eff}} \times 13.6$ eV.

(c) Using $\langle V_{ee}\rangle = \frac{5}{8}Z_{\text{eff}} \times 27.2$ eV, find the optimal $Z_{\text{eff}}$.

(d) Calculate the ground state energy and compare to experiment ($-78.98$ eV).

---

### Problem 8: Hydrogen with Nuclear Motion

For the hydrogen atom, account for nuclear motion by using reduced mass.

(a) What is the correction to the ground state energy?

(b) The muonic hydrogen atom has an electron replaced by a muon ($m_\mu = 207m_e$). What is its ground state energy?

---

### Problem 9: Two-Parameter Trial Function

For a particle in $V(x) = \frac{1}{2}m\omega^2 x^2 + \lambda x^4$:

(a) Use $\psi(x) = A\exp(-\alpha x^2 - \beta x^4)$ with $\beta = 0$ first. Find $E(\alpha)$.

(b) Include the $\beta$ parameter. Write the equations for $\partial E/\partial \alpha = 0$ and $\partial E/\partial\beta = 0$.

(c) For small $\lambda$, what is the first-order correction to the ground state energy?

---

## Part II: WKB Approximation

### Problem 10: WKB Validity

For the potential $V(x) = V_0(1 - e^{-x^2/a^2})$:

(a) Identify the classical turning points for energy $E < V_0$.

(b) Calculate the WKB validity criterion $|\hbar p'/p^2|$ near the turning point.

(c) Where does WKB break down?

---

### Problem 11: Harmonic Oscillator WKB

Apply the Bohr-Sommerfeld quantization condition to the harmonic oscillator.

(a) Find the turning points for energy $E$.

(b) Evaluate $\oint p\,dx$.

(c) Show that $E_n = (n + 1/2)\hbar\omega$.

---

### Problem 12: Infinite Square Well WKB

For a particle in a box of width $L$:

(a) What are the classical turning points?

(b) Apply WKB quantization. What result do you get?

(c) Compare to the exact result. Why is there a discrepancy?

---

### Problem 13: Linear Potential WKB

A particle moves in $V(x) = \beta x$ for $x > 0$ with an infinite wall at $x = 0$.

(a) Find the turning point $x_0$ for energy $E$.

(b) Apply WKB quantization to find $E_n$.

(c) Express your answer in terms of the Airy function zeros if appropriate.

---

### Problem 14: Morse Potential

The Morse potential is $V(x) = D(1 - e^{-ax})^2$.

(a) Find the classical turning points for $E < D$.

(b) Set up the WKB integral $\int p\,dx$.

(c) Show that the energy levels are: $E_n = \hbar\omega_0(n + 1/2) - \frac{[\hbar\omega_0(n+1/2)]^2}{4D}$ where $\omega_0 = a\sqrt{2D/m}$.

---

### Problem 15: Square Barrier Tunneling

A particle with energy $E$ encounters a rectangular barrier: $V = V_0$ for $0 < x < a$, zero elsewhere.

(a) Use WKB to find the transmission coefficient.

(b) Compare to the exact result for $E = V_0/2$ and $\kappa a = 2$.

---

### Problem 16: Triangular Barrier

A particle tunnels through a triangular barrier: $V(x) = V_0(1 - x/a)$ for $0 < x < a$.

(a) Find the classical turning point for energy $E < V_0$.

(b) Calculate the WKB transmission coefficient.

(c) How does $T$ depend on $E$?

---

### Problem 17: Parabolic Barrier

For an inverted harmonic oscillator barrier $V(x) = V_0 - \frac{1}{2}m\omega^2 x^2$:

(a) Find the turning points for $E < V_0$.

(b) Calculate the WKB transmission coefficient.

(c) Show that at $E = V_0$, the exact result is $T = 1/2$.

---

### Problem 18: Cold Emission

Electrons in a metal at the Fermi energy $E_F$ can tunnel out when an electric field $\mathcal{E}$ is applied. The barrier is approximately triangular.

(a) Sketch the potential and identify the turning points.

(b) Calculate the WKB transmission coefficient.

(c) Show that the emission current follows: $j \propto \mathcal{E}^2 \exp(-c/\mathcal{E})$, where $c$ is a constant.

---

## Part III: Applications and Combined Problems

### Problem 19: Alpha Decay (Caltech 2018)

An alpha particle with energy $E = 5$ MeV is inside a nucleus of radius $R = 7$ fm. Outside the nucleus, the Coulomb barrier is $V(r) = 2Ze^2/(4\pi\epsilon_0 r)$ with $Z = 90$.

(a) Find the classical turning point $r_0$.

(b) Calculate the Gamow factor $\gamma = \int_R^{r_0}\kappa\,dr / \hbar$.

(c) If the alpha particle bounces against the barrier $10^{21}$ times per second, estimate the decay rate and half-life.

---

### Problem 20: Variational WKB Comparison (MIT 2019)

For a particle in the potential $V(x) = \lambda|x|$:

(a) Use the variational method with $\psi(x) = Ae^{-\alpha|x|}$ to estimate $E_0$.

(b) Use WKB to find the quantized energy levels.

(c) Compare the two results for the ground state.

---

### Problem 21: Molecular Hydrogen Ion (Princeton 2020)

The $H_2^+$ molecular ion has one electron shared between two protons separated by distance $R$.

(a) Within the Born-Oppenheimer approximation, describe how you would find the electronic energy $E(R)$.

(b) Using the trial function $\psi = A[\phi_{1s}(\mathbf{r} - \mathbf{R}_A) + \phi_{1s}(\mathbf{r} - \mathbf{R}_B)]$, show that:
$$E(R) = E_{1s} + \frac{J(R) + K(R)}{1 + S(R)}$$
where $J$, $K$, and $S$ are Coulomb, exchange, and overlap integrals.

(c) Explain qualitatively why $H_2^+$ is stable (bonding orbital).

---

### Problem 22: Double Well Splitting (Berkeley 2018)

A particle moves in a symmetric double well with minima at $x = \pm a$.

(a) In the WKB approximation, show that the energy splitting between the two lowest states is:
$$\Delta E \approx \frac{\hbar\omega}{\pi}e^{-\gamma}$$
where $\omega$ is the classical oscillation frequency in one well and $\gamma$ is the Gamow factor.

(b) Interpret this result in terms of tunneling time.

(c) For ammonia (NH$_3$), the inversion frequency is about 24 GHz. Estimate the barrier height.

---

### Problem 23: 3D Variational Problem (Yale 2017)

For a particle in a 3D isotropic harmonic oscillator with a Gaussian perturbation:
$$H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 r^2 - V_0 e^{-r^2/a^2}$$

(a) Use the trial function $\psi(r) = A e^{-\alpha r^2}$. Calculate $E(\alpha)$.

(b) Find the optimal $\alpha$ in the limits $a \gg \sqrt{\hbar/(m\omega)}$ and $a \ll \sqrt{\hbar/(m\omega)}$.

(c) When is a bound state possible?

---

### Problem 24: Radial WKB (Harvard 2019)

For a central potential, the radial equation is:
$$-\frac{\hbar^2}{2m}\frac{d^2 u}{dr^2} + V_{\text{eff}}(r)u = Eu$$

where $V_{\text{eff}} = V(r) + \frac{\hbar^2\ell(\ell+1)}{2mr^2}$.

(a) State the WKB quantization condition for radial problems.

(b) Apply this to the hydrogen atom and derive the energy levels.

(c) What is the Langer correction and why is it needed?

---

### Problem 25: Stark Effect WKB (Chicago 2020)

A hydrogen atom is placed in a uniform electric field $\mathcal{E}$. The electron can tunnel out.

(a) Sketch the effective potential along the field direction.

(b) For the ground state, estimate the tunneling rate using WKB.

(c) At what field strength does the lifetime become comparable to the radiative lifetime ($\sim 10^{-8}$ s)?

---

### Problem 26: Above-Barrier Reflection (Cornell 2019)

A particle with energy $E > V_{\max}$ approaches a potential barrier.

(a) Show that WKB predicts zero reflection (complete transmission).

(b) For a smooth barrier $V(x) = V_0 \text{sech}^2(x/a)$, find the exact reflection coefficient.

(c) Explain the discrepancy. When is above-barrier reflection significant?

---

### Problem 27: Variational Excited State (Stanford 2021)

For the hydrogen atom, estimate the energy of the first excited state (2s).

(a) Use the trial function $\psi(r) = A(1 - \beta r)e^{-\alpha r}$ with the constraint that $\psi$ is orthogonal to the ground state.

(b) Determine the relationship between $\alpha$ and $\beta$ from orthogonality.

(c) Find the optimal parameters and compare to the exact result $E_{2s} = -3.4$ eV.

---

## Bonus Problems

### Problem 28: Instanton Tunneling

In the path integral formulation, tunneling is described by instantons.

(a) For a double-well potential, explain what an instanton is.

(b) How does the instanton action relate to the WKB Gamow factor?

(c) Why is this approach useful for quantum field theory?

---

### Problem 29: Density Functional Theory

The Hohenberg-Kohn theorem states that the ground state energy is a functional of the electron density.

(a) State the variational principle in DFT.

(b) How is this related to the standard variational principle?

(c) What is the Kohn-Sham scheme?

---

## Problem Checklist

| Problem | Topic | Status | Time | Difficulty |
|---------|-------|--------|------|------------|
| 1 | Variational proof | [ ] | ___ min | ___/5 |
| 2 | H exponential | [ ] | ___ min | ___/5 |
| 3 | H Gaussian | [ ] | ___ min | ___/5 |
| 4 | HO polynomial | [ ] | ___ min | ___/5 |
| 5 | Linear potential | [ ] | ___ min | ___/5 |
| 6 | Delta function | [ ] | ___ min | ___/5 |
| 7 | Helium | [ ] | ___ min | ___/5 |
| 8 | Reduced mass | [ ] | ___ min | ___/5 |
| 9 | Two-parameter | [ ] | ___ min | ___/5 |
| 10 | WKB validity | [ ] | ___ min | ___/5 |
| 11 | HO WKB | [ ] | ___ min | ___/5 |
| 12 | Square well WKB | [ ] | ___ min | ___/5 |
| 13 | Linear WKB | [ ] | ___ min | ___/5 |
| 14 | Morse potential | [ ] | ___ min | ___/5 |
| 15 | Square barrier | [ ] | ___ min | ___/5 |
| 16 | Triangular barrier | [ ] | ___ min | ___/5 |
| 17 | Parabolic barrier | [ ] | ___ min | ___/5 |
| 18 | Cold emission | [ ] | ___ min | ___/5 |
| 19 | Alpha decay | [ ] | ___ min | ___/5 |
| 20 | Var vs WKB | [ ] | ___ min | ___/5 |
| 21 | H2+ | [ ] | ___ min | ___/5 |
| 22 | Double well | [ ] | ___ min | ___/5 |
| 23 | 3D variational | [ ] | ___ min | ___/5 |
| 24 | Radial WKB | [ ] | ___ min | ___/5 |
| 25 | Stark effect | [ ] | ___ min | ___/5 |
| 26 | Above-barrier | [ ] | ___ min | ___/5 |
| 27 | Excited state | [ ] | ___ min | ___/5 |

---

**Target:** Complete at least 18 problems before moving to Week 155.
