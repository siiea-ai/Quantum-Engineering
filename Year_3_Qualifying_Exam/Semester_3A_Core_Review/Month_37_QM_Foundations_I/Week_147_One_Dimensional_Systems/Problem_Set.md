# Week 147: One-Dimensional Systems — Problem Set

## Instructions

25 qualifying-exam-level problems on 1D quantum systems. Problems organized by difficulty:

- **Level 1 (Problems 1-8):** Direct applications — 10-15 minutes each
- **Level 2 (Problems 9-18):** Intermediate — 15-25 minutes each
- **Level 3 (Problems 19-25):** Challenging — 25-40 minutes each

---

## Level 1: Direct Application

### Problem 1: Infinite Well Basics

A particle of mass $$m$$ is in an infinite square well of width $$L$$.

(a) Write down the normalized energy eigenfunctions and eigenvalues.

(b) Calculate $$\langle x \rangle$$ and $$\langle x^2 \rangle$$ for the ground state.

(c) Calculate $$\Delta x$$ for the ground state and compare to $$L$$.

---

### Problem 2: Infinite Well Superposition

A particle in an infinite well is in the state:
$$|\psi\rangle = \frac{1}{\sqrt{5}}(|1\rangle + 2|2\rangle)$$

(a) What is $$\langle E \rangle$$?

(b) What is $$\Delta E$$?

(c) What is the probability of finding the particle in the left half of the well?

---

### Problem 3: Harmonic Oscillator Ground State

For the 1D harmonic oscillator ground state:

(a) Calculate $$\langle \hat{x}^2 \rangle$$ and $$\langle \hat{p}^2 \rangle$$.

(b) Verify $$\Delta x \Delta p = \hbar/2$$.

(c) What fraction of the time does the particle spend in the classically forbidden region?

---

### Problem 4: Ladder Operator Practice

Using ladder operators:

(a) Calculate $$\langle 3|\hat{x}|5\rangle$$.

(b) Calculate $$\langle 2|\hat{x}^2|2\rangle$$.

(c) Calculate $$\langle n|\hat{p}^2|n\rangle$$.

---

### Problem 5: Delta Function Potential

For the attractive delta potential $$V(x) = -\alpha\delta(x)$$:

(a) Find the bound state energy.

(b) Sketch the wavefunction.

(c) Show that there is exactly one bound state.

---

### Problem 6: Harmonic Oscillator Selection Rules

For the harmonic oscillator:

(a) Show that $$\langle n|\hat{x}|m\rangle = 0$$ unless $$|n-m| = 1$$.

(b) Calculate $$\langle n|\hat{x}|n+1\rangle$$.

(c) What does this imply for electric dipole transitions?

---

### Problem 7: Time Evolution in Well

A particle in an infinite well starts in state $$\psi(x,0) = \sqrt{\frac{2}{L}}\sin\frac{2\pi x}{L}$$.

(a) What is $$\psi(x,t)$$?

(b) Is $$|\psi(x,t)|^2$$ time-dependent?

(c) What is $$\langle \hat{H} \rangle$$?

---

### Problem 8: Finite Well Bound States

A finite square well has depth $$V_0$$ and width $$2a$$.

(a) Write the general form of bound state wavefunctions in each region.

(b) What boundary conditions must be satisfied?

(c) For $$V_0 = 2\hbar^2/(ma^2)$$, approximately how many bound states exist?

---

## Level 2: Intermediate

### Problem 9: Infinite Well Perturbation

A particle is in an infinite well. A small perturbation $$V' = \lambda x$$ is added.

(a) Calculate the first-order energy correction for the ground state.

(b) Calculate the first-order correction to the ground state wavefunction.

(c) What symmetry is broken by this perturbation?

---

### Problem 10: Coherent State Properties

For a coherent state $$|\alpha\rangle$$ of the harmonic oscillator:

(a) Show that $$\langle\hat{n}\rangle = |\alpha|^2$$.

(b) Calculate $$\Delta n$$ and show that $$\Delta n/\langle n \rangle \to 0$$ as $$|\alpha| \to \infty$$.

(c) Show that the probability distribution $$P(n) = |\langle n|\alpha\rangle|^2$$ is Poissonian.

---

### Problem 11: Harmonic Oscillator Time Evolution

A harmonic oscillator starts in state $$|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$.

(a) Find $$|\psi(t)\rangle$$.

(b) Calculate $$\langle\hat{x}(t)\rangle$$.

(c) At what times is $$\langle\hat{x}\rangle$$ maximum?

---

### Problem 12: Expansion in Well

The initial wavefunction in an infinite well is $$\psi(x,0) = Ax(L-x)$$.

(a) Find the normalization constant $$A$$.

(b) Find the expansion coefficients $$c_n = \langle n|\psi\rangle$$.

(c) What is the probability of measuring the ground state energy?

---

### Problem 13: Double Delta Potential

Consider $$V(x) = -\alpha[\delta(x-a) + \delta(x+a)]$$.

(a) Show that there are two bound states (symmetric and antisymmetric).

(b) Find the transcendental equations for the bound state energies.

(c) Which state has lower energy and why?

---

### Problem 14: Finite Well: Deep Well Limit

For a finite well with $$V_0 \gg \hbar^2/(ma^2)$$:

(a) Show that bound state energies approach infinite well values.

(b) Estimate the correction to the ground state energy.

(c) How does the wavefunction "leak" into the classically forbidden region?

---

### Problem 15: Harmonic Oscillator Matrix Elements

Calculate the following for the harmonic oscillator:

(a) $$\langle n|(\hat{a}+\hat{a}^\dagger)^3|n\rangle$$

(b) $$\langle n|\hat{x}^4|n\rangle$$

(c) $$\langle 0|e^{\lambda\hat{x}}|0\rangle$$

---

### Problem 16: Wave Packet Dynamics

A Gaussian wave packet has initial width $$\sigma_0$$ and mean momentum $$p_0$$.

(a) Calculate the spreading time $$\tau = 2m\sigma_0^2/\hbar$$.

(b) For an electron with $$\sigma_0 = 1$$ nm, what is $$\tau$$?

(c) After time $$t \gg \tau$$, how does the width scale with time?

---

### Problem 17: Sudden Expansion

A particle is in the ground state of an infinite well of width $$L$$. At $$t=0$$, the well suddenly expands to width $$2L$$.

(a) What is the probability of remaining in the ground state of the new well?

(b) What is $$\langle E \rangle$$ after expansion?

(c) Is energy conserved? Explain.

---

### Problem 18: Scattering from Delta Barrier

For a delta barrier $$V(x) = \alpha\delta(x)$$ with $$\alpha > 0$$:

(a) Calculate the reflection coefficient $$R(E)$$.

(b) What is $$R$$ in the limit $$E \to 0$$?

(c) What is $$R$$ in the limit $$E \to \infty$$?

---

## Level 3: Challenging

### Problem 19: Half-Harmonic Oscillator (MIT)

Consider the half-harmonic oscillator:
$$V(x) = \begin{cases} \frac{1}{2}m\omega^2 x^2 & x > 0 \\ \infty & x \leq 0 \end{cases}$$

(a) What boundary condition must wavefunctions satisfy?

(b) Which harmonic oscillator states survive as eigenstates?

(c) What are the energy levels?

---

### Problem 20: Harmonic Oscillator with Electric Field (Berkeley)

A charged harmonic oscillator in electric field $$\mathcal{E}$$:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 - q\mathcal{E}\hat{x}$$

(a) Complete the square to find the new equilibrium position.

(b) Find the exact energy levels.

(c) If initially in ground state at $$\mathcal{E}=0$$, then $$\mathcal{E}$$ suddenly turns on, what is $$\langle E \rangle$$?

---

### Problem 21: Infinite Well with Delta Perturbation (Caltech)

An infinite well has a delta function perturbation at the center:
$$V'(x) = \lambda\delta(x - L/2)$$

(a) Calculate the first-order energy shift for state $$|n\rangle$$.

(b) Which states are unaffected by this perturbation?

(c) Calculate the second-order correction for the ground state.

---

### Problem 22: Finite Well Numerical Analysis (Yale)

For a finite well with $$z_0 = a\sqrt{2mV_0}/\hbar = 3$$:

(a) How many bound states exist?

(b) Solve the transcendental equation graphically or numerically to find the energies.

(c) Calculate the probability of finding the particle outside the well for the ground state.

---

### Problem 23: Squeezed States (Princeton)

Define the squeeze operator $$\hat{S}(\xi) = \exp[\frac{1}{2}(\xi^*\hat{a}^2 - \xi\hat{a}^{\dagger 2})]$$.

A squeezed vacuum is $$|\xi\rangle = \hat{S}(\xi)|0\rangle$$.

(a) Show that $$\hat{S}^\dagger\hat{a}\hat{S} = \hat{a}\cosh r - \hat{a}^\dagger e^{i\phi}\sinh r$$ where $$\xi = re^{i\phi}$$.

(b) Calculate $$\Delta x$$ and $$\Delta p$$ for the squeezed vacuum.

(c) Show that $$\Delta x \Delta p = \hbar/2$$ (still minimum uncertainty) but the uncertainties can be very different.

---

### Problem 24: Resonances in Finite Well

For a finite well with $$E > V_0$$ (scattering states):

(a) Derive the transmission coefficient $$T(E)$$.

(b) Show that $$T = 1$$ when $$k'a = n\pi/2$$ where $$k' = \sqrt{2mE}/\hbar$$.

(c) These are "transmission resonances." Explain physically.

---

### Problem 25: Anharmonic Oscillator

Consider $$V(x) = \frac{1}{2}m\omega^2x^2 + \lambda x^4$$ with $$\lambda$$ small.

(a) Calculate the first-order energy correction using $$\hat{x}^4 = \frac{\hbar^2}{4m^2\omega^2}(\hat{a}+\hat{a}^\dagger)^4$$.

(b) For the ground state, find $$E_0^{(1)}$$.

(c) Why is second-order perturbation theory more complicated for this problem?

---

## Exam Strategy

### Quick Checks
- Units: Energy should have dimensions $$\hbar^2/(mL^2)$$ or $$\hbar\omega$$
- Nodes: $$n$$th state has $$n-1$$ nodes (infinite well) or $$n$$ nodes (oscillator)
- Limits: Deep well → infinite well; $$\omega \to 0$$ → free particle

### Common Techniques
- Ladder operators faster than Hermite polynomials
- Use symmetry to eliminate zero matrix elements
- Completeness: $$\sum_n |n\rangle\langle n| = 1$$

---

*Problem Set for Week 147 — One-Dimensional Systems*
