# Week 146: Measurement and Dynamics — Problem Set

## Instructions

This problem set contains 26 qualifying-exam-level problems on measurement theory and quantum dynamics. Problems are organized by difficulty:

- **Level 1 (Problems 1-9):** Direct applications — 10-15 minutes each
- **Level 2 (Problems 10-19):** Intermediate — 15-25 minutes each
- **Level 3 (Problems 20-26):** Challenging — 25-40 minutes each

---

## Level 1: Direct Application

### Problem 1: Measurement Probabilities

A spin-1/2 particle is in the state:
$$|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle_z + \sqrt{\frac{2}{3}}|-\rangle_z$$

(a) What are the probabilities of measuring $$S_z = +\hbar/2$$ and $$S_z = -\hbar/2$$?

(b) What is $$\langle S_z \rangle$$?

(c) After measuring $$S_z = +\hbar/2$$, what is the new state?

---

### Problem 2: Sequential Measurements

A particle is in the state $$|\psi\rangle = \frac{1}{\sqrt{2}}(|+\rangle_z + |-\rangle_z)$$.

(a) What is the probability of measuring $$S_x = +\hbar/2$$?

(b) After measuring $$S_x = +\hbar/2$$, what is the probability of subsequently measuring $$S_z = +\hbar/2$$?

(c) What is the probability of obtaining $$S_x = +\hbar/2$$ followed by $$S_z = +\hbar/2$$?

---

### Problem 3: Stationary States

Consider a particle in an infinite square well with $$V(x) = 0$$ for $$0 < x < L$$.

(a) Write the general solution to the time-dependent Schrödinger equation as a superposition of stationary states.

(b) If $$|\psi(0)\rangle = |n\rangle$$, what is $$|\psi(t)\rangle$$?

(c) Show that $$|\psi(x,t)|^2$$ is time-independent for a stationary state.

---

### Problem 4: Time Evolution of Superposition

A particle in an infinite well is initially in state:
$$|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |2\rangle)$$

(a) What is $$|\psi(t)\rangle$$?

(b) Calculate $$|\psi(x,t)|^2$$.

(c) What is the "quantum beat" frequency $$\omega_{21} = (E_2 - E_1)/\hbar$$?

---

### Problem 5: Expectation Value Evolution

For a Hamiltonian $$\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$$, use Ehrenfest's theorem.

(a) Show that $$\frac{d\langle\hat{x}\rangle}{dt} = \frac{\langle\hat{p}\rangle}{m}$$.

(b) Show that $$\frac{d\langle\hat{p}\rangle}{dt} = -\langle V'(\hat{x})\rangle$$.

(c) For a harmonic oscillator, show $$\frac{d^2\langle\hat{x}\rangle}{dt^2} = -\omega^2\langle\hat{x}\rangle$$.

---

### Problem 6: Free Particle Propagator

Use the free particle propagator:
$$K_0(x,t;x',0) = \sqrt{\frac{m}{2\pi i\hbar t}}\exp\left[\frac{im(x-x')^2}{2\hbar t}\right]$$

(a) Verify that $$K_0(x,0;x',0) = \delta(x-x')$$ (in the distributional sense).

(b) For initial state $$\psi(x,0) = \delta(x)$$, find $$\psi(x,t)$$.

(c) Show that $$|\psi(x,t)|^2$$ spreads as $$1/t$$ for large $$t$$.

---

### Problem 7: Two-Level System Dynamics

A two-level system has Hamiltonian:
$$\hat{H} = \frac{\hbar\omega_0}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \frac{\hbar\omega_0}{2}\sigma_z$$

(a) Find the time evolution operator $$\hat{U}(t)$$.

(b) If $$|\psi(0)\rangle = |+\rangle_x = \frac{1}{\sqrt{2}}(|+\rangle_z + |-\rangle_z)$$, find $$|\psi(t)\rangle$$.

(c) Calculate $$\langle S_x(t)\rangle$$ and interpret the result.

---

### Problem 8: Heisenberg Picture

For a one-dimensional harmonic oscillator $$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$:

(a) Write the Heisenberg equations of motion for $$\hat{x}_H(t)$$ and $$\hat{p}_H(t)$$.

(b) Solve these coupled equations.

(c) Verify that $$[\hat{x}_H(t), \hat{p}_H(t)] = i\hbar$$ at all times.

---

### Problem 9: Conservation Laws

For a free particle $$\hat{H} = \frac{\hat{p}^2}{2m}$$:

(a) Show that $$[\hat{H}, \hat{p}] = 0$$ and interpret physically.

(b) Show that $$[\hat{H}, \hat{x}] \neq 0$$. Why isn't position conserved?

(c) Find a quantity involving $$\hat{x}$$ that IS conserved.

---

## Level 2: Intermediate

### Problem 10: Measurement of Non-Eigenstate

A particle in a harmonic oscillator is in state $$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$.

(a) Calculate the probability of measuring energy $$E = \frac{3}{2}\hbar\omega$$.

(b) Calculate $$\langle\hat{H}\rangle$$ and $$\Delta H$$.

(c) After measuring energy $$\frac{1}{2}\hbar\omega$$, what is $$\langle\hat{x}\rangle$$ of the resulting state?

---

### Problem 11: Time Evolution with Mixed Initial State

A spin-1/2 system is described by density matrix:
$$\hat{\rho}(0) = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

in the $$\hat{S}_z$$ basis. The Hamiltonian is $$\hat{H} = \omega\hat{S}_z$$.

(a) Find $$\hat{\rho}(t)$$.

(b) Calculate $$\langle\hat{S}_x(t)\rangle = \text{Tr}(\hat{\rho}(t)\hat{S}_x)$$.

(c) Is this a pure or mixed state? How can you tell?

---

### Problem 12: Propagator Construction

Construct the propagator for a particle in an infinite square well.

(a) Using the energy eigenstates $$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\frac{n\pi x}{L}$$, write $$K(x,t;x',0)$$.

(b) Verify that $$K(x,0;x',0) = \delta(x-x')$$ using the completeness relation.

(c) At what times does the wavefunction exactly reproduce itself (revival time)?

---

### Problem 13: Ehrenfest for Anharmonic Oscillator

Consider $$V(x) = \frac{1}{2}m\omega^2x^2 + \lambda x^4$$.

(a) Apply Ehrenfest's theorem to find $$\frac{d\langle\hat{p}\rangle}{dt}$$.

(b) Why can't we simply replace $$\langle x^3 \rangle$$ with $$\langle x \rangle^3$$?

(c) Under what conditions does classical behavior emerge?

---

### Problem 14: Comparing Pictures

For a harmonic oscillator, starting in state $$|\psi(0)\rangle = |n\rangle$$:

(a) In Schrödinger picture, find $$|\psi(t)\rangle_S$$ and $$\langle\hat{x}\rangle_S(t)$$.

(b) In Heisenberg picture, find $$\hat{x}_H(t)$$ and $$\langle\hat{x}\rangle_H(t)$$.

(c) Verify both methods give the same expectation value.

---

### Problem 15: Probability Current

The probability current is:
$$j(x,t) = \frac{\hbar}{2mi}\left(\psi^*\frac{\partial\psi}{\partial x} - \psi\frac{\partial\psi^*}{\partial x}\right)$$

(a) Show that $$\frac{\partial\rho}{\partial t} + \frac{\partial j}{\partial x} = 0$$ (continuity equation).

(b) Calculate $$j(x,t)$$ for a plane wave $$\psi = Ae^{i(kx-\omega t)}$$.

(c) Calculate $$j(x,t)$$ for a stationary state in a well. Interpret the result.

---

### Problem 16: Time-Energy Uncertainty

For a system with Hamiltonian $$\hat{H}$$:

(a) Using Ehrenfest's theorem, show that for any observable $$\hat{A}$$:
$$\Delta E \cdot \frac{\Delta A}{|d\langle\hat{A}\rangle/dt|} \geq \frac{\hbar}{2}$$

(b) Define the "characteristic time" $$\tau_A = \frac{\Delta A}{|d\langle\hat{A}\rangle/dt|}$$.

(c) Explain why this leads to $$\Delta E \cdot \Delta t \geq \hbar/2$$.

---

### Problem 17: Driven Two-Level System

A spin-1/2 in a magnetic field $$\mathbf{B} = B_0\hat{z} + B_1\cos(\omega t)\hat{x}$$ has:
$$\hat{H} = -\gamma(B_0\hat{S}_z + B_1\cos(\omega t)\hat{S}_x)$$

(a) What is $$\hat{H}_0$$ and $$\hat{V}(t)$$ for the interaction picture?

(b) Transform $$\hat{V}$$ to the interaction picture.

(c) Near resonance ($$\omega \approx \gamma B_0$$), what simplification occurs?

---

### Problem 18: Gaussian Wave Packet Evolution

A free particle starts with Gaussian wave packet:
$$\psi(x,0) = \left(\frac{1}{2\pi\sigma_0^2}\right)^{1/4}\exp\left(-\frac{x^2}{4\sigma_0^2}\right)$$

(a) Calculate $$\psi(x,t)$$ using the free propagator.

(b) Find $$\sigma(t)$$, the width of the packet at time $$t$$.

(c) At what time has the packet doubled in width?

---

### Problem 19: Measurement Statistics

A particle is prepared in state $$|\psi\rangle = c_1|1\rangle + c_2|2\rangle + c_3|3\rangle$$ where $$|n\rangle$$ are energy eigenstates.

(a) What is the probability of measuring $$E_2$$?

(b) If energy is measured many times (with fresh preparation each time), what is the average result?

(c) What is the variance in energy measurements?

---

## Level 3: Challenging

### Problem 20: Sudden Approximation (MIT)

A particle is in the ground state of an infinite well of width $$L$$. At $$t=0$$, the well instantly expands to width $$2L$$.

(a) What is the probability that the particle is found in the ground state of the new well?

(b) What is the probability of finding it in the first excited state?

(c) Calculate $$\langle E \rangle$$ after the expansion and compare to the initial energy.

---

### Problem 21: Adiabatic Theorem (Berkeley)

A spin-1/2 in magnetic field $$\mathbf{B}(t) = B_0(\sin\theta\cos\phi(t), \sin\theta\sin\phi(t), \cos\theta)$$ where $$\phi(t) = \omega t$$.

(a) Find the instantaneous energy eigenstates.

(b) If the system starts in the ground state at $$t=0$$ and $$\omega$$ is very small, what is the state at time $$t$$?

(c) Calculate the Berry phase acquired after one complete rotation ($$\phi: 0 \to 2\pi$$).

---

### Problem 22: Quantum Zeno Effect (Caltech)

A two-level system starts in state $$|1\rangle$$. The Hamiltonian couples it to $$|2\rangle$$:
$$\hat{H} = \hbar\Omega(|1\rangle\langle 2| + |2\rangle\langle 1|)$$

(a) Without measurements, what is the probability of finding the system in $$|1\rangle$$ at time $$t$$?

(b) If we measure "is the system in $$|1\rangle$$?" at times $$t/N, 2t/N, ..., t$$, what is the survival probability as $$N \to \infty$$?

(c) Explain the "quantum Zeno paradox."

---

### Problem 23: Path Integral Preview (Princeton)

The propagator can be written as a path integral:
$$K(x_f,t;x_i,0) = \int \mathcal{D}[x(t')] \exp\left[\frac{i}{\hbar}\int_0^t L(x,\dot{x}) dt'\right]$$

(a) For a free particle, show the classical path minimizes the action.

(b) The semiclassical approximation uses paths near the classical one. How does $$\hbar$$ control the width of this neighborhood?

(c) For a harmonic oscillator, verify the exact propagator matches the semiclassical result (the path integral is Gaussian-exact).

---

### Problem 24: Decay Rate from Fermi's Golden Rule (Yale)

A system starts in state $$|i\rangle$$ with energy $$E_i$$. A perturbation $$\hat{V}$$ couples it to a continuum of states $$|f\rangle$$ with density $$\rho(E)$$.

(a) From time-dependent perturbation theory, derive the transition probability $$P_{i\to f}(t)$$.

(b) Derive Fermi's Golden Rule:
$$\Gamma = \frac{2\pi}{\hbar}|\langle f|\hat{V}|i\rangle|^2\rho(E_f)$$

(c) How does the survival probability $$P_{i\to i}(t) = e^{-\Gamma t}$$ emerge?

---

### Problem 25: Quantum-Classical Correspondence

For $$\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$$, consider a minimum uncertainty wave packet centered at $$(x_0, p_0)$$.

(a) Show that to first order in $$\hbar$$, the center follows classical mechanics.

(b) Calculate the first quantum correction to the motion.

(c) For a harmonic oscillator, show the packet follows the classical trajectory exactly. Why?

---

### Problem 26: Decoherence

A qubit interacts with an environment, with total Hamiltonian:
$$\hat{H} = \frac{\hbar\omega}{2}\hat{\sigma}_z \otimes \hat{1}_E + \hat{\sigma}_z \otimes \hat{B}$$

where $$\hat{B}$$ is an environment operator.

(a) Show that the reduced density matrix of the qubit (tracing over environment) becomes diagonal in the $$\sigma_z$$ basis.

(b) This is "dephasing." What happens to off-diagonal coherences?

(c) What is the physical significance for quantum computing?

---

## Exam Strategy Notes

### Key Techniques
1. **Measurement problems:** Use Born rule carefully; don't forget normalization
2. **Time evolution:** Expand in energy eigenstates for time-independent $$\hat{H}$$
3. **Propagator problems:** Use completeness or direct integration
4. **Ehrenfest:** Compute commutators first, then take expectation values
5. **Picture conversion:** $$\hat{A}_H = \hat{U}^\dagger\hat{A}_S\hat{U}$$

### Common Errors
- Sign errors in $$e^{\pm i\hat{H}t/\hbar}$$
- Forgetting phase factors $$e^{-iE_nt/\hbar}$$
- Not normalizing after collapse
- Confusing $$|\langle a|\psi\rangle|^2$$ with $$\langle a|\psi\rangle$$

---

*Problem Set for Week 146 — Measurement and Dynamics*
*Solutions available in Problem_Solutions.md*
