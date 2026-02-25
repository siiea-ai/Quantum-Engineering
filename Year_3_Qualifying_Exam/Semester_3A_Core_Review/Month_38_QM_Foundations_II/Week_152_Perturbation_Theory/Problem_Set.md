# Week 152: Perturbation Theory - Problem Set

## Instructions

- **Total Problems:** 30
- **Recommended Time:** 5-6 hours (spread across the week)
- **Difficulty Levels:** Direct Application (D), Intermediate (I), Challenging (C)
- **Exam Conditions:** For problems marked with *, attempt under timed conditions (15-20 min each)

---

## Part A: Non-Degenerate Perturbation Theory (Problems 1-8)

### Problem 1 (D)
A harmonic oscillator $H_0 = \hbar\omega(a^{\dagger}a + 1/2)$ is perturbed by $H' = \lambda x^2$.
(a) Calculate the first-order energy correction for the ground state.
(b) Calculate the first-order energy correction for the n-th excited state.
(c) What is the exact result? Compare.

### Problem 2 (D)
For the infinite square well with walls at $x = 0$ and $x = a$, apply the perturbation $H' = V_0$ (constant) in the region $0 < x < a/2$.
(a) Calculate $E_n^{(1)}$ for all $n$.
(b) Why doesn't this perturbation split any degeneracies?

### Problem 3 (I)*
A harmonic oscillator is perturbed by $H' = \lambda x^4$.
(a) Calculate $E_n^{(1)}$ using ladder operators.
(b) Is $E_n^{(2)}$ positive or negative for the ground state? Explain without calculation.

### Problem 4 (I)
For a particle in a 1D box with $H' = V_0\sin(\pi x/a)$:
(a) Calculate $E_1^{(1)}$ and $E_2^{(1)}$.
(b) Calculate $E_1^{(2)}$ keeping only the contribution from the $n=2$ state.
(c) Why is the approximation in (b) valid if $V_0 \ll E_2 - E_1$?

### Problem 5 (I)*
A hydrogen atom is placed in a weak electric field $\mathbf{E} = E_0\hat{z}$. For the ground state:
(a) Explain why $E^{(1)} = 0$.
(b) Calculate $E^{(2)}$ by estimating which intermediate states contribute most.
(c) The result is called the quadratic Stark effect. Why is it quadratic?

### Problem 6 (C)
An electron in a hydrogen atom experiences a perturbation due to the finite nuclear size:
$$H' = \frac{Ze^2}{4\pi\epsilon_0}\left(\frac{1}{R} - \frac{1}{r}\right) \quad \text{for } r < R$$
$$H' = 0 \quad \text{for } r > R$$

where $R$ is the nuclear radius ($R \ll a_0$).
(a) Calculate the first-order energy shift for the ground state.
(b) For which states is this correction largest?

### Problem 7 (C)*
Consider a 1D potential with ground state $|0\rangle$ and first excited state $|1\rangle$, with energies $E_0$ and $E_1 = E_0 + \Delta$.

A perturbation $H' = \epsilon(|0\rangle\langle 1| + |1\rangle\langle 0|)$ is applied.
(a) Using second-order perturbation theory, find the energy shifts.
(b) Solve the problem exactly and expand to second order. Compare.
(c) For what range of $\epsilon$ is perturbation theory valid?

### Problem 8 (C)
Prove that for the ground state, the second-order energy correction is always negative:
$$E_0^{(2)} = \sum_{k\neq 0}\frac{|\langle k|H'|0\rangle|^2}{E_0 - E_k} < 0$$

What physical interpretation can you give this result?

---

## Part B: Degenerate Perturbation Theory (Problems 9-14)

### Problem 9 (D)
Two states $|1\rangle$ and $|2\rangle$ are degenerate with energy $E_0$. The perturbation matrix is:
$$W = \begin{pmatrix} a & b \\ b^* & c \end{pmatrix}$$

Find the first-order energy corrections and the "good" linear combinations.

### Problem 10 (I)*
The first excited state of a 2D isotropic harmonic oscillator is doubly degenerate: $|1,0\rangle$ and $|0,1\rangle$ (quantum numbers in x and y).

Apply the perturbation $H' = \lambda xy$.
(a) Calculate the matrix $W_{ij} = \langle i|H'|j\rangle$.
(b) Find the first-order energy shifts.
(c) What are the "good" states?

### Problem 11 (I)
For the hydrogen atom $n=2$ level with perturbation $H' = eE_0 z$ (linear Stark effect):
(a) Show that the only non-zero matrix element is between $|2,0,0\rangle$ and $|2,1,0\rangle$.
(b) Calculate this matrix element.
(c) Find the energy shifts and eigenstates.

### Problem 12 (C)*
A 3D isotropic harmonic oscillator has first excited level with degeneracy 3: $|1,0,0\rangle$, $|0,1,0\rangle$, $|0,0,1\rangle$.

Apply $H' = \lambda(x^2 - y^2)$.
(a) Construct the $3 \times 3$ perturbation matrix.
(b) Find all first-order energy shifts.
(c) Identify the "good" quantum numbers.

### Problem 13 (C)
For the hydrogen atom $n=3$ level (9-fold degenerate, ignoring spin):
(a) Which states are coupled by the perturbation $H' = eE_0 z$?
(b) Without calculating explicitly, how many distinct energy levels result?
(c) Which levels show linear Stark effect?

### Problem 14 (C)*
Two identical spin-1/2 particles are in a 1D harmonic oscillator. The spatial ground state is non-degenerate, but the spin state has degeneracy 4.

A spin-spin interaction $H' = J\mathbf{S}_1 \cdot \mathbf{S}_2$ is turned on.
(a) Find the energy shifts for all four spin states.
(b) What are the degeneracies of the resulting levels?

---

## Part C: Time-Dependent Perturbation Theory (Problems 15-20)

### Problem 15 (D)
A harmonic oscillator initially in the ground state is subject to a perturbation $H'(t) = F_0 x$ that is turned on suddenly at $t=0$ and held constant.
(a) Calculate $c_1(t)$ to first order.
(b) Find the transition probability $P_{0\to 1}(t)$.

### Problem 16 (D)
A two-level system with energy difference $\hbar\omega_0$ is subject to $H'(t) = V_0\cos(\omega t)$.
(a) Write the transition amplitude $c_f^{(1)}(t)$.
(b) Near resonance ($\omega \approx \omega_0$), simplify the expression.
(c) At exact resonance, how does $P_{i\to f}$ grow with time?

### Problem 17 (I)*
A hydrogen atom in its ground state is exposed to a time-dependent electric field $\mathbf{E}(t) = E_0 e^{-t/\tau}\hat{z}$ for $t > 0$.
(a) Calculate the probability of transition to the $|2,1,0\rangle$ state as $t \to \infty$.
(b) For what value of $\tau$ is this probability maximized?

### Problem 18 (I)
A perturbation $H'(t) = V_0\delta(t - t_0)$ (a delta-function "kick") is applied.
(a) Calculate the transition amplitude $c_f$ after the kick.
(b) This is called the sudden approximation. Explain why.

### Problem 19 (C)*
For a harmonic oscillator initially in state $|n\rangle$, a perturbation $H'(t) = F_0 x \cos(\omega t)$ is applied.
(a) To which states can transitions occur to first order?
(b) Calculate the transition probability to these states.
(c) At what frequency is the transition probability maximized?

### Problem 20 (C)
A spin-1/2 particle is initially in state $|\uparrow_z\rangle$ in a static field $B_0\hat{z}$. A weak oscillating field $B_1\cos(\omega t)\hat{x}$ is added.
(a) Using time-dependent perturbation theory, find the probability of transition to $|\downarrow_z\rangle$.
(b) Compare with the exact Rabi solution. When is perturbation theory valid?

---

## Part D: Fermi's Golden Rule (Problems 21-25)

### Problem 21 (D)
State Fermi's golden rule and explain when it is applicable. What assumptions are made in its derivation?

### Problem 22 (I)*
An atom in an excited state $|e\rangle$ decays to ground state $|g\rangle$ by emitting a photon. The decay rate is given by:
$$\Gamma = \frac{\omega^3}{3\pi\epsilon_0\hbar c^3}|\langle g|\mathbf{d}|e\rangle|^2$$

where $\mathbf{d} = e\mathbf{r}$ is the dipole moment.
(a) Identify the density of states factor.
(b) For hydrogen 2p $\to$ 1s, estimate the lifetime.

### Problem 23 (I)
A particle is in the ground state of an infinite square well. At $t=0$, one wall suddenly moves, changing the well width from $a$ to $2a$.
(a) What is the probability of finding the particle in the new ground state?
(b) Use Fermi's golden rule concepts to explain the transition.

### Problem 24 (C)*
In photoionization, an electron is ejected from a hydrogen atom by absorbing a photon.
(a) Apply Fermi's golden rule to find the ionization rate.
(b) What is the density of states for the ejected electron?
(c) How does the cross-section depend on photon energy?

### Problem 25 (C)
A nucleus undergoes beta decay: $n \to p + e^- + \bar{\nu}_e$.
(a) The transition rate is proportional to $|M_{fi}|^2\rho(E_e)$. What determines $\rho(E_e)$?
(b) Sketch the electron energy spectrum.
(c) Why is there a maximum electron energy?

---

## Part E: Adiabatic Theorem and Berry Phase (Problems 26-30)

### Problem 26 (D)
A particle is in the ground state of a harmonic oscillator with frequency $\omega_1$. The frequency is slowly changed to $\omega_2$.
(a) According to the adiabatic theorem, what is the final state?
(b) What is the condition for "slowly"?

### Problem 27 (I)*
A spin-1/2 particle is in a magnetic field $\mathbf{B}(t) = B_0(\sin\theta\cos\phi(t), \sin\theta\sin\phi(t), \cos\theta)$ where $\phi(t)$ increases slowly from 0 to $2\pi$.
(a) The field traces a cone. What is the solid angle enclosed?
(b) Calculate the Berry phase acquired by the spin-up state.

### Problem 28 (I)
For a spin-1/2 in a field that rotates from $+\hat{z}$ to $-\hat{z}$ along a great circle:
(a) What is the dynamical phase acquired?
(b) What is the Berry phase?
(c) Is the final state $|\uparrow\rangle$ or $|\downarrow\rangle$?

### Problem 29 (C)*
A particle moves adiabatically around a solenoid containing magnetic flux $\Phi$. The Aharonov-Bohm phase is $\gamma = e\Phi/\hbar$.
(a) Show this is a Berry phase.
(b) Define the Berry connection $A_i = i\langle n|\partial/\partial R_i|n\rangle$ and relate to the vector potential.

### Problem 30 (C)* - Qualifying Exam Style
A quantum system has a Hamiltonian $H(\mathbf{R})$ depending on parameters $\mathbf{R} = (R_1, R_2, R_3)$.

(a) Write the condition for adiabatic evolution in terms of the gap between energy levels.

(b) For a two-level system with $H = \mathbf{R}\cdot\boldsymbol{\sigma}$:
- Find the eigenvalues as functions of $|\mathbf{R}|$.
- Calculate the Berry phase when $\mathbf{R}$ traces a closed loop enclosing the origin.

(c) What happens at $\mathbf{R} = 0$? How does this affect adiabatic evolution?

(d) Apply this to a spin-1/2 in a magnetic field and verify the result $\gamma = -\Omega/2$.

---

## Answer Key (Quick Reference)

| Problem | Key Answer |
|---------|------------|
| 1a | $E_0^{(1)} = \frac{\lambda\hbar}{2m\omega}$ |
| 3a | $E_n^{(1)} = \frac{3\lambda\hbar^2}{4m^2\omega^2}(2n^2 + 2n + 1)$ |
| 5a | Zero by parity |
| 9 | $E^{(1)} = \frac{a+c}{2} \pm \sqrt{(\frac{a-c}{2})^2 + |b|^2}$ |
| 11b | $\langle 2,0,0|z|2,1,0\rangle = -3a_0$ |
| 14a | Triplet: $J\hbar^2/4$; Singlet: $-3J\hbar^2/4$ |
| 22b | $\tau \sim 1.6$ ns |
| 27b | $\gamma = -\pi(1-\cos\theta)$ |
| 28c | $|\downarrow\rangle$ |

---

**Detailed solutions in Problem_Solutions.md**
