# Week 153: Identical Particles - Problem Set

## Instructions

This problem set contains 28 problems at the PhD qualifying exam level. Problems are organized by difficulty:
- **Level A (1-10):** Fundamental concepts and direct applications
- **Level B (11-20):** Intermediate problems requiring synthesis
- **Level C (21-28):** Challenging problems at qualifying exam level

Time estimate: 15-25 minutes per problem for Levels A-B, 30-45 minutes for Level C.

---

## Level A: Fundamental Concepts

### Problem 1: Exchange Operator Properties

(a) Show that the exchange operator $\hat{P}_{12}$ is Hermitian.

(b) Prove that $\hat{P}_{12}^2 = \mathbf{1}$.

(c) What are the possible eigenvalues of $\hat{P}_{12}$? Prove your answer.

---

### Problem 2: Two-Particle Wavefunctions

Two identical particles are in single-particle states $\phi_a(\mathbf{r})$ and $\phi_b(\mathbf{r})$ where $\phi_a \neq \phi_b$.

(a) Write the properly normalized symmetric wavefunction.

(b) Write the properly normalized antisymmetric wavefunction.

(c) What is the probability density for finding both particles at the same position $\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}$ for each case?

---

### Problem 3: Pauli Exclusion

(a) Two fermions are both in the same single-particle state $\phi(\mathbf{r})$. Construct the antisymmetric wavefunction and show it vanishes.

(b) For two electrons in a 1D infinite square well, what is the minimum total energy if both electrons are in the same spin state? Different spin states?

(c) Explain why helium's ground state must be a spin singlet.

---

### Problem 4: Slater Determinant - Two Electrons

Construct the Slater determinant for two electrons in a 1D harmonic oscillator, with one electron in the ground state and one in the first excited state. Assume both have spin up.

---

### Problem 5: Slater Determinant - Three Electrons

Write the Slater determinant for three electrons in the ground state of lithium (1s$^2$2s$^1$). Specify the spin states.

---

### Problem 6: Bosonic Commutation Relations

Starting from the definition $a|n\rangle = \sqrt{n}|n-1\rangle$ and $a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$:

(a) Verify that $[a, a^\dagger] = 1$.

(b) Show that $[a, (a^\dagger)^n] = n(a^\dagger)^{n-1}$.

(c) Calculate $\langle n|a^\dagger a|n\rangle$.

---

### Problem 7: Fermionic Anticommutators

For fermionic operators satisfying $\{c, c^\dagger\} = 1$:

(a) Show that $(c^\dagger)^2 = 0$ and $(c)^2 = 0$.

(b) Show that $c^\dagger c + c c^\dagger = 1$.

(c) What are the eigenvalues of $\hat{n} = c^\dagger c$?

---

### Problem 8: Number Operator

For a system with two single-particle states (labeled 1 and 2):

(a) Write the total number operator in terms of $a_1^\dagger, a_1, a_2^\dagger, a_2$.

(b) Find $[\hat{N}, a_i^\dagger]$ and $[\hat{N}, a_i]$.

(c) What do these commutation relations tell us about how $a_i^\dagger$ and $a_i$ change particle number?

---

### Problem 9: Vacuum State

The vacuum state $|0\rangle$ is defined by $a_i|0\rangle = 0$ for all $i$.

(a) Show that $|1_i\rangle = a_i^\dagger|0\rangle$ is normalized if $\langle 0|0\rangle = 1$.

(b) Construct the normalized two-particle state with one particle in state 1 and one in state 2 for bosons.

(c) Repeat part (b) for fermions and show the antisymmetry.

---

### Problem 10: Helium Ground State Energy

The ground state of helium can be approximated as:
$$\psi_0(\mathbf{r}_1, \mathbf{r}_2) = \frac{Z^3}{\pi a_0^3}e^{-Z(r_1+r_2)/a_0}$$

(a) Why must this be multiplied by the spin singlet function?

(b) Using the result $\langle 1/r_{12}\rangle = 5Z/(8a_0)$ for this wavefunction, calculate the first-order perturbation theory estimate of the ground state energy.

(c) Compare to the experimental value of $-78.98$ eV.

---

## Level B: Intermediate Problems

### Problem 11: Exchange Integral Calculation

For two electrons in a 1D infinite square well of width $L$, with one in $\psi_1(x) = \sqrt{2/L}\sin(\pi x/L)$ and one in $\psi_2(x) = \sqrt{2/L}\sin(2\pi x/L)$:

(a) Calculate the direct integral for a delta-function interaction $V(x_1, x_2) = \lambda\delta(x_1 - x_2)$.

(b) Calculate the exchange integral for the same interaction.

(c) Which total spin state (singlet or triplet) has lower energy?

---

### Problem 12: Second Quantization of Kinetic Energy

The kinetic energy operator in first quantization is $\hat{T} = \sum_{i=1}^N \frac{\mathbf{p}_i^2}{2m}$.

(a) Express $\hat{T}$ in second quantization using creation and annihilation operators.

(b) For plane wave states $\phi_\mathbf{k}(\mathbf{r}) = \frac{1}{\sqrt{V}}e^{i\mathbf{k}\cdot\mathbf{r}}$, simplify your expression.

(c) What is the ground state energy of N non-interacting fermions in a box?

---

### Problem 13: Two-Body Interaction

Consider the two-body interaction $V = \sum_{i<j}v(|\mathbf{r}_i - \mathbf{r}_j|)$.

(a) Write this in second quantization using field operators $\hat{\psi}^\dagger(\mathbf{r}), \hat{\psi}(\mathbf{r})$.

(b) Transform to momentum space for a translationally invariant system.

(c) Identify the "direct" and "exchange" terms in the resulting expression.

---

### Problem 14: Fermi Gas Ground State

For N non-interacting spin-1/2 fermions in a 3D box of volume $V$:

(a) What is the ground state in occupation number representation?

(b) Calculate the Fermi energy $\epsilon_F$ in terms of the density $n = N/V$.

(c) Calculate the total ground state energy per particle.

---

### Problem 15: Bosonic Ground State

For N non-interacting bosons in a 3D harmonic oscillator with frequency $\omega$:

(a) What is the ground state energy?

(b) Write the ground state in second quantization notation.

(c) Compare the ground state energies of N bosons vs N spin-polarized fermions in the same potential.

---

### Problem 16: Spin-Orbit Basis States

For two electrons with $l_1 = l_2 = 1$ (p-orbitals):

(a) How many spatial states are there? How many spin states?

(b) For the singlet state, which spatial states are allowed by antisymmetry?

(c) For the triplet state, which spatial states are allowed?

---

### Problem 17: Helium Excited States

For the first excited states of helium (1s2s configuration):

(a) Construct the singlet spatial wavefunction.

(b) Construct the triplet spatial wavefunctions.

(c) Using the result that the exchange integral $K > 0$, determine which state (orthohelium or parahelium) is lower in energy.

---

### Problem 18: Jordan-Wigner Transformation

The Jordan-Wigner transformation maps spin operators to fermionic operators:
$$c_j = \left(\prod_{i<j}\sigma_i^z\right)\sigma_j^-$$

(a) Show that $\{c_j, c_j^\dagger\} = 1$.

(b) Show that $\{c_i, c_j\} = 0$ for $i \neq j$.

(c) What is the physical significance of the string operator $\prod_{i<j}\sigma_i^z$?

---

### Problem 19: Occupation Number Fluctuations

For a system of bosons in thermal equilibrium at temperature $T$:

(a) Show that the occupation number fluctuations satisfy $\langle(\Delta n)^2\rangle = \langle n\rangle(\langle n\rangle + 1)$ for the Bose-Einstein distribution.

(b) What is the corresponding result for fermions?

(c) Interpret the difference physically.

---

### Problem 20: Commutator Identities

Prove the following for bosonic operators:

(a) $[a^n, a^\dagger] = na^{n-1}$

(b) $[a, f(a^\dagger a)] = f(a^\dagger a + 1)a - af(a^\dagger a)$

(c) $e^{\alpha a^\dagger}a e^{-\alpha a^\dagger} = a - \alpha$ (Baker-Campbell-Hausdorff)

---

## Level C: Qualifying Exam Level

### Problem 21: Variational Treatment of Helium (Yale 2018)

Consider the helium atom with the trial wavefunction:
$$\psi(\mathbf{r}_1, \mathbf{r}_2) = \frac{Z_{\text{eff}}^3}{\pi a_0^3}e^{-Z_{\text{eff}}(r_1+r_2)/a_0}$$
where $Z_{\text{eff}}$ is a variational parameter.

(a) Calculate the expectation value of the kinetic energy in terms of $Z_{\text{eff}}$.

(b) Calculate the expectation value of the electron-nucleus potential energy.

(c) Using $\langle 1/r_{12}\rangle = 5Z_{\text{eff}}/(8a_0)$, find the optimal $Z_{\text{eff}}$ and the corresponding ground state energy.

(d) Compare to the experimental value and discuss the physical meaning of $Z_{\text{eff}} < 2$.

---

### Problem 22: Three-Fermion System (MIT 2019)

Three spin-1/2 fermions are confined to a 1D harmonic oscillator potential.

(a) What is the ground state configuration (specify spatial and spin quantum numbers for each particle)?

(b) Calculate the ground state energy.

(c) Write the ground state wavefunction as a Slater determinant.

(d) If the particles interact via a weak repulsive delta-function potential, qualitatively describe how the ground state energy changes.

---

### Problem 23: Exchange Hole (Caltech 2017)

For two electrons in a metal (free electron gas), consider the pair correlation function $g(\mathbf{r})$ which gives the probability of finding another electron at distance $r$ from a given electron.

(a) For electrons with the same spin, show that the antisymmetry of the wavefunction leads to $g(0) = 0$ (the "exchange hole").

(b) For electrons with opposite spin, what is $g(0)$?

(c) Estimate the size of the exchange hole in terms of the Fermi wavelength.

(d) How does the exchange hole affect the Coulomb energy of the electron gas?

---

### Problem 24: Second Quantized Hamiltonian (Princeton 2020)

Consider electrons in a tight-binding model on a 1D lattice with N sites and lattice spacing $a$.

(a) Starting from the first-quantized Hamiltonian with hopping $t$ between nearest neighbors, derive the second-quantized form:
$$H = -t\sum_{i,\sigma}(c_{i,\sigma}^\dagger c_{i+1,\sigma} + \text{h.c.})$$

(b) Transform to momentum space and diagonalize the Hamiltonian.

(c) Find the dispersion relation $\epsilon(k)$ and the bandwidth.

(d) Add an on-site Coulomb repulsion $U\sum_i n_{i\uparrow}n_{i\downarrow}$ and discuss the physics at half-filling for $U \gg t$ (Mott insulator).

---

### Problem 25: Identical Bosons in a Double Well (Berkeley 2018)

N identical bosons are in a double-well potential with single-particle ground states $|L\rangle$ and $|R\rangle$ (localized in left and right wells) with tunnel coupling $J$ and on-site interaction $U$.

The Hamiltonian in second quantization is:
$$H = -J(a_L^\dagger a_R + a_R^\dagger a_L) + \frac{U}{2}(n_L(n_L-1) + n_R(n_R-1))$$

(a) For $U = 0$, find the ground state and its energy.

(b) For $U \gg NJ$, find the ground state in the limit of strong interactions.

(c) Describe the quantum phase transition between these regimes.

(d) This is a model for what experimental system?

---

### Problem 26: Spin-Statistics from Field Theory (Harvard 2019)

Consider a scalar field $\phi(x)$ satisfying the Klein-Gordon equation.

(a) Write the mode expansion of $\phi(x)$ in terms of creation and annihilation operators.

(b) Show that requiring causality (commuting field operators at spacelike separations) is consistent only with commutation relations (bosons).

(c) For a spinor field, explain why anticommutation relations are required instead.

(d) This is the essence of the spin-statistics theorem. What are the key assumptions?

---

### Problem 27: Many-Body Perturbation Theory (Cornell 2020)

For a system of electrons described by:
$$H = \sum_{\mathbf{k},\sigma}\epsilon_\mathbf{k}c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma} + \frac{1}{2V}\sum_{\mathbf{k},\mathbf{k}',\mathbf{q},\sigma,\sigma'}V_\mathbf{q}c_{\mathbf{k}+\mathbf{q},\sigma}^\dagger c_{\mathbf{k}'-\mathbf{q},\sigma'}^\dagger c_{\mathbf{k}'\sigma'}c_{\mathbf{k}\sigma}$$

(a) Draw the Feynman diagram for the first-order Hartree energy.

(b) Draw the diagram for the first-order Fock (exchange) energy.

(c) Which diagram vanishes for a uniform electron gas and why?

(d) Calculate the exchange energy per particle for a 3D electron gas.

---

### Problem 28: Two-Electron Atom in a Magnetic Field (Chicago 2021)

A two-electron atom is placed in a weak magnetic field $\mathbf{B} = B\hat{z}$.

(a) Write the magnetic perturbation Hamiltonian including both orbital and spin contributions.

(b) For the ground state (1s$^2$), calculate the first-order energy shift.

(c) For the first excited state (1s2s), find the energy shifts for both singlet and triplet states.

(d) Draw a level diagram showing the splitting pattern (Zeeman effect).

---

## Bonus Challenge Problems

### Problem 29: Fermionic Coherent States

Define fermionic coherent states as eigenstates of the annihilation operator with Grassmann eigenvalues.

(a) Show why ordinary complex eigenvalues don't work for fermions.

(b) Construct the fermionic coherent state $|\eta\rangle$ satisfying $c|\eta\rangle = \eta|\eta\rangle$.

(c) Calculate the overlap $\langle\eta'|\eta\rangle$.

---

### Problem 30: Anyons in 2D

In two dimensions, particles can have statistics intermediate between bosons and fermions (anyons).

(a) Explain why the exchange statistics in 2D can be more general than in 3D.

(b) For anyons with exchange phase $e^{i\theta}$, write the two-particle wavefunction under exchange.

(c) What values of $\theta$ correspond to bosons and fermions?

(d) Where do anyons appear in condensed matter physics?

---

## Problem Checklist

| Problem | Status | Time Spent | Difficulty Rating |
|---------|--------|------------|-------------------|
| 1 | [ ] | _____ min | ___/5 |
| 2 | [ ] | _____ min | ___/5 |
| 3 | [ ] | _____ min | ___/5 |
| 4 | [ ] | _____ min | ___/5 |
| 5 | [ ] | _____ min | ___/5 |
| 6 | [ ] | _____ min | ___/5 |
| 7 | [ ] | _____ min | ___/5 |
| 8 | [ ] | _____ min | ___/5 |
| 9 | [ ] | _____ min | ___/5 |
| 10 | [ ] | _____ min | ___/5 |
| 11 | [ ] | _____ min | ___/5 |
| 12 | [ ] | _____ min | ___/5 |
| 13 | [ ] | _____ min | ___/5 |
| 14 | [ ] | _____ min | ___/5 |
| 15 | [ ] | _____ min | ___/5 |
| 16 | [ ] | _____ min | ___/5 |
| 17 | [ ] | _____ min | ___/5 |
| 18 | [ ] | _____ min | ___/5 |
| 19 | [ ] | _____ min | ___/5 |
| 20 | [ ] | _____ min | ___/5 |
| 21 | [ ] | _____ min | ___/5 |
| 22 | [ ] | _____ min | ___/5 |
| 23 | [ ] | _____ min | ___/5 |
| 24 | [ ] | _____ min | ___/5 |
| 25 | [ ] | _____ min | ___/5 |
| 26 | [ ] | _____ min | ___/5 |
| 27 | [ ] | _____ min | ___/5 |
| 28 | [ ] | _____ min | ___/5 |

---

**Target:** Complete at least 20 problems before moving to Week 154.
