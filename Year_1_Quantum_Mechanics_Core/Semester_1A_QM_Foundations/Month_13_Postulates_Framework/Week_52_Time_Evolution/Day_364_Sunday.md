# Day 364: Month 13 Capstone — Postulates and Mathematical Framework Complete

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Practice Qualifying Exam |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Solutions Review & Synthesis |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Project |

**Total Study Time:** 7 hours

---

## Congratulations!

You have completed **Month 13: Postulates and Mathematical Framework** — the foundational month of quantum mechanics. This capstone day consolidates your understanding of all five postulates and prepares you for Month 14's applications to one-dimensional systems.

---

## Month 13 Summary

### The Five Postulates of Quantum Mechanics

| Postulate | Statement | Week |
|-----------|-----------|------|
| **1. State Space** | The state of a quantum system is a vector $\|\psi\rangle$ in a complex Hilbert space $\mathcal{H}$ | 49 |
| **2. Observables** | Physical quantities are represented by Hermitian operators $\hat{A}^\dagger = \hat{A}$ | 50 |
| **3. Measurement** | Measuring $\hat{A}$ yields eigenvalue $a$ with probability $P(a) = \|\langle a\|\psi\rangle\|^2$ | 50 |
| **4. Collapse** | After measuring $a$, the state becomes $\|a\rangle$ | 50 |
| **5. Time Evolution** | $i\hbar\frac{\partial}{\partial t}\|\psi\rangle = \hat{H}\|\psi\rangle$ | 52 |

### Key Mathematical Structures

| Concept | Mathematical Object | Physical Role |
|---------|---------------------|---------------|
| States | Vectors in Hilbert space | System configuration |
| Observables | Hermitian operators | Measurable quantities |
| Measurements | Projections | Information extraction |
| Symmetries | Unitary operators | Physical transformations |
| Dynamics | Time evolution operator | State propagation |

### The Three Pictures

| Picture | States | Operators | Use |
|---------|--------|-----------|-----|
| Schrodinger | $\|\psi_S(t)\rangle = \hat{U}(t)\|\psi(0)\rangle$ | Fixed | Wave functions, TISE |
| Heisenberg | Fixed $\|\psi_H\rangle$ | $\hat{A}_H(t) = \hat{U}^\dagger\hat{A}_S\hat{U}$ | Classical correspondence |
| Interaction | $\|\psi_I(t)\rangle$ via $\hat{V}$ | $\hat{A}_I(t)$ via $\hat{H}_0$ | Perturbation theory |

---

## Practice Qualifying Exam

**Time Allowed:** 3 hours
**Total Points:** 150

### Instructions

- Show all work for full credit
- State any formulas you use
- Partial credit awarded for correct approach
- Calculators allowed for numerical computations

---

### Part A: Hilbert Space and Dirac Notation (30 points)

**Problem A1 (10 points):**
Consider a three-dimensional Hilbert space with orthonormal basis $\{|1\rangle, |2\rangle, |3\rangle\}$.

(a) Write the most general normalized state $|\psi\rangle$ in this space.

(b) For the state $|\psi\rangle = \frac{1}{\sqrt{3}}|1\rangle + \frac{i}{\sqrt{3}}|2\rangle + \frac{1}{\sqrt{3}}|3\rangle$, compute the density matrix $\hat{\rho} = |\psi\rangle\langle\psi|$.

(c) Verify that $\text{Tr}(\hat{\rho}) = 1$ and $\text{Tr}(\hat{\rho}^2) = 1$.

**Problem A2 (10 points):**
The operators $\hat{A}$ and $\hat{B}$ are represented in the $\{|1\rangle, |2\rangle\}$ basis as:
$$\hat{A} = \begin{pmatrix} 1 & 2 \\ 2 & -1 \end{pmatrix}, \quad \hat{B} = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

(a) Show that $\hat{A}$ and $\hat{B}$ are Hermitian.

(b) Compute the commutator $[\hat{A}, \hat{B}]$.

(c) Find the eigenvalues and normalized eigenvectors of $\hat{A}$.

**Problem A3 (10 points):**
Prove that for any Hermitian operator $\hat{A}$:

(a) All eigenvalues are real.

(b) Eigenvectors corresponding to different eigenvalues are orthogonal.

---

### Part B: Measurement and Uncertainty (40 points)

**Problem B1 (15 points):**
A spin-1/2 particle is prepared in the state $|\psi\rangle = \cos(\theta/2)|+z\rangle + e^{i\phi}\sin(\theta/2)|-z\rangle$.

(a) What are the probabilities of measuring $S_z = +\hbar/2$ and $S_z = -\hbar/2$?

(b) Calculate $\langle\hat{S}_z\rangle$, $\langle\hat{S}_z^2\rangle$, and $\Delta S_z$.

(c) If $S_x$ is measured instead, what are the possible outcomes and their probabilities? (Express your answer in terms of $\theta$ and $\phi$.)

**Problem B2 (15 points):**
Consider the uncertainty principle for position and momentum.

(a) Starting from the generalized uncertainty relation $\sigma_A\sigma_B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$, derive the Heisenberg uncertainty principle $\Delta x \cdot \Delta p \geq \hbar/2$.

(b) A particle is described by the wave function $\psi(x) = (\pi\sigma^2)^{-1/4}e^{-x^2/2\sigma^2}$. Calculate $\Delta x$ and $\Delta p$, and verify the uncertainty relation.

(c) Does this state saturate the uncertainty bound? What is special about Gaussian wave packets?

**Problem B3 (10 points):**
Two observables $\hat{A}$ and $\hat{B}$ satisfy $[\hat{A}, \hat{B}] = i\hbar\hat{C}$ where $\hat{C}$ is Hermitian.

(a) Can $\hat{A}$ and $\hat{B}$ have simultaneous eigenstates? Under what condition?

(b) If $|\psi\rangle$ is an eigenstate of $\hat{C}$ with eigenvalue 0, what can you say about measuring $\hat{A}$ and $\hat{B}$ in this state?

---

### Part C: Time Evolution (50 points)

**Problem C1 (15 points):**
A two-level system has Hamiltonian $\hat{H} = \epsilon\sigma_z + \Delta\sigma_x$ where $\epsilon, \Delta > 0$.

(a) Find the energy eigenvalues $E_\pm$.

(b) Find the normalized energy eigenstates $|E_+\rangle$ and $|E_-\rangle$ in the $\{|0\rangle, |1\rangle\}$ basis.

(c) If the system starts in state $|0\rangle$ at $t = 0$, find the probability of being in state $|1\rangle$ at time $t$.

**Problem C2 (15 points):**
For the harmonic oscillator $\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a} + 1/2)$:

(a) In the Heisenberg picture, solve for $\hat{a}_H(t)$ and $\hat{a}^\dagger_H(t)$.

(b) Using these, find $\hat{x}_H(t)$ and $\hat{p}_H(t)$.

(c) A coherent state $|\alpha\rangle$ has the property $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$. Show that:
$$\langle\hat{x}\rangle(t) = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(\alpha e^{-i\omega t})$$

**Problem C3 (20 points):**
Consider the Interaction picture for a system with $\hat{H} = \hat{H}_0 + \hat{V}$ where $\hat{H}_0 = \hbar\omega_0|1\rangle\langle 1|$ (taking $|0\rangle$ as zero energy) and $\hat{V} = \hbar g(|0\rangle\langle 1| + |1\rangle\langle 0|)$.

(a) Write the Interaction picture perturbation $\hat{V}_I(t)$.

(b) Starting from $|\psi(0)\rangle = |0\rangle$, calculate the first-order transition amplitude $c_1^{(1)}(t) = \langle 1|\hat{U}_I^{(1)}(t, 0)|0\rangle$.

(c) For what time $t$ does $|c_1^{(1)}(t)|^2 = 1/4$?

(d) The exact transition probability is $P_{1\leftarrow 0}(t) = \frac{g^2}{\omega_0^2 + g^2}\sin^2(\sqrt{\omega_0^2 + g^2}\cdot t/2)$. Under what condition is first-order perturbation theory valid?

---

### Part D: Synthesis and Applications (30 points)

**Problem D1 (15 points):**
A quantum system has the Hamiltonian $\hat{H} = \hat{p}^2/2m + V_0\Theta(x)$ where $\Theta(x)$ is the Heaviside step function (potential step).

(a) Is energy conserved? Justify using the commutator $[\hat{H}, \hat{H}]$.

(b) Is momentum conserved? Calculate $[\hat{H}, \hat{p}]$ and interpret the result.

(c) A wave packet incident from the left will partially reflect and partially transmit. Using Ehrenfest's theorem, explain qualitatively how $\langle\hat{x}\rangle$ and $\langle\hat{p}\rangle$ behave during the scattering process.

**Problem D2 (15 points):**
Connect quantum mechanics to quantum computing:

(a) A qubit gate $\hat{U}$ satisfies $\hat{U}|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$ and $\hat{U}|1\rangle = \frac{1}{\sqrt{2}}(i|0\rangle + |1\rangle)$. Write $\hat{U}$ as a matrix and verify it is unitary.

(b) Find a Hamiltonian $\hat{H}$ and time $T$ such that $\hat{U} = e^{-i\hat{H}T/\hbar}$. (Hint: $\hat{U}$ is related to a Pauli matrix.)

(c) If error rates are $10^{-4}$ per gate, how does this relate to the time evolution picture? (Discuss unitarity vs. non-unitary processes.)

---

## Solutions and Discussion

### Part A Solutions

**A1:**
(a) $|\psi\rangle = c_1|1\rangle + c_2|2\rangle + c_3|3\rangle$ with $|c_1|^2 + |c_2|^2 + |c_3|^2 = 1$

(b) $\hat{\rho} = \frac{1}{3}\begin{pmatrix} 1 & -i & 1 \\ i & 1 & i \\ 1 & -i & 1 \end{pmatrix}$

(c) $\text{Tr}(\hat{\rho}) = 1$ (sum of diagonal elements). $\text{Tr}(\hat{\rho}^2) = 1$ (pure state).

**A2:**
(a) $\hat{A}^\dagger = \hat{A}$ (real symmetric), $\hat{B}^\dagger = \hat{B}$ (check: $(-i)^* = i$, $(i)^* = -i$).

(b) $[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A} = \begin{pmatrix} 4i & -2i \\ -2i & -4i \end{pmatrix}$

(c) Eigenvalues of $\hat{A}$: $\det(\hat{A} - \lambda I) = 0 \Rightarrow \lambda^2 - 5 = 0 \Rightarrow \lambda = \pm\sqrt{5}$

**A3:**
(a) $\hat{A}|a\rangle = a|a\rangle$. Take adjoint: $\langle a|\hat{A}^\dagger = a^*\langle a|$. Since $\hat{A}^\dagger = \hat{A}$: $\langle a|\hat{A} = a^*\langle a|$. Then $\langle a|\hat{A}|a\rangle = a = a^*$.

(b) $\hat{A}|a\rangle = a|a\rangle$, $\hat{A}|a'\rangle = a'|a'\rangle$. $\langle a'|\hat{A}|a\rangle = a\langle a'|a\rangle = a'\langle a'|a\rangle$. So $(a - a')\langle a'|a\rangle = 0$. If $a \neq a'$, then $\langle a'|a\rangle = 0$.

### Part B Solutions

**B1:**
(a) $P(+\hbar/2) = \cos^2(\theta/2)$, $P(-\hbar/2) = \sin^2(\theta/2)$

(b) $\langle\hat{S}_z\rangle = \frac{\hbar}{2}\cos\theta$, $\langle\hat{S}_z^2\rangle = \frac{\hbar^2}{4}$, $\Delta S_z = \frac{\hbar}{2}|\sin\theta|$

(c) $P(S_x = +\hbar/2) = \frac{1}{2}(1 + \sin\theta\cos\phi)$, $P(S_x = -\hbar/2) = \frac{1}{2}(1 - \sin\theta\cos\phi)$

**B2:**
(a) $[\hat{x}, \hat{p}] = i\hbar \Rightarrow \sigma_x\sigma_p \geq \frac{1}{2}|i\hbar| = \frac{\hbar}{2}$

(b) $\Delta x = \sigma/\sqrt{2}$, $\Delta p = \hbar/(2\sigma\sqrt{2})$. $\Delta x \Delta p = \hbar/2$.

(c) Yes, Gaussians saturate the bound. They are minimum uncertainty states.

### Part C Solutions

**C1:**
(a) $E_\pm = \pm\sqrt{\epsilon^2 + \Delta^2}$

(b) $|E_+\rangle = \cos(\phi/2)|0\rangle + \sin(\phi/2)|1\rangle$ where $\tan\phi = \Delta/\epsilon$

(c) $P_{1\leftarrow 0}(t) = \frac{\Delta^2}{\epsilon^2 + \Delta^2}\sin^2\left(\frac{\sqrt{\epsilon^2 + \Delta^2}}{\hbar}t\right)$

**C2:**
(a) $\hat{a}_H(t) = \hat{a}(0)e^{-i\omega t}$, $\hat{a}^\dagger_H(t) = \hat{a}^\dagger(0)e^{i\omega t}$

(b) $\hat{x}_H(t) = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a}(0)e^{-i\omega t} + \hat{a}^\dagger(0)e^{i\omega t})$

(c) $\langle\hat{x}\rangle = \sqrt{\frac{\hbar}{2m\omega}}\langle\alpha|(\hat{a}e^{-i\omega t} + \hat{a}^\dagger e^{i\omega t})|\alpha\rangle = \sqrt{\frac{\hbar}{2m\omega}}(\alpha e^{-i\omega t} + \alpha^* e^{i\omega t}) = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(\alpha e^{-i\omega t})$

**C3:**
(a) $\hat{V}_I(t) = \hbar g(|0\rangle\langle 1|e^{i\omega_0 t} + |1\rangle\langle 0|e^{-i\omega_0 t})$

(b) $c_1^{(1)}(t) = -ig\int_0^t e^{-i\omega_0 t'}dt' = -ig\frac{1 - e^{-i\omega_0 t}}{i\omega_0} = \frac{g}{\omega_0}(e^{-i\omega_0 t} - 1)$

(c) $|c_1^{(1)}|^2 = \frac{4g^2}{\omega_0^2}\sin^2(\omega_0 t/2) = 1/4$ when $\sin^2(\omega_0 t/2) = \omega_0^2/16g^2$

(d) First-order valid when $g \ll \omega_0$ (weak coupling limit).

---

## Month 13 Comprehensive Formula Sheet

### Hilbert Space

$$\langle\psi|\phi\rangle = \langle\phi|\psi\rangle^*, \quad |\psi\rangle = \sum_n c_n|n\rangle, \quad c_n = \langle n|\psi\rangle$$
$$\hat{I} = \sum_n |n\rangle\langle n|, \quad \langle\psi|\psi\rangle = 1$$

### Operators

$$\hat{A}^\dagger = \hat{A} \text{ (Hermitian)}, \quad \hat{U}^\dagger\hat{U} = \hat{I} \text{ (Unitary)}$$
$$[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}, \quad [\hat{x}, \hat{p}] = i\hbar$$

### Measurement

$$P(a) = |\langle a|\psi\rangle|^2, \quad \langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle$$
$$(\Delta A)^2 = \langle\hat{A}^2\rangle - \langle\hat{A}\rangle^2$$

### Uncertainty

$$\sigma_A\sigma_B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|, \quad \Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

### Time Evolution

$$i\hbar\frac{\partial}{\partial t}|\psi\rangle = \hat{H}|\psi\rangle, \quad \hat{U}(t) = e^{-i\hat{H}t/\hbar}$$
$$|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar}|E_n\rangle$$

### Heisenberg Picture

$$\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t), \quad \frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H]$$

### Ehrenfest

$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{\langle\hat{p}\rangle}{m}, \quad \frac{d\langle\hat{p}\rangle}{dt} = -\left\langle\frac{\partial V}{\partial x}\right\rangle$$

---

## Computational Project: Month 13 Simulator

```python
"""
Day 364 Computational Project: Quantum Mechanics Simulator
Month 13 Capstone - Year 1

A comprehensive simulator demonstrating all concepts from Month 13.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh

class QuantumSystem:
    """A quantum system simulator supporting all Month 13 concepts."""

    def __init__(self, dim, name="Quantum System"):
        """
        Initialize a quantum system.

        Parameters:
        -----------
        dim : int
            Dimension of the Hilbert space
        name : str
            Name of the system
        """
        self.dim = dim
        self.name = name
        self.state = None
        self.hamiltonian = None
        self.hbar = 1.0

    def set_state(self, state):
        """Set the quantum state (normalizes automatically)."""
        state = np.array(state, dtype=complex).reshape(-1, 1)
        self.state = state / np.linalg.norm(state)
        print(f"State set (normalized): {self.state.flatten()}")

    def set_hamiltonian(self, H):
        """Set the Hamiltonian."""
        self.hamiltonian = np.array(H, dtype=complex)
        assert self.hamiltonian.shape == (self.dim, self.dim)
        assert np.allclose(self.hamiltonian, self.hamiltonian.conj().T), "H must be Hermitian"
        print(f"Hamiltonian set:\n{self.hamiltonian}")

    def evolve(self, t):
        """Evolve the state to time t (Schrodinger picture)."""
        U = expm(-1j * self.hamiltonian * t / self.hbar)
        self.state = U @ self.state

    def expectation(self, A):
        """Compute expectation value of operator A."""
        A = np.array(A, dtype=complex)
        return np.real((self.state.conj().T @ A @ self.state)[0, 0])

    def measure(self, A):
        """
        Simulate measurement of observable A.
        Returns eigenvalue and collapses state.
        """
        A = np.array(A, dtype=complex)
        eigenvalues, eigenvectors = eigh(A)

        # Compute probabilities
        probs = np.abs(eigenvectors.conj().T @ self.state)**2
        probs = probs.flatten()

        # Sample outcome
        outcome_idx = np.random.choice(len(eigenvalues), p=probs)
        outcome = eigenvalues[outcome_idx]

        # Collapse state
        self.state = eigenvectors[:, outcome_idx:outcome_idx+1]
        self.state = self.state / np.linalg.norm(self.state)

        return outcome

    def get_probabilities(self, A):
        """Get measurement probabilities for observable A."""
        A = np.array(A, dtype=complex)
        eigenvalues, eigenvectors = eigh(A)
        probs = np.abs(eigenvectors.conj().T @ self.state)**2
        return eigenvalues, probs.flatten()


class TwoLevelSystem(QuantumSystem):
    """Specialized two-level (qubit) system."""

    def __init__(self, name="Qubit"):
        super().__init__(2, name)

        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.I = np.eye(2, dtype=complex)

    def set_bloch_state(self, theta, phi):
        """Set state from Bloch sphere angles."""
        state = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        self.set_state(state)

    def get_bloch_vector(self):
        """Get Bloch vector (x, y, z)."""
        x = self.expectation(self.sigma_x)
        y = self.expectation(self.sigma_y)
        z = self.expectation(self.sigma_z)
        return x, y, z

    def rabi_oscillations(self, Omega, delta, t_max, n_points=200):
        """
        Simulate Rabi oscillations.

        Parameters:
        -----------
        Omega : float
            Rabi frequency
        delta : float
            Detuning
        t_max : float
            Maximum time
        n_points : int
            Number of time points

        Returns:
        --------
        t, P0, P1 : arrays
            Time, probabilities of |0> and |1>
        """
        # RWA Hamiltonian
        H_RWA = self.hbar/2 * (delta * self.sigma_z + Omega * self.sigma_x)

        t_values = np.linspace(0, t_max, n_points)
        P0, P1 = [], []

        initial_state = self.state.copy()

        for t in t_values:
            U = expm(-1j * H_RWA * t / self.hbar)
            psi_t = U @ initial_state
            P0.append(np.abs(psi_t[0, 0])**2)
            P1.append(np.abs(psi_t[1, 0])**2)

        self.state = initial_state  # Reset state

        return t_values, np.array(P0), np.array(P1)


def demo_month_13():
    """Demonstrate all Month 13 concepts."""

    print("=" * 70)
    print("MONTH 13 CAPSTONE: QUANTUM MECHANICS SIMULATOR")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Week 49: Hilbert Space
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEEK 49: HILBERT SPACE FORMALISM")
    print("=" * 70)

    qubit = TwoLevelSystem("Demo Qubit")
    qubit.set_state([1, 0])  # |0>

    print(f"\nInitial state: |0>")
    print(f"Bloch vector: {qubit.get_bloch_vector()}")

    # Superposition
    qubit.set_state([1, 1])  # |+>
    print(f"\nAfter setting |+> = (|0> + |1>)/sqrt(2):")
    print(f"Bloch vector: {qubit.get_bloch_vector()}")

    # -------------------------------------------------------------------------
    # Week 50: Measurement
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEEK 50: MEASUREMENT")
    print("=" * 70)

    qubit.set_state([1, 1])  # Reset to |+>

    # Get measurement probabilities
    eigenvalues, probs = qubit.get_probabilities(qubit.sigma_z)
    print(f"\nMeasuring sigma_z on |+> state:")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Probabilities: {probs}")
    print(f"  <sigma_z> = {qubit.expectation(qubit.sigma_z):.4f}")

    # Simulate measurement
    print("\nSimulating 1000 measurements:")
    results = []
    for _ in range(1000):
        qubit.set_state([1, 1])
        result = qubit.measure(qubit.sigma_z)
        results.append(result)

    print(f"  Mean outcome: {np.mean(results):.4f} (expected: 0)")
    print(f"  Fraction +1: {np.sum(np.array(results) > 0) / len(results):.3f} (expected: 0.5)")

    # -------------------------------------------------------------------------
    # Week 51: Uncertainty
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEEK 51: UNCERTAINTY")
    print("=" * 70)

    qubit.set_bloch_state(np.pi/3, 0)  # General state

    sx = qubit.expectation(qubit.sigma_x)
    sy = qubit.expectation(qubit.sigma_y)
    sz = qubit.expectation(qubit.sigma_z)

    sx2 = qubit.expectation(qubit.sigma_x @ qubit.sigma_x)
    sy2 = qubit.expectation(qubit.sigma_y @ qubit.sigma_y)

    delta_sx = np.sqrt(sx2 - sx**2)
    delta_sy = np.sqrt(sy2 - sy**2)

    # Commutator [sigma_x, sigma_y] = 2i*sigma_z
    commutator_exp = 2 * sz

    print(f"\nState: theta=pi/3, phi=0 on Bloch sphere")
    print(f"<sigma_x> = {sx:.4f}, <sigma_y> = {sy:.4f}, <sigma_z> = {sz:.4f}")
    print(f"Delta(sigma_x) = {delta_sx:.4f}")
    print(f"Delta(sigma_y) = {delta_sy:.4f}")
    print(f"Delta_x * Delta_y = {delta_sx * delta_sy:.4f}")
    print(f"|<[sigma_x, sigma_y]>|/2 = |i*<sigma_z>| = {np.abs(commutator_exp):.4f}")
    print(f"Uncertainty relation satisfied: {delta_sx * delta_sy >= np.abs(commutator_exp)/2}")

    # -------------------------------------------------------------------------
    # Week 52: Time Evolution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEEK 52: TIME EVOLUTION")
    print("=" * 70)

    # Set up driven qubit
    omega_0 = 1.0
    H = qubit.hbar * omega_0 / 2 * qubit.sigma_z
    qubit.set_hamiltonian(H)
    qubit.set_state([1, 1])  # |+>

    # Evolve and track
    print(f"\nEvolution under H = (hbar*omega/2)*sigma_z:")
    print(f"Initial Bloch vector: {qubit.get_bloch_vector()}")

    qubit.evolve(np.pi / omega_0)  # Half period
    print(f"After t = pi/omega: {qubit.get_bloch_vector()}")

    qubit.evolve(np.pi / omega_0)  # Full period
    print(f"After t = 2*pi/omega: {qubit.get_bloch_vector()}")

    # Rabi oscillations
    print("\n--- Rabi Oscillations ---")
    qubit.set_state([1, 0])  # Start in |0>

    Omega = 0.5
    t, P0, P1 = qubit.rabi_oscillations(Omega, delta=0, t_max=4*np.pi/Omega)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rabi oscillations
    ax1 = axes[0, 0]
    ax1.plot(t * Omega / (2*np.pi), P0, 'b-', label='P(|0>)', linewidth=2)
    ax1.plot(t * Omega / (2*np.pi), P1, 'r-', label='P(|1>)', linewidth=2)
    ax1.set_xlabel('Time (Rabi periods)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Resonant Rabi Oscillations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Off-resonant
    t_off, P0_off, P1_off = qubit.rabi_oscillations(Omega, delta=Omega, t_max=4*np.pi/Omega)

    ax2 = axes[0, 1]
    ax2.plot(t_off * Omega / (2*np.pi), P0_off, 'b-', label='P(|0>)', linewidth=2)
    ax2.plot(t_off * Omega / (2*np.pi), P1_off, 'r-', label='P(|1>)', linewidth=2)
    ax2.set_xlabel('Time (Rabi periods)')
    ax2.set_ylabel('Probability')
    ax2.set_title('Off-Resonant (Delta = Omega)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bloch sphere trajectory
    ax3 = axes[1, 0]

    qubit.set_state([1, 0])
    bloch_x, bloch_z = [], []

    for t_val in np.linspace(0, 2*np.pi/Omega, 100):
        H_RWA = qubit.hbar/2 * Omega * qubit.sigma_x
        U = expm(-1j * H_RWA * t_val / qubit.hbar)
        psi = U @ np.array([[1], [0]])
        rho = np.outer(psi, psi.conj())
        x = 2 * np.real(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        bloch_x.append(x)
        bloch_z.append(z)

    ax3.plot(bloch_x, bloch_z, 'purple', linewidth=2)
    ax3.plot(bloch_x[0], bloch_z[0], 'go', markersize=10, label='Start')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax3.add_patch(circle)
    ax3.set_xlabel('Bloch x')
    ax3.set_ylabel('Bloch z')
    ax3.set_title('Bloch Sphere: Resonant Rabi')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Month summary
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.9, 'MONTH 13 COMPLETE', fontsize=20, ha='center', va='top',
             transform=ax4.transAxes, fontweight='bold')
    ax4.text(0.5, 0.7, 'Postulates Mastered:', fontsize=14, ha='center', va='top',
             transform=ax4.transAxes)
    ax4.text(0.5, 0.55, '1. State Space (Hilbert)', fontsize=12, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.45, '2. Observables (Hermitian)', fontsize=12, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.35, '3. Measurement (Born Rule)', fontsize=12, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.25, '4. Collapse (Projection)', fontsize=12, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.15, '5. Evolution (Schrodinger)', fontsize=12, ha='center', transform=ax4.transAxes)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('day_364_month_13_capstone.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as 'day_364_month_13_capstone.png'")

    print("\n" + "=" * 70)
    print("MONTH 13 CAPSTONE COMPLETE!")
    print("Ready for Month 14: One-Dimensional Systems")
    print("=" * 70)


if __name__ == "__main__":
    demo_month_13()
```

---

## Transition to Month 14

### What You've Learned (Month 13)

- The mathematical structure of quantum mechanics
- How to represent states, observables, and measurements
- The uncertainty principle and its consequences
- Three equivalent pictures for time evolution
- Perturbation theory foundations

### What's Coming (Month 14: One-Dimensional Systems)

| Week | Topic | Key Systems |
|------|-------|-------------|
| 53 | Free Particle | Wave packets, dispersion, group velocity |
| 54 | Particle in a Box | Infinite well, quantization, boundary conditions |
| 55 | Harmonic Oscillator | Ladder operators, coherent states, applications |
| 56 | Barriers and Tunneling | Step potential, finite well, quantum tunneling |

---

## Daily Checklist

- [ ] Review all Week 49-52 summaries
- [ ] Complete the practice qualifying exam (3 hours, closed book)
- [ ] Review solutions and identify weak areas
- [ ] Work through the computational project
- [ ] Write a one-page summary of Month 13
- [ ] Identify top 3 areas needing more practice
- [ ] Preview Month 14 topics in Shankar Chapters 5-7

---

## Final Thoughts

You have now mastered the **postulates of quantum mechanics** — the logical foundation upon which all quantum physics is built. The mathematical framework you've learned (Hilbert spaces, Hermitian operators, unitary evolution, measurements) applies to:

- Atomic and molecular physics
- Condensed matter physics
- Particle physics
- Quantum computing
- Quantum information

From Month 14 onwards, we apply these principles to increasingly complex systems, building intuition and computational skills.

---

*"I think I can safely say that nobody understands quantum mechanics."*
— Richard Feynman

*Yet after Month 13, you understand its mathematical structure better than most!*

---

**Next:** Month 14, Week 53 — Free Particle and Wave Packets

---

## Month 14 Preview

In Month 14, we apply the postulate framework to one-dimensional quantum systems:

- **Week 53:** Free particle, wave packet dynamics, dispersion
- **Week 54:** Infinite and finite square wells, energy quantization
- **Week 55:** Quantum harmonic oscillator, ladder operators, coherent states
- **Week 56:** Potential barriers, tunneling, scattering theory preview

These exactly solvable systems provide the foundation for understanding more complex quantum phenomena.

---

**Congratulations on completing Month 13!**
