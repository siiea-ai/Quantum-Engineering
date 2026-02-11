# Week 152: Perturbation Theory - Comprehensive Review Guide

## Introduction

Perturbation theory is perhaps the most important approximation method in quantum mechanics. It allows us to find approximate solutions when exact solutions are unavailable, which is nearly always the case in realistic systems. This topic appears on virtually every PhD qualifying exam.

---

## Section 1: Non-Degenerate Time-Independent Perturbation Theory

### Setup

Consider a Hamiltonian that can be written as:
$$H = H_0 + \lambda H'$$

where:
- $H_0$ is exactly solvable with eigenstates $|n^{(0)}\rangle$ and energies $E_n^{(0)}$
- $H'$ is a small perturbation ($\lambda \ll 1$ is a formal parameter)
- The states $|n^{(0)}\rangle$ are non-degenerate

We seek the perturbed energies and states as power series:
$$E_n = E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2 E_n^{(2)} + \cdots$$
$$|n\rangle = |n^{(0)}\rangle + \lambda|n^{(1)}\rangle + \lambda^2|n^{(2)}\rangle + \cdots$$

### First-Order Corrections

**Energy correction:**
$$\boxed{E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle}$$

This is simply the expectation value of the perturbation in the unperturbed state.

**State correction:**
$$\boxed{|n^{(1)}\rangle = \sum_{k\neq n}\frac{\langle k^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_k^{(0)}}|k^{(0)}\rangle}$$

The perturbed state acquires components of other unperturbed states, with amplitudes proportional to the matrix element and inversely proportional to the energy difference.

### Second-Order Energy Correction

$$\boxed{E_n^{(2)} = \sum_{k\neq n}\frac{|\langle k^{(0)}|H'|n^{(0)}\rangle|^2}{E_n^{(0)} - E_k^{(0)}}}$$

**Key properties:**
1. For the ground state, all terms have $E_0^{(0)} - E_k^{(0)} < 0$, so $E_0^{(2)} < 0$. The ground state energy is always **lowered** by second-order effects.

2. Each term represents virtual transitions to state $k$ and back.

3. States close in energy contribute more (resonance enhancement).

### Validity Conditions

Perturbation theory is valid when:
$$\frac{|\langle k^{(0)}|H'|n^{(0)}\rangle|}{|E_n^{(0)} - E_k^{(0)}|} \ll 1 \quad \text{for all } k \neq n$$

The series breaks down when energy denominators become small (near degeneracy).

---

## Section 2: Examples of Non-Degenerate Perturbation

### Example 1: Anharmonic Oscillator

$H = H_0 + \lambda x^3$ where $H_0 = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2$

**First-order:** $E_n^{(1)} = \lambda\langle n|x^3|n\rangle = 0$ (by parity)

**Second-order:**
$$E_n^{(2)} = \lambda^2\sum_{k\neq n}\frac{|\langle k|x^3|n\rangle|^2}{E_n^{(0)} - E_k^{(0)}}$$

Using ladder operators: $x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^{\dagger})$

### Example 2: Infinite Square Well with Delta Perturbation

$H' = \alpha\delta(x - a/2)$ (spike at center of well)

**First-order:**
$$E_n^{(1)} = \alpha|\psi_n(a/2)|^2 = \frac{2\alpha}{a}\sin^2\left(\frac{n\pi}{2}\right)$$

For even $n$: $E_n^{(1)} = 0$
For odd $n$: $E_n^{(1)} = \frac{2\alpha}{a}$

---

## Section 3: Degenerate Perturbation Theory

### When Non-Degenerate Theory Fails

If $E_n^{(0)} = E_m^{(0)}$ for $n \neq m$, the formula for $|n^{(1)}\rangle$ contains a zero in the denominator. The perturbation mixes the degenerate states, and we must find the correct linear combinations first.

### The Procedure

1. Identify the degenerate subspace with dimension $g$
2. Construct the $g \times g$ matrix $W_{ij} = \langle i^{(0)}|H'|j^{(0)}\rangle$
3. Diagonalize $W$: the eigenvalues are $E^{(1)}$, eigenvectors are "good" states
4. The "good" states have definite first-order energy corrections
5. Continue with non-degenerate theory if needed for second-order

### Example: Linear Stark Effect in Hydrogen (n=2)

The $n=2$ level of hydrogen has four degenerate states: $|2,0,0\rangle$, $|2,1,0\rangle$, $|2,1,1\rangle$, $|2,1,-1\rangle$.

The perturbation is $H' = eEz$ (electric field along z).

**Matrix elements:**
- Only $\langle 2,0,0|z|2,1,0\rangle = -3a_0$ is non-zero
- All others vanish (selection rule: $\Delta m = 0$, $\Delta l = \pm 1$)

**The W matrix:**
$$W = \begin{pmatrix} 0 & -3eEa_0 & 0 & 0 \\ -3eEa_0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**Eigenvalues:** $E^{(1)} = \pm 3eEa_0, 0, 0$

The degeneracy is partially lifted: two states shift linearly with the field (linear Stark effect), while two remain unshifted.

---

## Section 4: Time-Dependent Perturbation Theory

### Setup

The Hamiltonian is $H = H_0 + H'(t)$ where $H'(t)$ is turned on at $t = 0$.

Initial state: $|\psi(0)\rangle = |i\rangle$, an eigenstate of $H_0$.

We work in the **interaction picture** where the time evolution from $H_0$ is absorbed.

### Transition Amplitude

To first order in $H'$:
$$\boxed{c_f^{(1)}(t) = -\frac{i}{\hbar}\int_0^t dt'\, \langle f|H'(t')|i\rangle e^{i\omega_{fi}t'}}$$

where $\omega_{fi} = (E_f - E_i)/\hbar$.

The transition probability is $P_{i\to f}(t) = |c_f(t)|^2$.

### Sinusoidal Perturbation

For $H'(t) = V\cos(\omega t) = \frac{V}{2}(e^{i\omega t} + e^{-i\omega t})$:

$$c_f^{(1)}(t) = -\frac{i}{2\hbar}\langle f|V|i\rangle\left[\frac{e^{i(\omega_{fi}+\omega)t} - 1}{\omega_{fi}+\omega} + \frac{e^{i(\omega_{fi}-\omega)t} - 1}{\omega_{fi}-\omega}\right]$$

**Near resonance** ($\omega \approx \omega_{fi}$), the second term dominates:

$$P_{i\to f}(t) \approx \frac{|\langle f|V|i\rangle|^2}{4\hbar^2}\frac{\sin^2[(\omega_{fi}-\omega)t/2]}{[(\omega_{fi}-\omega)/2]^2}$$

This is the **resonance formula**: maximum transition probability when $\omega = \omega_{fi}$.

---

## Section 5: Fermi's Golden Rule

### Transitions to a Continuum

When the final state belongs to a continuum (or quasi-continuum), we sum over final states weighted by the density of states $\rho(E_f)$.

### The Golden Rule

$$\boxed{\Gamma_{i\to f} = \frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho(E_f)}$$

**Key features:**
1. The transition rate is constant in time (for long times)
2. Proportional to $|$matrix element$|^2$
3. Proportional to density of states at the final energy
4. Energy conservation is enforced: $E_f = E_i + \hbar\omega$ (for $H' \propto e^{i\omega t}$)

### Applications

**Spontaneous emission rate:**
$$\Gamma = \frac{\omega^3}{3\pi\epsilon_0\hbar c^3}|\langle f|\mathbf{d}|i\rangle|^2$$

**Photoelectric effect:**
$$\sigma \propto |\langle f|\mathbf{r}|i\rangle|^2 \rho(E_f)$$

**Nuclear beta decay:**
$$\Gamma = \frac{2\pi}{\hbar}|M_{fi}|^2 \rho(E_e)$$

---

## Section 6: The Adiabatic Theorem

### Statement

If a Hamiltonian $H(t)$ varies **slowly** compared to the internal time scales of the system, and the system starts in an eigenstate $|n(0)\rangle$ of $H(0)$, then it remains in the instantaneous eigenstate $|n(t)\rangle$ of $H(t)$.

### Quantitative Condition

The adiabatic condition requires:
$$\left|\frac{\langle m|\dot{H}|n\rangle}{(E_n - E_m)^2}\right| \ll \frac{1}{\hbar}$$

for all $m \neq n$. This is satisfied when changes occur slowly compared to $\hbar/(E_n - E_m)$.

### The Phase

Under adiabatic evolution:
$$|\psi(t)\rangle = e^{i\gamma_n(t)}e^{-i\theta_n(t)}|n(t)\rangle$$

**Dynamical phase:**
$$\theta_n(t) = \frac{1}{\hbar}\int_0^t E_n(t')dt'$$

**Geometric (Berry) phase:**
$$\gamma_n(t) = i\int_0^t \langle n(t')|\dot{n}(t')\rangle dt'$$

---

## Section 7: Berry Phase

### Definition

For a Hamiltonian depending on parameters $\mathbf{R}(t)$, the Berry phase accumulated over a closed path in parameter space is:

$$\boxed{\gamma_n = i\oint \langle n(\mathbf{R})|\nabla_{\mathbf{R}}|n(\mathbf{R})\rangle \cdot d\mathbf{R}}$$

### Properties

1. **Geometric:** Depends only on the path in parameter space, not on the speed
2. **Gauge invariant:** The result is invariant under phase redefinitions of $|n\rangle$
3. **Observable:** Can be measured in interference experiments

### Example: Spin in a Rotating Magnetic Field

A spin-1/2 in a magnetic field $\mathbf{B}(\theta, \phi)$ with fixed magnitude but varying direction.

When $\mathbf{B}$ traces a closed path on the unit sphere enclosing solid angle $\Omega$:

$$\gamma = -\frac{\Omega}{2}$$

For a cone of half-angle $\alpha$, $\Omega = 2\pi(1 - \cos\alpha)$, so:
$$\gamma = -\pi(1 - \cos\alpha)$$

### Connection to Aharonov-Bohm

The Aharonov-Bohm phase $e^{i(e/\hbar)\oint \mathbf{A}\cdot d\mathbf{r}}$ is a Berry phase associated with magnetic flux.

---

## Summary Table

| Method | When to Use | Key Formula |
|--------|-------------|-------------|
| Non-degen PT, 1st order | Non-degenerate, estimate shift | $E^{(1)} = \langle H' \rangle$ |
| Non-degen PT, 2nd order | Need correction, convergent | $E^{(2)} = \sum \frac{|\langle k|H'|n\rangle|^2}{E_n - E_k}$ |
| Degenerate PT | Degenerate levels | Diagonalize $H'$ in subspace |
| Time-dep PT | Time-varying perturbation | $c_f = -\frac{i}{\hbar}\int \langle f|H'|i\rangle e^{i\omega_{fi}t}dt$ |
| Fermi's golden rule | Transitions to continuum | $\Gamma = \frac{2\pi}{\hbar}|M|^2\rho(E)$ |
| Adiabatic theorem | Slow changes | Stay in instantaneous eigenstate |
| Berry phase | Cyclic adiabatic evolution | $\gamma = i\oint\langle n|\nabla_R n\rangle \cdot dR$ |

---

## References

1. Shankar, R. *Principles of Quantum Mechanics*, Chapter 17
2. Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 5
3. Griffiths, D.J. *Introduction to Quantum Mechanics*, Chapters 7-9
4. Berry, M.V. "Quantal phase factors accompanying adiabatic changes," Proc. R. Soc. Lond. A 392, 45 (1984)
5. [MIT 8.06 Perturbation Theory Notes](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/)

---

**Word Count:** ~2800
**Last Updated:** February 9, 2026
