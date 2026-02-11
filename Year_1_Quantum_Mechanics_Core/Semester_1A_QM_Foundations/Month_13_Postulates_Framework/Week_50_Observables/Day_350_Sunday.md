# Day 350: Week 50 Review and Qiskit Lab

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive Review and Concept Integration |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Practice Exam (100 Points) |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Qiskit Lab: Measurements and Born Rule |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Synthesize all Week 50 concepts into a coherent understanding of quantum observables
2. Solve problems combining measurement, commutators, and representations
3. Apply the Born rule to calculate measurement probabilities
4. Implement quantum measurements in Qiskit
5. Verify measurement statistics experimentally through simulation
6. Transform between measurement bases in quantum circuits

---

## Week 50 Synthesis: The Physical Content of Quantum Mechanics

### The Big Picture

This week we learned how quantum mechanics makes **contact with experiment** through the theory of measurement.

**Central Theme:** Observables are represented by Hermitian operators, and measurements yield eigenvalues with probabilities given by the Born rule.

### Concept Map

```
                    OBSERVABLES
                         |
           +-------------+-------------+
           |                           |
    Hermitian Operators          Measurement
           |                           |
    +------+------+            +-------+-------+
    |             |            |               |
Eigenvalues  Eigenstates    Born Rule    State Collapse
    |             |            |               |
Possible      Complete    P(a) = |<a|psi>|^2   |
Outcomes       Basis                      Post-measurement
                                           state |a>

           COMPATIBLE OBSERVABLES
                    |
            [A, B] = 0
                    |
         Common Eigenstates
                    |
              +-----+-----+
              |           |
           CSCO    Simultaneous
                   Measurement

      POSITION AND MOMENTUM
              |
      [x, p] = ihbar
              |
    +----+----+----+
    |         |    |
Position  Momentum  Fourier
   x       -ihd/dx  Transform
   |         |         |
 psi(x)   phi(p)    psi <-> phi
```

### Key Results Summary

| Day | Topic | Key Result |
|-----|-------|------------|
| 344 | Measurement Postulate | Outcomes are eigenvalues; $P(a) = |\langle a|\psi\rangle|^2$ |
| 345 | State Collapse | After measuring $a$: $|\psi\rangle \to |a\rangle$ |
| 346 | Expectation Values | $\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle$ |
| 347 | Compatible Observables | $[\hat{A}, \hat{B}] = 0 \Leftrightarrow$ common eigenstates |
| 348 | Position & Momentum | $\hat{x} = x$, $\hat{p} = -i\hbar d/dx$, $[\hat{x}, \hat{p}] = i\hbar$ |
| 349 | Fourier Transform | $\phi(p) = \mathcal{F}[\psi(x)]$; uncertainty from Fourier |

---

## Master Formula Sheet

### Measurement Theory

| Concept | Formula |
|---------|---------|
| Measurement probability | $P(a) = |\langle a|\psi\rangle|^2$ |
| Degenerate case | $P(a) = \sum_i |\langle a,i|\psi\rangle|^2$ |
| Expectation value | $\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle = \sum_a a \cdot P(a)$ |
| Variance | $(\Delta A)^2 = \langle\hat{A}^2\rangle - \langle\hat{A}\rangle^2$ |
| Post-measurement state | $|\psi'\rangle = \frac{\hat{P}_a|\psi\rangle}{\sqrt{P(a)}}$ |

### Commutators

| Relation | Formula |
|----------|---------|
| Definition | $[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$ |
| Antisymmetry | $[\hat{A}, \hat{B}] = -[\hat{B}, \hat{A}]$ |
| Product rule | $[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]$ |
| Jacobi identity | $[\hat{A}, [\hat{B}, \hat{C}]] + \text{cyclic} = 0$ |
| Uncertainty | $\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$ |

### Position and Momentum

| Representation | Position | Momentum |
|----------------|----------|----------|
| Wave function | $\psi(x) = \langle x|\psi\rangle$ | $\phi(p) = \langle p|\psi\rangle$ |
| Position operator | $\hat{x} \to x$ | $\hat{x} \to i\hbar\frac{d}{dp}$ |
| Momentum operator | $\hat{p} \to -i\hbar\frac{d}{dx}$ | $\hat{p} \to p$ |
| Canonical commutation | $[\hat{x}, \hat{p}] = i\hbar$ | |

### Fourier Transform

| Direction | Formula |
|-----------|---------|
| Position to momentum | $\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx$ |
| Momentum to position | $\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{ipx/\hbar}\phi(p)dp$ |
| Parseval | $\int|\psi|^2 dx = \int|\phi|^2 dp$ |
| Shift theorem | $e^{ip_0x/\hbar}\psi(x) \leftrightarrow \phi(p-p_0)$ |

---

## Conceptual Review Questions

Answer these without looking at notes first, then verify:

1. **Why must observables be Hermitian?**
   - Real eigenvalues (physical measurement results)
   - Orthogonal eigenstates (distinguishable outcomes)
   - Complete basis (any state can be measured)

2. **What does $[\hat{A}, \hat{B}] = 0$ physically mean?**
   - Can measure $A$ and $B$ simultaneously
   - Share common eigenstates
   - Measurement order doesn't matter

3. **Why are position and momentum incompatible?**
   - $[\hat{x}, \hat{p}] = i\hbar \neq 0$
   - No simultaneous eigenstates
   - Uncertainty principle: $\Delta x \cdot \Delta p \geq \hbar/2$

4. **What is special about Gaussian wave packets?**
   - Minimum uncertainty states
   - Fourier transform of Gaussian is Gaussian
   - $\Delta x \cdot \Delta p = \hbar/2$ exactly

5. **How does measurement affect the quantum state?**
   - State collapses to eigenstate
   - Information about other observables may be lost
   - Repeated measurement gives same result

---

## Practice Exam (100 Points)

### Part A: Conceptual Questions (20 Points)

**Question A1 (5 points):** State the four postulates of quantum mechanics related to measurement (outcomes, probabilities, expectation values, collapse).

**Solution:**
1. Observable outcomes are eigenvalues of the corresponding Hermitian operator
2. Probability of outcome $a$ is $P(a) = |\langle a|\psi\rangle|^2$ (Born rule)
3. Expectation value is $\langle A\rangle = \langle\psi|\hat{A}|\psi\rangle$
4. After measurement of $a$, state collapses to $|a\rangle$

---

**Question A2 (5 points):** Explain why $\hat{x}$ and $\hat{p}$ cannot be simultaneously diagonalized.

**Solution:**
If they could be simultaneously diagonalized, there would exist a common eigenbasis $|x', p'\rangle$ where both operators are diagonal. But $[\hat{x}, \hat{p}] = i\hbar \neq 0$, which means they have no common eigenstates (except trivially). Alternatively, the commutator acting on any state gives $i\hbar|\psi\rangle \neq 0$ for normalized states, proving they don't commute.

---

**Question A3 (5 points):** What is a CSCO? Give an example for the hydrogen atom.

**Solution:**
A Complete Set of Commuting Observables (CSCO) is a minimal set of mutually commuting observables whose common eigenstates are non-degenerate (uniquely specified).

For hydrogen: $\{\hat{H}, \hat{L}^2, \hat{L}_z, \hat{S}_z\}$ is a CSCO.
States are uniquely labeled by quantum numbers $|n, \ell, m_\ell, m_s\rangle$.

---

**Question A4 (5 points):** Why does the Fourier transform relationship between $\psi(x)$ and $\phi(p)$ imply the uncertainty principle?

**Solution:**
The Fourier transform has a mathematical property: the product of the widths of a function and its Fourier transform is bounded below: $\Delta x \cdot \Delta k \geq 1/2$.

Since $p = \hbar k$, we have $\Delta p = \hbar\Delta k$, leading to:
$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

A narrow function in one domain must be broad in the conjugate domain.

---

### Part B: Calculation Problems (50 Points)

**Problem B1 (10 points):** A spin-1/2 particle is in state:
$$|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle + \sqrt{\frac{2}{3}}|-\rangle$$

(a) Find the probability of measuring $S_z = +\hbar/2$.
(b) Find $\langle S_z\rangle$.
(c) Find $\Delta S_z$.

**Solution:**

(a) $P(+\hbar/2) = |\langle +|\psi\rangle|^2 = |1/\sqrt{3}|^2 = 1/3$

(b) $\langle S_z\rangle = P(+)\cdot(+\hbar/2) + P(-)\cdot(-\hbar/2) = \frac{1}{3}\cdot\frac{\hbar}{2} + \frac{2}{3}\cdot(-\frac{\hbar}{2}) = -\frac{\hbar}{6}$

(c) $\langle S_z^2\rangle = P(+)\cdot(\hbar/2)^2 + P(-)\cdot(\hbar/2)^2 = \frac{\hbar^2}{4}$

$(\Delta S_z)^2 = \frac{\hbar^2}{4} - \frac{\hbar^2}{36} = \frac{9\hbar^2 - \hbar^2}{36} = \frac{8\hbar^2}{36} = \frac{2\hbar^2}{9}$

$\Delta S_z = \frac{\hbar\sqrt{2}}{3}$

---

**Problem B2 (10 points):** Calculate $[\hat{x}^2, \hat{p}^2]$.

**Solution:**

Using the commutator product rule repeatedly:

$[\hat{x}^2, \hat{p}^2] = [\hat{x}^2, \hat{p}]\hat{p} + \hat{p}[\hat{x}^2, \hat{p}]$

We know $[\hat{x}^2, \hat{p}] = 2i\hbar\hat{x}$ (from earlier):

$[\hat{x}^2, \hat{p}^2] = 2i\hbar\hat{x}\hat{p} + \hat{p}(2i\hbar\hat{x}) = 2i\hbar(\hat{x}\hat{p} + \hat{p}\hat{x})$

Note: $\hat{x}\hat{p} + \hat{p}\hat{x} = 2\hat{x}\hat{p} - [\hat{x},\hat{p}] = 2\hat{x}\hat{p} - i\hbar$

So: $[\hat{x}^2, \hat{p}^2] = 2i\hbar(2\hat{x}\hat{p} - i\hbar) = 4i\hbar\hat{x}\hat{p} + 2\hbar^2$

Or equivalently: $[\hat{x}^2, \hat{p}^2] = 2i\hbar(\hat{x}\hat{p} + \hat{p}\hat{x})$

---

**Problem B3 (10 points):** A particle has wave function:
$$\psi(x) = \begin{cases} A(a^2 - x^2) & |x| \leq a \\ 0 & |x| > a \end{cases}$$

(a) Find the normalization constant $A$.
(b) Calculate $\langle x^2\rangle$.

**Solution:**

(a) Normalization:
$$\int_{-a}^{a} A^2(a^2 - x^2)^2 dx = 1$$

Expand: $(a^2 - x^2)^2 = a^4 - 2a^2x^2 + x^4$

$$A^2\left[a^4 \cdot 2a - 2a^2 \cdot \frac{2a^3}{3} + \frac{2a^5}{5}\right] = A^2\left[2a^5 - \frac{4a^5}{3} + \frac{2a^5}{5}\right]$$

$$= A^2 \cdot 2a^5\left[1 - \frac{2}{3} + \frac{1}{5}\right] = A^2 \cdot 2a^5 \cdot \frac{15 - 10 + 3}{15} = A^2 \cdot \frac{16a^5}{15}$$

So $A = \sqrt{\frac{15}{16a^5}}$

(b) Calculate $\langle x^2\rangle$:
$$\langle x^2\rangle = A^2\int_{-a}^{a} x^2(a^2-x^2)^2 dx = A^2\int_{-a}^{a}(a^4x^2 - 2a^2x^4 + x^6)dx$$

$$= A^2\left[\frac{2a^7}{3} - \frac{4a^7}{5} + \frac{2a^7}{7}\right] = A^2 \cdot 2a^7\left[\frac{1}{3} - \frac{2}{5} + \frac{1}{7}\right]$$

$$= A^2 \cdot 2a^7 \cdot \frac{35 - 42 + 15}{105} = A^2 \cdot \frac{16a^7}{105}$$

$$\langle x^2\rangle = \frac{15}{16a^5} \cdot \frac{16a^7}{105} = \frac{a^2}{7}$$

---

**Problem B4 (10 points):** The momentum space wave function is:
$$\phi(p) = \frac{N}{(p^2 + p_0^2)}$$

(a) Find $N$ for normalization.
(b) Find $\langle p^2\rangle$.
(c) Discuss whether $\langle p^2\rangle$ is finite.

**Solution:**

(a) Normalization:
$$\int_{-\infty}^{\infty} \frac{N^2}{(p^2+p_0^2)^2}dp = 1$$

Using $\int_{-\infty}^{\infty}\frac{dp}{(p^2+a^2)^2} = \frac{\pi}{2a^3}$:

$$N^2 \cdot \frac{\pi}{2p_0^3} = 1 \implies N = \sqrt{\frac{2p_0^3}{\pi}}$$

(b) $\langle p^2\rangle = N^2\int_{-\infty}^{\infty}\frac{p^2 dp}{(p^2+p_0^2)^2}$

Using $\int_{-\infty}^{\infty}\frac{p^2 dp}{(p^2+a^2)^2} = \frac{\pi}{2a}$:

$$\langle p^2\rangle = \frac{2p_0^3}{\pi} \cdot \frac{\pi}{2p_0} = p_0^2$$

(c) $\langle p^2\rangle = p_0^2$ is finite. However, higher moments like $\langle p^4\rangle$ would diverge because the integrand decays too slowly at large $|p|$.

---

**Problem B5 (10 points):** Show that the operators $\hat{A} = \hat{p}^2$ and $\hat{B} = \hat{x}^2$ do NOT commute, but $[\hat{A}, \hat{B}]$ is proportional to a known operator combination.

**Solution:**

From Problem B2: $[\hat{x}^2, \hat{p}^2] = 2i\hbar(\hat{x}\hat{p} + \hat{p}\hat{x})$

So $[\hat{p}^2, \hat{x}^2] = -2i\hbar(\hat{x}\hat{p} + \hat{p}\hat{x})$

This is non-zero, so they don't commute.

The combination $\hat{x}\hat{p} + \hat{p}\hat{x}$ is related to the dilation operator $\hat{D} = \frac{1}{2}(\hat{x}\hat{p} + \hat{p}\hat{x})$, which generates scale transformations.

---

### Part C: Short Problems (30 Points)

**Problem C1 (6 points):** Two observables have matrix representations:
$$\hat{A} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}, \quad \hat{B} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

(a) Do $\hat{A}$ and $\hat{B}$ commute?
(b) Find the eigenvalues and eigenvectors of $\hat{B}$.

**Solution:**

(a) $\hat{A}\hat{B} = \begin{pmatrix} 2 & 2 \\ 1 & 1 \end{pmatrix}$, $\hat{B}\hat{A} = \begin{pmatrix} 3 & 1 \\ 3 & 1 \end{pmatrix}$

$[\hat{A}, \hat{B}] = \begin{pmatrix} -1 & 1 \\ -2 & 0 \end{pmatrix} \neq 0$

They do NOT commute.

(b) $\det(\hat{B} - \lambda I) = (1-\lambda)^2 - 1 = \lambda^2 - 2\lambda = \lambda(\lambda - 2) = 0$

Eigenvalues: $\lambda_1 = 0$, $\lambda_2 = 2$

For $\lambda = 0$: $|v_0\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$

For $\lambda = 2$: $|v_2\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

---

**Problem C2 (6 points):** Calculate $\langle p\rangle$ for the ground state of the harmonic oscillator:
$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/2\hbar}$$

**Solution:**

$$\langle p\rangle = \int_{-\infty}^{\infty}\psi_0^*(x)\left(-i\hbar\frac{d\psi_0}{dx}\right)dx$$

$$\frac{d\psi_0}{dx} = -\frac{m\omega x}{\hbar}\psi_0$$

$$\langle p\rangle = \int\psi_0^*\left(-i\hbar\right)\left(-\frac{m\omega x}{\hbar}\right)\psi_0 dx = im\omega\int x|\psi_0|^2 dx$$

The integrand is odd (x times even function), so:

$$\langle p\rangle = 0$$

---

**Problem C3 (6 points):** A system is in state $|\psi\rangle = \frac{1}{2}|a_1\rangle + \frac{\sqrt{3}}{2}|a_2\rangle$ where $|a_1\rangle, |a_2\rangle$ are eigenstates of $\hat{A}$ with eigenvalues $a_1 = 1$, $a_2 = 3$. Find:

(a) $P(a_1)$ and $P(a_2)$
(b) $\langle A\rangle$
(c) The state after measuring and obtaining $a_2$

**Solution:**

(a) $P(a_1) = |1/2|^2 = 1/4$, $P(a_2) = |\sqrt{3}/2|^2 = 3/4$

(b) $\langle A\rangle = \frac{1}{4}(1) + \frac{3}{4}(3) = \frac{1}{4} + \frac{9}{4} = \frac{10}{4} = 2.5$

(c) After measuring $a_2$, state collapses to $|a_2\rangle$

---

**Problem C4 (6 points):** Verify that $[\hat{L}_z, \hat{L}^2] = 0$ where $\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2$.

**Solution:**

$[\hat{L}_z, \hat{L}^2] = [\hat{L}_z, \hat{L}_x^2] + [\hat{L}_z, \hat{L}_y^2] + [\hat{L}_z, \hat{L}_z^2]$

The last term is 0 (any operator commutes with its powers).

For $[\hat{L}_z, \hat{L}_x^2]$:
Using $[\hat{A}, \hat{B}^2] = \hat{B}[\hat{A}, \hat{B}] + [\hat{A}, \hat{B}]\hat{B}$ and $[\hat{L}_z, \hat{L}_x] = i\hbar\hat{L}_y$:

$[\hat{L}_z, \hat{L}_x^2] = \hat{L}_x(i\hbar\hat{L}_y) + (i\hbar\hat{L}_y)\hat{L}_x = i\hbar(\hat{L}_x\hat{L}_y + \hat{L}_y\hat{L}_x)$

Similarly with $[\hat{L}_z, \hat{L}_y] = -i\hbar\hat{L}_x$:

$[\hat{L}_z, \hat{L}_y^2] = -i\hbar(\hat{L}_y\hat{L}_x + \hat{L}_x\hat{L}_y)$

Adding: $[\hat{L}_z, \hat{L}^2] = 0$ $\checkmark$

---

**Problem C5 (6 points):** The wave function $\psi(x) = Ae^{-(x-x_0)^2/2a^2}e^{ip_0x/\hbar}$ describes a Gaussian wave packet. Without detailed calculation, state:

(a) $\langle x\rangle$
(b) $\langle p\rangle$
(c) The uncertainty product $\Delta x \cdot \Delta p$

**Solution:**

(a) $\langle x\rangle = x_0$ (centered at $x_0$)

(b) $\langle p\rangle = p_0$ (the exponential factor gives mean momentum $p_0$)

(c) $\Delta x \cdot \Delta p = \hbar/2$ (Gaussian is minimum uncertainty; spatial shift and momentum boost don't change this)

---

## Comprehensive Qiskit Lab: Measurements and Born Rule

```python
"""
Day 350 Qiskit Lab: Measurements and Born Rule Verification
Comprehensive implementation of Week 50 concepts
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
import matplotlib.pyplot as plt

# Use Sampler for measurement simulation
sampler = Sampler()

print("=" * 60)
print("WEEK 50 QISKIT LAB: MEASUREMENTS AND BORN RULE")
print("=" * 60)

# ============================================
# Part 1: Born Rule Verification
# ============================================

print("\n" + "=" * 60)
print("Part 1: Born Rule Verification")
print("=" * 60)

def verify_born_rule(theta, phi=0, shots=10000):
    """
    Create state cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
    and verify P(0) = cos^2(theta/2), P(1) = sin^2(theta/2)
    """
    qc = QuantumCircuit(1, 1)

    # Create arbitrary state using U gate
    # U(theta, phi, lambda) creates cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
    qc.u(theta, phi, 0, 0)
    qc.measure(0, 0)

    # Run circuit
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.c.get_counts()

    # Extract probabilities
    p0_measured = counts.get('0', 0) / shots
    p1_measured = counts.get('1', 0) / shots

    # Theoretical predictions
    p0_theory = np.cos(theta/2)**2
    p1_theory = np.sin(theta/2)**2

    return p0_measured, p1_measured, p0_theory, p1_theory

# Test various states
theta_values = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]
results = []

print("\nVerifying Born Rule P(0) = cos^2(theta/2):")
print("-" * 50)
print(f"{'theta':^10} {'P(0) meas':^12} {'P(0) theory':^12} {'Error':^10}")
print("-" * 50)

for theta in theta_values:
    p0_m, p1_m, p0_t, p1_t = verify_born_rule(theta)
    error = abs(p0_m - p0_t)
    results.append((theta, p0_m, p0_t))
    print(f"{theta:^10.4f} {p0_m:^12.4f} {p0_t:^12.4f} {error:^10.4f}")

# Plot results
fig, ax = plt.subplots(figsize=(8, 5))
thetas = [r[0] for r in results]
p0_measured = [r[1] for r in results]
p0_theory = [r[2] for r in results]

theta_fine = np.linspace(0, np.pi, 100)
ax.plot(theta_fine, np.cos(theta_fine/2)**2, 'b-', linewidth=2, label='Theory: cos^2(theta/2)')
ax.scatter(thetas, p0_measured, color='red', s=100, zorder=5, label='Measured')
ax.set_xlabel('theta (radians)')
ax.set_ylabel('P(0)')
ax.set_title('Born Rule Verification')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('born_rule_verification.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 2: Measurement in Different Bases
# ============================================

print("\n" + "=" * 60)
print("Part 2: Measurement in Different Bases")
print("=" * 60)

def measure_in_basis(state_prep_circuit, basis='Z', shots=10000):
    """
    Measure a prepared state in the specified basis.
    basis: 'Z' (computational), 'X', or 'Y'
    """
    qc = state_prep_circuit.copy()

    # Add basis rotation before measurement
    if basis == 'X':
        qc.h(0)  # H rotates X basis to Z basis
    elif basis == 'Y':
        qc.sdg(0)  # S^dag then H rotates Y basis to Z basis
        qc.h(0)
    # Z basis: no rotation needed

    qc.measure_all()

    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    return counts

# Prepare |+> state
qc_plus = QuantumCircuit(1)
qc_plus.h(0)

print("\nState: |+> = (|0> + |1>)/sqrt(2)")
print("-" * 40)

for basis in ['Z', 'X', 'Y']:
    counts = measure_in_basis(qc_plus, basis)
    print(f"{basis}-basis measurement: {counts}")

print("\nExpected:")
print("Z-basis: 50% |0>, 50% |1>")
print("X-basis: 100% |+> (appears as |0>)")
print("Y-basis: 50% |+i>, 50% |-i>")

# Prepare |i> = (|0> + i|1>)/sqrt(2)
qc_i = QuantumCircuit(1)
qc_i.h(0)
qc_i.s(0)  # Adds i phase to |1>

print("\nState: |+i> = (|0> + i|1>)/sqrt(2)")
print("-" * 40)

for basis in ['Z', 'X', 'Y']:
    counts = measure_in_basis(qc_i, basis)
    print(f"{basis}-basis measurement: {counts}")

# ============================================
# Part 3: State Collapse Demonstration
# ============================================

print("\n" + "=" * 60)
print("Part 3: State Collapse Demonstration")
print("=" * 60)

def demonstrate_collapse(shots=10000):
    """
    Show that after measuring in Z, subsequent Z measurement gives same result.
    But measuring in X after Z gives random result.
    """
    # Circuit 1: Measure Z twice
    qc1 = QuantumCircuit(1, 2)
    qc1.h(0)  # Create |+>
    qc1.measure(0, 0)  # First Z measurement
    qc1.measure(0, 1)  # Second Z measurement

    # Circuit 2: Measure Z, then X
    qc2 = QuantumCircuit(1, 2)
    qc2.h(0)  # Create |+>
    qc2.measure(0, 0)  # First Z measurement
    qc2.h(0)  # Rotate to X basis
    qc2.measure(0, 1)  # X measurement (in computational basis after H)

    job1 = sampler.run([qc1], shots=shots)
    job2 = sampler.run([qc2], shots=shots)

    result1 = job1.result()[0].data.c.get_counts()
    result2 = job2.result()[0].data.c.get_counts()

    return result1, result2

r1, r2 = demonstrate_collapse()

print("\nStarting state: |+>")
print("\nExperiment 1: Z measurement followed by Z measurement")
print(f"Results (format: 'second_first'): {r1}")
print("Expected: Only '00' and '11' (measurements agree)")

print("\nExperiment 2: Z measurement followed by X measurement")
print(f"Results (format: 'X_result Z_result'): {r2}")
print("Expected: All four outcomes (Z collapse randomizes X)")

# ============================================
# Part 4: Expectation Value Calculation
# ============================================

print("\n" + "=" * 60)
print("Part 4: Expectation Value Calculation")
print("=" * 60)

def calculate_expectation(state_circuit, observable='Z', shots=10000):
    """
    Calculate expectation value of Pauli observable.
    <Z> = P(0) - P(1)
    <X> = P(+) - P(-) (measure in X basis)
    <Y> = P(+i) - P(-i) (measure in Y basis)
    """
    qc = state_circuit.copy()

    if observable == 'X':
        qc.h(0)
    elif observable == 'Y':
        qc.sdg(0)
        qc.h(0)

    qc.measure_all()

    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    p0 = counts.get('0', 0) / shots
    p1 = counts.get('1', 0) / shots

    return p0 - p1  # <observable>

# Test state: cos(theta/2)|0> + sin(theta/2)|1> at theta = pi/3
theta = np.pi / 3
qc_test = QuantumCircuit(1)
qc_test.ry(theta, 0)  # RY(theta) creates cos(theta/2)|0> + sin(theta/2)|1>

# Calculate all Pauli expectations
exp_X = calculate_expectation(qc_test, 'X')
exp_Y = calculate_expectation(qc_test, 'Y')
exp_Z = calculate_expectation(qc_test, 'Z')

print(f"\nState: cos({theta/2:.4f})|0> + sin({theta/2:.4f})|1>")
print("-" * 40)
print(f"<X> = {exp_X:.4f} (theory: {np.sin(theta):.4f})")
print(f"<Y> = {exp_Y:.4f} (theory: 0)")
print(f"<Z> = {exp_Z:.4f} (theory: {np.cos(theta):.4f})")

# Verify: <X>^2 + <Y>^2 + <Z>^2 <= 1 (equality for pure state on Bloch sphere)
bloch_radius = np.sqrt(exp_X**2 + exp_Y**2 + exp_Z**2)
print(f"\nBloch sphere radius: sqrt(<X>^2 + <Y>^2 + <Z>^2) = {bloch_radius:.4f}")
print("(Should be 1 for pure state)")

# ============================================
# Part 5: Compatible vs Incompatible Observables
# ============================================

print("\n" + "=" * 60)
print("Part 5: Compatible vs Incompatible Observables")
print("=" * 60)

def measure_two_qubits(obs1, obs2, shots=10000):
    """
    Measure two commuting single-qubit observables on different qubits.
    (Always compatible since they're on different qubits)
    """
    qc = QuantumCircuit(2, 2)

    # Prepare entangled state |00> + |11>
    qc.h(0)
    qc.cx(0, 1)

    # Rotate for measurement basis
    if obs1 == 'X':
        qc.h(0)
    elif obs1 == 'Y':
        qc.sdg(0)
        qc.h(0)

    if obs2 == 'X':
        qc.h(1)
    elif obs2 == 'Y':
        qc.sdg(1)
        qc.h(1)

    qc.measure([0, 1], [0, 1])

    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.c.get_counts()

    return counts

print("\nBell state |Phi+> = (|00> + |11>)/sqrt(2)")
print("-" * 40)

# ZZ measurement (always same)
counts_ZZ = measure_two_qubits('Z', 'Z')
print(f"Z1-Z2 measurement: {counts_ZZ}")
print("Expected: Perfect correlation (00 and 11 only)")

# XX measurement
counts_XX = measure_two_qubits('X', 'X')
print(f"\nX1-X2 measurement: {counts_XX}")
print("Expected: Perfect correlation (00 and 11 only)")

# XZ measurement (different bases)
counts_XZ = measure_two_qubits('X', 'Z')
print(f"\nX1-Z2 measurement: {counts_XZ}")
print("Expected: No correlation (all outcomes equally likely)")

# ============================================
# Part 6: Multi-Qubit Observable Measurement
# ============================================

print("\n" + "=" * 60)
print("Part 6: Multi-Qubit Observable (ZZ Parity)")
print("=" * 60)

def measure_ZZ_parity(shots=10000):
    """
    Measure the ZZ operator (parity) on a two-qubit state.
    ZZ eigenvalue +1: even parity (|00>, |11>)
    ZZ eigenvalue -1: odd parity (|01>, |10>)
    """
    # Create Bell state
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.c.get_counts()

    # Calculate ZZ expectation
    parity_plus = (counts.get('00', 0) + counts.get('11', 0)) / shots
    parity_minus = (counts.get('01', 0) + counts.get('10', 0)) / shots

    ZZ_expectation = parity_plus - parity_minus

    return counts, ZZ_expectation

counts, ZZ_exp = measure_ZZ_parity()
print(f"\nBell state measurement results: {counts}")
print(f"<ZZ> = P(even) - P(odd) = {ZZ_exp:.4f}")
print("Expected: <ZZ> = 1 (Bell state has even parity)")

# ============================================
# Part 7: Measurement Statistics Analysis
# ============================================

print("\n" + "=" * 60)
print("Part 7: Statistical Analysis of Measurements")
print("=" * 60)

def statistical_analysis(theta, trials=50, shots_per_trial=1000):
    """
    Run multiple measurement trials and analyze statistics.
    """
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    p0_values = []

    for _ in range(trials):
        job = sampler.run([qc], shots=shots_per_trial)
        result = job.result()
        counts = result[0].data.c.get_counts()
        p0 = counts.get('0', 0) / shots_per_trial
        p0_values.append(p0)

    p0_mean = np.mean(p0_values)
    p0_std = np.std(p0_values)
    p0_theory = np.cos(theta/2)**2

    # Expected standard deviation for binomial
    expected_std = np.sqrt(p0_theory * (1 - p0_theory) / shots_per_trial)

    return p0_values, p0_mean, p0_std, p0_theory, expected_std

theta = np.pi / 3
p0_vals, mean, std, theory, exp_std = statistical_analysis(theta)

print(f"\nState: RY({theta:.4f})|0>")
print(f"Trials: 50, Shots per trial: 1000")
print("-" * 40)
print(f"Measured P(0): mean = {mean:.4f}, std = {std:.4f}")
print(f"Theory: P(0) = {theory:.4f}, expected std = {exp_std:.4f}")

# Histogram of measured probabilities
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(p0_vals, bins=15, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(x=theory, color='red', linewidth=2, linestyle='--', label=f'Theory: {theory:.4f}')
ax.axvline(x=mean, color='green', linewidth=2, label=f'Measured mean: {mean:.4f}')
ax.set_xlabel('P(0)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Measured P(0) over 50 Trials')
ax.legend()
plt.savefig('measurement_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 8: Complete Measurement Demonstration
# ============================================

print("\n" + "=" * 60)
print("Part 8: Complete Measurement Workflow")
print("=" * 60)

def complete_measurement_demo():
    """
    Demonstrate complete measurement workflow:
    1. Prepare state
    2. Measure observable
    3. Analyze results
    4. Verify with Statevector
    """
    # Prepare a general state
    qc_prep = QuantumCircuit(2)
    qc_prep.h(0)
    qc_prep.cx(0, 1)
    qc_prep.rz(np.pi/4, 0)
    qc_prep.ry(np.pi/3, 1)

    # Get exact statevector
    sv = Statevector(qc_prep)

    print("Prepared 2-qubit state:")
    print(f"Statevector: {np.round(sv.data, 4)}")

    # Theoretical probabilities
    probs = sv.probabilities()
    print(f"\nTheoretical probabilities:")
    for i, p in enumerate(probs):
        print(f"  |{i:02b}>: {p:.4f}")

    # Measure
    qc_meas = qc_prep.copy()
    qc_meas.measure_all()

    job = sampler.run([qc_meas], shots=10000)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    print(f"\nMeasured counts (10000 shots):")
    for outcome, count in sorted(counts.items()):
        print(f"  |{outcome}>: {count} ({count/10000:.4f})")

    # Compare
    print("\nComparison (Theory vs Measured):")
    for i in range(4):
        bitstring = f'{i:02b}'
        rev_bitstring = bitstring[::-1]  # Qiskit uses little-endian
        theory_p = probs[i]
        meas_p = counts.get(rev_bitstring, 0) / 10000
        print(f"  |{bitstring}>: Theory={theory_p:.4f}, Measured={meas_p:.4f}, Error={abs(theory_p-meas_p):.4f}")

complete_measurement_demo()

print("\n" + "=" * 60)
print("QISKIT LAB COMPLETE!")
print("=" * 60)

# Summary of key concepts demonstrated
print("""
Summary of Demonstrated Concepts:
1. Born Rule: P(outcome) = |<outcome|state>|^2
2. Basis Rotation: Change measurement basis with H, S gates
3. State Collapse: Repeated measurements give consistent results
4. Expectation Values: <O> = sum of eigenvalues weighted by probabilities
5. Multi-qubit: ZZ parity measurement on Bell state
6. Statistics: Measurement outcomes follow predicted distributions
7. Workflow: State preparation -> Measurement -> Verification
""")
```

---

## Summary

### Week 50 Key Achievements

By completing this week, you have mastered:

1. **Measurement Postulate:** Understanding how quantum mechanics predicts experimental outcomes through eigenvalues and the Born rule

2. **State Collapse:** How measurement fundamentally changes the quantum state

3. **Expectation Values:** Calculating average outcomes and uncertainties

4. **Compatible Observables:** The commutator criterion $[\hat{A}, \hat{B}] = 0$ for simultaneous measurement

5. **Position-Momentum Duality:** The fundamental pair of conjugate variables and their representations

6. **Fourier Connection:** How wave functions transform between position and momentum space

### Connections to Quantum Computing

| QM Concept | Qiskit Implementation |
|------------|----------------------|
| Born rule | Shot statistics |
| Basis rotation | H, S, and general U gates |
| Expectation value | Weighted sum of outcomes |
| Compatible observables | Simultaneous measurements |
| State collapse | Mid-circuit measurement |

---

## Daily Checklist

- [ ] Complete conceptual review questions without notes
- [ ] Take practice exam (target: >80 points)
- [ ] Review any weak areas identified
- [ ] Complete Qiskit lab exercises
- [ ] Verify Born rule experimentally
- [ ] Implement measurements in different bases
- [ ] Run statistical analysis of measurements
- [ ] Summarize Week 50 in study journal
- [ ] Prepare questions for Week 51

---

## Preview: Week 51 - The Uncertainty Principle

Next week explores the profound consequences of incompatible observables:

**Topics:**
- Heisenberg uncertainty relation derivation
- Generalized uncertainty principle
- Energy-time uncertainty
- Minimum uncertainty states
- Squeeze states and quantum sensing
- Implications for quantum computing

**Key Equation:**
$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

**Preparation:** Review Week 50 commutator calculations thoroughly.

---

## End of Week Reflection

Take 15 minutes to reflect on Week 50:

1. **Most challenging concept:** What did you struggle with most? How did you overcome it?

2. **Key insight:** What was your "aha" moment this week?

3. **Remaining questions:** What would you like to understand better?

4. **Applications:** How do these concepts connect to quantum computing applications?

5. **Next steps:** What specific areas need more practice?

---

**Congratulations on completing Week 50!**

You now understand the physical content of quantum mechanics - how the mathematical formalism connects to laboratory measurements. This is the bridge between abstract theory and experimental reality.

---

**References:**
- Shankar, R. (1994). Principles of Quantum Mechanics, Chapter 4
- Sakurai, J.J. (2017). Modern Quantum Mechanics, Chapter 1
- Griffiths, D.J. (2018). Introduction to Quantum Mechanics, Chapter 3
- Nielsen, M.A. & Chuang, I.L. (2010). Quantum Computation and Quantum Information, Chapter 2
- Qiskit Documentation: https://qiskit.org/documentation/
