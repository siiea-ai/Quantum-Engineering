# Day 251: Stone's Theorem and Unitary Groups

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | One-Parameter Unitary Groups |
| **Morning II** | 2 hours | Stone's Theorem and Its Proof |
| **Afternoon** | 2 hours | The Schrödinger Equation |
| **Evening** | 2 hours | Computational Lab: Time Evolution |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** strongly continuous one-parameter unitary groups
2. **State and prove** Stone's theorem (both directions)
3. **Construct** the generator of a unitary group
4. **Derive** the Schrödinger equation from Stone's theorem
5. **Apply** Stone's theorem to quantum mechanical systems
6. **Compute** time evolution for specific Hamiltonians

## Core Content

### 1. One-Parameter Groups

**Definition (One-Parameter Group)**

A **one-parameter group** of operators is a family $\{T(t)\}_{t \in \mathbb{R}}$ satisfying:

1. **Group property**: $T(t)T(s) = T(t + s)$ for all $t, s \in \mathbb{R}$
2. **Identity**: $T(0) = I$

**Consequence**: $T(t)^{-1} = T(-t)$.

**Definition (Strongly Continuous)**

$\{T(t)\}$ is **strongly continuous** if for each $x \in \mathcal{H}$:
$$\lim_{t \to 0} T(t)x = x$$
(convergence in norm).

**Definition (One-Parameter Unitary Group)**

$\{U(t)\}_{t \in \mathbb{R}}$ is a **strongly continuous one-parameter unitary group** if:
1. Each $U(t)$ is unitary: $U(t)^*U(t) = U(t)U(t)^* = I$
2. $U(t)U(s) = U(t+s)$
3. $U(0) = I$
4. $t \mapsto U(t)x$ is continuous for each $x$

### 2. The Generator

**Definition (Infinitesimal Generator)**

The **generator** of a strongly continuous unitary group $\{U(t)\}$ is:
$$\boxed{A = i\lim_{t \to 0} \frac{U(t) - I}{t}}$$

with domain:
$$D(A) = \left\{x \in \mathcal{H} : \lim_{t \to 0} \frac{U(t)x - x}{t} \text{ exists}\right\}$$

**Remark**: The factor of $i$ is a convention ensuring $A$ is self-adjoint (not skew-adjoint).

**Alternative**: Some authors define $A' = -i\frac{d}{dt}U(t)|_{t=0}$ without the $i$ factor.

**Lemma (Properties of Generator)**

1. $D(A)$ is dense in $\mathcal{H}$
2. $A$ is closed
3. $U(t)D(A) \subseteq D(A)$ for all $t$
4. $\frac{d}{dt}U(t)x = -iAU(t)x = U(t)(-iAx)$ for $x \in D(A)$

### 3. Stone's Theorem

**Theorem (Stone's Theorem)**

There is a one-to-one correspondence:

$$\boxed{\text{Strongly continuous unitary groups } \{U(t)\} \quad \longleftrightarrow \quad \text{Self-adjoint operators } A}$$

given by:
$$\boxed{U(t) = e^{-iAt}}$$

**Part 1**: If $A$ is self-adjoint, then $U(t) = e^{-iAt}$ (defined via spectral calculus) is a strongly continuous unitary group with generator $A$.

**Part 2**: If $\{U(t)\}$ is a strongly continuous unitary group, then its generator $A$ is self-adjoint, and $U(t) = e^{-iAt}$.

### 4. Proof of Stone's Theorem

**Proof of Part 1** (Self-adjoint → Unitary Group):

Let $A$ be self-adjoint with spectral measure $E$. Define:
$$U(t) = e^{-iAt} = \int_{-\infty}^\infty e^{-i\lambda t} \, dE_\lambda$$

**Unitary**: For each $t$:
$$U(t)^*U(t) = \int |e^{-i\lambda t}|^2 \, dE_\lambda = \int 1 \, dE_\lambda = I$$

**Group property**:
$$U(t)U(s) = \int e^{-i\lambda t} \, dE_\lambda \int e^{-i\lambda s} \, dE_\lambda = \int e^{-i\lambda(t+s)} \, dE_\lambda = U(t+s)$$

**Strong continuity**: For $x \in \mathcal{H}$:
$$\|U(t)x - x\|^2 = \int |e^{-i\lambda t} - 1|^2 \, d\|E_\lambda x\|^2 \to 0$$
as $t \to 0$ by dominated convergence.

**Generator is $A$**: For $x \in D(A)$:
$$\frac{U(t)x - x}{t} = \int \frac{e^{-i\lambda t} - 1}{t} \, dE_\lambda x \to \int (-i\lambda) \, dE_\lambda x = -iAx$$

$\square$

**Proof of Part 2** (Unitary Group → Self-adjoint):

This direction is more subtle and uses the spectral theorem.

**Step 1**: Define the generator $A$ as above.

**Step 2**: Show $A$ is symmetric.

For $x, y \in D(A)$:
$$\langle Ax, y \rangle = \lim_{t \to 0} \left\langle i\frac{U(t)x - x}{t}, y \right\rangle$$
$$= \lim_{t \to 0} i\frac{\langle U(t)x, y \rangle - \langle x, y \rangle}{t}$$
$$= \lim_{t \to 0} i\frac{\langle x, U(-t)y \rangle - \langle x, y \rangle}{t}$$
$$= \lim_{t \to 0} \left\langle x, i\frac{U(-t)y - y}{t} \right\rangle = \langle x, Ay \rangle$$

**Step 3**: Show $A$ is self-adjoint using von Neumann's criterion.

Need: $\text{Range}(A \pm iI)$ is dense.

For any $y \perp \text{Range}(A + iI)$:
$$0 = \langle (A + iI)x, y \rangle = \frac{d}{dt}\langle U(t)x, y \rangle\big|_{t=0} - \langle x, y \rangle$$

This implies $\langle U(t)x, y \rangle = e^{-t}\langle x, y \rangle$ for all $x \in D(A)$.

Since $D(A)$ is dense and $|e^{-t}| = e^{-t}$, we need $\langle x, y \rangle = 0$ for all $x$, so $y = 0$.

Thus $\text{Range}(A + iI)$ is dense. Similarly for $A - iI$.

**Step 4**: Verify $U(t) = e^{-iAt}$.

Both sides satisfy the same differential equation and initial condition. $\square$

### 5. The Schrödinger Equation

**From Stone's Theorem to Quantum Mechanics**

Stone's theorem explains the mathematical structure of quantum dynamics.

**Postulate (Time Evolution)**

The state $|\psi(t)\rangle$ of a quantum system evolves according to:
$$\boxed{|\psi(t)\rangle = U(t)|\psi(0)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle}$$

where $H$ is the Hamiltonian (self-adjoint by Stone's theorem).

**Derivation of Schrödinger Equation**

Differentiate $|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle$:
$$\frac{d}{dt}|\psi(t)\rangle = -\frac{iH}{\hbar}e^{-iHt/\hbar}|\psi(0)\rangle = -\frac{iH}{\hbar}|\psi(t)\rangle$$

$$\boxed{i\hbar\frac{d}{dt}|\psi(t)\rangle = H|\psi(t)\rangle}$$

This is the **time-dependent Schrödinger equation**.

**Converse**: Given the Schrödinger equation, Stone's theorem guarantees $H$ must be self-adjoint for $U(t) = e^{-iHt/\hbar}$ to be unitary.

### 6. Properties of Unitary Evolution

**Conservation of Probability**

$$\langle\psi(t)|\psi(t)\rangle = \langle\psi(0)|U(t)^*U(t)|\psi(0)\rangle = \langle\psi(0)|\psi(0)\rangle = 1$$

Normalization is preserved—probability is conserved.

**Heisenberg Picture**

Alternatively, keep states fixed and evolve operators:
$$A_H(t) = U(t)^* A U(t) = e^{iHt/\hbar} A e^{-iHt/\hbar}$$

**Heisenberg Equation of Motion**:
$$\boxed{\frac{d}{dt}A_H(t) = \frac{i}{\hbar}[H, A_H(t)]}$$

**Energy Conservation**

Since $[H, H] = 0$:
$$\frac{d}{dt}H_H(t) = 0$$

Energy (Hamiltonian) is conserved in time.

**Expectation Value Evolution**

$$\frac{d}{dt}\langle A \rangle_{\psi(t)} = \frac{i}{\hbar}\langle [H, A] \rangle + \left\langle \frac{\partial A}{\partial t} \right\rangle$$

### 7. Examples of Unitary Groups

**Example 1: Free Particle**

Hamiltonian: $H = \frac{p^2}{2m} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}$

Evolution operator in momentum space:
$$\tilde{U}(t)\tilde{\psi}(p) = e^{-ip^2t/(2m\hbar)}\tilde{\psi}(p)$$

In position space (propagator):
$$\psi(x, t) = \int_{-\infty}^\infty K(x, x', t)\psi(x', 0)\, dx'$$
$$K(x, x', t) = \sqrt{\frac{m}{2\pi i\hbar t}}\exp\left(\frac{im(x-x')^2}{2\hbar t}\right)$$

**Example 2: Harmonic Oscillator**

Hamiltonian: $H = \hbar\omega(a^\dagger a + \frac{1}{2})$

Number states evolve by phases:
$$|n\rangle \to e^{-i\omega(n+1/2)t}|n\rangle$$

Coherent states rotate in phase space:
$$|\alpha\rangle \to |e^{-i\omega t}\alpha\rangle$$

**Example 3: Spin in Magnetic Field**

Hamiltonian: $H = -\gamma \vec{B} \cdot \vec{S}$

For $\vec{B} = B\hat{z}$: $H = -\gamma B S_z = \frac{\omega}{2}\sigma_z$

$$U(t) = e^{-i\omega t \sigma_z/2} = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

This is Larmor precession.

### 8. Applications to Quantum Information

**Quantum Gates as Unitary Evolution**

Single-qubit gates are elements of $U(2)$:

| Gate | Unitary | Generator |
|------|---------|-----------|
| Pauli-X | $\sigma_x$ | $\sigma_x$ (Hermitian) |
| Phase | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$ | $\frac{\phi}{2}(I - \sigma_z)$ |
| Hadamard | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | $\frac{\pi}{2\sqrt{2}}(\sigma_x + \sigma_z)$ |

**Quantum Computing Principle**

Any unitary operation can be written as $U = e^{-iHt}$ for some Hermitian $H$. This is realized by:
- Laser pulses (atomic qubits)
- Microwave pulses (superconducting qubits)
- Gate voltages (spin qubits)

---

## Worked Examples

### Example 1: Generator of Translation Group

**Problem**: On $L^2(\mathbb{R})$, the translation operators $(T_a\psi)(x) = \psi(x - a)$ form a unitary group. Find the generator.

**Solution**:

**Step 1**: Verify group property.

$(T_a T_b \psi)(x) = T_a[\psi(x-b)] = \psi(x-a-b) = (T_{a+b}\psi)(x)$

$(T_a)^* = T_{-a}$ (since translation preserves $L^2$ norm). ✓

**Step 2**: Compute the generator.

For smooth $\psi$ with compact support:
$$\frac{T_a\psi - \psi}{a} = \frac{\psi(x-a) - \psi(x)}{a} \to -\psi'(x) = -\frac{d\psi}{dx}$$

So the generator is $A = -i \cdot \frac{d}{dx} = \frac{1}{i}\frac{d}{dx}$.

**Step 3**: Relate to momentum.

In units with $\hbar = 1$:
$$\boxed{A = \hat{p} = -i\frac{d}{dx}}$$

**Conclusion**: The momentum operator generates spatial translations:
$$T_a = e^{-i\hat{p}a} = e^{a\frac{d}{dx}}$$

This is the mathematical content of "momentum is the generator of translations."

### Example 2: Time Evolution of Gaussian Wave Packet

**Problem**: A free particle starts in a Gaussian state $\psi(x, 0) = (\pi\sigma^2)^{-1/4}e^{-x^2/(2\sigma^2)}$. Find $\psi(x, t)$.

**Solution**:

**Step 1**: Transform to momentum space.

$$\tilde{\psi}(p, 0) = \int_{-\infty}^\infty \psi(x, 0) e^{-ipx/\hbar} dx = \left(\frac{\sigma^2}{\pi\hbar^2}\right)^{1/4} e^{-\sigma^2 p^2/(2\hbar^2)}$$

**Step 2**: Apply time evolution in momentum space.

Free particle: $H = p^2/(2m)$

$$\tilde{\psi}(p, t) = e^{-ip^2 t/(2m\hbar)}\tilde{\psi}(p, 0)$$

**Step 3**: Transform back to position space.

$$\psi(x, t) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^\infty \tilde{\psi}(p, t) e^{ipx/\hbar} dp$$

After Gaussian integration:

$$\boxed{\psi(x, t) = \frac{1}{(\pi\sigma_t^2)^{1/4}} \exp\left(-\frac{x^2}{2\sigma_t^2}\right)}$$

where the time-dependent width is:
$$\sigma_t^2 = \sigma^2 + \frac{i\hbar t}{m}$$

**Step 4**: Physical interpretation.

The probability density $|\psi(x,t)|^2$ spreads:
$$|\psi(x,t)|^2 = \frac{1}{\sqrt{\pi}|\sigma_t|}\exp\left(-\frac{x^2}{|\sigma_t|^2}\right)$$

Width: $|\sigma_t| = \sigma\sqrt{1 + \hbar^2 t^2/(m^2\sigma^4)}$

The wave packet spreads due to dispersion (different momenta travel at different speeds).

### Example 3: Rabi Oscillations

**Problem**: A two-level atom interacts with a resonant laser. The Hamiltonian in the rotating frame is:
$$H = \frac{\hbar\Omega}{2}\sigma_x$$
where $\Omega$ is the Rabi frequency. If the atom starts in $|g\rangle$ (ground state), find $|\psi(t)\rangle$.

**Solution**:

**Step 1**: Find the evolution operator.

$$U(t) = e^{-iHt/\hbar} = e^{-i\Omega t \sigma_x/2}$$

Using $e^{i\theta \sigma_x} = \cos\theta \cdot I + i\sin\theta \cdot \sigma_x$:

$$U(t) = \cos\left(\frac{\Omega t}{2}\right)I - i\sin\left(\frac{\Omega t}{2}\right)\sigma_x$$

$$= \begin{pmatrix} \cos(\Omega t/2) & -i\sin(\Omega t/2) \\ -i\sin(\Omega t/2) & \cos(\Omega t/2) \end{pmatrix}$$

**Step 2**: Apply to initial state.

$$|\psi(t)\rangle = U(t)|g\rangle = \cos\left(\frac{\Omega t}{2}\right)|g\rangle - i\sin\left(\frac{\Omega t}{2}\right)|e\rangle$$

**Step 3**: Compute populations.

$$P_g(t) = |\langle g|\psi(t)\rangle|^2 = \cos^2\left(\frac{\Omega t}{2}\right) = \frac{1 + \cos(\Omega t)}{2}$$

$$P_e(t) = |\langle e|\psi(t)\rangle|^2 = \sin^2\left(\frac{\Omega t}{2}\right) = \frac{1 - \cos(\Omega t)}{2}$$

**Conclusion**: The population oscillates between ground and excited states with frequency $\Omega$ — these are **Rabi oscillations**.

At $t = \pi/\Omega$ (a "$\pi$-pulse"), the atom completely inverts: $|g\rangle \to -i|e\rangle$.

---

## Practice Problems

### Level 1: Direct Application

1. **Translation Generator**: Show that $(e^{a\frac{d}{dx}}f)(x) = f(x + a)$ using the Taylor series.

2. **Rotation Group**: The rotation operators $R_z(\theta) = e^{-i\theta J_z}$ form a group. What is the generator?

3. **Phase Evolution**: If $H = E_0|0\rangle\langle 0| + E_1|1\rangle\langle 1|$, write $U(t)$ explicitly.

### Level 2: Intermediate

4. **Momentum Boost**: Show that $e^{ip_0\hat{x}/\hbar}$ shifts momentum: $e^{ip_0\hat{x}/\hbar}\psi(x) = e^{ip_0 x/\hbar}\psi(x)$.

5. **Unitary Equivalence**: If $U$ is unitary and $H' = UHU^*$, show that $e^{-iH't} = Ue^{-iHt}U^*$.

6. **Two-Level Dynamics**: For $H = \epsilon\sigma_z + \delta\sigma_x$, find the eigenvalues and $U(t)$.

### Level 3: Challenging

7. **Baker-Campbell-Hausdorff**: For $[A, B] = cI$ (a scalar), prove:
$$e^A e^B = e^{A+B+c/2}$$

8. **Coherent State Evolution**: Show that the coherent state $|\alpha\rangle$ of the harmonic oscillator evolves as $|\alpha(t)\rangle = |e^{-i\omega t}\alpha\rangle$.

9. **Adiabatic Theorem**: State the adiabatic theorem and explain its connection to Berry phase.

---

## Computational Lab: Time Evolution

```python
"""
Day 251 Computational Lab: Stone's Theorem and Unitary Evolution
Exploring time evolution and the Schrödinger equation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from typing import Tuple, Callable

np.random.seed(42)

# =============================================================================
# Part 1: Verifying Stone's Theorem for Finite Dimensions
# =============================================================================

print("="*70)
print("PART 1: STONE'S THEOREM VERIFICATION")
print("="*70)

def verify_stones_theorem(H: np.ndarray, t_values: np.ndarray) -> dict:
    """
    Verify that U(t) = exp(-iHt) forms a unitary group.
    """
    results = {
        'is_hermitian': np.allclose(H, H.conj().T),
        'unitary_check': [],
        'group_check': []
    }

    # Check H is Hermitian (self-adjoint)
    print(f"H is Hermitian: {results['is_hermitian']}")

    # Generate U(t) for various t
    U_list = []
    for t in t_values:
        U_t = linalg.expm(-1j * H * t)
        U_list.append(U_t)

        # Check unitarity
        is_unitary = np.allclose(U_t @ U_t.conj().T, np.eye(H.shape[0]))
        results['unitary_check'].append(is_unitary)

    # Check group property U(s)U(t) = U(s+t)
    for i, s in enumerate(t_values[:-1]):
        for j, t in enumerate(t_values[:-1]):
            if s + t <= t_values[-1]:
                U_s = linalg.expm(-1j * H * s)
                U_t = linalg.expm(-1j * H * t)
                U_sum = linalg.expm(-1j * H * (s + t))
                is_group = np.allclose(U_s @ U_t, U_sum)
                results['group_check'].append(is_group)

    print(f"All U(t) unitary: {all(results['unitary_check'])}")
    print(f"Group property satisfied: {all(results['group_check'])}")

    return results

# Test with a random Hermitian matrix
n = 4
H_random = np.random.randn(n, n) + 1j * np.random.randn(n, n)
H_random = (H_random + H_random.conj().T) / 2

t_values = np.linspace(0, 5, 20)
verify_stones_theorem(H_random, t_values)

# =============================================================================
# Part 2: Schrödinger Equation - Two-Level System
# =============================================================================

print("\n" + "="*70)
print("PART 2: SCHRÖDINGER EQUATION - TWO-LEVEL SYSTEM")
print("="*70)

def two_level_evolution(omega: float, t_max: float, n_steps: int) -> Tuple:
    """
    Solve Schrödinger equation for H = (omega/2) * sigma_z.
    """
    H = (omega / 2) * np.array([[1, 0], [0, -1]], dtype=complex)

    # Initial state: |+> = (|0> + |1>)/sqrt(2)
    psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)

    times = np.linspace(0, t_max, n_steps)
    states = []

    for t in times:
        U_t = linalg.expm(-1j * H * t)
        psi_t = U_t @ psi0
        states.append(psi_t)

    states = np.array(states)
    return times, states, H

omega = 2 * np.pi
times, states, H_tls = two_level_evolution(omega, t_max=2, n_steps=200)

# Compute expectation values
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

exp_x = [np.real(s.conj() @ sigma_x @ s) for s in states]
exp_y = [np.real(s.conj() @ sigma_y @ s) for s in states]
exp_z = [np.real(s.conj() @ sigma_z @ s) for s in states]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time evolution
ax1 = axes[0]
ax1.plot(times, exp_x, 'r-', linewidth=2, label='$\\langle\\sigma_x\\rangle$')
ax1.plot(times, exp_y, 'g-', linewidth=2, label='$\\langle\\sigma_y\\rangle$')
ax1.plot(times, exp_z, 'b-', linewidth=2, label='$\\langle\\sigma_z\\rangle$')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Expectation value')
ax1.set_title('Two-Level System: Larmor Precession')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bloch sphere projection
ax2 = axes[1]
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
ax2.plot(exp_x, exp_y, 'b-', linewidth=1.5)
ax2.scatter([exp_x[0]], [exp_y[0]], s=100, c='green', marker='o', zorder=5, label='Start')
ax2.scatter([exp_x[-1]], [exp_y[-1]], s=100, c='red', marker='s', zorder=5, label='End')
ax2.set_xlabel('$\\langle\\sigma_x\\rangle$')
ax2.set_ylabel('$\\langle\\sigma_y\\rangle$')
ax2.set_title('Bloch Sphere Trajectory')
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_level_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: two_level_evolution.png")

# =============================================================================
# Part 3: Rabi Oscillations
# =============================================================================

print("\n" + "="*70)
print("PART 3: RABI OSCILLATIONS")
print("="*70)

def rabi_oscillations(Omega: float, Delta: float, t_max: float, n_steps: int) -> Tuple:
    """
    Solve two-level system with driving: H = (Delta/2)*sigma_z + (Omega/2)*sigma_x
    """
    H = (Delta / 2) * np.array([[1, 0], [0, -1]]) + (Omega / 2) * np.array([[0, 1], [1, 0]])

    # Start in ground state |g> = |1>
    psi0 = np.array([0, 1], dtype=complex)

    times = np.linspace(0, t_max, n_steps)
    populations_e = []
    populations_g = []

    for t in times:
        U_t = linalg.expm(-1j * H * t)
        psi_t = U_t @ psi0
        populations_e.append(np.abs(psi_t[0])**2)
        populations_g.append(np.abs(psi_t[1])**2)

    return times, populations_e, populations_g

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Resonant case (Delta = 0)
ax1 = axes[0]
for Omega in [1, 2, 4]:
    times, P_e, P_g = rabi_oscillations(Omega, Delta=0, t_max=10, n_steps=200)
    ax1.plot(times, P_e, label=f'$\\Omega = {Omega}$')

ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Excited state population $P_e$')
ax1.set_title('Rabi Oscillations (Resonant: $\\Delta = 0$)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Detuned case
ax2 = axes[1]
Omega = 2
for Delta in [0, 1, 2, 4]:
    times, P_e, P_g = rabi_oscillations(Omega, Delta=Delta, t_max=10, n_steps=200)
    ax2.plot(times, P_e, label=f'$\\Delta = {Delta}$')

ax2.set_xlabel('Time $t$')
ax2.set_ylabel('Excited state population $P_e$')
ax2.set_title(f'Rabi Oscillations (Fixed $\\Omega = {Omega}$)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rabi_oscillations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rabi_oscillations.png")

# Generalized Rabi frequency
print("\nGeneralized Rabi frequency Omega' = sqrt(Omega^2 + Delta^2):")
for Delta in [0, 1, 2, 4]:
    Omega_gen = np.sqrt(Omega**2 + Delta**2)
    print(f"  Delta = {Delta}: Omega' = {Omega_gen:.4f}")

# =============================================================================
# Part 4: Free Particle Wave Packet Spreading
# =============================================================================

print("\n" + "="*70)
print("PART 4: FREE PARTICLE WAVE PACKET EVOLUTION")
print("="*70)

def free_particle_evolution(n: int, L: float, sigma: float,
                            t_values: np.ndarray, hbar: float = 1.0,
                            m: float = 1.0) -> Tuple:
    """
    Evolve a Gaussian wave packet under free particle Hamiltonian.
    """
    dx = 2*L / n
    x = np.linspace(-L, L, n, endpoint=False) + dx/2

    # Initial Gaussian
    psi0 = np.exp(-x**2 / (2*sigma**2))
    psi0 = psi0 / np.linalg.norm(psi0)

    # Hamiltonian: H = -hbar^2/(2m) * d^2/dx^2
    # Finite difference discretization
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        H[i, i] = 2
        H[i, (i+1) % n] = -1
        H[i, (i-1) % n] = -1
    H = hbar**2 / (2*m * dx**2) * H

    states = []
    for t in t_values:
        U_t = linalg.expm(-1j * H * t / hbar)
        psi_t = U_t @ psi0
        states.append(psi_t)

    return x, np.array(states), psi0

n = 256
L = 20
sigma = 2.0
t_values = np.array([0, 1, 2, 5, 10])

x, states, psi0 = free_particle_evolution(n, L, sigma, t_values)

fig, ax = plt.subplots(figsize=(12, 6))

for i, t in enumerate(t_values):
    prob = np.abs(states[i])**2
    ax.plot(x, prob + 0.1*i, label=f't = {t}')

ax.set_xlabel('Position $x$')
ax.set_ylabel('$|\\psi(x,t)|^2$ (offset)')
ax.set_title('Free Particle: Gaussian Wave Packet Spreading')
ax.legend()
ax.set_xlim([-15, 15])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wave_packet_spreading.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wave_packet_spreading.png")

# Compute width vs time
widths = []
for state in states:
    prob = np.abs(state)**2
    prob = prob / np.sum(prob)
    x_mean = np.sum(x * prob)
    x2_mean = np.sum(x**2 * prob)
    width = np.sqrt(x2_mean - x_mean**2)
    widths.append(width)

print("\nWave packet width vs time:")
for t, w in zip(t_values, widths):
    print(f"  t = {t:.1f}: width = {w:.4f}")

# =============================================================================
# Part 5: Quantum Harmonic Oscillator Evolution
# =============================================================================

print("\n" + "="*70)
print("PART 5: HARMONIC OSCILLATOR EVOLUTION")
print("="*70)

def harmonic_oscillator_evolution(n_states: int, omega: float,
                                  t_values: np.ndarray) -> Tuple:
    """
    Evolve states in the harmonic oscillator energy basis.
    """
    # Energy eigenvalues
    E = np.array([omega * (n + 0.5) for n in range(n_states)])
    H = np.diag(E)

    # Initial state: coherent state approximation
    # |alpha> ≈ sum_n (alpha^n / sqrt(n!)) * exp(-|alpha|^2/2) |n>
    alpha = 2  # coherent state parameter
    coeffs = np.zeros(n_states, dtype=complex)
    for n in range(n_states):
        coeffs[n] = (alpha**n / np.sqrt(np.math.factorial(n))) * np.exp(-np.abs(alpha)**2 / 2)
    psi0 = coeffs / np.linalg.norm(coeffs)

    # Expectation values of position (using a ~ x)
    # <x> ~ <a + a^dag> / sqrt(2)
    a = np.diag(np.sqrt(np.arange(1, n_states)), k=1)  # lowering operator
    x_op = (a + a.T) / np.sqrt(2)
    p_op = 1j * (a.T - a) / np.sqrt(2)

    x_expect = []
    p_expect = []

    for t in t_values:
        # Time evolution
        phase = np.exp(-1j * E * t)
        psi_t = phase * psi0

        x_expect.append(np.real(psi_t.conj() @ x_op @ psi_t))
        p_expect.append(np.real(psi_t.conj() @ p_op @ psi_t))

    return t_values, x_expect, p_expect

t_values_ho = np.linspace(0, 4*np.pi, 500)
omega = 1.0
_, x_expect, p_expect = harmonic_oscillator_evolution(30, omega, t_values_ho)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time evolution
ax1 = axes[0]
ax1.plot(t_values_ho, x_expect, 'b-', linewidth=2, label='$\\langle x \\rangle$')
ax1.plot(t_values_ho, p_expect, 'r-', linewidth=2, label='$\\langle p \\rangle$')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Expectation value')
ax1.set_title('Coherent State: Classical-like Oscillation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Phase space trajectory
ax2 = axes[1]
ax2.plot(x_expect, p_expect, 'b-', linewidth=1.5)
ax2.scatter([x_expect[0]], [p_expect[0]], s=100, c='green', marker='o', zorder=5, label='Start')
ax2.set_xlabel('$\\langle x \\rangle$')
ax2.set_ylabel('$\\langle p \\rangle$')
ax2.set_title('Phase Space Trajectory (Coherent State)')
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coherent_state_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: coherent_state_evolution.png")

# =============================================================================
# Part 6: Verifying Conservation Laws
# =============================================================================

print("\n" + "="*70)
print("PART 6: CONSERVATION LAWS")
print("="*70)

def check_conservation(H: np.ndarray, psi0: np.ndarray, t_values: np.ndarray) -> dict:
    """
    Check conservation of probability and energy.
    """
    norms = []
    energies = []

    for t in t_values:
        U_t = linalg.expm(-1j * H * t)
        psi_t = U_t @ psi0

        # Norm conservation
        norms.append(np.linalg.norm(psi_t))

        # Energy conservation
        energies.append(np.real(psi_t.conj() @ H @ psi_t))

    return {
        'times': t_values,
        'norms': np.array(norms),
        'energies': np.array(energies)
    }

# Test with harmonic oscillator
n_levels = 20
omega = 1.0
E = np.array([omega * (n + 0.5) for n in range(n_levels)])
H_ho = np.diag(E)

# Superposition initial state
psi0 = np.zeros(n_levels, dtype=complex)
psi0[0] = 1/np.sqrt(2)
psi0[1] = 1/np.sqrt(2)

t_values = np.linspace(0, 10*np.pi, 1000)
results = check_conservation(H_ho, psi0, t_values)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(results['times'], results['norms'], 'b-', linewidth=2)
ax1.axhline(y=1.0, color='r', linestyle='--', label='Expected: 1.0')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('$||\\psi(t)||$')
ax1.set_title('Norm Conservation (Probability)')
ax1.legend()
ax1.set_ylim([0.99, 1.01])
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(results['times'], results['energies'], 'b-', linewidth=2)
E0 = results['energies'][0]
ax2.axhline(y=E0, color='r', linestyle='--', label=f'Expected: {E0:.4f}')
ax2.set_xlabel('Time $t$')
ax2.set_ylabel('$\\langle H \\rangle$')
ax2.set_title('Energy Conservation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('conservation_laws.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: conservation_laws.png")

print("\nConservation verification:")
print(f"  Norm deviation: max|norm - 1| = {np.max(np.abs(results['norms'] - 1)):.2e}")
print(f"  Energy deviation: max|E - E0| = {np.max(np.abs(results['energies'] - E0)):.2e}")

# =============================================================================
# Part 7: Generator Computation
# =============================================================================

print("\n" + "="*70)
print("PART 7: COMPUTING THE GENERATOR")
print("="*70)

def compute_generator(U_func: Callable, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute the generator A from U(t) via A = i * lim_{t->0} (U(t) - I)/t.
    """
    I = np.eye(U_func(0).shape[0])
    A = 1j * (U_func(epsilon) - I) / epsilon
    return A

# Test: recover H from U(t) = exp(-iHt)
H_test = np.array([[1, 0.5], [0.5, 2]], dtype=complex)

def U_from_H(t):
    return linalg.expm(-1j * H_test * t)

A_computed = compute_generator(U_from_H, epsilon=1e-8)

print("Original Hamiltonian H:")
print(H_test)
print("\nRecovered generator A = i*d/dt U(t)|_{t=0}:")
print(np.real(A_computed))
print(f"\nMatch: {np.allclose(H_test, A_computed)}")

print("\n" + "="*70)
print("LAB COMPLETE")
print("="*70)
print("""
Key takeaways from Stone's Theorem:

1. CORRESPONDENCE: Self-adjoint operators <-> Strongly continuous unitary groups
   A self-adjoint <=> U(t) = exp(-iAt) is a unitary group

2. SCHRÖDINGER EQUATION: i*hbar * d|psi>/dt = H|psi>
   Solution: |psi(t)> = exp(-iHt/hbar)|psi(0)>

3. CONSERVATION: Unitarity preserves probability (norm) and energy

4. GENERATOR: A = i * lim_{t->0} (U(t) - I)/t
   The Hamiltonian generates time translations

5. QUANTUM COMPUTING: All quantum gates are unitary, generated by Hermitian ops
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Stone's theorem | $U(t) = e^{-iAt} \leftrightarrow A = A^*$ |
| Generator | $A = i\lim_{t \to 0} \frac{U(t) - I}{t}$ |
| Schrödinger equation | $i\hbar\frac{d}{dt}|\psi\rangle = H|\psi\rangle$ |
| Solution | $|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle$ |
| Heisenberg equation | $\frac{d}{dt}A_H(t) = \frac{i}{\hbar}[H, A_H]$ |
| Unitarity | $U(t)^*U(t) = I$ |

### Main Takeaways

1. **Stone's theorem**: There is a one-to-one correspondence between self-adjoint operators and strongly continuous unitary groups.

2. **Time evolution**: $U(t) = e^{-iHt/\hbar}$ is unitary precisely because $H$ is self-adjoint.

3. **Schrödinger equation**: The differential form $i\hbar\partial_t|\psi\rangle = H|\psi\rangle$ is equivalent to Stone's theorem.

4. **Conservation laws**: Unitarity guarantees conservation of probability; $[H, A] = 0$ implies conservation of $\langle A \rangle$.

5. **Generator interpretation**: The Hamiltonian generates time translations, just as momentum generates spatial translations.

6. **Quantum computing**: All quantum gates are unitary operators, hence generated by Hermitian operators.

---

## Daily Checklist

- [ ] I can define strongly continuous one-parameter unitary groups
- [ ] I understand both directions of Stone's theorem
- [ ] I can compute the generator of a unitary group
- [ ] I can derive the Schrödinger equation from Stone's theorem
- [ ] I understand why the Hamiltonian must be self-adjoint
- [ ] I can solve simple time evolution problems (two-level systems, free particles)
- [ ] I understand conservation of probability and energy
- [ ] I completed the computational lab

---

## Preview: Day 252

Tomorrow we conclude Month 9 with a comprehensive **review of Functional Analysis**. We'll synthesize:
- Banach and Hilbert spaces
- Bounded and compact operators
- The spectral theorem (compact, bounded, unbounded cases)
- Functional calculus
- Stone's theorem

This prepares us for applications in quantum mechanics, quantum field theory, and beyond.

---

*"Stone's theorem is the mathematical foundation of quantum dynamics. It explains why time evolution is unitary, why observables must be self-adjoint, and why the Schrödinger equation has the form it does."*
— Marshall Stone
