# Day 385: Week Review & Comprehensive Lab

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory Review & Synthesis |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Comprehensive QuTiP Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 385, you will be able to:

1. Synthesize the algebraic and analytic approaches to the QHO
2. Apply ladder operator methods fluently
3. Work with Fock states, coherent states, and cat states
4. Visualize quantum states using Wigner functions
5. Implement a complete QHO simulation using QuTiP
6. Connect QHO theory to quantum optics and computing applications

---

## Week 55 Review: The Quantum Harmonic Oscillator

### Key Concepts Summary

#### Day 379: QHO Setup & Motivation

$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$

**Key takeaways:**
- QHO describes any system near a stable equilibrium (Taylor expansion)
- Natural scales: $x_0 = \sqrt{\hbar/m\omega}$, $E_0 = \hbar\omega$
- Dimensionless TISE: $-\frac{d^2\psi}{d\xi^2} + \xi^2\psi = \varepsilon\psi$

---

#### Day 380: Ladder Operators

$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right), \quad \hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$$

$$[\hat{a}, \hat{a}^\dagger] = 1, \quad \hat{H} = \hbar\omega\left(\hat{N} + \frac{1}{2}\right)$$

**Key takeaways:**
- Algebraic method avoids differential equations
- $\hat{a}^\dagger$ creates quanta, $\hat{a}$ destroys quanta
- Position and momentum: $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$

---

#### Day 381: Number States |n⟩

$$|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle, \quad \hat{a}|0\rangle = 0$$

$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad \hat{a}|n\rangle = \sqrt{n}|n-1\rangle, \quad \hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

**Key takeaways:**
- Ground state defined by $\hat{a}|0\rangle = 0$
- Equally spaced spectrum with zero-point energy $E_0 = \frac{1}{2}\hbar\omega$
- Fock states form complete orthonormal basis

---

#### Day 382: Wave Functions

$$\psi_n(x) = \frac{1}{\sqrt{2^n n!}}\left(\frac{m\omega}{\pi\hbar}\right)^{1/4}H_n(\xi)e^{-\xi^2/2}$$

**Key takeaways:**
- Ground state is a Gaussian
- Hermite polynomials: $H_{n+1} = 2\xi H_n - 2nH_{n-1}$
- Parity: $\psi_n(-x) = (-1)^n\psi_n(x)$
- Number of nodes = $n$

---

#### Day 383: Coherent States

$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle, \quad |\alpha\rangle = e^{-|\alpha|^2/2}\sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

**Key takeaways:**
- Eigenstates of $\hat{a}$ (not $\hat{H}$)
- Poissonian photon statistics: $P(n) = e^{-\bar{n}}\frac{\bar{n}^n}{n!}$
- Minimum uncertainty: $\Delta x \cdot \Delta p = \frac{\hbar}{2}$
- Classical trajectory: $\alpha(t) = \alpha e^{-i\omega t}$

---

#### Day 384: Phase Space

$$W(x, p) = \frac{1}{\pi\hbar}\int\psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy$$

**Key takeaways:**
- Wigner function: quasi-probability distribution
- Can be negative (non-classical signature)
- Coherent states: positive Gaussian
- Fock states ($n \geq 1$): negative regions

---

### Master Formula Sheet

| Concept | Formula |
|---------|---------|
| **Hamiltonian** | $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 = \hbar\omega(\hat{N} + \frac{1}{2})$ |
| **Commutator** | $[\hat{a}, \hat{a}^\dagger] = 1$ |
| **Number operator** | $\hat{N} = \hat{a}^\dagger\hat{a}$ |
| **Energy spectrum** | $E_n = \hbar\omega(n + \frac{1}{2})$ |
| **Ground state** | $\psi_0 = (\frac{m\omega}{\pi\hbar})^{1/4}e^{-m\omega x^2/2\hbar}$ |
| **Coherent state** | $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$ |
| **Uncertainty** | $\Delta x \cdot \Delta p = \hbar(n + \frac{1}{2})$ for $|n\rangle$; $= \frac{\hbar}{2}$ for $|\alpha\rangle$ |
| **Wigner (ground)** | $W_0 = \frac{1}{\pi}e^{-q^2-p^2}$ |

---

## Comprehensive Problem Set

### Part A: Fundamentals (Warm-up)

**A1.** Starting from $[\hat{x}, \hat{p}] = i\hbar$, prove $[\hat{a}, \hat{a}^\dagger] = 1$.

**A2.** Calculate $\langle 3|\hat{x}^2|3\rangle$ using ladder operators.

**A3.** Write out $H_0$, $H_1$, $H_2$, $H_3$ using the recurrence relation.

### Part B: Advanced Manipulation

**B1.** Prove the identity $[\hat{a}, f(\hat{a}^\dagger)] = \frac{\partial f}{\partial \hat{a}^\dagger}$ for any analytic function $f$.

**B2.** For the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |3\rangle)$:
   - Calculate $\langle\hat{H}\rangle$ and $\Delta H$
   - Find $\langle\hat{x}\rangle(t)$ (show it oscillates)
   - What frequencies appear in $|\psi(t)|^2$?

**B3.** Show that the coherent state $|\alpha\rangle$ is NOT an eigenstate of $\hat{a}^\dagger$.

### Part C: Coherent States & Phase Space

**C1.** A coherent state has $\langle\hat{N}\rangle = 9$. Find:
   - $|\alpha|$
   - $\Delta N$
   - The probability of measuring exactly 9 photons

**C2.** For the even cat state $|\text{cat}_+\rangle = \mathcal{N}(|2\rangle + |-2\rangle)$:
   - Find the normalization $\mathcal{N}$
   - Calculate $\langle\hat{N}\rangle$
   - Explain why only even photon numbers appear

**C3.** At what value of $r = \sqrt{q^2 + p^2}$ does the Wigner function $W_2(q, p)$ first become zero?

### Part D: Applications

**D1.** A diatomic molecule has vibrational frequency $\omega = 6.3 \times 10^{13}$ rad/s and reduced mass $\mu = 1.6 \times 10^{-27}$ kg. Calculate:
   - The zero-point energy in eV
   - The spacing between vibrational levels in eV
   - The characteristic length $x_0$ in angstroms

**D2.** In a superconducting circuit, a resonator has $\omega/2\pi = 5$ GHz. At temperature $T = 20$ mK:
   - What is $\hbar\omega$ in mK (i.e., $\hbar\omega/k_B$)?
   - Estimate the thermal occupation $\langle n\rangle \approx 1/(e^{\hbar\omega/k_BT} - 1)$
   - Is the system in its quantum ground state?

**D3.** A GKP qubit encodes logical states using:
$$|0_L\rangle \propto \sum_n |2\sqrt{\pi}n\rangle_x, \quad |1_L\rangle \propto \sum_n |(2n+1)\sqrt{\pi}\rangle_x$$
where $|x\rangle_x$ denotes a position eigenstate. Why does this provide protection against small displacement errors?

---

## Solutions to Selected Problems

### Solution A2: $\langle 3|\hat{x}^2|3\rangle$

Using $\hat{x}^2 = \frac{\hbar}{2m\omega}(\hat{a} + \hat{a}^\dagger)^2 = \frac{\hbar}{2m\omega}(\hat{a}^2 + 2\hat{N} + 1 + (\hat{a}^\dagger)^2)$:

$$\langle 3|\hat{x}^2|3\rangle = \frac{\hbar}{2m\omega}\langle 3|(\hat{a}^2 + 2\hat{N} + 1 + (\hat{a}^\dagger)^2)|3\rangle$$

Since $\hat{a}^2|3\rangle \propto |1\rangle$ and $(\hat{a}^\dagger)^2|3\rangle \propto |5\rangle$:
$$\langle 3|\hat{a}^2|3\rangle = 0, \quad \langle 3|(\hat{a}^\dagger)^2|3\rangle = 0$$

Therefore:
$$\langle 3|\hat{x}^2|3\rangle = \frac{\hbar}{2m\omega}(0 + 2 \cdot 3 + 1 + 0) = \frac{7\hbar}{2m\omega}$$ ✓

### Solution C1: Coherent State with $\langle N\rangle = 9$

(a) $|\alpha|^2 = \langle\hat{N}\rangle = 9 \implies |\alpha| = 3$

(b) For coherent states: $(\Delta N)^2 = \langle N\rangle = 9 \implies \Delta N = 3$

(c) Poisson distribution: $P(9) = e^{-9}\frac{9^9}{9!} = \frac{9^9 e^{-9}}{362880} \approx 0.132$

### Solution D2: Superconducting Resonator

(a) $\hbar\omega/k_B = \frac{(1.055 \times 10^{-34})(2\pi \times 5 \times 10^9)}{1.38 \times 10^{-23}} \approx 240$ mK

(b) $\langle n\rangle = \frac{1}{e^{240/20} - 1} = \frac{1}{e^{12} - 1} \approx 6 \times 10^{-6} \approx 0$

(c) Yes! At 20 mK, the thermal occupation is essentially zero — the system is in its quantum ground state.

---

## Comprehensive QuTiP Lab

### Objective
Implement a complete QHO simulation using the Quantum Toolbox in Python (QuTiP), covering state creation, time evolution, and Wigner function visualization.

```python
"""
Day 385: Comprehensive QuTiP Lab - Quantum Harmonic Oscillator
Week 55 Review & Integration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Try to import QuTiP, fall back to manual implementation if not available
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
    print("QuTiP successfully imported!")
except ImportError:
    QUTIP_AVAILABLE = False
    print("QuTiP not available. Using manual implementations.")

# =============================================================================
# Part 1: State Creation (QuTiP or Manual)
# =============================================================================

print("=" * 70)
print("Part 1: Creating Quantum States")
print("=" * 70)

N = 30  # Hilbert space dimension

if QUTIP_AVAILABLE:
    # QuTiP state creation
    psi_0 = qt.fock(N, 0)          # Ground state
    psi_1 = qt.fock(N, 1)          # First excited
    psi_3 = qt.fock(N, 3)          # Third excited
    alpha = qt.coherent(N, 2.0)    # Coherent state α = 2
    cat_even = (qt.coherent(N, 2) + qt.coherent(N, -2)).unit()  # Even cat
    cat_odd = (qt.coherent(N, 2) - qt.coherent(N, -2)).unit()   # Odd cat

    # Operators
    a = qt.destroy(N)              # Annihilation operator
    a_dag = qt.create(N)           # Creation operator
    n_op = qt.num(N)               # Number operator
    x_op = (a + a_dag) / np.sqrt(2)
    p_op = 1j * (a_dag - a) / np.sqrt(2)

    print("\nUsing QuTiP for state creation")
else:
    # Manual implementation
    from scipy.special import factorial

    def fock_state(n, N_dim):
        state = np.zeros(N_dim, dtype=complex)
        state[n] = 1.0
        return state

    def coherent_state(alpha, N_dim):
        n = np.arange(N_dim)
        coeffs = np.exp(-np.abs(alpha)**2/2) * alpha**n / np.sqrt(factorial(n))
        return coeffs / np.linalg.norm(coeffs)

    def destroy_op(N_dim):
        a = np.zeros((N_dim, N_dim), dtype=complex)
        for n in range(1, N_dim):
            a[n-1, n] = np.sqrt(n)
        return a

    psi_0 = fock_state(0, N)
    psi_1 = fock_state(1, N)
    psi_3 = fock_state(3, N)
    alpha_state = coherent_state(2.0, N)
    cat_even_raw = coherent_state(2, N) + coherent_state(-2, N)
    cat_even = cat_even_raw / np.linalg.norm(cat_even_raw)
    cat_odd_raw = coherent_state(2, N) - coherent_state(-2, N)
    cat_odd = cat_odd_raw / np.linalg.norm(cat_odd_raw)

    a = destroy_op(N)
    a_dag = a.conj().T
    n_op = a_dag @ a
    x_op = (a + a_dag) / np.sqrt(2)
    p_op = 1j * (a_dag - a) / np.sqrt(2)

    print("\nUsing manual implementations")

print(f"Hilbert space dimension: N = {N}")

# =============================================================================
# Part 2: Verify Ladder Operator Properties
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Verifying Ladder Operator Algebra")
print("=" * 70)

if QUTIP_AVAILABLE:
    commutator = a * a_dag - a_dag * a
    print(f"[a, a†] diagonal elements: {np.diag(commutator.full())[:5].real}")
    print(f"Expected: [1, 1, 1, 1, 1]")

    # Test a|n⟩ = √n |n-1⟩
    for n in range(1, 5):
        psi_n = qt.fock(N, n)
        result = a * psi_n
        expected = np.sqrt(n) * qt.fock(N, n-1)
        match = np.allclose(result.full(), expected.full())
        print(f"a|{n}⟩ = √{n}|{n-1}⟩: {match}")
else:
    commutator = a @ a_dag - a_dag @ a
    print(f"[a, a†] diagonal: {np.diag(commutator)[:5].real}")

# =============================================================================
# Part 3: Time Evolution
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Time Evolution of Quantum States")
print("=" * 70)

# Time parameters
omega = 1.0  # Natural units
T = 2 * np.pi / omega  # One period
times = np.linspace(0, T, 100)

if QUTIP_AVAILABLE:
    # Hamiltonian
    H = omega * (n_op + 0.5 * qt.qeye(N))

    # Initial state: superposition |0⟩ + |1⟩
    psi_init = (qt.fock(N, 0) + qt.fock(N, 1)).unit()

    # Time evolution
    result = qt.mesolve(H, psi_init, times, [], [x_op, p_op, n_op])

    x_exp = result.expect[0]
    p_exp = result.expect[1]
    n_exp = result.expect[2]
else:
    # Manual time evolution
    from scipy.linalg import expm

    H = omega * (n_op + 0.5 * np.eye(N))
    psi_init = (fock_state(0, N) + fock_state(1, N))
    psi_init = psi_init / np.linalg.norm(psi_init)

    x_exp = []
    p_exp = []
    n_exp = []

    for t in times:
        U = expm(-1j * H * t)
        psi_t = U @ psi_init
        x_exp.append(np.real(np.vdot(psi_t, x_op @ psi_t)))
        p_exp.append(np.real(np.vdot(psi_t, p_op @ psi_t)))
        n_exp.append(np.real(np.vdot(psi_t, n_op @ psi_t)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(times/T, x_exp, 'b-', linewidth=2)
ax.set_xlabel('Time (periods)')
ax.set_ylabel('⟨x⟩')
ax.set_title('Position Expectation')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(times/T, p_exp, 'r-', linewidth=2)
ax.set_xlabel('Time (periods)')
ax.set_ylabel('⟨p⟩')
ax.set_title('Momentum Expectation')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(times/T, n_exp, 'g-', linewidth=2)
ax.set_xlabel('Time (periods)')
ax.set_ylabel('⟨N⟩')
ax.set_title('Number Expectation (constant!)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_385_time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Time evolution plots saved.")

# =============================================================================
# Part 4: Wigner Function Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Wigner Function Gallery")
print("=" * 70)

# Manual Wigner function computation
from scipy.special import genlaguerre

def wigner_fock(n, q, p):
    r_squared = q**2 + p**2
    L_n = genlaguerre(n, 0)
    return ((-1)**n / np.pi) * L_n(2 * r_squared) * np.exp(-r_squared)

def wigner_coherent(alpha, q, p):
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    return (1 / np.pi) * np.exp(-(q - q0)**2 - (p - p0)**2)

def wigner_cat(alpha, q, p, sign=1):
    overlap = np.exp(-2 * np.abs(alpha)**2)
    N_sq = 2 * (1 + sign * overlap)
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    W_plus = np.exp(-(q - q0)**2 - (p - p0)**2)
    W_minus = np.exp(-(q + q0)**2 - (p + p0)**2)
    W_interference = 2 * sign * np.exp(-q**2 - p**2) * np.cos(2 * (q * p0 - p * q0))
    return (W_plus + W_minus + W_interference) / (np.pi * N_sq)

# Create phase space grid
q_range = np.linspace(-5, 5, 200)
p_range = np.linspace(-5, 5, 200)
Q, P = np.meshgrid(q_range, p_range)

# Create gallery
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

states = [
    ('Vacuum |0⟩', lambda Q, P: wigner_fock(0, Q, P)),
    ('Fock |1⟩', lambda Q, P: wigner_fock(1, Q, P)),
    ('Fock |3⟩', lambda Q, P: wigner_fock(3, Q, P)),
    ('Coherent α=2', lambda Q, P: wigner_coherent(2, Q, P)),
    ('Even Cat α=2', lambda Q, P: wigner_cat(2, Q, P, 1)),
    ('Odd Cat α=2', lambda Q, P: wigner_cat(2, Q, P, -1)),
    ('Coherent α=1+i', lambda Q, P: wigner_coherent(1+1j, Q, P)),
    ('Fock |5⟩', lambda Q, P: wigner_fock(5, Q, P)),
]

for idx, (title, W_func) in enumerate(states):
    ax = axes[idx // 4, idx % 4]
    W = W_func(Q, P)
    vmax = np.max(np.abs(W))
    im = ax.contourf(Q, P, W, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(Q, P, W, levels=[0], colors='black', linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_aspect('equal')

fig.suptitle('Wigner Function Gallery: Various Quantum States', fontsize=14)
plt.tight_layout()
plt.savefig('day_385_wigner_gallery.png', dpi=150, bbox_inches='tight')
plt.show()

print("Wigner function gallery saved.")

# =============================================================================
# Part 5: Photon Number Statistics Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Photon Number Statistics")
print("=" * 70)

from scipy.special import factorial as fact

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

n_vals = np.arange(15)

# Fock state |3⟩
ax = axes[0, 0]
probs = np.zeros(15)
probs[3] = 1.0
ax.bar(n_vals, probs, color='blue', alpha=0.7)
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title('Fock State |3⟩: Definite number')

# Coherent state α = 2
ax = axes[0, 1]
alpha = 2.0
n_bar = np.abs(alpha)**2
probs = np.exp(-n_bar) * n_bar**n_vals / fact(n_vals)
ax.bar(n_vals, probs, color='green', alpha=0.7)
ax.axvline(n_bar, color='red', linestyle='--', label=f'⟨N⟩ = {n_bar}')
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title(f'Coherent State α={alpha}: Poissonian')
ax.legend()

# Even cat state
ax = axes[1, 0]
alpha = 2.0
overlap = np.exp(-2 * np.abs(alpha)**2)
N_sq = 2 * (1 + overlap)
c_n = np.exp(-np.abs(alpha)**2/2) * alpha**n_vals / np.sqrt(fact(n_vals))
c_n_minus = np.exp(-np.abs(alpha)**2/2) * (-alpha)**n_vals / np.sqrt(fact(n_vals))
coeffs = (c_n + c_n_minus) / np.sqrt(N_sq)
probs = np.abs(coeffs)**2
ax.bar(n_vals, probs, color='purple', alpha=0.7)
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title('Even Cat State: Only even n')

# Odd cat state
ax = axes[1, 1]
N_sq = 2 * (1 - overlap)
coeffs = (c_n - c_n_minus) / np.sqrt(N_sq)
probs = np.abs(coeffs)**2
ax.bar(n_vals, probs, color='orange', alpha=0.7)
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title('Odd Cat State: Only odd n')

plt.tight_layout()
plt.savefig('day_385_photon_stats.png', dpi=150, bbox_inches='tight')
plt.show()

print("Photon statistics comparison saved.")

# =============================================================================
# Part 6: Uncertainty Relations
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Uncertainty Relations for Various States")
print("=" * 70)

def compute_uncertainties(state_coeffs, a, a_dag, N_dim):
    """Compute Δx·Δp for a given state"""
    # Normalize
    state = state_coeffs / np.linalg.norm(state_coeffs)

    x_op = (a + a_dag) / np.sqrt(2)
    p_op = 1j * (a_dag - a) / np.sqrt(2)
    x2_op = x_op @ x_op
    p2_op = p_op @ p_op

    x_exp = np.real(np.vdot(state, x_op @ state))
    p_exp = np.real(np.vdot(state, p_op @ state))
    x2_exp = np.real(np.vdot(state, x2_op @ state))
    p2_exp = np.real(np.vdot(state, p2_op @ state))

    delta_x = np.sqrt(x2_exp - x_exp**2)
    delta_p = np.sqrt(p2_exp - p_exp**2)

    return delta_x, delta_p, delta_x * delta_p

a_manual = destroy_op(N)
a_dag_manual = a_manual.conj().T

print("\nUncertainty products (ℏ = 1, minimum = 0.5):")
print("-" * 50)

# Fock states
for n in range(5):
    state = fock_state(n, N)
    dx, dp, product = compute_uncertainties(state, a_manual, a_dag_manual, N)
    expected = n + 0.5
    print(f"|{n}⟩: Δx·Δp = {product:.4f} (expected: {expected:.4f})")

# Coherent states
for alpha_val in [0, 1, 2, 3]:
    state = coherent_state(alpha_val, N)
    dx, dp, product = compute_uncertainties(state, a_manual, a_dag_manual, N)
    print(f"|α={alpha_val}⟩: Δx·Δp = {product:.4f} (expected: 0.5)")

# =============================================================================
# Part 7: Coherent State Animation
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Creating Coherent State Animation Frames")
print("=" * 70)

# Create animation frames showing coherent state in phase space
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

alpha_0 = 2 + 1j
times_anim = np.linspace(0, 2*np.pi, 10)

for idx, t in enumerate(times_anim):
    ax = axes[idx // 5, idx % 5]

    alpha_t = alpha_0 * np.exp(-1j * t)
    W = wigner_coherent(alpha_t, Q, P)

    ax.contourf(Q, P, W, levels=30, cmap='Blues')

    # Trajectory
    t_traj = np.linspace(0, 2*np.pi, 100)
    alpha_traj = alpha_0 * np.exp(-1j * t_traj)
    ax.plot(np.sqrt(2)*np.real(alpha_traj), np.sqrt(2)*np.imag(alpha_traj),
            'r--', alpha=0.5)

    # Current position
    q0 = np.sqrt(2) * np.real(alpha_t)
    p0 = np.sqrt(2) * np.imag(alpha_t)
    ax.plot(q0, p0, 'r*', markersize=15)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f't = {t/np.pi:.1f}π', fontsize=11)

fig.suptitle('Coherent State Evolution: Follows Classical Trajectory', fontsize=14)
plt.tight_layout()
plt.savefig('day_385_coherent_animation.png', dpi=150, bbox_inches='tight')
plt.show()

print("Coherent state animation saved.")

# =============================================================================
# Part 8: Summary Statistics Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Summary of QHO State Properties")
print("=" * 70)

print("\n" + "=" * 70)
print("| State          | ⟨N⟩      | ΔN       | Δx·Δp    | Wigner   |")
print("|" + "-"*68 + "|")

states_info = [
    ("|0⟩", 0, 0, 0.5, "Positive"),
    ("|1⟩", 1, 0, 1.5, "Negative"),
    ("|3⟩", 3, 0, 3.5, "Negative"),
    ("|α=2⟩", 4, 2, 0.5, "Positive"),
    ("|cat+⟩", "≈4", "large", 0.5, "Negative"),
]

for name, N_exp, dN, dxdp, wigner in states_info:
    print(f"| {name:14} | {str(N_exp):8} | {str(dN):8} | {dxdp:<8} | {wigner:8} |")
print("|" + "-"*68 + "|")

# =============================================================================
# Part 9: Week 55 Concept Map
# =============================================================================

print("\n" + "=" * 70)
print("Part 9: Week 55 Concept Map")
print("=" * 70)

concept_map = """
                    QUANTUM HARMONIC OSCILLATOR
                              |
            +-----------------+------------------+
            |                                    |
      ALGEBRAIC                              ANALYTIC
      (Ladder Operators)                    (Wave Functions)
            |                                    |
    +-------+-------+                    +-------+-------+
    |               |                    |               |
   â, â†        Ĥ = ℏω(N̂+½)         Hermite Poly.  Ground State
    |               |                    |               |
[â,â†]=1      E_n = ℏω(n+½)          ψ_n(x)         Gaussian
    |               |                    |               |
    +-------+-------+                    +-------+-------+
            |                                    |
            +----------------+-------------------+
                             |
                       NUMBER STATES
                          |n⟩
                             |
            +----------------+------------------+
            |                                   |
      COHERENT STATES                     PHASE SPACE
          |α⟩                              Wigner W(x,p)
            |                                   |
    +-------+-------+                   +-------+-------+
    |       |       |                   |       |       |
 â|α⟩=α|α⟩ Poisson Min.Unc.        Negativity  Classical
            |                                Limit
    +-------+-------+
            |
      APPLICATIONS
    +-------+-------+
    |               |
Quantum Optics  Bosonic Qubits
  (Lasers)      (GKP, Cat codes)
"""

print(concept_map)

print("\n" + "=" * 70)
print("Week 55 Comprehensive Lab Complete!")
print("=" * 70)
print("\nKey achievements this week:")
print("  1. Mastered ladder operator algebra")
print("  2. Understood Fock states and their properties")
print("  3. Explored coherent states as 'most classical'")
print("  4. Visualized quantum states in phase space")
print("  5. Connected QHO to quantum optics and computing")
print("\nReady for Week 56: Tunneling & Barriers!")
```

---

## Weekly Self-Assessment

### Conceptual Understanding

Rate your understanding (1-5):

| Concept | Rating | Notes |
|---------|--------|-------|
| Ladder operators | ___ | Can I derive [a, a†] = 1? |
| Fock states | ___ | Can I construct |n⟩ from |0⟩? |
| Energy spectrum | ___ | Can I explain zero-point energy? |
| Wave functions | ___ | Do I know Hermite polynomials? |
| Coherent states | ___ | Can I prove minimum uncertainty? |
| Wigner function | ___ | Can I interpret negativity? |

### Problem-Solving Skills

- [ ] Can solve QHO problems using ladder operators
- [ ] Can compute expectation values algebraically
- [ ] Can work with coherent state expansions
- [ ] Can calculate photon number statistics
- [ ] Can analyze phase space distributions

### Computational Skills

- [ ] Can build ladder operator matrices
- [ ] Can construct Fock states numerically
- [ ] Can visualize Wigner functions
- [ ] Can animate coherent state evolution
- [ ] Comfortable with QuTiP (or manual implementations)

---

## Preview: Week 56 — Tunneling & Barriers

Next week we explore **quantum tunneling**, one of the most striking quantum phenomena:

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 386 | Step Potential | Reflection and transmission at discontinuity |
| 387 | Rectangular Barrier | Tunneling through classically forbidden region |
| 388 | WKB Approximation | Semi-classical method for tunneling |
| 389 | Alpha Decay | Nuclear physics application |
| 390 | STM Microscopy | Atomic-scale imaging via tunneling |
| 391 | Tunnel Diodes | Electronic applications |
| 392 | Month Review | Comprehensive assessment |

**Key questions to ponder:**
- How can a particle traverse a region where E < V?
- Why does tunneling probability depend exponentially on barrier width?
- What makes the scanning tunneling microscope so sensitive?

---

## Daily Checklist

- [ ] Review all Week 55 formula sheets
- [ ] Work through comprehensive problem set (Parts A-D)
- [ ] Check solutions and identify gaps
- [ ] Complete the full QuTiP lab
- [ ] Generate all visualization plots
- [ ] Self-assess understanding using the rubric
- [ ] Identify 2-3 topics for additional review
- [ ] Preview Week 56 materials

---

*"The simple harmonic oscillator is one of the most important problems in all of physics. Master it, and you have a tool that will serve you throughout your career."*
— J.J. Sakurai

---

**Congratulations on completing Week 55!**

You now have a deep understanding of the quantum harmonic oscillator — the foundation for quantum optics, quantum field theory, and bosonic quantum computing.

**Next:** [Week 56: Tunneling & Barriers](../Week_56_Tunneling/README.md)
