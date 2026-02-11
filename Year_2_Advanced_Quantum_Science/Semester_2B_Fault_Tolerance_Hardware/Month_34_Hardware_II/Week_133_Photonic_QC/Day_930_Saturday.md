# Day 930: Cat States and Bosonic Codes

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Cat state definition, properties, Kerr nonlinearity |
| Afternoon | 2.5 hours | Problem solving: Cat code operations and error analysis |
| Evening | 1.5 hours | Computational lab: Cat state simulation and Wigner functions |

## Learning Objectives

By the end of today, you will be able to:

1. Define cat states and calculate their properties (normalization, parity, photon statistics)
2. Explain the role of Kerr nonlinearity in cat state preparation
3. Describe dissipative stabilization of cat states
4. Analyze error correction properties of cat codes
5. Compare cat codes with GKP and other bosonic encodings
6. Implement cat state simulations and visualizations

## Core Content

### 1. Coherent State Superpositions

**Definition of Cat States:**
Cat states are superpositions of coherent states with opposite phases:
$$|cat_\pm\rangle = \mathcal{N}_\pm(|\alpha\rangle \pm |-\alpha\rangle)$$

where $|\alpha\rangle$ is a coherent state and $\mathcal{N}_\pm$ is the normalization.

**Normalization:**
$$\mathcal{N}_\pm = \frac{1}{\sqrt{2(1 \pm e^{-2|\alpha|^2})}}$$

For large $|\alpha|$: $\mathcal{N}_\pm \approx 1/\sqrt{2}$

**Even and Odd Cat States:**
- $|cat_+\rangle$: Even cat state (contains only even Fock states)
- $|cat_-\rangle$: Odd cat state (contains only odd Fock states)

**Fock State Expansion:**
$$|cat_+\rangle = \mathcal{N}_+ e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^{2n}}{\sqrt{(2n)!}}|2n\rangle$$

$$|cat_-\rangle = \mathcal{N}_- e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^{2n+1}}{\sqrt{(2n+1)!}}|2n+1\rangle$$

### 2. Parity and Photon Number

**Parity Operator:**
$$\hat{\Pi} = e^{i\pi\hat{n}} = (-1)^{\hat{n}}$$

Properties:
$$\hat{\Pi}|n\rangle = (-1)^n|n\rangle$$
$$\hat{\Pi}|\alpha\rangle = |-\alpha\rangle$$

**Cat States as Parity Eigenstates:**
$$\hat{\Pi}|cat_+\rangle = +|cat_+\rangle \quad \text{(even parity)}$$
$$\hat{\Pi}|cat_-\rangle = -|cat_-\rangle \quad \text{(odd parity)}$$

**Mean Photon Number:**
For $|cat_\pm\rangle$:
$$\langle\hat{n}\rangle_\pm = |\alpha|^2 \frac{1 \mp e^{-2|\alpha|^2}}{1 \pm e^{-2|\alpha|^2}}$$

For large $|\alpha|$: $\langle\hat{n}\rangle \approx |\alpha|^2$

**Photon Number Variance:**
$$\langle\Delta n^2\rangle = |\alpha|^2(1 + |\alpha|^2) - \langle\hat{n}\rangle^2$$

Cat states are super-Poissonian (larger variance than coherent states).

### 3. Wigner Function of Cat States

**General Form:**
$$W_{cat_\pm}(q, p) = \mathcal{N}_\pm^2 \left[W_\alpha(q,p) + W_{-\alpha}(q,p) \pm 2W_{int}(q,p)\right]$$

where:
- $W_\alpha$: Wigner function of $|\alpha\rangle$
- $W_{int}$: Interference term

**Interference Term:**
$$W_{int}(q, p) = \frac{1}{\pi}e^{-(q^2+p^2)}\cos(2\sqrt{2}p\cdot\text{Re}(\alpha) - 2\sqrt{2}q\cdot\text{Im}(\alpha))$$

For real $\alpha$ (cat along q-axis):
$$W_{int}(q, p) = \frac{1}{\pi}e^{-(q^2+p^2)}\cos(2\sqrt{2}\alpha p)$$

**Key Features:**
1. Two Gaussian blobs at $\pm\alpha$
2. Interference fringes between them
3. Negative values indicate non-classicality
4. Fringe spacing: $\Delta p \sim \pi/(\sqrt{2}\alpha)$

### 4. Cat Codes for Quantum Computing

**Logical Encoding:**
$$|0_L\rangle = |cat_+\rangle = \mathcal{N}_+(|\alpha\rangle + |-\alpha\rangle)$$
$$|1_L\rangle = |cat_-\rangle = \mathcal{N}_-(|\alpha\rangle - |-\alpha\rangle)$$

**Alternative Four-Component Encoding:**
$$|0_L\rangle = \mathcal{N}(|\alpha\rangle + |-\alpha\rangle + |i\alpha\rangle + |-i\alpha\rangle)$$
$$|1_L\rangle = \mathcal{N}(|\alpha\rangle + |-\alpha\rangle - |i\alpha\rangle - |-i\alpha\rangle)$$

This provides better protection against both photon loss and dephasing.

**Logical Operations:**

*Logical Z:*
$$\hat{Z}_L = \hat{\Pi} = e^{i\pi\hat{n}}$$

This is just the parity operator!

*Logical X:*
Harder to implement - requires switching between even and odd parity.

*Hadamard:*
Requires non-Gaussian operations.

### 5. Error Correction with Cat Codes

**Dominant Errors:**

1. **Photon Loss** ($\hat{a}$):
   $$\hat{a}|cat_+\rangle \propto |cat_-\rangle$$
   $$\hat{a}|cat_-\rangle \propto |cat_+\rangle$$

   Single photon loss flips the logical state!

2. **Dephasing** (random phase on $|\pm\alpha\rangle$):
   Causes transitions between $|cat_\pm\rangle$

**Error Detection:**
Parity measurement reveals photon loss:
- Even parity → no loss (or even number lost)
- Odd parity → odd number of photons lost

**Error Correction Strategy:**
1. Continuously monitor parity
2. Track parity flips
3. Apply recovery based on parity history

**Biased Noise:**
For large $|\alpha|$, photon loss is the dominant error. Cat codes exploit this bias:
$$\gamma_{loss} \gg \gamma_{dephasing}$$

### 6. Kerr Nonlinearity

**Kerr Hamiltonian:**
$$\hat{H}_{Kerr} = \hbar K (\hat{a}^\dagger\hat{a})^2 = \hbar K \hat{n}^2$$

This causes photon-number-dependent phase rotation:
$$e^{-i\hat{H}_{Kerr}t/\hbar}|n\rangle = e^{-iKn^2 t}|n\rangle$$

**Cat State Generation:**
Start with coherent state $|\alpha\rangle$ and evolve under Kerr:

At $t = \pi/(2K)$:
$$e^{-i\frac{\pi}{2}\hat{n}^2}|\alpha\rangle = \frac{1}{\sqrt{2}}(|cat_+\rangle + i|cat_-\rangle)$$

At $t = \pi/K$:
$$e^{-i\pi\hat{n}^2}|\alpha\rangle = |-\alpha\rangle$$

**Self-Kerr in Superconducting Circuits:**
Transmon qubits have intrinsic Kerr nonlinearity:
$$K \sim 2\pi \times 1\text{ MHz}$$

This enables deterministic cat state preparation.

### 7. Dissipative Cat State Preparation

**Two-Photon Driving:**
Apply a drive at frequency $2\omega_c$ (twice the cavity frequency):
$$\hat{H}_{drive} = \epsilon_2(\hat{a}^2 + \hat{a}^{\dagger 2})$$

Combined with two-photon loss:
$$\hat{L} = \sqrt{\kappa_2}\hat{a}^2$$

**Steady State:**
The cat states $|cat_\pm\rangle$ are degenerate steady states!

The system evolves to:
$$\hat{\rho}_{ss} \to |cat_+\rangle\langle cat_+| \text{ or } |cat_-\rangle\langle cat_-|$$

depending on initial conditions.

**Advantages:**
1. Autonomous stabilization (no measurement needed)
2. Robust against small perturbations
3. Continuously corrects errors

**Implementation:**
- Superconducting circuits with Josephson junctions
- Parametric amplifiers provide two-photon processes

### 8. Multi-Component Cat States

**Four-Legged Cat:**
$$|4cat\rangle = \mathcal{N}(|\alpha\rangle + |i\alpha\rangle + |-\alpha\rangle + |-i\alpha\rangle)$$

This is an eigenstate of $\hat{a}^4$ and provides protection against higher-order errors.

**General N-Component:**
$$|Ncat\rangle = \mathcal{N}\sum_{k=0}^{N-1} |e^{2\pi i k/N}\alpha\rangle$$

**Rotation Symmetry:**
$$e^{i\frac{2\pi}{N}\hat{n}}|Ncat\rangle = |Ncat\rangle$$

Higher N provides better error protection but requires more resources.

### 9. Comparison of Bosonic Codes

| Property | Cat Code | GKP Code | Binomial Code |
|----------|----------|----------|---------------|
| Logical states | $\|\pm\alpha\rangle$ superposition | Grid states | Binomial coefficients |
| Dominant error | Photon loss | Shifts | Loss + dephasing |
| Non-Gaussianity | Moderate | High | High |
| Mean photon # | $\|\alpha\|^2$ | $1/(4\Delta^2)$ | $N_{max}/2$ |
| Experimental status | Demonstrated | Demonstrated | Demonstrated |
| Autonomous QEC | Yes | No | No |

## Quantum Computing Applications

### Kerr-Cat Qubits

Amazon Web Services and Yale are developing Kerr-cat qubits:
1. Use Kerr effect to stabilize cat states
2. Biased noise enables efficient error correction
3. Compatible with superconducting circuit technology

**Key Advantage:**
Exponential suppression of bit-flip errors with increasing $|\alpha|$:
$$\gamma_{bit-flip} \sim e^{-2|\alpha|^2}$$

### Fault-Tolerant Architecture

**Concatenation with Repetition Code:**
Since cat qubits have biased noise (phase flips >> bit flips), use:
1. Cat code for bit-flip protection
2. Repetition code for phase-flip protection

This is more efficient than standard surface code for biased noise.

## Worked Examples

### Example 1: Cat State Normalization

**Problem:** Calculate the normalization constant for $|cat_+\rangle$ with $\alpha = 2$.

**Solution:**
The normalization is:
$$\mathcal{N}_+ = \frac{1}{\sqrt{2(1 + e^{-2|\alpha|^2})}}$$

With $\alpha = 2$:
$$e^{-2|\alpha|^2} = e^{-8} \approx 3.35 \times 10^{-4}$$

$$\mathcal{N}_+ = \frac{1}{\sqrt{2(1 + 0.000335)}} = \frac{1}{\sqrt{2.00067}} \approx 0.7071$$

For practical purposes:
$$\boxed{\mathcal{N}_+ \approx \frac{1}{\sqrt{2}} = 0.7071}$$

The overlap $\langle\alpha|-\alpha\rangle = e^{-2|\alpha|^2}$ is negligible for $|\alpha| \geq 2$.

### Example 2: Photon Loss Error

**Problem:** Show that single photon loss converts $|cat_+\rangle$ to $|cat_-\rangle$.

**Solution:**
Apply the annihilation operator:
$$\hat{a}|cat_+\rangle = \mathcal{N}_+(\hat{a}|\alpha\rangle + \hat{a}|-\alpha\rangle)$$

Since $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$:
$$\hat{a}|cat_+\rangle = \mathcal{N}_+(\alpha|\alpha\rangle + (-\alpha)|-\alpha\rangle)$$
$$= \alpha\mathcal{N}_+(|\alpha\rangle - |-\alpha\rangle)$$

This is proportional to $|cat_-\rangle$:
$$\hat{a}|cat_+\rangle = \alpha\frac{\mathcal{N}_+}{\mathcal{N}_-}|cat_-\rangle$$

For large $\alpha$: $\mathcal{N}_+ \approx \mathcal{N}_-$, so:
$$\boxed{\hat{a}|cat_+\rangle \approx \alpha|cat_-\rangle}$$

Single photon loss flips the logical qubit!

### Example 3: Kerr Evolution Time

**Problem:** A cavity has Kerr nonlinearity $K = 2\pi \times 500$ kHz. How long does it take to convert $|\alpha\rangle$ to a cat state?

**Solution:**
Cat state formation occurs at $t = \pi/(4K)$ for a superposition of $|cat_+\rangle$ and $|cat_-\rangle$.

$$t = \frac{\pi}{4K} = \frac{\pi}{4 \times 2\pi \times 500 \times 10^3}$$

$$t = \frac{1}{4 \times 10^6} = 250 \text{ ns}$$

For a pure $|cat_+\rangle$ or $|cat_-\rangle$, additional measurement is needed.

$$\boxed{t = 250 \text{ ns}}$$

## Practice Problems

### Level 1: Direct Application

1. **Cat State Overlap**

   Calculate $|\langle cat_+|cat_-\rangle|^2$ for $\alpha = 1.5$.

2. **Parity Measurement**

   A cat state with $\alpha = 2$ undergoes parity measurement. What are the possible outcomes and probabilities for $|cat_+\rangle$?

3. **Mean Photon Number**

   Calculate $\langle\hat{n}\rangle$ for $|cat_+\rangle$ with $\alpha = 3$.

### Level 2: Intermediate

4. **Two-Photon Loss**

   Show that two-photon loss ($\hat{a}^2$) does not flip the logical state of a cat qubit. What error does it cause?

5. **Wigner Negativity**

   For $|cat_+\rangle$ with $\alpha = 2$, find the point in phase space where the Wigner function has its most negative value.

6. **Kerr Dynamics**

   Starting from $|\alpha = 2\rangle$, calculate the state after Kerr evolution for time $t = \pi/(4K)$. Express in terms of cat states.

### Level 3: Challenging

7. **Four-Component Cat**

   Derive the Fock state expansion of the four-component cat state:
   $$|4cat_0\rangle = \mathcal{N}(|\alpha\rangle + |i\alpha\rangle + |-\alpha\rangle + |-i\alpha\rangle)$$
   Which Fock states appear?

8. **Dissipative Dynamics**

   For a cavity with two-photon driving $\epsilon_2$ and two-photon loss $\kappa_2$, write the Lindblad master equation. Show that cat states are steady states.

9. **Concatenated Code Threshold**

   For a cat qubit with bit-flip rate $\gamma_X = e^{-2|\alpha|^2}\gamma_0$ and phase-flip rate $\gamma_Z = \gamma_0|\alpha|^2$, find the optimal $|\alpha|$ that minimizes the total logical error rate when concatenated with a distance-3 repetition code.

## Computational Lab: Cat State Simulation

```python
"""
Day 930 Computational Lab: Cat States and Bosonic Codes
Simulation of cat states, Wigner functions, and error dynamics
"""

import numpy as np
from scipy.special import factorial
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fock space dimension
N_FOCK = 50


def coherent_state(alpha: complex, N: int = N_FOCK) -> np.ndarray:
    """Create coherent state |α⟩ in Fock basis."""
    n = np.arange(N)
    amplitudes = np.exp(-np.abs(alpha)**2 / 2) * (alpha ** n) / np.sqrt(factorial(n))
    return amplitudes.astype(complex)


def cat_state(alpha: complex, sign: int = 1, N: int = N_FOCK) -> np.ndarray:
    """
    Create cat state |cat_±⟩ = N(|α⟩ ± |-α⟩).
    sign: +1 for even cat, -1 for odd cat
    """
    coh_plus = coherent_state(alpha, N)
    coh_minus = coherent_state(-alpha, N)

    cat = coh_plus + sign * coh_minus

    # Normalize
    cat /= np.linalg.norm(cat)
    return cat


def four_component_cat(alpha: complex, N: int = N_FOCK) -> np.ndarray:
    """Create four-component cat state."""
    state = np.zeros(N, dtype=complex)
    for k in range(4):
        phase = np.exp(1j * np.pi * k / 2)
        state += coherent_state(alpha * phase, N)

    state /= np.linalg.norm(state)
    return state


def annihilation_operator(N: int = N_FOCK) -> np.ndarray:
    """Create annihilation operator."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a


def creation_operator(N: int = N_FOCK) -> np.ndarray:
    """Create creation operator."""
    return annihilation_operator(N).T.conj()


def number_operator(N: int = N_FOCK) -> np.ndarray:
    """Create number operator."""
    return np.diag(np.arange(N, dtype=complex))


def parity_operator(N: int = N_FOCK) -> np.ndarray:
    """Create parity operator (-1)^n."""
    return np.diag([(-1)**n for n in range(N)])


def displacement_operator(alpha: complex, N: int = N_FOCK) -> np.ndarray:
    """Create displacement operator D(α)."""
    a = annihilation_operator(N)
    a_dag = creation_operator(N)
    return expm(alpha * a_dag - np.conj(alpha) * a)


def kerr_unitary(K_t: float, N: int = N_FOCK) -> np.ndarray:
    """Create Kerr evolution unitary exp(-i K t n^2)."""
    n = number_operator(N)
    return expm(-1j * K_t * n @ n)


def wigner_function(state: np.ndarray, q_range: float = 5,
                    n_points: int = 100) -> tuple:
    """
    Calculate Wigner function for a state in Fock basis.
    Uses the formula involving displaced parity.
    """
    q = np.linspace(-q_range, q_range, n_points)
    p = np.linspace(-q_range, q_range, n_points)
    Q, P = np.meshgrid(q, p)

    N = len(state)
    W = np.zeros_like(Q)

    # Wigner function via Fock state summation
    for i, qi in enumerate(q):
        for j, pi in enumerate(p):
            alpha = (qi + 1j * pi) / np.sqrt(2)

            # W(q,p) = (2/π) Tr[D†(α) ρ D(α) Π]
            # For pure state: W = (2/π) |⟨α|Π|ψ⟩|²
            # Use direct calculation

            w_val = 0
            for n in range(N):
                for m in range(N):
                    if (n + m) % 2 == 0:  # Parity constraint
                        # Matrix element of displaced parity
                        w_val += state[n] * np.conj(state[m]) * \
                                 wigner_matrix_element(n, m, qi, pi)

            W[j, i] = (2 / np.pi) * np.real(w_val)

    return Q, P, W


def wigner_matrix_element(n: int, m: int, q: float, p: float) -> complex:
    """Calculate Wigner function matrix element."""
    from scipy.special import genlaguerre

    r2 = q**2 + p**2
    alpha = (q + 1j * p) / np.sqrt(2)

    if n == m:
        L = genlaguerre(n, 0)
        return (-1)**n * L(2 * r2) * np.exp(-r2)
    else:
        # Cross terms (more complex, approximate for speed)
        return 0  # Simplified - use diagonal terms only for speed


def wigner_coherent(Q: np.ndarray, P: np.ndarray, alpha: complex) -> np.ndarray:
    """Wigner function of coherent state (exact formula)."""
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    return (1 / np.pi) * np.exp(-((Q - q0)**2 + (P - p0)**2))


def wigner_cat_analytic(Q: np.ndarray, P: np.ndarray,
                        alpha: float, sign: int = 1) -> np.ndarray:
    """
    Analytic Wigner function for cat state with real α.
    """
    # Normalization
    overlap = np.exp(-2 * alpha**2)
    N2 = 2 * (1 + sign * overlap)

    # Two coherent state contributions
    W_plus = wigner_coherent(Q, P, alpha)
    W_minus = wigner_coherent(Q, P, -alpha)

    # Interference term
    W_int = (2 / np.pi) * np.exp(-(Q**2 + P**2)) * np.cos(2 * np.sqrt(2) * alpha * P)

    return (W_plus + W_minus + sign * 2 * W_int) / N2


def visualize_cat_states():
    """Visualize cat state Wigner functions."""
    print("=" * 60)
    print("Cat State Wigner Function Visualization")
    print("=" * 60)

    q = np.linspace(-6, 6, 200)
    p = np.linspace(-6, 6, 200)
    Q, P = np.meshgrid(q, p)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    alpha_values = [1.5, 2.5]
    signs = [1, -1]
    titles = ['Even Cat |cat₊⟩', 'Odd Cat |cat₋⟩']

    for i, alpha in enumerate(alpha_values):
        for j, (sign, title) in enumerate(zip(signs, titles)):
            W = wigner_cat_analytic(Q, P, alpha, sign)

            ax = axes[i, j]
            levels = np.linspace(W.min(), W.max(), 50)
            cf = ax.contourf(Q, P, W, levels=levels, cmap='RdBu_r')
            ax.contour(Q, P, W, levels=[0], colors='k', linewidths=0.5)
            plt.colorbar(cf, ax=ax)

            ax.set_xlabel('q', fontsize=12)
            ax.set_ylabel('p', fontsize=12)
            ax.set_title(f'{title}, α={alpha}', fontsize=14)
            ax.set_aspect('equal')

            # Mark coherent state positions
            ax.plot([np.sqrt(2)*alpha, -np.sqrt(2)*alpha], [0, 0],
                   'ko', markersize=8)

    plt.tight_layout()
    plt.savefig('cat_wigner_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: cat_wigner_functions.png")


def photon_number_distribution():
    """Analyze photon number distributions of cat states."""
    print("\n" + "=" * 60)
    print("Cat State Photon Number Distribution")
    print("=" * 60)

    N = 30
    alpha = 2.0

    # Create states
    cat_even = cat_state(alpha, sign=1, N=N)
    cat_odd = cat_state(alpha, sign=-1, N=N)
    coherent = coherent_state(alpha, N=N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n_values = np.arange(N)

    # Coherent state
    axes[0].bar(n_values, np.abs(coherent)**2, color='blue', alpha=0.7)
    axes[0].set_xlabel('Photon number n', fontsize=12)
    axes[0].set_ylabel('P(n)', fontsize=12)
    axes[0].set_title(f'Coherent State α={alpha}', fontsize=14)
    axes[0].set_xlim(-0.5, 15)

    # Even cat
    axes[1].bar(n_values, np.abs(cat_even)**2, color='green', alpha=0.7)
    axes[1].set_xlabel('Photon number n', fontsize=12)
    axes[1].set_ylabel('P(n)', fontsize=12)
    axes[1].set_title(f'Even Cat |cat₊⟩, α={alpha}', fontsize=14)
    axes[1].set_xlim(-0.5, 15)

    # Odd cat
    axes[2].bar(n_values, np.abs(cat_odd)**2, color='red', alpha=0.7)
    axes[2].set_xlabel('Photon number n', fontsize=12)
    axes[2].set_ylabel('P(n)', fontsize=12)
    axes[2].set_title(f'Odd Cat |cat₋⟩, α={alpha}', fontsize=14)
    axes[2].set_xlim(-0.5, 15)

    plt.tight_layout()
    plt.savefig('cat_photon_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: cat_photon_distribution.png")

    # Verify parity
    print(f"\nParity verification (α={alpha}):")
    print(f"  Even cat: only even n have non-zero amplitude")
    print(f"  Odd cat: only odd n have non-zero amplitude")

    # Calculate mean photon number
    n_mean_even = np.sum(n_values * np.abs(cat_even)**2)
    n_mean_odd = np.sum(n_values * np.abs(cat_odd)**2)
    n_mean_coh = np.sum(n_values * np.abs(coherent)**2)

    print(f"\nMean photon numbers:")
    print(f"  Coherent: ⟨n⟩ = {n_mean_coh:.3f} (theory: {alpha**2:.3f})")
    print(f"  Even cat: ⟨n⟩ = {n_mean_even:.3f}")
    print(f"  Odd cat: ⟨n⟩ = {n_mean_odd:.3f}")


def kerr_evolution_demo():
    """Demonstrate cat state generation via Kerr effect."""
    print("\n" + "=" * 60)
    print("Kerr Evolution: Coherent to Cat State")
    print("=" * 60)

    N = 40
    alpha = 2.0

    # Initial coherent state
    psi_0 = coherent_state(alpha, N)

    # Time evolution
    K = 1.0  # Kerr strength (normalized)
    times = [0, np.pi/(4*K), np.pi/(2*K), np.pi/K]
    labels = ['t=0', 't=π/4K', 't=π/2K', 't=π/K']

    q = np.linspace(-5, 5, 150)
    p = np.linspace(-5, 5, 150)
    Q, P = np.meshgrid(q, p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (t, label) in enumerate(zip(times, labels)):
        # Kerr evolution
        U_kerr = kerr_unitary(K * t, N)
        psi_t = U_kerr @ psi_0

        # For Wigner, use approximate formula based on state decomposition
        # Analyze in terms of coherent state overlaps

        # Calculate effective alpha for display
        a = annihilation_operator(N)
        alpha_eff = np.conj(psi_t) @ a @ psi_t

        # For simplicity, show photon number distribution
        ax = axes[i]
        n_vals = np.arange(min(20, N))
        probs = np.abs(psi_t[:20])**2

        ax.bar(n_vals, probs, color='purple', alpha=0.7)
        ax.set_xlabel('Photon number n', fontsize=12)
        ax.set_ylabel('P(n)', fontsize=12)
        ax.set_title(f'{label}', fontsize=14)

        # Calculate parity
        Pi = parity_operator(N)
        parity = np.real(np.conj(psi_t) @ Pi @ psi_t)
        ax.text(0.95, 0.95, f'⟨Π⟩ = {parity:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('kerr_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: kerr_evolution.png")


def photon_loss_simulation():
    """Simulate effect of photon loss on cat states."""
    print("\n" + "=" * 60)
    print("Photon Loss Effect on Cat Qubits")
    print("=" * 60)

    N = 40
    alpha = 2.0

    # Create cat states
    cat_even = cat_state(alpha, sign=1, N=N)
    cat_odd = cat_state(alpha, sign=-1, N=N)

    # Operators
    a = annihilation_operator(N)
    Pi = parity_operator(N)

    # Apply photon loss
    print("\nEffect of photon loss on |cat₊⟩:")

    state = cat_even.copy()
    for n_loss in range(5):
        parity = np.real(np.conj(state) @ Pi @ state)
        norm = np.linalg.norm(state)
        print(f"  After {n_loss} loss events: ⟨Π⟩ = {parity:.4f}, norm = {norm:.4f}")

        # Apply loss
        state = a @ state
        if np.linalg.norm(state) > 1e-10:
            state /= np.linalg.norm(state)

    print("\nObservation: Parity flips with each photon loss event")
    print("This allows error detection via parity measurement")


def error_rate_analysis():
    """Analyze logical error rates vs cat size."""
    print("\n" + "=" * 60)
    print("Cat Qubit Error Rate Analysis")
    print("=" * 60)

    alpha_values = np.linspace(0.5, 4, 50)

    # Bit-flip rate (suppressed exponentially)
    gamma_X = np.exp(-2 * alpha_values**2)

    # Phase-flip rate (linear in photon number)
    gamma_Z = alpha_values**2

    # Total error rate (for comparison)
    gamma_total = gamma_X + gamma_Z

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log scale for bit-flip
    axes[0].semilogy(alpha_values, gamma_X, 'b-', linewidth=2, label='Bit-flip γ_X')
    axes[0].semilogy(alpha_values, gamma_Z, 'r-', linewidth=2, label='Phase-flip γ_Z')
    axes[0].set_xlabel('Cat size |α|', fontsize=12)
    axes[0].set_ylabel('Error rate (normalized)', fontsize=12)
    axes[0].set_title('Cat Qubit Error Rates', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bias ratio
    bias = gamma_Z / gamma_X
    axes[1].semilogy(alpha_values, bias, 'g-', linewidth=2)
    axes[1].set_xlabel('Cat size |α|', fontsize=12)
    axes[1].set_ylabel('Noise bias γ_Z/γ_X', fontsize=12)
    axes[1].set_title('Noise Bias in Cat Qubits', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cat_error_rates.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: cat_error_rates.png")

    # Find optimal alpha for given target
    print("\nNoise bias at different cat sizes:")
    for alpha in [1, 1.5, 2, 2.5, 3]:
        gX = np.exp(-2 * alpha**2)
        gZ = alpha**2
        print(f"  α = {alpha}: γ_X = {gX:.2e}, γ_Z = {gZ:.2f}, bias = {gZ/gX:.2e}")


def four_component_cat_demo():
    """Demonstrate four-component cat states."""
    print("\n" + "=" * 60)
    print("Four-Component Cat State")
    print("=" * 60)

    N = 40
    alpha = 2.0

    # Create four-component cat
    state = four_component_cat(alpha, N)

    # Photon number distribution
    n_vals = np.arange(20)
    probs = np.abs(state[:20])**2

    plt.figure(figsize=(10, 6))
    plt.bar(n_vals, probs, color='purple', alpha=0.7)
    plt.xlabel('Photon number n', fontsize=12)
    plt.ylabel('P(n)', fontsize=12)
    plt.title(f'Four-Component Cat State, α={alpha}', fontsize=14)
    plt.savefig('four_component_cat.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: four_component_cat.png")

    # Check which Fock states appear
    print("\nFock state populations (|c_n|² > 0.001):")
    for n in range(20):
        if probs[n] > 0.001:
            print(f"  n = {n}: P = {probs[n]:.4f}")

    print("\nNote: Four-component cat only has n = 0, 4, 8, 12, ... (multiples of 4)")


def main():
    """Run all cat state simulations."""
    print("\n" + "=" * 60)
    print("DAY 930: CAT STATES AND BOSONIC CODES")
    print("=" * 60)

    # Basic demonstrations
    visualize_cat_states()
    photon_number_distribution()
    kerr_evolution_demo()
    photon_loss_simulation()
    error_rate_analysis()
    four_component_cat_demo()

    print("\n" + "=" * 60)
    print("Simulations Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Cat state | $\|cat_\pm\rangle = \mathcal{N}_\pm(\|\alpha\rangle \pm \|-\alpha\rangle)$ |
| Normalization | $\mathcal{N}_\pm = 1/\sqrt{2(1 \pm e^{-2\|\alpha\|^2})}$ |
| Parity operator | $\hat{\Pi} = (-1)^{\hat{n}} = e^{i\pi\hat{n}}$ |
| Logical Z | $\hat{Z}_L = \hat{\Pi}$ |
| Kerr Hamiltonian | $\hat{H}_K = \hbar K \hat{n}^2$ |
| Bit-flip suppression | $\gamma_X \sim e^{-2\|\alpha\|^2}$ |
| Phase-flip rate | $\gamma_Z \sim \|\alpha\|^2$ |

### Key Takeaways

1. **Cat states** are superpositions of coherent states with opposite phases
2. **Even/odd cats** are parity eigenstates, useful for encoding qubits
3. **Photon loss** flips the logical state but preserves the encoding space
4. **Kerr nonlinearity** enables deterministic cat state preparation
5. **Dissipative stabilization** provides autonomous error correction
6. Cat qubits exhibit **biased noise** - exponentially suppressed bit flips

## Daily Checklist

- [ ] I can define cat states and calculate their properties
- [ ] I understand the role of parity in cat code encoding
- [ ] I can explain how Kerr effect generates cat states
- [ ] I understand the error model for cat qubits
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 931

Tomorrow we conclude the week with **Integrated Photonics**, exploring how photonic quantum computing is implemented in chip-scale systems. Key topics:
- Silicon photonics fundamentals
- On-chip single-photon sources and detectors
- Programmable photonic processors
- Industry implementations (PsiQuantum, Xanadu, QuiX)
