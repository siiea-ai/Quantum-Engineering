# Day 781: Hardware-Efficient Codes

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Bosonic codes: Cat and GKP |
| Afternoon | 2.5 hours | Biased-noise codes and hardware tailoring |
| Evening | 2 hours | Wigner function visualization and simulations |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain cat code encoding** using coherent state superpositions
2. **Analyze GKP codes** and their grid-state structure in phase space
3. **Calculate error suppression** in biased-noise environments
4. **Design bias-preserving gates** for asymmetric noise channels
5. **Evaluate hardware-code matching** for different qubit technologies
6. **Visualize bosonic states** using Wigner functions

---

## Core Content

### 1. Why Hardware-Efficient Codes?

Standard QEC assumes **depolarizing noise**: equal probability for X, Y, Z errors. But real hardware often has **structured noise**:

| Hardware | Dominant Error | Noise Ratio |
|----------|---------------|-------------|
| Superconducting (transmon) | Amplitude decay ($T_1$) | Z-biased |
| Kerr-cat oscillators | Phase flips | $\kappa_2/\kappa_1 \sim 10^2$ |
| Trapped ions | Depolarizing | ~symmetric |
| Neutral atoms | Atom loss | Erasure-like |
| Photonic | Photon loss | Erasure |

**Key insight**: Codes tailored to the noise structure can achieve better performance with fewer resources.

### 2. Cat Codes

Cat codes encode quantum information in **superpositions of coherent states** in a bosonic mode (e.g., microwave cavity).

#### Coherent States

A coherent state with amplitude $\alpha$:

$$\boxed{|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}} |n\rangle}$$

Properties:
- $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$ (eigenstate of annihilation operator)
- Minimum uncertainty: $\Delta x = \Delta p = 1/2$
- Classical-like: Localized in phase space

#### Two-Component Cat States

The logical basis states:

$$\boxed{|\mathcal{C}_\alpha^+\rangle = \mathcal{N}_+(|\alpha\rangle + |-\alpha\rangle)}$$
$$\boxed{|\mathcal{C}_\alpha^-\rangle = \mathcal{N}_-(|\alpha\rangle - |-\alpha\rangle)}$$

where the normalization factors are:

$$\mathcal{N}_\pm = \frac{1}{\sqrt{2(1 \pm e^{-2|\alpha|^2})}}$$

For large $|\alpha|$: $\mathcal{N}_\pm \approx 1/\sqrt{2}$

#### Encoding

$$|\bar{0}\rangle = |\mathcal{C}_\alpha^+\rangle = \text{even photon number states}$$
$$|\bar{1}\rangle = |\mathcal{C}_\alpha^-\rangle = \text{odd photon number states}$$

The **photon number parity** distinguishes logical states:

$$\boxed{\bar{Z} = e^{i\pi \hat{n}} = (-1)^{\hat{n}}}$$

#### Error Model

**Phase-flip (Z error)**: Small rotation in phase space
$$|\alpha\rangle \to |e^{i\phi}\alpha\rangle \approx |\alpha\rangle + i\phi\alpha|\partial_\alpha\rangle$$

This maps $|\mathcal{C}_\alpha^+\rangle \leftrightarrow |\mathcal{C}_\alpha^-\rangle$ for large rotations.

**Bit-flip (X error)**: Photon loss changes parity
$$\hat{a}|\bar{0}\rangle \propto |\bar{1}\rangle$$

However, single-photon loss probability scales as:

$$\boxed{p_{\text{bit-flip}} \sim \kappa_1 \bar{n} \cdot t = \kappa_1 |\alpha|^2 \cdot t}$$

#### Engineered Dissipation (Kerr-Cat)

Using two-photon dissipation with rate $\kappa_2 \gg \kappa_1$:

$$\mathcal{L}_2[\rho] = \kappa_2 \mathcal{D}[\hat{a}^2 - \alpha^2]\rho$$

This **stabilizes** the cat states! The cat manifold becomes an attractor:

$$\boxed{\text{Bit-flip suppression: } p_X \propto e^{-2|\alpha|^2}}$$

while phase-flip rate grows:

$$\boxed{p_Z \sim \kappa_2 |\alpha|^2}$$

**Result**: Exponentially biased noise with $p_X \ll p_Z$.

### 3. GKP Codes

The **Gottesman-Kitaev-Preskill (GKP) code** encodes a qubit in the position/momentum of a harmonic oscillator using periodic grid states.

#### Ideal GKP States

$$\boxed{|\bar{0}\rangle_{\text{GKP}} = \sum_{s=-\infty}^{\infty} |q = 2s\sqrt{\pi}\rangle}$$
$$\boxed{|\bar{1}\rangle_{\text{GKP}} = \sum_{s=-\infty}^{\infty} |q = (2s+1)\sqrt{\pi}\rangle}$$

In momentum space:

$$|\bar{+}\rangle_{\text{GKP}} = \sum_{s=-\infty}^{\infty} |p = 2s\sqrt{\pi}\rangle$$

The spacing $\sqrt{\pi}$ ensures:
- $|\bar{0}\rangle$ and $|\bar{1}\rangle$ are orthogonal
- Small displacements are correctable

#### Stabilizers

$$\boxed{S_q = e^{i2\sqrt{\pi}\hat{q}}, \quad S_p = e^{-i2\sqrt{\pi}\hat{p}}}$$

Both stabilizers have eigenvalue +1 for GKP code states.

#### Error Correction

Displacement errors $D(\beta) = e^{\beta \hat{a}^\dagger - \beta^* \hat{a}}$ are correctable if:

$$\boxed{|\text{Re}(\beta)| < \frac{\sqrt{\pi}}{2}, \quad |\text{Im}(\beta)| < \frac{\sqrt{\pi}}{2}}$$

The syndrome is extracted by measuring $S_q$ and $S_p$ modulo $2\sqrt{\pi}$.

#### Finite-Energy GKP States

Physical GKP states have finite squeezing (Gaussian envelope):

$$|\bar{0}\rangle_{\Delta} \propto \sum_s e^{-\Delta^2 s^2} |q = 2s\sqrt{\pi}\rangle$$

The squeezing parameter $\Delta$ determines:
- **Quality**: Smaller $\Delta$ → better code performance
- **Energy**: $\bar{n} \approx (1/\Delta^2 - 1)/2$

**Threshold**: Useful QEC requires $\Delta \lesssim 0.5$ (about 10 dB squeezing).

$$\boxed{\Delta_{\text{threshold}} \approx 0.5 \quad (9.5 \text{ dB squeezing})}$$

### 4. Biased-Noise Codes

When one error type dominates, specialized codes outperform symmetric codes.

#### XZZX Surface Code

The standard surface code with $Z$-biased noise benefits from **XZZX variant**:

Stabilizers rotated by 45 degrees:
$$A_v = XZZX, \quad B_p = ZXXZ$$

**Threshold improvement** with biased noise:

| Bias ratio $\eta = p_Z/p_X$ | Standard threshold | XZZX threshold |
|----------------------------|-------------------|----------------|
| 1 (symmetric) | 1.0% | 1.0% |
| 10 | 0.8% | 2.5% |
| 100 | 0.5% | 5.0% |
| $\infty$ | 0% | 50% |

#### Repetition Code for Pure Z-Noise

Under pure dephasing ($p_X = p_Y = 0$), a simple repetition code suffices:

$$|0\rangle_L = |0\rangle^{\otimes n}, \quad |1\rangle_L = |1\rangle^{\otimes n}$$

This protects against Z errors perfectly (they become undetectable but harmless).

For pure Z noise:
$$\boxed{\text{Effective code: Repetition code, corrects bit-flips only}}$$

### 5. Bias-Preserving Gates

To maintain the noise advantage, gates must not convert low-rate errors into high-rate errors.

#### Bias-Preserving Clifford Gates

| Gate | Z-bias preserving? | Implementation |
|------|-------------------|----------------|
| Z, S, T | Yes | Phase gates preserve Z |
| CNOT | Yes (with care) | Conditional Z doesn't flip bits |
| H | **No** | Swaps X ↔ Z |
| CZ | Yes | Both controls are Z-type |

**Solution**: Use **Hadamard-free** universal gate set:
$$\{S, \text{CZ}, T\} + \text{magic state injection for H-like operations}$$

#### Cat Code Gates

For cat codes with Z-bias:

$$\boxed{\bar{Z} = e^{i\pi\hat{n}} \quad \text{(exact, no errors)}}$$

$$\boxed{\bar{X} = |\mathcal{C}_\alpha^+\rangle\langle\mathcal{C}_\alpha^-| + |\mathcal{C}_\alpha^-\rangle\langle\mathcal{C}_\alpha^+|}$$

The X gate requires moving between $\pm\alpha$, which can introduce phase errors but preserves bias.

### 6. Hardware-Code Matching

#### Superconducting Qubits

**Transmons**: High $T_2^*$, moderate $T_1$, crosstalk
- Best codes: Surface codes, some bias exploitation

**Kerr-cat**: Engineered bias, long coherence for phase
- Best codes: Repetition + cat hierarchy, bias-exploiting codes

**Fluxonium**: Lower frequency, longer $T_1$
- Best codes: Standard stabilizer codes benefit from low error rate

#### Trapped Ions

Nearly symmetric errors, high fidelity gates
- Best codes: Standard surface codes, color codes
- Advantage: High connectivity enables non-local codes

#### Neutral Atoms

Erasure-like errors (atom loss detectable)
- Best codes: Erasure codes outperform Pauli codes
- Recent work: Erasure conversion for standard codes

#### Photonics

Photon loss is dominant
- Best codes: GKP (with squeezing), bosonic codes
- Cat codes with loss-tolerant design

### 7. Concatenation with Bosonic Codes

Bosonic codes can serve as the **inner code** with a stabilizer **outer code**:

```
Physical oscillator → Cat/GKP code → Surface code → Logical qubit
```

This hierarchy:
1. Cat/GKP suppresses dominant error type (phase or displacement)
2. Surface code handles residual errors
3. Lower overhead than purely qubit-based approach

$$\boxed{N_{\text{hybrid}} \approx \frac{1}{\kappa_1} \cdot d^2 \quad \text{vs} \quad N_{\text{qubit}} \approx 2d^2}$$

---

## Quantum Mechanics Connection

### Phase Space Quantum Mechanics

Bosonic codes live in the continuous-variable (CV) Hilbert space, where:
- Position $\hat{q}$ and momentum $\hat{p}$ are conjugate observables
- $[\hat{q}, \hat{p}] = i\hbar$ (we set $\hbar = 1$)
- Coherent states minimize the uncertainty relation

The **Wigner function** provides a phase-space representation:

$$\boxed{W(q, p) = \frac{1}{\pi} \int_{-\infty}^{\infty} \langle q + y|\rho|q - y\rangle e^{2ipy} dy}$$

Cat states show interference fringes; GKP states show periodic peaks.

### Quantum Optics Connection

- Cat states: Schr\"odinger's cat in the lab
- Squeezing: Heisenberg uncertainty redistribution
- Two-photon processes: Parametric down-conversion physics

---

## Worked Examples

### Example 1: Cat Code Bit-Flip Suppression

**Problem:** A Kerr-cat qubit has $|\alpha|^2 = 4$ (mean photon number 4). Calculate the bit-flip suppression factor compared to phase-flip rate if $\kappa_1/\kappa_2 = 10^{-3}$.

**Solution:**

Bit-flip rate (from single-photon loss):
$$p_X \propto \kappa_1 e^{-2|\alpha|^2} = \kappa_1 e^{-8}$$

Phase-flip rate (from two-photon process):
$$p_Z \propto \kappa_2 |\alpha|^2 = 4\kappa_2$$

Bias ratio:
$$\eta = \frac{p_Z}{p_X} = \frac{4\kappa_2}{\kappa_1 e^{-8}} = \frac{4}{\kappa_1/\kappa_2} \cdot e^{8}$$

$$\eta = \frac{4}{10^{-3}} \cdot e^{8} = 4000 \times 2981 \approx 1.2 \times 10^7$$

$$\boxed{\text{Bias ratio } \eta \approx 10^7}$$

This enormous bias means a repetition code for X errors is extremely effective.

### Example 2: GKP Squeezing Requirement

**Problem:** A GKP code must correct displacement errors up to $|\beta| = 0.2$ (in phase space units). What squeezing parameter $\Delta$ is needed, and what is the corresponding squeezing in dB?

**Solution:**

GKP correction condition:
$$|\beta| < \frac{\sqrt{\pi}}{2} \approx 0.886$$

Our target $|\beta| = 0.2$ is within the correctable range.

For reliable correction, we need the GKP peaks to be well-resolved:
$$\Delta < \frac{|\beta|}{\sqrt{\pi}/2} \times 0.5$$

More precisely, the error probability scales as:
$$p_{\text{error}} \sim \text{erfc}\left(\frac{\sqrt{\pi}/2 - |\beta|}{\Delta}\right)$$

For $|\beta| = 0.2$ and targeting $p_{\text{error}} < 1\%$:
$$\frac{0.886 - 0.2}{\Delta} > 2.3 \quad \Rightarrow \quad \Delta < 0.3$$

Squeezing in dB:
$$\text{Squeezing (dB)} = -10\log_{10}(\Delta^2) = -10\log_{10}(0.09) = 10.5 \text{ dB}$$

$$\boxed{\Delta < 0.3 \quad \text{(10.5 dB squeezing)}}$$

### Example 3: Bias-Preserving CNOT

**Problem:** Show that CNOT preserves Z-bias, i.e., Z errors on control/target map to Z errors on output.

**Solution:**

CNOT action on Paulis:
$$\text{CNOT}: X_c \mapsto X_c X_t, \quad Z_c \mapsto Z_c$$
$$\text{CNOT}: X_t \mapsto X_t, \quad Z_t \mapsto Z_c Z_t$$

Consider Z errors:
- $Z_c$ before CNOT → $Z_c$ after (no spread)
- $Z_t$ before CNOT → $Z_c Z_t$ after (spreads to control)

Both outputs are still Z-type errors!

Consider X errors:
- $X_c$ before CNOT → $X_c X_t$ after (spreads to target)
- $X_t$ before CNOT → $X_t$ after (no spread)

X errors can spread, but:
- If $p_X \ll p_Z$, the spread X errors are still rare
- No X → Z conversion occurs

$$\boxed{\text{CNOT preserves Z-bias: } p_Z \text{ dominates after gate}}$$

---

## Practice Problems

### Level A: Direct Application

**A1.** Write the cat states $|\mathcal{C}_\alpha^+\rangle$ and $|\mathcal{C}_\alpha^-\rangle$ for $\alpha = 2$. What is the mean photon number?

**A2.** For an ideal GKP state, what is the spacing between position peaks in $|\bar{0}\rangle$? In $|\bar{1}\rangle$?

**A3.** If a system has phase-flip rate $p_Z = 10^{-3}$ and bit-flip rate $p_X = 10^{-6}$, what is the bias ratio? Which code family would be most efficient?

### Level B: Intermediate Analysis

**B1.** Derive the photon number distribution for the cat state $|\mathcal{C}_\alpha^+\rangle$. Show it contains only even photon numbers.

**B2.** Design a bias-preserving gate set for universal quantum computation. What non-Clifford resource is needed?

**B3.** Calculate the mean photon number for a finite-energy GKP state with $\Delta = 0.4$. How does this compare to a cat state with similar error protection?

### Level C: Research-Level Challenges

**C1.** Derive the threshold for GKP codes under Gaussian displacement noise. How does the logical error rate scale with squeezing $\Delta$?

**C2.** Analyze the break-even point for cat codes: at what $|\alpha|^2$ does the total error rate (bit-flip + phase-flip) reach a minimum?

**C3.** Design a concatenation scheme using cat codes (inner) and XZZX surface code (outer). Estimate the overhead reduction compared to standard surface codes.

---

## Computational Lab

```python
"""
Day 781: Hardware-Efficient Codes
Simulation of cat codes, GKP codes, and biased-noise analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, hermite
from scipy.linalg import expm
from typing import Tuple, List

# =============================================================================
# BOSONIC CODE UTILITIES
# =============================================================================

def coherent_state(alpha: complex, n_max: int = 50) -> np.ndarray:
    """
    Generate coherent state |α⟩ in Fock basis.

    Args:
        alpha: Complex amplitude
        n_max: Fock space truncation

    Returns:
        State vector in Fock basis
    """
    state = np.zeros(n_max, dtype=complex)
    for n in range(n_max):
        state[n] = np.exp(-np.abs(alpha)**2 / 2) * (alpha**n) / np.sqrt(factorial(n))
    return state


def cat_state(alpha: complex, parity: str = 'even', n_max: int = 50) -> np.ndarray:
    """
    Generate cat state (|α⟩ ± |-α⟩)/N.

    Args:
        alpha: Complex amplitude
        parity: 'even' (+) or 'odd' (-)
        n_max: Fock space truncation

    Returns:
        Normalized cat state
    """
    coh_plus = coherent_state(alpha, n_max)
    coh_minus = coherent_state(-alpha, n_max)

    if parity == 'even':
        state = coh_plus + coh_minus
    else:
        state = coh_plus - coh_minus

    return state / np.linalg.norm(state)


def parity_operator(n_max: int) -> np.ndarray:
    """Generate parity operator (-1)^n."""
    return np.diag([(-1)**n for n in range(n_max)])


def annihilation_operator(n_max: int) -> np.ndarray:
    """Generate annihilation operator a."""
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(1, n_max):
        a[n-1, n] = np.sqrt(n)
    return a


def creation_operator(n_max: int) -> np.ndarray:
    """Generate creation operator a†."""
    return annihilation_operator(n_max).T.conj()


def number_operator(n_max: int) -> np.ndarray:
    """Generate number operator n = a†a."""
    return np.diag(np.arange(n_max))


# =============================================================================
# CAT CODE ANALYSIS
# =============================================================================

def analyze_cat_code(alpha: float):
    """Analyze cat code properties."""

    print("=" * 60)
    print(f"CAT CODE ANALYSIS: α = {alpha}")
    print("=" * 60)

    n_max = 100

    # Generate cat states
    cat_0 = cat_state(alpha, 'even', n_max)  # |0⟩_L
    cat_1 = cat_state(alpha, 'odd', n_max)   # |1⟩_L

    # Mean photon number
    n_op = number_operator(n_max)
    n_bar_0 = np.real(cat_0.conj() @ n_op @ cat_0)
    n_bar_1 = np.real(cat_1.conj() @ n_op @ cat_1)

    print(f"\nLogical |0⟩ (even cat):")
    print(f"  Mean photon number: {n_bar_0:.3f}")

    print(f"\nLogical |1⟩ (odd cat):")
    print(f"  Mean photon number: {n_bar_1:.3f}")

    # Orthogonality
    overlap = np.abs(cat_0.conj() @ cat_1)**2
    print(f"\nOrthogonality: |⟨0|1⟩|² = {overlap:.2e}")

    # Bit-flip suppression
    # Single photon loss: a|cat⟩
    a = annihilation_operator(n_max)
    leaked_0 = a @ cat_0
    leaked_0 /= np.linalg.norm(leaked_0)

    # Overlap with |1⟩_L after loss
    flip_prob = np.abs(leaked_0.conj() @ cat_1)**2
    expected_suppression = np.exp(-2 * alpha**2)

    print(f"\nBit-flip analysis:")
    print(f"  P(|0⟩ → |1⟩ | photon loss) ∝ e^(-2|α|²) = {expected_suppression:.2e}")

    # Bias ratio (simplified model)
    kappa_ratio = 1e-3  # κ₁/κ₂
    p_Z = alpha**2  # Proportional
    p_X = np.exp(-2 * alpha**2)

    bias = p_Z / (p_X * kappa_ratio) if p_X > 0 else np.inf
    print(f"\nNoise bias (κ₁/κ₂ = {kappa_ratio}):")
    print(f"  η = p_Z/p_X ≈ {bias:.2e}")

    return {
        'alpha': alpha,
        'n_bar': (n_bar_0 + n_bar_1) / 2,
        'orthogonality_error': overlap,
        'bit_flip_suppression': expected_suppression,
        'bias_ratio': bias
    }


def plot_cat_states(alpha: float):
    """Visualize cat states in Fock and phase space."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n_max = 50

    # Generate states
    cat_even = cat_state(alpha, 'even', n_max)
    cat_odd = cat_state(alpha, 'odd', n_max)

    # Fock distribution - even cat
    ax = axes[0, 0]
    probs_even = np.abs(cat_even)**2
    ax.bar(range(n_max), probs_even, color='blue', alpha=0.7)
    ax.set_xlabel('Photon number n')
    ax.set_ylabel('Probability')
    ax.set_title(f'Even Cat |C_α^+⟩, α = {alpha}')
    ax.set_xlim(-0.5, min(30, n_max))

    # Fock distribution - odd cat
    ax = axes[0, 1]
    probs_odd = np.abs(cat_odd)**2
    ax.bar(range(n_max), probs_odd, color='red', alpha=0.7)
    ax.set_xlabel('Photon number n')
    ax.set_ylabel('Probability')
    ax.set_title(f'Odd Cat |C_α^-⟩, α = {alpha}')
    ax.set_xlim(-0.5, min(30, n_max))

    # Wigner function - even cat
    ax = axes[1, 0]
    W_even = compute_wigner(cat_even, n_max)
    q_range = np.linspace(-5, 5, 100)
    p_range = np.linspace(-5, 5, 100)
    im = ax.contourf(q_range, p_range, W_even, levels=50, cmap='RdBu_r')
    ax.set_xlabel('q (position)')
    ax.set_ylabel('p (momentum)')
    ax.set_title('Wigner Function: Even Cat')
    ax.plot([alpha, -alpha], [0, 0], 'ko', markersize=8)
    plt.colorbar(im, ax=ax)

    # Wigner function - odd cat
    ax = axes[1, 1]
    W_odd = compute_wigner(cat_odd, n_max)
    im = ax.contourf(q_range, p_range, W_odd, levels=50, cmap='RdBu_r')
    ax.set_xlabel('q (position)')
    ax.set_ylabel('p (momentum)')
    ax.set_title('Wigner Function: Odd Cat')
    ax.plot([alpha, -alpha], [0, 0], 'ko', markersize=8)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('day_781_cat_states.png', dpi=150, bbox_inches='tight')
    plt.show()


def compute_wigner(state: np.ndarray, n_max: int,
                   q_range: np.ndarray = None,
                   p_range: np.ndarray = None) -> np.ndarray:
    """
    Compute Wigner function on a grid.

    Uses the formula involving displaced parity operator.
    """
    if q_range is None:
        q_range = np.linspace(-5, 5, 100)
    if p_range is None:
        p_range = np.linspace(-5, 5, 100)

    W = np.zeros((len(p_range), len(q_range)))

    # Simplified computation using Fock basis
    for i, q in enumerate(q_range):
        for j, p in enumerate(p_range):
            alpha = (q + 1j * p) / np.sqrt(2)
            # Displaced state
            D_alpha = displacement_operator(alpha, n_max)
            displaced = D_alpha @ state

            # Parity expectation
            parity = parity_operator(n_max)
            W[j, i] = np.real(displaced.conj() @ parity @ displaced) / np.pi

    return W


def displacement_operator(alpha: complex, n_max: int) -> np.ndarray:
    """Compute displacement operator D(α) = exp(αa† - α*a)."""
    a = annihilation_operator(n_max)
    a_dag = creation_operator(n_max)
    return expm(alpha * a_dag - np.conj(alpha) * a)


# =============================================================================
# GKP CODE ANALYSIS
# =============================================================================

def gkp_state(delta: float, n_max: int = 100, logical: int = 0) -> np.ndarray:
    """
    Generate approximate GKP state with Gaussian envelope.

    Args:
        delta: Squeezing parameter (smaller = more ideal)
        n_max: Fock space truncation
        logical: 0 or 1 for logical state

    Returns:
        GKP state in Fock basis
    """
    # Position eigenstate in Fock basis
    def position_state(q: float, n_max: int) -> np.ndarray:
        """Approximate position eigenstate."""
        state = np.zeros(n_max, dtype=complex)
        for n in range(n_max):
            # Hermite polynomial / wave function
            Hn = hermite(n)
            state[n] = (np.pi**(-0.25) / np.sqrt(2**n * factorial(n))
                       * Hn(q) * np.exp(-q**2 / 2))
        return state

    # Sum over grid points with Gaussian envelope
    spacing = np.sqrt(np.pi)
    state = np.zeros(n_max, dtype=complex)

    for s in range(-10, 11):
        if logical == 0:
            q = 2 * s * spacing
        else:
            q = (2 * s + 1) * spacing

        weight = np.exp(-delta**2 * (2 * s + logical)**2 / 2)
        state += weight * position_state(q, n_max)

    return state / np.linalg.norm(state)


def analyze_gkp_code(delta: float):
    """Analyze GKP code properties."""

    print("\n" + "=" * 60)
    print(f"GKP CODE ANALYSIS: Δ = {delta}")
    print("=" * 60)

    n_max = 150

    # Generate GKP states
    gkp_0 = gkp_state(delta, n_max, logical=0)
    gkp_1 = gkp_state(delta, n_max, logical=1)

    # Mean photon number
    n_op = number_operator(n_max)
    n_bar_0 = np.real(gkp_0.conj() @ n_op @ gkp_0)
    n_bar_1 = np.real(gkp_1.conj() @ n_op @ gkp_1)

    # Squeezing in dB
    squeezing_dB = -10 * np.log10(delta**2)

    print(f"\nSqueezing: {squeezing_dB:.1f} dB")
    print(f"Mean photon number |0⟩: {n_bar_0:.1f}")
    print(f"Mean photon number |1⟩: {n_bar_1:.1f}")

    # Orthogonality
    overlap = np.abs(gkp_0.conj() @ gkp_1)**2
    print(f"Orthogonality error: {overlap:.2e}")

    # Correctable displacement
    max_displacement = np.sqrt(np.pi) / 2
    print(f"\nMax correctable displacement: {max_displacement:.3f}")

    return {
        'delta': delta,
        'squeezing_dB': squeezing_dB,
        'n_bar': (n_bar_0 + n_bar_1) / 2,
        'orthogonality_error': overlap
    }


# =============================================================================
# BIASED NOISE ANALYSIS
# =============================================================================

def biased_noise_threshold():
    """Analyze threshold improvement with biased noise."""

    print("\n" + "=" * 60)
    print("BIASED NOISE THRESHOLD ANALYSIS")
    print("=" * 60)

    # Bias ratios
    biases = [1, 3, 10, 30, 100, 1000]

    # Thresholds (approximate values from literature)
    # Standard surface code
    standard_th = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]

    # XZZX surface code (optimized for Z-bias)
    xzzx_th = [1.0, 1.5, 2.5, 4.0, 6.0, 10.0]

    print(f"\n{'Bias η':>10} | {'Standard (%)':>12} | {'XZZX (%)':>12} | {'Improvement':>12}")
    print("-" * 55)

    for i, eta in enumerate(biases):
        improvement = xzzx_th[i] / standard_th[i]
        print(f"{eta:>10} | {standard_th[i]:>12.1f} | {xzzx_th[i]:>12.1f} | {improvement:>12.1f}×")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(biases, standard_th, 'b-o', label='Standard Surface Code',
              linewidth=2, markersize=8)
    ax.loglog(biases, xzzx_th, 'r-s', label='XZZX Surface Code',
              linewidth=2, markersize=8)

    ax.set_xlabel('Noise Bias η = p_Z/p_X', fontsize=12)
    ax.set_ylabel('Threshold (%)', fontsize=12)
    ax.set_title('QEC Threshold vs Noise Bias', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_781_bias_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_cat_vs_qubit():
    """Compare cat code vs qubit-based surface code."""

    print("\n" + "=" * 60)
    print("CAT CODE vs SURFACE CODE COMPARISON")
    print("=" * 60)

    alphas = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    # Cat code overhead (simplified model)
    # Physical qubits: 1 oscillator = effective multiple qubits
    cat_effective_distance = 2 * alphas**2  # Simplified scaling
    cat_overhead = 1  # Single oscillator

    # Surface code for equivalent protection
    # Need d such that (p/p_th)^(d/2) matches cat suppression
    p_phys = 1e-3
    p_th = 0.01
    cat_suppression = np.exp(-2 * alphas**2)

    surface_d = 2 * np.log(cat_suppression) / np.log(p_phys / p_th)
    surface_d = np.maximum(3, np.ceil(np.abs(surface_d)))
    surface_overhead = 2 * surface_d**2

    print(f"\n{'|α|²':>6} | {'Cat eff-d':>10} | {'Surface d':>10} | {'Surface qubits':>15}")
    print("-" * 50)

    for i, a in enumerate(alphas):
        print(f"{a**2:>6.1f} | {cat_effective_distance[i]:>10.1f} | "
              f"{surface_d[i]:>10.0f} | {surface_overhead[i]:>15.0f}")

    print("\nCat codes: 1 oscillator per logical qubit")
    print("Surface codes: ~2d² qubits per logical qubit")


def plot_gkp_wigner(delta: float = 0.3):
    """Plot GKP state Wigner functions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_max = 100

    # GKP |0⟩
    gkp_0 = gkp_state(delta, n_max, logical=0)

    # Simple Wigner approximation - show probability in q
    q_range = np.linspace(-6, 6, 200)

    # Position space representation (simplified)
    prob_q_0 = np.zeros_like(q_range)
    prob_q_1 = np.zeros_like(q_range)

    spacing = np.sqrt(np.pi)
    for s in range(-5, 6):
        q_peak_0 = 2 * s * spacing
        q_peak_1 = (2 * s + 1) * spacing
        weight = np.exp(-delta**2 * (2 * s)**2)
        prob_q_0 += weight * np.exp(-(q_range - q_peak_0)**2 / (2 * delta**2))
        weight = np.exp(-delta**2 * (2 * s + 1)**2)
        prob_q_1 += weight * np.exp(-(q_range - q_peak_1)**2 / (2 * delta**2))

    prob_q_0 /= np.max(prob_q_0)
    prob_q_1 /= np.max(prob_q_1)

    ax = axes[0]
    ax.plot(q_range, prob_q_0, 'b-', linewidth=2, label='|0⟩_GKP')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for s in range(-2, 3):
        ax.axvline(x=2*s*spacing, color='blue', linestyle=':', alpha=0.3)
    ax.set_xlabel('Position q', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'GKP |0⟩ (Δ = {delta})', fontsize=14)
    ax.legend()

    ax = axes[1]
    ax.plot(q_range, prob_q_1, 'r-', linewidth=2, label='|1⟩_GKP')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for s in range(-2, 3):
        ax.axvline(x=(2*s+1)*spacing, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Position q', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'GKP |1⟩ (Δ = {delta})', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig('day_781_gkp_states.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Day 781: Hardware-Efficient Codes")
    print("=" * 60)

    # Cat code analysis
    for alpha in [2.0, 3.0]:
        analyze_cat_code(alpha)

    # GKP code analysis
    for delta in [0.5, 0.3]:
        analyze_gkp_code(delta)

    # Biased noise analysis
    biased_noise_threshold()

    # Comparison
    compare_cat_vs_qubit()

    # Generate visualizations
    plot_cat_states(alpha=2.5)
    plot_gkp_wigner(delta=0.35)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("\n1. Cat codes: Exponential bit-flip suppression e^(-2|α|²)")
    print("2. GKP codes: Correct small displacements, need ~10 dB squeezing")
    print("3. Biased noise: XZZX code threshold improves with bias")
    print("4. Hardware matching: Choose code based on dominant error type")
```

---

## Summary

### Key Formulas

| Code | Key Property | Formula |
|------|-------------|---------|
| Cat (even) | Logical zero | $\|\mathcal{C}_\alpha^+\rangle = \mathcal{N}(\|\alpha\rangle + \|-\alpha\rangle)$ |
| Cat (odd) | Logical one | $\|\mathcal{C}_\alpha^-\rangle = \mathcal{N}(\|\alpha\rangle - \|-\alpha\rangle)$ |
| Cat | Bit-flip suppression | $p_X \propto e^{-2\|\alpha\|^2}$ |
| GKP | Logical zero | $\|\bar{0}\rangle = \sum_s \|q = 2s\sqrt{\pi}\rangle$ |
| GKP | Squeezing threshold | $\Delta \lesssim 0.5$ (9.5 dB) |
| XZZX | Threshold at bias $\eta$ | $p_{\text{th}}(\eta) \propto \sqrt{\eta}$ |

### Main Takeaways

1. **Hardware-efficient codes exploit noise structure**: Cat codes for biased noise, GKP for displacement errors
2. **Cat codes achieve exponential bit-flip suppression**: At cost of linear phase-flip increase
3. **GKP codes require significant squeezing**: But provide powerful continuous-variable protection
4. **Bias-preserving gates maintain advantage**: Careful gate design prevents bias destruction
5. **Concatenation combines strengths**: Bosonic inner codes + stabilizer outer codes

---

## Daily Checklist

- [ ] I can explain cat code encoding and error suppression
- [ ] I understand GKP code structure and correction
- [ ] I can calculate bias ratios for different error models
- [ ] I know which codes match which hardware
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 782

Tomorrow we study **Near-Term QEC Experiments**, examining real experimental results from leading quantum computing groups:
- Google's surface code demonstrations
- IBM's heavy-hex architecture results
- IonQ's trapped-ion QEC experiments
- Lessons learned and path forward

*"Theory guides us, but experiments ground us in reality."*

---

*Day 781 of 2184 | Year 2, Month 28, Week 112, Day 4*
*Quantum Engineering PhD Curriculum*
