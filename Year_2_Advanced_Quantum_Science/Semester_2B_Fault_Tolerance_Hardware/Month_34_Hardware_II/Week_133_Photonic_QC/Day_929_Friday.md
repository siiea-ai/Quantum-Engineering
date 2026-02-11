# Day 929: GKP Encoding

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | GKP code construction, logical operations, error correction |
| Afternoon | 2.5 hours | Problem solving: Phase space analysis and syndrome extraction |
| Evening | 1.5 hours | Computational lab: GKP state simulation |

## Learning Objectives

By the end of today, you will be able to:

1. Construct ideal GKP logical states in position and momentum bases
2. Derive logical Pauli and Clifford operations for GKP codes
3. Explain the error correction mechanism for shift errors
4. Describe approximate (physical) GKP states and their properties
5. Analyze the resource requirements for GKP state preparation
6. Implement GKP state visualization and error correction simulation

## Core Content

### 1. The GKP Code Concept

**The Problem:**
Continuous-variable systems suffer from analog errors - small shifts in $q$ and $p$. How do we correct these?

**The GKP Solution (2001):**
Gottesman, Kitaev, and Preskill showed how to encode a **discrete qubit** into a **continuous-variable mode**, enabling:
1. Detection and correction of small shift errors
2. Use of standard qubit error-correcting codes on top
3. Fault-tolerant CV quantum computing

**Key Insight:**
Make the logical states periodic in phase space. Small shifts can be detected without destroying the encoded information.

### 2. Ideal GKP States

**Position Basis Definition:**
$$|0_L\rangle = \sum_{n=-\infty}^{\infty} |q = 2n\sqrt{\pi}\rangle$$
$$|1_L\rangle = \sum_{n=-\infty}^{\infty} |q = (2n+1)\sqrt{\pi}\rangle$$

These are infinite combs of delta functions in position space, spaced by $2\sqrt{\pi}$.

**Momentum Basis:**
Taking the Fourier transform:
$$|0_L\rangle = \sum_{n=-\infty}^{\infty} |p = n\sqrt{\pi}\rangle$$
$$|1_L\rangle = \sum_{n=-\infty}^{\infty} (-1)^n|p = n\sqrt{\pi}\rangle$$

**Logical Plus and Minus States:**
$$|+_L\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle) = \sum_n |q = n\sqrt{\pi}\rangle$$
$$|-_L\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle - |1_L\rangle) = \sum_n (-1)^n|q = n\sqrt{\pi}\rangle$$

**Phase Space Picture:**
The Wigner function of ideal GKP states is a 2D grid of delta functions:
$$W_{0_L}(q, p) \propto \sum_{n,m} \delta(q - 2n\sqrt{\pi})\delta(p - m\sqrt{\pi})$$

### 3. Stabilizer Formalism for GKP

**Stabilizer Operators:**
The GKP code is stabilized by displacement operators:
$$\hat{S}_q = e^{2i\sqrt{\pi}\hat{p}} = \hat{D}(i\sqrt{\pi})$$
$$\hat{S}_p = e^{-2i\sqrt{\pi}\hat{q}} = \hat{D}(\sqrt{\pi})$$

These shift the state by $2\sqrt{\pi}$ in position or momentum:
$$\hat{S}_q|q\rangle = |q + 2\sqrt{\pi}\rangle$$
$$\hat{S}_p|p\rangle = |p + 2\sqrt{\pi}\rangle$$

**Stabilizer Conditions:**
$$\hat{S}_q|0_L\rangle = |0_L\rangle, \quad \hat{S}_q|1_L\rangle = |1_L\rangle$$
$$\hat{S}_p|0_L\rangle = |0_L\rangle, \quad \hat{S}_p|1_L\rangle = |1_L\rangle$$

Both logical states are +1 eigenstates of both stabilizers.

**Commutation:**
$$[\hat{S}_q, \hat{S}_p] = 0$$

The stabilizers commute because:
$$\hat{S}_q\hat{S}_p = e^{2i\pi}\hat{S}_p\hat{S}_q = \hat{S}_p\hat{S}_q$$

### 4. Logical Pauli Operations

**Logical X:**
$$\hat{X}_L = e^{i\sqrt{\pi}\hat{p}} = \hat{D}(i\sqrt{\pi}/2)$$

This shifts position by $\sqrt{\pi}$:
$$\hat{X}_L|0_L\rangle = |1_L\rangle, \quad \hat{X}_L|1_L\rangle = |0_L\rangle$$

**Logical Z:**
$$\hat{Z}_L = e^{-i\sqrt{\pi}\hat{q}}$$

This applies alternating signs:
$$\hat{Z}_L|0_L\rangle = |0_L\rangle, \quad \hat{Z}_L|1_L\rangle = -|1_L\rangle$$

**Logical Y:**
$$\hat{Y}_L = i\hat{X}_L\hat{Z}_L = e^{i\sqrt{\pi}\hat{p}}e^{-i\sqrt{\pi}\hat{q}}$$

**Anticommutation:**
$$\{\hat{X}_L, \hat{Z}_L\} = 0$$

as required for Pauli operators.

**Relation to Stabilizers:**
$$\hat{X}_L^2 = \hat{S}_q, \quad \hat{Z}_L^2 = \hat{S}_p$$

### 5. Logical Clifford Gates

**Hadamard Gate:**
The Fourier transform swaps $q \leftrightarrow p$:
$$\hat{H}_L = e^{i\frac{\pi}{4}(\hat{q}^2 + \hat{p}^2 - 1)} = e^{i\frac{\pi}{4}(2\hat{n} + 1)}$$

This is just a $\pi/2$ rotation in phase space!
$$\hat{H}_L|q\rangle = |p = q\rangle$$

**Phase Gate:**
$$\hat{S}_L = e^{i\frac{\pi}{4}\hat{q}^2}$$

**CNOT Gate:**
For two GKP modes:
$$\hat{CNOT}_L = e^{-i\hat{q}_1\hat{p}_2}$$

This is the CV controlled-displacement (SUM gate):
$$\hat{q}_2 \to \hat{q}_2 + \hat{q}_1, \quad \hat{p}_1 \to \hat{p}_1 - \hat{p}_2$$

**Universal Gate Set:**
GKP Clifford gates are all Gaussian operations! For universality, need:
- Non-Clifford gate (e.g., T gate via magic state injection)
- Or cubic phase gate: $e^{i\gamma\hat{q}^3}$

### 6. Error Correction Mechanism

**Shift Errors:**
The dominant errors in CV systems are small displacements:
$$\hat{E}(\epsilon_q, \epsilon_p) = e^{i\epsilon_p\hat{q}}e^{-i\epsilon_q\hat{p}}$$

This shifts the state by $(\epsilon_q, \epsilon_p)$ in phase space.

**Error Detection:**
Measure the syndrome: $q \mod 2\sqrt{\pi}$ and $p \mod \sqrt{\pi}$.

If $|\epsilon_q| < \sqrt{\pi}/2$ and $|\epsilon_p| < \sqrt{\pi}/2$, the error can be corrected.

**Correction Procedure:**
1. Measure $q \mod 2\sqrt{\pi}$: gives estimate of $\epsilon_q$
2. Apply correction: shift by $-\epsilon_q$
3. Measure $p \mod \sqrt{\pi}$: gives estimate of $\epsilon_p$
4. Apply correction: shift by $-\epsilon_p$

**Correctable Region:**
Errors with $|\epsilon_q| < \sqrt{\pi}/2$ and $|\epsilon_p| < \sqrt{\pi}/2$ are correctable.

This defines a **Voronoi cell** in phase space - a square of side $\sqrt{\pi}$.

**Logical Error:**
Errors outside this region cause logical errors:
- $\epsilon_q \approx \sqrt{\pi}$: causes $\hat{X}_L$ error
- $\epsilon_p \approx \sqrt{\pi}$: causes $\hat{Z}_L$ error

### 7. Approximate GKP States

**The Problem with Ideal States:**
Ideal GKP states have:
- Infinite energy (infinite photon number)
- Infinite extent in phase space
- Zero width peaks (unphysical)

**Physical GKP States:**
Use Gaussian envelope to regularize:
$$|0_L^{\Delta}\rangle \propto \sum_{n=-\infty}^{\infty} e^{-\Delta^2(2n\sqrt{\pi})^2/2} \int e^{-(q-2n\sqrt{\pi})^2/(2\Delta^2)} |q\rangle dq$$

Parameters:
- $\Delta$: width of each peak (in quadrature units)
- Envelope: Gaussian decay of peak heights

**Wigner Function:**
$$W(q, p) \propto \sum_{n,m} e^{-\Delta^2(q-2n\sqrt{\pi})^2}e^{-\Delta^{-2}(p-m\sqrt{\pi})^2}e^{-\Delta^2((2n)^2 + m^2)\pi}$$

This is a grid of Gaussians with Gaussian envelope.

**Squeezing Requirement:**
The peak width $\Delta$ determines the squeezing:
$$\Delta = e^{-r}$$

For useful GKP states: $\Delta < 0.3$ requires $r > 1.2$ (about 10 dB squeezing).

**Mean Photon Number:**
$$\langle\hat{n}\rangle \approx \frac{1}{4\Delta^2} - \frac{1}{2}$$

For $\Delta = 0.2$: $\langle n \rangle \approx 6$ photons.

### 8. GKP State Preparation

**Preparation Methods:**

1. **Breeding Protocol:**
   - Start with squeezed states
   - Interfere on beam splitters
   - Measure and post-select
   - Repeat to improve quality

2. **Dissipative Preparation:**
   - Engineer dissipation to stabilize GKP manifold
   - Use nonlinear interactions (Kerr, SNAP)

3. **Modular Measurement:**
   - Measure $\hat{q} \mod 2\sqrt{\pi}$
   - Project onto GKP-like state

**Experimental Status:**
- First GKP states: superconducting circuits (2019)
- Photonic GKP: under development
- Best achieved: $\Delta \approx 0.25$ (about 12 dB equivalent)

### 9. Concatenated GKP Codes

**GKP + Surface Code:**
Encode each physical qubit of a surface code in a GKP mode:
$$|0_L^{surface}\rangle = \text{(surface code on GKP qubits)}$$

**Threshold Improvement:**
- Bare surface code threshold: ~1%
- GKP-surface code: can tolerate 10-20% loss per mode!

**Biased Noise:**
GKP errors are often biased ($\epsilon_p \neq \epsilon_q$). This can be exploited:
- XZZX surface code for biased noise
- Rectangular GKP grids

## Quantum Computing Applications

### Fault-Tolerant Photonic QC

GKP encoding is central to several fault-tolerant photonic architectures:

**Xanadu's Blueprint (2021):**
1. Prepare GKP states in squeezed light
2. Build GKP cluster states using beam splitters
3. Perform computation via homodyne measurement
4. Use GKP error correction between layers

**Resource Estimates:**
- ~1000 physical modes per logical qubit
- Clock speed: ~1 MHz
- Target: fault-tolerant universal quantum computing

### Comparison with Other Encodings

| Encoding | Physical System | Dominant Error | Threshold |
|----------|-----------------|----------------|-----------|
| GKP | Oscillator | Shift | ~10 dB squeezing |
| Cat | Oscillator | Photon loss | ~1% loss |
| Dual-rail | Photons | Loss | Very low |
| Superconducting | Transmon | T1, T2 | ~1% |

## Worked Examples

### Example 1: Stabilizer Eigenvalue

**Problem:** Verify that $|0_L\rangle = \sum_n |2n\sqrt{\pi}\rangle$ is a +1 eigenstate of $\hat{S}_q = e^{2i\sqrt{\pi}\hat{p}}$.

**Solution:**
The operator $\hat{S}_q$ shifts position by $2\sqrt{\pi}$:
$$\hat{S}_q|q\rangle = |q + 2\sqrt{\pi}\rangle$$

Acting on $|0_L\rangle$:
$$\hat{S}_q|0_L\rangle = \sum_n \hat{S}_q|2n\sqrt{\pi}\rangle = \sum_n |2n\sqrt{\pi} + 2\sqrt{\pi}\rangle$$

$$= \sum_n |2(n+1)\sqrt{\pi}\rangle = \sum_{m} |2m\sqrt{\pi}\rangle = |0_L\rangle$$

where we relabeled $m = n + 1$.

$$\boxed{\hat{S}_q|0_L\rangle = |0_L\rangle \checkmark}$$

### Example 2: Error Correction

**Problem:** A GKP state experiences a shift error $\epsilon_q = 0.3\sqrt{\pi}$. Describe the error correction procedure.

**Solution:**
**Step 1: Syndrome Measurement**
Measure $q \mod 2\sqrt{\pi}$.

For the shifted state, the peaks are at:
- $|0_L\rangle$: $q = (2n + 0.3)\sqrt{\pi}$
- $|1_L\rangle$: $q = (2n + 1 + 0.3)\sqrt{\pi} = (2n + 1.3)\sqrt{\pi}$

The syndrome (modular position) gives $s_q = 0.3\sqrt{\pi}$.

**Step 2: Error Estimation**
Since $|s_q| < \sqrt{\pi}/2$, we estimate $\epsilon_q \approx s_q = 0.3\sqrt{\pi}$.

**Step 3: Correction**
Apply displacement $\hat{D}(-i \cdot 0.3\sqrt{\pi}/\sqrt{2})$ to shift by $-0.3\sqrt{\pi}$ in position.

**Step 4: Result**
The state is restored to within the correctable region.

$$\boxed{\text{Error corrected: shift by } -0.3\sqrt{\pi} \text{ in } q}$$

### Example 3: Approximate GKP Photon Number

**Problem:** Calculate the mean photon number for an approximate GKP state with peak width $\Delta = 0.25$.

**Solution:**
For approximate GKP states, the mean photon number is approximately:
$$\langle\hat{n}\rangle \approx \frac{1}{4\Delta^2} - \frac{1}{2}$$

With $\Delta = 0.25$:
$$\langle\hat{n}\rangle \approx \frac{1}{4 \times 0.0625} - \frac{1}{2} = \frac{1}{0.25} - 0.5 = 4 - 0.5 = 3.5$$

The corresponding squeezing parameter:
$$r = -\ln(\Delta) = -\ln(0.25) \approx 1.39$$

In dB:
$$S_{dB} = 20r\log_{10}(e) \approx 12.0 \text{ dB}$$

$$\boxed{\langle\hat{n}\rangle \approx 3.5 \text{ photons}, \quad S \approx 12 \text{ dB}}$$

## Practice Problems

### Level 1: Direct Application

1. **Logical States in Momentum**

   Write out $|+_L\rangle$ and $|-_L\rangle$ in the momentum basis.

2. **Stabilizer Commutation**

   Show explicitly that $[\hat{S}_q, \hat{S}_p] = 0$ using the Baker-Campbell-Hausdorff formula.

3. **Logical X Action**

   Verify that $\hat{X}_L|1_L\rangle = |0_L\rangle$ for ideal GKP states.

### Level 2: Intermediate

4. **Correctable Error Region**

   A GKP state experiences a random shift with $\epsilon_q, \epsilon_p$ drawn from Gaussian distributions with standard deviation $\sigma = 0.2\sqrt{\pi}$. What fraction of errors are correctable?

5. **Hadamard on GKP**

   Show that the Fourier transform (Hadamard) takes $|0_L\rangle \to |+_L\rangle$ for ideal GKP states.

6. **Approximate State Overlap**

   Calculate the overlap $|\langle 0_L^\Delta | 1_L^\Delta \rangle|^2$ for approximate GKP states with peak width $\Delta = 0.3$. This determines the logical error rate from finite squeezing.

### Level 3: Challenging

7. **GKP-CNOT Implementation**

   For two GKP modes, show that the SUM gate $e^{-i\hat{q}_1\hat{p}_2}$ implements a logical CNOT. Verify by checking the action on all four computational basis states.

8. **Noise Threshold**

   For a GKP state with $\Delta = 0.2$, calculate the probability that a Gaussian shift error with $\sigma_q = \sigma_p = \sigma$ causes a logical error. Find the threshold $\sigma_c$ below which the logical error rate is less than 1%.

9. **Finite Energy Stabilizers**

   For approximate GKP states, the stabilizer eigenvalues are not exactly 1. Calculate $\langle 0_L^\Delta | \hat{S}_q | 0_L^\Delta \rangle$ as a function of $\Delta$.

## Computational Lab: GKP State Simulation

```python
"""
Day 929 Computational Lab: GKP Encoding
Simulation of GKP states and error correction
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from typing import Tuple, List

# Fundamental constants
SQRT_PI = np.sqrt(np.pi)


def ideal_gkp_wavefunction(q: np.ndarray, logical: int = 0,
                           n_peaks: int = 10) -> np.ndarray:
    """
    Approximate ideal GKP state wavefunction (sum of narrow Gaussians).
    logical: 0 for |0_L⟩, 1 for |1_L⟩
    """
    delta = 0.01  # Very narrow peaks to approximate delta functions
    psi = np.zeros_like(q, dtype=complex)

    for n in range(-n_peaks, n_peaks + 1):
        if logical == 0:
            center = 2 * n * SQRT_PI
        else:
            center = (2 * n + 1) * SQRT_PI

        psi += np.exp(-(q - center)**2 / (2 * delta**2))

    # Normalize (approximate)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * (q[1] - q[0]))
    return psi


def approximate_gkp_wavefunction(q: np.ndarray, logical: int = 0,
                                  Delta: float = 0.3, n_peaks: int = 5) -> np.ndarray:
    """
    Physical (approximate) GKP state wavefunction.
    Delta: peak width (squeezing parameter)
    """
    psi = np.zeros_like(q, dtype=complex)

    # Envelope parameter
    kappa = Delta**2

    for n in range(-n_peaks, n_peaks + 1):
        if logical == 0:
            center = 2 * n * SQRT_PI
        else:
            center = (2 * n + 1) * SQRT_PI

        # Gaussian peak with envelope
        envelope = np.exp(-kappa * center**2 / 2)
        peak = np.exp(-(q - center)**2 / (2 * Delta**2))
        psi += envelope * peak

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (q[1] - q[0]))
    return psi / norm


def gkp_wigner_function(Q: np.ndarray, P: np.ndarray,
                        logical: int = 0, Delta: float = 0.3,
                        n_peaks: int = 4) -> np.ndarray:
    """
    Wigner function for approximate GKP state.
    """
    W = np.zeros_like(Q)
    kappa = Delta**2

    for n in range(-n_peaks, n_peaks + 1):
        for m in range(-2*n_peaks, 2*n_peaks + 1):
            if logical == 0:
                qc = 2 * n * SQRT_PI
            else:
                qc = (2 * n + 1) * SQRT_PI

            pc = m * SQRT_PI

            # Envelope
            if logical == 0:
                envelope = np.exp(-kappa * ((2*n)**2 + m**2) * np.pi / 2)
            else:
                envelope = np.exp(-kappa * ((2*n+1)**2 + m**2) * np.pi / 2)

            # Gaussian blob
            W += envelope * np.exp(-((Q - qc)**2 / Delta**2 +
                                      (P - pc)**2 * Delta**2))

    # Normalize
    dq = Q[0, 1] - Q[0, 0]
    dp = P[1, 0] - P[0, 0]
    W /= (np.sum(W) * dq * dp)

    return W


def visualize_gkp_states():
    """Visualize GKP logical states in position space."""
    print("=" * 60)
    print("GKP State Visualization")
    print("=" * 60)

    q = np.linspace(-10, 10, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ideal-like states (very narrow peaks)
    psi_0_ideal = ideal_gkp_wavefunction(q, logical=0)
    psi_1_ideal = ideal_gkp_wavefunction(q, logical=1)

    axes[0, 0].plot(q / SQRT_PI, np.abs(psi_0_ideal)**2, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('q/√π', fontsize=12)
    axes[0, 0].set_ylabel('|ψ(q)|²', fontsize=12)
    axes[0, 0].set_title('Ideal GKP |0_L⟩', fontsize=14)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)

    axes[0, 1].plot(q / SQRT_PI, np.abs(psi_1_ideal)**2, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('q/√π', fontsize=12)
    axes[0, 1].set_ylabel('|ψ(q)|²', fontsize=12)
    axes[0, 1].set_title('Ideal GKP |1_L⟩', fontsize=14)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)

    # Approximate states (different Delta values)
    for Delta, style in [(0.4, 'b-'), (0.25, 'r-'), (0.15, 'g-')]:
        psi_0 = approximate_gkp_wavefunction(q, logical=0, Delta=Delta)
        psi_1 = approximate_gkp_wavefunction(q, logical=1, Delta=Delta)

        axes[1, 0].plot(q / SQRT_PI, np.abs(psi_0)**2, style,
                       linewidth=1.5, label=f'Δ={Delta}')
        axes[1, 1].plot(q / SQRT_PI, np.abs(psi_1)**2, style,
                       linewidth=1.5, label=f'Δ={Delta}')

    axes[1, 0].set_xlabel('q/√π', fontsize=12)
    axes[1, 0].set_ylabel('|ψ(q)|²', fontsize=12)
    axes[1, 0].set_title('Approximate GKP |0_L⟩', fontsize=14)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('q/√π', fontsize=12)
    axes[1, 1].set_ylabel('|ψ(q)|²', fontsize=12)
    axes[1, 1].set_title('Approximate GKP |1_L⟩', fontsize=14)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('gkp_wavefunctions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gkp_wavefunctions.png")


def visualize_gkp_wigner():
    """Visualize GKP Wigner functions."""
    print("\n" + "=" * 60)
    print("GKP Wigner Function Visualization")
    print("=" * 60)

    # Phase space grid
    q = np.linspace(-8, 8, 200)
    p = np.linspace(-8, 8, 200)
    Q, P = np.meshgrid(q, p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, logical in enumerate([0, 1]):
        W = gkp_wigner_function(Q, P, logical=logical, Delta=0.25)

        im = axes[idx].contourf(Q / SQRT_PI, P / SQRT_PI, W,
                                levels=30, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[idx], label='W(q,p)')

        # Draw grid lines at GKP lattice points
        for n in range(-3, 4):
            if logical == 0:
                axes[idx].axvline(x=2*n, color='gray', alpha=0.3, linestyle='--')
            else:
                axes[idx].axvline(x=2*n+1, color='gray', alpha=0.3, linestyle='--')
            axes[idx].axhline(y=n, color='gray', alpha=0.3, linestyle='--')

        axes[idx].set_xlabel('q/√π', fontsize=12)
        axes[idx].set_ylabel('p/√π', fontsize=12)
        axes[idx].set_title(f'GKP |{logical}_L⟩ Wigner Function (Δ=0.25)', fontsize=14)
        axes[idx].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('gkp_wigner.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gkp_wigner.png")


def error_correction_demo():
    """Demonstrate GKP error correction."""
    print("\n" + "=" * 60)
    print("GKP Error Correction Demonstration")
    print("=" * 60)

    q = np.linspace(-10, 10, 2000)
    Delta = 0.25

    # Original state
    psi_0 = approximate_gkp_wavefunction(q, logical=0, Delta=Delta)

    # Apply shift error
    shift_q = 0.3 * SQRT_PI  # Correctable error
    dq = q[1] - q[0]
    shift_bins = int(shift_q / dq)
    psi_shifted = np.roll(psi_0, shift_bins)

    # Syndrome measurement (position mod 2√π)
    # For simulation, find peak positions
    peaks = []
    for i in range(1, len(q) - 1):
        if (np.abs(psi_shifted[i])**2 > np.abs(psi_shifted[i-1])**2 and
            np.abs(psi_shifted[i])**2 > np.abs(psi_shifted[i+1])**2 and
            np.abs(psi_shifted[i])**2 > 0.01):
            peaks.append(q[i])

    # Estimate error from syndrome
    if peaks:
        # Find modular position
        mod_positions = [p % (2 * SQRT_PI) for p in peaks]
        # Average syndrome
        avg_syndrome = np.mean(mod_positions)
        if avg_syndrome > SQRT_PI:
            avg_syndrome -= 2 * SQRT_PI

        print(f"Applied shift: {shift_q/SQRT_PI:.3f}√π")
        print(f"Detected syndrome: {avg_syndrome/SQRT_PI:.3f}√π")

        # Correction
        correction = -avg_syndrome
        correction_bins = int(-correction / dq)
        psi_corrected = np.roll(psi_shifted, correction_bins)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(q / SQRT_PI, np.abs(psi_0)**2, 'b-', linewidth=1.5)
        axes[0].set_xlabel('q/√π', fontsize=12)
        axes[0].set_ylabel('|ψ(q)|²', fontsize=12)
        axes[0].set_title('Original |0_L⟩', fontsize=14)
        axes[0].set_xlim(-6, 6)

        axes[1].plot(q / SQRT_PI, np.abs(psi_shifted)**2, 'r-', linewidth=1.5)
        axes[1].set_xlabel('q/√π', fontsize=12)
        axes[1].set_title(f'After Shift Error ({shift_q/SQRT_PI:.2f}√π)', fontsize=14)
        axes[1].set_xlim(-6, 6)

        axes[2].plot(q / SQRT_PI, np.abs(psi_corrected)**2, 'g-', linewidth=1.5)
        axes[2].set_xlabel('q/√π', fontsize=12)
        axes[2].set_title('After Correction', fontsize=14)
        axes[2].set_xlim(-6, 6)

        plt.tight_layout()
        plt.savefig('gkp_error_correction.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: gkp_error_correction.png")

        # Calculate fidelity
        fidelity = np.abs(np.sum(np.conj(psi_0) * psi_corrected) * dq)**2
        print(f"Fidelity after correction: {fidelity:.4f}")


def logical_error_rate_simulation():
    """Simulate logical error rates vs noise strength."""
    print("\n" + "=" * 60)
    print("Logical Error Rate Analysis")
    print("=" * 60)

    # Parameters
    n_trials = 10000
    Delta_values = [0.4, 0.3, 0.25, 0.2, 0.15]
    sigma_values = np.linspace(0.05, 0.5, 20)  # In units of √π

    error_rates = {Delta: [] for Delta in Delta_values}

    for Delta in Delta_values:
        for sigma in sigma_values:
            n_logical_errors = 0

            for _ in range(n_trials):
                # Random shift error
                eps_q = np.random.normal(0, sigma * SQRT_PI)
                eps_p = np.random.normal(0, sigma * SQRT_PI)

                # Syndrome (modular measurement with uncertainty Delta)
                measurement_noise_q = np.random.normal(0, Delta)
                measurement_noise_p = np.random.normal(0, Delta)

                syndrome_q = (eps_q + measurement_noise_q) % (2 * SQRT_PI)
                if syndrome_q > SQRT_PI:
                    syndrome_q -= 2 * SQRT_PI

                syndrome_p = (eps_p + measurement_noise_p) % SQRT_PI
                if syndrome_p > SQRT_PI / 2:
                    syndrome_p -= SQRT_PI

                # Correction
                corrected_q = eps_q - syndrome_q
                corrected_p = eps_p - syndrome_p

                # Check for logical error
                if (np.abs(corrected_q) > SQRT_PI / 2 or
                    np.abs(corrected_p) > SQRT_PI / 2):
                    n_logical_errors += 1

            error_rate = n_logical_errors / n_trials
            error_rates[Delta].append(error_rate)

    # Plot
    plt.figure(figsize=(10, 6))
    for Delta in Delta_values:
        squeezing_db = -20 * np.log10(Delta)
        plt.semilogy(sigma_values, error_rates[Delta], 'o-',
                    label=f'Δ={Delta} ({squeezing_db:.1f} dB)', markersize=4)

    plt.xlabel('Noise σ (in units of √π)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('GKP Logical Error Rate vs Noise Strength', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    plt.savefig('gkp_logical_error_rate.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gkp_logical_error_rate.png")


def photon_number_analysis():
    """Analyze photon number for approximate GKP states."""
    print("\n" + "=" * 60)
    print("GKP Photon Number Analysis")
    print("=" * 60)

    Delta_values = np.linspace(0.1, 0.5, 50)

    # Approximate formula
    n_mean_approx = 1 / (4 * Delta_values**2) - 0.5

    # Squeezing in dB
    squeezing_db = -20 * np.log10(Delta_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(Delta_values, n_mean_approx, 'b-', linewidth=2)
    axes[0].set_xlabel('Peak width Δ', fontsize=12)
    axes[0].set_ylabel('Mean photon number ⟨n⟩', fontsize=12)
    axes[0].set_title('GKP Photon Number vs Peak Width', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.1, 0.5)

    axes[1].plot(squeezing_db, n_mean_approx, 'r-', linewidth=2)
    axes[1].set_xlabel('Equivalent squeezing (dB)', fontsize=12)
    axes[1].set_ylabel('Mean photon number ⟨n⟩', fontsize=12)
    axes[1].set_title('GKP Photon Number vs Squeezing', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gkp_photon_number.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gkp_photon_number.png")

    # Print table
    print("\nGKP State Parameters:")
    print("-" * 40)
    print(f"{'Δ':>8} {'Squeezing (dB)':>15} {'⟨n⟩':>8}")
    print("-" * 40)
    for Delta in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        sq = -20 * np.log10(Delta)
        n = 1 / (4 * Delta**2) - 0.5
        print(f"{Delta:>8.2f} {sq:>15.1f} {n:>8.1f}")


def stabilizer_expectation():
    """Calculate stabilizer expectation values for approximate GKP."""
    print("\n" + "=" * 60)
    print("Stabilizer Expectation Values")
    print("=" * 60)

    Delta_values = np.linspace(0.1, 0.5, 50)

    # For approximate GKP, ⟨S_q⟩ = exp(-2π Δ²)
    S_q_expectation = np.exp(-2 * np.pi * Delta_values**2)

    plt.figure(figsize=(10, 6))
    plt.plot(Delta_values, S_q_expectation, 'b-', linewidth=2)
    plt.xlabel('Peak width Δ', fontsize=12)
    plt.ylabel('⟨S_q⟩', fontsize=12)
    plt.title('Stabilizer Expectation Value for Approximate GKP', fontsize=14)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Ideal GKP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gkp_stabilizer_expectation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gkp_stabilizer_expectation.png")


def main():
    """Run all GKP simulations."""
    print("\n" + "=" * 60)
    print("DAY 929: GKP ENCODING SIMULATIONS")
    print("=" * 60)

    visualize_gkp_states()
    visualize_gkp_wigner()
    error_correction_demo()
    logical_error_rate_simulation()
    photon_number_analysis()
    stabilizer_expectation()

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
| GKP logical states | $\|0_L\rangle = \sum_n \|2n\sqrt{\pi}\rangle$, $\|1_L\rangle = \sum_n \|(2n+1)\sqrt{\pi}\rangle$ |
| Stabilizers | $\hat{S}_q = e^{2i\sqrt{\pi}\hat{p}}$, $\hat{S}_p = e^{-2i\sqrt{\pi}\hat{q}}$ |
| Logical X | $\hat{X}_L = e^{i\sqrt{\pi}\hat{p}}$ |
| Logical Z | $\hat{Z}_L = e^{-i\sqrt{\pi}\hat{q}}$ |
| Correctable region | $\|\epsilon_q\| < \sqrt{\pi}/2$, $\|\epsilon_p\| < \sqrt{\pi}/2$ |
| Mean photon number | $\langle n \rangle \approx 1/(4\Delta^2) - 1/2$ |

### Key Takeaways

1. **GKP encoding** embeds a discrete qubit in a continuous-variable oscillator
2. The logical states form a **periodic grid** in phase space
3. **Shift errors** can be detected via modular quadrature measurement
4. Errors within a **Voronoi cell** ($\sqrt{\pi}/2$ radius) are correctable
5. **Physical GKP states** require high squeezing (>10 dB for useful encoding)
6. GKP enables **concatenation** with standard qubit error-correcting codes

## Daily Checklist

- [ ] I can construct ideal GKP logical states in position and momentum bases
- [ ] I understand the stabilizer formalism for GKP codes
- [ ] I can explain the error correction mechanism for shift errors
- [ ] I know the resource requirements for physical GKP states
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 930

Tomorrow we explore **Cat States and Bosonic Codes**, another approach to encoding qubits in oscillators using superpositions of coherent states. Key topics:
- Cat state definition and properties
- Kerr nonlinearity for cat state preparation
- Dissipative preparation schemes
- Error correction with cat codes
