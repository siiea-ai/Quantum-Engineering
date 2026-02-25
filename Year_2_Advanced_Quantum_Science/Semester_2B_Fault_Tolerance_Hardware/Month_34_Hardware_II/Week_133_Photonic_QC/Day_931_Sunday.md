# Day 931: Integrated Photonics for Quantum Computing

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Silicon photonics fundamentals, waveguides, components |
| Afternoon | 2.5 hours | Problem solving: Chip design and performance analysis |
| Evening | 1.5 hours | Computational lab: Photonic circuit simulation |

## Learning Objectives

By the end of today, you will be able to:

1. Explain the physics of optical waveguides and integrated photonic components
2. Describe silicon photonics fabrication and its advantages for quantum computing
3. Analyze on-chip single-photon sources and detection schemes
4. Design programmable photonic processors using mesh architectures
5. Evaluate industry implementations and their trade-offs
6. Simulate integrated photonic circuits for quantum applications

## Core Content

### 1. Optical Waveguides: The Foundation

**Total Internal Reflection:**
Light is confined in a waveguide when the core has higher refractive index than the cladding:
$$n_{core} > n_{cladding}$$

Critical angle:
$$\theta_c = \sin^{-1}\left(\frac{n_{cladding}}{n_{core}}\right)$$

**Single-Mode Condition:**
For single-mode operation at wavelength $\lambda$:
$$V = \frac{2\pi a}{\lambda}\sqrt{n_{core}^2 - n_{cladding}^2} < 2.405$$

where $a$ is the waveguide core dimension.

**Common Materials:**

| Material | n @ 1550nm | Loss (dB/cm) | Nonlinearity |
|----------|------------|--------------|--------------|
| Silicon | 3.48 | 0.5-2 | High (Kerr) |
| Si₃N₄ | 2.0 | 0.01-0.1 | Low |
| SiO₂ | 1.45 | <0.01 | Very low |
| LiNbO₃ | 2.2 | 0.1 | High (χ²) |

**Silicon-on-Insulator (SOI):**
Standard platform for integrated photonics:
- Silicon core (220 nm thick)
- SiO₂ buried oxide (2-3 μm)
- Tight confinement: bend radius ~5 μm

### 2. Integrated Optical Components

**Directional Coupler (Beam Splitter):**
Two waveguides brought close together for evanescent coupling:
$$\kappa = \frac{\pi}{2L_c}$$

where $L_c$ is the coupling length.

Transfer matrix:
$$T = \begin{pmatrix} \cos(\kappa L) & i\sin(\kappa L) \\ i\sin(\kappa L) & \cos(\kappa L) \end{pmatrix}$$

50:50 splitting when $\kappa L = \pi/4$.

**Mach-Zehnder Interferometer (MZI):**
Two directional couplers with phase shifter in between:
$$U_{MZI}(\theta, \phi) = \begin{pmatrix} e^{i\phi}\cos\theta & i\sin\theta \\ ie^{i\phi}\sin\theta & \cos\theta \end{pmatrix}$$

Universal for single-mode operations.

**Phase Shifters:**
1. **Thermo-optic:** Heat changes refractive index
   $$\Delta n = \frac{dn}{dT}\Delta T$$
   - Slow (~μs), high power (~10 mW for π shift)

2. **Electro-optic:** Electric field changes index
   $$\Delta n = \frac{1}{2}n^3 r_{ij} E$$
   - Fast (~GHz), requires special materials (LiNbO₃)

**Ring Resonators:**
Circular waveguide coupled to bus waveguide:
$$T(f) = \frac{t - ae^{i\phi}}{1 - ta^*e^{i\phi}}$$

where $t$ is coupling coefficient, $a$ is round-trip loss, $\phi = 2\pi n_{eff}L/\lambda$.

Applications: filtering, delay lines, nonlinear enhancement.

### 3. On-Chip Single-Photon Sources

**Spontaneous Parametric Down-Conversion (SPDC):**
Pump photon → signal + idler photons
$$\omega_p = \omega_s + \omega_i$$
$$\vec{k}_p = \vec{k}_s + \vec{k}_i$$ (phase matching)

In waveguides: quasi-phase matching using periodic poling (PPLN, PPKTP).

**Spontaneous Four-Wave Mixing (SFWM):**
Two pump photons → signal + idler
$$2\omega_p = \omega_s + \omega_i$$

Enabled by Kerr nonlinearity in silicon or Si₃N₄.

**Quantum Dot Sources:**
Embedded in photonic crystal cavities:
- Near-unity efficiency possible
- Deterministic (on-demand)
- Requires cryogenic operation

**Source Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| Brightness | Pairs per pump power | >10⁶/s/mW |
| Heralding efficiency | P(signal | idler detected) | >90% |
| Indistinguishability | HOM visibility | >99% |
| g²(0) | Second-order correlation | <0.01 |

### 4. On-Chip Photon Detection

**Superconducting Nanowire SPDs (SNSPDs):**
- Efficiency: >95% at 1550 nm
- Timing jitter: <20 ps
- Dark count: <1 Hz
- Requires cryogenic operation (2-4 K)

**Waveguide-Integrated SNSPDs:**
NbN or WSi nanowires on top of waveguide:
- Absorption length: ~10-50 μm
- Near-unity on-chip detection efficiency

**Transition Edge Sensors (TES):**
- Photon-number resolving
- Lower efficiency than SNSPDs
- Slower (μs recovery)

**Detection Challenges:**
1. Cryogenic integration with room-temp sources
2. Readout multiplexing for many detectors
3. Dead time and recovery

### 5. Programmable Photonic Processors

**Mesh Architectures:**

**Reck Decomposition:**
Triangular mesh of MZIs:
- $N(N-1)/2$ MZIs for $N \times N$ unitary
- Depth: $2N-3$ layers

**Clements Decomposition:**
Rectangular mesh:
- Same MZI count
- Balanced depth: $N$ layers
- Better for fabrication tolerances

**Universal Photonic Chip:**
Program any linear optical unitary by setting MZI phases:
$$U = \prod_{layers} \left(\prod_{MZIs} U_{MZI}(\theta_i, \phi_i)\right)$$

**Commercial Systems:**
- Quandela: up to 12 modes
- QuiX Quantum: 12-20 modes
- Xanadu: specialized for GBS

### 6. Silicon Photonics Fabrication

**CMOS Compatibility:**
Silicon photonics uses standard semiconductor fabs:
- 193 nm or 248 nm lithography
- 200 mm or 300 mm wafers
- Cost: ~$1000/chip at scale

**Process Flow:**
1. SOI wafer preparation
2. Waveguide patterning (etching)
3. Cladding deposition
4. Metal heater deposition
5. Fiber coupling (edge or grating)

**Critical Dimensions:**
- Waveguide width: 400-500 nm
- Etch depth: 220 nm
- Gap in couplers: 100-200 nm

**Challenges for Quantum:**
1. Propagation loss reduces photon survival
2. Phase stability (thermal fluctuations)
3. Nonlinear losses at high power
4. Fiber-to-chip coupling loss

### 7. Photonic Quantum Computing Architectures

**Measurement-Based QC (MBQC):**
1. Generate large cluster state offline
2. Perform computation via measurements
3. Feed-forward classical information

**Fusion-Based QC (FBQC):**
1. Generate small entangled resource states
2. Fuse them into larger clusters
3. Use percolation above threshold

**Xanadu's Architecture:**
- Gaussian boson sampling as intermediate step
- GKP encoding for fault tolerance
- Squeezed light + linear optics + homodyne

**PsiQuantum's Architecture:**
- Silicon photonics at scale
- Single-photon sources and detectors
- Fusion gates for entanglement
- Target: millions of physical qubits

### 8. Industry Implementations

**PsiQuantum (USA):**
- Founded 2016
- Approach: Fault-tolerant, fusion-based
- Fab partner: GlobalFoundries
- Goal: 1 million+ qubit system

**Xanadu (Canada):**
- Founded 2016
- Approach: CV + GBS + GKP
- Borealis: 216 squeezed modes (2022)
- Cloud access: PennyLane + photonic hardware

**QuiX Quantum (Netherlands):**
- Founded 2019
- Product: Universal photonic processor
- 12-20 mode chips commercially available
- Si₃N₄ platform (low loss)

**Quandela (France):**
- Founded 2017
- Focus: Single-photon sources
- Quantum dot technology
- Integrated with photonic chips

**ORCA Computing (UK):**
- Founded 2019
- Approach: Quantum memories + photons
- Room-temperature operation

### 9. Challenges and Outlook

**Current Limitations:**

| Challenge | Status | Target |
|-----------|--------|--------|
| Photon loss | 1-10% per component | <0.1% |
| Source efficiency | 50-80% | >99% |
| Detection efficiency | 90-95% | >99.9% |
| Mode count | 10-100 | >10,000 |
| Clock rate | 1 MHz | 1 GHz |

**Path to Fault Tolerance:**
1. Improve component performance
2. Scale up mode count
3. Integrate error correction (GKP, surface code)
4. Demonstrate logical qubit operations

**Timeline Estimates:**
- 2025-2027: First logical qubit demonstrations
- 2028-2030: Small fault-tolerant systems
- 2030+: Commercially useful quantum computers

## Quantum Computing Applications

### Near-Term Applications

**Gaussian Boson Sampling:**
- Graph optimization problems
- Molecular vibronic spectra
- Machine learning kernels

**Quantum Communication:**
- Quantum key distribution (QKD)
- Quantum repeaters
- Quantum networks

### Long-Term Vision

**Fault-Tolerant Quantum Computing:**
- Drug discovery
- Materials simulation
- Cryptanalysis
- Optimization

**Quantum Internet:**
- Distributed quantum computing
- Secure communication networks
- Quantum sensing networks

## Worked Examples

### Example 1: Directional Coupler Design

**Problem:** Design a 50:50 directional coupler in silicon (n=3.48) at 1550 nm with gap g=200 nm.

**Solution:**
The coupling coefficient depends on waveguide geometry. For typical SOI:
$$\kappa \approx k_0 \frac{n_{eff}^2 - n_{clad}^2}{2n_{eff}} e^{-\gamma g}$$

where $\gamma$ is the evanescent decay constant.

For silicon waveguides at 200 nm gap: $\kappa \approx 0.1$ rad/μm (typical value)

For 50:50 splitting: $\kappa L = \pi/4$
$$L = \frac{\pi}{4\kappa} = \frac{\pi}{4 \times 0.1} = 7.85 \text{ μm}$$

Rounding up:
$$\boxed{L \approx 8 \text{ μm}}$$

### Example 2: Loss Budget

**Problem:** Calculate the total transmission for a photonic circuit with:
- Fiber-to-chip coupling: 2 dB loss per facet
- 10 MZIs each with 0.1 dB loss
- 50 cm total waveguide at 1 dB/cm
- Detector efficiency: 90%

**Solution:**
Total loss in dB:
$$L_{total} = 2 \times 2 + 10 \times 0.1 + 0.5 \times 1 = 4 + 1 + 0.5 = 5.5 \text{ dB}$$

Optical transmission:
$$T = 10^{-5.5/10} = 0.28$$

Including detector:
$$\eta_{total} = 0.28 \times 0.90 = 0.25$$

$$\boxed{\eta_{total} = 25\%}$$

This means 75% of photons are lost - a major challenge!

### Example 3: Programmable Unitary

**Problem:** How many phase shifters are needed to implement any 8x8 unitary on a Clements mesh?

**Solution:**
For an $N \times N$ unitary using Clements decomposition:
- Number of MZIs: $N(N-1)/2$
- Each MZI has 2 phase shifters (internal + external)
- Plus $N$ output phase shifters

For $N = 8$:
$$N_{MZI} = \frac{8 \times 7}{2} = 28$$

Phase shifters:
$$N_{phases} = 2 \times 28 + 8 = 64$$

$$\boxed{64 \text{ phase shifters}}$$

## Practice Problems

### Level 1: Direct Application

1. **Single-Mode Condition**

   For a silicon nitride waveguide (n=2.0) with SiO₂ cladding (n=1.45), what is the maximum core width for single-mode operation at 1550 nm?

2. **MZI Transfer Function**

   An MZI has internal phase $\theta = \pi/4$ and external phase $\phi = 0$. Calculate the transfer matrix and the power splitting ratio.

3. **Loss in dB**

   A waveguide has 0.5 dB/cm loss. What fraction of photons survive after 10 cm?

### Level 2: Intermediate

4. **Phase Shifter Power**

   Silicon has thermo-optic coefficient $dn/dT = 1.8 \times 10^{-4}$ K⁻¹. For a 100 μm long phase shifter, calculate the temperature change needed for a π phase shift at 1550 nm.

5. **SFWM Photon Rate**

   A silicon waveguide produces photon pairs via SFWM with rate $R = \gamma P^2 L^2$ where $\gamma = 200$ (W·m)⁻¹, $P = 1$ mW, $L = 1$ cm. Calculate the pair generation rate in MHz.

6. **Mesh Depth**

   Compare the circuit depth (number of layers) for Reck vs Clements decompositions of a 16-mode unitary. Which is better for reducing errors from sequential operations?

### Level 3: Challenging

7. **Fault-Tolerant Threshold**

   For a photonic fusion-based architecture, the loss threshold is approximately 10% per photon. If your chip has 2 dB fiber coupling, 0.5 dB/cm waveguide loss, and 90% detection efficiency:
   a) What is the maximum allowed waveguide length?
   b) How does this constrain chip design?

8. **GBS Advantage Scaling**

   For Gaussian boson sampling to maintain quantum advantage, the sampling must be faster than classical simulation. If classical time scales as $O(n^2 \cdot 2^n)$ for $n$ modes, and your photonic chip operates at 1 MHz, estimate the minimum $n$ for quantum advantage against a 10 PFLOP supercomputer.

9. **Integrated Source-Detector System**

   Design a complete on-chip two-photon interference experiment:
   a) Specify source type and expected pair rate
   b) Design the beam splitter and path matching
   c) Calculate expected HOM visibility given realistic component losses
   d) Estimate total system efficiency

## Computational Lab: Photonic Circuit Simulation

```python
"""
Day 931 Computational Lab: Integrated Photonic Circuit Simulation
Design and analysis of programmable photonic processors
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Physical constants
C = 3e8  # Speed of light (m/s)
WAVELENGTH = 1550e-9  # Telecom wavelength (m)


class MZI:
    """Mach-Zehnder Interferometer building block."""

    def __init__(self, theta: float = 0, phi: float = 0):
        """
        Initialize MZI with internal phase theta and external phase phi.
        """
        self.theta = theta
        self.phi = phi

    def matrix(self) -> np.ndarray:
        """Return 2x2 transfer matrix."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [np.exp(1j * self.phi) * c, 1j * s],
            [1j * np.exp(1j * self.phi) * s, c]
        ], dtype=complex)

    def set_splitting(self, ratio: float):
        """Set power splitting ratio (0 to 1)."""
        self.theta = np.arccos(np.sqrt(ratio))


class DirectionalCoupler:
    """Directional coupler model."""

    def __init__(self, coupling_length: float, kappa: float):
        """
        coupling_length: interaction length (m)
        kappa: coupling coefficient (rad/m)
        """
        self.L = coupling_length
        self.kappa = kappa

    def matrix(self) -> np.ndarray:
        """Return transfer matrix."""
        phi = self.kappa * self.L
        return np.array([
            [np.cos(phi), 1j * np.sin(phi)],
            [1j * np.sin(phi), np.cos(phi)]
        ], dtype=complex)

    def splitting_ratio(self) -> float:
        """Return cross-coupling power ratio."""
        return np.sin(self.kappa * self.L)**2


class PhotonicChip:
    """Programmable photonic processor using Clements mesh."""

    def __init__(self, n_modes: int):
        """Initialize n-mode photonic processor."""
        self.n_modes = n_modes
        self.n_mzis = n_modes * (n_modes - 1) // 2

        # Initialize MZI phases (random)
        self.mzis = []
        for _ in range(self.n_mzis):
            self.mzis.append(MZI(np.random.uniform(0, np.pi),
                                  np.random.uniform(0, 2*np.pi)))

        # Output phases
        self.output_phases = np.zeros(n_modes)

        # Loss per component (in linear scale)
        self.mzi_transmission = 0.98  # 98% per MZI
        self.waveguide_loss_per_cm = 0.01  # 1% per cm

    def set_unitary(self, U: np.ndarray):
        """
        Program the chip to implement unitary U.
        Uses Clements decomposition.
        """
        if U.shape != (self.n_modes, self.n_modes):
            raise ValueError(f"Unitary must be {self.n_modes}x{self.n_modes}")

        # Clements decomposition
        phases = self._clements_decompose(U)

        # Set MZI phases
        for i, (theta, phi) in enumerate(phases):
            self.mzis[i].theta = theta
            self.mzis[i].phi = phi

    def _clements_decompose(self, U: np.ndarray) -> List[Tuple[float, float]]:
        """
        Decompose unitary into MZI phases using Clements method.
        Simplified implementation - sets random phases for demo.
        """
        # In practice, this would compute the exact decomposition
        # Here we use a placeholder
        phases = []
        for _ in range(self.n_mzis):
            phases.append((np.random.uniform(0, np.pi),
                          np.random.uniform(0, 2*np.pi)))
        return phases

    def get_unitary(self, include_loss: bool = False) -> np.ndarray:
        """Calculate the implemented unitary matrix."""
        n = self.n_modes
        U = np.eye(n, dtype=complex)

        # Apply MZIs in Clements order
        mzi_idx = 0
        for layer in range(n):
            if layer % 2 == 0:  # Even layer
                for i in range(0, n - 1, 2):
                    U = self._apply_mzi(U, i, mzi_idx, include_loss)
                    mzi_idx += 1
            else:  # Odd layer
                for i in range(1, n - 1, 2):
                    U = self._apply_mzi(U, i, mzi_idx, include_loss)
                    mzi_idx += 1

        # Apply output phases
        for i in range(n):
            U[i, :] *= np.exp(1j * self.output_phases[i])

        return U

    def _apply_mzi(self, U: np.ndarray, mode: int, mzi_idx: int,
                   include_loss: bool) -> np.ndarray:
        """Apply MZI between mode and mode+1."""
        if mzi_idx >= len(self.mzis):
            return U

        mzi_matrix = self.mzis[mzi_idx].matrix()

        if include_loss:
            mzi_matrix *= np.sqrt(self.mzi_transmission)

        # Apply to modes
        U_new = U.copy()
        for j in range(U.shape[1]):
            vec = np.array([U[mode, j], U[mode + 1, j]])
            new_vec = mzi_matrix @ vec
            U_new[mode, j] = new_vec[0]
            U_new[mode + 1, j] = new_vec[1]

        return U_new


def simulate_hom_on_chip():
    """Simulate Hong-Ou-Mandel effect on integrated chip."""
    print("=" * 60)
    print("On-Chip Hong-Ou-Mandel Simulation")
    print("=" * 60)

    # Create 2-mode chip (just one MZI)
    chip = PhotonicChip(2)
    chip.mzis[0].theta = np.pi / 4  # 50:50 splitting
    chip.mzis[0].phi = np.pi / 2

    # Scan path length difference (simulated by phase)
    delays = np.linspace(-5, 5, 100)  # in units of coherence length
    coincidence_rates = []

    for delay in delays:
        # Distinguishability from path mismatch
        # HOM visibility decreases with delay
        visibility = np.exp(-delay**2)  # Gaussian coherence

        # Coincidence rate
        # For indistinguishable photons: 0
        # For distinguishable: 0.5
        p_coinc = 0.5 * (1 - visibility)
        coincidence_rates.append(p_coinc)

    plt.figure(figsize=(10, 6))
    plt.plot(delays, coincidence_rates, 'b-', linewidth=2)
    plt.xlabel('Path delay (coherence lengths)', fontsize=12)
    plt.ylabel('Coincidence rate (normalized)', fontsize=12)
    plt.title('Integrated HOM Dip', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Classical limit')
    plt.legend()
    plt.savefig('integrated_hom.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: integrated_hom.png")


def analyze_mesh_architecture():
    """Compare different mesh architectures."""
    print("\n" + "=" * 60)
    print("Photonic Mesh Architecture Analysis")
    print("=" * 60)

    modes = range(2, 17)

    reck_mzis = [n * (n-1) // 2 for n in modes]
    reck_depth = [2*n - 3 for n in modes]

    clements_mzis = [n * (n-1) // 2 for n in modes]
    clements_depth = [n for n in modes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(modes, reck_mzis, 'bo-', label='Reck', markersize=6)
    axes[0].plot(modes, clements_mzis, 'rs-', label='Clements', markersize=6)
    axes[0].set_xlabel('Number of modes', fontsize=12)
    axes[0].set_ylabel('Number of MZIs', fontsize=12)
    axes[0].set_title('MZI Count vs Modes', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(modes, reck_depth, 'bo-', label='Reck', markersize=6)
    axes[1].plot(modes, clements_depth, 'rs-', label='Clements', markersize=6)
    axes[1].set_xlabel('Number of modes', fontsize=12)
    axes[1].set_ylabel('Circuit depth (layers)', fontsize=12)
    axes[1].set_title('Circuit Depth vs Modes', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mesh_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: mesh_comparison.png")

    # Print table
    print("\nMesh Architecture Comparison:")
    print("-" * 50)
    print(f"{'Modes':>6} {'MZIs':>8} {'Reck Depth':>12} {'Clements Depth':>15}")
    print("-" * 50)
    for n in [4, 8, 12, 16]:
        print(f"{n:>6} {n*(n-1)//2:>8} {2*n-3:>12} {n:>15}")


def loss_budget_analysis():
    """Analyze loss budget for photonic quantum computing."""
    print("\n" + "=" * 60)
    print("Loss Budget Analysis")
    print("=" * 60)

    # Component losses (in dB)
    fiber_coupling = 2.0  # per facet
    waveguide_per_cm = 0.5  # dB/cm
    mzi_loss = 0.1  # per MZI
    detector_efficiency = 0.90  # linear

    # Scan number of modes
    modes = np.array([4, 8, 12, 16, 20, 24])

    # Assumptions
    chip_length = 1.0  # cm
    n_mzis = modes * (modes - 1) // 2

    total_loss_db = (2 * fiber_coupling +
                     chip_length * waveguide_per_cm +
                     n_mzis * mzi_loss)

    # Convert to transmission
    transmission = 10 ** (-total_loss_db / 10) * detector_efficiency

    plt.figure(figsize=(10, 6))
    plt.semilogy(modes, transmission, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Number of modes', fontsize=12)
    plt.ylabel('Total transmission (per photon)', fontsize=12)
    plt.title('Photonic Circuit Transmission vs Scale', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark thresholds
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='50% (usable)')
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% (FT threshold)')
    plt.legend()

    plt.savefig('loss_budget.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: loss_budget.png")

    # Print breakdown
    print("\nLoss Breakdown:")
    print("-" * 60)
    print(f"{'Component':>25} {'Loss':>15}")
    print("-" * 60)
    print(f"{'Fiber coupling (2 facets)':>25} {2*fiber_coupling:>12.1f} dB")
    print(f"{'Waveguide (1 cm)':>25} {chip_length * waveguide_per_cm:>12.1f} dB")
    print(f"{'Detection efficiency':>25} {-10*np.log10(detector_efficiency):>12.1f} dB")
    print("-" * 60)

    for n in modes:
        n_mzi = n * (n-1) // 2
        mzi_total = n_mzi * mzi_loss
        total = 2*fiber_coupling + chip_length*waveguide_per_cm + mzi_total
        trans = 10**(-total/10) * detector_efficiency
        print(f"{n:>3} modes ({n_mzi:>3} MZIs): {total:>6.1f} dB total, {trans*100:>5.1f}% transmission")


def programmable_unitary_demo():
    """Demonstrate programming a photonic chip."""
    print("\n" + "=" * 60)
    print("Programmable Unitary Demonstration")
    print("=" * 60)

    n_modes = 4

    # Create target unitaries
    # Hadamard-like
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H4 = np.kron(H, H)  # 4x4 Hadamard

    # DFT
    DFT4 = np.fft.fft(np.eye(4)) / 2

    # Random unitary
    from scipy.stats import unitary_group
    random_U = unitary_group.rvs(4)

    targets = [('Hadamard ⊗ Hadamard', H4),
               ('DFT', DFT4),
               ('Random', random_U)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (name, target_U) in enumerate(targets):
        # Create and program chip
        chip = PhotonicChip(n_modes)

        # For demo, we'll show target vs achieved
        # In practice, would use proper decomposition

        # Show target unitary magnitude
        im = axes[0, col].imshow(np.abs(target_U), cmap='viridis')
        axes[0, col].set_title(f'{name} (magnitude)', fontsize=12)
        plt.colorbar(im, ax=axes[0, col])

        # Show target unitary phase
        im = axes[1, col].imshow(np.angle(target_U), cmap='hsv')
        axes[1, col].set_title(f'{name} (phase)', fontsize=12)
        plt.colorbar(im, ax=axes[1, col])

    plt.tight_layout()
    plt.savefig('programmable_unitaries.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: programmable_unitaries.png")


def source_detector_integration():
    """Analyze integrated source-detector system."""
    print("\n" + "=" * 60)
    print("Integrated Source-Detector Analysis")
    print("=" * 60)

    # Source parameters
    pair_rates = np.logspace(4, 8, 50)  # pairs/s
    heralding_eff = 0.7

    # System losses
    waveguide_trans = 0.9
    mzi_trans = 0.98
    n_mzis = 10
    detector_eff = 0.9

    total_eff = waveguide_trans * (mzi_trans ** n_mzis) * detector_eff

    # Detected rates
    detected_rates = pair_rates * heralding_eff * total_eff

    # Multi-photon probability (unwanted)
    # If mean pairs per pulse μ, P(2+) ≈ μ²/2
    rep_rate = 1e6  # 1 MHz
    mean_pairs_per_pulse = pair_rates / rep_rate
    p_multi = mean_pairs_per_pulse**2 / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(pair_rates, detected_rates, 'b-', linewidth=2)
    axes[0].set_xlabel('Source pair rate (pairs/s)', fontsize=12)
    axes[0].set_ylabel('Detected heralded photons/s', fontsize=12)
    axes[0].set_title('Heralded Photon Detection Rate', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(pair_rates, p_multi, 'r-', linewidth=2)
    axes[1].set_xlabel('Source pair rate (pairs/s)', fontsize=12)
    axes[1].set_ylabel('Multi-pair probability', fontsize=12)
    axes[1].set_title('Multi-Photon Contamination', fontsize=14)
    axes[1].axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label='1% threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('source_detector_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: source_detector_tradeoff.png")

    # Find optimal operating point
    target_multi = 0.01
    optimal_rate = np.sqrt(2 * target_multi) * rep_rate
    optimal_detected = optimal_rate * heralding_eff * total_eff
    print(f"\nOptimal operating point (1% multi-photon):")
    print(f"  Pair generation rate: {optimal_rate:.2e} pairs/s")
    print(f"  Detected heralded rate: {optimal_detected:.2e} photons/s")


def industry_comparison():
    """Compare different photonic QC approaches."""
    print("\n" + "=" * 60)
    print("Industry Comparison")
    print("=" * 60)

    companies = ['PsiQuantum', 'Xanadu', 'QuiX', 'Quandela', 'ORCA']
    approaches = ['Fusion-based', 'GBS/CV', 'Universal LO', 'LOQC', 'Memory-based']
    modes = [100, 216, 20, 12, 4]  # Current demonstrated
    target_qubits = [1e6, 1e6, 1e4, 1e3, 1e4]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(companies))
    width = 0.35

    bars1 = ax.bar(x - width/2, modes, width, label='Current modes', color='steelblue')
    bars2 = ax.bar(x + width/2, np.log10(target_qubits), width,
                   label='Target (log10 qubits)', color='coral')

    ax.set_ylabel('Scale', fontsize=12)
    ax.set_title('Photonic QC Industry Landscape (2024)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(companies)
    ax.legend()

    # Add approach labels
    for i, approach in enumerate(approaches):
        ax.annotate(approach, (i, -15), ha='center', fontsize=9, rotation=45)

    plt.tight_layout()
    plt.savefig('industry_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: industry_comparison.png")


def main():
    """Run all integrated photonics simulations."""
    print("\n" + "=" * 60)
    print("DAY 931: INTEGRATED PHOTONICS SIMULATIONS")
    print("=" * 60)

    simulate_hom_on_chip()
    analyze_mesh_architecture()
    loss_budget_analysis()
    programmable_unitary_demo()
    source_detector_integration()
    industry_comparison()

    print("\n" + "=" * 60)
    print("Week 133: Photonic Quantum Computing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Single-mode condition | $V = \frac{2\pi a}{\lambda}\sqrt{n_{core}^2 - n_{clad}^2} < 2.405$ |
| Directional coupler | $T = \begin{pmatrix} \cos\kappa L & i\sin\kappa L \\ i\sin\kappa L & \cos\kappa L \end{pmatrix}$ |
| MZI transfer | $U_{MZI} = \begin{pmatrix} e^{i\phi}\cos\theta & i\sin\theta \\ ie^{i\phi}\sin\theta & \cos\theta \end{pmatrix}$ |
| MZI count (N modes) | $N_{MZI} = N(N-1)/2$ |
| Clements depth | $D = N$ layers |
| Thermo-optic shift | $\Delta\phi = \frac{2\pi L}{\lambda}\frac{dn}{dT}\Delta T$ |

### Key Takeaways

1. **Silicon photonics** enables scalable, CMOS-compatible quantum hardware
2. **MZIs and phase shifters** form universal programmable optical networks
3. **Loss is the primary challenge** - every component must be extremely efficient
4. **Clements mesh** provides optimal depth for programmable unitaries
5. Multiple companies pursuing different photonic QC approaches
6. **Fault-tolerant photonic QC** requires continued advances in sources, detectors, and error correction

## Weekly Summary: Photonic Quantum Computing

This week we explored the full landscape of photonic quantum computing:

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 925 | Linear Optical QC | Beam splitters and phase shifters enable single-qubit gates |
| 926 | KLM Protocol | Measurement provides nonlinearity for two-qubit gates |
| 927 | Boson Sampling | Permanents connect photonics to computational complexity |
| 928 | Continuous Variables | Infinite-dimensional Hilbert spaces using quadratures |
| 929 | GKP Encoding | Grid states enable discrete error correction in CV systems |
| 930 | Cat States | Coherent state superpositions with biased noise |
| 931 | Integrated Photonics | Chip-scale implementation of photonic quantum computing |

## Daily Checklist

- [ ] I understand optical waveguide physics and mode confinement
- [ ] I can analyze programmable photonic mesh architectures
- [ ] I understand on-chip single-photon source and detector technologies
- [ ] I can calculate loss budgets for photonic circuits
- [ ] I am familiar with industry approaches to photonic QC
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Week 134

Next week begins **Month 35: Advanced Error Correction**, diving deeper into:
- Topological codes beyond surface codes
- Color codes and gauge color codes
- Floquet codes and dynamic error correction
- Decoding algorithms and machine learning approaches
