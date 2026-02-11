# Day 937: Experimental Status

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Historical timeline, key experiments, and the 2021 retraction |
| Afternoon | 2 hours | Critical analysis: Interpreting experimental signatures |
| Evening | 2 hours | Computational lab: Data analysis techniques for Majorana detection |

## Learning Objectives

By the end of today, you will be able to:

1. **Trace the experimental history** of Majorana research from 2012 to present
2. **Analyze the 2018 Nature retraction** and its implications for the field
3. **Distinguish between** genuine and spurious zero-bias peaks
4. **Evaluate current experimental claims** with appropriate skepticism
5. **Identify remaining challenges** for definitive Majorana detection
6. **Assess the current confidence level** in Majorana existence

---

## Core Content

### 1. A Timeline of Majorana Experiments

The experimental pursuit of Majorana zero modes has been marked by both excitement and controversy.

#### 2012: The First Claims

**Mourik et al., Science 2012** (Delft/Microsoft collaboration)
- InSb nanowires with NbTiN contacts
- Zero-bias conductance peaks at finite magnetic field
- Interpreted as Majorana signature

**Initial excitement**: First experimental evidence after years of theoretical predictions!

**Caveats recognized even then**:
- Peak height not quantized ($G \ll 2e^2/h$)
- Could be Kondo effect or disorder
- Single-ended measurement only

#### 2014-2017: Building Evidence

Multiple groups reported similar observations:
- **Copenhagen**: InAs/Al epitaxial wires (cleaner interfaces)
- **Delft**: Improved devices, more systematic studies
- **Weizmann Institute**: Different geometries
- **China (PKU, USTC)**: Alternative material systems

Key improvements:
- Epitaxial superconductor shells (hard gap)
- Longer wires (better localization)
- More systematic gate and field dependence

**Persistent concerns**:
- Soft gaps allowing trivial states
- Lack of non-local correlations
- No demonstration of non-Abelian statistics

#### 2018: The Quantized Conductance Claim

**Zhang et al., Nature 2018** (Delft/Microsoft)
- Claimed quantized conductance plateau at $2e^2/h$
- Presented as "smoking gun" for Majoranas
- Widely celebrated in press and scientific community

This paper would later be retracted.

### 2. The 2021 Retraction

The 2021 retraction of the Nature 2018 paper was a watershed moment for the field.

#### What Happened

An investigation revealed:
1. **Data selection**: Only "good" traces shown; many non-quantized traces excluded
2. **Post-processing**: Data processing not fully disclosed
3. **Statistical significance**: Claims not supported by full dataset
4. **Reproducibility**: Other groups couldn't reproduce

#### The Investigation

Key findings from the investigation:
- Raw data showed highly variable peak heights
- Published traces were "cherry-picked"
- Processing steps altered apparent peak heights
- Authors retracted the paper

#### Lessons for the Field

The retraction highlighted:

1. **Publication pressure**: Strong incentive to claim breakthrough results
2. **Confirmation bias**: Tendency to interpret data favorably
3. **Replication importance**: Need for independent verification
4. **Transparency**: Raw data and processing must be available
5. **Skepticism**: Even prestigious papers need critical evaluation

### 3. What Signatures Are Really Being Observed?

Zero-bias conductance peaks can arise from multiple mechanisms. Understanding these is crucial.

#### Majorana Zero Modes (Topological)

True Majorana signatures:
- Zero-bias peak from perfect Andreev reflection
- Peak at both wire ends (non-local correlation)
- Robust to small perturbations
- Splitting exponential in wire length
- Quantized conductance at $T \to 0$

#### Trivial Mechanisms (Non-Topological)

**Andreev Bound States (ABS)**:
- Disorder or geometry can create near-zero energy states
- Can look identical to Majoranas in local measurements
- BUT: Not topologically protected, not at both ends

**Quasi-Majoranas**:
- Partially-separated Andreev states
- Smooth potential variations create overlapping states
- Mimic some Majorana properties but aren't topological

**Kondo Effect**:
- Magnetic impurities create zero-bias resonance
- Different temperature dependence
- Different magnetic field dependence

**Weak Antilocalization**:
- Coherent backscattering enhancement
- Broad zero-bias feature, not sharp peak
- Different gate voltage dependence

### 4. Current Experimental Approaches

Post-retraction, the field adopted more rigorous approaches.

#### The Topological Gap Protocol (TGP)

Microsoft's multi-metric approach (discussed Day 936):
1. Local and non-local conductance correlation
2. Gap closing at predicted field
3. Stability analysis
4. Length dependence
5. Multiple independent measurements

#### Three-Terminal Devices

Measuring correlations between different contacts:

```
   Contact A ─── Wire ─── Contact B
                  │
             Contact C

If Majorana: Correlated signals at A and B
If trivial ABS: Signal may be only at one end
```

#### Coulomb Blockade Spectroscopy

Using charging effects to probe parity:
- Island geometry isolates Majoranas
- Charge sensing detects parity switches
- 2e vs 1e periodicity distinguishes states

#### Fractional Josephson Effect

Majoranas predict $4\pi$ periodicity (instead of $2\pi$):
$$I(\phi) = I_c \sin(\phi/2)$$

Observing this would be strong evidence.

### 5. Current State of Evidence (2024-2025)

#### What Has Been Convincingly Shown

1. **Hard induced gaps**: Epitaxial Al-InAs shows clean superconductivity
2. **Gate-tunable transitions**: Can move between normal and superconducting regimes
3. **Zero-bias peaks**: Observed in many devices at appropriate field/gate values
4. **Phase diagrams**: Qualitative agreement with theoretical predictions

#### What Remains Controversial

1. **Quantized conductance**: Not reproducibly demonstrated
2. **Non-local correlations**: Mixed results, interpretation debated
3. **Topological protection**: Not definitively verified
4. **Non-Abelian statistics**: Never demonstrated

#### Where Are We Really?

A balanced assessment (as of late 2024/early 2025):
- **Optimistic view**: TGP passed, strong evidence for topology
- **Skeptical view**: Could still be explained by trivial mechanisms
- **Consensus**: Encouraging but not conclusive

### 6. Challenges for Definitive Proof

#### The "Smoking Gun" Problem

What would constitute definitive proof?

**Proposed tests**:
1. **Fusion rule verification**: Show $\sigma \times \sigma = 1 + \psi$ statistically
2. **Non-Abelian statistics**: Demonstrate braid-dependent outcomes
3. **Teleportation**: Information transfer via entangled Majoranas
4. **Exponential protection**: Systematically show error suppression

**Practical difficulties**:
- All require measuring small systems with high precision
- Thermal fluctuations mask signatures
- Device-to-device variation complicates systematics

#### Materials Challenges

Current limitations:
- **Disorder**: Random potential variations create trivial states
- **Interface quality**: Even epitaxial interfaces have defects
- **Strain**: Lattice mismatch affects band structure
- **Reproducibility**: Nominally identical devices behave differently

#### Measurement Challenges

Technical hurdles:
- **Temperature**: Need $T \ll \Delta_\text{ind}$, typically < 50 mK
- **Noise**: Electronic noise obscures small signals
- **Dynamics**: Some signatures require time-resolved measurement
- **Fabrication**: Many devices needed for statistical confidence

### 7. Alternative Platforms

The challenges with semiconductor nanowires have motivated exploration of alternatives.

#### Iron-Based Superconductors

Vortex cores in Fe(Se,Te) may host Majorana modes:
- **Advantages**: No external magnetic field needed
- **Challenges**: Material quality, vortex control

#### Planar Josephson Junctions

2D electron gas with superconductor stripes:
- **Advantages**: Scalable, gate-tunable
- **Challenges**: Narrow topological phase, disorder sensitivity

#### Magnetic Atom Chains

Self-assembled chains on superconductor surfaces:
- **Advantages**: Single-atom precision
- **Challenges**: Not easily scalable, limited control

#### Fractional Quantum Hall States

$\nu = 5/2$ state may have non-Abelian anyons:
- **Advantages**: Naturally 2D, high mobility
- **Challenges**: Requires high fields, very low temperature

### 8. Honest Assessment

#### What Would Change the Consensus?

For skeptics to be convinced:
1. Reproducible quantized conductance across many devices
2. Clear non-local correlations with correct theory match
3. Demonstration of parity lifetime exceeding thermal limits
4. Ideally: some signature of non-Abelian statistics

For optimists to be concerned:
1. Alternative explanations for all current data
2. Failure to improve with better devices
3. Fundamental physical limit preventing realization

#### The Scientific Process Working

Despite the setback of the retraction:
- Standards have improved dramatically
- Protocols are now more rigorous
- Community is more skeptical (appropriately)
- Progress continues with better methodology

This is how science should work - claims are tested, problematic results corrected, and the bar for evidence raised.

---

## Quantum Computing Applications

### Impact on the Roadmap

The experimental challenges affect practical timelines:

| Milestone | Original Estimate | Current Status |
|-----------|------------------|----------------|
| Single Majorana qubit | 2020-2022 | In progress |
| Two-qubit operations | 2023-2024 | Not demonstrated |
| Logical qubit | 2025-2027 | Delayed |
| Fault-tolerant QC | 2030+ | Timeline uncertain |

### Hedging Strategies

Given uncertainties, Microsoft has diversified:
- Continued topological research (primary)
- Partnerships with other hardware providers
- Azure Quantum cloud with multiple backends
- Investment in error correction research for all platforms

---

## Worked Examples

### Example 1: Distinguishing Peak Heights

**Problem**: A zero-bias conductance peak is observed with height $G = 0.7 \times 2e^2/h$. What might this indicate?

**Solution**:

Possible interpretations:

1. **Majorana with finite temperature**:
   $$G = \frac{2e^2}{h} \cdot f(T/\Gamma)$$
   where $\Gamma$ is the coupling strength. At finite $T$, $G < 2e^2/h$.

2. **Majorana with finite overlap**:
   If two Majoranas hybridize (wire too short):
   $$G = \frac{2e^2}{h} \cdot \frac{\Gamma^2}{\Gamma^2 + \delta E^2}$$
   where $\delta E$ is the splitting.

3. **Trivial Andreev bound state**:
   ABS at energy $E_0 \neq 0$ contributes:
   $$G(V=0) < 2e^2/h$$ generically

4. **Partially transparent contact**:
   Barrier reduces measured conductance.

**Conclusion**: 0.7 × quantized is suggestive but not definitive.

$$\boxed{G = 0.7 \times 2e^2/h \text{ is consistent with multiple scenarios}}$$

Additional tests needed: temperature dependence, magnetic field evolution, non-local measurement.

### Example 2: Non-Local Correlation

**Problem**: In a three-terminal device, explain what non-local conductance $G_{12}$ and $G_{13}$ should show for true Majoranas vs. trivial ABS.

**Solution**:

**For true Majoranas at positions 1 and 2**:
- Majoranas are at opposite ends of the topological segment
- They are components of a single non-local fermion
- Both should show correlated zero-bias features:
  - $G_{12}$: peak at zero bias (probes $\gamma_1$)
  - $G_{13}$: peak at zero bias (probes $\gamma_2$)
  - Peak heights should be correlated
  - Response to local gates should be anti-correlated

**For trivial ABS**:
- ABS is localized (not non-local)
- May appear at only one end
- $G_{12}$ and $G_{13}$ typically uncorrelated
- Local gate affects only nearby signal

**Mathematical form**:
True Majorana:
$$G_{12}(V=0) \propto |\langle \gamma_1 \rangle|^2, \quad G_{13}(V=0) \propto |\langle \gamma_2 \rangle|^2$$

Both should be maximal simultaneously when in topological phase.

$$\boxed{\text{Majorana: Correlated non-local signals; ABS: Uncorrelated}}$$

### Example 3: Temperature Dependence

**Problem**: A zero-bias peak has width $\Gamma = 20$ μeV at base temperature 20 mK. What is the expected temperature dependence?

**Solution**:

For a Majorana coupled to a lead with strength $\Gamma$:
$$G(V=0, T) = \frac{2e^2}{h} \cdot g(k_BT/\Gamma)$$

where $g$ is a universal function approaching 1 for $T \to 0$.

At $T = 20$ mK:
$$k_BT = 86 \text{ μeV/K} \times 0.020 \text{ K} = 1.72 \text{ μeV}$$

Ratio: $k_BT/\Gamma = 1.72/20 = 0.086$

This is in the low-temperature regime where $G$ should be close to quantized.

Expected behavior:
- $T < \Gamma/k_B \approx 0.23$ K: $G$ nearly quantized
- $T \sim \Gamma/k_B$: $G$ drops to ~half
- $T > \Gamma/k_B$: $G$ continues decreasing

$$\boxed{\text{Expect } G \approx 0.9 \times 2e^2/h \text{ at } T = 20 \text{ mK}}$$

If observed $G$ is much lower, suspect non-Majorana origin.

---

## Practice Problems

### Level 1: Direct Application

1. **Peak Height**: If the observed conductance is $1.5 e^2/h$ (instead of $2e^2/h$), list three possible reasons.

2. **Temperature Scale**: For induced gap $\Delta = 200$ μeV, what is the maximum temperature for observing Majorana signatures?

3. **Field Dependence**: Why should a Majorana zero-bias peak persist over a range of magnetic field values, while a Kondo peak should split?

### Level 2: Intermediate

4. **Soft Gap**: A device shows subgap conductance 10% of the normal-state value. Estimate the density of subgap states and their effect on Majorana lifetime.

5. **Length Dependence**: Design an experiment to test the exponential length dependence of Majorana splitting. What wire lengths and measurement precision are needed?

6. **Correlation Analysis**: Given conductance data from two ends of a wire, describe a statistical test to determine if zero-bias peaks are correlated.

### Level 3: Challenging

7. **Quasi-Majorana**: Model a smooth potential variation that creates a pair of quasi-Majorana states. Show how these can mimic true Majorana signatures in local measurements but fail non-local tests.

8. **Bayesian Analysis**: Given prior probability 50% for Majoranas, and a measurement showing zero-bias peak with 5% probability of arising from trivial sources, calculate the posterior probability of Majorana presence.

---

## Computational Lab: Data Analysis for Majorana Detection

```python
"""
Day 937 Computational Lab: Experimental Status Analysis
Data analysis techniques for evaluating Majorana claims
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Simulating Conductance Data
# =============================================================================

def majorana_conductance(V, G0, Gamma, E0=0, T=0.02):
    """
    Model conductance for a Majorana mode.

    G(V) = G0 * Γ² / ((V-E0)² + Γ²) * thermal_factor

    Parameters:
    -----------
    V : array - Bias voltage (mV)
    G0 : float - Peak conductance (2e²/h for ideal Majorana)
    Gamma : float - Coupling/broadening (meV)
    E0 : float - Energy of state (0 for Majorana)
    T : float - Temperature (K)
    """
    kB = 0.086  # meV/K
    thermal_width = kB * T

    # Lorentzian peak
    G = G0 * Gamma**2 / ((V - E0)**2 + Gamma**2)

    # Thermal broadening (convolution with derivative of Fermi function)
    # Simplified as additional Gaussian broadening
    if thermal_width > 0:
        V_grid = np.linspace(V.min(), V.max(), len(V))
        dV = V_grid[1] - V_grid[0]
        sigma_points = thermal_width / dV
        G = gaussian_filter1d(G, sigma_points)

    return G


def andreev_bound_state_conductance(V, G0, Gamma, E0, T=0.02):
    """
    Model conductance for a trivial Andreev bound state at energy E0.
    """
    kB = 0.086
    thermal_width = kB * T

    # Two peaks at ±E0
    G = G0 * Gamma**2 / ((V - E0)**2 + Gamma**2)
    G += G0 * Gamma**2 / ((V + E0)**2 + Gamma**2)

    # Thermal broadening
    if thermal_width > 0:
        dV = V[1] - V[0]
        sigma_points = thermal_width / dV
        G = gaussian_filter1d(G, sigma_points)

    return G


def generate_synthetic_data(scenario='majorana', noise_level=0.05):
    """
    Generate synthetic conductance vs voltage data.

    scenario: 'majorana', 'abs', 'kondo', 'mixed'
    """
    V = np.linspace(-0.3, 0.3, 500)  # mV

    G_quantum = 7.75e-5  # 2e²/h in Siemens

    if scenario == 'majorana':
        # Ideal Majorana: peak at V=0 with height approaching 2e²/h
        G = majorana_conductance(V, G0=0.95*G_quantum, Gamma=0.02, E0=0)

    elif scenario == 'abs':
        # Andreev bound state at finite energy
        G = andreev_bound_state_conductance(V, G0=0.8*G_quantum, Gamma=0.02, E0=0.05)

    elif scenario == 'kondo':
        # Kondo peak (similar to Majorana but different T, B dependence)
        G = majorana_conductance(V, G0=0.7*G_quantum, Gamma=0.05, E0=0)
        # Add asymmetry typical of Kondo
        G += 0.1 * G_quantum * V / V.max()

    elif scenario == 'mixed':
        # Quasi-Majorana: slightly split overlapping states
        G1 = majorana_conductance(V, G0=0.5*G_quantum, Gamma=0.015, E0=0.01)
        G2 = majorana_conductance(V, G0=0.5*G_quantum, Gamma=0.015, E0=-0.01)
        G = G1 + G2

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Add noise
    G_noisy = G + noise_level * G_quantum * np.random.randn(len(V))

    return V, G, G_noisy


# =============================================================================
# Part 2: Peak Analysis
# =============================================================================

def analyze_zero_bias_peak(V, G, verbose=True):
    """
    Analyze a conductance trace for zero-bias peak characteristics.

    Returns dict with:
    - peak_height: conductance at V=0
    - peak_width: FWHM
    - symmetry: measure of peak symmetry
    - quantization: ratio to 2e²/h
    """
    G_quantum = 7.75e-5  # 2e²/h

    # Find peak near V=0
    center_idx = len(V) // 2
    search_range = len(V) // 10
    local_G = G[center_idx - search_range:center_idx + search_range]
    local_V = V[center_idx - search_range:center_idx + search_range]

    peak_idx = np.argmax(local_G)
    peak_height = local_G[peak_idx]
    peak_voltage = local_V[peak_idx]

    # FWHM
    half_max = (peak_height + np.min(G)) / 2
    above_half = local_G > half_max
    if np.any(above_half):
        width_indices = np.where(above_half)[0]
        fwhm = local_V[width_indices[-1]] - local_V[width_indices[0]]
    else:
        fwhm = 0

    # Symmetry: compare left and right of peak
    left_G = G[:center_idx]
    right_G = G[center_idx:][::-1]
    min_len = min(len(left_G), len(right_G))
    symmetry = np.corrcoef(left_G[:min_len], right_G[:min_len])[0, 1]

    # Quantization
    quantization = peak_height / G_quantum

    results = {
        'peak_height': peak_height,
        'peak_voltage': peak_voltage,
        'fwhm': fwhm,
        'symmetry': symmetry,
        'quantization': quantization
    }

    if verbose:
        print("\nZero-Bias Peak Analysis:")
        print(f"  Peak height: {peak_height*1e6:.2f} μS")
        print(f"  Peak voltage: {peak_voltage*1000:.3f} μV")
        print(f"  FWHM: {fwhm*1000:.1f} μV")
        print(f"  Symmetry (correlation): {symmetry:.3f}")
        print(f"  Quantization (G/G_Q): {quantization:.3f}")

        if quantization > 0.9:
            print("  → Consistent with quantized Majorana peak")
        elif quantization > 0.5:
            print("  → Partial quantization (finite T, coupling, or trivial)")
        else:
            print("  → Well below quantized value")

    return results


def compare_scenarios():
    """Generate and analyze different scenarios."""
    print("=" * 60)
    print("Comparing Different Physical Scenarios")
    print("=" * 60)

    scenarios = ['majorana', 'abs', 'kondo', 'mixed']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    G_quantum = 7.75e-5

    for ax, scenario in zip(axes.flat, scenarios):
        V, G_ideal, G_noisy = generate_synthetic_data(scenario)

        ax.plot(V * 1000, G_noisy / G_quantum, 'b-', alpha=0.7, linewidth=0.5,
                label='Noisy data')
        ax.plot(V * 1000, G_ideal / G_quantum, 'r-', linewidth=2,
                label='Ideal')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5,
                   label='G = 2e²/h')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Bias Voltage (μV)')
        ax.set_ylabel('G / (2e²/h)')
        ax.set_title(f'{scenario.capitalize()}')
        ax.legend(loc='upper right')
        ax.set_xlim([-300, 300])
        ax.set_ylim([0, 1.2])

        # Analyze
        results = analyze_zero_bias_peak(V, G_noisy, verbose=False)
        ax.text(0.05, 0.95, f"G/G_Q = {results['quantization']:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top')

    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 3: Non-Local Correlation Analysis
# =============================================================================

def generate_correlated_data(correlation_strength=0.8):
    """
    Generate correlated zero-bias data from two ends of a wire.
    High correlation suggests Majorana; low correlation suggests trivial.
    """
    n_field_points = 100
    B = np.linspace(0, 2, n_field_points)  # Tesla

    # True Majorana: correlated signals at both ends
    np.random.seed(42)
    base_signal = np.exp(-(B - 1)**2 / 0.1)  # Peak at B = 1T

    # Add correlated and uncorrelated noise
    noise1 = np.random.randn(n_field_points) * 0.1
    noise2 = np.random.randn(n_field_points) * 0.1
    common_noise = np.random.randn(n_field_points) * 0.1

    G_left = base_signal + correlation_strength * common_noise + (1 - correlation_strength) * noise1
    G_right = base_signal + correlation_strength * common_noise + (1 - correlation_strength) * noise2

    return B, G_left, G_right


def analyze_correlation():
    """Analyze non-local correlations."""
    print("\n" + "=" * 60)
    print("Non-Local Correlation Analysis")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # High correlation (Majorana-like)
    B, G_left, G_right = generate_correlated_data(correlation_strength=0.9)
    corr_high, p_high = pearsonr(G_left, G_right)

    axes[0, 0].plot(B, G_left, 'b-', label='Left end')
    axes[0, 0].plot(B, G_right, 'r-', label='Right end')
    axes[0, 0].set_xlabel('Magnetic Field (T)')
    axes[0, 0].set_ylabel('G(V=0) (arb. units)')
    axes[0, 0].set_title(f'High Correlation (r = {corr_high:.2f})')
    axes[0, 0].legend()

    axes[0, 1].scatter(G_left, G_right, alpha=0.5)
    axes[0, 1].set_xlabel('G_left')
    axes[0, 1].set_ylabel('G_right')
    axes[0, 1].set_title(f'Correlation: {corr_high:.3f}, p = {p_high:.2e}')

    # Low correlation (trivial-like)
    B, G_left, G_right = generate_correlated_data(correlation_strength=0.2)
    corr_low, p_low = pearsonr(G_left, G_right)

    axes[1, 0].plot(B, G_left, 'b-', label='Left end')
    axes[1, 0].plot(B, G_right, 'r-', label='Right end')
    axes[1, 0].set_xlabel('Magnetic Field (T)')
    axes[1, 0].set_ylabel('G(V=0) (arb. units)')
    axes[1, 0].set_title(f'Low Correlation (r = {corr_low:.2f})')
    axes[1, 0].legend()

    axes[1, 1].scatter(G_left, G_right, alpha=0.5)
    axes[1, 1].set_xlabel('G_left')
    axes[1, 1].set_ylabel('G_right')
    axes[1, 1].set_title(f'Correlation: {corr_low:.3f}, p = {p_low:.2e}')

    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nHigh correlation case: r = {corr_high:.3f}")
    print(f"  Interpretation: Consistent with Majorana (non-local)")
    print(f"\nLow correlation case: r = {corr_low:.3f}")
    print(f"  Interpretation: Suggests trivial states (local)")


# =============================================================================
# Part 4: Gap Closing Analysis
# =============================================================================

def simulate_gap_closing(is_topological=True):
    """
    Simulate gap evolution across a topological phase transition.
    """
    B = np.linspace(0, 2, 200)  # Tesla
    B_c = 1.0  # Critical field

    if is_topological:
        # Gap closes and reopens at B_c
        gap = np.abs(B - B_c) * 0.5 + 0.05 * np.random.randn(len(B))
        gap = np.maximum(gap, 0.01)
    else:
        # Gap doesn't close (trivial transition)
        gap = 0.3 + 0.1 * np.sin(2 * np.pi * B / 2) + 0.02 * np.random.randn(len(B))
        gap = np.maximum(gap, 0.05)

    return B, gap


def analyze_gap_closing():
    """Analyze gap closing behavior."""
    print("\n" + "=" * 60)
    print("Gap Closing Analysis (Topological Gap Protocol)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Topological case
    B, gap = simulate_gap_closing(is_topological=True)
    axes[0].plot(B, gap, 'b-', linewidth=2)
    axes[0].axvline(x=1.0, color='r', linestyle='--', label='B_c (theory)')
    axes[0].set_xlabel('Magnetic Field (T)')
    axes[0].set_ylabel('Gap (meV)')
    axes[0].set_title('Topological: Gap Closes at B_c')
    axes[0].legend()
    axes[0].set_ylim([0, 0.6])

    min_idx = np.argmin(gap)
    axes[0].annotate(f'Min at B = {B[min_idx]:.2f} T',
                     xy=(B[min_idx], gap[min_idx]),
                     xytext=(1.3, 0.3),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10)

    # Non-topological case
    B, gap = simulate_gap_closing(is_topological=False)
    axes[1].plot(B, gap, 'b-', linewidth=2)
    axes[1].axvline(x=1.0, color='r', linestyle='--', label='B_c (theory)')
    axes[1].set_xlabel('Magnetic Field (T)')
    axes[1].set_ylabel('Gap (meV)')
    axes[1].set_title('Trivial: Gap Never Closes')
    axes[1].legend()
    axes[1].set_ylim([0, 0.6])

    plt.tight_layout()
    plt.savefig('gap_closing.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nKey test: Gap must close at predicted B_c for topological transition")
    print("Trivial states can show zero-bias peaks without gap closing")


# =============================================================================
# Part 5: Bayesian Evidence Analysis
# =============================================================================

def bayesian_analysis():
    """
    Demonstrate Bayesian approach to evaluating Majorana evidence.
    """
    print("\n" + "=" * 60)
    print("Bayesian Evidence Analysis")
    print("=" * 60)

    # Prior probability of Majorana
    P_majorana_prior = 0.5

    # Likelihoods for different observations
    observations = {
        'Zero-bias peak observed': (0.95, 0.40),  # P(obs|Maj), P(obs|trivial)
        'Peak near quantized': (0.30, 0.05),
        'Non-local correlation': (0.80, 0.10),
        'Gap closing at predicted B_c': (0.70, 0.15),
        'Exponential length dependence': (0.60, 0.05),
    }

    print(f"\nPrior P(Majorana) = {P_majorana_prior}")
    print("\nUpdating with observations:\n")

    P_maj = P_majorana_prior

    for obs_name, (p_if_maj, p_if_triv) in observations.items():
        # Bayes' theorem
        P_obs = P_maj * p_if_maj + (1 - P_maj) * p_if_triv
        P_maj_new = (p_if_maj * P_maj) / P_obs

        print(f"  {obs_name}:")
        print(f"    P(obs|Maj) = {p_if_maj}, P(obs|trivial) = {p_if_triv}")
        print(f"    P(Majorana) updated: {P_maj:.3f} → {P_maj_new:.3f}")

        P_maj = P_maj_new

    print(f"\nFinal posterior P(Majorana | all observations) = {P_maj:.3f}")

    # Visualize update process
    fig, ax = plt.subplots(figsize=(10, 6))

    P_values = [P_majorana_prior]
    P_curr = P_majorana_prior
    for obs_name, (p_if_maj, p_if_triv) in observations.items():
        P_obs = P_curr * p_if_maj + (1 - P_curr) * p_if_triv
        P_curr = (p_if_maj * P_curr) / P_obs
        P_values.append(P_curr)

    obs_names = ['Prior'] + list(observations.keys())
    x_pos = range(len(obs_names))

    ax.bar(x_pos, P_values, color='steelblue', edgecolor='black')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% confidence')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(obs_names, rotation=45, ha='right')
    ax.set_ylabel('P(Majorana)')
    ax.set_title('Bayesian Update of Majorana Probability')
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('bayesian_update.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 6: Statistical Significance of Quantization
# =============================================================================

def quantization_statistics():
    """
    Analyze statistical significance of conductance quantization.
    """
    print("\n" + "=" * 60)
    print("Conductance Quantization Statistics")
    print("=" * 60)

    # Simulate multiple device measurements
    n_devices = 50
    np.random.seed(123)

    # True case: Majorana with device-to-device variation
    G_quantum = 7.75e-5
    G_majorana = G_quantum * (0.8 + 0.3 * np.random.rand(n_devices))
    # Some devices have higher quality
    high_quality = np.random.rand(n_devices) > 0.7
    G_majorana[high_quality] = G_quantum * (0.95 + 0.05 * np.random.randn(np.sum(high_quality)))

    # Trivial case: random peak heights
    G_trivial = G_quantum * (0.2 + 0.6 * np.random.rand(n_devices))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of peak heights
    bins = np.linspace(0, 1.2, 20)
    axes[0].hist(G_majorana / G_quantum, bins=bins, alpha=0.7, label='Majorana-like',
                 color='blue', edgecolor='black')
    axes[0].hist(G_trivial / G_quantum, bins=bins, alpha=0.7, label='Trivial-like',
                 color='red', edgecolor='black')
    axes[0].axvline(x=1, color='green', linestyle='--', linewidth=2, label='Quantized')
    axes[0].set_xlabel('G / (2e²/h)')
    axes[0].set_ylabel('Number of devices')
    axes[0].set_title('Distribution of Peak Heights')
    axes[0].legend()

    # Q-Q plot or cumulative distribution
    G_sorted_maj = np.sort(G_majorana / G_quantum)
    G_sorted_triv = np.sort(G_trivial / G_quantum)

    axes[1].plot(np.linspace(0, 1, n_devices), G_sorted_maj, 'b-', linewidth=2,
                 label='Majorana-like')
    axes[1].plot(np.linspace(0, 1, n_devices), G_sorted_triv, 'r-', linewidth=2,
                 label='Trivial-like')
    axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Cumulative fraction')
    axes[1].set_ylabel('G / (2e²/h)')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('quantization_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Statistical tests
    from scipy.stats import ttest_1samp, kstest

    # Test if mean is consistent with quantization
    t_stat, p_val = ttest_1samp(G_majorana / G_quantum, 1.0)
    print(f"\nMajorana-like devices:")
    print(f"  Mean G/(2e²/h) = {np.mean(G_majorana/G_quantum):.3f} ± {np.std(G_majorana/G_quantum):.3f}")
    print(f"  Fraction > 0.9: {np.mean(G_majorana/G_quantum > 0.9)*100:.1f}%")

    print(f"\nTrivial-like devices:")
    print(f"  Mean G/(2e²/h) = {np.mean(G_trivial/G_quantum):.3f} ± {np.std(G_trivial/G_quantum):.3f}")
    print(f"  Fraction > 0.9: {np.mean(G_trivial/G_quantum > 0.9)*100:.1f}%")


# =============================================================================
# Part 7: Main Execution
# =============================================================================

def main():
    """Run all analysis demonstrations."""
    print("╔" + "=" * 58 + "╗")
    print("║  Day 937: Experimental Status - Data Analysis            ║")
    print("╚" + "=" * 58 + "╝")

    # 1. Compare different physical scenarios
    compare_scenarios()

    # 2. Analyze single zero-bias peak
    print("\n" + "=" * 60)
    print("Detailed Peak Analysis (Majorana case)")
    print("=" * 60)
    V, G_ideal, G_noisy = generate_synthetic_data('majorana')
    analyze_zero_bias_peak(V, G_noisy)

    # 3. Non-local correlation analysis
    analyze_correlation()

    # 4. Gap closing analysis
    analyze_gap_closing()

    # 5. Bayesian evidence
    bayesian_analysis()

    # 6. Quantization statistics
    quantization_statistics()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Analysis Techniques:
1. Peak shape: Majoranas should be Lorentzian, centered at V=0
2. Quantization: Ideal Majorana gives G = 2e²/h
3. Non-local correlation: True Majoranas show correlated signals
4. Gap closing: Must occur at predicted critical field
5. Bayesian updates: Combine multiple tests for confidence
6. Statistics: Multiple devices needed for significance

Current Status:
- Many devices show zero-bias peaks
- Quantization is not reliably achieved
- Non-local tests give mixed results
- Community confidence: Encouraging but not definitive
    """)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Observations

| Test | Majorana Signature | Current Status |
|------|-------------------|----------------|
| Zero-bias peak | At $V = 0$ | Observed in many devices |
| Quantized conductance | $G = 2e^2/h$ | Rarely achieved |
| Non-local correlation | Both ends show correlated peaks | Mixed results |
| Gap closing | At predicted $B_c$ | Often observed |
| Length dependence | Exponential splitting | Some evidence |
| Non-Abelian statistics | Braid-dependent outcomes | Never demonstrated |

### Main Takeaways

1. **The 2021 retraction** was a major setback but ultimately strengthened the field by improving standards.

2. **Zero-bias peaks** are necessary but not sufficient evidence - many trivial mechanisms can produce them.

3. **Distinguishing Majoranas** from trivial states requires multiple independent tests, especially non-local measurements.

4. **Current evidence** is encouraging but not conclusive - the community remains cautiously optimistic.

5. **Improved protocols** (like TGP) now set a higher bar for claiming topological states.

6. **Alternative platforms** are being explored as backup approaches.

---

## Daily Checklist

- [ ] I can trace the history of key Majorana experiments
- [ ] I understand the causes and lessons of the 2018 Nature retraction
- [ ] I can list mechanisms that produce zero-bias peaks besides Majoranas
- [ ] I know what measurements distinguish Majoranas from trivial states
- [ ] I can critically evaluate a new Majorana claim
- [ ] I have run the data analysis lab and understand the techniques

---

## Preview of Day 938

Tomorrow we conclude with **Topological QC Outlook** - looking forward at where the field is headed:

- Timeline projections for key milestones
- Hybrid approaches combining topological and conventional qubits
- Integration with quantum error correction
- Alternative implementations beyond nanowires
- The long-term vision for topological quantum computing

We'll assess realistic expectations and exciting possibilities!
