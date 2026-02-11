# Day 999: Semester 2B Review - Hardware Platforms

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Core Review: Quantum Hardware Platforms |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Qualifying Exam Problem Practice |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Platform Comparison and Trade-offs |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 999, you will be able to:

1. **Compare** the five major quantum computing platforms quantitatively
2. **Explain** the physical mechanisms behind qubit operation for each platform
3. **Analyze** coherence times, gate fidelities, and scalability trade-offs
4. **Evaluate** which platform is best suited for specific applications
5. **Discuss** the current state and future roadmaps for each platform
6. **Calculate** error budgets and system-level performance metrics

---

## Core Review Content

### 1. Superconducting Qubits

#### Physical Basis

**Core element:** Josephson junction - nonlinear inductor

$$\boxed{H = 4E_C(n - n_g)^2 - E_J\cos\phi}$$

- $E_C$: charging energy
- $E_J$: Josephson energy
- $n$: number of Cooper pairs
- $\phi$: phase across junction

#### Qubit Types

| Type | $E_J/E_C$ | Key Feature | Use Case |
|------|-----------|-------------|----------|
| Charge qubit | < 1 | Sensitive to charge | Historical |
| Transmon | 50-100 | Charge insensitive | IBM, Google |
| Flux qubit | > 100 | Flux sensitive | D-Wave |
| Fluxonium | Variable | Long coherence | Research |

#### Transmon Details

**Energy levels:**
$$E_n \approx \sqrt{8E_JE_C}\left(n + \frac{1}{2}\right) - \frac{E_C}{12}(6n^2 + 6n + 3)$$

**Anharmonicity:** $\alpha = E_{12} - E_{01} \approx -E_C \approx -200\text{ MHz}$

This negative anharmonicity allows selective addressing of $|0\rangle \leftrightarrow |1\rangle$.

#### Gate Mechanisms

**Single-qubit gates:** Microwave pulses at qubit frequency (~5 GHz)
$$H_{drive} = \Omega(t)\cos(\omega_d t + \phi) \sigma_x$$

**Two-qubit gates:**
- **Controlled-Z (CZ):** Tune qubits into resonance, accumulate phase
- **Cross-resonance (CR):** Drive one qubit at partner's frequency
- **iSWAP:** Exchange excitation via capacitive coupling

#### Key Metrics (2024-2025)

| Metric | Typical Value | Best Achieved |
|--------|---------------|---------------|
| T1 | 50-100 μs | 500 μs |
| T2 | 50-150 μs | 300 μs |
| 1Q gate fidelity | 99.9% | 99.99% |
| 2Q gate fidelity | 99-99.5% | 99.9% |
| 2Q gate time | 20-60 ns | 12 ns |
| Readout fidelity | 99% | 99.9% |

---

### 2. Trapped Ion Qubits

#### Physical Basis

**Trapping:** Paul trap uses oscillating electric fields
$$\Phi(x,y,z,t) = \frac{U_{DC}}{2}(x^2 + y^2 - 2z^2) + \frac{U_{RF}\cos(\Omega_{RF}t)}{2}(x^2 - y^2)$$

**Qubit encoding options:**
- **Hyperfine:** Ground state hyperfine levels (e.g., $^{171}$Yb$^+$)
- **Zeeman:** Magnetic sublevels
- **Optical:** Ground to metastable state (e.g., $^{40}$Ca$^+$)

#### Common Ion Species

| Ion | Qubit Type | Wavelength | Advantage |
|-----|------------|------------|-----------|
| $^{171}$Yb$^+$ | Hyperfine | 369 nm | Long coherence |
| $^{40}$Ca$^+$ | Optical | 729 nm | Visible lasers |
| $^{137}$Ba$^+$ | Hyperfine | 493 nm | Good visibility |
| $^9$Be$^+$ | Hyperfine | 313 nm | Fast gates |

#### Gate Mechanisms

**Single-qubit gates:** Laser or microwave pulses
$$H = \frac{\Omega}{2}(|0\rangle\langle 1|e^{i\phi} + \text{h.c.})$$

**Two-qubit gates (Molmer-Sorensen):**

Uses collective motional mode as bus:
$$H_{MS} = \sum_j \eta\Omega_j\sigma_j^x(ae^{i\delta t} + a^\dagger e^{-i\delta t})$$

Creates entangling operation:
$$U_{MS} = \exp\left(-i\frac{\pi}{4}\sigma_x^{(1)}\sigma_x^{(2)}\right)$$

#### Key Metrics (2024-2025)

| Metric | Typical Value | Best Achieved |
|--------|---------------|---------------|
| T1 | > 1 s | > 10 min |
| T2 | 1-10 s | 50 s |
| 1Q gate fidelity | 99.99% | 99.9999% |
| 2Q gate fidelity | 99-99.9% | 99.99% |
| 2Q gate time | 10-100 μs | 1.6 μs |
| Readout fidelity | 99.9% | 99.99% |

#### Scalability Approaches

- **Segmented traps:** Shuttle ions between zones
- **Photonic interconnects:** Entangle distant ions via photons
- **2D arrays:** Multiple trap zones

---

### 3. Neutral Atom Qubits

#### Physical Basis

**Trapping:** Optical tweezers using focused laser beams
$$U_{trap} = -\frac{1}{2}\alpha(\omega)|\vec{E}|^2$$

**Qubit encoding:** Hyperfine ground states (similar to ions)

#### Rydberg Interactions

**Key mechanism:** Excite atoms to Rydberg states ($n \sim 50-100$)

$$\boxed{V_{dd} = \frac{C_6}{R^6}}$$

$C_6 \propto n^{11}$ - interaction strength scales strongly with principal quantum number!

**Rydberg blockade:** If one atom is in Rydberg state, nearby atoms cannot be excited
$$|rr\rangle \text{ is shifted by } V_{dd} \gg \Omega_{Rydberg}$$

#### Gate Mechanisms

**Single-qubit gates:** Microwave or Raman transitions

**Two-qubit gates (CZ via blockade):**
1. Apply $\pi$ pulse: $|1\rangle \to |r\rangle$ on atom 1
2. Apply $2\pi$ pulse on atom 2 (blocked if atom 1 in $|r\rangle$)
3. Apply $\pi$ pulse: $|r\rangle \to |1\rangle$ on atom 1

Only $|11\rangle$ acquires $\pi$ phase shift.

#### Key Metrics (2024-2025)

| Metric | Typical Value | Best Achieved |
|--------|---------------|---------------|
| T1 | 1-10 s | > 10 s |
| T2 | 1-10 ms | 1 s |
| 1Q gate fidelity | 99.5% | 99.9% |
| 2Q gate fidelity | 97-99% | 99.5% |
| 2Q gate time | 0.1-1 μs | 100 ns |
| Array size | 100s | 6100 (QuEra) |

#### Unique Advantages

- **Reconfigurable arrays:** Move atoms with tweezers
- **Native multi-qubit gates:** Simultaneous Rydberg blockade
- **Large scale:** Thousands of atoms demonstrated
- **All-to-all connectivity:** Rearrange as needed

---

### 4. Photonic Qubits

#### Encoding Schemes

| Encoding | Basis States | Advantage |
|----------|--------------|-----------|
| Polarization | $|H\rangle, |V\rangle$ | Simple optics |
| Path | $|upper\rangle, |lower\rangle$ | Stable |
| Time-bin | $|early\rangle, |late\rangle$ | Fiber compatible |
| Number state | $|0\rangle, |1\rangle$ photons | Bosonic code ready |

#### Linear Optical Quantum Computing (LOQC)

**KLM Protocol (Knill-Laflamme-Milburn):**
- Uses beam splitters, phase shifters, single-photon sources, detectors
- CNOT requires ancilla photons and measurement
- Probabilistic but heralded

**Measurement-based QC:**
- Prepare cluster state of photons
- Perform single-qubit measurements to compute
- Xanadu's approach

#### GKP (Gottesman-Kitaev-Preskill) Encoding

Encode qubit in oscillator using grid states:
$$|0_L\rangle \propto \sum_n |2n\sqrt{\pi}\rangle_q$$
$$|1_L\rangle \propto \sum_n |(2n+1)\sqrt{\pi}\rangle_q$$

Protects against small displacements in phase space.

#### Key Metrics (2024-2025)

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Single-photon purity | 99% | Quantum dot sources |
| Indistinguishability | 99% | Critical for interference |
| Detector efficiency | 98% | Superconducting detectors |
| Gate success prob | 1/9 (CNOT) | Requires repeat attempts |
| Loss per km (fiber) | 0.2 dB | Long-distance advantage |

#### Unique Advantages

- **Room temperature operation** (mostly)
- **Natural networking:** Photons travel easily
- **No decoherence at rest:** Photons don't decohere in vacuum
- **High bandwidth:** Fast operations possible

---

### 5. Topological Qubits

#### Physical Basis: Majorana Fermions

**Majorana zero modes (MZMs):** Self-conjugate fermion states
$$\gamma = \gamma^\dagger, \quad \gamma^2 = 1$$

**Non-Abelian anyons:** Braiding changes quantum state:
$$\gamma_1\gamma_2 \to e^{i\pi/4}\gamma_2\gamma_1$$

#### Proposed Platforms

| Platform | Status | Key Challenge |
|----------|--------|---------------|
| Semiconductor-superconductor nanowires | Research | Disorder, reproducibility |
| Fractional quantum Hall | Theoretical | Extreme conditions |
| Iron-based superconductors | Research | Material quality |
| Quantum spin liquids | Theoretical | Identification |

#### Microsoft's Approach

**Goal:** Topological protection at hardware level

**Encoding:** Information in non-local fermion parity
$$|0\rangle = |even\rangle, \quad |1\rangle = |odd\rangle$$

Error requires moving Majorana across system - exponentially suppressed!

#### Current Status (2025)

- Microsoft claimed "topological signatures" (2023)
- Quantinuum achieving high fidelity without topology
- Still far from practical topological qubits
- Timeline: 5-10+ years for advantage

---

### 6. Platform Comparison

#### Quantitative Comparison Table

| Metric | Supercond. | Trapped Ion | Neutral Atom | Photonic |
|--------|------------|-------------|--------------|----------|
| T2/T_gate | 1000-5000 | 10^5-10^6 | 10^3-10^4 | N/A |
| 2Q fidelity | 99.5% | 99.9% | 99% | 99% |
| Connectivity | Nearest-neighbor | All-to-all | Configurable | Configurable |
| Qubit count | ~1000 | ~50 | ~1000 | ~100 |
| Gate speed | Fast (ns) | Slow (μs) | Medium (μs) | Fast (ns) |
| Operating temp | 10 mK | Room/laser | Room/laser | Room |

#### DiVincenzo Criteria Assessment

| Criterion | SC | Ion | Atom | Photon |
|-----------|----|----|------|--------|
| Scalable qubits | ★★★ | ★★ | ★★★★ | ★★★ |
| Initialization | ★★★★ | ★★★★ | ★★★★ | ★★★ |
| Long coherence | ★★ | ★★★★★ | ★★★ | ★★★★ |
| Universal gates | ★★★★ | ★★★★ | ★★★ | ★★★ |
| Qubit readout | ★★★★ | ★★★★★ | ★★★ | ★★★★ |

#### Best Use Cases

| Platform | Best For |
|----------|----------|
| Superconducting | Near-term demos, fast algorithms |
| Trapped Ion | High-fidelity algorithms, small problems |
| Neutral Atom | Combinatorial optimization, simulation |
| Photonic | Networking, distributed QC |
| Topological | Future fault-tolerant (if realized) |

---

## Concept Map: Hardware Platforms

```
Quantum Hardware Platforms
           │
     ┌─────┴─────┬─────────┬──────────┬──────────┐
     ▼           ▼         ▼          ▼          ▼
Superconducting  Trapped   Neutral   Photonic   Topological
     │           Ion       Atom        │          │
     │           │         │          │          │
     ▼           ▼         ▼          ▼          ▼
Josephson    Coulomb    Optical    Linear     Majorana
Junction     Crystal    Tweezers   Optics     Fermions
     │           │         │          │          │
     ▼           ▼         ▼          ▼          │
Microwave    Laser/MW   Rydberg    Photons    Future
Control      Control    Blockade   + Detectors  │
     │           │         │          │          │
     └─────┬─────┴─────────┴──────────┘          │
           ▼                                      │
    Current Leaders: Google, IBM,                 │
    IonQ, Quantinuum, QuEra, Xanadu              │
           │                                      │
           ▼                                      │
    Near-term: NISQ applications ◄────────────────┘
    Long-term: Fault-tolerant QC        (eventual?)
```

---

## Qualifying Exam Practice Problems

### Problem 1: Superconducting Qubit Analysis (25 points)

**Question:** A transmon qubit has $E_J/E_C = 50$ with $E_C = 250$ MHz.

(a) Calculate the qubit frequency $\omega_{01}$
(b) Calculate the anharmonicity $\alpha$
(c) If T1 = 50 μs and T2 = 80 μs, what is the pure dephasing time $T_\phi$?
(d) How many CZ gates (40 ns each) can be performed before losing coherence?

**Solution:**

**(a) Qubit frequency:**

$$\omega_{01} \approx \sqrt{8E_JE_C} - E_C$$

With $E_J = 50 \times E_C = 50 \times 250 = 12500$ MHz:

$$\omega_{01} = \sqrt{8 \times 12500 \times 250} - 250 = \sqrt{25,000,000} - 250$$
$$= 5000 - 250 = 4750 \text{ MHz} = 4.75 \text{ GHz}$$

**(b) Anharmonicity:**

$$\alpha = -E_C = -250 \text{ MHz}$$

This means $\omega_{12} = \omega_{01} + \alpha = 4500$ MHz.

**(c) Pure dephasing:**

$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

$$\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1} = \frac{1}{80} - \frac{1}{100} = \frac{100-80}{8000} = \frac{1}{400}$$

$$T_\phi = 400 \text{ μs}$$

**(d) Number of gates:**

Using $T_2 = 80$ μs as the relevant timescale:
$$N_{gates} = \frac{T_2}{t_{gate}} = \frac{80,000 \text{ ns}}{40 \text{ ns}} = 2000 \text{ gates}$$

But for high-fidelity operation, typically use $T_2/10$:
$$N_{useful} \approx 200 \text{ gates}$$

---

### Problem 2: Trapped Ion Gate Analysis (25 points)

**Question:** For a Molmer-Sorensen gate on two $^{171}$Yb$^+$ ions:

(a) If the radial trap frequency is $\omega_r = 2\pi \times 3$ MHz and ion spacing is 5 μm, estimate the Lamb-Dicke parameter $\eta$
(b) For gate time $\tau = 100$ μs, what Rabi frequency $\Omega$ is needed?
(c) If gate infidelity is dominated by motional heating with $\dot{\bar{n}} = 10$ quanta/s, estimate the infidelity
(d) Why are trapped ions slower than superconducting qubits?

**Solution:**

**(a) Lamb-Dicke parameter:**

$$\eta = k\sqrt{\frac{\hbar}{2m\omega_r}}$$

where $k = 2\pi/\lambda$ and $\lambda = 369$ nm for Yb+.

Ground state size: $x_0 = \sqrt{\hbar/2m\omega_r}$

For Yb+ ($m = 171$ amu):
$$x_0 = \sqrt{\frac{1.055 \times 10^{-34}}{2 \times 171 \times 1.66 \times 10^{-27} \times 2\pi \times 3 \times 10^6}}$$
$$= \sqrt{\frac{1.055 \times 10^{-34}}{5.35 \times 10^{-18}}} \approx 4.4 \text{ nm}$$

$$\eta = \frac{2\pi}{369 \times 10^{-9}} \times 4.4 \times 10^{-9} \approx 0.075$$

**(b) Rabi frequency:**

For MS gate: $\Omega \times \eta \times \tau = \pi$

$$\Omega = \frac{\pi}{\eta \tau} = \frac{\pi}{0.075 \times 100 \times 10^{-6}} = 420 \text{ kHz}$$

**(c) Heating-limited infidelity:**

Motional quanta added during gate:
$$\Delta \bar{n} = \dot{\bar{n}} \times \tau = 10 \times 100 \times 10^{-6} = 10^{-3}$$

Infidelity contribution:
$$1 - F \approx \eta^2 \Delta\bar{n} = (0.075)^2 \times 10^{-3} \approx 5.6 \times 10^{-6}$$

Very small contribution - heating not limiting for this gate time.

**(d) Speed limitation:**

Trapped ion gates are slow because:
1. **Need to avoid motional excitation:** $\Omega \ll \omega_{trap}$
2. **Lamb-Dicke regime:** Operations must be in $\eta \ll 1$ limit
3. **Spectral crowding:** Many motional modes must be avoided
4. **Laser power limits:** High Rabi frequencies require high intensity

Superconducting qubits have no motional degrees of freedom - direct microwave driving at GHz frequencies.

---

### Problem 3: Neutral Atom Blockade (20 points)

**Question:** For a neutral atom system with Rb-87:

(a) If atoms are separated by R = 4 μm and the Rydberg state has $n = 70$, estimate $C_6$ and the interaction strength
(b) What Rabi frequency is needed for strong blockade ($V_{dd} > 10\Omega$)?
(c) How does gate fidelity depend on R?
(d) What limits how close atoms can be placed?

**Solution:**

**(a) $C_6$ and interaction:**

For Rb at $n = 70$:
$$C_6 \approx 2\pi \times 862 \text{ GHz} \cdot \mu\text{m}^6$$ (empirical)

$$V_{dd} = \frac{C_6}{R^6} = \frac{2\pi \times 862 \times 10^9}{(4)^6} = \frac{862 \times 10^9}{4096} \times 2\pi$$
$$= 2\pi \times 210 \text{ MHz}$$

**(b) Blockade Rabi frequency:**

For strong blockade: $V_{dd} > 10\Omega$
$$\Omega < V_{dd}/10 = 2\pi \times 21 \text{ MHz}$$

**Maximum Rabi frequency: ~20 MHz** for good blockade

**(c) Fidelity vs R:**

Gate error from imperfect blockade:
$$1 - F \propto \left(\frac{\Omega}{V_{dd}}\right)^2 \propto R^{12}$$

Fidelity degrades extremely rapidly with distance (R^12 dependence)!

**(d) Minimum spacing limits:**

- **Optical resolution:** Diffraction limit ~λ/2 ≈ 0.5 μm
- **Tweezer crosstalk:** Beams overlap
- **Collisions:** If too close, atoms can collide and be lost
- **State-dependent forces:** Light shifts vary with position

Practical minimum: ~2-3 μm

---

### Problem 4: Error Budget (20 points)

**Question:** Create an error budget for a surface code on superconducting qubits:

Given:
- 1Q gate error: 0.1%
- 2Q gate error: 0.5%
- Readout error: 1%
- Idle error (per μs): 0.01%
- Syndrome cycle time: 1 μs

(a) What is the total error per syndrome cycle?
(b) Is this below the surface code threshold (~0.6%)?
(c) Which error source dominates?
(d) What improvement would help most?

**Solution:**

**(a) Error per syndrome cycle:**

Per syndrome round:
- 1Q gates: 4 per data qubit, error = $4 \times 0.1\% = 0.4\%$
- 2Q gates: 4 per data qubit, error = $4 \times 0.5\% = 2.0\%$
- Measurement: 1 per ancilla, error = $1\%$
- Idle: 1 μs, error = $0.01\%$

**Total per data qubit: ~2.4% (dominated by 2Q gates)**
**Total per ancilla: ~1.4% (dominated by measurement)**

Effective error rate: weighted average ≈ **1.5-2%**

**(b) Threshold comparison:**

At 1.5-2%, we are **above** the ~0.6% circuit-level threshold.

Surface code would NOT work with these error rates!

**(c) Dominant error source:**

**Two-qubit gates at 2%** contribution dominate.

**(d) Improvement priority:**

1. **2Q gate fidelity:** Need 99.9%+ (currently 99.5%)
   - Would reduce contribution from 2% to 0.4%

2. **Readout fidelity:** Need 99.5%+ (currently 99%)
   - Would reduce contribution from 1% to 0.5%

With these improvements: total ~0.9%, still marginal.

Need 2Q gates at 99.95% for comfortable below-threshold operation.

---

### Problem 5: Platform Selection (10 points)

**Question:** Recommend a platform for each application:

(a) Quantum simulation of a 100-spin Heisenberg model
(b) Running Shor's algorithm for 2048-bit factoring
(c) Distributed quantum computing between cities
(d) Variational quantum eigensolver for chemistry

**Solution:**

**(a) 100-spin Heisenberg model:**

**Neutral atoms** - best choice because:
- Native spin Hamiltonian simulation
- Can arrange atoms in desired geometry
- Rydberg interactions mimic spin-spin coupling
- 100+ atoms demonstrated
- Alternative: Trapped ions for smaller systems with all-to-all

**(b) 2048-bit Shor's algorithm:**

**Superconducting qubits** (eventually) - because:
- Fastest gate times for deep circuits
- Most mature error correction demonstrations
- Best path to million+ qubit systems
- BUT: No platform can do this today

**(c) Distributed quantum computing:**

**Photonic qubits** - because:
- Natural for long-distance transmission
- Fiber infrastructure exists
- Room temperature operation at nodes
- Entanglement distribution demonstrated over 1000+ km
- Alternative: Ion-photon interfaces for processing nodes

**(d) VQE for chemistry:**

**Trapped ions** for accuracy, **superconducting** for speed:
- Trapped ions: Highest gate fidelities for accurate energy
- Limited qubit count acceptable for small molecules
- Superconducting: Faster iteration for variational optimization
- Neutral atoms: Good for larger molecular systems

---

## Computational Review

```python
"""
Day 999 Computational Review: Hardware Platforms
Semester 2B Review - Week 143
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Platform Metrics Comparison
# =============================================================================

print("=" * 70)
print("Part 1: Platform Metrics Comparison")
print("=" * 70)

platforms = {
    'Superconducting': {
        'T1': 100e-6,        # seconds
        'T2': 150e-6,
        'gate_1Q': 20e-9,    # seconds
        'gate_2Q': 40e-9,
        'fidelity_1Q': 0.9999,
        'fidelity_2Q': 0.995,
        'qubits': 1000,
        'connectivity': 'nearest-neighbor'
    },
    'Trapped Ion': {
        'T1': 10,
        'T2': 1,
        'gate_1Q': 10e-6,
        'gate_2Q': 100e-6,
        'fidelity_1Q': 0.99999,
        'fidelity_2Q': 0.999,
        'qubits': 50,
        'connectivity': 'all-to-all'
    },
    'Neutral Atom': {
        'T1': 10,
        'T2': 0.01,
        'gate_1Q': 1e-6,
        'gate_2Q': 1e-6,
        'fidelity_1Q': 0.999,
        'fidelity_2Q': 0.99,
        'qubits': 1000,
        'connectivity': 'reconfigurable'
    },
    'Photonic': {
        'T1': float('inf'),
        'T2': float('inf'),
        'gate_1Q': 1e-9,
        'gate_2Q': 10e-9,
        'fidelity_1Q': 0.999,
        'fidelity_2Q': 0.99,
        'qubits': 100,
        'connectivity': 'linear'
    }
}

# Calculate derived metrics
for name, p in platforms.items():
    if p['T2'] != float('inf'):
        p['T2_over_gate'] = p['T2'] / p['gate_2Q']
    else:
        p['T2_over_gate'] = float('inf')
    p['error_per_gate'] = 1 - p['fidelity_2Q']

print("\nPlatform Comparison:")
print("-" * 70)
print(f"{'Platform':<15} {'T2 (s)':<12} {'2Q Gate (s)':<12} {'2Q Error':<10} {'Qubits':<10}")
print("-" * 70)

for name, p in platforms.items():
    t2_str = f"{p['T2']:.2e}" if p['T2'] != float('inf') else "inf"
    print(f"{name:<15} {t2_str:<12} {p['gate_2Q']:.2e}     {p['error_per_gate']:.1%}       {p['qubits']}")

# =============================================================================
# Part 2: Error Budget Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Surface Code Error Budget (Superconducting)")
print("=" * 70)

def surface_code_error_budget(
    error_1Q=0.001,
    error_2Q=0.005,
    error_readout=0.01,
    error_idle_per_us=0.0001,
    cycle_time_us=1.0
):
    """Calculate error budget for surface code syndrome cycle."""

    # Per data qubit per cycle
    gates_1Q = 4  # H gates
    gates_2Q = 4  # CNOTs to ancillas
    idle_time = 0.5 * cycle_time_us  # average idle

    error_from_1Q = gates_1Q * error_1Q
    error_from_2Q = gates_2Q * error_2Q
    error_from_idle = idle_time * error_idle_per_us

    # Per ancilla per cycle
    measurements = 1
    error_from_readout = measurements * error_readout

    total_error = error_from_1Q + error_from_2Q + error_from_idle + error_from_readout

    return {
        '1Q gates': error_from_1Q,
        '2Q gates': error_from_2Q,
        'Idle': error_from_idle,
        'Readout': error_from_readout,
        'Total': total_error
    }

# Current state-of-the-art
budget_current = surface_code_error_budget(
    error_1Q=0.001,
    error_2Q=0.005,
    error_readout=0.01
)

# Improved (Google Willow level)
budget_improved = surface_code_error_budget(
    error_1Q=0.0005,
    error_2Q=0.002,
    error_readout=0.005
)

print("\nError Budget - Current SOA:")
for source, error in budget_current.items():
    print(f"  {source}: {error:.2%}")

print(f"\nSurface code threshold: ~0.6%")
print(f"Status: {'ABOVE' if budget_current['Total'] > 0.006 else 'BELOW'} threshold")

print("\nError Budget - Improved (Willow-level):")
for source, error in budget_improved.items():
    print(f"  {source}: {error:.2%}")

print(f"Status: {'ABOVE' if budget_improved['Total'] > 0.006 else 'BELOW'} threshold")

# =============================================================================
# Part 3: Coherence Comparison Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Coherence Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: T2/T_gate ratio
ax1 = axes[0]
platform_names = ['Supercond.', 'Trapped Ion', 'Neutral Atom', 'Photonic']
t2_over_gate = [
    150e-6 / 40e-9,   # SC
    1 / 100e-6,       # Ion
    0.01 / 1e-6,      # Atom
    1e6               # Photon (approximate)
]

colors = ['blue', 'orange', 'green', 'purple']
bars = ax1.bar(platform_names, t2_over_gate, color=colors)
ax1.set_yscale('log')
ax1.set_ylabel('T2 / Gate Time', fontsize=12)
ax1.set_title('Coherence to Gate Time Ratio', fontsize=14)
ax1.axhline(y=1000, color='r', linestyle='--', label='Good QEC threshold')
ax1.legend()

# Plot 2: Error rates
ax2 = axes[1]
error_1q = [0.01, 0.001, 0.1, 0.1]
error_2q = [0.5, 0.1, 1.0, 1.0]

x = np.arange(len(platform_names))
width = 0.35

bars1 = ax2.bar(x - width/2, error_1q, width, label='1Q Error (%)', color='steelblue')
bars2 = ax2.bar(x + width/2, error_2q, width, label='2Q Error (%)', color='coral')

ax2.set_ylabel('Error Rate (%)', fontsize=12)
ax2.set_title('Gate Error Rates by Platform', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(platform_names)
ax2.legend()
ax2.axhline(y=0.6, color='r', linestyle='--', alpha=0.5)
ax2.text(3.5, 0.65, 'QEC threshold', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('day_999_platform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved platform comparison plots")

# =============================================================================
# Part 4: Rydberg Blockade Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Rydberg Blockade Analysis")
print("=" * 70)

def rydberg_interaction(R, n, species='Rb'):
    """
    Calculate Rydberg interaction strength.

    Args:
        R: interatomic distance in μm
        n: principal quantum number
        species: atomic species

    Returns:
        V: interaction in MHz
    """
    # C6 coefficient scales as n^11
    if species == 'Rb':
        C6_base = 862  # GHz μm^6 for n=70
        n_base = 70
    else:
        C6_base = 500
        n_base = 70

    C6 = C6_base * (n / n_base)**11
    V = C6 * 1e3 / R**6  # MHz

    return V

# Analyze blockade
R_values = np.linspace(2, 10, 100)
n_values = [50, 60, 70, 80]

plt.figure(figsize=(10, 6))

for n in n_values:
    V = [rydberg_interaction(R, n) for R in R_values]
    plt.semilogy(R_values, V, label=f'n = {n}', linewidth=2)

# Typical Rabi frequency
plt.axhline(y=10, color='r', linestyle='--', label='Typical Rabi freq (10 MHz)')
plt.axhline(y=100, color='r', linestyle=':', label='Strong blockade (100 MHz)')

plt.xlabel('Interatomic Distance (μm)', fontsize=12)
plt.ylabel('Interaction Strength (MHz)', fontsize=12)
plt.title('Rydberg Blockade: Interaction vs Distance', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([2, 10])
plt.ylim([1, 1e5])
plt.savefig('day_999_rydberg_blockade.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved Rydberg blockade plot")

# =============================================================================
# Part 5: Roadmap Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Platform Roadmaps")
print("=" * 70)

roadmaps = {
    'Google (Superconducting)': {
        '2024': 'Willow - below threshold',
        '2025': '1000+ qubits, logical operations',
        '2029': 'Useful error-corrected computer'
    },
    'IBM (Superconducting)': {
        '2024': 'Condor 1121 qubits',
        '2025': 'Flamingo - modular systems',
        '2029': '100,000+ qubits'
    },
    'IonQ (Trapped Ion)': {
        '2024': 'Forte - 35 algorithmic qubits',
        '2025': '64+ qubits',
        '2028': 'Fault-tolerant operations'
    },
    'Quantinuum (Trapped Ion)': {
        '2024': 'H2 - 56 qubits, 99.9% 2Q',
        '2025': 'Helios architecture',
        '2028': 'Logical qubit computer'
    },
    'QuEra (Neutral Atom)': {
        '2024': '256 qubits available',
        '2025': '1000+ qubit systems',
        '2028': 'Error-corrected logical qubits'
    }
}

for company, milestones in roadmaps.items():
    print(f"\n{company}:")
    for year, milestone in milestones.items():
        print(f"  {year}: {milestone}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Hardware Platforms Review Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Superconducting: Fastest gates, good scaling, ~0.5% 2Q error
   - Leaders: Google, IBM
   - Best for: Near-term demos, fast algorithms

2. Trapped Ions: Highest fidelity (99.9%), all-to-all connectivity
   - Leaders: IonQ, Quantinuum
   - Best for: High-fidelity small algorithms

3. Neutral Atoms: Largest arrays (1000s), native simulation
   - Leaders: QuEra, Pasqal, Atom Computing
   - Best for: Optimization, simulation

4. Photonic: Room temperature, natural networking
   - Leaders: Xanadu, PsiQuantum
   - Best for: Distributed QC, special applications

5. Topological: Future promise, not yet realized
   - Leader: Microsoft (research)
   - Best for: Long-term fault tolerance

Current Focus: Reducing 2Q error below 0.1% for QEC
""")

print("Review complete!")
```

---

## Summary Tables

### Platform Quick Reference

| Platform | Best Metric | Worst Metric | Maturity |
|----------|-------------|--------------|----------|
| Superconducting | Gate speed | Coherence | High |
| Trapped Ion | Fidelity | Gate speed | High |
| Neutral Atom | Scalability | 2Q fidelity | Medium |
| Photonic | Networking | Determinism | Medium |
| Topological | (Promised) protection | Existence | Low |

### Key Formulas by Platform

| Platform | Key Formula |
|----------|-------------|
| Superconducting | $\omega_{01} = \sqrt{8E_JE_C} - E_C$ |
| Trapped Ion | $\eta = k\sqrt{\hbar/2m\omega}$ |
| Neutral Atom | $V_{dd} = C_6/R^6$ |
| Photonic | $P_{CNOT} = 1/9$ (KLM) |

### Industry Leaders (2025)

| Platform | Leaders |
|----------|---------|
| Superconducting | Google, IBM, Rigetti |
| Trapped Ion | IonQ, Quantinuum, AQT |
| Neutral Atom | QuEra, Pasqal, Atom Computing |
| Photonic | Xanadu, PsiQuantum |
| Topological | Microsoft |

---

## Self-Assessment Checklist

### Physical Understanding
- [ ] Can explain transmon qubit operation
- [ ] Know trapped ion gate mechanisms
- [ ] Understand Rydberg blockade
- [ ] Know photonic encoding schemes

### Quantitative Analysis
- [ ] Can calculate coherence ratios
- [ ] Can estimate error budgets
- [ ] Know typical metric values

### Comparison Skills
- [ ] Can compare platforms for specific applications
- [ ] Understand trade-offs between approaches
- [ ] Know current state-of-the-art

---

## Preview: Day 1000 - MILESTONE!

Tomorrow is **Day 1000** - a major milestone in this curriculum!

We will review **Advanced Quantum Algorithms**, covering:
- HHL algorithm for linear systems
- Quantum simulation and Trotterization
- Quantum machine learning fundamentals
- VQE, QAOA, and variational methods
- Complexity theory and quantum advantage

Plus: Celebration of reaching 1000 days of quantum study!

---

*"Every quantum computer is a bet on which physics will be most controllable at scale."*
--- Anonymous

---

**Next:** [Day_1000_Saturday.md](Day_1000_Saturday.md) - Algorithms Review (Day 1000 Milestone!)
