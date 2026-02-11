# Day 837: Trapped-Ion and Neutral Atom Implementations

## Week 120, Day 4 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today we explore surface code implementations on trapped-ion and neutral atom platforms. These technologies offer fundamentally different advantages compared to superconducting qubits: trapped ions provide all-to-all connectivity and exceptional gate fidelities, while neutral atom arrays offer massive parallelism and reconfigurability. Both platforms have demonstrated significant progress toward fault-tolerant quantum computing.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Trapped-ion QEC fundamentals |
| **Afternoon** | 2.5 hours | Neutral atom surface codes |
| **Evening** | 1.5 hours | Computational lab: Platform comparison |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain trapped-ion surface codes** - Understand shuttling-based and photonic-linked architectures
2. **Analyze neutral atom implementations** - Evaluate Rydberg-based two-qubit gates for QEC
3. **Compare platform characteristics** - Quantify trade-offs in fidelity, speed, and scalability
4. **Assess real-time reconfigurability** - Understand dynamic qubit rearrangement
5. **Evaluate scaling approaches** - Compare modular vs. monolithic architectures
6. **Identify optimal applications** - Match platforms to quantum computing tasks

---

## Core Content

### 1. Trapped-Ion Quantum Computing

#### 1.1 Physical System

Trapped ions are individual atoms (typically $^{171}$Yb$^+$, $^{137}$Ba$^+$, or $^{40}$Ca$^+$) confined in electromagnetic traps:

**Paul Trap Potential:**
$$\Phi(x, y, z, t) = \frac{V_{RF}}{r_0^2}(x^2 - y^2)\cos(\Omega_{RF}t) + \frac{U_{DC}}{z_0^2}\left(z^2 - \frac{x^2 + y^2}{2}\right)$$

This creates an effective harmonic potential:
$$\boxed{V_{\text{eff}}(r) \approx \frac{1}{2}m\omega_r^2 r^2 + \frac{1}{2}m\omega_z^2 z^2}$$

where $\omega_r, \omega_z$ are the radial and axial trap frequencies.

#### 1.2 Qubit Encoding

**Hyperfine qubits** (e.g., $^{171}$Yb$^+$):
$$|0\rangle = |F=0, m_F=0\rangle, \quad |1\rangle = |F=1, m_F=0\rangle$$

**Coherence properties:**
- $T_1$: Effectively infinite (no spontaneous decay between hyperfine states)
- $T_2$: 1-10 seconds (limited by magnetic field fluctuations)
- Clock transitions: $T_2 > 1000$ s demonstrated

#### 1.3 Gate Operations

**Single-qubit gates:** Microwave or Raman transitions
$$R(\theta, \phi) = \exp\left(-i\frac{\theta}{2}(\cos\phi \, \sigma_x + \sin\phi \, \sigma_y)\right)$$

Fidelity: >99.99%

**Two-qubit gates:** Molmer-Sorensen (MS) gate using collective motional modes

$$\boxed{MS(\theta) = \exp\left(-i\frac{\theta}{4}\sigma_x^{(1)}\sigma_x^{(2)}\right)}$$

The MS gate creates entanglement through:
1. Off-resonant coupling to motional sidebands
2. Spin-dependent force creates displacement
3. Closed loop in phase space = geometric phase
4. Phase depends on product of spins

**Current state-of-the-art fidelities:**

| Company | 2Q Gate Fidelity | Gate Time |
|---------|------------------|-----------|
| IonQ | 99.5% | 600 μs |
| Quantinuum | 99.8% | 200 μs |
| Oxford Ionics | 99.9% | 100 μs |

### 2. Surface Codes on Trapped Ions

#### 2.1 Connectivity Advantage

Trapped ions in a linear chain have all-to-all connectivity:

$$\boxed{\text{Any qubit } i \text{ can interact with any qubit } j}$$

This eliminates the need for:
- SWAP gates to move information
- Local connectivity constraints
- Fixed stabilizer assignment

#### 2.2 Architectures for Scaling

**Linear Chain Approach:**
- Up to ~50 ions in one chain
- All-to-all connectivity maintained
- Syndrome extraction parallelized differently

**Quantum Charge-Coupled Device (QCCD):**
```
Zone 1          Zone 2          Zone 3
[■ ■ ■]---→---[■ ■]---→---[■ ■ ■]
Storage       Gate region      Storage
```

Ions shuttled between zones for:
- Storage (long coherence)
- Gates (interaction region)
- Measurement (detection region)

**Photonic Interconnects:**
```
Trap 1 ═══╗     ╔═══ Trap 2
          ║     ║
          ╚══◊══╝
         Photonic
          link
```

Remote entanglement via photon interference enables:
- Modular scaling
- Distributed quantum computing
- Lower per-module complexity

#### 2.3 Surface Code Implementation

For a distance-$d$ surface code on trapped ions:

**Option 1: Single Chain**
- $2d^2 - 1$ ions in one trap
- All-to-all enables flexible stabilizer measurement
- Limited to d ~ 5-7 by chain stability

**Option 2: QCCD Architecture**
- Multiple zones with shuttling
- Stabilizers measured by bringing ions together
- Scalable to larger distances

**Shuttling overhead:**
$$T_{\text{shuttle}} \approx 10-100 \text{ μs per move}$$

#### 2.4 Error Budget for Trapped Ions

| Error Source | Rate | Notes |
|--------------|------|-------|
| 2Q gate | 0.1-0.5% | MS gate infidelity |
| Measurement | 0.1% | State-dependent fluorescence |
| Memory (1 ms) | 0.01% | Exceptional coherence |
| Shuttling | 0.1% | Heating during transport |
| Crosstalk | 0.01% | Addressing errors |

**Total cycle error (estimated):**
$$p_{\text{cycle}}^{\text{ion}} \approx 0.5-1\%$$

Comparable to superconducting, but with much longer coherence.

### 3. Neutral Atom Quantum Computing

#### 3.1 Physical System

Neutral atoms (typically $^{87}$Rb, $^{133}$Cs, or $^{88}$Sr) trapped in optical tweezers:

**Dipole Trap Potential:**
$$U(\mathbf{r}) = -\frac{1}{2}\alpha(\omega)|\mathbf{E}(\mathbf{r})|^2$$

where $\alpha(\omega)$ is the polarizability.

**Optical Tweezer Array:**
- Individual focused laser beams
- Typical spacing: 3-10 μm
- Arrays of 100-1000+ atoms demonstrated
- Real-time reconfigurable positions

#### 3.2 Qubit Encoding

**Ground-state hyperfine qubits:**
$$|0\rangle = |F=1, m_F=0\rangle, \quad |1\rangle = |F=2, m_F=0\rangle$$

**Coherence:**
- $T_2^* \approx 1-10$ ms (magnetic field limited)
- $T_2^{\text{echo}} \approx 1-4$ s (with dynamical decoupling)

#### 3.3 Rydberg-Mediated Gates

Two-qubit gates use the Rydberg blockade mechanism:

**Rydberg states:** High principal quantum number ($n \sim 50-100$)
$$|r\rangle = |n, l, j, m_j\rangle$$

**Dipole-dipole interaction:**
$$\boxed{V_{dd}(R) = \frac{C_6}{R^6}}$$

where $C_6 \sim n^{11}$ scales dramatically with $n$.

**Blockade radius:**
$$R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}$$

For $n=70$ and $\Omega = 2\pi \times 5$ MHz: $R_b \approx 10$ μm

**CZ Gate Protocol:**
1. Apply π pulse: $|1\rangle \rightarrow |r\rangle$ on atom 1
2. Apply 2π pulse on atom 2 (blocked if atom 1 in $|r\rangle$)
3. Apply π pulse: $|r\rangle \rightarrow |1\rangle$ on atom 1

Result: Phase only on $|11\rangle$ state.

**Current fidelities:**
- 2Q gate: 99.5% (QuEra, Atom Computing)
- Gate time: 200-500 ns
- Parallel gates on entire array

#### 3.4 Reconfigurability Advantage

Neutral atoms can be physically rearranged:

```
Initial:            After rearrangement:
● ● ● ● ●          ● ● ○ ● ●
● ○ ● ● ●    →     ● ● ● ● ●
● ● ● ○ ●          ● ● ● ● ●

(○ = vacancy, filled by moving atoms)
```

**Advantages for QEC:**
- Defect-free arrays by rearranging
- Adapt to qubit loss
- Optimize stabilizer connectivity

**Rearrangement time:** ~10 ms (slow compared to gate operations)

### 4. QuEra's Approach to Surface Codes

#### 4.1 Hardware Overview

QuEra's Aquila processor:
- 256 atoms in programmable array
- Rydberg-based entanglement
- Native Ising-type Hamiltonian
- Global control with local addressing

#### 4.2 Surface Code Layout

```
Neutral atom surface code array:

    ●───●───●───●───●
    │ Z │ X │ Z │ X │
    ●───●───●───●───●
    │ X │ Z │ X │ Z │
    ●───●───●───●───●
    │ Z │ X │ Z │ X │
    ●───●───●───●───●

    ● = Atom (data or ancilla)
    Z, X = Stabilizer type
```

**Zoned architecture:**
- Data atoms: Fixed positions
- Ancilla atoms: Moved for measurement
- Rearrangement between syndrome rounds

#### 4.3 Parallel Operations

Key advantage: All stabilizers can be measured in parallel

$$\boxed{T_{\text{syndrome}} \propto \text{const}, \text{ not } O(\text{code size})}$$

For a d×d surface code:
- ~$d^2$ stabilizers
- All measured simultaneously with global Rydberg pulses
- Constant depth per syndrome cycle

### 5. Platform Comparison

#### 5.1 Quantitative Comparison

| Metric | Superconducting | Trapped Ion | Neutral Atom |
|--------|----------------|-------------|--------------|
| 2Q Gate Fidelity | 99.5-99.75% | 99.5-99.9% | 99-99.5% |
| 2Q Gate Time | 20-50 ns | 100-600 μs | 200-500 ns |
| T2 Coherence | 10-100 μs | 1-10 s | 1-1000 ms |
| Connectivity | Local (4) | All-to-all | Reconfigurable |
| Qubit Count (2024) | ~100-1000 | ~30-50 | ~100-1000 |
| Scalability Path | Monolithic | Modular | Monolithic |

#### 5.2 Threshold Considerations

**Superconducting:** Standard surface code threshold ~1%

**Trapped ions:** Higher fidelity but slower cycles
$$p_{\text{th}}^{\text{ion}} \sim 1\% \text{ (limited by cycle time vs coherence)}$$

**Neutral atoms:** Mid-range fidelity, fast parallel operations
$$p_{\text{th}}^{\text{atom}} \sim 0.5-1\% \text{ (limited by Rydberg fidelity)}$$

#### 5.3 Error Suppression Projections

All platforms can potentially achieve λ > 2 with continued improvement:

| Platform | Current λ (est.) | 2026 Target |
|----------|------------------|-------------|
| Google Willow | 2.14 | 3+ |
| IBM Heron | ~2 | 2.5+ |
| Quantinuum | ~1.5 | 2+ |
| QuEra | ~1.2 | 2+ |

### 6. Hybrid and Modular Approaches

#### 6.1 Photonic Networks

Connecting ion traps or neutral atom modules via photons:

$$|entangled\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)_{\text{remote}}$$

**Rate-fidelity trade-off:**
- High-fidelity links: ~10 Hz, 99%+ fidelity
- Fast links: ~10 kHz, 90% fidelity

#### 6.2 Distributed Surface Codes

For a distance-d surface code across multiple modules:

$$\text{Links needed} \approx O(d) \text{ between neighboring modules}$$

The logical error rate with imperfect links:
$$p_L^{\text{distributed}} \approx p_L^{\text{local}} + p_{\text{link}} \cdot d$$

This sets requirements on link fidelity for scaling.

---

## Worked Examples

### Example 1: Molmer-Sorensen Gate Fidelity

**Problem:** A Molmer-Sorensen gate is implemented with Rabi frequency $\Omega = 2\pi \times 100$ kHz, detuning $\delta = 2\pi \times 10$ kHz from the motional sideband, and motional heating rate $\dot{\bar{n}} = 100$ quanta/s. Estimate the gate fidelity if the gate time is $T_g = 200$ μs.

**Solution:**

**Ideal gate time:**
For an MS gate with detuning $\delta$ and Rabi frequency $\Omega$:
$$T_g = \frac{2\pi}{\delta} \cdot \frac{\delta^2}{2\Omega^2/\eta^2}$$

where $\eta \approx 0.1$ is the Lamb-Dicke parameter. Simplified:
$$T_g \approx \frac{\pi}{\eta^2 \Omega^2 / \delta}$$

**Heating error:**
Motional heating during the gate:
$$\Delta \bar{n} = \dot{\bar{n}} \cdot T_g = 100 \times 200 \times 10^{-6} = 0.02 \text{ quanta}$$

The infidelity from heating:
$$\epsilon_{\text{heat}} \approx \eta^2 \Delta \bar{n} = 0.01 \times 0.02 = 0.0002$$

**Off-resonant scattering:**
If scattering rate is $\Gamma_{\text{sc}} \approx 1$ Hz:
$$\epsilon_{\text{sc}} = \Gamma_{\text{sc}} \cdot T_g = 1 \times 200 \times 10^{-6} = 0.0002$$

**Total infidelity:**
$$\epsilon_{\text{total}} \approx \epsilon_{\text{heat}} + \epsilon_{\text{sc}} = 0.0004$$

$$\boxed{F = 1 - \epsilon \approx 99.96\%}$$

### Example 2: Rydberg Blockade Calculation

**Problem:** Two $^{87}$Rb atoms are separated by $R = 5$ μm. The Rydberg state $|70S\rangle$ has $C_6 = 2\pi \times 850$ GHz·μm$^6$. Calculate:
a) The interaction energy at this separation
b) The blockade shift
c) Whether this enables a high-fidelity CZ gate with $\Omega = 2\pi \times 3$ MHz

**Solution:**

a) **Interaction energy:**
$$V_{dd} = \frac{C_6}{R^6} = \frac{2\pi \times 850 \text{ GHz·μm}^6}{(5 \text{ μm})^6}$$
$$V_{dd} = \frac{2\pi \times 850 \times 10^9}{15625} \text{ Hz} = 2\pi \times 54.4 \text{ MHz}$$

b) **Blockade shift:**
The blockade radius is where $V_{dd} = \hbar\Omega$:
$$R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6} = \left(\frac{850 \text{ GHz·μm}^6}{3 \text{ MHz}}\right)^{1/6}$$
$$R_b = (283,000)^{1/6} \text{ μm} = 8.1 \text{ μm}$$

Since $R = 5$ μm $< R_b = 8.1$ μm, we are in the blockade regime.

c) **Gate fidelity consideration:**
The blockade-to-Rabi ratio:
$$\frac{V_{dd}}{\hbar\Omega} = \frac{54.4}{3} = 18.1$$

For high fidelity, we need this ratio $\gg 1$. With ratio ~18, the gate error is approximately:
$$\epsilon \approx \left(\frac{\Omega}{V_{dd}/\hbar}\right)^2 = \left(\frac{1}{18.1}\right)^2 \approx 0.003$$

$$\boxed{V_{dd} = 2\pi \times 54 \text{ MHz}, \quad R_b = 8.1 \text{ μm}, \quad F \approx 99.7\%}$$

### Example 3: Parallel vs Sequential Syndrome Extraction

**Problem:** Compare syndrome extraction time for a d=11 surface code on:
a) Superconducting qubits (sequential 4-layer CZ, 1 μs total)
b) Neutral atoms (parallel Rydberg gates, 500 ns per gate, but global addressing)

**Solution:**

**Surface code d=11:**
- Data qubits: $11^2 = 121$
- Stabilizers: $11^2 - 1 = 120$

a) **Superconducting:**
Circuit depth is constant (4 CZ layers + measurement):
$$T_{\text{SC}} = 4 \times 25 \text{ ns} + 500 \text{ ns (readout)} = 600 \text{ ns}$$

With cycle overhead: ~1 μs (as measured on Willow)

This is **independent of code distance** due to parallel operations.

b) **Neutral atoms:**
If all stabilizers measured in parallel:
- 4 Rydberg pulses per stabilizer (for weight-4)
- Global pulses: 4 × 500 ns = 2 μs
- Measurement: 10 μs (fluorescence imaging)

$$T_{\text{NA}} = 2 + 10 = 12 \text{ μs}$$

But if coherence $T_2 = 1$ s, the cycle-to-coherence ratio is:
$$\frac{T_{\text{NA}}}{T_2} = \frac{12 \text{ μs}}{1 \text{ s}} = 1.2 \times 10^{-5}$$

For superconducting with $T_2 = 30$ μs:
$$\frac{T_{\text{SC}}}{T_2} = \frac{1 \text{ μs}}{30 \text{ μs}} = 3.3 \times 10^{-2}$$

$$\boxed{\text{Neutral atoms: } 12 \text{ μs cycle, 1.2×10}^{-5} \text{ coherence fraction}}$$
$$\boxed{\text{Superconducting: } 1 \text{ μs cycle, 3.3×10}^{-2} \text{ coherence fraction}}$$

---

## Practice Problems

### Direct Application

**Problem 1:** A trapped-ion chain has 25 ions with all-to-all connectivity. How many native 2-qubit interactions are possible? Compare to a 25-qubit superconducting grid with nearest-neighbor only.

**Problem 2:** Calculate the Rydberg blockade radius for $^{133}$Cs in the $|60S\rangle$ state with $C_6 = 2\pi \times 150$ GHz·μm$^6$ and driving Rabi frequency $\Omega = 2\pi \times 5$ MHz.

**Problem 3:** A neutral atom array loses 5% of atoms during a computation. For a d=5 surface code with 49 qubits, what is the probability that the code remains operational (no data qubit lost)?

### Intermediate

**Problem 4:** The Quantinuum H2 processor has 32 qubits with 99.8% two-qubit gate fidelity and 1 s coherence time. Gates take 200 μs. Estimate:
a) The number of gates possible before coherence-limited error reaches 1%
b) Whether this supports below-threshold operation for d=5

**Problem 5:** QuEra proposes a 1000-atom array for surface codes. If each stabilizer measurement requires 4 parallel Rydberg gates (500 ns each) and 10 μs readout:
a) What is the syndrome cycle time?
b) For $T_2 = 2$ s, what is the coherence-limited error per cycle?
c) What code distance is achievable with λ = 1.5?

**Problem 6:** Compare the "error-per-gate-time" metric for:
- Google CZ: 0.25% in 25 ns
- Quantinuum MS: 0.2% in 200 μs
- QuEra Rydberg CZ: 0.5% in 400 ns

Which is most efficient in terms of error per nanosecond?

### Challenging

**Problem 7:** A distributed surface code uses photonic links with 90% fidelity at 1 kHz rate between modules. Each module holds a d=5 surface code patch. Derive the effective threshold for the distributed code as a function of the number of inter-module links per boundary.

**Problem 8:** Design a shuttling schedule for a d=3 surface code on a QCCD ion trap with 3 zones (each holding up to 5 ions). Minimize the total shuttling time for one syndrome extraction round.

---

## Computational Lab: Platform Comparison Tool

```python
"""
Day 837 Computational Lab: Trapped-Ion and Neutral Atom Platform Analysis
Comprehensive comparison of QEC platforms
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# =============================================================================
# Part 1: Platform Specifications
# =============================================================================

@dataclass
class QuantumPlatform:
    """Specifications for a quantum computing platform."""
    name: str
    two_q_fidelity: float  # Two-qubit gate fidelity
    two_q_time: float  # Two-qubit gate time in seconds
    t1: float  # T1 coherence in seconds
    t2: float  # T2 coherence in seconds
    measurement_time: float  # Measurement time in seconds
    measurement_fidelity: float
    connectivity: str  # 'all-to-all', 'nearest-neighbor', 'reconfigurable'
    max_qubits: int  # Current demonstrated qubit count

# Define platforms
platforms = {
    'Google Willow': QuantumPlatform(
        name='Google Willow',
        two_q_fidelity=0.9975,
        two_q_time=25e-9,
        t1=68e-6,
        t2=30e-6,
        measurement_time=500e-9,
        measurement_fidelity=0.993,
        connectivity='nearest-neighbor',
        max_qubits=105
    ),
    'IBM Heron': QuantumPlatform(
        name='IBM Heron',
        two_q_fidelity=0.997,
        two_q_time=400e-9,
        t1=300e-6,
        t2=200e-6,
        measurement_time=1e-6,
        measurement_fidelity=0.99,
        connectivity='nearest-neighbor',
        max_qubits=133
    ),
    'Quantinuum H2': QuantumPlatform(
        name='Quantinuum H2',
        two_q_fidelity=0.998,
        two_q_time=200e-6,
        t1=float('inf'),  # Hyperfine states
        t2=1.0,
        measurement_time=100e-6,
        measurement_fidelity=0.999,
        connectivity='all-to-all',
        max_qubits=32
    ),
    'IonQ Forte': QuantumPlatform(
        name='IonQ Forte',
        two_q_fidelity=0.995,
        two_q_time=600e-6,
        t1=float('inf'),
        t2=1.0,
        measurement_time=100e-6,
        measurement_fidelity=0.995,
        connectivity='all-to-all',
        max_qubits=36
    ),
    'QuEra Aquila': QuantumPlatform(
        name='QuEra Aquila',
        two_q_fidelity=0.995,
        two_q_time=400e-9,
        t1=10.0,  # Limited by trap lifetime
        t2=2.0,
        measurement_time=10e-6,
        measurement_fidelity=0.99,
        connectivity='reconfigurable',
        max_qubits=256
    ),
    'Atom Computing': QuantumPlatform(
        name='Atom Computing',
        two_q_fidelity=0.995,
        two_q_time=300e-9,
        t1=10.0,
        t2=4.0,  # With dynamical decoupling
        measurement_time=5e-6,
        measurement_fidelity=0.99,
        connectivity='reconfigurable',
        max_qubits=1000
    )
}

print("=" * 70)
print("QUANTUM PLATFORM SPECIFICATIONS")
print("=" * 70)
print(f"{'Platform':<20} {'2Q Fid.':<10} {'2Q Time':<12} {'T2':<12} {'Qubits':<8}")
print("-" * 70)
for name, p in platforms.items():
    t2_str = f"{p.t2*1e3:.0f} ms" if p.t2 < 10 else f"{p.t2:.0f} s"
    t_str = f"{p.two_q_time*1e9:.0f} ns" if p.two_q_time < 1e-6 else f"{p.two_q_time*1e6:.0f} μs"
    print(f"{name:<20} {p.two_q_fidelity*100:.2f}%{'':<4} {t_str:<12} {t2_str:<12} {p.max_qubits:<8}")

# =============================================================================
# Part 2: Syndrome Cycle Analysis
# =============================================================================

def estimate_cycle_time(platform: QuantumPlatform, code_distance: int) -> float:
    """
    Estimate syndrome extraction cycle time.

    Parameters:
    -----------
    platform : QuantumPlatform
        Platform specifications
    code_distance : int
        Surface code distance

    Returns:
    --------
    cycle_time : float
        Estimated cycle time in seconds
    """
    if platform.connectivity == 'all-to-all':
        # Ions: can do stabilizers with O(weight) gates, but sequentially
        n_stabilizers = code_distance**2 - 1
        # Simplified: each stabilizer takes 4 2Q gates + measurement
        cycle_time = n_stabilizers * (4 * platform.two_q_time + platform.measurement_time)
        # With parallelization tricks, reduce by factor
        cycle_time /= code_distance  # Approximate parallelization

    elif platform.connectivity == 'reconfigurable':
        # Neutral atoms: parallel gates on all stabilizers
        # 4 layers of Rydberg gates + measurement
        cycle_time = 4 * platform.two_q_time + platform.measurement_time

    else:
        # Superconducting: 4 CZ layers + measurement
        cycle_time = 4 * platform.two_q_time + platform.measurement_time

    return cycle_time

def estimate_cycle_error(platform: QuantumPlatform, code_distance: int) -> float:
    """
    Estimate error per syndrome cycle.

    Returns:
    --------
    cycle_error : float
        Estimated error per cycle
    """
    cycle_time = estimate_cycle_time(platform, code_distance)

    # Gate errors (4 2Q gates per data qubit per cycle, approximately)
    gate_error = 4 * (1 - platform.two_q_fidelity)

    # Coherence error
    if platform.t2 < float('inf'):
        coherence_error = cycle_time / platform.t2
    else:
        coherence_error = 0

    # Measurement error
    meas_error = 1 - platform.measurement_fidelity

    total_error = gate_error + coherence_error + meas_error
    return total_error

print("\n" + "=" * 70)
print("SYNDROME CYCLE ANALYSIS (d=7)")
print("=" * 70)
d = 7
print(f"{'Platform':<20} {'Cycle Time':<15} {'Cycle Error':<15}")
print("-" * 50)
for name, p in platforms.items():
    cycle_t = estimate_cycle_time(p, d)
    cycle_e = estimate_cycle_error(p, d)
    if cycle_t < 1e-3:
        t_str = f"{cycle_t*1e6:.1f} μs"
    else:
        t_str = f"{cycle_t*1e3:.1f} ms"
    print(f"{name:<20} {t_str:<15} {cycle_e*100:.2f}%")

# =============================================================================
# Part 3: Error Suppression Factor Estimation
# =============================================================================

def estimate_lambda(platform: QuantumPlatform, threshold: float = 0.01) -> float:
    """
    Estimate error suppression factor.

    lambda = p_th / p_eff
    """
    p_eff = estimate_cycle_error(platform, 7)  # Use d=7 as reference
    if p_eff >= threshold:
        return 0.5  # Above threshold
    return threshold / p_eff

print("\n" + "=" * 70)
print("ERROR SUPPRESSION FACTOR ESTIMATES")
print("=" * 70)
for name, p in platforms.items():
    lam = estimate_lambda(p)
    status = "Below threshold" if lam > 1 else "At/above threshold"
    print(f"{name:<20}: λ ≈ {lam:.2f} ({status})")

# =============================================================================
# Part 4: Rydberg Physics Calculations
# =============================================================================

def rydberg_blockade_radius(C6: float, Omega: float) -> float:
    """
    Calculate Rydberg blockade radius.

    Parameters:
    -----------
    C6 : float
        van der Waals coefficient in Hz·μm^6
    Omega : float
        Rabi frequency in Hz

    Returns:
    --------
    R_b : float
        Blockade radius in μm
    """
    return (C6 / Omega) ** (1/6)

def rydberg_interaction(C6: float, R: float) -> float:
    """
    Calculate Rydberg dipole-dipole interaction.

    Parameters:
    -----------
    C6 : float
        van der Waals coefficient in Hz·μm^6
    R : float
        Interatomic distance in μm

    Returns:
    --------
    V : float
        Interaction energy in Hz
    """
    return C6 / R**6

print("\n" + "=" * 70)
print("RYDBERG PHYSICS ANALYSIS")
print("=" * 70)

# Rubidium-87 parameters
C6_Rb_70S = 2 * np.pi * 850e9  # Hz·μm^6
Omega_typical = 2 * np.pi * 5e6  # 5 MHz Rabi frequency

R_b = rydberg_blockade_radius(C6_Rb_70S, Omega_typical)
print(f"\nRb-87 |70S⟩ state:")
print(f"  C6 = 2π × 850 GHz·μm^6")
print(f"  Ω = 2π × 5 MHz")
print(f"  Blockade radius: {R_b:.2f} μm")

# Interaction at various distances
print(f"\n  Interaction energy vs distance:")
for R in [3, 5, 7, 10]:
    V = rydberg_interaction(C6_Rb_70S, R)
    print(f"    R = {R} μm: V = 2π × {V/(2*np.pi*1e6):.1f} MHz")

# =============================================================================
# Part 5: Trapped Ion Gate Analysis
# =============================================================================

def ms_gate_fidelity(Omega: float, delta: float, n_bar: float,
                    eta: float = 0.1, T_gate: float = None) -> float:
    """
    Estimate Molmer-Sorensen gate fidelity.

    Parameters:
    -----------
    Omega : float
        Rabi frequency in Hz
    delta : float
        Detuning from sideband in Hz
    n_bar : float
        Mean phonon number
    eta : float
        Lamb-Dicke parameter
    T_gate : float
        Gate time (if None, calculated from Omega, delta)

    Returns:
    --------
    fidelity : float
        Estimated gate fidelity
    """
    if T_gate is None:
        # Simplified gate time estimate
        T_gate = np.pi * delta / (2 * (eta * Omega)**2)

    # Heating error
    epsilon_heat = eta**2 * n_bar * T_gate * 1e3  # Assuming 1000 quanta/s heating

    # Off-resonant error
    epsilon_off = (Omega / delta)**2

    # Total infidelity
    epsilon = epsilon_heat + epsilon_off

    return 1 - min(epsilon, 1)

print("\n" + "=" * 70)
print("TRAPPED ION GATE ANALYSIS")
print("=" * 70)

# Typical parameters
Omega_ion = 2 * np.pi * 100e3  # 100 kHz
delta_ion = 2 * np.pi * 10e3  # 10 kHz detuning
eta = 0.1
n_bar = 0.1  # After cooling

fid = ms_gate_fidelity(Omega_ion, delta_ion, n_bar, eta)
print(f"Molmer-Sorensen gate parameters:")
print(f"  Rabi frequency: 2π × 100 kHz")
print(f"  Detuning: 2π × 10 kHz")
print(f"  Lamb-Dicke parameter: {eta}")
print(f"  Mean phonon number: {n_bar}")
print(f"  Estimated fidelity: {fid*100:.2f}%")

# =============================================================================
# Part 6: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Platform comparison radar chart (simplified as bar chart)
ax1 = axes[0, 0]
metrics = ['2Q Fidelity\n(%)', 'Coherence\n(log10 T2/s)', 'Speed\n(-log10 gate/s)',
           'Scale\n(log10 qubits)']

platform_names = list(platforms.keys())[:4]  # First 4 platforms
x = np.arange(len(metrics))
width = 0.2

for i, name in enumerate(platform_names):
    p = platforms[name]
    values = [
        p.two_q_fidelity * 100,
        np.log10(p.t2) if p.t2 < float('inf') else 2,
        -np.log10(p.two_q_time),
        np.log10(p.max_qubits)
    ]
    # Normalize to 0-10 scale
    normalized = [v / max(values) * 10 for v in values]
    ax1.bar(x + i * width, values, width, label=name.split()[0])

ax1.set_xticks(x + 1.5 * width)
ax1.set_xticklabels(metrics)
ax1.legend(loc='upper right')
ax1.set_title('Platform Comparison (Raw Metrics)', fontsize=14)
ax1.set_ylabel('Value')

# Plot 2: Rydberg blockade
ax2 = axes[0, 1]
R_range = np.linspace(2, 15, 100)
V_range = [rydberg_interaction(C6_Rb_70S, R) / (2 * np.pi * 1e6) for R in R_range]

ax2.semilogy(R_range, V_range, 'b-', linewidth=2, label='V(R)')
ax2.axhline(y=Omega_typical / (2 * np.pi * 1e6), color='r', linestyle='--',
            linewidth=2, label=f'Ω = 5 MHz')
ax2.axvline(x=R_b, color='g', linestyle=':', linewidth=2, label=f'R_b = {R_b:.1f} μm')
ax2.fill_between(R_range[R_range < R_b],
                 [rydberg_interaction(C6_Rb_70S, R) / (2 * np.pi * 1e6) for R in R_range[R_range < R_b]],
                 alpha=0.3, color='green', label='Blockade regime')
ax2.set_xlabel('Interatomic Distance (μm)', fontsize=12)
ax2.set_ylabel('Interaction Energy (MHz)', fontsize=12)
ax2.set_title('Rydberg Blockade: Rb-87 |70S⟩', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2, 15)
ax2.set_ylim(0.1, 1000)

# Plot 3: Cycle time vs coherence
ax3 = axes[1, 0]
cycle_times = []
coherence_ratios = []
colors_p = ['#4285f4', '#0f62fe', '#00758f', '#c41e3a', '#9333ea', '#ea580c']

for i, (name, p) in enumerate(platforms.items()):
    ct = estimate_cycle_time(p, 7)
    cycle_times.append(ct)
    if p.t2 < float('inf'):
        coherence_ratios.append(ct / p.t2)
    else:
        coherence_ratios.append(ct / 10)  # Use 10s for "infinite"

ax3.scatter(cycle_times, coherence_ratios, s=200, c=colors_p, edgecolors='black',
            linewidths=2, zorder=3)
for i, name in enumerate(platforms.keys()):
    ax3.annotate(name.split()[0], (cycle_times[i], coherence_ratios[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Syndrome Cycle Time (s)', fontsize=12)
ax3.set_ylabel('Cycle Time / T2 (coherence error)', fontsize=12)
ax3.set_title('Coherence Budget per Cycle', fontsize=14)
ax3.grid(True, alpha=0.3, which='both')
ax3.axhline(y=0.01, color='r', linestyle='--', label='1% coherence error')
ax3.legend()

# Plot 4: Projected logical error rates
ax4 = axes[1, 1]
distances = np.arange(3, 22, 2)

for name, p in list(platforms.items())[:4]:
    lam = estimate_lambda(p)
    if lam > 0:
        p_d7 = estimate_cycle_error(p, 7)
        errors = [p_d7 * (1/max(lam, 0.1))**((d-7)/2) for d in distances]
        ax4.semilogy(distances, np.array(errors) * 100, 'o-', label=name.split()[0],
                    linewidth=2, markersize=6)

ax4.set_xlabel('Code Distance', fontsize=12)
ax4.set_ylabel('Logical Error Rate (%)', fontsize=12)
ax4.set_title('Projected Logical Error Rates', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim(2, 22)

plt.tight_layout()
plt.savefig('day_837_platform_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Visualization saved to: day_837_platform_comparison.png")
print("=" * 70)

# =============================================================================
# Part 7: Optimal Platform Selection
# =============================================================================

print("\n" + "=" * 70)
print("OPTIMAL PLATFORM SELECTION GUIDE")
print("=" * 70)

applications = {
    'NISQ Variational': 'High gate count, moderate fidelity → Neutral Atoms, Superconducting',
    'Deep Circuits': 'Very high fidelity needed → Trapped Ions (Quantinuum)',
    'Large QEC': 'Many qubits + error correction → Neutral Atoms, Superconducting',
    'Fast Feedback': 'Quick cycle times → Superconducting (Google, IBM)',
    'All-to-all': 'Complex connectivity patterns → Trapped Ions (IonQ)',
    'Research Prototype': 'Reconfigurability → Neutral Atoms (QuEra)'
}

for app, recommendation in applications.items():
    print(f"\n{app}:")
    print(f"  → {recommendation}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Paul trap potential | $V_{\text{eff}} = \frac{1}{2}m\omega^2 r^2$ |
| MS gate | $MS(\theta) = \exp(-i\frac{\theta}{4}\sigma_x^{(1)}\sigma_x^{(2)})$ |
| Rydberg interaction | $V_{dd} = C_6/R^6$ |
| Blockade radius | $R_b = (C_6/\hbar\Omega)^{1/6}$ |
| Coherence error | $p_{\text{coh}} = T_{\text{cycle}}/T_2$ |

### Main Takeaways

1. **Trapped ions achieve highest gate fidelities** - 99.8%+ demonstrated, ideal for deep circuits
2. **Neutral atoms offer massive parallelism** - 1000+ qubits with reconfigurable geometry
3. **All platforms face different trade-offs** - Speed vs. fidelity vs. connectivity
4. **Coherence advantage of atoms/ions is significant** - Seconds vs. microseconds
5. **Rydberg blockade enables fast neutral atom gates** - 100-500 ns for CZ
6. **No single platform dominates** - Application determines optimal choice

### Daily Checklist

- [ ] I understand the Molmer-Sorensen gate mechanism for trapped ions
- [ ] I can calculate Rydberg blockade parameters
- [ ] I can compare syndrome cycle times across platforms
- [ ] I understand the connectivity trade-offs (all-to-all vs local)
- [ ] I can estimate error suppression factors for different platforms
- [ ] I completed the platform comparison lab

---

## Preview: Day 838

Tomorrow we examine scaling roadmaps toward 1000+ logical qubits. We will analyze resource estimates for practical quantum algorithms, timeline projections from major companies, and the engineering challenges of building million-qubit systems.

**Key topics:**
- Resource estimates for fault-tolerant algorithms
- Roadmaps from Google, IBM, Microsoft, IonQ
- Cryogenic and control system scaling
- Timeline projections for quantum advantage
