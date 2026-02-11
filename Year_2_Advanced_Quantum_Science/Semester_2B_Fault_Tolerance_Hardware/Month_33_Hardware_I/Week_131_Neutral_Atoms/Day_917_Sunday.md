# Day 917: Mid-Circuit Measurement

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Dual-species systems, ancilla protocols, feedback |
| **Afternoon** | 2 hours | Problem solving: syndrome extraction |
| **Evening** | 2 hours | Computational lab: measurement simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Explain the challenges** of mid-circuit measurement in neutral atom systems
2. **Design dual-species architectures** for non-destructive measurement
3. **Implement ancilla-based protocols** for syndrome extraction
4. **Analyze real-time feedback** requirements and latency constraints
5. **Apply mid-circuit measurement** to quantum error correction
6. **Simulate measurement-based protocols** for fault-tolerant computation

## Core Content

### 1. The Mid-Circuit Measurement Challenge

#### Standard Measurement in Neutral Atoms

Conventional neutral atom measurement uses **state-dependent fluorescence**:
1. Apply resonant light to detect state $|1\rangle$
2. Collect scattered photons over 10-30 ms
3. Atom heated/lost during measurement

**Problem for mid-circuit measurement:**
- Measurement destroys coherence of neighboring qubits
- Photon scattering causes heating
- Long measurement time limits circuit depth
- Atom loss requires replacement

#### Requirements for Mid-Circuit Measurement

| Requirement | Challenge | Solution Approach |
|------------|-----------|-------------------|
| Non-destructive | Fluorescence heats atoms | Shelving, dual-species |
| Qubit-selective | Global light affects all | Local addressing |
| Fast feedback | 10+ ms latency | Fast electronics |
| Repeated | Atom loss compounds | Atom replacement |

### 2. Dual-Species Architectures

#### Concept

Use two atomic species with different resonances:
- **Data qubits**: Species A (e.g., Rb-87)
- **Ancilla qubits**: Species B (e.g., Cs-133 or Rb-85)

Measurement light for species B is far off-resonant for species A, providing natural isolation.

#### Species Combinations

| Data | Ancilla | Separation | Notes |
|------|---------|------------|-------|
| Rb-87 | Cs-133 | 70 nm | Different elements |
| Rb-87 | Rb-85 | 0.5 nm | Same element, isotopes |
| Sr-88 | Sr-87 | <1 nm | Nuclear spin qubit |
| Yb-171 | Yb-173 | <1 nm | Nuclear spin qubits |

**Rb-87 / Cs-133 example:**
- Rb D2: 780 nm
- Cs D2: 852 nm
- Detuning: 70 nm = 30 THz

Scattering rate of Cs light on Rb:
$$\Gamma_{sc,Rb} = \frac{\Gamma^2}{\Delta^2} \times I/I_{sat} \approx \left(\frac{6\,\text{MHz}}{30\,\text{THz}}\right)^2 \times 10^6 = 4 \times 10^{-8}\,\text{s}^{-1}$$

This is completely negligible!

#### Dual-Species Trapping

**Challenge:** Tweezers must trap both species efficiently.

Trap depth scales as:
$$U \propto \frac{\alpha(\omega)}{\Delta}$$

For 1064 nm trap:
- Rb: large positive polarizability → strong trap
- Cs: also positive polarizability → similar depth

**Magic wavelengths** exist where both species experience equal trap depths.

### 3. Ancilla-Based Measurement Protocols

#### Indirect Measurement Circuit

Instead of measuring data qubits directly:
1. Prepare ancilla in $|0\rangle$
2. Entangle ancilla with data via CNOT or CZ
3. Measure ancilla (destructive, but isolated)
4. Infer data qubit state

**Circuit for Z-basis measurement:**
```
Data:    ─────●─────────────
              │
Ancilla: |0⟩─X─────M
```

After measurement:
- If ancilla = 0: data was $|0\rangle$
- If ancilla = 1: data was $|1\rangle$

Data qubit is **projected** but not directly measured.

#### Syndrome Extraction for Stabilizer Codes

For surface code with stabilizers $S_Z = Z_1 Z_2 Z_3 Z_4$:

```
D1: ───●───────────
       │
D2: ───│───●───────
       │   │
D3: ───│───│───●───
       │   │   │
D4: ───│───│───│───●───
       │   │   │   │
A:  ───X───X───X───X───M
```

The ancilla parity encodes the stabilizer eigenvalue.

#### Repeated Measurement

For fault tolerance, measure same syndrome multiple times:
1. First round: raw syndrome
2. Second round: confirm or detect error
3. Third round: majority vote

Each round requires fresh or reset ancilla.

### 4. Ancilla Reset and Recycling

#### Fast Reset Protocols

After measurement, ancilla must return to $|0\rangle$:

**Optical pumping reset:**
- Apply σ⁺ light to pump to stretched state
- Duration: ~10 μs
- Fidelity: >99.9%

**Raman reset:**
- Coherent transfer to specific state
- Faster (~1 μs)
- Requires calibration

**Conditional reset:**
- Based on measurement outcome
- Only flip if measured $|1\rangle$
- Fastest for balanced outcomes

#### Atom Loss and Replacement

If ancilla is lost during measurement:
1. Detect loss via absence of fluorescence
2. Move replacement atom from reservoir
3. Initialize new ancilla
4. Continue circuit

**Loss probability per measurement:**
$$P_{loss} \approx 0.1-1\%$$

With ~1000 measurements per error correction cycle, replacement strategy essential.

### 5. Real-Time Classical Feedback

#### Feedback Requirements

Mid-circuit measurement requires fast classical processing:

| Stage | Latency | Technology |
|-------|---------|------------|
| Photon detection | 1 ns | APD/SPAD |
| Signal processing | 10-100 ns | FPGA |
| Decision logic | 10-100 ns | FPGA |
| Control signal | 10-100 ns | AWG |
| Total | 0.1-1 μs | Achievable |

**Comparison with gate time:**
- Two-qubit gate: 0.5 μs
- Feedback latency: 0.5 μs

Feedback can be applied before next gate completes!

#### Adaptive Circuits

**Teleportation example:**
```
|ψ⟩ ───X───────H───M─────────────────
       │           │
|0⟩ ───┼───●───────│───M──────────────
       │   │       │   │
|0⟩ ───────X───────X^m₁───Z^m₂───|ψ⟩
```

The final corrections depend on measurement outcomes.

#### Decoder Integration

For error correction, decoder processes syndromes:
1. Collect syndrome bits (μs)
2. Decode error pattern (varies widely)
3. Apply corrections (μs)

**Decoder latency challenge:**
- Simple codes: <1 μs possible
- Surface code: 10 μs - 1 ms typical
- Requires pipelining or predictive decoding

### 6. Applications to Error Correction

#### Surface Code Implementation

Neutral atoms naturally suited for 2D surface codes:
- Square lattice geometry
- Local stabilizer measurements
- Boundary qubits at edges

**Syndrome extraction cycle:**
1. Prepare ancillas (reset)
2. Four CNOT gates per stabilizer (Rydberg)
3. Measure ancillas (fluorescence)
4. Decode and correct

**Total cycle time estimate:**
- Reset: 10 μs
- 4 CNOTs: 4 × 0.5 μs = 2 μs
- Measurement: 20 μs
- Feedback: 10 μs
- **Total: ~40-50 μs per round**

#### Logical Qubit Memory Time

For surface code distance $d$:
$$\tau_{logical} = \tau_{physical} \times \left(\frac{1}{p}\right)^{(d+1)/2}$$

where $p$ is physical error rate.

With $\tau_{physical} \sim 1$ s and $p \sim 0.1\%$:
- $d=3$: $\tau_{logical} \sim 1000$ s
- $d=5$: $\tau_{logical} \sim 10^6$ s

This requires many syndrome extraction cycles.

### 7. Current State of the Art

#### Demonstrated Capabilities

| Group | Year | Achievement |
|-------|------|-------------|
| Lukin (Harvard) | 2023 | Logical qubit with mid-circuit measurement |
| QuEra | 2024 | 48 logical qubits |
| Atom Computing | 2024 | 1000+ physical qubits |
| Pasqal | 2023 | Dual-species demonstration |

#### Remaining Challenges

1. **Measurement fidelity**: Currently ~98%, need >99.9%
2. **Feedback latency**: Real-time decoding for deep circuits
3. **Atom loss**: Replacement strategies for long algorithms
4. **Crosstalk**: Isolation between data and ancilla

### 8. Future Directions

#### Erasure Conversion

Convert physical errors to **erasure errors** (known location):
- Detect if atom left Rydberg state incorrectly
- Flag the qubit as potentially erroneous
- Erasure errors easier to correct than general errors

**Advantage:** Erasure threshold ~50% vs ~1% for depolarizing

#### Measurement-Based Quantum Computing

Alternative paradigm:
1. Prepare large entangled resource state (cluster state)
2. Perform computation via single-qubit measurements
3. Measurement pattern determines algorithm

Neutral atoms naturally suited for cluster state preparation via global Rydberg gates.

## Worked Examples

### Example 1: Dual-Species Crosstalk Analysis

**Problem:** Calculate the probability that imaging a Cs ancilla causes a spin flip in a neighboring Rb-87 data qubit. The imaging uses 1 mW/cm² at 852 nm for 20 ms, and the Rb qubit is 5 μm away.

**Solution:**

**Step 1: Cs imaging parameters**
At 852 nm on Cs D2 line:
$$I/I_{sat} = \frac{1\,\text{mW/cm}^2}{1.1\,\text{mW/cm}^2} \approx 0.9$$

Scattering rate:
$$\Gamma_{sc,Cs} = \frac{\Gamma}{2}\frac{s}{1+s} = \frac{5.2\,\text{MHz}}{2} \times \frac{0.9}{1.9} = 1.2\,\text{MHz}$$

**Step 2: Effect on Rb**
The Rb D2 line is at 780 nm. Detuning from 852 nm:
$$\Delta = c\left(\frac{1}{780\,\text{nm}} - \frac{1}{852\,\text{nm}}\right) = 3.2 \times 10^{13}\,\text{Hz} = 32\,\text{THz}$$

Scattering rate on Rb:
$$\Gamma_{sc,Rb} = \frac{\Gamma_{Rb}^2}{4\Delta^2} \times I/I_{sat,Rb} = \frac{(6.07 \times 10^6)^2}{4(3.2 \times 10^{13})^2} \times 0.5$$
$$\Gamma_{sc,Rb} = 4.5 \times 10^{-9}\,\text{Hz}$$

**Step 3: Spin flip probability**
During 20 ms imaging:
$$P_{flip} = \Gamma_{sc,Rb} \times t = 4.5 \times 10^{-9} \times 0.020 = 9 \times 10^{-11}$$

This is completely negligible!

**Step 4: Consider intensity at Rb position**
The imaging light intensity at the Rb atom (5 μm away) may be different.
Assuming uniform illumination, the calculation above holds.

**Answer:** The spin flip probability is $<10^{-10}$, providing excellent isolation.

---

### Example 2: Syndrome Extraction Time

**Problem:** Calculate the minimum time for one round of surface code syndrome extraction on a distance-5 code using neutral atoms. Assume:
- Ancilla reset: 5 μs
- Single-qubit gate: 100 ns
- Two-qubit gate: 500 ns
- Measurement: 15 μs
- Feedback processing: 5 μs

**Solution:**

**Step 1: Surface code structure**
Distance-5 surface code has:
- 25 data qubits (5×5)
- 24 syndrome qubits (12 X-type, 12 Z-type)

**Step 2: Syndrome extraction circuit**
Each stabilizer requires:
- 1 ancilla reset
- 4 CNOT gates (can be partially parallelized)
- 1 measurement

**Step 3: Parallel scheduling**
With proper scheduling (see Fowler et al.):
- Depth 4 for CNOT gates (not all gates parallel due to conflicts)
- Total CNOT time: 4 × 500 ns = 2 μs

**Step 4: Total cycle time**
$$t_{cycle} = t_{reset} + t_{CNOTs} + t_{measure} + t_{feedback}$$
$$t_{cycle} = 5 + 2 + 15 + 5 = 27\,\mu\text{s}$$

**Step 5: Verify with realistic overhead**
Add 20% for imperfect scheduling:
$$t_{cycle,real} \approx 32\,\mu\text{s}$$

**Answer:** Approximately 30-35 μs per syndrome extraction round.

---

### Example 3: Logical Error Rate Calculation

**Problem:** A distance-3 surface code has physical error rate $p = 0.5\%$ per operation. The syndrome extraction has depth $d_{circuit} = 10$ gates. Calculate the logical error rate per round assuming the decoder is perfect.

**Solution:**

**Step 1: Error model**
For surface code, logical error requires $\geq (d+1)/2 = 2$ physical errors in a correctable pattern to go undetected.

**Step 2: Count error locations**
In one round:
- Data qubits: 9
- Ancilla qubits: 8 (4 X-type, 4 Z-type)
- Total operations: ~70 (gates + measurements)

**Step 3: Dominant error contribution**
Leading order logical error comes from weight-2 error chains:
$$P_{logical} \approx \binom{n_{locations}}{2} p^2 = C_{code} \cdot p^2$$

For distance-3 surface code: $C_{code} \approx 10$

$$P_{logical} \approx 10 \times (0.005)^2 = 2.5 \times 10^{-4}$$

**Step 4: Effect of circuit depth**
Each gate can fail independently. With 70 operations:
$$p_{eff} = 1 - (1-p)^{70} \approx 70 \times p = 35\%$$

This is too high! Need to account for syndrome structure.

**Step 5: More careful analysis**
Actually, syndrome extraction with repeated rounds (typically 3) allows detecting time-correlated errors. The effective logical error rate:

$$P_{logical} \approx 0.1 \times (p/p_{th})^{(d+1)/2}$$

With $p_{th} \approx 1\%$:
$$P_{logical} \approx 0.1 \times (0.5)^2 = 0.025 = 2.5\%$$

**Answer:** Logical error rate ~2.5% per round, indicating d=3 is marginal for this physical error rate.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the detuning between Rb-85 and Rb-87 D2 lines (766 nm vs 780 nm). What is the scattering ratio when imaging one isotope?

**Problem 1.2:** An ancilla qubit requires 4 CZ gates for stabilizer measurement. With 500 ns per gate, what is the gate portion of the syndrome extraction time?

**Problem 1.3:** A feedback system has 2 μs latency. How many two-qubit gates (at 500 ns each) can be executed in parallel with the feedback?

### Level 2: Intermediate Analysis

**Problem 2.1:** Design a syndrome extraction schedule for a distance-3 surface code that minimizes depth. Show the gate sequence and calculate total time.

**Problem 2.2:** Compare the crosstalk for Rb-87/Rb-85 dual-species vs Rb-87/Cs-133. Which provides better isolation and by what factor?

**Problem 2.3:** An atom is lost with 0.5% probability per measurement. After how many measurement rounds is the probability of all atoms surviving less than 90%? (Assume 24 ancillas)

### Level 3: Challenging Problems

**Problem 3.1:** Design an erasure detection scheme for neutral atoms using auxiliary Rydberg states. Calculate the detection fidelity and overhead.

**Problem 3.2:** Analyze the logical error rate for a distance-5 surface code with realistic error model including:
- Physical gate error: 0.3%
- Measurement error: 1%
- Atom loss: 0.1% per round
How does this compare to the simple $(p/p_{th})^3$ scaling?

**Problem 3.3:** Design a measurement-based quantum computation protocol for preparing a GHZ state on 8 qubits using cluster state resources. What are the resource and time overheads compared to circuit-based preparation?

## Computational Lab: Mid-Circuit Measurement Simulations

### Lab 1: Dual-Species Crosstalk Analysis

```python
"""
Day 917 Lab: Mid-Circuit Measurement Simulations
Analyzing dual-species systems and syndrome extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar

class AtomicSpecies:
    """Container for atomic species parameters."""
    def __init__(self, name, lambda_d2, Gamma, I_sat):
        self.name = name
        self.lambda_d2 = lambda_d2  # m
        self.omega_d2 = 2 * np.pi * c / lambda_d2  # rad/s
        self.Gamma = Gamma  # rad/s
        self.I_sat = I_sat  # W/m^2

# Define species
Rb87 = AtomicSpecies("Rb-87", 780e-9, 2*np.pi*6.07e6, 16.7)
Rb85 = AtomicSpecies("Rb-85", 780.2e-9, 2*np.pi*6.07e6, 16.7)
Cs133 = AtomicSpecies("Cs-133", 852e-9, 2*np.pi*5.22e6, 11.0)

def scattering_rate(species, wavelength, intensity):
    """
    Calculate scattering rate for given species and light.

    Parameters:
    -----------
    species : AtomicSpecies
    wavelength : float (m)
    intensity : float (W/m^2)
    """
    omega = 2 * np.pi * c / wavelength
    Delta = omega - species.omega_d2

    # Saturation parameter (off-resonant)
    s_eff = (intensity / species.I_sat) * (species.Gamma / (2*Delta))**2

    # Scattering rate
    Gamma_sc = (species.Gamma / 2) * s_eff / (1 + s_eff)

    return Gamma_sc

def crosstalk_ratio(data_species, ancilla_species, imaging_intensity):
    """
    Calculate the ratio of scattering rates (crosstalk).
    """
    # Imaging at ancilla wavelength
    wavelength = ancilla_species.lambda_d2

    Gamma_ancilla = scattering_rate(ancilla_species, wavelength, imaging_intensity)
    Gamma_data = scattering_rate(data_species, wavelength, imaging_intensity)

    return Gamma_data / Gamma_ancilla

# Analyze different species combinations
print("=== Dual-Species Crosstalk Analysis ===\n")

combinations = [
    (Rb87, Cs133, "Rb-87 / Cs-133"),
    (Rb87, Rb85, "Rb-87 / Rb-85"),
]

imaging_intensity = 10  # W/m^2 (typical)

for data, ancilla, name in combinations:
    wavelength_sep = abs(data.lambda_d2 - ancilla.lambda_d2) * 1e9
    detuning = abs(data.omega_d2 - ancilla.omega_d2) / (2 * np.pi * 1e12)

    ratio = crosstalk_ratio(data, ancilla, imaging_intensity)

    print(f"{name}:")
    print(f"  Wavelength separation: {wavelength_sep:.1f} nm")
    print(f"  Frequency separation: {detuning:.1f} THz")
    print(f"  Crosstalk ratio: {ratio:.2e}")
    print()

# Visualize crosstalk vs wavelength
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

wavelengths = np.linspace(750e-9, 900e-9, 200)

for species in [Rb87, Cs133]:
    rates = [scattering_rate(species, w, imaging_intensity) for w in wavelengths]
    axes[0].semilogy(wavelengths*1e9, rates, label=species.name, linewidth=2)

axes[0].axvline(x=780, color='blue', linestyle='--', alpha=0.5)
axes[0].axvline(x=852, color='orange', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Scattering rate (Hz)')
axes[0].set_title('Scattering Rate vs Wavelength')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Crosstalk during Cs imaging
I_values = np.logspace(-1, 3, 50)  # W/m^2
crosstalk = [crosstalk_ratio(Rb87, Cs133, I) for I in I_values]

axes[1].loglog(I_values, crosstalk, 'b-', linewidth=2)
axes[1].axhline(y=1e-10, color='r', linestyle='--', label='Negligible threshold')
axes[1].set_xlabel('Imaging intensity (W/m²)')
axes[1].set_ylabel('Crosstalk ratio')
axes[1].set_title('Rb-87 / Cs-133 Crosstalk')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dual_species_crosstalk.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Syndrome Extraction Circuit

```python
"""
Lab 2: Surface code syndrome extraction simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class SurfaceCodeLayout:
    """Simple surface code layout for syndrome extraction analysis."""

    def __init__(self, distance):
        self.d = distance
        self.n_data = distance ** 2
        self.n_ancilla = (distance - 1) * distance  # X and Z stabilizers

        # Create qubit positions
        self.data_positions = []
        self.x_ancilla_positions = []
        self.z_ancilla_positions = []

        for i in range(distance):
            for j in range(distance):
                self.data_positions.append((2*i, 2*j))

        # X stabilizers (on plaquettes)
        for i in range(distance - 1):
            for j in range(distance - 1):
                if (i + j) % 2 == 0:
                    self.x_ancilla_positions.append((2*i+1, 2*j+1))

        # Z stabilizers (on vertices)
        for i in range(distance - 1):
            for j in range(distance - 1):
                if (i + j) % 2 == 1:
                    self.z_ancilla_positions.append((2*i+1, 2*j+1))

    def get_stabilizer_data_qubits(self, ancilla_pos, stab_type):
        """Get the data qubits involved in a stabilizer."""
        i, j = ancilla_pos
        neighbors = [
            (i-1, j), (i+1, j), (i, j-1), (i, j+1)
        ]
        return [n for n in neighbors if n in self.data_positions]

def schedule_syndrome_extraction(layout):
    """
    Create a schedule for parallel syndrome extraction.

    Returns list of layers, each layer is a list of (control, target) pairs.
    """
    # Simple greedy scheduling
    layers = []

    # For each ancilla, we need 4 CNOTs
    # Order: N, E, S, W for Z-type; different for X-type

    all_gates = []

    for pos in layout.z_ancilla_positions + layout.x_ancilla_positions:
        data_qubits = layout.get_stabilizer_data_qubits(pos, 'Z')
        for dq in data_qubits:
            all_gates.append((dq, pos))  # data controls ancilla

    # Greedy scheduling
    remaining = list(all_gates)
    while remaining:
        layer = []
        used_qubits = set()

        for gate in remaining[:]:
            ctrl, tgt = gate
            if ctrl not in used_qubits and tgt not in used_qubits:
                layer.append(gate)
                used_qubits.add(ctrl)
                used_qubits.add(tgt)
                remaining.remove(gate)

        layers.append(layer)

    return layers

def calculate_syndrome_extraction_time(layers, t_gate=500e-9, t_reset=5e-6,
                                        t_measure=15e-6, t_feedback=5e-6):
    """
    Calculate total syndrome extraction time.

    Parameters:
    -----------
    layers : list of gate layers
    t_gate : float, time per two-qubit gate (s)
    t_reset : float, ancilla reset time (s)
    t_measure : float, measurement time (s)
    t_feedback : float, classical processing time (s)
    """
    n_layers = len(layers)
    t_gates = n_layers * t_gate
    total = t_reset + t_gates + t_measure + t_feedback
    return total, {'reset': t_reset, 'gates': t_gates,
                   'measure': t_measure, 'feedback': t_feedback}

# Analyze different code distances
distances = [3, 5, 7, 9]

print("=== Surface Code Syndrome Extraction ===\n")

results = {}
for d in distances:
    layout = SurfaceCodeLayout(d)
    schedule = schedule_syndrome_extraction(layout)

    total_time, breakdown = calculate_syndrome_extraction_time(schedule)

    results[d] = {
        'n_data': layout.n_data,
        'n_ancilla': len(layout.x_ancilla_positions) + len(layout.z_ancilla_positions),
        'n_layers': len(schedule),
        'total_time': total_time,
        'breakdown': breakdown
    }

    print(f"Distance {d}:")
    print(f"  Data qubits: {layout.n_data}")
    print(f"  Ancilla qubits: {results[d]['n_ancilla']}")
    print(f"  Gate layers: {len(schedule)}")
    print(f"  Total time: {total_time*1e6:.1f} μs")
    print()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

d_vals = list(results.keys())
times = [results[d]['total_time']*1e6 for d in d_vals]
layers = [results[d]['n_layers'] for d in d_vals]

axes[0].bar(d_vals, times, color='steelblue')
axes[0].set_xlabel('Code distance')
axes[0].set_ylabel('Syndrome extraction time (μs)')
axes[0].set_title('Extraction Time vs Distance')

# Breakdown for d=5
d = 5
breakdown = results[d]['breakdown']
labels = list(breakdown.keys())
values = [breakdown[k]*1e6 for k in labels]

axes[1].pie(values, labels=labels, autopct='%1.1f%%')
axes[1].set_title(f'Time Breakdown (d={d})')

# Layer scaling
axes[2].plot(d_vals, layers, 'go-', markersize=10, linewidth=2)
axes[2].set_xlabel('Code distance')
axes[2].set_ylabel('Number of gate layers')
axes[2].set_title('Gate Depth Scaling')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('syndrome_extraction.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: Logical Error Rate Analysis

```python
"""
Lab 3: Logical error rate analysis for error-corrected computation
"""

import numpy as np
import matplotlib.pyplot as plt

def logical_error_rate(p_phys, distance, p_threshold=0.01, prefactor=0.1):
    """
    Estimate logical error rate per syndrome extraction round.

    Uses simplified scaling: P_L = prefactor * (p/p_th)^((d+1)/2)
    """
    return prefactor * (p_phys / p_threshold)**((distance + 1) / 2)

def memory_lifetime_rounds(p_phys, distance, target_fidelity=0.99):
    """
    Estimate number of rounds before logical error probability exceeds threshold.
    """
    p_L = logical_error_rate(p_phys, distance)
    if p_L >= 1:
        return 0
    # Probability of no error after n rounds: (1-p_L)^n > target_fidelity
    # n < log(target_fidelity) / log(1-p_L)
    return int(np.log(target_fidelity) / np.log(1 - p_L))

# Analyze logical error rates
p_phys_values = np.logspace(-4, -1, 50)
distances = [3, 5, 7, 9, 11]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Logical error rate vs physical error rate
for d in distances:
    p_L = [logical_error_rate(p, d) for p in p_phys_values]
    axes[0].loglog(p_phys_values*100, p_L, label=f'd = {d}', linewidth=2)

axes[0].axhline(y=1e-6, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Physical error rate (%)')
axes[0].set_ylabel('Logical error rate per round')
axes[0].set_title('Logical Error Rate Scaling')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.01, 10)
axes[0].set_ylim(1e-12, 1)

# Memory lifetime (rounds) vs distance
p_phys_fixed = [0.001, 0.003, 0.005, 0.01]
d_range = np.arange(3, 15, 2)

for p in p_phys_fixed:
    lifetimes = [memory_lifetime_rounds(p, d) for d in d_range]
    axes[1].semilogy(d_range, lifetimes, 'o-', label=f'p = {p*100}%', linewidth=2)

axes[1].set_xlabel('Code distance')
axes[1].set_ylabel('Memory lifetime (rounds)')
axes[1].set_title('Logical Memory Lifetime')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Overhead analysis
# For given target logical error rate, what resources are needed?
p_L_target = 1e-6  # Target logical error rate

p_phys_scan = np.logspace(-3, -1.5, 30)
required_distances = []

for p in p_phys_scan:
    # Find minimum distance
    for d in range(3, 51, 2):
        if logical_error_rate(p, d) < p_L_target:
            required_distances.append(d)
            break
    else:
        required_distances.append(51)

axes[2].plot(p_phys_scan*100, required_distances, 'b-', linewidth=2)
axes[2].fill_between(p_phys_scan*100, 0, required_distances, alpha=0.3)
axes[2].set_xlabel('Physical error rate (%)')
axes[2].set_ylabel('Required code distance')
axes[2].set_title(f'Distance for P_L < {p_L_target}')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 25)

plt.tight_layout()
plt.savefig('logical_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary table
print("=== Resource Requirements Summary ===\n")
print(f"Target logical error rate: {p_L_target}")
print()
print(f"{'Physical error':>15} | {'Required d':>10} | {'# Qubits':>10} | {'Overhead':>10}")
print("-" * 55)

for p, d in zip(p_phys_scan[::5], required_distances[::5]):
    if d <= 50:
        n_qubits = d**2 + (d-1)*d  # data + ancilla
        overhead = n_qubits / 1  # per logical qubit
        print(f"{p*100:>14.2f}% | {d:>10} | {n_qubits:>10} | {overhead:>10.0f}x")

# Analyze circuit execution with mid-circuit measurement
print("\n=== Circuit Execution Analysis ===\n")

# Parameters
p_phys = 0.003  # 0.3% physical error rate
d = 5
t_syndrome = 30e-6  # 30 μs per round
t_two_qubit_gate = 500e-9

# For a 100-depth circuit
circuit_depth = 100

# Number of syndrome rounds per logical gate
rounds_per_gate = 3  # typical for fault tolerance

total_rounds = circuit_depth * rounds_per_gate
total_time = total_rounds * t_syndrome

p_L_round = logical_error_rate(p_phys, d)
p_L_circuit = 1 - (1 - p_L_round)**total_rounds

print(f"Physical error rate: {p_phys*100}%")
print(f"Code distance: {d}")
print(f"Circuit depth: {circuit_depth} logical gates")
print(f"Syndrome rounds per gate: {rounds_per_gate}")
print(f"Total syndrome rounds: {total_rounds}")
print(f"Total execution time: {total_time*1e3:.1f} ms")
print(f"Logical error per round: {p_L_round:.2e}")
print(f"Circuit success probability: {(1-p_L_circuit)*100:.1f}%")
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| Crosstalk ratio | $\Gamma_{data}/\Gamma_{anc} = (\Gamma/\Delta)^2$ | $<10^{-10}$ (Rb/Cs) |
| Syndrome time | $t_{syn} = t_{reset} + t_{gates} + t_{meas}$ | 30-50 μs |
| Logical error rate | $P_L \approx 0.1(p/p_{th})^{(d+1)/2}$ | Varies with d, p |
| Memory lifetime | $\tau_L = \tau_{phys}(p_{th}/p)^{(d+1)/2}$ | Seconds to hours |

### Main Takeaways

1. **Dual-species architectures** provide excellent measurement isolation, with crosstalk ratios below $10^{-10}$ for Rb/Cs combinations.

2. **Ancilla-based measurement** enables non-destructive syndrome extraction by measuring only ancilla qubits while preserving data qubit coherence.

3. **Real-time feedback** with μs latency is achievable with FPGA-based control, enabling adaptive circuits and error correction.

4. **Surface code implementation** requires ~30-50 μs per syndrome round, dominated by measurement and reset times.

5. **Logical error suppression** follows exponential scaling with code distance, requiring physical error rates below ~1% for practical benefit.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why standard measurement is destructive
- [ ] I understand dual-species isolation mechanisms
- [ ] I can describe syndrome extraction circuits
- [ ] I know the feedback latency requirements

### Mathematical Skills
- [ ] I can calculate crosstalk ratios
- [ ] I can estimate syndrome extraction times
- [ ] I can compute logical error rates

### Computational Skills
- [ ] I can simulate crosstalk effects
- [ ] I can schedule syndrome extraction
- [ ] I can analyze error correction overhead

## Week 131 Summary

This week covered the complete neutral atom quantum computing stack:

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 911 | Optical Tweezers | Dipole trapping, array generation |
| 912 | Rydberg States | Scaling laws, C₆ interactions |
| 913 | Blockade | Collective enhancement, entanglement |
| 914 | Single-Qubit Gates | Microwave and Raman transitions |
| 915 | Two-Qubit Gates | CZ via blockade, dressed gates |
| 916 | Array Preparation | Sorting algorithms, loading efficiency |
| 917 | Mid-Circuit Measurement | Dual-species, syndrome extraction |

Neutral atoms represent a promising platform for scalable quantum computing, with demonstrated capabilities including:
- Arrays of 1000+ qubits
- >99% two-qubit gate fidelity
- Native multi-qubit gates
- Mid-circuit measurement for error correction

Remaining challenges focus on improving gate fidelities, reducing preparation overhead, and demonstrating fault-tolerant logical operations at scale.
