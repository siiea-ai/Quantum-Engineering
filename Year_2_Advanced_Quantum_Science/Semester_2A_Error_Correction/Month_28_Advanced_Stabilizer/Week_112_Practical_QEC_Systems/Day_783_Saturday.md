# Day 783: Quantum Computer Architecture

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Full-stack architecture and control systems |
| Afternoon | 2.5 hours | Cryogenics and classical processing |
| Evening | 2 hours | System integration simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe the full quantum computing stack** from physical to application layer
2. **Explain control electronics requirements** for QEC systems
3. **Understand cryogenic engineering** challenges for superconducting qubits
4. **Design decoder architectures** meeting real-time constraints
5. **Analyze system bottlenecks** in fault-tolerant quantum computers
6. **Evaluate integration challenges** across the stack

---

## Core Content

### 1. The Quantum Computing Stack

A fault-tolerant quantum computer requires multiple integrated layers:

```
┌─────────────────────────────────────┐
│     Application Layer               │  Algorithms, high-level circuits
├─────────────────────────────────────┤
│     Logical Layer                   │  Logical qubits, logical gates
├─────────────────────────────────────┤
│     QEC Layer                       │  Error correction, decoding
├─────────────────────────────────────┤
│     Control Layer                   │  Pulse generation, timing
├─────────────────────────────────────┤
│     Physical Layer                  │  Qubits, couplers, readout
└─────────────────────────────────────┘
```

#### Layer Responsibilities

| Layer | Function | Key Challenges |
|-------|----------|----------------|
| Application | Algorithm design | Resource estimation |
| Logical | Fault-tolerant operations | Overhead management |
| QEC | Syndrome extraction, decoding | Real-time processing |
| Control | Pulse shaping, timing | Latency, calibration |
| Physical | Qubit manipulation | Coherence, fidelity |

### 2. Physical Layer: Qubit Technologies

#### Superconducting Qubits

**Components:**
- Transmon qubits: $E_J/E_C \approx 50$
- Coupling resonators: $f_r \sim 6-8$ GHz
- Readout resonators: $\kappa/2\pi \sim 1-10$ MHz

**Key parameters:**

$$\boxed{T_1 \approx 50-200 \mu s, \quad T_2 \approx 50-150 \mu s}$$

**Dilution refrigerator requirements:**
- Base temperature: $T \approx 10-20$ mK
- Cooling power: $\sim 10 \mu$W at 20 mK
- Wiring heat load: Major constraint

#### Trapped Ions

**Components:**
- Ion species: $^{171}$Yb$^+$, $^{40}$Ca$^+$, $^{137}$Ba$^+$
- Linear Paul traps
- Laser systems (Doppler cooling, gates, readout)

**Key parameters:**

$$\boxed{T_1 \approx 10^3 - 10^6 s, \quad T_2 \approx 1-100 s}$$

**Challenges:**
- Gate speed limited by motional frequency
- Crosstalk in multi-ion chains
- Scalability via shuttling or photonic interconnects

### 3. Control Layer: Electronics

#### Classical Control Requirements

For each qubit, control electronics must provide:
1. **Gate pulses**: Shaped microwave/laser pulses
2. **Timing**: Nanosecond precision
3. **Calibration**: Continuous drift compensation

**Control electronics per qubit:**
- DAC (Digital-to-Analog): 1-2 GS/s, 14+ bits
- ADC (Analog-to-Digital): 1+ GS/s, 12+ bits
- Frequency synthesis: 4-8 GHz range
- Arbitrary waveform generation

#### Scaling Challenge

$$\boxed{P_{\text{dissipated}} \propto N_{\text{qubits}} \times f_{\text{control}} \times V^2}$$

For $N = 10^6$ qubits at room temperature:
- Power: $\sim$ MW scale
- Wiring: $\sim 10^7$ coaxial cables (impossible!)

**Solutions:**
1. Cryogenic control electronics (cryo-CMOS)
2. Multiplexing and shared resources
3. Photonic interconnects

### 4. Cryogenic Engineering

#### Dilution Refrigerator Architecture

```
300 K  ─────────────────────────  Room temperature
         │
50 K   ─────────────────────────  First stage (pulse tubes)
         │
4 K    ─────────────────────────  Second stage (pulse tubes)
         │
800 mK ─────────────────────────  Still
         │
100 mK ─────────────────────────  Cold plate
         │
10 mK  ─────────────────────────  Mixing chamber (qubits here)
```

#### Thermal Budget

**Heat sources at each stage:**

| Stage | Source | Typical Load |
|-------|--------|--------------|
| 4 K | Coax cables, HEMT amps | 100 mW - 1 W |
| 100 mK | Attenuators, filters | 100 $\mu$W |
| 10 mK | Qubit control, noise | 10 $\mu$W |

**Constraint**: Total load at 10 mK must be $< 20 \mu$W typical.

**Wiring heat load per line:**
$$\boxed{Q = \kappa A \frac{\Delta T}{L}}$$

where $\kappa$ is thermal conductivity, $A$ is cross-section, $L$ is length.

#### Cooling Power Scaling

Current dilution refrigerators:
- 10-20 $\mu$W at 20 mK
- Can support ~1000 qubits with careful engineering

For million-qubit systems:
- Need ~100× more cooling power
- Modular cryostat architectures
- Cryogenic interconnects between modules

### 5. QEC Layer: Decoding Systems

#### Decoder Requirements

For real-time QEC with cycle time $t_c$:

$$\boxed{t_{\text{decode}} < t_c}$$

**Typical values:**
- Superconducting: $t_c \sim 1 \mu$s → $t_{\text{decode}} < 1 \mu$s
- Trapped ion: $t_c \sim 100 \mu$s → $t_{\text{decode}} < 100 \mu$s

#### Decoder Architectures

**1. FPGA-Based Decoders**

Advantages:
- Programmable, reconfigurable
- Low latency (<1 $\mu$s possible)
- Parallel processing

Challenges:
- Limited memory for large codes
- Complex algorithm implementation

**Union-Find decoder on FPGA:**
$$\boxed{t_{\text{UF}} = O(d^2) \text{ clock cycles}}$$

At 200 MHz clock: $t = d^2 \times 5$ ns $= 1.8 \mu$s for $d = 19$.

**2. ASIC Decoders**

Custom silicon for specific decoder:
- Even lower latency
- Higher throughput
- Fixed algorithm (less flexible)

**3. Neural Network Decoders**

Trained decoders using ML:
- Can handle correlated errors
- Require specialized hardware (TPU, GPU)
- Latency typically 10-100 $\mu$s

#### Decoder Data Flow

```
Syndrome      FPGA/ASIC       Correction
Readout  →   Decoder    →    Feed-forward
(analog)     (digital)       (control)
   ↓            ↓                ↓
 1 μs        <1 μs           <0.1 μs
```

Total latency budget: $< t_{\text{cycle}}$

### 6. Classical-Quantum Interface

#### Real-Time Processing Requirements

**Data rates:**
- Surface code $d = 11$: ~200 syndrome bits per round
- At 1 MHz cycle rate: 200 Mbits/s per logical qubit
- For 1000 logical qubits: 200 Gbits/s

**Processing pipeline:**

1. **Acquisition**: ADC samples → digital syndrome
2. **Decoding**: Syndrome → error estimate
3. **Correction**: Error estimate → control pulse
4. **Execution**: Apply correction pulse

#### Feedback Latency

Total feedback time:
$$\boxed{t_{\text{feedback}} = t_{\text{readout}} + t_{\text{decode}} + t_{\text{pulse}}}$$

For superconducting qubits:
$$t_{\text{feedback}} \approx 300 \text{ ns} + 500 \text{ ns} + 50 \text{ ns} = 850 \text{ ns}$$

This must fit within one syndrome cycle ($\sim 1 \mu$s).

### 7. System Integration

#### Modular Architecture

For million-qubit systems, modular design is essential:

```
┌──────────────┐    ┌──────────────┐
│   Module 1   │────│   Module 2   │
│  10K qubits  │    │  10K qubits  │
└──────────────┘    └──────────────┘
       │                   │
       └─────────┬─────────┘
                 │
         ┌──────────────┐
         │  Interconnect │
         │   Module     │
         └──────────────┘
```

**Inter-module operations:**
- Lattice surgery across boundaries
- Quantum interconnects (microwave, optical)
- Longer latency for cross-module gates

#### Compilation and Scheduling

**Compilation stack:**
1. High-level algorithm (Qiskit, Cirq)
2. Logical circuit optimization
3. Fault-tolerant compilation (magic states, surgery)
4. Physical gate decomposition
5. Pulse-level control

**Scheduling challenges:**
- Magic state factory allocation
- Lattice surgery routing
- Parallelism vs. resource conflicts

### 8. Power and Cost Analysis

#### System Power Budget

| Component | Power (10K qubits) | Power (1M qubits) |
|-----------|-------------------|-------------------|
| Dilution refrigerator | 50 kW | 500 kW |
| Control electronics | 100 kW | 10 MW |
| Classical computing | 50 kW | 5 MW |
| Infrastructure | 50 kW | 5 MW |
| **Total** | **250 kW** | **~20 MW** |

#### Cost Estimates

Current state-of-the-art systems (~1000 qubits):
- Hardware: $10-50M
- Installation: $5-10M
- Operating cost: $1-5M/year

Projected million-qubit systems:
- Hardware: $100M-1B
- Facility: $50-100M
- Operating: $10-50M/year

---

## Quantum Mechanics Connection

### The Measurement Problem in QEC

QEC syndrome measurement is a practical implementation of quantum measurement theory:

1. **Ancilla coupling**: $H_{\text{int}} = g \hat{S} \otimes \hat{\sigma}_z^{\text{anc}}$
2. **Measurement backaction**: Projection onto syndrome eigenspace
3. **Information gain**: Classical syndrome value
4. **Quantum preservation**: Logical information intact

The architecture must preserve quantum coherence while extracting classical syndrome information.

### Decoherence Across the Stack

Each layer contributes to effective decoherence:

$$\boxed{\Gamma_{\text{eff}} = \Gamma_{\text{physical}} + \Gamma_{\text{control}} + \Gamma_{\text{measurement}} + \Gamma_{\text{feedback}}}$$

Engineering minimizes each contribution within system constraints.

---

## Worked Examples

### Example 1: Wiring Thermal Load

**Problem:** A superconducting quantum computer with 1000 qubits requires 4 coaxial cables per qubit (2 control, 1 flux, 1 readout). Each cable has thermal conductivity $\kappa = 10$ W/(m·K) at 4K and cross-section $A = 0.5$ mm$^2$. Calculate the heat load at the 4K stage from a 50 cm cable length connecting to 300K.

**Solution:**

Heat flow per cable:
$$Q = \kappa A \frac{\Delta T}{L}$$

With temperature-dependent $\kappa$ integrated (simplified):
$$Q_{\text{per cable}} \approx \kappa_{\text{avg}} A \frac{T_{\text{high}} - T_{\text{low}}}{L}$$

Using $\kappa_{\text{avg}} \approx 100$ W/(m·K) for copper over 4K-300K:
$$Q_{\text{per cable}} = 100 \times 0.5 \times 10^{-6} \times \frac{296}{0.5} = 29.6 \text{ mW}$$

Total for 4000 cables:
$$Q_{\text{total}} = 4000 \times 29.6 \text{ mW} = 118 \text{ W}$$

$$\boxed{Q_{\text{4K}} \approx 120 \text{ W}}$$

This is significant but manageable with modern pulse-tube coolers.

### Example 2: FPGA Decoder Timing

**Problem:** Design an FPGA-based Union-Find decoder for distance-11 surface code. The FPGA runs at 250 MHz. Estimate if it can decode within a 1 $\mu$s syndrome cycle.

**Solution:**

Union-Find decoder complexity: $O(d^2 \alpha(d^2))$ where $\alpha$ is inverse Ackermann.

For practical purposes: $O(d^2)$ operations.

Operations needed: $\approx 10 \times d^2 = 10 \times 121 = 1210$ (with overhead)

Clock period: $t_{\text{clk}} = 1/250 \text{ MHz} = 4 \text{ ns}$

Decode time: $t_{\text{decode}} = 1210 \times 4 \text{ ns} = 4.84 \mu$s

This exceeds 1 $\mu$s! Solutions:
1. Pipeline the decoder (multiple syndromes in flight)
2. Use parallel processing (FPGA fabric)
3. Reduce clock cycles per operation

With 4× parallelism:
$$t_{\text{decode}} = 4.84/4 = 1.21 \mu s$$

With optimized implementation (fewer cycles):
$$t_{\text{decode}} \approx 0.8-1.0 \mu s$$

$$\boxed{\text{Feasible with optimization and pipelining}}$$

### Example 3: Modular System Design

**Problem:** Design a 100,000 physical qubit system using 10,000-qubit modules. Specify the number of modules, inter-module connections, and estimate the impact on logical error rates.

**Solution:**

**Module count:**
$$N_{\text{modules}} = 100,000 / 10,000 = 10 \text{ modules}$$

**Layout:** 2×5 or ring topology

**Inter-module connections:**
- For 2D lattice surgery: need connections along boundaries
- Boundary qubits: $\sqrt{10,000} = 100$ per edge
- Inter-module links needed: ~100 per adjacent pair
- Total pairs in 2×5 layout: 13
- Total inter-module links: $\approx 1300$

**Impact on logical operations:**

Inter-module operations have:
- Additional latency: $\Delta t \sim 10 \mu$s (microwave link)
- Higher error rate: $p_{\text{link}} \sim 10 \times p_{\text{internal}}$

For logical qubit spanning modules:
$$p_L^{\text{cross}} \approx p_L^{\text{internal}} + p_{\text{boundary}}$$

If boundary operations are 3× worse:
$$p_L^{\text{cross}} \approx 1.5 \times p_L^{\text{internal}}$$

$$\boxed{10 \text{ modules, } 1300 \text{ inter-module links, } 50\% \text{ penalty for cross-module operations}}$$

---

## Practice Problems

### Level A: Direct Application

**A1.** A dilution refrigerator provides 15 $\mu$W of cooling power at 15 mK. If each qubit control line dissipates 10 nW, how many qubits can be supported?

**A2.** An FPGA decoder processes $d^2$ syndrome bits. For $d = 17$ at 200 MHz clock, estimate the minimum decode time (1 clock per bit).

**A3.** Calculate the data rate for syndrome extraction from a distance-13 surface code with 0.5 $\mu$s cycle time.

### Level B: Intermediate Analysis

**B1.** Design the thermal attenuation chain for a qubit control line, specifying attenuation at each stage (300K, 4K, 100mK, 20mK) to achieve -60dB total while managing heat loads.

**B2.** Compare FPGA vs ASIC decoder implementations for a 1000-logical-qubit system. Consider latency, power, development cost, and flexibility.

**B3.** Estimate the total power consumption of a 100,000 physical qubit superconducting quantum computer, breaking down by subsystem.

### Level C: Research-Level Challenges

**C1.** Design an inter-module quantum interconnect using microwave photons. Specify the required link fidelity, bandwidth, and latency for lattice surgery operations.

**C2.** Analyze the trade-off between decoder complexity and decoder error rate. How does using a suboptimal (but fast) decoder affect the effective threshold?

**C3.** Propose an architecture for a billion-qubit quantum computer. Address cooling, control, error correction, and classical processing challenges.

---

## Computational Lab

```python
"""
Day 783: Quantum Computer Architecture
System-level simulation and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# SYSTEM COMPONENT MODELS
# =============================================================================

@dataclass
class QubitTechnology:
    """Model for a qubit technology."""
    name: str
    t1_us: float  # T1 in microseconds
    t2_us: float  # T2 in microseconds
    gate_time_ns: float  # Two-qubit gate time
    gate_fidelity: float  # Two-qubit gate fidelity
    readout_time_ns: float
    readout_fidelity: float
    operating_temp_mk: float


@dataclass
class ControlSystem:
    """Model for control electronics."""
    dac_rate_gsps: float  # DAC sample rate in GS/s
    adc_rate_gsps: float  # ADC sample rate in GS/s
    latency_ns: float  # Processing latency
    power_per_qubit_w: float  # Power dissipation per qubit
    channels_per_unit: int  # Channels per control unit


@dataclass
class CryoSystem:
    """Model for cryogenic system."""
    base_temp_mk: float
    cooling_power_uw: float  # at base temperature
    stages: List[Tuple[float, float]]  # (temp_K, power_W)


@dataclass
class Decoder:
    """Model for QEC decoder."""
    name: str
    latency_per_syndrome_ns: float  # Per syndrome bit
    power_w: float
    max_distance: int


# =============================================================================
# STANDARD CONFIGURATIONS
# =============================================================================

SUPERCONDUCTING = QubitTechnology(
    name="Superconducting Transmon",
    t1_us=100,
    t2_us=80,
    gate_time_ns=30,
    gate_fidelity=0.995,
    readout_time_ns=300,
    readout_fidelity=0.99,
    operating_temp_mk=15
)

TRAPPED_ION = QubitTechnology(
    name="Trapped Ion (Yb+)",
    t1_us=1e9,  # Very long
    t2_us=1e6,
    gate_time_ns=100000,  # 100 μs
    gate_fidelity=0.999,
    readout_time_ns=50000,
    readout_fidelity=0.999,
    operating_temp_mk=300000  # Room temp (in mK)
)

STANDARD_CONTROL = ControlSystem(
    dac_rate_gsps=2.0,
    adc_rate_gsps=1.0,
    latency_ns=100,
    power_per_qubit_w=0.1,
    channels_per_unit=8
)

STANDARD_CRYO = CryoSystem(
    base_temp_mk=15,
    cooling_power_uw=15,
    stages=[
        (50, 40),      # 50K stage, 40W
        (4, 2),        # 4K stage, 2W
        (0.8, 0.001),  # Still, 1mW
        (0.015, 15e-6) # Base, 15μW
    ]
)

UNION_FIND_DECODER = Decoder(
    name="Union-Find FPGA",
    latency_per_syndrome_ns=4,  # At 250 MHz
    power_w=50,
    max_distance=31
)


# =============================================================================
# SYSTEM ANALYZER
# =============================================================================

class QuantumComputerArchitecture:
    """Model a complete quantum computer architecture."""

    def __init__(self,
                 qubit_tech: QubitTechnology,
                 control: ControlSystem,
                 cryo: CryoSystem,
                 decoder: Decoder):
        self.qubit = qubit_tech
        self.control = control
        self.cryo = cryo
        self.decoder = decoder

    def syndrome_cycle_time(self, distance: int) -> float:
        """Calculate syndrome cycle time in nanoseconds."""
        # Components: gates + readout + processing
        gate_time = 4 * self.qubit.gate_time_ns  # 4 CNOTs per round
        readout_time = self.qubit.readout_time_ns
        processing_time = self.control.latency_ns

        return gate_time + readout_time + processing_time

    def decode_time(self, distance: int) -> float:
        """Calculate decode time in nanoseconds."""
        n_syndromes = 2 * distance**2
        return n_syndromes * self.decoder.latency_per_syndrome_ns

    def can_decode_realtime(self, distance: int) -> Tuple[bool, float]:
        """Check if real-time decoding is possible."""
        cycle_time = self.syndrome_cycle_time(distance)
        decode_time = self.decode_time(distance)
        return decode_time < cycle_time, decode_time / cycle_time

    def thermal_budget(self, n_qubits: int) -> Dict:
        """Calculate thermal loads at each stage."""
        # Simplified model
        lines_per_qubit = 4  # Control, flux, readout, spare
        n_lines = n_qubits * lines_per_qubit

        # Heat load per line at base temperature
        heat_per_line_nw = 0.01  # 10 pW typical with good filtering

        base_load = n_lines * heat_per_line_nw * 1e-3  # in μW

        return {
            'n_qubits': n_qubits,
            'n_lines': n_lines,
            'base_load_uw': base_load,
            'cooling_power_uw': self.cryo.cooling_power_uw,
            'margin_uw': self.cryo.cooling_power_uw - base_load,
            'sustainable': base_load < self.cryo.cooling_power_uw
        }

    def power_budget(self, n_qubits: int) -> Dict:
        """Calculate total system power."""
        # Control electronics
        control_power = n_qubits * self.control.power_per_qubit_w

        # Cryogenic system (scales with qubit count)
        cryo_base = 30000  # 30 kW base
        cryo_scaling = n_qubits * 0.01  # W per qubit
        cryo_power = cryo_base + cryo_scaling

        # Decoder (scales with logical qubits)
        n_logical = n_qubits / 1000  # Rough estimate
        decoder_power = n_logical * self.decoder.power_w

        # Classical compute
        classical_power = n_qubits * 0.001  # 1W per 1000 qubits

        return {
            'control_kw': control_power / 1000,
            'cryo_kw': cryo_power / 1000,
            'decoder_kw': decoder_power / 1000,
            'classical_kw': classical_power / 1000,
            'total_kw': (control_power + cryo_power +
                        decoder_power + classical_power) / 1000
        }

    def resource_estimate(self, n_logical: int, distance: int) -> Dict:
        """Estimate resources for fault-tolerant computation."""
        # Physical qubits for data
        data_qubits = n_logical * 2 * distance**2

        # Magic state factories (1 per 10 logical qubits)
        n_factories = max(1, n_logical // 10)
        factory_qubits = n_factories * 15 * 2 * distance**2

        total_qubits = data_qubits + factory_qubits

        # Time per logical operation
        t_clifford = self.syndrome_cycle_time(distance) * distance
        t_t_gate = t_clifford * 10  # Magic state injection

        return {
            'n_logical': n_logical,
            'distance': distance,
            'data_qubits': data_qubits,
            'factory_qubits': factory_qubits,
            'total_qubits': total_qubits,
            't_clifford_us': t_clifford / 1000,
            't_t_gate_us': t_t_gate / 1000,
            'n_factories': n_factories
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_architecture_scaling():
    """Visualize architecture scaling."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    arch = QuantumComputerArchitecture(
        SUPERCONDUCTING, STANDARD_CONTROL, STANDARD_CRYO, UNION_FIND_DECODER
    )

    # 1. Decode time vs distance
    ax = axes[0, 0]
    distances = np.arange(3, 31, 2)
    decode_times = [arch.decode_time(d) / 1000 for d in distances]  # μs
    cycle_times = [arch.syndrome_cycle_time(d) / 1000 for d in distances]

    ax.plot(distances, decode_times, 'b-o', label='Decode time', linewidth=2)
    ax.plot(distances, cycle_times, 'r--s', label='Cycle time', linewidth=2)
    ax.fill_between(distances, 0, cycle_times, alpha=0.2, color='green',
                   label='Real-time feasible')

    ax.set_xlabel('Code Distance', fontsize=12)
    ax.set_ylabel('Time (μs)', fontsize=12)
    ax.set_title('Decode Time vs Syndrome Cycle', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Thermal budget vs qubits
    ax = axes[0, 1]
    qubit_counts = np.array([100, 500, 1000, 2000, 5000, 10000])
    thermal_data = [arch.thermal_budget(n) for n in qubit_counts]

    loads = [t['base_load_uw'] for t in thermal_data]
    cooling = [t['cooling_power_uw'] for t in thermal_data]

    ax.semilogy(qubit_counts, loads, 'b-o', label='Heat load', linewidth=2)
    ax.axhline(y=cooling[0], color='r', linestyle='--',
              label=f'Cooling power ({cooling[0]} μW)')

    sustainable = [n for n, t in zip(qubit_counts, thermal_data) if t['sustainable']]
    if sustainable:
        ax.axvline(x=max(sustainable), color='g', linestyle=':',
                  label=f'Max sustainable: {max(sustainable)}')

    ax.set_xlabel('Physical Qubits', fontsize=12)
    ax.set_ylabel('Power at 15mK (μW)', fontsize=12)
    ax.set_title('Thermal Budget Scaling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Power budget breakdown
    ax = axes[1, 0]
    n_qubits_list = [1000, 10000, 100000, 1000000]
    categories = ['Control', 'Cryo', 'Decoder', 'Classical']
    colors = ['blue', 'cyan', 'green', 'orange']

    x = np.arange(len(n_qubits_list))
    width = 0.2

    for i, cat in enumerate(categories):
        values = []
        for n in n_qubits_list:
            power = arch.power_budget(n)
            key = cat.lower() + '_kw'
            values.append(power[key])
        ax.bar(x + i*width, values, width, label=cat, color=colors[i])

    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([f'{n//1000}K' for n in n_qubits_list])
    ax.set_xlabel('Physical Qubits', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_title('Power Budget Breakdown', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Resource requirements
    ax = axes[1, 1]
    logical_counts = [10, 50, 100, 500, 1000]
    distances_to_plot = [11, 17, 23]

    for d in distances_to_plot:
        total_qubits = [arch.resource_estimate(n, d)['total_qubits']
                       for n in logical_counts]
        ax.semilogy(logical_counts, total_qubits, 'o-',
                   label=f'd = {d}', linewidth=2)

    ax.set_xlabel('Logical Qubits', fontsize=12)
    ax.set_ylabel('Physical Qubits Required', fontsize=12)
    ax.set_title('Physical Qubit Requirements', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_783_architecture_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_technologies():
    """Compare different qubit technologies."""

    print("=" * 60)
    print("QUBIT TECHNOLOGY COMPARISON")
    print("=" * 60)

    technologies = [
        SUPERCONDUCTING,
        TRAPPED_ION
    ]

    print(f"\n{'Metric':<25} | ", end="")
    for tech in technologies:
        print(f"{tech.name[:20]:<20} | ", end="")
    print()
    print("-" * 70)

    metrics = [
        ('T1 (μs)', lambda t: f"{t.t1_us:.0e}"),
        ('T2 (μs)', lambda t: f"{t.t2_us:.0e}"),
        ('Gate time (ns)', lambda t: f"{t.gate_time_ns:.0f}"),
        ('Gate fidelity', lambda t: f"{t.gate_fidelity:.4f}"),
        ('Readout time (ns)', lambda t: f"{t.readout_time_ns:.0f}"),
        ('Operating temp (mK)', lambda t: f"{t.operating_temp_mk:.0f}"),
    ]

    for name, getter in metrics:
        print(f"{name:<25} | ", end="")
        for tech in technologies:
            print(f"{getter(tech):<20} | ", end="")
        print()


def analyze_modular_system():
    """Analyze modular quantum computer architecture."""

    print("\n" + "=" * 60)
    print("MODULAR SYSTEM ANALYSIS")
    print("=" * 60)

    target_qubits = 100000
    module_sizes = [1000, 5000, 10000, 20000]

    print(f"\nTarget: {target_qubits:,} physical qubits")
    print(f"\n{'Module Size':<15} | {'# Modules':<12} | {'Inter-module':<15} | {'Overhead':>10}")
    print("-" * 60)

    for size in module_sizes:
        n_modules = target_qubits // size
        # Assume 2D grid layout
        grid_dim = int(np.ceil(np.sqrt(n_modules)))

        # Inter-module connections
        boundary_qubits = int(np.sqrt(size))
        inter_connections = (grid_dim - 1) * grid_dim * 2 * boundary_qubits

        # Overhead from non-local operations
        overhead_fraction = inter_connections / target_qubits * 100

        print(f"{size:>12,} | {n_modules:>10} | {inter_connections:>13,} | {overhead_fraction:>9.1f}%")


def decoder_comparison():
    """Compare different decoder implementations."""

    print("\n" + "=" * 60)
    print("DECODER IMPLEMENTATION COMPARISON")
    print("=" * 60)

    decoders = [
        ("MWPM (CPU)", 1000, 100, "Exact"),
        ("Union-Find (FPGA)", 4, 50, "Near-optimal"),
        ("Neural Net (GPU)", 50, 200, "Learned"),
        ("Lookup Table", 1, 10, "Limited d"),
    ]

    print(f"\n{'Decoder':<20} | {'Latency (ns/syn)':<18} | {'Power (W)':<10} | {'Quality':<12}")
    print("-" * 70)

    for name, latency, power, quality in decoders:
        print(f"{name:<20} | {latency:>16} | {power:>8} | {quality:<12}")

    # Calculate max distance for real-time operation
    print(f"\nMax distance for 1μs cycle time:")
    cycle_time_ns = 1000
    for name, latency, _, _ in decoders:
        # latency * 2d² < cycle_time
        # d < sqrt(cycle_time / (2 * latency))
        max_d = int(np.sqrt(cycle_time_ns / (2 * latency)))
        print(f"  {name}: d ≤ {max_d}")


if __name__ == "__main__":
    print("Day 783: Quantum Computer Architecture")
    print("=" * 60)

    # Technology comparison
    compare_technologies()

    # System analysis
    arch = QuantumComputerArchitecture(
        SUPERCONDUCTING, STANDARD_CONTROL, STANDARD_CRYO, UNION_FIND_DECODER
    )

    # Real-time decoding analysis
    print("\n" + "=" * 60)
    print("REAL-TIME DECODING ANALYSIS")
    print("=" * 60)

    for d in [5, 11, 17, 23, 31]:
        can_rt, ratio = arch.can_decode_realtime(d)
        status = "YES" if can_rt else "NO"
        print(f"Distance {d:>2}: Real-time = {status}, ratio = {ratio:.2f}")

    # Modular analysis
    analyze_modular_system()

    # Decoder comparison
    decoder_comparison()

    # Resource estimates
    print("\n" + "=" * 60)
    print("RESOURCE ESTIMATES")
    print("=" * 60)

    for n_log in [100, 1000]:
        for d in [11, 17]:
            res = arch.resource_estimate(n_log, d)
            print(f"\n{n_log} logical qubits, d={d}:")
            print(f"  Physical qubits: {res['total_qubits']:,}")
            print(f"  T-factories: {res['n_factories']}")
            print(f"  Clifford time: {res['t_clifford_us']:.1f} μs")
            print(f"  T-gate time: {res['t_t_gate_us']:.1f} μs")

    # Power analysis
    print("\n" + "=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)

    for n in [10000, 100000, 1000000]:
        power = arch.power_budget(n)
        print(f"\n{n:,} physical qubits:")
        print(f"  Control: {power['control_kw']:.0f} kW")
        print(f"  Cryo: {power['cryo_kw']:.0f} kW")
        print(f"  Total: {power['total_kw']:.0f} kW")

    # Generate plots
    plot_architecture_scaling()
```

---

## Summary

### Key System Requirements

| Component | Current State | Million-Qubit Requirement |
|-----------|--------------|---------------------------|
| Physical qubits | ~1000 | 1,000,000 |
| Cooling at 15mK | 15 $\mu$W | 1 mW |
| Control power | 100 kW | 10 MW |
| Decode latency | <1 $\mu$s | <100 ns |
| Data rate | 100 Mb/s | 10 Tb/s |

### Main Takeaways

1. **Full stack integration is critical**: Each layer must meet tight specifications
2. **Cryogenics is a bottleneck**: Current dilution fridges support ~1000-10000 qubits
3. **Real-time decoding is challenging**: FPGA/ASIC decoders essential for large codes
4. **Modular architecture is necessary**: Single monolithic systems cannot scale to millions
5. **Power scales steeply**: Million-qubit systems require datacenter-scale infrastructure
6. **Classical-quantum interface is crucial**: Data rates and latencies are demanding

---

## Daily Checklist

- [ ] I can describe all layers of the quantum computing stack
- [ ] I understand control electronics requirements
- [ ] I can analyze cryogenic thermal budgets
- [ ] I know decoder architecture trade-offs
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 784

Tomorrow we conclude Month 28 with a **Comprehensive Synthesis** covering:
- Integration of all four weeks' material
- Key formulas from Month 28
- Open problems in fault-tolerant quantum computing
- Preparation for Month 29: Topological Error Correction

*"We have built the foundation; now we see the mountain we must climb."*

---

*Day 783 of 2184 | Year 2, Month 28, Week 112, Day 6*
*Quantum Engineering PhD Curriculum*
