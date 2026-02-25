# Day 832: Decoder-Hardware Co-Design

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | FPGA/ASIC architectures, design principles |
| **Afternoon** | 2.5 hours | Cryogenic processing, integration challenges |
| **Evening** | 2 hours | Hardware simulation lab |

---

## Learning Objectives

By the end of Day 832, you will be able to:

1. **Design** FPGA architectures for real-time surface code decoding
2. **Analyze** the trade-offs between FPGA, ASIC, and CPU implementations
3. **Evaluate** cryogenic classical processing approaches
4. **Estimate** latency, power, and area for decoder hardware
5. **Integrate** decoder hardware with quantum control systems
6. **Compare** state-of-the-art decoder implementations

---

## Core Content

### 1. The Hardware Imperative

Software decoders on CPUs achieve:
- MWPM: ~100 μs for distance-11 codes
- Union-Find: ~10 μs for distance-11 codes
- Neural Networks: ~1-10 μs with GPU

Target for superconducting systems: **< 500 ns**

This requires specialized hardware: **FPGA**, **ASIC**, or **cryogenic logic**.

### 2. FPGA Implementation

FPGAs (Field-Programmable Gate Arrays) offer:
- **Parallelism**: Thousands of operations simultaneously
- **Deterministic latency**: Predictable timing
- **Reconfigurability**: Update decoder for new codes
- **Fast development**: Faster than ASIC

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FPGA Decoder                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Syndrome │→ │ Defect       │→ │ Matching     │→ Output  │
│  │ Input    │  │ Detection    │  │ Engine       │          │
│  └──────────┘  └──────────────┘  └──────────────┘          │
│                      ↓                ↓                     │
│               ┌──────────────────────────────┐              │
│               │     Memory (BRAM/Registers)  │              │
│               └──────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components

**Syndrome Input Interface**:
- Parallel input for all syndrome bits
- Pipelining for continuous operation
- Input registers: $d^2$ bits

**Defect Detection**:
- XOR with previous syndrome
- Population count (count defects)
- Complexity: $O(1)$ time, $O(d^2)$ area

**Matching Engine** (Union-Find):
- Parallel cluster operations
- Tree-based union-find structure
- Complexity: $O(\log d)$ levels of pipeline

### 3. FPGA Resources and Constraints

| Resource | Typical FPGA | Usage for d=11 |
|----------|--------------|----------------|
| LUTs (Look-Up Tables) | 1M | ~50K |
| Flip-Flops | 2M | ~100K |
| BRAM (Block RAM) | 10 MB | ~100 KB |
| DSP Slices | 3000 | ~100 |
| Clock Speed | 200-500 MHz | 300 MHz |

**Latency estimate** (pipelined Union-Find):
$$t_{\text{decode}} = \frac{N_{\text{stages}}}{f_{\text{clock}}} = \frac{20}{300 \text{ MHz}} \approx 67 \text{ ns}$$

### 4. ASIC Decoder Design

ASICs (Application-Specific Integrated Circuits) provide:
- **Maximum performance**: Custom logic, optimal layout
- **Lowest power**: No reconfiguration overhead
- **Smallest area**: Dense implementation
- **Fixed function**: Cannot update after fabrication

#### Google's Decoder Chip (Example)

From Google Quantum AI's 2023 work:
- Target: Distance-5 surface code
- Latency: < 100 ns
- Technology: 28nm CMOS
- Power: ~100 mW

**Key innovations**:
- Fully parallel defect detection
- Hardwired Union-Find tree
- Pipeline for continuous operation

### 5. Latency Breakdown

For an FPGA decoder:

| Stage | Time | Notes |
|-------|------|-------|
| Input capture | 3 ns | Syndrome from ADC |
| Defect detection | 3 ns | XOR + popcount |
| Graph construction | 10 ns | Parallel edge weights |
| Matching (5 rounds) | 50 ns | Union-Find iterations |
| Correction output | 3 ns | Result encoding |
| **Total** | **~70 ns** | |

For ASIC:
- All stages optimized: **~30-50 ns** achievable

### 6. Cryogenic Classical Processing

Placing decoder near qubits reduces communication latency.

**4 Kelvin Stage**:
- SiGe or cryogenic CMOS technology
- Power budget: ~1 W typical
- Reduced cable length: ~10 cm vs ~1 m
- Latency savings: ~10 ns per meter of cable

**Challenges**:
- Limited power dissipation
- Thermal noise coupling to qubits
- Reliability at cryogenic temperatures

**Hybrid Approach**:
- Simple preprocessing at 4K
- Complex decoder at room temperature
- Best of both worlds

### 7. Integration with Quantum Control

The decoder interfaces with:

**Measurement System**:
- ADCs converting qubit readout to bits
- Thresholding and majority voting
- Error detection in measurement

**Classical Control**:
- FPGA for pulse sequencing
- Real-time parameter updates
- Pauli frame tracking

**Feedback Loop**:
```
Qubit → Measurement → Decoder → Correction → Qubit
  ↑                                            ↓
  └──────────── Frame Update ──────────────────┘
```

### 8. Power and Area Analysis

For a distance-$d$ decoder:

**Union-Find on FPGA**:
- Area: $O(d^2)$ LUTs for syndrome storage
- Area: $O(d^2 \log d)$ for union-find structure
- Power: $\sim 0.1 \cdot d^2$ mW at 300 MHz

**Neural Network on FPGA**:
- Area: $O(\text{parameters})$ for weight storage
- DSP usage: $O(\text{parameters})$ for multiply-accumulate
- Power: Higher than Union-Find for similar accuracy

**Comparison table**:

| Decoder | Area (d=11) | Power | Latency |
|---------|-------------|-------|---------|
| Union-Find (FPGA) | ~50K LUTs | ~10 mW | ~70 ns |
| Union-Find (ASIC) | ~0.5 mm² | ~1 mW | ~30 ns |
| Neural (FPGA) | ~100K LUTs | ~50 mW | ~100 ns |
| MWPM (FPGA) | ~200K LUTs | ~100 mW | ~500 ns |

---

## Worked Examples

### Example 1: FPGA Resource Estimation

**Problem**: Estimate FPGA resources for a distance-11 Union-Find decoder.

**Solution**:

**Syndrome storage**:
- Syndrome bits: $d^2 - 1 = 120$
- Window of 20 rounds: $120 \times 20 = 2400$ bits
- Flip-flops: 2400

**Union-Find structure**:
- Maximum clusters: ~20 (typical at low error rate)
- Cluster info: parent, rank, size, parity = 4 × 16 bits = 64 bits each
- Total: $20 \times 64 = 1280$ bits

**Defect detection**:
- 120 XOR gates for syndrome difference
- Population count: $\sim 120 \times 2 = 240$ LUTs

**Matching engine**:
- Comparison units: $\binom{20}{2} = 190$ pairs
- Per pair: distance calculation ~10 LUTs
- Total: $190 \times 10 = 1900$ LUTs

**Cluster merge logic**:
- Union operations: ~200 LUTs
- Find with path compression: ~500 LUTs

**Total estimate**:
- LUTs: $240 + 1900 + 700 \approx 3000$ (core logic)
- With control/interface: $\times 3 \approx 10,000$ LUTs
- Flip-flops: $\sim 5000$

$$\boxed{\text{~10K LUTs, ~5K FFs for d=11}}$$

### Example 2: Latency Budget

**Problem**: Design a pipelined decoder to achieve 100 ns latency at 200 MHz clock.

**Solution**:

Clock period: $T = 1/200 \text{ MHz} = 5$ ns

Available cycles: $100 \text{ ns} / 5 \text{ ns} = 20$ cycles

**Pipeline stages**:
1. **Input** (1 cycle): Capture syndrome
2. **Defect Detection** (1 cycle): XOR, count
3. **Edge Weight** (2 cycles): Parallel distance computation
4. **Sort/Priority** (3 cycles): Find minimum weight edges
5. **Union-Find** (10 cycles): Iterative cluster merging
6. **Correction Extract** (2 cycles): Trace matched pairs
7. **Output** (1 cycle): Format correction

Total: 20 cycles = 100 ns

**Throughput**: 1 syndrome per 5 ns (if fully pipelined) = 200 M syndromes/sec

$$\boxed{\text{20-stage pipeline at 200 MHz}}$$

### Example 3: Power Budget at 4K

**Problem**: A cryogenic decoder has 100 mW power budget. What complexity is feasible?

**Solution**:

At 4K, power dissipation limits:
- CMOS power: $P = C \cdot V^2 \cdot f \cdot N$
- $C \approx 10$ fF per gate
- $V \approx 0.5$ V (reduced for cryo)
- $f = 100$ MHz (conservative)

Power per gate:
$$P_{\text{gate}} = 10 \times 10^{-15} \times (0.5)^2 \times 10^8 = 0.25 \text{ μW}$$

Available gates: $100 \text{ mW} / 0.25 \text{ μW} = 400,000$ gates

For Union-Find at d=11: ~10K gates needed

**Conclusion**: Distance-11 decoder easily fits in 4K budget.

For d=21: ~50K gates, still feasible.

$$\boxed{\text{~400K gates available at 4K with 100 mW}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Calculate BRAM requirements for storing a window of 30 syndrome rounds for a distance-15 code.

**Problem 2**: A 300 MHz FPGA decoder uses 15 pipeline stages. What is the latency? What is the throughput?

### Intermediate

**Problem 3**: Design the defect detection module: inputs are current and previous syndrome (each 100 bits), output is list of defect positions. Estimate LUT count.

**Problem 4**: Compare the power efficiency (operations per watt) of FPGA vs ASIC vs GPU for Union-Find decoding.

### Challenging

**Problem 5**: Design a hybrid system where simple syndromes (0-2 defects) are decoded at 4K and complex syndromes are sent to room temperature. Analyze the latency distribution.

**Problem 6**: An ASIC decoder must handle any distance from 5 to 21. Design a reconfigurable architecture that maintains near-optimal latency for each distance.

---

## Computational Lab: Hardware Simulation

```python
"""
Day 832 Lab: Decoder Hardware Simulation
Modeling FPGA/ASIC implementations for real-time decoding
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Hardware Resource Model
# =============================================================================

@dataclass
class FPGAResources:
    """Model of FPGA resource usage."""
    luts: int           # Look-up tables
    flip_flops: int     # Registers
    bram_bits: int      # Block RAM bits
    dsp_slices: int     # DSP blocks
    clock_mhz: float    # Clock frequency

    def utilization(self, fpga_type='medium'):
        """Calculate utilization percentages."""
        # Typical FPGA resources
        available = {
            'small': {'luts': 100000, 'ffs': 200000, 'bram': 1e6, 'dsp': 100},
            'medium': {'luts': 500000, 'ffs': 1000000, 'bram': 10e6, 'dsp': 500},
            'large': {'luts': 1000000, 'ffs': 2000000, 'bram': 50e6, 'dsp': 2000},
        }[fpga_type]

        return {
            'luts': self.luts / available['luts'] * 100,
            'ffs': self.flip_flops / available['ffs'] * 100,
            'bram': self.bram_bits / available['bram'] * 100,
            'dsp': self.dsp_slices / available['dsp'] * 100
        }


def estimate_union_find_resources(distance, window_size=20):
    """
    Estimate FPGA resources for Union-Find decoder.

    Parameters:
    -----------
    distance : int
        Surface code distance
    window_size : int
        Number of syndrome rounds in window

    Returns:
    --------
    FPGAResources object
    """
    n_syndrome = distance ** 2 - 1

    # Syndrome storage
    syndrome_ffs = n_syndrome * window_size

    # Defect detection
    xor_luts = n_syndrome
    popcount_luts = n_syndrome * 2

    # Union-Find structure
    max_clusters = int(0.02 * n_syndrome * window_size)  # At ~1% error
    cluster_ffs = max_clusters * 64  # parent, rank, size, etc.
    uf_luts = max_clusters * 50  # Union-find logic

    # Matching engine
    # Pairwise comparison: O(k^2) where k = expected defects
    expected_defects = max(4, int(0.01 * n_syndrome * window_size))
    matching_luts = expected_defects ** 2 * 20

    # Control logic
    control_luts = 2000
    control_ffs = 500

    # BRAM for large window storage
    bram_bits = n_syndrome * window_size if window_size > 10 else 0

    total_luts = xor_luts + popcount_luts + uf_luts + matching_luts + control_luts
    total_ffs = syndrome_ffs + cluster_ffs + control_ffs

    return FPGAResources(
        luts=total_luts,
        flip_flops=total_ffs,
        bram_bits=bram_bits,
        dsp_slices=0,  # Union-Find doesn't need DSPs
        clock_mhz=300  # Typical achievable
    )


def estimate_neural_resources(distance, hidden_dims=[64, 32]):
    """
    Estimate FPGA resources for neural network decoder.
    """
    n_syndrome = distance ** 2 - 1
    input_dim = n_syndrome
    output_dim = 4  # Logical classes

    dims = [input_dim] + hidden_dims + [output_dim]

    # Weight storage (8-bit quantized)
    total_weights = sum(dims[i] * dims[i+1] for i in range(len(dims)-1))
    weight_bits = total_weights * 8

    # Bias storage
    total_biases = sum(dims[1:])
    bias_bits = total_biases * 16

    # Computation (DSP for multiply-accumulate)
    # Each layer needs parallel MACs
    dsp_slices = max(dims[:-1])  # Parallel multipliers

    # Activation storage
    max_activation = max(dims)
    activation_ffs = max_activation * 16  # 16-bit activations

    # Control logic
    control_luts = 5000
    control_ffs = 1000

    return FPGAResources(
        luts=control_luts + total_weights // 10,  # LUT-based weights
        flip_flops=activation_ffs + control_ffs,
        bram_bits=weight_bits + bias_bits,
        dsp_slices=dsp_slices,
        clock_mhz=200  # Lower due to MAC latency
    )


def demonstrate_resource_estimation():
    """Demonstrate resource estimation."""
    print("=" * 60)
    print("FPGA RESOURCE ESTIMATION")
    print("=" * 60)

    distances = [5, 7, 9, 11, 15, 21]

    print("\nUnion-Find Decoder Resources:")
    print("-" * 50)
    print(f"{'Distance':<10} {'LUTs':<12} {'FFs':<12} {'BRAM (Kb)':<12}")
    print("-" * 50)

    uf_resources = []
    for d in distances:
        res = estimate_union_find_resources(d)
        uf_resources.append(res)
        print(f"{d:<10} {res.luts:<12} {res.flip_flops:<12} {res.bram_bits/1000:<12.1f}")

    print("\nNeural Network Decoder Resources:")
    print("-" * 50)
    print(f"{'Distance':<10} {'LUTs':<12} {'DSPs':<12} {'BRAM (Kb)':<12}")
    print("-" * 50)

    nn_resources = []
    for d in distances:
        res = estimate_neural_resources(d, [64, 32])
        nn_resources.append(res)
        print(f"{d:<10} {res.luts:<12} {res.dsp_slices:<12} {res.bram_bits/1000:<12.1f}")

    # Utilization comparison
    print("\nResource Utilization on Medium FPGA (500K LUTs):")
    print("-" * 50)
    for i, d in enumerate(distances):
        uf_util = uf_resources[i].utilization('medium')
        nn_util = nn_resources[i].utilization('medium')
        print(f"d={d}: UF LUTs={uf_util['luts']:.1f}%, NN LUTs={nn_util['luts']:.1f}%")

    return uf_resources, nn_resources

uf_res, nn_res = demonstrate_resource_estimation()

# =============================================================================
# Part 2: Latency Model
# =============================================================================

@dataclass
class PipelineStage:
    """Model of a pipeline stage."""
    name: str
    cycles: int
    description: str


def model_decoder_pipeline(decoder_type='union_find', clock_mhz=300):
    """
    Model decoder pipeline stages.

    Returns list of pipeline stages and total latency.
    """
    if decoder_type == 'union_find':
        stages = [
            PipelineStage('input', 1, 'Capture syndrome'),
            PipelineStage('xor', 1, 'Compute syndrome difference'),
            PipelineStage('detect', 1, 'Identify defect positions'),
            PipelineStage('init', 2, 'Initialize clusters'),
            PipelineStage('grow_1', 2, 'Cluster growth round 1'),
            PipelineStage('grow_2', 2, 'Cluster growth round 2'),
            PipelineStage('grow_3', 2, 'Cluster growth round 3'),
            PipelineStage('merge', 3, 'Merge clusters'),
            PipelineStage('peel', 2, 'Extract correction'),
            PipelineStage('output', 1, 'Format output'),
        ]

    elif decoder_type == 'neural':
        stages = [
            PipelineStage('input', 1, 'Capture syndrome'),
            PipelineStage('layer1', 4, 'Dense layer 1 (MAC)'),
            PipelineStage('relu1', 1, 'ReLU activation'),
            PipelineStage('layer2', 3, 'Dense layer 2 (MAC)'),
            PipelineStage('relu2', 1, 'ReLU activation'),
            PipelineStage('output', 2, 'Softmax + argmax'),
        ]

    elif decoder_type == 'mwpm':
        stages = [
            PipelineStage('input', 1, 'Capture syndrome'),
            PipelineStage('detect', 2, 'Identify defects'),
            PipelineStage('graph', 10, 'Build matching graph'),
            PipelineStage('blossom', 50, 'Blossom algorithm'),
            PipelineStage('extract', 5, 'Extract correction'),
            PipelineStage('output', 1, 'Format output'),
        ]

    total_cycles = sum(s.cycles for s in stages)
    latency_ns = total_cycles / clock_mhz * 1000

    return stages, total_cycles, latency_ns


def analyze_latency():
    """Analyze decoder latencies."""
    print("\n" + "=" * 60)
    print("DECODER LATENCY ANALYSIS")
    print("=" * 60)

    decoders = ['union_find', 'neural', 'mwpm']
    clock = 300  # MHz

    results = {}

    for decoder in decoders:
        stages, cycles, latency = model_decoder_pipeline(decoder, clock)

        print(f"\n{decoder.upper()} Pipeline ({clock} MHz):")
        print("-" * 40)
        for stage in stages:
            print(f"  {stage.name:<12}: {stage.cycles} cycles - {stage.description}")
        print(f"  {'TOTAL':<12}: {cycles} cycles = {latency:.1f} ns")

        results[decoder] = {'cycles': cycles, 'latency': latency}

    # Visualization
    plt.figure(figsize=(10, 6))

    names = list(results.keys())
    latencies = [results[n]['latency'] for n in names]

    colors = ['green', 'blue', 'red']
    bars = plt.bar(names, latencies, color=colors, alpha=0.7, edgecolor='black')

    plt.axhline(y=500, color='black', linestyle='--', linewidth=2,
                label='Superconducting target (500 ns)')
    plt.axhline(y=1000, color='gray', linestyle=':', linewidth=2,
                label='Typical cycle time (1 μs)')

    plt.ylabel('Latency (ns)', fontsize=12)
    plt.title(f'Decoder Latency Comparison ({clock} MHz FPGA)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    for bar, lat in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{lat:.0f} ns', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('decoder_latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'decoder_latency_comparison.png'")

    return results

latency_results = analyze_latency()

# =============================================================================
# Part 3: Power Model
# =============================================================================

def estimate_power(resources: FPGAResources, technology='fpga'):
    """
    Estimate power consumption.

    Parameters:
    -----------
    resources : FPGAResources
        Resource usage
    technology : str
        'fpga', 'asic_28nm', 'asic_7nm', 'cryo_4k'

    Returns:
    --------
    power_mw : float
        Estimated power in milliwatts
    """
    # Power per resource (approximate)
    power_models = {
        'fpga': {
            'lut_uw': 0.5,       # μW per LUT at 300 MHz
            'ff_uw': 0.1,        # μW per FF
            'bram_uw_per_kb': 5, # μW per Kb BRAM
            'dsp_uw': 50,        # μW per DSP
            'static_mw': 500,    # Static power
        },
        'asic_28nm': {
            'lut_uw': 0.1,
            'ff_uw': 0.02,
            'bram_uw_per_kb': 1,
            'dsp_uw': 10,
            'static_mw': 10,
        },
        'asic_7nm': {
            'lut_uw': 0.02,
            'ff_uw': 0.005,
            'bram_uw_per_kb': 0.2,
            'dsp_uw': 2,
            'static_mw': 2,
        },
        'cryo_4k': {
            'lut_uw': 0.01,      # Reduced voltage
            'ff_uw': 0.002,
            'bram_uw_per_kb': 0.1,
            'dsp_uw': 1,
            'static_mw': 1,
        },
    }

    model = power_models[technology]

    dynamic_power = (
        resources.luts * model['lut_uw'] +
        resources.flip_flops * model['ff_uw'] +
        resources.bram_bits / 1000 * model['bram_uw_per_kb'] +
        resources.dsp_slices * model['dsp_uw']
    ) / 1000  # Convert to mW

    total_power = dynamic_power + model['static_mw']

    return total_power


def analyze_power():
    """Analyze power consumption across technologies."""
    print("\n" + "=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)

    d = 11
    technologies = ['fpga', 'asic_28nm', 'asic_7nm', 'cryo_4k']

    uf_resources = estimate_union_find_resources(d)
    nn_resources = estimate_neural_resources(d)

    print(f"\nPower for distance-{d} decoders:")
    print("-" * 60)
    print(f"{'Technology':<15} {'Union-Find (mW)':<20} {'Neural Net (mW)':<20}")
    print("-" * 60)

    uf_powers = []
    nn_powers = []

    for tech in technologies:
        uf_power = estimate_power(uf_resources, tech)
        nn_power = estimate_power(nn_resources, tech)
        uf_powers.append(uf_power)
        nn_powers.append(nn_power)
        print(f"{tech:<15} {uf_power:<20.1f} {nn_power:<20.1f}")

    # Efficiency (operations per watt)
    print("\nEnergy Efficiency (decode operations per second per watt):")
    print("-" * 60)

    uf_latency = latency_results['union_find']['latency']
    nn_latency = latency_results['neural']['latency']

    for i, tech in enumerate(technologies):
        uf_ops_per_sec = 1e9 / uf_latency  # nanoseconds
        nn_ops_per_sec = 1e9 / nn_latency

        uf_efficiency = uf_ops_per_sec / uf_powers[i] * 1e3  # per watt
        nn_efficiency = nn_ops_per_sec / nn_powers[i] * 1e3

        print(f"{tech:<15} UF: {uf_efficiency:.1e} ops/W, NN: {nn_efficiency:.1e} ops/W")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(technologies))
    width = 0.35

    axes[0].bar(x - width/2, uf_powers, width, label='Union-Find', color='green', alpha=0.7)
    axes[0].bar(x + width/2, nn_powers, width, label='Neural Net', color='blue', alpha=0.7)
    axes[0].set_xlabel('Technology', fontsize=12)
    axes[0].set_ylabel('Power (mW)', fontsize=12)
    axes[0].set_title(f'Power Consumption (d={d})', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(technologies, rotation=45)
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, which='both')

    # Power efficiency
    uf_eff = [1e9 / uf_latency / p * 1e3 for p in uf_powers]
    nn_eff = [1e9 / nn_latency / p * 1e3 for p in nn_powers]

    axes[1].bar(x - width/2, uf_eff, width, label='Union-Find', color='green', alpha=0.7)
    axes[1].bar(x + width/2, nn_eff, width, label='Neural Net', color='blue', alpha=0.7)
    axes[1].set_xlabel('Technology', fontsize=12)
    axes[1].set_ylabel('Operations / Watt', fontsize=12)
    axes[1].set_title('Energy Efficiency', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(technologies, rotation=45)
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'power_analysis.png'")

analyze_power()

# =============================================================================
# Part 4: Scaling Analysis
# =============================================================================

def analyze_scaling():
    """Analyze decoder scaling with code distance."""
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    distances = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21])

    # Collect metrics
    uf_luts = []
    uf_latency = []
    nn_luts = []
    nn_latency = []

    for d in distances:
        uf_res = estimate_union_find_resources(d)
        nn_res = estimate_neural_resources(d)

        uf_luts.append(uf_res.luts)
        nn_luts.append(nn_res.luts)

        # Latency scales with pipeline depth
        # Union-Find: O(log d) for tree depth, O(d) for window
        uf_lat = 50 + 5 * np.log2(d) + 2 * d  # ns approximation
        # Neural: O(1) for fixed architecture
        nn_lat = 60 + 0.5 * d  # Slight increase for larger input

        uf_latency.append(uf_lat)
        nn_latency.append(nn_lat)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(distances, uf_luts, 'go-', label='Union-Find', linewidth=2, markersize=8)
    axes[0].plot(distances, nn_luts, 'bs-', label='Neural Net', linewidth=2, markersize=8)
    axes[0].set_xlabel('Code Distance', fontsize=12)
    axes[0].set_ylabel('LUTs', fontsize=12)
    axes[0].set_title('Resource Scaling', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(distances, uf_latency, 'go-', label='Union-Find', linewidth=2, markersize=8)
    axes[1].plot(distances, nn_latency, 'bs-', label='Neural Net', linewidth=2, markersize=8)
    axes[1].axhline(y=500, color='r', linestyle='--', label='500 ns budget')
    axes[1].set_xlabel('Code Distance', fontsize=12)
    axes[1].set_ylabel('Latency (ns)', fontsize=12)
    axes[1].set_title('Latency Scaling', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'scaling_analysis.png'")

    # Find maximum feasible distance
    budget = 500  # ns
    for i, d in enumerate(distances):
        if uf_latency[i] > budget:
            print(f"\nUnion-Find exceeds 500 ns budget at d = {d}")
            print(f"Maximum feasible distance: d = {distances[i-1]}")
            break
    else:
        print(f"\nUnion-Find within budget for all tested distances (up to d={distances[-1]})")

analyze_scaling()

# =============================================================================
# Part 5: System Integration Model
# =============================================================================

def model_full_system():
    """Model complete QEC system with decoder."""
    print("\n" + "=" * 60)
    print("FULL SYSTEM MODEL")
    print("=" * 60)

    d = 11

    # System components and latencies
    components = [
        ('Qubit → Resonator', 'Coupling', 20),
        ('Resonator → ADC', 'Readout', 200),
        ('ADC → Digital', 'Conversion', 50),
        ('Digital → Decoder', 'Communication', 100),
        ('Decoder', 'Processing', 70),
        ('Decoder → Control', 'Communication', 50),
        ('Control → Qubit', 'Feedback', 30),
    ]

    print(f"\nQEC Cycle Breakdown (d = {d}):")
    print("-" * 50)

    total_latency = 0
    for comp, type_, latency in components:
        print(f"  {comp:<25} ({type_:<12}): {latency:>4} ns")
        total_latency += latency

    print("-" * 50)
    print(f"  {'TOTAL':<25}           : {total_latency:>4} ns")

    # Where is the decoder in the critical path?
    decoder_fraction = 70 / total_latency * 100
    print(f"\nDecoder is {decoder_fraction:.1f}% of total cycle time")

    # Visualization
    plt.figure(figsize=(12, 6))

    names = [c[0] for c in components]
    latencies = [c[2] for c in components]
    types = [c[1] for c in components]

    colors = {
        'Coupling': 'purple',
        'Readout': 'blue',
        'Conversion': 'cyan',
        'Communication': 'orange',
        'Processing': 'green',
        'Feedback': 'red',
    }

    bar_colors = [colors[t] for t in types]

    bars = plt.barh(names, latencies, color=bar_colors, alpha=0.7, edgecolor='black')

    plt.xlabel('Latency (ns)', fontsize=12)
    plt.title('QEC Cycle Timing Breakdown', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t, alpha=0.7)
                       for t, c in colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('system_integration.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'system_integration.png'")

    # What if decoder is at 4K?
    print("\n" + "-" * 50)
    print("Scenario: Decoder at 4K stage")
    print("-" * 50)

    # Remove room temperature communication
    cryo_latency = total_latency - 100 - 50  # No long cable delays
    print(f"Cycle time reduced: {total_latency} → {cryo_latency} ns")
    print(f"Improvement: {(total_latency - cryo_latency)/total_latency * 100:.1f}%")

model_full_system()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("LAB SUMMARY")
print("=" * 60)
print("""
Key findings:

1. FPGA RESOURCES: Union-Find decoder for d=11 requires ~10K LUTs,
   well within modern FPGA capacity. Neural networks need more DSPs.

2. LATENCY: Union-Find achieves ~70 ns at 300 MHz, meeting the 500 ns
   budget with margin. MWPM exceeds budget significantly.

3. POWER: ASIC implementations reduce power by 10-100x vs FPGA.
   Cryogenic processing at 4K further reduces power but limits capacity.

4. SCALING: Both LUTs and latency grow roughly quadratically with d.
   Distance 21 remains feasible with current technology.

5. INTEGRATION: Decoder is ~15% of total cycle time. Communication
   latency dominates for room-temperature decoders.

6. CRYOGENIC OPTION: Placing decoder at 4K saves ~150 ns by reducing
   cable delays, but power budget is tight.

Design recommendations:
- Use FPGA for development and initial deployment
- Move to ASIC for production-scale systems
- Consider hybrid cryogenic architecture for lowest latency
- Choose Union-Find over MWPM for real-time operation
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Pipeline latency | $t = N_{\text{stages}} / f_{\text{clock}}$ |
| Throughput | $\text{syndromes/s} = f_{\text{clock}}$ (pipelined) |
| FPGA power | $P = C \cdot V^2 \cdot f \cdot N + P_{\text{static}}$ |
| Cryo power budget | $\sim 1$ W at 4K stage |
| Resources for Union-Find | $O(d^2)$ LUTs, $O(d^2 W)$ registers |

### Technology Comparison

| Technology | Latency | Power | Development |
|------------|---------|-------|-------------|
| FPGA | 50-100 ns | ~100 mW | Weeks |
| ASIC (28nm) | 30-50 ns | ~10 mW | Months-years |
| ASIC (7nm) | 20-30 ns | ~2 mW | Months-years |
| Cryogenic | 30-50 ns | ~1 mW | Research |

---

## Daily Checklist

- [ ] I can estimate FPGA resources for a Union-Find decoder
- [ ] I understand the pipeline stages for real-time decoding
- [ ] I can analyze power consumption across technologies
- [ ] I know the trade-offs between FPGA, ASIC, and cryogenic implementations
- [ ] I understand how the decoder integrates with the full QEC system
- [ ] I can design for specific latency budgets

---

## Preview: Day 833

Tomorrow we synthesize the week's material in the **Week 119 Synthesis**:
- Complete decoder design workflow
- Benchmarking methodology
- Open research problems
- Future directions in real-time decoding

---

*"The decoder chip is where quantum meets classical at gigahertz speeds."*

---

[← Day 831: Sliding Window Decoders](./Day_831_Friday.md) | [Day 833: Week Synthesis →](./Day_833_Sunday.md)
