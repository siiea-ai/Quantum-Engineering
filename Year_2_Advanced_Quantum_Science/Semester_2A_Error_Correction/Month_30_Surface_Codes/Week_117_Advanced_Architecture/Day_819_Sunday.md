# Day 819: Week 117 Synthesis — Advanced Surface Code Architecture

## Week 117, Day 7 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

Today we synthesize the architectural knowledge acquired throughout Week 117 into a unified framework for surface code design. We will build a comprehensive analysis tool that incorporates geometry selection, boundary configuration, connectivity requirements, and error budgeting. This synthesis prepares us for the operational aspects of surface codes—particularly lattice surgery—that we'll explore in Week 118.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Concept integration, architecture decision framework |
| Afternoon | 2 hours | Comprehensive synthesis project |
| Evening | 2 hours | Complete architecture analyzer implementation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Integrate** all Week 117 concepts into a coherent architectural framework
2. **Compare** different surface code architectures quantitatively
3. **Design** surface code patches optimized for specific hardware constraints
4. **Analyze** trade-offs between competing design choices
5. **Implement** a complete architecture analysis tool
6. **Prepare** for lattice surgery by understanding patch requirements

---

## Week 117 Concept Integration

### 1. The Architecture Decision Tree

Designing a surface code implementation involves a sequence of interconnected decisions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE CONSTRAINTS                          │
│    (Connectivity, Error Rates, Qubit Count, Coherence Time)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LATTICE GEOMETRY                              │
│         (Square / Heavy-Hex / Hexagonal / Triangular)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CODE ORIENTATION                              │
│              (Rotated vs Unrotated Surface Code)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BOUNDARY CONFIGURATION                        │
│           (Standard / All-Smooth / All-Rough / Custom)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYNDROME EXTRACTION                           │
│            (Direct / Flagged / Staged / Parallel)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISTANCE SELECTION                            │
│              (Based on Error Budget and Target p_L)             │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Key Trade-offs Summary

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Geometry | Square (4-way) | Heavy-hex (3-way) | Threshold vs. Fabrication |
| Rotation | Rotated | Unrotated | Qubit count vs. Simplicity |
| Boundaries | Standard | Customized | Simplicity vs. Flexibility |
| Syndrome | Direct | Flagged | Gate count vs. Fault tolerance |
| Distance | Higher d | Lower d | Reliability vs. Qubit count |

### 3. Unified Performance Metric

We can define a **figure of merit** that captures architecture quality:

$$\boxed{F = \frac{p_{th}}{p_{\text{eff}}} \cdot \frac{1}{\sqrt{n_{\text{physical}}/n_{\text{logical}}}} \cdot \eta_{\text{cycle}}}$$

Where:
- $p_{th}/p_{\text{eff}}$ = threshold margin (larger is better)
- $n_{\text{physical}}/n_{\text{logical}}$ = qubit overhead (smaller is better)
- $\eta_{\text{cycle}}$ = syndrome cycle efficiency (fraction of time doing useful work)

### 4. Architecture Comparison Matrix

| Architecture | Google-like | IBM-like | Ideal |
|--------------|-------------|----------|-------|
| Geometry | Square | Heavy-hex | Square |
| Connectivity | 4-way | 3-way | 4-way |
| Threshold | ~1% | ~0.6% | ~1% |
| Flags needed | No | Yes | No |
| Relative overhead | 1.0x | 1.3x | 1.0x |
| Fabrication ease | Medium | High | - |

---

## Synthesis Framework

### 5. Complete Design Workflow

**Step 1: Characterize Hardware**
```python
hardware = {
    'connectivity': 4,  # or 3
    'p_gate': 0.003,
    'p_meas': 0.01,
    'p_idle': 0.001,
    'cycle_time_us': 1.0,
    'T1_us': 100,
    'T2_us': 150,
}
```

**Step 2: Select Geometry**
Based on connectivity:
- 4-way → Square lattice
- 3-way → Heavy-hex or hexagonal

**Step 3: Calculate Effective Error Rate**
$$p_{\text{eff}} = n_g \cdot p_g + p_m + n_i \cdot p_i$$

**Step 4: Determine Distance**
$$d = 2\left\lceil \frac{\log(A/p_L^{\text{target}})}{\log(p_{th}/p_{\text{eff}})} \right\rceil - 1$$

**Step 5: Calculate Resources**
$$n_{\text{physical}} = 2d^2 - 1$$

### 6. Boundary Engineering for Operations

Standard boundaries support basic operations:
- **Logical X measurement:** Connect smooth boundaries
- **Logical Z measurement:** Connect rough boundaries
- **Storage:** All stabilizers measured continuously

For lattice surgery (Week 118):
- Merging patches requires matching boundary types
- Splitting patches creates new boundaries
- Twists at corners enable certain gates

### 7. Error Budget Optimization

Given total error budget $p_{\text{total}}$, optimize allocation:

**Objective:** Minimize $p_{\text{eff}}$ subject to $p_g + p_m + p_i \leq p_{\text{total}}$

**Solution:** Weight by impact:
$$p_g : p_m : p_i = \frac{1}{n_g} : 1 : \frac{1}{n_i}$$

### 8. Scaling Projections

For large-scale quantum computing:

| Logical Qubits | Distance | Physical Qubits (per) | Total Physical |
|----------------|----------|----------------------|----------------|
| 100 | 11 | 241 | 24,100 |
| 1,000 | 15 | 449 | 449,000 |
| 10,000 | 21 | 881 | 8,810,000 |
| 100,000 | 27 | 1,457 | 145,700,000 |

---

## Comprehensive Practice: Architecture Analysis Project

### Problem Statement

Design and analyze a surface code architecture for a near-term quantum computer with the following specifications:

**Hardware Parameters:**
- Available qubits: 1,000
- Qubit connectivity: 4-way (square grid)
- Two-qubit gate error: 0.4%
- Measurement error: 1.5%
- T1 = 80 μs, T2 = 120 μs
- Syndrome cycle: 800 ns

**Target Application:**
- Algorithm requiring $10^6$ logical operations
- Target success probability: 99%
- Number of logical qubits needed: 10

**Tasks:**
1. Calculate effective error rate
2. Determine required code distance
3. Verify feasibility with available qubits
4. Propose optimization strategies if needed

### Solution

**Step 1: Effective Error Rate**

Syndrome cycle = 800 ns, with:
- 4 CNOT gates per data qubit
- ~4 idle periods of ~100 ns each
- Idle error per 100 ns: $\frac{0.1 \mu s}{80 \mu s} \approx 0.125\%$

$$p_{\text{eff}} = 4 \times 0.4\% + 1.5\% + 4 \times 0.125\%$$
$$p_{\text{eff}} = 1.6\% + 1.5\% + 0.5\% = 3.6\%$$

**Problem:** This exceeds the ~1% threshold!

**Step 2: Error Reduction Needed**

To achieve $p_{\text{eff}} < 1\%$, we need significant improvements:
- Reduce gate error to 0.2% → contribution: 0.8%
- Reduce measurement error to 0.5% → contribution: 0.5%
- Total: 0.8% + 0.5% + 0.5% = 1.8%

Still too high. Need:
- Gate error: 0.1% → 0.4%
- Measurement: 0.3% → 0.3%
- Idle: (same) → 0.5%
- Total: 1.2%

Still marginal. Let's assume aggressive improvements achieve $p_{\text{eff}} = 0.5\%$.

**Step 3: Distance Selection**

For $10^6$ operations at 99% success:
$$p_L^{\text{target}} = \frac{0.01}{10^6} = 10^{-8}$$

With $p_{\text{eff}} = 0.5\%$, $p_{th} = 1\%$, $\Lambda = 2$:

$$\frac{d+1}{2} = \frac{\log(0.1/10^{-8})}{\log(2)} = \frac{\log(10^7)}{\log(2)} = \frac{7}{0.301} = 23.3$$

$$d = 2 \times 24 - 1 = 47$$

**Step 4: Resource Calculation**

Qubits per logical qubit: $2 \times 47^2 - 1 = 4,417$

For 10 logical qubits: 44,170 physical qubits

**Feasibility Check:** We only have 1,000 qubits! This is infeasible.

**Step 5: Alternative Approaches**

**Option A: Lower target reliability**
- Accept 90% success (not 99%)
- Reduces required distance significantly

**Option B: Fewer operations**
- Use error mitigation for short circuits
- Reserve QEC for critical sections

**Option C: Hybrid approach**
- Use $d = 5$ code (49 qubits per logical)
- 10 logical qubits × 49 = 490 qubits (feasible!)
- Accept higher logical error rate (~$10^{-3}$ per cycle)
- Combine with post-selection and verification

**Recommendation:** With 1,000 qubits and current error rates, pursue:
1. Error mitigation for near-term algorithms
2. Demonstrate $d = 3$ or $d = 5$ logical operations
3. Validate scaling predictions for future hardware

---

## Computational Lab

### Lab 819: Complete Architecture Analyzer

```python
"""
Day 819 Computational Lab: Comprehensive Surface Code Architecture Analyzer
============================================================================

This lab provides a complete tool for analyzing and comparing
surface code architectures, integrating all Week 117 concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings

class Geometry(Enum):
    SQUARE = "square"
    HEAVY_HEX = "heavy_hex"
    HEXAGONAL = "hexagonal"
    TRIANGULAR = "triangular"

class BoundaryConfig(Enum):
    STANDARD = "standard"  # SRSR (smooth-rough-smooth-rough)
    ALL_SMOOTH = "all_smooth"
    ALL_ROUGH = "all_rough"
    CUSTOM = "custom"

@dataclass
class HardwareSpec:
    """Hardware specification."""
    connectivity: int  # 3 or 4
    p_gate: float  # Two-qubit gate error
    p_meas: float  # Measurement error
    p_idle_per_us: float  # Idle error per microsecond
    cycle_time_us: float  # Syndrome cycle duration
    total_qubits: int  # Available physical qubits
    T1_us: float = 100.0  # T1 relaxation time
    T2_us: float = 150.0  # T2 dephasing time

@dataclass
class CodeSpec:
    """Code specification."""
    geometry: Geometry
    distance: int
    boundaries: BoundaryConfig
    use_flags: bool = False

@dataclass
class PerformanceMetrics:
    """Performance analysis results."""
    p_effective: float
    p_logical: float
    threshold_margin: float  # p_th / p_eff
    qubits_per_logical: int
    total_qubits_needed: int
    is_feasible: bool
    limiting_factor: str
    recommendations: List[str] = field(default_factory=list)

class SurfaceCodeArchitect:
    """
    Comprehensive surface code architecture analyzer.
    """

    # Threshold values for different configurations
    THRESHOLDS = {
        (Geometry.SQUARE, False): 0.01,      # Square, no flags
        (Geometry.SQUARE, True): 0.008,       # Square, with flags
        (Geometry.HEAVY_HEX, False): 0.006,   # Heavy-hex, no flags
        (Geometry.HEAVY_HEX, True): 0.008,    # Heavy-hex, with flags
        (Geometry.HEXAGONAL, False): 0.007,   # Hexagonal
        (Geometry.TRIANGULAR, False): 0.01,   # Triangular
    }

    # Gate counts per data qubit per syndrome cycle
    GATE_COUNTS = {
        Geometry.SQUARE: 4,
        Geometry.HEAVY_HEX: 4,  # Effectively 4 with flags
        Geometry.HEXAGONAL: 6,
        Geometry.TRIANGULAR: 6,
    }

    # Qubit overhead factors
    OVERHEAD_FACTORS = {
        Geometry.SQUARE: 2.0,        # 2d^2 - 1
        Geometry.HEAVY_HEX: 2.5,     # Extra flag qubits
        Geometry.HEXAGONAL: 3.0,     # Larger stabilizers
        Geometry.TRIANGULAR: 1.5,    # Smaller stabilizers
    }

    def __init__(self, hardware: HardwareSpec, prefactor: float = 0.1):
        """
        Initialize architect with hardware specifications.

        Parameters:
        -----------
        hardware : HardwareSpec
            Hardware specification
        prefactor : float
            Prefactor A in logical error rate formula
        """
        self.hw = hardware
        self.A = prefactor

    def select_geometry(self) -> Geometry:
        """Select optimal geometry based on connectivity."""
        if self.hw.connectivity >= 4:
            return Geometry.SQUARE
        elif self.hw.connectivity == 3:
            return Geometry.HEAVY_HEX
        else:
            warnings.warn("Low connectivity - consider Bacon-Shor codes")
            return Geometry.HEAVY_HEX

    def get_threshold(self, geometry: Geometry, use_flags: bool) -> float:
        """Get threshold for given configuration."""
        return self.THRESHOLDS.get((geometry, use_flags),
                                   self.THRESHOLDS[(geometry, False)])

    def calculate_effective_rate(self, geometry: Geometry,
                                 use_flags: bool = False) -> float:
        """
        Calculate effective error rate per syndrome cycle.
        """
        n_gates = self.GATE_COUNTS[geometry]
        if use_flags:
            n_gates += 2  # Additional gates for flag protocol

        # Idle time estimate (4 idle periods)
        n_idle_periods = 4
        idle_time_per_period = self.hw.cycle_time_us / 8  # Rough estimate
        p_idle_total = n_idle_periods * idle_time_per_period * self.hw.p_idle_per_us

        p_eff = (n_gates * self.hw.p_gate +
                 self.hw.p_meas +
                 p_idle_total)

        return p_eff

    def calculate_logical_rate(self, p_eff: float, distance: int,
                              p_threshold: float) -> float:
        """Calculate logical error rate."""
        if p_eff >= p_threshold:
            # Above threshold - exponential increase
            return min(1.0, self.A * (p_eff / p_threshold) ** ((distance + 1) / 2))

        exponent = (distance + 1) / 2
        return self.A * (p_eff / p_threshold) ** exponent

    def minimum_distance(self, p_eff: float, p_threshold: float,
                        target_p_L: float) -> int:
        """Calculate minimum distance for target logical error rate."""
        if p_eff >= p_threshold:
            return -1  # Impossible

        if target_p_L >= self.A:
            return 3  # Minimum distance

        ratio = np.log(self.A / target_p_L) / np.log(p_threshold / p_eff)
        d_min = int(np.ceil(2 * ratio - 1))

        # Ensure odd
        if d_min % 2 == 0:
            d_min += 1

        return max(3, d_min)

    def calculate_qubits(self, geometry: Geometry, distance: int,
                        use_flags: bool = False) -> int:
        """Calculate physical qubits per logical qubit."""
        base = self.OVERHEAD_FACTORS[geometry] * distance ** 2
        if use_flags:
            base *= 1.3  # ~30% overhead for flags

        return int(np.ceil(base))

    def analyze(self, n_logical: int, n_operations: int,
               target_success_prob: float,
               geometry: Optional[Geometry] = None) -> PerformanceMetrics:
        """
        Perform complete architecture analysis.

        Parameters:
        -----------
        n_logical : int
            Number of logical qubits needed
        n_operations : int
            Number of logical operations in algorithm
        target_success_prob : float
            Target success probability (e.g., 0.99)
        geometry : Geometry, optional
            Force specific geometry, or auto-select

        Returns:
        --------
        PerformanceMetrics
            Complete analysis results
        """
        recommendations = []

        # Select geometry
        if geometry is None:
            geometry = self.select_geometry()

        # Determine if flags are needed
        use_flags = self.hw.connectivity < 4

        # Get threshold
        p_th = self.get_threshold(geometry, use_flags)

        # Calculate effective rate
        p_eff = self.calculate_effective_rate(geometry, use_flags)

        # Check if below threshold
        if p_eff >= p_th:
            return PerformanceMetrics(
                p_effective=p_eff,
                p_logical=1.0,
                threshold_margin=p_th / p_eff,
                qubits_per_logical=0,
                total_qubits_needed=float('inf'),
                is_feasible=False,
                limiting_factor="ABOVE_THRESHOLD",
                recommendations=[
                    f"Effective rate {p_eff*100:.2f}% exceeds threshold {p_th*100:.2f}%",
                    "Reduce gate error or measurement error",
                    f"Need p_eff < {p_th*100:.1f}%"
                ]
            )

        # Calculate required logical error rate
        target_p_L = (1 - target_success_prob) / n_operations

        # Calculate minimum distance
        d_min = self.minimum_distance(p_eff, p_th, target_p_L)

        # Calculate qubits
        qubits_per = self.calculate_qubits(geometry, d_min, use_flags)
        total_qubits = n_logical * qubits_per

        # Calculate actual logical rate
        p_L = self.calculate_logical_rate(p_eff, d_min, p_th)

        # Check feasibility
        is_feasible = total_qubits <= self.hw.total_qubits
        limiting_factor = "NONE" if is_feasible else "QUBIT_COUNT"

        # Generate recommendations
        if not is_feasible:
            recommendations.append(f"Need {total_qubits:,} qubits but only have {self.hw.total_qubits:,}")
            recommendations.append(f"Consider reducing logical qubits or target success probability")

            # Calculate what's achievable
            achievable_d = int(np.sqrt(self.hw.total_qubits / n_logical / 2))
            if achievable_d >= 3:
                achievable_p_L = self.calculate_logical_rate(p_eff, achievable_d, p_th)
                recommendations.append(f"With d={achievable_d}, can achieve p_L={achievable_p_L:.2e}")

        if p_eff / p_th > 0.5:
            recommendations.append("Operating close to threshold - consider error reduction")

        return PerformanceMetrics(
            p_effective=p_eff,
            p_logical=p_L,
            threshold_margin=p_th / p_eff,
            qubits_per_logical=qubits_per,
            total_qubits_needed=total_qubits,
            is_feasible=is_feasible,
            limiting_factor=limiting_factor,
            recommendations=recommendations
        )

    def compare_geometries(self, n_logical: int, n_operations: int,
                          target_success_prob: float) -> Dict[Geometry, PerformanceMetrics]:
        """Compare all geometry options."""
        results = {}
        for geom in Geometry:
            try:
                results[geom] = self.analyze(n_logical, n_operations,
                                            target_success_prob, geom)
            except Exception as e:
                print(f"Error analyzing {geom}: {e}")
        return results

    def print_analysis(self, metrics: PerformanceMetrics, title: str = ""):
        """Print formatted analysis results."""
        print(f"\n{'='*60}")
        if title:
            print(f"{title}")
            print('='*60)
        print(f"Effective error rate: {metrics.p_effective*100:.3f}%")
        print(f"Threshold margin (λ): {metrics.threshold_margin:.2f}")
        print(f"Logical error rate: {metrics.p_logical:.2e}")
        print(f"Qubits per logical: {metrics.qubits_per_logical:,}")
        print(f"Total qubits needed: {metrics.total_qubits_needed:,}")
        print(f"Feasible: {metrics.is_feasible}")
        if metrics.limiting_factor != "NONE":
            print(f"Limiting factor: {metrics.limiting_factor}")
        if metrics.recommendations:
            print("\nRecommendations:")
            for rec in metrics.recommendations:
                print(f"  - {rec}")


def visualize_architecture_comparison(architect: SurfaceCodeArchitect):
    """Create comprehensive architecture comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters for comparison
    n_logical = 10
    n_operations = 1e6
    target_success = 0.99

    # Compare geometries
    results = architect.compare_geometries(n_logical, int(n_operations), target_success)

    # Plot 1: Effective error rates
    ax1 = axes[0, 0]
    geoms = list(results.keys())
    p_effs = [results[g].p_effective * 100 for g in geoms]
    thresholds = [architect.get_threshold(g, architect.hw.connectivity < 4) * 100 for g in geoms]

    x = np.arange(len(geoms))
    width = 0.35

    ax1.bar(x - width/2, p_effs, width, label='Effective Rate', color='steelblue')
    ax1.bar(x + width/2, thresholds, width, label='Threshold', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.value for g in geoms])
    ax1.set_ylabel('Error Rate (%)')
    ax1.set_title('Effective Rate vs Threshold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Qubit overhead
    ax2 = axes[0, 1]
    qubits = [results[g].qubits_per_logical for g in geoms]
    colors = ['green' if results[g].is_feasible else 'red' for g in geoms]

    ax2.bar(x, qubits, color=colors)
    ax2.axhline(y=architect.hw.total_qubits / n_logical, color='black',
               linestyle='--', label='Available per logical')
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.value for g in geoms])
    ax2.set_ylabel('Qubits per Logical')
    ax2.set_title('Qubit Overhead (green=feasible)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Logical error rate vs distance
    ax3 = axes[1, 0]
    distances = np.arange(3, 25, 2)

    for geom in [Geometry.SQUARE, Geometry.HEAVY_HEX]:
        p_eff = architect.calculate_effective_rate(geom, architect.hw.connectivity < 4)
        p_th = architect.get_threshold(geom, architect.hw.connectivity < 4)

        if p_eff < p_th:
            p_Ls = [architect.calculate_logical_rate(p_eff, d, p_th) for d in distances]
            ax3.semilogy(distances, p_Ls, 'o-', label=geom.value, linewidth=2)

    ax3.axhline(y=(1 - target_success) / n_operations, color='red',
               linestyle=':', label=f'Target ({(1-target_success)/n_operations:.0e})')
    ax3.set_xlabel('Code Distance')
    ax3.set_ylabel('Logical Error Rate')
    ax3.set_title('Logical Error Rate Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Trade-off space
    ax4 = axes[1, 1]

    # Scatter plot of qubits vs logical error rate
    for geom in geoms:
        r = results[geom]
        if r.qubits_per_logical > 0:
            marker = 'o' if r.is_feasible else 'x'
            size = 200 if r.is_feasible else 100
            ax4.scatter(r.qubits_per_logical, r.p_logical,
                       s=size, marker=marker, label=geom.value)

    ax4.set_xlabel('Qubits per Logical')
    ax4.set_ylabel('Logical Error Rate')
    ax4.set_yscale('log')
    ax4.set_title('Trade-off: Overhead vs Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_architecture_report(architect: SurfaceCodeArchitect,
                                n_logical: int, n_operations: int,
                                target_success: float) -> str:
    """Generate a detailed architecture report."""
    report = []
    report.append("=" * 70)
    report.append("SURFACE CODE ARCHITECTURE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Hardware summary
    report.append("HARDWARE SPECIFICATION")
    report.append("-" * 40)
    report.append(f"  Connectivity: {architect.hw.connectivity}-way")
    report.append(f"  Two-qubit gate error: {architect.hw.p_gate*100:.3f}%")
    report.append(f"  Measurement error: {architect.hw.p_meas*100:.3f}%")
    report.append(f"  Idle error rate: {architect.hw.p_idle_per_us*100:.3f}%/μs")
    report.append(f"  Syndrome cycle: {architect.hw.cycle_time_us:.2f} μs")
    report.append(f"  Available qubits: {architect.hw.total_qubits:,}")
    report.append("")

    # Application requirements
    report.append("APPLICATION REQUIREMENTS")
    report.append("-" * 40)
    report.append(f"  Logical qubits: {n_logical}")
    report.append(f"  Logical operations: {n_operations:.0e}")
    report.append(f"  Target success: {target_success*100:.1f}%")
    report.append(f"  Required p_L: {(1-target_success)/n_operations:.2e}")
    report.append("")

    # Analyze each geometry
    report.append("GEOMETRY COMPARISON")
    report.append("-" * 40)

    best_geom = None
    best_margin = 0

    for geom in [Geometry.SQUARE, Geometry.HEAVY_HEX]:
        result = architect.analyze(n_logical, int(n_operations), target_success, geom)

        report.append(f"\n  {geom.value.upper()}")
        report.append(f"    Effective rate: {result.p_effective*100:.3f}%")
        report.append(f"    Threshold margin: {result.threshold_margin:.2f}")
        report.append(f"    Feasible: {result.is_feasible}")

        if result.is_feasible:
            report.append(f"    Qubits needed: {result.total_qubits_needed:,}")
            report.append(f"    Logical error: {result.p_logical:.2e}")

            if result.threshold_margin > best_margin:
                best_margin = result.threshold_margin
                best_geom = geom

    report.append("")
    report.append("RECOMMENDATION")
    report.append("-" * 40)
    if best_geom:
        report.append(f"  Recommended geometry: {best_geom.value}")
        result = architect.analyze(n_logical, int(n_operations), target_success, best_geom)
        report.append(f"  Required distance: d = {int(np.sqrt(result.qubits_per_logical/2))}")
        report.append(f"  Total physical qubits: {result.total_qubits_needed:,}")
    else:
        report.append("  No feasible configuration found with current hardware.")
        report.append("  Consider improving error rates or reducing requirements.")

    return "\n".join(report)


# Main execution
if __name__ == "__main__":
    print("Week 117 Synthesis: Complete Architecture Analysis")
    print("=" * 60)

    # Define hardware spec (similar to Google Willow)
    hardware = HardwareSpec(
        connectivity=4,
        p_gate=0.003,
        p_meas=0.01,
        p_idle_per_us=0.001,
        cycle_time_us=1.0,
        total_qubits=1000,
        T1_us=100,
        T2_us=150
    )

    # Create architect
    architect = SurfaceCodeArchitect(hardware)

    # Basic analysis
    metrics = architect.analyze(
        n_logical=10,
        n_operations=int(1e6),
        target_success_prob=0.99
    )
    architect.print_analysis(metrics, "Baseline Analysis")

    # Generate full report
    report = generate_architecture_report(
        architect,
        n_logical=10,
        n_operations=int(1e6),
        target_success=0.99
    )
    print("\n" + report)

    # Visualization
    fig = visualize_architecture_comparison(architect)
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved architecture_comparison.png")

    # Sensitivity analysis
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    for p_gate in [0.001, 0.003, 0.005, 0.01]:
        test_hw = HardwareSpec(
            connectivity=4, p_gate=p_gate, p_meas=0.01,
            p_idle_per_us=0.001, cycle_time_us=1.0,
            total_qubits=10000
        )
        test_arch = SurfaceCodeArchitect(test_hw)
        result = test_arch.analyze(10, int(1e6), 0.99)

        print(f"\np_gate = {p_gate*100:.2f}%:")
        print(f"  p_eff = {result.p_effective*100:.3f}%")
        print(f"  Feasible: {result.is_feasible}")
        if result.is_feasible:
            print(f"  Qubits needed: {result.total_qubits_needed:,}")

    plt.show()
```

---

## Week 117 Summary

### Key Concepts Mastered

| Day | Topic | Core Insight |
|-----|-------|--------------|
| 813 | Rotated Geometry | 45° rotation reduces qubits by ~50% |
| 814 | Boundary Conditions | Smooth/rough boundaries determine logical operators |
| 815 | Twist Defects | Boundary transitions enable topological gates |
| 816 | Alternative Lattices | Geometry trades off threshold vs. fabrication |
| 817 | Ancilla Design | Syndrome extraction requires careful circuit design |
| 818 | Error Budgets | Distance selection from $p_L \approx (p/p_{th})^{(d+1)/2}$ |
| 819 | Synthesis | Architecture design is constrained optimization |

### Key Formulas Reference

$$\boxed{\text{Rotated Surface Code: } [[d^2, 1, d]]}$$

$$\boxed{t = \lfloor (d-1)/2 \rfloor \text{ correctable errors}}$$

$$\boxed{p_L \approx A \left(\frac{p}{p_{th}}\right)^{(d+1)/2}}$$

$$\boxed{d_{\min} = 2\left\lceil \frac{\log(A/p_L^{\text{target}})}{\log(p_{th}/p)} \right\rceil - 1}$$

$$\boxed{n_{\text{physical}} = 2d^2 - 1 \text{ (rotated surface code)}}$$

---

## Looking Ahead: Week 118

Next week we explore **Lattice Surgery Operations**:
- Merge and split operations
- Logical CNOT implementation
- T gate via magic state injection
- Full fault-tolerant gate set

Lattice surgery is how we perform computation on surface-code-encoded logical qubits.

---

## Final Checklist

- [ ] I understand the complete decision tree for architecture design
- [ ] I can compare different geometries quantitatively
- [ ] I can calculate distance requirements for target logical error rates
- [ ] I can identify limiting factors in a proposed architecture
- [ ] I have implemented and run the complete architecture analyzer
- [ ] I am ready to learn lattice surgery operations

---

*"The surface code is not just a code—it is a framework. Its architecture determines what is possible, and understanding that architecture is the first step toward building a fault-tolerant quantum computer."*

— Week 117 Conclusion
