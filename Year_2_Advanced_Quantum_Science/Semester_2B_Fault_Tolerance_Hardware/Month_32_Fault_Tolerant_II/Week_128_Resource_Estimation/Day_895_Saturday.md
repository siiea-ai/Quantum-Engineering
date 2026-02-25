# Day 895: Computational Lab - Complete Resource Estimator

## Week 128, Day 6 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Building the unified resource estimation framework |
| Afternoon | 2.5 hours | Algorithm analysis and benchmarking |
| Evening | 1.5 hours | Visualization and reporting tools |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Build a complete resource estimation toolkit** integrating all week's components
2. **Implement modular, extensible code architecture** for resource analysis
3. **Analyze arbitrary quantum algorithms** end-to-end
4. **Generate comprehensive reports** with visualizations
5. **Validate estimates** against published benchmarks
6. **Create reusable tools** for future research projects

---

## Lab Overview

Today we synthesize all components from Week 128 into a unified **Quantum Resource Estimation Framework** (QREF). This framework will:

1. Calculate physical qubit requirements
2. Analyze space-time volume
3. Design T-factory configurations
4. Estimate runtimes
5. Compare error-correcting codes
6. Generate publication-quality reports

---

## Part 1: Framework Architecture

### 1.1 Core Module Structure

```python
"""
Quantum Resource Estimation Framework (QREF)
Day 895 Computational Lab

A comprehensive toolkit for fault-tolerant quantum resource estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class CodeType(Enum):
    """Supported quantum error-correcting codes."""
    SURFACE = "surface"
    COLOR = "color"
    CONCATENATED = "concatenated"
    LDPC = "ldpc"


@dataclass
class HardwareSpec:
    """Physical hardware specifications."""
    name: str = "Generic Superconducting"
    physical_error_rate: float = 1e-3
    cycle_time_us: float = 1.0
    connectivity: str = "2D_grid"
    max_qubits: Optional[int] = None

    def __post_init__(self):
        self.cycle_time_s = self.cycle_time_us * 1e-6


@dataclass
class AlgorithmSpec:
    """Complete algorithm specification."""
    name: str
    n_logical_qubits: int
    t_count: int
    circuit_depth: int
    toffoli_count: int = 0
    measurement_count: int = 0
    description: str = ""
    reference: str = ""

    @property
    def total_non_clifford(self) -> int:
        """Total non-Clifford gates (T + Toffoli contributions)."""
        return self.t_count + 4 * self.toffoli_count

    @property
    def t_fraction(self) -> float:
        """Fraction of operations that are T gates."""
        if self.circuit_depth == 0:
            return 0
        return self.t_count / self.circuit_depth


@dataclass
class FactorySpec:
    """T-factory specification."""
    n_factories: int
    levels: int = 2
    protocol: str = "15-to-1"
    area_per_factory_d2: float = 150  # in d² units
    cycles_per_t_state: int = 272  # 8d * 2 levels for d=17

    def production_rate(self) -> float:
        """T-states produced per cycle."""
        return self.n_factories / self.cycles_per_t_state


@dataclass
class ResourceEstimate:
    """Complete resource estimation result."""
    # Input parameters
    algorithm: str
    code_type: CodeType
    hardware: str

    # Core metrics
    logical_qubits: int
    physical_qubits: int
    code_distance: int

    # Breakdown
    data_qubits: int
    routing_qubits: int
    factory_qubits: int
    ancilla_qubits: int

    # Time metrics
    runtime_cycles: float
    runtime_seconds: float
    runtime_hours: float

    # Space-time
    spacetime_volume: float

    # Factory details
    n_factories: int
    factory_production_rate: float

    # Quality metrics
    target_error: float
    achieved_error: float
    overhead_ratio: float

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm,
            'code_type': self.code_type.value,
            'hardware': self.hardware,
            'logical_qubits': self.logical_qubits,
            'physical_qubits': self.physical_qubits,
            'code_distance': self.code_distance,
            'data_qubits': self.data_qubits,
            'routing_qubits': self.routing_qubits,
            'factory_qubits': self.factory_qubits,
            'ancilla_qubits': self.ancilla_qubits,
            'runtime_cycles': self.runtime_cycles,
            'runtime_seconds': self.runtime_seconds,
            'runtime_hours': self.runtime_hours,
            'spacetime_volume': self.spacetime_volume,
            'n_factories': self.n_factories,
            'factory_production_rate': self.factory_production_rate,
            'target_error': self.target_error,
            'achieved_error': self.achieved_error,
            'overhead_ratio': self.overhead_ratio,
            'timestamp': self.timestamp,
            'notes': self.notes
        }


# ============================================================================
# ERROR CORRECTION CODE MODELS
# ============================================================================

class ErrorCorrectionCode(ABC):
    """Abstract base class for error-correcting codes."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def threshold(self) -> float:
        pass

    @abstractmethod
    def logical_error_rate(self, d: int, p_phys: float) -> float:
        pass

    @abstractmethod
    def physical_qubits_per_logical(self, d: int) -> int:
        pass

    @abstractmethod
    def t_gate_cycles(self, d: int) -> int:
        pass

    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, p_phys: float) -> int:
        """Find minimum distance for target error rate."""
        if p_phys >= self.threshold():
            return 999

        for d in range(3, 101, 2):  # Odd distances only
            p_L = self.logical_error_rate(d, p_phys)
            total_error = p_L * n_logical * depth
            if total_error < target_error:
                return d

        return 999


class SurfaceCodeModel(ErrorCorrectionCode):
    """Rotated surface code model."""

    def name(self) -> str:
        return "Rotated Surface Code"

    def threshold(self) -> float:
        return 0.01

    def logical_error_rate(self, d: int, p_phys: float) -> float:
        """Logical error rate scaling."""
        ratio = p_phys / self.threshold()
        return 0.1 * (ratio ** ((d + 1) / 2))

    def physical_qubits_per_logical(self, d: int) -> int:
        """Physical qubits per logical qubit."""
        return 2 * d * d

    def t_gate_cycles(self, d: int) -> int:
        """Cycles for T-gate via distillation."""
        return 16 * d  # 8d per level, 2 levels


class ColorCodeModel(ErrorCorrectionCode):
    """4.8.8 color code model."""

    def name(self) -> str:
        return "4.8.8 Color Code"

    def threshold(self) -> float:
        return 0.001

    def logical_error_rate(self, d: int, p_phys: float) -> float:
        ratio = p_phys / self.threshold()
        return 0.05 * (ratio ** ((d + 1) / 2))

    def physical_qubits_per_logical(self, d: int) -> int:
        return int(0.75 * d * d)

    def t_gate_cycles(self, d: int) -> int:
        return 10  # Transversal T


class ConcatenatedCodeModel(ErrorCorrectionCode):
    """Concatenated [[7,1,3]] Steane code."""

    def __init__(self, base_n: int = 7):
        self.base_n = base_n

    def name(self) -> str:
        return "Concatenated [[7,1,3]]"

    def threshold(self) -> float:
        return 0.0001

    def logical_error_rate(self, d: int, p_phys: float) -> float:
        # d = 3^k for k levels
        k = int(np.log(d) / np.log(3))
        return (p_phys ** (2 ** k))

    def physical_qubits_per_logical(self, d: int) -> int:
        k = int(np.log(max(1, d)) / np.log(3))
        return self.base_n ** k

    def t_gate_cycles(self, d: int) -> int:
        k = int(np.log(max(1, d)) / np.log(3))
        return 50 * k


# ============================================================================
# RESOURCE ESTIMATOR
# ============================================================================

class QuantumResourceEstimator:
    """
    Main resource estimation engine.

    Integrates qubit counting, space-time analysis, factory design,
    and runtime estimation.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.codes = {
            CodeType.SURFACE: SurfaceCodeModel(),
            CodeType.COLOR: ColorCodeModel(),
            CodeType.CONCATENATED: ConcatenatedCodeModel()
        }

    def estimate(
        self,
        algorithm: AlgorithmSpec,
        code_type: CodeType = CodeType.SURFACE,
        target_error: float = 0.01,
        target_runtime_hours: Optional[float] = None,
        routing_overhead: float = 0.4
    ) -> ResourceEstimate:
        """
        Perform complete resource estimation.

        Returns comprehensive ResourceEstimate object.
        """
        code = self.codes[code_type]

        # Check viability
        if self.hardware.physical_error_rate >= code.threshold():
            raise ValueError(
                f"Physical error rate {self.hardware.physical_error_rate} "
                f"exceeds threshold {code.threshold()} for {code.name()}"
            )

        # Calculate required code distance
        d = code.required_distance(
            algorithm.n_logical_qubits,
            algorithm.circuit_depth,
            target_error,
            self.hardware.physical_error_rate
        )

        # Calculate achieved error
        p_L = code.logical_error_rate(d, self.hardware.physical_error_rate)
        achieved_error = p_L * algorithm.n_logical_qubits * algorithm.circuit_depth

        # Physical qubits for data
        qubits_per_logical = code.physical_qubits_per_logical(d)
        data_qubits = algorithm.n_logical_qubits * qubits_per_logical

        # Routing overhead
        routing_qubits = int(routing_overhead * data_qubits)

        # Factory design
        if code_type == CodeType.COLOR:
            # No factories needed for color code
            n_factories = 0
            factory_qubits = 0
            factory_production_rate = float('inf')
            t_cycles = code.t_gate_cycles(d)
        else:
            # Design factories based on target runtime or default
            factory_area = 150 * d * d
            distill_cycles = code.t_gate_cycles(d)

            if target_runtime_hours:
                # Calculate factories needed for target
                target_cycles = target_runtime_hours * 3600 / self.hardware.cycle_time_s
                required_rate = algorithm.total_non_clifford / target_cycles
                n_factories = max(1, int(np.ceil(required_rate * distill_cycles)))
            else:
                # Default: enough for ~10 hour runtime
                n_factories = max(1, algorithm.total_non_clifford // int(1e9))

            factory_qubits = n_factories * factory_area
            factory_production_rate = n_factories / distill_cycles
            t_cycles = distill_cycles

        # Ancilla qubits (rough estimate: 5% overhead)
        ancilla_qubits = int(0.05 * (data_qubits + routing_qubits))

        # Total physical qubits
        total_qubits = data_qubits + routing_qubits + factory_qubits + ancilla_qubits

        # Runtime calculation
        if code_type == CodeType.COLOR:
            runtime_cycles = algorithm.total_non_clifford * t_cycles
        else:
            runtime_cycles = algorithm.total_non_clifford / max(1, factory_production_rate)

        runtime_seconds = runtime_cycles * self.hardware.cycle_time_s
        runtime_hours = runtime_seconds / 3600

        # Space-time volume
        spacetime_volume = total_qubits * runtime_cycles

        return ResourceEstimate(
            algorithm=algorithm.name,
            code_type=code_type,
            hardware=self.hardware.name,
            logical_qubits=algorithm.n_logical_qubits,
            physical_qubits=total_qubits,
            code_distance=d,
            data_qubits=data_qubits,
            routing_qubits=routing_qubits,
            factory_qubits=factory_qubits,
            ancilla_qubits=ancilla_qubits,
            runtime_cycles=runtime_cycles,
            runtime_seconds=runtime_seconds,
            runtime_hours=runtime_hours,
            spacetime_volume=spacetime_volume,
            n_factories=n_factories,
            factory_production_rate=factory_production_rate,
            target_error=target_error,
            achieved_error=achieved_error,
            overhead_ratio=total_qubits / algorithm.n_logical_qubits
        )

    def compare_codes(
        self,
        algorithm: AlgorithmSpec,
        target_error: float = 0.01
    ) -> Dict[CodeType, ResourceEstimate]:
        """Compare all viable codes for an algorithm."""
        results = {}

        for code_type in CodeType:
            if code_type == CodeType.LDPC:
                continue  # Skip LDPC (not fully implemented)

            try:
                estimate = self.estimate(algorithm, code_type, target_error)
                results[code_type] = estimate
            except ValueError as e:
                print(f"  {code_type.value}: Not viable - {e}")

        return results

    def optimize(
        self,
        algorithm: AlgorithmSpec,
        target_error: float = 0.01,
        max_qubits: Optional[int] = None,
        max_runtime_hours: Optional[float] = None
    ) -> Tuple[CodeType, ResourceEstimate]:
        """
        Find optimal code configuration for given constraints.

        Returns best code type and corresponding estimate.
        """
        results = self.compare_codes(algorithm, target_error)

        valid_results = []
        for code_type, estimate in results.items():
            if max_qubits and estimate.physical_qubits > max_qubits:
                continue
            if max_runtime_hours and estimate.runtime_hours > max_runtime_hours:
                continue
            valid_results.append((code_type, estimate))

        if not valid_results:
            raise ValueError("No viable configuration found within constraints")

        # Optimize for minimum qubits (could use other objectives)
        best = min(valid_results, key=lambda x: x[1].physical_qubits)
        return best


# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

class ResourceVisualizer:
    """Generate visualizations for resource estimates."""

    def __init__(self, style: str = 'default'):
        self.style = style
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_qubit_breakdown(self, estimate: ResourceEstimate, save_path: Optional[str] = None):
        """Pie chart of qubit allocation."""
        fig, ax = plt.subplots(figsize=(10, 8))

        labels = ['Data Qubits', 'Routing', 'Factories', 'Ancilla']
        sizes = [
            estimate.data_qubits,
            estimate.routing_qubits,
            estimate.factory_qubits,
            estimate.ancilla_qubits
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        explode = (0.02, 0.02, 0.05, 0.02)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
        )

        ax.set_title(
            f'{estimate.algorithm}: Physical Qubit Breakdown\n'
            f'Total: {estimate.physical_qubits:,} qubits',
            fontsize=14, fontweight='bold'
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_comparison_bar(self, results: Dict[CodeType, ResourceEstimate],
                            metric: str = 'physical_qubits',
                            save_path: Optional[str] = None):
        """Bar chart comparing codes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        codes = list(results.keys())
        values = [getattr(results[c], metric) for c in codes]
        labels = [c.value for c in codes]

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax.bar(labels, values, color=colors[:len(codes)], edgecolor='black')

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Code Comparison: {metric.replace("_", " ").title()}', fontsize=14)

        # Add value labels
        for bar, val in zip(bars, values):
            if metric == 'physical_qubits':
                label = f'{val/1e6:.2f}M'
            elif metric == 'runtime_hours':
                label = f'{val:.1f}h'
            else:
                label = f'{val:.2e}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   label, ha='center', va='bottom', fontsize=11, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_scaling_analysis(self, estimator: QuantumResourceEstimator,
                              base_algorithm: AlgorithmSpec,
                              scale_factors: List[float],
                              save_path: Optional[str] = None):
        """Show how resources scale with problem size."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for code_type in [CodeType.SURFACE, CodeType.COLOR]:
            qubits = []
            runtimes = []

            for scale in scale_factors:
                scaled_algo = AlgorithmSpec(
                    name=f"{base_algorithm.name}_scaled",
                    n_logical_qubits=int(base_algorithm.n_logical_qubits * scale),
                    t_count=int(base_algorithm.t_count * scale ** 2),
                    circuit_depth=int(base_algorithm.circuit_depth * scale ** 2)
                )

                try:
                    estimate = estimator.estimate(scaled_algo, code_type)
                    qubits.append(estimate.physical_qubits)
                    runtimes.append(estimate.runtime_hours)
                except:
                    qubits.append(np.nan)
                    runtimes.append(np.nan)

            logical_counts = [int(base_algorithm.n_logical_qubits * s) for s in scale_factors]

            axes[0].loglog(logical_counts, qubits, '-o', label=code_type.value,
                          linewidth=2, markersize=8)
            axes[1].loglog(logical_counts, runtimes, '-o', label=code_type.value,
                          linewidth=2, markersize=8)

        axes[0].set_xlabel('Logical Qubits', fontsize=12)
        axes[0].set_ylabel('Physical Qubits', fontsize=12)
        axes[0].set_title('Physical Qubit Scaling', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Logical Qubits', fontsize=12)
        axes[1].set_ylabel('Runtime (hours)', fontsize=12)
        axes[1].set_title('Runtime Scaling', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class ResourceReporter:
    """Generate comprehensive reports."""

    def __init__(self):
        self.visualizer = ResourceVisualizer()

    def generate_report(self, estimate: ResourceEstimate) -> str:
        """Generate text report for a single estimate."""
        report = []
        report.append("=" * 70)
        report.append("QUANTUM RESOURCE ESTIMATION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {estimate.timestamp}")
        report.append("")

        report.append("ALGORITHM SUMMARY")
        report.append("-" * 40)
        report.append(f"  Name:            {estimate.algorithm}")
        report.append(f"  Logical qubits:  {estimate.logical_qubits:,}")
        report.append(f"  Code type:       {estimate.code_type.value}")
        report.append(f"  Hardware:        {estimate.hardware}")
        report.append("")

        report.append("PHYSICAL RESOURCES")
        report.append("-" * 40)
        report.append(f"  Code distance:      {estimate.code_distance}")
        report.append(f"  Total physical:     {estimate.physical_qubits:,}")
        report.append(f"    - Data qubits:    {estimate.data_qubits:,}")
        report.append(f"    - Routing:        {estimate.routing_qubits:,}")
        report.append(f"    - Factories:      {estimate.factory_qubits:,}")
        report.append(f"    - Ancilla:        {estimate.ancilla_qubits:,}")
        report.append(f"  Overhead ratio:     {estimate.overhead_ratio:.1f}x")
        report.append("")

        report.append("TIMING ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Runtime (cycles):   {estimate.runtime_cycles:.2e}")
        report.append(f"  Runtime (seconds):  {estimate.runtime_seconds:.2e}")
        report.append(f"  Runtime (hours):    {estimate.runtime_hours:.2f}")
        if estimate.runtime_hours >= 24:
            report.append(f"  Runtime (days):     {estimate.runtime_hours/24:.2f}")
        report.append("")

        report.append("FACTORY CONFIGURATION")
        report.append("-" * 40)
        report.append(f"  Number of factories:    {estimate.n_factories}")
        report.append(f"  Production rate:        {estimate.factory_production_rate:.4f} T/cycle")
        report.append("")

        report.append("ERROR BUDGET")
        report.append("-" * 40)
        report.append(f"  Target error:       {estimate.target_error:.2e}")
        report.append(f"  Achieved error:     {estimate.achieved_error:.2e}")
        report.append("")

        report.append("SPACE-TIME VOLUME")
        report.append("-" * 40)
        report.append(f"  Volume:             {estimate.spacetime_volume:.2e} qubit-cycles")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def generate_comparison_report(
        self,
        results: Dict[CodeType, ResourceEstimate],
        algorithm_name: str
    ) -> str:
        """Generate comparative report across codes."""
        report = []
        report.append("=" * 80)
        report.append(f"CODE COMPARISON REPORT: {algorithm_name}")
        report.append("=" * 80)
        report.append("")

        # Header
        header = f"{'Code':<20} {'Qubits':>15} {'Runtime':>12} {'Distance':>10} {'Factories':>10}"
        report.append(header)
        report.append("-" * 80)

        for code_type, estimate in results.items():
            row = (f"{code_type.value:<20} "
                   f"{estimate.physical_qubits/1e6:>12.2f}M "
                   f"{estimate.runtime_hours:>10.1f}h "
                   f"{estimate.code_distance:>10} "
                   f"{estimate.n_factories:>10}")
            report.append(row)

        report.append("-" * 80)

        # Find best
        best_qubits = min(results.values(), key=lambda x: x.physical_qubits)
        best_runtime = min(results.values(), key=lambda x: x.runtime_hours)

        report.append("")
        report.append(f"Minimum qubits: {best_qubits.code_type.value} ({best_qubits.physical_qubits/1e6:.2f}M)")
        report.append(f"Minimum runtime: {best_runtime.code_type.value} ({best_runtime.runtime_hours:.1f}h)")
        report.append("=" * 80)

        return "\n".join(report)


# ============================================================================
# BENCHMARK ALGORITHMS
# ============================================================================

def get_benchmark_algorithms() -> Dict[str, AlgorithmSpec]:
    """Standard benchmark algorithms for testing."""
    return {
        'RSA-2048': AlgorithmSpec(
            name="RSA-2048 Factoring",
            n_logical_qubits=6189,
            t_count=int(2.04e10),
            circuit_depth=int(2e10),
            toffoli_count=int(5.1e9),
            description="Shor's algorithm for 2048-bit RSA",
            reference="Gidney & Ekerå (2021)"
        ),
        'RSA-4096': AlgorithmSpec(
            name="RSA-4096 Factoring",
            n_logical_qubits=12000,
            t_count=int(2e11),
            circuit_depth=int(2e11),
            description="Shor's algorithm for 4096-bit RSA"
        ),
        'ECDLP-256': AlgorithmSpec(
            name="ECDLP-256",
            n_logical_qubits=2330,
            t_count=int(5e9),
            circuit_depth=int(5e9),
            description="Discrete log on 256-bit elliptic curve"
        ),
        'FeMoco': AlgorithmSpec(
            name="FeMoco Simulation",
            n_logical_qubits=4000,
            t_count=int(1e13),
            circuit_depth=int(1e13),
            description="Nitrogen fixation catalyst simulation"
        ),
        'Hubbard-10x10': AlgorithmSpec(
            name="Hubbard Model 10x10",
            n_logical_qubits=200,
            t_count=int(1e11),
            circuit_depth=int(1e11),
            description="2D Hubbard model ground state"
        ),
        'QAOA-100': AlgorithmSpec(
            name="QAOA-100",
            n_logical_qubits=100,
            t_count=int(1e8),
            circuit_depth=int(1e6),
            description="QAOA with 100 variables, p=100"
        ),
        'VQE-50': AlgorithmSpec(
            name="VQE-50",
            n_logical_qubits=50,
            t_count=int(1e6),
            circuit_depth=int(1e4),
            description="VQE for small molecule"
        )
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def run_comprehensive_analysis():
    """Run complete analysis on benchmark suite."""
    print("=" * 70)
    print("QUANTUM RESOURCE ESTIMATION FRAMEWORK")
    print("Day 895 Computational Lab")
    print("=" * 70)

    # Setup hardware
    hardware = HardwareSpec(
        name="Advanced Superconducting",
        physical_error_rate=5e-4,
        cycle_time_us=0.1  # 100 ns
    )

    # Create estimator
    estimator = QuantumResourceEstimator(hardware)
    reporter = ResourceReporter()
    visualizer = ResourceVisualizer()

    # Get benchmarks
    benchmarks = get_benchmark_algorithms()

    # Analyze each algorithm
    all_results = {}
    for name, algo in benchmarks.items():
        print(f"\nAnalyzing: {name}")
        print("-" * 40)

        try:
            results = estimator.compare_codes(algo)
            all_results[name] = results

            for code_type, estimate in results.items():
                print(f"  {code_type.value:15s}: "
                      f"{estimate.physical_qubits/1e6:.2f}M qubits, "
                      f"{estimate.runtime_hours:.1f}h")

        except Exception as e:
            print(f"  Error: {e}")

    # Detailed report for RSA-2048
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: RSA-2048")
    print("=" * 70)

    rsa_estimate = estimator.estimate(
        benchmarks['RSA-2048'],
        CodeType.SURFACE,
        target_error=0.01
    )
    print(reporter.generate_report(rsa_estimate))

    # Comparison report
    if 'RSA-2048' in all_results:
        print(reporter.generate_comparison_report(
            all_results['RSA-2048'],
            "RSA-2048 Factoring"
        ))

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Qubit breakdown
    visualizer.plot_qubit_breakdown(rsa_estimate, 'rsa2048_breakdown.png')

    # Code comparison
    if 'RSA-2048' in all_results:
        visualizer.plot_comparison_bar(
            all_results['RSA-2048'],
            'physical_qubits',
            'code_comparison_qubits.png'
        )
        visualizer.plot_comparison_bar(
            all_results['RSA-2048'],
            'runtime_hours',
            'code_comparison_runtime.png'
        )

    # Scaling analysis
    visualizer.plot_scaling_analysis(
        estimator,
        benchmarks['QAOA-100'],
        [0.5, 1, 2, 5, 10, 20],
        'scaling_analysis.png'
    )

    # Summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'Algorithm':<20} {'Best Code':<15} {'Qubits':>12} {'Runtime':>10}"
    print(header)
    print("-" * 80)

    for name, results in all_results.items():
        if results:
            best = min(results.values(), key=lambda x: x.physical_qubits)
            print(f"{name:<20} {best.code_type.value:<15} "
                  f"{best.physical_qubits/1e6:>10.2f}M {best.runtime_hours:>9.1f}h")

    print("=" * 80)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    run_comprehensive_analysis()
```

---

## Part 2: Extended Analysis Tools

### 2.1 Sensitivity Analyzer

```python
"""
Sensitivity analysis module for QREF.
"""

def sensitivity_analysis(
    estimator: QuantumResourceEstimator,
    algorithm: AlgorithmSpec,
    parameter: str,
    values: List[float]
) -> Dict:
    """
    Analyze sensitivity of resources to a parameter.

    Parameters:
    - parameter: 'physical_error_rate', 'n_factories', 'code_distance'
    - values: Range of values to test
    """
    results = {'parameter': parameter, 'values': values, 'qubits': [], 'runtime': []}

    for val in values:
        # Modify parameter
        if parameter == 'physical_error_rate':
            modified_hw = HardwareSpec(
                name=estimator.hardware.name,
                physical_error_rate=val,
                cycle_time_us=estimator.hardware.cycle_time_us
            )
            mod_estimator = QuantumResourceEstimator(modified_hw)
            try:
                estimate = mod_estimator.estimate(algorithm, CodeType.SURFACE)
                results['qubits'].append(estimate.physical_qubits)
                results['runtime'].append(estimate.runtime_hours)
            except:
                results['qubits'].append(np.nan)
                results['runtime'].append(np.nan)
        # Add other parameters as needed

    return results


def plot_sensitivity(results: Dict, save_path: Optional[str] = None):
    """Visualize sensitivity analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogx(results['values'], np.array(results['qubits'])/1e6, 'b-o')
    axes[0].set_xlabel(results['parameter'], fontsize=12)
    axes[0].set_ylabel('Physical Qubits (millions)', fontsize=12)
    axes[0].set_title('Qubit Sensitivity', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(results['values'], results['runtime'], 'r-o')
    axes[1].set_xlabel(results['parameter'], fontsize=12)
    axes[1].set_ylabel('Runtime (hours)', fontsize=12)
    axes[1].set_title('Runtime Sensitivity', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 2.2 Trade-off Explorer

```python
"""
Trade-off exploration tools.
"""

def explore_qubit_runtime_tradeoff(
    estimator: QuantumResourceEstimator,
    algorithm: AlgorithmSpec,
    factory_range: Tuple[int, int] = (10, 500)
) -> Dict:
    """
    Explore the qubit-runtime trade-off space.
    """
    factories = np.linspace(factory_range[0], factory_range[1], 50).astype(int)

    results = {
        'factories': [],
        'qubits': [],
        'runtime': [],
        'volume': []
    }

    code = estimator.codes[CodeType.SURFACE]
    d = code.required_distance(
        algorithm.n_logical_qubits,
        algorithm.circuit_depth,
        0.01,
        estimator.hardware.physical_error_rate
    )

    for n_f in factories:
        # Calculate qubits
        data_qubits = algorithm.n_logical_qubits * code.physical_qubits_per_logical(d)
        routing_qubits = int(0.4 * data_qubits)
        factory_qubits = n_f * 150 * d * d
        total_qubits = data_qubits + routing_qubits + factory_qubits

        # Calculate runtime
        t_cycles = code.t_gate_cycles(d)
        production_rate = n_f / t_cycles
        runtime_cycles = algorithm.total_non_clifford / production_rate
        runtime_hours = runtime_cycles * estimator.hardware.cycle_time_s / 3600

        # Volume
        volume = total_qubits * runtime_cycles

        results['factories'].append(n_f)
        results['qubits'].append(total_qubits)
        results['runtime'].append(runtime_hours)
        results['volume'].append(volume)

    return results


def plot_tradeoff_curve(results: Dict, save_path: Optional[str] = None):
    """Plot the qubit-runtime trade-off curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Qubit vs Runtime
    scatter = axes[0].scatter(
        np.array(results['qubits'])/1e6,
        results['runtime'],
        c=results['factories'],
        cmap='viridis',
        s=50
    )
    axes[0].set_xlabel('Physical Qubits (millions)', fontsize=12)
    axes[0].set_ylabel('Runtime (hours)', fontsize=12)
    axes[0].set_title('Qubit-Runtime Trade-off', fontsize=14)
    plt.colorbar(scatter, ax=axes[0], label='Factories')

    # Volume vs Factories
    axes[1].semilogy(results['factories'], results['volume'], 'g-', linewidth=2)
    axes[1].set_xlabel('Number of Factories', fontsize=12)
    axes[1].set_ylabel('Space-Time Volume', fontsize=12)
    axes[1].set_title('Volume vs Factory Count', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Part 3: Validation Against Published Results

### 3.1 Gidney-Ekerå Validation

```python
"""
Validate estimates against published benchmarks.
"""

def validate_rsa2048():
    """
    Validate RSA-2048 estimates against Gidney-Ekerå (2021).

    Published values:
    - Physical qubits: ~20 million
    - Runtime: ~8 hours
    - Code distance: 27
    - Factories: 28
    """
    print("\n" + "="*60)
    print("VALIDATION: RSA-2048 (Gidney-Ekerå 2021)")
    print("="*60)

    published = {
        'physical_qubits': 20_165_344,
        'runtime_hours': 8,
        'code_distance': 27,
        'n_factories': 28
    }

    # Our estimate
    hardware = HardwareSpec(
        name="Reference Hardware",
        physical_error_rate=1e-3,
        cycle_time_us=0.1
    )

    estimator = QuantumResourceEstimator(hardware)
    algo = AlgorithmSpec(
        name="RSA-2048",
        n_logical_qubits=6189,
        t_count=int(2.04e10),
        circuit_depth=int(2e10)
    )

    estimate = estimator.estimate(algo, CodeType.SURFACE, target_error=0.01)

    print(f"\n{'Metric':<25} {'Published':>15} {'Estimated':>15} {'Ratio':>10}")
    print("-"*60)

    metrics = [
        ('Physical qubits', published['physical_qubits'], estimate.physical_qubits),
        ('Runtime (hours)', published['runtime_hours'], estimate.runtime_hours),
        ('Code distance', published['code_distance'], estimate.code_distance),
        ('Factories', published['n_factories'], estimate.n_factories)
    ]

    for name, pub, est in metrics:
        ratio = est / pub if pub > 0 else float('inf')
        print(f"{name:<25} {pub:>15,.0f} {est:>15,.0f} {ratio:>10.2f}x")

    print("="*60)
    print("\nNote: Differences due to modeling assumptions and optimizations.")
```

---

## Summary

### Framework Components

| Component | Purpose | Key Functions |
|-----------|---------|---------------|
| `HardwareSpec` | Hardware parameters | Error rate, cycle time |
| `AlgorithmSpec` | Algorithm definition | Qubits, T-count, depth |
| `ResourceEstimate` | Results container | All metrics |
| `QuantumResourceEstimator` | Main engine | `estimate()`, `compare_codes()` |
| `ResourceVisualizer` | Plotting | Breakdowns, comparisons |
| `ResourceReporter` | Text reports | Detailed summaries |

### Key Capabilities

1. **Multi-code analysis**: Surface, color, concatenated
2. **Complete resource breakdown**: Data, routing, factory, ancilla
3. **Runtime estimation**: Factory-limited and depth-limited
4. **Space-time volume**: Full volume analysis
5. **Optimization**: Find best code for constraints
6. **Validation**: Compare with published benchmarks

### Usage Pattern

```python
# 1. Define hardware
hardware = HardwareSpec(physical_error_rate=1e-3, cycle_time_us=1.0)

# 2. Create estimator
estimator = QuantumResourceEstimator(hardware)

# 3. Define algorithm
algorithm = AlgorithmSpec(name="MyAlgo", n_logical_qubits=100, ...)

# 4. Estimate resources
estimate = estimator.estimate(algorithm, CodeType.SURFACE)

# 5. Generate report
reporter = ResourceReporter()
print(reporter.generate_report(estimate))
```

---

## Daily Checklist

- [ ] I can build a modular resource estimation framework
- [ ] I understand the data structures for hardware, algorithms, and results
- [ ] I can implement error correction code models
- [ ] I can generate comprehensive resource estimates
- [ ] I can create visualizations for qubit breakdown and comparisons
- [ ] I can validate estimates against published benchmarks
- [ ] I can extend the framework for new codes or algorithms

---

## Preview: Day 896

Tomorrow is the **Month 32 Capstone** and **Semester 2B Midpoint Review**:

- Complete fault-tolerant toolkit assembly
- Integration test across all FT concepts
- Comprehensive review of Months 29-32
- Preparation for Month 33: Hardware Implementations

We'll bring together all the pieces from the semester into a unified understanding of fault-tolerant quantum computing.

---

*Day 895 of 2184 | Week 128 of 312 | Month 32 of 72*

*"The best tool is one that you build yourself—then you understand every piece."*
