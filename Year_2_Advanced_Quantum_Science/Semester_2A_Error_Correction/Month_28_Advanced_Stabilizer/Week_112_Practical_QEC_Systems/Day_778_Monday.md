# Day 778: Resource Overhead Analysis

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Qubit overhead formulas and scaling |
| Afternoon | 2.5 hours | Gate overhead and compilation costs |
| Evening | 2 hours | Space-time volume computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Calculate physical qubit requirements** for surface code implementations of target algorithms
2. **Derive T-gate factory overhead** including distillation levels and parallelization
3. **Compute space-time volume** for fault-tolerant quantum circuits
4. **Analyze overhead scaling** with logical error rate targets
5. **Compare resource requirements** across different code families
6. **Estimate total runtime** for realistic quantum algorithms

---

## Core Content

### 1. The Resource Overhead Problem

Fault-tolerant quantum computing requires encoding logical qubits into many physical qubits and replacing logical gates with fault-tolerant implementations. The central question is: **How many physical resources are needed?**

#### Overhead Categories

1. **Qubit overhead**: Physical qubits per logical qubit
2. **Gate overhead**: Physical gates per logical gate
3. **Time overhead**: Additional syndrome measurement cycles
4. **Classical processing overhead**: Decoding computation

### 2. Surface Code Qubit Overhead

For a distance-$d$ surface code:

$$\boxed{N_{\text{data}} = d^2}$$

$$\boxed{N_{\text{ancilla}} = d^2 - 1}$$

$$\boxed{N_{\text{total}} = 2d^2 - 1 \approx 2d^2}$$

The logical error rate scales as:

$$\boxed{p_L = A \left(\frac{p}{p_{\text{th}}}\right)^{\lfloor(d+1)/2\rfloor}}$$

where $A \sim 0.1$ is a fitting constant and $p_{\text{th}} \approx 0.01$ is the threshold.

#### Distance Selection

For a target logical error rate $\epsilon$ with physical error rate $p$:

$$\boxed{d \geq 2\left\lceil \frac{\log(\epsilon/A)}{\log(p/p_{\text{th}})} \right\rceil - 1}$$

**Example:** For $p = 10^{-3}$, $p_{\text{th}} = 10^{-2}$, targeting $\epsilon = 10^{-15}$:
$$d \geq 2\left\lceil \frac{\log(10^{-14})}{\log(0.1)} \right\rceil - 1 = 2 \times 14 - 1 = 27$$

This requires $N \approx 2 \times 27^2 = 1458$ physical qubits per logical qubit.

### 3. T-Gate Factory Overhead

T-gates cannot be implemented transversally in the surface code. Magic state distillation provides the non-Clifford resource.

#### Basic Distillation (15-to-1)

The 15-to-1 protocol:
- Input: 15 noisy T-states with error $\epsilon_{\text{in}}$
- Output: 1 cleaner T-state with error $\epsilon_{\text{out}} \approx 35\epsilon_{\text{in}}^3$
- Success probability: $\approx 1 - 15\epsilon_{\text{in}}$

**Output error after $k$ levels:**
$$\boxed{\epsilon_k \approx \left(\frac{\epsilon_0}{0.14}\right)^{3^k}}$$

**Factory qubit cost (single level):**
$$\boxed{N_{\text{factory}}^{(1)} = 15 \times (2d^2) = 30d^2}$$

**Multi-level factory:**
$$\boxed{N_{\text{factory}}^{(k)} \approx 15^k \times 2d^2}$$

However, catalyzed protocols and more efficient layouts reduce this significantly.

#### Litinski Factory Analysis

State-of-the-art factories (Litinski, 2019) achieve:

$$\boxed{N_{\text{factory}} \approx 12d^2 \text{ per T-state/cycle}}$$

with output rate of approximately one T-state per $d$ code cycles.

### 4. Space-Time Volume

The total resource cost combines space (qubits) and time (cycles):

$$\boxed{V_{ST} = N_{\text{qubits}} \times T_{\text{cycles}}}$$

This is the fundamental metric for comparing fault-tolerant implementations.

#### Algorithm Space-Time Volume

For an algorithm with:
- $n$ logical qubits
- $C$ Clifford gates
- $T$ T-gates
- Target logical error rate $\epsilon$

**Total qubits needed:**
$$\boxed{N_{\text{total}} = n \times 2d^2 + N_{\text{factories}}}$$

**Total cycles:**
$$\boxed{T_{\text{cycles}} = \frac{T \cdot d}{\text{(number of factories)}} + T_{\text{Clifford}}}$$

### 5. Complete Resource Estimation Framework

#### Gidney-Ekera Analysis (2021) for RSA-2048

Breaking RSA-2048 using Shor's algorithm:

| Resource | Count |
|----------|-------|
| Logical qubits | ~6,000 |
| T-gates | ~$10^{12}$ |
| Code distance | $d = 27$ |
| Physical qubits | ~20 million |
| Runtime | ~8 hours |

#### Scaling Laws

**Physical qubits vs. problem size $N$:**
$$\boxed{Q_{\text{phys}} = O(N \cdot d^2 \cdot \text{polylog}(N/\epsilon))}$$

**Runtime vs. T-count:**
$$\boxed{T_{\text{runtime}} = O\left(\frac{T_{\text{count}} \cdot d}{k}\right) \times t_{\text{cycle}}}$$

where $k$ is the number of parallel magic state factories and $t_{\text{cycle}}$ is the physical cycle time.

### 6. Comparison Across Code Families

| Code | Qubits/Logical | T-Gate Method | Threshold |
|------|---------------|---------------|-----------|
| Surface | $2d^2$ | Magic state | ~1% |
| Color | $\frac{3d^2+1}{4}$ | Magic state (transversal CCZ) | ~0.5% |
| 3D Gauge Color | $O(d^3)$ | Transversal T | ~0.1% |
| Concatenated Steane | $7^L$ | Transversal T at top | ~$10^{-4}$ |

---

## Quantum Mechanics Connection

### Fundamental Trade-offs

The overhead in QEC reflects fundamental physics:

1. **No-Cloning Theorem**: Cannot amplify quantum information classically, requiring redundant encoding
2. **Quantum-Classical Boundary**: Syndrome measurement projects errors without revealing logical information
3. **Threshold Theorem**: Below threshold, overhead scales polylogarithmically with $1/\epsilon$

### Physical Implementation Constraints

Real overhead depends on:
- Qubit coherence times ($T_1$, $T_2$)
- Gate fidelities
- Measurement speed
- Connectivity constraints

The space-time volume connects to the **quantum volume** metric used for near-term devices.

---

## Worked Examples

### Example 1: Surface Code Distance for Quantum Chemistry

**Problem:** A quantum chemistry simulation requires $10^{10}$ logical gates with total failure probability $\leq 1\%$. Physical error rate is $p = 5 \times 10^{-4}$. Find the required code distance.

**Solution:**

Target per-gate logical error: $p_L = 0.01 / 10^{10} = 10^{-12}$

Using the logical error formula with $A = 0.1$ and $p_{\text{th}} = 0.01$:
$$p_L = 0.1 \left(\frac{5 \times 10^{-4}}{0.01}\right)^{(d+1)/2} = 0.1 \times (0.05)^{(d+1)/2}$$

Setting $p_L = 10^{-12}$:
$$10^{-12} = 0.1 \times (0.05)^{(d+1)/2}$$
$$(0.05)^{(d+1)/2} = 10^{-11}$$
$$\frac{d+1}{2} \log(0.05) = -11$$
$$\frac{d+1}{2} = \frac{11}{1.30} \approx 8.46$$
$$d = 2 \times 8.46 - 1 \approx 16$$

Rounding to nearest odd: $\boxed{d = 17}$

Physical qubits per logical: $N = 2 \times 17^2 = 578$

### Example 2: Magic State Factory Sizing

**Problem:** An algorithm requires $10^8$ T-gates in total, and we want to complete it in $10^6$ cycles. Each cycle takes 1 $\mu$s. How many magic state factories are needed?

**Solution:**

T-gate production rate needed:
$$\text{Rate} = \frac{10^8 \text{ T-gates}}{10^6 \text{ cycles}} = 100 \text{ T-gates/cycle}$$

Each factory produces approximately 1 T-state per $d$ cycles. For $d = 17$:
$$\text{Rate per factory} = \frac{1}{17} \text{ T-states/cycle}$$

Number of factories:
$$k = 100 \times 17 = 1700 \text{ factories}$$

Factory qubits (using $12d^2$ per factory):
$$N_{\text{factory}} = 1700 \times 12 \times 17^2 = 5.9 \times 10^6 \text{ qubits}$$

$$\boxed{k = 1700 \text{ factories, } N_{\text{factory}} \approx 6 \times 10^6 \text{ qubits}}$$

### Example 3: Space-Time Volume Comparison

**Problem:** Compare the space-time volume for running Grover search on $N = 2^{20}$ items using: (a) $d = 11$ with 100 factories, (b) $d = 17$ with 50 factories.

**Solution:**

Grover requires $O(\sqrt{N}) = O(2^{10}) \approx 1000$ iterations, each with $O(n) = O(20)$ Toffolis. Each Toffoli needs 7 T-gates.

Total T-gates: $T = 1000 \times 20 \times 7 = 1.4 \times 10^5$

Logical qubits: $n = 20$

**Case (a): $d = 11$, $k = 100$**
- Data qubits: $20 \times 2 \times 11^2 = 4840$
- Factory qubits: $100 \times 12 \times 11^2 = 1.45 \times 10^5$
- Total qubits: $N_a \approx 1.5 \times 10^5$
- Time: $T_a = (1.4 \times 10^5 \times 11)/100 = 1.54 \times 10^4$ cycles
- Volume: $V_a = 1.5 \times 10^5 \times 1.54 \times 10^4 = 2.3 \times 10^9$

**Case (b): $d = 17$, $k = 50$**
- Data qubits: $20 \times 2 \times 17^2 = 11560$
- Factory qubits: $50 \times 12 \times 17^2 = 1.73 \times 10^5$
- Total qubits: $N_b \approx 1.85 \times 10^5$
- Time: $T_b = (1.4 \times 10^5 \times 17)/50 = 4.76 \times 10^4$ cycles
- Volume: $V_b = 1.85 \times 10^5 \times 4.76 \times 10^4 = 8.8 \times 10^9$

$$\boxed{V_a / V_b = 0.26 \text{; Case (a) is more efficient}}$$

Note: Case (b) has lower logical error rate but higher space-time cost.

---

## Practice Problems

### Level A: Direct Application

**A1.** A surface code with distance $d = 9$ has what total physical qubit count? What is the logical error rate if $p = 10^{-3}$ and $p_{\text{th}} = 0.01$?

**A2.** A magic state factory uses the 15-to-1 protocol at a single level. If input error is $\epsilon_0 = 10^{-2}$, what is the output error?

**A3.** An algorithm requires 500 logical qubits at distance $d = 21$. Calculate the data qubit overhead.

### Level B: Intermediate Analysis

**B1.** Design a magic state factory to achieve output error $\epsilon < 10^{-10}$ starting from noisy T-states with $\epsilon_0 = 10^{-2}$. How many distillation levels are needed? What is the qubit cost for $d = 15$?

**B2.** A quantum computer has $10^7$ physical qubits. If 80% are allocated to magic state factories at $d = 13$, how many T-gates per second can be produced if cycle time is 1 $\mu$s?

**B3.** Compare the total physical qubit requirement for encoding 100 logical qubits in (a) surface code at $d = 15$, (b) concatenated Steane code at level 3.

### Level C: Research-Level Challenges

**C1.** Derive the optimal code distance as a function of physical error rate $p$ and target logical error $\epsilon$, accounting for the cost of magic state distillation at multiple levels.

**C2.** A fault-tolerant Shor's algorithm for 4096-bit RSA has approximately $4 \times 10^{12}$ T-gates. Estimate the total physical qubits and runtime assuming: $d = 31$, cycle time $= 1 \mu$s, with 2000 magic state factories.

**C3.** The space-time-error volume $V_{STE} = N \times T \times p_L$ should be minimized. Derive how code distance $d$ should scale with algorithm size (T-count) to minimize $V_{STE}$.

---

## Computational Lab

```python
"""
Day 778: Resource Overhead Analysis
Surface code resource estimation tools
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# =============================================================================
# SURFACE CODE OVERHEAD CALCULATOR
# =============================================================================

class SurfaceCodeCalculator:
    """Calculate resource overhead for surface code implementations."""

    def __init__(self, p_physical: float = 1e-3, p_threshold: float = 0.01):
        """
        Initialize calculator.

        Args:
            p_physical: Physical error rate
            p_threshold: Code threshold
        """
        self.p_phys = p_physical
        self.p_th = p_threshold
        self.A = 0.1  # Fitting constant for logical error rate

    def logical_error_rate(self, d: int) -> float:
        """Calculate logical error rate for distance d."""
        return self.A * (self.p_phys / self.p_th) ** ((d + 1) / 2)

    def qubits_per_logical(self, d: int) -> int:
        """Calculate physical qubits per logical qubit."""
        return 2 * d**2

    def required_distance(self, target_error: float) -> int:
        """Find minimum distance for target logical error rate."""
        if self.p_phys >= self.p_th:
            raise ValueError("Physical error rate must be below threshold")

        ratio = np.log(target_error / self.A) / np.log(self.p_phys / self.p_th)
        d = 2 * np.ceil(ratio) - 1

        # Ensure distance is odd and at least 3
        d = max(3, int(d))
        if d % 2 == 0:
            d += 1

        return d

    def factory_qubits(self, d: int, num_factories: int = 1) -> int:
        """Calculate qubits for magic state factories (Litinski-style)."""
        return num_factories * 12 * d**2

    def t_gate_rate(self, d: int, num_factories: int) -> float:
        """T-gates per code cycle with given factories."""
        return num_factories / d

    def algorithm_resources(
        self,
        n_logical: int,
        n_t_gates: int,
        target_error: float
    ) -> Dict:
        """
        Estimate resources for a complete algorithm.

        Args:
            n_logical: Number of logical qubits
            n_t_gates: Number of T-gates in circuit
            target_error: Target total failure probability

        Returns:
            Dictionary with resource estimates
        """
        # Per-gate error requirement
        p_per_gate = target_error / n_t_gates

        # Find required distance
        d = self.required_distance(p_per_gate)

        # Data qubit overhead
        data_qubits = n_logical * self.qubits_per_logical(d)

        # Factory sizing (aim for reasonable runtime)
        # Heuristic: enough factories to produce T-states in ~10^6 cycles
        target_cycles = 1e6
        n_factories = max(1, int(np.ceil(n_t_gates / (target_cycles * d))))
        factory_qubits = self.factory_qubits(d, n_factories)

        # Runtime in cycles
        t_rate = self.t_gate_rate(d, n_factories)
        runtime_cycles = n_t_gates / t_rate

        # Space-time volume
        total_qubits = data_qubits + factory_qubits
        spacetime = total_qubits * runtime_cycles

        return {
            'distance': d,
            'data_qubits': data_qubits,
            'n_factories': n_factories,
            'factory_qubits': factory_qubits,
            'total_qubits': total_qubits,
            'runtime_cycles': runtime_cycles,
            'spacetime_volume': spacetime,
            'logical_error_rate': self.logical_error_rate(d)
        }


class MagicStateFactory:
    """Model magic state distillation overhead."""

    def __init__(self, protocol: str = "15-to-1"):
        """
        Initialize factory model.

        Args:
            protocol: Distillation protocol name
        """
        self.protocol = protocol

        if protocol == "15-to-1":
            self.input_count = 15
            self.output_count = 1
            self.error_suppression = lambda e: 35 * e**3
        elif protocol == "20-to-4":
            # Reed-Muller based
            self.input_count = 20
            self.output_count = 4
            self.error_suppression = lambda e: 20 * e**2

    def output_error(self, input_error: float, levels: int = 1) -> float:
        """Calculate output error after multiple distillation levels."""
        error = input_error
        for _ in range(levels):
            error = self.error_suppression(error)
        return error

    def required_levels(self, input_error: float, target_error: float) -> int:
        """Find number of distillation levels needed."""
        levels = 0
        error = input_error
        while error > target_error and levels < 10:
            error = self.error_suppression(error)
            levels += 1
        return levels

    def qubit_cost(self, d: int, levels: int = 1) -> int:
        """Calculate qubit cost for multi-level factory."""
        if self.protocol == "15-to-1":
            return int(15**levels * 2 * d**2)
        else:
            return int((self.input_count/self.output_count)**levels * 2 * d**2)


def analyze_scaling():
    """Analyze resource scaling with algorithm parameters."""

    # Initialize calculator
    calc = SurfaceCodeCalculator(p_physical=1e-3)

    # Vary T-gate count
    t_counts = np.logspace(4, 12, 50)
    distances = []
    total_qubits = []
    spacetime_volumes = []

    for t in t_counts:
        resources = calc.algorithm_resources(
            n_logical=100,
            n_t_gates=int(t),
            target_error=0.01
        )
        distances.append(resources['distance'])
        total_qubits.append(resources['total_qubits'])
        spacetime_volumes.append(resources['spacetime_volume'])

    return t_counts, distances, total_qubits, spacetime_volumes


def plot_resource_analysis():
    """Generate resource scaling plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Logical error rate vs distance
    ax1 = axes[0, 0]
    distances = np.arange(3, 31, 2)
    for p in [1e-4, 5e-4, 1e-3, 2e-3]:
        calc = SurfaceCodeCalculator(p_physical=p)
        errors = [calc.logical_error_rate(d) for d in distances]
        ax1.semilogy(distances, errors, 'o-', label=f'p = {p:.0e}')

    ax1.axhline(1e-15, color='gray', linestyle='--', label='Target')
    ax1.set_xlabel('Code Distance d')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_title('Logical Error Rate vs Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Physical qubits vs distance
    ax2 = axes[0, 1]
    n_logical_values = [10, 50, 100, 500]
    for n in n_logical_values:
        qubits = [n * 2 * d**2 for d in distances]
        ax2.semilogy(distances, qubits, 's-', label=f'n = {n} logical')

    ax2.set_xlabel('Code Distance d')
    ax2.set_ylabel('Physical Qubits (data only)')
    ax2.set_title('Data Qubit Overhead')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distillation output error
    ax3 = axes[1, 0]
    factory = MagicStateFactory("15-to-1")
    input_errors = np.logspace(-3, -1, 30)

    for levels in [1, 2, 3]:
        output = [factory.output_error(e, levels) for e in input_errors]
        ax3.loglog(input_errors, output, label=f'{levels} level(s)')

    ax3.plot([1e-3, 1e-1], [1e-3, 1e-1], 'k--', label='Break-even')
    ax3.set_xlabel('Input Error Rate')
    ax3.set_ylabel('Output Error Rate')
    ax3.set_title('Magic State Distillation (15-to-1)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Resource scaling with T-count
    ax4 = axes[1, 1]
    t_counts, _, total_qubits, spacetime = analyze_scaling()

    ax4.loglog(t_counts, total_qubits, 'b-', linewidth=2, label='Total Qubits')
    ax4_twin = ax4.twinx()
    ax4_twin.loglog(t_counts, spacetime, 'r-', linewidth=2, label='Space-Time Volume')

    ax4.set_xlabel('T-Gate Count')
    ax4.set_ylabel('Physical Qubits', color='blue')
    ax4_twin.set_ylabel('Space-Time Volume', color='red')
    ax4.set_title('Resource Scaling (100 logical qubits)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_778_resource_overhead.png', dpi=150, bbox_inches='tight')
    plt.show()


def detailed_resource_estimate():
    """Print detailed resource estimate for a specific algorithm."""

    print("=" * 60)
    print("RESOURCE ESTIMATE: Quantum Chemistry Simulation")
    print("=" * 60)

    # Parameters for a typical quantum chemistry problem
    n_logical = 200  # Logical qubits
    n_t_gates = int(1e10)  # T-gate count
    target_error = 0.01  # 1% failure probability

    calc = SurfaceCodeCalculator(p_physical=1e-3)
    resources = calc.algorithm_resources(n_logical, n_t_gates, target_error)

    print(f"\nInput Parameters:")
    print(f"  Logical qubits: {n_logical}")
    print(f"  T-gate count: {n_t_gates:.2e}")
    print(f"  Target error: {target_error}")
    print(f"  Physical error rate: {calc.p_phys}")

    print(f"\nDerived Parameters:")
    print(f"  Code distance: d = {resources['distance']}")
    print(f"  Logical error rate: {resources['logical_error_rate']:.2e}")

    print(f"\nQubit Requirements:")
    print(f"  Data qubits: {resources['data_qubits']:,}")
    print(f"  Number of T-factories: {resources['n_factories']}")
    print(f"  Factory qubits: {resources['factory_qubits']:,}")
    print(f"  TOTAL QUBITS: {resources['total_qubits']:,}")

    print(f"\nTime Requirements:")
    print(f"  Runtime (cycles): {resources['runtime_cycles']:.2e}")
    cycle_time_us = 1  # microseconds
    runtime_s = resources['runtime_cycles'] * cycle_time_us * 1e-6
    print(f"  Runtime (seconds): {runtime_s:.2e}")
    print(f"  Runtime (hours): {runtime_s/3600:.1f}")

    print(f"\nSpace-Time Volume: {resources['spacetime_volume']:.2e}")
    print("=" * 60)


def compare_code_families():
    """Compare resource requirements across code families."""

    print("\n" + "=" * 60)
    print("CODE FAMILY COMPARISON")
    print("=" * 60)

    # Target: 50 logical qubits, 10^8 T-gates, 1% error
    n_logical = 50
    n_t_gates = int(1e8)
    target_error = 0.01

    results = []

    # Surface code
    calc_surface = SurfaceCodeCalculator(p_physical=1e-3)
    d_surface = calc_surface.required_distance(target_error/n_t_gates)
    qubits_surface = n_logical * 2 * d_surface**2
    results.append(('Surface', d_surface, qubits_surface))

    # Color code (slightly lower threshold)
    calc_color = SurfaceCodeCalculator(p_physical=1e-3, p_threshold=0.005)
    d_color = calc_color.required_distance(target_error/n_t_gates)
    qubits_color = n_logical * int((3*d_color**2 + 1)/4)
    results.append(('Color', d_color, qubits_color))

    # Concatenated Steane (much lower threshold but transversal T)
    # Level L has 7^L physical qubits
    p_cat = 1e-3
    p_L = p_cat
    level = 0
    while p_L > target_error/n_t_gates and level < 10:
        # Error goes as ~ p^(2^L) for concatenated codes
        p_L = p_cat ** (2 ** level)
        level += 1
    qubits_steane = n_logical * (7 ** level)
    results.append(('Steane (concat)', f'L={level}', qubits_steane))

    print(f"\nTarget: {n_logical} logical qubits, {n_t_gates:.0e} T-gates")
    print(f"Physical error rate: 10^-3\n")

    print(f"{'Code':<15} {'Distance/Level':<15} {'Qubits':<15}")
    print("-" * 45)
    for name, dist, qubits in results:
        print(f"{name:<15} {str(dist):<15} {qubits:,}")


if __name__ == "__main__":
    # Run demonstrations
    print("Day 778: Resource Overhead Analysis")
    print("=" * 60)

    # Detailed estimate for realistic algorithm
    detailed_resource_estimate()

    # Code family comparison
    compare_code_families()

    # Generate plots
    plot_resource_analysis()

    # Interactive examples
    print("\n" + "=" * 60)
    print("QUICK CALCULATIONS")
    print("=" * 60)

    calc = SurfaceCodeCalculator(p_physical=1e-3)

    # Example 1: Distance for target error
    target = 1e-10
    d = calc.required_distance(target)
    print(f"\nFor p_L < {target:.0e}:")
    print(f"  Required distance: d = {d}")
    print(f"  Qubits per logical: {calc.qubits_per_logical(d)}")
    print(f"  Actual p_L = {calc.logical_error_rate(d):.2e}")

    # Example 2: Factory analysis
    factory = MagicStateFactory("15-to-1")
    print(f"\n15-to-1 Distillation (input error = 1%):")
    for levels in range(1, 4):
        out_err = factory.output_error(0.01, levels)
        qubits = factory.qubit_cost(15, levels)
        print(f"  {levels} level(s): error = {out_err:.2e}, qubits = {qubits:,}")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Physical qubits per logical | $N = 2d^2$ |
| Logical error rate | $p_L = A(p/p_{\text{th}})^{(d+1)/2}$ |
| Distance from target error | $d \geq 2\lceil\log(\epsilon/A)/\log(p/p_{\text{th}})\rceil - 1$ |
| 15-to-1 output error | $\epsilon_{\text{out}} \approx 35\epsilon_{\text{in}}^3$ |
| Factory qubits (Litinski) | $N_{\text{factory}} \approx 12d^2$ |
| Space-time volume | $V_{ST} = N_{\text{qubits}} \times T_{\text{cycles}}$ |
| T-gate production rate | 1 T-state per $d$ cycles per factory |

### Main Takeaways

1. **Overhead is substantial**: A fault-tolerant quantum computer requires millions of physical qubits for practical algorithms
2. **T-gates dominate cost**: Magic state distillation factories consume most resources
3. **Distance scales logarithmically**: With target error, requiring polylog overhead
4. **Space-time volume is the metric**: Combining qubits and time captures total resource cost
5. **Hardware improvements critical**: Lower physical error rates dramatically reduce overhead

---

## Daily Checklist

- [ ] I can calculate physical qubit requirements for surface codes
- [ ] I understand the 15-to-1 magic state distillation protocol
- [ ] I can estimate space-time volume for algorithms
- [ ] I can compare overhead across code families
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 779

Tomorrow we study **Lattice Surgery Operations**, the primary method for implementing logical gates in surface codes. We will learn:
- Merge and split operations for joint measurements
- Lattice surgery CNOT implementation
- Twist defects and alternative topological approaches
- Time-optimal circuit compilation

*"In lattice surgery, we sculpt logical operations from the fabric of the surface code itself."*

---

*Day 778 of 2184 | Year 2, Month 28, Week 112, Day 1*
*Quantum Engineering PhD Curriculum*
