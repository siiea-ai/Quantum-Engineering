# Day 866: Resource Estimation Framework

## Week 124: Universal Fault-Tolerant Computation | Month 31: Fault-Tolerant QC I

---

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Resource metrics and formulas |
| Afternoon | 2.5 hrs | Case studies: Shor and simulation |
| Evening | 2.0 hrs | Practical estimation tools |

---

### Learning Objectives

By the end of today, you will be able to:

1. **Identify the key resource metrics** for fault-tolerant quantum computing
2. **Derive the complete resource formulas** for T-count, qubit count, and runtime
3. **Apply the framework to Shor's algorithm** for specific bit sizes
4. **Estimate resources for quantum simulation** algorithms
5. **Use the Azure/Google resource estimation methodologies**
6. **Make informed comparisons** between algorithmic approaches

---

### Core Content

#### Part 1: The Resource Estimation Problem

**Why Resource Estimation Matters:**

Fault-tolerant quantum computing requires massive overhead. Before building billion-dollar quantum computers, we need to know:
- How many physical qubits?
- How long will the computation take?
- What is the expected success probability?
- How does this compare to classical alternatives?

**The Three Pillars of FT Resources:**

$$\boxed{\text{Resources} = \langle \text{T-count}, \text{Qubit Count}, \text{Runtime} \rangle}$$

These are not independent---they trade off against each other:
- More qubits → faster runtime (parallelism)
- Lower T-count → fewer qubits needed for factories
- Longer runtime → requires higher code distance (more qubits)

---

#### Part 2: T-Count Analysis

**Why T-Count Dominates:**

In fault-tolerant QC with the surface code:
- Clifford gates are "free" (transversal, fast)
- T-gates require magic state distillation (slow, expensive)

**Total T-Count Formula:**

$$\boxed{T_{\text{total}} = T_{\text{algorithm}} \times N_{\text{iterations}} + T_{\text{overhead}}}$$

where:
- $T_{\text{algorithm}}$ = T-gates per iteration of the main algorithm
- $N_{\text{iterations}}$ = repetitions for success probability
- $T_{\text{overhead}}$ = auxiliary operations (state prep, measurement)

**T-Count Sources:**

| Source | Typical T-Count |
|--------|-----------------|
| Arbitrary rotation $R_z(\theta)$ | $\approx 3\log_2(1/\epsilon)$ |
| Toffoli gate | 7 (or 4 with measurement) |
| Controlled-rotation | $\approx 2 \times 3\log_2(1/\epsilon)$ |
| Modular exponentiation (Shor) | $O(n^2 \log n)$ for $n$-bit |
| Hamiltonian simulation | $O(\text{poly}(1/\epsilon) \cdot t)$ |

**T-Count Optimization Strategies:**

1. **Use T-optimal decompositions:** Toffoli = 4T with measurement
2. **Batch rotations:** Synthesize sums, not individual angles
3. **Trade qubits for T's:** Ancilla-heavy approaches
4. **Exploit structure:** Symmetries reduce redundant gates

---

#### Part 3: Qubit Count Analysis

**Physical Qubit Budget:**

$$\boxed{Q_{\text{total}} = Q_{\text{data}} + Q_{\text{factory}} + Q_{\text{routing}}}$$

**Data Qubits:**

$$Q_{\text{data}} = n_{\text{logical}} \times Q_{\text{per logical}}$$

For distance-$d$ surface code:
$$Q_{\text{per logical}} = 2d^2 \quad (\text{data + syndrome ancillas})$$

**Magic State Factory:**

The factory produces magic states for T-gates. Factory size depends on:
- T-rate needed: $r_T = T_{\text{total}} / \tau_{\text{runtime}}$
- Distillation protocol: 15-to-1, 20-to-4, etc.
- Code distance of factory

**Factory Qubit Formula (15-to-1 distillation):**

$$Q_{\text{factory}} = N_{\text{factories}} \times Q_{\text{per factory}}$$

$$Q_{\text{per factory}} \approx 16 \times d_{\text{factory}}^2$$

where $N_{\text{factories}}$ is chosen to match T-rate:

$$N_{\text{factories}} = \left\lceil \frac{r_T \cdot \tau_{\text{distill}}}{\text{states per batch}} \right\rceil$$

**Routing Overhead:**

Lattice surgery requires physical space for operations:

$$Q_{\text{routing}} \approx \alpha \cdot Q_{\text{data}}$$

where $\alpha \approx 0.5$ to $2$ depending on connectivity requirements.

---

#### Part 4: Runtime Analysis

**Runtime Components:**

$$\boxed{\tau_{\text{total}} = \max(\tau_{\text{gate}}, \tau_{\text{T-limited}})}$$

**Gate-Limited Runtime:**

Sequential gate execution:
$$\tau_{\text{gate}} = \sum_{\text{gates } g} \tau_g$$

With parallelism (depth $D$):
$$\tau_{\text{gate}} = D \times \tau_{\text{cycle}}$$

**T-Limited Runtime:**

If T-gates are the bottleneck:
$$\tau_{\text{T-limited}} = \frac{T_{\text{total}}}{r_T}$$

where $r_T$ is the T-state production rate.

**Code Cycle Time:**

One syndrome extraction round:
$$\tau_{\text{cycle}} \approx 1 \mu s \quad \text{(superconducting)}$$
$$\tau_{\text{cycle}} \approx 100 \mu s \quad \text{(ion trap)}$$

**Distillation Time:**

15-to-1 distillation:
$$\tau_{\text{distill}} \approx 10d \times \tau_{\text{cycle}}$$

---

#### Part 5: Case Study - Shor's Algorithm

**Factoring an $n$-bit Number:**

**Algorithm Parameters:**
- Quantum: $2n$ qubits for QPE register + $n$ qubits for modular arithmetic
- Classical post-processing: continued fractions

**Resource Breakdown (Gidney-Ekera 2019):**

For $n$-bit RSA modulus using optimized circuits:

| Component | Formula |
|-----------|---------|
| Logical qubits | $\approx 2n + O(\log n)$ |
| Toffoli count | $\approx 0.5n^3$ (modular exponentiation) |
| T-count | $\approx 4 \times 0.5n^3 = 2n^3$ |
| Circuit depth | $\approx O(n^2)$ |

**Concrete Example: Factoring 2048-bit RSA**

Using $n = 2048$ bits:

**Logical Resources:**
- Logical qubits: $\approx 4,100$
- T-count: $\approx 2 \times (2048)^3 \approx 1.7 \times 10^{10}$

**Physical Resources (d = 27):**
- Physical qubits per logical: $2 \times 27^2 = 1,458$
- Data qubits: $4,100 \times 1,458 \approx 6$ million
- Factory qubits: $\approx 14$ million (for adequate T-rate)
- Total: **$\approx 20$ million physical qubits**

**Runtime:**
- T-rate needed: $\approx 10^6$ T-states/second
- Distillation time: $\approx 270 \mu s$
- Total runtime: **$\approx 8$ hours**

$$\boxed{\text{RSA-2048: } \sim 20M \text{ qubits}, \sim 8 \text{ hours}}$$

---

#### Part 6: Case Study - Quantum Simulation

**Hamiltonian Simulation Problem:**

Simulate $e^{-iHt}$ for Hamiltonian $H$ with error $\epsilon$.

**Resource Scaling:**

For $k$-local Hamiltonian on $n$ qubits with $L$ terms:

| Method | T-count |
|--------|---------|
| Product formula | $O(L^2 t^2 / \epsilon)$ |
| Truncated Taylor | $O(L t \log(Lt/\epsilon) / \log\log(Lt/\epsilon))$ |
| Qubitization | $O(L t / \epsilon^{0+})$ |
| Quantum signal processing | $O(Lt + \log(1/\epsilon))$ |

**Example: Simulating FeMoco (nitrogen fixation)**

FeMoco is a 54-electron active space relevant to nitrogen fixation.

**Using qubitization (Berry et al. 2019):**
- Logical qubits: $\approx 2,200$
- Toffoli count: $\approx 10^{10}$ per step
- Total T-count (with QPE): $\approx 10^{12}$

**Physical Resources:**
- Code distance: $d \approx 31$ (for $10^{-10}$ error)
- Physical qubits: $\approx 4$ million
- Runtime: $\approx$ days to weeks

---

#### Part 7: The Resource Estimation Formula

**Complete Framework:**

Given:
- Algorithm: $n_L$ logical qubits, $T$ T-gates, depth $D$
- Target error: $\epsilon_{\text{total}}$
- Physical error rate: $p$

**Step 1: Determine code distance**

$$p_L = \epsilon_{\text{total}} / (n_L \cdot D)$$

$$d = \text{smallest odd } d \text{ such that } 0.1(p/p_{\text{th}})^{(d+1)/2} < p_L$$

**Step 2: Calculate physical qubits**

$$Q = n_L \cdot 2d^2 + N_{\text{factory}} \cdot 16d^2 + \alpha \cdot n_L \cdot 2d^2$$

**Step 3: Calculate runtime**

$$\tau = \max\left(D \cdot d \cdot \tau_{\text{cycle}}, \frac{T}{r_T}\right)$$

where $r_T = N_{\text{factory}} \cdot \text{states/distillation} / \tau_{\text{distill}}$.

**Step 4: Iterate for consistency**

The code distance depends on runtime (longer = more error accumulation).
Iterate until convergent.

---

### Algorithm Design Implications

**Resource-Aware Algorithm Design:**

| Goal | Strategy |
|------|----------|
| Minimize qubits | Accept longer runtime, smaller factories |
| Minimize runtime | More parallel factories, lower depth |
| Minimize T-count | Clever decompositions, RUS circuits |
| Balance | Optimize product $Q \times \tau$ |

**Comparison Metrics:**

1. **Qubit-time product:** $Q \times \tau$ (analogous to space-time in classical)
2. **T-count/useful operation:** Efficiency per oracle call
3. **Break-even point:** When quantum beats classical

---

### Worked Examples

#### Example 1: Resource Estimation for Grover Search

**Problem:** Estimate resources for Grover search of $N = 2^{40}$ items.

**Solution:**

**Step 1: Algorithm parameters**
- Qubits: $n = 40$ (for superposition)
- Iterations: $\sqrt{N} = 2^{20} \approx 10^6$
- Operations per iteration: 2 oracle calls + 2 diffusion

**Step 2: Oracle T-count**
Assume oracle is a Toffoli-based function with 100 Toffolis:
- T-count per oracle: $100 \times 7 = 700$
- Per iteration: $2 \times 700 + 2 \times 40 \times 7 = 1,960$ T-gates

**Step 3: Total T-count**
$$T_{\text{total}} = 10^6 \times 1,960 \approx 2 \times 10^9$$

**Step 4: Physical resources (d = 17)**
- Logical qubits: $\approx 100$ (including ancillas)
- Per logical: $2 \times 17^2 = 578$
- Data qubits: $100 \times 578 = 57,800$
- Factory: $\approx 500,000$ qubits (for adequate T-rate)
- **Total: $\approx 600,000$ physical qubits**

**Step 5: Runtime**
- T-rate: $\approx 10^5$/s
- Runtime: $2 \times 10^9 / 10^5 = 2 \times 10^4$ s $\approx$ **6 hours**

---

#### Example 2: Comparing Synthesis Strategies

**Problem:** An algorithm has 1000 rotations $R_z(\theta_i)$. Compare:
(a) Synthesize each independently at $\epsilon = 10^{-8}$
(b) Combine where possible, then synthesize at $\epsilon = 10^{-6}$

**Solution:**

**(a) Independent synthesis:**
- T-count per rotation: $3\log_2(10^8) \approx 80$
- Total: $1000 \times 80 = 80,000$ T-gates

**(b) Combined synthesis:**
- Assume 200 unique sums after combination
- T-count per rotation: $3\log_2(10^6) \approx 60$
- Total: $200 \times 60 = 12,000$ T-gates

**Savings:** $80,000 - 12,000 = 68,000$ T-gates (**85% reduction**)

Note: This assumes rotations can be combined, which requires circuit analysis.

---

#### Example 3: Factory Sizing

**Problem:** An algorithm needs $10^{10}$ T-states over 1 hour runtime. How many 15-to-1 distillation factories are needed?

**Solution:**

**Step 1: Required T-rate**
$$r_T = \frac{10^{10}}{3600 \text{ s}} \approx 2.8 \times 10^6 \text{ T-states/s}$$

**Step 2: Factory output rate**
- 15-to-1 produces 1 state per distillation
- Distillation time: $\approx 100 \mu s$ (for $d \approx 10$)
- Rate per factory: $10^4$ states/s

**Step 3: Number of factories**
$$N = \frac{2.8 \times 10^6}{10^4} = 280 \text{ factories}$$

**Step 4: Factory qubits**
- Per factory: $\approx 16 \times 10^2 = 1,600$ qubits
- Total: $280 \times 1,600 \approx 450,000$ qubits

---

### Practice Problems

#### Level 1: Direct Application

**Problem 1.1:** For a distance-13 surface code with 50 logical qubits, calculate the data qubit count.

**Problem 1.2:** If a distillation factory produces 1 magic state per $50\mu s$, how many states can 10 factories produce in 1 second?

**Problem 1.3:** An algorithm has T-count $10^8$ and T-depth $10^4$. What is the minimum number of parallel T-gates?

---

#### Level 2: Intermediate

**Problem 2.1:** Derive the relationship between code distance and target logical error rate, given physical error rate $p = 10^{-3}$ and threshold $p_{\text{th}} = 10^{-2}$.

**Problem 2.2:** For Shor's algorithm on a 1024-bit number, estimate:
(a) Logical qubit count
(b) T-count (using $T \approx 2n^3$)
(c) Physical qubits for $d = 21$

**Problem 2.3:** A quantum simulation requires $10^{12}$ Toffolis. If each Toffoli uses 4 T-gates (measurement-based), and the algorithm runs for 24 hours, what T-rate is needed?

---

#### Level 3: Challenging

**Problem 3.1:** **(Full Resource Estimation)**
Complete a resource estimate for quantum chemistry simulation:
- System: 100 qubits, $10^6$ Trotter steps
- Each step: 1000 rotation gates, 500 CNOTs
- Target precision: $\epsilon = 10^{-8}$
- Physical error rate: $p = 10^{-3}$

Calculate: (a) T-count, (b) Code distance needed, (c) Physical qubits, (d) Runtime

**Problem 3.2:** **(Trade-off Analysis)**
For the same algorithm, analyze the trade-off:
- Configuration A: 1 million qubits, maximize speed
- Configuration B: 100,000 qubits, accept longer runtime

What are the runtimes for each?

**Problem 3.3:** **(Break-even Analysis)**
Classical simulation of an $n$-qubit system takes $O(2^n)$ time. Quantum simulation takes $O(n^3 \cdot T)$ T-gates. At what $n$ does quantum become favorable, assuming:
- Classical: $10^{15}$ ops/second
- Quantum: T-rate $10^6$/second

---

### Computational Lab

```python
"""
Day 866 Computational Lab: Resource Estimation Framework
Complete toolkit for fault-tolerant resource analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings

@dataclass
class HardwareParams:
    """Physical hardware parameters"""
    physical_error_rate: float = 1e-3
    threshold_error_rate: float = 1e-2
    code_cycle_time_us: float = 1.0  # microseconds
    qubits_per_logical_d2: int = 2  # multiplied by d^2

@dataclass
class DistillationParams:
    """Magic state distillation parameters"""
    protocol: str = "15-to-1"
    input_states: int = 15
    output_states: int = 1
    qubits_per_factory_d2: int = 16
    cycles_per_distillation_d: int = 10

@dataclass
class AlgorithmSpec:
    """Algorithm specification"""
    name: str
    logical_qubits: int
    t_count: int
    t_depth: int
    cnot_count: int = 0
    total_depth: int = 0
    target_error: float = 1e-6

class ResourceEstimator:
    """Complete fault-tolerant resource estimation"""

    def __init__(self,
                 hardware: HardwareParams = None,
                 distillation: DistillationParams = None):
        self.hw = hardware or HardwareParams()
        self.dist = distillation or DistillationParams()

    def required_code_distance(self, n_logical: int, depth: int,
                                target_error: float) -> int:
        """
        Calculate minimum code distance for target error rate.
        Uses: p_L ≈ 0.1 * (p/p_th)^((d+1)/2) per logical qubit per cycle
        """
        p = self.hw.physical_error_rate
        p_th = self.hw.threshold_error_rate

        # Error budget per qubit per cycle
        error_per_cycle = target_error / (n_logical * depth)

        # Solve: 0.1 * (p/p_th)^((d+1)/2) < error_per_cycle
        ratio = p / p_th
        if ratio >= 1:
            warnings.warn("Physical error rate above threshold!")
            return 99

        # (d+1)/2 > log(error_per_cycle / 0.1) / log(ratio)
        if error_per_cycle <= 0:
            return 99

        rhs = np.log(error_per_cycle / 0.1) / np.log(ratio)
        d_min = 2 * rhs - 1

        # Round up to odd integer
        d = max(3, int(np.ceil(d_min)))
        if d % 2 == 0:
            d += 1

        return d

    def physical_qubits(self, n_logical: int, d: int,
                        n_factories: int, routing_factor: float = 0.5) -> Dict:
        """Calculate physical qubit breakdown"""
        data = n_logical * self.hw.qubits_per_logical_d2 * d**2
        factory = n_factories * self.dist.qubits_per_factory_d2 * d**2
        routing = int(routing_factor * data)

        return {
            'data': data,
            'factory': factory,
            'routing': routing,
            'total': data + factory + routing
        }

    def factory_count(self, t_count: int, runtime_us: float, d: int) -> int:
        """Calculate number of factories needed for T-rate"""
        if runtime_us <= 0:
            return 1

        t_rate_needed = t_count / runtime_us  # T-states per microsecond

        # Each factory produces output_states per distillation
        distill_time = self.dist.cycles_per_distillation_d * d * self.hw.code_cycle_time_us
        rate_per_factory = self.dist.output_states / distill_time

        n_factories = int(np.ceil(t_rate_needed / rate_per_factory))
        return max(1, n_factories)

    def runtime(self, algorithm: AlgorithmSpec, d: int, n_factories: int) -> Dict:
        """Calculate runtime breakdown"""
        cycle_time = self.hw.code_cycle_time_us

        # Gate-limited (assuming T-depth dominates for non-Clifford)
        # Each T layer takes d cycles (magic state injection)
        gate_limited = algorithm.t_depth * d * cycle_time

        # Add CNOT overhead: each CNOT is 2d cycles (lattice surgery)
        gate_limited += 2 * d * cycle_time * (algorithm.cnot_count / max(1, algorithm.t_depth))

        # T-limited
        distill_time = self.dist.cycles_per_distillation_d * d * cycle_time
        t_rate = n_factories * self.dist.output_states / distill_time
        t_limited = algorithm.t_count / t_rate if t_rate > 0 else float('inf')

        total_us = max(gate_limited, t_limited)

        return {
            'gate_limited_us': gate_limited,
            't_limited_us': t_limited,
            'total_us': total_us,
            'total_ms': total_us / 1e3,
            'total_s': total_us / 1e6,
            'total_hours': total_us / (1e6 * 3600),
        }

    def estimate(self, algorithm: AlgorithmSpec,
                 max_iterations: int = 10) -> Dict:
        """
        Complete resource estimation with iterative convergence.
        Code distance depends on runtime, which depends on factories,
        which depend on code distance. Iterate to convergence.
        """
        # Initial estimate
        d = self.required_code_distance(
            algorithm.logical_qubits,
            max(algorithm.t_depth, 1),
            algorithm.target_error
        )

        # Initial runtime estimate (rough)
        runtime_estimate = algorithm.t_depth * d * self.hw.code_cycle_time_us * 10

        for iteration in range(max_iterations):
            # Calculate factories needed
            n_factories = self.factory_count(algorithm.t_count, runtime_estimate, d)

            # Calculate actual runtime
            runtime_info = self.runtime(algorithm, d, n_factories)
            new_runtime = runtime_info['total_us']

            # Recalculate code distance with new runtime
            effective_depth = new_runtime / (d * self.hw.code_cycle_time_us)
            new_d = self.required_code_distance(
                algorithm.logical_qubits,
                max(effective_depth, algorithm.t_depth),
                algorithm.target_error
            )

            # Check convergence
            if new_d == d and abs(new_runtime - runtime_estimate) / max(runtime_estimate, 1) < 0.01:
                break

            d = new_d
            runtime_estimate = new_runtime

        # Final calculations
        n_factories = self.factory_count(algorithm.t_count, runtime_estimate, d)
        qubits = self.physical_qubits(algorithm.logical_qubits, d, n_factories)
        runtime_info = self.runtime(algorithm, d, n_factories)

        return {
            'algorithm': algorithm.name,
            'logical_qubits': algorithm.logical_qubits,
            't_count': algorithm.t_count,
            't_depth': algorithm.t_depth,
            'code_distance': d,
            'n_factories': n_factories,
            'physical_qubits': qubits,
            'runtime': runtime_info,
            'iterations': iteration + 1,
        }


def shor_resources(n_bits: int) -> AlgorithmSpec:
    """Generate algorithm spec for Shor's algorithm on n-bit number"""
    # Based on Gidney-Ekera optimizations
    logical_qubits = 2 * n_bits + int(np.ceil(np.log2(n_bits)))
    t_count = int(2 * n_bits**3)  # Simplified estimate
    t_depth = int(n_bits**2)

    return AlgorithmSpec(
        name=f"Shor-{n_bits}bit",
        logical_qubits=logical_qubits,
        t_count=t_count,
        t_depth=t_depth,
        target_error=1e-3
    )

def grover_resources(n_bits: int, oracle_toffolis: int) -> AlgorithmSpec:
    """Generate algorithm spec for Grover's algorithm"""
    iterations = int(np.sqrt(2**n_bits))
    logical_qubits = n_bits + oracle_toffolis + 10  # work qubits
    t_per_iteration = oracle_toffolis * 7 * 2 + n_bits * 7 * 2  # Oracle + Diffusion
    t_count = iterations * t_per_iteration
    t_depth = iterations * (oracle_toffolis + n_bits)

    return AlgorithmSpec(
        name=f"Grover-{n_bits}bit",
        logical_qubits=logical_qubits,
        t_count=t_count,
        t_depth=t_depth,
        target_error=1e-3
    )

def simulation_resources(n_qubits: int, trotter_steps: int,
                         gates_per_step: int) -> AlgorithmSpec:
    """Generate algorithm spec for Hamiltonian simulation"""
    t_per_gate = 60  # Average for rotation synthesis
    t_count = trotter_steps * gates_per_step * t_per_gate
    t_depth = trotter_steps * gates_per_step // n_qubits

    return AlgorithmSpec(
        name=f"Simulation-{n_qubits}q",
        logical_qubits=n_qubits * 2,  # with ancillas
        t_count=t_count,
        t_depth=t_depth,
        target_error=1e-6
    )


# Main demonstrations
print("="*70)
print("Fault-Tolerant Resource Estimation Framework")
print("="*70)

estimator = ResourceEstimator()

# Example 1: Shor's algorithm for various bit sizes
print("\n--- Shor's Algorithm Resource Scaling ---")
print(f"{'Bits':<8} {'Logical Q':<12} {'T-count':<15} {'Distance':<10} {'Phys Q':<15} {'Runtime':<12}")
print("-" * 75)

shor_data = []
for n_bits in [512, 1024, 2048, 4096]:
    algo = shor_resources(n_bits)
    result = estimator.estimate(algo)

    print(f"{n_bits:<8} {result['logical_qubits']:<12} {result['t_count']:<15.2e} "
          f"{result['code_distance']:<10} {result['physical_qubits']['total']:<15,.0f} "
          f"{result['runtime']['total_hours']:<12.1f} hrs")

    shor_data.append((n_bits, result['physical_qubits']['total'],
                      result['runtime']['total_hours']))

# Example 2: Grover's algorithm
print("\n--- Grover's Algorithm Resource Scaling ---")
print(f"{'Bits':<8} {'Iterations':<12} {'T-count':<15} {'Phys Q':<15} {'Runtime':<12}")
print("-" * 65)

for n_bits in [20, 30, 40, 50]:
    if n_bits <= 40:  # Don't estimate huge cases
        algo = grover_resources(n_bits, oracle_toffolis=100)
        result = estimator.estimate(algo)

        iterations = int(np.sqrt(2**n_bits))
        runtime_str = f"{result['runtime']['total_s']:.1f} s" if result['runtime']['total_hours'] < 1 else f"{result['runtime']['total_hours']:.1f} hrs"

        print(f"{n_bits:<8} {iterations:<12,} {result['t_count']:<15.2e} "
              f"{result['physical_qubits']['total']:<15,.0f} {runtime_str}")

# Example 3: Quantum simulation
print("\n--- Quantum Simulation Resources ---")
print(f"{'Qubits':<10} {'Steps':<12} {'T-count':<15} {'Phys Q':<15} {'Runtime':<12}")
print("-" * 65)

for n_q in [50, 100, 200]:
    for steps in [1000, 10000]:
        algo = simulation_resources(n_q, trotter_steps=steps, gates_per_step=n_q*10)
        result = estimator.estimate(algo)

        print(f"{n_q:<10} {steps:<12,} {result['t_count']:<15.2e} "
              f"{result['physical_qubits']['total']:<15,.0f} "
              f"{result['runtime']['total_hours']:.2f} hrs")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Shor's algorithm scaling
ax1 = axes[0, 0]
bits = [512, 1024, 2048, 4096]
qubits = [shor_resources(b).t_count for b in bits]
ax1.semilogy(bits, qubits, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('RSA Key Size (bits)')
ax1.set_ylabel('T-count')
ax1.set_title("Shor's Algorithm: T-count Scaling")
ax1.grid(True, alpha=0.3)

# Plot 2: Physical qubit scaling with code distance
ax2 = axes[0, 1]
distances = np.arange(3, 31, 2)
n_logical = 100
qubits_vs_d = [2 * d**2 * n_logical for d in distances]
ax2.plot(distances, qubits_vs_d, 'g-', linewidth=2)
ax2.set_xlabel('Code Distance d')
ax2.set_ylabel('Data Qubits (100 logical)')
ax2.set_title('Physical Qubit Scaling with Code Distance')
ax2.grid(True, alpha=0.3)

# Plot 3: Error rate vs code distance
ax3 = axes[1, 0]
p = 1e-3
p_th = 1e-2
logical_errors = [0.1 * (p/p_th)**((d+1)/2) for d in distances]
ax3.semilogy(distances, logical_errors, 'r-', linewidth=2)
ax3.axhline(y=1e-10, color='k', linestyle='--', label='Target: 10^-10')
ax3.set_xlabel('Code Distance d')
ax3.set_ylabel('Logical Error Rate (per cycle)')
ax3.set_title('Error Suppression vs Code Distance')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Factory requirements
ax4 = axes[1, 1]
t_counts = [1e6, 1e8, 1e10, 1e12]
runtime_hours = 1
runtime_us = runtime_hours * 3600 * 1e6
d = 15

factories_needed = []
for tc in t_counts:
    n_f = estimator.factory_count(tc, runtime_us, d)
    factories_needed.append(n_f)

ax4.loglog(t_counts, factories_needed, 'mo-', linewidth=2, markersize=8)
ax4.set_xlabel('T-count')
ax4.set_ylabel('Number of Factories')
ax4.set_title(f'Factory Requirements (1 hour runtime, d={d})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resource_estimation.png', dpi=150, bbox_inches='tight')
plt.show()

# Detailed breakdown for RSA-2048
print("\n" + "="*70)
print("Detailed Resource Breakdown: RSA-2048 Factoring")
print("="*70)

algo = shor_resources(2048)
result = estimator.estimate(algo)

print(f"\nAlgorithm: {result['algorithm']}")
print(f"\nLogical Resources:")
print(f"  Logical qubits: {result['logical_qubits']:,}")
print(f"  T-count: {result['t_count']:,.0f}")
print(f"  T-depth: {result['t_depth']:,}")

print(f"\nPhysical Resources (d={result['code_distance']}):")
print(f"  Data qubits: {result['physical_qubits']['data']:,}")
print(f"  Factory qubits: {result['physical_qubits']['factory']:,}")
print(f"  Routing qubits: {result['physical_qubits']['routing']:,}")
print(f"  TOTAL: {result['physical_qubits']['total']:,}")

print(f"\nRuntime:")
print(f"  Gate-limited: {result['runtime']['gate_limited_us']/1e6:.1f} seconds")
print(f"  T-limited: {result['runtime']['t_limited_us']/1e6:.1f} seconds")
print(f"  Total: {result['runtime']['total_hours']:.1f} hours")

print(f"\nFactory Details:")
print(f"  Number of factories: {result['n_factories']}")
t_rate = result['t_count'] / result['runtime']['total_us'] * 1e6
print(f"  T-rate achieved: {t_rate:,.0f} T-states/second")

# Comparison table
print("\n" + "="*70)
print("Algorithm Comparison Summary")
print("="*70)

algorithms = [
    shor_resources(2048),
    grover_resources(40, 100),
    simulation_resources(100, 10000, 1000),
]

print(f"\n{'Algorithm':<25} {'Logical Q':<12} {'T-count':<12} {'Phys Q':<15} {'Runtime':<15}")
print("-" * 80)

for algo in algorithms:
    result = estimator.estimate(algo)
    runtime_str = f"{result['runtime']['total_hours']:.1f} hours"
    print(f"{algo.name:<25} {result['logical_qubits']:<12} {result['t_count']:<12.2e} "
          f"{result['physical_qubits']['total']:<15,.0f} {runtime_str}")

print("\n" + "="*70)
print("Resource Estimation Lab Complete")
print("="*70)
```

---

### Summary

#### Key Formulas

| Metric | Formula |
|--------|---------|
| Physical qubits | $Q = n_L \cdot 2d^2 + N_f \cdot 16d^2 + \text{routing}$ |
| Code distance | $0.1(p/p_{th})^{(d+1)/2} < \epsilon / (n_L \cdot D)$ |
| T-rate | $r_T = N_f \cdot 1 / (10d \cdot \tau_{cycle})$ |
| Runtime | $\tau = \max(D \cdot d \cdot \tau_{cycle}, T/r_T)$ |
| RSA-2048 estimate | ~20M qubits, ~8 hours |

#### Main Takeaways

1. **T-count is the dominant resource** determining factory size and runtime
2. **Code distance grows logarithmically** with target error reduction
3. **Qubit count scales as $d^2$** making high-distance codes expensive
4. **Runtime is often T-limited** not gate-depth limited
5. **Trade-offs exist** between qubits, time, and error rates

---

### Daily Checklist

- [ ] Understand the three pillars of FT resources
- [ ] Can calculate code distance from target error rate
- [ ] Can size magic state factories for given T-rate
- [ ] Applied framework to Shor's algorithm
- [ ] Completed computational lab with resource estimator
- [ ] Worked through at least 2 practice problems per level

---

### Preview: Day 867

Tomorrow is our **Computational Lab** day, where we implement complete Solovay-Kitaev decomposition, build T-count analyzers for sample algorithms, and create visualization tools for understanding gate synthesis and resource trade-offs. This hands-on session solidifies all concepts from the week.

---

*Day 866 provides the quantitative framework for resource estimation---tomorrow we build practical tools.*
