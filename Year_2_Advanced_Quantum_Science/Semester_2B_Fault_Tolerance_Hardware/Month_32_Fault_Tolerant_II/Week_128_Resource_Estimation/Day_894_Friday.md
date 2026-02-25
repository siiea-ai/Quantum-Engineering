# Day 894: Code Choice Comparison

## Week 128, Day 5 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Comparing error-correcting code families |
| Afternoon | 2.5 hours | Problem solving: Code selection optimization |
| Evening | 2 hours | Computational lab: Multi-code resource estimator |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Compare surface code overhead** with alternative quantum error-correcting codes
2. **Analyze color code advantages** including transversal T-gates and reduced overhead
3. **Evaluate concatenated code trade-offs** for different error regimes
4. **Assess LDPC code potential** for future fault-tolerant systems
5. **Select optimal codes** for specific algorithm types and hardware constraints
6. **Build comparison tools** for multi-code resource estimation

---

## Core Content

### 1. The Code Selection Problem

#### Why Code Choice Matters

Different quantum error-correcting codes offer vastly different:
- **Physical qubit overhead**: Qubits per logical qubit
- **Threshold error rates**: Tolerance to physical errors
- **Native gate sets**: Which gates are "easy"
- **Connectivity requirements**: Hardware layout needs
- **Decoding complexity**: Classical processing overhead

#### The Main Contenders

| Code Family | Example | Native Gates | Overhead | Threshold |
|-------------|---------|--------------|----------|-----------|
| **Surface codes** | Rotated planar | Clifford + lattice surgery | $O(d^2)$ | ~1% |
| **Color codes** | 4.8.8 color | Clifford + transversal T | $O(d^2)$ | ~0.1% |
| **Concatenated** | [[7,1,3]] × k | Level-dependent | $O((\log n)^k)$ | ~$10^{-4}$ |
| **LDPC** | Hypergraph product | Varies | $O(1)$ to $O(n^α)$ | Unknown |

### 2. Surface Code: The Baseline

#### Strengths

1. **Highest threshold**: ~1% for depolarizing noise
2. **Local connectivity**: Only nearest-neighbor interactions
3. **Well-understood**: Extensive theoretical and experimental study
4. **Efficient decoding**: Minimum-weight perfect matching in polynomial time

#### Weaknesses

1. **No transversal non-Clifford**: Requires magic state distillation
2. **High overhead**: $2d^2$ physical qubits per logical
3. **2D layout only**: Limited by planar geometry

#### Overhead Formula

$$\boxed{Q_{surface} = n_{logical} \times 2d^2 + n_{factories} \times A_{factory}}$$

For RSA-2048 at $d = 27$:
$$Q = 6189 \times 2 \times 729 + 28 \times 150 \times 729 \approx 20 \text{ million}$$

### 3. Color Codes: Transversal T Advantage

#### The Key Advantage

Color codes on certain lattices support **transversal T gates**:

$$T_L = T^{\otimes n}$$

This eliminates the need for magic state distillation for T gates!

#### The 4.8.8 Color Code

```
       ●───●
      /│   │\
     ● │   │ ●
     │ │   │ │      Three-colorable lattice
     ● │   │ ●      Supports transversal T
      \│   │/
       ●───●
```

**Parameters:**
- Code distance $d$
- Physical qubits: $\frac{3d^2 + 1}{4}$ (approximately)
- Threshold: ~0.1% (lower than surface code)

#### Overhead Comparison

Without magic state distillation:

$$\boxed{Q_{color} \approx n_{logical} \times \frac{3d^2}{4} \times f_{routing}}$$

For the same RSA-2048:
- No factory overhead!
- But need higher $d$ due to lower threshold

If $p_{phys} = 10^{-3}$, color code needs $d_{color} \approx 1.5 \times d_{surface}$:

$$Q_{color} \approx 6189 \times \frac{3 \times 40^2}{4} \times 1.4 = 10.4 \text{ million}$$

**Result**: ~50% savings despite larger distance!

#### Caveats

1. **Lower threshold**: Requires better physical qubits
2. **More complex connectivity**: Non-planar for some implementations
3. **Harder decoding**: More complex than surface codes
4. **3D color codes**: Full transversal Clifford + T requires 3D

### 4. Concatenated Codes

#### The Concept

Build a hierarchy of codes:

$$\text{Level } k: [[n_1, k_1, d_1]]^{\otimes k}$$

**Example**: [[7,1,3]] Steane code concatenated:

| Level | Physical qubits | Distance |
|-------|-----------------|----------|
| 0 | 1 | 1 |
| 1 | 7 | 3 |
| 2 | 49 | 9 |
| 3 | 343 | 27 |
| k | $7^k$ | $3^k$ |

#### Error Suppression

Each level squares the error rate (for CSS codes):

$$p_{level\, k} = c \cdot p_{level\, k-1}^2 \approx c^{2^k - 1} \cdot p_{phys}^{2^k}$$

**Extremely rapid** error suppression with levels.

#### Overhead Formula

$$\boxed{Q_{concat} = n_{logical} \times n_1^k}$$

For $[[7,1,3]]$ with target $p_L = 10^{-15}$, $p_{phys} = 10^{-4}$:

$$k = \lceil \log_2(\log(p_L / c) / \log(p_{phys})) \rceil \approx 4$$

$$Q_{concat} = n_{logical} \times 7^4 = n_{logical} \times 2401$$

#### Comparison for RSA-2048

| Code | Qubits/Logical | Total (6189 logical) |
|------|----------------|----------------------|
| Surface (d=27) | ~3,000 | ~20M |
| Color (d=40) | ~1,700 | ~10M |
| Concat (k=4) | ~2,400 | ~15M |

**Trade-off**: Concatenated codes need lower physical error rates but can have comparable overhead.

### 5. LDPC Codes: The Future

#### Low-Density Parity-Check Codes

LDPC codes have **constant-weight** stabilizers:

$$\text{Each stabilizer involves } O(1) \text{ qubits (not } O(d) \text{)}$$

This enables potentially **constant overhead**:

$$Q_{LDPC} \propto n_{logical}$$

#### Hypergraph Product Construction

The hypergraph product of classical LDPC codes yields quantum LDPC:

$$[[n, k, d]] \text{ with } n = O(k^2), \quad d = O(\sqrt{k})$$

**Example**: Good LDPC codes achieve:
- $k = \Theta(n)$ (constant rate!)
- $d = \Theta(n^{1/2})$ or better

#### Challenges

1. **Non-local connectivity**: Qubits may need long-range interactions
2. **Unknown thresholds**: Less well-characterized than surface codes
3. **Complex decoding**: May require different algorithms
4. **No clear magic state approach**: T-gate implementation unclear

#### Potential Benefits

For large-scale computation:

$$\lim_{n \to \infty} \frac{Q_{LDPC}}{Q_{surface}} = O\left(\frac{1}{d_{surface}}\right) \to 0$$

LDPC codes could provide **orders of magnitude** improvement for large algorithms.

### 6. Code Comparison Framework

#### Normalized Overhead Metric

Define the **overhead ratio**:

$$\eta = \frac{Q_{physical}}{n_{logical}}$$

| Code | $\eta$ (approximate) | Scaling |
|------|---------------------|---------|
| Surface | $2d^2 + \text{factory}$ | $O(d^2)$ |
| Color | $0.75d^2$ | $O(d^2)$ |
| Concatenated | $n_1^k$ | $O((\log 1/p)^{\log_2 n_1})$ |
| LDPC | $O(1)$ to $O(d)$ | $O(1)$ (asymptotic) |

#### When to Use Each Code

| Scenario | Best Code | Rationale |
|----------|-----------|-----------|
| High physical error ($p > 10^{-3}$) | Surface | Highest threshold |
| Low physical error ($p < 10^{-5}$) | Concatenated or LDPC | Lower overhead |
| T-heavy algorithm | Color | No distillation |
| Near-term hardware | Surface | Simplest implementation |
| Long-term scaling | LDPC | Asymptotic efficiency |

### 7. Algorithm-Specific Optimization

#### T-Gate Fraction Analysis

Define the **T-fraction**:

$$f_T = \frac{T_{count}}{G_{total}}$$

| Algorithm | $f_T$ | Code Recommendation |
|-----------|-------|---------------------|
| Shor's (modular exp) | >90% | Color code |
| Grover's | ~50% | Surface or color |
| VQE | ~30% | Surface code |
| QAOA | <20% | Surface code |
| Chemistry (Trotter) | ~80% | Color code |

#### Break-Even Analysis

When does color code beat surface code?

**Color advantage** when:
$$Q_{color} < Q_{surface}$$

$$n \times \frac{3d_c^2}{4} < n \times 2d_s^2 + n_f \times A_f$$

Given $d_c \approx 1.5 d_s$ and solving:

$$\frac{3 \times 2.25 d_s^2}{4} < 2d_s^2 + n_f \times A_f / n$$

$$1.69 d_s^2 < 2d_s^2 + \text{factory overhead}$$

**Always true when factories dominate!**

For T-heavy algorithms, **color codes win**.

---

## Practical Benchmarks

### Head-to-Head Comparison: RSA-2048

| Metric | Surface | Color | Concatenated |
|--------|---------|-------|--------------|
| Logical qubits | 6,189 | 6,189 | 6,189 |
| Code distance | 27 | 40 | N/A (k=4) |
| Qubits/logical | ~3,200 | ~1,700 | ~2,400 |
| **Total qubits** | **20M** | **10M** | **15M** |
| Factories needed | 28 | 0 | 10* |
| Factory overhead | ~3M | 0 | ~1M |
| Threshold | 1% | 0.1% | 0.01% |

*Concatenated may still need factories for ancilla preparation.

### Cross-Code Runtime Comparison

| Code | T-gate time | RSA-2048 Runtime |
|------|-------------|------------------|
| Surface + distillation | ~300 cycles | 8 hours |
| Color (transversal T) | ~10 cycles | 15 minutes |
| Concatenated | ~100 cycles | 3 hours |

**Color codes are 30× faster for T-heavy algorithms!**

### Physical Error Rate Requirements

| Code | Required $p_{phys}$ for RSA-2048 |
|------|----------------------------------|
| Surface | $< 10^{-3}$ |
| Color | $< 10^{-4}$ |
| Concatenated | $< 10^{-4}$ |
| LDPC | Unknown (likely $< 10^{-3}$) |

---

## Worked Examples

### Example 1: Code Selection for Shor's Algorithm

**Problem:** Select the optimal code for factoring a 1024-bit number with $p_{phys} = 5 \times 10^{-4}$, targeting 1-hour runtime.

**Solution:**

Step 1: Algorithm parameters
- Logical qubits: ~3000
- T-count: ~$5 \times 10^9$
- T-fraction: >95% (very T-heavy)

Step 2: Evaluate surface code
- Distance needed: $d \approx 21$ (for $p_L < 10^{-12}$)
- Physical qubits: $3000 \times 2 \times 441 + 20 \times 150 \times 441 \approx 4M$
- Runtime: With 20 factories at 300 cycles/T: $5 \times 10^9 / (20/300) \approx 7.5 \times 10^{10}$ cycles ≈ 21 hours

Step 3: Evaluate color code
- Threshold check: $5 \times 10^{-4} < 10^{-3}$ ✓ (marginal)
- Distance needed: $d \approx 35$ (higher for lower threshold)
- Physical qubits: $3000 \times 0.75 \times 1225 \approx 2.8M$
- Runtime: Transversal T at 10 cycles: $5 \times 10^9 \times 10 \approx 5 \times 10^{10}$ cycles ≈ 14 hours

Step 4: Evaluate concatenated
- Below concatenated threshold (needs $p < 10^{-4}$) ✗

**Recommendation**: Color code
- Qubits: 2.8M (vs. 4M surface)
- Runtime: ~14 hours (vs. 21 hours surface)

$$\boxed{\text{Color code: 2.8M qubits, ~14 hours}}$$

---

### Example 2: Break-Even T-Count

**Problem:** At what T-count does surface code (with factories) become more expensive than color code for the same algorithm?

**Solution:**

Let $n$ = logical qubits, $d_s$ = surface distance, $d_c = 1.5 d_s$.

Surface qubit count:
$$Q_s = n \times 2d_s^2 + n_f \times 150 d_s^2$$

where $n_f \propto T_{count} / T_{target\_time}$.

Simplify: $n_f = \alpha \times T_{count}$ for some constant $\alpha$.

$$Q_s = n \times 2d_s^2 + \alpha \times T_{count} \times 150 d_s^2$$

Color qubit count (no factories):
$$Q_c = n \times 0.75 \times (1.5)^2 d_s^2 = n \times 1.69 d_s^2$$

Break-even: $Q_s = Q_c$
$$n \times 2d_s^2 + \alpha \times T_{count} \times 150 d_s^2 = n \times 1.69 d_s^2$$

$$\alpha \times T_{count} \times 150 = n \times (1.69 - 2) = -0.31n$$

This is negative, meaning **color code always wins** when the threshold requirement is met!

The break-even is actually about **threshold**: color wins whenever $p_{phys}$ is low enough.

$$\boxed{\text{Color code wins for all } T_{count} \text{ when } p_{phys} < p_{threshold}^{color}}$$

---

### Example 3: LDPC Scaling Advantage

**Problem:** At what scale ($n_{logical}$) does a hypothetical LDPC code with overhead 10 qubits/logical beat surface code with overhead $2d^2$ qubits/logical, assuming $d = 0.1 \times \log(n)$?

**Solution:**

Surface overhead:
$$Q_s = n \times 2d^2 = n \times 2 \times (0.1 \log n)^2 = 0.02 n (\log n)^2$$

LDPC overhead:
$$Q_{LDPC} = 10n$$

Break-even:
$$0.02 n (\log n)^2 = 10n$$
$$(\log n)^2 = 500$$
$$\log n = 22.4$$
$$n = e^{22.4} \approx 5 \times 10^9$$

**Result**: LDPC dominates for $n > 5 \times 10^9$ logical qubits.

For practical algorithms ($n < 10^6$), surface code may still be competitive. But for **very large-scale** computation (future quantum internet, etc.), LDPC wins.

$$\boxed{n > 5 \times 10^9 \text{ for LDPC advantage}}$$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the physical qubit count for:
- 500 logical qubits
- Code distance d = 15
- Compare: surface code (with 10 factories at 150d²) vs. color code (d' = 22, no factories)

**Problem 1.2:** What is the overhead ratio η for each code in Problem 1.1?

**Problem 1.3:** If a [[7,1,3]] concatenated code is used at level k=3, how many physical qubits are needed for 100 logical qubits?

### Level 2: Intermediate Analysis

**Problem 2.1:** An algorithm has T-fraction 0.6. Compare the expected runtime using:
- Surface code (300 cycles per T-gate via distillation)
- Color code (10 cycles per transversal T)
for T-count = $10^8$, assuming 1 μs cycles.

**Problem 2.2:** Derive the relationship between surface code threshold ($p_{th,s}$) and color code threshold ($p_{th,c}$) that makes color code preferable for a given algorithm.

**Problem 2.3:** A hypothetical LDPC code has parameters [[1000, 100, 10]]. Calculate:
- Rate k/n
- Physical qubits for 1000 logical qubits
- Compare to surface code at d=15

### Level 3: Challenging Problems

**Problem 3.1:** **Hybrid Code Design**
Design a hybrid approach using:
- Surface code for data storage
- Color code patches for T-gate execution
Analyze the routing overhead and potential benefits.

**Problem 3.2:** **Threshold Optimization**
Given a target logical error rate $\epsilon$ and physical error rate $p$, find the code family (surface, color, concatenated) that minimizes total physical qubits. Express as a decision tree based on $p$ and $\epsilon$.

**Problem 3.3:** **Future Scaling**
Assuming Moore's law for quantum error rates (halving every 2 years from $10^{-3}$ today), project when:
a) Color codes become universally preferable
b) Simple concatenated codes become viable
c) LDPC codes dominate
Start from 2024 baseline.

---

## Computational Lab

### Multi-Code Resource Estimator

```python
"""
Day 894: Code Choice Comparison Tool
Compare resource requirements across quantum error-correcting codes.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum

class CodeFamily(Enum):
    SURFACE = "Surface Code"
    COLOR = "Color Code"
    CONCATENATED = "Concatenated"
    LDPC = "LDPC"


@dataclass
class CodeParameters:
    """Base parameters for any quantum error-correcting code."""
    name: str
    family: CodeFamily
    threshold: float
    description: str = ""


class QuantumCode(ABC):
    """Abstract base class for quantum error-correcting codes."""

    @abstractmethod
    def physical_qubits(self, n_logical: int, target_error: float,
                        physical_error: float) -> int:
        """Calculate physical qubit requirement."""
        pass

    @abstractmethod
    def t_gate_time(self, code_distance: int) -> int:
        """Time (in cycles) to execute a T gate."""
        pass

    @abstractmethod
    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, physical_error: float) -> int:
        """Calculate required code distance."""
        pass


class SurfaceCode(QuantumCode):
    """Rotated surface code implementation."""

    def __init__(self, factory_area_coeff: float = 150,
                 routing_overhead: float = 0.4):
        self.params = CodeParameters(
            name="Rotated Surface Code",
            family=CodeFamily.SURFACE,
            threshold=0.01,
            description="Planar topological code with high threshold"
        )
        self.factory_area_coeff = factory_area_coeff
        self.routing_overhead = routing_overhead

    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, physical_error: float) -> int:
        """Calculate required code distance."""
        if physical_error >= self.params.threshold:
            return 999  # Invalid regime

        ratio = physical_error / self.params.threshold
        p_logical_req = target_error / (n_logical * depth)

        # p_L = 0.1 * ratio^((d+1)/2)
        log_term = np.log(p_logical_req / 0.1)
        log_ratio = np.log(ratio)

        d_float = 2 * log_term / log_ratio - 1
        d = int(np.ceil(d_float))
        if d % 2 == 0:
            d += 1
        return max(3, d)

    def physical_qubits(self, n_logical: int, target_error: float,
                        physical_error: float, depth: int = 10**8,
                        t_count: int = 10**9, n_factories: int = None) -> int:
        """Calculate total physical qubits including factories."""
        d = self.required_distance(n_logical, depth, target_error, physical_error)

        # Logical qubit area
        logical_area = n_logical * 2 * d**2

        # Routing overhead
        routing_area = self.routing_overhead * logical_area

        # Factory area
        if n_factories is None:
            # Estimate factories needed
            n_factories = max(1, t_count // (10**9))  # Rough heuristic

        factory_area = n_factories * self.factory_area_coeff * d**2

        return int(logical_area + routing_area + factory_area)

    def t_gate_time(self, code_distance: int, n_levels: int = 2) -> int:
        """Time for T gate via distillation."""
        return 8 * code_distance * n_levels


class ColorCode(QuantumCode):
    """4.8.8 color code with transversal T."""

    def __init__(self, routing_overhead: float = 0.5):
        self.params = CodeParameters(
            name="4.8.8 Color Code",
            family=CodeFamily.COLOR,
            threshold=0.001,  # Lower threshold than surface
            description="Topological code with transversal T gate"
        )
        self.routing_overhead = routing_overhead

    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, physical_error: float) -> int:
        """Calculate required code distance (higher than surface for same error)."""
        if physical_error >= self.params.threshold:
            return 999

        ratio = physical_error / self.params.threshold
        p_logical_req = target_error / (n_logical * depth)

        # Color code has steeper suppression but lower threshold
        # Approximate: need ~1.5x distance of surface code
        log_term = np.log(p_logical_req / 0.05)
        log_ratio = np.log(ratio)

        d_float = 2 * log_term / log_ratio - 1
        d = int(np.ceil(d_float * 1.3))  # Factor for lower threshold
        if d % 2 == 0:
            d += 1
        return max(3, d)

    def physical_qubits(self, n_logical: int, target_error: float,
                        physical_error: float, depth: int = 10**8,
                        **kwargs) -> int:
        """Calculate physical qubits (no factory overhead)."""
        d = self.required_distance(n_logical, depth, target_error, physical_error)

        # Color code area: approximately 3d²/4 per logical
        logical_area = n_logical * 0.75 * d**2

        # Routing overhead
        routing_area = self.routing_overhead * logical_area

        # No factory overhead for T gates!
        return int(logical_area + routing_area)

    def t_gate_time(self, code_distance: int, **kwargs) -> int:
        """Transversal T gate is fast."""
        return 10  # Just a few cycles for transversal gate


class ConcatenatedCode(QuantumCode):
    """Concatenated [[7,1,3]] Steane code."""

    def __init__(self, base_n: int = 7, base_d: int = 3):
        self.params = CodeParameters(
            name="Concatenated [[7,1,3]]",
            family=CodeFamily.CONCATENATED,
            threshold=0.0001,  # Much lower threshold
            description="Hierarchical code with exponential error suppression"
        )
        self.base_n = base_n
        self.base_d = base_d

    def required_levels(self, target_error: float, physical_error: float) -> int:
        """Calculate number of concatenation levels needed."""
        if physical_error >= self.params.threshold:
            return 999

        # Error at level k: p_k ~ c^(2^k - 1) * p^(2^k)
        # Need p_k < target_error
        c = 1.0  # Constant factor

        current_error = physical_error
        levels = 0
        while current_error > target_error and levels < 10:
            current_error = c * current_error ** 2
            levels += 1

        return levels

    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, physical_error: float) -> int:
        """Distance for concatenated code is 3^k."""
        k = self.required_levels(target_error / (n_logical * depth), physical_error)
        return self.base_d ** k

    def physical_qubits(self, n_logical: int, target_error: float,
                        physical_error: float, depth: int = 10**8,
                        **kwargs) -> int:
        """Calculate physical qubits for k concatenation levels."""
        k = self.required_levels(target_error / (n_logical * depth), physical_error)
        qubits_per_logical = self.base_n ** k
        return int(n_logical * qubits_per_logical)

    def t_gate_time(self, code_distance: int, **kwargs) -> int:
        """T gate time depends on levels (transversal at each level)."""
        # Approximate: log_3(d) levels, each taking ~50 cycles
        levels = int(np.log(code_distance) / np.log(self.base_d))
        return 50 * levels


class LDPCCode(QuantumCode):
    """Hypothetical good LDPC code for future comparison."""

    def __init__(self, overhead_constant: float = 10):
        self.params = CodeParameters(
            name="Hypergraph Product LDPC",
            family=CodeFamily.LDPC,
            threshold=0.005,  # Estimated
            description="Asymptotically efficient quantum LDPC code"
        )
        self.overhead_constant = overhead_constant

    def required_distance(self, n_logical: int, depth: int,
                          target_error: float, physical_error: float) -> int:
        """LDPC distance scaling."""
        if physical_error >= self.params.threshold:
            return 999

        # For good LDPC: d ~ sqrt(n) where n = physical qubits
        # Very rough approximation
        ratio = physical_error / self.params.threshold
        d = int(10 * np.sqrt(-np.log(target_error / (n_logical * depth))))
        return max(5, d)

    def physical_qubits(self, n_logical: int, target_error: float,
                        physical_error: float, **kwargs) -> int:
        """Near-constant overhead in asymptotic limit."""
        # Constant overhead: just multiply by overhead constant
        return int(n_logical * self.overhead_constant)

    def t_gate_time(self, code_distance: int, **kwargs) -> int:
        """Estimated T gate time (unclear for LDPC)."""
        return 100  # Placeholder


class CodeComparator:
    """Compare resource requirements across code families."""

    def __init__(self, physical_error: float = 1e-3):
        self.physical_error = physical_error
        self.codes = {
            CodeFamily.SURFACE: SurfaceCode(),
            CodeFamily.COLOR: ColorCode(),
            CodeFamily.CONCATENATED: ConcatenatedCode(),
            CodeFamily.LDPC: LDPCCode()
        }

    def compare_qubits(self, n_logical: int, target_error: float = 1e-10,
                       depth: int = 10**8, t_count: int = 10**9) -> Dict:
        """Compare physical qubit requirements across codes."""
        results = {}

        for family, code in self.codes.items():
            try:
                qubits = code.physical_qubits(
                    n_logical, target_error, self.physical_error,
                    depth=depth, t_count=t_count
                )
                distance = code.required_distance(
                    n_logical, depth, target_error, self.physical_error
                )
                t_time = code.t_gate_time(distance)

                results[family] = {
                    'qubits': qubits,
                    'distance': distance,
                    't_gate_time': t_time,
                    'overhead_ratio': qubits / n_logical,
                    'threshold': code.params.threshold,
                    'viable': self.physical_error < code.params.threshold
                }
            except:
                results[family] = {'viable': False, 'error': 'Calculation failed'}

        return results

    def compare_runtime(self, t_count: int, results: Dict) -> Dict:
        """Add runtime estimates to comparison."""
        for family, data in results.items():
            if data.get('viable', False):
                t_time = data['t_gate_time']
                # Assume 1 μs cycles, 100 parallel T operations
                runtime_cycles = t_count * t_time / 100
                runtime_hours = runtime_cycles * 1e-6 / 3600
                data['runtime_hours'] = runtime_hours
        return results

    def print_comparison(self, results: Dict, n_logical: int):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print(f"CODE COMPARISON: {n_logical} logical qubits")
        print("="*80)

        header = f"{'Code':<25} {'Qubits':>12} {'Distance':>10} {'T-time':>10} {'Viable':>8}"
        print(header)
        print("-"*80)

        for family, data in results.items():
            if data.get('viable', False):
                row = (f"{family.value:<25} {data['qubits']:>12,} "
                       f"{data['distance']:>10} {data['t_gate_time']:>10} "
                       f"{'Yes':>8}")
            else:
                row = f"{family.value:<25} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'No':>8}"
            print(row)

        print("="*80)


def plot_code_comparison():
    """Visualize code comparison across different scales."""
    logical_qubits = [10, 50, 100, 500, 1000, 5000, 10000]

    comparator = CodeComparator(physical_error=5e-4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Physical qubits comparison
    for family in [CodeFamily.SURFACE, CodeFamily.COLOR, CodeFamily.CONCATENATED]:
        qubits = []
        for n in logical_qubits:
            results = comparator.compare_qubits(n)
            if results[family].get('viable', False):
                qubits.append(results[family]['qubits'])
            else:
                qubits.append(np.nan)

        axes[0].loglog(logical_qubits, qubits, '-o', label=family.value,
                       linewidth=2, markersize=8)

    axes[0].set_xlabel('Logical Qubits', fontsize=12)
    axes[0].set_ylabel('Physical Qubits', fontsize=12)
    axes[0].set_title('Physical Qubit Scaling by Code', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')

    # Overhead ratio comparison
    for family in [CodeFamily.SURFACE, CodeFamily.COLOR, CodeFamily.CONCATENATED]:
        ratios = []
        for n in logical_qubits:
            results = comparator.compare_qubits(n)
            if results[family].get('viable', False):
                ratios.append(results[family]['overhead_ratio'])
            else:
                ratios.append(np.nan)

        axes[1].semilogx(logical_qubits, ratios, '-o', label=family.value,
                         linewidth=2, markersize=8)

    axes[1].set_xlabel('Logical Qubits', fontsize=12)
    axes[1].set_ylabel('Overhead Ratio (physical/logical)', fontsize=12)
    axes[1].set_title('Overhead Ratio by Code', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('code_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_sensitivity():
    """Show how code choice depends on physical error rate."""
    error_rates = np.logspace(-5, -2, 20)

    n_logical = 1000
    target_error = 1e-10

    surface_qubits = []
    color_qubits = []
    concat_qubits = []

    for p in error_rates:
        comparator = CodeComparator(physical_error=p)
        results = comparator.compare_qubits(n_logical, target_error)

        surface_qubits.append(
            results[CodeFamily.SURFACE]['qubits']
            if results[CodeFamily.SURFACE].get('viable') else np.nan
        )
        color_qubits.append(
            results[CodeFamily.COLOR]['qubits']
            if results[CodeFamily.COLOR].get('viable') else np.nan
        )
        concat_qubits.append(
            results[CodeFamily.CONCATENATED]['qubits']
            if results[CodeFamily.CONCATENATED].get('viable') else np.nan
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(error_rates, surface_qubits, 'b-o', label='Surface', linewidth=2)
    ax.loglog(error_rates, color_qubits, 'g-s', label='Color', linewidth=2)
    ax.loglog(error_rates, concat_qubits, 'r-^', label='Concatenated', linewidth=2)

    ax.set_xlabel('Physical Error Rate', fontsize=12)
    ax.set_ylabel('Physical Qubits', fontsize=12)
    ax.set_title(f'Physical Qubits vs. Error Rate ({n_logical} logical qubits)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Mark thresholds
    ax.axvline(x=0.01, color='blue', linestyle='--', alpha=0.5, label='Surface threshold')
    ax.axvline(x=0.001, color='green', linestyle='--', alpha=0.5, label='Color threshold')
    ax.axvline(x=0.0001, color='red', linestyle='--', alpha=0.5, label='Concat threshold')

    plt.tight_layout()
    plt.savefig('threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()


def benchmark_rsa2048():
    """Detailed comparison for RSA-2048."""
    print("\n" + "="*80)
    print("RSA-2048 FACTORING: CODE COMPARISON")
    print("="*80)

    n_logical = 6189
    t_count = int(2e10)
    target_error = 1e-12
    depth = int(2e10)

    for p_phys in [1e-3, 5e-4, 1e-4]:
        print(f"\nPhysical error rate: {p_phys:.0e}")
        print("-"*60)

        comparator = CodeComparator(physical_error=p_phys)
        results = comparator.compare_qubits(n_logical, target_error, depth, t_count)
        results = comparator.compare_runtime(t_count, results)

        for family, data in results.items():
            if data.get('viable', False):
                print(f"{family.value:20s}: {data['qubits']/1e6:.1f}M qubits, "
                      f"d={data['distance']}, "
                      f"~{data.get('runtime_hours', 'N/A'):.1f} hours")
            else:
                print(f"{family.value:20s}: Not viable (below threshold)")


# Main demonstration
if __name__ == "__main__":
    print("Code Choice Comparison Tool - Day 894")
    print("="*50)

    # Create comparator
    comparator = CodeComparator(physical_error=5e-4)

    # Compare for various scales
    for n in [100, 1000, 6189]:
        results = comparator.compare_qubits(n, target_error=1e-12, t_count=int(n * 1e6))
        comparator.print_comparison(results, n)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_code_comparison()
    plot_threshold_sensitivity()

    # RSA-2048 benchmark
    benchmark_rsa2048()

    print("\nCode comparison complete!")
```

---

## Summary

### Code Family Comparison

| Code | Threshold | Overhead | T-Gate | Best For |
|------|-----------|----------|--------|----------|
| Surface | ~1% | $2d^2$ + factories | Slow (distill) | High error rates |
| Color | ~0.1% | $0.75d^2$ | Fast (transversal) | T-heavy algorithms |
| Concatenated | ~0.01% | $7^k$ | Medium | Low error rates |
| LDPC | ~0.5% (est.) | $O(1)$ | Unknown | Future scaling |

### Selection Guidelines

1. **Physical error rate > 0.1%**: Only surface code viable
2. **Physical error rate 0.01-0.1%**: Color code for T-heavy, surface otherwise
3. **Physical error rate < 0.01%**: Consider concatenated codes
4. **Very large scale**: LDPC codes (future)

### Key Insight

$$\text{Optimal code} = f(\text{physical error}, \text{T-fraction}, \text{scale}, \text{hardware})$$

No single code is universally best—the choice depends on the specific context.

---

## Daily Checklist

- [ ] I can compare surface, color, and concatenated code overheads
- [ ] I understand threshold requirements for each code family
- [ ] I know when color codes are advantageous (T-heavy algorithms)
- [ ] I can calculate overhead ratios for each code
- [ ] I understand LDPC codes' potential for future scaling
- [ ] I can select optimal codes based on algorithm characteristics
- [ ] I can use the code comparison tool

---

## Preview: Day 895

Tomorrow is **Computational Lab Saturday**:

- Build a complete resource estimation toolkit
- Integrate qubit counting, space-time analysis, factory design
- Analyze sample algorithms end-to-end
- Generate comprehensive reports and visualizations
- Create reusable estimation framework

We'll synthesize all the components from this week into a unified resource estimation platform.

---

*Day 894 of 2184 | Week 128 of 312 | Month 32 of 72*

*"The right code is not the most powerful—it's the one that matches your constraints."*
