# Day 887: Parallelization & Pipelining

## Overview

**Day:** 887 of 1008
**Week:** 127 (Logical Gate Compilation)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Exploiting Parallelism and Magic State Production Pipelining

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Parallelism analysis and extraction |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Magic state pipelining and T-factories |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Identify** parallel execution opportunities in quantum circuits
2. **Compute** theoretical parallelism bounds for circuits
3. **Design** magic state production pipelines
4. **Schedule** T-factories to match gate consumption rates
5. **Optimize** space-time volume through parallelization
6. **Analyze** bottlenecks in fault-tolerant execution

---

## Circuit Parallelism

### Parallelism Fundamentals

**Definition:** Two operations are **parallel** if they share no data dependencies.

For quantum circuits:
- Operations on disjoint qubits can run in parallel
- Sequential gates on the same qubit must serialize

### Parallelism Metrics

**Work (W):** Total number of operations
$$W = \sum_{\text{gates } g} T_g$$

**Depth (D):** Length of critical path (minimum sequential time)
$$D = \text{longest path in dependency DAG}$$

**Parallelism (P):** Average parallelism
$$\boxed{P = \frac{W}{D}}$$

**Maximum Parallelism:** Maximum concurrent operations at any time
$$P_{\max} = \max_t |\{g : t \in [t_g^{\text{start}}, t_g^{\text{end}})\}|$$

### Amdahl's Law for Quantum Circuits

If fraction $f$ of work is inherently sequential:

$$\boxed{S_{\max} = \frac{1}{f + \frac{1-f}{P}}}$$

where $S_{\max}$ is the maximum speedup with $P$ parallel resources.

### Example: Parallelism Analysis

**Circuit:**
```
q0: ──H──●──────●──T──
         │      │
q1: ─────X──T───┼─────
                │
q2: ────────────X──H──
```

**Gates:**
1. H(q0): time 0-1
2. CNOT(q0,q1): time 1-2
3. T(q1): time 2-3
4. CNOT(q0,q2): time 2-3 (parallel with T!)
5. T(q0): time 3-4
6. H(q2): time 3-4 (parallel with T(q0)!)

**Analysis:**
- Work W = 6 gates
- Depth D = 4 time steps
- Parallelism P = 6/4 = 1.5

---

## Critical Path Optimization

### Identifying the Critical Path

The **critical path** is the longest dependency chain in the circuit DAG.

**Algorithm:**
```
CRITICAL_PATH(DAG):
    # Compute earliest finish time for each node
    for node in topological_order(DAG):
        node.eft = max(pred.eft for pred in predecessors) + node.duration

    # Find node with maximum eft
    critical_end = argmax(node.eft for node in DAG)

    # Trace back
    path = [critical_end]
    while path[-1] has predecessors:
        path.append(predecessor with max eft)

    return reverse(path)
```

### Critical Path Reduction Strategies

#### 1. Gate Decomposition Alternatives

Some gates have multiple decompositions with different depths:

| Gate | Decomposition A | Decomposition B |
|------|-----------------|-----------------|
| Toffoli | Depth 5, T-count 7 | Depth 3, T-count 4 (with ancilla) |
| CSWAP | Depth 6 | Depth 4 (with ancilla) |

#### 2. Ancilla Introduction

Adding ancilla qubits can break dependencies:

**Before (sequential):**
```
q: ──U₁──U₂──U₃──
```

**After (parallel with ancilla):**
```
q: ──U₁──────●─────────
             │
a: ──U₂prep──X──U₃──M──
```

#### 3. Commutation

Reorder commuting gates to reduce depth:

If $[U, V] = 0$:
```
Before: ──U──V──  →  After: ──V──U──
```

This flexibility can help scheduling.

---

## T-Gate Bottleneck

### The T-Gate Dominance

In fault-tolerant circuits, T gates dominate execution time:

| Gate | Cycles | Resources |
|------|--------|-----------|
| Clifford | O(1) | Minimal |
| T | O(distillation) | Magic state factory |

**Typical ratio:** T gate takes 100-1000× longer than Clifford gates.

### T-Gate Critical Path

The **T-depth** is the number of T gates on the critical path:

$$\boxed{T_{\text{depth}} = \max_{\text{paths}} |\{T \text{ gates on path}\}|}$$

**Key insight:** Reducing T-depth often matters more than reducing total T-count.

### T-Count vs. T-Depth Trade-offs

| Approach | T-count | T-depth | Notes |
|----------|---------|---------|-------|
| Sequential | Low | High | Standard decomposition |
| Parallel | Higher | Low | Uses more ancillas |
| Hybrid | Medium | Medium | Balance |

---

## Magic State Factories

### Factory Architecture

A **magic state factory** produces T-states through distillation:

```
┌─────────────────────────────────────┐
│           T-Factory                 │
│  ┌─────┐  ┌─────┐  ┌─────┐         │
│  │ L1  │→│ L1  │→│ L2  │→ |T⟩out  │
│  │dist │  │dist │  │dist │         │
│  └─────┘  └─────┘  └─────┘         │
│     ↑        ↑                      │
│   noisy T  noisy T                  │
└─────────────────────────────────────┘
```

### Factory Parameters

**Level-1 (15-to-1) factory:**
- Input: 15 noisy T states
- Output: 1 high-fidelity T state
- Error: $p_{\text{out}} \approx 35 p_{\text{in}}^3$
- Footprint: ~$15 \times d \times d$ physical qubits
- Time: ~$5d$ cycles

**Level-2 factory:**
- Input: 15 Level-1 outputs
- Output: 1 ultra-high-fidelity T state
- Error: $p_{\text{out}} \approx 35 (35 p_{\text{in}}^3)^3$

### Factory Throughput

**Single factory throughput:**
$$\boxed{R_{\text{factory}} = \frac{1}{T_{\text{distill}}}}$$

where $T_{\text{distill}}$ is the distillation time.

**Required factories for consumption rate $R_T$:**
$$\boxed{n_{\text{factories}} = \lceil R_T \cdot T_{\text{distill}} \rceil}$$

---

## Magic State Pipelining

### The Pipelining Concept

**Problem:** Distillation takes many cycles; gates must wait for magic states.

**Solution:** Start distillation early; pipeline production with consumption.

```
Time:    0   10   20   30   40   50   60   70   80
Factory: [==D1==][==D2==][==D3==][==D4==][==D5==]...
Gates:            T1    T2    T3    T4    T5
                   ↑     ↑     ↑     ↑     ↑
                  use   use   use   use   use
```

### Pipeline Scheduling

**Full pipeline scheduling:**
1. Analyze circuit to count T gates and their timing
2. Schedule factory to produce magic states just-in-time
3. Match production rate to consumption rate

**Key constraint:**
$$\text{Production rate} \geq \text{Peak consumption rate}$$

### Pipeline Depth

**Pipeline depth** = number of magic states in flight:

$$\boxed{d_{\text{pipeline}} = T_{\text{distill}} \times R_{\text{consumption}}}$$

**Example:** If distillation takes 100 cycles and we consume 1 T per 20 cycles:
$$d_{\text{pipeline}} = 100 \times (1/20) = 5 \text{ magic states}$$

Need to start distilling 5 magic states before they're needed.

### Lookahead Scheduling

```
LOOKAHEAD_SCHEDULE(circuit, factory):
    t_distill = factory.distillation_time

    for each T gate g at time t_g:
        # Schedule distillation to complete at t_g
        t_start = t_g - t_distill
        schedule_distillation(factory, t_start)

    # Handle conflicts (overlapping distillations)
    resolve_conflicts_with_additional_factories()
```

---

## Space-Time Volume Optimization

### The Space-Time Trade-off

**Space-time volume:**
$$\boxed{V = Q \times T}$$

where $Q$ is qubit count (space) and $T$ is execution time.

**Trade-off:** More qubits → faster execution (more parallelism)

### Minimum Space-Time Volume

For a circuit with work $W$:
$$\boxed{V_{\min} = W \times t_{\text{gate}}}$$

achievable only with infinite parallelism.

**Realistic bound:**
$$V = D \times Q_{\max}$$

where $D$ is depth and $Q_{\max}$ is peak qubit usage.

### Optimization Strategies

#### 1. Right-Size the Factory Count

**Under-provisioned:** Gates wait for magic states → increased time
**Over-provisioned:** Factories idle → wasted space

**Optimal:**
$$n_{\text{factories}}^* = \frac{n_T}{T / T_{\text{distill}}}$$

where $n_T$ is T-count and $T$ is target execution time.

#### 2. Adaptive Factory Allocation

Allocate factories dynamically based on circuit phase:

```
Phase 1 (T-heavy): 10 factories active
Phase 2 (Clifford): 2 factories (banking states)
Phase 3 (T-heavy): 10 factories active
```

#### 3. Magic State Banking

Produce magic states during Clifford-heavy phases and store for later use.

**Storage overhead:** Each stored T state requires $O(d^2)$ qubits.

---

## Multi-Core Architectures

### Modular Quantum Computing

Large algorithms may use multiple **quantum modules**:

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Module 1 │───│ Module 2 │───│ Module 3 │
│  (data)  │   │ (factory)│   │  (data)  │
└──────────┘   └──────────┘   └──────────┘
```

### Inter-Module Communication

Modules communicate via:
1. **Lattice surgery** between adjacent modules
2. **Teleportation** for distant modules
3. **Entanglement distribution** networks

**Communication cost:** $O(d)$ cycles per logical operation

### Load Balancing

Distribute work across modules to balance:
- T-factory load
- Data qubit utilization
- Communication overhead

---

## Worked Examples

### Example 1: Parallelism Calculation

**Problem:** Calculate the parallelism metrics for:

```
q0: ──T──●──────
         │
q1: ──T──X──T──
```

**Solution:**

**Gates:**
1. T(q0): depends on nothing
2. T(q1): depends on nothing
3. CNOT(q0,q1): depends on T(q0), T(q1)
4. T(q1): depends on CNOT

**Depth calculation:**
- T(q0), T(q1) can run in parallel: depth contribution = 1
- CNOT: depth += 1
- T(q1): depth += 1
- Total depth D = 3

**Work:** W = 4 (assuming each gate = 1 unit)

**Parallelism:** P = W/D = 4/3 ≈ 1.33

### Example 2: Factory Requirements

**Problem:** A circuit has 1000 T gates and executes over 50,000 cycles. Each distillation takes 500 cycles. How many factories are needed?

**Solution:**

**Consumption rate:**
$$R_T = \frac{1000}{50000} = 0.02 \text{ T/cycle}$$

**Factory throughput:**
$$R_{\text{factory}} = \frac{1}{500} = 0.002 \text{ T/cycle}$$

**Required factories:**
$$n = \frac{R_T}{R_{\text{factory}}} = \frac{0.02}{0.002} = 10 \text{ factories}$$

### Example 3: Pipeline Depth

**Problem:** With 10 factories each taking 500 cycles, and T gates occurring every 50 cycles, calculate the pipeline depth.

**Solution:**

**Aggregate throughput:**
$$R_{\text{total}} = 10 \times \frac{1}{500} = 0.02 \text{ T/cycle}$$

This matches consumption rate of 1/50 = 0.02 T/cycle. Good!

**Pipeline depth per factory:**
$$d_{\text{pipeline}} = \frac{T_{\text{distill}}}{T_{\text{interval}}} = \frac{500}{50 \times 10} = 1$$

With 10 factories sharing the load, each factory produces 1 T every 500 cycles, which is consumed every 500 cycles. Pipeline depth = 1 per factory, 10 total in flight.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** A circuit has 50 gates with depth 10. Calculate:
a) Average parallelism
b) If each gate takes 1 cycle, what is the execution time with unlimited parallelism? With 2-way parallelism?

**P1.2** A T-factory produces one magic state every 200 cycles. How many factories are needed to support a circuit with one T gate every 25 cycles?

**P1.3** Calculate the pipeline depth for a system with:
- Distillation time: 1000 cycles
- T-gate interval: 100 cycles

### Level 2: Intermediate

**P2.1** For the circuit:
```
q0: ──T──●──T──●──
         │     │
q1: ──T──X──T──X──
```
a) Identify all parallelism opportunities
b) Calculate T-depth
c) Design a schedule that minimizes depth

**P2.2** A computation requires 10,000 T gates over 2 seconds. Physical syndrome cycle time is 1 microsecond. Factory footprint is 10,000 physical qubits.
a) Calculate required number of factories
b) Estimate total factory qubit overhead
c) Compare to data qubit requirements (assume 100 logical qubits, distance 17)

**P2.3** Design a magic state banking strategy for a circuit with:
- Phase 1: 1000 Clifford gates, 100 T gates
- Phase 2: 100 Clifford gates, 500 T gates

### Level 3: Challenging

**P3.1** Prove that for any circuit, the space-time volume satisfies $V \geq W \cdot t_{\text{gate}}$.

**P3.2** Design an optimal factory allocation strategy for a circuit with time-varying T-gate density. Formulate as an optimization problem.

**P3.3** Analyze the parallelism limits of quantum error correction itself. How does syndrome extraction limit achievable parallelism?

---

## Computational Lab

```python
"""
Day 887: Parallelization & Pipelining
=====================================

Implementing parallelism analysis and magic state scheduling.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


# =============================================================================
# Circuit and Gate Representations
# =============================================================================

@dataclass
class Gate:
    """Represents a quantum gate."""
    gate_id: int
    gate_type: str
    qubits: Tuple[int, ...]
    is_t_gate: bool = False
    duration: int = 1  # in cycles

    # Scheduling info
    start_time: Optional[int] = None
    end_time: Optional[int] = None


class CircuitDAG:
    """
    Circuit represented as a DAG for parallelism analysis.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: Dict[int, Gate] = {}
        self.dependencies: Dict[int, Set[int]] = defaultdict(set)
        self.dependents: Dict[int, Set[int]] = defaultdict(set)
        self._next_id = 0

        # Track last gate on each qubit
        self._last_on_qubit: Dict[int, int] = {}

    def add_gate(self, gate_type: str, qubits: Tuple[int, ...],
                 duration: int = 1) -> int:
        """Add a gate and automatically compute dependencies."""
        gate_id = self._next_id
        self._next_id += 1

        is_t = gate_type in ['T', 'Tdg']

        gate = Gate(
            gate_id=gate_id,
            gate_type=gate_type,
            qubits=qubits,
            is_t_gate=is_t,
            duration=duration
        )

        self.gates[gate_id] = gate

        # Add dependencies based on qubit usage
        for q in qubits:
            if q in self._last_on_qubit:
                pred_id = self._last_on_qubit[q]
                self.dependencies[gate_id].add(pred_id)
                self.dependents[pred_id].add(gate_id)
            self._last_on_qubit[q] = gate_id

        return gate_id

    def t(self, qubit: int): return self.add_gate('T', (qubit,))
    def h(self, qubit: int): return self.add_gate('H', (qubit,))
    def cnot(self, c: int, t: int): return self.add_gate('CNOT', (c, t))

    def topological_order(self) -> List[int]:
        """Return gates in topological order."""
        in_degree = {gid: len(self.dependencies[gid]) for gid in self.gates}
        ready = [gid for gid, deg in in_degree.items() if deg == 0]
        result = []

        while ready:
            gid = ready.pop(0)
            result.append(gid)
            for succ in self.dependents[gid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        return result

    def asap_schedule(self) -> Dict[int, Tuple[int, int]]:
        """ASAP scheduling."""
        schedule = {}
        for gid in self.topological_order():
            gate = self.gates[gid]
            start = 0
            for pred_id in self.dependencies[gid]:
                start = max(start, schedule[pred_id][1])
            end = start + gate.duration
            schedule[gid] = (start, end)
            gate.start_time = start
            gate.end_time = end
        return schedule

    def depth(self) -> int:
        """Compute circuit depth (critical path length)."""
        schedule = self.asap_schedule()
        if not schedule:
            return 0
        return max(end for _, end in schedule.values())

    def work(self) -> int:
        """Total work (sum of gate durations)."""
        return sum(g.duration for g in self.gates.values())

    def parallelism(self) -> float:
        """Average parallelism."""
        d = self.depth()
        if d == 0:
            return 0
        return self.work() / d

    def t_count(self) -> int:
        """Count T gates."""
        return sum(1 for g in self.gates.values() if g.is_t_gate)

    def t_depth(self) -> int:
        """Compute T-depth (critical path counting only T gates)."""
        # DP: longest path of T gates to each node
        t_dist = {}

        for gid in self.topological_order():
            gate = self.gates[gid]
            pred_max = 0
            for pred_id in self.dependencies[gid]:
                pred_max = max(pred_max, t_dist[pred_id])

            if gate.is_t_gate:
                t_dist[gid] = pred_max + 1
            else:
                t_dist[gid] = pred_max

        return max(t_dist.values()) if t_dist else 0

    def parallelism_profile(self) -> Dict[int, int]:
        """Compute number of active gates at each time step."""
        schedule = self.asap_schedule()
        events = []
        for gid, (start, end) in schedule.items():
            events.append((start, 1))
            events.append((end, -1))

        events.sort()

        profile = {}
        current = 0
        prev_time = 0

        for time, delta in events:
            if time > prev_time and current > 0:
                for t in range(prev_time, time):
                    profile[t] = current
            current += delta
            prev_time = time

        return profile


# =============================================================================
# Magic State Factory
# =============================================================================

@dataclass
class MagicStateFactory:
    """
    Model of a T-factory.
    """
    factory_id: int
    distillation_time: int  # cycles per magic state
    footprint: int  # physical qubits

    # State
    busy_until: int = 0
    states_produced: int = 0


class FactoryScheduler:
    """
    Schedules magic state production to meet circuit demands.
    """

    def __init__(self, num_factories: int, distill_time: int,
                 footprint_per_factory: int = 10000):
        self.factories = [
            MagicStateFactory(
                factory_id=i,
                distillation_time=distill_time,
                footprint=footprint_per_factory
            )
            for i in range(num_factories)
        ]
        self.distill_time = distill_time

    def total_footprint(self) -> int:
        """Total physical qubits for all factories."""
        return sum(f.footprint for f in self.factories)

    def throughput(self) -> float:
        """Aggregate throughput (magic states per cycle)."""
        return len(self.factories) / self.distill_time

    def schedule_production(self, t_gate_times: List[int]) -> Dict[int, int]:
        """
        Schedule factory production for T gates at given times.

        Returns: dict mapping t_gate_time to production_start_time
        """
        # Sort T gate times
        demands = sorted(t_gate_times)

        # Priority queue of (busy_until, factory_id)
        factory_heap = [(f.busy_until, f.factory_id) for f in self.factories]
        heapq.heapify(factory_heap)

        schedule = {}

        for demand_time in demands:
            # Get next available factory
            available_time, fid = heapq.heappop(factory_heap)

            # When should distillation start?
            # Should complete by demand_time
            start_time = max(available_time, demand_time - self.distill_time)

            # If start_time + distill_time > demand_time, gate must wait
            completion_time = start_time + self.distill_time

            schedule[demand_time] = {
                'factory': fid,
                'start': start_time,
                'ready': completion_time,
                'wait': max(0, completion_time - demand_time)
            }

            # Update factory availability
            new_busy = start_time + self.distill_time
            heapq.heappush(factory_heap, (new_busy, fid))
            self.factories[fid].busy_until = new_busy
            self.factories[fid].states_produced += 1

        return schedule

    def analyze_schedule(self, schedule: Dict) -> Dict:
        """Analyze factory schedule."""
        if not schedule:
            return {}

        waits = [s['wait'] for s in schedule.values()]
        return {
            'total_t_gates': len(schedule),
            'total_wait_cycles': sum(waits),
            'max_wait': max(waits),
            'avg_wait': np.mean(waits),
            'gates_with_wait': sum(1 for w in waits if w > 0),
        }


# =============================================================================
# Pipeline Analysis
# =============================================================================

def compute_pipeline_depth(distill_time: int, t_interval: int,
                           num_factories: int) -> float:
    """
    Compute the effective pipeline depth.

    This is the number of magic states "in flight" at steady state.
    """
    # Time between magic states from all factories combined
    combined_interval = distill_time / num_factories

    # If T gates arrive faster than combined production, we have a bottleneck
    if t_interval < combined_interval:
        return float('inf')  # Bottleneck - can't keep up

    # Pipeline depth = distillation time / consumption interval
    return distill_time / t_interval


def optimal_factory_count(t_count: int, execution_time: int,
                          distill_time: int) -> int:
    """
    Compute optimal number of factories.
    """
    # Average T consumption rate
    rate = t_count / execution_time

    # Required factories
    n = np.ceil(rate * distill_time)

    return int(n)


# =============================================================================
# Space-Time Volume Analysis
# =============================================================================

@dataclass
class ResourceProfile:
    """Resource usage over time."""
    data_qubits: int
    factory_qubits: int
    ancilla_qubits: int
    time_steps: int

    def space_time_volume(self) -> int:
        """Compute space-time volume."""
        total_qubits = self.data_qubits + self.factory_qubits + self.ancilla_qubits
        return total_qubits * self.time_steps

    def __repr__(self):
        return (f"ResourceProfile(data={self.data_qubits}, "
                f"factory={self.factory_qubits}, "
                f"ancilla={self.ancilla_qubits}, "
                f"time={self.time_steps}, "
                f"volume={self.space_time_volume()})")


def estimate_resources(circuit: CircuitDAG,
                       code_distance: int,
                       factory_count: int,
                       factory_footprint: int = 10000) -> ResourceProfile:
    """
    Estimate resource requirements for a circuit.
    """
    # Data qubits: n_logical * d^2
    data_qubits = circuit.num_qubits * (code_distance ** 2)

    # Factory qubits
    factory_qubits = factory_count * factory_footprint

    # Ancilla qubits (rough estimate: 50% of data qubits)
    ancilla_qubits = data_qubits // 2

    # Time: circuit depth * d (syndrome cycles per logical operation)
    time_steps = circuit.depth() * code_distance

    return ResourceProfile(
        data_qubits=data_qubits,
        factory_qubits=factory_qubits,
        ancilla_qubits=ancilla_qubits,
        time_steps=time_steps
    )


# =============================================================================
# Visualization
# =============================================================================

def print_parallelism_profile(circuit: CircuitDAG):
    """Print parallelism over time."""
    profile = circuit.parallelism_profile()
    if not profile:
        print("Empty circuit")
        return

    max_time = max(profile.keys())
    max_parallel = max(profile.values())

    print(f"\nParallelism Profile (max = {max_parallel}):")
    print("-" * 50)

    for t in range(max_time + 1):
        p = profile.get(t, 0)
        bar = "█" * p + "░" * (max_parallel - p)
        print(f"t={t:3d}: {bar} ({p})")


def print_factory_schedule(schedule: Dict, max_show: int = 20):
    """Print factory schedule."""
    print("\nFactory Schedule:")
    print("-" * 60)
    print(f"{'T-gate Time':>12} {'Factory':>8} {'Start':>8} {'Ready':>8} {'Wait':>8}")
    print("-" * 60)

    for i, (demand_time, info) in enumerate(sorted(schedule.items())):
        if i >= max_show:
            print(f"... and {len(schedule) - max_show} more")
            break
        print(f"{demand_time:>12} {info['factory']:>8} "
              f"{info['start']:>8} {info['ready']:>8} {info['wait']:>8}")


# =============================================================================
# Demo
# =============================================================================

def demo_parallelism():
    """Demonstrate parallelism and pipelining analysis."""

    print("=" * 70)
    print("Day 887: Parallelization & Pipelining - Demonstration")
    print("=" * 70)

    # Example 1: Parallelism analysis
    print("\n1. Circuit Parallelism Analysis")
    print("-" * 40)

    circuit = CircuitDAG(3)
    circuit.t(0)
    circuit.t(1)  # Parallel with T(0)
    circuit.cnot(0, 1)  # Depends on both
    circuit.t(1)
    circuit.cnot(0, 2)
    circuit.h(2)

    print(f"Number of gates: {len(circuit.gates)}")
    print(f"Work: {circuit.work()}")
    print(f"Depth: {circuit.depth()}")
    print(f"Parallelism: {circuit.parallelism():.2f}")
    print(f"T-count: {circuit.t_count()}")
    print(f"T-depth: {circuit.t_depth()}")

    print_parallelism_profile(circuit)

    # Example 2: Factory throughput analysis
    print("\n2. T-Factory Throughput Analysis")
    print("-" * 40)

    for n_factories in [1, 2, 5, 10]:
        scheduler = FactoryScheduler(
            num_factories=n_factories,
            distill_time=500
        )
        print(f"  {n_factories} factories: throughput = {scheduler.throughput():.4f} T/cycle, "
              f"footprint = {scheduler.total_footprint()} qubits")

    # Example 3: Production scheduling
    print("\n3. Magic State Production Scheduling")
    print("-" * 40)

    # Simulate T gates occurring at specific times
    t_gate_times = list(range(100, 2100, 50))  # T gate every 50 cycles
    print(f"T gates: {len(t_gate_times)} gates, interval = 50 cycles")

    for n_factories in [2, 5, 10, 20]:
        scheduler = FactoryScheduler(n_factories, distill_time=500)
        schedule = scheduler.schedule_production(t_gate_times.copy())
        analysis = scheduler.analyze_schedule(schedule)

        print(f"\n  {n_factories} factories:")
        print(f"    Total wait: {analysis['total_wait_cycles']} cycles")
        print(f"    Max wait: {analysis['max_wait']} cycles")
        print(f"    Avg wait: {analysis['avg_wait']:.1f} cycles")
        print(f"    Gates with wait: {analysis['gates_with_wait']}/{analysis['total_t_gates']}")

    # Example 4: Pipeline depth
    print("\n4. Pipeline Depth Analysis")
    print("-" * 40)

    distill_time = 500
    for t_interval in [25, 50, 100, 200]:
        for n_factories in [5, 10]:
            depth = compute_pipeline_depth(distill_time, t_interval, n_factories)
            status = "OK" if depth != float('inf') else "BOTTLENECK"
            print(f"  Interval={t_interval}, factories={n_factories}: "
                  f"pipeline depth = {depth:.1f} ({status})")

    # Example 5: Resource estimation
    print("\n5. Resource Estimation")
    print("-" * 40)

    # Create a larger circuit
    big_circuit = CircuitDAG(50)
    for i in range(50):
        big_circuit.h(i)
    for i in range(49):
        big_circuit.cnot(i, i+1)
    for i in range(50):
        big_circuit.t(i)

    for distance in [11, 17, 23]:
        n_factories = optimal_factory_count(
            t_count=big_circuit.t_count(),
            execution_time=big_circuit.depth() * distance,
            distill_time=500
        )

        resources = estimate_resources(
            big_circuit,
            code_distance=distance,
            factory_count=n_factories
        )

        print(f"\n  Distance d={distance}:")
        print(f"    {resources}")
        print(f"    Optimal factories: {n_factories}")

    # Example 6: Detailed factory schedule
    print("\n6. Detailed Factory Schedule (sample)")
    print("-" * 40)

    scheduler = FactoryScheduler(5, distill_time=200)
    t_times = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    schedule = scheduler.schedule_production(t_times)
    print_factory_schedule(schedule)


if __name__ == "__main__":
    demo_parallelism()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Parallelism | $P = W/D$ (work/depth) |
| Factory throughput | $R = n_{\text{factories}}/T_{\text{distill}}$ |
| Required factories | $n = \lceil R_T \cdot T_{\text{distill}} \rceil$ |
| Pipeline depth | $d = T_{\text{distill}} / T_{\text{interval}}$ |
| Space-time volume | $V = Q \times T$ |
| Amdahl's law | $S = 1/(f + (1-f)/P)$ |

### Main Takeaways

1. **Parallelism** is limited by data dependencies in the circuit DAG
2. **T-gates dominate** execution time and create bottlenecks
3. **T-depth** often matters more than T-count for parallel execution
4. **Magic state factories** must be sized to match T-gate consumption rate
5. **Pipelining** enables continuous magic state production
6. **Space-time volume** captures the total resource cost

---

## Daily Checklist

- [ ] I can calculate work, depth, and parallelism for a circuit
- [ ] I understand the T-gate bottleneck problem
- [ ] I can determine required factory count for a given T-rate
- [ ] I know how to compute pipeline depth
- [ ] I understand space-time trade-offs
- [ ] I can apply Amdahl's law to quantum circuits

---

## Preview: Day 888

Tomorrow is **Computational Lab** day:

- Building a complete logical compiler
- Implementing T-count optimization
- Lattice surgery scheduling visualization
- Resource estimation for benchmark circuits
- End-to-end compilation pipeline

We'll put together everything from this week into a working compiler.
