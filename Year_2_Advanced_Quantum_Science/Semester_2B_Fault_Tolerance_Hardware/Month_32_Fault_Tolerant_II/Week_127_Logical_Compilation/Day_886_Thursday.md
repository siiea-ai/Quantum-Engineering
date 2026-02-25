# Day 886: Lattice Surgery Scheduling

## Overview

**Day:** 886 of 1008
**Week:** 127 (Logical Gate Compilation)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Translating Logical Operations to Lattice Surgery Instructions

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Surgery primitives and instruction mapping |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Scheduling algorithms and optimization |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Map** logical Clifford gates to lattice surgery primitives
2. **Construct** dependency graphs for surgery operations
3. **Schedule** operations respecting spatial and temporal constraints
4. **Manage** ancilla patch allocation and deallocation
5. **Optimize** surgery schedules for minimum time or space
6. **Analyze** the time cost of complex logical circuits

---

## Lattice Surgery Primitives

### Review: Surface Code Patches

A **logical qubit** is encoded in a surface code **patch**:

```
┌───────────────────┐
│  ● ─ ● ─ ● ─ ● ─ ●│  ← Rough boundary (Z logical)
│  │   │   │   │   ││
│  ● ─ ● ─ ● ─ ● ─ ●│
│  │   │   │   │   ││
│  ● ─ ● ─ ● ─ ● ─ ●│
│  │   │   │   │   ││
│  ● ─ ● ─ ● ─ ● ─ ●│
└───────────────────┘
        ↑
    Smooth boundary (X logical)
```

**Patch size:** $d \times d$ physical qubits for distance $d$

### Lattice Surgery Operations

#### 1. Merge (Multi-Patch Measurement)

**XX Merge:** Measures $X_L \otimes X_L$ between two patches

```
Before:           After merge:
┌─────┐ ┌─────┐   ┌─────────────┐
│  A  │ │  B  │ → │    A ⊗ B    │
└─────┘ └─────┘   └─────────────┘
```

**ZZ Merge:** Measures $Z_L \otimes Z_L$ between two patches

**Time cost:** $d$ syndrome measurement rounds

#### 2. Split (Patch Division)

Reverse of merge: divides one patch into two.

```
Before:           After split:
┌─────────────┐   ┌─────┐ ┌─────┐
│    A ⊗ B    │ → │  A  │ │  B  │
└─────────────┘   └─────┘ └─────┘
```

**Time cost:** $d$ syndrome measurement rounds

#### 3. Patch Rotation/Twist

Rotates the logical operator orientation (changes which boundary is rough/smooth).

**Time cost:** $O(d)$ rounds

### Surgery Timing

$$\boxed{T_{\text{surgery}} = O(d) \text{ syndrome cycles}}$$

Each syndrome cycle takes time $t_{\text{cycle}}$ (physical measurement + classical processing).

**Total time per operation:**
$$T_{\text{op}} = d \cdot t_{\text{cycle}}$$

---

## Mapping Logical Gates to Surgery

### Clifford Gates via Surgery

| Logical Gate | Surgery Implementation | Time ($d$ cycles) |
|--------------|----------------------|-------------------|
| Pauli X, Y, Z | Pauli frame update | 0 |
| H | Twist/rotation | $d$ |
| S | Pauli frame + corner | $d$ |
| CNOT | XX and ZZ measurements | $2d$ |

### CNOT via Lattice Surgery

The key operation: **CNOT** between logical qubits A and B.

**Protocol:**
1. Prepare ancilla patch in $|+\rangle_L$
2. Merge A with ancilla (measure $Z_A Z_{\text{anc}}$)
3. Split, then merge ancilla with B (measure $X_{\text{anc}} X_B$)
4. Measure ancilla in X basis
5. Apply corrections based on measurements

```
     A: ──●──────────────────
          │
   anc: ──⊕── ... ──●── M_X
                    │
     B: ────────────⊕────────
```

**Surgery sequence:**
```
Time 0-d:    Prepare |+⟩_anc
Time d-2d:   ZZ merge (A, anc)
Time 2d-3d:  Split
Time 3d-4d:  XX merge (anc, B)
Time 4d-5d:  Split + measure anc
```

**Total time:** ~$5d$ cycles

### T Gate via Magic State Injection

T gates require **magic state injection**:

1. Consume magic state $|T\rangle = T|+\rangle$
2. Perform CNOT between data and magic state
3. Measure magic state in X basis
4. Apply S correction if measurement is 1

**Surgery sequence:**
```
Time 0:      Magic state |T⟩ available
Time 0-2d:   CNOT(data, magic) via surgery
Time 2d:     Measure magic, apply correction
```

**Total time:** ~$2d$ cycles (assuming magic state ready)

---

## Instruction Dependency Graph

### Building the Dependency Graph

A **surgery schedule** is a valid ordering of operations respecting:

1. **Data dependencies:** Gate B depends on gate A if B uses output of A
2. **Spatial constraints:** Merges require adjacent patches
3. **Resource constraints:** Limited ancilla patches

### Example: Two-CNOT Circuit

Circuit:
```
q0: ──●────────●──
      │        │
q1: ──X────────┼──
               │
q2: ───────────X──
```

**Dependency graph:**
```
CNOT(q0,q1) ──→ CNOT(q0,q2)
```

The second CNOT depends on the first (both use q0).

### Surgery Operations DAG

For each CNOT, expand into surgery primitives:

```
CNOT_1 expansion:
  prep_anc_1 → zz_merge(q0, anc1) → split_1 → xx_merge(anc1, q1) → measure_1

CNOT_2 expansion:
  prep_anc_2 → zz_merge(q0, anc2) → split_2 → xx_merge(anc2, q2) → measure_2

Dependencies:
  split_1 → zz_merge(q0, anc2)  [q0 must be free]
```

### Topological Sorting

A valid schedule is any topological ordering of the DAG.

**Algorithm:**
```
TOPOLOGICAL_SORT(DAG):
    result = []
    ready = nodes with no incoming edges

    while ready not empty:
        node = ready.pop()
        result.append(node)

        for successor in node.successors:
            remove edge (node, successor)
            if successor has no incoming edges:
                ready.add(successor)

    return result
```

---

## Scheduling Algorithms

### ASAP Scheduling (As Soon As Possible)

Schedule each operation at the earliest possible time.

```
ASAP_SCHEDULE(DAG):
    for node in topological_order(DAG):
        t_start = max(t_end for all predecessors)
        t_end = t_start + node.duration
        schedule[node] = (t_start, t_end)
    return schedule
```

**Pros:** Minimizes total time
**Cons:** May have high peak resource usage

### ALAP Scheduling (As Late As Possible)

Schedule each operation at the latest possible time.

```
ALAP_SCHEDULE(DAG, deadline):
    for node in reverse_topological_order(DAG):
        t_end = min(t_start for all successors)
        t_start = t_end - node.duration
        schedule[node] = (t_start, t_end)
    return schedule
```

**Pros:** Delays resource allocation
**Cons:** Sensitive to deadline choice

### List Scheduling

Balances time and resources using priority-based allocation.

```
LIST_SCHEDULE(DAG, num_resources):
    ready_queue = priority_queue()  # sorted by priority
    time = 0
    active = []

    # Initialize with source nodes
    for node in DAG.sources():
        ready_queue.push(node)

    while not done:
        # Free completed operations
        for op in active:
            if op.end_time <= time:
                active.remove(op)
                for succ in op.successors:
                    if succ.all_preds_done():
                        ready_queue.push(succ)

        # Schedule ready operations
        while len(active) < num_resources and ready_queue:
            op = ready_queue.pop()
            op.start_time = time
            op.end_time = time + op.duration
            active.append(op)

        # Advance time
        time = min(op.end_time for op in active)

    return schedule
```

### Surgery-Specific Scheduling

Surface code surgery has additional constraints:

1. **Patch adjacency:** Merge operations require patches to be physically adjacent
2. **Routing:** May need to move patches or create routing ancillas
3. **T-factory integration:** Magic states arrive from factories at specific times

---

## Ancilla Patch Management

### Ancilla Lifecycle

```
ALLOCATE → PREPARE → USE → MEASURE → DEALLOCATE
    ↑                                      │
    └──────────────────────────────────────┘
                   (recycle)
```

### Allocation Strategies

#### Static Allocation

Pre-allocate all ancillas at compile time.

**Pros:** Simple, predictable
**Cons:** May waste space

#### Dynamic Allocation

Allocate ancillas on-demand during execution.

```
DYNAMIC_ALLOCATE(operation):
    if free_pool not empty:
        return free_pool.pop()
    else:
        return create_new_patch()

DYNAMIC_DEALLOCATE(patch):
    free_pool.push(patch)
```

**Pros:** Efficient space usage
**Cons:** More complex scheduling

### Ancilla Sharing

Multiple operations can share ancilla patches if they don't overlap in time:

```
Time:    0    d   2d   3d   4d   5d
Op1:    [====]
Op2:              [====]
Op3:         [====]

Ancilla assignment:
- anc_1: Op1, Op2 (no overlap)
- anc_2: Op3
```

**Optimal sharing** is equivalent to graph coloring on the interval graph.

---

## Space-Time Trade-offs

### The Space-Time Volume

**Definition:** Total resources consumed:

$$\boxed{V = \text{(number of patches)} \times \text{(time)} = \int_0^T n(t) \, dt}$$

where $n(t)$ is the number of active patches at time $t$.

### Trading Space for Time

**More patches → More parallelism → Less time**

Example: Execute two independent CNOTs

**Sequential (1 ancilla):**
```
Time:    0    d   2d   3d   4d   5d   6d   7d   8d   9d  10d
CNOT1:  [=======================================]
CNOT2:                                            [===========]
```
Time: ~$10d$, Space: 3 patches

**Parallel (2 ancillas):**
```
Time:    0    d   2d   3d   4d   5d
CNOT1:  [=====================]
CNOT2:  [=====================]
```
Time: ~$5d$, Space: 4 patches

### Optimal Schedule

Finding the optimal space-time schedule is generally NP-hard.

**Heuristics:**
1. Critical path analysis
2. Resource leveling
3. Genetic algorithms
4. Constraint programming

---

## Worked Examples

### Example 1: CNOT Surgery Scheduling

**Problem:** Schedule CNOT(A, B) using lattice surgery primitives.

**Solution:**

**Primitives:**
1. `prep_anc`: Prepare ancilla in $|+\rangle_L$ (time: $d$)
2. `zz_merge`: Merge A with ancilla (time: $d$)
3. `split`: Split merged patch (time: $d$)
4. `xx_merge`: Merge ancilla with B (time: $d$)
5. `measure`: Measure ancilla in X (time: $d$)

**Dependencies:**
```
prep_anc → zz_merge → split → xx_merge → measure
```

**Schedule:**
| Time | Operation | Patches Used |
|------|-----------|--------------|
| 0-d | prep_anc | anc |
| d-2d | zz_merge(A, anc) | A, anc |
| 2d-3d | split | A, anc |
| 3d-4d | xx_merge(anc, B) | anc, B |
| 4d-5d | measure(anc) | anc |

**Total time:** $5d$
**Peak patches:** 3 (A, B, anc)

### Example 2: Two Independent CNOTs

**Problem:** Schedule CNOT(A,B) and CNOT(C,D) where A,B,C,D are distinct qubits.

**Solution:**

Since the CNOTs are independent, they can execute in parallel.

**Parallel Schedule:**
| Time | Operations |
|------|------------|
| 0-d | prep_anc1, prep_anc2 |
| d-2d | zz_merge(A,anc1), zz_merge(C,anc2) |
| 2d-3d | split1, split2 |
| 3d-4d | xx_merge(anc1,B), xx_merge(anc2,D) |
| 4d-5d | measure1, measure2 |

**Total time:** $5d$ (same as single CNOT!)
**Peak patches:** 6 (A, B, C, D, anc1, anc2)

### Example 3: Dependent CNOTs

**Problem:** Schedule CNOT(A,B) followed by CNOT(B,C).

**Solution:**

The second CNOT depends on B from the first.

**Sequential Schedule:**
| Time | Operation | Notes |
|------|-----------|-------|
| 0-5d | CNOT(A,B) | Full surgery sequence |
| 5d-10d | CNOT(B,C) | Must wait for B |

**Total time:** $10d$

**Optimized:** Notice that after CNOT(A,B), we can start prep for second CNOT early:

| Time | CNOT1 | CNOT2 |
|------|-------|-------|
| 0-d | prep_anc1 | - |
| d-2d | zz_merge | - |
| 2d-3d | split | - |
| 3d-4d | xx_merge | prep_anc2 |
| 4d-5d | measure | - |
| 5d-6d | - | zz_merge |
| ... | - | ... |

Slight overlap saves $d$ cycles.

**Optimized time:** $9d$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** How many syndrome cycles does a single ZZ merge operation take in a distance-7 surface code?

**P1.2** Draw the dependency graph for the circuit:
```
q0: ──H──●──T──
         │
q1: ─────X─────
```

**P1.3** List the lattice surgery primitives needed for a SWAP gate (SWAP = 3 CNOTs).

### Level 2: Intermediate

**P2.1** Given the circuit:
```
q0: ──●──────●──
      │      │
q1: ──X──●───┼──
         │   │
q2: ─────X───X──
```
a) Build the dependency graph
b) Find the critical path
c) What is the minimum execution time in units of $d$?

**P2.2** Design an ancilla sharing strategy for the circuit in P2.1 that uses only 2 ancilla patches.

**P2.3** Calculate the space-time volume (in units of patch·$d$) for both sequential and parallel execution of 4 independent CNOTs.

### Level 3: Challenging

**P3.1** Prove that the minimum time for executing a circuit with $n$ CNOTs in series is $\Omega(nd)$.

**P3.2** Design a scheduling algorithm that minimizes peak patch count while respecting a time deadline.

**P3.3** Analyze the lattice surgery scheduling for a 3-qubit QFT circuit. Include:
- All required surgery primitives
- Dependency graph
- ASAP schedule
- Total execution time

---

## Computational Lab

```python
"""
Day 886: Lattice Surgery Scheduling
===================================

Implementing surgery scheduling algorithms.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


# =============================================================================
# Surgery Primitive Definitions
# =============================================================================

@dataclass
class SurgeryOp:
    """
    A single lattice surgery operation.

    Attributes:
        op_id: Unique identifier
        op_type: Type of operation (prep, merge, split, measure)
        patches: Patches involved
        duration: Time in units of d
        dependencies: Operations that must complete first
    """
    op_id: int
    op_type: str
    patches: Tuple[str, ...]
    duration: int = 1  # in units of d
    dependencies: Set[int] = field(default_factory=set)

    # Scheduling attributes (filled by scheduler)
    start_time: Optional[int] = None
    end_time: Optional[int] = None

    def __hash__(self):
        return hash(self.op_id)

    def __repr__(self):
        return f"{self.op_type}({', '.join(self.patches)})"


class SurgeryPrimitives:
    """Factory for creating surgery primitives."""

    _next_id = 0

    @classmethod
    def _get_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    @classmethod
    def prep_plus(cls, patch: str) -> SurgeryOp:
        """Prepare patch in |+⟩ state."""
        return SurgeryOp(cls._get_id(), 'prep_plus', (patch,), duration=1)

    @classmethod
    def prep_zero(cls, patch: str) -> SurgeryOp:
        """Prepare patch in |0⟩ state."""
        return SurgeryOp(cls._get_id(), 'prep_zero', (patch,), duration=1)

    @classmethod
    def zz_merge(cls, patch1: str, patch2: str) -> SurgeryOp:
        """ZZ merge between two patches."""
        return SurgeryOp(cls._get_id(), 'zz_merge', (patch1, patch2), duration=1)

    @classmethod
    def xx_merge(cls, patch1: str, patch2: str) -> SurgeryOp:
        """XX merge between two patches."""
        return SurgeryOp(cls._get_id(), 'xx_merge', (patch1, patch2), duration=1)

    @classmethod
    def split(cls, patch: str) -> SurgeryOp:
        """Split a merged patch."""
        return SurgeryOp(cls._get_id(), 'split', (patch,), duration=1)

    @classmethod
    def measure_x(cls, patch: str) -> SurgeryOp:
        """Measure patch in X basis."""
        return SurgeryOp(cls._get_id(), 'measure_x', (patch,), duration=1)

    @classmethod
    def measure_z(cls, patch: str) -> SurgeryOp:
        """Measure patch in Z basis."""
        return SurgeryOp(cls._get_id(), 'measure_z', (patch,), duration=1)

    @classmethod
    def twist(cls, patch: str) -> SurgeryOp:
        """Rotate patch (for H gate)."""
        return SurgeryOp(cls._get_id(), 'twist', (patch,), duration=1)

    @classmethod
    def reset(cls):
        """Reset ID counter."""
        cls._next_id = 0


# =============================================================================
# Logical Gate to Surgery Translation
# =============================================================================

class GateToSurgery:
    """Translate logical gates to surgery primitives."""

    def __init__(self):
        self.ancilla_counter = 0

    def _new_ancilla(self) -> str:
        self.ancilla_counter += 1
        return f"anc_{self.ancilla_counter}"

    def cnot(self, control: str, target: str) -> List[SurgeryOp]:
        """
        CNOT gate via lattice surgery.

        Returns list of surgery operations with dependencies set.
        """
        anc = self._new_ancilla()

        # Create operations
        prep = SurgeryPrimitives.prep_plus(anc)
        zz = SurgeryPrimitives.zz_merge(control, anc)
        split = SurgeryPrimitives.split(anc)
        xx = SurgeryPrimitives.xx_merge(anc, target)
        measure = SurgeryPrimitives.measure_x(anc)

        # Set dependencies
        zz.dependencies.add(prep.op_id)
        split.dependencies.add(zz.op_id)
        xx.dependencies.add(split.op_id)
        measure.dependencies.add(xx.op_id)

        return [prep, zz, split, xx, measure]

    def hadamard(self, qubit: str) -> List[SurgeryOp]:
        """H gate via patch twist."""
        return [SurgeryPrimitives.twist(qubit)]

    def t_gate(self, qubit: str, magic_state: str) -> List[SurgeryOp]:
        """
        T gate via magic state injection.

        Assumes magic_state patch is already prepared.
        """
        # CNOT from qubit to magic state, then measure
        ops = self.cnot(qubit, magic_state)

        # The measurement determines the correction
        # (handled classically)

        return ops


# =============================================================================
# Dependency Graph
# =============================================================================

class DependencyGraph:
    """
    DAG of surgery operations with dependencies.
    """

    def __init__(self):
        self.operations: Dict[int, SurgeryOp] = {}
        self.successors: Dict[int, Set[int]] = defaultdict(set)
        self.predecessors: Dict[int, Set[int]] = defaultdict(set)

    def add_operation(self, op: SurgeryOp):
        """Add an operation to the graph."""
        self.operations[op.op_id] = op
        for dep_id in op.dependencies:
            self.successors[dep_id].add(op.op_id)
            self.predecessors[op.op_id].add(dep_id)

    def add_operations(self, ops: List[SurgeryOp]):
        """Add multiple operations."""
        for op in ops:
            self.add_operation(op)

    def add_dependency(self, from_id: int, to_id: int):
        """Add a dependency edge."""
        self.successors[from_id].add(to_id)
        self.predecessors[to_id].add(from_id)
        self.operations[to_id].dependencies.add(from_id)

    def topological_sort(self) -> List[int]:
        """Return operations in topological order."""
        in_degree = {op_id: len(self.predecessors[op_id])
                     for op_id in self.operations}

        ready = [op_id for op_id, deg in in_degree.items() if deg == 0]
        result = []

        while ready:
            op_id = ready.pop(0)
            result.append(op_id)

            for succ_id in self.successors[op_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    ready.append(succ_id)

        return result

    def critical_path_length(self) -> int:
        """Compute the critical path length."""
        if not self.operations:
            return 0

        # Dynamic programming: longest path to each node
        longest = {}

        for op_id in self.topological_sort():
            op = self.operations[op_id]
            pred_max = 0
            for pred_id in self.predecessors[op_id]:
                if pred_id in longest:
                    pred_max = max(pred_max, longest[pred_id])
            longest[op_id] = pred_max + op.duration

        return max(longest.values()) if longest else 0

    def find_critical_path(self) -> List[int]:
        """Find one critical path."""
        if not self.operations:
            return []

        # Compute earliest finish times
        finish = {}
        for op_id in self.topological_sort():
            op = self.operations[op_id]
            start = 0
            for pred_id in self.predecessors[op_id]:
                start = max(start, finish[pred_id])
            finish[op_id] = start + op.duration

        # Find end of critical path
        end_op = max(finish.keys(), key=lambda x: finish[x])

        # Trace back
        path = [end_op]
        current = end_op

        while True:
            preds = list(self.predecessors[current])
            if not preds:
                break

            # Find predecessor on critical path
            current_start = finish[current] - self.operations[current].duration
            critical_pred = None
            for pred_id in preds:
                if finish[pred_id] == current_start:
                    critical_pred = pred_id
                    break

            if critical_pred is None:
                break

            path.append(critical_pred)
            current = critical_pred

        return list(reversed(path))


# =============================================================================
# Schedulers
# =============================================================================

def asap_schedule(graph: DependencyGraph) -> Dict[int, Tuple[int, int]]:
    """
    As-Soon-As-Possible scheduling.

    Returns dict mapping op_id to (start_time, end_time).
    """
    schedule = {}

    for op_id in graph.topological_sort():
        op = graph.operations[op_id]

        # Start as soon as all predecessors finish
        start = 0
        for pred_id in graph.predecessors[op_id]:
            if pred_id in schedule:
                start = max(start, schedule[pred_id][1])

        end = start + op.duration
        schedule[op_id] = (start, end)
        op.start_time = start
        op.end_time = end

    return schedule


def alap_schedule(graph: DependencyGraph,
                  deadline: Optional[int] = None) -> Dict[int, Tuple[int, int]]:
    """
    As-Late-As-Possible scheduling.

    Returns dict mapping op_id to (start_time, end_time).
    """
    if deadline is None:
        deadline = graph.critical_path_length()

    schedule = {}
    topo_order = graph.topological_sort()

    for op_id in reversed(topo_order):
        op = graph.operations[op_id]

        # End as late as possible before successors start
        end = deadline
        for succ_id in graph.successors[op_id]:
            if succ_id in schedule:
                end = min(end, schedule[succ_id][0])

        start = end - op.duration
        schedule[op_id] = (start, end)
        op.start_time = start
        op.end_time = end

    return schedule


def list_schedule(graph: DependencyGraph,
                  max_parallel: int = float('inf')) -> Dict[int, Tuple[int, int]]:
    """
    List scheduling with resource constraint.

    max_parallel: maximum number of concurrent operations.
    """
    schedule = {}
    in_degree = {op_id: len(graph.predecessors[op_id])
                 for op_id in graph.operations}

    # Priority queue: (finish_time, op_id)
    active = []  # heap of (finish_time, op_id)
    ready = []   # ops ready to start

    # Initialize ready queue
    for op_id, deg in in_degree.items():
        if deg == 0:
            ready.append(op_id)

    time = 0

    while ready or active:
        # Complete finished operations
        while active and active[0][0] <= time:
            finish_time, op_id = heapq.heappop(active)

            for succ_id in graph.successors[op_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    ready.append(succ_id)

        # Schedule ready operations (respecting resource limit)
        while ready and len(active) < max_parallel:
            op_id = ready.pop(0)
            op = graph.operations[op_id]

            start = time
            end = time + op.duration
            schedule[op_id] = (start, end)
            op.start_time = start
            op.end_time = end

            heapq.heappush(active, (end, op_id))

        # Advance time
        if active:
            time = active[0][0]
        elif ready:
            # No active ops but ready ops waiting (shouldn't happen normally)
            pass

    return schedule


# =============================================================================
# Schedule Analysis
# =============================================================================

def compute_makespan(schedule: Dict[int, Tuple[int, int]]) -> int:
    """Compute total execution time."""
    if not schedule:
        return 0
    return max(end for _, end in schedule.values())


def compute_patch_usage(graph: DependencyGraph,
                        schedule: Dict[int, Tuple[int, int]]) -> Dict[int, int]:
    """
    Compute patch usage over time.

    Returns dict mapping time -> number of patches in use.
    """
    events = []  # (time, delta)

    for op_id, (start, end) in schedule.items():
        op = graph.operations[op_id]
        num_patches = len(op.patches)
        events.append((start, num_patches))
        events.append((end, -num_patches))

    events.sort()

    usage = {}
    current = 0
    for time, delta in events:
        current += delta
        usage[time] = current

    return usage


def compute_space_time_volume(graph: DependencyGraph,
                              schedule: Dict[int, Tuple[int, int]]) -> int:
    """Compute space-time volume (patches × time units)."""
    volume = 0
    for op_id, (start, end) in schedule.items():
        op = graph.operations[op_id]
        volume += len(op.patches) * op.duration
    return volume


# =============================================================================
# Visualization
# =============================================================================

def print_schedule(graph: DependencyGraph,
                   schedule: Dict[int, Tuple[int, int]],
                   width: int = 60):
    """Print a text-based Gantt chart."""
    if not schedule:
        print("Empty schedule")
        return

    makespan = compute_makespan(schedule)
    scale = width / makespan if makespan > 0 else 1

    print(f"\nSchedule (makespan = {makespan}d cycles):")
    print("-" * (width + 20))

    for op_id in sorted(schedule.keys()):
        op = graph.operations[op_id]
        start, end = schedule[op_id]

        # Create bar
        bar_start = int(start * scale)
        bar_length = max(1, int((end - start) * scale))
        bar = " " * bar_start + "█" * bar_length

        label = f"{op.op_type[:8]:8s}"
        print(f"{label} |{bar}")

    print("-" * (width + 20))
    print(f"Time:    |{'0':^10}|{makespan//2:^{width//2-10}}|{makespan:>{width//2}}")


def print_gantt_chart(graph: DependencyGraph,
                      schedule: Dict[int, Tuple[int, int]]):
    """Print detailed Gantt chart."""
    if not schedule:
        return

    makespan = compute_makespan(schedule)

    print(f"\nGantt Chart (makespan = {makespan}d):")
    print("=" * 70)

    # Sort by start time
    sorted_ops = sorted(schedule.items(), key=lambda x: x[1][0])

    for op_id, (start, end) in sorted_ops:
        op = graph.operations[op_id]
        print(f"  t={start:3d}-{end:3d}: {op}")

    print("=" * 70)


# =============================================================================
# Demo
# =============================================================================

def demo_scheduling():
    """Demonstrate surgery scheduling."""

    print("=" * 70)
    print("Day 886: Lattice Surgery Scheduling - Demonstration")
    print("=" * 70)

    SurgeryPrimitives.reset()
    translator = GateToSurgery()

    # Example 1: Single CNOT
    print("\n1. Single CNOT Surgery Schedule")
    print("-" * 40)

    graph1 = DependencyGraph()
    cnot_ops = translator.cnot("q0", "q1")
    graph1.add_operations(cnot_ops)

    schedule1 = asap_schedule(graph1)

    print("Operations:")
    for op in cnot_ops:
        print(f"  {op.op_id}: {op}")

    print(f"\nCritical path length: {graph1.critical_path_length()}d")
    print(f"Makespan: {compute_makespan(schedule1)}d")

    print_gantt_chart(graph1, schedule1)

    # Example 2: Two independent CNOTs
    print("\n2. Two Independent CNOTs (Parallel)")
    print("-" * 40)

    SurgeryPrimitives.reset()
    translator = GateToSurgery()

    graph2 = DependencyGraph()
    cnot1_ops = translator.cnot("q0", "q1")
    cnot2_ops = translator.cnot("q2", "q3")
    graph2.add_operations(cnot1_ops)
    graph2.add_operations(cnot2_ops)

    schedule2 = asap_schedule(graph2)

    print(f"Critical path length: {graph2.critical_path_length()}d")
    print(f"Makespan: {compute_makespan(schedule2)}d")
    print(f"Space-time volume: {compute_space_time_volume(graph2, schedule2)} patch·d")

    # Example 3: Two dependent CNOTs
    print("\n3. Two Dependent CNOTs (Sequential)")
    print("-" * 40)

    SurgeryPrimitives.reset()
    translator = GateToSurgery()

    graph3 = DependencyGraph()
    cnot1_ops = translator.cnot("q0", "q1")
    cnot2_ops = translator.cnot("q1", "q2")  # Depends on q1

    graph3.add_operations(cnot1_ops)
    graph3.add_operations(cnot2_ops)

    # Add dependency: second CNOT's ZZ merge depends on first CNOT completing
    # (simplified: depend on last op of first CNOT)
    last_cnot1 = max(op.op_id for op in cnot1_ops)
    first_cnot2 = min(op.op_id for op in cnot2_ops)
    graph3.add_dependency(last_cnot1, first_cnot2)

    schedule3 = asap_schedule(graph3)

    print(f"Critical path length: {graph3.critical_path_length()}d")
    print(f"Makespan: {compute_makespan(schedule3)}d")

    # Example 4: Resource-constrained scheduling
    print("\n4. Resource-Constrained Scheduling")
    print("-" * 40)

    SurgeryPrimitives.reset()
    translator = GateToSurgery()

    graph4 = DependencyGraph()
    for i in range(4):
        ops = translator.cnot(f"q{2*i}", f"q{2*i+1}")
        graph4.add_operations(ops)

    # Unlimited parallelism
    schedule_unlimited = asap_schedule(graph4)

    # Limited to 2 concurrent operations
    schedule_limited = list_schedule(graph4, max_parallel=2)

    print(f"Unlimited parallelism: makespan = {compute_makespan(schedule_unlimited)}d")
    print(f"Max 2 parallel ops:    makespan = {compute_makespan(schedule_limited)}d")

    # Example 5: Critical path analysis
    print("\n5. Critical Path Analysis")
    print("-" * 40)

    SurgeryPrimitives.reset()
    translator = GateToSurgery()

    # Build a more complex circuit
    graph5 = DependencyGraph()

    # q0 --●-- q1 --●-- q2
    ops1 = translator.cnot("q0", "q1")
    graph5.add_operations(ops1)

    ops2 = translator.cnot("q1", "q2")
    graph5.add_operations(ops2)
    # Dependency
    graph5.add_dependency(max(o.op_id for o in ops1),
                          min(o.op_id for o in ops2))

    critical_path = graph5.find_critical_path()
    print(f"Critical path: {critical_path}")
    print(f"Critical path length: {graph5.critical_path_length()}d")

    # Print critical path operations
    print("Critical path operations:")
    for op_id in critical_path:
        op = graph5.operations[op_id]
        print(f"  {op}")


if __name__ == "__main__":
    demo_scheduling()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Surgery operation time | $T_{\text{op}} = O(d)$ cycles |
| CNOT surgery time | $T_{\text{CNOT}} \approx 5d$ cycles |
| Critical path | Longest path in dependency DAG |
| Space-time volume | $V = \int n(t) \, dt$ |
| Makespan | $T = \max_{\text{ops}} t_{\text{end}}$ |

### Main Takeaways

1. **Lattice surgery** translates logical gates into merge/split/measure primitives
2. **Each operation** takes $O(d)$ syndrome cycles, where $d$ is code distance
3. **Dependency graphs** capture the ordering constraints between operations
4. **ASAP scheduling** minimizes time but may spike resource usage
5. **Resource-constrained scheduling** trades time for reduced patch count
6. **Critical path** determines the minimum possible execution time

---

## Daily Checklist

- [ ] I can list the basic lattice surgery primitives
- [ ] I understand how CNOT is implemented via surgery
- [ ] I can build a dependency graph for a circuit
- [ ] I know the difference between ASAP and ALAP scheduling
- [ ] I can compute critical path length
- [ ] I understand space-time trade-offs in scheduling

---

## Preview: Day 887

Tomorrow we explore **Parallelization & Pipelining**:

- Extracting maximum parallelism from circuits
- Pipelining magic state production
- T-factory scheduling and integration
- Optimizing for minimum space-time volume
- Advanced scheduling heuristics

Parallelism is essential for practical fault-tolerant quantum computing performance.
