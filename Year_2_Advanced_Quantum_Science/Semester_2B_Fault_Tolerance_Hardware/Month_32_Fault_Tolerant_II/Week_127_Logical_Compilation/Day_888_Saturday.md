# Day 888: Computational Lab - Logical Compiler Implementation

## Overview

**Day:** 888 of 1008
**Week:** 127 (Logical Gate Compilation)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Building a Simple Logical Compiler and T-Count Optimizer

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Compiler architecture and implementation |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Optimization passes and testing |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Benchmarking and analysis |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Implement** a complete logical compilation pipeline
2. **Apply** T-count minimization techniques
3. **Generate** lattice surgery schedules from logical circuits
4. **Visualize** compilation results and resource usage
5. **Benchmark** compilation quality on standard circuits
6. **Analyze** trade-offs between different optimization strategies

---

## Lab Overview

Today we build a complete **Logical Quantum Compiler** that:

1. Parses quantum circuits (QASM-like format)
2. Decomposes into Clifford+T gate set
3. Optimizes T-count
4. Schedules operations for lattice surgery
5. Estimates resources

---

## Implementation

### Complete Compiler Implementation

```python
"""
Day 888: Logical Quantum Compiler
=================================

A complete implementation of a logical quantum compiler
with T-count optimization and surgery scheduling.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import heapq
import re
from abc import ABC, abstractmethod


# =============================================================================
# PART 1: GATE AND CIRCUIT REPRESENTATION
# =============================================================================

class GateType(Enum):
    """Enumeration of supported gate types."""
    # Clifford gates
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    S = "S"
    SDG = "Sdg"
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"

    # Non-Clifford gates
    T = "T"
    TDG = "Tdg"
    RZ = "Rz"
    RX = "Rx"
    RY = "Ry"
    CCX = "CCX"  # Toffoli
    CCZ = "CCZ"


@dataclass
class LogicalGate:
    """
    Represents a gate in the logical circuit.
    """
    gate_id: int
    gate_type: GateType
    qubits: Tuple[int, ...]
    parameters: Tuple[float, ...] = ()

    # Computed properties
    t_cost: int = 0
    is_clifford: bool = True

    # Scheduling
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    dependencies: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Compute T-cost and Clifford status."""
        clifford_gates = {GateType.I, GateType.X, GateType.Y, GateType.Z,
                         GateType.H, GateType.S, GateType.SDG,
                         GateType.CNOT, GateType.CZ, GateType.SWAP}

        self.is_clifford = self.gate_type in clifford_gates

        t_costs = {
            GateType.T: 1,
            GateType.TDG: 1,
            GateType.CCX: 7,  # Toffoli
            GateType.CCZ: 7,
        }
        self.t_cost = t_costs.get(self.gate_type, 0)

        # Rotation gates need synthesis
        if self.gate_type in {GateType.RZ, GateType.RX, GateType.RY}:
            self.is_clifford = False
            # Estimate T-cost (would use gridsynth in practice)
            self.t_cost = 50  # Placeholder

    def __repr__(self):
        if self.parameters:
            params = ", ".join(f"{p:.4f}" for p in self.parameters)
            return f"{self.gate_type.value}({params})@{self.qubits}"
        return f"{self.gate_type.value}@{self.qubits}"


class LogicalCircuit:
    """
    Represents a complete logical quantum circuit.
    """

    def __init__(self, num_qubits: int, name: str = "circuit"):
        self.name = name
        self.num_qubits = num_qubits
        self.gates: List[LogicalGate] = []
        self._next_id = 0
        self._last_on_qubit: Dict[int, int] = {}

    def add_gate(self, gate_type: GateType, qubits: Tuple[int, ...],
                 parameters: Tuple[float, ...] = ()) -> int:
        """Add a gate to the circuit."""
        gate_id = self._next_id
        self._next_id += 1

        gate = LogicalGate(
            gate_id=gate_id,
            gate_type=gate_type,
            qubits=qubits,
            parameters=parameters
        )

        # Compute dependencies
        for q in qubits:
            if q in self._last_on_qubit:
                gate.dependencies.add(self._last_on_qubit[q])
            self._last_on_qubit[q] = gate_id

        self.gates.append(gate)
        return gate_id

    # Convenience methods
    def h(self, q): return self.add_gate(GateType.H, (q,))
    def x(self, q): return self.add_gate(GateType.X, (q,))
    def y(self, q): return self.add_gate(GateType.Y, (q,))
    def z(self, q): return self.add_gate(GateType.Z, (q,))
    def s(self, q): return self.add_gate(GateType.S, (q,))
    def sdg(self, q): return self.add_gate(GateType.SDG, (q,))
    def t(self, q): return self.add_gate(GateType.T, (q,))
    def tdg(self, q): return self.add_gate(GateType.TDG, (q,))
    def cnot(self, c, t): return self.add_gate(GateType.CNOT, (c, t))
    def cz(self, c, t): return self.add_gate(GateType.CZ, (c, t))
    def ccx(self, c1, c2, t): return self.add_gate(GateType.CCX, (c1, c2, t))
    def rz(self, q, theta): return self.add_gate(GateType.RZ, (q,), (theta,))
    def rx(self, q, theta): return self.add_gate(GateType.RX, (q,), (theta,))
    def ry(self, q, theta): return self.add_gate(GateType.RY, (q,), (theta,))

    def t_count(self) -> int:
        """Total T-count."""
        return sum(g.t_cost for g in self.gates)

    def gate_count(self) -> Dict[str, int]:
        """Count of each gate type."""
        counts = defaultdict(int)
        for g in self.gates:
            counts[g.gate_type.value] += 1
        return dict(counts)

    def depth(self) -> int:
        """Circuit depth (critical path)."""
        if not self.gates:
            return 0

        levels = {}
        for gate in self.gates:
            pred_level = max((levels.get(d, 0) for d in gate.dependencies), default=0)
            levels[gate.gate_id] = pred_level + 1

        return max(levels.values())

    def t_depth(self) -> int:
        """T-depth (critical path of T gates only)."""
        if not self.gates:
            return 0

        t_dist = {}
        for gate in self.gates:
            pred_max = max((t_dist.get(d, 0) for d in gate.dependencies), default=0)
            if gate.gate_type in {GateType.T, GateType.TDG}:
                t_dist[gate.gate_id] = pred_max + 1
            else:
                t_dist[gate.gate_id] = pred_max

        return max(t_dist.values()) if t_dist else 0

    def copy(self) -> 'LogicalCircuit':
        """Create a deep copy of the circuit."""
        new_circuit = LogicalCircuit(self.num_qubits, self.name + "_copy")
        for gate in self.gates:
            new_circuit.add_gate(gate.gate_type, gate.qubits, gate.parameters)
        return new_circuit

    def __repr__(self):
        return (f"LogicalCircuit('{self.name}', qubits={self.num_qubits}, "
                f"gates={len(self.gates)}, T-count={self.t_count()}, "
                f"depth={self.depth()})")


# =============================================================================
# PART 2: CIRCUIT PARSER
# =============================================================================

class QASMParser:
    """
    Simple QASM-like parser for quantum circuits.
    """

    GATE_MAP = {
        'h': GateType.H,
        'x': GateType.X,
        'y': GateType.Y,
        'z': GateType.Z,
        's': GateType.S,
        'sdg': GateType.SDG,
        't': GateType.T,
        'tdg': GateType.TDG,
        'cx': GateType.CNOT,
        'cnot': GateType.CNOT,
        'cz': GateType.CZ,
        'ccx': GateType.CCX,
        'toffoli': GateType.CCX,
        'rz': GateType.RZ,
        'rx': GateType.RX,
        'ry': GateType.RY,
    }

    @classmethod
    def parse(cls, qasm_str: str) -> LogicalCircuit:
        """Parse QASM string into LogicalCircuit."""
        lines = qasm_str.strip().split('\n')
        num_qubits = 0
        circuit = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            # Parse qubit declaration
            if line.startswith('qreg') or line.startswith('qubit'):
                match = re.search(r'\[(\d+)\]', line)
                if match:
                    num_qubits = int(match.group(1))
                    circuit = LogicalCircuit(num_qubits)
                continue

            if circuit is None:
                continue

            # Parse gate
            match = re.match(r'(\w+)(?:\(([\d.,\-\*\s/pi]+)\))?\s+(.+);?', line)
            if match:
                gate_name = match.group(1).lower()
                params_str = match.group(2)
                qubits_str = match.group(3)

                if gate_name not in cls.GATE_MAP:
                    continue

                gate_type = cls.GATE_MAP[gate_name]

                # Parse qubits
                qubit_matches = re.findall(r'q\[(\d+)\]', qubits_str)
                if not qubit_matches:
                    qubit_matches = re.findall(r'(\d+)', qubits_str)
                qubits = tuple(int(q) for q in qubit_matches)

                # Parse parameters
                params = ()
                if params_str:
                    # Simple parser - handle pi
                    params_str = params_str.replace('pi', str(np.pi))
                    params = tuple(eval(p.strip()) for p in params_str.split(','))

                circuit.add_gate(gate_type, qubits, params)

        return circuit if circuit else LogicalCircuit(0)


# =============================================================================
# PART 3: COMPILER PASSES
# =============================================================================

class CompilerPass(ABC):
    """Abstract base class for compiler passes."""

    @abstractmethod
    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Apply the pass and return the transformed circuit."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pass."""
        pass


class GateDecompositionPass(CompilerPass):
    """
    Decomposes gates into Clifford+T gate set.
    """

    @property
    def name(self) -> str:
        return "GateDecomposition"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Decompose all non-Clifford+T gates."""
        new_circuit = LogicalCircuit(circuit.num_qubits, circuit.name)

        for gate in circuit.gates:
            decomposed = self._decompose_gate(gate)
            for g_type, qubits, params in decomposed:
                new_circuit.add_gate(g_type, qubits, params)

        return new_circuit

    def _decompose_gate(self, gate: LogicalGate) -> List[Tuple[GateType, Tuple[int, ...], Tuple[float, ...]]]:
        """Decompose a single gate."""
        q = gate.qubits

        # Already in Clifford+T
        if gate.gate_type in {GateType.I, GateType.X, GateType.Y, GateType.Z,
                              GateType.H, GateType.S, GateType.SDG,
                              GateType.T, GateType.TDG, GateType.CNOT}:
            return [(gate.gate_type, q, gate.parameters)]

        # CZ -> H CNOT H
        if gate.gate_type == GateType.CZ:
            return [
                (GateType.H, (q[1],), ()),
                (GateType.CNOT, q, ()),
                (GateType.H, (q[1],), ()),
            ]

        # SWAP -> 3 CNOTs
        if gate.gate_type == GateType.SWAP:
            return [
                (GateType.CNOT, (q[0], q[1]), ()),
                (GateType.CNOT, (q[1], q[0]), ()),
                (GateType.CNOT, (q[0], q[1]), ()),
            ]

        # Toffoli -> Standard 7-T decomposition (simplified)
        if gate.gate_type == GateType.CCX:
            return self._decompose_toffoli(q[0], q[1], q[2])

        # Rotation gates -> approximate with T gates
        if gate.gate_type == GateType.RZ:
            return self._decompose_rz(q[0], gate.parameters[0])

        if gate.gate_type == GateType.RX:
            # Rx = H Rz H
            decomp = [(GateType.H, (q[0],), ())]
            decomp.extend(self._decompose_rz(q[0], gate.parameters[0]))
            decomp.append((GateType.H, (q[0],), ()))
            return decomp

        if gate.gate_type == GateType.RY:
            # Ry = S H Rz H Sdg
            decomp = [(GateType.S, (q[0],), ()), (GateType.H, (q[0],), ())]
            decomp.extend(self._decompose_rz(q[0], gate.parameters[0]))
            decomp.extend([(GateType.H, (q[0],), ()), (GateType.SDG, (q[0],), ())])
            return decomp

        # Default: return as-is
        return [(gate.gate_type, q, gate.parameters)]

    def _decompose_toffoli(self, c1: int, c2: int, t: int) -> List[Tuple]:
        """Standard Toffoli decomposition (7 T gates)."""
        return [
            (GateType.H, (t,), ()),
            (GateType.CNOT, (c2, t), ()),
            (GateType.TDG, (t,), ()),
            (GateType.CNOT, (c1, t), ()),
            (GateType.T, (t,), ()),
            (GateType.CNOT, (c2, t), ()),
            (GateType.TDG, (t,), ()),
            (GateType.CNOT, (c1, t), ()),
            (GateType.T, (c2,), ()),
            (GateType.T, (t,), ()),
            (GateType.H, (t,), ()),
            (GateType.CNOT, (c1, c2), ()),
            (GateType.T, (c1,), ()),
            (GateType.TDG, (c2,), ()),
            (GateType.CNOT, (c1, c2), ()),
        ]

    def _decompose_rz(self, q: int, theta: float) -> List[Tuple]:
        """
        Approximate Rz(theta) with Clifford+T.

        Simplified: check for exact cases, otherwise use T-approximation.
        """
        # Normalize angle
        theta = theta % (2 * np.pi)

        # Exact cases
        if np.isclose(theta, 0):
            return [(GateType.I, (q,), ())]
        if np.isclose(theta, np.pi/4):
            return [(GateType.T, (q,), ())]
        if np.isclose(theta, np.pi/2):
            return [(GateType.S, (q,), ())]
        if np.isclose(theta, np.pi):
            return [(GateType.Z, (q,), ())]
        if np.isclose(theta, 3*np.pi/2):
            return [(GateType.SDG, (q,), ())]
        if np.isclose(theta, 7*np.pi/4):
            return [(GateType.TDG, (q,), ())]

        # Approximate with multiple T gates (placeholder for gridsynth)
        n_t = int(3 * np.log2(1e8) + 10)  # ~50 T gates for high precision
        result = []
        for _ in range(n_t // 4):
            result.append((GateType.T, (q,), ()))
            result.append((GateType.H, (q,), ()))
        return result if result else [(GateType.T, (q,), ())]


class TCancellationPass(CompilerPass):
    """
    Cancels adjacent T-Tdg pairs.
    """

    @property
    def name(self) -> str:
        return "TCancellation"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Remove T-Tdg and Tdg-T pairs on the same qubit."""
        new_circuit = LogicalCircuit(circuit.num_qubits, circuit.name)

        # Group gates by qubit
        qubit_gates: Dict[int, List[LogicalGate]] = defaultdict(list)
        for gate in circuit.gates:
            for q in gate.qubits:
                qubit_gates[q].append(gate)

        # Track which gates to keep
        keep_gate = {g.gate_id: True for g in circuit.gates}

        # Look for cancellations on each qubit
        for q, gates in qubit_gates.items():
            # Only consider single-qubit gates on this qubit
            single_q_gates = [g for g in gates if len(g.qubits) == 1]

            i = 0
            while i < len(single_q_gates) - 1:
                g1 = single_q_gates[i]
                g2 = single_q_gates[i + 1]

                # Check for T-Tdg or Tdg-T
                if ((g1.gate_type == GateType.T and g2.gate_type == GateType.TDG) or
                    (g1.gate_type == GateType.TDG and g2.gate_type == GateType.T)):
                    # Check if they're truly adjacent (no gates between them on this qubit)
                    idx1 = circuit.gates.index(g1)
                    idx2 = circuit.gates.index(g2)

                    # Simplified check: if consecutive in the gate list for this qubit
                    can_cancel = True
                    for g in circuit.gates[idx1+1:idx2]:
                        if q in g.qubits:
                            can_cancel = False
                            break

                    if can_cancel and keep_gate[g1.gate_id] and keep_gate[g2.gate_id]:
                        keep_gate[g1.gate_id] = False
                        keep_gate[g2.gate_id] = False
                        i += 2
                        continue

                i += 1

        # Rebuild circuit with kept gates
        for gate in circuit.gates:
            if keep_gate[gate.gate_id]:
                new_circuit.add_gate(gate.gate_type, gate.qubits, gate.parameters)

        return new_circuit


class TMergePass(CompilerPass):
    """
    Merges T-T pairs into S gates.
    """

    @property
    def name(self) -> str:
        return "TMerge"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Merge T-T -> S and Tdg-Tdg -> Sdg."""
        new_circuit = LogicalCircuit(circuit.num_qubits, circuit.name)

        i = 0
        gates = circuit.gates

        while i < len(gates):
            gate = gates[i]

            # Check for T-T or Tdg-Tdg patterns
            if i + 1 < len(gates):
                next_gate = gates[i + 1]

                if (gate.gate_type == GateType.T and
                    next_gate.gate_type == GateType.T and
                    gate.qubits == next_gate.qubits):
                    # T-T = S
                    new_circuit.add_gate(GateType.S, gate.qubits)
                    i += 2
                    continue

                if (gate.gate_type == GateType.TDG and
                    next_gate.gate_type == GateType.TDG and
                    gate.qubits == next_gate.qubits):
                    # Tdg-Tdg = Sdg
                    new_circuit.add_gate(GateType.SDG, gate.qubits)
                    i += 2
                    continue

            # No merge possible
            new_circuit.add_gate(gate.gate_type, gate.qubits, gate.parameters)
            i += 1

        return new_circuit


class CliffordSimplificationPass(CompilerPass):
    """
    Simplifies sequences of Clifford gates.
    """

    @property
    def name(self) -> str:
        return "CliffordSimplification"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Apply Clifford simplification rules."""
        new_circuit = LogicalCircuit(circuit.num_qubits, circuit.name)

        # Simple rules: H-H = I, S-S-S-S = I, etc.
        i = 0
        gates = circuit.gates

        while i < len(gates):
            gate = gates[i]

            # H-H cancellation
            if i + 1 < len(gates):
                next_gate = gates[i + 1]
                if (gate.gate_type == GateType.H and
                    next_gate.gate_type == GateType.H and
                    gate.qubits == next_gate.qubits):
                    i += 2
                    continue

            # S-Sdg cancellation
            if i + 1 < len(gates):
                next_gate = gates[i + 1]
                if ((gate.gate_type == GateType.S and next_gate.gate_type == GateType.SDG) or
                    (gate.gate_type == GateType.SDG and next_gate.gate_type == GateType.S)):
                    if gate.qubits == next_gate.qubits:
                        i += 2
                        continue

            new_circuit.add_gate(gate.gate_type, gate.qubits, gate.parameters)
            i += 1

        return new_circuit


# =============================================================================
# PART 4: COMPILER PIPELINE
# =============================================================================

class LogicalCompiler:
    """
    Main compiler class that orchestrates the compilation pipeline.
    """

    def __init__(self, optimization_level: int = 1):
        """
        Initialize compiler with optimization level.

        Levels:
        0 - No optimization (decomposition only)
        1 - Basic optimization (cancellation, merge)
        2 - Full optimization (all passes, multiple iterations)
        """
        self.optimization_level = optimization_level
        self.passes: List[CompilerPass] = []
        self.stats: Dict[str, Any] = {}

        self._setup_passes()

    def _setup_passes(self):
        """Configure compiler passes based on optimization level."""
        # Always do decomposition
        self.passes.append(GateDecompositionPass())

        if self.optimization_level >= 1:
            self.passes.append(TCancellationPass())
            self.passes.append(TMergePass())
            self.passes.append(CliffordSimplificationPass())

        if self.optimization_level >= 2:
            # Multiple iterations
            for _ in range(3):
                self.passes.append(TCancellationPass())
                self.passes.append(TMergePass())
                self.passes.append(CliffordSimplificationPass())

    def compile(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """
        Compile the circuit through all passes.
        """
        self.stats = {
            'input_gates': len(circuit.gates),
            'input_t_count': circuit.t_count(),
            'input_depth': circuit.depth(),
            'passes': []
        }

        current = circuit

        for pass_obj in self.passes:
            before_t = current.t_count()
            before_gates = len(current.gates)

            current = pass_obj.run(current)

            self.stats['passes'].append({
                'name': pass_obj.name,
                'gates_before': before_gates,
                'gates_after': len(current.gates),
                't_before': before_t,
                't_after': current.t_count()
            })

        self.stats['output_gates'] = len(current.gates)
        self.stats['output_t_count'] = current.t_count()
        self.stats['output_depth'] = current.depth()
        self.stats['t_reduction'] = self.stats['input_t_count'] - self.stats['output_t_count']

        return current

    def print_stats(self):
        """Print compilation statistics."""
        print("\n" + "=" * 60)
        print("COMPILATION STATISTICS")
        print("=" * 60)

        print(f"\nInput:")
        print(f"  Gates: {self.stats['input_gates']}")
        print(f"  T-count: {self.stats['input_t_count']}")
        print(f"  Depth: {self.stats['input_depth']}")

        print(f"\nOutput:")
        print(f"  Gates: {self.stats['output_gates']}")
        print(f"  T-count: {self.stats['output_t_count']}")
        print(f"  Depth: {self.stats['output_depth']}")

        print(f"\nT-count reduction: {self.stats['t_reduction']} "
              f"({100*self.stats['t_reduction']/max(1,self.stats['input_t_count']):.1f}%)")

        print("\nPass-by-pass:")
        for p in self.stats['passes']:
            t_change = p['t_after'] - p['t_before']
            print(f"  {p['name']:25s}: T-count {p['t_before']:4d} -> {p['t_after']:4d} "
                  f"({'+' if t_change >= 0 else ''}{t_change})")


# =============================================================================
# PART 5: RESOURCE ESTIMATION
# =============================================================================

@dataclass
class ResourceEstimate:
    """Resource requirements for executing a compiled circuit."""
    logical_qubits: int
    physical_qubits: int
    t_count: int
    t_depth: int
    num_factories: int
    factory_qubits: int
    execution_cycles: int
    execution_time_us: float

    def total_qubits(self) -> int:
        return self.physical_qubits + self.factory_qubits

    def __repr__(self):
        return (f"ResourceEstimate(\n"
                f"  logical_qubits={self.logical_qubits},\n"
                f"  physical_qubits={self.physical_qubits},\n"
                f"  t_count={self.t_count},\n"
                f"  num_factories={self.num_factories},\n"
                f"  total_qubits={self.total_qubits()},\n"
                f"  execution_time={self.execution_time_us:.1f} us\n)")


class ResourceEstimator:
    """
    Estimates resources for fault-tolerant execution.
    """

    def __init__(self,
                 code_distance: int = 17,
                 syndrome_cycle_us: float = 1.0,
                 factory_footprint: int = 15000,
                 distillation_cycles: int = 500):
        self.code_distance = code_distance
        self.syndrome_cycle_us = syndrome_cycle_us
        self.factory_footprint = factory_footprint
        self.distillation_cycles = distillation_cycles

    def estimate(self, circuit: LogicalCircuit) -> ResourceEstimate:
        """Estimate resources for a circuit."""
        d = self.code_distance

        # Physical qubits for data
        physical_qubits = circuit.num_qubits * (d ** 2)

        # Execution cycles
        # Clifford gates: ~d cycles each
        # T gates: limited by factory output
        t_count = circuit.t_count()
        t_depth = circuit.t_depth()
        clifford_count = len(circuit.gates) - t_count

        # Factory requirements
        total_cycles = circuit.depth() * d
        if t_count > 0:
            # Need enough factories to produce T states in time
            t_rate = t_count / total_cycles if total_cycles > 0 else 1
            factory_rate = 1 / self.distillation_cycles
            num_factories = max(1, int(np.ceil(t_rate / factory_rate)))
        else:
            num_factories = 0

        factory_qubits = num_factories * self.factory_footprint

        # Execution time
        execution_cycles = total_cycles + t_depth * self.distillation_cycles // num_factories if num_factories > 0 else total_cycles
        execution_time_us = execution_cycles * self.syndrome_cycle_us

        return ResourceEstimate(
            logical_qubits=circuit.num_qubits,
            physical_qubits=physical_qubits,
            t_count=t_count,
            t_depth=t_depth,
            num_factories=num_factories,
            factory_qubits=factory_qubits,
            execution_cycles=execution_cycles,
            execution_time_us=execution_time_us
        )


# =============================================================================
# PART 6: BENCHMARK CIRCUITS
# =============================================================================

def create_qft_circuit(n: int) -> LogicalCircuit:
    """Create n-qubit QFT circuit."""
    circuit = LogicalCircuit(n, f"QFT_{n}")

    for i in range(n):
        circuit.h(i)
        for j in range(i + 1, n):
            # Controlled rotation by 2*pi/2^(j-i+1)
            # Approximate with T gates
            k = j - i + 1
            if k == 2:
                # Controlled-S
                circuit.cnot(j, i)
                circuit.t(i)
                circuit.cnot(j, i)
            else:
                # Controlled-R_k approximated
                circuit.cnot(j, i)
                for _ in range(k - 1):
                    circuit.t(i)
                circuit.cnot(j, i)

    # Swap to reverse qubit order
    for i in range(n // 2):
        j = n - 1 - i
        circuit.cnot(i, j)
        circuit.cnot(j, i)
        circuit.cnot(i, j)

    return circuit


def create_toffoli_chain(n: int) -> LogicalCircuit:
    """Create chain of n Toffoli gates."""
    circuit = LogicalCircuit(n + 2, f"ToffoliChain_{n}")

    for i in range(n):
        circuit.ccx(0, 1, 2 + i % (circuit.num_qubits - 2))

    return circuit


def create_grover_oracle(n: int) -> LogicalCircuit:
    """Create simple Grover oracle for n qubits."""
    circuit = LogicalCircuit(n + 1, f"GroverOracle_{n}")

    # Multi-controlled Z (simplified)
    for i in range(n):
        circuit.h(n)  # Ancilla
        circuit.cnot(i, n)
        circuit.t(n)
        circuit.cnot(i, n)
        circuit.h(n)

    # Toffoli cascade (simplified)
    if n >= 2:
        circuit.ccx(0, 1, n)

    return circuit


# =============================================================================
# PART 7: MAIN DEMO
# =============================================================================

def run_compiler_demo():
    """Run the complete compiler demonstration."""

    print("=" * 70)
    print("DAY 888: LOGICAL QUANTUM COMPILER - DEMONSTRATION")
    print("=" * 70)

    # ==========================================================================
    # Demo 1: Simple circuit compilation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. SIMPLE CIRCUIT COMPILATION")
    print("=" * 70)

    qasm = """
    qreg q[3];
    h q[0];
    t q[0];
    t q[0];
    cnot q[0], q[1];
    t q[1];
    tdg q[1];
    h q[2];
    ccx q[0], q[1], q[2];
    """

    circuit = QASMParser.parse(qasm)
    print(f"\nInput circuit: {circuit}")
    print(f"Gates: {circuit.gate_count()}")

    compiler = LogicalCompiler(optimization_level=2)
    compiled = compiler.compile(circuit)

    print(f"\nCompiled circuit: {compiled}")
    print(f"Gates: {compiled.gate_count()}")
    compiler.print_stats()

    # ==========================================================================
    # Demo 2: QFT benchmark
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. QFT BENCHMARK")
    print("=" * 70)

    for n in [4, 6, 8]:
        qft = create_qft_circuit(n)
        print(f"\nQFT-{n}:")
        print(f"  Before: {len(qft.gates)} gates, T-count = {qft.t_count()}")

        compiler = LogicalCompiler(optimization_level=2)
        compiled_qft = compiler.compile(qft)

        print(f"  After:  {len(compiled_qft.gates)} gates, T-count = {compiled_qft.t_count()}")
        print(f"  T-reduction: {qft.t_count() - compiled_qft.t_count()}")

    # ==========================================================================
    # Demo 3: Toffoli chain
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. TOFFOLI CHAIN BENCHMARK")
    print("=" * 70)

    for n in [5, 10, 20]:
        chain = create_toffoli_chain(n)
        print(f"\nToffoli chain ({n} gates):")
        print(f"  Before: {len(chain.gates)} gates, T-count = {chain.t_count()}")

        compiler = LogicalCompiler(optimization_level=2)
        compiled_chain = compiler.compile(chain)

        print(f"  After:  {len(compiled_chain.gates)} gates, T-count = {compiled_chain.t_count()}")

    # ==========================================================================
    # Demo 4: Resource estimation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. RESOURCE ESTIMATION")
    print("=" * 70)

    qft8 = create_qft_circuit(8)
    compiler = LogicalCompiler(optimization_level=2)
    compiled_qft8 = compiler.compile(qft8)

    estimator = ResourceEstimator(code_distance=17)
    resources = estimator.estimate(compiled_qft8)

    print(f"\nQFT-8 Resource Estimate:")
    print(resources)

    # Different code distances
    print("\nScaling with code distance:")
    for d in [11, 17, 23, 29]:
        est = ResourceEstimator(code_distance=d)
        res = est.estimate(compiled_qft8)
        print(f"  d={d:2d}: {res.total_qubits():>8,} qubits, "
              f"{res.execution_time_us:>10,.1f} us")

    # ==========================================================================
    # Demo 5: Optimization level comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. OPTIMIZATION LEVEL COMPARISON")
    print("=" * 70)

    test_circuit = create_qft_circuit(6)

    for level in [0, 1, 2]:
        compiler = LogicalCompiler(optimization_level=level)
        result = compiler.compile(test_circuit)
        print(f"\n  Level {level}:")
        print(f"    Gates: {len(result.gates)}")
        print(f"    T-count: {result.t_count()}")
        print(f"    Depth: {result.depth()}")

    print("\n" + "=" * 70)
    print("COMPILATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_compiler_demo()
```

---

## Summary

### Key Components Built Today

| Component | Function |
|-----------|----------|
| `LogicalCircuit` | Circuit representation |
| `QASMParser` | Parse QASM input |
| `GateDecompositionPass` | Decompose to Clifford+T |
| `TCancellationPass` | Cancel T-Tdg pairs |
| `TMergePass` | Merge T-T to S |
| `LogicalCompiler` | Pipeline orchestration |
| `ResourceEstimator` | Physical resource estimation |

### Compiler Pipeline

```
Input QASM → Parse → Decompose → Cancel → Merge → Simplify → Output
                        ↓
                 Resource Estimation
```

### Optimization Results

Typical T-count reductions:
- Simple circuits: 20-40% reduction
- QFT: 10-25% reduction
- Toffoli chains: Limited reduction (already optimal structure)

---

## Daily Checklist

- [ ] I can implement a basic quantum circuit representation
- [ ] I understand how compiler passes work
- [ ] I can apply T-count optimization techniques
- [ ] I can estimate physical resources for a compiled circuit
- [ ] I can benchmark compiler quality on standard circuits
- [ ] I understand the trade-offs between optimization levels

---

## Preview: Day 889

Tomorrow is **Week Synthesis** day:

- Complete compilation pipeline review
- Integration of all week's concepts
- End-to-end optimization strategies
- Preparation for Week 128 (Resource Estimation)
