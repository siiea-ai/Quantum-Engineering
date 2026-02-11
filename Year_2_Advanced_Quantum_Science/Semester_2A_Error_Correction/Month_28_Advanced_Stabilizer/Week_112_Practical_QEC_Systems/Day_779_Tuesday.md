# Day 779: Lattice Surgery Operations

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Merge and split operations theory |
| Afternoon | 2.5 hours | Logical CNOT and multi-qubit gates |
| Evening | 2 hours | Twist defects and scheduling simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain merge and split operations** on surface code patches
2. **Implement logical CNOT** through lattice surgery joint measurements
3. **Analyze time overhead** of lattice surgery versus braiding
4. **Describe twist defect** encoding and operations
5. **Design lattice surgery circuits** for multi-qubit Clifford operations
6. **Optimize surgery schedules** to minimize space-time volume

---

## Core Content

### 1. Introduction to Lattice Surgery

Lattice surgery provides a method for implementing logical operations by manipulating the boundaries of surface code patches. Unlike braiding (which moves defects around each other), lattice surgery **merges and splits** patches to create entanglement.

#### Advantages of Lattice Surgery
- No need for holes or defects within the lattice
- More efficient use of physical qubits
- Simpler routing requirements
- Naturally compatible with planar architectures

### 2. Surface Code Patch Basics

A **surface code patch** is a finite region encoding one logical qubit:

$$\boxed{|\bar{0}\rangle = \frac{1}{\sqrt{2^{d-1}}} \sum_{\text{even weight } z} |z\rangle}$$

The patch has:
- **Rough edges**: Where $Z$-stabilizers terminate (support $\bar{Z}$ operator)
- **Smooth edges**: Where $X$-stabilizers terminate (support $\bar{X}$ operator)

The logical operators are strings connecting opposite edges:

$$\boxed{\bar{Z} = \prod_{i \in \text{rough path}} Z_i, \quad \bar{X} = \prod_{j \in \text{smooth path}} X_j}$$

### 3. Merge Operation

Merging joins two patches by measuring the product of their logical operators across the boundary.

#### Rough Merge (Z-type)

Connect two patches along their rough edges by measuring $Z$-stabilizers across the junction:

$$\boxed{M_{ZZ} = \bar{Z}_1 \bar{Z}_2}$$

The measurement result $m \in \{+1, -1\}$ determines the parity of the merged logical qubits.

**Before merge:** Two independent logical qubits $|\psi_1\rangle |\psi_2\rangle$

**After merge:** Projected state with $\bar{Z}_1\bar{Z}_2 = m$

If initially $|\psi_1\rangle = \alpha|0\rangle + \beta|1\rangle$ and $|\psi_2\rangle = |+\rangle$:

After measuring $\bar{Z}_1\bar{Z}_2 = +1$:
$$|\psi_{\text{merged}}\rangle = \alpha|0\rangle|+\rangle_{ZZ=+1} + \beta|1\rangle|+\rangle_{ZZ=+1} = \frac{1}{\sqrt{2}}(\alpha|00\rangle + \beta|11\rangle + \alpha|01\rangle + \beta|10\rangle)$$

Wait, this simplifies to:
$$|\psi_{\text{merged}}\rangle = \frac{1}{\sqrt{2}}[(\alpha+\beta)|0+\rangle + (\alpha-\beta)|1-\rangle]$$

#### Smooth Merge (X-type)

Connect along smooth edges by measuring $X$-stabilizers:

$$\boxed{M_{XX} = \bar{X}_1 \bar{X}_2}$$

### 4. Split Operation

Splitting reverses the merge by introducing new boundaries and measuring logical operators.

#### Split Process
1. Create new stabilizers along the cut line
2. Measure these stabilizers to establish boundaries
3. Result: Two independent patches, each encoding a logical qubit

The split operation takes $O(d)$ code cycles to establish reliable syndrome information at the new boundary.

### 5. Lattice Surgery CNOT

The CNOT gate via lattice surgery uses an **ancilla patch** prepared in $|+\rangle$:

$$\boxed{\text{CNOT}_{c \to t} = M_{XX}^{c,a} \to M_{ZZ}^{a,t} \to \text{Split}}$$

#### Step-by-Step Protocol

**Initial state:** $|\psi_c\rangle = \alpha|0\rangle + \beta|1\rangle$, $|\psi_t\rangle = |\phi\rangle$, $|a\rangle = |+\rangle$

**Step 1:** Rough merge control and ancilla (measure $\bar{Z}_c\bar{Z}_a$)

If measurement gives $m_1$:
$$|\psi\rangle \to \frac{1}{\sqrt{2}}(\alpha|0\rangle + (-1)^{m_1}\beta|1\rangle)|+\rangle|\phi\rangle + \frac{1}{\sqrt{2}}(\alpha|0\rangle - (-1)^{m_1}\beta|1\rangle)|-\rangle|\phi\rangle$$

Actually, let's be more careful. After measuring $\bar{Z}_c\bar{Z}_a = s_1 \in \{\pm 1\}$:

**Step 2:** Smooth merge ancilla and target (measure $\bar{X}_a\bar{X}_t$)

Result $s_2$: This effectively applies controlled-Z between ancilla and target.

**Step 3:** Split ancilla from both and measure $\bar{X}_a$

Result $s_3$: Apply corrections based on $s_1, s_2, s_3$.

The final result is:
$$\boxed{\text{CNOT}_{c\to t} \cdot |\psi_c\rangle|\psi_t\rangle = \alpha|0\rangle|\phi\rangle + \beta|1\rangle X|\phi\rangle}$$

with Pauli corrections determined by measurement outcomes.

### 6. Time Overhead Analysis

#### CNOT Time Cost

Each merge/split takes $O(d)$ code cycles for reliable syndrome extraction:

$$\boxed{T_{\text{CNOT}} = d \text{ syndrome cycles (per merge/split)}}$$

For a full CNOT with two merges and one split:
$$T_{\text{total}} \approx 3d \text{ cycles}$$

However, operations can be pipelined:

$$\boxed{T_{\text{pipelined}} \approx d \text{ cycles (amortized)}}$$

#### Comparison with Braiding

| Method | Time per CNOT | Space overhead | Parallelism |
|--------|---------------|----------------|-------------|
| Lattice surgery | $O(d)$ | Low | High |
| Defect braiding | $O(d)$ | Higher | Limited |
| Transversal | $O(1)$ | N/A | Perfect |

### 7. Twist Defects

**Twist defects** are topological features that change the type of string operators passing through them.

#### Twist Defect Properties

A twist defect converts:
- $X$-string $\leftrightarrow$ $Z$-string as it passes through

This enables encoding multiple qubits in a single patch with fewer physical qubits.

$$\boxed{\bar{X}_{\text{twist}} = X_1 Z_2 Z_3 \cdots Z_{n-1} X_n}$$

#### Twist Defect Braiding

Moving a twist defect around a logical operator implements:

$$\boxed{\text{Braid}(\text{twist}, \bar{Z}) = \bar{Y}}$$

This provides an alternative route to non-Clifford operations (though still requires magic states for T).

### 8. Multi-Qubit Operations

#### ZZZ Measurement

For a 3-qubit Toffoli decomposition, measure $\bar{Z}\bar{Z}\bar{Z}$:

1. Prepare 3 patches and one ancilla in $|+\rangle$
2. Merge all four patches simultaneously (if geometry permits)
3. Extract the joint parity

#### Arbitrary Pauli Products

Lattice surgery naturally implements measurement of arbitrary Pauli products:

$$\boxed{M_P = P_1 \otimes P_2 \otimes \cdots \otimes P_n}$$

by appropriate choice of merge types (rough vs. smooth) for each patch.

### 9. Routing and Scheduling

In a 2D architecture, patches must be routed for surgery:

**Routing graph:** Patches as nodes, possible surgery connections as edges

**Scheduling problem:** Minimize makespan (total time) subject to:
- Spatial constraints (patches cannot overlap)
- Temporal dependencies (from circuit structure)

$$\boxed{T_{\text{circuit}} = \text{max}_{\text{paths}} (\text{critical path length})}$$

---

## Quantum Mechanics Connection

### Measurement-Based Quantum Computing

Lattice surgery is a form of **measurement-based computation**:
- The resource state is the surface code fabric
- Measurements (merges) drive computation
- Feed-forward corrections complete operations

This connects to the cluster state model where single-qubit measurements on entangled resource states implement quantum gates.

### Topological Protection

The $O(d)$ time requirement ensures that any errors during surgery are detected by repeated syndrome measurements. The merge "anneals" into a stable configuration:

$$p_{\text{surgery error}} \sim (p/p_{\text{th}})^{d/2}$$

---

## Worked Examples

### Example 1: Lattice Surgery Bell State Preparation

**Problem:** Use lattice surgery to prepare $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ from two patches initially in $|0\rangle$ and $|+\rangle$.

**Solution:**

Initial state: $|0\rangle_1 |+\rangle_2 = |0\rangle_1 \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)_2$

Perform rough merge (measure $\bar{Z}_1 \bar{Z}_2$):

The state projects onto the $+1$ or $-1$ eigenspace of $\bar{Z}_1\bar{Z}_2$.

For outcome $+1$ (eigenspace where $Z_1 = Z_2$):
$$|\psi\rangle_{+1} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$

For outcome $-1$ (eigenspace where $Z_1 \neq Z_2$):
$$|\psi\rangle_{-1} = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) = |\Psi^+\rangle$$

Apply $\bar{X}_1$ correction if outcome is $-1$:
$$\bar{X}_1 |\Psi^+\rangle = |\Phi^+\rangle$$

$$\boxed{|\Phi^+\rangle = M_{ZZ}(|0\rangle|+\rangle) \text{ with } X^{(1-m)/2} \text{ correction}}$$

### Example 2: Multi-Control Toffoli via Lattice Surgery

**Problem:** Implement a Toffoli gate on three logical qubits using lattice surgery. Estimate the time overhead at distance $d = 11$.

**Solution:**

The Toffoli gate decomposes into:
$$\text{Toffoli} = H_3 \cdot \text{CCZ} \cdot H_3$$

where CCZ can be implemented via:
1. Four T-gates (from magic states)
2. Plus Clifford operations

Using the T-gate decomposition:
$$\text{CCZ} = T_1 T_2 T_3 T_{12}^\dagger T_{23}^\dagger T_{13}^\dagger T_{123}$$

Actually, a more efficient approach uses measurement:

**Direct CCZ via lattice surgery:**
1. Prepare magic state $|CCZ\rangle$ (via distillation)
2. Measure logical qubits with magic state ancilla

**Time estimate:**
- Each merge/split: $d = 11$ cycles
- Total operations: ~6 merge/split operations
- Total time: $\approx 6 \times 11 = 66$ cycles

At 1 $\mu$s per cycle: $\boxed{T_{\text{Toffoli}} \approx 66 \text{ } \mu\text{s}}$

### Example 3: Scheduling Two Parallel CNOTs

**Problem:** Schedule two CNOT gates, $\text{CNOT}_{1\to2}$ and $\text{CNOT}_{3\to4}$, on a 2D grid where qubits are arranged in a line: 1-2-3-4. Can they execute in parallel?

**Solution:**

Qubit layout: $[1]-[2]-[3]-[4]$

For $\text{CNOT}_{1\to2}$:
- Need ancilla between 1 and 2: $[1]-[a_1]-[2]-[3]-[4]$
- Merge 1-$a_1$, then $a_1$-2

For $\text{CNOT}_{3\to4}$:
- Need ancilla between 3 and 4: $[1]-[2]-[3]-[a_2]-[4]$
- Merge 3-$a_2$, then $a_2$-4

Space requirement: Can fit both ancillas if we expand the grid:
$$[1]-[a_1]-[2]-[3]-[a_2]-[4]$$

**Parallel execution:** Yes, if the grid has sufficient width. The operations don't interfere spatially.

**Time:** Both complete in $\approx 2d$ cycles (two sequential merges).

$$\boxed{\text{Parallel CNOTs: } T = 2d \text{ cycles, independent of number}}$$

---

## Practice Problems

### Level A: Direct Application

**A1.** A surface code patch has distance $d = 7$ with rough edges on top/bottom and smooth edges on left/right. Draw the logical $\bar{Z}$ and $\bar{X}$ operators.

**A2.** Two patches in states $|+\rangle$ and $|0\rangle$ undergo a smooth merge (measuring $\bar{X}_1\bar{X}_2$). What are the possible output states for each measurement outcome?

**A3.** Calculate the time for a lattice surgery CNOT at distance $d = 15$ assuming 1 $\mu$s syndrome cycles.

### Level B: Intermediate Analysis

**B1.** Design a lattice surgery circuit to implement the GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ from three $|0\rangle$ patches plus one ancilla.

**B2.** A circuit has 100 sequential CNOTs forming a chain: $\text{CNOT}_{1\to2}, \text{CNOT}_{2\to3}, \ldots$ Compare the runtime using (a) sequential surgery, (b) pipelined surgery.

**B3.** Explain why a smooth merge followed by a rough split implements an $\bar{X}$ measurement on the separated qubit. Draw the stabilizer flow.

### Level C: Research-Level Challenges

**C1.** Derive the error model for a lattice surgery merge, showing that the logical error probability scales as $(p/p_{\text{th}})^{d/2}$ for the merge duration.

**C2.** Design an optimal lattice surgery schedule for the 4-qubit Clifford circuit: $H_1 \to \text{CNOT}_{1\to2} \to \text{CNOT}_{2\to3} \to \text{CNOT}_{3\to4} \to H_4$. Minimize space-time volume.

**C3.** Analyze how twist defects could reduce the qubit overhead for encoding multiple logical qubits. What is the optimal configuration for 4 logical qubits?

---

## Computational Lab

```python
"""
Day 779: Lattice Surgery Operations
Simulation of merge, split, and logical gate operations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import networkx as nx
from typing import List, Tuple, Dict, Optional

# =============================================================================
# SURFACE CODE PATCH MODEL
# =============================================================================

class SurfaceCodePatch:
    """Represents a logical qubit in a surface code patch."""

    def __init__(self, name: str, position: Tuple[int, int], distance: int = 5):
        """
        Initialize a surface code patch.

        Args:
            name: Identifier for this patch
            position: (x, y) grid position
            distance: Code distance
        """
        self.name = name
        self.position = position
        self.distance = distance
        self.state = np.array([1, 0], dtype=complex)  # |0⟩
        self.merged_with = None

    def apply_gate(self, gate: np.ndarray):
        """Apply a single-qubit gate."""
        self.state = gate @ self.state
        self.state /= np.linalg.norm(self.state)

    def measure_z(self) -> int:
        """Measure in Z basis, return 0 or 1."""
        prob_0 = np.abs(self.state[0])**2
        outcome = 0 if np.random.random() < prob_0 else 1
        self.state = np.array([1, 0]) if outcome == 0 else np.array([0, 1])
        return outcome

    def __repr__(self):
        return f"Patch({self.name}, pos={self.position}, d={self.distance})"


class LatticeSurgerySimulator:
    """Simulate lattice surgery operations."""

    # Standard gates
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    def __init__(self, distance: int = 5, cycle_time_us: float = 1.0):
        """
        Initialize simulator.

        Args:
            distance: Default code distance
            cycle_time_us: Time per syndrome cycle in microseconds
        """
        self.distance = distance
        self.cycle_time = cycle_time_us
        self.patches: Dict[str, SurfaceCodePatch] = {}
        self.time_elapsed = 0  # in cycles

    def create_patch(self, name: str, position: Tuple[int, int],
                     initial_state: str = '0') -> SurfaceCodePatch:
        """Create a new surface code patch."""
        patch = SurfaceCodePatch(name, position, self.distance)

        if initial_state == '+':
            patch.apply_gate(self.H)
        elif initial_state == '-':
            patch.apply_gate(self.H)
            patch.apply_gate(self.Z)
        elif initial_state == '1':
            patch.apply_gate(self.X)

        self.patches[name] = patch
        return patch

    def rough_merge(self, patch1: str, patch2: str) -> int:
        """
        Perform rough merge (ZZ measurement).

        Returns:
            Measurement outcome: +1 or -1
        """
        p1 = self.patches[patch1]
        p2 = self.patches[patch2]

        # Time cost
        self.time_elapsed += self.distance

        # Joint state
        state_joint = np.kron(p1.state, p2.state)

        # ZZ eigenprojectors
        ZZ = np.kron(self.Z, self.Z)
        P_plus = (np.eye(4) + ZZ) / 2  # ZZ = +1 subspace
        P_minus = (np.eye(4) - ZZ) / 2  # ZZ = -1 subspace

        prob_plus = np.real(state_joint.conj() @ P_plus @ state_joint)

        if np.random.random() < prob_plus:
            outcome = +1
            projected = P_plus @ state_joint
        else:
            outcome = -1
            projected = P_minus @ state_joint

        projected /= np.linalg.norm(projected)

        # Store merged state in first patch (simplified model)
        p1.state = projected[:2] if np.linalg.norm(projected[:2]) > 0.5 else projected[2:]
        p1.state /= np.linalg.norm(p1.state)
        p1.merged_with = patch2

        return outcome

    def smooth_merge(self, patch1: str, patch2: str) -> int:
        """
        Perform smooth merge (XX measurement).

        Returns:
            Measurement outcome: +1 or -1
        """
        p1 = self.patches[patch1]
        p2 = self.patches[patch2]

        self.time_elapsed += self.distance

        state_joint = np.kron(p1.state, p2.state)

        XX = np.kron(self.X, self.X)
        P_plus = (np.eye(4) + XX) / 2
        P_minus = (np.eye(4) - XX) / 2

        prob_plus = np.real(state_joint.conj() @ P_plus @ state_joint)

        if np.random.random() < prob_plus:
            outcome = +1
            projected = P_plus @ state_joint
        else:
            outcome = -1
            projected = P_minus @ state_joint

        projected /= np.linalg.norm(projected)

        # For XX, transform to computational basis interpretation
        p1.state = projected[:2]
        if np.linalg.norm(p1.state) < 0.1:
            p1.state = projected[2:]
        p1.state /= np.linalg.norm(p1.state)
        p1.merged_with = patch2

        return outcome

    def split(self, patch: str) -> None:
        """Split a merged patch back into independent qubits."""
        self.time_elapsed += self.distance
        self.patches[patch].merged_with = None

    def lattice_surgery_cnot(self, control: str, target: str,
                             ancilla: str = 'anc') -> Dict:
        """
        Implement CNOT via lattice surgery.

        Args:
            control: Control qubit patch name
            target: Target qubit patch name
            ancilla: Ancilla patch name (will be created)

        Returns:
            Dictionary with measurement outcomes and corrections
        """
        # Create ancilla in |+⟩
        ctrl_pos = self.patches[control].position
        anc_pos = (ctrl_pos[0] + 1, ctrl_pos[1])
        self.create_patch(ancilla, anc_pos, '+')

        # Step 1: Rough merge control-ancilla
        m1 = self.rough_merge(control, ancilla)

        # Step 2: Smooth merge ancilla-target
        m2 = self.smooth_merge(ancilla, target)

        # Step 3: Split and measure ancilla in X
        self.split(ancilla)
        self.patches[ancilla].apply_gate(self.H)
        m3 = self.patches[ancilla].measure_z()

        # Apply corrections based on outcomes
        corrections = {'Z_control': (m2 == -1), 'X_target': (m3 == 1)}

        if corrections['Z_control']:
            self.patches[control].apply_gate(self.Z)
        if corrections['X_target']:
            self.patches[target].apply_gate(self.X)

        return {
            'm1': m1, 'm2': m2, 'm3': m3,
            'corrections': corrections,
            'time': 3 * self.distance
        }

    def get_time_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.time_elapsed * self.cycle_time


def simulate_bell_state_preparation():
    """Demonstrate Bell state preparation via lattice surgery."""

    print("=" * 60)
    print("BELL STATE PREPARATION VIA LATTICE SURGERY")
    print("=" * 60)

    sim = LatticeSurgerySimulator(distance=7)

    # Create patches
    sim.create_patch('A', (0, 0), '0')
    sim.create_patch('B', (2, 0), '+')

    print(f"\nInitial states:")
    print(f"  Patch A: |0⟩")
    print(f"  Patch B: |+⟩")

    # Rough merge
    outcome = sim.rough_merge('A', 'B')

    print(f"\nRough merge (ZZ measurement):")
    print(f"  Outcome: {'+' if outcome == 1 else '-'}1")
    print(f"  Time: {sim.time_elapsed} cycles")

    if outcome == -1:
        print("  Applying X correction to patch A")
        sim.patches['A'].apply_gate(sim.X)

    print(f"\nFinal state: |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)")
    print(f"Total time: {sim.get_time_us()} μs")


def simulate_cnot_gate():
    """Demonstrate CNOT via lattice surgery."""

    print("\n" + "=" * 60)
    print("CNOT VIA LATTICE SURGERY")
    print("=" * 60)

    sim = LatticeSurgerySimulator(distance=11)

    # Create control in superposition, target in |0⟩
    sim.create_patch('ctrl', (0, 0), '+')  # |+⟩ = (|0⟩ + |1⟩)/√2
    sim.create_patch('tgt', (4, 0), '0')

    print(f"\nInitial states:")
    print(f"  Control: |+⟩")
    print(f"  Target: |0⟩")
    print(f"  Expected result: (|00⟩ + |11⟩)/√2 = |Φ⁺⟩")

    # Perform CNOT
    result = sim.lattice_surgery_cnot('ctrl', 'tgt')

    print(f"\nLattice surgery CNOT:")
    print(f"  ZZ measurement (ctrl-anc): {'+' if result['m1'] == 1 else '-'}1")
    print(f"  XX measurement (anc-tgt): {'+' if result['m2'] == 1 else '-'}1")
    print(f"  X measurement (anc): {result['m3']}")
    print(f"\nCorrections applied:")
    print(f"  Z on control: {result['corrections']['Z_control']}")
    print(f"  X on target: {result['corrections']['X_target']}")
    print(f"\nTotal time: {result['time']} cycles = {result['time'] * sim.cycle_time} μs")


def scheduling_analysis():
    """Analyze scheduling of multiple lattice surgery operations."""

    print("\n" + "=" * 60)
    print("LATTICE SURGERY SCHEDULING ANALYSIS")
    print("=" * 60)

    # Create a simple circuit graph
    G = nx.DiGraph()

    # Circuit: 4 CNOTs with dependencies
    # CNOT(1,2) -> CNOT(2,3) -> CNOT(3,4)
    #           -> CNOT(1,3) -> CNOT(3,4)
    G.add_edge('CNOT_12', 'CNOT_23')
    G.add_edge('CNOT_23', 'CNOT_34')
    G.add_edge('CNOT_12', 'CNOT_13')
    G.add_edge('CNOT_13', 'CNOT_34')

    print("\nCircuit dependency graph:")
    print("  CNOT(1,2) → CNOT(2,3) → CNOT(3,4)")
    print("          ↘ CNOT(1,3) ↗")

    # Find critical path
    critical_path = nx.dag_longest_path(G)
    print(f"\nCritical path: {' → '.join(critical_path)}")
    print(f"Critical path length: {len(critical_path)} operations")

    d = 11  # Code distance
    cnot_time = 3 * d  # cycles

    sequential_time = len(G.nodes) * cnot_time
    parallel_time = len(critical_path) * cnot_time

    print(f"\nTime analysis (d = {d}):")
    print(f"  Sequential: {len(G.nodes)} × {cnot_time} = {sequential_time} cycles")
    print(f"  Parallel: {len(critical_path)} × {cnot_time} = {parallel_time} cycles")
    print(f"  Speedup: {sequential_time / parallel_time:.2f}×")


def visualize_lattice_surgery():
    """Create visualization of lattice surgery operations."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    def draw_patch(ax, x, y, w, h, color, label=''):
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=12, fontweight='bold')

    # 1. Initial patches
    ax = axes[0, 0]
    draw_patch(ax, 0, 0, 2, 2, 'lightblue', 'Ctrl\n|ψ⟩')
    draw_patch(ax, 4, 0, 2, 2, 'lightgreen', 'Tgt\n|φ⟩')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 0: Initial Patches', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 2. Ancilla added
    ax = axes[0, 1]
    draw_patch(ax, 0, 0, 2, 2, 'lightblue', 'Ctrl')
    draw_patch(ax, 2.5, 0, 1, 2, 'lightyellow', 'Anc\n|+⟩')
    draw_patch(ax, 4, 0, 2, 2, 'lightgreen', 'Tgt')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 1: Add Ancilla in |+⟩', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 3. ZZ merge
    ax = axes[0, 2]
    draw_patch(ax, 0, 0, 3.5, 2, 'lightcoral', 'Ctrl + Anc\n(ZZ merge)')
    draw_patch(ax, 4, 0, 2, 2, 'lightgreen', 'Tgt')
    ax.annotate('', xy=(3.5, 1), xytext=(4, 1),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 2: Rough Merge (ZZ)', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 4. XX merge
    ax = axes[1, 0]
    draw_patch(ax, 0, 0, 2, 2, 'lightblue', 'Ctrl')
    draw_patch(ax, 2.5, 0, 3.5, 2, 'lightsalmon', 'Anc + Tgt\n(XX merge)')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 3: Smooth Merge (XX)', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. Split and measure
    ax = axes[1, 1]
    draw_patch(ax, 0, 0, 2, 2, 'lightblue', 'Ctrl\'')
    draw_patch(ax, 2.5, 0, 1, 2, 'gray', 'Anc\n(meas)')
    draw_patch(ax, 4, 0, 2, 2, 'lightgreen', 'Tgt\'')
    ax.annotate('X basis', xy=(3, -0.3), fontsize=10, ha='center')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 4: Split & Measure Ancilla', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 6. Final result
    ax = axes[1, 2]
    draw_patch(ax, 0, 0, 2, 2, 'lightblue', 'Ctrl\'')
    draw_patch(ax, 4, 0, 2, 2, 'lightgreen', 'Tgt\'')
    ax.annotate('CNOT(Ctrl, Tgt)', xy=(3, 1), fontsize=12, ha='center',
               fontweight='bold')
    ax.annotate('+ Pauli corrections', xy=(3, 0.3), fontsize=10, ha='center')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 3)
    ax.set_title('Step 5: CNOT Complete!', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('Lattice Surgery CNOT Protocol', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('day_779_lattice_surgery.png', dpi=150, bbox_inches='tight')
    plt.show()


def time_overhead_comparison():
    """Compare time overhead across methods."""

    fig, ax = plt.subplots(figsize=(10, 6))

    distances = np.arange(3, 31, 2)

    # Lattice surgery: O(d) per operation
    ls_time = 3 * distances  # 3 merge/split ops per CNOT

    # Braiding with defects: O(d) but larger constant
    braid_time = 5 * distances

    # Transversal (if available): O(1)
    trans_time = np.ones_like(distances) * 5

    ax.plot(distances, ls_time, 'b-o', label='Lattice Surgery', linewidth=2)
    ax.plot(distances, braid_time, 'r-s', label='Defect Braiding', linewidth=2)
    ax.plot(distances, trans_time, 'g--', label='Transversal (if available)', linewidth=2)

    ax.set_xlabel('Code Distance d', fontsize=12)
    ax.set_ylabel('Time (code cycles)', fontsize=12)
    ax.set_title('CNOT Time Overhead Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_779_time_overhead.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Day 779: Lattice Surgery Operations")
    print("=" * 60)

    # Run demonstrations
    simulate_bell_state_preparation()
    simulate_cnot_gate()
    scheduling_analysis()

    # Generate visualizations
    visualize_lattice_surgery()
    time_overhead_comparison()

    print("\n" + "=" * 60)
    print("SUMMARY: Lattice Surgery Key Metrics")
    print("=" * 60)

    d = 11
    print(f"\nFor code distance d = {d}:")
    print(f"  Single merge/split time: {d} cycles")
    print(f"  CNOT time: {3*d} cycles (3 operations)")
    print(f"  Pipelined CNOT: {d} cycles (amortized)")

    print("\nKey advantages:")
    print("  - No defects inside patches")
    print("  - Efficient use of 2D qubit array")
    print("  - Natural parallel execution")
    print("  - Compatible with planar architectures")
```

---

## Summary

### Key Formulas

| Operation | Time Cost | Description |
|-----------|-----------|-------------|
| Single merge/split | $d$ cycles | Establish boundary stabilizers |
| ZZ measurement | $d$ cycles | Rough merge (Z-string connection) |
| XX measurement | $d$ cycles | Smooth merge (X-string connection) |
| Lattice surgery CNOT | $3d$ cycles | Two merges + one split |
| Pipelined CNOT | $d$ cycles | Amortized over many operations |

### Main Takeaways

1. **Merge creates entanglement**: Joint Pauli measurements on patches create logical entanglement
2. **Split restores independence**: New boundaries separate logical qubits
3. **CNOT uses ancilla patch**: Three-step protocol with measurement-based corrections
4. **Time scales with distance**: Each surgery operation takes $O(d)$ cycles
5. **Scheduling enables parallelism**: Independent operations execute concurrently
6. **Twist defects offer alternatives**: Different encoding with potentially lower overhead

---

## Daily Checklist

- [ ] I can explain rough and smooth merge operations
- [ ] I understand the lattice surgery CNOT protocol
- [ ] I can calculate time overhead for surgery circuits
- [ ] I understand twist defect encoding
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 780

Tomorrow we study **Code Switching and Gauge Fixing**, techniques for accessing gates not natively transversal in a single code:
- Switching between Steane and Reed-Muller codes
- Gauge fixing in subsystem codes
- Trade-offs between code switching and magic states

*"When one code cannot do it all, we choreograph a dance between codes."*

---

*Day 779 of 2184 | Year 2, Month 28, Week 112, Day 2*
*Quantum Engineering PhD Curriculum*
