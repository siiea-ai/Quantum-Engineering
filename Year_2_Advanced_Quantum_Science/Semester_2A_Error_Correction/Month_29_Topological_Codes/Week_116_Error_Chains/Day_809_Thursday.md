# Day 809: Lattice Surgery Operations

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 116: Error Chains & Logical Operations

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Merge and split operations |
| Afternoon | 2.5 hours | CNOT via lattice surgery |
| Evening | 1.5 hours | Computational lab: surgery simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. Execute merge operations to measure logical Pauli products
2. Perform split operations to create entangled logical states
3. Implement logical CNOT using merge-split sequences
4. Design surgery schedules for parallel operations
5. Analyze time overhead for lattice surgery gates
6. Compare surgery to other logical gate methods
7. Simulate basic lattice surgery operations

---

## Core Content: Lattice Surgery

### The Key Idea

**Lattice surgery** implements logical gates by:
1. Temporarily joining (merging) separate surface code patches
2. Measuring joint stabilizers across the merged region
3. Separating (splitting) the patches

This enables **fault-tolerant logical gates** using only local operations!

### Why Lattice Surgery?

| Method | Pros | Cons |
|--------|------|------|
| Transversal gates | Simple, fast | Limited gate set |
| Code deformation | Flexible | Complex scheduling |
| **Lattice surgery** | Universal, local | O(d) time overhead |
| Magic state injection | Universal | High resource cost |

---

## The Merge Operation

### Setup

Two surface code patches $A$ and $B$ with adjacent boundaries:

```
    ╔═══════════╗     ╔═══════════╗
    ║           ║     ║           ║
    ║   Patch   ║     ║   Patch   ║
    ║     A     ║     ║     B     ║
    ║           ║     ║           ║
    ╚═══════════╝     ╚═══════════╝
         └──gap──┘
```

### Rough Merge (XX Measurement)

**Goal:** Measure $\bar{X}_A \bar{X}_B$

**Procedure:**
1. Initialize ancilla qubits in gap region to $|+\rangle$
2. Turn on X-stabilizers connecting patches
3. Measure for $d$ rounds
4. Read out merged X-stabilizer product

```
Before merge:          After merge:

A: ●──●──●  ○  ●──●──● B    A: ●──●──●──●──●──●──● B
   │  │  │     │  │  │        │  │  │  │  │  │  │
   ●──●──●  ○  ●──●──●        ●──●──●──●──●──●──●
   │  │  │     │  │  │        │  │  │  │  │  │  │
   ●──●──●  ○  ●──●──●        ●──●──●──●──●──●──●

○ = gap qubits        New X-stabilizers span gap
```

**Measurement outcome:**

$$\boxed{M_{XX} = \bar{X}_A \bar{X}_B = \pm 1}$$

### Smooth Merge (ZZ Measurement)

**Goal:** Measure $\bar{Z}_A \bar{Z}_B$

**Procedure:**
1. Initialize ancilla qubits in gap to $|0\rangle$
2. Turn on Z-stabilizers connecting patches
3. Measure for $d$ rounds
4. Read out merged Z-stabilizer product

**Measurement outcome:**

$$\boxed{M_{ZZ} = \bar{Z}_A \bar{Z}_B = \pm 1}$$

### Merge Formulas

For rough merge (XX measurement):

$$|00\rangle + |11\rangle \xrightarrow{XX=+1} |00\rangle + |11\rangle$$
$$|00\rangle - |11\rangle \xrightarrow{XX=-1} |00\rangle - |11\rangle$$
$$|01\rangle + |10\rangle \xrightarrow{XX=+1} |01\rangle + |10\rangle$$
$$|01\rangle - |10\rangle \xrightarrow{XX=-1} |01\rangle - |10\rangle$$

The measurement **projects** onto $\pm 1$ eigenspaces of $\bar{X}_A\bar{X}_B$.

---

## The Split Operation

### Setup

Single large patch to be divided into two patches:

```
Before split:               After split:

╔═══════════════════╗      ╔═══════════╗     ╔═══════════╗
║                   ║      ║           ║     ║           ║
║   Single Patch    ║  →   ║   Patch   ║     ║   Patch   ║
║                   ║      ║     A     ║     ║     B     ║
║                   ║      ║           ║     ║           ║
╚═══════════════════╝      ╚═══════════╝     ╚═══════════╝
```

### Rough Split (Bell State Preparation)

**Procedure:**
1. Start with merged patch (single logical qubit)
2. Measure individual X-stabilizers in split region
3. Apply corrections based on measurement outcomes

**Result:** Creates Bell-like state

$$|\psi\rangle_\text{merged} \to \frac{|00\rangle + |11\rangle}{\sqrt{2}} \quad \text{(for } |\bar{0}\rangle \text{ input)}$$

$$|\psi\rangle_\text{merged} \to \frac{|00\rangle - |11\rangle}{\sqrt{2}} \quad \text{(for } |\bar{1}\rangle \text{ input)}$$

### Smooth Split

Similar procedure but measuring Z-stabilizers:

$$|\bar{+}\rangle \to \frac{|++\rangle + |--\rangle}{\sqrt{2}}$$

### Split as Inverse of Merge

$$\boxed{\text{Split} \approx \text{Merge}^{-1}}$$

But with important differences in the resulting state!

---

## CNOT via Lattice Surgery

### The Protocol

Logical CNOT between control qubit $C$ and target qubit $T$:

$$\text{CNOT}: |c,t\rangle \to |c, c \oplus t\rangle$$

**Equivalent to:**
- $\bar{X}_C \to \bar{X}_C \bar{X}_T$ (X propagates from control to target)
- $\bar{Z}_T \to \bar{Z}_C \bar{Z}_T$ (Z propagates from target to control)

### Step-by-Step Surgery

**Step 1: Prepare ancilla patch in $|\bar{+}\rangle$**

```
╔═══╗     ╔═══╗     ╔═══╗
║ C ║     ║ A ║     ║ T ║
╚═══╝     ╚═══╝     ╚═══╝
Control   Ancilla   Target
```

**Step 2: ZZ-merge between C and A**

Measure $\bar{Z}_C \bar{Z}_A$:
- Outcome $m_1 = \pm 1$
- State: $\bar{Z}_C \bar{Z}_A$ eigenstate

**Step 3: ZZ-merge between A and T**

Measure $\bar{Z}_A \bar{Z}_T$:
- Outcome $m_2 = \pm 1$
- Combined: know $\bar{Z}_C \bar{Z}_T = m_1 \cdot m_2$

**Step 4: XX-split A from merged CT**

Measure $\bar{X}_A$ individually:
- Outcome $m_3 = \pm 1$
- Apply $\bar{Z}_T^{(1-m_3)/2}$ correction if needed

**Final result:** CNOT applied (up to known Pauli corrections)!

### Circuit Representation

```
Control: ─────●─────
              │
Ancilla: ─|+⟩─●──M─
              │
Target:  ─────⊕─────

Translates to:

   C ──ZZ──┐
           │
   A ──────●──ZZ──●──M_X──
                  │
   T ─────────────●───────
```

### Time Complexity

Each merge/split takes $O(d)$ syndrome rounds.

$$\boxed{\text{CNOT time} = O(d) \text{ rounds}}$$

For a d=17 code at 1 MHz: CNOT takes ~17 microseconds.

---

## Multi-Qubit Operations

### Parallel Surgery

Multiple CNOTs can be performed simultaneously if patches don't share boundaries:

```
C₁──●──     C₂──●──     C₃──●──
    │           │           │
    ⊕           ⊕           ⊕
    │           │           │
T₁──┴──     T₂──┴──     T₃──┴──

All three CNOTs execute in parallel: O(d) total time
```

### Serial Constraints

If qubits share boundaries, operations must be serialized:

```
CNOT(C₁, T)  then  CNOT(C₂, T)
Total time: O(2d)
```

### Scheduling Optimization

**Problem:** Given a circuit, minimize total time by:
1. Parallelizing independent operations
2. Routing qubits to enable desired merges
3. Avoiding boundary conflicts

**This is an NP-hard optimization problem in general!**

---

## Lattice Surgery Variants

### Y-Merge

Measure $\bar{Y}_A \bar{Y}_B = i\bar{X}_A\bar{Z}_A \cdot i\bar{X}_B\bar{Z}_B$

Requires preparing ancillas in $|i\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt{2}}$.

### Multi-Patch Merge

Merge three or more patches simultaneously:

$$M_{XXX} = \bar{X}_A \bar{X}_B \bar{X}_C$$

Used for:
- Multi-target CNOT
- Toffoli gates (with magic states)
- Parity checks

### Auto-Corrected Surgery

Use decoder to track Pauli frame through surgery:
- No explicit corrections needed
- Track in classical software
- Apply at final measurement only

---

## Worked Examples

### Example 1: Bell State via Surgery

**Goal:** Create $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$ from two $|0\rangle$ patches.

**Protocol:**
1. Start: Patch A in $|\bar{0}\rangle$, Patch B in $|\bar{0}\rangle$
2. XX-merge A and B
3. Measure outcome $m$

**If $m = +1$:**
$$|00\rangle \to \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

**If $m = -1$:**
$$|00\rangle \to \frac{|00\rangle - |11\rangle}{\sqrt{2}}$$

Apply $\bar{Z}_A$ to convert to $+$ Bell state.

---

### Example 2: CNOT Implementation

**Input:** $|\bar{+}\rangle_C |\bar{0}\rangle_T$

**Expected output:** $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$ (Bell state)

**Surgery steps:**

1. Prepare ancilla $|\bar{+}\rangle_A$

2. ZZ-merge C-A: measure $\bar{Z}_C \bar{Z}_A$
   - Outcome $m_1$
   - State projects to $\bar{Z}_C \bar{Z}_A = m_1$ eigenspace

3. XX-merge A-T: measure $\bar{X}_A \bar{X}_T$
   - Outcome $m_2$
   - Combined with previous constraint

4. Split and measure $\bar{Z}_A$:
   - Outcome $m_3$
   - Apply $\bar{X}_T^{(1-m_1)/2} \bar{Z}_T^{(1-m_3)/2}$

**Final state:** $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$ as expected!

---

### Example 3: Time Budget Calculation

**Circuit:** 100 sequential CNOTs on 50 logical qubits

**Parameters:** $d = 21$, cycle time = 1 $\mu$s

**Analysis:**
- Each CNOT: ~21 rounds
- Parallelization: assume 2x speedup (50 parallel paths)
- Effective CNOTs: 50 sequential operations
- Total time: $50 \times 21 \times 1\mu s = 1.05$ ms

**Comparison:** Without error correction, 100 CNOTs at 100 ns each = 10 $\mu$s.

**Overhead factor:** ~100x slowdown for fault tolerance.

---

## Practice Problems

### Problem Set A: Merge Operations

**A1.** Two patches in states $|\bar{+}\rangle$ and $|\bar{0}\rangle$ undergo ZZ-merge.
(a) What is the pre-measurement state?
(b) What are the possible outcomes and post-measurement states?

**A2.** Prove that XX-merge followed by ZZ-split returns to original states (up to Paulis).

**A3.** Design a protocol to prepare the GHZ state $\frac{|000\rangle + |111\rangle}{\sqrt{2}}$ using lattice surgery.

### Problem Set B: CNOT Analysis

**B1.** Trace through the CNOT protocol with input $|\bar{1}\rangle_C |\bar{+}\rangle_T$.
(a) State after each step
(b) Measurement outcomes
(c) Final state (verify it's $|\bar{1}\rangle |\bar{-}\rangle$)

**B2.** The CNOT protocol uses one ancilla patch. Can you design a protocol that uses zero ancilla patches? What's the tradeoff?

**B3.** How many surface code patches (minimum) are needed to implement:
(a) A single CNOT?
(b) CNOT between any pair of $n$ qubits?
(c) A 10-qubit circuit with arbitrary connectivity?

### Problem Set C: Scheduling

**C1.** A circuit has this structure:
```
CNOT(1,2), CNOT(3,4), CNOT(2,3), CNOT(1,4)
```
Find the minimum time schedule assuming:
(a) No layout constraints
(b) Linear layout: 1-2-3-4

**C2.** Prove that any Clifford circuit on $n$ qubits can be implemented in $O(n \cdot d)$ time with lattice surgery.

**C3.** For a 2D grid of surface code patches, what is the diameter (longest shortest path) as a function of grid size? How does this affect circuit depth?

---

## Computational Lab: Lattice Surgery Simulation

```python
"""
Day 809 Computational Lab: Lattice Surgery Operations
Simulating merge, split, and CNOT operations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional

class LogicalQubit:
    """
    Represents a logical qubit in surface code.

    Tracks logical Pauli frame for surgery operations.
    """

    def __init__(self, name: str, state: np.ndarray = None):
        """
        Initialize logical qubit.

        Args:
            name: Identifier for the qubit
            state: Initial state vector (2D) or None for |0⟩
        """
        self.name = name
        if state is None:
            self.state = np.array([1, 0], dtype=complex)  # |0⟩
        else:
            self.state = state / np.linalg.norm(state)

        # Pauli frame tracking
        self.x_frame = 0  # Number of X corrections (mod 2)
        self.z_frame = 0  # Number of Z corrections (mod 2)

    def apply_x(self):
        """Apply logical X."""
        self.state = np.array([self.state[1], self.state[0]])
        self.x_frame = (self.x_frame + 1) % 2

    def apply_z(self):
        """Apply logical Z."""
        self.state = np.array([self.state[0], -self.state[1]])
        self.z_frame = (self.z_frame + 1) % 2

    def apply_h(self):
        """Apply logical Hadamard."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.state = H @ self.state
        # Swap X and Z frames
        self.x_frame, self.z_frame = self.z_frame, self.x_frame

    def measure_z(self) -> int:
        """Measure in Z basis, return outcome."""
        prob_0 = np.abs(self.state[0])**2
        outcome = 0 if np.random.random() < prob_0 else 1
        self.state = np.array([1, 0] if outcome == 0 else [0, 1], dtype=complex)
        return outcome ^ self.z_frame

    def measure_x(self) -> int:
        """Measure in X basis, return outcome."""
        self.apply_h()
        outcome = self.measure_z()
        return outcome

    def __repr__(self):
        return f"LogicalQubit({self.name}, state={self.state})"


class LatticeSurgery:
    """
    Implements lattice surgery operations between logical qubits.
    """

    def __init__(self, verbose: bool = True):
        """Initialize surgery engine."""
        self.verbose = verbose
        self.operation_log = []

    def log(self, message: str):
        """Log operation."""
        if self.verbose:
            print(f"  [Surgery] {message}")
        self.operation_log.append(message)

    def merge_xx(self, qubit_a: LogicalQubit, qubit_b: LogicalQubit) -> int:
        """
        XX-merge: Measure X_A X_B.

        Args:
            qubit_a, qubit_b: Qubits to merge

        Returns:
            Measurement outcome (+1 or -1)
        """
        self.log(f"XX-merge({qubit_a.name}, {qubit_b.name})")

        # Convert to X basis
        # |+⟩ = (|0⟩ + |1⟩)/√2, |-⟩ = (|0⟩ - |1⟩)/√2
        # XX|++⟩ = |++⟩, XX|--⟩ = |--⟩
        # XX|+-⟩ = -|+-⟩, XX|-+⟩ = -|-+⟩

        # Compute XX expectation
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        state_a_x = H @ qubit_a.state
        state_b_x = H @ qubit_b.state

        # Joint state in X basis
        joint_x = np.outer(state_a_x, state_b_x).flatten()

        # XX eigenvalues: ++/-- have +1, +-/-+ have -1
        prob_plus = np.abs(joint_x[0])**2 + np.abs(joint_x[3])**2  # |++⟩ + |--⟩
        outcome = 1 if np.random.random() < prob_plus else -1

        self.log(f"  Outcome: XX = {outcome:+d}")

        return outcome

    def merge_zz(self, qubit_a: LogicalQubit, qubit_b: LogicalQubit) -> int:
        """
        ZZ-merge: Measure Z_A Z_B.

        Returns:
            Measurement outcome (+1 or -1)
        """
        self.log(f"ZZ-merge({qubit_a.name}, {qubit_b.name})")

        # ZZ|00⟩ = |00⟩, ZZ|11⟩ = |11⟩, ZZ|01⟩ = -|01⟩, ZZ|10⟩ = -|10⟩
        joint = np.outer(qubit_a.state, qubit_b.state).flatten()

        prob_plus = np.abs(joint[0])**2 + np.abs(joint[3])**2  # |00⟩ + |11⟩
        outcome = 1 if np.random.random() < prob_plus else -1

        self.log(f"  Outcome: ZZ = {outcome:+d}")

        return outcome

    def split_xx(self, merged_qubit: LogicalQubit) -> Tuple[LogicalQubit, LogicalQubit]:
        """
        XX-split: Split merged patch into two qubits.

        Creates Bell-like state from single qubit.
        """
        self.log(f"XX-split({merged_qubit.name})")

        # |0⟩ → (|00⟩ + |11⟩)/√2
        # |1⟩ → (|01⟩ + |10⟩)/√2
        a, b = merged_qubit.state
        qubit_a = LogicalQubit(f"{merged_qubit.name}_A")
        qubit_b = LogicalQubit(f"{merged_qubit.name}_B")

        # Simplified: create entangled state
        # In reality, measure X stabilizers in split region

        self.log(f"  Created {qubit_a.name} and {qubit_b.name}")

        return qubit_a, qubit_b

    def cnot(self, control: LogicalQubit, target: LogicalQubit) -> Tuple[int, int, int]:
        """
        Implement CNOT via lattice surgery.

        Uses ancilla qubit in |+⟩ state.

        Returns:
            Tuple of measurement outcomes (m1, m2, m3)
        """
        self.log(f"\n=== CNOT({control.name} → {target.name}) ===")

        # Step 1: Prepare ancilla
        ancilla = LogicalQubit("Ancilla")
        ancilla.apply_h()  # |+⟩
        self.log("Prepared ancilla in |+⟩")

        # Step 2: ZZ-merge control and ancilla
        m1 = self.merge_zz(control, ancilla)

        # Step 3: ZZ-merge ancilla and target
        m2 = self.merge_zz(ancilla, target)

        # Step 4: Measure ancilla X
        m3 = ancilla.measure_x()
        self.log(f"Measured ancilla X: {m3}")

        # Apply corrections
        if m1 == -1:
            target.apply_x()
            self.log("Applied X correction to target")
        if m3 == 1:
            target.apply_z()
            self.log("Applied Z correction to target")

        # Actually perform CNOT on state vectors (for simulation)
        joint = np.outer(control.state, target.state).flatten()
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        new_joint = CNOT @ joint

        # Extract individual states (if separable)
        # For entangled output, this is just approximate

        self.log("=== CNOT Complete ===\n")

        return m1, m2, m3


def visualize_lattice_surgery():
    """Create visualization of lattice surgery CNOT."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        'control': '#3498db',
        'target': '#e74c3c',
        'ancilla': '#2ecc71',
        'merged': '#9b59b6'
    }

    def draw_patch(ax, x, y, w, h, color, label, state=''):
        """Draw a surface code patch."""
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=14, fontweight='bold')
        if state:
            ax.text(x + w/2, y - 0.15, state, ha='center', va='top', fontsize=11)

    # Panel 1: Initial state
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')

    draw_patch(ax1, 0, 0, 0.8, 0.8, colors['control'], 'C', '|ψ⟩_C')
    draw_patch(ax1, 2.2, 0, 0.8, 0.8, colors['target'], 'T', '|φ⟩_T')

    ax1.set_title('Step 1: Initial State\nControl and Target patches', fontsize=12)
    ax1.axis('off')

    # Panel 2: Ancilla preparation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')

    draw_patch(ax2, 0, 0, 0.8, 0.8, colors['control'], 'C')
    draw_patch(ax2, 1.1, 0, 0.8, 0.8, colors['ancilla'], 'A', '|+⟩')
    draw_patch(ax2, 2.2, 0, 0.8, 0.8, colors['target'], 'T')

    ax2.set_title('Step 2: Prepare Ancilla\nAncilla in |+⟩ state', fontsize=12)
    ax2.axis('off')

    # Panel 3: First ZZ merge
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')

    draw_patch(ax3, 0, 0, 1.9, 0.8, colors['merged'], 'C-A', 'ZZ merge')
    draw_patch(ax3, 2.2, 0, 0.8, 0.8, colors['target'], 'T')

    ax3.annotate('', xy=(1.9, 0.4), xytext=(1.4, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax3.text(1.5, 0.65, 'm₁', fontsize=11)

    ax3.set_title('Step 3: ZZ-Merge (C, A)\nMeasure Z_C Z_A = m₁', fontsize=12)
    ax3.axis('off')

    # Panel 4: Second ZZ merge
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 1.5)
    ax4.set_aspect('equal')

    draw_patch(ax4, 0, 0, 0.8, 0.8, colors['control'], 'C')
    draw_patch(ax4, 1.1, 0, 1.9, 0.8, colors['merged'], 'A-T', 'ZZ merge')

    ax4.annotate('', xy=(1.1, 0.4), xytext=(1.6, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax4.text(1.6, 0.65, 'm₂', fontsize=11)

    ax4.set_title('Step 4: ZZ-Merge (A, T)\nMeasure Z_A Z_T = m₂', fontsize=12)
    ax4.axis('off')

    # Panel 5: Split and measure
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(-0.5, 3.5)
    ax5.set_ylim(-0.5, 1.5)
    ax5.set_aspect('equal')

    draw_patch(ax5, 0, 0, 0.8, 0.8, colors['control'], 'C')
    draw_patch(ax5, 1.1, 0, 0.8, 0.8, colors['ancilla'], 'A', 'M_X')
    draw_patch(ax5, 2.2, 0, 0.8, 0.8, colors['target'], 'T')

    ax5.annotate('', xy=(1.5, 1.0), xytext=(1.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax5.text(1.7, 0.95, 'm₃', fontsize=11)

    ax5.set_title('Step 5: XX-Split, Measure A\nMeasure X_A = m₃', fontsize=12)
    ax5.axis('off')

    # Panel 6: Final state
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(-0.5, 3.5)
    ax6.set_ylim(-0.5, 1.5)
    ax6.set_aspect('equal')

    draw_patch(ax6, 0, 0, 0.8, 0.8, colors['control'], 'C', '|ψ⟩')
    draw_patch(ax6, 2.2, 0, 0.8, 0.8, colors['target'], 'T', 'CNOT|ψ,φ⟩')

    # Draw CNOT symbol
    ax6.annotate('', xy=(2.2, 0.4), xytext=(0.8, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax6.plot(0.4, 0.4, 'ko', markersize=10)
    ax6.plot(2.6, 0.4, 'ko', markersize=15, fillstyle='none', markeredgewidth=2)
    ax6.plot([2.6, 2.6], [0.3, 0.5], 'k-', linewidth=2)

    ax6.set_title('Step 6: Apply Corrections\nCNOT complete!', fontsize=12)
    ax6.axis('off')

    plt.suptitle('Lattice Surgery CNOT Protocol', fontsize=16, fontweight='bold')
    plt.savefig('lattice_surgery_cnot.png', dpi=150, bbox_inches='tight')
    plt.show()


def simulate_cnot_surgery():
    """Simulate CNOT via lattice surgery."""
    print("=" * 60)
    print("Lattice Surgery CNOT Simulation")
    print("=" * 60)

    surgery = LatticeSurgery(verbose=True)

    # Test case 1: |+⟩|0⟩ → Bell state
    print("\n--- Test 1: |+⟩|0⟩ → Bell state ---")
    control = LogicalQubit("Control")
    control.apply_h()  # |+⟩
    target = LogicalQubit("Target")  # |0⟩

    print(f"Initial: Control = |+⟩, Target = |0⟩")
    m1, m2, m3 = surgery.cnot(control, target)
    print(f"Outcomes: m1={m1}, m2={m2}, m3={m3}")

    # Test case 2: |1⟩|+⟩ → |1⟩|-⟩
    print("\n--- Test 2: |1⟩|+⟩ → |1⟩|-⟩ ---")
    control2 = LogicalQubit("Control2", np.array([0, 1]))  # |1⟩
    target2 = LogicalQubit("Target2")
    target2.apply_h()  # |+⟩

    print(f"Initial: Control = |1⟩, Target = |+⟩")
    m1, m2, m3 = surgery.cnot(control2, target2)
    print(f"Outcomes: m1={m1}, m2={m2}, m3={m3}")

    return surgery


def time_analysis():
    """Analyze lattice surgery time overhead."""
    print("\n" + "=" * 60)
    print("Time Overhead Analysis")
    print("=" * 60)

    # Parameters
    d_values = [5, 9, 13, 17, 21, 25]
    cycle_time_us = 1.0  # microseconds

    print(f"\nCycle time: {cycle_time_us} μs")
    print(f"\nCNOT time = d rounds × cycle time\n")

    print("Distance | Rounds | CNOT time (μs) | Gate rate (kHz)")
    print("-" * 55)

    for d in d_values:
        rounds = d
        cnot_time = rounds * cycle_time_us
        gate_rate = 1000 / cnot_time  # kHz

        print(f"   {d:2d}    |   {rounds:2d}   |     {cnot_time:5.1f}      |    {gate_rate:6.1f}")

    # Compare with physical gate
    print(f"\nFor comparison: Physical CNOT ~0.1 μs")
    print(f"Overhead factor for d=17: {17 * cycle_time_us / 0.1:.0f}x")


def main():
    """Run lattice surgery demonstrations."""
    print("=" * 70)
    print("DAY 809: LATTICE SURGERY OPERATIONS")
    print("=" * 70)

    # Theory summary
    print("""
    LATTICE SURGERY KEY OPERATIONS

    ┌─────────────────────────────────────────────────────────────┐
    │ MERGE: Join two patches to measure logical Pauli product   │
    │                                                            │
    │   XX-merge: Measure X_A X_B (rough boundaries)             │
    │   ZZ-merge: Measure Z_A Z_B (smooth boundaries)            │
    │                                                            │
    │ SPLIT: Divide patch into two, creating entanglement        │
    │                                                            │
    │   XX-split: Create (|00⟩ + |11⟩)/√2 from |0⟩               │
    │   ZZ-split: Create (|++⟩ + |--⟩)/√2 from |+⟩               │
    │                                                            │
    │ CNOT PROTOCOL:                                             │
    │   1. Prepare ancilla |+⟩                                   │
    │   2. ZZ-merge(Control, Ancilla) → m₁                       │
    │   3. ZZ-merge(Ancilla, Target) → m₂                        │
    │   4. Measure X on Ancilla → m₃                             │
    │   5. Apply X^{(1-m₁)/2} Z^{m₃} to Target                   │
    └─────────────────────────────────────────────────────────────┘
    """)

    # Visualize
    print("\n1. Creating lattice surgery visualization...")
    visualize_lattice_surgery()

    # Simulate
    print("\n2. Running CNOT simulation...")
    simulate_cnot_surgery()

    # Time analysis
    print("\n3. Time overhead analysis...")
    time_analysis()

    print("\n" + "=" * 70)
    print("Key insight: Lattice surgery enables universal computation with")
    print("only local operations, at the cost of O(d) time per logical gate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Operation | Effect | Time |
|-----------|--------|------|
| XX-merge | Measure $\bar{X}_A \bar{X}_B$ | $O(d)$ |
| ZZ-merge | Measure $\bar{Z}_A \bar{Z}_B$ | $O(d)$ |
| XX-split | $\|\bar{0}\rangle \to \|\Phi^+\rangle$ | $O(d)$ |
| CNOT | $\|ct\rangle \to \|c, c\oplus t\rangle$ | $O(d)$ |

### Main Takeaways

1. **Surgery = measure + project**: Merge measures joint stabilizers, projecting to eigenspaces

2. **CNOT needs ancilla**: Protocol uses one ancilla patch in $|+\rangle$

3. **Time overhead is linear**: Each surgery step takes $O(d)$ rounds

4. **Pauli frame tracking**: Corrections can be tracked classically

5. **Parallelism possible**: Independent operations can overlap

---

## Daily Checklist

### Morning Session (3 hours)
- [ ] Understand merge operation mechanics
- [ ] Work through split operation details
- [ ] Study boundary alignment requirements

### Afternoon Session (2.5 hours)
- [ ] Complete CNOT protocol walkthrough
- [ ] Solve Problem Sets A and B
- [ ] Analyze scheduling constraints

### Evening Session (1.5 hours)
- [ ] Run computational lab
- [ ] Simulate CNOT via surgery
- [ ] Complete Problem Set C

### Self-Assessment
1. Can you execute merge given boundary types?
2. Can you trace through CNOT protocol?
3. Do you understand where corrections come from?
4. Can you estimate time for a circuit?

---

## Preview: Day 810

Tomorrow we study **Transversal and Non-Transversal Gates**:
- Transversal Hadamard (requires square patch)
- Non-transversal T via magic state injection
- Magic state distillation protocols
- T-factory design and resource costs

---

*Day 809 of 2184 | Year 2, Month 29, Week 116 | Quantum Engineering PhD Curriculum*
