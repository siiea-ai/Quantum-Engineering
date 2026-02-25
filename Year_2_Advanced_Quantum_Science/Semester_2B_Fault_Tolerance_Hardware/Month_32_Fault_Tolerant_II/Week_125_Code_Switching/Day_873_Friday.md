# Day 873: Lattice Surgery as Code Switching

## Overview

**Day:** 873 of 1008
**Week:** 125 (Code Switching & Gauge Fixing)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Lattice Surgery Operations as Code Transitions and Topological Interpretation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Lattice surgery fundamentals |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Code switching perspective |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational exploration |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Describe** lattice surgery merge and split operations in the surface code
2. **Interpret** lattice surgery as transitions between different code configurations
3. **Analyze** the topological interpretation of code switching
4. **Design** surface code to color code conversion protocols
5. **Apply** the generalized lattice surgery framework to various codes
6. **Connect** lattice surgery to subsystem code concepts

---

## Lattice Surgery Fundamentals

### What is Lattice Surgery?

**Lattice surgery** is a technique for performing logical operations between encoded qubits by **merging and splitting** code patches.

Introduced by Horsman et al. (2012) for surface codes, it has become the dominant paradigm for surface code quantum computing.

**Key Operations:**
1. **Merge:** Combine two patches into one (measures $\bar{Z}_1 \bar{Z}_2$ or $\bar{X}_1 \bar{X}_2$)
2. **Split:** Divide one patch into two (prepares entangled state)

### Surface Code Patches

A **surface code patch** encodes one logical qubit:

```
Rough Boundary (X)
    ├─────────────────┤
    │  Z   Z   Z   Z  │
    │ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ │
S   │X   X   X   X   X│ S
m   │ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ │ m
o   │  Z   Z   Z   Z  │ o
o   │ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ │ o
t   │X   X   X   X   X│ t
h   │ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ │ h
    │  Z   Z   Z   Z  │
    ├─────────────────┤
    Rough Boundary (X)
```

**Boundaries:**
- **Rough boundary:** Logical $\bar{X}$ string terminates here
- **Smooth boundary:** Logical $\bar{Z}$ string terminates here

### The Merge Operation

**XX Merge** (along rough boundaries):

```
Before:
┌─────┐       ┌─────┐
│  A  │       │  B  │
│ |0⟩ │ gap   │ |ψ⟩ │
└─────┘       └─────┘

After:
┌─────────────────────┐
│    Merged Patch     │
│   Measures X_A X_B  │
└─────────────────────┘
```

**Process:**
1. Fill the gap with new data qubits
2. Measure new X-type stabilizers across the seam
3. This measures $\bar{X}_A \bar{X}_B$

**Effect on logical state:**
$$|\psi\rangle_A |\phi\rangle_B \xrightarrow{\text{XX merge}} \text{projects onto } \bar{X}_A \bar{X}_B = \pm 1$$

### The Split Operation

**Reverse of merge:**

```
Before:
┌─────────────────────┐
│    Single Patch     │
│       |ψ⟩           │
└─────────────────────┘

After:
┌─────┐       ┌─────┐
│  A  │       │  B  │
│     │ gap   │     │
└─────┘       └─────┘
```

**Process:**
1. Stop measuring stabilizers along a line
2. The line becomes new boundaries
3. Creates entanglement between A and B

---

## Lattice Surgery as Code Switching

### The Key Insight

**Observation:** Merge and split operations change the code structure:

- **Before merge:** Two separate $[[n,1,d]]$ codes
- **After merge:** One $[[2n+k,1,d]]$ code (approximately)

This is a form of **code switching**!

### Formal Description

**Two-patch system before merge:**
$$\mathcal{H} = \mathcal{H}_{C_A} \otimes \mathcal{H}_{C_B}$$

Stabilizer group: $\mathcal{S} = \mathcal{S}_A \otimes \mathcal{S}_B$

Logical operators: $\bar{X}_A, \bar{Z}_A, \bar{X}_B, \bar{Z}_B$

**After merge:**
$$\mathcal{H}' = \mathcal{H}_{C_{AB}}$$

New stabilizer group: $\mathcal{S}' = \langle \mathcal{S}_A, \mathcal{S}_B, \text{seam stabilizers} \rangle$

**Merged logical operators:**
- $\bar{X}_{AB} = \bar{X}_A = \bar{X}_B$ (they become equal!)
- $\bar{Z}_{AB} = \bar{Z}_A \bar{Z}_B$

### The Measurement Interpretation

**Merge = Measure + Project:**

$$\boxed{\text{XX Merge} \equiv \text{Measure } \bar{X}_A \bar{X}_B}$$

The merge operation:
1. Enlarges the stabilizer group
2. Promotes $\bar{X}_A \bar{X}_B$ from logical to stabilizer
3. Projects onto an eigenspace

### Subsystem Code Perspective

The seam region during merge can be viewed as a **subsystem code**:

**Seam qubits:** New data qubits in the gap
**Gauge operators:** Initial random parity of seam measurements
**Gauge fixing:** Repeated measurement stabilizes the seam

$$\text{Two patches} \xrightarrow{\text{add seam}} \text{Subsystem code} \xrightarrow{\text{gauge fix}} \text{Merged patch}$$

---

## Topological Interpretation

### Anyons and String Operators

In the topological picture:
- **Logical operators** are string operators connecting boundaries
- **Stabilizers** are closed loops (contractible)

**Rough boundary:** Endpoint for $\bar{X}$ string (e-anyon)
**Smooth boundary:** Endpoint for $\bar{Z}$ string (m-anyon)

### Merge as Boundary Fusion

**XX Merge (rough boundaries):**

```
Before:           After:
  |                 |
──┼── ──┼──   →   ──┼──
  |     |           |
 A     B          A=B
```

The rough boundaries **fuse**, making $\bar{X}_A$ and $\bar{X}_B$ equivalent.

**Result:** $\bar{X}_A \bar{X}_B$ becomes a stabilizer (trivial loop).

### ZZ Merge (smooth boundaries)

```
Before:           After:
══╪══ ══╪══   →   ══╪══
  A     B          A=B
```

Smooth boundaries fuse, making $\bar{Z}_A$ and $\bar{Z}_B$ equivalent.

**Measures:** $\bar{Z}_A \bar{Z}_B$

### Topological Code Switching

**Surface code $\leftrightarrow$ Color code:**

Both are topological codes, but with different anyon structures:
- **Surface code:** $e$ and $m$ anyons (abelian)
- **Color code:** Three types of anyons

**Conversion:** Requires changing the lattice structure (code switching!)

---

## Surface Code to Color Code Conversion

### Why Convert?

| Property | Surface Code | Color Code |
|----------|--------------|------------|
| Transversal H | No | Yes |
| Transversal S | No | Yes |
| Transversal T | No | No |
| Qubit overhead | Lower | Higher |

Converting to color code enables transversal Clifford gates!

### The Conversion Protocol

**Kubica & Beverland (2015):**

**Step 1:** Prepare surface code patch with encoded state $|\psi_L\rangle$

**Step 2:** Reshape patch using lattice surgery:
- Split and merge operations to create triangular structure
- This is a sequence of code switches

**Step 3:** Measure new stabilizers:
- Color code has 3-colorable faces
- Measure X and Z stabilizers on each colored face

**Step 4:** Result is color code with same logical state

### Circuit-Level Description

```
Surface Code          Intermediate           Color Code
┌─────────┐         ┌─────────────┐       ┌───────────┐
│ □ □ □ □ │         │   △ △ △     │       │  △   △    │
│ □ □ □ □ │   →     │  △ △ △ △    │   →   │ △ △ △ △   │
│ □ □ □ □ │         │   △ △ △     │       │  △   △    │
└─────────┘         └─────────────┘       └───────────┘
```

**Operations:**
1. Lattice deformation (split/merge)
2. Ancilla preparation
3. New stabilizer measurement
4. Error correction

### Fault Tolerance

The conversion is fault-tolerant if:
- Each step is a valid QEC operation
- Errors are tracked through the conversion
- Final error correction handles accumulated errors

---

## Generalized Lattice Surgery

### The ZX-Calculus Framework

Lattice surgery can be understood in the **ZX-calculus**:

**Merge:** Spider fusion in ZX-calculus
**Split:** Spider splitting

This provides a graphical language for reasoning about lattice surgery.

### Multi-Patch Operations

**Multi-merge (n patches):**

Measure $\bar{X}_1 \bar{X}_2 \cdots \bar{X}_n$ simultaneously:

```
     ┌───┐ ┌───┐ ┌───┐
     │ A │ │ B │ │ C │
     └─┬─┘ └─┬─┘ └─┬─┘
       │     │     │
     ┌─┴─────┴─────┴─┐
     │   Merged      │
     └───────────────┘
```

**Result:** Measures $\bar{X}_A \bar{X}_B \bar{X}_C$ parity.

### Lattice Surgery for Non-Clifford Gates

**Twist-based T gate:**

Using "twist defects" in the surface code:

```
     ┌─────┐
     │     │
    ∨│  τ  │∧   ← Twist defect
     │     │
     └─────┘
```

A twist defect carries non-abelian anyon structure.
Braiding around twists can implement T-gate!

This is another form of topological code switching.

---

## Subsystem Lattice Surgery (SLS)

### Unifying Framework

**Subsystem Lattice Surgery (2017)** provides a unified view:

1. Start with two codes $C_1$ and $C_2$
2. Create combined subsystem code $C_{12}$ with gauge qubits on boundary
3. Gauge fix to get merged code
4. Different gauge fixings give different operations

### SLS for Code Switching

**Key insight:** SLS naturally describes code switching!

$$C_1 \otimes C_2 \xrightarrow{\text{enlarge}} \text{Subsystem}(C_1, C_2) \xrightarrow{\text{gauge fix}} C_{\text{target}}$$

The target code depends on which gauge fixing is applied.

### Advantages of SLS Framework

1. **Unified theory:** Merge, split, code switch all in one framework
2. **Flexibility:** Can design new operations
3. **Analysis:** Systematic error analysis
4. **Generalization:** Applies beyond surface codes

---

## Worked Examples

### Example 1: XX Merge Logical Action

**Problem:** Two surface code patches A and B are in states $|+_L\rangle_A$ and $|\psi_L\rangle_B = \alpha|0_L\rangle + \beta|1_L\rangle$. Compute the state after XX merge.

**Solution:**

**Initial state:**
$$|\Psi_0\rangle = |+_L\rangle_A \otimes |\psi_L\rangle_B = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle)_A \otimes (\alpha|0_L\rangle + \beta|1_L\rangle)_B$$

**Expand:**
$$= \frac{1}{\sqrt{2}}[\alpha|0_L 0_L\rangle + \beta|0_L 1_L\rangle + \alpha|1_L 0_L\rangle + \beta|1_L 1_L\rangle]$$

**XX Merge measures $\bar{X}_A \bar{X}_B$:**

Eigenvalues of $\bar{X}_A \bar{X}_B$:
- $|0_L 0_L\rangle$: $\bar{X}_A \bar{X}_B = (+1)(+1) = +1$...

Wait, $\bar{X}|0_L\rangle = |1_L\rangle$, so $|0_L\rangle$ is NOT an eigenstate of $\bar{X}$.

Let me reconsider. Express in X-basis:
- $|0_L\rangle = \frac{1}{\sqrt{2}}(|+_L\rangle + |-_L\rangle)$
- $|1_L\rangle = \frac{1}{\sqrt{2}}(|+_L\rangle - |-_L\rangle)$

For $\bar{X}_A \bar{X}_B$ eigenstates:
- $|+_L +_L\rangle$: eigenvalue $+1$
- $|+_L -_L\rangle$: eigenvalue $-1$
- $|-_L +_L\rangle$: eigenvalue $-1$
- $|-_L -_L\rangle$: eigenvalue $+1$

**Rewrite initial state:**
$$|\Psi_0\rangle = |+_L\rangle_A \otimes |\psi_L\rangle_B$$

In X-basis for B:
$$|\psi_L\rangle_B = \frac{\alpha+\beta}{\sqrt{2}}|+_L\rangle_B + \frac{\alpha-\beta}{\sqrt{2}}|-_L\rangle_B$$

So:
$$|\Psi_0\rangle = \frac{\alpha+\beta}{\sqrt{2}}|+_L +_L\rangle + \frac{\alpha-\beta}{\sqrt{2}}|+_L -_L\rangle$$

**Measurement projects:**
- Outcome $+1$: $|+_L +_L\rangle$ with probability $|\alpha+\beta|^2/2$
- Outcome $-1$: $|+_L -_L\rangle$ with probability $|\alpha-\beta|^2/2$

**After merge (outcome +1):**
$$\boxed{|\Psi_{\text{merged}}\rangle = |+_L\rangle_{AB} \text{ (single logical qubit)}}$$

The merged patch is in $|+_L\rangle$.

### Example 2: Implementing CNOT via Lattice Surgery

**Problem:** Use lattice surgery to implement logical CNOT between control C and target T.

**Solution:**

**Protocol (Horsman et al.):**

**Step 1:** Prepare ancilla patch A in $|+_L\rangle$

**Step 2:** ZZ merge C with A:
- Measures $\bar{Z}_C \bar{Z}_A$
- Outcome $m_1 \in \{+1, -1\}$

**Step 3:** Split C from A (rough split):
- Creates entanglement

**Step 4:** XX merge A with T:
- Measures $\bar{X}_A \bar{X}_T$
- Outcome $m_2 \in \{+1, -1\}$

**Step 5:** Classical correction:
- If $m_1 = -1$: Apply $\bar{Z}_T$
- If $m_2 = -1$: Apply $\bar{X}_C$

**Net effect:**
$$\boxed{\text{CNOT}_{C \to T}}$$

**Verification:**
Track computational basis states through the protocol to verify CNOT action.

### Example 3: Code Distance During Merge

**Problem:** Two distance-$d$ surface code patches are merged. What is the distance of the merged code during and after the merge?

**Solution:**

**Before merge:**
- Each patch: distance $d$
- Total system: $(d, d)$ - errors on A and B independent

**During merge (first round):**
- Seam stabilizers not yet reliable
- Effective distance across seam: 1
- Overall: reduced protection during transition

**After $d$ rounds:**
- Seam stabilizers have been measured $d$ times
- Errors can be tracked across seam
- Effective distance: $d$ (restored)

$$\boxed{d_{\text{during}} < d, \quad d_{\text{after}} = d}$$

**Implication:** Merge/split operations are "instantaneous" in terms of logical operation but require $O(d)$ syndrome rounds for full error correction.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Draw the surface code patch configuration for a ZZ merge (smooth boundaries touching).

**P1.2** If two patches in $|0_L\rangle$ are XX-merged, what is the measurement outcome and final state?

**P1.3** How many syndrome measurement rounds are needed for a merge operation on distance-5 patches?

### Level 2: Intermediate

**P2.1** Design a lattice surgery protocol to implement $\bar{Z}_A \bar{Z}_B$ measurement without physically merging the patches (hint: use an ancilla).

**P2.2** Prove that XX merge followed by XX split on the same boundary returns the original two-patch configuration (up to Pauli corrections).

**P2.3** Calculate the space-time volume (qubits $\times$ time) for implementing a CNOT via lattice surgery on distance-$d$ surface codes.

### Level 3: Challenging

**P3.1** Design a lattice surgery protocol to convert a surface code patch to a color code patch. Specify all intermediate steps and stabilizer measurements.

**P3.2** Analyze the error model during lattice surgery. How do errors on seam qubits propagate to logical errors?

**P3.3** The twist-based approach uses defects for non-Clifford gates. Explain how this relates to code switching and compare resources with magic state injection.

---

## Computational Lab

```python
"""
Day 873: Lattice Surgery as Code Switching
==========================================

Simulation of lattice surgery operations and their code-switching interpretation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


class BoundaryType(Enum):
    ROUGH = "rough"  # X-type boundary
    SMOOTH = "smooth"  # Z-type boundary


class MergeType(Enum):
    XX = "XX"  # Merge along rough boundaries
    ZZ = "ZZ"  # Merge along smooth boundaries


@dataclass
class SurfaceCodePatch:
    """Represents a surface code patch."""
    name: str
    distance: int
    logical_state: np.ndarray  # 2D state vector [alpha, beta]

    def __post_init__(self):
        # Normalize
        norm = np.linalg.norm(self.logical_state)
        if norm > 0:
            self.logical_state = self.logical_state / norm

    @property
    def alpha(self):
        return self.logical_state[0]

    @property
    def beta(self):
        return self.logical_state[1]

    def apply_X(self):
        """Apply logical X."""
        self.logical_state = np.array([self.beta, self.alpha])

    def apply_Z(self):
        """Apply logical Z."""
        self.logical_state = np.array([self.alpha, -self.beta])

    def __repr__(self):
        return f"Patch({self.name}, d={self.distance}, state=[{self.alpha:.3f}, {self.beta:.3f}])"


def logical_inner_product(psi: np.ndarray, phi: np.ndarray) -> complex:
    """Compute inner product of logical states."""
    return np.vdot(psi, phi)


class LatticeSurgery:
    """
    Implements lattice surgery operations.
    """

    def __init__(self):
        # Basis states
        self.zero = np.array([1, 0], dtype=complex)
        self.one = np.array([0, 1], dtype=complex)
        self.plus = (self.zero + self.one) / np.sqrt(2)
        self.minus = (self.zero - self.one) / np.sqrt(2)

    def merge_xx(self, patch_a: SurfaceCodePatch,
                 patch_b: SurfaceCodePatch,
                 verbose: bool = True) -> Tuple[SurfaceCodePatch, int]:
        """
        Perform XX merge (rough boundary merge).

        Measures X_A X_B and projects onto an eigenstate.

        Returns:
        --------
        (merged_patch, measurement_outcome)
        """
        if verbose:
            print(f"\nXX Merge: {patch_a.name} + {patch_b.name}")
            print("-" * 40)

        # Express states in X basis
        # |0⟩ = (|+⟩ + |-⟩)/√2
        # |1⟩ = (|+⟩ - |-⟩)/√2

        alpha_a, beta_a = patch_a.alpha, patch_a.beta
        alpha_b, beta_b = patch_b.alpha, patch_b.beta

        # X-basis amplitudes
        plus_a = (alpha_a + beta_a) / np.sqrt(2)
        minus_a = (alpha_a - beta_a) / np.sqrt(2)
        plus_b = (alpha_b + beta_b) / np.sqrt(2)
        minus_b = (alpha_b - beta_b) / np.sqrt(2)

        # Joint state in X basis: |++⟩, |+-⟩, |-+⟩, |--⟩
        # X_A X_B eigenvalue +1: |++⟩, |--⟩
        # X_A X_B eigenvalue -1: |+-⟩, |-+⟩

        prob_plus1 = np.abs(plus_a * plus_b)**2 + np.abs(minus_a * minus_b)**2
        prob_minus1 = np.abs(plus_a * minus_b)**2 + np.abs(minus_a * plus_b)**2

        # Normalize
        total = prob_plus1 + prob_minus1
        prob_plus1 /= total
        prob_minus1 /= total

        if verbose:
            print(f"  P(X_A X_B = +1) = {prob_plus1:.4f}")
            print(f"  P(X_A X_B = -1) = {prob_minus1:.4f}")

        # Simulate measurement
        outcome = np.random.choice([+1, -1], p=[prob_plus1, prob_minus1])

        if verbose:
            print(f"  Measurement outcome: {'+1' if outcome == +1 else '-1'}")

        # Project and compute merged state
        if outcome == +1:
            # Project onto |++⟩ + |--⟩ (up to normalization)
            # Merged state is in |+⟩
            merged_state = self.plus
        else:
            # Project onto |+-⟩ + |-+⟩
            # Merged state is in |-⟩
            merged_state = self.minus

        merged_patch = SurfaceCodePatch(
            name=f"{patch_a.name}+{patch_b.name}",
            distance=max(patch_a.distance, patch_b.distance),
            logical_state=merged_state.copy()
        )

        if verbose:
            print(f"  Merged patch: {merged_patch}")

        return merged_patch, outcome

    def merge_zz(self, patch_a: SurfaceCodePatch,
                 patch_b: SurfaceCodePatch,
                 verbose: bool = True) -> Tuple[SurfaceCodePatch, int]:
        """
        Perform ZZ merge (smooth boundary merge).

        Measures Z_A Z_B and projects onto an eigenstate.
        """
        if verbose:
            print(f"\nZZ Merge: {patch_a.name} + {patch_b.name}")
            print("-" * 40)

        alpha_a, beta_a = patch_a.alpha, patch_a.beta
        alpha_b, beta_b = patch_b.alpha, patch_b.beta

        # Z basis is computational basis
        # Z_A Z_B eigenvalue +1: |00⟩, |11⟩
        # Z_A Z_B eigenvalue -1: |01⟩, |10⟩

        prob_plus1 = np.abs(alpha_a * alpha_b)**2 + np.abs(beta_a * beta_b)**2
        prob_minus1 = np.abs(alpha_a * beta_b)**2 + np.abs(beta_a * alpha_b)**2

        # Normalize
        total = prob_plus1 + prob_minus1
        prob_plus1 /= total
        prob_minus1 /= total

        if verbose:
            print(f"  P(Z_A Z_B = +1) = {prob_plus1:.4f}")
            print(f"  P(Z_A Z_B = -1) = {prob_minus1:.4f}")

        # Simulate measurement
        outcome = np.random.choice([+1, -1], p=[prob_plus1, prob_minus1])

        if verbose:
            print(f"  Measurement outcome: {'+1' if outcome == +1 else '-1'}")

        # The merged state depends on projection
        # This is simplified - actual state depends on input superposition
        if outcome == +1:
            merged_state = self.zero  # Simplified
        else:
            merged_state = self.one  # Simplified

        merged_patch = SurfaceCodePatch(
            name=f"{patch_a.name}+{patch_b.name}",
            distance=max(patch_a.distance, patch_b.distance),
            logical_state=merged_state.copy()
        )

        if verbose:
            print(f"  Merged patch: {merged_patch}")

        return merged_patch, outcome

    def split(self, patch: SurfaceCodePatch,
              split_type: str = "XX",
              verbose: bool = True) -> Tuple[SurfaceCodePatch, SurfaceCodePatch]:
        """
        Split a patch into two patches.

        This creates entanglement between the two resulting patches.
        """
        if verbose:
            print(f"\n{split_type} Split: {patch.name}")
            print("-" * 40)

        # Splitting creates a Bell pair essentially
        # The original logical state is distributed

        # For XX split (rough boundaries created):
        if split_type == "XX":
            # Creates |Φ+⟩-like state tensored with original state info
            patch_a = SurfaceCodePatch(
                name=f"{patch.name}_L",
                distance=patch.distance,
                logical_state=patch.logical_state.copy()
            )
            patch_b = SurfaceCodePatch(
                name=f"{patch.name}_R",
                distance=patch.distance,
                logical_state=self.plus.copy()
            )
        else:  # ZZ split
            patch_a = SurfaceCodePatch(
                name=f"{patch.name}_L",
                distance=patch.distance,
                logical_state=patch.logical_state.copy()
            )
            patch_b = SurfaceCodePatch(
                name=f"{patch.name}_R",
                distance=patch.distance,
                logical_state=self.zero.copy()
            )

        if verbose:
            print(f"  Created: {patch_a}")
            print(f"  Created: {patch_b}")

        return patch_a, patch_b


def cnot_via_lattice_surgery():
    """Demonstrate CNOT implementation via lattice surgery."""
    print("\n" + "=" * 60)
    print("CNOT via Lattice Surgery")
    print("=" * 60)

    ls = LatticeSurgery()

    # Prepare control and target
    control = SurfaceCodePatch("C", 3, np.array([1, 0]))  # |0⟩
    target = SurfaceCodePatch("T", 3, np.array([1, 0]))   # |0⟩

    print(f"\nInitial control: {control}")
    print(f"Initial target: {target}")

    # For a proper CNOT, we need:
    # |00⟩ → |00⟩
    # |01⟩ → |01⟩
    # |10⟩ → |11⟩
    # |11⟩ → |10⟩

    print("\nCNOT Protocol:")
    print("-" * 40)

    # Step 1: Prepare ancilla in |+⟩
    ancilla = SurfaceCodePatch("A", 3, (np.array([1, 0]) + np.array([0, 1])) / np.sqrt(2))
    print(f"\n1. Ancilla prepared: {ancilla}")

    # Step 2: ZZ merge control with ancilla
    print("\n2. ZZ merge Control with Ancilla")
    merged_ca, m1 = ls.merge_zz(control, ancilla, verbose=True)

    # Step 3: Split (simplified - in practice more complex)
    print("\n3. Split Control from Ancilla")
    control_new, ancilla_new = ls.split(merged_ca, "ZZ", verbose=True)

    # Step 4: XX merge ancilla with target
    print("\n4. XX merge Ancilla with Target")
    merged_at, m2 = ls.merge_xx(ancilla_new, target, verbose=True)

    # Step 5: Apply corrections
    print("\n5. Apply corrections based on measurements")
    print(f"   m1 = {'+1' if m1 == +1 else '-1'}: ", end="")
    if m1 == -1:
        print("Apply Z to Target")
        merged_at.apply_Z()
    else:
        print("No correction")

    print(f"   m2 = {'+1' if m2 == +1 else '-1'}: ", end="")
    if m2 == -1:
        print("Apply X to Control")
        control_new.apply_X()
    else:
        print("No correction")

    print(f"\nFinal control: {control_new}")
    print(f"Final target: {merged_at}")


def code_switching_perspective():
    """Explain lattice surgery as code switching."""
    print("\n" + "=" * 60)
    print("Lattice Surgery as Code Switching")
    print("=" * 60)

    print("\nTwo-Patch System → Merged System")
    print("-" * 40)

    print("""
    Before Merge:
    ┌─────────────────────────────────────────┐
    │  Code Structure: C_A ⊗ C_B              │
    │  Stabilizers: S_A ⊗ I, I ⊗ S_B          │
    │  Logical ops: X̄_A, Z̄_A, X̄_B, Z̄_B       │
    │  Logical qubits: 2                      │
    └─────────────────────────────────────────┘

    After Merge:
    ┌─────────────────────────────────────────┐
    │  Code Structure: C_AB (merged)          │
    │  Stabilizers: S_A, S_B, seam operators  │
    │  New stabilizer: X̄_A X̄_B or Z̄_A Z̄_B    │
    │  Logical ops: X̄_AB, Z̄_AB               │
    │  Logical qubits: 1                      │
    └─────────────────────────────────────────┘
    """)

    print("\nThis is code switching!")
    print("  - Code parameters change: [[2n,2,d]] → [[2n+k,1,d]]")
    print("  - Stabilizer group is enlarged")
    print("  - Logical operator count reduced")


def subsystem_lattice_surgery_view():
    """Explain the subsystem code perspective on lattice surgery."""
    print("\n" + "=" * 60)
    print("Subsystem Lattice Surgery (SLS) View")
    print("=" * 60)

    print("""
    Merge Process through SLS Lens:
    ───────────────────────────────

    Step 1: Two Patches
    ┌─────┐   ┌─────┐
    │  A  │   │  B  │    Two independent codes
    └─────┘   └─────┘

    Step 2: Add Seam (Gauge Qubits)
    ┌─────┬───┬─────┐
    │  A  │ G │  B  │    G = gauge qubits on seam
    └─────┴───┴─────┘    This is a SUBSYSTEM CODE!

    Step 3: Gauge Fix (Measure Seam)
    ┌─────────────────┐
    │    Merged       │    Gauge fixed to stabilizer code
    └─────────────────┘

    Key Insight:
    - Seam qubits act as gauge qubits
    - Measuring seam = gauge fixing
    - Different measurements → different merged codes
    """)


def resource_analysis():
    """Analyze resources for lattice surgery operations."""
    print("\n" + "=" * 60)
    print("Resource Analysis: Lattice Surgery")
    print("=" * 60)

    d = 5  # Code distance

    print(f"\nFor distance d = {d} surface code:")
    print("-" * 40)

    # Merge operation
    print("\nMerge Operation:")
    print(f"  Seam qubits added: ~{d}")
    print(f"  Syndrome rounds: {d} (for reliability)")
    print(f"  Time: {d} × τ_syndrome")

    # CNOT via lattice surgery
    print("\nCNOT via Lattice Surgery:")
    print(f"  Ancilla patch: d² = {d**2} qubits")
    print(f"  Operations: 2 merges + 1 split")
    print(f"  Total time: ~{3*d} syndrome rounds")
    print(f"  Total qubits: ~{3*d**2} (3 patches)")

    # Compare to transversal (if available)
    print("\nComparison to Transversal CNOT:")
    print(f"  Transversal: 1 time step, n = d² gates")
    print(f"  Lattice surgery: {3*d} time steps, uses ancilla")
    print(f"  Trade-off: LS uses more time but simpler connectivity")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 873: Lattice Surgery as Code Switching")
    print("=" * 60)

    # Basic lattice surgery
    ls = LatticeSurgery()

    # Example: Merge two patches
    patch_a = SurfaceCodePatch("A", 3, np.array([1, 0]))
    patch_b = SurfaceCodePatch("B", 3, (np.array([1, 0]) + np.array([0, 1])) / np.sqrt(2))

    print(f"\nPatch A: {patch_a} (|0⟩)")
    print(f"Patch B: {patch_b} (|+⟩)")

    merged, outcome = ls.merge_xx(patch_a, patch_b, verbose=True)

    # CNOT demonstration
    cnot_via_lattice_surgery()

    # Conceptual explanations
    code_switching_perspective()
    subsystem_lattice_surgery_view()
    resource_analysis()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Lattice surgery = merge/split operations on code patches")
    print("2. Merge measures logical operator products (X̄_AX̄_B or Z̄_AZ̄_B)")
    print("3. This is CODE SWITCHING: changes code structure")
    print("4. SLS view: seam = gauge qubits, merge = gauge fixing")
    print("5. Enables logical gates without transversal operations")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| XX Merge | Measures $\bar{X}_A \bar{X}_B$ |
| ZZ Merge | Measures $\bar{Z}_A \bar{Z}_B$ |
| Merged logical X | $\bar{X}_{AB} = \bar{X}_A = \bar{X}_B$ |
| Merged logical Z | $\bar{Z}_{AB} = \bar{Z}_A \bar{Z}_B$ |
| Merge time | $O(d)$ syndrome rounds |
| CNOT resources | 2 merges + 1 split + ancilla |

### Main Takeaways

1. **Lattice surgery** performs logical operations by merging/splitting patches
2. **Merge = measurement** of logical operator products
3. **This is code switching:** the code structure changes during merge
4. **Subsystem view:** seam qubits are gauge qubits; merge is gauge fixing
5. **Universal computation:** CNOT, measurement, and state prep enable universality
6. **Topological interpretation:** boundary fusion in anyon picture

---

## Daily Checklist

- [ ] I can describe XX and ZZ merge operations
- [ ] I understand how merge measures logical operators
- [ ] I can explain lattice surgery as code switching
- [ ] I know the subsystem code perspective (SLS)
- [ ] I can outline CNOT implementation via lattice surgery
- [ ] I understand the resource costs of lattice surgery

---

## Preview: Day 874

Tomorrow is **Computational Lab Day**:

- Implement complete code switching simulations
- Verify Steane → Reed-Muller protocols
- Simulate gauge fixing on Bacon-Shor code
- Compare error rates with magic state approach
- Benchmark resource requirements
- Analyze fault-tolerance numerically
