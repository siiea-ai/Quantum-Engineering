# Day 750: Color Codes

## Overview

**Day:** 750 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Color Codes and Topological Structure

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Color code construction |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Transversal gates |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Construct** color codes on 3-colorable lattices
2. **Identify** X and Z stabilizers from face structure
3. **Verify** the CSS structure of color codes
4. **Implement** transversal Clifford gates
5. **Compare** color codes with surface codes
6. **Analyze** code parameters and properties

---

## Color Code Introduction

### Motivation

**Surface codes:** Transversal X, Z, CNOT but NOT H or S.

**Color codes:** Transversal **full Clifford group** including H and S!

This is a significant advantage for fault-tolerant computation.

### 3-Colorable Lattices

A lattice is **3-colorable** (or tripartite) if faces can be colored with 3 colors such that no two adjacent faces share the same color.

**Examples:**
- Triangular (kagome) lattice
- Hexagonal (honeycomb) lattice
- Square-octagon lattice

### Color Code Definition

**Qubits:** On vertices of the lattice

**Stabilizers:** For each face f:
$$A_f = \prod_{v \in f} X_v \quad \text{(X stabilizer)}$$
$$B_f = \prod_{v \in f} Z_v \quad \text{(Z stabilizer)}$$

Both X and Z stabilizers are supported on the same faces!

---

## The 7-Qubit Color Code

### Smallest Example

The **[[7, 1, 3]]** color code is the smallest color code, equivalent to the Steane code.

**Lattice:** Triangular patch with 7 vertices and 3 faces (one of each color).

```
        1
       / \
      /   \
     2-----3
    /|\   /|\
   / | \ / | \
  4--5--6--7--4
```

Wait, let me draw this correctly. The 7-qubit color code uses:

```
      1
     /|\
    / | \
   2--3--4
   |\ | /|
   | \|/ |
   5--6--7
```

Actually, the standard representation is a triangle with 7 vertices:

**Vertices:** 1-7
**Faces:**
- Red face: {1, 2, 3}
- Green face: {3, 4, 5}
- Blue face: {5, 6, 7}
- Central face: {2, 3, 5} (different coloring schemes exist)

### Steane Code Connection

The 7-qubit color code has the **same stabilizers** as the Steane code:
- 3 X stabilizers (one per face)
- 3 Z stabilizers (one per face)
- 1 logical qubit

$$\text{Color}[[7,1,3]] \cong \text{Steane}[[7,1,3]]$$

---

## General Color Code Structure

### CSS Property

Color codes are **CSS codes**:
- All X stabilizers commute (products of X)
- All Z stabilizers commute (products of Z)
- X and Z stabilizers on same face commute (even overlap)

**Commutation check:** For X stabilizer on face f and Z stabilizer on face g:
$$[A_f, B_g] = 0 \Leftrightarrow |f \cap g| \equiv 0 \pmod 2$$

For 3-colorable lattices, adjacent faces share exactly 2 vertices, so they commute.

### Code Parameters

For distance-d color code on appropriate lattice:
$$[[n, k, d]]$$

**Typical:** $n = O(d^2)$, $k = 1$ or more depending on topology.

### Logical Operators

**Logical X:** String of X operators connecting boundaries of same color
**Logical Z:** String of Z operators connecting boundaries of same color

The structure mirrors surface codes but with 3-fold symmetry.

---

## Transversal Gates

### Full Clifford Group

**Theorem:** Color codes support transversal implementation of the entire Clifford group.

**Transversal gates:**
- $\bar{X} = X^{\otimes n}$
- $\bar{Z} = Z^{\otimes n}$
- $\bar{H} = H^{\otimes n}$ (swaps X ↔ Z stabilizers)
- $\bar{S} = S^{\otimes n}$ (for appropriately chosen codes)
- $\overline{CNOT} = CNOT^{\otimes n}$

### Why H is Transversal

For color codes, X and Z stabilizers have **identical support** on each face:
$$A_f = \prod_{v \in f} X_v, \quad B_f = \prod_{v \in f} Z_v$$

Under $H^{\otimes n}$:
$$A_f \to B_f, \quad B_f \to A_f$$

The stabilizer group is preserved! ✓

### Why S is Transversal

Under $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$:
$$X \to Y = iXZ, \quad Z \to Z$$

For stabilizer $A_f = \prod X_v$:
$$S^{\otimes n} A_f (S^\dagger)^{\otimes n} = \prod_v Y_v = i^{|f|} A_f B_f$$

For color codes with all faces having the **same parity** of vertices:
- If |f| ≡ 0 mod 4: $S A_f S^\dagger = A_f B_f$
- This is still a stabilizer element ✓

**Condition:** All faces must have |f| ≡ 0 mod 4 for transversal S.

---

## Color Code vs Surface Code

### Comparison Table

| Property | Surface Code | Color Code |
|----------|--------------|------------|
| Lattice | Square | 3-colorable |
| Qubits per face | 4 (typically) | Varies (often 4 or 6) |
| Transversal X | ✓ | ✓ |
| Transversal Z | ✓ | ✓ |
| Transversal H | ✗ | ✓ |
| Transversal S | ✗ | ✓ (if face sizes ≡ 0 mod 4) |
| Transversal CNOT | ✓ | ✓ |
| Threshold | ~1% | ~0.1% (lower) |

### Trade-offs

**Advantages of color codes:**
- Full transversal Clifford group
- Potential for simpler logical gates

**Disadvantages:**
- Lower threshold (more qubits needed for same error rate)
- More complex stabilizer structure

---

## Triangular Color Code

### Construction

On a triangular lattice (3 vertices per face):

**Parameters:** [[n, 1, d]] where n = 3d(d-1)/2 + 1 for distance d

**Example: d = 3**
- n = 3·3·2/2 + 1 = 10 qubits

Actually, the formula depends on the specific triangular patch.

### Hexagonal Color Code

On a hexagonal lattice (6 vertices per face):

**Advantages:**
- Each face has 6 vertices (|f| = 6 ≡ 2 mod 4)
- Need modification for transversal S

**Parameters:** Depend on boundary conditions.

---

## Worked Examples

### Example 1: Verify Color Code Commutation

**Problem:** Show that X and Z stabilizers on adjacent faces commute.

**Solution:**

Consider two adjacent faces f and g sharing edge (v₁, v₂).

X stabilizer on f: $A_f = X_{v_1} X_{v_2} \cdots$ (includes v₁, v₂)
Z stabilizer on g: $B_g = Z_{v_1} Z_{v_2} \cdots$ (includes v₁, v₂)

Overlap: {v₁, v₂} has 2 elements.

Commutation: $(-1)^2 = 1$, so they commute. ✓

### Example 2: 7-Qubit Color Code Stabilizers

**Problem:** Write the stabilizers for the [[7,1,3]] color code.

**Solution:**

Using the Steane code representation:

**X Stabilizers:**
- $A_{\text{red}} = X_4 X_5 X_6 X_7$ (face with vertices 4,5,6,7)
- $A_{\text{green}} = X_2 X_3 X_6 X_7$ (face with vertices 2,3,6,7)
- $A_{\text{blue}} = X_1 X_3 X_5 X_7$ (face with vertices 1,3,5,7)

**Z Stabilizers:**
- $B_{\text{red}} = Z_4 Z_5 Z_6 Z_7$
- $B_{\text{green}} = Z_2 Z_3 Z_6 Z_7$
- $B_{\text{blue}} = Z_1 Z_3 Z_5 Z_7$

Same structure as Steane code! The 7-qubit color code IS the Steane code.

### Example 3: Transversal H Verification

**Problem:** Show $H^{\otimes 7}$ preserves the 7-qubit color code.

**Solution:**

Under H: $X \to Z$, $Z \to X$

$A_{\text{red}} = X_4 X_5 X_6 X_7 \to Z_4 Z_5 Z_6 Z_7 = B_{\text{red}}$
$B_{\text{red}} = Z_4 Z_5 Z_6 Z_7 \to X_4 X_5 X_6 X_7 = A_{\text{red}}$

Similarly for green and blue faces.

The stabilizer group maps to itself! ✓

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Draw a triangular color code with 10 qubits and label the three face colors.

**P1.2** For a color code face with 6 vertices, write the X and Z stabilizers.

**P1.3** Verify that two faces sharing 2 vertices have commuting X/Z stabilizers.

### Level 2: Intermediate

**P2.1** Prove that $H^{\otimes n}$ is transversal for any color code (not just d=3).

**P2.2** Determine the condition on face sizes for $S^{\otimes n}$ to be transversal.

**P2.3** Compare the 7-qubit color code and Steane code:
a) Are they the same code?
b) Do they have the same transversal gates?

### Level 3: Challenging

**P3.1** Construct a color code with parameters [[19, 1, 5]] on a triangular lattice.

**P3.2** Prove that no 2D topological code can have transversal non-Clifford gates.

**P3.3** Design a color code that supports transversal T gate (hint: need 3D or other structure).

---

## Computational Lab

```python
"""
Day 750: Color Codes
====================

Implementing color code constructions and analysis.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
import matplotlib.pyplot as plt


class ColorCodeFace:
    """A face in a color code lattice."""

    def __init__(self, vertices: List[int], color: str):
        self.vertices = set(vertices)
        self.color = color  # 'red', 'green', or 'blue'

    def x_stabilizer(self, n: int) -> np.ndarray:
        """Return X stabilizer as binary vector."""
        stab = np.zeros(n, dtype=int)
        for v in self.vertices:
            stab[v] = 1
        return stab

    def z_stabilizer(self, n: int) -> np.ndarray:
        """Return Z stabilizer as binary vector."""
        return self.x_stabilizer(n)  # Same support for color codes

    def __repr__(self) -> str:
        return f"{self.color} face: {self.vertices}"


class ColorCode:
    """Color code on a 3-colorable lattice."""

    def __init__(self, n_qubits: int, faces: List[ColorCodeFace]):
        """
        Initialize color code.

        Parameters:
        -----------
        n_qubits : int
            Number of physical qubits (vertices)
        faces : List[ColorCodeFace]
            List of faces with colors
        """
        self.n = n_qubits
        self.faces = faces
        self._verify_3_coloring()
        self._build_stabilizers()

    def _verify_3_coloring(self):
        """Verify valid 3-coloring (adjacent faces have different colors)."""
        for i, f1 in enumerate(self.faces):
            for f2 in self.faces[i+1:]:
                overlap = len(f1.vertices & f2.vertices)
                if overlap >= 2 and f1.color == f2.color:
                    raise ValueError(f"Adjacent faces have same color: {f1}, {f2}")

    def _build_stabilizers(self):
        """Build X and Z stabilizer matrices."""
        m = len(self.faces)
        self.H_X = np.zeros((m, self.n), dtype=int)
        self.H_Z = np.zeros((m, self.n), dtype=int)

        for i, face in enumerate(self.faces):
            self.H_X[i] = face.x_stabilizer(self.n)
            self.H_Z[i] = face.z_stabilizer(self.n)

    def verify_css(self) -> bool:
        """Verify CSS commutation."""
        # X-Z commutation: check overlap parity
        for i, f1 in enumerate(self.faces):
            for j, f2 in enumerate(self.faces):
                overlap = len(f1.vertices & f2.vertices)
                if overlap % 2 != 0:
                    return False
        return True

    def code_parameters(self) -> Tuple[int, int]:
        """Return (n, k) parameters."""
        rank_X = np.linalg.matrix_rank(self.H_X)
        rank_Z = np.linalg.matrix_rank(self.H_Z)
        k = self.n - rank_X - rank_Z
        return self.n, k

    def x_stabilizer_strings(self) -> List[str]:
        """Return X stabilizers as Pauli strings."""
        stabs = []
        for row in self.H_X:
            pauli = ''.join('X' if b else 'I' for b in row)
            stabs.append(pauli)
        return stabs

    def z_stabilizer_strings(self) -> List[str]:
        """Return Z stabilizers as Pauli strings."""
        stabs = []
        for row in self.H_Z:
            pauli = ''.join('Z' if b else 'I' for b in row)
            stabs.append(pauli)
        return stabs

    def check_transversal_H(self) -> bool:
        """Check if H^⊗n is transversal."""
        # H swaps X ↔ Z. For color codes, X and Z have same support,
        # so H always works.
        return np.array_equal(self.H_X, self.H_Z)

    def check_transversal_S(self) -> bool:
        """Check if S^⊗n is transversal."""
        # Need all face sizes ≡ 0 mod 4
        for face in self.faces:
            if len(face.vertices) % 4 != 0:
                return False
        return True

    def __repr__(self) -> str:
        n, k = self.code_parameters()
        return f"Color Code [[{n}, {k}]] with {len(self.faces)} faces"


def steane_color_code() -> ColorCode:
    """
    The [[7,1,3]] color code (= Steane code).

    Vertices labeled 0-6.
    Three faces with 4 vertices each.
    """
    faces = [
        ColorCodeFace([3, 4, 5, 6], 'red'),    # IIIXXXX
        ColorCodeFace([1, 2, 5, 6], 'green'),  # IXXIIXX
        ColorCodeFace([0, 2, 4, 6], 'blue')    # XIXIXIX
    ]
    return ColorCode(7, faces)


def triangular_color_code(L: int) -> ColorCode:
    """
    Triangular color code of size L.

    Simplified construction for demonstration.
    """
    # This would build a proper triangular lattice
    # For now, return the L=2 case (7-qubit code)
    if L == 2:
        return steane_color_code()
    else:
        raise NotImplementedError("Only L=2 implemented")


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 750: Color Codes")
    print("=" * 60)

    # Example 1: Steane/7-qubit color code
    print("\n1. [[7,1,3]] Color Code (= Steane)")
    print("-" * 40)

    steane = steane_color_code()
    n, k = steane.code_parameters()
    print(f"Code: {steane}")
    print(f"CSS valid: {steane.verify_css()}")

    print("\nFaces:")
    for face in steane.faces:
        print(f"  {face}")

    print("\nX Stabilizers:")
    for s in steane.x_stabilizer_strings():
        print(f"  {s}")

    print("\nZ Stabilizers:")
    for s in steane.z_stabilizer_strings():
        print(f"  {s}")

    # Example 2: Transversal gates
    print("\n2. Transversal Gate Analysis")
    print("-" * 40)

    print(f"H^⊗7 transversal: {steane.check_transversal_H()}")
    print(f"S^⊗7 transversal: {steane.check_transversal_S()}")

    # All faces have 4 vertices, 4 ≡ 0 mod 4, so S works!
    print("\nFace sizes:")
    for face in steane.faces:
        print(f"  {face.color}: {len(face.vertices)} vertices")

    # Example 3: Verify X-Z commutation
    print("\n3. X-Z Commutation Check")
    print("-" * 40)

    for i, f1 in enumerate(steane.faces):
        for j, f2 in enumerate(steane.faces):
            overlap = len(f1.vertices & f2.vertices)
            commutes = "✓" if overlap % 2 == 0 else "✗"
            print(f"  {f1.color}-X with {f2.color}-Z: "
                  f"overlap = {overlap}, commutes: {commutes}")

    # Example 4: Compare with surface code
    print("\n4. Color Code vs Surface Code")
    print("-" * 40)

    print(f"{'Property':<25} {'Surface':<12} {'Color':<12}")
    print("-" * 50)
    print(f"{'Transversal X':<25} {'✓':<12} {'✓':<12}")
    print(f"{'Transversal Z':<25} {'✓':<12} {'✓':<12}")
    print(f"{'Transversal H':<25} {'✗':<12} {'✓':<12}")
    print(f"{'Transversal S':<25} {'✗':<12} {'✓ (if |f|≡0 mod 4)':<12}")
    print(f"{'Transversal CNOT':<25} {'✓':<12} {'✓':<12}")
    print(f"{'Error threshold':<25} {'~1%':<12} {'~0.1%':<12}")

    print("\n" + "=" * 60)
    print("Color codes: full Clifford group transversally!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| X stabilizer | $A_f = \prod_{v \in f} X_v$ |
| Z stabilizer | $B_f = \prod_{v \in f} Z_v$ |
| Transversal H | $\bar{H} = H^{\otimes n}$ (always works) |
| Transversal S | $\bar{S} = S^{\otimes n}$ (if $|f| \equiv 0 \pmod 4$) |
| Commutation | $|f \cap g| \equiv 0 \pmod 2$ |

### Main Takeaways

1. **Color codes** are CSS codes on 3-colorable lattices
2. X and Z stabilizers have **same support** on each face
3. **Transversal H** works because X ↔ Z swap preserves stabilizers
4. **Transversal S** requires face sizes ≡ 0 mod 4
5. Color codes have **lower threshold** than surface codes
6. The [[7,1,3]] color code equals the Steane code

---

## Daily Checklist

- [ ] I can construct color code stabilizers from a colored lattice
- [ ] I understand why color codes are CSS codes
- [ ] I can verify the 3-coloring property
- [ ] I know which gates are transversal for color codes
- [ ] I understand the trade-offs vs surface codes
- [ ] I can identify the [[7,1,3]] color code = Steane code

---

## Preview: Day 751

Tomorrow we explore **Reed-Muller quantum codes**:

- Classical Reed-Muller code construction
- Quantum RM codes and their duality
- The remarkable transversal T gate
- Applications to magic state distillation

Reed-Muller codes have special structure enabling non-Clifford transversal gates!
