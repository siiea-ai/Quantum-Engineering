# Week 115: Surface Code Implementation

## Month 29: Topological Codes | Semester 2A: Error Correction
### Year 2: Advanced Quantum Science

---

## Week Overview

Week 115 transitions from the abstract elegance of the toric code to the practical reality of the **planar surface code**—the leading architecture for fault-tolerant quantum computation. While the toric code's periodic boundary conditions provide mathematical simplicity, they require embedding a torus in three-dimensional space, which is physically impossible with planar chip layouts. The surface code resolves this by introducing boundaries that break translational symmetry while preserving the essential topological protection.

This week covers the complete theory and implementation of surface codes: boundary conditions and their stabilizer modifications, syndrome extraction circuits with careful attention to hook errors, logical operator construction on planar geometries, defect-based encodings for universal gates, and hardware-optimized variants like the rotated surface code.

---

## Learning Objectives

By the end of Week 115, you will be able to:

1. **Explain** why periodic boundaries are impractical and how smooth/rough boundaries replace them
2. **Construct** planar surface code layouts with correct data and ancilla qubit placement
3. **Design** syndrome extraction circuits that avoid hook error propagation
4. **Identify** logical operators as boundary-to-boundary strings on planar codes
5. **Describe** defects and holes as resources for encoding and logic
6. **Compare** rotated vs. unrotated surface codes for hardware efficiency

---

## Daily Schedule

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| **Day 799** | Monday | From Torus to Plane: Boundaries | ⬜ Not Started |
| **Day 800** | Tuesday | Planar Surface Code Structure | ⬜ Not Started |
| **Day 801** | Wednesday | Syndrome Extraction Circuits | ⬜ Not Started |
| **Day 802** | Thursday | Logical Operators on Planar Code | ⬜ Not Started |
| **Day 803** | Friday | Defects and Holes | ⬜ Not Started |
| **Day 804** | Saturday | Rotated Surface Code | ⬜ Not Started |
| **Day 805** | Sunday | Week 115 Synthesis | ⬜ Not Started |

---

## Prerequisites

Before beginning Week 115, ensure mastery of:

- **Week 114**: Toric code fundamentals, homological interpretation, anyon theory
- **Week 113**: Stabilizer formalism, error syndromes, CSS codes
- **Quantum Circuits**: CNOT gates, ancilla-based measurement
- **Graph Theory**: Planar graphs, boundaries, dual lattices

---

## Key Concepts Overview

### From Toric to Planar

The toric code lives on a torus (periodic boundaries), encoding 2 logical qubits. The **planar surface code** replaces periodicity with boundaries:

- **Smooth boundaries** (X-type): Terminate plaquette stabilizers
- **Rough boundaries** (Z-type): Terminate vertex stabilizers

A rectangular surface code with 2 smooth and 2 rough boundaries encodes exactly **1 logical qubit**.

### The $[[d^2, 1, d]]$ Code

For a distance-$d$ unrotated surface code:
- **Physical qubits**: $d^2$ data qubits
- **Logical qubits**: 1
- **Code distance**: $d$ (minimum weight of logical operator)

The **rotated surface code** achieves the same distance with only $\frac{d^2 + 1}{2}$ data qubits.

### Syndrome Extraction

Stabilizer measurements use **ancilla qubits** with CNOT circuits:
- X-stabilizers: Hadamard-CNOT pattern
- Z-stabilizers: Direct CNOT pattern

**Hook errors** occur when a single ancilla fault creates correlated data errors—mitigated by CNOT ordering.

### Defects and Holes

**Twist defects** (genons) and **holes** in the surface code:
- Provide alternative logical qubit encodings
- Enable non-Clifford gates via braiding
- Trade space for gate complexity

---

## Mathematical Framework

### Boundary Stabilizers

At smooth boundary (X-type):
$$S_p^{\text{boundary}} = \prod_{q \in \partial p} X_q \quad \text{(weight-3 or weight-2)}$$

At rough boundary (Z-type):
$$A_v^{\text{boundary}} = \prod_{q \sim v} Z_q \quad \text{(weight-3 or weight-2)}$$

### Logical Operators

For a surface code with smooth boundaries on left/right and rough on top/bottom:

$$\bar{X} = \prod_{i \in \text{horizontal path}} X_i \quad \text{(smooth to smooth)}$$

$$\bar{Z} = \prod_{j \in \text{vertical path}} Z_j \quad \text{(rough to rough)}$$

### Rotated Code Efficiency

Qubit count comparison for distance $d$:

| Variant | Data Qubits | Ancilla Qubits | Total |
|---------|-------------|----------------|-------|
| Unrotated | $d^2$ | $d^2 - 1$ | $2d^2 - 1$ |
| Rotated | $(d^2+1)/2$ | $(d^2-1)/2$ | $d^2$ |

---

## Computational Tools

This week's Python implementations include:

```python
# Core modules for Week 115
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from itertools import product

# Surface code utilities
class SurfaceCode:
    """Planar surface code with configurable boundaries."""

    def __init__(self, d, rotated=False):
        self.d = d
        self.rotated = rotated
        self.build_lattice()
        self.build_stabilizers()

    def build_lattice(self):
        """Construct qubit positions and neighbor relations."""
        pass  # Implemented in daily lessons

    def build_stabilizers(self):
        """Generate X and Z stabilizers with boundary terms."""
        pass  # Implemented in daily lessons
```

---

## Connections to Experiment

### IBM Quantum

IBM's Eagle and Heron processors implement **heavy-hex** surface codes:
- Modified connectivity for reduced crosstalk
- Native two-qubit gates aligned with surface code structure

### Google Quantum AI

Google's Sycamore demonstrated:
- Distance-3 and distance-5 surface codes
- Exponential suppression of logical errors with distance
- Real-time decoder integration

### Quantinuum

Ion trap implementations explore:
- All-to-all connectivity enabling non-local syndrome extraction
- Defect-based encodings without physical boundaries

---

## Week 115 Synthesis Goals

By Sunday, you should be able to:

1. Draw a distance-5 surface code with correct boundary types
2. Write syndrome extraction circuits avoiding hook errors
3. Trace logical X and Z operators between appropriate boundaries
4. Explain defect braiding for Clifford gates
5. Calculate qubit savings in rotated vs. unrotated layouts
6. Evaluate hardware tradeoffs for different surface code variants

---

## References

### Primary Sources

1. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." *Physical Review A* 86, 032324 (2012)
2. Dennis, E., et al. "Topological quantum memory." *Journal of Mathematical Physics* 43, 4452 (2002)
3. Bombin, H., & Martin-Delgado, M. A. "Topological quantum distillation." *Physical Review Letters* 97, 180501 (2006)

### Review Articles

4. Terhal, B. M. "Quantum error correction for quantum memories." *Reviews of Modern Physics* 87, 307 (2015)
5. Campbell, E. T., Terhal, B. M., & Vuillot, C. "Roads towards fault-tolerant universal quantum computation." *Nature* 549, 172 (2017)

### Implementation Guides

6. Google Quantum AI. "Suppressing quantum errors by scaling a surface code logical qubit." *Nature* 614, 676 (2023)
7. IBM Quantum. "High-threshold and low-overhead fault-tolerant quantum memory." arXiv:2308.07915 (2023)

---

## Navigation

- **Previous**: [Week 114 - Toric Code and Anyons](../Week_114_Toric_Code_Anyons/)
- **Next**: [Week 116 - Decoding and Thresholds](../Week_116_Decoding_Thresholds/)
- **Month 29 Overview**: [Topological Codes](../README.md)

---

*Week 115 brings topological codes from mathematical abstraction to physical implementation—the surface code is the workhorse of fault-tolerant quantum computing.*
