# Week 118: Lattice Surgery & Logical Gates

## Month 30: Surface Codes | Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Week Overview

**Focus:** Lattice Surgery Operations for Universal Fault-Tolerant Quantum Computation

**Duration:** Days 820-826 (7 days, ~49 hours total study time)

**Prerequisites:** Week 117 (Surface Code Fundamentals), Stabilizer formalism, Topological error correction basics

---

## Learning Goals

By the end of this week, you will be able to:

1. **Understand Lattice Surgery** - Master the code deformation paradigm for implementing logical gates without transversal operations
2. **Implement Merge Operations** - Execute XX and ZZ joint measurements to entangle surface code patches
3. **Perform Split Operations** - Separate merged patches while preserving logical information
4. **Construct Fault-Tolerant CNOT** - Build two-qubit gates using the merge-and-split protocol
5. **Design Multi-Patch Architectures** - Organize surface code patches for efficient parallel computation
6. **Integrate Magic States** - Inject T-gates via magic state distillation and lattice surgery

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **820** | Lattice Surgery Fundamentals | Code deformation, boundary types, rough/smooth edges |
| **821** | Merge Operations (XX and ZZ) | Joint measurements, boundary growth, parity extraction |
| **822** | Split Operations & State Preparation | Patch separation, logical state initialization |
| **823** | Surface Code CNOT via Lattice Surgery | ZZ measurement → XOR → ZZ correction protocol |
| **824** | Multi-Patch Architectures | Parallel gates, routing, defect-based operations |
| **825** | T-Gate Injection & Magic States | Magic state factories, gate teleportation, S correction |
| **826** | Week Synthesis | Complete protocols, resource estimation, algorithm compilation |

---

## Key Concepts

### Lattice Surgery Paradigm

Unlike transversal gates (which apply gates qubit-by-qubit across the code block), **lattice surgery** implements logical operations by:
- Growing and shrinking code boundaries
- Temporarily merging separate surface code patches
- Performing joint measurements on merged regions

This approach is **hardware-efficient** because:
- Only nearest-neighbor interactions required
- No long-range qubit connections needed
- Natural fit for 2D planar architectures (superconducting qubits, ion traps)

### Boundary Types

Surface codes have two boundary types:
- **Rough boundaries (Z-type):** Terminate Z stabilizers, support logical Z operators
- **Smooth boundaries (X-type):** Terminate X stabilizers, support logical X operators

### Merge and Split Operations

$$\boxed{\text{Merge: } |\psi\rangle \otimes |\phi\rangle \xrightarrow{M_{XX}} \frac{1}{\sqrt{2}}(|\psi\phi\rangle + X|\psi\rangle \otimes X|\phi\rangle)}$$

$$\boxed{\text{Split: Measured eigenvalue determines Pauli correction}}$$

---

## Mathematical Framework

### Logical Operators on Merged Patches

For two patches A and B merged along smooth boundaries:
- **Logical XX:** Product of original logical X operators
- **Logical ZZ:** Chain spanning both patches through merge region

### CNOT via Lattice Surgery

The canonical protocol:
1. Prepare ancilla patch in $|+\rangle_L$
2. ZZ merge between control and ancilla → measure $Z_C Z_A$
3. XX merge between ancilla and target → measure $X_A X_T$
4. Apply corrections based on measurement outcomes

$$\boxed{\text{CNOT} = H_T \cdot CZ \cdot H_T = (\text{ZZ merge}) \cdot (\text{XOR}) \cdot (\text{Pauli corrections})}$$

### T-Gate via Magic State Injection

$$T|+\rangle = |T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

Protocol:
1. Prepare magic state $|T\rangle_L$ via distillation
2. Merge with target patch (ZZ measurement)
3. Apply S correction if measurement outcome is -1

$$\boxed{T|\psi\rangle = S^m |T\rangle\langle T| |\psi\rangle, \text{ where } m = \text{measurement outcome}}$$

---

## Resources

### Primary References
- Horsman, Fowler, Devitt, Van Meter (2012). "Surface code quantum computing by lattice surgery"
- Litinski (2019). "A Game of Surface Codes: Large-scale quantum computing with lattice surgery"
- Fowler et al. (2012). "Surface codes: Towards practical large-scale quantum computation"

### Software Tools
- Stim: Fast Clifford circuit simulator with surface code support
- PyMatching: Minimum-weight perfect matching decoder
- Qiskit: General quantum circuit simulation

---

## Weekly Project

**Goal:** Implement a complete lattice surgery compiler that:
1. Takes a quantum circuit as input
2. Decomposes into Clifford + T gates
3. Converts to lattice surgery primitives
4. Estimates space-time volume (qubit-cycles)
5. Visualizes the surgery schedule

---

## Assessment Criteria

- [ ] Correctly simulate merge/split operations
- [ ] Implement fault-tolerant CNOT protocol
- [ ] Calculate resource overhead for T-gate injection
- [ ] Design multi-patch layout for 3+ logical qubits
- [ ] Compile a simple algorithm (e.g., Toffoli) to lattice surgery

---

## Connection to Research Frontiers

Lattice surgery is the **leading approach** for fault-tolerant quantum computation in:
- Google's surface code roadmap
- IBM's heavy-hex architecture adaptations
- AWS/IonQ trapped-ion 2D array plans

Recent advances (2024-2025):
- Twist-based operations for reduced overhead
- Floquet code adaptations of surgery protocols
- Hardware demonstrations of merge operations in superconducting systems

---

*"Lattice surgery transforms the abstract mathematics of topological codes into practical engineering blueprints for large-scale quantum computers."*
