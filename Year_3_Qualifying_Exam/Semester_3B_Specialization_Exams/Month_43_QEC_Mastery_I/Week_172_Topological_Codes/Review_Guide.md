# Week 172: Topological Codes - Review Guide

## Introduction

Topological quantum error-correcting codes represent the most promising approach to building large-scale fault-tolerant quantum computers. These codes encode quantum information in global, topological properties of a system that are inherently protected from local perturbations. The surface code, in particular, has emerged as the leading candidate for near-term quantum computing due to its high error threshold and compatibility with 2D architectures.

This review covers the toric code, anyonic excitations, the surface code, and practical implementation considerations.

---

## 1. Kitaev's Toric Code

### 1.1 Definition and Setup

The toric code is defined on a square lattice with periodic boundary conditions (a torus). Qubits live on the **edges** of the lattice.

For an $$L \times L$$ lattice:
- Number of edges (qubits): $$n = 2L^2$$
- Number of vertices: $$L^2$$
- Number of plaquettes (faces): $$L^2$$

### 1.2 Stabilizer Generators

**Vertex operators (star operators):**
$$A_v = \prod_{e \ni v} X_e$$

For each vertex $$v$$, apply $$X$$ to all four edges meeting at $$v$$.

**Plaquette operators:**
$$B_p = \prod_{e \in \partial p} Z_e$$

For each plaquette (face) $$p$$, apply $$Z$$ to all four edges on the boundary.

### 1.3 Code Space

The code space is the simultaneous +1 eigenspace of all $$A_v$$ and $$B_p$$:

$$\mathcal{C} = \{|\psi\rangle : A_v|\psi\rangle = |\psi\rangle, B_p|\psi\rangle = |\psi\rangle \text{ for all } v, p\}$$

**Counting generators:**
- $$L^2$$ vertex operators, but $$\prod_v A_v = I$$ (one constraint)
- $$L^2$$ plaquette operators, but $$\prod_p B_p = I$$ (one constraint)
- Independent generators: $$2L^2 - 2$$

**Code dimension:**
$$k = n - (2L^2 - 2) = 2L^2 - 2L^2 + 2 = 2$$

The toric code encodes **2 logical qubits**.

### 1.4 Logical Operators

The logical operators are **non-contractible loops** on the torus:

**Logical $$\overline{Z}_1$$:** Product of $$Z$$ on edges along a horizontal non-contractible cycle.

**Logical $$\overline{X}_1$$:** Product of $$X$$ on edges crossing a vertical non-contractible cycle.

**Logical $$\overline{Z}_2$$:** Product of $$Z$$ along a vertical non-contractible cycle.

**Logical $$\overline{X}_2$$:** Product of $$X$$ crossing a horizontal non-contractible cycle.

These operators commute with all stabilizers but anticommute in pairs:
$$\{\overline{X}_i, \overline{Z}_i\} = 0, \quad [\overline{X}_i, \overline{Z}_j] = 0 \text{ for } i \neq j$$

### 1.5 Code Distance

The distance is the minimum weight of a logical operator = minimum length of a non-contractible cycle = $$L$$.

**Parameters:** $$[[2L^2, 2, L]]$$

### 1.6 Ground State Structure

The ground state manifold is 4-dimensional (encoding 2 qubits). The four ground states can be labeled by the eigenvalues of $$\overline{Z}_1$$ and $$\overline{Z}_2$$:

$$|00\rangle_L, |01\rangle_L, |10\rangle_L, |11\rangle_L$$

This degeneracy is **topological**: it cannot be lifted by any local perturbation and is robust to local errors.

---

## 2. Anyonic Excitations

### 2.1 Excitations as Anyons

Violations of stabilizer constraints create localized excitations:

**Electric charges (e):** Violations of vertex operators ($$A_v = -1$$)
- Created by $$Z$$ errors on edges
- A string of $$Z$$ operators creates $$e$$ particles at endpoints

**Magnetic vortices (m):** Violations of plaquette operators ($$B_p = -1$$)
- Created by $$X$$ errors on edges
- A string of $$X$$ operators creates $$m$$ particles at endpoints

### 2.2 Fusion Rules

When two excitations of the same type meet, they can annihilate:

$$e \times e = 1$$ (vacuum)
$$m \times m = 1$$

When $$e$$ and $$m$$ meet:
$$e \times m = \epsilon$$

where $$\epsilon$$ is a **fermion** (composite excitation).

### 2.3 Braiding Statistics

The key topological property: braiding an $$e$$ particle around an $$m$$ particle (or vice versa) produces a phase of $$-1$$.

**Proof:**
Let $$\gamma_e$$ be the path of $$e$$ (a string of $$Z$$ operators).
Let $$\gamma_m$$ be the path of $$m$$ (a string of $$X$$ operators).

When $$\gamma_e$$ encloses $$\gamma_m$$, the operators intersect an odd number of times:
$$Z \cdot X = -X \cdot Z$$

This gives the $$-1$$ braiding phase.

### 2.4 Topological Order

The toric code exhibits **Z₂ topological order**:
- Ground state degeneracy depends on topology (2^{2g} for genus g)
- No local order parameter distinguishes ground states
- Excitations have non-trivial braiding statistics

---

## 3. The Surface Code

### 3.1 From Torus to Plane

The surface code is the toric code on a plane with boundaries. Boundaries break the periodic structure and reduce the encoded qubits.

**Two boundary types:**

**Rough boundary:** Terminates plaquettes, $$Z$$ stabilizers incomplete
- Allows $$X$$-type logical operators to end on boundary

**Smooth boundary:** Terminates vertices, $$X$$ stabilizers incomplete
- Allows $$Z$$-type logical operators to end on boundary

### 3.2 Standard Surface Code

The standard surface code has:
- Rough boundaries on top and bottom
- Smooth boundaries on left and right

This configuration encodes **1 logical qubit**.

### 3.3 Code Parameters

For a distance-$$d$$ surface code:

**Rotated layout (most efficient):**
$$[[d^2, 1, d]]$$

**Standard layout:**
$$[[(2d-1)^2, 1, d]]$$

The rotated layout uses fewer qubits by tilting the lattice 45°.

### 3.4 Logical Operators

**Logical $$\overline{Z}$$:** String of $$Z$$ operators connecting left and right boundaries (smooth to smooth).

**Logical $$\overline{X}$$:** String of $$X$$ operators connecting top and bottom boundaries (rough to rough).

The distance equals the minimum path length between opposite boundaries.

### 3.5 Stabilizer Measurements

Each stabilizer involves 4 qubits (or 2-3 at boundaries):

**X-stabilizer measurement circuit:**
1. Initialize ancilla in $$|+\rangle$$
2. Apply CNOT from each data qubit to ancilla
3. Measure ancilla in X basis

**Z-stabilizer measurement circuit:**
1. Initialize ancilla in $$|0\rangle$$
2. Apply CNOT from ancilla to each data qubit
3. Measure ancilla in Z basis

### 3.6 Error Correction Process

1. **Syndrome extraction:** Measure all stabilizers
2. **Syndrome interpretation:** Identify which stabilizers give $$-1$$
3. **Decoding:** Match $$-1$$ outcomes to infer error
4. **Correction:** Apply recovery operation (or track in software)

---

## 4. Decoding the Surface Code

### 4.1 Error Model

Under the standard depolarizing model, errors create pairs of excitations:
- $$X$$ error → pair of $$m$$ excitations
- $$Z$$ error → pair of $$e$$ excitations
- $$Y$$ error → both types

### 4.2 Minimum-Weight Perfect Matching (MWPM)

The standard decoder for the surface code:

1. **Build syndrome graph:**
   - Nodes: syndrome locations (violated stabilizers)
   - Edges: weighted by distance (error probability)

2. **Find minimum-weight perfect matching:**
   - Pair up all syndrome nodes
   - Minimize total edge weight
   - Use Edmonds' blossom algorithm (polynomial time)

3. **Determine correction:**
   - For each matched pair, errors occurred on connecting path
   - Apply correction (or update Pauli frame)

### 4.3 Error Threshold

**Theorem (Dennis et al., 2002):** The surface code has an error threshold of approximately 11% for independent $$X$$ and $$Z$$ errors.

For depolarizing noise: $$p_{\text{th}} \approx 1\%$$

With measurement errors (phenomenological): $$p_{\text{th}} \approx 3\%$$

With circuit-level noise: $$p_{\text{th}} \approx 0.5-1\%$$

### 4.4 Logical Error Rate

Below threshold, the logical error rate scales as:

$$p_L \sim \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

Increasing distance $$d$$ exponentially suppresses errors.

---

## 5. Logical Operations

### 5.1 Transversal Gates

The surface code has limited transversal gates:
- $$\overline{X} = X^{\otimes n}$$ (not exactly, but related)
- $$\overline{Z} = Z^{\otimes n}$$

**No transversal:** H, S, T, CNOT (within a single patch)

### 5.2 Lattice Surgery

**Lattice surgery** performs logical gates by merging and splitting code patches:

**Merge operation:**
- Bring two patches together
- Measure joint stabilizers
- Projects onto joint eigenspace
- Implements measurements like $$\overline{Z}_1 \overline{Z}_2$$

**Split operation:**
- Reverse of merge
- Separate a patch into two
- Implements state preparation/teleportation

**Logical CNOT via surgery:**
1. Merge patches to measure $$\overline{Z}_1 \overline{Z}_2$$
2. Apply corrections based on measurement
3. Split patches

### 5.3 Magic State Distillation

For T gates, use **magic state distillation**:

1. **Prepare noisy magic states:** $$|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$$
2. **Distillation protocol:** Use multiple noisy states to produce fewer, higher-fidelity states
3. **Inject into computation:** Consume magic states for T gates

**15-to-1 distillation:** Uses 15 noisy $$|T\rangle$$ states to produce 1 higher-quality state.

**Overhead:** Magic state distillation dominates the resource cost of surface code quantum computing.

### 5.4 Universal Computation

Combining:
- Lattice surgery for Clifford gates
- Magic state injection for T gates

gives universal quantum computation on the surface code.

---

## 6. Practical Considerations

### 6.1 Physical Requirements

**Connectivity:** 2D nearest-neighbor (matches superconducting architectures)

**Measurement:** Mid-circuit measurement required for syndrome extraction

**Cycle time:** Syndrome measurement every ~1 μs

### 6.2 Resource Estimates

For useful quantum computation:
- Thousands to millions of physical qubits
- Distance $$d \sim 15-30$$ for target logical error rates
- Magic state factories consume significant resources

**Example:** 1000 logical qubits at $$10^{-10}$$ error rate with $$p = 10^{-3}$$:
- ~$$10^6$$ physical qubits for data
- ~$$10^7$$ physical qubits including magic states

### 6.3 Current Status (2026)

- Demonstrations of small surface codes
- Logical qubit lifetimes exceeding physical
- Distance-3 and distance-5 implementations
- Break-even point reached for some operations

---

## 7. Summary

### Key Results

| Concept | Result |
|---------|--------|
| Toric code parameters | $$[[2L^2, 2, L]]$$ |
| Surface code parameters | $$[[d^2, 1, d]]$$ (rotated) |
| Anyon braiding | $$e$$ around $$m$$ gives $$-1$$ |
| Error threshold | ~1% (circuit-level) |
| Logical error scaling | $$p_L \sim (p/p_{\text{th}})^{(d+1)/2}$$ |

### Exam Preparation

**Know:**
1. Toric code construction and stabilizers
2. Anyon types and braiding
3. Surface code boundaries and logical operators
4. MWPM decoding principle
5. Lattice surgery operations

**Be able to:**
- Draw surface code patches with boundaries
- Identify logical operators
- Explain why threshold exists
- Describe path to universal computation

---

## References

1. Kitaev, A. "Fault-tolerant quantum computation by anyons" [arXiv:quant-ph/9707021](https://arxiv.org/abs/quant-ph/9707021)

2. Dennis, E. et al. "Topological quantum memory" [arXiv:quant-ph/0110143](https://arxiv.org/abs/quant-ph/0110143)

3. Fowler, A.G. et al. "Surface codes: Towards practical large-scale quantum computation" [arXiv:1208.0928](https://arxiv.org/abs/1208.0928)

4. Horsman, C. et al. "Surface code quantum computing by lattice surgery" [arXiv:1111.4022](https://arxiv.org/abs/1111.4022)

---

**Word Count:** ~2300 words
**Review Guide Created:** February 10, 2026
