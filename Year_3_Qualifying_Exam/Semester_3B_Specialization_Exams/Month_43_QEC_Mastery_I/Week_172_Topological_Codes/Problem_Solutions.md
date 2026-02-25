# Week 172: Topological Codes - Problem Solutions

## Part A: The Toric Code

### Solution 1

**(a)** For $$L = 2$$ toric code:

```
v1----e1----v2
|           |
e2    p1    e3
|           |
v3----e4----v4
|           |
e5    p2    e6
|           |
v1----e7----v2   (wraps around)
```

With periodic boundaries, we have:
- 4 vertices: $$v_1, v_2, v_3, v_4$$
- 4 plaquettes: $$p_1, p_2, p_3, p_4$$
- 8 edges (qubits)

**(b)** Number of qubits: $$n = 2L^2 = 2 \times 4 = 8$$

**(c)** Vertex operators (4 total):
- $$A_{v_1} = X_{e_1}X_{e_2}X_{e_5}X_{e_7}$$ (edges meeting at $$v_1$$)
- $$A_{v_2} = X_{e_1}X_{e_3}X_{e_6}X_{e_7}$$
- $$A_{v_3} = X_{e_2}X_{e_4}X_{e_5}X_{e_8}$$
- $$A_{v_4} = X_{e_3}X_{e_4}X_{e_6}X_{e_8}$$

**(d)** Plaquette operators (4 total):
- $$B_{p_1} = Z_{e_1}Z_{e_2}Z_{e_3}Z_{e_4}$$
- Similar for $$p_2, p_3, p_4$$

---

### Solution 2

**(a)** $$\prod_v A_v$$:

Each edge appears in exactly two vertex operators (its two endpoints).
$$X_e \cdot X_e = I$$ for each edge.
Therefore $$\prod_v A_v = I$$ ✓

**(b)** $$\prod_p B_p$$:

Each edge appears in exactly two plaquettes (one on each side).
$$Z_e \cdot Z_e = I$$ for each edge.
Therefore $$\prod_p B_p = I$$ ✓

**(c)** Independent generators:
- 4 vertex operators with 1 constraint: 3 independent
- 4 plaquette operators with 1 constraint: 3 independent
- Total: 6 independent generators

**(d)** Code parameters:
- $$n = 8$$ qubits
- $$k = n - 6 = 2$$ logical qubits
- $$d = L = 2$$

$$[[8, 2, 2]]$$

---

### Solution 3

**(a)** If $$v$$ and $$p$$ share no edges:
$$A_v$$ acts on edges at $$v$$, $$B_p$$ acts on edges around $$p$$.
Disjoint support → operators commute trivially.

**(b)** If $$v$$ and $$p$$ share exactly 2 edges:
$$A_v = X_{e_1} X_{e_2} \cdots$$ (includes $$e_1, e_2$$ shared with $$p$$)
$$B_p = Z_{e_1} Z_{e_2} \cdots$$

At each shared edge: $$X$$ and $$Z$$ anticommute.
Two anticommutations: $$(-1)^2 = 1$$
Therefore $$[A_v, B_p] = 0$$ ✓

**(c)** Geometry prevents 1 or 3 shared edges:
- A vertex has 4 edges radiating out
- A plaquette has 4 edges forming a square
- A vertex is either a corner of a plaquette (2 shared) or not (0 shared)

---

### Solution 4

**(a)** $$\overline{Z}_1$$: Horizontal loop around the torus.
For $$L = 2$$: $$\overline{Z}_1 = Z_{e_1}Z_{e_7}$$ (horizontal edges in one row)

**(b)** $$\overline{X}_1$$: Dual path (perpendicular edges).
$$\overline{X}_1 = X_{e_2}X_{e_5}$$ (vertical edges in one column)

**(c)** $$\overline{X}_1\overline{Z}_1 = X_{e_2}X_{e_5} \cdot Z_{e_1}Z_{e_7}$$

If $$e_2$$ and $$e_1$$ share a vertex: one $$XZ$$ anticommutation.
Check geometry: they share vertex $$v_1$$.
Only one shared position → anticommute.

$$\{\overline{X}_1, \overline{Z}_1\} = 0$$ ✓

**(d)** $$\overline{Z}_2$$: Vertical loop.
$$\overline{X}_2$$: Horizontal dual path.

**(e)** $$[\overline{X}_1, \overline{X}_2] = 0$$ (both X-type, always commute)
$$[\overline{Z}_1, \overline{Z}_2] = 0$$ (both Z-type)
$$[\overline{X}_1, \overline{Z}_2] = 0$$ (different logical qubits, perpendicular)
$$[\overline{X}_2, \overline{Z}_1] = 0$$ (perpendicular)
$$\{\overline{X}_1, \overline{Z}_1\} = 0$$ ✓
$$\{\overline{X}_2, \overline{Z}_2\} = 0$$ ✓

---

### Solution 5

**(a)** $$|\psi\rangle = \prod_v \frac{1+A_v}{2}|0\rangle^{\otimes n}$$

$$A_v \frac{1+A_v}{2} = \frac{A_v + A_v^2}{2} = \frac{A_v + I}{2} = \frac{1+A_v}{2}$$

So $$A_v|\psi\rangle = |\psi\rangle$$ ✓

**(b)** For plaquettes: $$B_p|0\rangle^{\otimes n} = |0\rangle^{\otimes n}$$ since $$Z|0\rangle = |0\rangle$$.

The projectors $$\frac{1+A_v}{2}$$ preserve this because $$[A_v, B_p] = 0$$.

So $$B_p|\psi\rangle = |\psi\rangle$$ ✓

**(c)** The state is a superposition of loop configurations (closed strings of 1s).

For $$L = 2$$: $$|GS\rangle$$ includes $$|00000000\rangle$$ plus states with closed loops.

---

### Solution 6

**(a)** For genus $$g$$: $$2g$$ non-contractible cycles ($$g$$ pairs).

**(b)** Ground state degeneracy: $$2^{2g} = 4^g$$

**(c)** Parameters: $$[[n, 2g, d]]$$ where $$n$$ and $$d$$ depend on the specific lattice.

**(d)** For $$g = 2$$: $$k = 2 \times 2 = 4$$ logical qubits.

---

## Part B: Anyonic Excitations

### Solution 8

**(a)** $$Z$$ on edge $$e$$ anticommutes with $$A_v$$ for $$v$$ at both endpoints of $$e$$.

Violated stabilizers: $$A_{v_1}$$ and $$A_{v_2}$$ where $$e = (v_1, v_2)$$.

**(b)** Two $$e$$ particles at $$v_1$$ and $$v_2$$.

**(c)** Syndrome: $$-1$$ on $$A_{v_1}$$ and $$A_{v_2}$$, $$+1$$ elsewhere.

---

### Solution 9

**(a)** Closed contractible loop: Each vertex on the loop is touched twice by the string.
$$(-1)^2 = +1$$ for each vertex.
Syndrome: all $$+1$$ (no excitations).

**(b)** Path from $$v_1$$ to $$v_2$$: Endpoints touched once, interior vertices touched twice.
Syndrome: $$-1$$ on $$A_{v_1}$$ and $$A_{v_2}$$ ($$e$$ particles at endpoints).

**(c)** Non-contractible loop: All vertices touched twice (loop closes on torus).
No local excitations, but applies logical $$\overline{Z}$$!

---

### Solution 10

**(a)-(c)** Let $$Z_{\gamma_e}$$ be the $$Z$$ string moving $$e$$ around a closed path encircling plaquette $$p$$.

The $$m$$ particle at $$p$$ was created by an $$X$$ string $$X_\mu$$ from the boundary of $$p$$.

When $$\gamma_e$$ encircles $$p$$:
$$Z_{\gamma_e}$$ crosses $$X_\mu$$ at one edge (the entry point into the encircled region).

$$Z \cdot X = -X \cdot Z$$ at that edge.

The state picks up a phase: $$Z_{\gamma_e}|\psi_m\rangle = -|\psi_m'\rangle$$

This is the $$-1$$ braiding phase. ✓

---

### Solution 11

**(a)** Two $$e$$ particles at same vertex: Both created by $$Z$$ strings ending there.
Combined: $$Z$$-string ending at vertex twice = $$Z$$-string passing through.
Net effect: no excitation at that vertex.
$$e \times e = 1$$ ✓

**(b)** Self-statistics of $$e$$: Exchange two $$e$$ particles.
This is equivalent to braiding $$e$$ around itself... trivial phase.
$$e$$ is a **boson**.

**(c)** $$\epsilon = e \times m$$ picks up $$-1$$ when exchanged with another $$\epsilon$$.
$$\epsilon$$ is a **fermion**.

---

## Part C: Surface Code

### Solution 13

Distance-3 surface code (rotated):

```
    [Z]     [Z]     [Z]
  /  |  \ /  |  \ /  |  \
 D---X---D---X---D---X---D
  \  |  / \  |  / \  |  /
    [Z]     [Z]     [Z]
  /  |  \ /  |  \ /  |  \
 D---X---D---X---D---X---D
  \  |  / \  |  / \  |  /
    [Z]     [Z]     [Z]
  /  |  \ /  |  \ /  |  \
 D---X---D---X---D---X---D
```

Where D = data qubit, X = X-stabilizer ancilla, [Z] = Z-stabilizer ancilla.

**(a)** Data qubits: 9 (for $$d = 3$$)
Measurement qubits: 8 (4 X-type + 4 Z-type)

**(b)** X-stabilizers: weight-4 in interior, weight-2 on rough boundaries
Z-stabilizers: weight-4 in interior, weight-2 on smooth boundaries

**(c)** $$d^2 = 9$$ data qubits.

---

### Solution 14

**(a)** $$\overline{Z}$$: String of $$Z$$ connecting left to right (smooth to smooth boundaries).
$$\overline{Z} = Z_1 Z_4 Z_7$$ (leftmost column)

**(b)** $$\overline{X}$$: String of $$X$$ connecting top to bottom (rough to rough).
$$\overline{X} = X_1 X_2 X_3$$ (top row)

**(c)** $$\overline{X}$$ and $$\overline{Z}$$ share qubit 1.
$$X_1 Z_1 = -Z_1 X_1$$
They anticommute. ✓

**(d)** Minimum weight: $$d = 3$$ for both.

---

### Solution 15

**(a)** For rotated distance-$$d$$ code: $$n = d^2$$ data qubits.

**(b)** X-stabilizers: $$(d^2 - 1)/2$$ for odd $$d$$.

**(c)** Z-stabilizers: $$(d^2 - 1)/2$$.

**(d)** $$k = d^2 - (d^2-1)/2 - (d^2-1)/2 = d^2 - d^2 + 1 = 1$$ ✓

---

### Solution 16

**(a)-(b)** Z-stabilizer measurement circuit:

```
Ancilla: |0⟩---CNOT---CNOT---CNOT---CNOT---M
               |      |      |      |
Data:    [q1]--●------+------+------+------
         [q2]---------●------+------+------
         [q3]----------------●------+------
         [q4]-----------------------●------
```

**(c)** Weight-2 boundary stabilizer: Only 2 CNOTs needed.

**(d)** Single CNOT failure: Ancilla may flip, giving wrong syndrome for one round.
Solved by repeated measurements and majority voting / space-time decoding.

---

### Solution 17

**(a)** Single $$X$$ error at qubit $$q$$: Detected by adjacent Z-stabilizers.
Typically 2 Z-stabilizers (4 if at a corner, 2 at edge, 1 impossible).

**(b)** Syndrome: Two $$-1$$ values at adjacent Z-stabilizers.

**(c)** MWPM creates a graph:
- Nodes at syndrome locations
- Edge weights proportional to distance
- Matches the two nodes (pairs them)
- Infers error on shortest path between them

**(d)** Correction: Apply $$X$$ on the edge connecting the matched syndromes.
(Or track in Pauli frame.)

---

## Part D: Logical Operations

### Solution 20

**(a)** Transversal gates: $$\overline{X} = X^{\otimes n}$$, $$\overline{Z} = Z^{\otimes n}$$, identity.

Actually, these aren't quite right for the surface code. The true transversal gates are just Paulis.

**(b)** Hadamard swaps X and Z. But in the surface code, X and Z stabilizers have different boundary conditions. $$H^{\otimes n}$$ maps X-stabilizers to Z-stabilizers with different support structure.

This doesn't preserve the code → not transversal.

**(c)** CNOT between two patches: Would need to entangle corresponding qubits, but the stabilizer structures don't align properly.

Not transversal in the naive sense.

---

### Solution 21

**(a)** Two patches side by side:
```
[Patch 1]  |  [Patch 2]
           |
```

**(b)** Merge: Measure joint stabilizers across the boundary.
New Z-stabilizers span the merged region.

**(c)** Effect: Projects onto eigenspace of $$\overline{Z}_1 \overline{Z}_2$$.
Measurement outcome determines eigenvalue.

**(d)** Split: Re-measure individual stabilizers, disentangling the patches.

---

### Solution 22

**(a)** For CNOT: Measure $$\overline{Z}_1 \overline{Z}_2$$ and $$\overline{X}_1 \overline{X}_2$$ via merges.

**(b)** Sequence:
1. Merge horizontally (measures $$\overline{Z}_c \overline{Z}_t$$)
2. Split
3. Merge vertically (measures $$\overline{X}_c$$)
4. Apply corrections based on measurements

**(c)** Time overhead: Several code cycles per merge/split.
Much slower than transversal CNOT (~10-100x).

**(d)** Pauli corrections from each measurement must be tracked through the circuit.

---

### Solution 23

**(a)** Magic state: $$|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2} = T|+\rangle$$

**(b)** 15-to-1 protocol:
- Encode 15 noisy $$|T\rangle$$ states in a [[15, 1, 3]] code
- Measure stabilizers
- If all pass: output 1 high-quality state
- Error rate: $$p_{\text{out}} \approx 35 p^3$$ (cubic suppression)

**(c)** With $$p_{\text{in}} = p$$: $$p_{\text{out}} \approx 35 p^3$$

**(d)** To reach $$10^{-15}$$ from $$10^{-2}$$:

Level 1: $$35 \times (10^{-2})^3 = 3.5 \times 10^{-5}$$
Level 2: $$35 \times (3.5 \times 10^{-5})^3 \approx 10^{-12}$$
Level 3: $$\approx 10^{-34}$$ (well below target)

**2-3 levels** of distillation needed.

---

### Solution 24

**(a)** For 2048-bit factoring: ~4000 logical qubits (rough estimate).

**(b)** T gates: ~$$10^{10}$$ (varies by implementation).

**(c)** With $$d = 17$$: $$d^2 = 289$$ data qubits per logical.
$$4000 \times 289 \approx 1.2 \times 10^6$$ data qubits.

**(d)** Magic state factories: Need to produce $$~10^{10}$$ T states.
Factory size ~$$10^4$$ qubits, need multiple factories.
Total: ~$$10^7$$ qubits including factories.

---

## Part E: Advanced Topics

### Solution 25

**(a)** Surface code = toric code with boundaries.
Boundaries break periodicity and reduce logical qubits.

**(b)** Toric code: 2 logical qubits (two non-contractible cycles).
Surface code: 1 logical qubit (one pair of boundaries).

**(c)** Surface code is more practical:
- Flat 2D layout (no periodic boundaries)
- Easier to implement in hardware
- Natural termination for finite systems

---

### Solution 26

**(a)** 3D surface code: Qubits on faces of a 3D cubic lattice.
X-stabilizers on edges, Z-stabilizers on cubes.

**(b)** Threshold in 3D: Higher than 2D (~3-4% vs ~1%).
3D has better percolation properties.

**(c)** 3D codes can have transversal CCZ gate.
Some constructions give transversal T.

**(d)** Trade-off: 3D is harder to build physically.
Most hardware is inherently 2D.

---

**Solutions Document Created:** February 10, 2026
