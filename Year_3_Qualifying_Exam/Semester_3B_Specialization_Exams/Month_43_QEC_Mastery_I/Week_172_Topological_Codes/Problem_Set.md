# Week 172: Topological Codes - Problem Set

## Instructions

This problem set contains 28 problems covering the toric code, anyonic excitations, surface codes, and practical implementation. Problems range from basic stabilizer calculations to research-level analysis.

**Levels:**
- **Level I:** Direct application of definitions
- **Level II:** Multi-step analysis and proofs
- **Level III:** Challenging synthesis problems

Time estimate: 15-20 hours total

---

## Part A: The Toric Code (Problems 1-7)

### Problem 1 (Level I)
For the toric code on an $$L \times L$$ square lattice with periodic boundary conditions:

(a) Draw the lattice for $$L = 2$$, labeling edges (qubits), vertices, and plaquettes.

(b) How many qubits are there?

(c) Write down all vertex operators $$A_v$$ explicitly.

(d) Write down all plaquette operators $$B_p$$ explicitly.

### Problem 2 (Level I)
For the $$L = 2$$ toric code:

(a) Verify that $$\prod_v A_v = I$$.

(b) Verify that $$\prod_p B_p = I$$.

(c) How many independent stabilizer generators are there?

(d) What are the code parameters $$[[n, k, d]]$$?

### Problem 3 (Level II)
Prove that vertex operators commute with plaquette operators: $$[A_v, B_p] = 0$$ for all $$v, p$$.

(a) Consider a vertex $$v$$ and plaquette $$p$$ that share no edges. Why do they commute?

(b) Consider $$v$$ and $$p$$ sharing exactly two edges. Show they commute.

(c) Can $$v$$ and $$p$$ share one or three edges? Explain.

### Problem 4 (Level II)
Find the logical operators for the $$L = 2$$ toric code.

(a) Identify $$\overline{Z}_1$$ as a horizontal non-contractible loop.

(b) Identify $$\overline{X}_1$$ as the dual vertical path.

(c) Verify $$\{\overline{X}_1, \overline{Z}_1\} = 0$$.

(d) Find $$\overline{X}_2$$ and $$\overline{Z}_2$$ similarly.

(e) Verify all commutation/anticommutation relations.

### Problem 5 (Level II)
The ground state of the toric code can be constructed as:

$$|GS\rangle = \prod_v \frac{1 + A_v}{2} |0\rangle^{\otimes n}$$

(a) Show this is a +1 eigenstate of all $$A_v$$.

(b) Show this is a +1 eigenstate of all $$B_p$$.

(c) Write the ground state as a superposition of computational basis states for $$L = 2$$.

### Problem 6 (Level III)
Consider the toric code on a surface of genus $$g$$ (a surface with $$g$$ handles).

(a) How many non-contractible cycles are there?

(b) What is the ground state degeneracy?

(c) What are the code parameters?

(d) For $$g = 2$$, how many logical qubits are encoded?

### Problem 7 (Level III)
The toric code Hamiltonian is:

$$H = -\sum_v A_v - \sum_p B_p$$

(a) What is the energy of the ground state(s)?

(b) What is the energy gap to the first excited state?

(c) How does a single $$X$$ error affect the energy?

(d) Explain why the code is stable at zero temperature but not at finite temperature (in 2D).

---

## Part B: Anyonic Excitations (Problems 8-12)

### Problem 8 (Level I)
Consider applying a single $$Z$$ error on edge $$e$$ in the toric code.

(a) Which stabilizers are violated?

(b) Draw the locations of the resulting excitations (e particles).

(c) What is the syndrome?

### Problem 9 (Level I)
Consider applying a string of $$Z$$ operators along a path $$\gamma$$.

(a) If $$\gamma$$ is a closed contractible loop, what is the syndrome?

(b) If $$\gamma$$ connects two vertices $$v_1$$ and $$v_2$$, what is the syndrome?

(c) What happens if $$\gamma$$ is a non-contractible loop?

### Problem 10 (Level II)
Prove that braiding an $$e$$ particle around an $$m$$ particle gives a phase of $$-1$$.

(a) Let $$\gamma_e$$ be the path of the $$e$$ particle ($$Z$$ string).

(b) Let the $$m$$ particle sit at plaquette $$p$$.

(c) If $$\gamma_e$$ encircles $$p$$ once, show the state picks up a $$-1$$ phase.

*Hint:* Consider the commutation of $$Z_{\gamma_e}$$ with the $$X$$ string creating $$m$$.

### Problem 11 (Level II)
The fusion rules for toric code anyons are:

$$e \times e = 1, \quad m \times m = 1, \quad e \times m = \epsilon$$

(a) Verify $$e \times e = 1$$ by considering what happens when two $$e$$ particles meet.

(b) What is the self-statistics of $$e$$? (Bosonic or fermionic?)

(c) What is the statistics of $$\epsilon$$?

### Problem 12 (Level III)
The toric code is an example of $$\mathbb{Z}_2$$ topological order.

(a) Define topological order in terms of ground state properties.

(b) Explain why no local operator can distinguish the four ground states.

(c) What is the "topological S-matrix" for the toric code?

(d) How is the S-matrix related to braiding?

---

## Part C: Surface Code (Problems 13-19)

### Problem 13 (Level I)
Draw a distance-3 surface code with:
- Rough boundaries on top and bottom
- Smooth boundaries on left and right

(a) Label all data qubits and measurement qubits.

(b) Identify X-type and Z-type stabilizers.

(c) How many data qubits are there?

### Problem 14 (Level I)
For the distance-3 surface code:

(a) Draw the logical $$\overline{Z}$$ operator.

(b) Draw the logical $$\overline{X}$$ operator.

(c) Verify they anticommute.

(d) What is the minimum weight of each logical operator?

### Problem 15 (Level II)
For a distance-$$d$$ rotated surface code:

(a) How many data qubits are there?

(b) How many X-stabilizers?

(c) How many Z-stabilizers?

(d) Verify $$k = n - (\text{X-stabilizers}) - (\text{Z-stabilizers}) = 1$$.

### Problem 16 (Level II)
Design the syndrome measurement circuit for a weight-4 Z-stabilizer.

(a) Draw the circuit using an ancilla qubit.

(b) What is the gate sequence?

(c) How do you handle the case where the stabilizer is on the boundary (weight 2)?

(d) Analyze what happens if one CNOT gate fails.

### Problem 17 (Level II)
Consider a single $$X$$ error on the surface code.

(a) Which stabilizers detect this error?

(b) Draw the syndrome pattern.

(c) How does the MWPM decoder process this syndrome?

(d) What correction is applied?

### Problem 18 (Level III)
The error threshold can be related to a phase transition.

(a) Explain the mapping from surface code decoding to the random-bond Ising model.

(b) The threshold corresponds to what physical quantity in the Ising model?

(c) Why does dimensionality matter for the threshold?

(d) What is the approximate threshold for independent X/Z errors?

### Problem 19 (Level III)
Analyze measurement errors in the surface code.

(a) In the phenomenological noise model, what is a measurement error?

(b) How does the decoder handle measurement errors?

(c) Why is "space-time" decoding necessary?

(d) How does the threshold change with measurement errors?

---

## Part D: Logical Operations (Problems 20-24)

### Problem 20 (Level I)
The surface code has very limited transversal gates.

(a) What gates can be implemented transversally?

(b) Why can't Hadamard be implemented transversally on a single patch?

(c) What about CNOT between two patches?

### Problem 21 (Level II)
Explain lattice surgery for performing a logical $$\overline{Z}_1 \overline{Z}_2$$ measurement.

(a) Draw two surface code patches before the merge.

(b) What stabilizers are measured during the merge?

(c) What is the effect on the logical state?

(d) How is the merge operation reversed (split)?

### Problem 22 (Level II)
A logical CNOT can be implemented via lattice surgery.

(a) What measurements are needed?

(b) Draw the sequence of merge/split operations.

(c) What is the time overhead compared to a transversal CNOT?

(d) How do Pauli corrections propagate through the surgery?

### Problem 23 (Level III)
Magic state distillation is required for T gates.

(a) Define the magic state $$|T\rangle$$.

(b) Describe the 15-to-1 distillation protocol.

(c) If input magic states have error rate $$p$$, what is the output error rate?

(d) How many levels of distillation are needed to achieve error rate $$10^{-15}$$ from $$p = 10^{-2}$$?

### Problem 24 (Level III)
Estimate the resource overhead for running Shor's algorithm on the surface code.

(a) How many logical qubits are needed to factor a 2048-bit number?

(b) How many T gates are required?

(c) With physical error rate $$10^{-3}$$ and code distance 17, estimate physical qubits for data.

(d) Estimate magic state factory requirements.

---

## Part E: Advanced Topics (Problems 25-28)

### Problem 25 (Level II)
Compare the toric code and surface code.

(a) What is the relationship between them?

(b) Why does the surface code encode fewer qubits?

(c) Which is more practical for implementation? Why?

### Problem 26 (Level III)
The surface code can be generalized to higher dimensions.

(a) Describe the 3D surface code.

(b) What is the threshold behavior in 3D?

(c) Does the 3D code have a transversal T gate?

(d) What is the trade-off with 2D codes?

### Problem 27 (Level III)
Decoders beyond MWPM exist for the surface code.

(a) Describe the Union-Find decoder and its complexity.

(b) What is a neural network decoder?

(c) Compare MWPM, Union-Find, and neural network decoders in terms of threshold and speed.

(d) What is the "optimal" decoder and why is it impractical?

### Problem 28 (Level III)
Analyze the surface code for a specific hardware platform.

(a) For superconducting qubits with 2D connectivity, what are the main noise sources?

(b) How does leakage to non-computational states affect decoding?

(c) What is the current experimental state-of-the-art for surface code demonstrations?

(d) What improvements are needed to reach fault-tolerant quantum computing?

---

## Submission Guidelines

- Draw clear diagrams for lattice problems
- Show all stabilizer calculations explicitly
- For decoder problems, trace through the algorithm step-by-step
- Resource estimates should include assumptions

---

**Problem Set Created:** February 10, 2026
**Total Problems:** 28
**Estimated Time:** 15-20 hours
