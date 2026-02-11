# Week 171: Code Families - Problem Set

## Instructions

This problem set contains 27 problems covering quantum Reed-Muller codes, color codes, Reed-Solomon codes, concatenated codes, and fundamental bounds. Problems require both computational skills and conceptual understanding.

**Levels:**
- **Level I:** Direct application of formulas and definitions
- **Level II:** Multi-step analysis and proofs
- **Level III:** Challenging synthesis and design problems

Time estimate: 15-20 hours total

---

## Part A: Quantum Reed-Muller Codes (Problems 1-6)

### Problem 1 (Level I)
For the classical Reed-Muller code $$RM(r, m)$$:

(a) Compute the parameters $$[n, k, d]$$ for $$RM(1, 4)$$.

(b) Compute the parameters for $$RM(2, 4)$$.

(c) Verify that $$RM(1, 4)^\perp = RM(4-1-1, 4) = RM(2, 4)$$.

(d) What is the relationship between $$RM(r, m)$$ and $$RM(m-r-1, m)$$?

### Problem 2 (Level I)
Construct the quantum Reed-Muller code $$QRM(1, 4) = CSS(RM(1, 4), RM(2, 4)^\perp)$$.

(a) What is $$RM(2, 4)^\perp$$?

(b) Verify that $$RM(2, 4)^\perp \subset RM(1, 4)$$.

(c) Calculate the quantum code parameters $$[[n, k, d]]$$.

(d) This gives the famous $$[[15, ?, 3]]$$ code. What is the dimension $$k$$?

### Problem 3 (Level II)
The Clifford hierarchy is defined by $$\mathcal{C}_1 = \mathcal{G}_n$$ and $$\mathcal{C}_{k+1} = \{U : UPU^\dagger \in \mathcal{C}_k \text{ for all } P\}$$.

(a) Show that $$\mathcal{C}_2$$ is the Clifford group.

(b) Show that the T gate $$T = \text{diag}(1, e^{i\pi/4})$$ is in $$\mathcal{C}_3$$ by computing $$TXT^\dagger$$ and $$TZT^\dagger$$.

(c) The $$[[15, 1, 3]]$$ Reed-Muller code supports transversal T. What level of the hierarchy does this code access?

### Problem 4 (Level II)
The transversality of gates on Reed-Muller codes depends on the code structure.

(a) Explain why transversal gates must map stabilizers to stabilizers.

(b) For the $$[[15, 1, 3]]$$ code, verify that $$T^{\otimes 15}$$ maps the stabilizer group to itself (up to phases).

(c) Why can't the Steane code $$[[7, 1, 3]]$$ have a transversal T gate?

### Problem 5 (Level III)
Design a quantum Reed-Muller code with distance $$d \geq 7$$.

(a) What is the minimum $$m$$ needed for $$QRM(r, m)$$ to have $$d \geq 7$$?

(b) For this $$m$$, what values of $$r$$ are valid?

(c) Calculate the code parameters.

(d) What transversal gates does this code support?

### Problem 6 (Level III)
The $$[[256, 0, 16]]$$ quantum Reed-Muller code is notable.

(a) What $$RM(r, m)$$ construction gives this?

(b) Why does this code have $$k = 0$$? (Hint: it's a "quantum state" code)

(c) What is the significance of having $$k = 0$$?

---

## Part B: Color Codes (Problems 7-12)

### Problem 7 (Level I)
The smallest 2D color code is the $$[[7, 1, 3]]$$ code on a triangular lattice.

(a) Draw the lattice with 7 vertices (qubits) and faces colored Red, Green, Blue.

(b) Write the stabilizer generators (3 X-type, 3 Z-type).

(c) Verify that all generators commute.

### Problem 8 (Level I)
For the $$[[7, 1, 3]]$$ color code:

(a) Find logical operators $$\overline{X}$$ and $$\overline{Z}$$.

(b) Verify that $$\overline{X}$$ anticommutes with $$\overline{Z}$$.

(c) Find the minimum weight representatives of the logical operators.

### Problem 9 (Level II)
Color codes support transversal Clifford gates.

(a) Explain why $$H^{\otimes n}$$ is transversal for color codes but not for general CSS codes.

(b) Show that $$S^{\otimes n}$$ implements $$\overline{S}$$ on the $$[[7, 1, 3]]$$ color code.

(c) For CNOT between two $$[[7, 1, 3]]$$ color code blocks, show that $$CNOT^{\otimes 7}$$ implements $$\overline{CNOT}$$.

### Problem 10 (Level II)
The 2D color code can be defined on any 2-colorable lattice.

(a) What constraint must the lattice satisfy for a valid color code?

(b) Design a $$[[19, 1, 5]]$$ color code by extending the triangular lattice.

(c) What is the general relationship between code distance and lattice size?

### Problem 11 (Level III)
3D color codes support transversal non-Clifford gates.

(a) Describe the structure of a 3D color code on a 4-colorable lattice.

(b) The $$[[15, 1, 3]]$$ 3D color code supports transversal T. How does this compare to the Reed-Muller $$[[15, 1, 3]]$$ code?

(c) Can a 2D color code ever support a transversal T gate? Prove or disprove.

### Problem 12 (Level III)
Compare error thresholds for different code families.

(a) The 2D color code has threshold $$\sim 0.1\%$$ while the surface code has $$\sim 1\%$$. What accounts for this difference?

(b) For a physical error rate of $$p = 10^{-3}$$, which code family would you recommend?

(c) At what physical error rate does color code become preferable to surface code (considering the transversal Clifford advantage)?

---

## Part C: Quantum Reed-Solomon Codes (Problems 13-17)

### Problem 13 (Level I)
Classical Reed-Solomon codes over $$\mathbb{F}_q$$ have parameters $$[n, k, n-k+1]_q$$.

(a) What is the minimum field size $$q$$ for a $$[7, 3, 5]$$ RS code?

(b) Verify this code achieves the Singleton bound.

(c) What makes RS codes "MDS" (Maximum Distance Separable)?

### Problem 14 (Level II)
Construct a quantum Reed-Solomon code using CSS construction.

(a) For $$q = 8$$, construct $$CSS(RS(7, 5), RS(7, 3)^\perp)$$.

(b) What are the quantum code parameters?

(c) Does this code achieve the quantum Singleton bound?

### Problem 15 (Level II)
The quantum Singleton bound states $$k \leq n - 2d + 2$$.

(a) Prove this bound using the no-cloning theorem.

(b) For a quantum MDS code $$[[n, k, d]]_q$$, express $$d$$ in terms of $$n$$ and $$k$$.

(c) Do quantum MDS codes exist for all valid parameters?

### Problem 16 (Level III)
Quantum RS codes require qudits rather than qubits.

(a) Explain why binary quantum RS codes are limited.

(b) How can qudit codes be used to protect qubit information?

(c) What is the overhead of encoding qubits in qudits for error correction?

### Problem 17 (Level III)
Compare the practicality of quantum RS codes vs. qubit stabilizer codes.

(a) List advantages of quantum RS codes.

(b) List disadvantages for near-term implementation.

(c) For quantum communication (where rate matters), which family is preferable?

---

## Part D: Concatenated Codes (Problems 18-23)

### Problem 18 (Level I)
Concatenation of an outer code $$[[n_1, 1, d_1]]$$ with inner code $$[[n_2, 1, d_2]]$$ gives:

(a) What are the parameters of the concatenated code?

(b) Calculate the parameters for Steane $$[[7, 1, 3]]$$ concatenated with itself.

(c) Calculate for three levels of Steane concatenation.

### Problem 19 (Level I)
The Shor code $$[[9, 1, 3]]$$ can be viewed as concatenated.

(a) Identify the inner and outer codes.

(b) Verify the distance formula $$d = d_1 \cdot d_2$$.

(c) Why is the Shor code considered "doubly concatenated"?

### Problem 20 (Level II)
The threshold theorem states that for $$p < p_{\text{th}}$$, arbitrarily reliable computation is possible.

(a) For a distance-3 code, failures occur when $$\geq 2$$ locations fail. Express the failure probability in terms of $$p$$.

(b) Derive the threshold condition $$p_{\text{th}} = 1/C$$ where $$C$$ is code-dependent.

(c) Estimate $$C$$ for the Steane code (consider all pairs of qubit errors in a syndrome extraction circuit).

### Problem 21 (Level II)
Calculate the overhead for concatenated codes.

(a) For $$L$$ levels of concatenation with base code $$[[n, 1, d]]$$, how many physical qubits are needed?

(b) To achieve logical error rate $$\epsilon$$ with physical rate $$p$$, how many levels are needed?

(c) What is the total physical qubit count as a function of $$\epsilon$$?

### Problem 22 (Level III)
Compare concatenated codes to topological codes.

(a) Concatenated Steane has threshold $$\sim 10^{-4}$$, surface code has $$\sim 10^{-2}$$. Plot the logical error rate vs. physical error rate for both.

(b) At what physical error rate do they achieve equal logical error rate?

(c) Below this crossover, which is more efficient in physical qubits?

### Problem 23 (Level III)
Design an optimal concatenation scheme for a target application.

(a) You need logical error rate $$10^{-15}$$ with physical error rate $$10^{-4}$$. Design a concatenated code scheme.

(b) Calculate total physical qubit count.

(c) Compare to a surface code achieving the same logical error rate.

---

## Part E: Bounds and Comparisons (Problems 24-27)

### Problem 24 (Level I)
Apply the quantum Hamming bound to various codes.

(a) Verify the $$[[5, 1, 3]]$$ code saturates the bound.

(b) Does the Steane $$[[7, 1, 3]]$$ code saturate the bound?

(c) Does the Shor $$[[9, 1, 3]]$$ code saturate the bound?

### Problem 25 (Level II)
The quantum Singleton bound is $$k \leq n - 2d + 2$$.

(a) Verify this bound for codes $$[[5, 1, 3]]$$, $$[[7, 1, 3]]$$, $$[[9, 1, 3]]$$.

(b) Which of these codes are "close" to optimal by Singleton?

(c) Can any code with $$d = 3$$ on fewer than 5 qubits exist?

### Problem 26 (Level III)
Asymptotic bounds describe the behavior of codes as $$n \to \infty$$.

(a) State the quantum Gilbert-Varshamov bound.

(b) For a family of codes with $$d/n \to \delta$$ as $$n \to \infty$$, what is the maximum asymptotic rate $$k/n$$?

(c) Do any known code families achieve the Gilbert-Varshamov bound?

### Problem 27 (Level III)
Design a code selection algorithm.

Given:
- Physical error rate $$p$$
- Target logical error rate $$\epsilon$$
- Available connectivity (2D grid, all-to-all, etc.)
- Gate set requirements

(a) Outline an algorithm to select the optimal code family.

(b) Apply your algorithm for $$p = 10^{-3}$$, $$\epsilon = 10^{-10}$$, 2D connectivity, need T gates.

(c) Apply for $$p = 10^{-5}$$, $$\epsilon = 10^{-15}$$, all-to-all connectivity, Clifford gates only.

---

## Submission Guidelines

- For comparison problems, use tables and justify rankings
- Show all parameter calculations explicitly
- For design problems, verify all constraints are satisfied
- Include diagrams for color code problems

---

**Problem Set Created:** February 10, 2026
**Total Problems:** 27
**Estimated Time:** 15-20 hours
