# Week 175: QLDPC Codes - Problem Set

## Instructions

This problem set contains 26 problems on quantum LDPC codes, from classical foundations to the Panteleev-Kalachev breakthrough. Problems test understanding of constructions, parameter analysis, and implications for fault tolerance.

**Difficulty Levels:**
- **(B)** Basic - Direct application
- **(I)** Intermediate - Requires synthesis
- **(A)** Advanced - Research-level

**Time Estimate:** 8-10 hours total

---

## Section 1: Classical LDPC Foundations (Problems 1-5)

### Problem 1 (B)

Consider the classical LDPC code with parity-check matrix:
$$H = \begin{pmatrix} 1 & 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 1 & 0 \\ 1 & 0 & 0 & 0 & 1 & 1 \end{pmatrix}$$

(a) What are the row weight and column weight?

(b) Draw the Tanner graph.

(c) Find the code parameters $$[n, k, d]$$.

(d) Is this a "good" code in the asymptotic sense? Why or why not?

---

### Problem 2 (I)

**Gallager's LDPC Construction**

Construct a $$(3, 6)$$-regular LDPC code of length $$n = 12$$:

(a) What are the code rate bounds?

(b) Draw a valid Tanner graph with 6 check nodes and 12 variable nodes.

(c) Verify your construction satisfies the regularity constraints.

---

### Problem 3 (I)

**Expansion and Distance**

For a $$(d_v, d_c)$$-regular bipartite graph:

(a) Define the expansion ratio $$\alpha$$ for a set $$S$$ of variable nodes.

(b) Prove that if $$\alpha > d_v/2$$ for all sets $$|S| \leq d_{\min}/2$$, then the code has minimum distance $$\geq d_{\min}$$.

(c) What is the relationship between spectral gap and expansion?

---

### Problem 4 (B)

**Rate Calculation**

For a classical LDPC code with $$(3, 6)$$-regular Tanner graph on $$n$$ variable nodes:

(a) How many check nodes are there?

(b) What is the design rate?

(c) Why might the actual rate differ from the design rate?

---

### Problem 5 (A)

**Capacity-Achieving Codes**

(a) State Shannon's channel coding theorem for the binary symmetric channel.

(b) Explain why random LDPC codes with BP decoding achieve capacity.

(c) What modifications are needed for quantum channels?

---

## Section 2: CSS Construction and Constraints (Problems 6-9)

### Problem 6 (B)

For a CSS code with check matrices $$H_X$$ and $$H_Z$$:

(a) Write the commutativity constraint.

(b) Verify that $$H_Z = H_X^T$$ satisfies this constraint.

(c) For the repetition code $$H = (1, 1, 1)$$, find the corresponding CSS code parameters.

---

### Problem 7 (I)

**Self-Orthogonal Codes**

A classical code $$C$$ is self-orthogonal if $$C \subseteq C^\perp$$.

(a) Prove that self-orthogonal codes give valid CSS codes with $$H_X = H_Z$$.

(b) What constraint does this place on the parity-check matrix?

(c) Give an example of a self-orthogonal LDPC code.

---

### Problem 8 (I)

**QLDPC Definition**

Consider a quantum code with:
- $$n$$ physical qubits
- $$m$$ X-type stabilizers and $$m$$ Z-type stabilizers
- Maximum stabilizer weight $$w$$
- Maximum qubit degree (number of stabilizers per qubit) $$\Delta$$

(a) What constraints must $$w$$ and $$\Delta$$ satisfy for QLDPC?

(b) For the surface code, what are $$w$$ and $$\Delta$$?

(c) Why is the surface code considered QLDPC despite having only $$k = O(1)$$?

---

### Problem 9 (A)

**CSS from LDPC Obstruction**

Explain why it is difficult to construct CSS codes directly from two classical LDPC codes:

(a) What is the dimension of the kernel of $$H_X H_Z^T$$?

(b) How does sparsity of $$H_X$$ and $$H_Z$$ affect this?

(c) What is the typical dimension $$k$$ for naive constructions?

---

## Section 3: Hypergraph Product Codes (Problems 10-14)

### Problem 10 (B)

**Hypergraph Product Definition**

Given classical codes $$C_1 = [4, 2, 2]$$ and $$C_2 = [4, 2, 2]$$ with parity-check matrices:
$$H_1 = H_2 = \begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}$$

Compute the hypergraph product code parameters $$[[n, k, d]]$$.

---

### Problem 11 (I)

**Parity-Check Matrix Construction**

For the hypergraph product of $$H_1 = H_2 = (1, 1)$$ (repetition code):

(a) Write out $$H_X$$ and $$H_Z$$ explicitly.

(b) Verify $$H_X H_Z^T = 0$$.

(c) Identify the resulting quantum code (it should be familiar).

---

### Problem 12 (I)

**Asymptotic Analysis**

Using classical codes with parameters $$[n, k, d]$$ where $$k = cn$$ and $$d = c'n$$:

(a) Show the hypergraph product gives quantum rate $$R = \Theta(1)$$.

(b) Show the quantum distance is $$d_Q = \Theta(\sqrt{n_Q})$$.

(c) Why does the distance not scale linearly?

---

### Problem 13 (A)

**Hypergraph Product Optimality**

(a) Prove that for any CSS code from classical codes $$C_1 \supseteq C_2^\perp$$:
$$d_X \cdot d_Z \leq n$$

(b) How does the hypergraph product saturate this bound?

(c) What would be needed to exceed $$\sqrt{n}$$ distance?

---

### Problem 14 (I)

**Comparison to Surface Code**

Compare the hypergraph product code (from $$[n, \Theta(n), \Theta(n)]$$ classical codes) to the surface code:

| Property | Hypergraph Product | Surface Code |
|----------|-------------------|--------------|
| $$k$$ | | |
| $$d$$ | | |
| Stabilizer weight | | |
| Connectivity | | |

Fill in the table and discuss trade-offs.

---

## Section 4: Lifted Product and Expanders (Problems 15-18)

### Problem 15 (B)

**Cayley Graph Definition**

For group $$G = \mathbb{Z}_4$$ with generators $$S = \{1, 3\}$$:

(a) Draw the Cayley graph.

(b) What is the degree of each vertex?

(c) Is this graph an expander? (Qualitative answer acceptable)

---

### Problem 16 (I)

**Lifting Operation**

Consider a base graph with 2 vertices and 3 edges, lifted by $$G = \mathbb{Z}_3$$:

(a) How many vertices does the lifted graph have?

(b) How many edges?

(c) Describe the structure of the lifted graph.

---

### Problem 17 (I)

**Expansion and Distance**

For an $$(n, d, \lambda)$$-expander graph:

(a) Define the spectral gap.

(b) State the expander mixing lemma.

(c) Explain intuitively why expansion implies good distance for codes built on expanders.

---

### Problem 18 (A)

**Abelian vs Non-Abelian Lifting**

(a) For Abelian group lifting, why is the distance limited to $$O(n/\log n)$$?

(b) What property of non-Abelian groups overcomes this limitation?

(c) Give an example of a non-Abelian group used in the Panteleev-Kalachev construction.

---

## Section 5: Panteleev-Kalachev Codes (Problems 19-22)

### Problem 19 (B)

**Main Result Statement**

State the main theorem of Panteleev-Kalachev (2021-2022):

(a) What are the code parameters?

(b) What makes this result significant?

(c) How long was the QLDPC conjecture open?

---

### Problem 20 (I)

**Construction Outline**

Describe the high-level steps of the Panteleev-Kalachev construction:

(a) What is the role of the Cayley graph?

(b) What are Tanner codes and how are they used?

(c) What is the lifted product operation?

---

### Problem 21 (A)

**Proof Ideas**

(a) Explain how expansion implies linear distance.

(b) Why is the rate constant?

(c) What is the stabilizer weight in terms of the local code and graph degree?

---

### Problem 22 (A)

**Alternative Constructions**

Besides Panteleev-Kalachev, two other groups achieved asymptotically good QLDPC in 2021-2022:

(a) Name the other constructions (Leverrier-Zémor and Dinur et al.).

(b) How do their approaches differ?

(c) What are the relative advantages of each construction?

---

## Section 6: Constant-Overhead Fault Tolerance (Problems 23-26)

### Problem 23 (I)

**Magic State Distillation Overhead**

(a) Define the distillation exponent $$\gamma$$.

(b) What is $$\gamma$$ for standard 15-to-1 distillation?

(c) What is $$\gamma$$ for QLDPC-based distillation?

---

### Problem 24 (I)

**How QLDPC Enables Constant Overhead**

Explain the mechanism:

(a) How does linear distance help with error suppression?

(b) How does constant rate help with overhead?

(c) Why does one round of distillation suffice?

---

### Problem 25 (A)

**Single-Shot Error Correction**

(a) Define single-shot error correction.

(b) Which QLDPC codes have this property?

(c) How does this reduce the time overhead?

---

### Problem 26 (A)

**Practical Crossover**

At what scale do QLDPC codes become advantageous over surface codes?

(a) Compare qubit counts for target error $$\epsilon = 10^{-10}$$.

(b) What are the connectivity costs?

(c) Estimate the crossover point in number of logical qubits.

---

## Bonus Problems

### Problem 27 (A)

**Geometrically Local QLDPC**

(a) State the BPT bound for 2D topological codes.

(b) What parameters are achievable in 3D?

(c) Describe the recent "almost optimal" constructions.

---

### Problem 28 (A)

**Decoding QLDPC**

(a) Why doesn't MWPM apply directly to QLDPC codes?

(b) Describe BP+OSD decoding for QLDPC.

(c) What are the threshold estimates for Panteleev-Kalachev codes?

---

### Problem 29 (A)

**Open Problem**

Choose one open problem in QLDPC codes and:

(a) State the problem precisely.

(b) Explain why it matters.

(c) Describe partial progress or approaches.

---

## Submission Guidelines

1. Show all calculations and justify claims
2. For construction problems, be explicit about matrices/graphs
3. For proof problems, structure arguments clearly
4. Reference relevant theorems when applicable
5. Discuss limitations of your answers where appropriate

**Solutions available in:** [Problem_Solutions.md](Problem_Solutions.md)
