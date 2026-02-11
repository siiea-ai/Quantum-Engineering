# Week 175: Quantum Low-Density Parity-Check Codes

## Overview

**Days:** 1219-1225
**Theme:** QLDPC Codes and Constant-Overhead Fault Tolerance

This week explores quantum low-density parity-check (QLDPC) codes, culminating in the groundbreaking Panteleev-Kalachev construction that achieves asymptotically good parameters. These codes represent a paradigm shift in quantum error correction, offering the potential for constant-overhead fault-tolerant quantum computation.

## Learning Objectives

By the end of this week, students will be able to:

1. Construct QLDPC codes from classical LDPC components
2. Analyze the hypergraph product construction and its limitations
3. Understand lifted product codes and expander graphs
4. Prove asymptotic goodness of Panteleev-Kalachev codes
5. Explain constant-overhead fault tolerance implications
6. Evaluate practical challenges for QLDPC implementation

## Daily Schedule

### Day 1219 (Monday): Classical LDPC Review

**Topics:**
- Sparse parity-check matrices
- Tanner graph representation
- Belief propagation decoding
- Asymptotically good classical LDPC

**Key Concepts:**
- **LDPC code:** Parity-check matrix $$H$$ with $$O(1)$$ ones per row and column
- **Rate:** $$R = k/n = 1 - \text{rank}(H)/n$$
- **Distance:** For good codes, $$d = \Theta(n)$$
- **Capacity-achieving:** Random LDPC codes approach Shannon limit

**Classical LDPC parameters:**
$$\boxed{[n, k = \Theta(n), d = \Theta(n)] \text{ with } O(1) \text{ weight checks}}$$

---

### Day 1220 (Tuesday): Quantum LDPC Construction

**Topics:**
- CSS codes from classical LDPC
- Commutativity constraints ($$H_X H_Z^T = 0$$)
- Quantum LDPC definition
- Early constructions and limitations

**Key Challenge:**
For CSS codes: $$H_X H_Z^T = 0 \mod 2$$

This constraint makes direct LDPC construction difficult.

**QLDPC Definition:**
$$\boxed{\text{QLDPC: Stabilizer weight } w = O(1), \text{ qubit degree } = O(1)}$$

---

### Day 1221 (Wednesday): Hypergraph Product Codes

**Topics:**
- Tillich-Zémor construction
- Product of two classical codes
- Parameter analysis
- Distance limitations

**Construction:**
Given classical codes $$C_1 = [n_1, k_1, d_1]$$ and $$C_2 = [n_2, k_2, d_2]$$:

$$[[n, k, d]]$$ where:
- $$n = n_1 n_2 + m_1 m_2$$ ($$m_i = n_i - k_i$$)
- $$k = k_1 k_2$$
- $$d = \min(d_1, d_2)$$

**Limitation:**
Using good classical codes: $$d = \Theta(\sqrt{n})$$, not linear.

---

### Day 1222 (Thursday): Lifted Product Codes

**Topics:**
- Group lifting operation
- Cayley graphs and group structure
- Expander graphs for quantum codes
- Improved distance scaling

**Key Insight:**
Lifting by non-Abelian groups can improve distance.

**Construction Steps:**
1. Start with Cayley graph of finite group $$G$$
2. Take double cover for CSS compatibility
3. Create Tanner codes on expander graphs
4. Apply lifted product construction

**Parameters (Abelian lifting):**
$$[[n, k = \Theta(n), d = \Theta(n/\log n)]]$$

---

### Day 1223 (Friday): Panteleev-Kalachev Codes

**Topics:**
- Asymptotically good QLDPC
- Non-Abelian group construction
- Proof of linear distance
- Historical significance

**Main Result (2021-2022):**
$$\boxed{[[n, k = \Theta(n), d = \Theta(n)]] \text{ with } O(1) \text{ stabilizer weight}}$$

**Key Innovation:**
Use non-Abelian groups (specifically, certain semidirect products) for the lifting.

**Proof Outline:**
1. Construct expander Cayley graph
2. Build Tanner codes with small local codes
3. Lifted product inherits expansion properties
4. Expansion implies linear distance

---

### Day 1224 (Saturday): Constant-Overhead Fault Tolerance

**Topics:**
- Single-shot error correction
- Constant-overhead magic state distillation
- Fault-tolerant gates on QLDPC
- Resource estimates

**Breakthrough Result:**
$$\boxed{\gamma = 0: \text{ Constant overhead for magic state distillation}}$$

**Implications:**
1. Total overhead for fault-tolerant computation is $$O(1)$$ per logical gate
2. No polylogarithmic blowup from distillation
3. Asymptotically optimal resource scaling

**Challenges:**
- Gate implementation (no transversal gates beyond Paulis for most QLDPC)
- Non-local connectivity requirements
- Decoding complexity

---

### Day 1225 (Sunday): QLDPC Frontiers

**Topics:**
- Geometrically local QLDPC
- Implementation challenges
- Decoders for QLDPC
- Current research directions

**Open Problems:**
1. Geometrically local QLDPC with good parameters in 2D/3D
2. Efficient decoders matching MWPM performance
3. Practical magic state factories with QLDPC
4. Threshold estimates for realistic noise

**Recent Progress (2024-2025):**
- Almost optimal geometrically local codes in any dimension
- BP+OSD decoding achieving reasonable thresholds
- First experimental demonstrations on small QLDPC codes

## Key Theorems and Results

### Hypergraph Product (Tillich-Zémor, 2009)

For classical $$[n, k, d]$$ codes:
$$\boxed{[[n^2 + m^2, k^2, d]] \text{ where } m = n-k}$$

### Panteleev-Kalachev (2021-2022)

First asymptotically good QLDPC:
$$\boxed{[[n, \Theta(n), \Theta(n)]] \text{ with } O(1) \text{ check weight}}$$

### Constant-Overhead FT (2024-2025)

Magic state distillation with:
$$\boxed{N_{\text{magic}} = O(1) \text{ per logical T gate}}$$

## Week 175 Files

| File | Description |
|------|-------------|
| [Review_Guide.md](Review_Guide.md) | Comprehensive review of QLDPC codes |
| [Problem_Set.md](Problem_Set.md) | 25-30 practice problems |
| [Problem_Solutions.md](Problem_Solutions.md) | Detailed solutions |
| [Oral_Practice.md](Oral_Practice.md) | Oral exam preparation questions |
| [Self_Assessment.md](Self_Assessment.md) | Self-evaluation checklist |

## Prerequisites Review

Before starting this week, ensure mastery of:
- CSS code construction
- Classical coding theory (rate, distance, parity-check matrix)
- Graph theory (Cayley graphs, expanders)
- Group theory basics (Abelian vs non-Abelian)

## Mathematical Background

### Expander Graphs

A graph $$G = (V, E)$$ is a $$\lambda$$-expander if:
$$\lambda_2(A_G) \leq \lambda < 1$$

where $$\lambda_2$$ is the second largest eigenvalue of the normalized adjacency matrix.

**Key Property:** Expansion implies:
1. High connectivity
2. Rapid mixing
3. Error correction capability

### Tanner Codes

Given graph $$G$$ and local code $$C_0$$:
- Bits on edges of $$G$$
- Check at each vertex: bits on incident edges form codeword of $$C_0$$
- Good expansion $$\implies$$ good distance

## Navigation

- **Previous:** [Week 174: Decoding](../Week_174_Decoding/README.md)
- **Next:** [Week 176: QEC Integration Exam](../Week_176_QEC_Integration_Exam/README.md)
- **Month:** [Month 44: QEC Mastery II](../README.md)
