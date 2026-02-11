# Week 167: Quantum Information Theory

## Overview

**Days:** 1163-1169
**Theme:** Entropy, Channel Capacity, and Information-Theoretic Limits
**Hours:** 45 hours (7.5 hours/day × 6 days)

---

## Learning Objectives

By the end of this week, you should be able to:

1. Define and calculate von Neumann entropy and its properties
2. Prove key entropy inequalities (subadditivity, strong subadditivity)
3. State and apply the Holevo bound
4. Calculate classical and quantum channel capacities
5. Explain quantum data compression (Schumacher compression)
6. Apply quantum Shannon theory to communication problems
7. Connect information-theoretic quantities to physical systems

---

## Daily Schedule

### Day 1163 (Monday): Von Neumann Entropy

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Entropy calculations |
| 2:00-5:00 | 3 hrs | Review: Properties and proofs |
| 7:00-8:30 | 1.5 hrs | Computational: Entropy simulations |

**Topics:**
- Definition: $S(\rho) = -\text{Tr}(\rho \log \rho)$
- Properties: non-negativity, concavity, maximum value
- Entropy of pure states (S = 0) and mixed states
- Connection to eigenvalue spectrum

### Day 1164 (Tuesday): Conditional Entropy and Mutual Information

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Composite systems |
| 2:00-5:00 | 3 hrs | Review: Quantum conditional entropy |
| 7:00-8:30 | 1.5 hrs | Oral practice: Entropy concepts |

**Topics:**
- Quantum conditional entropy: $S(A|B) = S(AB) - S(B)$
- Can be negative! (entanglement signature)
- Quantum mutual information: $I(A:B) = S(A) + S(B) - S(AB)$
- Subadditivity and strong subadditivity

### Day 1165 (Wednesday): Holevo Bound

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Holevo bound applications |
| 2:00-5:00 | 3 hrs | Review: Proof and implications |
| 7:00-8:30 | 1.5 hrs | Written practice: Bound calculations |

**Topics:**
- Holevo quantity: $\chi(\{p_i, \rho_i\}) = S(\sum_i p_i\rho_i) - \sum_i p_i S(\rho_i)$
- Holevo bound: accessible information ≤ Holevo quantity
- Why you can't extract more than log(d) bits from a qudit
- Achievability with HSW theorem

### Day 1166 (Thursday): Quantum Channel Capacity

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Channel capacity calculations |
| 2:00-5:00 | 3 hrs | Review: Classical vs quantum capacity |
| 7:00-8:30 | 1.5 hrs | Computational: Channel simulations |

**Topics:**
- Classical capacity of quantum channel (HSW theorem)
- Quantum capacity (coherent information)
- Private capacity
- Entanglement-assisted capacity

### Day 1167 (Friday): Data Compression

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Compression schemes |
| 2:00-5:00 | 3 hrs | Review: Schumacher's theorem |
| 7:00-8:30 | 1.5 hrs | Lab: Compression implementation |

**Topics:**
- Classical data compression (Shannon)
- Quantum data compression (Schumacher)
- Typical subspace and projections
- Compression rate = von Neumann entropy

### Day 1168 (Saturday): Advanced Topics

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Advanced problems |
| 2:00-5:00 | 3 hrs | Review: Entanglement distillation, LOCC |
| 7:00-8:30 | 1.5 hrs | Written practice: Comprehensive problems |

**Topics:**
- Entanglement entropy and area laws
- Quantum relative entropy
- Entanglement distillation rates
- State merging and decoupling

### Day 1169 (Sunday): Integration and Assessment

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Comprehensive problem set completion |
| 2:00-5:00 | 3 hrs | Oral exam practice session |
| 7:00-8:30 | 1.5 hrs | Self-assessment and gap analysis |

---

## Key Equations

### Von Neumann Entropy
$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where $\{\lambda_i\}$ are eigenvalues of $\rho$.

### Entropy Properties
- **Non-negativity:** $S(\rho) \geq 0$
- **Maximum:** $S(\rho) \leq \log d$ for $d$-dimensional system
- **Concavity:** $S(\sum_i p_i \rho_i) \geq \sum_i p_i S(\rho_i)$
- **Pure state:** $S(|\psi\rangle\langle\psi|) = 0$

### Subadditivity
$$S(AB) \leq S(A) + S(B)$$

Equality iff $\rho_{AB} = \rho_A \otimes \rho_B$

### Strong Subadditivity
$$S(ABC) + S(B) \leq S(AB) + S(BC)$$

Equivalently: $I(A:C|B) \geq 0$

### Holevo Bound
$$I(X:B) \leq \chi(\{p_x, \rho_x\}) = S(\rho) - \sum_x p_x S(\rho_x)$$

where $\rho = \sum_x p_x \rho_x$

### Channel Capacities

**Classical capacity:**
$$C(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \chi^*(\mathcal{N}^{\otimes n})$$

**Quantum capacity:**
$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} I_c^*(\mathcal{N}^{\otimes n})$$

where $I_c = S(\mathcal{N}(\rho)) - S((\mathcal{N} \otimes I)(\psi_{AR}))$ is coherent information.

### Schumacher Compression
For source emitting states $\{p_i, |\psi_i\rangle\}$ with density matrix $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$:

$$\text{Compression rate} = S(\rho) \text{ qubits per source symbol}$$

---

## Key Theorems

### Theorem 1: Holevo Bound
For an ensemble $\{p_x, \rho_x\}$ encoded in quantum states and any measurement:
$$I(X:Y) \leq S(\rho) - \sum_x p_x S(\rho_x)$$

### Theorem 2: HSW Theorem
The classical capacity of a quantum channel equals the regularized Holevo capacity.

### Theorem 3: Schumacher Compression
Quantum sources can be compressed to $S(\rho)$ qubits per symbol asymptotically.

### Theorem 4: Strong Subadditivity (Lieb-Ruskai)
For any tripartite state $\rho_{ABC}$:
$$S(A|BC) \leq S(A|B)$$

---

## Connections

### From Previous Weeks
| Previous Topic | Connection to This Week |
|----------------|------------------------|
| Density matrices (Week 157) | Entropy defined on density matrices |
| Quantum channels (Week 160) | Channel capacity uses channel formalism |
| Entanglement (Week 159) | Entanglement entropy measures entanglement |

### To Integration Week
| This Week's Topic | Integration Application |
|-------------------|------------------------|
| Holevo bound | Limits on communication protocols |
| Channel capacity | Fundamental limits for quantum computing |
| Entropy | Universal measure in QI/QC |

---

## Resources

### Primary Reading
- Nielsen & Chuang, Chapters 11-12
- Wilde, *Quantum Information Theory*, Chapters 10-15
- Preskill, Ph219 Lecture Notes, Chapter 5

### Research Papers
- Holevo, "The Capacity of a Quantum Communications Channel" (1973)
- Schumacher, "Quantum Coding" (1995)
- Lieb & Ruskai, "Proof of Strong Subadditivity" (1973)

### Online Resources
- [Preskill Notes Chapter 5](http://theory.caltech.edu/~preskill/ph219/chap5_15.pdf)
- [Wilde Textbook (arXiv)](https://arxiv.org/abs/1106.1445)

---

**Created:** February 9, 2026
**Status:** Not Started
