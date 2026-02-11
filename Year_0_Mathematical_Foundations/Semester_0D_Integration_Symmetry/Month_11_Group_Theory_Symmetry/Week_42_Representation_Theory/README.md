# Week 42: Representation Theory (Days 288-294)

## Overview

Week 42 introduces **representation theory**, the study of how groups act on vector spaces. This is where abstract group theory meets linear algebra and becomes directly applicable to quantum mechanics. In QM, quantum states form vector spaces (Hilbert spaces), and symmetry groups act on these spaces via representations. Understanding representations is essential for classifying particles, understanding selection rules, and working with angular momentum.

## Learning Objectives

By the end of Week 42, you will be able to:

1. Define group representations and recognize matrix representations
2. Identify reducible and irreducible representations
3. State and apply Schur's lemma
4. Compute and use character tables
5. Decompose representations using orthogonality relations
6. Apply representation theory to quantum mechanical systems

## Daily Schedule

| Day | Topic | Key Concepts | QM Connection |
|-----|-------|--------------|---------------|
| 288 | Group Representations | Definition, matrix representations, examples | States under symmetry |
| 289 | Reducibility | Invariant subspaces, complete reducibility | Multiplets, degeneracy |
| 290 | Schur's Lemma | Statement, proof, consequences | Selection rules |
| 291 | Characters | Definition, properties, character tables | State counting |
| 292 | Orthogonality | Great Orthogonality Theorem | Decomposition |
| 293 | Representations of S_n | Young tableaux, partitions | Identical particles |
| 294 | Week Review | Synthesis, physical applications | Full integration |

## Key Concepts

### What is a Representation?

A **representation** of group $G$ on vector space $V$ is a homomorphism:
$$\rho: G \to GL(V)$$

This means:
- Each group element $g$ acts as an invertible linear transformation $\rho(g)$
- The group operation is preserved: $\rho(gh) = \rho(g)\rho(h)$

### Matrix Representations

Choosing a basis for $V$ gives **matrix representations**:
$$D: G \to GL_n(\mathbb{C}), \quad g \mapsto D(g)$$

where $D(g)$ is an $n \times n$ matrix.

### Irreducible Representations

A representation is **irreducible** if $V$ has no proper invariant subspaces.

Irreducible representations (irreps) are the "building blocks"—every representation decomposes into irreps.

### Characters

The **character** of a representation is the trace:
$$\chi(g) = \text{Tr}(D(g))$$

Characters are:
- Constant on conjugacy classes
- Determine representations up to equivalence
- Satisfy orthogonality relations

## Quantum Mechanics Connections

| Rep Theory Concept | QM Application |
|-------------------|----------------|
| Representation | How states transform |
| Irreducible rep | Multiplet of states |
| Dimension of irrep | Degeneracy |
| Character | Observable expectation |
| Decomposition | Finding quantum numbers |
| Schur's lemma | Selection rules |

## Key Formulas

### Group Representation Axiom
$$D(g_1 g_2) = D(g_1) D(g_2)$$

### Character Orthogonality (Rows)
$$\sum_{g \in G} \chi^{(\alpha)}(g)^* \chi^{(\beta)}(g) = |G| \delta_{\alpha\beta}$$

### Character Orthogonality (Columns)
$$\sum_{\alpha} \chi^{(\alpha)}(g)^* \chi^{(\alpha)}(h) = \frac{|G|}{|C_g|} \delta_{C_g, C_h}$$

### Decomposition Formula
$$n_\alpha = \frac{1}{|G|} \sum_{g \in G} \chi^{(\alpha)}(g)^* \chi(g)$$

## Python Libraries

```python
import numpy as np
from numpy.linalg import eig, norm
from scipy.linalg import block_diag
import sympy
from sympy.combinatorics import Permutation
```

## Resources

### Primary Texts
- Tinkham, *Group Theory and Quantum Mechanics*, Chapters 3-5
- Serre, *Linear Representations of Finite Groups*
- Fulton & Harris, *Representation Theory*

### Physics Applications
- Georgi, *Lie Algebras in Particle Physics*
- Cornwell, *Group Theory in Physics*

### Online
- MIT OCW 18.712 (Introduction to Representation Theory)
- Mathematical Physics notes on representation theory

## Assessment Goals

By Day 294, you should be able to:
- [ ] Construct matrix representations for small groups
- [ ] Determine if a representation is reducible
- [ ] Build and interpret character tables
- [ ] Use orthogonality to decompose representations
- [ ] Apply representation theory to quantum systems

## Week Progress

| Day | File | Status |
|-----|------|--------|
| 288 | Day_288_Monday.md | 🔄 In Progress |
| 289 | Day_289_Tuesday.md | ⬜ Not Started |
| 290 | Day_290_Wednesday.md | ⬜ Not Started |
| 291 | Day_291_Thursday.md | ⬜ Not Started |
| 292 | Day_292_Friday.md | ⬜ Not Started |
| 293 | Day_293_Saturday.md | ⬜ Not Started |
| 294 | Day_294_Sunday.md | ⬜ Not Started |

**Week Progress:** 0/7 days (0%)
