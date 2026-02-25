# Day 412: Wigner 3j Symbols

## Overview
**Day 412** | Year 1, Month 15, Week 59 | Symmetric Coupling Coefficients

Wigner 3j symbols are a symmetric form of Clebsch-Gordan coefficients, with elegant symmetry properties under permutation and sign changes.

---

## Core Content

### Definition

$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = \frac{(-1)^{j_1-j_2-m_3}}{\sqrt{2j_3+1}} \langle j_1, m_1; j_2, m_2 | j_3, -m_3\rangle$$

### Relation to CG Coefficients

$$\langle j_1, m_1; j_2, m_2 | j, m\rangle = (-1)^{j_1-j_2+m}\sqrt{2j+1}\begin{pmatrix} j_1 & j_2 & j \\ m_1 & m_2 & -m \end{pmatrix}$$

### Selection Rules

The 3j symbol vanishes unless:
1. m₁ + m₂ + m₃ = 0
2. Triangle condition: j₁, j₂, j₃ satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂
3. j₁ + j₂ + j₃ is an integer

### Symmetry Properties

**Even permutations** (unchanged):
$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = \begin{pmatrix} j_2 & j_3 & j_1 \\ m_2 & m_3 & m_1 \end{pmatrix} = \begin{pmatrix} j_3 & j_1 & j_2 \\ m_3 & m_1 & m_2 \end{pmatrix}$$

**Odd permutations** (factor of (-1)^{j₁+j₂+j₃}):
$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = (-1)^{j_1+j_2+j_3}\begin{pmatrix} j_2 & j_1 & j_3 \\ m_2 & m_1 & m_3 \end{pmatrix}$$

**Sign flip:**
$$\begin{pmatrix} j_1 & j_2 & j_3 \\ -m_1 & -m_2 & -m_3 \end{pmatrix} = (-1)^{j_1+j_2+j_3}\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}$$

### Orthogonality

$$\sum_{m_1,m_2}(2j_3+1)\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}\begin{pmatrix} j_1 & j_2 & j'_3 \\ m_1 & m_2 & m'_3 \end{pmatrix} = \delta_{j_3j'_3}\delta_{m_3m'_3}$$

---

## Practice Problems
1. Calculate the 3j symbol for j₁=j₂=j₃=1/2, m₁=m₂=1/2, m₃=-1.
2. Verify the symmetry under column exchange.
3. Show that the 3j symbol vanishes when j₁+j₂+j₃ is odd and all m=0.

---

**Next:** [Day_413_Sunday.md](Day_413_Sunday.md) — Week 59 Review
