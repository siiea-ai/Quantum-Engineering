# Day 633: Coin and Shift Operators

## Overview
**Day 633** | Week 91, Day 3 | Year 1, Month 23 | Quantum Walks

Today we examine the coin and shift operators in detail, exploring different coin choices and their effects on walk dynamics.

---

## Learning Objectives

1. Analyze different coin operators
2. Understand the role of coin in creating interference
3. Construct shift operators for various graphs
4. Study coined walks on higher-dimensional lattices
5. Explore the Grover coin for search applications
6. Connect coin choice to algorithmic performance

---

## Core Content

### Coin Operators

**Hadamard Coin:**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Creates equal superposition but with phase asymmetry.

**Grover Coin (for d directions):**
$$G = 2|s\rangle\langle s| - I, \quad |s\rangle = \frac{1}{\sqrt{d}}\sum_{j=1}^{d}|j\rangle$$

For d=2: $G = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ (Pauli X)

**Y-rotation Coin:**
$$C(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Parameterized family allowing bias control.

### Shift Operators

**Standard Shift (Line):**
$$S = |L\rangle\langle L| \otimes T^- + |R\rangle\langle R| \otimes T^+$$

where $T^{\pm}|x\rangle = |x \pm 1\rangle$.

**Flip-Flop Shift:**
$$S_{FF}|c\rangle|x\rangle = |\bar{c}\rangle|x + (-1)^c\rangle$$

Flips coin after moving.

**Graph Shift (General):**
For vertex $v$ with neighbors $\{u_1, ..., u_d\}$:
$$S|j\rangle|v\rangle = |j'\rangle|u_j\rangle$$

where $j'$ is determined by how $v$ appears in $u_j$'s neighbor list.

### Grover Coin for Search

For a d-regular graph, the Grover coin at each vertex:
$$G_v = 2|s_v\rangle\langle s_v| - I$$

where $|s_v\rangle = \frac{1}{\sqrt{d}}\sum_j|j\rangle$ is uniform over directions.

This coin choice is optimal for quantum walk search!

### Multi-Dimensional Walks

**2D Lattice:**
- Coin space: 4-dimensional (up, down, left, right)
- Shift in each direction based on coin state
- Grover coin gives optimal spreading

**d-Dimensional Hypercube:**
- Coin space: d-dimensional
- Each coin state corresponds to dimension to step in

---

## Worked Examples

### Example 1: Grover vs Hadamard Coin
Compare spreading rates for different coins.

**Solution:**
Hadamard (d=2): Asymmetric walk, peaks at $\pm t/\sqrt{2}$

Grover (d=2): $G = X$, deterministic oscillation!

For 2D with 4-direction coin:
- Hadamard tensor: $H \otimes H$, fast but asymmetric
- 4D Grover: optimal symmetric spreading

### Example 2: Flip-Flop Walk
Show the flip-flop shift creates same walk as standard shift.

**Solution:**
They're unitarily equivalent via coin relabeling at each position.

---

## Practice Problems

### Problem 1: Custom Coin
Design a coin that creates equal left-right probabilities from $|R\rangle$.

### Problem 2: 2D Walk Operator
Write the walk operator for a 2D grid with Grover coin.

### Problem 3: Graph Walk
Construct the walk operator for a complete graph $K_4$.

---

## Computational Lab

```python
"""Day 633: Coin and Shift Operators"""
import numpy as np
import matplotlib.pyplot as plt

def hadamard_coin():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def grover_coin_2d():
    return np.array([[0, 1], [1, 0]])  # Same as Pauli X for d=2

def grover_coin(d):
    """d-dimensional Grover coin."""
    s = np.ones(d) / np.sqrt(d)
    return 2 * np.outer(s, s) - np.eye(d)

def y_rotation_coin(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def compare_coins():
    """Compare different coin operators."""
    coins = [
        (hadamard_coin(), "Hadamard"),
        (grover_coin_2d(), "Grover (d=2)"),
        (y_rotation_coin(np.pi/4), "Y(π/4)"),
        (y_rotation_coin(np.pi/3), "Y(π/3)"),
    ]

    for C, name in coins:
        eigvals = np.linalg.eigvals(C)
        print(f"{name}:")
        print(f"  Matrix:\n{np.round(C, 3)}")
        print(f"  Eigenvalues: {np.round(eigvals, 3)}")
        print()

# Run comparison
print("="*50)
print("Coin Operator Comparison")
print("="*50)
compare_coins()

# Grover coin for higher dimensions
print("\n4D Grover Coin:")
G4 = grover_coin(4)
print(np.round(G4, 3))
```

---

## Summary

### Key Formulas

| Coin | Matrix | Property |
|------|--------|----------|
| Hadamard | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | Asymmetric |
| Grover (d=2) | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Symmetric |
| Grover (d) | $2\|s\rangle\langle s\| - I$ | Optimal for search |

---

## Daily Checklist

- [ ] I understand different coin choices
- [ ] I can construct Grover coins
- [ ] I know how coin affects walk dynamics
- [ ] I can build shift operators for graphs
- [ ] I understand multi-dimensional walks

---

*Next: Day 634 — Continuous-Time Quantum Walks*
