# Day 636: Graph Problems

## Overview
**Day 636** | Week 91, Day 6 | Year 1, Month 23 | Quantum Walks

Today we explore how quantum walks provide speedups for graph problems like element distinctness, triangle finding, and graph connectivity.

---

## Learning Objectives

1. Apply quantum walks to element distinctness
2. Analyze triangle finding algorithm
3. Understand the walk-based framework
4. Compare query complexities
5. Study graph connectivity
6. Appreciate the power of quantum walk algorithms

---

## Core Content

### Element Distinctness

**Problem:** Given $x_1, ..., x_n$, are any two equal?

**Classical:** $O(n)$ queries (or $O(n \log n)$ for sorting)

**Quantum (Ambainis 2004):** $O(n^{2/3})$ queries using quantum walk!

### The Algorithm

1. Maintain a subset $S$ of size $r = n^{2/3}$
2. Walk on Johnson graph $J(n, r)$
3. Mark vertices where subset contains collision
4. Quantum walk search on this graph

**Analysis:** Hitting time $O(n^{2/3})$, giving total $O(n^{2/3})$ complexity.

### Triangle Finding

**Problem:** Does graph $G$ have a triangle (3-clique)?

**Classical:** $O(n^2)$ or $O(n^\omega)$ via matrix multiplication

**Quantum (Magniez et al. 2007):** $O(n^{5/4})$

### Algorithm Sketch

1. For each vertex $v$, check its neighborhood
2. Element distinctness on edges determines if triangle
3. Combine with amplitude amplification

### Graph Connectivity

**Problem:** Is graph $G$ connected?

**Classical:** $O(n + m)$ via BFS/DFS

**Quantum:** Also $O(n + m)$ — no speedup for this!

Some graph problems don't benefit from quantum speedups.

### Summary of Results

| Problem | Classical | Quantum | Speedup |
|---------|-----------|---------|---------|
| Element Distinctness | $O(n)$ | $O(n^{2/3})$ | $n^{1/3}$ |
| Triangle Finding | $O(n^2)$ | $O(n^{5/4})$ | $n^{3/4}$ |
| k-Clique | $O(n^k)$ | $O(n^{k/2 + O(1)})$ | $\sim n^{k/2}$ |
| Matrix Product Verify | $O(n^2)$ | $O(n^{5/3})$ | $n^{1/3}$ |

---

## Computational Lab

```python
"""Day 636: Graph Problems"""
import numpy as np
import matplotlib.pyplot as plt

def complexity_comparison():
    """Compare classical vs quantum for graph problems."""
    n = np.logspace(1, 6, 100)

    problems = {
        'Element Distinctness': (n, n**(2/3)),
        'Triangle Finding': (n**2, n**(5/4)),
        'k-Clique (k=4)': (n**4, n**2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, (classical, quantum)) in enumerate(problems.items()):
        axes[idx].loglog(n, classical, 'b-', label='Classical', linewidth=2)
        axes[idx].loglog(n, quantum, 'r-', label='Quantum', linewidth=2)
        axes[idx].set_xlabel('n')
        axes[idx].set_ylabel('Query Complexity')
        axes[idx].set_title(name)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graph_problems.png', dpi=150)
    plt.show()

complexity_comparison()

# Speedup factors
print("\nSpeedup Factors:")
for n in [100, 1000, 10000, 100000]:
    ed_speedup = n / n**(2/3)
    tri_speedup = n**2 / n**(5/4)
    print(f"n = {n:>6}: Element Dist. = {ed_speedup:.1f}x, Triangle = {tri_speedup:.1f}x")
```

---

## Summary

### Key Results

| Algorithm | Complexity | Based On |
|-----------|------------|----------|
| Element Distinctness | $O(n^{2/3})$ | Quantum walk on Johnson graph |
| Triangle Finding | $O(n^{5/4})$ | Walk + collision finding |
| Graph Connectivity | No speedup | Structure prevents speedup |

---

## Daily Checklist

- [ ] I understand element distinctness algorithm
- [ ] I know the triangle finding speedup
- [ ] I can identify which problems get speedups
- [ ] I understand the walk-based framework

---

*Next: Day 637 — Week Review*
