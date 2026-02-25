# Day 538: GHZ and W States

## Overview
**Day 538** | Week 77, Day 6 | Year 1, Month 20 | Multipartite Entanglement

Today we study the two fundamental classes of three-qubit entanglement: GHZ (Greenberger-Horne-Zeilinger) and W states, which cannot be converted into each other by LOCC.

---

## Learning Objectives
1. Define GHZ and W states mathematically
2. Understand SLOCC classification of multipartite entanglement
3. Analyze entanglement properties of each class
4. Compute reduced density matrices
5. Distinguish GHZ vs W via operational properties
6. Extend to N-qubit generalizations

---

## Core Content

### GHZ State

$$\boxed{|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)}$$

**Properties:**
- Maximally entangled in a specific sense
- Highly non-classical (maximum Bell violation)
- **Fragile:** Tracing out any qubit gives a separable state!

### W State

$$\boxed{|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)}$$

**Properties:**
- Symmetric under particle exchange
- **Robust:** Tracing out any qubit leaves entanglement!
- Single excitation superposition

### SLOCC Classification

**SLOCC** (Stochastic LOCC): Local operations + classical communication with some success probability.

**Theorem (Dür, Vidal, Cirac 2000):** For three qubits, there are exactly six SLOCC classes:

1. **Separable:** $|000\rangle$
2. **Biseparable A-BC:** $|0\rangle|ψ\rangle_{BC}$
3. **Biseparable B-AC:** $|ψ\rangle_{AC}|0\rangle$
4. **Biseparable C-AB:** $|ψ\rangle_{AB}|0\rangle$
5. **W class:** $|W\rangle$ and SLOCC equivalents
6. **GHZ class:** $|GHZ\rangle$ and SLOCC equivalents

**Key result:** GHZ and W are in different SLOCC classes—cannot convert one to other with any probability!

### Reduced Density Matrices

**GHZ reduced states:**
$$\rho_{AB}^{GHZ} = \text{Tr}_C(|GHZ\rangle\langle GHZ|) = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$

This is **separable** (classical mixture)! No bipartite entanglement.

**W reduced states:**
$$\rho_{AB}^{W} = \text{Tr}_C(|W\rangle\langle W|) = \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10| + |01\rangle\langle 10| + |10\rangle\langle 01|)$$

This is **entangled**! The state $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ with some mixing.

### Comparison

| Property | GHZ | W |
|----------|-----|---|
| Bipartite reduced | Separable | Entangled |
| Loss of one qubit | All entanglement lost | Partial entanglement remains |
| Bell violation | Maximal (Mermin) | Submaximal |
| Quantum secret sharing | Yes (perfect) | No |
| Robustness | Fragile | Robust |

### N-Qubit Generalizations

**N-qubit GHZ:**
$$|GHZ_N\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes N} + |1\rangle^{\otimes N})$$

**N-qubit W:**
$$|W_N\rangle = \frac{1}{\sqrt{N}}(|10...0\rangle + |01...0\rangle + ... + |00...1\rangle)$$

**Dicke states** generalize W:
$$|D_N^k\rangle = \binom{N}{k}^{-1/2} \sum_{\text{permutations}} |1\rangle^{\otimes k} |0\rangle^{\otimes (N-k)}$$

### GHZ Paradox (All-or-Nothing)

GHZ states reveal quantum non-locality without inequalities!

**Setup:** Measure each qubit in $X$ or $Y$ basis.

**QM predictions:**
- $X_A X_B X_C |GHZ\rangle = +|GHZ\rangle$
- $X_A Y_B Y_C |GHZ\rangle = +|GHZ\rangle$
- $Y_A X_B Y_C |GHZ\rangle = +|GHZ\rangle$
- $Y_A Y_B X_C |GHZ\rangle = -|GHZ\rangle$

**Classical (LHV):** Would require $x_A x_B x_C = +1$ always, but:
$$x_A y_B y_C \cdot y_A x_B y_C \cdot y_A y_B x_C = x_A x_B x_C \cdot (y_A y_B y_C)^2 = x_A x_B x_C = +1$$

Yet QM predicts: $(+1)(+1)(+1)(-1)^{-1} = -1$. **Contradiction!**

---

## Worked Examples

### Example 1: GHZ Reduced State
Compute $\rho_{AB} = \text{Tr}_C(|GHZ\rangle\langle GHZ|)$.

**Solution:**
$$|GHZ\rangle\langle GHZ| = \frac{1}{2}(|000\rangle + |111\rangle)(\langle 000| + \langle 111|)$$
$$= \frac{1}{2}(|000\rangle\langle 000| + |000\rangle\langle 111| + |111\rangle\langle 000| + |111\rangle\langle 111|)$$

Trace over C (sum over $\langle 0|$ and $\langle 1|$ on third qubit):
$$\rho_{AB} = \frac{1}{2}(\langle 0_C|000\rangle\langle 000|0_C\rangle + \langle 1_C|111\rangle\langle 111|1_C\rangle)$$
$$= \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$

This is a classical mixture—no entanglement! ∎

### Example 2: W Reduced State Entanglement
Show $\rho_{AB}^W$ is entangled.

**Solution:**
$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

Trace over C:
$$\rho_{AB} = \frac{1}{3}(\langle 1|001\rangle\langle 001|1\rangle + \langle 0|010\rangle\langle 010|0\rangle + \langle 0|100\rangle\langle 100|0\rangle)$$
$$= \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10|)$$

Wait—need cross terms from $|010\rangle\langle 100| + |100\rangle\langle 010|$:
$$\rho_{AB} = \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10| + |01\rangle\langle 10| + |10\rangle\langle 01|)$$

PPT check: compute partial transpose and eigenvalues. Result: one negative eigenvalue → entangled! ∎

### Example 3: SLOCC Invariant
The **3-tangle** distinguishes GHZ from W:
$$\tau_3 = |\langle \psi| \sigma_y^{\otimes 3} |\psi^*\rangle|^2$$

For GHZ: $\tau_3 = 1$
For W: $\tau_3 = 0$

---

## Practice Problems

### Problem 1: Four-Qubit States
Construct 4-qubit GHZ and W states. What are the classes?

### Problem 2: Mermin Inequality
Derive the Mermin inequality for GHZ states and show quantum violation.

### Problem 3: Graph States
Show that $|GHZ\rangle$ is a graph state. What is the graph?

---

## Computational Lab

```python
"""Day 538: GHZ and W States"""
import numpy as np
from scipy.linalg import eigvalsh
from itertools import product

def computational_basis(n):
    """Return computational basis states for n qubits"""
    return [np.array([int(b) for b in format(i, f'0{n}b')]) for i in range(2**n)]

def ket(bits):
    """Create state vector from bit string"""
    n = len(bits)
    state = np.zeros(2**n, dtype=complex)
    idx = sum(b * 2**(n-1-i) for i, b in enumerate(bits))
    state[idx] = 1
    return state

def ghz_state(n=3):
    """N-qubit GHZ state"""
    state = ket([0]*n) + ket([1]*n)
    return state / np.linalg.norm(state)

def w_state(n=3):
    """N-qubit W state"""
    state = np.zeros(2**n, dtype=complex)
    for i in range(n):
        bits = [0]*n
        bits[i] = 1
        state += ket(bits)
    return state / np.linalg.norm(state)

def partial_trace(rho, keep, dims):
    """
    Partial trace over subsystems.
    keep: list of indices to keep
    dims: list of subsystem dimensions
    """
    n = len(dims)
    trace_out = [i for i in range(n) if i not in keep]

    # Reshape
    shape = dims + dims
    rho = rho.reshape(shape)

    # Trace out unwanted subsystems
    for i in sorted(trace_out, reverse=True):
        rho = np.trace(rho, axis1=i, axis2=i+n)
        n -= 1
        # Adjust remaining indices
        shape = list(rho.shape[:n]) + list(rho.shape[n:])

    kept_dim = np.prod([dims[i] for i in keep])
    return rho.reshape(kept_dim, kept_dim)

def negativity(rho, dim_A, dim_B):
    """Compute negativity"""
    rho_TB = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_TB = rho_TB.transpose(0, 3, 2, 1).reshape(dim_A*dim_B, dim_A*dim_B)
    eigenvalues = eigvalsh(rho_TB)
    return np.sum(np.abs(eigenvalues[eigenvalues < 0]))

def projector(psi):
    return np.outer(psi, psi.conj())

# Create states
print("=== GHZ and W States (3 qubits) ===\n")

ghz = ghz_state(3)
w = w_state(3)

print("GHZ state:")
for i, amp in enumerate(ghz):
    if np.abs(amp) > 1e-10:
        bits = format(i, '03b')
        print(f"  |{bits}⟩: {amp:.4f}")

print("\nW state:")
for i, amp in enumerate(w):
    if np.abs(amp) > 1e-10:
        bits = format(i, '03b')
        print(f"  |{bits}⟩: {amp:.4f}")

# Reduced density matrices
print("\n=== Reduced Density Matrices (trace out C) ===\n")

rho_ghz = projector(ghz)
rho_w = projector(w)

# Trace out qubit C (index 2)
rho_ghz_AB = partial_trace(rho_ghz, [0, 1], [2, 2, 2])
rho_w_AB = partial_trace(rho_w, [0, 1], [2, 2, 2])

print("GHZ reduced ρ_AB:")
print(np.round(rho_ghz_AB.real, 4))
print(f"Negativity: {negativity(rho_ghz_AB, 2, 2):.4f}")

print("\nW reduced ρ_AB:")
print(np.round(rho_w_AB.real, 4))
print(f"Negativity: {negativity(rho_w_AB, 2, 2):.4f}")

# Single qubit reduced states
print("\n=== Single Qubit Reduced States ===\n")

rho_ghz_A = partial_trace(rho_ghz, [0], [2, 2, 2])
rho_w_A = partial_trace(rho_w, [0], [2, 2, 2])

print("GHZ reduced ρ_A:")
print(np.round(rho_ghz_A.real, 4))

print("\nW reduced ρ_A:")
print(np.round(rho_w_A.real, 4))

# GHZ paradox measurements
print("\n=== GHZ Paradox (Stabilizer Check) ===\n")

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def measure_operator(psi, op):
    """Compute expectation value"""
    return np.real(psi.conj() @ op @ psi)

# Check stabilizer eigenvalues
XXX = np.kron(np.kron(X, X), X)
XYY = np.kron(np.kron(X, Y), Y)
YXY = np.kron(np.kron(Y, X), Y)
YYX = np.kron(np.kron(Y, Y), X)

print("GHZ stabilizer eigenvalues:")
print(f"  XXX: {measure_operator(ghz, XXX):+.4f}")
print(f"  XYY: {measure_operator(ghz, XYY):+.4f}")
print(f"  YXY: {measure_operator(ghz, YXY):+.4f}")
print(f"  YYX: {measure_operator(ghz, YYX):+.4f}")

print("\nProduct of last three should be -XXX for LHV, but QM gives:")
print(f"  XYY × YXY × YYX = +1, while YYX = -1")
print("  Contradiction with local hidden variables!")

# N-qubit generalization
print("\n=== N-Qubit Generalizations ===\n")

for n in [3, 4, 5]:
    ghz_n = ghz_state(n)
    w_n = w_state(n)

    rho_ghz_n = projector(ghz_n)
    rho_w_n = projector(w_n)

    # Single qubit reduced state
    rho_ghz_1 = partial_trace(rho_ghz_n, [0], [2]*n)
    rho_w_1 = partial_trace(rho_w_n, [0], [2]*n)

    # Purity of single qubit
    purity_ghz = np.trace(rho_ghz_1 @ rho_ghz_1).real
    purity_w = np.trace(rho_w_1 @ rho_w_1).real

    print(f"N = {n}:")
    print(f"  GHZ single-qubit purity: {purity_ghz:.4f}")
    print(f"  W single-qubit purity: {purity_w:.4f}")
```

**Expected Output:**
```
=== GHZ and W States (3 qubits) ===

GHZ state:
  |000⟩: 0.7071
  |111⟩: 0.7071

W state:
  |001⟩: 0.5774
  |010⟩: 0.5774
  |100⟩: 0.5774

=== Reduced Density Matrices (trace out C) ===

GHZ reduced ρ_AB:
[[0.5 0.  0.  0. ]
 [0.  0.  0.  0. ]
 [0.  0.  0.  0. ]
 [0.  0.  0.  0.5]]
Negativity: 0.0000

W reduced ρ_AB:
[[0.3333 0.     0.     0.    ]
 [0.     0.3333 0.3333 0.    ]
 [0.     0.3333 0.3333 0.    ]
 [0.     0.     0.     0.    ]]
Negativity: 0.1667
```

---

## Summary

### Key States

| State | Form | Class |
|-------|------|-------|
| GHZ | $(|000\rangle + |111\rangle)/\sqrt{2}$ | GHZ class |
| W | $(|001\rangle + |010\rangle + |100\rangle)/\sqrt{3}$ | W class |

### Key Takeaways
1. **GHZ** is maximally entangled but fragile
2. **W** has robust bipartite entanglement
3. **SLOCC inequivalent:** cannot convert between classes
4. **GHZ paradox** shows quantum nonlocality without inequalities
5. **Applications:** GHZ for quantum secret sharing, W for quantum networks

---

## Daily Checklist

- [ ] I can write GHZ and W states
- [ ] I understand SLOCC classification
- [ ] I can compute reduced density matrices
- [ ] I understand the robustness difference
- [ ] I can explain the GHZ paradox

---

*Next: Day 539 — Week Review*
