# Day 534: Bell States

## Overview
**Day 534** | Week 77, Day 2 | Year 1, Month 20 | Maximally Entangled States

Today we study the Bell states—the four maximally entangled two-qubit states that form a complete orthonormal basis and serve as the fundamental resource for quantum communication protocols.

---

## Learning Objectives
1. Define and construct the four Bell states
2. Prove they form an orthonormal basis
3. Show maximal entanglement via Schmidt coefficients
4. Understand Bell state creation circuits
5. Compute Bell state measurements
6. Apply Bell states in quantum protocols

---

## Core Content

### The Four Bell States

$$\boxed{|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)}$$
$$\boxed{|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)}$$
$$\boxed{|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)}$$
$$\boxed{|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)}$$

### Naming Convention

| State | Same bits | Relative phase |
|-------|-----------|----------------|
| $\|\Phi^+\rangle$ | Yes (00,11) | + |
| $\|\Phi^-\rangle$ | Yes (00,11) | − |
| $\|\Psi^+\rangle$ | No (01,10) | + |
| $\|\Psi^-\rangle$ | No (01,10) | − |

### Orthonormality

The Bell states satisfy:
$$\langle\Phi^+|\Phi^+\rangle = \langle\Phi^-|\Phi^-\rangle = \langle\Psi^+|\Psi^+\rangle = \langle\Psi^-|\Psi^-\rangle = 1$$
$$\langle\Phi^+|\Phi^-\rangle = \langle\Phi^+|\Psi^+\rangle = \cdots = 0$$

They form a complete basis for $\mathbb{C}^2 \otimes \mathbb{C}^2$.

### Maximal Entanglement

Each Bell state has Schmidt decomposition:
$$|\Phi^+\rangle = \sqrt{\frac{1}{2}}|0\rangle|0\rangle + \sqrt{\frac{1}{2}}|1\rangle|1\rangle$$

Schmidt coefficients: $\lambda_1 = \lambda_2 = 1/2$

**Entanglement entropy:**
$$S = -\sum_i \lambda_i \log_2 \lambda_i = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1 \text{ ebit}$$

This is the **maximum** for two qubits.

### Bell State Circuit

```
|0⟩ ─────[H]─────●───── |Φ⁺⟩ (first qubit)
                 │
|0⟩ ─────────────⊕───── |Φ⁺⟩ (second qubit)
```

1. Apply Hadamard to first qubit: $|0\rangle \to |+\rangle$
2. Apply CNOT: $|+0\rangle \to |\Phi^+\rangle$

### Creating All Four Bell States

Starting from $|00\rangle$:
- $|\Phi^+\rangle$: H on qubit 1, CNOT
- $|\Phi^-\rangle$: X on qubit 1, H on qubit 1, CNOT
- $|\Psi^+\rangle$: H on qubit 1, CNOT, X on qubit 2
- $|\Psi^-\rangle$: X on qubit 1, H on qubit 1, CNOT, X on qubit 2

Or equivalently, using Pauli operations on $|\Phi^+\rangle$:
$$|\Phi^-\rangle = (Z \otimes I)|\Phi^+\rangle$$
$$|\Psi^+\rangle = (X \otimes I)|\Phi^+\rangle$$
$$|\Psi^-\rangle = (iY \otimes I)|\Phi^+\rangle$$

### Bell Measurement

The Bell basis measurement distinguishes between the four Bell states.

**Circuit:**
```
─────●─────[H]───M───  (outcome a)
     │
─────⊕─────────────M───  (outcome b)
```

| Input | After CNOT | After H | Measurement (a,b) |
|-------|------------|---------|-------------------|
| $\|\Phi^+\rangle$ | $\|+0\rangle$ | $\|00\rangle$ | 0,0 |
| $\|\Phi^-\rangle$ | $\|-0\rangle$ | $\|10\rangle$ | 1,0 |
| $\|\Psi^+\rangle$ | $\|+1\rangle$ | $\|01\rangle$ | 0,1 |
| $\|\Psi^-\rangle$ | $\|-1\rangle$ | $\|11\rangle$ | 1,1 |

### Density Matrices

For $|\Phi^+\rangle$:
$$\rho_{\Phi^+} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

### Reduced Density Matrix

For any Bell state $|\beta\rangle$:
$$\rho_A = \text{Tr}_B(|\beta\rangle\langle\beta|) = \frac{I}{2}$$

The reduced state is **maximally mixed**—this is characteristic of maximal entanglement.

---

## Worked Examples

### Example 1: Verify Bell State Orthonormality
Show $\langle\Phi^+|\Psi^+\rangle = 0$.

**Solution:**
$$\langle\Phi^+| = \frac{1}{\sqrt{2}}(\langle 00| + \langle 11|)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$

$$\langle\Phi^+|\Psi^+\rangle = \frac{1}{2}(\langle 00|01\rangle + \langle 00|10\rangle + \langle 11|01\rangle + \langle 11|10\rangle)$$
$$= \frac{1}{2}(0 + 0 + 0 + 0) = 0$$ ∎

### Example 2: Bell State from CNOT
Show that CNOT transforms $|+0\rangle$ to $|\Phi^+\rangle$.

**Solution:**
$$|+0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

CNOT action: $|ab\rangle \to |a, a\oplus b\rangle$
$$\text{CNOT}|+0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$ ∎

### Example 3: Reduced State Calculation
Compute $\rho_A = \text{Tr}_B(|\Psi^-\rangle\langle\Psi^-|)$.

**Solution:**
$$|\Psi^-\rangle\langle\Psi^-| = \frac{1}{2}(|01\rangle - |10\rangle)(\langle 01| - \langle 10|)$$
$$= \frac{1}{2}(|01\rangle\langle 01| - |01\rangle\langle 10| - |10\rangle\langle 01| + |10\rangle\langle 10|)$$

Taking partial trace (trace over B):
$$\rho_A = \frac{1}{2}(\langle 1|01\rangle\langle 01|1\rangle|0\rangle\langle 0| + \text{cross terms} + |1\rangle\langle 1|\langle 0|10\rangle\langle 10|0\rangle)$$

After careful calculation:
$$\rho_A = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$ ∎

---

## Practice Problems

### Problem 1: Bell State Completeness
Show that $\sum_{\beta} |\beta\rangle\langle\beta| = I_4$ where $\beta$ ranges over all four Bell states.

### Problem 2: Local Pauli Equivalence
Verify that $(X \otimes I)|\Phi^+\rangle = |\Psi^+\rangle$ explicitly.

### Problem 3: Bell Measurement
If the state $\frac{1}{\sqrt{2}}(|00\rangle + i|11\rangle)$ undergoes Bell measurement, what are the outcome probabilities?

---

## Computational Lab

```python
"""Day 534: Bell States"""
import numpy as np
from scipy.linalg import sqrtm

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def bell_states():
    """Return the four Bell states as vectors"""
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return {'Φ⁺': phi_plus, 'Φ⁻': phi_minus, 'Ψ⁺': psi_plus, 'Ψ⁻': psi_minus}

def cnot():
    """CNOT gate matrix"""
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)

def create_bell_state():
    """Create |Φ⁺⟩ using H and CNOT"""
    # Start with |00⟩
    state = np.array([1, 0, 0, 0], dtype=complex)
    # Apply H ⊗ I
    HI = np.kron(H, I)
    state = HI @ state
    print(f"After H⊗I: {state}")
    # Apply CNOT
    state = cnot() @ state
    print(f"After CNOT: {state}")
    return state

def partial_trace_B(rho_AB, dim_A=2, dim_B=2):
    """Compute partial trace over B"""
    rho = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=1, axis2=3)

def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

# Test Bell states
print("=== Bell States ===")
bells = bell_states()
for name, state in bells.items():
    print(f"|{name}⟩ = {state}")

# Verify orthonormality
print("\n=== Orthonormality Check ===")
for n1, s1 in bells.items():
    for n2, s2 in bells.items():
        inner = np.abs(np.vdot(s1, s2))
        if inner > 1e-10:
            print(f"⟨{n1}|{n2}⟩ = {inner:.4f}")

# Create Bell state via circuit
print("\n=== Bell State Creation Circuit ===")
created = create_bell_state()
print(f"Created state matches |Φ⁺⟩: {np.allclose(created, bells['Φ⁺'])}")

# Reduced density matrices
print("\n=== Reduced Density Matrices ===")
for name, state in bells.items():
    rho = np.outer(state, state.conj())
    rho_A = partial_trace_B(rho)
    print(f"ρ_A for |{name}⟩:")
    print(rho_A)
    print(f"Entropy: {von_neumann_entropy(rho_A):.4f} ebits\n")

# Bell measurement simulation
print("=== Bell Measurement ===")
def bell_measurement(state):
    """Simulate Bell measurement, return outcomes"""
    # Apply CNOT then H⊗I
    state = cnot() @ state
    state = np.kron(H, I) @ state
    probs = np.abs(state)**2
    return probs

print("Bell measurement probabilities:")
for name, state in bells.items():
    probs = bell_measurement(state.copy())
    print(f"|{name}⟩: |00⟩:{probs[0]:.2f}, |01⟩:{probs[1]:.2f}, |10⟩:{probs[2]:.2f}, |11⟩:{probs[3]:.2f}")
```

**Expected Output:**
```
=== Bell States ===
|Φ⁺⟩ = [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
|Φ⁻⟩ = [ 0.70710678+0.j  0.        +0.j  0.        +0.j -0.70710678+0.j]
|Ψ⁺⟩ = [0.        +0.j 0.70710678+0.j 0.70710678+0.j 0.        +0.j]
|Ψ⁻⟩ = [ 0.        +0.j  0.70710678+0.j -0.70710678+0.j  0.        +0.j]

=== Orthonormality Check ===
⟨Φ⁺|Φ⁺⟩ = 1.0000
⟨Φ⁻|Φ⁻⟩ = 1.0000
⟨Ψ⁺|Ψ⁺⟩ = 1.0000
⟨Ψ⁻|Ψ⁻⟩ = 1.0000

=== Reduced Density Matrices ===
ρ_A for |Φ⁺⟩:
[[0.5+0.j 0. +0.j]
 [0. +0.j 0.5+0.j]]
Entropy: 1.0000 ebits
```

---

## Summary

### Key Formulas

| Bell State | Vector Form |
|------------|-------------|
| $\|\Phi^+\rangle$ | $(|00\rangle + |11\rangle)/\sqrt{2}$ |
| $\|\Phi^-\rangle$ | $(|00\rangle - |11\rangle)/\sqrt{2}$ |
| $\|\Psi^+\rangle$ | $(|01\rangle + |10\rangle)/\sqrt{2}$ |
| $\|\Psi^-\rangle$ | $(|01\rangle - |10\rangle)/\sqrt{2}$ |

### Key Takeaways
1. **Four Bell states** form orthonormal basis for two-qubit Hilbert space
2. **Maximal entanglement**: 1 ebit of entanglement entropy
3. **Reduced states** are maximally mixed (I/2)
4. **Circuit**: H + CNOT creates Bell states
5. **Local Paulis** interconvert Bell states

---

## Daily Checklist

- [ ] I can write all four Bell states from memory
- [ ] I understand why they are maximally entangled
- [ ] I can derive the Bell state creation circuit
- [ ] I understand the Bell measurement procedure
- [ ] I can compute reduced density matrices

---

*Next: Day 535 — Entanglement Detection*
