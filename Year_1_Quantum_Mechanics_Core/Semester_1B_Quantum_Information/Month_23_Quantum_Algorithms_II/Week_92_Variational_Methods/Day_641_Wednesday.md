# Day 641: Parameterized Quantum Circuits

## Overview
**Day 641** | Week 92, Day 4 | Year 1, Month 23 | Variational Methods

Today we study parameterized quantum circuits (PQCs), also called variational ansatze, analyzing their expressibility, entanglement capabilities, and suitability for different problems.

---

## Learning Objectives

1. Define parameterized quantum circuits
2. Analyze circuit expressibility
3. Understand entangling capability
4. Design hardware-efficient ansatze
5. Compare problem-specific vs general ansatze
6. Quantify circuit properties

---

## Core Content

### Parameterized Quantum Circuits

**Definition:** A PQC is a unitary $U(\theta)$ that depends on classical parameters $\theta = (\theta_1, ..., \theta_m)$.

**Standard form:**
$$U(\theta) = \prod_{l=1}^{L} U_l(\theta_l) W_l$$

where $U_l$ are parameterized gates and $W_l$ are fixed gates.

### Common Gate Families

**Single-qubit rotations:**
$$R_x(\theta) = e^{-i\theta X/2}, \quad R_y(\theta) = e^{-i\theta Y/2}, \quad R_z(\theta) = e^{-i\theta Z/2}$$

**General single-qubit:**
$$U(\theta, \phi, \lambda) = R_z(\phi)R_y(\theta)R_z(\lambda)$$

**Entangling gates:** CNOT, CZ, $R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z/2}$

### Hardware-Efficient Ansatz

**Structure:**
```
Layer 1:     R_y ─ R_z ─●─
                        │
             R_y ─ R_z ─X─

Layer 2:     R_y ─ R_z ─●─
                        │
             R_y ─ R_z ─X─
```

**Advantages:**
- Native to hardware topology
- Minimal circuit depth
- All gates easily implementable

**Disadvantages:**
- May not capture problem structure
- Can suffer from barren plateaus

### Expressibility

**Definition:** Ability of PQC to generate diverse quantum states.

**Measure:** Compare ensemble of states to Haar-random distribution.

$$\text{Expr} = D_{KL}\left(P_{PQC}(F) \| P_{Haar}(F)\right)$$

where $F$ is fidelity between pairs of states.

More expressive = can represent more states.

### Entangling Capability

**Definition:** Ability to generate entangled states.

**Measure:** Meyer-Wallach entanglement over parameter space.

$$Q = \frac{1}{|\Theta|}\sum_\theta Q(|\psi(\theta)\rangle)$$

Higher $Q$ = more entanglement on average.

### Problem-Inspired Ansatze

**Chemistry (UCCSD):**
$$U = e^{T - T^\dagger}$$

Preserves particle number, physically motivated.

**QAOA:**
$$U = \prod_p e^{-i\beta H_M} e^{-i\gamma H_C}$$

Problem structure built in.

### Tradeoffs

| Aspect | Hardware-Efficient | Problem-Specific |
|--------|-------------------|------------------|
| Depth | Low | Higher |
| Expressibility | High | Targeted |
| Trainability | May have issues | Often better |
| Hardware fit | Excellent | May need adaptation |

---

## Computational Lab

```python
"""Day 641: Parameterized Quantum Circuits"""
import numpy as np
import matplotlib.pyplot as plt

def ry_gate(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

def rz_gate(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]])

def cnot():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def hardware_efficient_layer(params, n_qubits):
    """One layer of hardware-efficient ansatz."""
    # Single-qubit rotations
    U = np.eye(1)
    for q in range(n_qubits):
        U = np.kron(U, ry_gate(params[2*q]) @ rz_gate(params[2*q+1]))

    # CNOT ladder
    for q in range(n_qubits - 1):
        CNOT = np.eye(2**n_qubits)
        # Apply CNOT between q and q+1
        # Simplified: assume 2 qubits
        if n_qubits == 2:
            CNOT = cnot()
            U = CNOT @ U

    return U

def random_state_ensemble(ansatz_func, n_samples=100):
    """Generate ensemble of random PQC states."""
    states = []
    for _ in range(n_samples):
        params = np.random.uniform(0, 2*np.pi, 4)  # 2 qubits, 2 params each
        state = ansatz_func(params) @ np.array([1, 0, 0, 0])
        states.append(state)
    return states

def compute_fidelities(states):
    """Compute pairwise fidelities."""
    fidelities = []
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            F = np.abs(states[i].conj() @ states[j])**2
            fidelities.append(F)
    return fidelities

# Generate states and compute expressibility
states = random_state_ensemble(
    lambda p: hardware_efficient_layer(p, 2),
    n_samples=200
)
fidelities = compute_fidelities(states)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(fidelities, bins=30, density=True, alpha=0.7, label='PQC')
# Haar random for 2 qubits: Beta distribution
F_range = np.linspace(0, 1, 100)
haar_dist = 3 * (1 - F_range)**2  # Haar for d=4
plt.plot(F_range, haar_dist, 'r-', label='Haar', linewidth=2)
plt.xlabel('Fidelity', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Expressibility Analysis', fontsize=14)
plt.legend()

plt.subplot(1, 2, 2)
# Entanglement distribution
def concurrence(state):
    """Compute concurrence for 2-qubit pure state."""
    rho = np.outer(state, state.conj())
    Y = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(Y, Y)
    rho_tilde = YY @ rho.conj() @ YY
    from scipy.linalg import sqrtm
    R = sqrtm(sqrtm(rho) @ rho_tilde @ sqrtm(rho))
    eigvals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    return max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])

concurrences = [concurrence(s) for s in states]
plt.hist(concurrences, bins=30, alpha=0.7)
plt.xlabel('Concurrence', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Entanglement Distribution', fontsize=14)

plt.tight_layout()
plt.savefig('pqc_analysis.png', dpi=150)
plt.show()

print(f"Average concurrence: {np.mean(concurrences):.4f}")
```

---

## Summary

### Key Concepts

| Property | Definition |
|----------|------------|
| Expressibility | Diversity of achievable states |
| Entangling capability | Ability to create entanglement |
| Hardware-efficient | Adapted to device topology |
| Problem-specific | Structured for particular problem |

---

## Daily Checklist

- [ ] I understand PQC structure
- [ ] I know about expressibility
- [ ] I understand entangling capability
- [ ] I can design hardware-efficient ansatze
- [ ] I ran the computational lab

---

*Next: Day 642 — Optimization Landscapes*
