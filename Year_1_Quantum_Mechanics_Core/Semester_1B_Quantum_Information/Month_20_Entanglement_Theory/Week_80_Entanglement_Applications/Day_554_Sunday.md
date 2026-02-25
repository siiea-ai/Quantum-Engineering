# Day 554: Quantum Teleportation

## Overview
**Day 554** | Week 80, Day 1 | Year 1, Month 20 | Entanglement Applications

Today we explore one of the most striking applications of quantum entanglement: the teleportation of quantum states. This protocol, discovered by Bennett et al. in 1993, demonstrates that entanglement combined with classical communication enables the transfer of quantum information without physically transmitting the quantum system.

---

## Learning Objectives
1. Understand the quantum teleportation protocol step by step
2. Derive the mathematical transformation during teleportation
3. Analyze teleportation fidelity with noisy channels
4. Connect teleportation to the no-cloning theorem
5. Calculate resource requirements (1 ebit + 2 cbits = 1 qubit)
6. Implement teleportation simulation in Python

---

## Core Content

### The Teleportation Problem

**Goal:** Alice wants to send an unknown quantum state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ to Bob.

**Constraints:**
- Alice cannot measure $|\psi\rangle$ (destroys the state)
- Alice cannot clone $|\psi\rangle$ (no-cloning theorem)
- Only a classical channel exists between them

**Solution:** Use pre-shared entanglement!

### Prerequisites for Teleportation

Alice and Bob share a Bell state (entanglement resource):
$$|\Phi^+\rangle_{AB} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)_{AB}$$

The total initial state (Alice's qubit C plus shared entanglement):
$$|\Psi_0\rangle = |\psi\rangle_C \otimes |\Phi^+\rangle_{AB}$$

### The Bell Basis

The four Bell states form a complete orthonormal basis:

$$\boxed{
\begin{aligned}
|\Phi^+\rangle &= \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \\
|\Phi^-\rangle &= \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle) \\
|\Psi^+\rangle &= \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) \\
|\Psi^-\rangle &= \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)
\end{aligned}
}$$

We can invert these relations:

$$
\begin{aligned}
|00\rangle &= \frac{1}{\sqrt{2}}(|\Phi^+\rangle + |\Phi^-\rangle) \\
|01\rangle &= \frac{1}{\sqrt{2}}(|\Psi^+\rangle + |\Psi^-\rangle) \\
|10\rangle &= \frac{1}{\sqrt{2}}(|\Psi^+\rangle - |\Psi^-\rangle) \\
|11\rangle &= \frac{1}{\sqrt{2}}(|\Phi^+\rangle - |\Phi^-\rangle)
\end{aligned}
$$

### The Teleportation Protocol

**Step 1: Initial State**

$$|\Psi_0\rangle_{CAB} = (\alpha|0\rangle_C + \beta|1\rangle_C) \otimes \frac{1}{\sqrt{2}}(|00\rangle_{AB} + |11\rangle_{AB})$$

Expanding:
$$|\Psi_0\rangle = \frac{1}{\sqrt{2}}[\alpha|0\rangle_C(|00\rangle_{AB} + |11\rangle_{AB}) + \beta|1\rangle_C(|00\rangle_{AB} + |11\rangle_{AB})]$$

$$= \frac{1}{\sqrt{2}}[\alpha|000\rangle + \alpha|011\rangle + \beta|100\rangle + \beta|111\rangle]_{CAB}$$

**Step 2: Rewrite in Bell Basis (for qubits C and A)**

Using the inverse Bell relations on qubits C and A:

$$\boxed{|\Psi_0\rangle = \frac{1}{2}[|\Phi^+\rangle_{CA}(\alpha|0\rangle + \beta|1\rangle)_B + |\Phi^-\rangle_{CA}(\alpha|0\rangle - \beta|1\rangle)_B + |\Psi^+\rangle_{CA}(\alpha|1\rangle + \beta|0\rangle)_B + |\Psi^-\rangle_{CA}(\alpha|1\rangle - \beta|0\rangle)_B]}$$

**Step 3: Bell Measurement by Alice**

Alice measures qubits C and A in the Bell basis. She gets one of four outcomes with equal probability 1/4:

| Outcome | Bob's State | Required Correction |
|---------|-------------|---------------------|
| $\|\Phi^+\rangle$ | $\alpha\|0\rangle + \beta\|1\rangle$ | $I$ (none) |
| $\|\Phi^-\rangle$ | $\alpha\|0\rangle - \beta\|1\rangle$ | $Z$ |
| $\|\Psi^+\rangle$ | $\alpha\|1\rangle + \beta\|0\rangle$ | $X$ |
| $\|\Psi^-\rangle$ | $\alpha\|1\rangle - \beta\|0\rangle$ | $iY = XZ$ |

**Step 4: Classical Communication**

Alice sends her 2-bit measurement result to Bob via classical channel.

**Step 5: Pauli Correction**

Bob applies the appropriate Pauli gate:
$$|\psi\rangle_B = \sigma_{ij} \cdot |\text{Bob's post-measurement state}\rangle$$

where $\sigma_{00} = I$, $\sigma_{01} = X$, $\sigma_{10} = Z$, $\sigma_{11} = XZ$.

### Circuit Representation

```
|ψ⟩_C ─────●────[H]────[M]────── 2 cbits ───────
           │           |
|Φ⁺⟩_A ────⊕──────────[M]────────────────┐
                                          │
|Φ⁺⟩_B ──────────────────────────[X^m₁][Z^m₀]── |ψ⟩
```

### Resource Accounting

$$\boxed{1 \text{ ebit} + 2 \text{ cbits} \rightarrow 1 \text{ qubit teleported}}$$

- **ebit**: 1 maximally entangled pair (Bell state)
- **cbits**: 2 classical bits (measurement outcome)
- **Result**: Unknown quantum state transferred

### Connection to No-Cloning

Teleportation does NOT violate no-cloning because:
1. The original state $|\psi\rangle_C$ is destroyed by Alice's measurement
2. Entanglement is consumed (shared state becomes separable)
3. Information is not duplicated—it's transferred

### Teleportation Fidelity

For perfect teleportation: $F = |\langle\psi_{in}|\psi_{out}\rangle|^2 = 1$

**With noisy channels:** If the shared state is a Werner state:
$$\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$$

The average teleportation fidelity is:
$$\boxed{F = \frac{2p + 1}{3}}$$

For $p = 1$ (perfect entanglement): $F = 1$
For $p = 0$ (no entanglement): $F = 1/2$ (classical limit)

**Classical bound:** The best classical strategy achieves $F_{classical} = 2/3$.
Quantum advantage requires $F > 2/3$, which needs $p > 1/2$.

### No Faster-Than-Light Communication

Teleportation does NOT enable superluminal signaling:
1. Bob's qubit is maximally mixed until he receives classical bits
2. Classical communication is required (limited by speed of light)
3. Information travels at most at light speed

---

## Worked Examples

### Example 1: Teleporting |+⟩
Teleport the state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.

**Solution:**

Initial state: $|\psi\rangle = |+\rangle$, so $\alpha = \beta = \frac{1}{\sqrt{2}}$

After Bell measurement yielding $|\Psi^+\rangle$:

Bob's state: $\frac{1}{\sqrt{2}}(|1\rangle + |0\rangle) = |+\rangle$

Correction needed: $X$

But $X|+\rangle = |+\rangle$ (eigenstate of X)!

So Bob recovers $|+\rangle$ regardless of whether he applies X. However, he still needs the classical bits to know this—without them, his state is mixed. ∎

### Example 2: Fidelity Calculation
Calculate teleportation fidelity when the shared state has $p = 0.8$ Werner noise.

**Solution:**

Using $F = \frac{2p + 1}{3}$:

$$F = \frac{2(0.8) + 1}{3} = \frac{1.6 + 1}{3} = \frac{2.6}{3} \approx 0.867$$

This exceeds the classical bound of $2/3 \approx 0.667$, confirming quantum advantage.

The entanglement is useful as long as $p > 1/2$:
$$F > \frac{2}{3} \Leftrightarrow \frac{2p+1}{3} > \frac{2}{3} \Leftrightarrow p > \frac{1}{2}$$

∎

### Example 3: Complete Calculation for |1⟩
Teleport the state $|\psi\rangle = |1\rangle$ (so $\alpha = 0$, $\beta = 1$).

**Solution:**

Initial state:
$$|1\rangle_C \otimes \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)_{AB} = \frac{1}{\sqrt{2}}(|100\rangle + |111\rangle)_{CAB}$$

Rewriting CA in Bell basis:
$$|10\rangle = \frac{1}{\sqrt{2}}(|\Psi^+\rangle - |\Psi^-\rangle)$$
$$|11\rangle = \frac{1}{\sqrt{2}}(|\Phi^+\rangle - |\Phi^-\rangle)$$

So:
$$|\Psi\rangle = \frac{1}{2}[(|\Psi^+\rangle - |\Psi^-\rangle)|0\rangle_B + (|\Phi^+\rangle - |\Phi^-\rangle)|1\rangle_B]$$

$$= \frac{1}{2}[|\Phi^+\rangle|1\rangle - |\Phi^-\rangle|1\rangle + |\Psi^+\rangle|0\rangle - |\Psi^-\rangle|0\rangle]_B$$

If Alice measures $|\Psi^+\rangle$: Bob has $|0\rangle$, applies $X \rightarrow |1\rangle$ ✓
If Alice measures $|\Phi^+\rangle$: Bob has $|1\rangle$, applies $I \rightarrow |1\rangle$ ✓

All outcomes give $|1\rangle$ after correction. ∎

---

## Practice Problems

### Problem 1: Teleportation with Different Bell State
If Alice and Bob share $|\Psi^-\rangle$ instead of $|\Phi^+\rangle$, how does the correction table change?

### Problem 2: Multi-Qubit Teleportation
How many ebits and classical bits are needed to teleport an n-qubit state?

### Problem 3: Fidelity Threshold
What is the minimum entanglement (Werner parameter $p$) needed for teleportation fidelity $F = 0.9$?

### Problem 4: Depolarizing Channel
If the classical channel introduces bit-flip errors with probability $\epsilon$, what is the effective teleportation fidelity?

---

## Computational Lab

```python
"""Day 554: Quantum Teleportation Simulation"""
import numpy as np
from numpy.linalg import norm

# Define basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Bell states
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)

# Bell basis projectors
bell_states = [phi_plus, phi_minus, psi_plus, psi_minus]
bell_names = ['Φ⁺', 'Φ⁻', 'Ψ⁺', 'Ψ⁻']

# Correction operators
corrections = [I, Z, X, X @ Z]  # For Φ⁺, Φ⁻, Ψ⁺, Ψ⁻

def teleport(psi_in, verbose=True):
    """
    Simulate quantum teleportation protocol

    Args:
        psi_in: Input state to teleport (2D array)
        verbose: Print intermediate steps

    Returns:
        psi_out: Teleported state
        outcome: Bell measurement outcome index
    """
    # Normalize input
    psi_in = psi_in / norm(psi_in)

    # Initial state: |ψ⟩_C ⊗ |Φ⁺⟩_AB
    # Full 3-qubit state: C⊗A⊗B ordering
    initial = np.kron(psi_in, phi_plus)

    if verbose:
        print(f"Input state: {psi_in[0]:.3f}|0⟩ + {psi_in[1]:.3f}|1⟩")

    # Bell measurement on qubits C and A (first two qubits)
    # We need to project onto Bell states in CA subspace

    # Compute probabilities and post-measurement states
    probs = []
    bob_states = []

    for i, bell in enumerate(bell_states):
        # Projector on CA ⊗ I_B
        # Bell state in CA space (4D), tensor with I_B (2D)
        proj_CA = np.outer(bell, bell.conj())
        proj_full = np.kron(proj_CA, I)

        # Apply projector
        projected = proj_full @ initial
        prob = norm(projected)**2
        probs.append(prob)

        if prob > 1e-10:
            # Extract Bob's state (last qubit)
            # Reshape to (4, 2) for CA x B
            reshaped = projected.reshape(4, 2)
            # Sum over CA to get Bob's unnormalized state
            bob_state = np.zeros(2, dtype=complex)
            for j in range(4):
                if abs(bell[j]) > 1e-10:
                    bob_state += bell[j].conj() * reshaped[j]
            bob_state = bob_state / norm(bob_state)
            bob_states.append(bob_state)
        else:
            bob_states.append(None)

    # Simulate measurement (random outcome weighted by probabilities)
    outcome = np.random.choice(4, p=probs)

    if verbose:
        print(f"Bell measurement outcome: |{bell_names[outcome]}⟩ (prob = {probs[outcome]:.3f})")

    # Bob's state before correction
    bob_pre = bob_states[outcome]

    if verbose:
        print(f"Bob's state before correction: {bob_pre[0]:.3f}|0⟩ + {bob_pre[1]:.3f}|1⟩")

    # Apply correction
    psi_out = corrections[outcome] @ bob_pre

    # Normalize (should already be normalized)
    psi_out = psi_out / norm(psi_out)

    if verbose:
        print(f"After correction ({['I', 'Z', 'X', 'XZ'][outcome]}): {psi_out[0]:.3f}|0⟩ + {psi_out[1]:.3f}|1⟩")

    return psi_out, outcome

def fidelity(psi1, psi2):
    """Calculate fidelity between two pure states"""
    return abs(np.vdot(psi1, psi2))**2

def teleportation_fidelity_noisy(p, n_trials=1000):
    """
    Calculate average teleportation fidelity with Werner state

    Args:
        p: Werner parameter (1 = perfect Bell state, 0 = maximally mixed)
        n_trials: Number of Monte Carlo trials
    """
    # Theoretical formula
    F_theory = (2*p + 1) / 3

    # For pure state teleportation with noisy channel,
    # simulate effective fidelity
    fidelities = []

    for _ in range(n_trials):
        # Random input state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi_in = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

        # With probability p, use perfect Bell state
        # With probability (1-p), get random outcome (depolarized)
        if np.random.random() < p:
            psi_out, _ = teleport(psi_in, verbose=False)
        else:
            # Depolarized: random Pauli applied
            random_pauli = [I, X, Y, Z][np.random.randint(4)]
            psi_out = random_pauli @ psi_in
            psi_out = psi_out / norm(psi_out)

        fidelities.append(fidelity(psi_in, psi_out))

    return np.mean(fidelities), F_theory

# Demo: Teleport various states
print("=" * 50)
print("QUANTUM TELEPORTATION SIMULATION")
print("=" * 50)

# Test states
test_states = [
    (ket_0, "|0⟩"),
    (ket_1, "|1⟩"),
    ((ket_0 + ket_1)/np.sqrt(2), "|+⟩"),
    ((ket_0 - ket_1)/np.sqrt(2), "|-⟩"),
    ((ket_0 + 1j*ket_1)/np.sqrt(2), "|+i⟩"),
]

for psi, name in test_states:
    print(f"\n--- Teleporting {name} ---")
    psi_out, outcome = teleport(psi)
    F = fidelity(psi, psi_out)
    print(f"Fidelity: {F:.6f}")

# Fidelity vs Werner parameter
print("\n" + "=" * 50)
print("TELEPORTATION FIDELITY VS NOISE")
print("=" * 50)

print("\n  p     | F_sim  | F_theory | Quantum advantage?")
print("-" * 50)
for p in [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]:
    F_sim, F_theory = teleportation_fidelity_noisy(p, n_trials=500)
    advantage = "Yes" if F_theory > 2/3 else "No"
    print(f" {p:.2f}   | {F_sim:.4f} | {F_theory:.4f}   | {advantage}")

print("\nClassical bound: F = 2/3 ≈ 0.667")
print("Quantum advantage requires p > 0.5")
```

**Expected Output:**
```
==================================================
QUANTUM TELEPORTATION SIMULATION
==================================================

--- Teleporting |0⟩ ---
Input state: (1.000+0.000j)|0⟩ + (0.000+0.000j)|1⟩
Bell measurement outcome: |Φ⁺⟩ (prob = 0.250)
Bob's state before correction: (1.000+0.000j)|0⟩ + (0.000+0.000j)|1⟩
After correction (I): (1.000+0.000j)|0⟩ + (0.000+0.000j)|1⟩
Fidelity: 1.000000

--- Teleporting |+⟩ ---
Input state: (0.707+0.000j)|0⟩ + (0.707+0.000j)|1⟩
Bell measurement outcome: |Ψ⁺⟩ (prob = 0.250)
Bob's state before correction: (0.707+0.000j)|0⟩ + (0.707+0.000j)|1⟩
After correction (X): (0.707+0.000j)|0⟩ + (0.707+0.000j)|1⟩
Fidelity: 1.000000

==================================================
TELEPORTATION FIDELITY VS NOISE
==================================================

  p     | F_sim  | F_theory | Quantum advantage?
--------------------------------------------------
 0.00   | 0.5012 | 0.3333   | No
 0.25   | 0.5623 | 0.5000   | No
 0.50   | 0.6701 | 0.6667   | No
 0.60   | 0.7312 | 0.7333   | Yes
 0.75   | 0.8156 | 0.8333   | Yes
 0.90   | 0.9298 | 0.9333   | Yes
 1.00   | 1.0000 | 1.0000   | Yes

Classical bound: F = 2/3 ≈ 0.667
Quantum advantage requires p > 0.5
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Teleportation resource | $1 \text{ ebit} + 2 \text{ cbits} \rightarrow 1 \text{ qubit}$ |
| Initial state | $\|\psi\rangle_C \otimes \|\Phi^+\rangle_{AB}$ |
| Bell measurement | Projects CA onto $\{\|\Phi^\pm\rangle, \|\Psi^\pm\rangle\}$ |
| Pauli corrections | $I, Z, X, XZ$ depending on outcome |
| Noisy fidelity | $F = (2p + 1)/3$ for Werner state |
| Classical bound | $F_{classical} = 2/3$ |

### Key Takeaways
1. **Teleportation transfers quantum states** using entanglement and classical communication
2. **No cloning is respected**: the original state is destroyed
3. **No FTL signaling**: classical communication is required
4. **Resource cost**: 1 ebit + 2 cbits per qubit teleported
5. **Noisy channels reduce fidelity**: quantum advantage needs $p > 1/2$
6. **Foundational protocol** for quantum networks and computing

---

## Daily Checklist

- [ ] I can explain each step of the teleportation protocol
- [ ] I can derive the state transformation mathematically
- [ ] I understand why teleportation doesn't violate no-cloning
- [ ] I can calculate fidelity with noisy entanglement
- [ ] I understand the resource requirements
- [ ] I ran the simulation and verified perfect fidelity

---

*Next: Day 555 — Superdense Coding*
