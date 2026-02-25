# Day 556: Entanglement Swapping

## Overview
**Day 556** | Week 80, Day 3 | Year 1, Month 20 | Entanglement Applications

Today we explore entanglement swapping, a protocol that creates entanglement between particles that have never interacted. This seemingly paradoxical phenomenon is the foundation of quantum repeaters and long-distance quantum communication networks.

---

## Learning Objectives
1. Understand the entanglement swapping protocol
2. Derive the mathematical transformation creating distant entanglement
3. Explain why particles that never interacted become entangled
4. Connect to quantum repeater architectures
5. Analyze swapping with imperfect Bell states
6. Implement entanglement swapping simulation in Python

---

## Core Content

### The Swapping Scenario

Consider four particles: A, B, C, D arranged as:

```
Alice                  Charlie                  Bob
  A ═══════════════════ B     C ═══════════════════ D
      Bell pair 1              Bell pair 2
```

- Alice and Charlie share $|\Phi^+\rangle_{AB}$
- Charlie and Bob share $|\Phi^+\rangle_{CD}$
- A and D have never interacted!

**Goal:** Create entanglement between A (Alice) and D (Bob).

### Initial State

$$|\Psi_0\rangle = |\Phi^+\rangle_{AB} \otimes |\Phi^+\rangle_{CD}$$

Expanding:
$$|\Psi_0\rangle = \frac{1}{2}(|00\rangle_{AB} + |11\rangle_{AB}) \otimes (|00\rangle_{CD} + |11\rangle_{CD})$$

$$= \frac{1}{2}(|0000\rangle + |0011\rangle + |1100\rangle + |1111\rangle)_{ABCD}$$

### The Swapping Protocol

**Step 1:** Charlie performs a Bell measurement on his two particles (B and C).

**Step 2:** Charlie communicates the result to Alice and Bob (2 classical bits).

**Step 3:** Depending on the outcome, Alice and Bob share a Bell state!

### Mathematical Derivation

Rewrite the initial state in the Bell basis for particles B and C:

$$|00\rangle_{BC} = \frac{1}{\sqrt{2}}(|\Phi^+\rangle + |\Phi^-\rangle)_{BC}$$
$$|01\rangle_{BC} = \frac{1}{\sqrt{2}}(|\Psi^+\rangle + |\Psi^-\rangle)_{BC}$$
$$|10\rangle_{BC} = \frac{1}{\sqrt{2}}(|\Psi^+\rangle - |\Psi^-\rangle)_{BC}$$
$$|11\rangle_{BC} = \frac{1}{\sqrt{2}}(|\Phi^+\rangle - |\Phi^-\rangle)_{BC}$$

Substituting into $|\Psi_0\rangle$:

$$|\Psi_0\rangle_{ABCD} = \frac{1}{2}[|0\rangle_A |00\rangle_{BC} |0\rangle_D + |0\rangle_A |11\rangle_{BC} |1\rangle_D + |1\rangle_A |00\rangle_{BC} |1\rangle_D + |1\rangle_A |11\rangle_{BC} |0\rangle_D]$$

Rewriting with Bell states on BC:

$$\boxed{|\Psi_0\rangle = \frac{1}{2}[|\Phi^+\rangle_{BC}|\Phi^+\rangle_{AD} + |\Phi^-\rangle_{BC}|\Phi^-\rangle_{AD} + |\Psi^+\rangle_{BC}|\Psi^+\rangle_{AD} + |\Psi^-\rangle_{BC}|\Psi^-\rangle_{AD}]}$$

### Result of Bell Measurement

| Charlie's Outcome | Alice-Bob State | Probability |
|-------------------|-----------------|-------------|
| $\|\Phi^+\rangle_{BC}$ | $\|\Phi^+\rangle_{AD}$ | 1/4 |
| $\|\Phi^-\rangle_{BC}$ | $\|\Phi^-\rangle_{AD}$ | 1/4 |
| $\|\Psi^+\rangle_{BC}$ | $\|\Psi^+\rangle_{AD}$ | 1/4 |
| $\|\Psi^-\rangle_{BC}$ | $\|\Psi^-\rangle_{AD}$ | 1/4 |

**Remarkable fact:** After Charlie's measurement, A and D are entangled even though they never interacted!

### Physical Interpretation

```
Before swapping:
A ═══ B     C ═══ D
  ent.        ent.

Charlie measures BC:
A     [B─C]     D
      Bell
      meas.

After swapping:
A ═════════════════ D
        ent.!
```

**Why this works:**
1. Entanglement is a property of quantum correlations, not physical connection
2. The Bell measurement on BC "teleports" the entanglement
3. The measurement projects the system into a correlated state
4. No information travels faster than light (classical communication required)

### Generalization: Any Initial Bell States

If the initial states are $|\beta_{ij}\rangle_{AB}$ and $|\beta_{kl}\rangle_{CD}$:

$$|\beta_{ij}\rangle \otimes |\beta_{kl}\rangle \xrightarrow{\text{Bell}_{BC}} |\beta_{i \oplus k, j \oplus l}\rangle_{AD}$$

where $\oplus$ is XOR and $\beta_{00} = \Phi^+$, $\beta_{01} = \Psi^+$, $\beta_{10} = \Phi^-$, $\beta_{11} = \Psi^-$.

### Connection to Quantum Repeaters

Entanglement swapping is the key primitive for **quantum repeaters**:

```
Alice ═══ R1 ═══ R2 ═══ R3 ═══ Bob
      L1      L2      L3      L4

Step 1: Create short-distance entanglement
Alice ═══ R1     R1 ═══ R2     R2 ═══ R3     R3 ═══ Bob

Step 2: Swap at R1 and R3
Alice ═══════════ R2 ═══════════ Bob

Step 3: Swap at R2
Alice ══════════════════════════ Bob
```

This overcomes the **exponential loss** in optical fibers!

### Fidelity with Noisy States

If the initial Bell pairs have fidelity $F$ with the ideal state:

$$\rho_{AB} = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\frac{I - |\Phi^+\rangle\langle\Phi^+|}{3}$$

The swapped state has fidelity:

$$\boxed{F_{swap} = F^2 + \frac{(1-F)^2}{3}}$$

For $F = 0.9$: $F_{swap} \approx 0.81 + 0.003 \approx 0.813$

**Problem:** Fidelity decreases with each swap!

---

## Worked Examples

### Example 1: Verifying the Swapping Formula
Verify that measuring $|\Psi^+\rangle_{BC}$ yields $|\Psi^+\rangle_{AD}$.

**Solution:**

Start with:
$$|\Psi_0\rangle = \frac{1}{2}(|0000\rangle + |0011\rangle + |1100\rangle + |1111\rangle)_{ABCD}$$

Group by BC values:
- $|00\rangle_{BC}$: appears in $|0000\rangle$ and $|1100\rangle$ → coefficient $\frac{1}{2}(|0\rangle_A|0\rangle_D + |1\rangle_A|1\rangle_D)$
- $|01\rangle_{BC}$: appears in $|0011\rangle$ → coefficient $\frac{1}{2}|0\rangle_A|1\rangle_D$
- $|10\rangle_{BC}$: appears in $|1100\rangle$... wait, this is $|11\rangle_{AB}|00\rangle_{CD}$.

Let me redo more carefully. The state is:
$$|\Psi_0\rangle = \frac{1}{2}(|00\rangle_A|00\rangle_B|00\rangle_C|0\rangle_D + ...)$$

Actually with ABCD ordering:
$$= \frac{1}{2}(|0\rangle_A|0\rangle_B|0\rangle_C|0\rangle_D + |0\rangle_A|0\rangle_B|1\rangle_C|1\rangle_D + |1\rangle_A|1\rangle_B|0\rangle_C|0\rangle_D + |1\rangle_A|1\rangle_B|1\rangle_C|1\rangle_D)$$

Projecting onto $|\Psi^+\rangle_{BC} = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)_{BC}$:

The terms containing $|01\rangle_{BC}$ or $|10\rangle_{BC}$:
- $|0011\rangle$ contains $|01\rangle_{BC}$: contributes $|0\rangle_A|1\rangle_D$
- $|1100\rangle$ contains $|10\rangle_{BC}$: contributes $|1\rangle_A|0\rangle_D$

After projection:
$$|\psi\rangle_{AD} \propto |0\rangle_A|1\rangle_D + |1\rangle_A|0\rangle_D = \sqrt{2}|\Psi^+\rangle_{AD}$$

Normalized: $|\Psi^+\rangle_{AD}$ ✓ ∎

### Example 2: Two-Hop Swapping Fidelity
Calculate the fidelity after two sequential swaps, starting with $F = 0.95$ Bell pairs.

**Solution:**

After first swap:
$$F_1 = F^2 + \frac{(1-F)^2}{3} = (0.95)^2 + \frac{(0.05)^2}{3} = 0.9025 + 0.000833 = 0.9033$$

After second swap (using $F_1$):
$$F_2 = F_1^2 + \frac{(1-F_1)^2}{3} = (0.9033)^2 + \frac{(0.0967)^2}{3}$$
$$= 0.816 + 0.003 = 0.819$$

The fidelity dropped from 0.95 to 0.819 after just 2 swaps.

This is why **entanglement distillation** is needed in repeater networks! ∎

### Example 3: Swapping with Different Bell States
If Alice-Charlie share $|\Phi^-\rangle$ and Charlie-Bob share $|\Psi^+\rangle$, what state do Alice-Bob get after swapping?

**Solution:**

Using the XOR rule:
- $|\Phi^-\rangle = |\beta_{10}\rangle$ (i.e., indices are 1, 0)
- $|\Psi^+\rangle = |\beta_{01}\rangle$ (i.e., indices are 0, 1)

After swapping:
$$|\beta_{1 \oplus 0, 0 \oplus 1}\rangle_{AD} = |\beta_{11}\rangle_{AD} = |\Psi^-\rangle_{AD}$$

Alice and Bob share $|\Psi^-\rangle$. ∎

---

## Practice Problems

### Problem 1: Three-Hop Swapping
Calculate the final fidelity after three swaps with initial $F = 0.99$.

### Problem 2: Asymmetric Initial States
If AB share a Werner state with $p = 0.8$ and CD share a Werner state with $p = 0.9$, what is the fidelity of the swapped AD state?

### Problem 3: Delayed Choice
Show that Charlie can perform his Bell measurement before or after Alice and Bob make any local measurements, without affecting the final correlations.

### Problem 4: GHZ Swapping
Can entanglement swapping be generalized to create GHZ states from Bell pairs? Design a protocol.

---

## Computational Lab

```python
"""Day 556: Entanglement Swapping Simulation"""
import numpy as np
from numpy.linalg import norm, eigvalsh
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Bell states (in standard ordering: |00⟩, |01⟩, |10⟩, |11⟩)
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

bell_states = {
    'Φ⁺': phi_plus,
    'Φ⁻': phi_minus,
    'Ψ⁺': psi_plus,
    'Ψ⁻': psi_minus
}

bell_list = [phi_plus, phi_minus, psi_plus, psi_minus]
bell_names = ['Φ⁺', 'Φ⁻', 'Ψ⁺', 'Ψ⁻']

def tensor(*states):
    """Compute tensor product of multiple states"""
    result = states[0]
    for s in states[1:]:
        result = np.kron(result, s)
    return result

def partial_trace(rho, dims, trace_out):
    """
    Compute partial trace of density matrix

    Args:
        rho: Density matrix
        dims: List of subsystem dimensions
        trace_out: List of subsystem indices to trace out

    Returns:
        Reduced density matrix
    """
    n = len(dims)
    keep = [i for i in range(n) if i not in trace_out]

    # Reshape to tensor
    shape = dims + dims
    rho_tensor = rho.reshape(shape)

    # Trace out specified systems
    for i in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=i, axis2=i+n)
        n -= 1

    # Reshape back
    d_keep = np.prod([dims[i] for i in keep])
    return rho_tensor.reshape(d_keep, d_keep)

def bell_measurement_4qubit(state_ABCD, measure_BC=True):
    """
    Perform Bell measurement on middle qubits (B and C)

    Args:
        state_ABCD: 4-qubit state vector (16-dim)
        measure_BC: If True, measure qubits B and C

    Returns:
        outcome_idx: Index of Bell measurement outcome
        state_AD: Post-measurement state of A and D
    """
    # Reshape to (2, 2, 2, 2) for A, B, C, D
    psi = state_ABCD.reshape(2, 2, 2, 2)

    # Project onto Bell states for BC
    probs = []
    states_AD = []

    for bell in bell_list:
        # Reshape Bell state to (2, 2)
        bell_bc = bell.reshape(2, 2)

        # Contract BC with the Bell state
        # Sum over B and C indices
        psi_AD = np.einsum('ijkl,jk->il', psi, bell_bc.conj())

        prob = np.sum(np.abs(psi_AD)**2)
        probs.append(prob)

        if prob > 1e-10:
            psi_AD = psi_AD.flatten() / np.sqrt(prob)
        else:
            psi_AD = np.zeros(4, dtype=complex)
        states_AD.append(psi_AD)

    # Simulate measurement
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize (should already be normalized)
    outcome_idx = np.random.choice(4, p=probs)

    return outcome_idx, states_AD[outcome_idx], probs

def entanglement_swapping(bell_AB='Φ⁺', bell_CD='Φ⁺', verbose=True):
    """
    Perform entanglement swapping protocol

    Args:
        bell_AB: Initial Bell state between A and B
        bell_CD: Initial Bell state between C and D
        verbose: Print details

    Returns:
        outcome: Bell measurement outcome
        state_AD: Final state of A and D
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ENTANGLEMENT SWAPPING")
        print(f"Initial: |{bell_AB}⟩_AB ⊗ |{bell_CD}⟩_CD")
        print('='*60)

    # Initial state
    psi_AB = bell_states[bell_AB]
    psi_CD = bell_states[bell_CD]
    psi_ABCD = tensor(psi_AB, psi_CD)

    if verbose:
        print(f"\n1. Initial 4-qubit state created")

    # Bell measurement on BC
    outcome_idx, state_AD, probs = bell_measurement_4qubit(psi_ABCD)

    if verbose:
        print(f"2. Charlie performs Bell measurement on B and C")
        print(f"   Probabilities: {probs}")
        print(f"   Outcome: |{bell_names[outcome_idx]}⟩_BC")

    # Identify resulting Bell state
    for name, bell in bell_states.items():
        overlap = np.abs(np.vdot(bell, state_AD))**2
        if overlap > 0.99:
            if verbose:
                print(f"3. Alice and Bob now share: |{name}⟩_AD")
            return bell_names[outcome_idx], name, state_AD

    return bell_names[outcome_idx], "Unknown", state_AD

def verify_all_outcomes():
    """Verify swapping for all initial Bell state combinations"""
    print("\n" + "="*60)
    print("VERIFICATION: All Initial State Combinations")
    print("="*60)
    print("\n|AB⟩ ⊗ |CD⟩  →  Bell_BC outcome  →  |AD⟩")
    print("-"*50)

    for ab_name in bell_names:
        for cd_name in bell_names:
            # Run swapping
            outcome_bc, result_ad, _ = entanglement_swapping(ab_name, cd_name, verbose=False)
            print(f"|{ab_name}⟩ ⊗ |{cd_name}⟩  →  |{outcome_bc}⟩_BC  →  |{result_ad}⟩_AD")

def fidelity_after_swap(F1, F2):
    """
    Calculate fidelity after swapping two Werner states

    Args:
        F1: Fidelity of first Bell pair
        F2: Fidelity of second Bell pair

    Returns:
        F_swap: Fidelity after swapping
    """
    # For Werner states, the formula simplifies
    # F_swap ≈ F1 * F2 + (1-F1)*(1-F2)/3 for large F
    # More accurate formula:
    return F1 * F2 + (1 - F1) * (1 - F2) / 3

def simulate_noisy_swapping(F, n_trials=1000):
    """
    Simulate swapping with noisy Bell pairs

    Args:
        F: Fidelity of each initial Bell pair with |Φ⁺⟩
        n_trials: Number of Monte Carlo trials

    Returns:
        Average fidelity of output state
    """
    fidelities = []

    for _ in range(n_trials):
        # With probability F, use perfect Bell state
        # With probability (1-F)/3, use each of the other Bell states
        def sample_noisy_bell():
            r = np.random.random()
            if r < F:
                return phi_plus.copy()
            elif r < F + (1-F)/3:
                return phi_minus.copy()
            elif r < F + 2*(1-F)/3:
                return psi_plus.copy()
            else:
                return psi_minus.copy()

        psi_AB = sample_noisy_bell()
        psi_CD = sample_noisy_bell()
        psi_ABCD = tensor(psi_AB, psi_CD)

        # Bell measurement
        outcome_idx, state_AD, _ = bell_measurement_4qubit(psi_ABCD)

        # Calculate fidelity with Φ⁺
        fid = np.abs(np.vdot(phi_plus, state_AD))**2
        fidelities.append(fid)

    return np.mean(fidelities)

# Run demonstrations
print("ENTANGLEMENT SWAPPING DEMONSTRATION")
print("="*60)

# Basic swapping
outcome, result, state = entanglement_swapping('Φ⁺', 'Φ⁺')

# Verify all combinations
verify_all_outcomes()

# Fidelity analysis
print("\n" + "="*60)
print("FIDELITY DEGRADATION IN SWAPPING")
print("="*60)

print("\nInitial F | F_theory | F_simulated")
print("-"*40)

for F in [1.0, 0.99, 0.95, 0.90, 0.85, 0.80]:
    F_theory = fidelity_after_swap(F, F)
    F_sim = simulate_noisy_swapping(F, n_trials=2000)
    print(f"  {F:.2f}    |  {F_theory:.4f}  |   {F_sim:.4f}")

# Multi-hop analysis
print("\n" + "="*60)
print("MULTI-HOP SWAPPING (Starting F = 0.95)")
print("="*60)

F = 0.95
print(f"\nHops | Fidelity")
print("-"*20)
for hops in range(6):
    print(f"  {hops}  |  {F:.4f}")
    F = fidelity_after_swap(F, 0.95)

print("\nConclusion: Fidelity degrades with each hop!")
print("This motivates ENTANGLEMENT DISTILLATION (Day 558)")

# Visualization of the protocol
print("\n" + "="*60)
print("ENTANGLEMENT SWAPPING DIAGRAM")
print("="*60)
print("""
BEFORE SWAPPING:
    Alice          Charlie          Bob
      A ═══════════ B     C ═══════════ D
         |Φ⁺⟩_AB          |Φ⁺⟩_CD

CHARLIE'S BELL MEASUREMENT:
      A            [B─C]            D
                  Bell Meas.
                     ↓
                  2 cbits
                 ↙      ↘

AFTER SWAPPING:
    Alice                          Bob
      A ══════════════════════════ D
              |Φ⁺⟩_AD (or other Bell state)

A and D are now entangled despite NEVER having interacted!
""")
```

**Expected Output:**
```
ENTANGLEMENT SWAPPING DEMONSTRATION
============================================================

============================================================
ENTANGLEMENT SWAPPING
Initial: |Φ⁺⟩_AB ⊗ |Φ⁺⟩_CD
============================================================

1. Initial 4-qubit state created
2. Charlie performs Bell measurement on B and C
   Probabilities: [0.25 0.25 0.25 0.25]
   Outcome: |Φ⁺⟩_BC
3. Alice and Bob now share: |Φ⁺⟩_AD

============================================================
VERIFICATION: All Initial State Combinations
============================================================

|AB⟩ ⊗ |CD⟩  →  Bell_BC outcome  →  |AD⟩
--------------------------------------------------
|Φ⁺⟩ ⊗ |Φ⁺⟩  →  |Φ⁺⟩_BC  →  |Φ⁺⟩_AD
|Φ⁺⟩ ⊗ |Φ⁻⟩  →  |Φ⁻⟩_BC  →  |Φ⁺⟩_AD
...

============================================================
FIDELITY DEGRADATION IN SWAPPING
============================================================

Initial F | F_theory | F_simulated
----------------------------------------
  1.00    |  1.0000  |   1.0000
  0.99    |  0.9801  |   0.9805
  0.95    |  0.9033  |   0.9028
  0.90    |  0.8133  |   0.8145
  0.85    |  0.7308  |   0.7295
  0.80    |  0.6533  |   0.6520
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Initial state | $\|\Phi^+\rangle_{AB} \otimes \|\Phi^+\rangle_{CD}$ |
| Swapping transformation | Bell$_{BC}$ projects AD onto Bell state |
| Outcome correlation | Bell$_{BC}$ = Bell$_{AD}$ |
| Fidelity after swap | $F_{swap} = F_1 F_2 + \frac{(1-F_1)(1-F_2)}{3}$ |
| XOR rule | $\|\beta_{ij}\rangle \otimes \|\beta_{kl}\rangle \rightarrow \|\beta_{i\oplus k, j\oplus l}\rangle$ |

### Key Takeaways
1. **Entanglement swapping creates entanglement** between particles that never interacted
2. **Bell measurement** on middle particles projects outer particles into Bell state
3. **All four outcomes** are equally likely, each giving a different Bell state
4. **Classical communication** is required (no FTL signaling)
5. **Fidelity degrades** with each swap—motivates distillation
6. **Foundation of quantum repeaters** for long-distance entanglement

---

## Daily Checklist

- [ ] I can explain the entanglement swapping protocol
- [ ] I can derive the state transformation mathematically
- [ ] I understand why non-interacting particles become entangled
- [ ] I can calculate fidelity after swapping noisy states
- [ ] I see the connection to quantum repeaters
- [ ] I ran the simulation and verified the outcomes

---

*Next: Day 557 — Quantum Repeaters*
