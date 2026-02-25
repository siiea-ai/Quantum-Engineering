# Day 822: Split Operations & Logical State Preparation

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Split operation theory, boundary separation, state disentanglement |
| **Afternoon** | 2.5 hours | State preparation protocols, practice problems |
| **Evening** | 1.5 hours | Computational lab: Split simulation and state initialization |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 822, you will be able to:

1. **Define split operations** as the inverse of merge operations
2. **Analyze the measurement** that occurs during split and its effect on states
3. **Prepare logical basis states** $|0_L\rangle$, $|1_L\rangle$, $|+_L\rangle$, $|-_L\rangle$ fault-tolerantly
4. **Design state injection protocols** for arbitrary logical states
5. **Calculate the resource overhead** for state preparation via lattice surgery
6. **Implement split operations** in simulation

---

## 1. Introduction: The Split Operation

### Reversing the Merge

The **split operation** is conceptually the reverse of merge: it takes a single large surface code patch and separates it into two independent patches.

**Key distinction from merge:**
- Merge: Measures joint observable, creates correlation
- Split: Measures individual observables, breaks correlation

$$\boxed{\text{Split} = \text{Boundary separation} + \text{Individual stabilizer measurement} \rightarrow \text{State projection}}$$

### Why Split Matters

Split operations are essential for:
1. **Completing logical gates** (merge-and-split protocols)
2. **Preparing logical states** from scratch
3. **Teleportation-based computation** (consuming ancilla patches)
4. **Resource state distribution** across the processor

---

## 2. Split Operation Theory

### Setup: Merged Patch

Consider a merged patch encoding one logical qubit, formed from regions A and B:

```
┌─────────────────────────────────────────────┐
│                                             │
│   Region A ──── Merge Zone ──── Region B    │
│                                             │
└─────────────────────────────────────────────┘
         Single logical qubit encoded
```

The logical operators span the full patch:
- $\bar{Z}$: Path through A, merge zone, and B
- $\bar{X}$: Path through A, merge zone, and B (perpendicular)

### Split Procedure

**Step 1: Identify split boundary**

Choose where to cut the merged patch, typically through the middle of the merge zone.

**Step 2: Measure split stabilizers**

Introduce new stabilizers that "cut" the logical operators:

For **ZZ split** (cutting Z logical operator):
$$S_Z^{\text{cut}} = Z_{\text{left}} Z_{\text{right}}$$

For **XX split** (cutting X logical operator):
$$S_X^{\text{cut}} = X_{\text{left}} X_{\text{right}}$$

**Step 3: Remove merge zone qubits**

Measure out the qubits in the merge zone in the appropriate basis.

**Step 4: Stabilize independent patches**

Each patch now has its own complete stabilizer group.

### Mathematical Effect of Split

**ZZ Split:** Measures $Z_A \otimes Z_B$ on the merged patch

If merged state is $|\Psi_{\text{merged}}\rangle$:

$$|\Psi_{\text{merged}}\rangle \xrightarrow{\text{ZZ split}} |\psi_A\rangle \otimes |\psi_B\rangle$$

where the product state depends on:
1. The original merged state
2. The split measurement outcome

$$\boxed{\text{ZZ split converts } |\Psi_{AB}\rangle \text{ to product state via } Z_AZ_B \text{ measurement}}$$

---

## 3. State Projection During Split

### Initial Merged State

Consider a general merged state in the computational basis:
$$|\Psi_{AB}\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle + \delta|11\rangle$$

### ZZ Split Outcomes

Measuring $Z_A Z_B$:

**Outcome +1 (eigenspace: $|00\rangle$, $|11\rangle$):**
$$|\Psi_+\rangle \propto \alpha|00\rangle + \delta|11\rangle$$

This is generally **entangled**! The split doesn't complete until we project further.

**Additional measurement needed:**
To get a product state, we must also measure one qubit individually.

### Complete Split Protocol

**Full ZZ split:**
1. Measure $Z_A Z_B$ → outcome $m_1 \in \{+1, -1\}$
2. Measure $Z_A$ (or $Z_B$) → outcome $m_2 \in \{+1, -1\}$

This fully determines both $Z_A$ and $Z_B$:
- $Z_A = m_2$
- $Z_B = m_1 \cdot m_2$

$$\boxed{\text{Complete split: } Z_AZ_B \text{ measurement} + Z_A \text{ measurement} \rightarrow \text{product state}}$$

---

## 4. Logical State Preparation

### Preparing $|0_L\rangle$

**Protocol 1: Direct initialization**

1. Initialize all data qubits in $|0\rangle$
2. Measure all stabilizers
3. Apply corrections based on syndrome

This works but requires careful handling of measurement errors.

**Protocol 2: Split from merged patch**

1. Start with merged patch in known state
2. Perform ZZ split with outcome tracking
3. One patch contains $|0_L\rangle$ or $|1_L\rangle$
4. Apply $X_L$ correction if needed

### Preparing $|+_L\rangle$

**Protocol 1: Direct initialization**

1. Initialize all data qubits in $|+\rangle$
2. Measure all X-type stabilizers (trivially satisfied)
3. Measure Z-type stabilizers (random outcomes)
4. Apply X corrections based on Z syndrome

**Protocol 2: XX Split**

1. Start with merged patch
2. Perform XX split
3. Outcomes determine $|+_L\rangle$ vs $|-_L\rangle$
4. Apply $Z_L$ correction if needed

$$\boxed{|+_L\rangle: \text{ Initialize in } |+\rangle^{\otimes n}, \text{ correct based on Z syndrome}}$$

### Preparing Arbitrary States

For arbitrary $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

**Gate teleportation protocol:**

1. Prepare ancilla in $|+_L\rangle$
2. Prepare resource state $|\psi\rangle$ (possibly noisy)
3. Use merge operations to teleport state into code
4. Apply Pauli corrections

$$\boxed{\text{State injection: } |\psi\rangle_{\text{noisy}} \xrightarrow{\text{teleport}} |\psi\rangle_L}$$

---

## 5. Fault-Tolerant State Preparation

### Why Standard Preparation Fails

**Problem:** Initializing qubits one-by-one and measuring stabilizers can spread errors.

Example: An X error on one qubit during preparation can propagate to logical error before stabilizers "lock in" the state.

### Post-Selection Method

**Protocol:**
1. Prepare state using standard method
2. Measure all stabilizers multiple times
3. **Reject** if any syndrome indicates error
4. Accept only if all syndromes are trivial

**Success probability:**
$$P_{\text{success}} \approx (1-p)^{n \cdot r}$$

where $n$ = number of qubits, $r$ = rounds, $p$ = error rate.

$$\boxed{\text{Post-selection: Accept if } d \text{ rounds show no errors}}$$

### State Distillation

For non-Clifford states (like $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$):

1. Prepare many noisy copies of $|T\rangle$
2. Apply distillation circuit (uses Clifford gates)
3. Measure some copies to detect errors
4. Output fewer, higher-fidelity copies

**Distillation overhead:**
$$n_{\text{input}} \sim O\left(\log^{\gamma}\left(\frac{1}{\epsilon_{\text{out}}}\right)\right)$$

where $\gamma \approx 1-2$ depending on protocol.

---

## 6. Worked Examples

### Example 1: ZZ Split of Bell State

**Problem:** A merged patch is in the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$. Perform a ZZ split and determine all possible outcomes.

**Solution:**

The Bell state is already in the $Z_AZ_B = +1$ eigenspace:
$$Z_AZ_B|\Phi^+\rangle = +1 \cdot |\Phi^+\rangle$$

**Step 1: ZZ measurement**
Outcome: $m_1 = +1$ (deterministic)
State after: Still $|\Phi^+\rangle$ (unchanged)

**Step 2: Measure $Z_A$**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|0\rangle_A|0\rangle_B + |1\rangle_A|1\rangle_B)$$

- Outcome $m_2 = +1$ (probability 1/2): State → $|0\rangle_A|0\rangle_B$
- Outcome $m_2 = -1$ (probability 1/2): State → $|1\rangle_A|1\rangle_B$

**Final states:**
$$\boxed{|\Phi^+\rangle \xrightarrow{\text{ZZ split}} \begin{cases} |0_L\rangle_A \otimes |0_L\rangle_B & P = 1/2 \\ |1_L\rangle_A \otimes |1_L\rangle_B & P = 1/2 \end{cases}}$$

---

### Example 2: Preparing $|0_L\rangle$ via Split

**Problem:** Design a protocol to prepare $|0_L\rangle$ starting from two patches in $|+_L\rangle$.

**Solution:**

**Step 1:** Start with patches A and B in $|+_L\rangle$:
$$|+_L\rangle_A \otimes |+_L\rangle_B = \frac{1}{2}(|0\rangle + |1\rangle)_A(|0\rangle + |1\rangle)_B$$

**Step 2:** ZZ merge
$$|+_L,+_L\rangle \xrightarrow{ZZ} \text{random outcome}$$

For $|+_L,+_L\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$:
- $P(Z_AZ_B = +1) = 1/2$ → projects to $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
- $P(Z_AZ_B = -1) = 1/2$ → projects to $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$

**Step 3:** ZZ split
After merge, perform split:
- Measure $Z_A$ to get definite computational basis states

**Step 4:** Apply corrections
If $Z_A = -1$, apply $X_A$ to get $|0_L\rangle$.

$$\boxed{|+_L\rangle \otimes |+_L\rangle \xrightarrow{ZZ\text{ merge}} \xrightarrow{ZZ\text{ split}} |0_L\rangle \otimes |\pm_L\rangle}$$

---

### Example 3: Resource Counting for State Preparation

**Problem:** Calculate the space-time cost of preparing $|0_L\rangle$ using the split method for distance $d=7$.

**Solution:**

**Space (qubits):**
- Patch A: $\sim 2d^2 = 2(49) = 98$ qubits
- Patch B: $\sim 2d^2 = 98$ qubits
- Merge zone: $\sim d = 7$ qubits
- Total: $\sim 203$ qubits

**Time (cycles):**
- Initial $|+_L\rangle$ prep: $d = 7$ cycles (stabilizer measurement)
- ZZ merge: $d = 7$ cycles
- ZZ split: $d = 7$ cycles
- Total: $\sim 21$ cycles

**Space-time volume:**
$$V = (\text{qubits}) \times (\text{cycles}) \approx 200 \times 21 = 4200 \text{ qubit-cycles}$$

$$\boxed{V_{|0_L\rangle} \approx O(d^3) \text{ qubit-cycles}}$$

---

## 7. Practice Problems

### Problem Set A: Direct Application

**A1.** A merged patch is in state $|01\rangle + |10\rangle)/\sqrt{2}$. What is the ZZ split $Z_AZ_B$ measurement outcome?

**A2.** Starting from $|0_L\rangle$, how do you prepare $|1_L\rangle$ using logical gates?

**A3.** What is the initial state of merge-zone qubits for an XX split? For a ZZ split?

---

### Problem Set B: Intermediate

**B1.** Design a protocol to prepare the Bell state $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ using merge and split operations.

**B2.** Calculate the logical error rate for state preparation using post-selection with:
- Physical error rate $p = 10^{-3}$
- Distance $d = 5$
- Post-selection over $r = 5$ rounds

**B3.** Show that XX split on the state $|0_L\rangle$ produces random $|+_L\rangle$ or $|-_L\rangle$ outcomes.

---

### Problem Set C: Challenging

**C1.** Prove that split operations are "self-inverse": performing merge followed immediately by split (on the same boundary type) returns independent patches in product states.

**C2.** Design a fault-tolerant protocol to prepare an arbitrary state $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$ at the logical level. What is the dominant source of infidelity?

**C3.** Compare the resource efficiency (qubit-cycles per logical qubit prepared) of:
(a) Direct initialization with post-selection
(b) Merge-split protocol
(c) Teleportation from ancilla factory

---

## 8. Computational Lab: Split Operations and State Preparation

```python
"""
Day 822 Computational Lab: Split Operations & State Preparation
Simulating split operations and logical state initialization protocols

This lab implements split operations and demonstrates various
state preparation techniques using lattice surgery primitives.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Pauli matrices and basis states
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)


def tensor(*args):
    """Compute tensor product of multiple matrices/vectors."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def outer(v1, v2=None):
    """Compute outer product |v1><v2|."""
    if v2 is None:
        v2 = v1
    return v1 @ v2.conj().T


def partial_trace(rho, keep_subsystem, dims):
    """
    Compute partial trace of density matrix.

    Parameters:
    -----------
    rho : ndarray
        Density matrix
    keep_subsystem : int
        Which subsystem to keep (0 or 1 for 2-qubit system)
    dims : list
        Dimensions of subsystems [d1, d2]
    """
    d1, d2 = dims
    if keep_subsystem == 0:
        # Trace out second subsystem
        rho_reduced = np.zeros((d1, d1), dtype=complex)
        for i in range(d2):
            rho_reduced += rho[i::d2, i::d2]
        return rho_reduced / np.trace(rho_reduced)
    else:
        # Trace out first subsystem
        rho_reshaped = rho.reshape(d1, d2, d1, d2)
        rho_reduced = np.trace(rho_reshaped, axis1=0, axis2=2)
        return rho_reduced / np.trace(rho_reduced)


def measure_observable(state, observable, return_all=False):
    """
    Perform projective measurement of an observable.

    Returns outcome, post-measurement state, and probability.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(observable)
    eigenvalues = np.round(eigenvalues).astype(int)

    P_plus = np.zeros_like(observable, dtype=complex)
    P_minus = np.zeros_like(observable, dtype=complex)

    for i, ev in enumerate(eigenvalues):
        v = eigenvectors[:, i:i+1]
        if ev == 1:
            P_plus += outer(v)
        else:
            P_minus += outer(v)

    if state.shape[1] == 1:
        rho = outer(state)
    else:
        rho = state

    p_plus = np.real(np.trace(P_plus @ rho))
    p_minus = np.real(np.trace(P_minus @ rho))

    if return_all:
        return [(+1, P_plus @ state / np.linalg.norm(P_plus @ state) if p_plus > 1e-10 else None, p_plus),
                (-1, P_minus @ state / np.linalg.norm(P_minus @ state) if p_minus > 1e-10 else None, p_minus)]

    outcome = np.random.choice([1, -1], p=[p_plus, p_minus])

    if outcome == 1:
        post_state = P_plus @ state
    else:
        post_state = P_minus @ state

    post_state = post_state / np.linalg.norm(post_state)
    prob = p_plus if outcome == 1 else p_minus

    return outcome, post_state, prob


def zz_split(merged_state, verbose=True):
    """
    Perform ZZ split on a two-qubit merged state.

    This measures Z_A ⊗ Z_B, then measures Z_A to fully determine
    the product state.

    Parameters:
    -----------
    merged_state : ndarray
        Two-qubit state vector
    verbose : bool
        Print detailed output

    Returns:
    --------
    outcomes : tuple
        (m_ZZ, m_ZA) measurement outcomes
    state_A, state_B : ndarray
        Resulting single-qubit states
    """
    if verbose:
        print("\n" + "="*50)
        print("ZZ SPLIT OPERATION")
        print("="*50)
        print(f"\nInput merged state: {merged_state.flatten()}")

    # Step 1: Measure ZZ
    ZZ = tensor(Z, Z)
    m_zz, post_zz, p_zz = measure_observable(merged_state, ZZ)

    if verbose:
        print(f"\nStep 1: Measure Z_A Z_B")
        print(f"  Outcome: {'+1' if m_zz == 1 else '-1'}")
        print(f"  Probability: {p_zz:.4f}")
        print(f"  State after ZZ: {post_zz.flatten()}")

    # Step 2: Measure Z_A (to fully split)
    ZI = tensor(Z, I)
    m_za, post_za, p_za = measure_observable(post_zz, ZI)

    if verbose:
        print(f"\nStep 2: Measure Z_A")
        print(f"  Outcome: {'+1' if m_za == 1 else '-1'}")
        print(f"  Probability: {p_za:.4f}")

    # Determine individual Z values
    z_a = m_za
    z_b = m_zz * m_za  # Since ZZ = Z_A * Z_B

    # The final state should be a product state
    # |z_a, z_b> where z_a, z_b ∈ {0, 1}
    state_A = ket_0 if z_a == 1 else ket_1
    state_B = ket_0 if z_b == 1 else ket_1

    if verbose:
        print(f"\nFinal product state:")
        print(f"  |ψ_A⟩ = |{'0' if z_a == 1 else '1'}⟩")
        print(f"  |ψ_B⟩ = |{'0' if z_b == 1 else '1'}⟩")

    return (m_zz, m_za), state_A, state_B


def xx_split(merged_state, verbose=True):
    """
    Perform XX split on a two-qubit merged state.

    Measures X_A ⊗ X_B, then X_A to fully determine product state
    in the X basis.

    Parameters:
    -----------
    merged_state : ndarray
        Two-qubit state vector
    verbose : bool
        Print detailed output

    Returns:
    --------
    outcomes : tuple
        (m_XX, m_XA) measurement outcomes
    state_A, state_B : ndarray
        Resulting single-qubit states (in X basis)
    """
    if verbose:
        print("\n" + "="*50)
        print("XX SPLIT OPERATION")
        print("="*50)
        print(f"\nInput merged state: {merged_state.flatten()}")

    # Step 1: Measure XX
    XX = tensor(X, X)
    m_xx, post_xx, p_xx = measure_observable(merged_state, XX)

    if verbose:
        print(f"\nStep 1: Measure X_A X_B")
        print(f"  Outcome: {'+1' if m_xx == 1 else '-1'}")
        print(f"  Probability: {p_xx:.4f}")

    # Step 2: Measure X_A
    XI = tensor(X, I)
    m_xa, post_xa, p_xa = measure_observable(post_xx, XI)

    if verbose:
        print(f"\nStep 2: Measure X_A")
        print(f"  Outcome: {'+1' if m_xa == 1 else '-1'}")
        print(f"  Probability: {p_xa:.4f}")

    # Determine X eigenvalues
    x_a = m_xa
    x_b = m_xx * m_xa

    state_A = ket_plus if x_a == 1 else ket_minus
    state_B = ket_plus if x_b == 1 else ket_minus

    if verbose:
        print(f"\nFinal product state:")
        print(f"  |ψ_A⟩ = |{'+' if x_a == 1 else '-'}⟩")
        print(f"  |ψ_B⟩ = |{'+' if x_b == 1 else '-'}⟩")

    return (m_xx, m_xa), state_A, state_B


def prepare_logical_zero(method='split', verbose=True):
    """
    Prepare |0_L⟩ using different methods.

    Parameters:
    -----------
    method : str
        'split' - Use merge-split protocol
        'direct' - Direct initialization

    Returns:
    --------
    state : ndarray
        Prepared logical state
    resources : dict
        Resource usage (cycles, qubits, etc.)
    """
    if verbose:
        print("\n" + "="*50)
        print(f"PREPARING |0_L⟩ VIA {method.upper()} METHOD")
        print("="*50)

    if method == 'split':
        # Start with two patches in |+⟩
        patch_A = ket_plus.copy()
        patch_B = ket_plus.copy()

        if verbose:
            print("\nStep 1: Initialize patches in |+_L⟩")
            print(f"  Patch A: |+⟩")
            print(f"  Patch B: |+⟩")

        # Merge (ZZ)
        initial = tensor(patch_A, patch_B)
        ZZ = tensor(Z, Z)
        m_zz, merged, _ = measure_observable(initial, ZZ)

        if verbose:
            print(f"\nStep 2: ZZ Merge")
            print(f"  Outcome: {'+1' if m_zz == 1 else '-1'}")

        # Split (ZZ)
        (m_zz2, m_za), state_A, state_B = zz_split(merged, verbose=verbose)

        # Apply correction if needed
        if m_za == -1:
            state_A = X @ state_A
            if verbose:
                print(f"\nStep 4: Apply X correction (m_ZA = -1)")
                print(f"  Final state: |0_L⟩")

        resources = {'method': 'split', 'cycles': 3, 'ancilla_patches': 1}
        return state_A, resources

    else:  # direct
        # Simply return |0⟩
        if verbose:
            print("\nDirect initialization in |0⟩")
        return ket_0.copy(), {'method': 'direct', 'cycles': 1, 'ancilla_patches': 0}


def prepare_logical_plus(method='split', verbose=True):
    """
    Prepare |+_L⟩ using different methods.
    """
    if verbose:
        print("\n" + "="*50)
        print(f"PREPARING |+_L⟩ VIA {method.upper()} METHOD")
        print("="*50)

    if method == 'split':
        # Start with two patches in |0⟩
        patch_A = ket_0.copy()
        patch_B = ket_0.copy()

        if verbose:
            print("\nStep 1: Initialize patches in |0_L⟩")

        # Merge (XX)
        initial = tensor(patch_A, patch_B)
        XX = tensor(X, X)
        m_xx, merged, _ = measure_observable(initial, XX)

        if verbose:
            print(f"\nStep 2: XX Merge")
            print(f"  Outcome: {'+1' if m_xx == 1 else '-1'}")

        # Split (XX)
        (m_xx2, m_xa), state_A, state_B = xx_split(merged, verbose=verbose)

        # Apply correction if needed
        if m_xa == -1:
            state_A = Z @ state_A
            if verbose:
                print(f"\nApply Z correction")

        resources = {'method': 'split', 'cycles': 3, 'ancilla_patches': 1}
        return state_A, resources

    else:
        return ket_plus.copy(), {'method': 'direct', 'cycles': 1, 'ancilla_patches': 0}


def demonstrate_split_statistics():
    """
    Analyze split operation statistics for various input states.
    """
    print("\n" + "="*60)
    print("SPLIT OPERATION STATISTICS")
    print("="*60)

    # Test states
    bell_plus = tensor(ket_0, ket_0) + tensor(ket_1, ket_1)
    bell_plus = bell_plus / np.linalg.norm(bell_plus)

    bell_minus = tensor(ket_0, ket_1) + tensor(ket_1, ket_0)
    bell_minus = bell_minus / np.linalg.norm(bell_minus)

    product_state = tensor(ket_plus, ket_0)

    test_cases = [
        (bell_plus, "Bell |Φ+⟩ = (|00⟩+|11⟩)/√2"),
        (bell_minus, "Bell |Ψ+⟩ = (|01⟩+|10⟩)/√2"),
        (product_state, "Product |+⟩⊗|0⟩")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (state, name) in enumerate(test_cases):
        outcomes_zz = []
        outcomes_za = []

        for _ in range(1000):
            (m_zz, m_za), _, _ = zz_split(state, verbose=False)
            outcomes_zz.append(m_zz)
            outcomes_za.append(m_za)

        outcomes_zz = np.array(outcomes_zz)
        outcomes_za = np.array(outcomes_za)

        # Plot outcome distribution
        ax = axes[idx]
        categories = ['ZZ=+1,ZA=+1', 'ZZ=+1,ZA=-1', 'ZZ=-1,ZA=+1', 'ZZ=-1,ZA=-1']
        counts = [
            np.sum((outcomes_zz == 1) & (outcomes_za == 1)),
            np.sum((outcomes_zz == 1) & (outcomes_za == -1)),
            np.sum((outcomes_zz == -1) & (outcomes_za == 1)),
            np.sum((outcomes_zz == -1) & (outcomes_za == -1))
        ]
        probs = np.array(counts) / 1000

        bars = ax.bar(range(4), probs, color=['blue', 'lightblue', 'red', 'salmon'])
        ax.set_xticks(range(4))
        ax.set_xticklabels(['|00⟩', '|11⟩', '|01⟩', '|10⟩'], fontsize=10)
        ax.set_ylabel('Probability')
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1.1)

        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.annotate(f'{prob:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, prob),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.suptitle('ZZ Split Outcome Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig('split_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to 'split_statistics.png'")


def visualize_state_preparation_protocol():
    """
    Visualize the merge-split state preparation protocol.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Protocol steps (conceptual)
    steps = [
        ("Initial: |+⟩ ⊗ |+⟩", "Two separate patches\nin |+_L⟩ state"),
        ("ZZ Merge", "Patches connected\nvia rough boundaries"),
        ("Merged State", "Joint ZZ measured\n→ entangled state"),
        ("ZZ Split", "Separate patches\nvia ZZ measurement"),
        ("Measure Z_A", "Determine individual\nZ eigenvalue"),
        ("Correction", "Apply X if Z_A = -1"),
        ("Final: |0⟩ ⊗ |?⟩", "Patch A in |0_L⟩\nPatch B discarded"),
        ("Success!", "|0_L⟩ prepared\nfault-tolerantly")
    ]

    for idx, (title, description) in enumerate(steps):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        # Draw conceptual diagram
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title and description
        ax.text(5, 9, title, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(5, 1, description, ha='center', va='bottom', fontsize=10,
               style='italic', wrap=True)

        # Draw patches based on step
        if idx == 0:  # Initial
            ax.add_patch(plt.Rectangle((1, 4), 2, 3, facecolor='lightblue', edgecolor='blue'))
            ax.add_patch(plt.Rectangle((7, 4), 2, 3, facecolor='lightblue', edgecolor='blue'))
            ax.text(2, 5.5, '|+⟩', ha='center', va='center', fontsize=14)
            ax.text(8, 5.5, '|+⟩', ha='center', va='center', fontsize=14)
        elif idx == 1:  # Merge
            ax.add_patch(plt.Rectangle((1, 4), 2, 3, facecolor='lightblue', edgecolor='blue'))
            ax.add_patch(plt.Rectangle((7, 4), 2, 3, facecolor='lightblue', edgecolor='blue'))
            ax.annotate('', xy=(6.8, 5.5), xytext=(3.2, 5.5),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(5, 5.5, 'ZZ', ha='center', va='center', fontsize=12, color='red')
        elif idx == 2:  # Merged
            ax.add_patch(plt.Rectangle((1, 4), 8, 3, facecolor='lightgreen', edgecolor='green'))
            ax.text(5, 5.5, 'Merged Patch', ha='center', va='center', fontsize=12)
        elif idx == 3:  # Split
            ax.add_patch(plt.Rectangle((1, 4), 3, 3, facecolor='lightyellow', edgecolor='orange'))
            ax.add_patch(plt.Rectangle((6, 4), 3, 3, facecolor='lightyellow', edgecolor='orange'))
            ax.plot([4.5, 5.5], [4, 7], 'r--', lw=2)
            ax.text(5, 5.5, '✂', ha='center', va='center', fontsize=20)
        elif idx == 4:  # Measure
            ax.add_patch(plt.Rectangle((1, 4), 3, 3, facecolor='lightcoral', edgecolor='red'))
            ax.add_patch(plt.Rectangle((6, 4), 3, 3, facecolor='lightgray', edgecolor='gray'))
            ax.text(2.5, 5.5, 'Z?', ha='center', va='center', fontsize=14)
        elif idx == 5:  # Correction
            ax.add_patch(plt.Rectangle((1, 4), 3, 3, facecolor='lightblue', edgecolor='blue'))
            ax.text(2.5, 5.5, 'X?', ha='center', va='center', fontsize=14)
            ax.text(2.5, 3.5, 'if needed', ha='center', fontsize=10)
        elif idx == 6:  # Final
            ax.add_patch(plt.Rectangle((1, 4), 3, 3, facecolor='lightgreen', edgecolor='green'))
            ax.text(2.5, 5.5, '|0⟩', ha='center', va='center', fontsize=14)
        elif idx == 7:  # Success
            ax.add_patch(plt.Circle((5, 5.5), 2, facecolor='gold', edgecolor='orange'))
            ax.text(5, 5.5, '|0_L⟩', ha='center', va='center', fontsize=16, fontweight='bold')

    plt.suptitle('Fault-Tolerant |0_L⟩ Preparation via Merge-Split Protocol', fontsize=14)
    plt.tight_layout()
    plt.savefig('state_preparation_protocol.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nProtocol visualization saved to 'state_preparation_protocol.png'")


def resource_comparison():
    """
    Compare resource requirements for different state preparation methods.
    """
    print("\n" + "="*60)
    print("RESOURCE COMPARISON: STATE PREPARATION METHODS")
    print("="*60)

    distances = [3, 5, 7, 9, 11, 13, 15]

    # Resource scaling (simplified models)
    # Direct: O(d^2) qubits, O(d) time
    # Split: O(d^2) qubits * 2 patches, O(3d) time
    # Teleport: O(d^2) qubits * 3 patches, O(2d) time

    direct_qubits = [d**2 for d in distances]
    direct_time = [d for d in distances]
    direct_volume = [q*t for q, t in zip(direct_qubits, direct_time)]

    split_qubits = [2 * d**2 + d for d in distances]  # 2 patches + merge zone
    split_time = [3 * d for d in distances]  # merge + split
    split_volume = [q*t for q, t in zip(split_qubits, split_time)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Qubit count
    axes[0].plot(distances, direct_qubits, 'bo-', label='Direct', markersize=8)
    axes[0].plot(distances, split_qubits, 'rs-', label='Merge-Split', markersize=8)
    axes[0].set_xlabel('Code Distance')
    axes[0].set_ylabel('Qubits')
    axes[0].set_title('Qubit Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Time
    axes[1].plot(distances, direct_time, 'bo-', label='Direct', markersize=8)
    axes[1].plot(distances, split_time, 'rs-', label='Merge-Split', markersize=8)
    axes[1].set_xlabel('Code Distance')
    axes[1].set_ylabel('Cycles')
    axes[1].set_title('Time (Stabilizer Cycles)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Space-time volume
    axes[2].plot(distances, direct_volume, 'bo-', label='Direct', markersize=8)
    axes[2].plot(distances, split_volume, 'rs-', label='Merge-Split', markersize=8)
    axes[2].set_xlabel('Code Distance')
    axes[2].set_ylabel('Qubit-Cycles')
    axes[2].set_title('Space-Time Volume')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('State Preparation Resource Scaling', fontsize=14)
    plt.tight_layout()
    plt.savefig('state_prep_resources.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResource comparison saved to 'state_prep_resources.png'")


def main():
    """Run all Day 822 demonstrations."""
    print("Day 822: Split Operations & Logical State Preparation")
    print("="*60)

    # Basic ZZ split demonstration
    print("\n--- Example 1: ZZ Split of Bell State ---")
    bell = (tensor(ket_0, ket_0) + tensor(ket_1, ket_1)) / np.sqrt(2)
    zz_split(bell)

    # XX split demonstration
    print("\n--- Example 2: XX Split of Product State ---")
    product = tensor(ket_plus, ket_0)
    xx_split(product)

    # State preparation
    print("\n--- Example 3: Prepare |0_L⟩ via Split Method ---")
    state, resources = prepare_logical_zero(method='split')
    print(f"\nResources used: {resources}")

    print("\n--- Example 4: Prepare |+_L⟩ via Split Method ---")
    state, resources = prepare_logical_plus(method='split')
    print(f"\nResources used: {resources}")

    # Statistics
    demonstrate_split_statistics()

    # Protocol visualization
    visualize_state_preparation_protocol()

    # Resource comparison
    resource_comparison()

    print("\n" + "="*60)
    print("Day 822 Computational Lab Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 9. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| ZZ split measurement | $M_{ZZ} = Z_A \otimes Z_B$ followed by $M_{Z_A}$ |
| XX split measurement | $M_{XX} = X_A \otimes X_B$ followed by $M_{X_A}$ |
| Product state after ZZ split | $\|z_A, z_B\rangle$ where $z_i \in \{0,1\}$ |
| Post-selection success | $P_{\text{success}} \approx (1-p)^{n \cdot r}$ |
| State prep space-time volume | $V \sim O(d^3)$ qubit-cycles |
| Correction for $|0_L\rangle$ | $X_L^{(1-m_Z)/2}$ |
| Correction for $\|+_L\rangle$ | $Z_L^{(1-m_X)/2}$ |

### Key Takeaways

1. **Split operations** reverse merges by measuring joint then individual observables
2. **State preparation** uses merge-split sequences with Pauli corrections
3. **Fault-tolerant initialization** requires $O(d)$ rounds of syndrome measurement
4. **Post-selection** improves fidelity by rejecting states with detected errors
5. **Resource overhead** for state prep scales as $O(d^3)$ qubit-cycles
6. **Magic states** require additional distillation due to non-Clifford nature

---

## 10. Daily Checklist

- [ ] I understand the difference between merge and split operations
- [ ] I can trace through the ZZ and XX split measurement sequences
- [ ] I know how to prepare $|0_L\rangle$ and $|+_L\rangle$ fault-tolerantly
- [ ] I can calculate Pauli corrections based on split measurement outcomes
- [ ] I understand why post-selection improves state preparation fidelity
- [ ] I completed the computational lab and visualized split statistics

---

## 11. Preview: Day 823

Tomorrow we combine merge and split operations to build the **Surface Code CNOT gate**:

- The canonical ZZ-then-XX lattice surgery CNOT
- Measurement outcome tracking and Pauli frame updates
- Comparison with transversal and braiding approaches
- Resource analysis for multi-CNOT circuits

This is the core two-qubit gate that, combined with single-qubit Cliffords and T gates, enables universal quantum computation.

---

*"State preparation is where abstract stabilizer theory meets physical reality - every logical qubit begins its journey through a carefully orchestrated dance of measurements."*
