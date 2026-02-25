# Day 821: Merge Operations (XX and ZZ)

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Merge operation theory, joint measurements, boundary fusion |
| **Afternoon** | 2.5 hours | Merge protocol problems, measurement outcome analysis |
| **Evening** | 1.5 hours | Computational lab: Simulating merge operations |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 821, you will be able to:

1. **Define XX and ZZ merge operations** mathematically and operationally
2. **Describe the boundary fusion process** that connects two surface code patches
3. **Analyze measurement outcomes** and their effect on logical states
4. **Calculate Pauli corrections** required after merge operations
5. **Verify fault-tolerance** of the merge protocol through syndrome analysis
6. **Simulate merge operations** on logical qubit states

---

## 1. Introduction: Merging Surface Code Patches

### The Merge Concept

A **merge operation** in lattice surgery combines two separate surface code patches into a single larger patch, creating a **joint logical operator** that spans both original patches.

**Key insight:** Measuring stabilizers across the boundary between patches performs a **logical joint measurement** on the encoded qubits.

$$\boxed{\text{Merge} = \text{Boundary fusion} + \text{Joint stabilizer measurement} \rightarrow \text{Logical parity measurement}}$$

### Types of Merge Operations

**XX Merge (Smooth boundary merge):**
- Connect patches along smooth (X-type) boundaries
- Creates joint logical XX operator
- Measures logical $X_A X_B$ eigenvalue

**ZZ Merge (Rough boundary merge):**
- Connect patches along rough (Z-type) boundaries
- Creates joint logical ZZ operator
- Measures logical $Z_A Z_B$ eigenvalue

---

## 2. Mathematical Framework for Merge Operations

### Initial State

Consider two surface code patches A and B, each encoding one logical qubit:

$$|\Psi\rangle = |\psi_A\rangle \otimes |\psi_B\rangle$$

where $|\psi_A\rangle = \alpha_A|0_L\rangle + \beta_A|1_L\rangle$ and similarly for B.

### Joint Measurement Effect

**ZZ Merge:** Measures the observable $\bar{Z}_A \otimes \bar{Z}_B$

The state projects onto an eigenstate of $Z_A Z_B$:

$$|\Psi\rangle \xrightarrow{M_{ZZ}} \begin{cases}
\frac{1}{\sqrt{p_+}} \Pi_+ |\Psi\rangle & \text{with prob } p_+ \\
\frac{1}{\sqrt{p_-}} \Pi_- |\Psi\rangle & \text{with prob } p_-
\end{cases}$$

where:
- $\Pi_+ = \frac{1}{2}(I + Z_A Z_B)$ projects onto $Z_A Z_B = +1$ eigenspace
- $\Pi_- = \frac{1}{2}(I - Z_A Z_B)$ projects onto $Z_A Z_B = -1$ eigenspace

$$\boxed{P(\text{outcome } m) = \langle\Psi|\Pi_m|\Psi\rangle, \quad m \in \{+, -\}}$$

### Explicit State Transformation

For $|\Psi\rangle = (\alpha_A|0\rangle + \beta_A|1\rangle) \otimes (\alpha_B|0\rangle + \beta_B|1\rangle)$:

**ZZ measurement outcome +1:**
$$|\Psi_+\rangle \propto \alpha_A\alpha_B|00\rangle + \beta_A\beta_B|11\rangle$$

**ZZ measurement outcome -1:**
$$|\Psi_-\rangle \propto \alpha_A\beta_B|01\rangle + \beta_A\alpha_B|10\rangle$$

$$\boxed{\text{ZZ merge with outcome } m \text{ creates state in } Z_AZ_B = m \text{ eigenspace}}$$

---

## 3. ZZ Merge Protocol

### Step-by-Step Procedure

**Setup:** Patches A and B with rough boundaries facing each other

```
    Patch A             Merge Zone           Patch B
┌──────────────┐    ┌────────────┐    ┌──────────────┐
│  ●  ●  ●  ●  │    │            │    │  ●  ●  ●  ●  │
│  ●  ●  ●  ● R│ → │  ● ─ ● ─ ● │ ← │R ●  ●  ●  ●  │
│  ●  ●  ●  ●  │    │            │    │  ●  ●  ●  ●  │
└──────────────┘    └────────────┘    └──────────────┘
   Rough boundary    New qubits        Rough boundary
```

**Step 1: Initialize merge qubits**

Add new data qubits in the merge zone, initialized in $|0\rangle$:
$$|\text{new}\rangle = |0\rangle^{\otimes d}$$

**Step 2: Introduce connecting stabilizers**

New Z-type stabilizers span across the merge region:
$$S_Z^{\text{merge}} = Z_{A,\text{edge}} \cdot Z_{\text{new}_i} \cdot Z_{\text{new}_{i+1}} \cdot Z_{B,\text{edge}}$$

**Step 3: Measure merged stabilizers for d rounds**

Repeat stabilizer measurements for $d$ rounds to ensure fault-tolerance.

**Step 4: Extract logical measurement outcome**

The product of all Z stabilizer measurements in the merge zone gives $Z_A Z_B$:

$$m_{ZZ} = \prod_{\text{merge stabilizers}} (\text{measurement outcomes})$$

**Step 5: Update syndrome decoding**

The decoder now treats A+merge+B as a single extended patch.

$$\boxed{t_{\text{ZZ merge}} = d \times \tau_{\text{cycle}}}$$

---

## 4. XX Merge Protocol

### Setup and Procedure

**Setup:** Patches A and B with smooth boundaries facing each other

```
    Patch A                              Patch B
┌──────────────┐                    ┌──────────────┐
│  ●  ●  ●  ●  │                    │  ●  ●  ●  ●  │
│  ●  ●  ●  ●  │ ←── Smooth ───→   │  ●  ●  ●  ●  │
│  ●  ●  ●  ●  │    boundaries      │  ●  ●  ●  ●  │
└──────────────┘                    └──────────────┘
       S                                   S
```

**Step 1: Initialize merge qubits in $|+\rangle$**

$$|\text{new}\rangle = |+\rangle^{\otimes d}$$

This is the key difference from ZZ merge!

**Step 2: Introduce X-type connecting stabilizers**

$$S_X^{\text{merge}} = X_{A,\text{edge}} \cdot X_{\text{new}_i} \cdot X_{\text{new}_{i+1}} \cdot X_{B,\text{edge}}$$

**Step 3: Measure for d rounds**

**Step 4: Extract logical $X_A X_B$ measurement**

$$m_{XX} = \prod_{\text{merge X-stabilizers}} (\text{measurement outcomes})$$

$$\boxed{\text{XX merge: Initialize in } |+\rangle, \text{ measure X stabilizers}}$$

---

## 5. Logical Operator Transformation

### Before and After Merge

**Before ZZ merge:**
- Patch A: $\bar{Z}_A$ (vertical path), $\bar{X}_A$ (horizontal path)
- Patch B: $\bar{Z}_B$ (vertical path), $\bar{X}_B$ (horizontal path)
- Independent logical qubits

**After ZZ merge:**
- $\bar{Z}_A$ and $\bar{Z}_B$ become equivalent (connected through merge)
- The **merged patch** encodes only **one logical qubit**
- $\bar{X}_{\text{merged}} = \bar{X}_A \cdot \bar{X}_B$ (product of original X operators)

### Logical Operator Fate

$$\boxed{\text{ZZ merge: } \bar{Z}_A \equiv \bar{Z}_B, \quad \bar{X}_{\text{new}} = \bar{X}_A \bar{X}_B}$$

$$\boxed{\text{XX merge: } \bar{X}_A \equiv \bar{X}_B, \quad \bar{Z}_{\text{new}} = \bar{Z}_A \bar{Z}_B}$$

### Information Flow

The merge operation:
1. **Measures** one joint observable ($Z_AZ_B$ or $X_AX_B$)
2. **Destroys** one logical qubit (merged into single patch)
3. **Preserves** information about the orthogonal observable

---

## 6. Measurement Outcomes and Pauli Corrections

### Tracking Measurement Results

Let $m \in \{+1, -1\}$ be the merge measurement outcome.

**ZZ merge with outcome $m = -1$:**

The state is in the $Z_AZ_B = -1$ eigenspace. To correct to $+1$:
$$X_A \text{ or } X_B \text{ flips the relative phase}$$

**XX merge with outcome $m = -1$:**

Apply $Z_A$ or $Z_B$ to correct.

### Pauli Frame Tracking

In practice, we don't physically apply corrections. Instead, we **track the Pauli frame**:

$$|\psi_{\text{corrected}}\rangle = P \cdot |\psi_{\text{measured}}\rangle$$

where $P \in \{I, X, Y, Z\}$ is tracked classically.

$$\boxed{\text{Correction after merge: } X^{(1-m)/2} \text{ for ZZ}, \quad Z^{(1-m)/2} \text{ for XX}}$$

---

## 7. Fault-Tolerance of Merge Operations

### Error Propagation During Merge

**Concern:** Does the merge create opportunities for errors to spread?

**Analysis:**

1. **Errors on merge qubits** before stabilizer measurement:
   - Detected by initial stabilizer measurements
   - Correction applied before merge completes

2. **Measurement errors** on merge stabilizers:
   - Handled by $d$ rounds of repeated measurement
   - Majority voting / minimum weight matching

3. **Errors during merge**:
   - Code distance reduced to $\sim d/2$ at merge boundary
   - But only for $O(1)$ positions, not global reduction

### Distance During Merge

$$\boxed{d_{\text{effective}} \geq \frac{d}{2} \text{ during merge}}$$

The logical error rate scales as:
$$P_L^{\text{merge}} \sim \left(\frac{p}{p_{\text{th}}}\right)^{d/4}$$

This is worse than the static code but still exponentially suppressed.

### Requirements for Fault-Tolerant Merge

1. Syndrome measurement for $d$ rounds minimum
2. Proper decoder handling of the merge region
3. Soft boundary between patches (distance scales with merge progress)

---

## 8. Worked Examples

### Example 1: ZZ Merge on Computational Basis States

**Problem:** Two patches A and B are prepared in $|0_L\rangle_A$ and $|1_L\rangle_B$. Perform a ZZ merge and determine the outcome.

**Solution:**

Initial state:
$$|\Psi\rangle = |0_L\rangle_A \otimes |1_L\rangle_B = |0_L, 1_L\rangle$$

ZZ measurement on $|0\rangle|1\rangle$:
$$Z_A|0\rangle = +1 \cdot |0\rangle$$
$$Z_B|1\rangle = -1 \cdot |1\rangle$$

Therefore:
$$Z_A Z_B |0_L, 1_L\rangle = (+1)(-1)|0_L, 1_L\rangle = -1|0_L, 1_L\rangle$$

**Outcome:** $m_{ZZ} = -1$ with probability 1.

The state remains $|0_L, 1_L\rangle$ after measurement (eigenstate of $Z_AZ_B$).

$$\boxed{|0_L\rangle \otimes |1_L\rangle \xrightarrow{ZZ \text{ merge}} |0_L, 1_L\rangle, \quad m = -1}$$

---

### Example 2: XX Merge Creating Entanglement

**Problem:** Patches A and B are both in $|+_L\rangle$. Perform an XX merge. What is the resulting state?

**Solution:**

Initial state:
$$|\Psi\rangle = |+_L\rangle_A \otimes |+_L\rangle_B = |+_L, +_L\rangle$$

Express in computational basis:
$$|+_L, +_L\rangle = \frac{1}{2}(|0_L\rangle + |1_L\rangle)(|0_L\rangle + |1_L\rangle)$$
$$= \frac{1}{2}(|0_L0_L\rangle + |0_L1_L\rangle + |1_L0_L\rangle + |1_L1_L\rangle)$$

XX measurement:
- $|++\rangle$ and $|--\rangle$ have $X_AX_B = +1$
- $|+-\rangle$ and $|-+\rangle$ have $X_AX_B = -1$

Since $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$ and $|1\rangle = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$:

$$|+_L, +_L\rangle = |+_L\rangle|+_L\rangle$$

This is an eigenstate of $X_AX_B$ with eigenvalue $+1$.

**Outcome:** $m_{XX} = +1$ with probability 1.

$$\boxed{|+_L\rangle \otimes |+_L\rangle \xrightarrow{XX \text{ merge}} |+_L, +_L\rangle, \quad m = +1}$$

---

### Example 3: Merge Creating Bell State

**Problem:** Patch A is in $|+_L\rangle$ and patch B is in $|0_L\rangle$. Perform a ZZ merge. What are the possible outcomes and resulting states?

**Solution:**

Initial state:
$$|\Psi\rangle = |+_L\rangle_A \otimes |0_L\rangle_B = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle) \otimes |0_L\rangle$$
$$= \frac{1}{\sqrt{2}}(|0_L, 0_L\rangle + |1_L, 0_L\rangle)$$

ZZ eigenvalues:
- $|0_L, 0_L\rangle$: $Z_AZ_B = (+1)(+1) = +1$
- $|1_L, 0_L\rangle$: $Z_AZ_B = (-1)(+1) = -1$

The state has equal components in both eigenspaces!

**Outcome +1 (probability 1/2):**
$$|\Psi_+\rangle = |0_L, 0_L\rangle$$

**Outcome -1 (probability 1/2):**
$$|\Psi_-\rangle = |1_L, 0_L\rangle$$

After merge, the patches are connected. The merged patch encodes:
- If $m = +1$: State where original A and B had same Z-parity
- If $m = -1$: State where original A and B had opposite Z-parity

$$\boxed{|+_L\rangle|0_L\rangle \xrightarrow{ZZ} \begin{cases} |0_L,0_L\rangle & m=+1 \\ |1_L,0_L\rangle & m=-1 \end{cases}, \quad P(m) = \frac{1}{2}}$$

---

## 9. Practice Problems

### Problem Set A: Direct Application

**A1.** Two patches are in states $|1_L\rangle_A$ and $|1_L\rangle_B$. What is the ZZ merge outcome?

**A2.** Patches are in $|-_L\rangle_A$ and $|+_L\rangle_B$. Calculate the XX merge outcome probability.

**A3.** After a ZZ merge with outcome $m=-1$, what Pauli correction brings the state to the $Z_AZ_B = +1$ eigenspace?

---

### Problem Set B: Intermediate

**B1.** Show that for arbitrary states $|\psi\rangle_A$ and $|\phi\rangle_B$, the ZZ merge outcome probabilities are:
$$P(+1) = |\langle\psi|0\rangle\langle\phi|0\rangle|^2 + |\langle\psi|1\rangle\langle\phi|1\rangle|^2$$

**B2.** Derive the state after an XX merge of $|0_L\rangle_A$ and $|0_L\rangle_B$. Express in terms of Bell states.

**B3.** Two patches in the Bell state $\frac{1}{\sqrt{2}}(|0_L0_L\rangle + |1_L1_L\rangle)$ already exist as a merged patch. If we split them with a ZZ measurement, what are the possible outcomes and final states?

---

### Problem Set C: Challenging

**C1.** Design a protocol using merge operations to measure the three-body operator $Z_A Z_B Z_C$ on three surface code patches.

**C2.** Analyze the error probability when a single physical X error occurs during the ZZ merge process. At what stage is this error most damaging?

**C3.** Prove that the merge operation preserves the stabilizer structure: show that all stabilizers of the merged patch commute.

---

## 10. Computational Lab: Simulating Merge Operations

```python
"""
Day 821 Computational Lab: Merge Operation Simulation
Simulating XX and ZZ merge operations on logical qubit states

This lab implements merge operations as quantum channels and
visualizes the effect on logical states.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Computational basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def outer(v1, v2=None):
    """Compute outer product |v1><v2|."""
    if v2 is None:
        v2 = v1
    return v1 @ v2.conj().T


def measure_observable(state, observable):
    """
    Perform projective measurement of an observable.

    Parameters:
    -----------
    state : ndarray
        State vector or density matrix
    observable : ndarray
        Hermitian observable with eigenvalues +1, -1

    Returns:
    --------
    outcome : int
        Measurement outcome (+1 or -1)
    post_state : ndarray
        Post-measurement state
    probability : float
        Probability of this outcome
    """
    # Get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(observable)

    # Round eigenvalues to +1 or -1
    eigenvalues = np.round(eigenvalues).astype(int)

    # Projectors onto +1 and -1 eigenspaces
    P_plus = np.zeros_like(observable)
    P_minus = np.zeros_like(observable)

    for i, ev in enumerate(eigenvalues):
        v = eigenvectors[:, i:i+1]
        proj = outer(v)
        if ev == 1:
            P_plus += proj
        else:
            P_minus += proj

    # Handle state as vector
    if state.shape[1] == 1:
        rho = outer(state)
    else:
        rho = state

    # Calculate probabilities
    p_plus = np.real(np.trace(P_plus @ rho))
    p_minus = np.real(np.trace(P_minus @ rho))

    # Sample outcome
    outcome = np.random.choice([1, -1], p=[p_plus, p_minus])

    # Post-measurement state
    if outcome == 1:
        post_state = P_plus @ state
        prob = p_plus
    else:
        post_state = P_minus @ state
        prob = p_minus

    # Normalize
    post_state = post_state / np.linalg.norm(post_state)

    return outcome, post_state, prob


def zz_merge(state_A, state_B, verbose=True):
    """
    Perform ZZ merge on two logical qubit states.

    Parameters:
    -----------
    state_A, state_B : ndarray
        Single-qubit state vectors
    verbose : bool
        Print detailed output

    Returns:
    --------
    outcome : int
        ZZ measurement result (+1 or -1)
    merged_state : ndarray
        Post-measurement two-qubit state
    """
    if verbose:
        print("\n" + "="*50)
        print("ZZ MERGE OPERATION")
        print("="*50)

    # Form initial product state
    initial_state = tensor(state_A, state_B)

    if verbose:
        print("\nInitial state (product):")
        print(f"  |ψ_A⟩ = {state_A.flatten()}")
        print(f"  |ψ_B⟩ = {state_B.flatten()}")

    # ZZ observable
    ZZ = tensor(Z, I) @ tensor(I, Z)

    # Measure ZZ
    outcome, post_state, prob = measure_observable(initial_state, ZZ)

    if verbose:
        print(f"\nZZ Measurement:")
        print(f"  Outcome: {'+1' if outcome == 1 else '-1'}")
        print(f"  Probability: {prob:.4f}")
        print(f"\nPost-measurement state:")
        print(f"  |ψ_merged⟩ = {post_state.flatten()}")

    return outcome, post_state


def xx_merge(state_A, state_B, verbose=True):
    """
    Perform XX merge on two logical qubit states.

    Parameters:
    -----------
    state_A, state_B : ndarray
        Single-qubit state vectors
    verbose : bool
        Print detailed output

    Returns:
    --------
    outcome : int
        XX measurement result (+1 or -1)
    merged_state : ndarray
        Post-measurement two-qubit state
    """
    if verbose:
        print("\n" + "="*50)
        print("XX MERGE OPERATION")
        print("="*50)

    # Form initial product state
    initial_state = tensor(state_A, state_B)

    if verbose:
        print("\nInitial state (product):")
        print(f"  |ψ_A⟩ = {state_A.flatten()}")
        print(f"  |ψ_B⟩ = {state_B.flatten()}")

    # XX observable
    XX = tensor(X, I) @ tensor(I, X)

    # Measure XX
    outcome, post_state, prob = measure_observable(initial_state, XX)

    if verbose:
        print(f"\nXX Measurement:")
        print(f"  Outcome: {'+1' if outcome == 1 else '-1'}")
        print(f"  Probability: {prob:.4f}")
        print(f"\nPost-measurement state:")
        print(f"  |ψ_merged⟩ = {post_state.flatten()}")

    return outcome, post_state


def analyze_merge_statistics(state_A, state_B, merge_type='ZZ', num_samples=10000):
    """
    Analyze merge outcome statistics through repeated sampling.

    Parameters:
    -----------
    state_A, state_B : ndarray
        Input states
    merge_type : str
        'ZZ' or 'XX'
    num_samples : int
        Number of samples for statistics
    """
    print(f"\n{'='*50}")
    print(f"MERGE STATISTICS ({merge_type})")
    print(f"{'='*50}")

    outcomes = []
    merge_func = zz_merge if merge_type == 'ZZ' else xx_merge

    for _ in range(num_samples):
        outcome, _ = merge_func(state_A, state_B, verbose=False)
        outcomes.append(outcome)

    outcomes = np.array(outcomes)
    p_plus = np.sum(outcomes == 1) / num_samples
    p_minus = np.sum(outcomes == -1) / num_samples

    print(f"\nInput states:")
    print(f"  |ψ_A⟩ = {state_A.flatten()}")
    print(f"  |ψ_B⟩ = {state_B.flatten()}")
    print(f"\nEmpirical probabilities ({num_samples} samples):")
    print(f"  P(+1) = {p_plus:.4f}")
    print(f"  P(-1) = {p_minus:.4f}")

    # Theoretical calculation
    initial_state = tensor(state_A, state_B)
    if merge_type == 'ZZ':
        obs = tensor(Z, I) @ tensor(I, Z)
    else:
        obs = tensor(X, I) @ tensor(I, X)

    eigenvalues, eigenvectors = np.linalg.eigh(obs)
    P_plus = np.zeros_like(obs)
    P_minus = np.zeros_like(obs)
    for i, ev in enumerate(np.round(eigenvalues).astype(int)):
        v = eigenvectors[:, i:i+1]
        if ev == 1:
            P_plus += outer(v)
        else:
            P_minus += outer(v)

    rho = outer(initial_state)
    p_plus_theory = np.real(np.trace(P_plus @ rho))
    p_minus_theory = np.real(np.trace(P_minus @ rho))

    print(f"\nTheoretical probabilities:")
    print(f"  P(+1) = {p_plus_theory:.4f}")
    print(f"  P(-1) = {p_minus_theory:.4f}")

    return p_plus, p_minus


def visualize_merge_process():
    """
    Create visualization of merge operation effect on Bloch sphere.
    """
    fig = plt.figure(figsize=(15, 5))

    # Test states
    states_A = [ket_0, ket_plus, (ket_0 + 1j*ket_1)/np.sqrt(2)]
    states_B = [ket_1, ket_plus, ket_0]
    labels_A = ['|0⟩', '|+⟩', '|i⟩']
    labels_B = ['|1⟩', '|+⟩', '|0⟩']

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axes = [ax1, ax2, ax3]

    for idx, (sA, sB, lA, lB, ax) in enumerate(zip(states_A, states_B,
                                                    labels_A, labels_B, axes)):
        # Run multiple trials
        outcomes_zz = []
        outcomes_xx = []
        for _ in range(1000):
            m_zz, _ = zz_merge(sA, sB, verbose=False)
            m_xx, _ = xx_merge(sA, sB, verbose=False)
            outcomes_zz.append(m_zz)
            outcomes_xx.append(m_xx)

        p_zz_plus = np.mean(np.array(outcomes_zz) == 1)
        p_xx_plus = np.mean(np.array(outcomes_xx) == 1)

        # Bar chart
        x = np.arange(2)
        width = 0.35

        bars1 = ax.bar(x - width/2, [p_zz_plus, 1-p_zz_plus], width,
                       label='ZZ merge', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, [p_xx_plus, 1-p_xx_plus], width,
                       label='XX merge', color='red', alpha=0.7)

        ax.set_ylabel('Probability')
        ax.set_title(f'{lA} ⊗ {lB}')
        ax.set_xticks(x)
        ax.set_xticklabels(['+1', '-1'])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add probability labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.suptitle('Merge Operation Outcome Probabilities', fontsize=14)
    plt.tight_layout()
    plt.savefig('merge_probabilities.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to 'merge_probabilities.png'")


def demonstrate_entanglement_creation():
    """
    Show how merge operations can create entanglement.
    """
    print("\n" + "="*60)
    print("ENTANGLEMENT CREATION VIA MERGE")
    print("="*60)

    # Start with |+⟩|0⟩
    state_A = ket_plus.copy()
    state_B = ket_0.copy()

    print("\nInitial state: |+⟩ ⊗ |0⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩")
    print("This is a product state (not entangled)")

    # ZZ merge
    print("\n--- Performing ZZ Merge ---")
    outcome, merged = zz_merge(state_A, state_B)

    # Check entanglement via Schmidt decomposition
    # Reshape to matrix for SVD
    merged_matrix = merged.reshape(2, 2)
    U, S, Vh = np.linalg.svd(merged_matrix)

    print(f"\nSchmidt coefficients: {S}")
    print(f"Number of non-zero Schmidt coefficients: {np.sum(S > 1e-10)}")

    if np.sum(S > 1e-10) > 1:
        print("State is ENTANGLED (Schmidt rank > 1)")
    else:
        print("State is PRODUCT (Schmidt rank = 1)")

    # Compare with |+⟩|+⟩ -> stays product under ZZ
    print("\n" + "-"*40)
    print("Contrast: |+⟩ ⊗ |+⟩ under ZZ merge")
    state_A2 = ket_plus.copy()
    state_B2 = ket_plus.copy()
    outcome2, merged2 = zz_merge(state_A2, state_B2, verbose=False)

    merged_matrix2 = merged2.reshape(2, 2)
    U2, S2, Vh2 = np.linalg.svd(merged_matrix2)
    print(f"Schmidt coefficients: {S2}")
    print(f"State remains PRODUCT (was already ZZ eigenstate)")


def main():
    """Run all Day 821 demonstrations."""
    print("Day 821: Merge Operations (XX and ZZ)")
    print("="*60)

    # Example 1: Basic ZZ merge
    print("\n--- Example 1: ZZ Merge of |0⟩ and |1⟩ ---")
    zz_merge(ket_0, ket_1)

    # Example 2: Basic XX merge
    print("\n--- Example 2: XX Merge of |+⟩ and |+⟩ ---")
    xx_merge(ket_plus, ket_plus)

    # Example 3: Superposition state
    print("\n--- Example 3: ZZ Merge of |+⟩ and |0⟩ ---")
    zz_merge(ket_plus, ket_0)

    # Statistics analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    analyze_merge_statistics(ket_plus, ket_0, 'ZZ', 10000)
    analyze_merge_statistics(ket_0, ket_plus, 'XX', 10000)

    # Visualization
    visualize_merge_process()

    # Entanglement demonstration
    demonstrate_entanglement_creation()

    print("\n" + "="*60)
    print("Day 821 Computational Lab Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 11. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| ZZ merge outcome probabilities | $P(\pm 1) = |\alpha_A\alpha_B|^2 + |\beta_A\beta_B|^2$ or $|\alpha_A\beta_B|^2 + |\beta_A\alpha_B|^2$ |
| XX merge outcome probabilities | Similar with X-basis amplitudes |
| Post-ZZ-merge state (+1) | $\propto \alpha_A\alpha_B|00\rangle + \beta_A\beta_B|11\rangle$ |
| Post-ZZ-merge state (-1) | $\propto \alpha_A\beta_B|01\rangle + \beta_A\alpha_B|10\rangle$ |
| Pauli correction (ZZ) | $X_A^{(1-m)/2}$ or $X_B^{(1-m)/2}$ |
| Pauli correction (XX) | $Z_A^{(1-m)/2}$ or $Z_B^{(1-m)/2}$ |
| Merge time | $t_{\text{merge}} = d \times \tau_{\text{cycle}}$ |
| Logical operators after ZZ merge | $\bar{Z}_A \equiv \bar{Z}_B$, $\bar{X}_{\text{merged}} = \bar{X}_A\bar{X}_B$ |

### Key Takeaways

1. **Merge operations** perform joint logical measurements by connecting patch boundaries
2. **ZZ merge** measures $Z_AZ_B$ by fusing rough boundaries with $|0\rangle$-initialized qubits
3. **XX merge** measures $X_AX_B$ by fusing smooth boundaries with $|+\rangle$-initialized qubits
4. **Measurement outcomes** are random; Pauli corrections restore desired eigenspace
5. **Merged patches** share logical operators, reducing total encoded information by one qubit
6. **Fault-tolerance** is maintained through repeated syndrome measurement

---

## 12. Daily Checklist

- [ ] I can derive outcome probabilities for merge operations on arbitrary states
- [ ] I understand why boundary type determines merge type (rough→ZZ, smooth→XX)
- [ ] I can calculate the post-measurement state after a merge
- [ ] I know how to track and apply Pauli corrections based on measurement outcomes
- [ ] I understand how logical operators transform during merge
- [ ] I ran the computational lab and observed merge statistics

---

## 13. Preview: Day 822

Tomorrow we explore **Split Operations and Logical State Preparation**:

- Reverse of merge: separating a large patch into independent patches
- Split measurement and its effect on logical states
- Preparing logical basis states using split operations
- State injection protocols for fault-tolerant initialization

Split operations complete the surgery toolkit, enabling us to both combine and separate logical qubits on demand.

---

*"The merge operation is where the magic happens - local operations on physical qubits create global entanglement on logical qubits."*
