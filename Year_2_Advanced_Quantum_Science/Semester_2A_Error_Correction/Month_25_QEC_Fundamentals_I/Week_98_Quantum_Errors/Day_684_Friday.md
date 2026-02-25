# Day 684: Three-Qubit Bit-Flip Code

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Bit-Flip Code Theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 684, you will be able to:

1. **Construct the three-qubit bit-flip code** and its encoding
2. **Implement syndrome measurement** without destroying quantum information
3. **Derive error correction circuits** for single-qubit X errors
4. **Analyze the code's limitations** against phase errors
5. **Connect to the classical repetition code** from Week 97
6. **Understand the encoding rate and distance** of $[[3,1,1]]$

---

## The Quantum Repetition Code

### Motivation: Protecting Against Bit-Flips

We learned that the X operator (bit-flip) swaps $|0\rangle \leftrightarrow |1\rangle$. How can we protect quantum information against such errors?

**Classical Solution (Week 97):** The repetition code: $0 \to 000$, $1 \to 111$

**Quantum Problem:** We cannot copy quantum states (no-cloning theorem)!

**Quantum Solution:** Encode using **entanglement**, not copying.

### The Three-Qubit Bit-Flip Code

**Logical states:**

$$\boxed{|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle}$$

**General encoded state:**

$$|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

This is **not** three copies of $|\psi\rangle$! It's an entangled state.

**Important distinction:**
- Copying: $|\psi\rangle \to |\psi\rangle|\psi\rangle|\psi\rangle$ (forbidden by no-cloning)
- Encoding: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle \to \alpha|000\rangle + \beta|111\rangle$ (allowed!)

---

## Code Parameters: [[3, 1, 1]]

### Quantum Code Notation

Quantum codes are denoted $[[n, k, d]]$:
- $n$: number of physical qubits
- $k$: number of logical qubits encoded
- $d$: code distance (minimum weight of undetectable error)

**Three-qubit bit-flip code:** $[[3, 1, 1]]$

Wait — why distance 1? Against bit-flips, shouldn't it be distance 3?

### Understanding the Distance

The **distance** is the minimum weight Pauli operator that:
1. Takes a valid codeword to another valid codeword, OR
2. Is undetectable by syndrome measurement

For bit-flip code:
- $X_1, X_2, X_3$: detectable and correctable ✓
- But $Z_1|0_L\rangle = |000\rangle$, $Z_1|1_L\rangle = -|111\rangle$

A single Z error is **undetectable** — it only introduces a phase!

$$d_{bit-flip} = 1 \text{ (against all Pauli errors)}$$

$$d_{bit-flip}^X = 3 \text{ (against X errors only)}$$

---

## Encoding Circuit

### The Encoding Operation

To encode $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ into $|\psi_L\rangle = \alpha|000\rangle + \beta|111\rangle$:

```
|ψ⟩ ─────●─────●───── |ψ_L⟩ qubit 1
         │     │
|0⟩ ─────⊕─────┼───── |ψ_L⟩ qubit 2
               │
|0⟩ ───────────⊕───── |ψ_L⟩ qubit 3
```

Two CNOT gates with the data qubit as control.

### Verification

Starting with $|\psi\rangle|0\rangle|0\rangle = (\alpha|0\rangle + \beta|1\rangle)|00\rangle$:

1. After first CNOT (control 1, target 2):
   $$\alpha|0\rangle|0\rangle|0\rangle + \beta|1\rangle|1\rangle|0\rangle = \alpha|000\rangle + \beta|110\rangle$$

2. After second CNOT (control 1, target 3):
   $$\alpha|000\rangle + \beta|111\rangle = |\psi_L\rangle \quad \checkmark$$

### Encoding as a Unitary

$$U_{enc} = \text{CNOT}_{13} \cdot \text{CNOT}_{12}$$

This is a unitary transformation on 3 qubits.

---

## Syndrome Measurement

### The Key Insight

In classical error correction, syndrome = parity checks on received bits.

In quantum, we need to extract **error information without measuring the encoded data**.

### Syndrome Operators

For the bit-flip code, define:

$$\boxed{Z_1Z_2 \text{ and } Z_2Z_3}$$

These are the **stabilizer generators** (more on this in Week 99).

**Properties:**
- $Z_1Z_2$ measures the parity of qubits 1 and 2
- $Z_2Z_3$ measures the parity of qubits 2 and 3
- Both have eigenvalues $\pm 1$
- Both commute with $X_L = X_1X_2X_3$ and $Z_L = Z_1Z_2Z_3$

### Syndrome Table

| Error | $Z_1Z_2$ | $Z_2Z_3$ | Syndrome |
|-------|----------|----------|----------|
| I (no error) | +1 | +1 | (0, 0) |
| $X_1$ | −1 | +1 | (1, 0) |
| $X_2$ | −1 | −1 | (1, 1) |
| $X_3$ | +1 | −1 | (0, 1) |

Each single-qubit X error produces a **unique syndrome**!

### Syndrome Measurement Circuit

To measure $Z_1Z_2$ without collapsing the encoded state:

```
|0⟩ ──H──●──────●──H──M──
         │      │
  q1 ────Z──────┼─────────
                │
  q2 ───────────Z─────────
```

**Alternative with CNOTs:**

```
|0⟩ ────●────●────M──  ancilla
        │    │
 q1 ────⊕────┼────────  data qubit 1
             │
 q2 ─────────⊕────────  data qubit 2
```

The ancilla measures the parity $q_1 \oplus q_2$ in the computational basis.

---

## Complete Error Correction Protocol

### Step 1: Encode

$$|\psi\rangle|00\rangle \xrightarrow{\text{CNOT}_{12}\text{CNOT}_{13}} |\psi_L\rangle$$

### Step 2: Error Occurs

Some X error affects the encoded state:
$$|\psi_L\rangle \to E|\psi_L\rangle$$

where $E \in \{I, X_1, X_2, X_3\}$.

### Step 3: Syndrome Extraction

Measure $Z_1Z_2$ and $Z_2Z_3$ using ancilla qubits.

### Step 4: Apply Correction

Based on syndrome, apply correction:

| Syndrome | Correction |
|----------|------------|
| (0, 0) | I |
| (1, 0) | $X_1$ |
| (1, 1) | $X_2$ |
| (0, 1) | $X_3$ |

### Step 5: Decode (if needed)

Reverse the encoding to extract $|\psi\rangle$.

---

## Why This Works: Detailed Analysis

### Superposition Preservation

Consider the encoded state after error $X_1$:

$$X_1(\alpha|000\rangle + \beta|111\rangle) = \alpha|100\rangle + \beta|011\rangle$$

Measuring $Z_1Z_2$ gives:
- $Z_1Z_2|100\rangle = (-1)(+1)|100\rangle = -|100\rangle$
- $Z_1Z_2|011\rangle = (+1)(-1)|011\rangle = -|011\rangle$

Both components have the **same eigenvalue** (-1), so the superposition is preserved!

$$\alpha|100\rangle + \beta|011\rangle \xrightarrow{Z_1Z_2 \text{ meas.}} \alpha|100\rangle + \beta|011\rangle$$

The measurement tells us $X_1$ error occurred, but doesn't reveal $\alpha, \beta$.

### Commutativity Argument

Why does syndrome measurement preserve the code space?

$$[Z_1Z_2, X_L] = [Z_1Z_2, X_1X_2X_3] = 0$$

Because $Z_1Z_2$ anticommutes with $X_1$ and $X_2$, but commutes with $X_3$:
$$Z_1Z_2 \cdot X_1X_2X_3 = (-1)(-1)(+1) X_1X_2X_3 \cdot Z_1Z_2 = X_1X_2X_3 \cdot Z_1Z_2$$

---

## Limitations of the Bit-Flip Code

### No Protection Against Phase Errors

Consider a Z error on qubit 1:

$$Z_1|\psi_L\rangle = Z_1(\alpha|000\rangle + \beta|111\rangle) = \alpha|000\rangle - \beta|111\rangle$$

What syndrome does this produce?

$$Z_1Z_2(Z_1|\psi_L\rangle) = Z_1Z_2(\alpha|000\rangle - \beta|111\rangle) = \alpha|000\rangle - \beta|111\rangle$$

Eigenvalue +1 for both terms! Similarly for $Z_2Z_3$: eigenvalue +1.

**Syndrome: (0, 0)** — same as no error!

The code **cannot detect Z errors**.

### Rate and Overhead

- **Encoding rate:** $k/n = 1/3$ (encoding 1 qubit into 3)
- **Overhead:** 200% (need 2 extra physical qubits per logical qubit)
- **Distance:** $d = 1$ against general errors; $d_X = 3$ against X-only

### When Is It Useful?

The bit-flip code is useful for:
1. **Pedagogical purposes** — simplest quantum code
2. **Highly biased noise** — when Z errors are negligible
3. **Building block** — combines with phase-flip code to make Shor code

---

## Connection to Classical Repetition Code

### Parallel Structure

| Classical Repetition | Quantum Bit-Flip |
|---------------------|------------------|
| $0 \to 000$ | $\|0\rangle \to \|000\rangle$ |
| $1 \to 111$ | $\|1\rangle \to \|111\rangle$ |
| Parity checks | $Z_1Z_2$, $Z_2Z_3$ |
| Syndrome: XOR of bits | Syndrome: eigenvalues |
| Correct via majority | Correct via syndrome lookup |

### Key Difference

Classical: Can measure all bits, then vote.

Quantum: Cannot measure data qubits! Must use ancilla-based syndrome extraction.

---

## Worked Examples

### Example 1: Encoding |+⟩

**Problem:** Encode $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ into the bit-flip code.

**Solution:**

$$|\psi_L\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle) = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

This is the GHZ state! The encoding maps computational basis states to GHZ-type entangled states.

### Example 2: Syndrome Calculation

**Problem:** The received state is $|\phi\rangle = \alpha|010\rangle + \beta|101\rangle$. Find the syndrome and identify the error.

**Solution:**

Measure $Z_1Z_2$:
- $Z_1Z_2|010\rangle = (+1)(-1)|010\rangle = -|010\rangle$
- $Z_1Z_2|101\rangle = (-1)(+1)|101\rangle = -|101\rangle$

Both components: eigenvalue −1. So $s_1 = 1$.

Measure $Z_2Z_3$:
- $Z_2Z_3|010\rangle = (-1)(+1)|010\rangle = -|010\rangle$
- $Z_2Z_3|101\rangle = (+1)(-1)|101\rangle = -|101\rangle$

Both components: eigenvalue −1. So $s_2 = 1$.

**Syndrome: (1, 1)** → Error $X_2$.

Apply $X_2$ correction:
$$X_2(\alpha|010\rangle + \beta|101\rangle) = \alpha|000\rangle + \beta|111\rangle = |\psi_L\rangle$$

Corrected!

### Example 3: Two-Qubit Error

**Problem:** What happens if $X_1X_2$ error occurs?

**Solution:**

$$X_1X_2|\psi_L\rangle = \alpha|110\rangle + \beta|001\rangle$$

Syndrome:
- $Z_1Z_2|110\rangle = (-1)(-1)|110\rangle = +|110\rangle$ → $s_1 = 0$
- $Z_2Z_3|110\rangle = (-1)(+1)|110\rangle = -|110\rangle$ → $s_2 = 1$

Syndrome: (0, 1) → Code thinks it's $X_3$ error!

Applying $X_3$ "correction":
$$X_3(\alpha|110\rangle + \beta|001\rangle) = \alpha|111\rangle + \beta|000\rangle$$

This is $|1_L\rangle$ when we wanted $|0_L\rangle$! We've **made it worse**.

The code fails for two-qubit errors.

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Write out the encoding circuit for the bit-flip code using quantum circuit notation.

**A.2** If the encoded state is $\frac{1}{\sqrt{2}}|000\rangle + \frac{1}{\sqrt{2}}|111\rangle$ and error $X_3$ occurs, what is the resulting state?

**A.3** Compute the syndrome for the state $|011\rangle$ and identify what single-qubit error could have caused this.

### Problem Set B: Intermediate

**B.1** Prove that $[Z_1Z_2, Z_2Z_3] = 0$ (the syndrome operators commute).

**B.2** Show that encoding followed by decoding (reverse CNOTs) gives the identity, assuming no errors.

**B.3** What syndrome does the error $X_1X_3$ produce? What correction will be applied? What is the final logical state?

### Problem Set C: Challenging

**C.1** Prove that any single X error produces a unique syndrome in the three-qubit bit-flip code.

**C.2** Generalize to a 5-qubit repetition code. What are the syndrome operators? How many syndromes can you distinguish?

**C.3** Design a circuit that implements the full error correction protocol (encoding, syndrome measurement, correction) using only CNOT and measurements.

---

## Computational Lab: Simulating the Bit-Flip Code

```python
"""
Day 684 Computational Lab: Three-Qubit Bit-Flip Code
====================================================

Implementing and simulating the simplest quantum error correcting code.
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Basic Quantum Operations
# =============================================================================

# Single qubit states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*args):
    """Compute tensor product of multiple matrices/vectors."""
    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)
    return result

def CNOT(control: int, target: int, n_qubits: int) -> np.ndarray:
    """
    Create CNOT gate acting on n_qubits, with specified control and target.
    Qubit numbering: 0, 1, 2, ... (from left in tensor product)
    """
    dim = 2**n_qubits
    result = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        # Convert index to binary representation
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]

        # If control bit is 1, flip target bit
        new_bits = bits.copy()
        if bits[control] == 1:
            new_bits[target] = 1 - bits[target]

        # Convert back to index
        j = sum(b << (n_qubits - 1 - q) for q, b in enumerate(new_bits))
        result[j, i] = 1

    return result

print("=" * 60)
print("PART 1: Three-Qubit Bit-Flip Code Encoding")
print("=" * 60)

# =============================================================================
# Part 2: Encoding
# =============================================================================

def encode_bit_flip(state_1q: np.ndarray) -> np.ndarray:
    """
    Encode a single-qubit state into the 3-qubit bit-flip code.
    |ψ⟩ → |ψ_L⟩ = α|000⟩ + β|111⟩
    """
    # Start with |ψ⟩|0⟩|0⟩
    state_3q = tensor(state_1q, ket_0, ket_0)

    # Apply CNOT_{0,1} then CNOT_{0,2}
    cnot_01 = CNOT(0, 1, 3)
    cnot_02 = CNOT(0, 2, 3)

    state_3q = cnot_01 @ state_3q
    state_3q = cnot_02 @ state_3q

    return state_3q

# Test encoding
alpha, beta = 1/np.sqrt(3), np.sqrt(2/3)
psi = alpha * ket_0 + beta * ket_1

psi_L = encode_bit_flip(psi)

print(f"\nOriginal state: |ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")
print(f"\nEncoded state |ψ_L⟩:")
labels = ['000', '001', '010', '011', '100', '101', '110', '111']
for i, label in enumerate(labels):
    if np.abs(psi_L[i]) > 1e-10:
        print(f"  |{label}⟩: {psi_L[i]:.4f}")

print(f"\nVerify: |000⟩ amplitude = α = {alpha:.4f}")
print(f"        |111⟩ amplitude = β = {beta:.4f}")

# =============================================================================
# Part 3: Syndrome Measurement
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Syndrome Measurement")
print("=" * 60)

def measure_syndrome_Z1Z2(state: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Measure the Z1Z2 stabilizer.
    Returns: (syndrome bit, post-measurement state)
    """
    # Z1Z2 = Z ⊗ Z ⊗ I
    Z1Z2 = tensor(Z, Z, I)

    # Project onto +1 eigenspace: P_+ = (I + Z1Z2)/2
    P_plus = (np.eye(8) + Z1Z2) / 2
    P_minus = (np.eye(8) - Z1Z2) / 2

    # Measurement probabilities
    p_plus = np.real(state.conj() @ P_plus @ state)
    p_minus = np.real(state.conj() @ P_minus @ state)

    # Determine outcome (for simulation, we pick deterministically based on state)
    if p_plus > 0.99:
        return 0, state
    elif p_minus > 0.99:
        return 1, state
    else:
        # State is in superposition of eigenspaces - shouldn't happen for valid codewords with errors
        print(f"  Warning: p(+1)={p_plus:.4f}, p(-1)={p_minus:.4f}")
        return 0 if p_plus > p_minus else 1, state

def measure_syndrome_Z2Z3(state: np.ndarray) -> Tuple[int, np.ndarray]:
    """Measure the Z2Z3 stabilizer."""
    Z2Z3 = tensor(I, Z, Z)

    P_plus = (np.eye(8) + Z2Z3) / 2
    P_minus = (np.eye(8) - Z2Z3) / 2

    p_plus = np.real(state.conj() @ P_plus @ state)
    p_minus = np.real(state.conj() @ P_minus @ state)

    if p_plus > 0.99:
        return 0, state
    elif p_minus > 0.99:
        return 1, state
    else:
        return 0 if p_plus > p_minus else 1, state

def get_syndrome(state: np.ndarray) -> Tuple[int, int]:
    """Get full syndrome (s1, s2) for Z1Z2 and Z2Z3."""
    s1, _ = measure_syndrome_Z1Z2(state)
    s2, _ = measure_syndrome_Z2Z3(state)
    return s1, s2

# Test syndrome on encoded state with various errors
print("\nSyndrome table for encoded state with single X errors:")
print("-" * 40)
print(f"{'Error':<10} {'Z1Z2':<8} {'Z2Z3':<8} {'Syndrome':<12}")
print("-" * 40)

# Pauli X operators on 3 qubits
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
X3 = tensor(I, I, X)

errors = {'I': np.eye(8), 'X1': X1, 'X2': X2, 'X3': X3}

for name, E in errors.items():
    error_state = E @ psi_L
    s1, s2 = get_syndrome(error_state)
    print(f"{name:<10} {'+1' if s1==0 else '-1':<8} {'+1' if s2==0 else '-1':<8} ({s1}, {s2})")

# =============================================================================
# Part 4: Error Correction
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Complete Error Correction Protocol")
print("=" * 60)

def correct_error(state: np.ndarray) -> np.ndarray:
    """Apply error correction based on syndrome."""
    s1, s2 = get_syndrome(state)

    correction_table = {
        (0, 0): np.eye(8),       # No error
        (1, 0): X1,              # X1 error
        (1, 1): X2,              # X2 error
        (0, 1): X3               # X3 error
    }

    correction = correction_table[(s1, s2)]
    return correction @ state

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity between two pure states."""
    return np.abs(np.vdot(state1, state2))**2

# Test error correction
print("\nError correction demonstration:")
print("-" * 50)

for name, E in errors.items():
    # Apply error
    error_state = E @ psi_L

    # Get syndrome
    syndrome = get_syndrome(error_state)

    # Apply correction
    corrected_state = correct_error(error_state)

    # Check fidelity with original
    fid = fidelity(corrected_state, psi_L)

    print(f"{name}: syndrome {syndrome}, fidelity after correction = {fid:.6f}")

# =============================================================================
# Part 5: Two-Qubit Errors (Code Failure)
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Code Failure with Two-Qubit Errors")
print("=" * 60)

X1X2 = X1 @ X2
X1X3 = X1 @ X3
X2X3 = X2 @ X3

two_errors = {'X1X2': X1X2, 'X1X3': X1X3, 'X2X3': X2X3}

print("\nTwo-qubit error analysis:")
print("-" * 60)

for name, E in two_errors.items():
    # Apply error
    error_state = E @ psi_L

    # Get syndrome
    syndrome = get_syndrome(error_state)

    # Apply "correction"
    corrected_state = correct_error(error_state)

    # Check fidelity
    fid = fidelity(corrected_state, psi_L)

    # What correction was applied?
    correction_applied = {(0,0): 'I', (1,0): 'X1', (1,1): 'X2', (0,1): 'X3'}[syndrome]

    print(f"{name}: syndrome {syndrome} → applies {correction_applied}, fidelity = {fid:.6f}")

print("\nNote: Two-qubit errors cause miscorrection (low fidelity)!")

# =============================================================================
# Part 6: Phase Error Vulnerability
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Vulnerability to Phase (Z) Errors")
print("=" * 60)

Z1 = tensor(Z, I, I)
Z2 = tensor(I, Z, I)
Z3 = tensor(I, I, Z)

z_errors = {'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

print("\nZ error analysis on encoded |+_L⟩:")
print("-" * 50)

# Use |+⟩ to see phase effects
psi_plus = (ket_0 + ket_1) / np.sqrt(2)
psi_plus_L = encode_bit_flip(psi_plus)

for name, E in z_errors.items():
    # Apply Z error
    error_state = E @ psi_plus_L

    # Get syndrome
    syndrome = get_syndrome(error_state)

    # Check if state changed
    fid_original = fidelity(error_state, psi_plus_L)

    print(f"{name}: syndrome {syndrome}, fidelity with original = {fid_original:.6f}")

print("\nZ errors produce syndrome (0,0) but corrupt the logical state!")
print("The bit-flip code CANNOT detect Z errors.")

# =============================================================================
# Part 7: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Syndrome table visualization
ax1 = axes[0]
syndrome_data = np.array([
    [0, 0],  # I
    [1, 0],  # X1
    [1, 1],  # X2
    [0, 1],  # X3
])
errors_list = ['I', 'X₁', 'X₂', 'X₃']

im = ax1.imshow(syndrome_data, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Z₁Z₂', 'Z₂Z₃'])
ax1.set_yticks(range(4))
ax1.set_yticklabels(errors_list)
ax1.set_title('Syndrome Table\n(0=+1, 1=-1)')
for i in range(4):
    for j in range(2):
        ax1.text(j, i, f'{syndrome_data[i,j]}', ha='center', va='center',
                color='white' if syndrome_data[i,j]==1 else 'black', fontsize=14)

# Plot 2: Error correction success rate
ax2 = axes[1]

# Simulate with noise
n_trials = 1000
error_probs = np.linspace(0, 0.3, 20)
success_rates = []

for p in error_probs:
    successes = 0
    for _ in range(n_trials):
        # Random initial state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi_random = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1
        psi_random_L = encode_bit_flip(psi_random)

        # Random single-qubit X error with probability p
        error_occurred = np.random.random() < p
        if error_occurred:
            qubit = np.random.choice([0, 1, 2])
            error_ops = [X1, X2, X3]
            psi_error = error_ops[qubit] @ psi_random_L
        else:
            psi_error = psi_random_L

        # Correct
        psi_corrected = correct_error(psi_error)

        # Check success
        if fidelity(psi_corrected, psi_random_L) > 0.99:
            successes += 1

    success_rates.append(successes / n_trials)

ax2.plot(error_probs * 100, np.array(success_rates) * 100, 'b-', linewidth=2)
ax2.axhline(100, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Single X Error Probability (%)')
ax2.set_ylabel('Successful Correction Rate (%)')
ax2.set_title('Error Correction Performance\n(Single X Errors)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

# Plot 3: Code space visualization
ax3 = axes[2]
ax3.axis('off')

code_diagram = """
┌─────────────────────────────────────────┐
│     Three-Qubit Bit-Flip Code [[3,1,1]] │
├─────────────────────────────────────────┤
│                                         │
│  Logical Basis:                         │
│    |0_L⟩ = |000⟩                        │
│    |1_L⟩ = |111⟩                        │
│                                         │
│  Stabilizers:                           │
│    S₁ = Z₁Z₂                            │
│    S₂ = Z₂Z₃                            │
│                                         │
│  Logical Operators:                     │
│    X_L = X₁X₂X₃                         │
│    Z_L = Z₁Z₂Z₃                         │
│                                         │
│  Correctable: Single X errors           │
│  NOT correctable: Z errors, 2+ X errors │
│                                         │
│  Code Rate: k/n = 1/3                   │
│  Distance: d = 1 (general), d_X = 3     │
└─────────────────────────────────────────┘
"""
ax3.text(0.05, 0.95, code_diagram, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('day_684_bit_flip_code.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_684_bit_flip_code.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Three-Qubit Bit-Flip Code")
print("=" * 60)

summary = """
┌────────────────────────────────────────────────────────────────────┐
│                Three-Qubit Bit-Flip Code [[3,1,1]]                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ENCODING:                                                           │
│   |ψ⟩ = α|0⟩ + β|1⟩  →  |ψ_L⟩ = α|000⟩ + β|111⟩                   │
│   Circuit: CNOT₀₁ then CNOT₀₂                                       │
│                                                                     │
│ SYNDROME MEASUREMENT:                                               │
│   Measure Z₁Z₂ and Z₂Z₃ (eigenvalues ±1)                           │
│   Maps each single X error to unique syndrome                       │
│                                                                     │
│ CORRECTION:                                                         │
│   (0,0) → I,  (1,0) → X₁,  (1,1) → X₂,  (0,1) → X₃                │
│                                                                     │
│ LIMITATIONS:                                                        │
│   ✗ Cannot detect Z errors                                          │
│   ✗ Cannot correct 2+ qubit X errors                                │
│   ✗ True distance d=1 (against all Paulis)                          │
│                                                                     │
│ NEXT: Phase-flip code to protect against Z errors                   │
└────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 684 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Logical states | $\|0_L\rangle = \|000\rangle$, $\|1_L\rangle = \|111\rangle$ |
| Stabilizers | $Z_1Z_2$, $Z_2Z_3$ |
| Logical X | $X_L = X_1X_2X_3$ |
| Code parameters | $[[3, 1, 1]]$ (distance 1 against general errors) |
| Encoding | $\|\psi\rangle\|00\rangle \to \text{CNOT}_{12}\text{CNOT}_{13} \to \|\psi_L\rangle$ |

### Syndrome Table

| Error | $Z_1Z_2$ | $Z_2Z_3$ | Correction |
|-------|----------|----------|------------|
| None | +1 | +1 | I |
| $X_1$ | −1 | +1 | $X_1$ |
| $X_2$ | −1 | −1 | $X_2$ |
| $X_3$ | +1 | −1 | $X_3$ |

### Main Takeaways

1. **Bit-flip code** is the simplest quantum error correcting code
2. **Entanglement replaces copying** — we encode, not clone
3. **Syndrome measurement** extracts error info without measuring data
4. **Each syndrome uniquely identifies** a single X error
5. **Major limitation:** cannot detect or correct Z (phase) errors
6. **True distance d=1** against general Pauli errors

---

## Daily Checklist

- [ ] I can construct the encoded logical states
- [ ] I understand the encoding circuit
- [ ] I can compute syndromes for arbitrary error states
- [ ] I know why syndrome measurement doesn't collapse the encoded state
- [ ] I understand why Z errors go undetected

---

## Preview: Day 685

Tomorrow we study the **three-qubit phase-flip code**:
- Dual to the bit-flip code
- Protects against Z errors instead of X
- Combining both codes leads to the **Shor 9-qubit code**

---

**Day 684 Complete!** Week 98: 5/7 days (71%)
