# Day 964: Qubitization and Block Encoding

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3.5 hours | Block encoding fundamentals and LCU |
| Afternoon | 2.5 hours | Qubitization and quantum walks |
| Evening | 1 hour | Computational lab: Block encoding implementation |

## Learning Objectives

By the end of today, you will be able to:

1. Define block encoding and explain its role in quantum algorithms
2. Construct block encodings using Linear Combination of Unitaries (LCU)
3. Apply qubitization to create quantum walk operators from block encodings
4. Analyze the overhead costs of block encoding for local Hamiltonians
5. Connect block encoding to QSVT for optimal simulation
6. Implement block-encoded matrix operations in code

## Core Content

### 1. What is Block Encoding?

**Block encoding** is a technique to embed a non-unitary matrix $A$ inside a larger unitary $U_A$.

**Definition:** A unitary $U_A$ is an $(\alpha, a, \epsilon)$-block encoding of matrix $A$ if:

$$\boxed{\left\| A - \alpha \langle 0|^{\otimes a} U_A |0\rangle^{\otimes a} \right\| \leq \epsilon}$$

Pictorially:

```
            ┌─────────┐
|0⟩^a  ────┤         ├──── |0⟩^a projected
           │   U_A   │
|ψ⟩    ────┤         ├──── A|ψ⟩/α (if ancilla measured |0⟩)
            └─────────┘
```

**Parameters:**
- $\alpha$: **Normalization factor** (1-norm or spectral norm scale)
- $a$: **Number of ancilla qubits**
- $\epsilon$: **Approximation error**

---

### 2. Why Block Encoding?

The power of block encoding lies in **composability**:

| Operation | Block Encoding |
|-----------|---------------|
| $A$ | $U_A$ is $(\alpha, a, \epsilon)$-block encoding |
| $A + B$ | $U_{A+B}$ from $U_A$ and $U_B$ |
| $A \cdot B$ | $U_{AB}$ from $U_A$ and $U_B$ |
| $P(A)$ | QSVT with $U_A$ |
| $e^{-iAt}$ | Hamiltonian simulation via QSVT |

Once you have a block encoding, QSVT gives you polynomial transformations "for free."

---

### 3. Linear Combination of Unitaries (LCU)

**LCU** is the primary method for constructing block encodings.

Given a Hamiltonian decomposition:

$$H = \sum_{j=0}^{L-1} \alpha_j U_j$$

where $U_j$ are unitary and $\alpha_j \geq 0$ (can absorb signs into $U_j$).

**The LCU Block Encoding:**

$$\boxed{U_H = (V^\dagger \otimes I) \cdot \text{SELECT} \cdot (V \otimes I)}$$

where:

**PREPARE ($V$):**

$$V|0\rangle^{\otimes a} = \sum_{j=0}^{L-1} \sqrt{\frac{\alpha_j}{\lambda}} |j\rangle$$

with $\lambda = \sum_j \alpha_j$ (the 1-norm).

**SELECT:**

$$\text{SELECT} = \sum_{j=0}^{L-1} |j\rangle\langle j| \otimes U_j$$

Applies $U_j$ controlled on register being in state $|j\rangle$.

---

### 4. LCU Analysis

**Theorem:** The LCU circuit $U_H$ is a $(\lambda, a, 0)$-block encoding of $H$:

$$\langle 0|^{\otimes a} U_H |0\rangle^{\otimes a} = \frac{H}{\lambda}$$

**Proof:**

$$\langle 0|^{\otimes a} U_H |0\rangle^{\otimes a} = \langle 0|^{\otimes a} V^\dagger \cdot \text{SELECT} \cdot V |0\rangle^{\otimes a}$$

$$= \sum_{j,k} \sqrt{\frac{\alpha_j \alpha_k}{\lambda^2}} \langle j|k\rangle U_j = \sum_j \frac{\alpha_j}{\lambda} U_j = \frac{H}{\lambda}$$

$\square$

**Resource costs:**
- Ancilla qubits: $a = \lceil \log_2 L \rceil$
- PREPARE: $O(L)$ gates (using QROM or arithmetic circuits)
- SELECT: $O(L)$ controlled operations

---

### 5. Qubitization: From Block Encoding to Quantum Walk

**Qubitization** converts a block encoding into a quantum walk operator that enables efficient eigenvalue processing.

**The Qubitized Walk Operator:**

Given $(\alpha, a, 0)$-block encoding $U_A$ of Hermitian $A$:

$$\boxed{W = (2|0\rangle\langle 0|^{\otimes a} - I) \cdot U_A}$$

This is a reflection composed with the block encoding.

**Key Property:** If $A$ has eigenvalues $\lambda_j$ with eigenstates $|\psi_j\rangle$, then $W$ has eigenvalues:

$$e^{\pm i \arccos(\lambda_j / \alpha)}$$

in a 2-dimensional subspace for each eigenvalue.

---

### 6. Eigenvalue Structure of Qubitized Operator

For each eigenpair $(|\psi_j\rangle, \lambda_j)$ of $A/\alpha$, the walk operator $W$ has a 2D invariant subspace spanned by:

$$|+_j\rangle = |0\rangle^{\otimes a} |\psi_j\rangle + \text{garbage}$$
$$|-_j\rangle = \text{orthogonal complement}$$

In this subspace:

$$W|+_j\rangle = e^{+i\arccos(\lambda_j/\alpha)}|+_j\rangle + \ldots$$

The eigenphases of $W$ encode $\arccos(\lambda_j/\alpha)$!

**Connection to QSP:** The signal $x = \lambda_j/\alpha$ appears through $\arccos(x)$, exactly matching the QSP signal operator $W(x) = e^{i\arccos(x) X}$.

---

### 7. QSVT with Block Encoding

**The Complete Picture:**

1. **Block encode** $H$ with normalization $\alpha$: construct $U_H$
2. **Qubitize** to get walk operator $W$
3. **Apply QSP phases** to implement polynomial $P$:
   $$U_P = e^{i\phi_0 Z} W e^{i\phi_1 Z} W \cdots$$
4. **Result:** Block encoding of $P(H/\alpha)$

For Hamiltonian simulation with $P(x) \approx e^{-ix\alpha t}$:

$$\boxed{\langle 0|^{\otimes a} U_P |0\rangle^{\otimes a} \approx e^{-iHt}}$$

**Total Complexity:**

$$O\left(\alpha t + \frac{\log(1/\epsilon)}{\log\log(1/\epsilon)}\right)$$

uses of $U_H$ and $U_H^\dagger$.

---

### 8. Block Encoding Local Hamiltonians

For a local Hamiltonian $H = \sum_{j=1}^{L} h_j H_j$ where each $H_j$ is a Pauli string:

**Method 1: Direct LCU**
- $L$ terms, each a Pauli string
- PREPARE: $O(\log L)$ ancillas, $O(L)$ gates
- SELECT: Controlled Pauli operations

**Method 2: Pauli Access Model**
- Store Pauli strings in quantum database (QROM)
- Access pattern: $|j\rangle|0\rangle \to |j\rangle |P_j\rangle$
- Controlled application of $P_j$

**Normalization:**

$$\lambda = \sum_j |h_j|$$

This 1-norm determines the simulation cost!

---

### 9. Comparison: Block Encoding vs. Product Formulas

| Aspect | Product Formulas | Block Encoding + QSVT |
|--------|------------------|----------------------|
| Query complexity | $O((\lambda t)^{1+o(1)}/\epsilon^{o(1)})$ | $O(\lambda t + \log(1/\epsilon))$ |
| Ancilla qubits | 0 | $O(\log L)$ |
| Circuit depth | $O(L \cdot n_{\text{steps}})$ | $O(\lambda t \cdot L)$ |
| Implementation | Simple | Complex |
| Best for | Short time, low precision | Long time, high precision |

---

## Worked Examples

### Example 1: Block Encoding $\sigma_z$

**Problem:** Construct a block encoding of $H = Z$.

**Solution:**

Step 1: Recognize $Z$ is already unitary.
Since $Z^\dagger Z = I$, we can use $Z$ directly:

$$\langle 0| I |0\rangle = 1 \cdot Z = Z$$

Wait, that's not right. We need:

$$\langle 0|^{\otimes a} U |0\rangle^{\otimes a} = Z/\alpha$$

Step 2: For $\alpha = 1$, use no ancilla.
Actually, if $A$ is unitary with $\|A\| = 1$, we can set:

$$U_A = A$$

and $\alpha = 1$, $a = 0$.

This is trivial: unitary matrices are their own $(1, 0, 0)$-block encoding!

Step 3: More interesting case: $H = 0.5 Z$.
Use $\alpha = 0.5$:

$$\langle 0| U |0\rangle = 0.5 Z$$

We need $U$ such that projecting gives $0.5 Z$. Use LCU with:

$$H = 0.5 Z = 0.5 \cdot Z$$

PREPARE: $V|0\rangle = |0\rangle$ (single term, $\sqrt{0.5/0.5} = 1$)
SELECT: $|0\rangle\langle 0| \otimes Z$

Actually for a single term, $U = Z$ directly works with $\alpha = 0.5$ understood.

**Answer:** $Z$ is a $(1, 0, 0)$-block encoding of itself. For scaled versions, adjust $\alpha$ accordingly.

$\square$

---

### Example 2: LCU for $H = X + Z$

**Problem:** Construct the LCU block encoding for $H = X + Z$.

**Solution:**

Step 1: Identify terms.
$$H = 1 \cdot X + 1 \cdot Z$$
$$\alpha_0 = 1, U_0 = X$$
$$\alpha_1 = 1, U_1 = Z$$
$$\lambda = 1 + 1 = 2$$

Step 2: PREPARE circuit.
Need: $V|0\rangle = \sqrt{1/2}|0\rangle + \sqrt{1/2}|1\rangle = |+\rangle$

$$V = H \text{ (Hadamard gate)}$$

Step 3: SELECT circuit.
$$\text{SELECT} = |0\rangle\langle 0| \otimes X + |1\rangle\langle 1| \otimes Z$$

This is:
- If ancilla is $|0\rangle$, apply $X$ to system
- If ancilla is $|1\rangle$, apply $Z$ to system

Circuit:
```
a: ──●────●──
     │    │
s: ──X────Z──
    (control (control
     on 0)   on 1)
```

Using standard controlled gates:
- Controlled-$X$ with control on $|0\rangle$: $X$ gate on ancilla, then CNOT, then $X$ on ancilla
- Controlled-$Z$ with control on $|1\rangle$: CZ gate

Step 4: Full LCU circuit.

```
|0⟩_a ──H──●───●──H──
           │   │
|ψ⟩_s ─────X───Z─────
```

Wait, the SELECT structure needs fixing. Let me be more careful:

```
a: ──H──●──────●──H──
        │      │
s: ─────X─(CZ)─Z─────
```

Actually:
- Controlled-X on $|0\rangle$: Need to flip control. Use $X_a$-CNOT-$X_a$:

```
a: ──H──X──●──X──●──H──
           │     │
s: ────────X─────Z─────
          (CX)  (CZ)
```

Step 5: Verify block encoding property.

$$\langle 0|_a (H \otimes I) \cdot \text{SELECT} \cdot (H \otimes I) |0\rangle_a$$

$$= \langle +|_a \text{SELECT} |+\rangle_a$$

$$= \langle +| \left( |0\rangle\langle 0| \otimes X + |1\rangle\langle 1| \otimes Z \right) |+\rangle$$

$$= \frac{1}{2}(X + Z) = \frac{H}{2}$$

So $U_H$ is a $(2, 1, 0)$-block encoding of $H = X + Z$. ✓

$\square$

---

### Example 3: Qubitized Walk Operator

**Problem:** Given the block encoding from Example 2, construct the qubitized walk operator.

**Solution:**

Step 1: The block encoding unitary.
$$U_H = (H \otimes I) \cdot \text{SELECT} \cdot (H \otimes I)$$

Step 2: Reflection operator.
$$R = 2|0\rangle\langle 0|_a \otimes I - I = Z_a \otimes I$$

(Reflection about $|0\rangle$ on ancilla)

Step 3: Walk operator.
$$W = R \cdot U_H = (Z_a \otimes I) \cdot U_H$$

Step 4: Eigenvalue analysis.
$H = X + Z$ has eigenvalues $\pm\sqrt{2}$ (easy to verify: $\text{det}(H - \lambda I) = \lambda^2 - 2$).

Normalized by $\alpha = 2$:
$$\lambda_\pm / \alpha = \pm\sqrt{2}/2 = \pm 1/\sqrt{2}$$

Walk operator eigenphases:
$$\theta_\pm = \arccos(\pm 1/\sqrt{2}) = \pi/4 \text{ or } 3\pi/4$$

So $W$ has eigenvalues $e^{\pm i\pi/4}$ and $e^{\pm 3i\pi/4}$.

$\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Simple block encoding:** Verify that the Pauli matrix $Y$ is a $(1, 0, 0)$-block encoding of itself.

2. **LCU for Ising:** For $H = Z_1 Z_2 + X_1 + X_2$, identify the terms, compute $\lambda$, and describe the PREPARE state.

3. **Reflection operator:** Show that $2|0\rangle\langle 0| - I = Z$ for a single qubit.

### Level 2: Intermediate Analysis

4. **Gate count:** For an $L$-term LCU with Pauli terms, estimate the total gate count for PREPARE and SELECT.

5. **Normalization scaling:** If $H = \sum_{j=1}^{n} X_j$ (sum of $n$ single-qubit terms), what is $\lambda$? How does simulation cost scale with $n$?

6. **Qubitization eigenpairs:** For $H = X$ (single Pauli), compute the qubitized walk operator eigenvalues explicitly.

### Level 3: Challenging Problems

7. **Optimal normalization:** For a Hamiltonian given in different decompositions, show that choosing the decomposition minimizing $\lambda$ improves simulation efficiency.

8. **Block encoding composition:** Given $(\alpha_A, a_A, 0)$-encoding of $A$ and $(\alpha_B, a_B, 0)$-encoding of $B$, construct a block encoding of $A + B$ and analyze its parameters.

9. **QSVT construction:** Outline the complete QSVT circuit for implementing $e^{-iHt}$ given a block encoding, including QSP phases for Chebyshev approximation.

---

## Computational Lab: Block Encoding Implementation

### Lab Objective

Implement LCU block encoding and verify the structure.

```python
"""
Day 964 Lab: Block Encoding and Qubitization
Week 138: Quantum Simulation
"""

import numpy as np
from scipy.linalg import expm, block_diag
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================
# Part 1: Basic Block Encoding
# =============================================================

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def verify_block_encoding(U: np.ndarray, A: np.ndarray,
                         alpha: float, n_ancilla: int) -> Tuple[float, bool]:
    """
    Verify that U is an (alpha, n_ancilla, epsilon)-block encoding of A.

    Returns (epsilon, success).
    """
    # Dimension of system
    dim_system = A.shape[0]
    dim_ancilla = 2**n_ancilla
    dim_total = dim_system * dim_ancilla

    assert U.shape == (dim_total, dim_total), "Dimension mismatch"

    # Project onto |0...0⟩ on ancilla
    proj_0 = np.zeros((dim_ancilla, dim_ancilla))
    proj_0[0, 0] = 1

    # Block encoding extraction: <0|^a U |0>^a
    extracted = np.zeros((dim_system, dim_system), dtype=complex)

    for i in range(dim_system):
        for j in range(dim_system):
            # |i⟩_sys ⊗ |0⟩_anc → index i * dim_ancilla + 0
            row = i * dim_ancilla
            col = j * dim_ancilla
            extracted[i, j] = U[row, col]

    # Should equal A/alpha
    target = A / alpha
    epsilon = np.linalg.norm(extracted - target, ord=2)

    return epsilon, epsilon < 1e-10

print("=" * 60)
print("Part 1: Verifying Block Encodings")
print("=" * 60)

# Test 1: Pauli Z is its own block encoding
print("\nTest 1: Z is (1, 0, 0)-block encoding of Z")
epsilon, success = verify_block_encoding(Z, Z, 1.0, 0)
print(f"  epsilon = {epsilon:.2e}, success = {success}")

# Test 2: Create block encoding of 0.5*Z using ancilla
# We need U such that <0|U|0> = 0.5*Z on system
# This requires embedding in larger space

print("\nTest 2: Block encoding of 0.5*Z")
# Use 1 ancilla, construct U = |0><0| ⊗ Z + |1><1| ⊗ (something)
# For simplicity, use controlled-Z structure

# =============================================================
# Part 2: LCU Block Encoding
# =============================================================

print("\n" + "=" * 60)
print("Part 2: LCU Block Encoding")
print("=" * 60)

def lcu_block_encoding(coefficients: List[float],
                       unitaries: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Construct LCU block encoding for H = sum_j alpha_j U_j.

    Returns (U_H, lambda) where U_H is the block encoding unitary
    and lambda is the normalization (1-norm).
    """
    L = len(coefficients)
    assert L > 0 and L == len(unitaries)

    # Normalization
    lambda_norm = sum(abs(c) for c in coefficients)

    # Number of ancilla qubits
    n_ancilla = int(np.ceil(np.log2(L))) if L > 1 else 1
    dim_ancilla = 2**n_ancilla
    dim_system = unitaries[0].shape[0]
    dim_total = dim_ancilla * dim_system

    # PREPARE: V|0> = sum_j sqrt(alpha_j/lambda) |j>
    prepare_state = np.zeros(dim_ancilla, dtype=complex)
    for j, alpha in enumerate(coefficients):
        prepare_state[j] = np.sqrt(abs(alpha) / lambda_norm)

    # PREPARE unitary (simplified: assume we can construct it)
    # In practice, this uses quantum arithmetic
    # Here we construct it from the target state

    # V|0> = prepare_state, extend to full unitary
    V = np.eye(dim_ancilla, dtype=complex)
    V[:, 0] = prepare_state
    # Gram-Schmidt for remaining columns
    for k in range(1, dim_ancilla):
        v = np.zeros(dim_ancilla, dtype=complex)
        v[k] = 1
        for j in range(k):
            v -= np.vdot(V[:, j], v) * V[:, j]
        v /= np.linalg.norm(v)
        V[:, k] = v

    # SELECT: sum_j |j><j| ⊗ U_j
    SELECT = np.zeros((dim_total, dim_total), dtype=complex)
    for j, U_j in enumerate(unitaries):
        # Tensor |j><j| with U_j
        proj_j = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
        proj_j[j, j] = 1
        # Include sign if coefficient is negative
        sign = np.sign(coefficients[j]) if coefficients[j] != 0 else 1
        SELECT += np.kron(proj_j, sign * U_j)

    # Handle unused ancilla states
    for j in range(L, dim_ancilla):
        proj_j = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
        proj_j[j, j] = 1
        SELECT += np.kron(proj_j, np.eye(dim_system))

    # LCU: U_H = (V† ⊗ I) SELECT (V ⊗ I)
    V_tensor = np.kron(V, np.eye(dim_system))
    V_dag_tensor = np.kron(V.conj().T, np.eye(dim_system))

    U_H = V_dag_tensor @ SELECT @ V_tensor

    return U_H, lambda_norm

# Test: H = X + Z
print("\nLCU for H = X + Z:")
coeffs = [1.0, 1.0]
unitaries = [X, Z]
U_H, lambda_norm = lcu_block_encoding(coeffs, unitaries)

H = X + Z
epsilon, success = verify_block_encoding(U_H, H, lambda_norm, 1)
print(f"  lambda = {lambda_norm:.2f}")
print(f"  Block encoding error: {epsilon:.2e}")
print(f"  Success: {success}")

# Test: H = 0.5*X + 0.3*Y + 0.2*Z
print("\nLCU for H = 0.5*X + 0.3*Y + 0.2*Z:")
coeffs2 = [0.5, 0.3, 0.2]
unitaries2 = [X, Y, Z]
U_H2, lambda_norm2 = lcu_block_encoding(coeffs2, unitaries2)

H2 = 0.5*X + 0.3*Y + 0.2*Z
epsilon2, success2 = verify_block_encoding(U_H2, H2, lambda_norm2, 2)
print(f"  lambda = {lambda_norm2:.2f}")
print(f"  Block encoding error: {epsilon2:.2e}")
print(f"  Success: {success2}")

# =============================================================
# Part 3: Qubitization
# =============================================================

print("\n" + "=" * 60)
print("Part 3: Qubitization Walk Operator")
print("=" * 60)

def qubitize(U_block: np.ndarray, n_ancilla: int) -> np.ndarray:
    """
    Construct qubitized walk operator W = R * U_block.

    R = 2|0><0|^a ⊗ I - I (reflection about |0> on ancilla)
    """
    dim_total = U_block.shape[0]
    dim_ancilla = 2**n_ancilla
    dim_system = dim_total // dim_ancilla

    # Reflection: 2|0><0| - I on ancilla
    R_ancilla = 2 * np.outer(np.eye(dim_ancilla)[0], np.eye(dim_ancilla)[0]) - np.eye(dim_ancilla)
    R = np.kron(R_ancilla, np.eye(dim_system))

    # Walk operator
    W = R @ U_block

    return W

# Qubitize H = X + Z
W = qubitize(U_H, 1)

print("\nQubitized walk operator for H = X + Z:")
print(f"  Shape: {W.shape}")

# Check eigenvalues
eigvals_W = np.linalg.eigvals(W)
print(f"  Eigenvalues of W:")
for ev in eigvals_W:
    phase = np.angle(ev)
    print(f"    {ev:.4f} (phase = {phase:.4f} rad = {np.degrees(phase):.1f}°)")

# Expected: eigenvalues of H are ±sqrt(2)
# Normalized: ±sqrt(2)/2 ≈ ±0.707
# arccos(0.707) ≈ 0.785 rad = 45°
# arccos(-0.707) ≈ 2.356 rad = 135°

eigvals_H = np.linalg.eigvalsh(H)
print(f"\nEigenvalues of H: {eigvals_H}")
expected_phases = [np.arccos(ev / lambda_norm) for ev in eigvals_H]
print(f"Expected W phases: {expected_phases} rad")

# =============================================================
# Part 4: QSVT Structure
# =============================================================

print("\n" + "=" * 60)
print("Part 4: QSVT for Polynomial Transformation")
print("=" * 60)

def qsvt_step(W: np.ndarray, phi: float, n_ancilla: int) -> np.ndarray:
    """
    Apply one QSVT rotation: exp(i*phi*Z) on ancilla, tensored with identity.
    """
    dim_total = W.shape[0]
    dim_ancilla = 2**n_ancilla
    dim_system = dim_total // dim_ancilla

    Z_ancilla = np.diag([1, -1] * (dim_ancilla // 2)) if dim_ancilla > 2 else Z
    Z_ancilla = np.array([[1, 0], [0, -1]], dtype=complex)  # For 1 ancilla
    if n_ancilla > 1:
        Z_ancilla = np.kron(Z_ancilla, np.eye(2**(n_ancilla-1)))

    phase_gate = np.kron(expm(1j * phi * Z_ancilla[:2, :2]), np.eye(dim_system))

    return phase_gate

def apply_qsvt(W: np.ndarray, phases: List[float], n_ancilla: int) -> np.ndarray:
    """
    Apply QSVT sequence: e^{i*phi_0*Z} W e^{i*phi_1*Z} W ... e^{i*phi_d*Z}
    """
    dim_total = W.shape[0]
    dim_ancilla = 2**n_ancilla
    dim_system = dim_total // dim_ancilla

    # Phase rotation on ancilla
    def phase_rotation(phi):
        Z_rot = expm(1j * phi * Z)
        return np.kron(Z_rot, np.eye(dim_system))

    d = len(phases) - 1
    U = phase_rotation(phases[0])
    for j in range(d):
        U = U @ W @ phase_rotation(phases[j + 1])

    return U

# Test: Chebyshev T_2 with zero phases
print("\nTesting QSVT with zero phases (should give Chebyshev T_d):")
phases_T2 = [0.0, 0.0, 0.0]
U_T2 = apply_qsvt(W, phases_T2, 1)

# Extract polynomial from block encoding
def extract_from_block(U: np.ndarray, n_ancilla: int) -> np.ndarray:
    dim_total = U.shape[0]
    dim_ancilla = 2**n_ancilla
    dim_system = dim_total // dim_ancilla

    extracted = np.zeros((dim_system, dim_system), dtype=complex)
    for i in range(dim_system):
        for j in range(dim_system):
            extracted[i, j] = U[i * dim_ancilla, j * dim_ancilla]
    return extracted

P_T2 = extract_from_block(U_T2, 1)
print(f"  Extracted matrix (should be T_2(H/lambda)):")
print(f"  {P_T2}")

# Compute T_2(H/lambda) = 2*(H/lambda)^2 - I
H_normalized = H / lambda_norm
T2_expected = 2 * H_normalized @ H_normalized - np.eye(2)
print(f"\n  Expected T_2(H/lambda):")
print(f"  {T2_expected}")

error_T2 = np.linalg.norm(P_T2 - T2_expected)
print(f"\n  Error: {error_T2:.2e}")

# =============================================================
# Part 5: Two-Qubit Hamiltonian Block Encoding
# =============================================================

print("\n" + "=" * 60)
print("Part 5: Two-Qubit Ising Block Encoding")
print("=" * 60)

# H = J * Z_1 Z_2 + h * (X_1 + X_2)
J, h = 1.0, 0.5

# Construct terms
ZZ = np.kron(Z, Z)
XI = np.kron(X, I)
IX = np.kron(I, X)

H_ising = J * ZZ + h * XI + h * IX

print(f"Ising Hamiltonian (J={J}, h={h}):")
print(f"  H = {J}*ZZ + {h}*XI + {h}*IX")

# LCU decomposition
coeffs_ising = [J, h, h]
unitaries_ising = [ZZ, XI, IX]

U_ising, lambda_ising = lcu_block_encoding(coeffs_ising, unitaries_ising)

print(f"\n  1-norm lambda = {lambda_ising:.2f}")
print(f"  Number of ancilla qubits: {int(np.ceil(np.log2(len(coeffs_ising))))}")

# Verify block encoding
epsilon_ising, success_ising = verify_block_encoding(U_ising, H_ising, lambda_ising, 2)
print(f"  Block encoding error: {epsilon_ising:.2e}")
print(f"  Success: {success_ising}")

# Eigenvalues
eigvals_ising = np.linalg.eigvalsh(H_ising)
print(f"\n  Eigenvalues of H: {eigvals_ising}")

# =============================================================
# Part 6: Simulation Cost Analysis
# =============================================================

print("\n" + "=" * 60)
print("Part 6: Simulation Cost Comparison")
print("=" * 60)

def block_encoding_cost(L: int, t: float, epsilon: float,
                        lambda_norm: float) -> dict:
    """Estimate QSVT/block encoding simulation cost."""
    # QSP degree
    degree = int(np.ceil(lambda_norm * t + np.log(1/epsilon)))

    # Each query uses the block encoding once
    queries = degree

    # PREPARE cost: O(L) gates
    prepare_gates = L

    # SELECT cost: O(L) controlled operations, each O(1) for Paulis
    select_gates = L

    # Per query: PREPARE + SELECT + PREPARE†
    gates_per_query = 2 * prepare_gates + select_gates

    total_gates = queries * gates_per_query

    return {
        'queries': queries,
        'gates_per_query': gates_per_query,
        'total_gates': total_gates,
        'degree': degree
    }

def trotter_cost(L: int, t: float, epsilon: float, order: int = 2) -> dict:
    """Estimate Trotter simulation cost."""
    if order == 1:
        steps = int(np.ceil(t**2 / epsilon))
    elif order == 2:
        steps = int(np.ceil((t**3 / epsilon)**0.5))
    else:
        p = order
        steps = int(np.ceil((t**(p+1) / epsilon)**(1.0/p)))

    gates_per_step = (2 * L - 1) * (5**(order//2 - 1)) if order > 1 else L

    return {
        'steps': steps,
        'gates_per_step': gates_per_step,
        'total_gates': steps * gates_per_step
    }

# Compare for 100-term Hamiltonian
L = 100
t = 10.0
lambda_norm = L  # Assume unit coefficients

print(f"\nCost comparison for L={L} terms, t={t}:")
print("-" * 60)
print(f"{'Method':<25} {'ε=1e-3':>15} {'ε=1e-6':>15}")
print("-" * 60)

for eps in [1e-3, 1e-6]:
    qsvt = block_encoding_cost(L, t, eps, lambda_norm)
    trotter2 = trotter_cost(L, t, eps, 2)
    trotter4 = trotter_cost(L, t, eps, 4)

    if eps == 1e-3:
        print(f"{'QSVT':<25} {qsvt['total_gates']:>15,}", end="")
    else:
        print(f" {qsvt['total_gates']:>15,}")

for eps in [1e-3, 1e-6]:
    trotter2 = trotter_cost(L, t, eps, 2)
    if eps == 1e-3:
        print(f"{'Trotter (order 2)':<25} {trotter2['total_gates']:>15,}", end="")
    else:
        print(f" {trotter2['total_gates']:>15,}")

for eps in [1e-3, 1e-6]:
    trotter4 = trotter_cost(L, t, eps, 4)
    if eps == 1e-3:
        print(f"{'Trotter (order 4)':<25} {trotter4['total_gates']:>15,}", end="")
    else:
        print(f" {trotter4['total_gates']:>15,}")

# =============================================================
# Part 7: Visualization
# =============================================================

print("\n" + "=" * 60)
print("Part 7: Visualization")
print("=" * 60)

# Plot block encoding structure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Block encoding matrix visualization
ax = axes[0]
U_plot = np.abs(U_ising)
im = ax.imshow(U_plot, cmap='Blues')
ax.set_title('Block Encoding |U_H| for 2-Qubit Ising', fontsize=12)
ax.set_xlabel('Column index')
ax.set_ylabel('Row index')
plt.colorbar(im, ax=ax)

# Mark the block-encoded region
ax.add_patch(plt.Rectangle((0-0.5, 0-0.5), 4, 4, fill=False,
                            edgecolor='red', linewidth=2))
ax.text(1.5, -1, 'Block-encoded\nregion', ha='center', va='bottom', color='red')

# Cost scaling
ax = axes[1]
L_range = np.arange(10, 201, 10)
t = 10.0
eps = 1e-6

qsvt_costs = [block_encoding_cost(L, t, eps, L)['total_gates'] for L in L_range]
trotter2_costs = [trotter_cost(L, t, eps, 2)['total_gates'] for L in L_range]
trotter4_costs = [trotter_cost(L, t, eps, 4)['total_gates'] for L in L_range]

ax.semilogy(L_range, qsvt_costs, 'b-', linewidth=2, label='QSVT')
ax.semilogy(L_range, trotter2_costs, 'g--', linewidth=2, label='Trotter-2')
ax.semilogy(L_range, trotter4_costs, 'r-.', linewidth=2, label='Trotter-4')
ax.set_xlabel('Number of Hamiltonian Terms (L)', fontsize=12)
ax.set_ylabel('Total Gate Count', fontsize=12)
ax.set_title(f'Gate Scaling (t={t}, ε={eps})', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_964_block_encoding.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
print("Figure saved: day_964_block_encoding.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Block encoding | $A = \alpha \langle 0|^{\otimes a} U_A |0\rangle^{\otimes a}$ |
| PREPARE state | $V|0\rangle = \sum_j \sqrt{\alpha_j/\lambda} |j\rangle$ |
| SELECT | $\text{SELECT} = \sum_j |j\rangle\langle j| \otimes U_j$ |
| LCU | $U_H = (V^\dagger \otimes I) \cdot \text{SELECT} \cdot (V \otimes I)$ |
| 1-norm | $\lambda = \sum_j |\alpha_j|$ |
| Walk operator | $W = (2|0\rangle\langle 0|^{\otimes a} - I) \cdot U_H$ |
| QSVT complexity | $O(\lambda t + \log(1/\epsilon))$ |

### Key Takeaways

1. **Block encoding** embeds non-unitary matrices in larger unitary operations.

2. **LCU** constructs block encodings using PREPARE and SELECT subroutines.

3. **The 1-norm** $\lambda = \sum_j |\alpha_j|$ determines the simulation cost.

4. **Qubitization** creates a walk operator whose eigenphases encode the Hamiltonian eigenvalues.

5. **QSVT + Block encoding** achieves optimal Hamiltonian simulation.

6. **Trade-off:** Block encoding requires ancilla qubits but achieves better asymptotic scaling.

---

## Daily Checklist

- [ ] I can define block encoding and its parameters $(\alpha, a, \epsilon)$
- [ ] I understand how LCU constructs block encodings
- [ ] I can describe the PREPARE and SELECT operations
- [ ] I understand qubitization and the walk operator structure
- [ ] I know how QSVT uses block encoding for simulation
- [ ] I completed the computational lab

---

## Preview of Day 965

Tomorrow we apply quantum simulation to **chemistry and materials science**. We will:

- Learn second quantization and fermionic operators
- Apply Jordan-Wigner transformation to map fermions to qubits
- Understand electronic structure Hamiltonians
- Implement VQE for molecular ground states
- Explore applications in drug discovery and materials design

Chemistry simulation is the most promising near-term application of quantum computers!

---

*"The underlying physical laws necessary for the mathematical theory of a large part of physics and the whole of chemistry are thus completely known."*
*— Paul Dirac, 1929*

---

**Next:** [Day_965_Saturday.md](Day_965_Saturday.md) - Chemistry and Materials Simulation
