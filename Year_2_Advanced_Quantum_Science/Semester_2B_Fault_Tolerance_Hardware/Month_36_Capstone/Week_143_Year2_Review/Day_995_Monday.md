# Day 995: Semester 2A Review - QEC Fundamentals

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Core Review: Classical to Quantum Error Correction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Qualifying Exam Problem Practice |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Synthesis and Concept Mapping |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 995, you will be able to:

1. **Summarize** the complete hierarchy from classical to quantum error correction
2. **Derive** the Knill-Laflamme conditions and explain their significance
3. **Analyze** quantum error channels and their discretization
4. **Construct** the 3-qubit and 9-qubit codes from first principles
5. **Compare** error detection versus error correction capabilities
6. **Solve** qualifying-level problems on QEC fundamentals

---

## Core Review Content

### 1. Classical Error Correction Foundations

#### The Communication Problem
Shannon's noisy channel coding theorem (1948) established that reliable communication is possible over noisy channels up to channel capacity $C$:

$$C = \max_{p(x)} I(X;Y) = \max_{p(x)} \left[ H(Y) - H(Y|X) \right]$$

**Key Classical Codes:**

| Code | Parameters | Distance | Rate |
|------|------------|----------|------|
| Repetition | [n, 1, n] | n | 1/n |
| Hamming | [7, 4, 3] | 3 | 4/7 |
| Reed-Solomon | [n, k, n-k+1] | n-k+1 | k/n |

#### Hamming Distance and Error Correction

For a code with minimum distance $d$:
- **Detect** up to $d-1$ errors
- **Correct** up to $\lfloor(d-1)/2\rfloor$ errors

$$\boxed{d = \min_{c_1 \neq c_2} d_H(c_1, c_2)}$$

---

### 2. Quantum Error Models

#### The Three Fundamental Errors

**Bit-flip (X error):**
$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

**Phase-flip (Z error):**
$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$$

**Bit-phase-flip (Y error):**
$$Y = iXZ, \quad Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

#### Depolarizing Channel

The most common error model:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Kraus operators: $\{K_0 = \sqrt{1-p}I, K_1 = \sqrt{p/3}X, K_2 = \sqrt{p/3}Y, K_3 = \sqrt{p/3}Z\}$

#### Error Discretization Theorem

**Critical insight:** Continuous errors can be corrected discretely!

If a code corrects errors $\{E_a\}$, it also corrects any linear combination:
$$E = \sum_a c_a E_a$$

This works because measurement projects the error into a discrete subspace.

---

### 3. The Knill-Laflamme Conditions

#### Statement

A quantum code $\mathcal{C}$ with projector $P$ can correct error set $\{E_a\}$ if and only if:

$$\boxed{P E_a^\dagger E_b P = C_{ab} P}$$

where $C_{ab}$ is a Hermitian matrix (depends only on error operators, not code states).

#### Equivalent Formulation

For orthonormal codewords $\{|c_i\rangle\}$:

$$\langle c_i | E_a^\dagger E_b | c_j \rangle = C_{ab} \delta_{ij}$$

**Physical interpretation:**
- Errors map code space to orthogonal subspaces (correctable)
- Or errors act identically on all codewords (detectable but not distinguishable)

#### Derivation Sketch

1. After error $E_a$, state becomes $E_a|\psi\rangle$
2. Syndrome measurement projects to error subspace
3. For correction: different errors must be distinguishable by syndrome
4. This requires $\langle c_i|E_a^\dagger E_b|c_j\rangle \propto \delta_{ij}$

---

### 4. Three-Qubit Codes

#### Bit-Flip Code

**Encoding:**
$$|0\rangle_L = |000\rangle, \quad |1\rangle_L = |111\rangle$$

**Stabilizer generators:** $\{Z_1Z_2, Z_2Z_3\}$

**Error syndromes:**

| Error | $Z_1Z_2$ | $Z_2Z_3$ | Syndrome |
|-------|----------|----------|----------|
| None | +1 | +1 | (0,0) |
| $X_1$ | -1 | +1 | (1,0) |
| $X_2$ | -1 | -1 | (1,1) |
| $X_3$ | +1 | -1 | (0,1) |

**Parameters:** [[3, 1, 1]] for bit-flip errors only

#### Phase-Flip Code

**Encoding:**
$$|0\rangle_L = |{+}{+}{+}\rangle, \quad |1\rangle_L = |{-}{-}{-}\rangle$$

where $|{\pm}\rangle = (|0\rangle \pm |1\rangle)/\sqrt{2}$

**Stabilizer generators:** $\{X_1X_2, X_2X_3\}$

---

### 5. Shor's Nine-Qubit Code

#### Construction

Combine bit-flip and phase-flip codes:

$$|0\rangle_L = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$
$$|1\rangle_L = \frac{1}{2\sqrt{2}}(|000\rangle - |111\rangle)^{\otimes 3}$$

**Expanded:**
$$|0\rangle_L = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)$$

#### Stabilizer Generators

**Bit-flip detection (6 generators):**
$$Z_1Z_2, Z_2Z_3, Z_4Z_5, Z_5Z_6, Z_7Z_8, Z_8Z_9$$

**Phase-flip detection (2 generators):**
$$X_1X_2X_3X_4X_5X_6, \quad X_4X_5X_6X_7X_8X_9$$

**Parameters:** [[9, 1, 3]]

---

### 6. Code Distance and Parameters

#### Quantum Code Notation

$$[[n, k, d]]$$

- $n$: number of physical qubits
- $k$: number of logical qubits
- $d$: code distance

#### Quantum Singleton Bound

$$k \leq n - 2(d-1)$$

Equivalently: $n \geq 2d + k - 2$

For $k=1$: need at least $n = 2d - 1$ qubits for distance $d$

#### Quantum Hamming Bound

$$2^k \sum_{j=0}^{t} \binom{n}{j} 3^j \leq 2^n$$

where $t = \lfloor(d-1)/2\rfloor$

**Perfect codes** achieve equality (5-qubit code is perfect).

---

### 7. Error Detection vs. Error Correction

| Capability | Distance Required | Errors Handled |
|------------|-------------------|----------------|
| Detect $t$ errors | $d \geq t + 1$ | Know error occurred |
| Correct $t$ errors | $d \geq 2t + 1$ | Can identify and fix |

**Key trade-off:** Detection is cheaper but requires discarding data

---

## Concept Map: QEC Fundamentals

```
Classical Codes (Shannon 1948)
        │
        ▼
Hamming Distance ──────────────────────┐
        │                              │
        ▼                              ▼
Quantum No-Cloning ──► Cannot copy ──► Need clever encoding
        │
        ▼
Discretization Theorem ──► Only need Pauli corrections
        │
        ▼
Knill-Laflamme Conditions ◄──── Mathematical foundation
        │
        ├──► 3-Qubit Codes (detect one type)
        │
        ├──► 9-Qubit Code (detect all types)
        │
        └──► General [[n,k,d]] codes
```

---

## Qualifying Exam Practice Problems

### Problem 1: Knill-Laflamme Application (20 points)

**Question:** Consider a code with codewords $|0_L\rangle = |00\rangle$ and $|1_L\rangle = |11\rangle$. Determine which of the following error sets can be corrected:

(a) $\{I, X_1\}$
(b) $\{I, Z_1\}$
(c) $\{I, X_1, X_2\}$
(d) $\{I, Z_1Z_2\}$

**Solution:**

Apply Knill-Laflamme: $P E_a^\dagger E_b P = C_{ab} P$

**(a) $\{I, X_1\}$:**
- $\langle 0_L|I \cdot I|0_L\rangle = 1$, $\langle 1_L|I \cdot I|1_L\rangle = 1$ (consistent)
- $\langle 0_L|X_1^\dagger X_1|0_L\rangle = 1$, $\langle 1_L|X_1^\dagger X_1|1_L\rangle = 1$ (consistent)
- $\langle 0_L|I \cdot X_1|0_L\rangle = \langle 00|X_1|00\rangle = 0$
- $\langle 1_L|I \cdot X_1|1_L\rangle = \langle 11|X_1|11\rangle = 0$ (consistent)

**Answer: (a) CORRECTABLE**

**(b) $\{I, Z_1\}$:**
- $\langle 0_L|Z_1|0_L\rangle = \langle 00|Z_1|00\rangle = +1$
- $\langle 1_L|Z_1|1_L\rangle = \langle 11|Z_1|11\rangle = -1$ (INCONSISTENT!)

**Answer: (b) NOT correctable** ($Z_1$ distinguishes logical states)

**(c) $\{I, X_1, X_2\}$:**
- $\langle 0_L|X_1 X_2|0_L\rangle = \langle 00|X_1X_2|00\rangle = \langle 10|X_2|00\rangle = 0$
- But $\langle 0_L|X_1|1_L\rangle = \langle 00|X_1|11\rangle = 0$ (need to check all)

After checking: $X_1|0_L\rangle = |10\rangle$, $X_2|0_L\rangle = |01\rangle$ - orthogonal subspaces

**Answer: (c) CORRECTABLE**

**(d) $\{I, Z_1Z_2\}$:**
- $Z_1Z_2|00\rangle = |00\rangle$, $Z_1Z_2|11\rangle = |11\rangle$
- $Z_1Z_2$ acts as $+I$ on code space!

**Answer: (d) CORRECTABLE** (trivially - error is undetectable)

---

### Problem 2: Error Channel Analysis (20 points)

**Question:** For the amplitude damping channel with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

(a) Verify $\sum_i K_i^\dagger K_i = I$
(b) Find the output state for input $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$
(c) Explain why standard QEC codes don't fully correct this error

**Solution:**

**(a)**
$$K_0^\dagger K_0 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix}$$
$$K_1^\dagger K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix}$$
$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \checkmark$$

**(b)** Input: $\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}$

Output: $\mathcal{E}(\rho) = K_0\rho K_0^\dagger + K_1\rho K_1^\dagger$

$$= \begin{pmatrix} |\alpha|^2 + \gamma|\beta|^2 & \alpha\beta^*\sqrt{1-\gamma} \\ \alpha^*\beta\sqrt{1-\gamma} & (1-\gamma)|\beta|^2 \end{pmatrix}$$

**(c)** Amplitude damping is **non-Pauli**: $K_1$ is not proportional to a Pauli operator. Standard stabilizer codes assume errors from Pauli group. The $|1\rangle \to |0\rangle$ transition can't be expressed as a linear combination of $\{I, X, Y, Z\}$ acting on the state.

---

### Problem 3: Code Construction (25 points)

**Question:** Design a [[4, 1, 2]] code that encodes 1 logical qubit in 4 physical qubits with distance 2.

(a) Write the stabilizer generators
(b) Give explicit expressions for $|0_L\rangle$ and $|1_L\rangle$
(c) Show this code can detect any single-qubit error

**Solution:**

**(a) Stabilizer generators:**
$$S = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle$$

With 2 generators and 4 qubits: $k = 4 - 2 = 2$ initially, but we need to add structure.

Better approach - use stabilizers:
$$g_1 = Z_1Z_2, \quad g_2 = Z_2Z_3, \quad g_3 = Z_3Z_4$$

No wait - this gives [[4,1,1]]. For distance 2:

$$g_1 = X_1X_2, \quad g_2 = Z_1Z_2, \quad g_3 = X_1X_3Z_2Z_4$$

Actually, use the [[4,2,2]] code and project:
$$g_1 = X_1X_2X_3X_4, \quad g_2 = Z_1Z_2Z_3Z_4, \quad g_3 = X_1Z_2Z_3X_4$$

**(b) Codewords:**
$$|0_L\rangle = \frac{1}{2}(|0000\rangle + |0011\rangle + |1100\rangle + |1111\rangle)$$
$$|1_L\rangle = \frac{1}{2}(|0101\rangle + |0110\rangle + |1001\rangle + |1010\rangle)$$

**(c) Single-qubit error detection:**
For any single-qubit Pauli $P_i$, at least one stabilizer anticommutes:
- $X_1$ anticommutes with $g_2$
- $Z_1$ anticommutes with $g_1$

All single-qubit errors produce non-trivial syndrome, hence detectable.

---

### Problem 4: Threshold Analysis (15 points)

**Question:** A concatenated code has threshold $p_{th} = 1\%$.

(a) If physical error rate is $p = 0.1\%$, what is the logical error rate after 3 levels?
(b) How many levels needed to achieve $p_L < 10^{-15}$?

**Solution:**

**(a)** Using $p_L^{(k)} \leq p_{th}(p/p_{th})^{2^k}$:

$p/p_{th} = 0.001/0.01 = 0.1$

Level 1: $p_L^{(1)} = 0.01 \times 0.1^2 = 10^{-4}$
Level 2: $p_L^{(2)} = 0.01 \times 0.1^4 = 10^{-6}$
Level 3: $p_L^{(3)} = 0.01 \times 0.1^8 = 10^{-10}$

**(b)** Need: $0.01 \times 0.1^{2^k} < 10^{-15}$
$$0.1^{2^k} < 10^{-13}$$
$$2^k \cdot \log(0.1) < -13$$
$$2^k > 13$$
$$k > \log_2(13) \approx 3.7$$

**Answer: k = 4 levels needed**

---

### Problem 5: Error Discretization (20 points)

**Question:** Prove that if a code corrects the error set $\{I, X, Z, XZ\}$ on a given qubit, it also corrects any rotation error:
$$R_{\vec{n}}(\theta) = \cos(\theta/2)I - i\sin(\theta/2)(n_x X + n_y Y + n_z Z)$$

**Solution:**

Any single-qubit unitary can be written as:
$$U = aI + bX + cY + dZ$$
where $a, b, c, d \in \mathbb{C}$ with $|a|^2 + |b|^2 + |c|^2 + |d|^2 = 1$.

Since $Y = iXZ$, we have:
$$U = aI + bX + ic \cdot XZ + dZ = aI + bX + dZ + (ic)XZ$$

By linearity of the Knill-Laflamme conditions:

If $PE_a^\dagger E_b P = C_{ab}P$ for $E_a, E_b \in \{I, X, Z, XZ\}$, then for:
$$E = \sum_j \alpha_j E_j$$

we have:
$$P(\sum_i \alpha_i^* E_i^\dagger)(\sum_j \alpha_j E_j)P = \sum_{i,j} \alpha_i^*\alpha_j C_{ij} P$$

which still satisfies Knill-Laflamme with $C'_{ab}$ depending on coefficients.

**Key insight:** Syndrome measurement projects the continuous error onto one of the discrete error operators, then standard correction applies.

---

## Computational Review

```python
"""
Day 995 Computational Review: QEC Fundamentals
Semester 2A Review - Week 143
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Knill-Laflamme Condition Verification
# =============================================================================

print("=" * 70)
print("Part 1: Knill-Laflamme Condition Checker")
print("=" * 70)

# Define Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def tensor(*ops):
    """Tensor product of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def check_knill_laflamme(codewords, errors):
    """
    Check if code satisfies Knill-Laflamme conditions for given errors.

    Args:
        codewords: list of column vectors (codewords)
        errors: list of error operators

    Returns:
        (is_correctable, C_matrix)
    """
    k = len(codewords)
    m = len(errors)

    # Build C matrix
    C = np.zeros((m, m), dtype=complex)

    for a, Ea in enumerate(errors):
        for b, Eb in enumerate(errors):
            # Check condition for all pairs of codewords
            values = []
            for i, ci in enumerate(codewords):
                for j, cj in enumerate(codewords):
                    # <c_i|E_a^dag E_b|c_j>
                    val = ci.conj().T @ (Ea.conj().T @ Eb) @ cj
                    if i == j:
                        values.append(('diag', val[0, 0]))
                    else:
                        values.append(('offdiag', val[0, 0]))

            # Check consistency
            diag_vals = [v[1] for v in values if v[0] == 'diag']
            offdiag_vals = [v[1] for v in values if v[0] == 'offdiag']

            # All diagonal should be same
            if not np.allclose(diag_vals, diag_vals[0]):
                return False, None

            # All off-diagonal should be zero
            if not np.allclose(offdiag_vals, 0):
                return False, None

            C[a, b] = diag_vals[0]

    return True, C

# Test: 3-qubit bit-flip code
print("\n3-Qubit Bit-Flip Code:")
print("-" * 40)

# Codewords
c0 = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])  # |000>
c1 = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])  # |111>

# Errors: I, X1, X2, X3
I3 = tensor(I, I, I)
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
X3 = tensor(I, I, X)

errors = [I3, X1, X2, X3]
is_corr, C = check_knill_laflamme([c0, c1], errors)
print(f"Corrects {{I, X1, X2, X3}}: {is_corr}")
if is_corr:
    print(f"C matrix:\n{np.real(C)}")

# Test phase errors
Z1 = tensor(Z, I, I)
errors_z = [I3, Z1]
is_corr_z, _ = check_knill_laflamme([c0, c1], errors_z)
print(f"\nCorrects {{I, Z1}}: {is_corr_z}")

# =============================================================================
# Part 2: Depolarizing Channel Simulation
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Depolarizing Channel Simulation")
print("=" * 70)

def depolarizing_channel(rho, p):
    """Apply depolarizing channel with error probability p."""
    return (1 - p) * rho + (p/3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)

def fidelity(rho, sigma):
    """Compute fidelity between two density matrices."""
    sqrt_rho = linalg.sqrtm(rho)
    return np.real(np.trace(linalg.sqrtm(sqrt_rho @ sigma @ sqrt_rho))**2)

# Initial pure state |+>
psi = np.array([[1], [1]]) / np.sqrt(2)
rho_init = psi @ psi.conj().T

# Apply channel for various error rates
p_values = np.linspace(0, 1, 50)
fidelities = []

for p in p_values:
    rho_out = depolarizing_channel(rho_init, p)
    fidelities.append(fidelity(rho_init, rho_out))

plt.figure(figsize=(8, 5))
plt.plot(p_values, fidelities, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Classical limit')
plt.xlabel('Error probability p', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.title('Depolarizing Channel: Fidelity vs Error Rate', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('day_995_depolarizing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved depolarizing channel plot")

# =============================================================================
# Part 3: Code Distance Verification
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Shor Code Syndrome Table")
print("=" * 70)

def compute_syndrome(state, stabilizers):
    """Compute syndrome for a state given stabilizers."""
    syndrome = []
    for stab in stabilizers:
        # Eigenvalue +1 or -1
        exp_val = state.conj().T @ stab @ state
        syndrome.append(int(np.real(exp_val[0, 0]) < 0))
    return tuple(syndrome)

# Shor code stabilizers (simplified for 2 blocks)
# For full 9-qubit code, would have 8 stabilizers

print("\nShor 9-qubit code protects against arbitrary single-qubit errors")
print("Syndromes allow identification of error type and location")
print("\nSyndrome structure:")
print("- 6 generators detect bit-flip location within blocks")
print("- 2 generators detect phase-flip between blocks")

# =============================================================================
# Part 4: Threshold Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Concatenation Threshold Visualization")
print("=" * 70)

def logical_error_rate(p, p_th, levels):
    """Compute logical error rate after concatenation."""
    ratio = p / p_th
    return p_th * (ratio ** (2**levels))

p_th = 0.01
p_physical = np.linspace(0.0001, 0.02, 100)
levels = [1, 2, 3, 4]

plt.figure(figsize=(10, 6))
plt.loglog(p_physical, p_physical, 'k--', label='No encoding', linewidth=2)

for L in levels:
    p_logical = [logical_error_rate(p, p_th, L) for p in p_physical]
    plt.loglog(p_physical, p_logical, label=f'{L} level(s)', linewidth=2)

plt.axvline(x=p_th, color='r', linestyle=':', label=f'Threshold = {p_th}')
plt.xlabel('Physical error rate p', fontsize=12)
plt.ylabel('Logical error rate', fontsize=12)
plt.title('Concatenated Code Performance', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0.0001, 0.02])
plt.ylim([1e-20, 1])
plt.savefig('day_995_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved threshold visualization")

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("QEC Fundamentals Summary")
print("=" * 70)

print("""
Key Results:
1. Quantum errors discretize to Pauli basis
2. Knill-Laflamme: P E_a^dag E_b P = C_ab P
3. [[n,k,d]] notation: n physical, k logical, distance d
4. Threshold theorem enables arbitrarily reliable computation
5. Below threshold: p_L ~ (p/p_th)^(2^levels)
""")

print("Review complete!")
```

---

## Summary Tables

### QEC Hierarchy

| Level | Concept | Key Result |
|-------|---------|------------|
| Classical | Shannon coding | Reliable comm. up to capacity |
| Quantum challenge | No-cloning | Cannot copy to protect |
| Resolution | Error discretization | Pauli errors suffice |
| Theory | Knill-Laflamme | Necessary and sufficient conditions |
| Practice | [[n,k,d]] codes | Systematic constructions |
| Scalability | Threshold theorem | Polynomial overhead possible |

### Important Codes

| Code | Parameters | Errors Corrected |
|------|------------|------------------|
| 3-qubit bit-flip | [[3,1,1]]* | Single X |
| 3-qubit phase-flip | [[3,1,1]]* | Single Z |
| Shor 9-qubit | [[9,1,3]] | Any single-qubit |
| Steane 7-qubit | [[7,1,3]] | Any single-qubit |
| 5-qubit perfect | [[5,1,3]] | Any single-qubit (optimal) |

*Distance for specified error type only

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $d = 2t + 1$ | Distance to correct $t$ errors |
| $n \geq 2d + k - 2$ | Singleton bound |
| $p_L \leq p_{th}(p/p_{th})^{2^k}$ | Concatenation scaling |

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] Can explain why quantum error correction is possible despite no-cloning
- [ ] Can derive Knill-Laflamme conditions from first principles
- [ ] Can explain error discretization theorem
- [ ] Understand difference between detection and correction

### Problem Solving
- [ ] Can verify if a code corrects a given error set
- [ ] Can compute syndromes for simple codes
- [ ] Can analyze error channels using Kraus operators
- [ ] Can calculate logical error rates for concatenated codes

### Connections
- [ ] Can relate classical codes to quantum codes
- [ ] Understand how stabilizer formalism emerges from Knill-Laflamme
- [ ] Can explain role of QEC in fault-tolerant computation

---

## Preview: Day 996

Tomorrow we review **Stabilizer Formalism and Topological Codes**, covering:
- Pauli group structure and stabilizer subgroups
- CSS code construction from classical codes
- Gottesman-Knill theorem
- Toric code and anyonic excitations
- Topological protection mechanisms

---

*"The remarkable thing is not that quantum error correction exists, but that it exists despite no-cloning."*
--- Peter Shor

---

**Next:** [Day_996_Tuesday.md](Day_996_Tuesday.md) - Stabilizer & Topological Codes Review
