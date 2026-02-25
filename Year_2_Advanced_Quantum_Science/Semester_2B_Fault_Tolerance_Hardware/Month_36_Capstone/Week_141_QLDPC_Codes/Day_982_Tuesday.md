# Day 982: Quantum LDPC Code Construction

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | CSS Construction & Hypergraph Products |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Quantum BP & Degeneracy |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: qLDPC Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 982, you will be able to:

1. Construct CSS quantum codes from classical LDPC codes
2. Derive the hypergraph product code construction (Tillich-Zémor)
3. Calculate qLDPC code parameters from classical code inputs
4. Explain degeneracy and its impact on quantum decoding
5. Adapt belief propagation for quantum error correction
6. Implement basic qLDPC code construction and analysis

---

## Core Content

### 1. From Classical to Quantum LDPC

The transition from classical to quantum LDPC codes requires handling two fundamental differences:

**Classical vs Quantum:**

| Aspect | Classical LDPC | Quantum LDPC |
|--------|----------------|--------------|
| Errors | Bit flips | Pauli X, Y, Z |
| Redundancy | Parity checks | Stabilizer generators |
| Constraint | $Hc = 0$ | Commutation: $[S_i, S_j] = 0$ |
| Degeneracy | None | Multiple errors $\to$ same syndrome |
| Decoding | Maximize likelihood | Equivalence class selection |

**Key Challenge:** Quantum codes must handle X and Z errors simultaneously while respecting the commutation constraint between stabilizers.

---

### 2. CSS Code Construction

Calderbank-Shor-Steane (CSS) codes provide the natural framework for quantum LDPC.

**CSS Code Definition:**

Given two classical codes $C_1$ and $C_2$ with parity-check matrices $H_1$ and $H_2$ satisfying:

$$\boxed{C_2^\perp \subseteq C_1 \iff H_1 H_2^T = 0}$$

The CSS code has:
- **X-stabilizers:** Rows of $H_X = H_1$ (detect Z errors)
- **Z-stabilizers:** Rows of $H_Z = H_2$ (detect X errors)

**Stabilizer Group:**
$$S = \langle X^{h_1^{(i)}}, Z^{h_2^{(j)}} \rangle$$

where $h_1^{(i)}$ is the $i$-th row of $H_1$ and $h_2^{(j)}$ is the $j$-th row of $H_2$.

**Code Parameters:**

$$[[n, k, d]] = [[n, k_1 - (n - k_2), \min(d_1, d_2)]]$$

where:
- $n$: number of physical qubits
- $k_1 = n - \text{rank}(H_1)$: dimension of $C_1$
- $k_2 = n - \text{rank}(H_2)$: dimension of $C_2$
- $d_1, d_2$: minimum distances

For **self-orthogonal** codes ($C \subseteq C^\perp$, equivalently $H H^T = 0$):
$$[[n, k, d]] = [[n, n - 2\text{rank}(H), d_C]]$$

---

### 3. The Hypergraph Product Construction

The breakthrough for quantum LDPC codes came from Tillich and Zémor's hypergraph product construction (2014).

**Construction:**

Given classical codes $C_1[n_1, k_1, d_1]$ and $C_2[n_2, k_2, d_2]$ with parity-check matrices $H_1$ ($m_1 \times n_1$) and $H_2$ ($m_2 \times n_2$):

**Physical Qubits:**
$$n = n_1 n_2 + m_1 m_2$$

Arranged in two "blocks":
- Block A: $n_1 \times n_2$ qubits (identified with $H_1$ columns $\otimes$ $H_2$ columns)
- Block B: $m_1 \times m_2$ qubits (identified with $H_1$ rows $\otimes$ $H_2$ rows)

**Stabilizer Generators:**

$$\boxed{H_X = [H_1 \otimes I_{n_2}, \quad I_{m_1} \otimes H_2^T]}$$
$$\boxed{H_Z = [I_{n_1} \otimes H_2, \quad H_1^T \otimes I_{m_2}]}$$

**Verification of CSS Constraint:**
$$H_X H_Z^T = (H_1 \otimes I)(I \otimes H_2^T) + (I \otimes H_2^T)(H_1^T \otimes I)$$
$$= H_1 \otimes H_2^T + H_2^T \otimes H_1^T = 0 \pmod{2}$$

(since each term equals the other in $\mathbb{F}_2$)

**Code Parameters:**

$$[[n_1 n_2 + m_1 m_2, \, k_1 k_2, \, \min(d_1, d_2)]]$$

---

### 4. Hypergraph Product Example

**Example:** Take the $[7, 4, 3]$ Hamming code with:

$$H = \begin{pmatrix}
1 & 1 & 1 & 0 & 1 & 0 & 0 \\
1 & 1 & 0 & 1 & 0 & 1 & 0 \\
1 & 0 & 1 & 1 & 0 & 0 & 1
\end{pmatrix}$$

Product with itself:

- $n_1 = n_2 = 7$, $m_1 = m_2 = 3$
- $k_1 = k_2 = 4$, $d_1 = d_2 = 3$

**Resulting qLDPC code:**

$$n = 7 \times 7 + 3 \times 3 = 49 + 9 = 58$$
$$k = 4 \times 4 = 16$$
$$d \geq \min(3, 3) = 3$$

$$\boxed{[[58, 16, 3]]}$$

**Rate:** $k/n = 16/58 \approx 0.28$

This is much better than surface codes of similar distance!

---

### 5. Weight and Degree Analysis

**Stabilizer Weight:**

For hypergraph product of $(w_c, w_r)$-regular LDPC codes:
- X-stabilizer weight: $w_r + w_c$ (from row of $H_X$)
- Z-stabilizer weight: $w_r + w_c$ (from row of $H_Z$)

**Qubit Participation:**

Each qubit participates in:
- Block A qubits: $w_c + w_c$ stabilizers (column weights from both $H_1$ and $H_2$)
- Block B qubits: $w_r + w_r$ stabilizers (row weights)

**Locality:**

Unlike surface codes (weight-4 local stabilizers), hypergraph product codes typically have:
- Higher weight stabilizers
- Non-local connectivity
- Complex Tanner graph structure

---

### 6. Degeneracy in Quantum Codes

**Definition:** A quantum code is **degenerate** if there exist distinct errors $E_1 \neq E_2$ such that $E_1^\dagger E_2 \in S$ (stabilizer group).

**Equivalence Classes:**

Errors are equivalent if they differ by a stabilizer:
$$E_1 \sim E_2 \iff E_1 E_2^\dagger \in S$$

**Impact on Decoding:**

Classical BP finds the most likely error. Quantum BP must find the most likely **equivalence class**.

$$P(\text{class } [E]) = \sum_{E' \in [E]} P(E')$$

**Example:**

In the $[[5, 1, 3]]$ code, if $E_1 = X_1 X_2$ and $E_2 = X_3 X_4 X_5$, and $X_1 X_2 X_3 X_4 X_5 \in S$, then $E_1 \sim E_2$.

Correcting either $E_1$ or $E_2$ gives the same logical result!

---

### 7. Quantum Belief Propagation

**Challenge:** Standard BP returns a single error estimate, not an equivalence class.

**Approaches:**

**1. Syndrome-Based BP:**

Run classical BP on syndrome:
$$\sigma_X = H_Z \cdot e_X \pmod{2}$$
$$\sigma_Z = H_X \cdot e_Z \pmod{2}$$

**2. Degeneracy-Aware BP:**

Modify messages to sum over equivalent errors:

$$P(e_i = 1 | \sigma) = \sum_{e : e_i = 1, \, H \cdot e = \sigma} P(e)$$

**3. OSD Post-Processing:**

After BP converges (or fails), use Ordered Statistics Decoding:
1. Order bits by reliability
2. Fix most reliable bits
3. Solve linear system for remaining bits
4. Check multiple candidates

**4. Neural Network Enhancement:**

Train neural network to select correct equivalence class from BP marginals.

---

### 8. The Chain Complex Perspective

Modern qLDPC constructions use **chain complexes** from algebraic topology.

**Chain Complex:**
$$\cdots \xrightarrow{\partial_3} C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0 \xrightarrow{\partial_0} \cdots$$

where $\partial_i \circ \partial_{i+1} = 0$ (boundary of boundary is zero).

**Correspondence:**
- $C_1$: Qubits (edges)
- $C_0$: X-checks (vertices)
- $C_2$: Z-checks (faces)
- $\partial_1 = H_X^T$: X-check incidence
- $\partial_2 = H_Z$: Z-check incidence

**CSS Constraint as Topology:**
$$H_X H_Z^T = 0 \iff \partial_1 \partial_2 = 0$$

This is automatically satisfied in a chain complex!

**Code Parameters from Homology:**
- $k = \dim H_1 = \dim \ker \partial_1 - \dim \text{im } \partial_2$
- $d_X = $ minimum weight of non-trivial 1-cycle (X logical)
- $d_Z = $ minimum weight of non-trivial 1-cocycle (Z logical)

---

## Practical Applications

### qLDPC for Near-Term Devices

**Current Limitations:**
- Non-locality requires all-to-all connectivity
- High-weight stabilizers need more ancillas
- Complex syndrome extraction circuits

**Proposed Solutions:**
1. **Modular architectures:** Groups of qubits with inter-module links
2. **3D hardware:** Stack layers with vertical connections
3. **Photonic systems:** Long-range entanglement naturally available
4. **Neutral atoms:** Reconfigurable arrays with transport

**Intermediate Codes:**

Some qLDPC codes have partial locality:
- **Fiber bundle codes:** 3D local with good parameters
- **Floquet codes:** Dynamic stabilizers with better locality
- **Bivariate bicycle codes:** Recent IBM focus, moderate non-locality

---

## Worked Examples

### Example 1: CSS Code from Self-Orthogonal LDPC

**Problem:** Verify that the following matrix defines a valid CSS code:

$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0
\end{pmatrix}$$

**Solution:**

**Step 1: Check self-orthogonality**

$$H H^T = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 1 \\
1 & 0 & 0 \\
1 & 1 & 1 \\
1 & 1 & 0 \\
0 & 1 & 1 \\
0 & 1 & 0
\end{pmatrix}$$

Computing each entry modulo 2:
- $(1,1)$: $1+1+1+1 = 4 \equiv 0$
- $(1,2)$: $0+0+1+1 = 2 \equiv 0$
- $(1,3)$: $1+0+1+0 = 2 \equiv 0$
- $(2,2)$: $0+0+1+1+1+1 = 4 \equiv 0$
- $(2,3)$: $0+0+1+0+1+0 = 2 \equiv 0$
- $(3,3)$: $1+0+1+0+1+0 = 3 \equiv 1$ **NOT zero!**

$$H H^T = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix} \neq 0$$

**This matrix is NOT self-orthogonal and cannot define a CSS code with $H_X = H_Z = H$.**

**Step 2: Find a valid modification**

Modify row 3 to ensure orthogonality:
$$H' = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 1 \\
1 & 1 & 0 & 0 & 1 & 1
\end{pmatrix}$$

Now verify: Row 3 has weight 4, and $\langle r_1, r_3 \rangle = 1+1+0+0 = 2 \equiv 0$, etc.

---

### Example 2: Hypergraph Product Parameters

**Problem:** Compute the parameters of the hypergraph product of a $[15, 11, 3]$ code with itself.

**Solution:**

**Classical code:** $[n_1, k_1, d_1] = [15, 11, 3]$

Parity-check matrix: $m_1 = n_1 - k_1 = 4$ rows

**Hypergraph product with itself:**

$$n = n_1 \cdot n_2 + m_1 \cdot m_2 = 15 \times 15 + 4 \times 4 = 225 + 16 = 241$$

$$k = k_1 \cdot k_2 = 11 \times 11 = 121$$

$$d \geq \min(d_1, d_2) = 3$$

**Result:** $[[241, 121, \geq 3]]$

**Rate:** $k/n = 121/241 \approx 0.502$ (about 50%!)

**Comparison with surface code at $d=3$:**
- Surface code: $[[9, 1, 3]]$, rate = 0.11
- This qLDPC: $[[241, 121, 3]]$, rate = 0.50

The qLDPC encodes **121x more logical qubits** for the same distance!

---

### Example 3: Degeneracy Analysis

**Problem:** In a $[[7, 1, 3]]$ Steane code, show that single-qubit X errors on qubits 1, 2, and 4 are degenerate.

**Solution:**

The Steane code has X-stabilizers:
$$S_1^X = X_1 X_3 X_5 X_7$$
$$S_2^X = X_2 X_3 X_6 X_7$$
$$S_3^X = X_4 X_5 X_6 X_7$$

**Single-qubit errors:**
- $E_1 = X_1$: syndrome from Z-checks
- $E_2 = X_2$: different syndrome
- $E_4 = X_4$: different syndrome

These have **different syndromes**, so they are NOT degenerate.

**Degenerate example in Steane code:**

Consider $E_a = X_1 X_3 X_5 X_7$ (which equals $S_1^X$).

Error $E_a$ has syndrome 0 (it's a stabilizer).

The identity $I$ also has syndrome 0.

$E_a$ and $I$ are degenerate: applying $E_a$ to the code space returns the same state!

**Practical degeneracy:** Weight-3 errors:
- $E = X_1 X_2 X_3$: some syndrome $\sigma$
- $E' = E \cdot S_1 = X_2 X_5 X_7$: same syndrome $\sigma$!

Both corrections are valid!

---

## Practice Problems

### Level 1: Direct Application

1. **CSS Parameters:** A CSS code uses $H_X$ with 50 rows and $H_Z$ with 30 rows on 100 qubits. What is the number of logical qubits (assuming full rank)?

2. **Hypergraph Product:** Compute $n$ for the hypergraph product of $[7, 4, 3]$ with $[15, 11, 3]$.

3. **Stabilizer Weight:** If both input codes to hypergraph product are (3, 6)-regular, what is the maximum weight of stabilizers in the resulting qLDPC code?

### Level 2: Intermediate

4. **Homology:** Explain why the number of logical qubits equals $k_1 \cdot k_2$ in the hypergraph product, using the chain complex perspective.

5. **Syndrome Decoding:** A qLDPC code has syndrome $\sigma_X = (1, 0, 1, 1)$. The decoder returns error $e_X = (0, 1, 0, 1, 1, 0, 0, 0)$. How do you verify this is correct?

6. **Rate Comparison:** Derive the rate $k/n$ of the hypergraph product in terms of the input code rates $R_1$ and $R_2$.

### Level 3: Challenging

7. **Distance Lower Bound:** Prove that the distance of the hypergraph product is at least $\min(d_1, d_2)$ by analyzing logical operators.

8. **BP on qLDPC:** Describe how to modify classical BP for the X-error channel on a CSS code. What changes for a depolarizing channel?

9. **Fiber Bundle Codes:** Research and summarize how fiber bundle codes achieve 3D locality while maintaining linear distance. What is the trade-off?

---

## Computational Lab

### Objective
Implement hypergraph product code construction and analyze resulting qLDPC parameters.

```python
"""
Day 982 Computational Lab: Quantum LDPC Code Construction
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: CSS Code Verification
# =============================================================================

print("=" * 70)
print("Part 1: CSS Code Construction and Verification")
print("=" * 70)

def check_css_valid(Hx, Hz):
    """
    Check if Hx and Hz define a valid CSS code.

    Returns: (valid, message)
    """
    # CSS constraint: Hx @ Hz.T = 0 mod 2
    product = (Hx @ Hz.T) % 2

    if np.sum(product) == 0:
        return True, "Valid CSS code: Hx @ Hz.T = 0"
    else:
        nonzero = np.sum(product)
        return False, f"Invalid: Hx @ Hz.T has {nonzero} non-zero entries"

def css_parameters(Hx, Hz, n):
    """Compute CSS code parameters [[n, k, d]]."""
    # k = n - rank(Hx) - rank(Hz)
    rank_x = np.linalg.matrix_rank(Hx)
    rank_z = np.linalg.matrix_rank(Hz)
    k = n - rank_x - rank_z

    return n, k

# Example: Steane code
print("\nSteane [[7, 1, 3]] Code:")
Hx_steane = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])
Hz_steane = Hx_steane.copy()  # Steane code has Hx = Hz

valid, msg = check_css_valid(Hx_steane, Hz_steane)
print(f"CSS validity: {msg}")
n, k = css_parameters(Hx_steane, Hz_steane, 7)
print(f"Parameters: [[{n}, {k}, 3]]")

# =============================================================================
# Part 2: Hypergraph Product Construction
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Hypergraph Product Code Construction")
print("=" * 70)

def hypergraph_product(H1, H2):
    """
    Construct hypergraph product CSS code from two classical codes.

    Args:
        H1: m1 x n1 parity-check matrix of code C1
        H2: m2 x n2 parity-check matrix of code C2

    Returns:
        Hx, Hz: X and Z stabilizer matrices
        n: number of physical qubits
        parameters: dict with code info
    """
    m1, n1 = H1.shape
    m2, n2 = H2.shape

    # Number of physical qubits
    n = n1 * n2 + m1 * m2

    # Construct Hx = [H1 ⊗ I_n2 | I_m1 ⊗ H2^T]
    H1_kron_I = np.kron(H1, np.eye(n2, dtype=int))  # m1*n2 x n1*n2
    I_kron_H2T = np.kron(np.eye(m1, dtype=int), H2.T)  # m1*n2 x m1*m2

    Hx = np.hstack([H1_kron_I, I_kron_H2T]) % 2

    # Construct Hz = [I_n1 ⊗ H2 | H1^T ⊗ I_m2]
    I_kron_H2 = np.kron(np.eye(n1, dtype=int), H2)  # n1*m2 x n1*n2
    H1T_kron_I = np.kron(H1.T, np.eye(m2, dtype=int))  # n1*m2 x m1*m2

    Hz = np.hstack([I_kron_H2, H1T_kron_I]) % 2

    # Compute parameters
    k1 = n1 - np.linalg.matrix_rank(H1)
    k2 = n2 - np.linalg.matrix_rank(H2)
    k = k1 * k2

    parameters = {
        'n': n,
        'k': k,
        'n1': n1, 'm1': m1, 'k1': k1,
        'n2': n2, 'm2': m2, 'k2': k2,
        'Hx_shape': Hx.shape,
        'Hz_shape': Hz.shape,
        'rate': k / n if n > 0 else 0
    }

    return Hx.astype(int), Hz.astype(int), n, parameters

# Example: Hypergraph product of Hamming [7,4,3] with itself
print("\nHamming [7, 4, 3] Code Parity-Check Matrix:")
H_hamming = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
], dtype=int)
print(H_hamming)

print("\nHypergraph Product: Hamming ⊗ Hamming")
Hx, Hz, n, params = hypergraph_product(H_hamming, H_hamming)

print(f"\nResulting qLDPC Code Parameters:")
print(f"  Physical qubits n = {params['n1']}×{params['n2']} + {params['m1']}×{params['m2']} = {params['n']}")
print(f"  Logical qubits k = {params['k1']}×{params['k2']} = {params['k']}")
print(f"  Distance d ≥ min(d1, d2) = 3")
print(f"  Rate k/n = {params['rate']:.4f}")
print(f"  Hx shape: {params['Hx_shape']}")
print(f"  Hz shape: {params['Hz_shape']}")

# Verify CSS constraint
valid, msg = check_css_valid(Hx, Hz)
print(f"\nCSS Verification: {msg}")

# =============================================================================
# Part 3: Stabilizer Weight Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Stabilizer Weight Analysis")
print("=" * 70)

def analyze_weights(H, name="H"):
    """Analyze row and column weights of a matrix."""
    row_weights = np.sum(H, axis=1)
    col_weights = np.sum(H, axis=0)

    print(f"\n{name} Analysis:")
    print(f"  Shape: {H.shape}")
    print(f"  Sparsity: {np.sum(H) / H.size:.6f}")
    print(f"  Row weights: min={row_weights.min()}, max={row_weights.max()}, mean={row_weights.mean():.2f}")
    print(f"  Col weights: min={col_weights.min()}, max={col_weights.max()}, mean={col_weights.mean():.2f}")

    return row_weights, col_weights

hx_rows, hx_cols = analyze_weights(Hx, "Hx (X-stabilizers)")
hz_rows, hz_cols = analyze_weights(Hz, "Hz (Z-stabilizers)")

# Plot weight distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
ax1.hist([hx_rows, hz_rows], bins=range(1, max(hx_rows.max(), hz_rows.max())+2),
         label=['X-stabilizers', 'Z-stabilizers'], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Stabilizer Weight')
ax1.set_ylabel('Count')
ax1.set_title('Stabilizer Weight Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.hist([hx_cols, hz_cols], bins=range(1, max(hx_cols.max(), hz_cols.max())+2),
         label=['Qubit in X-stabs', 'Qubit in Z-stabs'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Number of Stabilizers')
ax2.set_ylabel('Count')
ax2.set_title('Qubit Participation Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_982_weight_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nWeight analysis saved to 'day_982_weight_analysis.png'")

# =============================================================================
# Part 4: Comparison with Surface Code
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Comparison with Surface Code")
print("=" * 70)

def surface_code_params(d):
    """Parameters of a distance-d surface code."""
    n = d**2 + (d-1)**2  # Rotated surface code
    k = 1
    return n, k, d

def hypergraph_product_family(base_size_range):
    """Generate family of hypergraph product codes."""
    results = []

    for size in base_size_range:
        # Create a simple (3, 6)-regular-like code
        n1 = size
        m1 = n1 // 2
        k1 = n1 - m1

        # Random sparse matrix (simplified)
        np.random.seed(42 + size)
        H = np.zeros((m1, n1), dtype=int)
        for j in range(n1):
            rows = np.random.choice(m1, size=3, replace=False)
            H[rows, j] = 1

        # Estimate distance (simplified: weight of lightest row)
        d1 = min(np.sum(H, axis=1))

        # Hypergraph product with itself
        n = n1**2 + m1**2
        k = k1**2

        results.append({
            'base_n': n1,
            'n': n,
            'k': k,
            'd_est': d1,
            'rate': k / n
        })

    return results

# Generate comparison data
base_sizes = [10, 15, 20, 25, 30]
hp_codes = hypergraph_product_family(base_sizes)

# Surface codes at various distances
distances = [3, 5, 7, 9, 11]
surface_codes = [{'d': d, 'n': surface_code_params(d)[0], 'k': 1,
                  'rate': 1 / surface_code_params(d)[0]} for d in distances]

print("\nCode Comparison:")
print("\nSurface Codes:")
print(f"{'d':>5} {'n':>10} {'k':>5} {'rate':>10}")
for sc in surface_codes:
    print(f"{sc['d']:>5} {sc['n']:>10} {sc['k']:>5} {sc['rate']:>10.6f}")

print("\nHypergraph Product Codes (from base codes):")
print(f"{'base_n':>7} {'n':>10} {'k':>5} {'rate':>10} {'d_est':>7}")
for hp in hp_codes:
    print(f"{hp['base_n']:>7} {hp['n']:>10} {hp['k']:>5} {hp['rate']:>10.4f} {hp['d_est']:>7}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
sc_n = [sc['n'] for sc in surface_codes]
sc_k = [sc['k'] for sc in surface_codes]
hp_n = [hp['n'] for hp in hp_codes]
hp_k = [hp['k'] for hp in hp_codes]

ax1.loglog(sc_n, sc_k, 'ro-', markersize=10, linewidth=2, label='Surface Code')
ax1.loglog(hp_n, hp_k, 'bs-', markersize=10, linewidth=2, label='Hypergraph Product')
ax1.set_xlabel('Physical Qubits (n)')
ax1.set_ylabel('Logical Qubits (k)')
ax1.set_title('Encoding Efficiency')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
sc_rate = [sc['rate'] for sc in surface_codes]
hp_rate = [hp['rate'] for hp in hp_codes]

ax2.semilogy(range(len(distances)), sc_rate, 'ro-', markersize=10, linewidth=2, label='Surface Code')
ax2.semilogy(range(len(base_sizes)), hp_rate, 'bs-', markersize=10, linewidth=2, label='Hypergraph Product')
ax2.set_xlabel('Code Index')
ax2.set_ylabel('Rate (k/n)')
ax2.set_title('Code Rate Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_982_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nComparison saved to 'day_982_comparison.png'")

# =============================================================================
# Part 5: Degeneracy Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Degeneracy Analysis")
print("=" * 70)

def find_degenerate_errors(Hx, Hz, max_weight=3):
    """
    Find pairs of degenerate errors (same syndrome, different error).

    For CSS code:
    - X errors: syndrome given by Hz
    - Z errors: syndrome given by Hx
    """
    n = Hx.shape[1]
    degenerate_pairs = []

    # Check X errors (detected by Z-checks, i.e., Hz)
    print(f"\nSearching for degenerate X-errors up to weight {max_weight}...")

    syndromes = {}
    for w in range(1, max_weight + 1):
        # Generate all weight-w errors
        from itertools import combinations
        for positions in combinations(range(n), w):
            error = np.zeros(n, dtype=int)
            error[list(positions)] = 1

            syndrome = tuple((Hz @ error) % 2)

            if syndrome in syndromes:
                # Found degenerate pair!
                degenerate_pairs.append((syndromes[syndrome], tuple(positions)))
            else:
                syndromes[syndrome] = tuple(positions)

    print(f"Found {len(degenerate_pairs)} degenerate X-error pairs")

    if degenerate_pairs:
        print("Examples:")
        for i, (e1, e2) in enumerate(degenerate_pairs[:3]):
            print(f"  Error 1: qubits {e1}, Error 2: qubits {e2}")

    return degenerate_pairs

# Analyze degeneracy in the hypergraph product code
# (Using smaller example for computational feasibility)
H_small = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
], dtype=int)

Hx_small, Hz_small, n_small, params_small = hypergraph_product(H_small, H_small)
print(f"\nSmall qLDPC code: [[{n_small}, {params_small['k']}, d]]")

# This is expensive, so only do for small codes
if n_small <= 50:
    deg_pairs = find_degenerate_errors(Hx_small, Hz_small, max_weight=2)

# =============================================================================
# Part 6: Syndrome Extraction Circuit Depth
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Syndrome Extraction Analysis")
print("=" * 70)

def estimate_syndrome_depth(H):
    """
    Estimate depth of syndrome extraction circuit.

    Each stabilizer requires:
    - Ancilla preparation
    - CNOT for each qubit in stabilizer
    - Measurement

    Parallelization limited by overlapping stabilizers.
    """
    m, n = H.shape

    # Build conflict graph: stabilizers that share qubits
    conflicts = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(i+1, m):
            # Check if stabilizers i and j share any qubit
            shared = np.sum(H[i, :] * H[j, :])
            if shared > 0:
                conflicts[i, j] = 1
                conflicts[j, i] = 1

    # Greedy coloring for parallelization
    colors = [-1] * m
    for i in range(m):
        neighbor_colors = set()
        for j in range(m):
            if conflicts[i, j] and colors[j] >= 0:
                neighbor_colors.add(colors[j])

        # Find smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[i] = color

    num_layers = max(colors) + 1

    # Depth per layer: max stabilizer weight
    max_weight = np.max(np.sum(H, axis=1))
    total_depth = num_layers * (max_weight + 2)  # +2 for prep and measurement

    return {
        'num_stabilizers': m,
        'parallel_layers': num_layers,
        'max_weight': max_weight,
        'estimated_depth': total_depth
    }

print("\nSyndrome Extraction for Hypergraph Product Code:")
depth_x = estimate_syndrome_depth(Hx)
depth_z = estimate_syndrome_depth(Hz)

print(f"\nX-syndrome extraction:")
print(f"  Stabilizers: {depth_x['num_stabilizers']}")
print(f"  Parallel layers needed: {depth_x['parallel_layers']}")
print(f"  Max stabilizer weight: {depth_x['max_weight']}")
print(f"  Estimated circuit depth: {depth_x['estimated_depth']}")

print(f"\nZ-syndrome extraction:")
print(f"  Stabilizers: {depth_z['num_stabilizers']}")
print(f"  Parallel layers needed: {depth_z['parallel_layers']}")
print(f"  Max stabilizer weight: {depth_z['max_weight']}")
print(f"  Estimated circuit depth: {depth_z['estimated_depth']}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Quantum LDPC Code Construction")
print("=" * 70)

summary = f"""
Hypergraph Product Code: [[{params['n']}, {params['k']}, ≥3]]

Construction:
- Input: Hamming [7, 4, 3] code
- Output: [[58, 16, ≥3]] qLDPC code
- Rate: {params['rate']:.4f} (vs ~0.11 for surface code at d=3)

Key Properties:
- X-stabilizer weight: {hx_rows.mean():.1f} average
- Z-stabilizer weight: {hz_rows.mean():.1f} average
- Non-local: requires long-range connectivity

Trade-offs vs Surface Code:
+ Much higher rate (more logical qubits per physical)
+ Better asymptotic scaling
- Non-local connectivity required
- More complex syndrome extraction
- Harder to implement on 2D chips

Tomorrow: Good qLDPC codes with constant rate AND linear distance!
"""
print(summary)

print("\n" + "=" * 70)
print("Day 982 Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CSS constraint | $H_X H_Z^T = 0$ |
| Hypergraph product $H_X$ | $[H_1 \otimes I_{n_2}, I_{m_1} \otimes H_2^T]$ |
| Hypergraph product $H_Z$ | $[I_{n_1} \otimes H_2, H_1^T \otimes I_{m_2}]$ |
| Physical qubits | $n = n_1 n_2 + m_1 m_2$ |
| Logical qubits | $k = k_1 k_2$ |
| Distance bound | $d \geq \min(d_1, d_2)$ |

### Main Takeaways

1. **CSS codes** separate X and Z error correction via two classical codes
2. **Hypergraph product** creates qLDPC from classical LDPC with preserved rate
3. **Degeneracy** means multiple errors map to same syndrome - decoder must handle equivalence classes
4. **Higher rate** than surface codes comes at cost of non-locality
5. **Chain complex perspective** unifies code construction with topology
6. **Syndrome extraction** is more complex due to higher-weight stabilizers

---

## Daily Checklist

- [ ] Verify CSS constraint for given matrices
- [ ] Compute hypergraph product parameters
- [ ] Understand degeneracy and equivalence classes
- [ ] Analyze stabilizer weights
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab
- [ ] Compare with surface code properties

---

## Preview: Day 983

Tomorrow we explore the breakthrough of **Good qLDPC Codes** that achieve:

$$[[n, k = \Theta(n), d = \Theta(n)]]$$

These codes have **constant rate** AND **linear distance** - something thought impossible for years! We will examine:
- Why hypergraph product codes are "not quite good enough"
- The expansion property requirements
- Historical attempts and barriers
- The 2021-2022 breakthrough constructions

---

*"The hypergraph product is the bridge between classical and quantum LDPC - it preserves rate while satisfying the CSS constraint."*
--- Tillich and Zémor

---

**Next:** Day 983 - Good qLDPC Codes: Constant Rate & Distance
