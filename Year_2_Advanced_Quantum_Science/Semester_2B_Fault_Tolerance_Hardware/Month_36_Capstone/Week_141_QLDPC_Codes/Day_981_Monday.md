# Day 981: Classical LDPC Codes & Belief Propagation

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Classical LDPC Theory & Tanner Graphs |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Belief Propagation Algorithm |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: LDPC Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 981, you will be able to:

1. Define classical LDPC codes via sparse parity-check matrices
2. Construct and analyze Tanner graph representations
3. Distinguish regular and irregular LDPC code structures
4. Implement the sum-product (belief propagation) algorithm
5. Explain why LDPC codes approach Shannon capacity
6. Connect classical LDPC concepts to quantum error correction foundations

---

## Core Content

### 1. Introduction to LDPC Codes

Low-Density Parity-Check (LDPC) codes, invented by Robert Gallager in 1962 and rediscovered in the 1990s, are among the most powerful classical error-correcting codes. They achieve near-Shannon-limit performance and form the foundation for quantum LDPC codes.

**Definition:** An LDPC code is a linear block code defined by a sparse parity-check matrix $H$.

$$\boxed{H \cdot c^T = 0 \pmod{2}}$$

where $c \in \{0,1\}^n$ is a valid codeword.

**Sparse Matrix Properties:**
- $H$ is an $m \times n$ matrix over $\mathbb{F}_2$
- Number of 1s is $O(n)$, not $O(n^2)$
- Column weight $w_c$: number of 1s per column (typically 3-6)
- Row weight $w_r$: number of 1s per row (typically 6-20)

**Code Parameters:**
- Block length: $n$ (number of bits)
- Number of checks: $m$ (rows of $H$)
- Rate: $R = k/n \geq 1 - m/n$ (with equality for full-rank $H$)
- Minimum distance: $d$ (weight of lightest non-zero codeword)

---

### 2. Regular vs Irregular LDPC Codes

**Regular LDPC Codes:**
- All columns have the same weight $w_c$
- All rows have the same weight $w_r$
- Denoted $(w_c, w_r)$-regular
- Constraint: $n \cdot w_c = m \cdot w_r$

**Example: (3, 6)-Regular Code**
$$H = \begin{pmatrix}
1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1
\end{pmatrix}$$

Each column has exactly 3 ones, each row has exactly 6 ones.

**Irregular LDPC Codes:**
- Variable degree distributions
- Described by polynomials:
$$\lambda(x) = \sum_{i=2}^{d_v} \lambda_i x^{i-1}$$ (variable node degrees)
$$\rho(x) = \sum_{i=2}^{d_c} \rho_i x^{i-1}$$ (check node degrees)
- Can approach capacity more closely than regular codes
- Optimized via density evolution

---

### 3. Tanner Graph Representation

The Tanner graph is a bipartite graph representation of the parity-check matrix.

**Components:**
- **Variable nodes** (circles): One per codeword bit ($n$ nodes)
- **Check nodes** (squares): One per parity check ($m$ nodes)
- **Edges**: Connect variable $j$ to check $i$ if $H_{ij} = 1$

**Graph Properties:**
- Girth: Length of shortest cycle (larger is better)
- Degree distribution: Node connectivity pattern
- Expansion: Neighborhood growth properties

**Example Tanner Graph for Small LDPC:**

```
Variable nodes:    v1    v2    v3    v4    v5    v6
                   |  \   |   / |  \  |   / |  /
                   |   \  |  /  |   \ | /   | /
                   |    \ | /   |    \|/    |/
Check nodes:       c1 ---- c2 ---- c3 ---- c4
```

**Mathematical Representation:**

$$G = (V \cup C, E)$$

where:
- $V = \{v_1, \ldots, v_n\}$ are variable nodes
- $C = \{c_1, \ldots, c_m\}$ are check nodes
- $(v_j, c_i) \in E \iff H_{ij} = 1$

---

### 4. Why Sparsity Matters

**Decoding Complexity:**
- Dense codes: Syndrome decoding is NP-hard in general
- Sparse codes: Message-passing algorithms run in $O(n \cdot w_c \cdot \text{iterations})$

**Minimum Distance:**
- Sparse matrices allow $d = O(n)$ in some constructions
- Trade-off with density: too sparse $\Rightarrow$ low distance

**Belief Propagation:**
- Works well when Tanner graph has few short cycles
- High girth ensures near-independence of messages

---

### 5. The Sum-Product Algorithm (Belief Propagation)

The BP algorithm iteratively computes posterior probabilities for each bit, given channel observations.

**Setup:**
- Received word: $y = c + e$ (codeword plus error)
- Channel model: Binary Symmetric Channel (BSC) or AWGN
- Goal: Compute $P(c_j = 0 | y)$ for each bit $j$

**Log-Likelihood Ratios (LLRs):**

$$L_j = \log \frac{P(c_j = 0 | y_j)}{P(c_j = 1 | y_j)}$$

For BSC with crossover probability $p$:
$$L_j^{\text{init}} = (-1)^{y_j} \log \frac{1-p}{p}$$

**Message Passing:**

**Variable-to-Check Messages:**
$$L_{v \to c}^{(t)} = L_v^{\text{init}} + \sum_{c' \in \mathcal{N}(v) \setminus c} L_{c' \to v}^{(t-1)}$$

**Check-to-Variable Messages:**
$$\boxed{L_{c \to v}^{(t)} = 2 \tanh^{-1}\left(\prod_{v' \in \mathcal{N}(c) \setminus v} \tanh\left(\frac{L_{v' \to c}^{(t)}}{2}\right)\right)}$$

**Posterior Belief:**
$$L_v^{(t)} = L_v^{\text{init}} + \sum_{c \in \mathcal{N}(v)} L_{c \to v}^{(t)}$$

**Decision:**
$$\hat{c}_j = \begin{cases} 0 & \text{if } L_j^{(t)} > 0 \\ 1 & \text{if } L_j^{(t)} < 0 \end{cases}$$

---

### 6. Algorithm Pseudocode

```
Algorithm: Sum-Product BP for LDPC
Input: Received y, parity-check H, channel LLRs
Output: Decoded codeword c_hat

1. Initialize: L_init[j] = channel LLR for bit j
              L_v_to_c[v][c] = L_init[v] for all edges

2. For iteration t = 1 to max_iter:

   # Check-to-variable messages
   For each check c:
       For each v in N(c):
           product = 1
           For each v' in N(c) \ {v}:
               product *= tanh(L_v_to_c[v'][c] / 2)
           L_c_to_v[c][v] = 2 * atanh(product)

   # Variable-to-check messages
   For each variable v:
       sum_incoming = sum(L_c_to_v[c][v] for c in N(v))
       For each c in N(v):
           L_v_to_c[v][c] = L_init[v] + sum_incoming - L_c_to_v[c][v]

   # Compute posteriors and decisions
   For each variable v:
       L_posterior[v] = L_init[v] + sum(L_c_to_v[c][v] for c in N(v))
       c_hat[v] = 0 if L_posterior[v] > 0 else 1

   # Check if valid codeword
   If H @ c_hat == 0:
       Return c_hat

3. Return c_hat (possibly with errors)
```

---

### 7. Performance and Shannon Capacity

**Shannon Limit for BSC:**
$$C = 1 - H_2(p)$$

where $H_2(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ is binary entropy.

**LDPC Performance:**
- Properly designed irregular LDPC codes can achieve rates within 0.0045 dB of Shannon limit
- Regular codes achieve within 0.5 dB
- Performance improves with block length $n$

**Threshold Phenomenon:**
- Below threshold: Block error rate $\to 0$ as $n \to \infty$
- Above threshold: Finite error probability persists
- BP threshold can be computed via density evolution

---

### 8. Connection to Quantum Error Correction

**Classical-Quantum Parallels:**

| Classical LDPC | Quantum LDPC |
|----------------|--------------|
| Bit | Qubit |
| Parity check | Stabilizer generator |
| $H \cdot c = 0$ | $S_i \ket{\psi} = \ket{\psi}$ |
| Sparse $H$ | Low-weight stabilizers |
| Belief propagation | Quantum BP (with degeneracy) |

**Key Insight:** The Tanner graph structure translates directly to stabilizer code design, with check nodes becoming syndrome measurements.

**Challenges in Quantum:**
1. **Degeneracy:** Multiple errors can have the same syndrome
2. **CSS constraint:** Must satisfy $H_X H_Z^T = 0$
3. **Syndrome extraction:** Non-destructive measurement is costly
4. **Correlated errors:** X and Z errors must be handled together

---

## Practical Applications

### Classical LDPC in Modern Systems

**5G Wireless (NR):**
- LDPC for data channels (replacing turbo codes)
- Block lengths: 68 to 8448 bits
- Rates: 1/3 to 8/9

**Wi-Fi (802.11n/ac/ax):**
- LDPC optional in 802.11n, standard in 802.11ac/ax
- Block lengths: 648, 1296, 1944 bits
- Rates: 1/2, 2/3, 3/4, 5/6

**Storage Systems:**
- Flash memory error correction
- Hard disk drives
- Solid-state drives

---

## Worked Examples

### Example 1: Constructing a Tanner Graph

**Problem:** Draw the Tanner graph for the following parity-check matrix:

$$H = \begin{pmatrix}
1 & 1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
1 & 0 & 1 & 1 & 0
\end{pmatrix}$$

**Solution:**

The matrix has 3 checks (rows) and 5 variables (columns).

1. **Identify connections from H:**
   - $c_1$ connects to $v_1, v_2, v_4$ (row 1 has 1s in columns 1, 2, 4)
   - $c_2$ connects to $v_2, v_3, v_5$ (row 2 has 1s in columns 2, 3, 5)
   - $c_3$ connects to $v_1, v_3, v_4$ (row 3 has 1s in columns 1, 3, 4)

2. **Variable degrees:** $v_1: 2$, $v_2: 2$, $v_3: 2$, $v_4: 2$, $v_5: 1$

3. **Check degrees:** All checks have degree 3

4. **Girth calculation:**
   - Looking for shortest cycle
   - Path: $v_1 \to c_1 \to v_2 \to c_2 \to v_3 \to c_3 \to v_1$
   - Girth = 6

```
   v1    v2    v3    v4    v5
   /\    /\    /\    |
  /  \  /  \  /  \   |
 c1    c2    c3      (c2--v5)
```

---

### Example 2: One Iteration of BP

**Problem:** Given a (3,4)-regular LDPC code, compute one BP iteration for a received word with initial LLRs: $L^{(0)} = [2, -1, 3, 2]$ on a simple check connecting all 4 variables.

**Solution:**

**Step 1: Variable-to-check messages (initial)**
$$L_{v_i \to c} = L_i^{(0)}$$

So: $L_{v_1 \to c} = 2$, $L_{v_2 \to c} = -1$, $L_{v_3 \to c} = 3$, $L_{v_4 \to c} = 2$

**Step 2: Check-to-variable messages**

For variable $v_1$:
$$L_{c \to v_1} = 2\tanh^{-1}\left(\tanh\frac{-1}{2} \cdot \tanh\frac{3}{2} \cdot \tanh\frac{2}{2}\right)$$

Computing:
- $\tanh(-0.5) \approx -0.462$
- $\tanh(1.5) \approx 0.905$
- $\tanh(1.0) \approx 0.762$

Product: $(-0.462)(0.905)(0.762) \approx -0.319$

$$L_{c \to v_1} = 2\tanh^{-1}(-0.319) \approx 2 \times (-0.331) \approx -0.66$$

**Step 3: Updated posterior for $v_1$**
$$L_{v_1}^{(1)} = L_1^{(0)} + L_{c \to v_1} = 2 + (-0.66) = 1.34$$

Decision: $\hat{c}_1 = 0$ (since $L > 0$)

The check message slightly reduced confidence due to the likely error at position 2.

---

### Example 3: Code Rate Calculation

**Problem:** A (3, 6)-regular LDPC code has 1200 variable nodes. Calculate the number of check nodes and the code rate.

**Solution:**

**Edge counting:**
$$n \cdot w_c = m \cdot w_r$$
$$1200 \times 3 = m \times 6$$
$$m = 600 \text{ check nodes}$$

**Code rate:**
$$R = \frac{n - m}{n} = \frac{1200 - 600}{1200} = \frac{1}{2} = 0.5$$

(Assuming full rank, which is typical for random LDPC constructions)

**Information bits:** $k = n - m = 600$ bits

---

## Practice Problems

### Level 1: Direct Application

1. **Matrix Sparsity:** An LDPC parity-check matrix has 10,000 columns, each with weight 4. What is the total number of 1s in the matrix? If the matrix has 5,000 rows, what is the average row weight?

2. **Tanner Graph:** Draw the Tanner graph for:
$$H = \begin{pmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 1
\end{pmatrix}$$

3. **LLR Sign:** If a variable has LLR = -3.5, what is the hard decision? What is the probability that this decision is correct (approximately)?

### Level 2: Intermediate

4. **Girth Analysis:** Prove that a (2, 3)-regular LDPC code must have girth at most 6 by counting argument.

5. **BP Convergence:** A check node receives messages with LLRs $[5, 5, 5]$. Compute the outgoing message to one of the connected variables. What happens as the incoming LLRs increase to infinity?

6. **Rate-Threshold Trade-off:** Explain why higher-rate LDPC codes typically have lower thresholds. What is the theoretical limit?

### Level 3: Challenging

7. **Density Evolution:** For a (3, 6)-regular LDPC code on the BSC, derive the first step of density evolution: given input erasure probability $\epsilon$, compute the erasure probability after one iteration of BP.

8. **Finite Length Effects:** Why does BP performance degrade for finite-length codes compared to the asymptotic (density evolution) prediction? Discuss at least two effects.

9. **Quantum Connection:** Explain why the CSS construction applied to classical LDPC codes produces quantum codes with the same sparsity properties. What additional constraint must the classical codes satisfy?

---

## Computational Lab

### Objective
Implement a classical LDPC code simulator with Tanner graph visualization and belief propagation decoding.

```python
"""
Day 981 Computational Lab: Classical LDPC Codes
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import expit  # Sigmoid function

# =============================================================================
# Part 1: LDPC Code Construction
# =============================================================================

print("=" * 70)
print("Part 1: LDPC Parity-Check Matrix Construction")
print("=" * 70)

def construct_regular_ldpc(n, wc, wr, seed=42):
    """
    Construct a (wc, wr)-regular LDPC parity-check matrix.

    Args:
        n: number of variable nodes (code length)
        wc: column weight (variable degree)
        wr: row weight (check degree)
        seed: random seed

    Returns:
        H: parity-check matrix (m x n)
    """
    np.random.seed(seed)

    # Number of check nodes
    m = (n * wc) // wr
    assert n * wc == m * wr, "Invalid parameters: n*wc must equal m*wr"

    # Create the matrix using progressive edge growth (simplified)
    H = np.zeros((m, n), dtype=int)

    for j in range(n):
        # For each variable, connect to wc checks
        available_checks = [i for i in range(m) if np.sum(H[i, :]) < wr]

        if len(available_checks) >= wc:
            selected = np.random.choice(available_checks, size=wc, replace=False)
        else:
            # Fallback: allow some violations
            all_checks = list(range(m))
            probs = 1.0 / (1 + np.sum(H, axis=1))
            probs = probs / probs.sum()
            selected = np.random.choice(all_checks, size=wc, replace=False, p=probs)

        for i in selected:
            H[i, j] = 1

    return H

# Create a (3, 6)-regular LDPC code
n = 24  # Code length
wc = 3  # Column weight
wr = 6  # Row weight

H = construct_regular_ldpc(n, wc, wr)
m = H.shape[0]

print(f"Code parameters: n = {n}, m = {m}")
print(f"Rate: R = {(n-m)/n:.3f}")
print(f"Column weights: min = {H.sum(axis=0).min()}, max = {H.sum(axis=0).max()}")
print(f"Row weights: min = {H.sum(axis=1).min()}, max = {H.sum(axis=1).max()}")
print(f"Total edges: {H.sum()}")
print(f"Sparsity: {H.sum() / H.size:.4f}")

print("\nParity-check matrix H (first 8x12 submatrix):")
print(H[:8, :12])

# =============================================================================
# Part 2: Tanner Graph Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Tanner Graph Visualization")
print("=" * 70)

def create_tanner_graph(H):
    """Create a NetworkX bipartite graph from parity-check matrix."""
    m, n = H.shape
    G = nx.Graph()

    # Add variable nodes (bottom)
    var_nodes = [f'v{i}' for i in range(n)]
    G.add_nodes_from(var_nodes, bipartite=0)

    # Add check nodes (top)
    check_nodes = [f'c{i}' for i in range(m)]
    G.add_nodes_from(check_nodes, bipartite=1)

    # Add edges
    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                G.add_edge(f'c{i}', f'v{j}')

    return G, var_nodes, check_nodes

def compute_girth(G, var_nodes):
    """Compute the girth (shortest cycle) of the Tanner graph."""
    min_cycle = float('inf')

    for node in var_nodes[:min(10, len(var_nodes))]:  # Sample for efficiency
        try:
            cycle_basis = nx.cycle_basis(G, root=node)
            for cycle in cycle_basis:
                if len(cycle) < min_cycle:
                    min_cycle = len(cycle)
        except:
            pass

    return min_cycle if min_cycle < float('inf') else None

# Create and analyze Tanner graph
G, var_nodes, check_nodes = create_tanner_graph(H)
girth = compute_girth(G, var_nodes)

print(f"Tanner graph: {len(var_nodes)} variable nodes, {len(check_nodes)} check nodes")
print(f"Total edges: {G.number_of_edges()}")
print(f"Estimated girth: {girth}")

# Visualize Tanner graph (for smaller portion)
fig, ax = plt.subplots(figsize=(14, 6))

# Use a smaller subset for visualization
H_small = H[:6, :12]
G_small, var_small, check_small = create_tanner_graph(H_small)

pos = {}
for i, v in enumerate(var_small):
    pos[v] = (i, 0)
for i, c in enumerate(check_small):
    pos[c] = (i * 2, 1)

nx.draw_networkx_nodes(G_small, pos, nodelist=var_small,
                        node_color='lightblue', node_size=500,
                        node_shape='o', ax=ax)
nx.draw_networkx_nodes(G_small, pos, nodelist=check_small,
                        node_color='lightcoral', node_size=400,
                        node_shape='s', ax=ax)
nx.draw_networkx_edges(G_small, pos, alpha=0.5, ax=ax)
nx.draw_networkx_labels(G_small, pos, font_size=8, ax=ax)

ax.set_title('Tanner Graph (subset: 12 variables, 6 checks)')
ax.set_ylim(-0.5, 1.5)
ax.axis('off')

# Add legend
ax.text(0.02, 0.98, 'Circle = Variable node\nSquare = Check node',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.savefig('day_981_tanner_graph.png', dpi=150, bbox_inches='tight')
plt.show()
print("Tanner graph saved to 'day_981_tanner_graph.png'")

# =============================================================================
# Part 3: Belief Propagation Implementation
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Sum-Product Belief Propagation")
print("=" * 70)

def bp_decode(H, y, p, max_iter=50):
    """
    Belief propagation decoding for LDPC codes on BSC.

    Args:
        H: parity-check matrix
        y: received word
        p: channel crossover probability
        max_iter: maximum iterations

    Returns:
        decoded: decoded codeword
        success: whether decoding succeeded
        iterations: number of iterations used
    """
    m, n = H.shape

    # Initialize LLRs from channel
    L_init = (1 - 2*y) * np.log((1-p)/p)

    # Message storage
    # L_v_to_c[i, j] = message from variable j to check i
    # L_c_to_v[i, j] = message from check i to variable j
    L_v_to_c = np.zeros((m, n))
    L_c_to_v = np.zeros((m, n))

    # Initialize variable-to-check messages
    for i in range(m):
        for j in range(n):
            if H[i, j]:
                L_v_to_c[i, j] = L_init[j]

    for iteration in range(max_iter):
        # Check-to-variable messages
        for i in range(m):
            connected = np.where(H[i, :] == 1)[0]
            for j in connected:
                others = [k for k in connected if k != j]
                prod = 1.0
                for k in others:
                    prod *= np.tanh(L_v_to_c[i, k] / 2)
                # Clip to avoid numerical issues
                prod = np.clip(prod, -0.999999, 0.999999)
                L_c_to_v[i, j] = 2 * np.arctanh(prod)

        # Variable-to-check messages
        for j in range(n):
            connected = np.where(H[:, j] == 1)[0]
            sum_incoming = sum(L_c_to_v[i, j] for i in connected)
            for i in connected:
                L_v_to_c[i, j] = L_init[j] + sum_incoming - L_c_to_v[i, j]

        # Compute posteriors and make decisions
        L_posterior = np.zeros(n)
        for j in range(n):
            connected = np.where(H[:, j] == 1)[0]
            L_posterior[j] = L_init[j] + sum(L_c_to_v[i, j] for i in connected)

        decoded = (L_posterior < 0).astype(int)

        # Check if valid codeword
        syndrome = H @ decoded % 2
        if np.sum(syndrome) == 0:
            return decoded, True, iteration + 1

    return decoded, False, max_iter

# Test BP decoding
print("Testing BP decoder...")

# Create a codeword (all zeros is always valid)
codeword = np.zeros(n, dtype=int)

# Simulate BSC channel
p_error = 0.05
np.random.seed(123)
noise = (np.random.random(n) < p_error).astype(int)
received = (codeword + noise) % 2

print(f"Transmitted: all zeros")
print(f"Channel error probability: {p_error}")
print(f"Number of errors: {np.sum(noise)}")
print(f"Error positions: {np.where(noise)[0]}")

# Decode
decoded, success, iterations = bp_decode(H, received, p_error)

print(f"\nDecoding result:")
print(f"Success: {success}")
print(f"Iterations: {iterations}")
print(f"Residual errors: {np.sum(decoded != codeword)}")

# =============================================================================
# Part 4: Performance Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: BP Performance Analysis")
print("=" * 70)

def simulate_ldpc_performance(H, p_range, num_trials=100):
    """Simulate block error rate for different error probabilities."""
    n = H.shape[1]
    bler = []

    for p in p_range:
        errors = 0
        for trial in range(num_trials):
            codeword = np.zeros(n, dtype=int)
            noise = (np.random.random(n) < p).astype(int)
            received = (codeword + noise) % 2

            decoded, success, _ = bp_decode(H, received, p, max_iter=30)

            if not success or np.sum(decoded != codeword) > 0:
                errors += 1

        bler.append(errors / num_trials)

    return bler

# Simulate for range of error probabilities
p_range = np.linspace(0.01, 0.15, 8)
print("Simulating BLER (this may take a moment)...")
bler = simulate_ldpc_performance(H, p_range, num_trials=50)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BLER curve
ax1 = axes[0]
ax1.semilogy(p_range, bler, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Channel Error Probability p')
ax1.set_ylabel('Block Error Rate (BLER)')
ax1.set_title(f'LDPC ({wc},{wr})-Regular, n={n}')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.001, 1.1])

# Threshold estimation
ax1.axvline(x=0.10, color='red', linestyle='--', label='Typical threshold')
ax1.legend()

# Degree distribution
ax2 = axes[1]
var_degrees = H.sum(axis=0)
check_degrees = H.sum(axis=1)

ax2.hist([var_degrees, check_degrees], bins=range(1, max(wc, wr)+3),
         label=['Variable degrees', 'Check degrees'], alpha=0.7,
         edgecolor='black')
ax2.set_xlabel('Degree')
ax2.set_ylabel('Count')
ax2.set_title('Degree Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_981_ldpc_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Performance plot saved to 'day_981_ldpc_performance.png'")

# =============================================================================
# Part 5: Connection to Quantum Codes
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Preview - Connection to Quantum LDPC")
print("=" * 70)

def check_css_compatibility(H1, H2):
    """Check if two classical codes can form a CSS code."""
    # CSS requires H1 @ H2.T = 0 (mod 2)
    product = (H1 @ H2.T) % 2
    return np.sum(product) == 0

# For CSS quantum codes, we need H1 @ H2^T = 0
# Self-orthogonal codes satisfy H @ H^T = 0

print("Checking self-orthogonality of H for CSS construction:")
orthogonal_check = (H @ H.T) % 2
print(f"H @ H^T has {np.sum(orthogonal_check)} non-zero entries")
print(f"Self-orthogonal: {np.sum(orthogonal_check) == 0}")

# Create a dual-containing code (H @ H^T = 0) by construction
print("\nConstructing a self-orthogonal LDPC matrix...")

def construct_self_orthogonal_ldpc(n, wc):
    """
    Construct a self-orthogonal LDPC matrix for CSS codes.
    Uses hypergraph product with itself.
    """
    # Start with a simple self-orthogonal base
    # This is a simplified construction
    m = n // 2
    H = np.zeros((m, n), dtype=int)

    # Construct carefully to maintain orthogonality
    for i in range(m):
        H[i, 2*i] = 1
        H[i, 2*i + 1] = 1
        H[i, (2*i + 2) % n] = 1
        H[i, (2*i + 3) % n] = 1

    return H

H_css = construct_self_orthogonal_ldpc(12, 4)
css_check = (H_css @ H_css.T) % 2

print(f"Self-orthogonal matrix shape: {H_css.shape}")
print(f"H_css @ H_css^T = 0: {np.sum(css_check) == 0}")

print("\nThis self-orthogonal structure is key for CSS quantum LDPC codes!")
print("Tomorrow: Quantum LDPC code construction in detail.")

# =============================================================================
# Part 6: Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary")
print("=" * 70)

summary = f"""
Classical LDPC Code Summary:
----------------------------
Code type: ({wc}, {wr})-regular LDPC
Block length n: {n}
Check nodes m: {m}
Code rate R: {(n-m)/n:.3f}
Sparsity: {H.sum() / H.size:.4f}
Estimated girth: {girth}

BP Decoding:
- Complexity: O(n * wc * iterations)
- Typical iterations: 10-50
- Near-capacity performance for large n

Quantum Connection:
- CSS construction requires H @ H^T = 0
- Tanner graph becomes stabilizer structure
- BP extends to quantum (with degeneracy handling)
"""
print(summary)

print("\n" + "=" * 70)
print("Day 981 Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| LDPC constraint | $H \cdot c^T = 0 \pmod{2}$ |
| Code rate | $R = k/n \geq 1 - m/n$ |
| V-to-C message | $L_{v \to c} = L_v^{\text{init}} + \sum_{c' \neq c} L_{c' \to v}$ |
| C-to-V message | $L_{c \to v} = 2\tanh^{-1}\left(\prod_{v' \neq v} \tanh(L_{v' \to c}/2)\right)$ |
| Posterior LLR | $L_v = L_v^{\text{init}} + \sum_c L_{c \to v}$ |

### Main Takeaways

1. **LDPC codes** use sparse parity-check matrices enabling efficient decoding
2. **Tanner graphs** provide a visual representation connecting variables to checks
3. **Belief propagation** iteratively passes messages to compute posterior probabilities
4. **Near-capacity performance** makes LDPC codes foundational for modern communications
5. **Quantum extension** requires self-orthogonality ($H H^T = 0$) for CSS construction
6. **Sparsity is key** for both efficient decoding and quantum syndrome extraction

---

## Daily Checklist

- [ ] Understand sparse parity-check matrix representation
- [ ] Draw Tanner graphs from parity-check matrices
- [ ] Compute variable and check degrees
- [ ] Execute BP algorithm by hand for small examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab
- [ ] Explain connection to quantum codes

---

## Preview: Day 982

Tomorrow we transition to **Quantum LDPC Code Construction**, where we will:
- Apply CSS construction to classical LDPC codes
- Explore hypergraph product codes (Tillich-Zémor)
- Handle quantum-specific challenges: degeneracy, X-Z correlations
- Implement quantum BP decoding

The sparse structure of classical LDPC naturally extends to quantum codes, but with crucial additional constraints that we will explore in detail.

---

*"The belief propagation algorithm transforms the NP-hard decoding problem into a tractable iterative process through the magic of sparsity."*
--- Contemporary coding theory perspective

---

**Next:** Day 982 - Quantum LDPC Code Construction
