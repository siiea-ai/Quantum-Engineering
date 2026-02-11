# Day 863: Solovay-Kitaev Theorem

## Week 124: Universal Fault-Tolerant Computation | Month 31: Fault-Tolerant QC I

---

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Theorem statement and significance |
| Afternoon | 2.5 hrs | Proof and recursive algorithm |
| Evening | 2.0 hrs | Computational implementation |

---

### Learning Objectives

By the end of today, you will be able to:

1. **State the Solovay-Kitaev theorem** precisely with all conditions
2. **Explain the significance** of the $O(\log^c(1/\epsilon))$ bound
3. **Describe the recursive decomposition algorithm** and its key steps
4. **Analyze the group-theoretic foundations** of net refinement
5. **Implement a basic Solovay-Kitaev decomposition** for SU(2)
6. **Compare SK efficiency** with other approximation methods

---

### Core Content

#### Part 1: Statement of the Solovay-Kitaev Theorem

**Theorem (Solovay-Kitaev):** Let $G$ be a finite set of elements in $SU(d)$ such that:
1. $G$ is closed under inverses: $g \in G \implies g^{-1} \in G$
2. $G$ generates a dense subgroup of $SU(d)$

Then for any $\epsilon > 0$ and any $U \in SU(d)$, there exists a sequence $g_1, g_2, \ldots, g_\ell$ of elements from $G$ such that:

$$\boxed{\|U - g_1 g_2 \cdots g_\ell\| < \epsilon}$$

with sequence length:

$$\boxed{\ell = O\left(\log^c\left(\frac{1}{\epsilon}\right)\right)}$$

where $c$ is a constant (originally $c \approx 4$, improved to $c \approx 3.97$).

**Operator Norm:** Throughout, we use $\|A\| = \max_{|v|=1} |Av|$ (spectral norm).

**Historical Note:** The theorem was independently discovered by Solovay (unpublished) and Kitaev in the 1990s. The first detailed proof was published by Dawson and Nielsen (2005).

---

#### Part 2: Significance and Implications

**Why This Matters:**

1. **Polylogarithmic scaling:** The dependence on $1/\epsilon$ is *poly-logarithmic*, not polynomial
   - Naive random walk would give $O(1/\epsilon^2)$
   - Brute force enumeration gives $O(1/\epsilon)$ but with huge constants

2. **Universal compilation:** Any quantum algorithm can be compiled to Clifford+T with only logarithmic overhead per gate

3. **Practical algorithms:** Modern synthesis achieves even better constants:
   - Gridsynth: Optimal T-count $\approx 3 \log_2(1/\epsilon)$
   - Ross-Selinger: Provably optimal to within additive constant

**Comparison of Approximation Methods:**

| Method | Gate Count | Optimality | Notes |
|--------|-----------|------------|-------|
| Random walk | $O(1/\epsilon^2)$ | Poor | Simple but inefficient |
| Brute force | $O(1/\epsilon)$ | OK | Huge hidden constants |
| Solovay-Kitaev | $O(\log^{3.97}(1/\epsilon))$ | Good | General algorithm |
| Gridsynth | $\approx 3\log_2(1/\epsilon)$ | Optimal | For Clifford+T |

**Key Insight:** The SK algorithm trades *time* (recursion) for *space* (gate count), achieving exponential compression.

---

#### Part 3: Group-Theoretic Foundations

**$\epsilon$-nets in SU(2):**

An **$\epsilon$-net** for $SU(2)$ is a finite set $N_\epsilon \subset SU(2)$ such that:

$$\forall U \in SU(2), \exists V \in N_\epsilon : \|U - V\| < \epsilon$$

**Lemma (Net Size):** The minimum size of an $\epsilon$-net for $SU(d)$ is:

$$|N_\epsilon| = \Theta\left(\frac{1}{\epsilon^{d^2-1}}\right)$$

For $SU(2)$: $|N_\epsilon| = \Theta(1/\epsilon^3)$ (dimension of SU(2) is 3).

**Generating $\epsilon$-nets from Gate Sets:**

If $G$ generates a dense subgroup, then products of elements from $G$ of length $\ell$ form increasingly fine nets:

$$G^\ell = \{g_1 g_2 \cdots g_\ell : g_i \in G\}$$

**Lemma:** There exist constants $c_1, c_2 > 0$ such that $G^\ell$ forms a $c_1 e^{-c_2 \ell}$-net.

This gives the naive bound: $\ell = O(\log(1/\epsilon))$ for an $\epsilon$-net, but with $|G|^\ell$ elements to search.

---

#### Part 4: The Recursive Algorithm

The Solovay-Kitaev algorithm works by recursively refining approximations.

**Key Construction: Group Commutator Shrinking**

If $U, V \in SU(2)$ are close to identity:
$$\|U - I\| \leq \delta, \quad \|V - I\| \leq \delta$$

Then their group commutator $[U, V] = UVU^{-1}V^{-1}$ satisfies:

$$\boxed{\|[U, V] - I\| \leq O(\delta^2)}$$

**Proof:**
$$UVU^{-1}V^{-1} = (I + A)(I + B)(I - A + O(\delta^2))(I - B + O(\delta^2))$$
$$= I + A + B - A - B + AB - BA + O(\delta^2) = I + [A, B] + O(\delta^2)$$

where $U = I + A$, $V = I + B$. The commutator $[A, B] = O(\delta^2)$.

**Inverse Construction:**

From the shrinking property, we can *un-shrink*: given $W$ close to $I$ with $\|W - I\| \leq \delta^2$, find $U, V$ with $\|U - I\|, \|V - I\| \leq O(\delta)$ such that $W \approx [U, V]$.

---

**The Solovay-Kitaev Algorithm:**

```
Algorithm SK(U, n):
    Input: U ∈ SU(2), recursion depth n
    Output: Product of gates from G approximating U

    if n == 0:
        return nearest element of G⁰ to U  // Base case: lookup table

    U_{n-1} = SK(U, n-1)                   // Coarse approximation

    Δ = U · U_{n-1}^†                       // Error: Δ ≈ I

    (V, W) = GC-Decompose(Δ, n-1)          // Find V, W s.t. Δ ≈ [V, W]

    V_{n-1} = SK(V, n-1)
    W_{n-1} = SK(W, n-1)

    return V_{n-1} · W_{n-1} · V_{n-1}^† · W_{n-1}^† · U_{n-1}
```

**Group Commutator Decomposition (GC-Decompose):**

```
Algorithm GC-Decompose(Δ, n):
    Input: Δ ∈ SU(2) with ||Δ - I|| < ε
    Output: V, W such that Δ ≈ [V, W]

    // Δ is a rotation by angle θ around axis n̂
    // We want [V, W] ≈ Δ

    θ = rotation_angle(Δ)
    n̂ = rotation_axis(Δ)

    // Find V, W rotations around different axes
    // with angle ≈ √θ such that [V, W] has angle θ

    φ = √(θ / c)  // c is a geometric constant

    V = Rotation around x-axis by φ
    W = Rotation around (some computed axis) by φ

    return (V, W)
```

---

**Correctness Analysis:**

Let $\epsilon_n$ denote the approximation error at recursion depth $n$.

At level $n-1$: $\|U - U_{n-1}\| \leq \epsilon_{n-1}$

The error $\Delta = U \cdot U_{n-1}^\dagger$ satisfies $\|\Delta - I\| \leq \epsilon_{n-1}$.

After GC-decomposition and recursive approximation:
$$\|\Delta - [V_{n-1}, W_{n-1}]\| \leq c \cdot \epsilon_{n-1}^2 + 4\epsilon_{n-1}$$

where the $\epsilon_{n-1}^2$ comes from GC-shrinking and $4\epsilon_{n-1}$ from approximating $V, W$.

For small enough $\epsilon_0$, we get:

$$\boxed{\epsilon_n \leq \epsilon_0^{(3/2)^n}}$$

Solving for $n$ to achieve $\epsilon$:
$$(3/2)^n \log(1/\epsilon_0) \geq \log(1/\epsilon)$$
$$n = O(\log \log(1/\epsilon))$$

**Total Gate Count:**

Let $\ell_n$ denote gates used at depth $n$. Then:
$$\ell_n = 5 \ell_{n-1} + O(1)$$

(We call SK five times: on $U$, $V$, $W$, $V^{-1}$, $W^{-1}$.)

Solution: $\ell_n = O(5^n) = O(5^{\log_{3/2}\log(1/\epsilon)}) = O(\log^c(1/\epsilon))$

where $c = \log_2 5 / \log_2(3/2) \approx 3.97$.

---

#### Part 5: Improvements and Variations

**Improved Constants:**

The original SK algorithm has $c \approx 3.97$. Improvements include:

1. **Better recurrences:** Using $\ell_n = 4\ell_{n-1}$ instead of 5
2. **Balanced decomposition:** Equal recursion on all branches
3. **Gridsynth (Ross-Selinger):** Optimal for Clifford+T, $\approx 3\log_2(1/\epsilon)$

**Gridsynth Key Idea:**

For $R_z(\theta)$ rotations:
1. Find integers $a, b, c, d$ with $|ad - bc| = 1$
2. Such that $\frac{a + b\omega}{c + d\omega} \approx e^{i\theta}$ where $\omega = e^{i\pi/4}$
3. Use continued fraction-like algorithm on the ring $\mathbb{Z}[\omega]$

**Lower Bounds:**

**Theorem (Harrow-Recht-Chuang):** Any algorithm approximating arbitrary unitaries to precision $\epsilon$ using a finite gate set requires:

$$\Omega\left(\log\left(\frac{1}{\epsilon}\right)\right)$$

gates. Thus Gridsynth is optimal to within a constant factor.

---

#### Part 6: Multi-Qubit Extension

**Extension to SU(2^n):**

The SK theorem extends to $SU(d)$ for any $d$, with:

$$\ell = O\left(\log^c\left(\frac{1}{\epsilon}\right)\right) \cdot d^2$$

**Practical Approach for Multi-Qubit Gates:**

Rather than applying SK directly to $SU(2^n)$:
1. Decompose multi-qubit gate into single-qubit + CNOT
2. Apply SK to each single-qubit gate
3. Total T-count scales as $O(\text{CNOT count} + n \cdot \log^c(1/\epsilon))$

**KAK Decomposition for Two-Qubit Gates:**

Any $U \in SU(4)$ can be written as:

$$U = (A_1 \otimes A_2) \cdot \exp\left(-i\sum_{j=1}^3 \theta_j \sigma_j \otimes \sigma_j\right) \cdot (B_1 \otimes B_2)$$

where $A_1, A_2, B_1, B_2 \in SU(2)$ and $\theta_j$ are the interaction coefficients.

---

### Algorithm Design Implications

**Practical Compilation Pipeline:**

```
Quantum Algorithm
      ↓
High-level gates (Rx, Ry, Rz, arbitrary angles)
      ↓
Solovay-Kitaev / Gridsynth
      ↓
Clifford+T circuit
      ↓
Fault-tolerant implementation
```

**Resource Trade-offs:**

| Factor | Impact on T-count |
|--------|-------------------|
| Target precision ε | $\log(1/\epsilon)$ per rotation |
| Rotation count | Linear in algorithm size |
| Synthesis algorithm | 3-4x difference between SK and optimal |
| Parallelization | Can reduce depth, not T-count |

---

### Worked Examples

#### Example 1: SK Recursion Trace

**Problem:** Trace one level of SK recursion for approximating $R_x(\pi/7)$.

**Solution:**

**Step 1:** Base approximation $U_0$

Let $\epsilon_0 = 0.1$ (base net resolution). Find closest gate sequence of length $\leq 3$:

$$R_x(\pi/7) \approx R_x(0.449) \text{ radians}$$

From our gate set, $HTH = R_x(\pi/4) = R_x(0.785)$ is one option.

Let's say $U_0 = \text{some Clifford+T sequence}$ with $\|R_x(\pi/7) - U_0\| = 0.1$.

**Step 2:** Compute error

$$\Delta = R_x(\pi/7) \cdot U_0^\dagger$$

This $\Delta$ is close to identity: $\|\Delta - I\| \approx 0.1$.

**Step 3:** GC-Decomposition

$\Delta$ is a rotation by angle $\approx 0.1$ around some axis.

We need $V, W$ with $\|V - I\|, \|W - I\| \approx \sqrt{0.1} \approx 0.32$ such that:
$$[V, W] \approx \Delta$$

**Step 4:** Recurse

Apply SK recursively to find $V_0, W_0$ approximating $V, W$.

**Step 5:** Combine

$$U_1 = V_0 W_0 V_0^\dagger W_0^\dagger U_0$$

Error: $\epsilon_1 \approx \epsilon_0^{3/2} = 0.1^{1.5} \approx 0.032$

After $n$ levels: $\epsilon_n \approx 0.1^{(3/2)^n}$

For $\epsilon = 10^{-10}$: need $(3/2)^n \geq 10$, so $n \geq 6$.

---

#### Example 2: Gate Count Estimation

**Problem:** Estimate the T-count for approximating $R_z(\theta)$ to precision $\epsilon = 10^{-8}$ using (a) SK algorithm, (b) Gridsynth.

**Solution:**

**(a) Solovay-Kitaev:**

$$\ell = O(\log^{3.97}(1/\epsilon)) = O(\log^{3.97}(10^8))$$
$$= O((8 \ln 10)^{3.97}) = O((18.4)^{3.97}) \approx O(100,000)$$

With constants, typically $\sim 10^4 - 10^5$ gates.

**(b) Gridsynth:**

$$\ell \approx 3\log_2(1/\epsilon) = 3\log_2(10^8) = 3 \times 26.6 \approx 80$$

T-count $\approx 80$ T-gates (much more efficient!).

**Comparison:** Gridsynth is $\sim 1000\times$ more efficient than basic SK for this precision.

---

#### Example 3: Verify Commutator Shrinking

**Problem:** Verify numerically that $\|[U, V] - I\| = O(\delta^2)$ for rotations $U, V$ near identity.

**Solution:**

Let $U = R_z(\delta)$, $V = R_x(\delta)$ for $\delta = 0.1$.

$$U = \begin{pmatrix} e^{-i\delta/2} & 0 \\ 0 & e^{i\delta/2} \end{pmatrix} \approx \begin{pmatrix} 1 - i\delta/2 & 0 \\ 0 & 1 + i\delta/2 \end{pmatrix}$$

$$V = \begin{pmatrix} \cos(\delta/2) & -i\sin(\delta/2) \\ -i\sin(\delta/2) & \cos(\delta/2) \end{pmatrix} \approx \begin{pmatrix} 1 & -i\delta/2 \\ -i\delta/2 & 1 \end{pmatrix}$$

Computing $[U, V] = UVU^{-1}V^{-1}$:

The result is approximately:
$$[U, V] \approx I + \begin{pmatrix} 0 & -\delta^2/4 \\ \delta^2/4 & 0 \end{pmatrix}$$

So $\|[U, V] - I\| \approx \delta^2/4 = 0.0025$ when $\delta = 0.1$.

This confirms the $O(\delta^2)$ shrinking: $0.1^2/4 = 0.0025$ $\checkmark$

---

### Practice Problems

#### Level 1: Direct Application

**Problem 1.1:** State the three key properties required for a gate set to satisfy SK theorem conditions.

**Problem 1.2:** If the base approximation error is $\epsilon_0 = 0.05$, calculate the error after 5 levels of SK recursion.

**Problem 1.3:** For a gate set $G$ with $|G| = 10$, how many distinct products of length 5 are there? (Ignore equivalences.)

---

#### Level 2: Intermediate

**Problem 2.1:** Prove that $\|[U, V] - I\| \leq 2\|U - I\| + 2\|V - I\|$ for any unitaries $U, V$.

**Problem 2.2:** Show that if $\epsilon_n = \epsilon_0^{\alpha^n}$ for $\alpha > 1$, then the number of recursion levels needed for target precision $\epsilon$ is:
$$n = \log_\alpha\left(\frac{\log(1/\epsilon)}{\log(1/\epsilon_0)}\right)$$

**Problem 2.3:** Explain why the SK algorithm doesn't directly give the optimal Clifford+T count. What additional structure does Gridsynth exploit?

---

#### Level 3: Challenging

**Problem 3.1:** **(Full Recursion Analysis)**
Derive the recurrence $\ell_n = 5\ell_{n-1} + O(1)$ from the SK algorithm structure. Then show how using $\ell_n = 4\ell_{n-1}$ improves the exponent $c$.

**Problem 3.2:** **(Lower Bound Proof)**
Prove that $\Omega(\log(1/\epsilon))$ gates are necessary to approximate arbitrary unitaries to precision $\epsilon$.

Hint: Count the number of distinct $\epsilon$-balls needed to cover $SU(2)$ and compare to the number of circuits of length $\ell$.

**Problem 3.3:** **(GC-Decomposition Details)**
Given $\Delta = R_{\hat{n}}(\phi)$ for small $\phi$, construct explicit $V = R_{\hat{u}}(\psi)$ and $W = R_{\hat{v}}(\psi)$ such that $[V, W] \approx \Delta$ with $\psi = O(\sqrt{\phi})$.

Hint: Use the formula $[R_{\hat{u}}(\psi), R_{\hat{v}}(\psi)] \approx R_{\hat{u} \times \hat{v}}(2\psi^2)$ for small $\psi$.

---

### Computational Lab

```python
"""
Day 863 Computational Lab: Solovay-Kitaev Algorithm Implementation
Basic implementation for SU(2) with Clifford+T gate set
"""

import numpy as np
from scipy.spatial.transform import Rotation
from functools import lru_cache
import time

# Gate definitions
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
S_dag = np.conj(S).T
T_dag = np.conj(T).T

# Basic gate set (closed under inverse)
GATES = {
    'I': I,
    'H': H,
    'S': S,
    'S†': S_dag,
    'T': T,
    'T†': T_dag,
    'HS': H @ S,
    'SH': S @ H,
    'HSH': H @ S @ H,
}

def matrix_distance(A, B):
    """Compute operator norm distance ||A - B||"""
    return np.linalg.norm(A - B, ord=2)

def normalize_su2(U):
    """Project matrix to SU(2) by normalizing determinant"""
    det = np.linalg.det(U)
    return U / np.sqrt(det)

def matrix_to_rotation(U):
    """
    Convert SU(2) matrix to rotation angle and axis.
    U = cos(θ/2)I - i sin(θ/2)(n·σ)
    """
    U = normalize_su2(U)
    trace = np.trace(U)
    cos_half = np.real(trace) / 2
    cos_half = np.clip(cos_half, -1, 1)
    theta = 2 * np.arccos(cos_half)

    if np.abs(np.sin(theta/2)) < 1e-10:
        return 0, np.array([0, 0, 1])

    nx = np.imag(U[0, 1] + U[1, 0]) / (2 * np.sin(theta/2))
    ny = np.real(U[0, 1] - U[1, 0]) / (2 * np.sin(theta/2))
    nz = np.imag(U[0, 0] - U[1, 1]) / (2 * np.sin(theta/2))

    axis = np.array([nx, ny, nz])
    norm = np.linalg.norm(axis)
    if norm > 1e-10:
        axis = axis / norm

    return theta, axis

def rotation_to_matrix(theta, axis):
    """Convert rotation angle and axis to SU(2) matrix"""
    axis = np.array(axis) / np.linalg.norm(axis)
    nx, ny, nz = axis

    c = np.cos(theta/2)
    s = np.sin(theta/2)

    return np.array([
        [c - 1j*s*nz, -s*ny - 1j*s*nx],
        [s*ny - 1j*s*nx, c + 1j*s*nz]
    ], dtype=complex)

class SolovayKitaev:
    """Solovay-Kitaev algorithm for gate decomposition"""

    def __init__(self, base_gates=GATES, base_depth=4):
        """
        Initialize SK algorithm with base gate set.

        Args:
            base_gates: Dictionary of gate name -> matrix
            base_depth: Depth of base approximation lookup table
        """
        self.base_gates = base_gates
        self.base_depth = base_depth
        self.epsilon_net = {}
        self.build_base_approximations()

    def build_base_approximations(self):
        """Build lookup table of base approximations"""
        print(f"Building base approximation table (depth {self.base_depth})...")

        # Start with identity
        self.epsilon_net = {self._matrix_key(I): ('I', I)}

        # BFS to generate all products up to base_depth
        gate_names = list(self.base_gates.keys())
        current_level = [('I', I)]

        for depth in range(1, self.base_depth + 1):
            next_level = []
            for seq, mat in current_level:
                for name, gate in self.base_gates.items():
                    new_mat = gate @ mat
                    new_seq = name + '·' + seq if seq != 'I' else name
                    key = self._matrix_key(new_mat)
                    if key not in self.epsilon_net:
                        self.epsilon_net[key] = (new_seq, new_mat)
                        next_level.append((new_seq, new_mat))
            current_level = next_level
            print(f"  Depth {depth}: {len(self.epsilon_net)} total unitaries")

        print(f"Base table complete: {len(self.epsilon_net)} elements")

    def _matrix_key(self, M, precision=4):
        """Convert matrix to hashable key"""
        # Normalize global phase
        if np.abs(M[0, 0]) > 0.01:
            phase = np.exp(-1j * np.angle(M[0, 0]))
        else:
            phase = 1
        M_norm = M * phase
        return tuple(np.round(np.real(M_norm.flatten()), precision).tolist() +
                     np.round(np.imag(M_norm.flatten()), precision).tolist())

    def base_approximation(self, U):
        """Find closest element in epsilon-net to U"""
        best_dist = float('inf')
        best_seq = 'I'
        best_mat = I

        for key, (seq, mat) in self.epsilon_net.items():
            dist = matrix_distance(U, mat)
            if dist < best_dist:
                best_dist = dist
                best_seq = seq
                best_mat = mat

        return best_seq, best_mat, best_dist

    def gc_decompose(self, Delta):
        """
        Group commutator decomposition.
        Find V, W such that [V, W] ≈ Delta.
        """
        theta, axis = matrix_to_rotation(Delta)

        if np.abs(theta) < 1e-10:
            return I, I

        # For [Ru(φ), Rv(φ)] ≈ R_{u×v}(2φ²)
        # We need φ ≈ √(θ/2)
        phi = np.sqrt(np.abs(theta) / 2)

        # Choose V, W axes perpendicular to each other
        # and such that their cross product is along the target axis

        # Find two perpendicular axes whose cross product is 'axis'
        if np.abs(axis[2]) < 0.9:
            v_axis = np.cross(axis, [0, 0, 1])
        else:
            v_axis = np.cross(axis, [1, 0, 0])
        v_axis = v_axis / np.linalg.norm(v_axis)

        w_axis = np.cross(v_axis, axis)
        w_axis = w_axis / np.linalg.norm(w_axis)

        V = rotation_to_matrix(phi, v_axis)
        W = rotation_to_matrix(phi, w_axis)

        return V, W

    def decompose(self, U, depth=3):
        """
        Solovay-Kitaev decomposition.

        Args:
            U: Target SU(2) matrix
            depth: Recursion depth

        Returns:
            (sequence, matrix, error)
        """
        U = normalize_su2(U)

        if depth == 0:
            return self.base_approximation(U)

        # Recursive approximation of U
        seq_prev, U_prev, _ = self.decompose(U, depth - 1)

        # Compute error Delta = U · U_prev†
        Delta = U @ np.conj(U_prev).T

        # Check if already good enough
        delta_dist = matrix_distance(Delta, I)
        if delta_dist < 1e-12:
            return seq_prev, U_prev, matrix_distance(U, U_prev)

        # GC decomposition: find V, W such that [V, W] ≈ Delta
        V, W = self.gc_decompose(Delta)

        # Recursively approximate V and W
        seq_V, V_approx, _ = self.decompose(V, depth - 1)
        seq_W, W_approx, _ = self.decompose(W, depth - 1)

        # Construct approximation: [V, W] · U_prev
        commutator = V_approx @ W_approx @ np.conj(V_approx).T @ np.conj(W_approx).T
        U_new = commutator @ U_prev

        # Build sequence string
        seq_V_inv = self._invert_sequence(seq_V)
        seq_W_inv = self._invert_sequence(seq_W)
        new_seq = f"[{seq_V}·{seq_W}·{seq_V_inv}·{seq_W_inv}]·{seq_prev}"

        error = matrix_distance(U, U_new)
        return new_seq, U_new, error

    def _invert_sequence(self, seq):
        """Invert a gate sequence"""
        if seq == 'I':
            return 'I'

        parts = seq.split('·')
        inverted_parts = []
        for part in reversed(parts):
            if part.endswith('†'):
                inverted_parts.append(part[:-1])
            elif part in ['H']:  # Self-inverse
                inverted_parts.append(part)
            else:
                inverted_parts.append(part + '†')
        return '·'.join(inverted_parts)

# Test the implementation
print("="*70)
print("Solovay-Kitaev Algorithm Implementation")
print("="*70)

sk = SolovayKitaev(base_depth=4)

# Test targets
test_angles = [np.pi/7, np.pi/5, 0.123, 1.234, 2.718]
test_axes = [
    [1, 0, 0],  # X-rotation
    [0, 1, 0],  # Y-rotation
    [0, 0, 1],  # Z-rotation
    [1, 1, 0],  # XY-rotation
    [1, 1, 1],  # XYZ-rotation
]

print("\n" + "="*70)
print("Testing Solovay-Kitaev Decomposition")
print("="*70)

results = []
for angle in test_angles[:3]:
    for axis in test_axes[:2]:
        target = rotation_to_matrix(angle, axis)
        target_name = f"R({angle:.3f}, [{axis[0]},{axis[1]},{axis[2]}])"

        print(f"\nTarget: {target_name}")
        for depth in range(4):
            start = time.time()
            seq, approx, error = sk.decompose(target, depth=depth)
            elapsed = time.time() - start

            # Count gates
            gate_count = seq.count('·') + 1 if seq != 'I' else 0

            print(f"  Depth {depth}: error = {error:.2e}, gates ≈ {gate_count:4d}, time = {elapsed:.3f}s")

            if depth == 3:
                results.append((target_name, error, gate_count))

# Demonstrate commutator shrinking
print("\n" + "="*70)
print("Commutator Shrinking Demonstration")
print("="*70)

deltas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
for delta in deltas:
    U = rotation_to_matrix(delta, [0, 0, 1])
    V = rotation_to_matrix(delta, [1, 0, 0])

    commutator = U @ V @ np.conj(U).T @ np.conj(V).T
    comm_dist = matrix_distance(commutator, I)

    print(f"δ = {delta:.3f}: ||U-I|| = {matrix_distance(U, I):.4f}, "
          f"||[U,V]-I|| = {comm_dist:.6f}, ratio = {comm_dist/(delta**2):.2f}")

# Visualize error scaling
print("\n" + "="*70)
print("Error Scaling with Recursion Depth")
print("="*70)

target = rotation_to_matrix(0.5, [1, 1, 1])
errors = []
gate_counts = []

for depth in range(6):
    _, _, error = sk.decompose(target, depth=depth)
    errors.append(error)
    print(f"Depth {depth}: error = {error:.2e}")

# Theoretical prediction: ε_n ≈ ε_0^(1.5^n)
print("\nTheoretical vs Actual:")
eps_0 = errors[0]
for n in range(len(errors)):
    theoretical = eps_0 ** (1.5 ** n)
    print(f"  n={n}: actual = {errors[n]:.2e}, theoretical ≈ {theoretical:.2e}")

print("\n" + "="*70)
print("Solovay-Kitaev Lab Complete")
print("="*70)
```

---

### Summary

#### Key Formulas

| Concept | Formula |
|---------|---------|
| SK bound | $\ell = O(\log^{3.97}(1/\epsilon))$ |
| Commutator shrinking | $\|[U,V] - I\| = O(\delta^2)$ if $\|U-I\|, \|V-I\| \leq \delta$ |
| Error recursion | $\epsilon_n \approx \epsilon_0^{(3/2)^n}$ |
| Gate recursion | $\ell_n = 5\ell_{n-1} + O(1)$ |
| Optimal bound | $\Omega(\log(1/\epsilon))$ (information-theoretic) |

#### Main Takeaways

1. **Solovay-Kitaev guarantees efficient approximation** with polylogarithmic gate count
2. **Commutator shrinking** is the key insight: $[U,V]$ squares the distance to identity
3. **Recursive structure** trades time for gates, achieving exponential compression
4. **Modern algorithms (Gridsynth)** achieve optimal $O(\log(1/\epsilon))$ for Clifford+T
5. **SK applies to any dense gate set**, not just Clifford+T

---

### Daily Checklist

- [ ] Can state Solovay-Kitaev theorem with all conditions
- [ ] Understand the significance of polylogarithmic scaling
- [ ] Can explain the recursive algorithm structure
- [ ] Understand commutator shrinking and its role
- [ ] Completed computational lab with working SK implementation
- [ ] Worked through at least 2 practice problems per level

---

### Preview: Day 864

Tomorrow we focus on **T-Gate Synthesis**---specialized algorithms for decomposing rotations into Clifford+T circuits. We'll study the Ross-Selinger Gridsynth algorithm that achieves optimal T-count $\approx 3\log_2(1/\epsilon)$, Repeat-Until-Success (RUS) circuits that trade ancillas for T-count reduction, and practical synthesis tools used in real quantum compilers.

---

*Day 863 establishes the theoretical foundation for gate approximation---tomorrow we learn optimal practical algorithms.*
