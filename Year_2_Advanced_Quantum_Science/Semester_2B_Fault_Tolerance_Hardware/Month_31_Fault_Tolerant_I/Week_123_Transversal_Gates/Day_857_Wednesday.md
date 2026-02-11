# Day 857: Eastin-Knill Theorem Statement

## Overview

**Day:** 857 of 1008
**Week:** 123 (Transversal Gates & Eastin-Knill)
**Month:** 31 (Fault-Tolerant Quantum Computing I)
**Topic:** Formal Statement and Implications of the Eastin-Knill Theorem

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Theorem statement and context |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Discreteness and proof sketch |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Implications exploration |

---

## Learning Objectives

By the end of today, you should be able to:

1. **State** the Eastin-Knill theorem precisely
2. **Explain** the connection between transversality and discreteness
3. **Understand** why discrete subgroups of SU(d) cannot be universal
4. **Describe** the proof sketch using Lie group theory
5. **Articulate** the fundamental implications for fault-tolerant quantum computing
6. **Connect** the theorem to the magic state paradigm

---

## Historical Context

### The Search for Fault-Tolerant Universality

By the mid-2000s, the quantum error correction community faced a puzzle:

**Observation 1:** Transversal gates are naturally fault-tolerant (no error spread).

**Observation 2:** No known code had a transversal universal gate set.

**Question:** Is this a limitation of our ingenuity, or a fundamental barrier?

### The Eastin-Knill Result (2009)

Bryan Eastin and Emanuel Knill proved a remarkable no-go theorem:

> **There exists no quantum error-detecting code that admits a universal, transversal gate set.**

This answered the question definitively: the limitation is **fundamental**, not just technical.

---

## The Theorem

### Precise Statement

**Theorem (Eastin-Knill, 2009):**
Let $\mathcal{C}$ be a quantum error-detecting code encoding $k$ logical qubits into $n$ physical qubits. Let $\mathcal{T} \subseteq U(2^k)$ be the group of logical operations implementable by transversal physical gates. Then:

$$\boxed{\mathcal{T} \text{ is a finite group}}$$

**Corollary:** Since $U(2^k)$ has infinitely many elements (it's a continuous Lie group), $\mathcal{T}$ cannot be universal for quantum computation.

### Equivalent Formulations

**Formulation 1 (Discreteness):**
The transversal gate group $\mathcal{T}$ is a discrete subgroup of $U(2^k)$.

**Formulation 2 (Non-Density):**
$\mathcal{T}$ is not dense in $SU(2^k)$ under any reasonable topology.

**Formulation 3 (Finite Approximation):**
Any transversal gate can only approximate finitely many distinct logical operations.

### Scope of the Theorem

The theorem applies to:
- All stabilizer codes
- All CSS codes
- All topological codes
- **Any** code that can detect at least one error

The theorem does **not** apply to:
- Trivial encodings (no error detection)
- Approximate error correction (where the theorem is modified)

---

## Key Concepts

### Definition: Error-Detecting Code

**Definition:** A quantum code $\mathcal{C}$ is **error-detecting** if there exists a non-trivial error $E$ such that:

$$P_{\mathcal{C}} E P_{\mathcal{C}} = \lambda P_{\mathcal{C}}$$

for some scalar $\lambda$, where $P_{\mathcal{C}}$ is the projector onto the code space.

This means the code can detect at least one type of error.

**Note:** All codes with distance $d \geq 2$ are error-detecting.

### Definition: Transversal Gate (Refined)

**Definition:** A transversal gate on $\mathcal{C}$ is a unitary $\tilde{U}$ on the physical space such that:

1. $\tilde{U} = \bigotimes_{i=1}^{n} U_i$ (tensor product structure)
2. $\tilde{U} \mathcal{C} = \mathcal{C}$ (preserves code space)
3. $\tilde{U}$ induces a logical operation $U_L$ on the encoded qubits

### The Transversal Gate Group

**Definition:** The transversal gate group is:

$$\mathcal{T} = \{U_L \in U(2^k) : U_L = \tilde{U}|_{\mathcal{C}} \text{ for some transversal } \tilde{U}\}$$

**Properties of $\mathcal{T}$:**
- Closed under composition: $U_L, V_L \in \mathcal{T} \Rightarrow U_L V_L \in \mathcal{T}$
- Closed under inverse: $U_L \in \mathcal{T} \Rightarrow U_L^{-1} \in \mathcal{T}$
- Contains identity: $I \in \mathcal{T}$

Therefore $\mathcal{T}$ is a subgroup of $U(2^k)$.

---

## The Discreteness Argument

### Why Transversal Gates Form a Discrete Group

**Key Insight:** The constraint that transversal gates must preserve the code space forces discreteness.

**Intuitive Argument:**

1. Consider a continuous family of transversal gates $\tilde{U}(\theta) = U(\theta)^{\otimes n}$
2. For small $\theta$, $U(\theta) \approx I + i\theta H$ for some Hermitian $H$
3. The logical action: $U_L(\theta) = \tilde{U}(\theta)|_{\mathcal{C}}$
4. If $U(\theta)^{\otimes n}$ preserves the code space for all $\theta$, it must preserve the error structure
5. But infinitesimal generators would map code space to error spaces!

**Contradiction:** Continuous transversal gates cannot preserve error-detecting codes.

### Formal Discreteness Theorem

**Theorem:** Let $\mathcal{C}$ be an error-detecting code and $\mathcal{T}$ its transversal gate group. There exists $\epsilon > 0$ such that:

$$U_L, V_L \in \mathcal{T}, \, U_L \neq V_L \Rightarrow \|U_L - V_L\| > \epsilon$$

**Meaning:** Distinct elements of $\mathcal{T}$ are separated by a finite gap.

### Discreteness Implies Finiteness

**Theorem:** A discrete subgroup of a compact Lie group is finite.

**Proof Sketch:**
1. $U(2^k)$ is a compact Lie group
2. Compact spaces have finite covers by $\epsilon$-balls
3. If $\mathcal{T}$ is discrete with separation $\epsilon$, each element occupies its own ball
4. Finite cover implies finite number of elements $\square$

---

## Proof Sketch

### Overview

The proof proceeds in three main steps:

1. **Local Structure:** Show transversal gates near identity are highly constrained
2. **Lie Algebra Constraint:** The Lie algebra of $\mathcal{T}$ is trivial
3. **Discreteness:** Conclude $\mathcal{T}$ has no continuous part

### Step 1: Infinitesimal Analysis

Consider a transversal unitary near identity:
$$\tilde{U} = I + i\epsilon G + O(\epsilon^2)$$

where $G = \sum_i G_i$ with $G_i$ acting on qubit $i$.

For $\tilde{U}$ to preserve the code space:
$$P_{\mathcal{C}} G P_{\mathcal{C}} = G|_{\mathcal{C}}$$

But for error detection, there exists error $E$ with:
$$P_{\mathcal{C}} E P_{\mathcal{C}} = \lambda P_{\mathcal{C}}$$

The interplay between $G$ and $E$ creates constraints.

### Step 2: Error Operator Constraints

**Key Lemma:** If $\mathcal{C}$ detects error $E_j$ on qubit $j$, then for any transversal generator $G = \sum_i G_i$:

$$[G_j, E_j]_{\mathcal{C}} = 0$$

This means the local generator must commute with detectable errors when restricted to the code space.

**Consequence:** The generator $G$ is severely restricted.

### Step 3: Trivial Lie Algebra

**Theorem:** The Lie algebra of $\mathcal{T}$ (infinitesimal generators) is zero-dimensional.

**Proof Idea:**
- Transversal generators must commute with all detectable errors
- For a non-trivial error-detecting code, this forces $G|_{\mathcal{C}} = \lambda I$ for some scalar $\lambda$
- Scalar multiples of identity generate only phases, not non-trivial unitaries
- Therefore, no non-trivial infinitesimal transversal gates exist $\square$

### Conclusion

With trivial Lie algebra, $\mathcal{T}$ has no continuous component. Combined with compactness of $U(2^k)$, this implies $\mathcal{T}$ is finite.

---

## Examples and Non-Examples

### Example 1: Steane Code

The [[7,1,3]] Steane code has transversal gate group:
$$\mathcal{T}_{\text{Steane}} = \langle X, Z, H, S, \text{CNOT} \rangle / \text{phases}$$

This is the **Clifford group** modulo phases, which is finite:
$$|\text{Cliff}_1 / U(1)| = 24$$

Not universal! Consistent with Eastin-Knill.

### Example 2: 15-Qubit Reed-Muller Code

The [[15,1,3]] Reed-Muller code has:
- Transversal T gate
- Transversal X, Z
- But NO transversal H or S!

$$\mathcal{T}_{RM} = \langle X, Z, T \rangle / \text{phases}$$

Still finite. The T gate doesn't help because we lose H.

### Non-Example: Trivial Encoding

Consider the "code" $\mathcal{C} = \mathbb{C}^2$ (single qubit, no encoding):
- Every unitary is "transversal" (acts on the single qubit)
- $\mathcal{T} = U(2)$ is continuous and universal

But this is NOT error-detecting! Eastin-Knill doesn't apply.

### Non-Example: Infinite Distance Limit

Consider a family of codes with distance $d \to \infty$.

Theoretically, one might hope the transversal group grows. But Eastin-Knill shows it remains finite regardless of $d$.

---

## Implications for Fault-Tolerant Computing

### The Magic State Necessity

**Corollary:** Universal fault-tolerant quantum computation requires NON-transversal gates.

**Solution:** Magic state injection + distillation

1. Prepare "magic states" like $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$
2. Use transversal Clifford gates to implement T via gate teleportation
3. Purify noisy magic states through distillation

### The Code Switching Approach

**Alternative:** Use multiple codes with complementary transversal gates.

- Code A: Transversal Clifford
- Code B: Transversal T

Switch between codes to access full gate set.

**Challenge:** The switching itself must be fault-tolerant!

### Resource Overhead

Eastin-Knill implies fundamental resource overhead:
- Magic state distillation requires $O(\log^c(1/\epsilon))$ resources
- Code switching requires constant overhead per gate
- No scheme achieves universality with purely transversal gates

---

## Connection to Lie Group Theory

### The Unitary Group U(d)

The unitary group $U(d)$ is a Lie group of dimension $d^2$:

$$\dim U(d) = d^2, \quad \dim SU(d) = d^2 - 1$$

For a single logical qubit: $\dim SU(2) = 3$

### Discrete Subgroups

Famous discrete subgroups of $SU(2)$:
- Cyclic groups $\mathbb{Z}_n$
- Dihedral groups $D_n$
- Binary tetrahedral group (order 24)
- Binary octahedral group (order 48)
- Binary icosahedral group (order 120)

These are ALL that exist! (A, D, E classification)

### The Clifford Group

The single-qubit Clifford group (mod phases) has order 24, isomorphic to $S_4$.

For $n$ qubits:
$$|\text{Cliff}_n / U(1)| = 2^{n^2 + 2n} \prod_{j=1}^{n}(4^j - 1)$$

Large but finite.

---

## Worked Examples

### Example 1: Verify Clifford Group is Finite

**Problem:** Show the single-qubit Clifford group has exactly 24 elements (mod phases).

**Solution:**

The Clifford group is generated by $H$ and $S$:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

Count elements:
- $S$ has order 4: $\{I, S, S^2 = Z, S^3\}$
- $H$ has order 2: $\{I, H\}$
- But they don't commute, so the group is larger

The group acts on the Bloch sphere, permuting the octahedron vertices (up/down, left/right, front/back).

Symmetries of the octahedron = $S_4$ (order 24). $\checkmark$

### Example 2: Why T Makes the Group Infinite (Without Code Constraints)

**Problem:** Show that $\langle H, T \rangle$ generates a dense subgroup of $SU(2)$.

**Solution:**

The T gate: $T = e^{i\pi/8} \begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix}$

Ignoring global phases: rotation by $\pi/4$ about the Z-axis.

$HTH$ gives a rotation about a different axis.

**Solovay-Kitaev:** Any two non-commuting rotations by irrational angles generate a dense subgroup.

Since $\pi/4$ is irrational as a multiple of $\pi$, and $H$ changes the axis:
$$\langle H, T \rangle \text{ is dense in } SU(2)$$

But on a code, T is constrained! The transversal version doesn't give all these rotations.

### Example 3: Discrete Gap in the Steane Code

**Problem:** Estimate the minimum distance between distinct transversal gates on the Steane code.

**Solution:**

The transversal gates form the Clifford group (24 elements).

The "nearest" distinct Clifford gates are related by single generators.

For example, $I$ and $S$ differ by:
$$\|I - S\| = \|I - \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}\| = \max(0, |1-i|) = \sqrt{2}$$

Similarly, $I$ and $H$:
$$\|I - H\| = \|I - \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\| \approx 1.08$$

The minimum gap is approximately $\epsilon \approx 1.08$ in operator norm.

This finite gap is the discrete structure that Eastin-Knill predicts. $\checkmark$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** State the Eastin-Knill theorem in your own words. What does "discrete" mean in this context?

**P1.2** The Toffoli gate is universal for classical computation. Why doesn't adding transversal Toffoli violate Eastin-Knill?

**P1.3** List three codes and their transversal gate groups. Verify each group is finite.

### Level 2: Intermediate

**P2.1** Prove that the Clifford group on $n$ qubits is finite by counting its elements (use the formula given above for $n=2$).

**P2.2** Explain why a code with distance $d=1$ (error-detecting but not correcting) still falls under Eastin-Knill.

**P2.3** The [[4,2,2]] code has two logical qubits. Its transversal gates include controlled operations between the logical qubits. How does this fit with Eastin-Knill?

### Level 3: Challenging

**P3.1** Prove that if $\mathcal{T}$ contains a continuous one-parameter subgroup $U(\theta) = e^{i\theta G}$, then $G|_{\mathcal{C}} = 0$ for any error-detecting code.

**P3.2** Construct a code where the transversal gate group is exactly the Pauli group (much smaller than Clifford). What properties must it have?

**P3.3** Investigate: Does Eastin-Knill extend to approximate error-correcting codes? State a version of the theorem for this case.

---

## Computational Lab

```python
"""
Day 857: Eastin-Knill Theorem - Discreteness Exploration
=========================================================

Exploring the discrete nature of transversal gate groups.
"""

import numpy as np
from typing import List, Tuple, Set
from itertools import product
import matplotlib.pyplot as plt

# Pauli and Clifford gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def normalize_phase(U: np.ndarray) -> np.ndarray:
    """Normalize a unitary to have determinant 1 (SU form)."""
    det = np.linalg.det(U)
    phase = det ** (1 / U.shape[0])
    return U / phase


def are_equal_up_to_phase(U: np.ndarray, V: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if U and V differ only by a global phase."""
    # Normalize both to SU
    U_norm = normalize_phase(U)
    V_norm = normalize_phase(V)
    return np.allclose(U_norm, V_norm, atol=tol) or \
           np.allclose(U_norm, -V_norm, atol=tol)


def generate_clifford_group() -> List[np.ndarray]:
    """Generate the single-qubit Clifford group."""
    generators = [H, S]
    group = {tuple(I.flatten()): I}

    # Generate by multiplying all combinations
    queue = [I, H, S]
    while queue:
        current = queue.pop(0)
        for gen in generators:
            new = current @ gen
            key = tuple(normalize_phase(new).flatten().round(10))
            if key not in group:
                group[key] = new
                queue.append(new)

            new2 = gen @ current
            key2 = tuple(normalize_phase(new2).flatten().round(10))
            if key2 not in group:
                group[key2] = new2
                queue.append(new2)

    return list(group.values())


def compute_gate_distances(gates: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise distances between gates."""
    n = len(gates)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Operator norm distance
            diff = gates[i] - gates[j]
            dist = np.linalg.norm(diff, ord=2)  # Spectral norm
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def minimum_gap(distances: np.ndarray) -> float:
    """Find minimum non-zero distance."""
    nonzero = distances[distances > 1e-10]
    if len(nonzero) == 0:
        return 0
    return np.min(nonzero)


def generate_ht_approximations(depth: int) -> List[np.ndarray]:
    """Generate gates from H and T up to given depth."""
    gates = [I]
    current_level = [I]

    for d in range(depth):
        next_level = []
        for g in current_level:
            for gen in [H, T]:
                new = g @ gen
                # Check if already have this (up to phase)
                is_new = True
                for existing in gates:
                    if are_equal_up_to_phase(new, existing):
                        is_new = False
                        break
                if is_new:
                    gates.append(new)
                    next_level.append(new)
        current_level = next_level

    return gates


def bloch_coordinates(U: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract the axis-angle representation and return Bloch sphere point.
    For visualization of where gates "point".
    """
    # U = exp(i*theta/2 * (n_x X + n_y Y + n_z Z)) up to phase
    U_su = normalize_phase(U)

    # Extract angle
    trace = np.trace(U_su)
    cos_theta_2 = trace.real / 2
    theta = 2 * np.arccos(np.clip(cos_theta_2, -1, 1))

    if abs(theta) < 1e-10:
        return (0, 0, 0)  # Identity

    sin_theta_2 = np.sin(theta / 2)
    if abs(sin_theta_2) < 1e-10:
        return (0, 0, 1)  # Pure Z rotation

    # Extract axis from U - U^dag = 2i sin(theta/2) (n.sigma)
    diff = U_su - U_su.conj().T
    n_x = -diff[0, 1].imag / (2 * sin_theta_2) if abs(sin_theta_2) > 1e-10 else 0
    n_y = diff[0, 1].real / (2 * sin_theta_2) if abs(sin_theta_2) > 1e-10 else 0
    n_z = -diff[0, 0].imag / (2 * sin_theta_2) if abs(sin_theta_2) > 1e-10 else 1

    return (n_x * theta, n_y * theta, n_z * theta)


def visualize_discrete_vs_dense():
    """Visualize Clifford (discrete) vs H+T (dense) groups."""
    print("\n" + "=" * 60)
    print("Visualizing Discrete vs Dense Gate Groups")
    print("=" * 60)

    # Generate Clifford group
    clifford = generate_clifford_group()
    print(f"\nClifford group size: {len(clifford)}")

    # Generate H+T approximations
    ht_depth_4 = generate_ht_approximations(4)
    ht_depth_6 = generate_ht_approximations(6)
    print(f"H+T depth 4: {len(ht_depth_4)} gates")
    print(f"H+T depth 6: {len(ht_depth_6)} gates")

    # Compute minimum gaps
    cliff_dist = compute_gate_distances(clifford)
    cliff_gap = minimum_gap(cliff_dist)
    print(f"\nClifford minimum gap: {cliff_gap:.4f}")

    ht4_dist = compute_gate_distances(ht_depth_4)
    ht4_gap = minimum_gap(ht4_dist)
    print(f"H+T depth 4 minimum gap: {ht4_gap:.4f}")

    ht6_dist = compute_gate_distances(ht_depth_6)
    ht6_gap = minimum_gap(ht6_dist)
    print(f"H+T depth 6 minimum gap: {ht6_gap:.4f}")

    print("\nObservation: Clifford gap is large and fixed.")
    print("H+T gap shrinks with depth -> becomes dense!")


def verify_eastin_knill_constraint():
    """Demonstrate the Eastin-Knill constraint numerically."""
    print("\n" + "=" * 60)
    print("Eastin-Knill Constraint Verification")
    print("=" * 60)

    # For a code to have transversal U(theta), we need
    # U(theta)^{otimes n} to preserve the code space for all theta

    # Simulate: if U(theta) = exp(i theta Z) is transversal,
    # what constraints does this place?

    print("\nConsider a hypothetical transversal rotation exp(i theta Z)^{otimes n}")
    print("For this to preserve an error-detecting code:")
    print("  - Must map code space to code space")
    print("  - Must map error spaces to error spaces")

    print("\nFor a stabilizer code with Z stabilizers:")
    print("  - exp(i theta Z)^{otimes n} preserves Z stabilizers")
    print("  - But may not preserve X stabilizers!")

    # Check: Z rotation on all qubits
    theta = 0.1
    Rz = np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]])

    print(f"\nRz({theta:.2f}) X Rz({theta:.2f})^dag =")
    result = Rz @ X @ Rz.conj().T
    print(result.round(4))
    print(f"Is this X? {np.allclose(result, X)}")
    print(f"Is this a Pauli? {np.allclose(result, X) or np.allclose(result, Y) or np.allclose(result, Z)}")

    print("\nResult: Continuous Z-rotation does NOT preserve X!")
    print("Therefore, transversal Rz(theta) is NOT allowed on stabilizer codes.")


def analyze_transversal_group_structure():
    """Analyze the structure of known transversal gate groups."""
    print("\n" + "=" * 60)
    print("Transversal Gate Group Structure Analysis")
    print("=" * 60)

    # Steane code: Clifford group
    clifford = generate_clifford_group()
    print(f"\nSteane [[7,1,3]] transversal group: Clifford")
    print(f"  Size: {len(clifford)}")
    print(f"  Universal: No (missing T)")

    # Theoretical maximum
    print(f"\n  SU(2) has dimension 3 (continuous)")
    print(f"  Clifford is finite subset -> cannot be dense")

    # Surface code: Pauli + controlled-Z type
    print(f"\nSurface code transversal group: Very restricted")
    print(f"  Only X, Z, and limited controlled operations")
    print(f"  Much smaller than Clifford")

    # Color code
    print(f"\nColor code transversal group: Full Clifford")
    print(f"  Similar to Steane but different implementation")

    # Reed-Muller [[15,1,3]]
    print(f"\nReed-Muller [[15,1,3]] transversal group: T-like")
    print(f"  Has transversal T, missing H")
    print(f"  Complementary to Steane!")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 857: Eastin-Knill Theorem - Discreteness")
    print("=" * 60)

    # Part 1: Visualize discrete vs dense
    visualize_discrete_vs_dense()

    # Part 2: Verify constraint
    verify_eastin_knill_constraint()

    # Part 3: Analyze group structures
    analyze_transversal_group_structure()

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("1. Clifford group is discrete with finite gap")
    print("2. H+T group becomes dense (gap -> 0 with depth)")
    print("3. On codes, we're stuck with Clifford-like groups")
    print("4. This is Eastin-Knill: discreteness is forced!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Statement |
|---------|-----------|
| Eastin-Knill | $\mathcal{T}$ is finite for any error-detecting code |
| Discreteness | $\exists \epsilon > 0: U \neq V \Rightarrow \|U - V\| > \epsilon$ |
| Lie algebra | $\text{Lie}(\mathcal{T}) = \{0\}$ (trivial) |
| Clifford size | $\|\text{Cliff}_1\| = 24$ (mod phases) |
| Implication | Transversal gates cannot be universal |

### Main Takeaways

1. **Eastin-Knill Theorem:** No error-detecting code has a transversal universal gate set
2. **Discreteness:** Transversal gates form a discrete (hence finite) group
3. **Proof idea:** Infinitesimal generators conflict with error detection
4. **Clifford is maximal:** For CSS codes, Clifford is often the best achievable
5. **Magic states are necessary:** Non-transversal gates require external resources
6. **Code switching:** Alternative approach using multiple codes

---

## Daily Checklist

- [ ] I can state the Eastin-Knill theorem precisely
- [ ] I understand why transversal gates form a discrete group
- [ ] I can explain the connection to Lie group theory
- [ ] I know the proof sketch (infinitesimal analysis)
- [ ] I understand why this forces us to use magic states
- [ ] I can identify the transversal group for common codes

---

## Preview: Day 858

Tomorrow we dive into the **detailed proof of Eastin-Knill**:

- Rigorous infinitesimal analysis
- The cleaning lemma for transversal gates
- Discreteness from error detection
- Extensions to subsystem codes
- The Bravyi-Koenig classification for topological codes

The proof reveals deep connections between error correction and gate implementation.
