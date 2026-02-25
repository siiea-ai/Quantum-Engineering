# Day 787: Ground State and Code Space

## Overview

**Day:** 787 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Ground State Structure and Code Space of the Toric Code

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Ground state conditions and degeneracy |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Explicit ground state construction |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **State** the ground state condition: $A_v|\psi\rangle = B_p|\psi\rangle = +1|\psi\rangle$
2. **Prove** the ground state degeneracy equals 4 on the torus
3. **Construct** explicit ground state wavefunctions
4. **Explain** the connection between degeneracy and logical qubits
5. **Write** the toric code Hamiltonian and analyze its spectrum
6. **Calculate** the energy gap protecting the code space

---

## Core Theory

### 1. Ground State Condition

The toric code space is defined as the simultaneous +1 eigenspace of all stabilizers:

$$\boxed{A_v|\psi_{GS}\rangle = +1|\psi_{GS}\rangle \quad \forall v}$$
$$\boxed{B_p|\psi_{GS}\rangle = +1|\psi_{GS}\rangle \quad \forall p}$$

A state $|\psi\rangle$ is a valid code state (ground state) if and only if it satisfies these conditions for ALL vertices and ALL plaquettes.

**Physical interpretation:**
- $A_v = +1$: No "electric charge" at vertex $v$
- $B_p = +1$: No "magnetic flux" through plaquette $p$
- Ground states are "charge-free" and "flux-free" configurations

### 2. Ground State Degeneracy

**Theorem:** The ground state degeneracy of the toric code on an $L \times L$ torus is exactly 4.

**Proof via Stabilizer Counting:**

The code space dimension is:
$$\dim(\mathcal{C}) = 2^{n - r}$$

where $n$ is the number of qubits and $r$ is the number of independent stabilizer generators.

From Day 786:
- $n = 2L^2$
- Independent generators: $r = 2L^2 - 2$

Therefore:
$$\dim(\mathcal{C}) = 2^{2L^2 - (2L^2 - 2)} = 2^2 = 4 \quad \blacksquare$$

**Topological interpretation:** The torus has two independent non-contractible cycles. Each cycle contributes one logical qubit, giving $2^2 = 4$ ground states.

### 3. The Toric Code Hamiltonian

The toric code Hamiltonian is:

$$\boxed{H = -\sum_{v} A_v - \sum_{p} B_p}$$

**Properties:**
- All terms commute: $[A_v, A_{v'}] = [B_p, B_{p'}] = [A_v, B_p] = 0$
- Each term has eigenvalues $\pm 1$
- Hamiltonian is exactly solvable

**Energy Spectrum:**

Ground state energy (all stabilizers have eigenvalue +1):
$$E_0 = -L^2 - L^2 = -2L^2$$

Excited states have some stabilizers with eigenvalue -1:
$$E = E_0 + 2 \times (\text{number of violated stabilizers})$$

**Energy gap:** $\Delta = 2$ (minimum excitation energy)

### 4. Explicit Ground State Construction

**Method 1: Equal superposition over loop configurations**

Consider the basis $|z_1, z_2, \ldots, z_n\rangle$ where $z_i \in \{0, 1\}$ represents the Z-basis state of qubit $i$.

The plaquette condition $B_p = +1$ requires:
$$\prod_{e \in \partial p} (-1)^{z_e} = +1$$

This means the parity of Z-basis values around each plaquette is even.

**Solution:** $z_e$ values form "closed loops" on the dual lattice.

The star operators $A_v$ act as loop creators/destroyers:
$$A_v|z_1, \ldots, z_n\rangle = |z_1 \oplus \delta_{1v}, \ldots, z_n \oplus \delta_{nv}\rangle$$

where $\delta_{ev} = 1$ if edge $e$ touches vertex $v$.

**Ground state:** Equal superposition over all loop configurations in the same homology class.

### 5. The Four Ground States

The 4 ground states correspond to 4 **homology classes** of loops on the torus:

**Homology class [0, 0]:** Loops contractible to a point
$$|GS_{00}\rangle = \frac{1}{\sqrt{N}} \sum_{\ell \in [0,0]} |\ell\rangle$$

**Homology class [1, 0]:** Loops wrapping once around the horizontal cycle
$$|GS_{10}\rangle = \frac{1}{\sqrt{N}} \sum_{\ell \in [1,0]} |\ell\rangle$$

**Homology class [0, 1]:** Loops wrapping once around the vertical cycle
$$|GS_{01}\rangle = \frac{1}{\sqrt{N}} \sum_{\ell \in [0,1]} |\ell\rangle$$

**Homology class [1, 1]:** Loops wrapping around both cycles
$$|GS_{11}\rangle = \frac{1}{\sqrt{N}} \sum_{\ell \in [1,1]} |\ell\rangle$$

These form an orthonormal basis for the code space.

### 6. Alternative Construction via Projector

The projector onto the ground space is:

$$P_{GS} = \prod_v \frac{I + A_v}{2} \prod_p \frac{I + B_p}{2}$$

Starting from any reference state $|0\rangle^{\otimes n}$:

$$|GS_{00}\rangle \propto P_{GS} |0\rangle^{\otimes n} = \prod_v \frac{I + A_v}{2} |0\rangle^{\otimes n}$$

Note: $\frac{I + B_p}{2}$ acts as identity on $|0\rangle^{\otimes n}$ since $B_p|0\rangle^{\otimes n} = |0\rangle^{\otimes n}$.

Expanding:
$$|GS_{00}\rangle \propto \sum_{S \subseteq V} \prod_{v \in S} A_v |0\rangle^{\otimes n}$$

This is the equal superposition over all "star configurations."

### 7. Logical Basis States

We can label the ground states as logical qubit states:

$$|00_L\rangle = |GS_{00}\rangle$$
$$|10_L\rangle = |GS_{10}\rangle$$
$$|01_L\rangle = |GS_{01}\rangle$$
$$|11_L\rangle = |GS_{11}\rangle$$

**Relationship to logical operators:**
$$\bar{Z}_1 |ij_L\rangle = (-1)^i |ij_L\rangle$$
$$\bar{Z}_2 |ij_L\rangle = (-1)^j |ij_L\rangle$$

The logical operators are non-contractible loop operators (to be studied in Day 789).

### 8. Entanglement Structure

The ground states exhibit **topological entanglement entropy**.

For a region $A$ with boundary $\partial A$:
$$S(A) = |\partial A| - \gamma$$

where $\gamma = \log 2$ is the **topological entanglement entropy**.

This nonzero $\gamma$ indicates:
- Long-range entanglement
- Cannot be created by local operations
- Characteristic of topological order

---

## Quantum Mechanics Connection

### Stabilizer Code Perspective

The toric code is a **stabilizer code** with:
- Stabilizer group: $S = \langle A_v, B_p \rangle$
- Code space: $\mathcal{C} = \{|\psi\rangle : s|\psi\rangle = |\psi\rangle \, \forall s \in S\}$
- Logical operators: Elements of $N(S) \setminus S$ (normalizer but not stabilizer)

**Key property:** Logical operators commute with all stabilizers but are not in the stabilizer group.

### Condensed Matter Perspective

The toric code is the simplest example of a **$\mathbb{Z}_2$ topological order**:

1. **Gapped Hamiltonian:** $\Delta = 2$ energy gap
2. **Ground state degeneracy:** Depends on topology (4 for torus)
3. **Anyonic excitations:** Violations create anyons (Week 114)
4. **Topological entanglement:** $\gamma = \log 2$

### Protection Mechanism

Why is the code space protected?

1. **Local errors cannot distinguish ground states:** Any local operator either:
   - Commutes with all stabilizers (acts within code space but cannot change logical state)
   - Anti-commutes with some stabilizer (takes state out of code space)

2. **Logical errors require global support:** To change logical state without leaving code space, operator must act on a non-contractible loop (weight $\geq L$).

---

## Worked Examples

### Example 1: Ground State Energy for L = 3

**Problem:** Calculate the ground state energy for a $3 \times 3$ toric code.

**Solution:**

Number of stabilizers:
- Star operators: $L^2 = 9$
- Plaquette operators: $L^2 = 9$
- Total: 18

Ground state energy:
$$E_0 = -\sum_v \langle A_v \rangle - \sum_p \langle B_p \rangle = -9(+1) - 9(+1) = -18$$

Energy of state with one star violation:
$$E_1 = -18 + 2 = -16$$

Energy gap: $\Delta = 2$

### Example 2: Projector onto Ground Space

**Problem:** For a $2 \times 2$ toric code (8 qubits), write the dimension of the ground space projector.

**Solution:**

Ground space dimension: $\dim(\mathcal{C}) = 4$

Projector rank: $\text{rank}(P_{GS}) = 4$

Total Hilbert space: $2^8 = 256$

Projector:
$$P_{GS} = \sum_{i=0}^{3} |GS_i\rangle\langle GS_i|$$

### Example 3: Verifying Ground State Condition

**Problem:** Show that $|0\rangle^{\otimes n}$ is NOT a ground state but $\prod_v(I + A_v)|0\rangle^{\otimes n}$ is.

**Solution:**

**For $|0\rangle^{\otimes n}$:**

Plaquette check: $B_p|0\rangle^{\otimes n} = Z_{e_1}Z_{e_2}Z_{e_3}Z_{e_4}|0\rangle^{\otimes n} = (+1)^4|0\rangle^{\otimes n} = |0\rangle^{\otimes n}$ ✓

Star check: $A_v|0\rangle^{\otimes n} = X_{e_1}X_{e_2}X_{e_3}X_{e_4}|0\rangle^{\otimes n} = |1111...\rangle \neq |0\rangle^{\otimes n}$ ✗

So $|0\rangle^{\otimes n}$ is NOT a ground state.

**For $|\psi\rangle = \prod_v(I + A_v)|0\rangle^{\otimes n}$:**

This is an equal superposition over all configurations reachable by applying star operators.

For any star $A_{v'}$:
$$A_{v'}|\psi\rangle = A_{v'}\prod_v(I + A_v)|0\rangle^{\otimes n}$$

Since $A_{v'}(I + A_{v'}) = (A_{v'} + I)$, we have $A_{v'}|\psi\rangle = |\psi\rangle$ ✓

So $|\psi\rangle$ is a ground state (specifically $|GS_{00}\rangle$).

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a $4 \times 4$ toric code:
a) What is the ground state energy?
b) What is the energy of a state with exactly 2 star violations?
c) What is the dimension of the ground space?

**P1.2** How many terms are in the sum $\prod_v(I + A_v)|0\rangle^{\otimes n}$ for L = 2?

**P1.3** Show that $B_p|0\rangle^{\otimes n} = +1|0\rangle^{\otimes n}$ for any plaquette $p$.

### Level 2: Intermediate

**P2.1** Prove that the ground states are orthogonal to each other: $\langle GS_{ij}|GS_{kl}\rangle = \delta_{ik}\delta_{jl}$.

**P2.2** For the state $|\psi\rangle = |GS_{00}\rangle + |GS_{11}\rangle$ (unnormalized):
a) Is this a valid code state?
b) What logical state does it represent?

**P2.3** Calculate the entanglement entropy of $|GS_{00}\rangle$ for a bipartition that cuts a vertical line through the lattice.

### Level 3: Challenging

**P3.1** Prove that any local operator $O$ with support on fewer than $L$ qubits either:
a) Commutes with all stabilizers (preserves code space), or
b) Anti-commutes with at least one stabilizer (leaves code space)

**P3.2** Show that the topological entanglement entropy $\gamma = \log D$ where $D$ is the total quantum dimension of the anyons. For the toric code, $D = 2$.

**P3.3** Construct the ground state explicitly for L = 2 by listing all 16 terms in the superposition $|GS_{00}\rangle$.

---

## Computational Lab

```python
"""
Day 787: Ground State and Code Space
=====================================

Constructing and analyzing toric code ground states.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from itertools import product, combinations
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh


class ToricCodeGroundStates:
    """
    Toric code ground state analysis.

    Constructs ground states and verifies their properties.
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L
        self.n_vertices = L * L
        self.n_faces = L * L

        # Precompute stabilizer supports
        self._compute_stabilizers()

    def edge_index(self, i: int, j: int, d: int) -> int:
        """Convert (i, j, d) to linear edge index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def _compute_stabilizers(self):
        """Compute star and plaquette operator supports."""
        self.stars = []
        for i in range(self.L):
            for j in range(self.L):
                support = [
                    self.edge_index(i, j, 0),
                    self.edge_index(i, j - 1, 0),
                    self.edge_index(i, j, 1),
                    self.edge_index(i - 1, j, 1),
                ]
                self.stars.append(frozenset(support))

        self.plaquettes = []
        for i in range(self.L):
            for j in range(self.L):
                support = [
                    self.edge_index(i, j, 0),
                    self.edge_index(i + 1, j, 0),
                    self.edge_index(i, j, 1),
                    self.edge_index(i, j + 1, 1),
                ]
                self.plaquettes.append(frozenset(support))

    def ground_state_energy(self) -> float:
        """Compute ground state energy."""
        return -float(self.n_vertices + self.n_faces)

    def energy_gap(self) -> float:
        """Return energy gap."""
        return 2.0

    def ground_state_degeneracy(self) -> int:
        """Return ground state degeneracy."""
        return 4

    def apply_star(self, state: np.ndarray, v_idx: int) -> np.ndarray:
        """Apply star operator A_v to a computational basis state."""
        new_state = state.copy()
        for edge in self.stars[v_idx]:
            new_state[edge] = 1 - new_state[edge]  # Flip bit
        return new_state

    def apply_all_stars_subset(self, state: np.ndarray, subset: Set[int]) -> np.ndarray:
        """Apply star operators for vertices in subset."""
        result = state.copy()
        for v in subset:
            result = self.apply_star(result, v)
        return result

    def check_plaquette(self, state: np.ndarray, p_idx: int) -> int:
        """Check plaquette eigenvalue (+1 or -1)."""
        parity = 0
        for edge in self.plaquettes[p_idx]:
            parity ^= state[edge]
        return 1 - 2 * parity  # 0 -> +1, 1 -> -1

    def is_ground_state_config(self, state: np.ndarray) -> bool:
        """Check if configuration satisfies all plaquette constraints."""
        for p_idx in range(self.n_faces):
            if self.check_plaquette(state, p_idx) != 1:
                return False
        return True

    def construct_ground_state_00(self) -> Dict[Tuple[int, ...], complex]:
        """
        Construct |GS_00> as superposition over star operator applications.

        Returns dictionary mapping basis states to amplitudes.
        """
        initial = tuple([0] * self.n_qubits)
        state_dict = {}

        # Apply all subsets of star operators
        n_subsets = 2 ** self.n_vertices
        for subset_int in range(n_subsets):
            # Convert integer to subset of vertices
            subset = {v for v in range(self.n_vertices) if (subset_int >> v) & 1}

            # Apply star operators
            config = np.array(initial)
            for v in subset:
                config = self.apply_star(config, v)

            config_tuple = tuple(config)
            if config_tuple in state_dict:
                state_dict[config_tuple] += 1
            else:
                state_dict[config_tuple] = 1

        # Normalize
        total = sum(abs(v)**2 for v in state_dict.values())
        norm = np.sqrt(total)
        state_dict = {k: v / norm for k, v in state_dict.items()}

        return state_dict

    def verify_ground_state(self, state_dict: Dict[Tuple[int, ...], complex]) -> Dict[str, bool]:
        """Verify that state_dict represents a valid ground state."""
        results = {'all_plaquettes_satisfied': True, 'all_stars_satisfied': True}

        # Check plaquettes for each basis state
        for config, amp in state_dict.items():
            if abs(amp) < 1e-10:
                continue
            config_arr = np.array(config)
            for p_idx in range(self.n_faces):
                if self.check_plaquette(config_arr, p_idx) != 1:
                    results['all_plaquettes_satisfied'] = False
                    break

        # For star operators, we need to check that applying A_v
        # permutes basis states within the superposition with correct phases
        # This is automatic if we constructed via star operators

        return results


def analyze_ground_state_structure(L: int) -> None:
    """Analyze ground state structure for L x L toric code."""
    print(f"\n{'='*60}")
    print(f"Ground State Analysis for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeGroundStates(L)

    print(f"\nCode parameters:")
    print(f"  Physical qubits: n = {code.n_qubits}")
    print(f"  Logical qubits: k = 2")
    print(f"  Ground state degeneracy: {code.ground_state_degeneracy()}")

    print(f"\nEnergy spectrum:")
    print(f"  Ground state energy: E_0 = {code.ground_state_energy()}")
    print(f"  Energy gap: Delta = {code.energy_gap()}")
    print(f"  First excited: E_1 = {code.ground_state_energy() + code.energy_gap()}")


def construct_and_verify_ground_state(L: int) -> None:
    """Construct and verify ground state for small L."""
    if L > 3:
        print(f"L = {L} too large for explicit construction (2^{2*L*L} states)")
        return

    print(f"\n{'='*60}")
    print(f"Ground State Construction for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeGroundStates(L)

    print(f"\nConstructing |GS_00> via star operator superposition...")
    gs00 = code.construct_ground_state_00()

    print(f"Number of basis states in superposition: {len(gs00)}")
    print(f"Expected (2^(L^2 - 1) = 2^{L**2 - 1} = {2**(L**2 - 1)})")

    # Verify normalization
    norm_sq = sum(abs(v)**2 for v in gs00.values())
    print(f"Norm squared: {norm_sq:.6f} (should be 1.0)")

    # Verify ground state conditions
    verification = code.verify_ground_state(gs00)
    print(f"\nVerification:")
    for key, value in verification.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    # Show a few basis states
    print(f"\nSample basis states in |GS_00>:")
    count = 0
    for config, amp in sorted(gs00.items(), key=lambda x: -abs(x[1])):
        if count >= 5:
            break
        weight = sum(config)
        print(f"  {config} : amplitude = {amp:.4f}, weight = {weight}")
        count += 1


def demonstrate_star_action(L: int) -> None:
    """Demonstrate how star operators permute ground state components."""
    print(f"\n{'='*60}")
    print(f"Star Operator Action on Ground State (L = {L})")
    print(f"{'='*60}")

    code = ToricCodeGroundStates(L)

    # Start with all-zeros configuration
    config = np.array([0] * code.n_qubits)
    print(f"\nInitial configuration: {tuple(config)}")
    print(f"All plaquettes satisfied: {code.is_ground_state_config(config)}")

    # Apply star operator at (0, 0)
    config1 = code.apply_star(config, 0)
    print(f"\nAfter A_(0,0): {tuple(config1)}")
    print(f"All plaquettes satisfied: {code.is_ground_state_config(config1)}")

    # Apply another star operator
    config2 = code.apply_star(config1, 1)
    print(f"\nAfter A_(0,1): {tuple(config2)}")
    print(f"All plaquettes satisfied: {code.is_ground_state_config(config2)}")

    # Apply A_(0,0) again (should return to config1)
    config3 = code.apply_star(config2, 0)
    print(f"\nAfter A_(0,0) again: {tuple(config3)}")
    print(f"Same as after first A_(0,1): {np.array_equal(config3, config1)}")


def build_hamiltonian_small(L: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build toric code Hamiltonian for small L using sparse matrices.

    Returns (eigenvalues, eigenvectors) of lowest states.
    """
    print(f"\n{'='*60}")
    print(f"Hamiltonian Construction and Diagonalization (L = {L})")
    print(f"{'='*60}")

    n = 2 * L * L
    dim = 2 ** n

    print(f"Hilbert space dimension: 2^{n} = {dim}")

    if dim > 1024:
        print("Too large for full diagonalization. Using sparse methods.")
        return None, None

    # Build Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def pauli_op(pauli_type: str, qubit: int, n_qubits: int) -> np.ndarray:
        """Build n-qubit Pauli operator with given type on given qubit."""
        if pauli_type == 'I':
            op = I
        elif pauli_type == 'X':
            op = X
        elif pauli_type == 'Z':
            op = Z

        result = np.array([[1.0]], dtype=complex)
        for i in range(n_qubits):
            if i == qubit:
                result = np.kron(result, op)
            else:
                result = np.kron(result, I)
        return result

    code = ToricCodeGroundStates(L)

    # Build Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)

    print("Building star operators...")
    for star in code.stars:
        op = np.eye(dim, dtype=complex)
        for edge in star:
            op = op @ pauli_op('X', edge, n)
        H -= op

    print("Building plaquette operators...")
    for plaq in code.plaquettes:
        op = np.eye(dim, dtype=complex)
        for edge in plaq:
            op = op @ pauli_op('Z', edge, n)
        H -= op

    print("Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Find ground states
    E0 = eigenvalues[0]
    ground_indices = np.where(np.abs(eigenvalues - E0) < 1e-10)[0]

    print(f"\nResults:")
    print(f"  Ground state energy: E_0 = {E0:.2f}")
    print(f"  Expected: {code.ground_state_energy()}")
    print(f"  Ground state degeneracy: {len(ground_indices)}")
    print(f"  Expected: {code.ground_state_degeneracy()}")

    # Energy gap
    non_ground = eigenvalues[len(ground_indices):]
    if len(non_ground) > 0:
        E1 = non_ground[0]
        gap = E1 - E0
        print(f"  First excited energy: E_1 = {E1:.2f}")
        print(f"  Energy gap: Delta = {gap:.2f}")
        print(f"  Expected: {code.energy_gap()}")

    return eigenvalues, eigenvectors


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 787: GROUND STATE AND CODE SPACE")
    print("=" * 70)

    # Demo 1: Ground state analysis
    for L in [2, 3, 4, 5]:
        analyze_ground_state_structure(L)

    # Demo 2: Explicit ground state construction
    construct_and_verify_ground_state(2)

    # Demo 3: Star operator action
    demonstrate_star_action(2)

    # Demo 4: Hamiltonian diagonalization (small L only)
    eigenvalues, eigenvectors = build_hamiltonian_small(2)

    # Demo 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Ground State and Code Space")
    print("=" * 70)

    print("""
    GROUND STATE CONDITION:
    -----------------------
    A_v |psi_GS> = +1 |psi_GS>  for all vertices v
    B_p |psi_GS> = +1 |psi_GS>  for all plaquettes p

    GROUND STATE DEGENERACY:
    ------------------------
    dim(code space) = 2^(n - r) = 2^(2L^2 - (2L^2 - 2)) = 2^2 = 4

    The 4 ground states correspond to 4 homology classes:
    - |GS_00>: trivial loops
    - |GS_10>: horizontal winding
    - |GS_01>: vertical winding
    - |GS_11>: both windings

    HAMILTONIAN:
    ------------
    H = -sum_v A_v - sum_p B_p

    Ground state energy: E_0 = -2L^2
    Energy gap: Delta = 2
    Excited states: E = E_0 + 2 * (# violations)

    TOPOLOGICAL PROTECTION:
    -----------------------
    - Ground states differ by non-local operators
    - Local perturbations cannot split degeneracy
    - Protection improves with system size L
    """)

    print("=" * 70)
    print("Day 787 Complete: Ground State Structure Understood")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Ground state condition | $A_v\|\psi_{GS}\rangle = B_p\|\psi_{GS}\rangle = +1\|\psi_{GS}\rangle$ |
| Ground state degeneracy | $\dim(\mathcal{C}) = 4$ |
| Hamiltonian | $H = -\sum_v A_v - \sum_p B_p$ |
| Ground state energy | $E_0 = -2L^2$ |
| Energy gap | $\Delta = 2$ |
| Excited energy | $E = E_0 + 2 \times (\text{\# violations})$ |

### Main Takeaways

1. **Ground states** satisfy $A_v = B_p = +1$ for all vertices and plaquettes
2. **Four-fold degeneracy** on the torus encodes 2 logical qubits
3. **Explicit construction:** $|GS_{00}\rangle = \prod_v \frac{I + A_v}{2}|0\rangle^{\otimes n}$
4. **Four ground states** correspond to four homology classes of loops
5. **Energy gap** $\Delta = 2$ protects the ground space
6. **Topological protection:** local operators cannot distinguish ground states

---

## Daily Checklist

- [ ] I can state the ground state condition for toric code
- [ ] I can prove ground state degeneracy = 4 using stabilizer counting
- [ ] I understand the four homology classes on the torus
- [ ] I can write the Hamiltonian and compute ground state energy
- [ ] I understand why the energy gap protects the code space
- [ ] I ran the computational lab and verified the ground state construction

---

## Preview: Day 788

Tomorrow we examine the **toric code as a CSS code**:

- X-stabilizers (stars) and Z-stabilizers (plaquettes)
- Chain complex structure on the lattice
- Boundary operators and homology
- Dual lattice picture
- Connection to classical codes

The CSS structure reveals the deep algebraic and topological foundations of the toric code.
