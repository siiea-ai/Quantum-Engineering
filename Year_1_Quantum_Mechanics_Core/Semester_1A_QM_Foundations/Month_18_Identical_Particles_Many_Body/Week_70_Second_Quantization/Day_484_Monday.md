# Day 484: Occupation Number Representation

## Overview

**Day 484 of 2520 | Week 70, Day 1 | Month 18: Identical Particles & Many-Body Physics**

Today we enter the formalism of second quantization, the most powerful framework for describing systems of many identical particles. The occupation number representation provides an elegant way to describe quantum states without explicitly writing out cumbersome symmetrized or antisymmetrized wave functions. This formalism is essential for quantum field theory, condensed matter physics, and modern quantum computing.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Motivation & First vs Second Quantization | 60 min |
| 10:00 AM | Fock Space Construction | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Occupation Number States | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Number Operator & Algebra | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Connection to First Quantization | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the motivation for second quantization and contrast with first quantization
2. **Define** Fock space as a direct sum of N-particle Hilbert spaces
3. **Construct** occupation number states |n₁, n₂, n₃, ...⟩
4. **Apply** the number operator to extract occupation numbers
5. **Derive** properties of occupation number basis states
6. **Connect** second quantization notation to symmetrized wave functions

---

## 1. From First to Second Quantization

### The Challenge of Many-Particle Wave Functions

In first quantization, an N-particle state is written as:

$$\Psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N)$$

For **identical bosons**, this must be symmetric:

$$\Psi(\ldots, \mathbf{r}_i, \ldots, \mathbf{r}_j, \ldots) = \Psi(\ldots, \mathbf{r}_j, \ldots, \mathbf{r}_i, \ldots)$$

For **identical fermions**, antisymmetric:

$$\Psi(\ldots, \mathbf{r}_i, \ldots, \mathbf{r}_j, \ldots) = -\Psi(\ldots, \mathbf{r}_j, \ldots, \mathbf{r}_i, \ldots)$$

**Problems with first quantization:**

1. **Complexity:** Symmetrization requires N! terms (Slater determinants)
2. **Variable particle number:** Cannot easily describe particle creation/annihilation
3. **Labeling ambiguity:** Particle labels are unphysical for identical particles
4. **Practical limitations:** Formulas become unwieldy for large N

### The Second Quantization Philosophy

Instead of asking "where are particles 1, 2, 3, ...?", we ask:

**"How many particles occupy each quantum state?"**

This shift in perspective:
- Eliminates the need for explicit symmetrization
- Naturally handles variable particle numbers
- Treats particle number as a quantum observable
- Connects directly to quantum field theory

### Historical Context

- **1927:** Dirac introduces second quantization for photons
- **1928:** Jordan and Wigner extend to fermions
- **1930s:** Fock develops the systematic framework
- **Today:** Foundation of quantum field theory and many-body physics

---

## 2. Fock Space: The Arena for Second Quantization

### Construction of Fock Space

Let $\mathcal{H}_1$ be the single-particle Hilbert space with basis $\{|\phi_\alpha\rangle\}$.

**N-particle Hilbert space:**

For bosons: $\mathcal{H}_N^{(+)} = \text{Sym}^N(\mathcal{H}_1)$ (symmetric N-fold tensor product)

For fermions: $\mathcal{H}_N^{(-)} = \wedge^N(\mathcal{H}_1)$ (antisymmetric N-fold tensor product)

**Fock space** is the direct sum over all particle numbers:

$$\boxed{\mathcal{F} = \bigoplus_{N=0}^{\infty} \mathcal{H}_N = \mathcal{H}_0 \oplus \mathcal{H}_1 \oplus \mathcal{H}_2 \oplus \cdots}$$

### The Vacuum State

The N = 0 sector contains exactly one state: the **vacuum**

$$|0\rangle = |\text{vac}\rangle$$

**Properties of the vacuum:**
- $\langle 0 | 0 \rangle = 1$ (normalized)
- Contains no particles
- **Not zero!** It is a state, not the absence of a state
- Foundation from which all states are built

### Fock Space Structure

```
Fock Space F
│
├── H₀: |0⟩ (vacuum, 1-dimensional)
│
├── H₁: |φ₁⟩, |φ₂⟩, |φ₃⟩, ... (single-particle states)
│
├── H₂: Symmetrized 2-particle states
│       |φ₁, φ₁⟩, |φ₁, φ₂⟩, |φ₂, φ₂⟩, ...
│
├── H₃: Symmetrized 3-particle states
│
└── ... (continues to all N)
```

### Dimension of Fock Space

For a single-particle space of dimension d:

**Bosons:** dim($\mathcal{H}_N^{(+)}$) = $\binom{N+d-1}{N}$ (stars and bars)

**Fermions:** dim($\mathcal{H}_N^{(-)}$) = $\binom{d}{N}$ (Pauli exclusion)

For infinite-dimensional single-particle spaces (like position), Fock space is infinite-dimensional.

---

## 3. Occupation Number States

### Definition

Given a complete set of single-particle states $\{|\phi_1\rangle, |\phi_2\rangle, |\phi_3\rangle, \ldots\}$, the **occupation number state** is:

$$\boxed{|n_1, n_2, n_3, \ldots\rangle \equiv |\{n_\alpha\}\rangle}$$

where $n_\alpha$ = number of particles in single-particle state $|\phi_\alpha\rangle$.

### Bosons vs Fermions

**Bosons:** $n_\alpha \in \{0, 1, 2, 3, \ldots\}$ (any non-negative integer)

**Fermions:** $n_\alpha \in \{0, 1\}$ (Pauli exclusion principle)

### Total Particle Number

$$N = \sum_\alpha n_\alpha$$

The state $|n_1, n_2, \ldots\rangle$ belongs to the N-particle sector $\mathcal{H}_N$.

### Examples

**Vacuum state:**
$$|0\rangle = |0, 0, 0, \ldots\rangle$$

**Single particle in state 2:**
$$|0, 1, 0, 0, \ldots\rangle$$

**3 bosons: 2 in state 1, 1 in state 3:**
$$|2, 0, 1, 0, \ldots\rangle$$

**2 fermions in states 1 and 3:**
$$|1, 0, 1, 0, \ldots\rangle$$

### Orthonormality

Occupation number states form an orthonormal basis:

$$\boxed{\langle n_1', n_2', \ldots | n_1, n_2, \ldots \rangle = \prod_\alpha \delta_{n_\alpha', n_\alpha}}$$

### Completeness

$$\boxed{\sum_{\{n_\alpha\}} |n_1, n_2, \ldots\rangle \langle n_1, n_2, \ldots| = \hat{I}}$$

where the sum runs over all valid occupation configurations.

---

## 4. The Number Operator

### Definition

The **number operator** for single-particle state α:

$$\boxed{\hat{n}_\alpha |n_1, n_2, \ldots\rangle = n_\alpha |n_1, n_2, \ldots\rangle}$$

The occupation number state is an eigenstate of $\hat{n}_\alpha$ with eigenvalue $n_\alpha$.

### Total Number Operator

$$\boxed{\hat{N} = \sum_\alpha \hat{n}_\alpha}$$

$$\hat{N} |n_1, n_2, \ldots\rangle = \left(\sum_\alpha n_\alpha\right) |n_1, n_2, \ldots\rangle = N |n_1, n_2, \ldots\rangle$$

### Properties of Number Operators

**Hermitian:** $\hat{n}_\alpha^\dagger = \hat{n}_\alpha$

**Commutation:** $[\hat{n}_\alpha, \hat{n}_\beta] = 0$ (different modes commute)

**Eigenvalues:**
- Bosons: $n_\alpha = 0, 1, 2, \ldots$
- Fermions: $n_\alpha = 0, 1$

### Physical Interpretation

The number operator is an **observable**:
- $\langle \hat{n}_\alpha \rangle$ = average occupation of state α
- $\Delta n_\alpha$ = fluctuation in occupation number

For a number eigenstate: $\Delta n_\alpha = 0$ (definite particle number)

---

## 5. Connection to First Quantization Wave Functions

### Bosonic Case: 2 Particles

In first quantization, a symmetrized 2-particle state with particles in $|\phi_1\rangle$ and $|\phi_2\rangle$:

$$|\psi\rangle_{1st} = \frac{1}{\sqrt{2}}(|\phi_1\rangle_1 |\phi_2\rangle_2 + |\phi_2\rangle_1 |\phi_1\rangle_2)$$

In second quantization:
$$|1, 1, 0, 0, \ldots\rangle$$

**Much simpler!** No explicit symmetrization needed.

### Bosonic Case: 2 Particles in Same State

First quantization:
$$|\psi\rangle_{1st} = |\phi_1\rangle_1 |\phi_1\rangle_2$$

Second quantization:
$$|2, 0, 0, \ldots\rangle$$

### Fermionic Case: 2 Particles

Antisymmetrized (Slater determinant):
$$|\psi\rangle_{1st} = \frac{1}{\sqrt{2}}(|\phi_1\rangle_1 |\phi_2\rangle_2 - |\phi_2\rangle_1 |\phi_1\rangle_2)$$

Second quantization:
$$|1, 1, 0, 0, \ldots\rangle$$

**The formalism automatically enforces fermionic antisymmetry!**

### General N-Particle State

**First quantization N-fermion state** (Slater determinant):
$$\Psi = \frac{1}{\sqrt{N!}} \begin{vmatrix} \phi_{\alpha_1}(\mathbf{r}_1) & \phi_{\alpha_1}(\mathbf{r}_2) & \cdots \\ \phi_{\alpha_2}(\mathbf{r}_1) & \phi_{\alpha_2}(\mathbf{r}_2) & \cdots \\ \vdots & \vdots & \ddots \end{vmatrix}$$

**Second quantization:** Simply list occupations!

$$|\ldots, \underbrace{1}_{\alpha_1}, \ldots, \underbrace{1}_{\alpha_2}, \ldots, \underbrace{1}_{\alpha_N}, \ldots\rangle$$

---

## 6. Worked Examples

### Example 1: Counting States

**Problem:** Consider 3 bosons that can occupy 2 single-particle states. List all possible occupation number states and verify the dimension formula.

**Solution:**

Possible states with $n_1 + n_2 = 3$:

| $n_1$ | $n_2$ | State |
|-------|-------|-------|
| 3 | 0 | $\|3, 0\rangle$ |
| 2 | 1 | $\|2, 1\rangle$ |
| 1 | 2 | $\|1, 2\rangle$ |
| 0 | 3 | $\|0, 3\rangle$ |

**4 states total.**

Dimension formula: $\binom{N+d-1}{N} = \binom{3+2-1}{3} = \binom{4}{3} = \boxed{4}$ ✓

### Example 2: Fermionic Constraints

**Problem:** For 2 fermions in a system with 4 single-particle states, how many distinct occupation number states exist?

**Solution:**

Fermions: each $n_\alpha \in \{0, 1\}$ with constraint $\sum_\alpha n_\alpha = 2$.

Must choose 2 states out of 4 to occupy:

$$\binom{4}{2} = 6$$

The states are:
- $|1, 1, 0, 0\rangle$
- $|1, 0, 1, 0\rangle$
- $|1, 0, 0, 1\rangle$
- $|0, 1, 1, 0\rangle$
- $|0, 1, 0, 1\rangle$
- $|0, 0, 1, 1\rangle$

$$\boxed{6 \text{ states}}$$

### Example 3: Superposition State

**Problem:** A bosonic system is in the state:
$$|\psi\rangle = \frac{1}{\sqrt{2}}|2, 0\rangle + \frac{1}{\sqrt{2}}|0, 2\rangle$$

Find $\langle \hat{n}_1 \rangle$, $\langle \hat{n}_2 \rangle$, and $\langle \hat{N} \rangle$.

**Solution:**

$$\langle \hat{n}_1 \rangle = \frac{1}{2}\langle 2,0|\hat{n}_1|2,0\rangle + \frac{1}{2}\langle 0,2|\hat{n}_1|0,2\rangle$$

$$= \frac{1}{2}(2) + \frac{1}{2}(0) = \boxed{1}$$

By symmetry: $\langle \hat{n}_2 \rangle = \boxed{1}$

Total: $\langle \hat{N} \rangle = \langle \hat{n}_1 \rangle + \langle \hat{n}_2 \rangle = \boxed{2}$

**Note:** The total particle number is definite (N = 2), but individual occupations fluctuate!

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Write the occupation number representation for:
(a) 5 bosons all in state $|\phi_3\rangle$
(b) 3 fermions in states $|\phi_1\rangle$, $|\phi_2\rangle$, and $|\phi_5\rangle$

**Problem 1.2:** Calculate $\hat{n}_2 |1, 3, 2, 0\rangle$ and $\hat{N}|1, 3, 2, 0\rangle$.

**Problem 1.3:** For 4 bosons in 3 single-particle states, how many distinct occupation number states exist?

### Level 2: Intermediate

**Problem 2.1:** A system is in state $|\psi\rangle = \frac{1}{\sqrt{3}}|2, 1\rangle + \sqrt{\frac{2}{3}}|1, 2\rangle$. Calculate:
(a) $\langle \hat{n}_1 \rangle$ and $\langle \hat{n}_2 \rangle$
(b) $\langle \hat{n}_1^2 \rangle$ and variance $(\Delta n_1)^2$
(c) Is particle number definite?

**Problem 2.2:** Show that occupation number states for different total N are orthogonal: $\langle N' | N \rangle = \delta_{N'N}$ where $|N\rangle$ represents any state in $\mathcal{H}_N$.

**Problem 2.3:** For fermions, prove that if the same single-particle state is "doubly occupied," the state must be zero: $|..., 2, ...\rangle = 0$.

### Level 3: Challenging

**Problem 3.1:** Consider a coherent superposition in Fock space:
$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}} |n\rangle$$
Show that this is a coherent state with $\langle \hat{n} \rangle = |\alpha|^2$ and $\Delta n = |\alpha|$.

**Problem 3.2:** Derive the dimension of the fermionic Fock space for N fermions in d single-particle states: $\dim(\mathcal{H}_N^{(-)}) = \binom{d}{N}$.

**Problem 3.3:** For a system of N identical bosons in the thermodynamic limit, the Bose-Einstein distribution gives $\langle n_k \rangle = 1/(e^{\beta(\epsilon_k - \mu)} - 1)$. Show that this can lead to macroscopic occupation of the ground state (BEC).

---

## 8. Computational Lab: Fock Space Implementation

```python
"""
Day 484 Computational Lab: Occupation Number Representation
Implementing Fock space states and number operators for many-body systems.
"""

import numpy as np
from scipy.special import comb
from itertools import combinations_with_replacement, combinations
import matplotlib.pyplot as plt

class FockSpace:
    """
    Implementation of Fock space for bosons or fermions.
    """

    def __init__(self, num_modes, particle_type='boson', max_occupation=10):
        """
        Initialize Fock space.

        Parameters:
        -----------
        num_modes : int
            Number of single-particle states
        particle_type : str
            'boson' or 'fermion'
        max_occupation : int
            Maximum occupation per mode (for bosons)
        """
        self.num_modes = num_modes
        self.particle_type = particle_type
        self.max_occupation = max_occupation if particle_type == 'boson' else 1

    def generate_basis(self, N):
        """
        Generate all basis states with N particles.

        Returns:
        --------
        list : List of tuples representing occupation numbers
        """
        if self.particle_type == 'fermion':
            # Choose N modes to occupy
            if N > self.num_modes:
                return []
            return [self._indices_to_occupation(combo)
                    for combo in combinations(range(self.num_modes), N)]
        else:
            # Bosonic: partitions with repetition
            return list(self._boson_partitions(N, self.num_modes))

    def _indices_to_occupation(self, indices):
        """Convert occupied indices to occupation tuple."""
        occupation = [0] * self.num_modes
        for i in indices:
            occupation[i] = 1
        return tuple(occupation)

    def _boson_partitions(self, N, modes, max_per_mode=None):
        """Generate all ways to distribute N bosons in modes."""
        if max_per_mode is None:
            max_per_mode = N

        if modes == 1:
            if N <= max_per_mode:
                yield (N,)
            return

        for n in range(min(N, max_per_mode) + 1):
            for rest in self._boson_partitions(N - n, modes - 1, max_per_mode):
                yield (n,) + rest

    def dimension(self, N):
        """Calculate dimension of N-particle Hilbert space."""
        if self.particle_type == 'fermion':
            if N > self.num_modes:
                return 0
            return int(comb(self.num_modes, N))
        else:
            return int(comb(N + self.num_modes - 1, N))

    def total_dimension(self, max_N):
        """Calculate total Fock space dimension up to max_N particles."""
        return sum(self.dimension(N) for N in range(max_N + 1))


class OccupationState:
    """
    Represents an occupation number state |n1, n2, ...⟩.
    """

    def __init__(self, occupations, particle_type='boson'):
        """
        Initialize occupation number state.

        Parameters:
        -----------
        occupations : list or tuple
            Occupation numbers for each mode
        particle_type : str
            'boson' or 'fermion'
        """
        self.occupations = tuple(occupations)
        self.particle_type = particle_type
        self.num_modes = len(occupations)

        # Validate
        if particle_type == 'fermion':
            if any(n > 1 for n in occupations):
                raise ValueError("Fermionic occupation cannot exceed 1")
        if any(n < 0 for n in occupations):
            raise ValueError("Occupation numbers must be non-negative")

    @property
    def total_particles(self):
        """Total particle number N."""
        return sum(self.occupations)

    def __repr__(self):
        return f"|{', '.join(map(str, self.occupations))}⟩"

    def __eq__(self, other):
        return self.occupations == other.occupations

    def __hash__(self):
        return hash(self.occupations)

    def inner_product(self, other):
        """Calculate ⟨self|other⟩."""
        if self.occupations == other.occupations:
            return 1.0
        return 0.0

    def apply_number_operator(self, mode):
        """
        Apply number operator n_mode to this state.
        Returns (eigenvalue, state).
        """
        if mode >= self.num_modes:
            raise ValueError(f"Mode {mode} does not exist")
        return self.occupations[mode], self


class FockStateVector:
    """
    Represents a general state in Fock space as superposition.
    """

    def __init__(self, particle_type='boson'):
        """Initialize empty state vector."""
        self.particle_type = particle_type
        self.components = {}  # {OccupationState: amplitude}

    def add_component(self, state, amplitude):
        """Add a component to the state."""
        if isinstance(state, (list, tuple)):
            state = OccupationState(state, self.particle_type)

        if state in self.components:
            self.components[state] += amplitude
        else:
            self.components[state] = amplitude

    def normalize(self):
        """Normalize the state vector."""
        norm_sq = sum(abs(amp)**2 for amp in self.components.values())
        norm = np.sqrt(norm_sq)
        for state in self.components:
            self.components[state] /= norm

    def expectation_value(self, operator_func):
        """
        Calculate ⟨ψ|O|ψ⟩ for a diagonal operator.
        operator_func: function that takes OccupationState and returns eigenvalue.
        """
        result = 0
        for state, amp in self.components.items():
            eigenvalue = operator_func(state)
            result += abs(amp)**2 * eigenvalue
        return result

    def expectation_n(self, mode):
        """Calculate ⟨n_mode⟩."""
        return self.expectation_value(lambda s: s.occupations[mode])

    def expectation_N(self):
        """Calculate ⟨N⟩ total particle number."""
        return self.expectation_value(lambda s: s.total_particles)

    def variance_n(self, mode):
        """Calculate variance (Δn_mode)²."""
        mean = self.expectation_n(mode)
        mean_sq = self.expectation_value(lambda s: s.occupations[mode]**2)
        return mean_sq - mean**2

    def __repr__(self):
        terms = []
        for state, amp in self.components.items():
            if abs(amp) > 1e-10:
                terms.append(f"({amp:.4f}){state}")
        return " + ".join(terms)


def demonstrate_fock_space():
    """Demonstrate Fock space construction."""

    print("=" * 60)
    print("FOCK SPACE CONSTRUCTION")
    print("=" * 60)

    # Bosonic Fock space
    print("\n--- Bosonic System: 3 modes ---")
    fock_boson = FockSpace(num_modes=3, particle_type='boson')

    for N in range(5):
        basis = fock_boson.generate_basis(N)
        dim = fock_boson.dimension(N)
        print(f"N = {N}: dimension = {dim}")
        if N <= 2:
            for state in basis:
                print(f"    |{state}⟩")

    # Fermionic Fock space
    print("\n--- Fermionic System: 4 modes ---")
    fock_fermion = FockSpace(num_modes=4, particle_type='fermion')

    for N in range(5):
        basis = fock_fermion.generate_basis(N)
        dim = fock_fermion.dimension(N)
        print(f"N = {N}: dimension = {dim}")
        if N <= 2:
            for state in basis:
                print(f"    |{state}⟩")


def demonstrate_number_operators():
    """Demonstrate number operator operations."""

    print("\n" + "=" * 60)
    print("NUMBER OPERATOR DEMONSTRATION")
    print("=" * 60)

    # Create a state
    state = OccupationState([2, 1, 3, 0], particle_type='boson')
    print(f"\nState: {state}")
    print(f"Total particles: N = {state.total_particles}")

    for mode in range(4):
        n, _ = state.apply_number_operator(mode)
        print(f"n_{mode}|ψ⟩ = {n}|ψ⟩")

    # Superposition state
    print("\n--- Superposition State ---")
    psi = FockStateVector(particle_type='boson')
    psi.add_component([2, 0], 1/np.sqrt(2))
    psi.add_component([0, 2], 1/np.sqrt(2))

    print(f"|ψ⟩ = {psi}")
    print(f"⟨n_1⟩ = {psi.expectation_n(0):.4f}")
    print(f"⟨n_2⟩ = {psi.expectation_n(1):.4f}")
    print(f"⟨N⟩ = {psi.expectation_N():.4f}")
    print(f"Var(n_1) = {psi.variance_n(0):.4f}")


def plot_fock_space_dimensions():
    """Visualize Fock space dimensions."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bosonic dimensions
    ax1 = axes[0]
    modes_list = [2, 3, 4, 5]
    for modes in modes_list:
        fock = FockSpace(num_modes=modes, particle_type='boson')
        N_values = range(8)
        dims = [fock.dimension(N) for N in N_values]
        ax1.plot(N_values, dims, 'o-', label=f'd = {modes} modes')

    ax1.set_xlabel('Number of Particles N', fontsize=12)
    ax1.set_ylabel('Dimension of H_N', fontsize=12)
    ax1.set_title('Bosonic Hilbert Space Dimension\n' +
                  r'$\dim(H_N) = \binom{N+d-1}{N}$', fontsize=12)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Fermionic dimensions
    ax2 = axes[1]
    for modes in modes_list:
        fock = FockSpace(num_modes=modes, particle_type='fermion')
        N_values = range(modes + 1)
        dims = [fock.dimension(N) for N in N_values]
        ax2.plot(N_values, dims, 's-', label=f'd = {modes} modes')

    ax2.set_xlabel('Number of Particles N', fontsize=12)
    ax2.set_ylabel('Dimension of H_N', fontsize=12)
    ax2.set_title('Fermionic Hilbert Space Dimension\n' +
                  r'$\dim(H_N) = \binom{d}{N}$', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fock_space_dimensions.png', dpi=150, bbox_inches='tight')
    plt.show()


def quantum_computing_connection():
    """Demonstrate connection to quantum computing."""

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 60)

    print("""
    OCCUPATION NUMBER REPRESENTATION AND QUBITS
    ============================================

    A fermionic mode with occupation n ∈ {0, 1} is equivalent to a QUBIT!

    Mapping:
    --------
    |0⟩_mode  ←→  |0⟩_qubit  (unoccupied / computational 0)
    |1⟩_mode  ←→  |1⟩_qubit  (occupied / computational 1)

    Multi-mode Systems:
    -------------------
    |n1, n2, n3, ...⟩  ←→  |q1, q2, q3, ...⟩ (multi-qubit state)

    Example: 4 fermionic modes = 4 qubits
    |1, 0, 1, 0⟩ ←→ |1010⟩ in binary notation

    Number Operator:
    ----------------
    n̂_j = |1⟩⟨1|_j  (projection onto occupied state)

    In qubit language: n̂ = (I - Z)/2 = |1⟩⟨1|

    This mapping is the foundation of:
    - Quantum simulation of fermionic systems
    - Jordan-Wigner transformation
    - Variational quantum eigensolvers (VQE)
    """)

    # Example: 3-qubit fermionic state
    print("--- Example: 3-mode Fermionic System ---")
    fock = FockSpace(num_modes=3, particle_type='fermion')

    print("\nAll 2-fermion states (equivalent to 2-electron configurations):")
    for state in fock.generate_basis(2):
        qubit_string = ''.join(map(str, state))
        print(f"  |{state}⟩ ←→ |{qubit_string}⟩")


def simulate_measurement():
    """Simulate measurement in Fock space."""

    print("\n" + "=" * 60)
    print("SIMULATING OCCUPATION NUMBER MEASUREMENTS")
    print("=" * 60)

    # Create superposition
    psi = FockStateVector(particle_type='boson')
    psi.add_component([3, 0], 0.5)
    psi.add_component([2, 1], np.sqrt(0.5))
    psi.add_component([1, 2], 0.5)
    psi.normalize()

    print(f"\nState: |ψ⟩ = {psi}")
    print(f"⟨n_1⟩ = {psi.expectation_n(0):.4f}")
    print(f"⟨n_2⟩ = {psi.expectation_n(1):.4f}")

    # Simulate measurements
    n_measurements = 10000
    results = {tuple(s.occupations): 0 for s in psi.components}
    probabilities = {tuple(s.occupations): abs(amp)**2
                    for s, amp in psi.components.items()}

    # Sample from distribution
    states = list(probabilities.keys())
    probs = list(probabilities.values())

    samples = np.random.choice(len(states), size=n_measurements, p=probs)
    for idx in samples:
        results[states[idx]] += 1

    print(f"\n{n_measurements} measurements:")
    for state, count in results.items():
        freq = count / n_measurements
        prob = probabilities[state]
        print(f"  |{state}⟩: {count} times ({freq:.4f}) [theory: {prob:.4f}]")


# Main execution
if __name__ == "__main__":
    print("Day 484: Occupation Number Representation")
    print("=" * 60)

    demonstrate_fock_space()
    demonstrate_number_operators()
    plot_fock_space_dimensions()
    quantum_computing_connection()
    simulate_measurement()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Fock space | Direct sum of N-particle Hilbert spaces: $\mathcal{F} = \bigoplus_N \mathcal{H}_N$ |
| Vacuum state | Zero-particle state $\|0\rangle$, foundation of Fock space |
| Occupation numbers | $n_\alpha$ = particles in single-particle state α |
| Number operator | $\hat{n}_\alpha\|n_1, n_2, ...\rangle = n_\alpha\|n_1, n_2, ...\rangle$ |
| Bosons vs Fermions | $n_\alpha \in \{0,1,2,...\}$ vs $n_\alpha \in \{0,1\}$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\mathcal{F} = \bigoplus_{N=0}^{\infty} \mathcal{H}_N$$ | Fock space definition |
| $$\hat{N} = \sum_\alpha \hat{n}_\alpha$$ | Total number operator |
| $$\dim(\mathcal{H}_N^{(+)}) = \binom{N+d-1}{N}$$ | Bosonic dimension |
| $$\dim(\mathcal{H}_N^{(-)}) = \binom{d}{N}$$ | Fermionic dimension |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can explain why second quantization is more convenient than first quantization
- [ ] I understand the structure of Fock space as a direct sum
- [ ] I can construct occupation number states for bosons and fermions
- [ ] I understand the role of the vacuum state

### Mathematical Skills
- [ ] I can calculate dimensions of N-particle Hilbert spaces
- [ ] I can apply number operators to occupation states
- [ ] I can compute expectation values in superposition states
- [ ] I can verify orthonormality of occupation number states

### Computational Skills
- [ ] I implemented Fock space basis generation
- [ ] I computed Hilbert space dimensions for bosons and fermions
- [ ] I simulated measurements in occupation number basis

### Quantum Computing Connection
- [ ] I see the mapping between fermionic modes and qubits
- [ ] I understand occupation number as qubit computational basis
- [ ] I recognize relevance to quantum simulation

---

## 11. Preview: Day 485

Tomorrow we introduce **bosonic creation and annihilation operators**:

- Definition of $\hat{a}^\dagger$ and $\hat{a}$
- Canonical commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$
- Action on Fock states: building states from vacuum
- Connection to harmonic oscillator ladder operators
- Multi-mode commutation relations

These operators are the fundamental building blocks for constructing and manipulating many-body states.

---

## References

1. Fetter, A.L. & Walecka, J.D. (2003). *Quantum Theory of Many-Particle Systems*. Dover, Ch. 1.

2. Negele, J.W. & Orland, H. (1998). *Quantum Many-Particle Systems*. Westview Press, Ch. 1.

3. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.

4. Coleman, P. (2015). *Introduction to Many-Body Physics*. Cambridge University Press, Ch. 2.

---

*"The occupation number representation is the natural language for systems where the number of particles can change, or where the particles are truly indistinguishable."*
— Alexander Fetter

---

**Day 484 Complete.** Tomorrow: Bosonic Creation and Annihilation Operators.
