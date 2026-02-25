# Day 305: Addition of Angular Momenta

## Overview

**Month 11, Week 44, Day 4 — Thursday**

Today we learn how to combine angular momenta from different sources — orbital plus spin, or two spins, or the angular momenta of two particles. This is the mathematical foundation for understanding atomic structure, nuclear physics, and quantum entanglement.

## Learning Objectives

1. Understand the tensor product structure of composite systems
2. Master the uncoupled and coupled bases
3. Derive the range of total angular momentum
4. Apply to physical examples: singlet/triplet, spin-orbit coupling
5. Prepare for Clebsch-Gordan coefficients

---

## 1. The Physical Problem

### Why Add Angular Momenta?

**Orbital + Spin:** An electron has both orbital ($\mathbf{L}$) and spin ($\mathbf{S}$) angular momentum.

**Two Electrons:** Two electrons in an atom each have spin.

**Nuclear + Electronic:** Nuclear spin couples to electronic angular momentum.

In all cases, we need the **total angular momentum**:
$$\mathbf{J} = \mathbf{J}_1 + \mathbf{J}_2$$

### The Question

Given systems with angular momenta $j_1$ and $j_2$, what are the allowed values of total $j$?

---

## 2. Tensor Product Space

### The Combined Hilbert Space

$$\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2$$

Dimension: $(2j_1 + 1)(2j_2 + 1)$

### Operators on Composite System

$$\hat{J}_{1z} \to \hat{J}_{1z} \otimes \hat{I}_2$$
$$\hat{J}_{2z} \to \hat{I}_1 \otimes \hat{J}_{2z}$$

Total operators:
$$\hat{J}_z = \hat{J}_{1z} + \hat{J}_{2z}$$
$$\hat{\mathbf{J}} = \hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2$$

### Commutation Relations

$$[\hat{J}_{1i}, \hat{J}_{2j}] = 0 \quad \text{(different systems)}$$

The total $\hat{\mathbf{J}}$ satisfies:
$$[\hat{J}_i, \hat{J}_j] = i\hbar\epsilon_{ijk}\hat{J}_k$$

---

## 3. The Uncoupled Basis

### Product States

$$|j_1, m_1\rangle \otimes |j_2, m_2\rangle \equiv |j_1, m_1; j_2, m_2\rangle$$

These are simultaneous eigenstates of $\{\hat{J}_1^2, \hat{J}_{1z}, \hat{J}_2^2, \hat{J}_{2z}\}$.

### Example: Two Spin-1/2 Particles

Dimension: $2 \times 2 = 4$

Basis: $|\uparrow\uparrow\rangle, |\uparrow\downarrow\rangle, |\downarrow\uparrow\rangle, |\downarrow\downarrow\rangle$

where $|\uparrow\downarrow\rangle \equiv |{+}\rangle_1 \otimes |{-}\rangle_2$.

### Total $m$ in Uncoupled Basis

$$\hat{J}_z|j_1, m_1; j_2, m_2\rangle = \hbar(m_1 + m_2)|j_1, m_1; j_2, m_2\rangle$$

The total magnetic quantum number is:
$$m = m_1 + m_2$$

---

## 4. The Coupled Basis

### A Different Set of Quantum Numbers

We seek eigenstates of $\{\hat{J}_1^2, \hat{J}_2^2, \hat{J}^2, \hat{J}_z\}$:

$$|j_1, j_2; j, m\rangle$$

where:
- $\hat{J}^2|j_1, j_2; j, m\rangle = \hbar^2 j(j+1)|j_1, j_2; j, m\rangle$
- $\hat{J}_z|j_1, j_2; j, m\rangle = \hbar m|j_1, j_2; j, m\rangle$

### The Key Question

**What values of $j$ are possible given $j_1$ and $j_2$?**

---

## 5. Range of Total Angular Momentum

### The Triangle Rule

$$\boxed{j = |j_1 - j_2|, |j_1 - j_2| + 1, \ldots, j_1 + j_2 - 1, j_1 + j_2}$$

### Derivation via Counting

**Maximum $m$:** $m_{\max} = j_1 + j_2$ (unique state)

**States with $m = m_{\max}$:** Only $|j_1, j_1; j_2, j_2\rangle$

This must be $|j_1, j_2; j_{\max}, j_{\max}\rangle$ with $j_{\max} = j_1 + j_2$.

**States with $m = m_{\max} - 1$:**
$$|j_1, j_1-1; j_2, j_2\rangle \quad \text{and} \quad |j_1, j_1; j_2, j_2-1\rangle$$

Two states. One belongs to $j = j_1 + j_2$, the other starts $j = j_1 + j_2 - 1$.

**Continue counting** to find all $j$ values.

### Dimension Check

$$(2j_1 + 1)(2j_2 + 1) = \sum_{j=|j_1-j_2|}^{j_1+j_2}(2j+1)$$

**Example:** $j_1 = 1, j_2 = 1/2$

LHS: $3 \times 2 = 6$

RHS: $j = 1/2$ gives $2$, $j = 3/2$ gives $4$. Total = $6$ ✓

---

## 6. The Basis Change

### Relation Between Bases

$$|j_1, j_2; j, m\rangle = \sum_{m_1, m_2} |j_1, m_1; j_2, m_2\rangle \langle j_1, m_1; j_2, m_2|j_1, j_2; j, m\rangle$$

The coefficients $\langle j_1, m_1; j_2, m_2|j_1, j_2; j, m\rangle$ are the **Clebsch-Gordan coefficients**.

### Notation

$$C^{jm}_{j_1 m_1; j_2 m_2} \equiv \langle j_1, m_1; j_2, m_2|j_1, j_2; j, m\rangle$$

### Selection Rule

$$C^{jm}_{j_1 m_1; j_2 m_2} = 0 \quad \text{unless} \quad m = m_1 + m_2$$

---

## 7. Example: Two Spin-1/2 Particles

### The Setup

$j_1 = j_2 = 1/2$

Possible total $j$: $|1/2 - 1/2| = 0$ to $1/2 + 1/2 = 1$

So $j = 0$ (singlet) or $j = 1$ (triplet).

### Finding the States

**$j = 1, m = 1$:**
$$|1, 1\rangle = |\uparrow\uparrow\rangle$$

**$j = 1, m = 0$:**
Apply $J_- = J_{1-} + J_{2-}$ to $|1, 1\rangle$:
$$J_-|1, 1\rangle = \hbar\sqrt{2}|1, 0\rangle$$
$$(J_{1-} + J_{2-})|\uparrow\uparrow\rangle = \hbar(|\downarrow\uparrow\rangle + |\uparrow\downarrow\rangle)$$

$$\boxed{|1, 0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)}$$

**$j = 1, m = -1$:**
$$|1, -1\rangle = |\downarrow\downarrow\rangle$$

**$j = 0, m = 0$:**
Must be orthogonal to $|1, 0\rangle$:
$$\boxed{|0, 0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)}$$

### Summary: Singlet and Triplet

**Triplet ($j = 1$, symmetric):**
$$|1, 1\rangle = |\uparrow\uparrow\rangle$$
$$|1, 0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$$
$$|1, -1\rangle = |\downarrow\downarrow\rangle$$

**Singlet ($j = 0$, antisymmetric):**
$$|0, 0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

---

## 8. Physical Applications

### Spin-Orbit Coupling

For an electron: $\mathbf{J} = \mathbf{L} + \mathbf{S}$

If $\ell = 1, s = 1/2$: possible $j = 1/2$ or $3/2$.

The Hamiltonian $\hat{H}_{SO} \propto \mathbf{L}\cdot\mathbf{S}$ splits these levels.

$$\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(\mathbf{J}^2 - \mathbf{L}^2 - \mathbf{S}^2)$$

### Helium Atom

Two electrons with spins: total spin $S = 0$ (para-helium) or $S = 1$ (ortho-helium).

**Para-helium:** Singlet spin, symmetric spatial wavefunction
**Ortho-helium:** Triplet spin, antisymmetric spatial wavefunction

### Nuclear Physics

Proton + neutron in deuteron: $j_p = j_n = 1/2$

Total spin $S = 0$ or $1$. Ground state has $S = 1$ (triplet).

---

## 9. Representation Theory Perspective

### Tensor Product of Representations

From Week 43:
$$D^{(j_1)} \otimes D^{(j_2)} = \bigoplus_{j=|j_1-j_2|}^{j_1+j_2} D^{(j)}$$

This is the **Clebsch-Gordan decomposition** of representations.

### Dimension Formula

$$\dim(D^{(j_1)}) \times \dim(D^{(j_2)}) = \sum_j \dim(D^{(j)})$$

$$(2j_1+1)(2j_2+1) = \sum_{j=|j_1-j_2|}^{j_1+j_2}(2j+1)$$

### Group Theory Interpretation

- Uncoupled basis: block diagonal under separate rotations
- Coupled basis: block diagonal under total rotation
- Clebsch-Gordan coefficients: unitary transformation between bases

---

## 10. Computational Lab

```python
"""
Day 305: Addition of Angular Momenta
"""

import numpy as np
from scipy.linalg import block_diag
from itertools import product

class AngularMomentumAddition:
    """
    Tools for adding two angular momenta.
    """

    def __init__(self, j1, j2):
        """Initialize with two angular momentum quantum numbers."""
        self.j1 = j1
        self.j2 = j2
        self.dim1 = int(2*j1 + 1)
        self.dim2 = int(2*j2 + 1)
        self.dim_total = self.dim1 * self.dim2

        # Allowed j values
        self.j_values = np.arange(abs(j1 - j2), j1 + j2 + 1)

    def uncoupled_basis(self):
        """Return list of (m1, m2) pairs for uncoupled basis."""
        m1_values = np.arange(self.j1, -self.j1 - 1, -1)
        m2_values = np.arange(self.j2, -self.j2 - 1, -1)
        return [(m1, m2) for m1 in m1_values for m2 in m2_values]

    def coupled_basis(self):
        """Return list of (j, m) pairs for coupled basis."""
        basis = []
        for j in self.j_values:
            for m in np.arange(j, -j - 1, -1):
                basis.append((j, m))
        return basis

    def verify_dimension(self):
        """Verify dimension counting."""
        uncoupled_dim = self.dim_total
        coupled_dim = sum(int(2*j + 1) for j in self.j_values)
        return uncoupled_dim, coupled_dim, uncoupled_dim == coupled_dim


class TwoSpinHalf:
    """
    Special case: two spin-1/2 particles.
    """

    # Basis ordering: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
    # Indices:          0      1      2      3

    def __init__(self):
        """Initialize two-spin system."""
        self.dim = 4

        # Uncoupled basis states
        self.uu = np.array([1, 0, 0, 0], dtype=complex)  # |↑↑⟩
        self.ud = np.array([0, 1, 0, 0], dtype=complex)  # |↑↓⟩
        self.du = np.array([0, 0, 1, 0], dtype=complex)  # |↓↑⟩
        self.dd = np.array([0, 0, 0, 1], dtype=complex)  # |↓↓⟩

        # Coupled basis states
        self.triplet_p1 = self.uu  # |1, 1⟩
        self.triplet_0 = (self.ud + self.du) / np.sqrt(2)  # |1, 0⟩
        self.triplet_m1 = self.dd  # |1, -1⟩
        self.singlet = (self.ud - self.du) / np.sqrt(2)  # |0, 0⟩

        self._build_operators()

    def _build_operators(self):
        """Build spin operators for two-particle system."""
        # Single-particle Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        I = np.eye(2, dtype=complex)

        # Two-particle operators (tensor products)
        self.S1x = np.kron(sx, I)
        self.S1y = np.kron(sy, I)
        self.S1z = np.kron(sz, I)

        self.S2x = np.kron(I, sx)
        self.S2y = np.kron(I, sy)
        self.S2z = np.kron(I, sz)

        # Total spin operators
        self.Sx = self.S1x + self.S2x
        self.Sy = self.S1y + self.S2y
        self.Sz = self.S1z + self.S2z

        # S^2 = Sx^2 + Sy^2 + Sz^2
        self.S2 = self.Sx @ self.Sx + self.Sy @ self.Sy + self.Sz @ self.Sz

        # S1·S2 = (S^2 - S1^2 - S2^2)/2
        S1_squared = self.S1x @ self.S1x + self.S1y @ self.S1y + self.S1z @ self.S1z
        S2_squared = self.S2x @ self.S2x + self.S2y @ self.S2y + self.S2z @ self.S2z
        self.S1_dot_S2 = (self.S2 - S1_squared - S2_squared) / 2

    def verify_coupled_states(self):
        """Verify that coupled states are eigenstates of S^2 and Sz."""
        print("Verifying coupled basis states:")
        print("-" * 50)

        states = [
            ('|1, 1⟩', self.triplet_p1, 1, 1),
            ('|1, 0⟩', self.triplet_0, 1, 0),
            ('|1,-1⟩', self.triplet_m1, 1, -1),
            ('|0, 0⟩', self.singlet, 0, 0),
        ]

        for name, state, j, m in states:
            # Check S^2 eigenvalue
            S2_state = self.S2 @ state
            S2_eigenvalue = np.real(state.conj() @ S2_state)
            S2_expected = j * (j + 1)

            # Check Sz eigenvalue
            Sz_state = self.Sz @ state
            Sz_eigenvalue = np.real(state.conj() @ Sz_state)
            Sz_expected = m

            print(f"{name}: S² = {S2_eigenvalue:.4f} (expected {S2_expected}), "
                  f"Sz = {Sz_eigenvalue:.4f} (expected {Sz_expected})")

    def exchange_operator(self):
        """
        Compute the exchange operator P12.
        P12|m1, m2⟩ = |m2, m1⟩
        """
        P = np.zeros((4, 4), dtype=complex)
        P[0, 0] = 1  # |↑↑⟩ → |↑↑⟩
        P[1, 2] = 1  # |↑↓⟩ → |↓↑⟩
        P[2, 1] = 1  # |↓↑⟩ → |↑↓⟩
        P[3, 3] = 1  # |↓↓⟩ → |↓↓⟩
        return P

    def verify_symmetry(self):
        """Verify triplet is symmetric, singlet is antisymmetric."""
        print("\nVerifying exchange symmetry:")
        print("-" * 50)

        P = self.exchange_operator()

        states = [
            ('Triplet |1, 1⟩', self.triplet_p1),
            ('Triplet |1, 0⟩', self.triplet_0),
            ('Triplet |1,-1⟩', self.triplet_m1),
            ('Singlet |0, 0⟩', self.singlet),
        ]

        for name, state in states:
            exchanged = P @ state
            if np.allclose(exchanged, state):
                print(f"{name}: symmetric (eigenvalue +1)")
            elif np.allclose(exchanged, -state):
                print(f"{name}: antisymmetric (eigenvalue -1)")
            else:
                print(f"{name}: not an eigenstate of exchange")


def demonstrate_addition():
    """Demonstrate angular momentum addition."""
    print("=" * 60)
    print("ANGULAR MOMENTUM ADDITION")
    print("=" * 60)

    test_cases = [
        (0.5, 0.5),
        (1, 0.5),
        (1, 1),
        (1.5, 1),
        (2, 1.5),
    ]

    for j1, j2 in test_cases:
        am = AngularMomentumAddition(j1, j2)
        uncoupled, coupled, match = am.verify_dimension()

        j_range = f"{abs(j1-j2)} to {j1+j2}"
        print(f"\nj1 = {j1}, j2 = {j2}:")
        print(f"  j values: {am.j_values}")
        print(f"  Dimension: {uncoupled} = {' + '.join(str(int(2*j+1)) for j in am.j_values)} ✓")


def demonstrate_two_spin_half():
    """Demonstrate two spin-1/2 system."""
    print("\n" + "=" * 60)
    print("TWO SPIN-1/2 SYSTEM")
    print("=" * 60)

    system = TwoSpinHalf()
    system.verify_coupled_states()
    system.verify_symmetry()

    # Show S1·S2 eigenvalues
    print("\nS1·S2 eigenvalues:")
    print("-" * 50)

    for name, state, j in [('Triplet', system.triplet_0, 1),
                           ('Singlet', system.singlet, 0)]:
        S1S2 = system.S1_dot_S2
        eigenvalue = np.real(state.conj() @ S1S2 @ state)
        # S1·S2 = (S^2 - S1^2 - S2^2)/2 = (j(j+1) - 3/4 - 3/4)/2
        expected = (j*(j+1) - 3/4 - 3/4) / 2
        print(f"  {name}: ⟨S1·S2⟩ = {eigenvalue:.4f} (expected {expected:.4f})")


def ladder_operator_action():
    """Show how ladder operators connect states."""
    print("\n" + "=" * 60)
    print("LADDER OPERATOR ACTION")
    print("=" * 60)

    system = TwoSpinHalf()

    # J- = S1- + S2-
    S1m = np.array([[0, 0], [1, 0]], dtype=complex)
    S2m = np.array([[0, 0], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    Jm = np.kron(S1m, I) + np.kron(I, S2m)

    print("\nApplying J- to triplet states:")

    # J-|1,1⟩ should give sqrt(2)|1,0⟩
    result = Jm @ system.triplet_p1
    coeff = np.sqrt(1*2 - 1*0)  # sqrt(j(j+1) - m(m-1))
    expected = coeff * system.triplet_0
    print(f"  J-|1,1⟩ = {result}")
    print(f"  Expected √2|1,0⟩ = {expected.round(4)}")
    print(f"  Match: {np.allclose(result, expected)}")


def spin_orbit_coupling_example():
    """Demonstrate spin-orbit coupling for p-electron."""
    print("\n" + "=" * 60)
    print("SPIN-ORBIT COUPLING (l=1, s=1/2)")
    print("=" * 60)

    # For l=1, s=1/2, possible j = 1/2, 3/2
    l, s = 1, 0.5

    print("\nPossible total angular momentum values:")
    for j in [abs(l-s) + i for i in range(int(2*min(l,s)+1))]:
        multiplicity = int(2*j + 1)
        # L·S = (J^2 - L^2 - S^2)/2 = [j(j+1) - l(l+1) - s(s+1)]/2
        LS_eigenvalue = (j*(j+1) - l*(l+1) - s*(s+1)) / 2
        print(f"  j = {j}: {multiplicity} states, ⟨L·S⟩/ℏ² = {LS_eigenvalue:.4f}")


def entanglement_demonstration():
    """Show that singlet state is maximally entangled."""
    print("\n" + "=" * 60)
    print("ENTANGLEMENT IN SINGLET STATE")
    print("=" * 60)

    system = TwoSpinHalf()

    # Compute reduced density matrix for particle 1
    # ρ1 = Tr_2(|ψ⟩⟨ψ|)

    def reduced_density_matrix(state):
        """Compute reduced density matrix for first particle."""
        # Reshape 4-component vector to 2x2 matrix
        psi = state.reshape(2, 2)
        # ρ1 = Σ_j ⟨j|ψ⟩⟨ψ|j⟩ where |j⟩ are basis states of particle 2
        rho1 = psi @ psi.conj().T
        return rho1

    # Von Neumann entropy S = -Tr(ρ log ρ)
    def von_neumann_entropy(rho):
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    states = [
        ('Product |↑↑⟩', system.uu),
        ('Triplet |1,0⟩', system.triplet_0),
        ('Singlet |0,0⟩', system.singlet),
    ]

    for name, state in states:
        rho = reduced_density_matrix(state)
        S = von_neumann_entropy(rho)
        print(f"\n{name}:")
        print(f"  ρ1 = \n{rho.round(4)}")
        print(f"  Entanglement entropy: S = {S:.4f} bits")


# Main execution
if __name__ == "__main__":
    demonstrate_addition()
    demonstrate_two_spin_half()
    ladder_operator_action()
    spin_orbit_coupling_example()
    entanglement_demonstration()
```

---

## 11. Practice Problems

### Problem 1: Dimension Counting

Verify that $(2 \cdot 2 + 1)(2 \cdot 1 + 1) = \sum_{j=1}^{3}(2j+1)$ for $j_1 = 2, j_2 = 1$.

### Problem 2: Three Spin-1/2

What values of total $j$ are possible when coupling three spin-1/2 particles? Find the multiplicity of each.

### Problem 3: Singlet Projection

Show that the projection operator onto the singlet state is:
$$\hat{P}_{singlet} = \frac{1}{4}(I - \vec{\sigma}_1 \cdot \vec{\sigma}_2)$$

### Problem 4: Exchange Operator

Prove that the exchange operator $\hat{P}_{12}$ can be written as:
$$\hat{P}_{12} = \frac{1}{2}(I + \vec{\sigma}_1 \cdot \vec{\sigma}_2)$$

### Problem 5: Spin-Orbit Splitting

For a p-electron ($\ell = 1$) with spin-orbit Hamiltonian $H_{SO} = A\mathbf{L}\cdot\mathbf{S}$, calculate the energy splitting between $j = 3/2$ and $j = 1/2$ states.

---

## Summary

### Angular Momentum Addition Rule

$$\boxed{j_1 \otimes j_2 = |j_1 - j_2| \oplus |j_1 - j_2| + 1 \oplus \cdots \oplus (j_1 + j_2)}$$

### Two Spin-1/2: Singlet and Triplet

| State | $j$ | $m$ | Spin Part | Exchange |
|-------|-----|-----|-----------|----------|
| Triplet | 1 | +1 | $\|\uparrow\uparrow\rangle$ | Symmetric |
| Triplet | 1 | 0 | $(\|\uparrow\downarrow\rangle + \|\downarrow\uparrow\rangle)/\sqrt{2}$ | Symmetric |
| Triplet | 1 | -1 | $\|\downarrow\downarrow\rangle$ | Symmetric |
| Singlet | 0 | 0 | $(\|\uparrow\downarrow\rangle - \|\downarrow\uparrow\rangle)/\sqrt{2}$ | Antisymmetric |

### Key Relations

$$\mathbf{S}_1 \cdot \mathbf{S}_2 = \frac{1}{2}(\mathbf{S}^2 - \mathbf{S}_1^2 - \mathbf{S}_2^2)$$

---

## Preview: Day 306

Tomorrow we dive deep into **Clebsch-Gordan coefficients** — the transformation matrices between coupled and uncoupled bases. These are essential for atomic physics calculations.
