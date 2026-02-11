# Day 307: Applications — Selection Rules and the Wigner-Eckart Theorem

## Overview

**Month 11, Week 44, Day 6 — Saturday**

Today we apply the angular momentum formalism to physical problems. The Wigner-Eckart theorem dramatically simplifies calculations by separating geometric (Clebsch-Gordan) factors from dynamical (reduced matrix element) factors. This leads to powerful selection rules that govern atomic transitions.

## Learning Objectives

1. State and apply the Wigner-Eckart theorem
2. Derive selection rules for electric dipole transitions
3. Calculate transition probabilities using reduced matrix elements
4. Apply to atomic spectra and quantum information
5. Connect group theory to observable physics

---

## 1. Tensor Operators

### Definition

A **tensor operator** $T^{(k)}_q$ of rank $k$ transforms under rotations like spherical harmonics $Y_k^q$:

$$[J_z, T^{(k)}_q] = \hbar q \, T^{(k)}_q$$
$$[J_\pm, T^{(k)}_q] = \hbar\sqrt{k(k+1) - q(q \pm 1)} \, T^{(k)}_{q \pm 1}$$

### Examples

**Rank 0 (Scalar):** $T^{(0)}_0$
- Example: $H$, $\mathbf{J}^2$, $\mathbf{r} \cdot \mathbf{p}$

**Rank 1 (Vector):** $T^{(1)}_q$
- Components: $T^{(1)}_{+1} = -\frac{1}{\sqrt{2}}(T_x + iT_y)$, $T^{(1)}_0 = T_z$, $T^{(1)}_{-1} = \frac{1}{\sqrt{2}}(T_x - iT_y)$
- Examples: $\mathbf{r}$, $\mathbf{p}$, $\mathbf{L}$, $\mathbf{S}$, electric dipole operator

**Rank 2 (Tensor):** $T^{(2)}_q$
- Examples: Quadrupole moments, stress tensor

---

## 2. The Wigner-Eckart Theorem

### Statement

$$\boxed{\langle j', m' | T^{(k)}_q | j, m \rangle = \langle j' \| T^{(k)} \| j \rangle \, C^{j'm'}_{jm; kq}}$$

where:
- $\langle j' \| T^{(k)} \| j \rangle$ is the **reduced matrix element** (independent of $m, m', q$)
- $C^{j'm'}_{jm; kq}$ is the Clebsch-Gordan coefficient

### Significance

The theorem **factorizes** matrix elements:
1. **Geometric factor:** Clebsch-Gordan coefficient (universal, tabulated)
2. **Dynamical factor:** Reduced matrix element (depends on specific operator and states)

### Selection Rules from CG Coefficients

$$C^{j'm'}_{jm;kq} \neq 0 \quad \text{requires:}$$
- $m' = m + q$ (z-component conservation)
- $|j - k| \leq j' \leq j + k$ (triangle rule)

---

## 3. Electric Dipole Transitions

### The Electric Dipole Operator

$$\hat{\mathbf{d}} = e\hat{\mathbf{r}}$$

In spherical tensor form:
$$d^{(1)}_0 = ez, \quad d^{(1)}_{\pm 1} = \mp\frac{e}{\sqrt{2}}(x \pm iy)$$

### Selection Rules

For electric dipole transitions $|n', \ell', m'\rangle \to |n, \ell, m\rangle$:

$$\boxed{\Delta \ell = \pm 1, \quad \Delta m = 0, \pm 1}$$

Also: $\Delta m = 0$ for $\pi$-polarization (linear along z)
$\Delta m = \pm 1$ for $\sigma^\pm$-polarization (circular)

### Derivation

The position operator $\mathbf{r}$ is a rank-1 tensor.

Matrix element $\langle \ell', m' | r^{(1)}_q | \ell, m \rangle$ requires:
- $|\ell - 1| \leq \ell' \leq \ell + 1$ (triangle)
- $m' = m + q$

Since $\ell, \ell'$ are non-negative integers and $\ell' = \ell$ is forbidden by parity, we get $\Delta \ell = \pm 1$.

### Parity Selection Rule

Under parity: $Y_\ell^m \to (-1)^\ell Y_\ell^m$

Electric dipole operator: $\mathbf{r} \to -\mathbf{r}$ (odd parity)

For nonzero matrix element: $(-1)^{\ell'} \times (-1) \times (-1)^\ell = 1$

Thus $\ell' + \ell = \text{odd}$, meaning $\Delta \ell = \text{odd}$.

Combined with triangle rule: $\Delta \ell = \pm 1$.

---

## 4. Transition Probabilities

### Fermi's Golden Rule

Transition rate:
$$W_{i \to f} = \frac{2\pi}{\hbar}|\langle f | H' | i \rangle|^2 \rho(E_f)$$

For electric dipole transitions in an electromagnetic field:
$$W \propto |\langle f | \hat{\mathbf{d}} \cdot \hat{\epsilon} | i \rangle|^2$$

where $\hat{\epsilon}$ is the polarization direction.

### Using Wigner-Eckart

$$|\langle n', \ell', m' | d^{(1)}_q | n, \ell, m \rangle|^2 = |\langle n', \ell' \| d^{(1)} \| n, \ell \rangle|^2 |C^{\ell' m'}_{\ell m; 1 q}|^2$$

### Sum Rules

Summing over $m'$ and $q$:
$$\sum_{m', q} |C^{\ell' m'}_{\ell m; 1 q}|^2 = 1$$

The total transition rate depends only on the reduced matrix element:
$$W \propto |\langle n', \ell' \| d^{(1)} \| n, \ell \rangle|^2$$

---

## 5. Magnetic Dipole and Electric Quadrupole

### Magnetic Dipole Transitions

Operator: $\boldsymbol{\mu} = -\mu_B(\mathbf{L} + g_s\mathbf{S})/\hbar$

Selection rules:
$$\Delta \ell = 0, \quad \Delta m = 0, \pm 1$$
(Parity conserved, so same $\ell$)

### Electric Quadrupole Transitions

Operator: $Q_{ij} = er_i r_j$

Rank-2 tensor, selection rules:
$$\Delta \ell = 0, \pm 2, \quad \Delta m = 0, \pm 1, \pm 2$$

### Comparison of Transition Strengths

| Type | Operator Rank | Selection Rules | Relative Strength |
|------|---------------|-----------------|-------------------|
| E1 (dipole) | 1 | $\Delta\ell = \pm 1$ | 1 |
| M1 (magnetic dipole) | 1 | $\Delta\ell = 0$ | $\sim 10^{-5}$ |
| E2 (quadrupole) | 2 | $\Delta\ell = 0, \pm 2$ | $\sim 10^{-7}$ |

---

## 6. Atomic Fine Structure

### Spin-Orbit Coupling

$$\hat{H}_{SO} = A(r)\hat{\mathbf{L}} \cdot \hat{\mathbf{S}}$$

Using $\mathbf{J} = \mathbf{L} + \mathbf{S}$:
$$\hat{\mathbf{L}} \cdot \hat{\mathbf{S}} = \frac{1}{2}(\hat{\mathbf{J}}^2 - \hat{\mathbf{L}}^2 - \hat{\mathbf{S}}^2)$$

### Energy Levels

$$E_{SO} = \frac{A}{2}[j(j+1) - \ell(\ell+1) - s(s+1)]$$

For $s = 1/2$:
- $j = \ell + 1/2$: $E = A\ell/2$
- $j = \ell - 1/2$: $E = -A(\ell+1)/2$

**Splitting:** $\Delta E = A(2\ell + 1)/2$

### Fine Structure Selection Rules

Including spin:
$$\Delta j = 0, \pm 1 \quad (j = 0 \nrightarrow j' = 0)$$
$$\Delta \ell = \pm 1$$
$$\Delta m_j = 0, \pm 1$$

---

## 7. Quantum Computing Applications

### Qubit Operations as Rotations

A qubit state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ on the Bloch sphere.

Single-qubit gates = SU(2) rotations = angular momentum operations.

### Selection Rules in Quantum Gates

**Resonant driving** of a two-level system:
$$H_{int} = \hbar\Omega(\sigma_+ e^{-i\omega t} + \sigma_- e^{i\omega t})$$

This is a rank-1 tensor interaction, connecting $|0\rangle \leftrightarrow |1\rangle$ (like dipole transitions).

### Forbidden Transitions in Quantum Error Correction

Selection rules protect certain states:
- States with different total angular momentum don't mix under symmetric noise
- Decoherence-free subspaces exploit this

---

## 8. Computational Lab

```python
"""
Day 307: Selection Rules and Wigner-Eckart Applications
"""

import numpy as np
from scipy.special import factorial
from itertools import product
import matplotlib.pyplot as plt

def clebsch_gordan(j1, m1, j2, m2, j, m):
    """Compute CG coefficient."""
    if m != m1 + m2:
        return 0.0
    if not (abs(j1 - j2) <= j <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0

    def delta(j1, j2, j):
        num = factorial(j1+j2-j) * factorial(j1-j2+j) * factorial(-j1+j2+j)
        den = factorial(j1+j2+j+1)
        return np.sqrt(num / den)

    prefactor = np.sqrt(2*j + 1) * delta(j1, j2, j)
    prefactor *= np.sqrt(
        factorial(j1+m1) * factorial(j1-m1) *
        factorial(j2+m2) * factorial(j2-m2) *
        factorial(j+m) * factorial(j-m)
    )

    total = 0.0
    for k in range(100):
        args = [k, j1+j2-j-k, j1-m1-k, j2+m2-k, j-j2+m1+k, j-j1-m2+k]
        if all(a >= 0 and a == int(a) for a in args):
            denom = np.prod([factorial(int(a)) for a in args])
            total += (-1)**k / denom

    return prefactor * total


class SelectionRules:
    """Tools for analyzing selection rules."""

    @staticmethod
    def check_electric_dipole(ell_i, m_i, ell_f, m_f):
        """
        Check if electric dipole transition is allowed.
        Returns: (allowed, polarization)
        """
        delta_ell = ell_f - ell_i
        delta_m = m_f - m_i

        if abs(delta_ell) != 1:
            return False, None

        if delta_m == 0:
            return True, 'π (linear z)'
        elif delta_m == 1:
            return True, 'σ+ (left circular)'
        elif delta_m == -1:
            return True, 'σ- (right circular)'
        else:
            return False, None

    @staticmethod
    def transition_table(ell_max):
        """Generate table of allowed E1 transitions."""
        print("Electric Dipole (E1) Allowed Transitions")
        print("=" * 50)

        for ell_i in range(ell_max + 1):
            for ell_f in range(ell_max + 1):
                if abs(ell_f - ell_i) == 1:
                    print(f"\nℓ = {ell_i} → ℓ' = {ell_f}:")
                    for m_i in range(-ell_i, ell_i + 1):
                        for delta_m in [-1, 0, 1]:
                            m_f = m_i + delta_m
                            if abs(m_f) <= ell_f:
                                pol = {-1: 'σ-', 0: 'π', 1: 'σ+'}[delta_m]
                                # Get CG coefficient
                                cg = clebsch_gordan(ell_i, m_i, 1, delta_m, ell_f, m_f)
                                print(f"  m={m_i:+d} → m'={m_f:+d} ({pol}): "
                                      f"|CG|² = {cg**2:.4f}")


class AtomicSpectrum:
    """Calculate atomic transition properties."""

    def __init__(self, n_max=4, ell_max=3):
        self.n_max = n_max
        self.ell_max = ell_max

    def energy_level(self, n, ell, j, Z=1):
        """
        Calculate energy including fine structure.
        Simple hydrogen-like model.
        """
        # Bohr energy (in eV)
        E_n = -13.6 * Z**2 / n**2

        # Fine structure correction (leading order)
        alpha = 1/137  # Fine structure constant
        E_fs = E_n * (alpha * Z)**2 / n * (1/j - 3/(4*n))

        return E_n + E_fs

    def allowed_transitions(self, n_i, ell_i, j_i, n_f, ell_f, j_f):
        """Check if transition is E1 allowed."""
        # Selection rules
        if abs(ell_f - ell_i) != 1:
            return False
        if abs(j_f - j_i) > 1 or (j_i == 0 and j_f == 0):
            return False
        return True

    def generate_spectrum(self):
        """Generate all allowed E1 transitions."""
        transitions = []

        # Generate all states
        states = []
        for n in range(1, self.n_max + 1):
            for ell in range(min(n, self.ell_max + 1)):
                for j in [ell - 0.5, ell + 0.5]:
                    if j >= 0:
                        E = self.energy_level(n, ell, j)
                        states.append((n, ell, j, E))

        # Find all allowed transitions
        for i, (n_i, l_i, j_i, E_i) in enumerate(states):
            for f, (n_f, l_f, j_f, E_f) in enumerate(states):
                if E_f < E_i:  # Emission
                    if self.allowed_transitions(n_i, l_i, j_i, n_f, l_f, j_f):
                        delta_E = E_i - E_f
                        transitions.append({
                            'initial': (n_i, l_i, j_i),
                            'final': (n_f, l_f, j_f),
                            'energy': delta_E,
                            'wavelength': 1240 / delta_E  # nm
                        })

        return transitions


def wigner_eckart_example():
    """Demonstrate Wigner-Eckart theorem application."""
    print("=" * 60)
    print("WIGNER-ECKART THEOREM APPLICATION")
    print("=" * 60)

    # Electric dipole matrix elements between p and s states
    ell_i, ell_f = 1, 0  # p → s transition

    print(f"\nMatrix elements <ℓ'=0, m'| d^(1)_q |ℓ=1, m>:")
    print("-" * 50)

    # The reduced matrix element is the same for all m, m', q
    print("All matrix elements share the same reduced matrix element")
    print("<0 || d || 1>, but differ by CG coefficients:")
    print()

    for m_i in [-1, 0, 1]:
        for q in [-1, 0, 1]:
            m_f = m_i + q
            if abs(m_f) <= ell_f:  # m_f must satisfy |m_f| <= ell_f
                cg = clebsch_gordan(ell_i, m_i, 1, q, ell_f, m_f)
                if abs(cg) > 1e-10:
                    print(f"  <0,{m_f:+d}| d^(1)_{q:+d} |1,{m_i:+d}> = "
                          f"<0||d||1> × {cg:+.4f}")

    print("\nRelative intensities (|CG|²):")
    for m_i in [-1, 0, 1]:
        for q in [-1, 0, 1]:
            m_f = m_i + q
            if abs(m_f) <= ell_f:
                cg = clebsch_gordan(ell_i, m_i, 1, q, ell_f, m_f)
                if abs(cg) > 1e-10:
                    pol = {-1: 'σ-', 0: 'π', 1: 'σ+'}[q]
                    print(f"  |1,{m_i:+d}> → |0,{m_f:+d}> ({pol}): I ∝ {cg**2:.4f}")


def spin_orbit_splitting():
    """Calculate spin-orbit fine structure."""
    print("\n" + "=" * 60)
    print("SPIN-ORBIT FINE STRUCTURE")
    print("=" * 60)

    print("\nFor hydrogen p-state (ℓ=1, s=1/2):")
    print("Possible j values: 1/2, 3/2")
    print()

    ell = 1
    s = 0.5
    A = 1  # Spin-orbit constant (normalized)

    for j in [0.5, 1.5]:
        LS = 0.5 * (j*(j+1) - ell*(ell+1) - s*(s+1))
        E = A * LS
        print(f"  j = {j}: <L·S>/ℏ² = {LS:+.2f}, E = {E:+.2f}A")

    splitting = A * (2*ell + 1) / 2
    print(f"\n  Fine structure splitting: ΔE = {splitting:.2f}A")
    print(f"  (= A(2ℓ+1)/2 = A × {2*ell+1}/2)")


def visualize_term_diagram():
    """Create atomic term diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Energy levels (arbitrary units)
    levels = {
        '1s1/2': (0, -13.6),
        '2s1/2': (1, -3.4),
        '2p1/2': (2, -3.4 + 0.05),
        '2p3/2': (2, -3.4 - 0.05),
        '3s1/2': (3, -1.51),
        '3p1/2': (4, -1.51 + 0.02),
        '3p3/2': (4, -1.51 - 0.02),
        '3d3/2': (5, -1.51 + 0.01),
        '3d5/2': (5, -1.51 - 0.01),
    }

    # Plot levels
    for name, (x, E) in levels.items():
        ax.hlines(E, x - 0.3, x + 0.3, colors='blue', linewidth=2)
        ax.text(x, E + 0.3, name, ha='center', fontsize=10)

    # Plot allowed transitions
    transitions = [
        ('2p1/2', '1s1/2'),
        ('2p3/2', '1s1/2'),
        ('3s1/2', '2p1/2'),
        ('3s1/2', '2p3/2'),
        ('3p1/2', '2s1/2'),
        ('3p3/2', '2s1/2'),
        ('3d3/2', '2p1/2'),
        ('3d3/2', '2p3/2'),
        ('3d5/2', '2p3/2'),
    ]

    for upper, lower in transitions:
        x1, E1 = levels[upper]
        x2, E2 = levels[lower]
        ax.annotate('', xy=(x2, E2 + 0.2), xytext=(x1, E1 - 0.2),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    ax.set_xlim(-1, 7)
    ax.set_ylim(-15, 0)
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Hydrogen-like Atom Term Diagram with E1 Transitions')
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig('term_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: term_diagram.png")


def branching_ratios():
    """Calculate branching ratios for spontaneous emission."""
    print("\n" + "=" * 60)
    print("BRANCHING RATIOS FROM WIGNER-ECKART")
    print("=" * 60)

    print("\nFor 2p → 1s transitions in hydrogen:")
    print("Initial state: |n=2, ℓ=1, m⟩")
    print("Final state: |n=1, ℓ=0, m'=0⟩")
    print()

    # Sum CG coefficients squared for each m
    for m in [-1, 0, 1]:
        total = 0
        print(f"From m = {m:+d}:")
        for q in [-1, 0, 1]:
            m_f = m + q
            if m_f == 0:  # Only m_f = 0 allowed for ℓ = 0
                cg = clebsch_gordan(1, m, 1, q, 0, 0)
                if abs(cg) > 1e-10:
                    pol = {-1: 'σ-', 0: 'π', 1: 'σ+'}[q]
                    print(f"  {pol} polarization: |CG|² = {cg**2:.4f}")
                    total += cg**2
        print(f"  Total: {total:.4f}")
        print()


# Main execution
if __name__ == "__main__":
    SelectionRules.transition_table(2)
    wigner_eckart_example()
    spin_orbit_splitting()
    branching_ratios()
    visualize_term_diagram()
```

---

## 9. Practice Problems

### Problem 1: Selection Rules

Which of the following electric dipole transitions are allowed?
- (a) $3d \to 2p$
- (b) $3s \to 2s$
- (c) $4f \to 3d$
- (d) $2p \to 2s$

### Problem 2: Wigner-Eckart

Using the Wigner-Eckart theorem, show that $\langle j', m' | J_z | j, m \rangle = m\hbar\delta_{jj'}\delta_{mm'}$.

### Problem 3: Reduced Matrix Element

For the position operator $r^{(1)}_0 = z$, show that:
$$\langle \ell' = \ell-1 \| r \| \ell \rangle \propto \sqrt{\ell}$$
$$\langle \ell' = \ell+1 \| r \| \ell \rangle \propto \sqrt{\ell+1}$$

### Problem 4: Polarization

An atom in state $|2p, m=1\rangle$ decays to $|1s\rangle$. What polarization of light is emitted?

### Problem 5: Fine Structure

For the $3p$ states of sodium, calculate the ratio of energies $E_{3p_{3/2}}/E_{3p_{1/2}}$ in terms of the spin-orbit constant.

---

## Summary

### Wigner-Eckart Theorem

$$\boxed{\langle j', m' | T^{(k)}_q | j, m \rangle = \langle j' \| T^{(k)} \| j \rangle \, C^{j'm'}_{jm; kq}}$$

### Electric Dipole Selection Rules

$$\boxed{\Delta \ell = \pm 1, \quad \Delta m = 0, \pm 1, \quad \Delta j = 0, \pm 1}$$

### Key Applications

| Application | Relevant Selection Rules |
|-------------|-------------------------|
| Atomic spectra | $\Delta\ell = \pm 1$ for E1 |
| Laser transitions | Dipole-allowed only |
| NMR/ESR | $\Delta m = \pm 1$ |
| Quantum computing | Resonant driving |

---

## Preview: Day 308

Tomorrow: **Month 11 Capstone Review** — synthesizing all of group theory and its applications to quantum mechanics.
