# Day 933: Majorana Fermions

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Majorana operators, zero modes, and the Kitaev chain |
| Afternoon | 2 hours | Problem solving: Kitaev chain phase diagram |
| Evening | 2 hours | Computational lab: Kitaev chain simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Define Majorana operators** and their algebraic properties
2. **Decompose fermion operators** into Majorana operators
3. **Solve the Kitaev chain** Hamiltonian exactly
4. **Identify topological phases** from the band structure
5. **Locate Majorana zero modes** at chain ends in the topological phase
6. **Explain topological protection** through energy gap and non-locality

---

## Core Content

### 1. What Are Majorana Fermions?

Ettore Majorana proposed in 1937 that there might exist fermions that are their own antiparticles. In particle physics, neutrinos may be Majorana fermions (still unconfirmed). In condensed matter, we don't find fundamental Majorana particles, but we can create **Majorana quasiparticles** - collective excitations that behave mathematically like Majorana fermions.

#### The Defining Property

A Majorana operator $\gamma$ satisfies:

$$\boxed{\gamma = \gamma^\dagger}$$

This is the self-conjugate or "self-antiparticle" property. Contrast with regular fermion operators:
- $c^\dagger$ creates a particle
- $c$ destroys a particle (creates an antiparticle)
- For Majorana: creation = annihilation!

#### Additional Properties

Majorana operators are Hermitian and satisfy:

$$\boxed{\gamma^2 = 1}$$

$$\boxed{\{\gamma_i, \gamma_j\} = 2\delta_{ij}}$$

where $\{A, B\} = AB + BA$ is the anticommutator.

### 2. From Fermions to Majoranas

Any ordinary fermion can be decomposed into two Majorana operators:

$$\boxed{c = \frac{1}{2}(\gamma_1 + i\gamma_2)}$$

$$\boxed{c^\dagger = \frac{1}{2}(\gamma_1 - i\gamma_2)}$$

We can verify:
- $c + c^\dagger = \gamma_1$ (Hermitian - real part)
- $i(c^\dagger - c) = \gamma_2$ (Hermitian - imaginary part)

Equivalently, inverting:
$$\gamma_1 = c + c^\dagger, \quad \gamma_2 = i(c^\dagger - c)$$

#### Physical Interpretation

Think of a complex fermion as having two real components:
- A regular fermion lives at one site with states $|0\rangle$ and $|1\rangle$
- Two Majorana operators at that site encode the same information
- The key insight: **Majoranas can be spatially separated**

When Majoranas are far apart:
- The fermion becomes **non-local**
- Local perturbations cannot flip the fermion state
- This is the origin of topological protection!

### 3. The Kitaev Chain Model

Alexei Kitaev's 1D chain model is the simplest system hosting Majorana zero modes.

#### The Hamiltonian

$$\boxed{H = -\mu \sum_{j=1}^{N} n_j - t \sum_{j=1}^{N-1} (c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j) + \Delta \sum_{j=1}^{N-1} (c_j c_{j+1} + c_{j+1}^\dagger c_j^\dagger)}$$

where:
- $\mu$ = chemical potential
- $t$ = hopping amplitude
- $\Delta$ = superconducting pairing (p-wave)
- $n_j = c_j^\dagger c_j$ = number operator
- $N$ = number of sites

The key ingredient is **p-wave pairing** $\Delta$, which pairs electrons at adjacent sites.

#### Rewriting in Majorana Operators

Define Majorana operators at each site:
$$c_j = \frac{1}{2}(\gamma_{2j-1} + i\gamma_{2j})$$
$$c_j^\dagger = \frac{1}{2}(\gamma_{2j-1} - i\gamma_{2j})$$

The Hamiltonian becomes:
$$H = -\frac{\mu}{2}\sum_{j=1}^{N}(1 + i\gamma_{2j-1}\gamma_{2j}) + \frac{i}{4}\sum_{j=1}^{N-1}[(-t+\Delta)\gamma_{2j}\gamma_{2j+1} + (t+\Delta)\gamma_{2j-1}\gamma_{2j+2}]$$

### 4. Two Limiting Cases

#### Case 1: Trivial Phase ($\mu < 0$, $t = \Delta = 0$)

$$H = -\mu \sum_j n_j = -\frac{\mu}{2}\sum_j (1 + i\gamma_{2j-1}\gamma_{2j})$$

The Majoranas pair **on-site**:
```
Site 1      Site 2      Site 3
(γ₁-γ₂)    (γ₃-γ₄)    (γ₅-γ₆)
  ↑___↑      ↑___↑      ↑___↑
```

Each site has a local fermion. No zero modes. Ground state is unique.

#### Case 2: Topological Phase ($\mu = 0$, $t = \Delta \neq 0$)

$$H = -it\sum_{j=1}^{N-1} \gamma_{2j}\gamma_{2j+1}$$

The Majoranas pair **between sites**:
```
Site 1      Site 2      Site 3
 γ₁  γ₂    γ₃  γ₄    γ₅  γ₆
 ↑   ↑_____↑   ↑_____↑   ↑
 |               unpaired!
unpaired!
```

**Critical observation**: $\gamma_1$ and $\gamma_{2N}$ are unpaired! These are the **Majorana zero modes**.

### 5. Majorana Zero Modes

In the topological phase, the edge Majoranas satisfy:
$$[H, \gamma_1] = 0, \quad [H, \gamma_{2N}] = 0$$

They are zero-energy modes because they commute with the Hamiltonian.

#### Non-Local Fermion

The unpaired Majoranas form a non-local fermion:
$$\tilde{c} = \frac{1}{2}(\gamma_1 + i\gamma_{2N})$$

This fermion is delocalized across the entire chain!

#### Ground State Degeneracy

The non-local fermion can be empty or occupied:
- $|\tilde{0}\rangle$: fermion empty
- $|\tilde{1}\rangle = \tilde{c}^\dagger|\tilde{0}\rangle$: fermion occupied

Both states have the same energy. The ground state is **two-fold degenerate**.

$$\boxed{\text{Topological phase: 2-fold degenerate ground state}}$$

This encodes one qubit!

### 6. Phase Diagram

The phase boundary is determined by the gap closing condition.

#### Bulk Spectrum

For periodic boundary conditions (to find bulk properties), Fourier transform:
$$c_j = \frac{1}{\sqrt{N}}\sum_k e^{ikj} c_k$$

The Hamiltonian in momentum space:
$$H = \sum_k \begin{pmatrix} c_k^\dagger & c_{-k} \end{pmatrix} \begin{pmatrix} -\mu - 2t\cos k & 2i\Delta\sin k \\ -2i\Delta\sin k & \mu + 2t\cos k \end{pmatrix} \begin{pmatrix} c_k \\ c_{-k}^\dagger \end{pmatrix}$$

The energy spectrum:
$$\boxed{E_k = \pm\sqrt{(\mu + 2t\cos k)^2 + 4\Delta^2\sin^2 k}}$$

#### Gap Closing

The gap closes when $E_k = 0$ for some $k$:
- At $k = 0$: requires $|\mu| = 2t$
- At $k = \pi$: requires $|\mu| = -2t$ (not physical for $t > 0$)

Phase diagram for $\Delta \neq 0$:

$$\boxed{|\mu| < 2t: \text{Topological phase (MZMs present)}}$$
$$\boxed{|\mu| > 2t: \text{Trivial phase (no MZMs)}}$$

### 7. Topological Invariant

The phases are distinguished by a **topological invariant** - the Majorana number or $\mathbb{Z}_2$ invariant.

$$\mathcal{M} = \text{sgn}[\text{Pf}(A(0)) \cdot \text{Pf}(A(\pi))]$$

where $\text{Pf}$ is the Pfaffian and $A(k)$ is the Hamiltonian in Majorana basis.

Simpler criterion using the winding number:
$$\nu = \frac{1}{2\pi i}\oint \frac{dz}{z}, \quad z = d_x(k) + id_y(k)$$

where $\vec{d}(k)$ parametrizes the Hamiltonian.

$$\boxed{\nu = 1: \text{Topological}, \quad \nu = 0: \text{Trivial}}$$

### 8. Wavefunctions of Zero Modes

In the topological phase away from the sweet spot, the Majorana zero modes have finite extent.

For $|\mu| < 2t$ and $\Delta = t$:
$$\gamma_L \sim \sum_j e^{-j/\xi} \gamma_{2j-1}$$
$$\gamma_R \sim \sum_j e^{-(N-j)/\xi} \gamma_{2j}$$

The localization length:
$$\boxed{\xi = \frac{1}{\ln|2t/\mu|}}$$

At the phase transition ($|\mu| = 2t$), $\xi \to \infty$ - the zero modes delocalize into bulk.

---

## Quantum Computing Applications

### Encoding a Qubit

With two Majorana zero modes $\gamma_L$ and $\gamma_R$:
- Define parity: $P = i\gamma_L\gamma_R = \pm 1$
- Two states: $|0\rangle$ ($P = +1$) and $|1\rangle$ ($P = -1$)

$$\boxed{|0\rangle_\text{topo}: i\gamma_L\gamma_R = +1}$$
$$\boxed{|1\rangle_\text{topo}: i\gamma_L\gamma_R = -1}$$

### Protection Mechanism

Local perturbations $V$ can't flip the qubit:
- Need to act on both $\gamma_L$ and $\gamma_R$ simultaneously
- Matrix element: $\langle 0|V|1\rangle \sim e^{-L/\xi}$
- Exponentially suppressed for large separation $L$!

### Operations

**Z gate**: Measure parity $i\gamma_L\gamma_R$
**X gate**: Requires braiding or non-topological operation
**Initialization**: Fuse Majoranas, measure outcome

### Multiple Qubits

For $2n$ Majoranas, ground state degeneracy is $2^{n-1}$:
- One constraint: total parity is conserved
- Encodes $n-1$ logical qubits

---

## Worked Examples

### Example 1: Majorana Algebra

**Problem**: Verify that $c = (\gamma_1 + i\gamma_2)/2$ satisfies fermionic anticommutation relations $\{c, c^\dagger\} = 1$ and $\{c, c\} = 0$.

**Solution**:

First, compute $\{c, c^\dagger\}$:
$$\{c, c^\dagger\} = cc^\dagger + c^\dagger c$$

$$c \cdot c^\dagger = \frac{1}{4}(\gamma_1 + i\gamma_2)(\gamma_1 - i\gamma_2) = \frac{1}{4}(\gamma_1^2 + i\gamma_2\gamma_1 - i\gamma_1\gamma_2 + \gamma_2^2)$$

$$= \frac{1}{4}(1 + 1 + i\gamma_2\gamma_1 - i\gamma_1\gamma_2) = \frac{1}{2} + \frac{i}{2}\gamma_2\gamma_1$$

Similarly:
$$c^\dagger \cdot c = \frac{1}{2} - \frac{i}{2}\gamma_2\gamma_1$$

Therefore:
$$\{c, c^\dagger\} = 1 \checkmark$$

Now compute $\{c, c\}$:
$$c \cdot c = \frac{1}{4}(\gamma_1 + i\gamma_2)^2 = \frac{1}{4}(\gamma_1^2 + 2i\gamma_1\gamma_2 - \gamma_2^2)$$
$$= \frac{1}{4}(1 + 2i\gamma_1\gamma_2 - 1) = \frac{i}{2}\gamma_1\gamma_2$$

$$\{c, c\} = 2c^2 = i\gamma_1\gamma_2$$

Wait, this should be zero! Let me recalculate:
$$c \cdot c = \frac{1}{4}(\gamma_1 + i\gamma_2)(\gamma_1 + i\gamma_2) = \frac{1}{4}(\gamma_1^2 + i\gamma_1\gamma_2 + i\gamma_2\gamma_1 + i^2\gamma_2^2)$$
$$= \frac{1}{4}(1 + i\gamma_1\gamma_2 + i\gamma_2\gamma_1 - 1) = \frac{i}{4}(\gamma_1\gamma_2 + \gamma_2\gamma_1) = \frac{i}{4}\{\gamma_1, \gamma_2\} = 0 \checkmark$$

$$\boxed{\{c, c^\dagger\} = 1, \quad \{c, c\} = 0}$$

### Example 2: Kitaev Chain Sweet Spot

**Problem**: For the Kitaev chain with $\mu = 0$ and $t = \Delta$, explicitly show that $\gamma_1$ and $\gamma_{2N}$ are zero modes.

**Solution**:

At $\mu = 0$, $t = \Delta$, the Hamiltonian is:
$$H = -it\sum_{j=1}^{N-1}\gamma_{2j}\gamma_{2j+1}$$

Compute the commutator $[H, \gamma_1]$:
$$[H, \gamma_1] = -it\sum_{j=1}^{N-1}[\gamma_{2j}\gamma_{2j+1}, \gamma_1]$$

Using $[\gamma_i\gamma_j, \gamma_k] = \gamma_i\{\gamma_j, \gamma_k\} - \{\gamma_i, \gamma_k\}\gamma_j$:

For $j \geq 1$: $\gamma_{2j}$ and $\gamma_{2j+1}$ have indices $\geq 2$, so:
$$\{\gamma_{2j}, \gamma_1\} = 0, \quad \{\gamma_{2j+1}, \gamma_1\} = 0$$

Therefore:
$$[\gamma_{2j}\gamma_{2j+1}, \gamma_1] = 0 \text{ for all } j$$

$$\boxed{[H, \gamma_1] = 0}$$

Similarly, $[H, \gamma_{2N}] = 0$ because the sum only involves $\gamma_2, \gamma_3, ..., \gamma_{2N-1}$.

$$\boxed{\gamma_1 \text{ and } \gamma_{2N} \text{ are exact zero modes}}$$

### Example 3: Phase Boundary

**Problem**: Find the gap at $k = 0$ and $k = \pi$ and determine when it closes.

**Solution**:

From the dispersion:
$$E_k = \sqrt{(\mu + 2t\cos k)^2 + 4\Delta^2\sin^2 k}$$

At $k = 0$:
$$E_0 = \sqrt{(\mu + 2t)^2 + 0} = |\mu + 2t|$$

Gap closes when $\mu = -2t$.

At $k = \pi$:
$$E_\pi = \sqrt{(\mu - 2t)^2 + 0} = |\mu - 2t|$$

Gap closes when $\mu = 2t$.

For $t > 0$ and $\Delta \neq 0$:
- Topological phase: $-2t < \mu < 2t$ (equivalently $|\mu| < 2|t|$)
- Trivial phase: $|\mu| > 2|t|$

$$\boxed{\text{Phase transition at } |\mu| = 2|t|}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Majorana Counting**: A system has 6 Majorana zero modes. How many qubits can it encode? What is the ground state degeneracy?

2. **Operator Check**: Show that if $\gamma = \gamma^\dagger$, then $\gamma^2 = 1$ implies $\gamma$ has eigenvalues $\pm 1$.

3. **Fermion Parity**: For a single fermion mode $c$, express the parity operator $P = (-1)^{n} = (-1)^{c^\dagger c}$ in terms of Majorana operators.

### Level 2: Intermediate

4. **Off-Diagonal Pairing**: In the Kitaev chain, show that the pairing term $\Delta(c_j c_{j+1} + h.c.)$ breaks U(1) particle number conservation but preserves $\mathbb{Z}_2$ parity.

5. **Localization Length**: For $\mu = t$ and $\Delta = t$, calculate the localization length $\xi$ of the Majorana zero modes.

6. **Three-Site Chain**: For a 3-site Kitaev chain at the sweet spot ($\mu = 0$, $t = \Delta$), find all energy eigenvalues.

### Level 3: Challenging

7. **Winding Number**: For the Kitaev chain, the Hamiltonian can be written as $H(k) = \vec{d}(k) \cdot \vec{\sigma}$ where $d_x = 2\Delta\sin k$, $d_y = 0$, $d_z = \mu + 2t\cos k$. Calculate the winding number $\nu$ in both phases.

8. **Finite Size Splitting**: For a finite chain of length $N$ in the topological phase, the two ground states split by an energy $\delta E \sim e^{-N/\xi}$. Derive this using perturbation theory in the overlap of the two Majorana wavefunctions.

---

## Computational Lab: Kitaev Chain Simulation

```python
"""
Day 933 Computational Lab: Majorana Fermions and the Kitaev Chain
Exact diagonalization and visualization of Majorana zero modes
"""

import numpy as np
from scipy.linalg import eigh, block_diag
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Building the Kitaev Chain Hamiltonian
# =============================================================================

def kitaev_chain_hamiltonian(N, mu, t, Delta):
    """
    Construct the Kitaev chain Hamiltonian in the BdG formalism.

    H = -μ Σ n_j - t Σ (c†_j c_{j+1} + h.c.) + Δ Σ (c_j c_{j+1} + h.c.)

    In BdG form with basis (c_1, c_2, ..., c_N, c†_1, c†_2, ..., c†_N):
    H = (1/2) Ψ† H_BdG Ψ

    Parameters:
    -----------
    N : int - Number of sites
    mu : float - Chemical potential
    t : float - Hopping amplitude
    Delta : float - Pairing amplitude

    Returns:
    --------
    H_BdG : ndarray - 2N x 2N BdG Hamiltonian matrix
    """
    # Particle sector (upper left N×N block)
    h = np.zeros((N, N), dtype=complex)

    # On-site terms: -μ
    for j in range(N):
        h[j, j] = -mu

    # Hopping: -t (c†_j c_{j+1} + h.c.)
    for j in range(N - 1):
        h[j, j + 1] = -t
        h[j + 1, j] = -t

    # Pairing sector (upper right N×N block)
    delta_mat = np.zeros((N, N), dtype=complex)

    # Pairing: Δ (c_j c_{j+1} + h.c.) -> Δ in upper right, -Δ* in lower left
    for j in range(N - 1):
        delta_mat[j, j + 1] = Delta
        delta_mat[j + 1, j] = -Delta  # Antisymmetric for fermions

    # Build BdG Hamiltonian
    # H_BdG = [[h, Δ], [-Δ*, -h*]]
    H_BdG = np.block([
        [h, delta_mat],
        [-delta_mat.conj(), -h.conj()]
    ])

    return H_BdG


def solve_kitaev_chain(N, mu, t, Delta):
    """
    Solve the Kitaev chain and return energies and wavefunctions.
    """
    H_BdG = kitaev_chain_hamiltonian(N, mu, t, Delta)
    energies, states = eigh(H_BdG)
    return energies, states


# =============================================================================
# Part 2: Phase Diagram
# =============================================================================

def compute_phase_diagram(N=50, t=1.0, Delta=1.0, n_points=100):
    """
    Compute the energy gap as a function of chemical potential.
    """
    mu_values = np.linspace(-4*t, 4*t, n_points)
    gaps = []
    min_energies = []

    for mu in mu_values:
        energies, _ = solve_kitaev_chain(N, mu, t, Delta)
        # Gap is the smallest positive energy
        positive_energies = energies[energies > 1e-10]
        if len(positive_energies) > 0:
            gap = np.min(positive_energies)
        else:
            gap = 0
        gaps.append(gap)

        # Track lowest energy for zero modes
        min_energies.append(np.min(np.abs(energies)))

    return mu_values, np.array(gaps), np.array(min_energies)


def plot_phase_diagram(t=1.0, Delta=1.0):
    """
    Plot the phase diagram showing topological vs trivial phases.
    """
    print("=" * 50)
    print("Kitaev Chain Phase Diagram")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute for different system sizes
    for N, color in zip([20, 50, 100], ['blue', 'green', 'red']):
        mu_values, gaps, min_E = compute_phase_diagram(N, t, Delta)

        axes[0].plot(mu_values / t, gaps, color=color, label=f'N = {N}')
        axes[1].semilogy(mu_values / t, min_E + 1e-16, color=color, label=f'N = {N}')

    # Mark phase boundaries
    for ax in axes:
        ax.axvline(x=-2, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=2, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    axes[0].set_xlabel('μ/t')
    axes[0].set_ylabel('Energy Gap')
    axes[0].set_title('Bulk Energy Gap')
    axes[0].legend()
    axes[0].set_xlim([-4, 4])

    # Add phase labels
    axes[0].text(-3, axes[0].get_ylim()[1]*0.8, 'Trivial', fontsize=12, ha='center')
    axes[0].text(0, axes[0].get_ylim()[1]*0.8, 'Topological', fontsize=12, ha='center')
    axes[0].text(3, axes[0].get_ylim()[1]*0.8, 'Trivial', fontsize=12, ha='center')

    axes[1].set_xlabel('μ/t')
    axes[1].set_ylabel('Lowest |Energy|')
    axes[1].set_title('Zero Mode Energy (log scale)')
    axes[1].legend()
    axes[1].set_xlim([-4, 4])

    plt.tight_layout()
    plt.savefig('kitaev_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPhase boundaries at μ = ±2t")
    print("Topological phase: |μ| < 2t (zero modes present)")
    print("Trivial phase: |μ| > 2t (no zero modes)")


# =============================================================================
# Part 3: Majorana Zero Mode Wavefunctions
# =============================================================================

def extract_majorana_wavefunction(states, N):
    """
    Extract the Majorana zero mode wavefunctions from BdG states.

    For zero modes, the BdG eigenvector has special structure.
    """
    # Zero modes are at E ≈ 0
    # In BdG, particle-hole symmetry means states come in ±E pairs
    # We look at the states closest to E = 0

    # The BdG eigenvector is (u, v) where u is particle, v is hole component
    # Majorana mode: u = v* (self-conjugate)

    zero_mode_idx = N - 1  # Index of state closest to E = 0 (middle of spectrum)

    u = states[:N, zero_mode_idx]  # Particle component
    v = states[N:, zero_mode_idx]  # Hole component

    # Majorana mode γ = c + c† has wavefunction |u + v*|
    # γ' = i(c† - c) has wavefunction |u - v*|

    gamma_L = np.abs(u + v.conj())
    gamma_R = np.abs(u - v.conj())

    return gamma_L, gamma_R


def plot_zero_modes(N=50, t=1.0, Delta=1.0):
    """
    Plot Majorana zero mode wavefunctions in different phases.
    """
    print("\n" + "=" * 50)
    print("Majorana Zero Mode Wavefunctions")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    mu_values = [-3*t, 0, 1.5*t]  # Trivial, topological sweet spot, topological
    titles = ['Trivial (μ = -3t)', 'Topological Sweet Spot (μ = 0)',
              'Topological (μ = 1.5t)']

    sites = np.arange(1, N + 1)

    for idx, (mu, title) in enumerate(zip(mu_values, titles)):
        energies, states = solve_kitaev_chain(N, mu, t, Delta)

        # Energy spectrum
        axes[0, idx].bar(range(len(energies)), energies, color='steelblue', width=0.8)
        axes[0, idx].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, idx].set_xlabel('State index')
        axes[0, idx].set_ylabel('Energy')
        axes[0, idx].set_title(f'{title}\nSpectrum')
        axes[0, idx].set_ylim([-3, 3])

        # Extract zero mode wavefunctions
        # Find states closest to E = 0
        sorted_idx = np.argsort(np.abs(energies))
        zero_idx = sorted_idx[:2]  # Two zero modes

        for i, zi in enumerate(zero_idx):
            u = states[:N, zi]
            v = states[N:, zi]
            psi = np.abs(u)**2 + np.abs(v)**2
            psi = psi / np.max(psi)  # Normalize for visualization

            axes[1, idx].plot(sites, psi, 'o-', label=f'Mode {i+1}, E={energies[zi]:.4f}',
                            markersize=3)

        axes[1, idx].set_xlabel('Site j')
        axes[1, idx].set_ylabel('|ψ|² (normalized)')
        axes[1, idx].set_title('Lowest energy mode(s)')
        axes[1, idx].legend()
        axes[1, idx].set_yscale('log')
        axes[1, idx].set_ylim([1e-4, 2])

    plt.tight_layout()
    plt.savefig('majorana_wavefunctions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nKey observations:")
    print("- Trivial phase: No edge-localized states")
    print("- Topological phase: Two modes localized at opposite edges")
    print("- Sweet spot (μ=0): Perfect edge localization")


# =============================================================================
# Part 4: Localization Length
# =============================================================================

def compute_localization_length(N=100, t=1.0, Delta=1.0):
    """
    Compute how the localization length depends on μ.
    """
    print("\n" + "=" * 50)
    print("Majorana Localization Length")
    print("=" * 50)

    mu_values = np.linspace(0.1*t, 1.9*t, 20)
    xi_numerical = []
    xi_theory = []

    for mu in mu_values:
        energies, states = solve_kitaev_chain(N, mu, t, Delta)

        # Find zero mode
        sorted_idx = np.argsort(np.abs(energies))
        zero_idx = sorted_idx[0]

        u = states[:N, zero_idx]
        v = states[N:, zero_idx]
        psi = np.abs(u)**2 + np.abs(v)**2

        # Fit exponential decay from left edge
        # ψ(j) ~ exp(-j/ξ)
        log_psi = np.log(psi[:N//3] + 1e-16)
        sites = np.arange(1, N//3 + 1)

        # Linear fit to get decay rate
        coeffs = np.polyfit(sites, log_psi, 1)
        xi_num = -1 / coeffs[0] if coeffs[0] < 0 else np.inf
        xi_numerical.append(xi_num)

        # Theoretical prediction (simplified)
        # ξ = 1 / ln(2t/|μ|) for μ < 2t and Δ = t
        if np.abs(mu) < 2*t:
            xi_th = 1 / np.log(2*t / np.abs(mu)) if mu != 0 else np.inf
        else:
            xi_th = 0
        xi_theory.append(xi_th)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_values/t, xi_numerical, 'o-', label='Numerical', markersize=6)
    ax.plot(mu_values/t, xi_theory, 's--', label='Theory: ξ = 1/ln(2t/μ)', markersize=4)
    ax.set_xlabel('μ/t')
    ax.set_ylabel('Localization length ξ (sites)')
    ax.set_title('Majorana Zero Mode Localization')
    ax.legend()
    ax.set_ylim([0, 20])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('localization_length.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 5: Majorana Parity Operator
# =============================================================================

def demonstrate_parity():
    """
    Show how the ground state degeneracy splits in finite systems.
    """
    print("\n" + "=" * 50)
    print("Ground State Splitting vs System Size")
    print("=" * 50)

    t, Delta = 1.0, 1.0
    mu = 0.5 * t  # In topological phase

    N_values = np.arange(10, 101, 5)
    splittings = []

    for N in N_values:
        energies, _ = solve_kitaev_chain(N, mu, t, Delta)
        # Splitting is twice the lowest positive energy
        positive_E = np.sort(np.abs(energies))
        splitting = 2 * positive_E[0]
        splittings.append(splitting)

    # Fit to exponential
    log_split = np.log(np.array(splittings) + 1e-16)
    coeffs = np.polyfit(N_values, log_split, 1)
    xi_eff = -1 / coeffs[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(N_values, splittings, 'o-', markersize=6, label='Numerical')
    ax.semilogy(N_values, np.exp(coeffs[1]) * np.exp(coeffs[0] * N_values),
                '--', label=f'Fit: exp(-N/{xi_eff:.1f})')
    ax.set_xlabel('Chain length N')
    ax.set_ylabel('Energy splitting δE')
    ax.set_title(f'Ground State Splitting (μ = 0.5t, topological phase)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('energy_splitting.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSplitting decreases exponentially: δE ~ exp(-N/ξ)")
    print(f"Effective localization length ξ ≈ {xi_eff:.1f} sites")
    print(f"For N = 100: δE ≈ {splittings[-1]:.2e}")


# =============================================================================
# Part 6: Main Execution
# =============================================================================

def main():
    """Run all demonstrations."""
    print("╔" + "=" * 58 + "╗")
    print("║  Day 933: Majorana Fermions and the Kitaev Chain          ║")
    print("╚" + "=" * 58 + "╝")

    # 1. Phase diagram
    plot_phase_diagram()

    # 2. Zero mode wavefunctions
    plot_zero_modes()

    # 3. Localization length
    compute_localization_length()

    # 4. Finite-size splitting
    demonstrate_parity()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Results:
1. Majorana operators: γ = γ†, γ² = 1, {γᵢ, γⱼ} = 2δᵢⱼ
2. One complex fermion = two Majorana operators
3. Kitaev chain: topological for |μ| < 2t
4. Sweet spot (μ=0): exact edge Majorana zero modes
5. Zero mode splitting ~ exp(-L/ξ): topological protection
6. Two Majorana zero modes = one topological qubit
    """)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Majorana property | $\gamma = \gamma^\dagger$, $\gamma^2 = 1$ |
| Anticommutation | $\{\gamma_i, \gamma_j\} = 2\delta_{ij}$ |
| Fermion decomposition | $c = \frac{1}{2}(\gamma_1 + i\gamma_2)$ |
| Kitaev Hamiltonian | $H = -\mu\sum n_j - t\sum(c_j^\dagger c_{j+1} + h.c.) + \Delta\sum(c_jc_{j+1} + h.c.)$ |
| Phase boundary | $|\mu| = 2t$ |
| Localization length | $\xi = 1/\ln(2t/|\mu|)$ |
| Energy splitting | $\delta E \sim e^{-L/\xi}$ |

### Main Takeaways

1. **Majorana operators** are self-adjoint fermion operators ($\gamma = \gamma^\dagger$) that square to the identity.

2. **Every fermion has two Majorana components** - like real and imaginary parts of a complex number.

3. **The Kitaev chain** is a 1D model that hosts Majorana zero modes at its ends when in the topological phase ($|\mu| < 2|t|$).

4. **Topological protection** comes from the spatial separation of Majoranas - errors require tunneling across the entire chain.

5. **The energy splitting** between degenerate ground states decreases exponentially with system size, enabling long coherence times.

6. **Two Majorana zero modes** define a non-local fermion, encoding one topological qubit in their parity.

---

## Daily Checklist

- [ ] I can derive Majorana operators from fermion operators
- [ ] I understand the Kitaev chain Hamiltonian and its sweet spot
- [ ] I can identify the topological phase from the phase diagram
- [ ] I understand why Majorana zero modes appear at the chain edges
- [ ] I can explain the origin of ground state degeneracy
- [ ] I have run the computational lab and observed Majorana localization

---

## Preview of Day 934

Tomorrow we explore **Topological Superconductors** - the physical systems where Majorana zero modes can actually be realized:

- Semiconductor nanowires with spin-orbit coupling
- Proximity-induced superconductivity
- The recipe for engineering a Kitaev chain
- Zero-bias conductance peaks as Majorana signatures

We'll bridge from theoretical models to experimental reality!
