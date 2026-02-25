# Day 489: Applications of Second Quantization

## Overview

**Day 489 of 2520 | Week 70, Day 6 | Month 18: Identical Particles & Many-Body Physics**

Today we apply the second quantization formalism to important physical models. We study the tight-binding model for electrons in crystals, the Hubbard model for strongly correlated systems, and preview BCS theory for superconductivity. These models are cornerstones of condensed matter physics and represent some of the most promising applications for quantum simulation on near-term quantum computers.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Tight-Binding Model | 60 min |
| 10:00 AM | Band Structure from Second Quantization | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Hubbard Model Introduction | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | BCS Theory Preview | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Quantum Simulation Applications | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Construct** the tight-binding Hamiltonian in second quantization
2. **Derive** energy bands from the tight-binding model
3. **Explain** the Hubbard model and its parameters
4. **Identify** key features of strongly correlated systems
5. **Describe** the BCS pairing mechanism in superconductors
6. **Discuss** quantum simulation applications of these models

---

## 1. The Tight-Binding Model

### Physical Motivation

In crystalline solids, electrons can hop between atomic sites:
- Atoms form a regular lattice
- Electron wave functions overlap between neighbors
- Hopping creates energy bands

### The Hamiltonian

For electrons hopping on a lattice with sites $i, j$:

$$\boxed{\hat{H}_{TB} = -t \sum_{\langle i,j \rangle, \sigma} \left(\hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + \hat{c}_{j\sigma}^\dagger \hat{c}_{i\sigma}\right) + \epsilon_0 \sum_{i,\sigma} \hat{n}_{i\sigma}}$$

**Parameters:**
- $t$ = hopping amplitude (typically 0.1-1 eV)
- $\epsilon_0$ = on-site energy (sets energy zero)
- $\langle i,j \rangle$ = nearest-neighbor pairs
- $\sigma = \uparrow, \downarrow$ = spin index

### Physical Interpretation

$$\hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma}$$

- Annihilates electron with spin $\sigma$ at site $j$
- Creates electron with spin $\sigma$ at site $i$
- **Net effect:** Electron hops from $j$ to $i$

The Hermitian conjugate describes hopping in the opposite direction.

### One-Dimensional Chain

For a 1D chain with N sites and periodic boundary conditions:

$$\hat{H} = -t \sum_{i=1}^{N} \sum_\sigma \left(\hat{c}_{i\sigma}^\dagger \hat{c}_{i+1,\sigma} + \hat{c}_{i+1,\sigma}^\dagger \hat{c}_{i\sigma}\right)$$

with $\hat{c}_{N+1} \equiv \hat{c}_1$ (periodic).

### Diagonalization via Fourier Transform

Define momentum-space operators:
$$\hat{c}_{k\sigma} = \frac{1}{\sqrt{N}} \sum_{j=1}^{N} e^{-ikR_j} \hat{c}_{j\sigma}$$

where $k = \frac{2\pi n}{Na}$ for $n = 0, 1, \ldots, N-1$ and $R_j = ja$.

The Hamiltonian becomes diagonal:
$$\boxed{\hat{H} = \sum_{k,\sigma} \epsilon(k) \hat{c}_{k\sigma}^\dagger \hat{c}_{k\sigma}}$$

with dispersion relation:
$$\boxed{\epsilon(k) = -2t\cos(ka)}$$

### Band Structure

The dispersion $\epsilon(k) = -2t\cos(ka)$ gives:
- **Bandwidth:** $W = 4t$ (from $-2t$ to $+2t$)
- **Band bottom:** $k = 0$, $\epsilon = -2t$
- **Band top:** $k = \pi/a$, $\epsilon = +2t$
- **Fermi surface:** At half-filling, $k_F = \pm\pi/(2a)$

### Higher Dimensions

**2D square lattice:**
$$\epsilon(\mathbf{k}) = -2t[\cos(k_x a) + \cos(k_y a)]$$

**3D cubic lattice:**
$$\epsilon(\mathbf{k}) = -2t[\cos(k_x a) + \cos(k_y a) + \cos(k_z a)]$$

---

## 2. The Hubbard Model

### Motivation: Electron-Electron Interaction

The tight-binding model ignores electron-electron interaction. When two electrons occupy the same site, they repel each other!

### The Hubbard Hamiltonian

$$\boxed{\hat{H}_{Hubbard} = -t \sum_{\langle i,j \rangle, \sigma} \left(\hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + h.c.\right) + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}}$$

**New term:** $U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}$
- $U$ = on-site Coulomb repulsion (typically 1-10 eV)
- $\hat{n}_{i\uparrow} \hat{n}_{i\downarrow}$ = doubly occupied site
- Energy cost $U$ when two electrons share a site

### Parameter Regimes

**Weak coupling ($U \ll t$):**
- Electrons delocalized
- Metallic behavior
- Fermi liquid theory applies

**Strong coupling ($U \gg t$):**
- Electrons localized (one per site at half-filling)
- Mott insulator behavior
- Magnetic ordering (antiferromagnetism)

**Intermediate ($U \sim t$):**
- Strongly correlated regime
- Non-perturbative physics
- High-$T_c$ superconductivity region

### Two-Site Hubbard Model

The simplest case: two sites, two electrons (one per site on average).

Hilbert space: $\{|\uparrow,\uparrow\rangle, |\uparrow,\downarrow\rangle, |\downarrow,\uparrow\rangle, |\downarrow,\downarrow\rangle, |\uparrow\downarrow, 0\rangle, |0, \uparrow\downarrow\rangle\}$

In the $S_z = 0$ sector (two states with opposite spins):

$$\hat{H} = \begin{pmatrix} 0 & -t & -t & 0 \\ -t & U & 0 & -t \\ -t & 0 & U & -t \\ 0 & -t & -t & 0 \end{pmatrix}$$

(In basis: $|\uparrow,\downarrow\rangle$, $|\uparrow\downarrow, 0\rangle$, $|0, \uparrow\downarrow\rangle$, $|\downarrow,\uparrow\rangle$)

**Ground state energy:**
$$E_0 = \frac{U - \sqrt{U^2 + 16t^2}}{2}$$

For $U \gg t$: $E_0 \approx -\frac{4t^2}{U}$ (superexchange)

### Magnetic Correlations

At half-filling ($\langle n \rangle = 1$) and large $U$:
- Double occupation suppressed
- Virtual hopping creates antiferromagnetic exchange:
$$J = \frac{4t^2}{U}$$
- Effective Heisenberg model: $\hat{H}_{eff} = J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j$

### Away from Half-Filling: Doped Hubbard Model

- Adding/removing electrons creates "holes" or "doublons"
- Holes can move freely (metallic)
- Believed to describe high-$T_c$ superconductors
- **Still unsolved after 40+ years!**

---

## 3. BCS Theory Preview

### The Superconducting State

Below critical temperature $T_c$:
- Zero electrical resistance
- Perfect diamagnetism (Meissner effect)
- Energy gap in excitation spectrum

### Cooper Pairs

BCS insight: electrons near Fermi surface can form bound pairs!

**Effective attraction:** Mediated by phonons (lattice vibrations)
- Electron 1 polarizes lattice
- Electron 2 attracted to polarization
- Net attraction if $|\omega| < \omega_D$ (Debye frequency)

### The BCS Hamiltonian

$$\boxed{\hat{H}_{BCS} = \sum_{\mathbf{k},\sigma} \epsilon_\mathbf{k} \hat{c}_{\mathbf{k}\sigma}^\dagger \hat{c}_{\mathbf{k}\sigma} - g \sum_{\mathbf{k}, \mathbf{k}'} \hat{c}_{\mathbf{k}\uparrow}^\dagger \hat{c}_{-\mathbf{k}\downarrow}^\dagger \hat{c}_{-\mathbf{k}'\downarrow} \hat{c}_{\mathbf{k}'\uparrow}}$$

**Pairing term:** $\hat{c}_{\mathbf{k}\uparrow}^\dagger \hat{c}_{-\mathbf{k}\downarrow}^\dagger$ creates a Cooper pair
- Opposite momenta: $\mathbf{k}$ and $-\mathbf{k}$
- Opposite spins: $\uparrow$ and $\downarrow$
- Total momentum zero: center-of-mass at rest

### Mean-Field Approximation

Define the gap parameter:
$$\Delta = g \sum_\mathbf{k} \langle \hat{c}_{-\mathbf{k}\downarrow} \hat{c}_{\mathbf{k}\uparrow} \rangle$$

Mean-field Hamiltonian:
$$\hat{H}_{MF} = \sum_\mathbf{k} \left[\epsilon_\mathbf{k} (\hat{c}_{\mathbf{k}\uparrow}^\dagger \hat{c}_{\mathbf{k}\uparrow} + \hat{c}_{-\mathbf{k}\downarrow}^\dagger \hat{c}_{-\mathbf{k}\downarrow}) - \Delta(\hat{c}_{\mathbf{k}\uparrow}^\dagger \hat{c}_{-\mathbf{k}\downarrow}^\dagger + h.c.)\right]$$

### Bogoliubov Transformation

Diagonalize via:
$$\hat{\gamma}_{\mathbf{k}\uparrow} = u_\mathbf{k} \hat{c}_{\mathbf{k}\uparrow} - v_\mathbf{k} \hat{c}_{-\mathbf{k}\downarrow}^\dagger$$
$$\hat{\gamma}_{-\mathbf{k}\downarrow}^\dagger = u_\mathbf{k} \hat{c}_{-\mathbf{k}\downarrow}^\dagger + v_\mathbf{k} \hat{c}_{\mathbf{k}\uparrow}$$

with $|u_\mathbf{k}|^2 + |v_\mathbf{k}|^2 = 1$.

**Quasiparticle energies:**
$$\boxed{E_\mathbf{k} = \sqrt{\epsilon_\mathbf{k}^2 + |\Delta|^2}}$$

The gap $\Delta$ opens at the Fermi surface!

### Self-Consistent Gap Equation

$$\Delta = g \sum_\mathbf{k} \frac{\Delta}{2E_\mathbf{k}} \tanh\left(\frac{E_\mathbf{k}}{2k_B T}\right)$$

**At T = 0:**
$$\Delta(0) = 2\hbar\omega_D \exp\left(-\frac{1}{N(0)g}\right)$$

where $N(0)$ is the density of states at Fermi level.

---

## 4. Quantum Simulation Applications

### Why Quantum Simulation?

Classical computers struggle with:
- Exponential Hilbert space growth
- Fermionic sign problem in Monte Carlo
- Real-time dynamics of quantum systems

**Quantum computers are natural simulators of quantum systems!**

### Tight-Binding/Hubbard on Quantum Computers

**Mapping:**
- Each site → one or more qubits
- Fermion operators → Pauli operators (Jordan-Wigner)
- Hopping terms → two-qubit gates
- Interaction terms → one-qubit gates (Z)

**Example: 4-site Hubbard model**
- 8 qubits needed (2 spins × 4 sites)
- Hopping: $\hat{c}_i^\dagger \hat{c}_j$ → string of Z gates + XX/YY gates
- Interaction: $\hat{n}_{i\uparrow}\hat{n}_{i\downarrow}$ → ZZ gate

### Current Achievements

1. **Fermi-Hubbard model:**
   - Simulated on Google's Sycamore (2020)
   - Observed metal-insulator crossover
   - 10-20 site systems

2. **Molecular simulation:**
   - H₂, LiH, BeH₂ ground states via VQE
   - Error rates limit larger molecules

3. **Dynamics:**
   - Quench dynamics in Hubbard model
   - Thermalization studies

### Near-Term Goals

- **Materials discovery:** Predict new superconductors
- **Catalyst design:** Understand reaction mechanisms
- **Quantum phases:** Map phase diagrams
- **Benchmarking:** Compare with classical methods

### Challenges

1. **Qubit number:** Need 100+ for useful chemistry
2. **Gate fidelity:** Current ~99.5%, need ~99.99%
3. **Coherence time:** Need longer for deep circuits
4. **Measurement:** Shot noise limits precision

---

## 5. Worked Examples

### Example 1: 1D Tight-Binding Dispersion

**Problem:** For a 1D chain with lattice constant $a$ and hopping $t$, find the group velocity at the band center.

**Solution:**

Dispersion: $\epsilon(k) = -2t\cos(ka)$

Group velocity: $v_g = \frac{1}{\hbar}\frac{d\epsilon}{dk}$

$$v_g = \frac{1}{\hbar} \cdot 2ta\sin(ka) = \frac{2ta}{\hbar}\sin(ka)$$

At band center, $k = \pi/(2a)$:
$$v_g = \frac{2ta}{\hbar}\sin(\pi/2) = \boxed{\frac{2ta}{\hbar}}$$

This is the maximum group velocity in the band.

### Example 2: Hubbard Model Ground State

**Problem:** For the two-site Hubbard model with $U = 4t$, calculate the ground state energy and the probability of double occupation.

**Solution:**

Ground state energy:
$$E_0 = \frac{U - \sqrt{U^2 + 16t^2}}{2} = \frac{4t - \sqrt{16t^2 + 16t^2}}{2} = \frac{4t - 4\sqrt{2}t}{2}$$
$$E_0 = 2t(1 - \sqrt{2}) \approx \boxed{-0.83t}$$

The ground state is:
$$|GS\rangle = \alpha(|\uparrow,\downarrow\rangle + |\downarrow,\uparrow\rangle) + \beta(|\uparrow\downarrow,0\rangle + |0,\uparrow\downarrow\rangle)$$

where $\beta/\alpha = 2t/(E_0 - U) = 2t/(-0.83t - 4t) = -0.41$.

Double occupation probability:
$$P_{double} = 2|\beta|^2 = \frac{2 \times 0.41^2}{1 + 0.41^2 \times 2} \approx \boxed{0.25}$$

Without interaction ($U=0$): $P_{double} = 0.5$. Interaction reduces double occupation.

### Example 3: BCS Gap

**Problem:** For a BCS superconductor with $\Delta = 1$ meV, calculate the minimum energy to break a Cooper pair.

**Solution:**

Breaking a Cooper pair creates two quasiparticles at minimum energy (at Fermi surface, $\epsilon_\mathbf{k} = 0$):

$$E_{pair} = 2E_\mathbf{k}|_{k=k_F} = 2\sqrt{0 + \Delta^2} = 2\Delta$$

$$E_{pair} = 2 \times 1 \text{ meV} = \boxed{2 \text{ meV}}$$

This is the **superconducting gap** observable in tunneling experiments.

---

## 6. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** For a 2D square lattice tight-binding model, find $\epsilon(k_x, k_y)$ and identify the bandwidth.

**Problem 1.2:** Write the Hubbard Hamiltonian for 3 sites in a ring geometry.

**Problem 1.3:** In BCS theory, if $\Delta = 2$ meV and $\epsilon_\mathbf{k} = 1$ meV, what is the quasiparticle energy?

### Level 2: Intermediate

**Problem 2.1:** Show that the tight-binding model on a bipartite lattice (like a square lattice) has particle-hole symmetry: $\epsilon(-\mathbf{k}) = -\epsilon(\mathbf{k})$ after shifting energy.

**Problem 2.2:** For the half-filled Hubbard model at large $U$, derive the effective antiferromagnetic exchange $J = 4t^2/U$ using second-order perturbation theory.

**Problem 2.3:** Calculate the BCS coherence length $\xi_0 = \hbar v_F / (\pi \Delta)$ for aluminum with $\Delta \approx 0.17$ meV and $v_F \approx 2 \times 10^6$ m/s.

### Level 3: Challenging

**Problem 3.1:** Include next-nearest-neighbor hopping $t'$ in the 2D tight-binding model. Show that this breaks particle-hole symmetry and calculate the modified dispersion.

**Problem 3.2:** For the negative-U Hubbard model (attractive interaction), show that the ground state at half-filling has on-site pairs and calculate the pair binding energy.

**Problem 3.3:** Derive the BCS gap equation at finite temperature and show that $\Delta(T) \to 0$ as $T \to T_c$ with $T_c \approx 0.57\Delta(0)/k_B$.

---

## 7. Computational Lab: Model Hamiltonians

```python
"""
Day 489 Computational Lab: Applications of Second Quantization
Implementing tight-binding, Hubbard, and BCS models.
"""

import numpy as np
from scipy.linalg import eigh
from itertools import combinations, product
import matplotlib.pyplot as plt

class TightBindingModel:
    """
    Tight-binding model on various lattices.
    """

    def __init__(self, lattice_type='1D', N_sites=10, t=1.0, periodic=True):
        """
        Initialize tight-binding model.

        Parameters:
        -----------
        lattice_type : str
            '1D' or '2D_square'
        N_sites : int
            Number of sites (total for 1D, per side for 2D)
        t : float
            Hopping amplitude
        periodic : bool
            Periodic boundary conditions
        """
        self.lattice_type = lattice_type
        self.N = N_sites
        self.t = t
        self.periodic = periodic

        if lattice_type == '1D':
            self.total_sites = N_sites
        else:
            self.total_sites = N_sites**2

    def get_dispersion_1D(self, k):
        """1D dispersion relation."""
        return -2 * self.t * np.cos(k)

    def get_dispersion_2D(self, kx, ky):
        """2D square lattice dispersion."""
        return -2 * self.t * (np.cos(kx) + np.cos(ky))

    def build_hamiltonian_1D(self):
        """Build 1D tight-binding Hamiltonian matrix."""
        H = np.zeros((self.N, self.N))

        for i in range(self.N - 1):
            H[i, i+1] = -self.t
            H[i+1, i] = -self.t

        if self.periodic:
            H[0, self.N-1] = -self.t
            H[self.N-1, 0] = -self.t

        return H

    def build_hamiltonian_2D(self):
        """Build 2D square lattice Hamiltonian matrix."""
        N = self.N
        dim = N * N
        H = np.zeros((dim, dim))

        for i in range(N):
            for j in range(N):
                site = i * N + j

                # Right neighbor
                if j < N - 1:
                    neighbor = i * N + (j + 1)
                    H[site, neighbor] = -self.t
                    H[neighbor, site] = -self.t
                elif self.periodic:
                    neighbor = i * N + 0
                    H[site, neighbor] = -self.t
                    H[neighbor, site] = -self.t

                # Down neighbor
                if i < N - 1:
                    neighbor = (i + 1) * N + j
                    H[site, neighbor] = -self.t
                    H[neighbor, site] = -self.t
                elif self.periodic:
                    neighbor = 0 * N + j
                    H[site, neighbor] = -self.t
                    H[neighbor, site] = -self.t

        return H

    def calculate_band_structure(self, n_k=100):
        """Calculate band structure."""
        if self.lattice_type == '1D':
            k = np.linspace(-np.pi, np.pi, n_k)
            E = self.get_dispersion_1D(k)
            return k, E
        else:
            # High-symmetry path: Γ → X → M → Γ
            n_segment = n_k // 3

            # Γ to X: (0,0) → (π,0)
            kx_GX = np.linspace(0, np.pi, n_segment)
            ky_GX = np.zeros(n_segment)

            # X to M: (π,0) → (π,π)
            kx_XM = np.pi * np.ones(n_segment)
            ky_XM = np.linspace(0, np.pi, n_segment)

            # M to Γ: (π,π) → (0,0)
            kx_MG = np.linspace(np.pi, 0, n_segment)
            ky_MG = np.linspace(np.pi, 0, n_segment)

            kx = np.concatenate([kx_GX, kx_XM, kx_MG])
            ky = np.concatenate([ky_GX, ky_XM, ky_MG])

            E = self.get_dispersion_2D(kx, ky)

            # Path parameter
            path_lengths = [n_segment, n_segment, n_segment]
            k_path = np.arange(len(kx))

            return k_path, E, path_lengths


class HubbardModel:
    """
    Hubbard model for small systems.
    """

    def __init__(self, N_sites, t=1.0, U=4.0, periodic=True):
        """
        Initialize Hubbard model.

        Parameters:
        -----------
        N_sites : int
            Number of lattice sites
        t : float
            Hopping amplitude
        U : float
            On-site interaction
        periodic : bool
            Periodic boundary conditions
        """
        self.N = N_sites
        self.t = t
        self.U = U
        self.periodic = periodic

    def generate_basis(self, N_up, N_down):
        """
        Generate basis states for given number of up/down electrons.
        Each state is a tuple (up_config, down_config).
        """
        up_configs = list(combinations(range(self.N), N_up))
        down_configs = list(combinations(range(self.N), N_down))

        basis = []
        for up in up_configs:
            for down in down_configs:
                basis.append((up, down))

        return basis

    def config_to_occupation(self, config):
        """Convert configuration to occupation list."""
        occ = [0] * self.N
        for site in config:
            occ[site] = 1
        return occ

    def build_hamiltonian(self, N_up, N_down):
        """Build Hubbard Hamiltonian matrix."""
        basis = self.generate_basis(N_up, N_down)
        dim = len(basis)
        H = np.zeros((dim, dim))

        for i, state_i in enumerate(basis):
            up_i, down_i = state_i
            occ_up_i = self.config_to_occupation(up_i)
            occ_down_i = self.config_to_occupation(down_i)

            # Diagonal: U term
            for site in range(self.N):
                if site in up_i and site in down_i:
                    H[i, i] += self.U

            # Off-diagonal: hopping
            for j, state_j in enumerate(basis):
                up_j, down_j = state_j
                occ_up_j = self.config_to_occupation(up_j)
                occ_down_j = self.config_to_occupation(down_j)

                # Up-spin hopping
                if down_i == down_j:
                    hopping = self._hopping_element(up_i, up_j)
                    H[i, j] += -self.t * hopping

                # Down-spin hopping
                if up_i == up_j:
                    hopping = self._hopping_element(down_i, down_j)
                    H[i, j] += -self.t * hopping

        return H, basis

    def _hopping_element(self, config_i, config_j):
        """Calculate hopping matrix element between configurations."""
        # Find differences
        list_i = sorted(config_i)
        list_j = sorted(config_j)

        diff_i = [x for x in list_i if x not in list_j]
        diff_j = [x for x in list_j if x not in list_i]

        # Must differ by exactly one particle
        if len(diff_i) != 1 or len(diff_j) != 1:
            return 0

        site_i = diff_i[0]
        site_j = diff_j[0]

        # Check if neighbors
        if abs(site_i - site_j) == 1:
            return 1
        if self.periodic and abs(site_i - site_j) == self.N - 1:
            return 1

        return 0


def demonstrate_tight_binding():
    """Demonstrate tight-binding model."""

    print("=" * 60)
    print("TIGHT-BINDING MODEL")
    print("=" * 60)

    # 1D band structure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    tb1d = TightBindingModel('1D', N_sites=100, t=1.0)
    k, E = tb1d.calculate_band_structure(200)
    ax.plot(k, E, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(k, E, -3, where=(E < 0), alpha=0.3, color='blue')
    ax.set_xlabel(r'$k$ (units of $1/a$)', fontsize=12)
    ax.set_ylabel(r'$\epsilon(k)/t$', fontsize=12)
    ax.set_title('1D Tight-Binding Band', fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)

    # 2D band structure
    ax = axes[1]
    tb2d = TightBindingModel('2D_square', N_sites=50, t=1.0)
    k_path, E_2d, path_lengths = tb2d.calculate_band_structure(150)
    ax.plot(k_path, E_2d, 'b-', linewidth=2)

    # Mark high-symmetry points
    cumsum = [0, path_lengths[0], path_lengths[0]+path_lengths[1], sum(path_lengths)]
    for x in cumsum:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(cumsum)
    ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
    ax.set_ylabel(r'$\epsilon(k)/t$', fontsize=12)
    ax.set_title('2D Square Lattice Band', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Density of states
    ax = axes[2]
    # Sample many k points
    kx = np.random.uniform(-np.pi, np.pi, 100000)
    ky = np.random.uniform(-np.pi, np.pi, 100000)
    E_dos = tb2d.get_dispersion_2D(kx, ky)
    ax.hist(E_dos, bins=100, density=True, alpha=0.7, color='green')
    ax.set_xlabel(r'$\epsilon/t$', fontsize=12)
    ax.set_ylabel('DOS (arb. units)', fontsize=12)
    ax.set_title('2D Density of States', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tight_binding.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Numerical vs analytic
    print("\n1D tight-binding eigenvalues comparison:")
    H = tb1d.build_hamiltonian_1D()[:10, :10]
    numerical_E = np.linalg.eigvalsh(H)
    k_values = 2*np.pi*np.arange(10)/10
    analytic_E = sorted(-2*tb1d.t*np.cos(k_values))
    print(f"  Numerical (first 5): {np.sort(numerical_E)[:5]}")
    print(f"  Analytic (first 5): {np.array(analytic_E)[:5]}")


def demonstrate_hubbard():
    """Demonstrate Hubbard model."""

    print("\n" + "=" * 60)
    print("HUBBARD MODEL")
    print("=" * 60)

    # Two-site Hubbard model
    print("\nTwo-site Hubbard model at half-filling (1 up, 1 down):")

    U_values = [0, 1, 2, 4, 8, 16]
    t = 1.0

    ground_energies = []
    double_occ = []

    for U in U_values:
        hubbard = HubbardModel(N_sites=2, t=t, U=U, periodic=False)
        H, basis = hubbard.build_hamiltonian(N_up=1, N_down=1)

        eigenvalues, eigenvectors = eigh(H)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        ground_energies.append(E0)

        # Calculate double occupation
        P_double = 0
        for idx, state in enumerate(basis):
            up, down = state
            for site in range(2):
                if site in up and site in down:
                    P_double += np.abs(psi0[idx])**2
        double_occ.append(P_double)

        print(f"  U/t = {U/t:.1f}: E_0 = {E0:.4f}t, P_double = {P_double:.4f}")

    # Analytic comparison
    print("\n  Analytic: E_0 = (U - sqrt(U² + 16t²))/2")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    U_fine = np.linspace(0.01, 16, 100)
    E_analytic = (U_fine - np.sqrt(U_fine**2 + 16*t**2)) / 2
    ax.plot(U_fine/t, E_analytic/t, 'b-', linewidth=2, label='Analytic')
    ax.plot(np.array(U_values)/t, np.array(ground_energies)/t, 'ro',
            markersize=8, label='Numerical')
    ax.set_xlabel('U/t', fontsize=12)
    ax.set_ylabel(r'$E_0/t$', fontsize=12)
    ax.set_title('Two-site Hubbard Ground State Energy', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.array(U_values)/t, double_occ, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Non-interacting')
    ax.set_xlabel('U/t', fontsize=12)
    ax.set_ylabel('Double Occupation', fontsize=12)
    ax.set_title('Suppression of Double Occupation', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hubbard_model.png', dpi=150, bbox_inches='tight')
    plt.show()


def bcs_gap_equation():
    """Solve BCS gap equation."""

    print("\n" + "=" * 60)
    print("BCS SUPERCONDUCTIVITY")
    print("=" * 60)

    # Model parameters
    omega_D = 0.01  # Debye frequency (in units of bandwidth)
    g = 0.3  # Coupling constant
    N0 = 1.0  # DOS at Fermi level

    def gap_equation_T0(Delta, g, omega_D, N0):
        """Gap equation at T=0."""
        # Δ = g * N0 * ∫_0^ωD dε Δ / (2√(ε² + Δ²))
        # = g * N0 * Δ * arcsinh(ωD/Δ)
        if Delta < 1e-10:
            return np.inf
        return 1 - g * N0 * np.arcsinh(omega_D / Delta)

    # Solve for gap
    from scipy.optimize import brentq
    Delta_0 = brentq(gap_equation_T0, 1e-10, omega_D, args=(g, omega_D, N0))

    print(f"BCS gap at T=0: Δ(0) = {Delta_0:.6f}")
    print(f"Weak-coupling prediction: Δ ≈ 2ωD exp(-1/gN0) = {2*omega_D*np.exp(-1/(g*N0)):.6f}")

    # Temperature dependence
    def gap_equation_T(Delta, T, g, omega_D, N0):
        """Gap equation at finite T."""
        if Delta < 1e-10:
            return 1
        # Numerical integration
        epsilon = np.linspace(0, omega_D, 1000)
        E = np.sqrt(epsilon**2 + Delta**2)
        if T < 1e-10:
            integrand = Delta / (2 * E)
        else:
            integrand = Delta / (2 * E) * np.tanh(E / (2 * T))
        return 1 - g * N0 * np.trapz(integrand, epsilon)

    # Find Tc
    T_values = np.linspace(0.001, 0.01, 50)
    Delta_T = []

    for T in T_values:
        try:
            Delta = brentq(gap_equation_T, 1e-10, Delta_0 * 2, args=(T, g, omega_D, N0))
        except:
            Delta = 0
        Delta_T.append(Delta)

    Delta_T = np.array(Delta_T)

    # Find Tc
    Tc_idx = np.argmax(Delta_T < 1e-6)
    if Tc_idx > 0:
        Tc = T_values[Tc_idx]
    else:
        Tc = T_values[-1]

    print(f"Critical temperature: Tc ≈ {Tc:.6f}")
    print(f"Ratio Δ(0)/Tc = {Delta_0/Tc:.3f} (BCS prediction: 1.76)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(T_values / Tc, Delta_T / Delta_0, 'b-', linewidth=2)
    ax.set_xlabel(r'$T/T_c$', fontsize=12)
    ax.set_ylabel(r'$\Delta(T)/\Delta(0)$', fontsize=12)
    ax.set_title('BCS Gap Temperature Dependence', fontsize=12)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Quasiparticle spectrum
    ax = axes[1]
    epsilon = np.linspace(-3*Delta_0, 3*Delta_0, 200)
    E_qp = np.sqrt(epsilon**2 + Delta_0**2)

    ax.plot(epsilon / Delta_0, E_qp / Delta_0, 'b-', linewidth=2,
            label='Superconducting')
    ax.plot(epsilon / Delta_0, np.abs(epsilon) / Delta_0, 'r--', linewidth=2,
            label='Normal')
    ax.fill_between(epsilon / Delta_0, 0, 1, alpha=0.2, color='blue',
                    label='Gap region')
    ax.set_xlabel(r'$\epsilon/\Delta$', fontsize=12)
    ax.set_ylabel(r'$E_k/\Delta$', fontsize=12)
    ax.set_title('BCS Quasiparticle Dispersion', fontsize=12)
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bcs_theory.png', dpi=150, bbox_inches='tight')
    plt.show()


def quantum_simulation_demo():
    """Demonstrate quantum simulation concepts."""

    print("\n" + "=" * 60)
    print("QUANTUM SIMULATION APPLICATIONS")
    print("=" * 60)

    print("""
    MAPPING HUBBARD MODEL TO QUANTUM COMPUTER
    =========================================

    System: 2-site Hubbard model
    Qubits needed: 4 (2 sites × 2 spins)

    Qubit assignment:
      q0 = site 1, spin up
      q1 = site 1, spin down
      q2 = site 2, spin up
      q3 = site 2, spin down

    Jordan-Wigner mapping:
      c†_1↑ = σ⁺₀
      c†_1↓ = Z₀ ⊗ σ⁺₁
      c†_2↑ = Z₀ ⊗ Z₁ ⊗ σ⁺₂
      c†_2↓ = Z₀ ⊗ Z₁ ⊗ Z₂ ⊗ σ⁺₃

    Hamiltonian terms:
      Hopping: c†_1σ c_2σ + h.c. → XX + YY gates
      Interaction: n_1↑ n_1↓ → (I-Z₀)(I-Z₁)/4

    Circuit depth scales with system size!
    """)

    # Calculate number of terms for larger systems
    print("\nScaling of Hubbard model simulation:")
    print("-" * 50)
    for N in [2, 4, 6, 8, 10]:
        qubits = 2 * N
        hopping_terms = 2 * N  # 1D with PBC
        interaction_terms = N
        total_terms = hopping_terms + interaction_terms
        print(f"  {N} sites: {qubits} qubits, {total_terms} Hamiltonian terms")


# Main execution
if __name__ == "__main__":
    print("Day 489: Applications of Second Quantization")
    print("=" * 60)

    demonstrate_tight_binding()
    demonstrate_hubbard()
    bcs_gap_equation()
    quantum_simulation_demo()
```

---

## 8. Summary

### Key Concepts

| Model | Hamiltonian | Key Physics |
|-------|-------------|-------------|
| Tight-binding | $-t\sum_{\langle i,j\rangle} c_i^\dagger c_j$ | Band structure, metals |
| Hubbard | Tight-binding + $U\sum_i n_{i\uparrow}n_{i\downarrow}$ | Mott transition, magnetism |
| BCS | Kinetic + pairing interaction | Superconductivity |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\epsilon(k) = -2t\cos(ka)$$ | 1D tight-binding dispersion |
| $$J = 4t^2/U$$ | Superexchange coupling |
| $$E_k = \sqrt{\epsilon_k^2 + \Delta^2}$$ | BCS quasiparticle energy |
| $$\Delta(0) \approx 2\omega_D e^{-1/N(0)g}$$ | BCS gap |

---

## 9. Daily Checklist

### Conceptual Understanding
- [ ] I can write the tight-binding Hamiltonian
- [ ] I understand the Hubbard model parameters and regimes
- [ ] I know the BCS pairing mechanism
- [ ] I understand quantum simulation applications

### Mathematical Skills
- [ ] I can derive band structures from tight-binding
- [ ] I can calculate Hubbard model ground states
- [ ] I can solve the BCS gap equation
- [ ] I can map models to qubit representations

### Computational Skills
- [ ] I implemented tight-binding band structure calculation
- [ ] I diagonalized small Hubbard models
- [ ] I solved the BCS gap equation numerically

### Quantum Computing Connection
- [ ] I understand qubit requirements for Hubbard simulation
- [ ] I know current experimental achievements
- [ ] I recognize the scaling challenges

---

## 10. Preview: Day 490

Tomorrow is the **Week 70 Review**, covering:

- Comprehensive summary of second quantization
- Problem set spanning all topics
- Self-assessment checklist
- Preparation for upcoming topics in many-body physics

---

## References

1. Ashcroft, N.W. & Mermin, N.D. (1976). *Solid State Physics*. Cengage Learning.

2. Fazekas, P. (1999). *Lecture Notes on Electron Correlation and Magnetism*. World Scientific.

3. Tinkham, M. (2004). *Introduction to Superconductivity*, 2nd ed. Dover.

4. Qin, M. et al. (2022). "The Hubbard Model: A Computational Perspective." *Annu. Rev. Condens. Matter Phys.* 13, 275.

---

*"The Hubbard model is deceptively simple to write down, but captures the essence of strongly correlated electron physics."*
— Patrick Fazekas

---

**Day 489 Complete.** Tomorrow: Week 70 Review.
