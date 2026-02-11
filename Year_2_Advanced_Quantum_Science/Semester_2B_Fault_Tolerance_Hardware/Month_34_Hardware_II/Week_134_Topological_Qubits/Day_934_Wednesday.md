# Day 934: Topological Superconductors

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Semiconductor-superconductor heterostructures and the Oreg-Lutchyn recipe |
| Afternoon | 2 hours | Problem solving: Band structure and topological transitions |
| Evening | 2 hours | Computational lab: Nanowire simulations |

## Learning Objectives

By the end of today, you will be able to:

1. **Explain the key ingredients** for engineering topological superconductivity
2. **Derive the effective Hamiltonian** for semiconductor-superconductor nanowires
3. **Calculate the topological phase diagram** as a function of Zeeman field and chemical potential
4. **Predict zero-bias conductance peaks** as signatures of Majorana modes
5. **Identify experimental challenges** and distinguish Majoranas from trivial states
6. **Design optimal device parameters** for robust Majorana zero modes

---

## Core Content

### 1. From Kitaev Chain to Real Materials

The Kitaev chain requires spinless p-wave superconductivity - a pairing symmetry not found naturally in any known material. The breakthrough came from realizing we can **engineer** an effective Kitaev chain using:

1. **Semiconductor with strong spin-orbit coupling** (InAs, InSb)
2. **Conventional s-wave superconductor** (Al, NbTiN)
3. **External magnetic field** (Zeeman splitting)

This "Oreg-Lutchyn recipe" (2010) creates effective p-wave pairing from ordinary s-wave pairing!

### 2. The Semiconductor Nanowire Hamiltonian

Consider a 1D semiconductor nanowire:

$$\boxed{H_\text{wire} = \frac{p^2}{2m^*} - \mu + \alpha_R p \sigma_y + V_Z \sigma_x}$$

where:
- $m^*$ = effective electron mass
- $\mu$ = chemical potential
- $\alpha_R$ = Rashba spin-orbit coupling strength
- $V_Z = g\mu_B B/2$ = Zeeman energy from magnetic field $B$
- $\sigma_{x,y}$ = Pauli matrices for spin

#### Physical Interpretation

**Spin-orbit coupling** ($\alpha_R p \sigma_y$):
- Couples momentum to spin
- Electrons with $+p$ feel a different spin-dependent potential than those with $-p$
- Creates spin-momentum locking

**Zeeman field** ($V_Z \sigma_x$):
- Opens a gap at $p = 0$
- Polarizes spins
- Combined with SOC, creates an effective spinless regime

### 3. Proximity-Induced Superconductivity

When the nanowire is placed in contact with an s-wave superconductor, Cooper pairs tunnel into the wire:

$$\boxed{H_\text{SC} = \Delta_\text{ind} \sum_j (c_{j\uparrow}^\dagger c_{j\downarrow}^\dagger + c_{j\downarrow} c_{j\uparrow})}$$

The induced gap $\Delta_\text{ind}$ inherits its magnitude from the parent superconductor but is typically smaller due to interface transparency.

#### The Full Hamiltonian

In momentum space:
$$H(k) = \begin{pmatrix} h(k) & i\sigma_y\Delta_\text{ind} \\ -i\sigma_y\Delta_\text{ind}^* & -h^*(-k) \end{pmatrix}$$

where:
$$h(k) = \frac{\hbar^2 k^2}{2m^*} - \mu + \alpha_R k \sigma_y + V_Z \sigma_x$$

This is a $4 \times 4$ Bogoliubov-de Gennes (BdG) Hamiltonian.

### 4. The Topological Phase Transition

The energy spectrum reveals two distinct phases.

#### Band Structure at $k = 0$

At $k = 0$, the eigenvalues are:
$$E = \pm\sqrt{(\mu \pm V_Z)^2 + \Delta_\text{ind}^2}$$

Gap closes when:
$$V_Z^2 = \mu^2 + \Delta_\text{ind}^2$$

#### Topological Criterion

$$\boxed{V_Z > \sqrt{\mu^2 + \Delta_\text{ind}^2} \quad \Rightarrow \quad \text{Topological phase}}$$

$$\boxed{V_Z < \sqrt{\mu^2 + \Delta_\text{ind}^2} \quad \Rightarrow \quad \text{Trivial phase}}$$

At the transition, the bulk gap closes and reopens with inverted band character.

#### Phase Diagram

The phase boundary is a hyperbola in the $(V_Z, \mu)$ plane:
$$V_Z^2 - \mu^2 = \Delta_\text{ind}^2$$

### 5. Effective p-Wave Pairing

Why does s-wave pairing become effective p-wave?

In the limit $V_Z \gg \Delta_\text{ind}, \alpha_R k$:
- Only one spin species (aligned with field) is near the Fermi level
- The s-wave pairing acts on this single spin band
- With spin-orbit coupling, this becomes odd-parity (p-wave) in the effective 1D model

The effective pairing gap:
$$\boxed{\Delta_\text{eff}(k) \approx \frac{\alpha_R k \Delta_\text{ind}}{V_Z}}$$

This has p-wave symmetry: odd in $k$!

### 6. Zero-Bias Conductance Peak

The primary experimental signature of Majorana zero modes is a **quantized zero-bias conductance peak**.

#### Normal-Superconductor Junction

In a normal metal-superconductor (NS) junction:
- For $E < \Delta$: normal reflection or Andreev reflection
- Andreev reflection: electron reflects as hole, creating Cooper pair
- Conductance depends on Andreev reflection probability

#### Majorana-Induced Andreev Reflection

A Majorana mode provides a **perfect Andreev reflection channel**:

$$\boxed{G(V=0) = \frac{2e^2}{h} \quad \text{(quantized!)}}$$

This is because:
- Majorana is its own antiparticle: $\gamma = \gamma^\dagger$
- Incoming electron must convert to outgoing hole (and vice versa)
- Perfect reflection probability: $R_A = 1$

#### Non-Quantized Peaks

In practice, many effects can produce zero-bias peaks that aren't from Majoranas:
- Disorder-induced Andreev bound states
- Kondo effect
- Weak antilocalization
- Soft gap due to quasiparticle poisoning

The **height** and **width** of the peak provide diagnostic information.

### 7. Experimental Platform Design

#### Material Requirements

**Semiconductor**:
| Property | InAs | InSb |
|----------|------|------|
| $m^*/m_e$ | 0.023 | 0.014 |
| $g$-factor | 15 | 50 |
| $\alpha_R$ (meV nm) | 20-30 | 50-100 |
| Mobility | Good | Better |

InSb has larger SOC and g-factor, but InAs has better epitaxial growth with Al.

**Superconductor**:
- Aluminum (Al): $T_c \approx 1.2$ K, very clean interface with InAs
- NbTiN: Higher $T_c \approx 15$ K, harder interface

#### Device Geometry

```
     Gate electrodes
          ↓ ↓ ↓
    ═══════════════════  ← Tunnel barrier
    ─────────────────── ← Semiconductor nanowire (InAs)
    ███████████████████ ← Superconductor shell (Al)
    ↑
    Magnetic field B along wire axis
```

**Critical parameters**:
- Wire diameter: 50-100 nm
- Wire length: 1-3 μm
- Al shell: half-shell or full-shell coverage
- Gate voltage: tunes $\mu$
- Magnetic field: tunes $V_Z$

### 8. Hard Gap vs. Soft Gap

A key quality metric is the "hardness" of the induced superconducting gap.

**Hard gap**: Zero states inside the gap (except Majoranas)
$$\rho(E) = 0 \text{ for } |E| < \Delta_\text{ind}$$

**Soft gap**: Finite density of states inside gap (disorder, interface quality)
$$\rho(E) \neq 0 \text{ for } |E| < \Delta_\text{ind}$$

A soft gap leads to:
- Quasiparticle poisoning
- Reduced Majorana coherence
- Difficulty in braiding operations

Recent advances (epitaxial Al on InAs) have achieved hard gaps with subgap conductance $< 10^{-3}$ of normal state.

---

## Quantum Computing Applications

### Device Integration

For a topological qubit, we need:
1. **Two or more Majorana zero modes** (minimum 4 for one qubit)
2. **Tunable couplings** between Majoranas
3. **Readout mechanism** (typically charge sensing)
4. **Scalable architecture** for multiple qubits

### Qubit Designs

**Linear geometry**:
```
   γ₁ ══════════ γ₂     γ₃ ══════════ γ₄
   ←── Wire 1 ───→     ←── Wire 2 ───→
```
Parity of each wire pair encodes qubit state.

**T-junction geometry** (for braiding):
```
        γ₂
        ║
   γ₁ ══╬══ γ₃
        ║
        γ₄
```
Allows moving Majoranas for braiding operations.

### Measurement-Based Operations

Since physical braiding is challenging:
1. Measure joint parity of Majorana pairs
2. Use measurement outcomes to track logical state
3. Implement gates through sequences of measurements
4. Enables all Clifford gates

---

## Worked Examples

### Example 1: Topological Transition Field

**Problem**: A nanowire has $\mu = 0.5$ meV and induced gap $\Delta_\text{ind} = 0.25$ meV. What is the critical Zeeman field for the topological transition?

**Solution**:

The topological criterion is:
$$V_Z^c = \sqrt{\mu^2 + \Delta_\text{ind}^2}$$

Substituting values:
$$V_Z^c = \sqrt{(0.5)^2 + (0.25)^2} = \sqrt{0.25 + 0.0625} = \sqrt{0.3125}$$
$$V_Z^c = 0.559 \text{ meV}$$

For InSb with $g = 50$:
$$B_c = \frac{2V_Z^c}{g\mu_B} = \frac{2 \times 0.559 \text{ meV}}{50 \times 0.058 \text{ meV/T}}$$
$$B_c = \frac{1.118}{2.9} \text{ T} = 0.39 \text{ T}$$

$$\boxed{V_Z^c = 0.56 \text{ meV}, \quad B_c \approx 0.4 \text{ T}}$$

### Example 2: Spin-Orbit Length

**Problem**: Calculate the spin-orbit length $l_{SO}$ for InAs with $\alpha_R = 30$ meV nm and $m^* = 0.023 m_e$.

**Solution**:

The spin-orbit length is defined as:
$$l_{SO} = \frac{\hbar^2}{m^* \alpha_R}$$

Converting units:
$$\hbar = 0.658 \text{ eV fs} = 658 \text{ meV fs}$$
$$\hbar^2 = 433 \text{ meV}^2 \text{ fs}^2$$

We need consistent units. Using:
$$m^* = 0.023 \times 0.511 \text{ MeV}/c^2 = 0.0118 \text{ MeV}/c^2$$

Or more directly with energy-length units:
$$l_{SO} = \frac{\hbar^2/(2m^*)}{m^* \alpha_R^2/\hbar^2} \times \frac{1}{\alpha_R/\hbar}$$

The spin-orbit energy is $E_{SO} = m^* \alpha_R^2/(2\hbar^2)$ and:
$$l_{SO} = \frac{\hbar}{m^* \alpha_R}$$

Numerically (using $\hbar^2/(2m^*) \approx 1.6$ eV nm² for InAs):
$$l_{SO} = \frac{2 \times 1.6 \times 10^3 \text{ meV nm}^2}{30 \text{ meV nm}} \times \frac{1}{m^*/m_e}$$

Actually, let's use the direct formula:
$$l_{SO} = \frac{\hbar^2}{m^* \alpha_R}$$

With $\hbar^2/(2m_e) = 3.81$ eV Å² and InAs effective mass $m^* = 0.023 m_e$:
$$\frac{\hbar^2}{2m^*} = \frac{3.81}{0.023} = 165.7 \text{ eV Å}^2 = 1657 \text{ meV nm}^2$$

$$l_{SO} = \frac{2 \times 1657 \text{ meV nm}^2}{30 \text{ meV nm}} = \frac{3314}{30} \text{ nm} = 110 \text{ nm}$$

$$\boxed{l_{SO} \approx 110 \text{ nm}}$$

This sets the scale for Majorana localization.

### Example 3: Effective p-Wave Gap

**Problem**: For a nanowire with $V_Z = 1$ meV, $\Delta_\text{ind} = 0.3$ meV, $\alpha_R = 30$ meV nm, and Fermi wavevector $k_F = 0.02$ nm⁻¹, estimate the effective p-wave gap.

**Solution**:

The effective gap at the Fermi level:
$$\Delta_\text{eff} = \frac{\alpha_R k_F \Delta_\text{ind}}{\sqrt{V_Z^2 - \Delta_\text{ind}^2}}$$

In the topological regime where $V_Z > \Delta_\text{ind}$:
$$\Delta_\text{eff} = \frac{30 \times 0.02 \times 0.3}{\sqrt{1^2 - 0.3^2}}$$
$$= \frac{0.18}{\sqrt{0.91}} = \frac{0.18}{0.954} = 0.19 \text{ meV}$$

$$\boxed{\Delta_\text{eff} \approx 0.19 \text{ meV}}$$

This gap protects the Majorana modes. For temperature $T$, we need $k_B T \ll \Delta_\text{eff}$, i.e., $T \ll 2.2$ K.

---

## Practice Problems

### Level 1: Direct Application

1. **Critical Field**: A nanowire has $\mu = 0$. At what Zeeman field does the topological transition occur?

2. **Gap Closing**: At the topological transition, which wavevector has zero energy gap: $k = 0$ or $k = k_F$?

3. **Material Selection**: Which has a larger g-factor, InAs or InSb? Why does this matter?

### Level 2: Intermediate

4. **Spin-Orbit Strength**: Show that stronger spin-orbit coupling increases the effective p-wave gap. What is the tradeoff?

5. **Finite Length Effects**: For a wire of length $L = 1$ μm with Majorana localization length $\xi = 100$ nm, estimate the energy splitting between the two lowest states.

6. **Phase Diagram**: Sketch the phase diagram in the $(V_Z/\Delta, \mu/\Delta)$ plane. Label the topological and trivial regions.

### Level 3: Challenging

7. **Multi-Band Effects**: Real nanowires have multiple transverse subbands. How does the topological criterion generalize? When can higher subbands induce additional Majorana pairs?

8. **Disorder Effects**: Disorder can localize Majoranas inside the wire (not at ends). Derive the Thouless criterion for when disorder destroys the topological phase.

---

## Computational Lab: Nanowire Simulations

```python
"""
Day 934 Computational Lab: Topological Superconductor Nanowires
Semiconductor-superconductor heterostructure simulations
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Part 1: Nanowire BdG Hamiltonian
# =============================================================================

def nanowire_hamiltonian(N, params):
    """
    Build the BdG Hamiltonian for a semiconductor nanowire with
    proximity-induced superconductivity.

    Parameters:
    -----------
    N : int - Number of lattice sites
    params : dict with keys:
        - t : hopping (sets energy scale, typically ~25 meV for InAs with a=10nm)
        - mu : chemical potential
        - alpha : Rashba spin-orbit coupling (in units of t*a)
        - Vz : Zeeman energy
        - Delta : induced superconducting gap

    Returns:
    --------
    H : 4N x 4N complex array - BdG Hamiltonian
    """
    t = params['t']
    mu = params['mu']
    alpha = params['alpha']
    Vz = params['Vz']
    Delta = params['Delta']

    # Pauli matrices
    s0 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])

    # Particle-hole matrix structure: (e↑, e↓, h↑, h↓)
    # BdG Hamiltonian is 4N × 4N

    H = np.zeros((4*N, 4*N), dtype=complex)

    for j in range(N):
        # On-site terms
        # Electron part: (2t - μ)σ₀ + V_Z σ_x
        h_e = (2*t - mu) * s0 + Vz * sx

        # Hole part: -(2t - μ)σ₀ - V_Z σ_x (particle-hole conjugate)
        h_h = -(2*t - mu) * s0 - Vz * sx

        # Pairing: Δ(iσ_y)
        Delta_mat = Delta * (1j * sy)

        # Block indices
        idx_e = slice(4*j, 4*j + 2)  # electron block
        idx_h = slice(4*j + 2, 4*j + 4)  # hole block

        H[idx_e, idx_e] = h_e
        H[idx_h, idx_h] = h_h
        H[idx_e, idx_h] = Delta_mat
        H[idx_h, idx_e] = Delta_mat.conj().T

    # Hopping terms
    for j in range(N - 1):
        # Electron hopping: -t σ₀ - i(α/2) σ_y
        hop_e = -t * s0 - 1j * (alpha/2) * sy

        # Hole hopping: t σ₀ - i(α/2) σ_y (conjugate of electron hopping)
        hop_h = t * s0 - 1j * (alpha/2) * sy

        idx_e = slice(4*j, 4*j + 2)
        idx_e_next = slice(4*(j+1), 4*(j+1) + 2)
        idx_h = slice(4*j + 2, 4*j + 4)
        idx_h_next = slice(4*(j+1) + 2, 4*(j+1) + 4)

        # Electron hopping
        H[idx_e, idx_e_next] = hop_e
        H[idx_e_next, idx_e] = hop_e.conj().T

        # Hole hopping
        H[idx_h, idx_h_next] = hop_h
        H[idx_h_next, idx_h] = hop_h.conj().T

    return H


def solve_nanowire(N, params):
    """Solve the nanowire BdG Hamiltonian."""
    H = nanowire_hamiltonian(N, params)
    energies, states = eigh(H)
    return energies, states


# =============================================================================
# Part 2: Phase Diagram
# =============================================================================

def compute_phase_diagram(N=100, n_Vz=50, n_mu=50):
    """
    Compute the topological phase diagram.
    """
    print("=" * 50)
    print("Topological Phase Diagram")
    print("=" * 50)

    # Parameters (in units of hopping t)
    t = 1.0
    Delta = 0.2
    alpha = 0.5

    Vz_values = np.linspace(0, 0.8, n_Vz)
    mu_values = np.linspace(-0.5, 0.5, n_mu)

    gap_map = np.zeros((n_Vz, n_mu))
    lowest_E = np.zeros((n_Vz, n_mu))

    for i, Vz in enumerate(Vz_values):
        for j, mu in enumerate(mu_values):
            params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
            energies, _ = solve_nanowire(N, params)

            # Gap is smallest |E|
            gap_map[i, j] = np.min(np.abs(energies))
            lowest_E[i, j] = np.sort(np.abs(energies))[0]

    # Theoretical phase boundary: Vz = sqrt(μ² + Δ²)
    mu_theory = np.linspace(-0.5, 0.5, 100)
    Vz_boundary = np.sqrt(mu_theory**2 + Delta**2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gap map
    im = axes[0].imshow(gap_map, extent=[mu_values[0]/t, mu_values[-1]/t,
                                          Vz_values[0]/Delta, Vz_values[-1]/Delta],
                        origin='lower', aspect='auto', cmap='viridis')
    axes[0].plot(mu_theory/t, Vz_boundary/Delta, 'r--', linewidth=2, label='Theory')
    axes[0].set_xlabel('μ/t')
    axes[0].set_ylabel('V_Z/Δ')
    axes[0].set_title('Energy Gap (color: gap magnitude)')
    axes[0].legend()
    plt.colorbar(im, ax=axes[0], label='Gap/t')

    # Zero mode energy
    im2 = axes[1].imshow(np.log10(lowest_E + 1e-10),
                          extent=[mu_values[0]/t, mu_values[-1]/t,
                                  Vz_values[0]/Delta, Vz_values[-1]/Delta],
                          origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=-4, vmax=0)
    axes[1].plot(mu_theory/t, Vz_boundary/Delta, 'k--', linewidth=2, label='Phase boundary')
    axes[1].set_xlabel('μ/t')
    axes[1].set_ylabel('V_Z/Δ')
    axes[1].set_title('log₁₀(Lowest |E|) - Blue = zero modes')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='log₁₀(E/t)')

    plt.tight_layout()
    plt.savefig('nanowire_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nParameters: Δ/t = {Delta}, α/t = {alpha}")
    print(f"Phase boundary: V_Z = √(μ² + Δ²)")


# =============================================================================
# Part 3: Majorana Wavefunctions in Nanowires
# =============================================================================

def plot_majorana_wavefunctions(N=100):
    """
    Visualize Majorana zero mode wavefunctions.
    """
    print("\n" + "=" * 50)
    print("Majorana Wavefunctions in Nanowires")
    print("=" * 50)

    t = 1.0
    Delta = 0.2
    alpha = 0.5
    mu = 0.0

    Vz_values = [0.15, 0.25, 0.4]  # Below, at, above transition
    labels = ['Trivial (V_Z < V_c)', 'Near transition (V_Z ≈ V_c)',
              'Topological (V_Z > V_c)']

    Vc = np.sqrt(mu**2 + Delta**2)
    print(f"Critical Zeeman field: V_c = {Vc:.3f}t")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sites = np.arange(N)

    for idx, (Vz, label) in enumerate(zip(Vz_values, labels)):
        params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
        energies, states = solve_nanowire(N, params)

        # Plot energy spectrum
        axes[0, idx].bar(range(len(energies)), energies, width=0.8, color='steelblue')
        axes[0, idx].axhline(y=0, color='red', linestyle='--')
        axes[0, idx].set_xlim([len(energies)//2 - 20, len(energies)//2 + 20])
        axes[0, idx].set_ylim([-0.5, 0.5])
        axes[0, idx].set_xlabel('State index')
        axes[0, idx].set_ylabel('E/t')
        axes[0, idx].set_title(f'{label}\nV_Z = {Vz:.2f}t')

        # Find zero modes
        sorted_idx = np.argsort(np.abs(energies))
        zero_indices = sorted_idx[:2]

        for zi in zero_indices:
            # Extract spatial profile
            # States are ordered as (e↑, e↓, h↑, h↓) at each site
            psi = states[:, zi]

            # Probability density at each site
            prob = np.zeros(N)
            for j in range(N):
                prob[j] = np.sum(np.abs(psi[4*j:4*j+4])**2)

            prob = prob / np.max(prob)  # Normalize
            axes[1, idx].plot(sites, prob, 'o-', markersize=2,
                            label=f'E = {energies[zi]:.4f}t')

        axes[1, idx].set_xlabel('Site')
        axes[1, idx].set_ylabel('|ψ|² (normalized)')
        axes[1, idx].legend()
        axes[1, idx].set_yscale('log')
        axes[1, idx].set_ylim([1e-4, 2])

    plt.tight_layout()
    plt.savefig('nanowire_majorana_wf.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 4: Zero-Bias Conductance Peak
# =============================================================================

def local_density_of_states(N, params, energies_sample, broadening=0.01):
    """
    Compute the local density of states at the wire end.
    """
    H = nanowire_hamiltonian(N, params)
    energies, states = eigh(H)

    ldos = np.zeros_like(energies_sample)

    for i, E in enumerate(energies_sample):
        # Lorentzian broadening
        for n, En in enumerate(energies):
            # Weight by wavefunction at first site (electron component)
            psi_end = np.abs(states[0:4, n])**2  # First site, all components
            weight = np.sum(psi_end)  # Total weight at edge
            ldos[i] += weight * broadening / (np.pi * ((E - En)**2 + broadening**2))

    return ldos


def plot_zbcp():
    """
    Plot the zero-bias conductance peak signature.
    """
    print("\n" + "=" * 50)
    print("Zero-Bias Conductance Peak")
    print("=" * 50)

    N = 100
    t = 1.0
    Delta = 0.2
    alpha = 0.5
    mu = 0.0

    E_sample = np.linspace(-0.4, 0.4, 500)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    Vz_values = [0.15, 0.3, 0.5]
    Vc = np.sqrt(mu**2 + Delta**2)

    for idx, Vz in enumerate(Vz_values):
        params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
        ldos = local_density_of_states(N, params, E_sample, broadening=0.005)

        axes[idx].plot(E_sample/Delta, ldos, 'b-', linewidth=1.5)
        axes[idx].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].fill_between(E_sample/Delta, ldos, alpha=0.3)
        axes[idx].set_xlabel('E/Δ')
        axes[idx].set_ylabel('LDOS (arb. units)')

        if Vz < Vc:
            phase = "Trivial"
        else:
            phase = "Topological"
        axes[idx].set_title(f'V_Z = {Vz:.2f}t ({phase})')

    plt.tight_layout()
    plt.savefig('zbcp_signature.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nCritical field: V_c = {Vc:.3f}t = {Vc/Delta:.2f}Δ")
    print("In topological phase: Sharp zero-bias peak from Majorana mode")


# =============================================================================
# Part 5: Field Sweep - Transition Signature
# =============================================================================

def field_sweep():
    """
    Simulate an experimental field sweep showing the topological transition.
    """
    print("\n" + "=" * 50)
    print("Magnetic Field Sweep - Topological Transition")
    print("=" * 50)

    N = 100
    t = 1.0
    Delta = 0.2
    alpha = 0.5
    mu = 0.0

    Vz_values = np.linspace(0, 0.6, 60)
    E_sample = np.linspace(-0.3, 0.3, 200)

    ldos_map = np.zeros((len(Vz_values), len(E_sample)))

    for i, Vz in enumerate(Vz_values):
        params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
        ldos_map[i, :] = local_density_of_states(N, params, E_sample, broadening=0.005)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [E_sample[0]/Delta, E_sample[-1]/Delta,
              Vz_values[0]/Delta, Vz_values[-1]/Delta]

    im = ax.imshow(ldos_map, extent=extent, origin='lower', aspect='auto',
                   cmap='hot', vmin=0, vmax=np.percentile(ldos_map, 95))

    # Mark phase boundary
    Vc = np.sqrt(mu**2 + Delta**2)
    ax.axhline(y=Vc/Delta, color='cyan', linestyle='--', linewidth=2,
              label=f'V_c/Δ = {Vc/Delta:.2f}')

    ax.set_xlabel('Bias Voltage E/Δ')
    ax.set_ylabel('Zeeman Energy V_Z/Δ')
    ax.set_title('Local DOS at Wire End vs Magnetic Field')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='LDOS')

    plt.tight_layout()
    plt.savefig('field_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nKey features:")
    print("- Below V_c: Gap with no zero-energy states")
    print("- At V_c: Gap closing (phase transition)")
    print("- Above V_c: Zero-bias peak from Majorana mode")


# =============================================================================
# Part 6: Parameter Optimization
# =============================================================================

def optimize_parameters():
    """
    Explore how parameters affect topological gap and Majorana localization.
    """
    print("\n" + "=" * 50)
    print("Parameter Optimization for Topological Qubits")
    print("=" * 50)

    N = 80
    t = 1.0
    Delta = 0.2
    mu = 0.0

    # Vary spin-orbit coupling
    alpha_values = np.linspace(0.1, 1.0, 10)
    Vz = 0.4  # In topological regime

    gaps = []
    zero_energies = []

    for alpha in alpha_values:
        params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
        energies, _ = solve_nanowire(N, params)

        sorted_E = np.sort(np.abs(energies))
        zero_energies.append(sorted_E[0])  # Zero mode energy
        gaps.append(sorted_E[2] if len(sorted_E) > 2 else 0)  # Bulk gap

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(alpha_values, zero_energies, 'o-', label='Zero mode E')
    axes[0].semilogy(alpha_values, gaps, 's-', label='Bulk gap')
    axes[0].set_xlabel('Spin-orbit coupling α/t')
    axes[0].set_ylabel('Energy/t')
    axes[0].set_title('Effect of Spin-Orbit Coupling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Vary Zeeman field in topological regime
    Vz_values = np.linspace(0.25, 0.8, 15)
    alpha = 0.5

    effective_gaps = []
    for Vz in Vz_values:
        params = {'t': t, 'mu': mu, 'alpha': alpha, 'Vz': Vz, 'Delta': Delta}
        energies, _ = solve_nanowire(N, params)
        sorted_E = np.sort(np.abs(energies))
        # Effective gap is the bulk gap (3rd smallest energy)
        effective_gaps.append(sorted_E[2] if len(sorted_E) > 2 else 0)

    axes[1].plot(Vz_values/Delta, effective_gaps, 'o-')
    axes[1].axvline(x=1, color='r', linestyle='--', label='V_c/Δ = 1')
    axes[1].set_xlabel('V_Z/Δ')
    axes[1].set_ylabel('Effective Gap/t')
    axes[1].set_title('Topological Gap vs Zeeman Field')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nOptimization insights:")
    print("- Stronger SOC: Better localized Majoranas, larger effective gap")
    print("- Optimal V_Z: Not too close to transition (small gap)")
    print("           Not too large (gap closes again in real materials)")


# =============================================================================
# Part 7: Main Execution
# =============================================================================

def main():
    """Run all demonstrations."""
    print("╔" + "=" * 58 + "╗")
    print("║  Day 934: Topological Superconductor Nanowires            ║")
    print("╚" + "=" * 58 + "╝")

    # 1. Phase diagram
    compute_phase_diagram(N=60, n_Vz=40, n_mu=40)

    # 2. Majorana wavefunctions
    plot_majorana_wavefunctions()

    # 3. Zero-bias peak
    plot_zbcp()

    # 4. Field sweep
    field_sweep()

    # 5. Parameter optimization
    optimize_parameters()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Results:
1. Topological phase when V_Z > √(μ² + Δ²)
2. Majorana modes localized at wire ends in topological phase
3. Zero-bias conductance peak is Majorana signature
4. Strong spin-orbit coupling improves topological gap
5. Need careful balance of V_Z: above transition but not too high
    """)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Nanowire Hamiltonian | $H = \frac{p^2}{2m^*} - \mu + \alpha_R p \sigma_y + V_Z \sigma_x$ |
| Topological criterion | $V_Z > \sqrt{\mu^2 + \Delta_\text{ind}^2}$ |
| Effective p-wave gap | $\Delta_\text{eff} \approx \alpha_R k_F \Delta_\text{ind}/V_Z$ |
| Zeeman energy | $V_Z = g\mu_B B/2$ |
| Quantized conductance | $G(V=0) = 2e^2/h$ |

### Main Takeaways

1. **The Oreg-Lutchyn recipe** creates effective p-wave superconductivity: spin-orbit coupling + Zeeman field + s-wave pairing.

2. **The topological transition** occurs when Zeeman energy exceeds the combined gap from chemical potential and superconducting pairing.

3. **Zero-bias conductance peaks** are the primary experimental signature, arising from perfect Andreev reflection through Majorana modes.

4. **Material choice matters**: High g-factor (InSb) reduces required fields; strong SOC increases the topological gap.

5. **Hard induced gaps** are essential for qubit operation - requires high-quality interfaces.

6. **Distinguishing Majoranas** from trivial states requires multiple complementary measurements (conductance quantization, non-local correlations, exponential protection).

---

## Daily Checklist

- [ ] I can explain the three ingredients for topological superconductivity
- [ ] I understand how spin-orbit coupling enables effective p-wave pairing
- [ ] I can derive the topological phase criterion
- [ ] I can predict zero-bias conductance peak signatures
- [ ] I understand the difference between hard and soft induced gaps
- [ ] I have run the nanowire simulation and observed the phase transition

---

## Preview of Day 935

Tomorrow we explore **Braiding Operations** - how to actually compute with Majorana zero modes:

- Non-Abelian exchange statistics in action
- The braiding matrices for Ising anyons
- T-junction geometries for Majorana manipulation
- Why braiding alone gives only Clifford gates
- Fibonacci anyons and universal quantum computation

We'll see how topology constrains and protects quantum operations!
