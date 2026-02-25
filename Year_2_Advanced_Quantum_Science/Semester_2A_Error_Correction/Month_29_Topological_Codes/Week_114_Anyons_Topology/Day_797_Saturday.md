# Day 797: Topological Order

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 114: Anyons & Topological Order

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Long-range entanglement and topological invariants |
| Afternoon | 2.5 hours | Problem solving: topological quantities |
| Evening | 1.5 hours | Computational lab: entanglement calculations |

---

## Learning Objectives

By the end of today, you will be able to:

1. Define topological order through long-range entanglement
2. Calculate ground state degeneracy from topology: $\text{GSD} = 4^g$ for genus $g$
3. Compute topological entanglement entropy: $S_{\text{topo}} = -\log D$
4. Explain robustness to local perturbations
5. Distinguish topological order from symmetry-breaking order
6. Connect topological order to quantum error correction

---

## Core Content

### 1. What is Topological Order?

**Topological order** is a type of quantum order that:
- Cannot be characterized by local order parameters
- Has ground state degeneracy depending on spatial topology
- Features long-range entanglement that cannot be removed locally
- Is robust to arbitrary local perturbations

#### Historical Context

Introduced by Wen (1989) in the context of the fractional quantum Hall effect:
- FQHE states have the same symmetry but different physics
- Distinguished by topological invariants, not symmetry breaking
- "Quantum order beyond Landau paradigm"

#### Comparison with Symmetry Breaking

| Property | Symmetry Breaking | Topological Order |
|----------|-------------------|-------------------|
| Local order parameter | Yes | No |
| Ground state degeneracy | From symmetry | From topology |
| Excitations | Goldstone bosons | Anyons |
| Robustness | To symmetric perturbations | To all local perturbations |
| Example | Ferromagnet | Toric code |

### 2. Long-Range Entanglement

#### Definition

A state has **long-range entanglement** if it cannot be transformed to a product state by any finite-depth local unitary circuit:

$$|\psi_{\text{topo}}\rangle \neq U_{\text{local}} |\psi_{\text{product}}\rangle$$

for any local unitary $U_{\text{local}} = \prod_i U_i$ with bounded range.

#### The Toric Code Ground State

The toric code ground state:
$$|\Omega\rangle = \prod_v \frac{1 + A_v}{2} \prod_p \frac{1 + B_p}{2} |0\rangle^{\otimes n}$$

This state is **highly entangled**:
- Cannot be written as a product state
- Entanglement spans the entire system
- Correlations are "topological" rather than decaying with distance

#### Correlation Functions

In topologically ordered states:
- Local correlation functions are short-ranged or trivial
- Wilson loop correlations are long-ranged (topological)

For toric code:
$$\langle A_v \rangle = 1, \quad \langle A_v A_{v'} \rangle = 1$$
$$\langle Z_e \rangle = 0, \quad \langle Z_e Z_{e'} \rangle = \delta_{e,e'}$$

No local operator can detect the topological order!

### 3. Ground State Degeneracy

#### Torus: Genus 1

On a torus (genus $g = 1$):
$$\boxed{\text{GSD} = 4}$$

The four ground states are labeled by logical qubit values:
$$|00\rangle, |01\rangle, |10\rangle, |11\rangle$$

Distinguished by eigenvalues of logical operators $\bar{Z}_1, \bar{Z}_2$.

#### General Surface: Genus g

On a closed orientable surface of genus $g$ (number of handles):
$$\boxed{\text{GSD} = 4^g}$$

**Derivation**:
- Each handle contributes 2 independent non-contractible loops
- Each loop supports one logical qubit
- Total: $2g$ logical qubits
- Dimension: $2^{2g} = 4^g$

#### Examples

| Surface | Genus | GSD |
|---------|-------|-----|
| Sphere | 0 | 1 |
| Torus | 1 | 4 |
| Double torus | 2 | 16 |
| g-torus | g | $4^g$ |

### 4. Topological Entanglement Entropy

#### Area Law for Entanglement

In gapped systems, entanglement entropy typically follows an **area law**:
$$S(A) = \alpha |\partial A| - \gamma + \mathcal{O}(1/|\partial A|)$$

where:
- $|\partial A|$ = boundary length (perimeter)
- $\alpha$ = non-universal constant
- $\gamma$ = **topological entanglement entropy**

#### Definition of TEE

The **topological entanglement entropy** (TEE) is:
$$\boxed{S_{\text{topo}} = -\gamma = \log D}$$

where $D$ is the **total quantum dimension**:
$$D = \sqrt{\sum_a d_a^2}$$

with $d_a$ being the quantum dimension of anyon $a$.

#### TEE for Toric Code

For the toric code:
- All anyons have $d_a = 1$ (Abelian)
- Four anyon types: $|\mathcal{A}| = 4$
- Total quantum dimension: $D = \sqrt{1^2 + 1^2 + 1^2 + 1^2} = 2$

$$\boxed{S_{\text{topo}} = \log 2 \approx 0.693}$$

### 5. Kitaev-Preskill / Levin-Wen Construction

To extract the TEE from the entanglement structure, we use a clever cancellation scheme:

#### The Construction

Divide the system into regions A, B, C and compute:
$$S_{\text{topo}} = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}$$

This combination cancels the area law contributions, leaving only the topological part.

#### Alternative: Disk Geometry

For a disk-shaped region $A$ of radius $R$:
$$S(A) = \alpha \cdot 2\pi R - \log D + \mathcal{O}(1/R)$$

The constant term $-\log D$ is the TEE.

### 6. Robustness to Local Perturbations

#### Stability Theorem

**Theorem** (Bravyi-Hastings-Michalakis): Topological order is stable under arbitrary local perturbations below a finite threshold.

Specifically, if:
$$H = H_{\text{toric}} + \lambda V$$

where $V$ is a sum of local terms, then for $|\lambda| < \lambda_c$:
- Ground state degeneracy is preserved (up to exponentially small splitting)
- Anyon properties are unchanged
- The gap remains open

#### Why Robustness?

1. **No local order parameter**: Local perturbations can't change global topology
2. **Energy gap**: Excitations cost finite energy
3. **Anyonic protection**: Changing anyon type requires string operators (non-local)

#### Splitting of Ground State Degeneracy

Under perturbation, the 4-fold degeneracy splits by:
$$\Delta E \sim e^{-L/\xi}$$

where $L$ is the system size and $\xi$ is a correlation length. This splitting is exponentially small!

### 7. Topological Order vs Quantum Error Correction

#### Deep Connection

The properties of topological order map directly to error correction:

| Topological Order | Error Correction |
|-------------------|------------------|
| Ground state degeneracy | Code space dimension |
| Long-range entanglement | Non-local encoding |
| Anyonic excitations | Error syndromes |
| Robustness to perturbations | Fault tolerance |
| TEE = log D | Code distance growth |

#### The Toric Code as an Example

- **4 ground states** → 2 logical qubits
- **String operators** → Logical operations
- **Star/plaquette violations** → Error syndromes
- **Energy gap** → Error suppression

### 8. Detecting Topological Order

#### Experimental Signatures

1. **Ground state degeneracy**: Count distinct ground states on different topologies
2. **Braiding statistics**: Measure phase from anyon exchange
3. **Entanglement spectrum**: Look for characteristic structure
4. **Edge modes**: Detect gapless boundary excitations

#### Numerical Signatures

1. **TEE extraction**: Use Kitaev-Preskill or Levin-Wen
2. **Modular matrices**: Compute S and T from ground state overlaps
3. **Minimum entanglement entropy**: Check it matches $\log D$

---

## Quantum Computing Connection

### Topological Protection of Quantum Information

The deep principle behind topological quantum computing:

1. **Encode in ground state manifold**: Logical qubits live in degenerate ground states
2. **Gates by braiding**: Logical operations from anyon exchange
3. **Protection from noise**: Local errors create localized anyons, don't affect encoded info
4. **Measurement by fusion**: Read out by bringing anyons together

### Beyond the Toric Code

For universal topological QC, we need non-Abelian anyons:
- Fibonacci anyons: Universal by braiding alone
- Ising anyons: Need additional "magic" for universality
- Toric code: Abelian, but foundation for more complex codes

### Current Experimental Status

- **Google/IBM**: Surface code implementations approaching error threshold
- **Microsoft**: Pursuing Majorana-based topological qubits
- **Quantinuum**: Trapped-ion simulations of topological states

---

## Worked Examples

### Example 1: Counting Ground States on a Double Torus

**Problem**: How many ground states does the toric code have on a surface of genus 2?

**Solution**:

A genus-2 surface (double torus) has:
- 4 independent non-contractible loops (2 per handle)
- Each pair supports a logical qubit
- Total: 2 logical qubits per handle × 2 handles = 4 logical qubits

Wait, let me reconsider.

Each handle (genus 1 piece) contributes:
- 2 non-contractible cycles (meridian and longitude)
- These support 2 logical operators $\bar{Z}, \bar{X}$
- Together encoding 1 logical qubit

For genus $g$:
- $2g$ cycles total
- $2g$ logical operators
- But they come in conjugate pairs → $g$ logical qubits...

Actually, for the toric code specifically:
$$\text{GSD} = |\mathcal{A}|^g = 4^g$$

For genus 2: $\text{GSD} = 4^2 = 16$.

$$\boxed{\text{GSD on genus-2 surface} = 16}$$

### Example 2: Computing TEE from Entanglement

**Problem**: A disk region $A$ of radius $R = 10$ (in lattice units) in the toric code ground state has entanglement entropy $S(A) = 62.1$ bits. What is the TEE?

**Solution**:

The area law:
$$S(A) = \alpha \cdot 2\pi R - S_{\text{topo}}$$

We need another measurement to extract $\alpha$. Suppose for $R = 20$:
$$S(A') = 124.9 \text{ bits}$$

Then:
$$S(A') - S(A) = \alpha \cdot 2\pi (20 - 10) = 124.9 - 62.1 = 62.8$$
$$\alpha = \frac{62.8}{20\pi} \approx 1.0$$

Now:
$$S_{\text{topo}} = \alpha \cdot 2\pi R - S(A) = 1.0 \times 2\pi \times 10 - 62.1 = 62.83 - 62.1 = 0.73$$

$$\boxed{S_{\text{topo}} \approx 0.73 \approx \log 2}$$

This matches the expected value for the toric code!

### Example 3: Ground State Splitting

**Problem**: A 10×10 toric code is subject to a perturbation of strength $\lambda = 0.01$. If the correlation length is $\xi = 2$ lattice spacings, estimate the ground state splitting.

**Solution**:

The splitting goes as:
$$\Delta E \sim \lambda^k e^{-L/\xi}$$

where $k$ depends on the perturbation order and $L$ is the smallest linear dimension.

For $L = 10$, $\xi = 2$:
$$e^{-L/\xi} = e^{-10/2} = e^{-5} \approx 0.0067$$

The splitting is suppressed by a factor of roughly 150 from the exponential alone.

If the perturbation enters at first order with coefficient ~1:
$$\Delta E \sim 0.01 \times 0.0067 \approx 6.7 \times 10^{-5}$$

$$\boxed{\Delta E \sim 10^{-4} \text{ (in units of the gap)}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Genus-3 Surface
What is the ground state degeneracy of the toric code on a genus-3 surface?

**Problem 2**: TEE for $\mathbb{Z}_N$ Gauge Theory
The $\mathbb{Z}_N$ toric code has $N^2$ anyon types, all with quantum dimension 1. What is its TEE?

**Problem 3**: Scaling of Splitting
If the system size doubles (L → 2L), by what factor does the ground state splitting decrease?

### Intermediate

**Problem 4**: Non-Abelian TEE
For Fibonacci anyons with quantum dimensions $d_1 = 1$ and $d_\tau = \phi$ (golden ratio), compute the total quantum dimension $D$ and TEE.

**Problem 5**: Kitaev-Preskill Construction
Draw the region configuration for the Kitaev-Preskill TEE extraction. Explain why the area law terms cancel.

**Problem 6**: Correlation Length
In a perturbed toric code, the correlation length diverges as $\xi \sim 1/\sqrt{\lambda_c - \lambda}$ near the phase transition. Estimate $\lambda_c$ if $\xi = 10$ at $\lambda = 0.05$.

### Challenging

**Problem 7**: Modular Matrices
The modular T-matrix gives the topological spin: $T_{ab} = \delta_{ab} \theta_a$. Write down the T-matrix for the toric code and verify $T S T = S T^{-1} S$ (modular relation).

**Problem 8**: Beyond Area Law
In 1D, the area law is just a constant (zero-dimensional boundary). Why doesn't 1D support topological order in gapped phases?

---

## Computational Lab: Topological Order Calculations

```python
"""
Day 797 Computational Lab: Topological Order
Computing entanglement entropy and topological invariants
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
from itertools import product

class TopologicalOrderAnalysis:
    """
    Tools for analyzing topological order in the toric code
    """

    def __init__(self, L):
        """Initialize L x L toric code"""
        self.L = L
        self.n_qubits = 2 * L * L

    def compute_gsd(self, genus):
        """Ground state degeneracy for surface of given genus"""
        return 4 ** genus

    def compute_total_quantum_dimension(self, quantum_dims):
        """
        Compute total quantum dimension D = sqrt(sum d_a^2)
        quantum_dims: list of quantum dimensions [d_1, d_e, d_m, d_eps]
        """
        return np.sqrt(sum(d**2 for d in quantum_dims))

    def compute_tee(self, D):
        """Topological entanglement entropy"""
        return np.log(D)


def visualize_gsd_vs_genus():
    """Plot ground state degeneracy vs genus"""
    fig, ax = plt.subplots(figsize=(10, 6))

    genus = np.arange(0, 6)
    gsd = 4 ** genus

    ax.bar(genus, gsd, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Genus g', fontsize=14)
    ax.set_ylabel('Ground State Degeneracy', fontsize=14)
    ax.set_title('Toric Code: GSD = 4^g', fontsize=16)
    ax.set_xticks(genus)

    # Add labels on bars
    for g, d in zip(genus, gsd):
        ax.annotate(f'{d}', (g, d), ha='center', va='bottom', fontsize=12)

    # Add log scale inset
    ax_inset = ax.inset_axes([0.6, 0.5, 0.35, 0.4])
    ax_inset.semilogy(genus, gsd, 'bo-', markersize=8)
    ax_inset.set_xlabel('g', fontsize=10)
    ax_inset.set_ylabel('GSD (log)', fontsize=10)
    ax_inset.set_title('Log scale', fontsize=10)
    ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gsd_vs_genus.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_tee():
    """Visualize topological entanglement entropy"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Area law with TEE
    ax = axes[0]

    perimeters = np.linspace(10, 100, 50)
    alpha = 1.0  # Coefficient
    tee = np.log(2)  # For toric code

    S = alpha * perimeters - tee

    ax.plot(perimeters, S, 'b-', linewidth=2, label='S(A)')
    ax.plot(perimeters, alpha * perimeters, 'r--', linewidth=2, label='Area law (no TEE)')
    ax.fill_between(perimeters, S, alpha * perimeters, alpha=0.3, color='green',
                    label=f'TEE = log(2) ≈ {tee:.3f}')

    ax.set_xlabel('Perimeter |∂A|', fontsize=14)
    ax.set_ylabel('Entanglement Entropy S(A)', fontsize=14)
    ax.set_title('Area Law with Topological Correction', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: TEE for different theories
    ax = axes[1]

    theories = ['Trivial', 'Toric Code\n(Z₂)', 'Z₃ Gauge', 'Fibonacci', 'Ising']
    D_values = [1, 2, 3, (1 + np.sqrt(5))/2 + 1, np.sqrt(2) + 1]  # Approximate
    D_values = [1, 2, 3, 1 + (1+np.sqrt(5))/2, 1 + np.sqrt(2)]

    # Recalculate properly
    D_toric = 2
    D_z3 = 3
    D_fib = np.sqrt(1 + ((1+np.sqrt(5))/2)**2)  # 1 and tau
    D_ising = np.sqrt(1 + 1 + 2)  # 1, psi, sigma

    D_values = [1, D_toric, D_z3, D_fib, D_ising]
    tee_values = [np.log(D) if D > 1 else 0 for D in D_values]

    colors = ['gray', 'blue', 'green', 'purple', 'orange']
    bars = ax.bar(theories, tee_values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('TEE = log(D)', fontsize=14)
    ax.set_title('Topological Entanglement Entropy\nfor Different Topological Phases', fontsize=16)

    # Add D values as labels
    for bar, D in zip(bars, D_values):
        height = bar.get_height()
        ax.annotate(f'D={D:.2f}', (bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('tee_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def simulate_kitaev_preskill():
    """Illustrate the Kitaev-Preskill construction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: The region configuration
    ax = axes[0]

    # Draw three regions A, B, C meeting at a point
    theta_A = np.linspace(0, 2*np.pi/3, 50)
    theta_B = np.linspace(2*np.pi/3, 4*np.pi/3, 50)
    theta_C = np.linspace(4*np.pi/3, 2*np.pi, 50)

    R = 1.0
    r = 0.3

    for theta, color, label in [(theta_A, 'red', 'A'),
                                  (theta_B, 'blue', 'B'),
                                  (theta_C, 'green', 'C')]:
        x_outer = R * np.cos(theta)
        y_outer = R * np.sin(theta)
        x_inner = r * np.cos(theta)
        y_inner = r * np.sin(theta)

        # Fill the region
        verts = list(zip(np.concatenate([x_outer, x_inner[::-1]]),
                        np.concatenate([y_outer, y_inner[::-1]])))
        from matplotlib.patches import Polygon
        poly = Polygon(verts, facecolor=color, alpha=0.5, edgecolor=color, linewidth=2)
        ax.add_patch(poly)

        # Label
        mid_theta = (theta[0] + theta[-1]) / 2
        ax.text(0.65 * np.cos(mid_theta), 0.65 * np.sin(mid_theta), label,
               fontsize=16, fontweight='bold', ha='center', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Kitaev-Preskill Regions', fontsize=14)
    ax.axis('off')

    # Panel 2: Entropies for each region
    ax = axes[1]

    # Hypothetical values
    alpha = 1.0  # Boundary coefficient
    gamma = np.log(2)  # TEE

    # Perimeters (illustrative)
    L_ext = 2.0  # External boundary length per region
    L_int = 1.0  # Internal boundary length between regions

    S_A = alpha * (L_ext + L_int) - gamma/3
    S_B = alpha * (L_ext + L_int) - gamma/3
    S_C = alpha * (L_ext + L_int) - gamma/3
    S_AB = alpha * (2*L_ext + L_int) - gamma/3
    S_BC = alpha * (2*L_ext + L_int) - gamma/3
    S_AC = alpha * (2*L_ext + L_int) - gamma/3
    S_ABC = alpha * 3*L_ext - gamma

    regions = ['A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC']
    entropies = [S_A, S_B, S_C, S_AB, S_BC, S_AC, S_ABC]

    ax.bar(regions, entropies, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Entanglement Entropy', fontsize=12)
    ax.set_title('Entropies of Regions', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 3: The combination
    ax = axes[2]

    combination = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    tee = combination

    ax.text(0.5, 0.8, 'Kitaev-Preskill Formula:', fontsize=14, ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.6, r'$S_{topo} = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}$',
            fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'= {S_A:.2f} + {S_B:.2f} + {S_C:.2f}', fontsize=11, ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.3, f'  - {S_AB:.2f} - {S_BC:.2f} - {S_AC:.2f}', fontsize=11, ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.2, f'  + {S_ABC:.2f}', fontsize=11, ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.05, f'= {tee:.3f} ≈ log(2) = {np.log(2):.3f}', fontsize=14,
            ha='center', transform=ax.transAxes, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.axis('off')
    ax.set_title('TEE Extraction', fontsize=14)

    plt.tight_layout()
    plt.savefig('kitaev_preskill.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_robustness():
    """Visualize robustness of topological order"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Ground state splitting vs system size
    ax = axes[0]

    L_values = np.arange(4, 25)
    xi = 2  # Correlation length
    lambda_pert = 0.01

    splitting = lambda_pert * np.exp(-L_values / xi)

    ax.semilogy(L_values, splitting, 'b-', linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('System Size L', fontsize=14)
    ax.set_ylabel('Ground State Splitting ΔE', fontsize=14)
    ax.set_title('Exponential Suppression of Splitting\n$\\Delta E \\sim \\lambda e^{-L/\\xi}$', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(f'ξ = {xi}', (15, splitting[11]), fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Panel 2: Phase diagram
    ax = axes[1]

    lambda_vals = np.linspace(0, 0.15, 100)
    lambda_c = 0.1

    # Order parameter (schematic)
    order = np.zeros_like(lambda_vals)
    order[lambda_vals < lambda_c] = 1

    # Gap (schematic)
    gap = np.ones_like(lambda_vals)
    gap[lambda_vals > lambda_c] = 0

    ax.fill_between(lambda_vals[lambda_vals < lambda_c], 0, 1,
                    alpha=0.3, color='blue', label='Topological Phase')
    ax.fill_between(lambda_vals[lambda_vals >= lambda_c], 0, 1,
                    alpha=0.3, color='red', label='Trivial Phase')
    ax.axvline(x=lambda_c, color='black', linestyle='--', linewidth=2,
               label=f'λ_c = {lambda_c}')

    ax.set_xlabel('Perturbation Strength λ', fontsize=14)
    ax.set_ylabel('', fontsize=14)
    ax.set_title('Phase Diagram\n(Topological Order is Stable for λ < λ_c)', fontsize=14)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')

    # Add labels
    ax.text(0.05, 0.5, 'Topological\nOrder', fontsize=14, ha='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.85, 0.5, 'Trivial', fontsize=14, ha='center',
            transform=ax.transAxes, fontweight='bold')

    plt.tight_layout()
    plt.savefig('robustness.png', dpi=150, bbox_inches='tight')
    plt.show()


def compute_modular_matrices():
    """Compute and verify modular matrices for toric code"""
    print("=" * 60)
    print("Modular Matrices for Toric Code")
    print("=" * 60)

    # S-matrix
    S = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ]) / 2

    # T-matrix (topological spins on diagonal)
    topological_spins = [1, 1, 1, -1]  # 1, e, m, epsilon
    T = np.diag(topological_spins)

    print("\nS-matrix:")
    print(S)

    print("\nT-matrix:")
    print(T)

    # Verify modular relations
    print("\nVerifying modular relations:")

    # S^2 = C (charge conjugation, identity for toric code)
    S2 = S @ S
    print(f"\nS² (should be identity):")
    print(np.round(S2, 10))
    print(f"S² = I: {np.allclose(S2, np.eye(4))}")

    # (ST)^3 = exp(i*pi*c/4) * C, where c is central charge (c=0 for toric code)
    ST = S @ T
    ST3 = ST @ ST @ ST
    print(f"\n(ST)³ (should be identity for c=0):")
    print(np.round(ST3, 10))

    # T S T = S T^(-1) S  (alternative form of modular relation)
    T_inv = np.diag(1/np.array(topological_spins))
    TST = T @ S @ T
    ST_inv_S = S @ T_inv @ S
    print(f"\nT S T:")
    print(np.round(TST, 10))
    print(f"\nS T⁻¹ S:")
    print(np.round(ST_inv_S, 10))

    return S, T


def compare_orderings():
    """Compare topological order with symmetry breaking"""
    print("\n" + "=" * 60)
    print("Comparison: Topological Order vs Symmetry Breaking")
    print("=" * 60)

    comparison = """
    Property                | Symmetry Breaking        | Topological Order
    ========================|==========================|=========================
    Local order parameter   | Yes (e.g., magnetization)| No
    Ground state degeneracy | From broken symmetry     | From topology (4^g)
    Excitations             | Gapless (Goldstone)      | Gapped (anyons)
    Robustness              | To symmetric perturb.    | To ALL local perturb.
    Example                 | Ising ferromagnet        | Toric code
    Entanglement            | Short-range              | Long-range
    Detectable by           | Local measurements       | Non-local (Wilson loops)
    """
    print(comparison)


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Day 797: Topological Order - Computational Lab")
    print("=" * 60)

    print("\n1. Ground state degeneracy vs genus...")
    visualize_gsd_vs_genus()

    print("\n2. Topological entanglement entropy...")
    visualize_tee()

    print("\n3. Kitaev-Preskill construction...")
    simulate_kitaev_preskill()

    print("\n4. Robustness to perturbations...")
    visualize_robustness()

    print("\n5. Modular matrices...")
    S, T = compute_modular_matrices()

    print("\n6. Comparison of orderings...")
    compare_orderings()

    print("\n" + "=" * 60)
    print("Key Insights from Lab:")
    print("=" * 60)
    print("1. GSD = 4^g depends only on topology, not local details")
    print("2. TEE = log(D) = log(2) for toric code")
    print("3. Kitaev-Preskill cancels area law, extracts TEE")
    print("4. Ground state splitting ~ exp(-L/ξ) is exponentially small")
    print("5. Modular matrices encode all topological data")
    print("6. Topological order ≠ symmetry breaking order")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Ground state degeneracy (genus g) | $\text{GSD} = 4^g$ |
| Total quantum dimension | $D = \sqrt{\sum_a d_a^2}$ |
| TEE (toric code) | $S_{\text{topo}} = \log 2$ |
| Area law | $S(A) = \alpha \|\partial A\| - S_{\text{topo}}$ |
| Ground state splitting | $\Delta E \sim e^{-L/\xi}$ |
| Kitaev-Preskill | $S_{\text{topo}} = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}$ |

### Main Takeaways

1. **Long-range entanglement**: Topological states cannot be created by local operations
2. **GSD from topology**: Ground state degeneracy reveals the genus of the surface
3. **TEE as fingerprint**: The constant $-\log D$ in entanglement identifies the phase
4. **Robustness**: Topological order survives arbitrary local perturbations below threshold
5. **No local order parameter**: Cannot detect topological order locally
6. **Foundation for QEC**: Topological protection = fault-tolerant quantum memory

---

## Daily Checklist

### Morning Theory (3 hours)
- [ ] Understand long-range entanglement
- [ ] Derive GSD = 4^g formula
- [ ] Master the area law and TEE
- [ ] Study robustness to perturbations

### Afternoon Problems (2.5 hours)
- [ ] Complete all Direct Application problems
- [ ] Work through at least 2 Intermediate problems
- [ ] Attempt at least 1 Challenging problem

### Evening Lab (1.5 hours)
- [ ] Run all visualization code
- [ ] Verify modular matrix properties
- [ ] Explore the Kitaev-Preskill construction

### Self-Assessment Questions
1. Why is the ground state degeneracy on a sphere equal to 1?
2. How does TEE distinguish topological phases from trivial ones?
3. What happens to topological order at finite temperature?

---

## Preview: Day 798

Tomorrow we synthesize the entire week's material on **Anyons & Topological Order**. We'll create comprehensive classification tables, connect all concepts to topological quantum computing, and preview the surface code boundaries that enable practical implementations.

---

*Day 797 of 2184 | Year 2, Month 29, Week 114 | Quantum Engineering PhD Curriculum*
