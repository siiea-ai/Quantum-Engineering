# Day 426: 3D Isotropic Harmonic Oscillator

## Overview
**Day 426** | Year 1, Month 16, Week 61 | The Most Important Model in Physics

Today we solve the 3D isotropic harmonic oscillator — perhaps the most important model in all of physics, appearing in molecular vibrations, quantum optics, and as the foundation of quantum field theory.

---

## Learning Objectives

By the end of today, you will be able to:
1. Solve the 3D harmonic oscillator using separation in spherical coordinates
2. Derive the energy spectrum E = ℏω(n + 3/2)
3. Calculate degeneracies using combinatorics
4. Understand the connection to Cartesian separation
5. Recognize the hidden SU(3) symmetry
6. Apply to molecular vibrations and trapped atoms

---

## Core Content

### The Potential

$$V(r) = \frac{1}{2}m\omega^2 r^2 = \frac{1}{2}m\omega^2(x^2 + y^2 + z^2)$$

This is isotropic (spherically symmetric).

### Two Approaches

**1. Cartesian separation:**
$$H = H_x + H_y + H_z$$

where each H_i is a 1D harmonic oscillator.

**2. Spherical separation:**
$$\psi(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$

### Cartesian Solution

Three independent 1D oscillators:
$$E = \hbar\omega\left(n_x + n_y + n_z + \frac{3}{2}\right) = \hbar\omega\left(N + \frac{3}{2}\right)$$

where N = n_x + n_y + n_z = 0, 1, 2, ...

### Energy Spectrum

$$\boxed{E_N = \hbar\omega\left(N + \frac{3}{2}\right)}$$

Ground state: E₀ = (3/2)ℏω (zero-point energy from three dimensions)

### Degeneracy

For total quantum number N, how many states exist?

Need to count non-negative integer solutions to:
$$n_x + n_y + n_z = N$$

$$\boxed{g_N = \frac{(N+1)(N+2)}{2}}$$

| N | g_N | States |
|---|-----|--------|
| 0 | 1 | (0,0,0) |
| 1 | 3 | (1,0,0), (0,1,0), (0,0,1) |
| 2 | 6 | (2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1) |
| 3 | 10 | ... |

### Spherical Solution: Radial Equation

Using u = rR:
$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + \left[\frac{1}{2}m\omega^2 r^2 + \frac{\hbar^2 l(l+1)}{2mr^2}\right]u = Eu$$

### Radial Wavefunctions

Define ξ = (mω/ℏ)^{1/2} r and look for solutions:
$$R_{nl}(r) = N_{nl} \xi^l e^{-\xi^2/2} L_{n_r}^{l+1/2}(\xi^2)$$

where L are associated Laguerre polynomials and n_r = (N - l)/2.

### Relation Between Quantum Numbers

$$N = 2n_r + l$$

where:
- N = principal quantum number (0, 1, 2, ...)
- l = angular momentum (N, N-2, N-4, ..., 0 or 1)
- n_r = radial quantum number (0, 1, 2, ...)

### Angular Momentum Content

For given N, allowed l values:
$$l = N, N-2, N-4, \ldots, \begin{cases} 0 & N \text{ even} \\ 1 & N \text{ odd} \end{cases}$$

Example for N = 4: l = 4, 2, 0

### Verifying Degeneracy

For N = 2:
- l = 2: m = -2, -1, 0, 1, 2 → 5 states
- l = 0: m = 0 → 1 state
- Total: 6 states = g₂ ✓

### Hidden Symmetry: SU(3)

The extra degeneracy (beyond SO(3) rotation symmetry) comes from:
- The Runge-Lenz-like operators
- SU(3) symmetry of the isotropic oscillator
- This is why l values with same N are degenerate

### Creation/Annihilation Operators

In 3D, we have three sets:
$$\hat{a}_i = \sqrt{\frac{m\omega}{2\hbar}}\hat{x}_i + i\frac{\hat{p}_i}{\sqrt{2m\hbar\omega}}, \quad i = x, y, z$$

Number operators: N̂_i = â†_i â_i

$$\hat{H} = \hbar\omega\left(\hat{N}_x + \hat{N}_y + \hat{N}_z + \frac{3}{2}\right)$$

---

## Quantum Computing Connection

### Bosonic Modes

The harmonic oscillator is the quantum description of:
- **Electromagnetic field modes** (photons)
- **Phonons** in solids
- **Mechanical oscillators** in optomechanics

### Continuous Variable QC

In CV quantum computing:
- Qumodes are harmonic oscillator states
- Cat states: superpositions of coherent states
- GKP states: grid states for error correction

### Trapped Ions

Ion trap qubits use:
- Electronic states for qubit
- Motional states (3D oscillator) for bus

---

## Worked Examples

### Example 1: First Excited State

**Problem:** What are the l values for N = 1, and how many total states?

**Solution:**
For N = 1: l = N - 2k where N - 2k ≥ 0 and same parity as N.

l = 1 only (l = -1 not allowed).

For l = 1: m = -1, 0, +1 → 3 states.

Verify: g₁ = (1+1)(1+2)/2 = 3 ✓

In Cartesian: (1,0,0), (0,1,0), (0,0,1) — same 3 states!

### Example 2: Zero-Point Energy

**Problem:** Calculate the zero-point energy for a molecule with ω = 10¹⁴ rad/s.

**Solution:**
$$E_0 = \frac{3}{2}\hbar\omega = \frac{3}{2}(1.055 \times 10^{-34})(10^{14})$$
$$= 1.58 \times 10^{-20} \text{ J} = 0.099 \text{ eV} \approx 0.1 \text{ eV}$$

This is the irreducible energy of a 3D oscillator at T = 0.

### Example 3: Angular Momentum Selection

**Problem:** A particle is in state |N=4, l=2, m=1⟩. What states can it transition to via dipole radiation (Δl = ±1)?

**Solution:**
Dipole: l' = l ± 1 = 1 or 3

For l' = 1: N can be 1, 3, 5, ...
For l' = 3: N can be 3, 5, 7, ...

If staying in same N = 4 shell: neither l = 1 nor l = 3 appears (only l = 4, 2, 0).

So dipole transitions must change N!

---

## Practice Problems

### Direct Application
1. What is the degeneracy for N = 5?
2. List all (l, m) quantum numbers for N = 3.
3. What is the energy gap between N = 0 and N = 1?

### Intermediate
4. Show that g_N = (N+1)(N+2)/2 using the "stars and bars" counting method.
5. Verify that the sum of degeneracies: g_0 + g_1 + ... = n³/6 for large n.
6. Calculate ⟨r²⟩ for the ground state.

### Challenging
7. Show that the accidental degeneracy comes from conserved operators beyond L̂.
8. Derive the radial equation from the full Schrödinger equation.

---

## Computational Lab

```python
"""
Day 426: 3D Isotropic Harmonic Oscillator
Energy levels, degeneracies, and wavefunctions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial
from mpl_toolkits.mplot3d import Axes3D

# Degeneracy calculation
def degeneracy(N):
    """Calculate degeneracy for principal quantum number N"""
    return (N + 1) * (N + 2) // 2

# Energy levels and degeneracies
print("=== 3D Harmonic Oscillator Energy Levels ===")
print("\n N    E/(ℏω)    g_N    Angular momentum content")
print("-" * 55)

for N in range(8):
    E = N + 1.5
    g = degeneracy(N)

    # Find allowed l values
    l_values = []
    for l in range(N, -1, -2):
        if l >= 0:
            l_values.append(l)

    l_str = ', '.join([f'l={l}' for l in l_values])
    print(f" {N}    {E:5.1f}      {g:3d}    {l_str}")

# Verify degeneracy
print("\nVerification of degeneracy formula:")
for N in range(6):
    # Count states by summing (2l+1) over allowed l
    total = sum(2*l + 1 for l in range(N, -1, -2))
    formula = (N + 1) * (N + 2) // 2
    print(f"N = {N}: Σ(2l+1) = {total}, formula = {formula}")

# Plot energy level diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel 1: Energy levels with angular momentum structure
ax = axes[0]

colors = {'s': 'blue', 'p': 'green', 'd': 'red', 'f': 'purple', 'g': 'orange', 'h': 'brown'}
l_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}

for N in range(7):
    E = N + 1.5
    x_offset = 0
    for l in range(N, -1, -2):
        if l >= 0:
            multiplicity = 2 * l + 1
            l_name = l_names.get(l, f'l={l}')
            color = colors.get(l_name, 'black')

            # Draw level
            ax.hlines(E, x_offset, x_offset + 0.8, colors=color, linewidth=3)
            ax.text(x_offset + 0.4, E + 0.15, f'{l_name}({multiplicity})',
                    ha='center', fontsize=9, color=color)
            x_offset += 1.2

    # Label total
    g = degeneracy(N)
    ax.text(-0.5, E, f'N={N}', ha='right', va='center', fontsize=10)
    ax.text(x_offset + 0.3, E, f'g={g}', ha='left', va='center', fontsize=10)

ax.set_xlabel('Angular momentum states', fontsize=12)
ax.set_ylabel('E / ℏω', fontsize=12)
ax.set_title('3D Harmonic Oscillator Energy Levels', fontsize=14)
ax.set_xlim(-1, 8)
ax.set_ylim(1, 8.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Degeneracy growth
ax = axes[1]

N_range = np.arange(0, 15)
g_values = [(N+1)*(N+2)//2 for N in N_range]

ax.bar(N_range, g_values, color='steelblue', edgecolor='black')
ax.plot(N_range, N_range**2 / 2, 'r--', linewidth=2, label='N²/2 (asymptotic)')
ax.set_xlabel('Principal quantum number N', fontsize=12)
ax.set_ylabel('Degeneracy g_N', fontsize=12)
ax.set_title('Degeneracy vs Quantum Number', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day426_3D_oscillator_levels.png', dpi=150)
plt.show()

# Radial wavefunctions
def radial_wavefunction(n_r, l, r, mw_hbar=1.0):
    """
    Radial wavefunction R_{n_r,l}(r) for 3D harmonic oscillator
    n_r = radial quantum number, l = angular momentum
    mw_hbar = mω/ℏ (sets length scale)
    """
    xi = np.sqrt(mw_hbar) * r
    xi2 = xi**2

    # Associated Laguerre polynomial L_{n_r}^{l+1/2}(xi^2)
    L = genlaguerre(n_r, l + 0.5)(xi2)

    # Normalization (simplified)
    norm = np.sqrt(2 * factorial(n_r) / factorial(n_r + l + 0.5))
    norm *= (mw_hbar)**(3/4)

    R = norm * xi**l * np.exp(-xi2/2) * L

    return R

# Plot radial wavefunctions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

r = np.linspace(0, 5, 500)

# Ground state (N=0, l=0)
ax = axes[0, 0]
R_00 = radial_wavefunction(0, 0, r)
R_00 = R_00 / np.max(np.abs(R_00))  # Normalize for display

ax.plot(r, R_00, 'b-', linewidth=2, label='R₀₀(r)')
ax.fill_between(r, 0, R_00**2 * r**2 * 5, alpha=0.3, label='P(r) scaled')
ax.set_xlabel('r (oscillator units)', fontsize=12)
ax.set_ylabel('R(r)', fontsize=12)
ax.set_title('Ground State (N=0, l=0)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# N=1 (l=1)
ax = axes[0, 1]
R_01 = radial_wavefunction(0, 1, r)
R_01 = R_01 / np.max(np.abs(R_01))

ax.plot(r, R_01, 'g-', linewidth=2, label='R₀₁(r)')
ax.fill_between(r, 0, R_01**2 * r**2 * 5, alpha=0.3, label='P(r) scaled')
ax.set_xlabel('r (oscillator units)', fontsize=12)
ax.set_ylabel('R(r)', fontsize=12)
ax.set_title('N=1, l=1 (p-wave)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# N=2 states
ax = axes[1, 0]
R_10 = radial_wavefunction(1, 0, r)  # n_r=1, l=0
R_10 = R_10 / np.max(np.abs(R_10))

R_02 = radial_wavefunction(0, 2, r)  # n_r=0, l=2
R_02 = R_02 / np.max(np.abs(R_02))

ax.plot(r, R_10, 'r-', linewidth=2, label='n_r=1, l=0 (2s-like)')
ax.plot(r, R_02, 'b-', linewidth=2, label='n_r=0, l=2 (1d-like)')
ax.set_xlabel('r (oscillator units)', fontsize=12)
ax.set_ylabel('R(r)', fontsize=12)
ax.set_title('N=2 States (same energy, different l)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Comparison of all low-lying states
ax = axes[1, 1]

states = [
    (0, 0, 'N=0, l=0', 'blue'),
    (0, 1, 'N=1, l=1', 'green'),
    (1, 0, 'N=2, l=0', 'red'),
    (0, 2, 'N=2, l=2', 'orange'),
    (0, 3, 'N=3, l=3', 'purple'),
]

for n_r, l, label, color in states:
    R = radial_wavefunction(n_r, l, r)
    R = R / np.max(np.abs(R))
    P = R**2 * r**2
    P = P / np.max(P)
    ax.plot(r, P, color=color, linewidth=2, label=label)

ax.set_xlabel('r (oscillator units)', fontsize=12)
ax.set_ylabel('P(r) normalized', fontsize=12)
ax.set_title('Radial Probability Densities', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day426_radial_wavefunctions.png', dpi=150)
plt.show()

# Applications
print("\n=== Physical Applications ===")

print("\n1. Molecular Vibrations:")
print("   Diatomic molecules have 3D center-of-mass motion")
print("   Internal vibration: 1D oscillator")
print("   Total: 3 + 1 = 4 oscillator modes")

print("\n2. Trapped Atoms:")
print("   Optical traps create harmonic confinement")
print("   Typical frequencies: 10-100 kHz")
print("   Used in quantum simulation and computing")

print("\n3. Quantum Field Theory:")
print("   Each mode of EM field is a harmonic oscillator")
print("   Photon number states |n⟩ = excited states")
print("   Coherent states: displaced ground states")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Potential | V(r) = ½mω²r² |
| Energy | E_N = ℏω(N + 3/2) |
| Degeneracy | g_N = (N+1)(N+2)/2 |
| Allowed l | l = N, N-2, ..., 0 or 1 |
| Relation | N = 2n_r + l |
| Zero-point | E₀ = (3/2)ℏω |

### Key Insights

1. **Two approaches** (Cartesian/spherical) give same energies
2. **Extra degeneracy** beyond rotation symmetry (SU(3))
3. **Angular momentum mixing** — different l at same energy
4. **Creation operators** provide algebraic solution
5. **Universal model** for oscillatory systems

---

## Daily Checklist

- [ ] I can derive the 3D oscillator energy spectrum
- [ ] I understand the degeneracy formula
- [ ] I can list allowed (l, m) for given N
- [ ] I know the relation N = 2n_r + l
- [ ] I can connect Cartesian and spherical solutions
- [ ] I see applications to molecules and trapped atoms

---

## Preview: Day 427

Tomorrow we consolidate the week with a comprehensive review of **central potential methods**, preparing for the hydrogen atom.

---

**Next:** [Day_427_Sunday.md](Day_427_Sunday.md) — Week 61 Review
