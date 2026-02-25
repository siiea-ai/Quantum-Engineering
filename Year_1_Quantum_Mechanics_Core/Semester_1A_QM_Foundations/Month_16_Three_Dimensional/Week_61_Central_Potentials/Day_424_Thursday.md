# Day 424: Infinite Spherical Well

## Overview
**Day 424** | Year 1, Month 16, Week 61 | Particle in a 3D Box (Spherical)

Today we solve the infinite spherical well — a particle confined within radius a. This simple model illustrates how boundary conditions quantize energy in three dimensions.

---

## Learning Objectives

By the end of today, you will be able to:
1. Set up the boundary value problem for the spherical well
2. Apply the boundary condition R(a) = 0
3. Find energy eigenvalues from zeros of Bessel functions
4. Calculate degeneracies for each energy level
5. Compare with the 3D rectangular box
6. Understand the nuclear shell model connection

---

## Core Content

### The Infinite Spherical Well Potential

$$V(r) = \begin{cases} 0 & r < a \\ \infty & r \geq a \end{cases}$$

**Boundary condition:** ψ(a, θ, φ) = 0 for all angles.

Since ψ = R(r)Y_l^m(θ,φ), we need:
$$R(a) = 0$$

### Interior Solution (r < a)

Inside the well, V = 0, so we have the free-particle radial equation:
$$R_{nl}(r) = A_l j_l(kr)$$

where k = √(2mE)/ℏ.

We use only j_l (not n_l) because n_l diverges at the origin.

### Quantization Condition

The boundary condition:
$$\boxed{j_l(ka) = 0}$$

Let x_{nl} denote the n-th zero of j_l(x), then:
$$k_{nl} a = x_{nl}$$
$$k_{nl} = \frac{x_{nl}}{a}$$

### Energy Eigenvalues

$$\boxed{E_{nl} = \frac{\hbar^2 x_{nl}^2}{2ma^2}}$$

The energy depends on both n (radial quantum number) and l (angular momentum).

### Zeros of Spherical Bessel Functions

| l | n=1 | n=2 | n=3 | n=4 |
|---|-----|-----|-----|-----|
| 0 | 3.142 (π) | 6.283 (2π) | 9.425 (3π) | 12.566 (4π) |
| 1 | 4.493 | 7.725 | 10.904 | 14.066 |
| 2 | 5.763 | 9.095 | 12.323 | 15.515 |
| 3 | 6.988 | 10.417 | 13.698 | 16.924 |

Note: For l = 0, zeros are at x = nπ (since j₀(x) = sin(x)/x).

### Energy Level Ordering

Ordering levels by increasing energy:

| Level | (n, l) | x_{nl} | Degeneracy |
|-------|--------|--------|------------|
| 1 | (1, 0) | π ≈ 3.14 | 1 |
| 2 | (1, 1) | 4.49 | 3 |
| 3 | (1, 2) | 5.76 | 5 |
| 4 | (2, 0) | 2π ≈ 6.28 | 1 |
| 5 | (1, 3) | 6.99 | 7 |
| 6 | (2, 1) | 7.73 | 3 |

### Degeneracy

For each (n, l), there are (2l + 1) states (different m values):
$$g_{nl} = 2l + 1$$

Including spin (for fermions):
$$g_{nl}^{\text{spin}} = 2(2l + 1)$$

### Wavefunctions

$$\psi_{nlm}(r, \theta, \phi) = A_{nl} j_l\left(\frac{x_{nl} r}{a}\right) Y_l^m(\theta, \phi)$$

Normalization:
$$\int_0^a |R_{nl}|^2 r^2 dr = 1$$

This gives:
$$A_{nl} = \sqrt{\frac{2}{a^3}} \frac{1}{j_{l+1}(x_{nl})}$$

### Comparison with Rectangular Box

**Rectangular box (side a):**
$$E_{n_x n_y n_z} = \frac{\hbar^2 \pi^2}{2ma^2}(n_x^2 + n_y^2 + n_z^2)$$

Different geometry → different level ordering and degeneracies!

---

## Quantum Computing Connection

### Nuclear Shell Model

The infinite spherical well is a zeroth approximation to nuclear physics:
- Nucleons confined within nuclear radius
- Magic numbers (2, 8, 20, 28, 50, 82, 126) correspond to filled shells
- Adding spin-orbit coupling improves the model

### Quantum Dots

Semiconductor quantum dots approximate spherical wells:
- Artificial atoms with controllable properties
- Used as qubits in some quantum computing architectures
- Level structure similar to this model

---

## Worked Examples

### Example 1: Ground State Energy

**Problem:** Find the ground state energy for an electron in a spherical well of radius a = 1 nm.

**Solution:**
Ground state: (n, l) = (1, 0), x₁₀ = π

$$E_{10} = \frac{\hbar^2 \pi^2}{2m_e a^2}$$

With ℏ = 1.055 × 10⁻³⁴ J·s, m_e = 9.11 × 10⁻³¹ kg, a = 10⁻⁹ m:

$$E_{10} = \frac{(1.055 \times 10^{-34})^2 \pi^2}{2(9.11 \times 10^{-31})(10^{-9})^2}$$
$$= \frac{1.097 \times 10^{-67}}{1.822 \times 10^{-48}} \approx 6.02 \times 10^{-20} \text{ J}$$
$$= 0.376 \text{ eV}$$

### Example 2: First Excited State

**Problem:** What is the first excited state and its degeneracy?

**Solution:**
Looking at the zeros: x₁₁ = 4.493 > x₁₀ = π ≈ 3.14

So the first excited state is (n, l) = (1, 1).

$$E_{11} = \frac{\hbar^2 (4.493)^2}{2ma^2} = \frac{(4.493)^2}{\pi^2} E_{10} \approx 2.05 E_{10}$$

Degeneracy: 2l + 1 = 3 (states with m = -1, 0, +1)

### Example 3: Wavefunction at Origin

**Problem:** Show that only s-waves (l = 0) have nonzero probability at the origin.

**Solution:**
$$|\psi_{nlm}(0)|^2 \propto |R_{nl}(0)|^2 |Y_l^m(0, \phi)|^2$$

For r → 0: j_l(kr) ~ (kr)^l

So R_{nl}(0) ~ 0^l = 0 for l > 0.

Only l = 0 states have finite probability density at r = 0.

---

## Practice Problems

### Direct Application
1. What is E₂₀ in terms of E₁₀?
2. How many states have E < 10E₁₀?
3. Find the radius where R₁₀(r) has its maximum.

### Intermediate
4. Verify that x₁₀ = π is a zero of j₀(x) = sin(x)/x.
5. Calculate the normalization constant for the ground state.
6. What is the probability of finding the particle in the inner half (r < a/2) for the ground state?

### Challenging
7. Derive the recursion relation for zeros of j_l from the recurrence for Bessel functions.
8. Show that the degeneracy grows as n² l for large quantum numbers.

---

## Computational Lab

```python
"""
Day 424: Infinite Spherical Well
Energy levels and wavefunctions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.optimize import brentq

# Find zeros of spherical Bessel functions
def find_bessel_zeros(l, num_zeros=5):
    """Find zeros of j_l(x)"""
    zeros = []
    x = 0.1
    while len(zeros) < num_zeros:
        try:
            # Search in intervals
            x_start = x
            x_end = x + np.pi
            # Check for sign change
            if spherical_jn(l, x_start) * spherical_jn(l, x_end) < 0:
                zero = brentq(lambda x: spherical_jn(l, x), x_start, x_end)
                zeros.append(zero)
        except:
            pass
        x += 0.5
        if x > 100:
            break
    return np.array(zeros)

# Calculate zeros for l = 0 to 5
print("=== Zeros of Spherical Bessel Functions ===")
print("\n      n=1      n=2      n=3      n=4      n=5")
zeros_dict = {}
for l in range(6):
    zeros = find_bessel_zeros(l, 5)
    zeros_dict[l] = zeros
    print(f"l={l}: ", end="")
    for z in zeros:
        print(f"{z:8.4f} ", end="")
    print()

# Calculate energy levels
print("\n\n=== Energy Levels (in units of ℏ²/2ma²) ===")

energies = []
for l in range(6):
    for n, x_nl in enumerate(zeros_dict[l][:5], 1):
        E = x_nl**2
        degeneracy = 2*l + 1
        energies.append((E, n, l, x_nl, degeneracy))

# Sort by energy
energies.sort()

print("\nLevel  (n,l)    E/(ℏ²/2ma²)    x_nl      Degeneracy")
print("-" * 55)
for i, (E, n, l, x_nl, deg) in enumerate(energies[:15], 1):
    spectroscopic = ['s', 'p', 'd', 'f', 'g', 'h'][l] if l < 6 else f'l={l}'
    print(f"  {i:2d}   ({n},{l})={n}{spectroscopic}    {E:8.3f}        {x_nl:.4f}       {deg}")

# Plot wavefunctions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

r = np.linspace(0, 1, 500)  # r/a

# Ground state R_10
ax = axes[0, 0]
x_10 = zeros_dict[0][0]  # π
R_10 = spherical_jn(0, x_10 * r)
R_10_norm = R_10 / np.sqrt(np.trapezoid(R_10**2 * r**2, r))
ax.plot(r, R_10_norm, 'b-', linewidth=2, label='R₁₀(r)')
ax.fill_between(r, 0, R_10_norm**2 * r**2 * 5, alpha=0.3, label='P(r) = |R₁₀|²r² (scaled)')
ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('r/a', fontsize=12)
ax.set_ylabel('R(r)', fontsize=12)
ax.set_title('Ground State (n=1, l=0): 1s', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# First excited states
ax = axes[0, 1]
x_11 = zeros_dict[1][0]
R_11 = spherical_jn(1, x_11 * r)
R_11_norm = R_11 / np.sqrt(np.trapezoid(R_11**2 * r**2, r))

x_20 = zeros_dict[0][1]  # 2π
R_20 = spherical_jn(0, x_20 * r)
R_20_norm = R_20 / np.sqrt(np.trapezoid(R_20**2 * r**2, r))

ax.plot(r, R_11_norm, 'r-', linewidth=2, label='R₁₁(r) - 1p')
ax.plot(r, R_20_norm, 'g-', linewidth=2, label='R₂₀(r) - 2s')
ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('r/a', fontsize=12)
ax.set_ylabel('R(r)', fontsize=12)
ax.set_title('First Excited States', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Energy level diagram
ax = axes[1, 0]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for i, (E, n, l, x_nl, deg) in enumerate(energies[:12]):
    spectroscopic = ['s', 'p', 'd', 'f', 'g', 'h'][l]
    # Draw energy level
    ax.hlines(E, l-0.3, l+0.3, colors=colors[l], linewidth=3)
    ax.text(l+0.35, E, f'{n}{spectroscopic}', fontsize=10, va='center')

ax.set_xlabel('l', fontsize=12)
ax.set_ylabel('E (ℏ²/2ma²)', fontsize=12)
ax.set_title('Energy Level Diagram', fontsize=14)
ax.set_xticks(range(6))
ax.set_xlim(-0.5, 5.5)
ax.grid(True, alpha=0.3, axis='y')

# Comparison with rectangular box
ax = axes[1, 1]

# Spherical well levels
sphere_E = [e[0] for e in energies[:10]]
sphere_deg = [e[4] for e in energies[:10]]

# Rectangular box: E ∝ (n_x² + n_y² + n_z²)π²
# Find first 10 levels
rect_levels = []
for nx in range(1, 10):
    for ny in range(1, 10):
        for nz in range(1, 10):
            E = (nx**2 + ny**2 + nz**2) * np.pi**2
            rect_levels.append(E)

rect_levels = sorted(set(rect_levels))[:10]

# Plot comparison
x_pos = np.arange(10)
width = 0.35

bars1 = ax.bar(x_pos - width/2, sphere_E, width, label='Spherical well')
bars2 = ax.bar(x_pos + width/2, rect_levels, width, label='Rectangular box')

ax.set_xlabel('Level number', fontsize=12)
ax.set_ylabel('E (arb. units)', fontsize=12)
ax.set_title('Spherical vs Rectangular Well', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day424_spherical_well.png', dpi=150)
plt.show()

# Nuclear magic numbers
print("\n\n=== Nuclear Shell Model (Simplified) ===")
print("\nFilling levels with protons/neutrons (spin degeneracy = 2):")
print("\nLevel  Config   Particles   Cumulative")
print("-" * 45)

cumulative = 0
for i, (E, n, l, x_nl, deg) in enumerate(energies[:10], 1):
    spectroscopic = ['s', 'p', 'd', 'f', 'g', 'h'][l]
    particles = 2 * deg  # Include spin
    cumulative += particles
    config = f'{n}{spectroscopic}'
    print(f"  {i:2d}    {config:4s}      {particles:3d}          {cumulative}")

print("\nWithout spin-orbit: Magic numbers would be 2, 8, 18, 20, 34, ...")
print("Real magic numbers (with spin-orbit): 2, 8, 20, 28, 50, 82, 126")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Potential | V = 0 (r < a), V = ∞ (r ≥ a) |
| Boundary condition | j_l(ka) = 0 |
| Wavenumber | k_{nl} = x_{nl}/a |
| Energy | E_{nl} = ℏ²x²_{nl}/(2ma²) |
| Degeneracy | g_{nl} = 2l + 1 |
| Ground state | E₁₀ = ℏ²π²/(2ma²) |

### Key Insights

1. **Boundary condition** j_l(ka) = 0 quantizes energy
2. **Level ordering** different from rectangular box
3. **Degeneracy** (2l+1) from angular momentum
4. **s-waves** penetrate to origin; higher l suppressed
5. **Nuclear shell model** is improved spherical well

---

## Daily Checklist

- [ ] I can set up the infinite spherical well problem
- [ ] I understand why j_l(ka) = 0 quantizes energy
- [ ] I can find energies from Bessel function zeros
- [ ] I know the degeneracy for each level
- [ ] I can calculate ground state properties
- [ ] I see the connection to nuclear physics

---

## Preview: Day 425

Tomorrow we study the **finite spherical well**, where particles can tunnel beyond r = a. This introduces the important physics of bound vs scattering states.

---

**Next:** [Day_425_Friday.md](Day_425_Friday.md) — Finite Spherical Well
