# Day 427: Week 61 Review — Central Potentials

## Overview
**Day 427** | Year 1, Month 16, Week 61 | Comprehensive Review

Today we consolidate our understanding of three-dimensional quantum mechanics and central potentials, preparing for the hydrogen atom next week.

---

## Week 61 Summary

### Day 421: 3D Schrödinger Equation
- Extended QM to three dimensions
- Laplacian in spherical coordinates
- Separation: ψ = R(r)Y_l^m(θ,φ)
- Angular part → spherical harmonics

### Day 422: Radial Equation
- Substitution u = rR simplifies equation
- Effective potential: V_eff = V(r) + ℏ²l(l+1)/(2mr²)
- Centrifugal barrier for l > 0
- Boundary conditions determine quantization

### Day 423: Free Particle in 3D
- Spherical Bessel functions j_l, n_l
- Plane wave expansion (Rayleigh)
- Foundation for scattering theory

### Day 424: Infinite Spherical Well
- Boundary condition j_l(ka) = 0
- Zeros of Bessel functions give energies
- Level ordering differs from rectangular box

### Day 425: Finite Spherical Well
- Matching conditions at boundary
- Critical depth for bound states
- Nuclear physics applications

### Day 426: 3D Harmonic Oscillator
- E = ℏω(N + 3/2)
- Degeneracy g_N = (N+1)(N+2)/2
- Hidden SU(3) symmetry

---

## Master Formula Sheet

### 3D Schrödinger Equation
$$-\frac{\hbar^2}{2m}\nabla^2\psi + V(r)\psi = E\psi$$

### Separation of Variables
$$\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$

### Radial Equation (u = rR)
$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + V_{\text{eff}}(r)u = Eu$$

### Effective Potential
$$V_{\text{eff}}(r) = V(r) + \frac{\hbar^2 l(l+1)}{2mr^2}$$

### Spherical Bessel Functions

| Function | Definition | r → 0 | r → ∞ |
|----------|------------|-------|-------|
| j_l(ρ) | sin → derivative | ρ^l/(2l+1)!! | sin(ρ-lπ/2)/ρ |
| n_l(ρ) | cos → derivative | -(2l-1)!!/ρ^{l+1} | -cos(ρ-lπ/2)/ρ |

### Plane Wave Expansion
$$e^{ikz} = \sum_{l=0}^{\infty} i^l (2l+1) j_l(kr) P_l(\cos\theta)$$

### Key Results

| System | Energy Formula | Degeneracy |
|--------|----------------|------------|
| Infinite spherical well | E = ℏ²x²_{nl}/(2ma²) | 2l+1 |
| Finite well (critical) | V₀^{crit} = ℏ²π²/(8ma²) | — |
| 3D harmonic oscillator | E = ℏω(N + 3/2) | (N+1)(N+2)/2 |

---

## Conceptual Framework

### The Central Force Problem

```
Central Potential V(r)
        ↓
Spherical Symmetry [H, L²] = [H, L_z] = 0
        ↓
Good Quantum Numbers: n, l, m
        ↓
Separation: ψ = R(r) × Y_l^m(θ,φ)
        ↓
Angular → Spherical Harmonics (solved)
Radial → System-specific (solve)
```

### When to Use Which Functions

| Situation | Use | Reason |
|-----------|-----|--------|
| Regular at origin | j_l(kr) | n_l diverges |
| Bound states (exterior) | exp(-κr) | must decay |
| Scattering (outgoing) | h_l^{(1)} | outgoing wave |
| Scattering (incoming) | h_l^{(2)} | incoming wave |

### The Role of Angular Momentum

- l = 0: s-waves reach origin
- l > 0: centrifugal barrier excludes origin
- Higher l → larger barrier → states pushed outward

---

## Problem-Solving Strategies

### General Approach for Central Potentials

1. **Identify the potential** V(r)
2. **Separate variables** in spherical coordinates
3. **Angular part**: spherical harmonics (done!)
4. **Radial part**: solve for R(r) or u(r) = rR(r)
5. **Apply boundary conditions**
   - r = 0: regularity (u(0) = 0)
   - r → ∞: normalizability or scattering condition
6. **Quantize**: boundary conditions give discrete E

### Matching at Boundaries

For piecewise potentials:
1. Solve in each region
2. Match ψ and dψ/dr at boundaries
3. Equivalently: match R and R'/R (log derivative)

### Counting States

For degeneracy:
- Each (n, l) gives (2l + 1) states from m = -l, ..., +l
- Accidental degeneracy may combine different l

---

## Comprehensive Problems

### Problem 1: Mixed Potential
A particle experiences:
$$V(r) = \begin{cases} \frac{1}{2}m\omega^2 r^2 & r < a \\ V_0 & r \geq a \end{cases}$$

a) What is the effective potential for l = 1?
b) Sketch V_eff(r) for this case.
c) Qualitatively, how do bound states differ from the pure oscillator?

### Problem 2: Bessel Function Identity
Prove that:
$$j_0(x) j_0(y) = \frac{1}{2}\int_{-1}^{1} j_0\left(\sqrt{x^2 + y^2 - 2xy t}\right) dt$$

(Hint: Use plane wave expansion and orthogonality of Legendre polynomials)

### Problem 3: Dimensional Analysis
For the infinite spherical well:
a) Using only ℏ, m, and a, construct a quantity with dimensions of energy.
b) Verify this matches the actual ground state formula.
c) Why doesn't the result depend on any other constants?

### Problem 4: Quantum Numbers
A system has energy levels E_{nl} = E_0(2n + l + 3/2).
a) What is the ground state energy?
b) What is the degeneracy of the first excited level?
c) What physical system might this describe?

---

## Quantum Computing Connections

### This Week's QC Links

| Physics Concept | QC Application |
|-----------------|----------------|
| 3D Schrödinger equation | Molecular simulation (VQE) |
| Spherical Bessel functions | Scattering calculations |
| Harmonic oscillator | Bosonic modes, CV QC |
| Finite well | Nuclear structure simulation |
| Degeneracy | State space dimension |

### Looking Ahead: Hydrogen

The hydrogen atom (next week) combines:
- Central potential (Coulomb)
- All techniques from this week
- Foundation for atomic physics
- Model for trapped ion qubits

---

## Computational Lab

```python
"""
Day 427: Week 61 Review - Central Potentials Summary
Comprehensive visualization of concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, genlaguerre
from scipy.integrate import odeint

# Create comprehensive summary figure
fig = plt.figure(figsize=(16, 12))

# 1. Effective potentials for different l
ax1 = fig.add_subplot(2, 3, 1)
r = np.linspace(0.1, 5, 500)

# Coulomb-like: V = -1/r (atomic units)
for l in range(4):
    V_eff = -1/r + l*(l+1)/(2*r**2)
    ax1.plot(r, V_eff, linewidth=2, label=f'l = {l}')

ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.set_xlabel('r', fontsize=11)
ax1.set_ylabel('V_eff(r)', fontsize=11)
ax1.set_title('Effective Potential (Coulomb)', fontsize=12)
ax1.set_xlim(0, 5)
ax1.set_ylim(-2, 2)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Spherical Bessel functions
ax2 = fig.add_subplot(2, 3, 2)
rho = np.linspace(0.01, 15, 500)

for l in range(4):
    jl = spherical_jn(l, rho)
    ax2.plot(rho, jl, linewidth=2, label=f'j_{l}(ρ)')

ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel('ρ = kr', fontsize=11)
ax2.set_ylabel('j_l(ρ)', fontsize=11)
ax2.set_title('Spherical Bessel Functions', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 15)
ax2.set_ylim(-0.4, 1.1)

# 3. Energy levels comparison
ax3 = fig.add_subplot(2, 3, 3)

# Three systems
systems = {
    'Infinite Well': [np.pi**2, (4.493)**2, (5.763)**2, (2*np.pi)**2, (6.988)**2],
    'Harmonic Osc.': [1.5, 2.5, 3.5, 4.5, 5.5],
    '3D Box (rect)': [3, 6, 9, 11, 12],
}

x = np.arange(5)
width = 0.25
multiplier = 0

for name, levels in systems.items():
    # Normalize to first excited state
    levels = np.array(levels)
    levels = levels / levels[0]
    offset = width * multiplier
    ax3.bar(x + offset, levels, width, label=name)
    multiplier += 1

ax3.set_xlabel('Level number', fontsize=11)
ax3.set_ylabel('E / E₀ (normalized)', fontsize=11)
ax3.set_title('Energy Level Comparison', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Radial probability densities
ax4 = fig.add_subplot(2, 3, 4)

# Hydrogen-like wavefunctions
def hydrogen_R(n, l, r, a0=1):
    """Simplified hydrogen radial wavefunctions"""
    if n == 1 and l == 0:
        return 2 * np.exp(-r/a0)
    elif n == 2 and l == 0:
        return (1/np.sqrt(2)) * (1 - r/(2*a0)) * np.exp(-r/(2*a0))
    elif n == 2 and l == 1:
        return (1/np.sqrt(24)) * (r/a0) * np.exp(-r/(2*a0))
    elif n == 3 and l == 0:
        return (2/np.sqrt(27)) * (1 - 2*r/(3*a0) + 2*r**2/(27*a0**2)) * np.exp(-r/(3*a0))
    return np.zeros_like(r)

r = np.linspace(0.01, 20, 500)

for n, l, label in [(1, 0, '1s'), (2, 0, '2s'), (2, 1, '2p'), (3, 0, '3s')]:
    R = hydrogen_R(n, l, r)
    P = np.abs(R)**2 * r**2
    P = P / np.max(P)  # Normalize for display
    ax4.plot(r, P, linewidth=2, label=label)

ax4.set_xlabel('r / a₀', fontsize=11)
ax4.set_ylabel('P(r) (normalized)', fontsize=11)
ax4.set_title('Hydrogen Radial Probability', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 20)

# 5. Finite well bound states
ax5 = fig.add_subplot(2, 3, 5)

z = np.linspace(0.01, 4*np.pi, 1000)
lhs = z / np.tan(z)
lhs[np.abs(np.diff(lhs, prepend=lhs[0])) > 10] = np.nan

ax5.plot(z, lhs, 'b-', linewidth=2, label='z cot(z)')

for z0 in [2, 5, 8]:
    rhs = -np.sqrt(z0**2 - z**2)
    rhs[z > z0] = np.nan
    ax5.plot(z, rhs, '--', linewidth=2, label=f'z₀ = {z0}')

ax5.axvline(x=np.pi/2, color='red', linestyle=':', alpha=0.7)
ax5.set_xlabel('z = ka', fontsize=11)
ax5.set_ylabel('f(z)', fontsize=11)
ax5.set_title('Finite Well Quantization', fontsize=12)
ax5.legend(fontsize=9, loc='lower right')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 10)
ax5.set_ylim(-12, 5)

# 6. Degeneracy patterns
ax6 = fig.add_subplot(2, 3, 6)

N_max = 8

# Harmonic oscillator
N_vals = np.arange(N_max)
g_ho = [(N+1)*(N+2)//2 for N in N_vals]

# Hydrogen (including spin)
g_h = [2*n**2 for n in range(1, N_max+1)]

# Spherical well (first few levels)
g_sw = [1, 3, 5, 1, 7, 3, 9, 1]  # Rough approximation

width = 0.25
ax6.bar(N_vals - width, g_ho[:N_max], width, label='3D Harmonic Osc.')
ax6.bar(N_vals, g_h[:N_max], width, label='Hydrogen (with spin)')
ax6.bar(N_vals + width, g_sw[:N_max], width, label='Spherical Well')

ax6.set_xlabel('Principal quantum number', fontsize=11)
ax6.set_ylabel('Degeneracy', fontsize=11)
ax6.set_title('Degeneracy Comparison', fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day427_week61_review.png', dpi=150)
plt.show()

# Summary tables
print("=" * 60)
print("WEEK 61 SUMMARY: CENTRAL POTENTIALS")
print("=" * 60)

print("\n┌─────────────────────────────────────────────────────────┐")
print("│                   KEY EQUATIONS                         │")
print("├─────────────────────────────────────────────────────────┤")
print("│ Separation:    ψ(r,θ,φ) = R(r) Y_l^m(θ,φ)               │")
print("│ Radial (u=rR): -ℏ²u''/2m + V_eff u = Eu                 │")
print("│ V_effective:   V(r) + ℏ²l(l+1)/(2mr²)                   │")
print("│ Plane wave:    e^{ikz} = Σ i^l(2l+1)j_l(kr)P_l(cosθ)    │")
print("└─────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│                   KEY RESULTS                           │")
print("├─────────────────────────────────────────────────────────┤")
print("│ Infinite Well:     E = ℏ²x²_{nl}/(2ma²)                 │")
print("│ Finite Well:       V₀^{crit} = ℏ²π²/(8ma²)              │")
print("│ 3D Oscillator:     E = ℏω(N + 3/2), g = (N+1)(N+2)/2    │")
print("│ Small r behavior:  u(r) ~ r^{l+1}                        │")
print("└─────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│               QUANTUM NUMBERS                           │")
print("├─────────────────────────────────────────────────────────┤")
print("│ n:  principal (radial quantization)                     │")
print("│ l:  orbital angular momentum (0,1,2,...,n-1 for H)      │")
print("│ m:  magnetic quantum number (-l,...,+l)                 │")
print("│ Degeneracy per (n,l): 2l+1                              │")
print("└─────────────────────────────────────────────────────────┘")

print("\n✓ Ready for Week 62: THE HYDROGEN ATOM")
```

---

## Week 61 Checklist

### Concepts
- [ ] 3D Schrödinger equation and spherical coordinates
- [ ] Separation of variables for central potentials
- [ ] Effective potential and centrifugal barrier
- [ ] Spherical Bessel functions (first/second kind)
- [ ] Plane wave expansion
- [ ] Boundary conditions and quantization

### Problem Types
- [ ] Find energy levels from Bessel function zeros
- [ ] Determine critical depth for bound states
- [ ] Calculate degeneracies
- [ ] Analyze wavefunction behavior at boundaries
- [ ] Apply matching conditions

### Connections
- [ ] Link to Month 15 angular momentum
- [ ] Understand role in scattering theory
- [ ] See applications to nuclear/atomic physics
- [ ] Recognize quantum computing relevance

---

## Preview: Week 62 — The Hydrogen Atom

Next week we solve the most important problem in atomic physics: the hydrogen atom.

**Topics:**
- Coulomb potential -e²/r
- Bohr model review and quantum solution
- Energy levels E_n = -13.6 eV/n²
- Radial wavefunctions (Laguerre polynomials)
- Full 3D probability densities
- Degeneracy and spectroscopic notation

The hydrogen atom is:
- First exact solution of a realistic atom
- Foundation for all of chemistry
- Model for trapped ion qubits
- Starting point for perturbation theory

---

**Congratulations on completing Week 61!**

**Next:** [Week_62_Hydrogen_Atom](../Week_62_Hydrogen_Atom/README.md)
