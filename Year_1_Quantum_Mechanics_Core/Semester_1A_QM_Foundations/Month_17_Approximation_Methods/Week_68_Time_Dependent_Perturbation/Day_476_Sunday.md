# Day 476: Month 17 Capstone — Approximation Methods Integration

## Overview
**Day 476** | Year 1, Month 17, Week 68 | Comprehensive Review & Synthesis

Today we integrate all approximation methods learned in Month 17, comparing their domains of applicability and applying them to realistic quantum systems.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Method comparison & integration |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Comprehensive problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Multi-method simulation project |

---

## Learning Objectives

By the end of today, you will be able to:
1. Choose the appropriate approximation method for a given problem
2. Combine multiple methods for complex systems
3. Estimate errors and validate approximations
4. Apply these methods to real quantum computing scenarios
5. Solve comprehensive problems using multiple techniques
6. Connect approximation methods to modern quantum research

---

## Month 17 Review: The Four Pillars

### Overview of Methods

| Method | Best For | Key Assumption | Error Scaling |
|--------|----------|----------------|---------------|
| **Perturbation Theory** | Small corrections | V ≪ H₀ | O(λ²) |
| **Variational Method** | Ground states | Trial function overlap | Upper bound |
| **WKB** | Semiclassical regime | ℏ → 0, slowly varying | O(ℏ²) |
| **Time-Dependent** | Transitions | Weak, slow perturbation | O(V²) |

---

## Pillar 1: Time-Independent Perturbation Theory

### Non-Degenerate Case

**Energy corrections:**
$$E_n = E_n^{(0)} + \lambda\langle n^{(0)}|V|n^{(0)}\rangle + \lambda^2\sum_{m\neq n}\frac{|\langle m^{(0)}|V|n^{(0)}\rangle|^2}{E_n^{(0)}-E_m^{(0)}} + \cdots$$

**State corrections:**
$$|n\rangle = |n^{(0)}\rangle + \lambda\sum_{m\neq n}\frac{\langle m^{(0)}|V|n^{(0)}\rangle}{E_n^{(0)}-E_m^{(0)}}|m^{(0)}\rangle + \cdots$$

### Degenerate Case

1. Diagonalize V in the degenerate subspace
2. Use lifted degeneracy for higher orders

### When to Use

- Small perturbations to known systems
- Calculating energy shifts (Stark, Zeeman)
- Fine/hyperfine structure
- Understanding qubit frequency shifts

---

## Pillar 2: Variational Method

### The Variational Principle

$$E_{ground} \leq \langle\psi_{trial}|\hat{H}|\psi_{trial}\rangle$$

### Procedure

1. Choose trial wavefunction ψ(α, β, ...)
2. Calculate ⟨H⟩ = f(α, β, ...)
3. Minimize: ∂⟨H⟩/∂α = 0, etc.
4. Result is upper bound on E₀

### Linear Variational Method

$$|\psi\rangle = \sum_i c_i|\phi_i\rangle$$

Leads to generalized eigenvalue problem:
$$\mathbf{H}\mathbf{c} = E\mathbf{S}\mathbf{c}$$

### When to Use

- No exact solution available
- Ground state estimation
- Molecular orbital calculations
- VQE in quantum computing

---

## Pillar 3: WKB Approximation

### Semiclassical Wavefunction

**Classically allowed (E > V):**
$$\psi(x) = \frac{C}{\sqrt{p(x)}}\exp\left(\pm\frac{i}{\hbar}\int p(x)\,dx\right)$$

**Classically forbidden (E < V):**
$$\psi(x) = \frac{C}{\sqrt{|p(x)|}}\exp\left(\pm\frac{1}{\hbar}\int|p(x)|\,dx\right)$$

### Connection Formulas

At turning point x = a:
$$\frac{C}{\sqrt{|p|}}\exp\left(-\frac{1}{\hbar}\int|p|\,dx\right) \leftrightarrow \frac{2C}{\sqrt{p}}\cos\left(\frac{1}{\hbar}\int p\,dx - \frac{\pi}{4}\right)$$

### Bohr-Sommerfeld Quantization

$$\oint p\,dx = 2\pi\hbar\left(n + \frac{1}{2}\right)$$

### When to Use

- High quantum numbers
- Tunneling probability
- Bound state energies (quick estimate)
- Semiclassical correspondence

---

## Pillar 4: Time-Dependent Perturbation Theory

### First-Order Amplitude

$$c_f^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle f|V(t')|i\rangle e^{i\omega_{fi}t'}\,dt'$$

### Fermi's Golden Rule

For transitions to continuum:
$$W = \frac{2\pi}{\hbar}|\langle f|V|i\rangle|^2\rho(E_f)$$

### Harmonic Perturbations

Absorption/emission resonances:
$$W = \frac{\pi|V_{fi}|^2}{2\hbar^2}[\delta(\omega_{fi}-\omega) + \delta(\omega_{fi}+\omega)]$$

### When to Use

- Light-matter interaction
- Decay rates and lifetimes
- Qubit gate operations
- Transition probabilities

---

## Method Selection Guide

### Decision Tree

```
Is the perturbation small?
├── YES → Is it time-dependent?
│   ├── YES → TIME-DEPENDENT PERTURBATION
│   │   ├── Sudden: Use sudden approximation
│   │   ├── Harmonic: Fermi's Golden Rule
│   │   └── Adiabatic: Adiabatic theorem
│   └── NO → TIME-INDEPENDENT PERTURBATION
│       ├── Degenerate: Degenerate PT
│       └── Non-degenerate: Standard PT
└── NO → Is ground state needed?
    ├── YES → VARIATIONAL METHOD
    │   ├── Simple system: Trial wavefunction
    │   └── Complex system: Linear variational / VQE
    └── NO → Is it semiclassical (ℏ small)?
        ├── YES → WKB APPROXIMATION
        │   ├── Bound states: Bohr-Sommerfeld
        │   └── Tunneling: Connection formulas
        └── NO → NUMERICAL METHODS required
```

### Quick Reference Table

| Problem Type | Recommended Method |
|--------------|-------------------|
| Energy shift from weak field | Perturbation theory |
| Decay rate, transition probability | Fermi's Golden Rule |
| Ground state of complex system | Variational |
| Tunneling through barrier | WKB |
| High-n energy levels | WKB |
| Rabi oscillations | Time-dependent PT / exact 2-level |
| Molecular ground state | Linear variational / VQE |

---

## Quantum Computing Applications

### Qubit Frequency Calibration

**Perturbation theory:** Calculate charge dispersion
$$E_{01}(\phi) = E_{01}^{(0)} + \epsilon_1\cos(2\pi\phi/\phi_0) + \cdots$$

### VQE for Molecular Simulation

**Variational method:** Quantum-classical hybrid
$$E_{min} = \min_{\boldsymbol{\theta}} \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle$$

### Gate Error from Leakage

**Time-dependent PT:** Calculate |1⟩ → |2⟩ transitions
$$P_{leakage} \approx \frac{\Omega^2}{(\omega_{12}-\omega_{01})^2}$$

### Tunneling Between Qubit States

**WKB:** Double-well flux qubit tunneling
$$\Delta = \hbar\omega_0\exp\left(-\frac{1}{\hbar}\int_{-a}^{a}|p|\,dx\right)$$

---

## Comprehensive Examples

### Example 1: Hydrogen in Electric and Magnetic Fields

**Problem:** Find the energy levels of hydrogen n=2 in combined E and B fields.

**Method Combination:**
1. Start with degenerate 2s, 2p states
2. Use degenerate perturbation theory for Stark effect
3. Add Zeeman splitting (perturbation or exact)
4. Calculate transition rates between split levels (time-dependent)

**First-order energies (Stark):**
$$E = E_2^{(0)} \pm 3eEa_0$$

The 2s and 2p states mix to form linear combinations.

### Example 2: Alpha Decay

**Problem:** Estimate the alpha decay half-life of ²³⁸U.

**Method:** WKB for barrier penetration

The alpha particle tunnels through the Coulomb barrier:
$$V(r) = \frac{2Ze^2}{4\pi\epsilon_0 r} \text{ for } r > R_{nucleus}$$

Gamow factor:
$$G = \frac{2}{\hbar}\int_{R}^{R'}\sqrt{2m(V-E)}\,dr$$

$$T = e^{-2G}$$

For ²³⁸U: T ≈ 10⁻³⁸, giving t₁/₂ ≈ 4.5 billion years.

### Example 3: Laser Cooling

**Problem:** Calculate the scattering rate for laser cooling of atoms.

**Methods:** Time-dependent perturbation + Fermi's Golden Rule

Scattering rate (low intensity):
$$R = \frac{\Gamma}{2}\frac{s}{1+s+(2\delta/\Gamma)^2}$$

where s = I/I_sat and Γ = natural linewidth.

At saturation, R → Γ/2 (spontaneous emission limited).

---

## Practice Problems

### Comprehensive Problem Set

**Problem 1: Anharmonic Oscillator (Multiple Methods)**

For $H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2x^2 + \lambda x^4$:

a) Use perturbation theory to find E₀ to second order in λ.
b) Use a variational trial function ψ = (β/π)^(1/4) exp(-βx²/2) to estimate E₀.
c) Compare the two results for λ = 0.1 ℏω/a₀⁴ where a₀ = √(ℏ/mω).

**Problem 2: Double-Well Tunneling (WKB + Perturbation)**

A particle is in a symmetric double well V(x) = V₀[(x/a)² - 1]².

a) Use WKB to estimate the tunneling splitting Δ.
b) Treat Δ as a perturbation coupling the two localized states. Find the bonding/antibonding energies.
c) Calculate the tunneling time.

**Problem 3: Atom-Light Interaction (Time-Dependent)**

A two-level atom (ω₀ = 3×10¹⁵ rad/s, dipole moment d = 2ea₀) is exposed to:
- Weak monochromatic light at ω = ω₀
- Intensity I = 1 W/cm²

a) Calculate the Rabi frequency.
b) Find the time for a π-pulse.
c) If the laser is detuned by δ = 100 MHz, what is the maximum excitation probability?
d) Calculate the spontaneous emission rate. Is the coherent dynamics observable?

**Problem 4: VQE for H₂ (Variational)**

The hydrogen molecule in minimal basis has the Hamiltonian:
$$H = h_1(c_1^\dagger c_1 + c_2^\dagger c_2) + h_2 c_1^\dagger c_1 c_2^\dagger c_2 + \cdots$$

a) Map to qubit Hamiltonian using Jordan-Wigner.
b) Propose a hardware-efficient ansatz.
c) How many parameters are needed? How many measurements to estimate ⟨H⟩?

**Problem 5: Combined Effects in Quantum Dots**

A quantum dot (modeled as 3D harmonic oscillator, ℏω = 10 meV) contains one electron and is subject to:
- Electric field E = 10⁵ V/m (Stark effect)
- Magnetic field B = 1 T (Zeeman + diamagnetic)

a) Use perturbation theory to find the ground state energy shift.
b) Calculate the transition rate for 0→1 absorption at the shifted frequency.
c) What is the natural linewidth of the transition?

---

## Computational Lab

```python
"""
Day 476 Capstone Lab: Multi-Method Quantum Calculation
Compares perturbation theory, variational method, and WKB for the same system
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad, odeint
from scipy.linalg import eigh

# Physical constants
hbar = 1.0  # Natural units
m = 1.0
omega = 1.0

print("=" * 70)
print("MONTH 17 CAPSTONE: COMPARING APPROXIMATION METHODS")
print("=" * 70)

# ============================================================
# SYSTEM: Anharmonic Oscillator H = p²/2m + ½mω²x² + λx⁴
# ============================================================

print("\n" + "=" * 70)
print("SYSTEM: ANHARMONIC OSCILLATOR")
print("H = p²/2m + ½mω²x² + λx⁴")
print("=" * 70)

lambda_values = np.linspace(0, 0.5, 20)

# ============================================================
# METHOD 1: PERTURBATION THEORY
# ============================================================

def perturbation_energy(lam, order=2):
    """
    Ground state energy using perturbation theory.

    E₀ = ½ℏω + λ⟨0|x⁴|0⟩ + λ²(second order) + ...

    For harmonic oscillator: ⟨0|x⁴|0⟩ = 3/(4m²ω²) (in natural units: 3/4)
    Second order: -λ²(21/8)/(ℏω)
    """
    E0_unperturbed = 0.5 * hbar * omega

    # First order: ⟨0|x⁴|0⟩ = 3ℏ²/(4m²ω²)
    E1 = 3/4 * (hbar / (m * omega))**2

    # Second order correction
    E2 = -21/8 * (hbar / (m * omega))**4 / (hbar * omega)

    if order == 1:
        return E0_unperturbed + lam * E1
    elif order == 2:
        return E0_unperturbed + lam * E1 + lam**2 * E2
    else:
        return E0_unperturbed

E_pert_1 = [perturbation_energy(l, order=1) for l in lambda_values]
E_pert_2 = [perturbation_energy(l, order=2) for l in lambda_values]

print("\nMethod 1: PERTURBATION THEORY")
print("-" * 40)
print(f"E₀ = ½ℏω + (3/4)λ(ℏ/mω)² - (21/8)λ²(ℏ/mω)⁴/(ℏω) + ...")
print(f"At λ=0.1: E₀^(1) = {perturbation_energy(0.1, 1):.6f}")
print(f"At λ=0.1: E₀^(2) = {perturbation_energy(0.1, 2):.6f}")

# ============================================================
# METHOD 2: VARIATIONAL METHOD
# ============================================================

def variational_energy(beta, lam):
    """
    Energy expectation value for Gaussian trial function.

    ψ(x) = (β/π)^(1/4) exp(-βx²/2)

    ⟨T⟩ = ℏ²β/(4m)
    ⟨V_ho⟩ = mω²/(4β)
    ⟨x⁴⟩ = 3/(4β²)
    """
    T = hbar**2 * beta / (4 * m)
    V_ho = m * omega**2 / (4 * beta)
    V_anh = lam * 3 / (4 * beta**2)
    return T + V_ho + V_anh

def optimize_variational(lam):
    """Find optimal β and minimum energy."""
    result = minimize_scalar(lambda b: variational_energy(b, lam),
                             bounds=(0.1, 10), method='bounded')
    return result.fun, result.x

E_var = []
beta_opt = []
for l in lambda_values:
    E, b = optimize_variational(l)
    E_var.append(E)
    beta_opt.append(b)

print("\nMethod 2: VARIATIONAL METHOD")
print("-" * 40)
print(f"Trial: ψ(x) = (β/π)^(1/4) exp(-βx²/2)")
E_opt, b_opt = optimize_variational(0.1)
print(f"At λ=0.1: β_opt = {b_opt:.6f}, E₀ = {E_opt:.6f}")

# ============================================================
# METHOD 3: NUMERICAL (EXACT FOR COMPARISON)
# ============================================================

def numerical_ground_state(lam, N=100):
    """
    Solve in harmonic oscillator basis.
    """
    # Matrix elements in HO basis
    H = np.zeros((N, N))

    for n in range(N):
        # Diagonal: harmonic oscillator + <n|x⁴|n>
        H[n, n] = hbar * omega * (n + 0.5)
        # <n|x⁴|n> = (3/4)(2n² + 2n + 1) / (mω/ℏ)²
        x4_nn = 3/4 * (2*n**2 + 2*n + 1) * (hbar / (m * omega))**2
        H[n, n] += lam * x4_nn

        # Off-diagonal: <n|x⁴|n±2>, <n|x⁴|n±4>
        if n >= 2:
            x4_n2 = (1/4) * np.sqrt(n*(n-1)) * (4*n - 2) * (hbar / (m * omega))**2
            H[n, n-2] += lam * x4_n2
            H[n-2, n] += lam * x4_n2
        if n >= 4:
            x4_n4 = (1/4) * np.sqrt(n*(n-1)*(n-2)*(n-3)) * (hbar / (m * omega))**2
            H[n, n-4] += lam * x4_n4
            H[n-4, n] += lam * x4_n4

    eigenvalues, _ = eigh(H)
    return eigenvalues[0]

E_exact = [numerical_ground_state(l) for l in lambda_values]

print("\nMethod 3: NUMERICAL (EXACT)")
print("-" * 40)
print(f"At λ=0.1: E₀ = {numerical_ground_state(0.1):.6f}")

# ============================================================
# COMPARISON PLOT
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON OF METHODS")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Absolute energies
ax = axes[0]
ax.plot(lambda_values, E_exact, 'k-', linewidth=3, label='Exact (numerical)')
ax.plot(lambda_values, E_pert_1, 'b--', linewidth=2, label='Perturbation (1st order)')
ax.plot(lambda_values, E_pert_2, 'b-', linewidth=2, label='Perturbation (2nd order)')
ax.plot(lambda_values, E_var, 'r-', linewidth=2, label='Variational')

ax.set_xlabel('Anharmonicity λ', fontsize=12)
ax.set_ylabel('Ground State Energy E₀', fontsize=12)
ax.set_title('Ground State Energy: Method Comparison', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Relative errors
ax = axes[1]
err_pert_1 = np.abs(np.array(E_pert_1) - np.array(E_exact)) / np.array(E_exact) * 100
err_pert_2 = np.abs(np.array(E_pert_2) - np.array(E_exact)) / np.array(E_exact) * 100
err_var = np.abs(np.array(E_var) - np.array(E_exact)) / np.array(E_exact) * 100

ax.semilogy(lambda_values[1:], err_pert_1[1:], 'b--', linewidth=2, label='Perturbation (1st)')
ax.semilogy(lambda_values[1:], err_pert_2[1:], 'b-', linewidth=2, label='Perturbation (2nd)')
ax.semilogy(lambda_values[1:], err_var[1:], 'r-', linewidth=2, label='Variational')

ax.set_xlabel('Anharmonicity λ', fontsize=12)
ax.set_ylabel('Relative Error (%)', fontsize=12)
ax.set_title('Approximation Errors', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(1e-4, 100)

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# WKB FOR TUNNELING
# ============================================================

print("\n" + "=" * 70)
print("WKB: DOUBLE-WELL TUNNELING")
print("=" * 70)

def double_well_potential(x, V0, a):
    """V(x) = V0[(x/a)² - 1]²"""
    return V0 * ((x/a)**2 - 1)**2

def wkb_tunneling_integral(E, V0, a):
    """
    Calculate the WKB tunneling integral.
    """
    # Find turning points where V(x) = E
    # For E < V0, turning points are at x = ±a√(1 - √(E/V0))
    if E >= V0:
        return 0  # No barrier

    inner = np.sqrt(1 - np.sqrt(E / V0))
    x1 = -a * inner
    x2 = a * inner

    def integrand(x):
        V = double_well_potential(x, V0, a)
        if V > E:
            return np.sqrt(2 * m * (V - E))
        return 0

    integral, _ = quad(integrand, x1, x2)
    return integral / hbar

# Parameters
V0 = 5.0
a = 1.0

print(f"\nDouble well: V(x) = V₀[(x/a)² - 1]² with V₀ = {V0}, a = {a}")

# Calculate tunneling splitting for ground state (E ≈ bottom of well)
E_well_bottom = 0  # Minimum of each well

# WKB tunneling splitting: Δ ≈ ℏω₀ exp(-S/ℏ)
S = wkb_tunneling_integral(E_well_bottom * 0.5, V0, a)  # Approximate
omega_well = np.sqrt(8 * V0 / (m * a**2))  # Curvature at well bottom
Delta_WKB = hbar * omega_well * np.exp(-S)

print(f"Well frequency: ω = {omega_well:.3f}")
print(f"WKB integral: S/ℏ = {S:.3f}")
print(f"Tunneling splitting: Δ ≈ {Delta_WKB:.6f}")
print(f"Tunneling time: τ ≈ {np.pi * hbar / Delta_WKB:.3f}" if Delta_WKB > 0 else "Tunneling time: τ ≈ ∞")

# Plot double well and energy levels
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-2*a, 2*a, 500)
V = double_well_potential(x, V0, a)

ax.plot(x/a, V/V0, 'b-', linewidth=2, label='V(x)/V₀')
ax.axhline(0.1, color='r', linestyle='--', alpha=0.7, label='E₀ (symmetric)')
ax.axhline(0.1 + Delta_WKB/V0, color='orange', linestyle='--', alpha=0.7, label='E₁ (antisymmetric)')

ax.fill_between(x/a, 0, V/V0, where=(V/V0 > 0.1) & (np.abs(x) < a),
                alpha=0.3, color='gray', label='Tunneling region')

ax.set_xlabel('x/a', fontsize=12)
ax.set_ylabel('V/V₀', fontsize=12)
ax.set_title('Double-Well Potential and WKB Tunneling', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-0.2, 1.5)

plt.tight_layout()
plt.savefig('double_well_wkb.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# TIME-DEPENDENT: RABI + DECAY
# ============================================================

print("\n" + "=" * 70)
print("TIME-DEPENDENT: RABI OSCILLATIONS WITH DECAY")
print("=" * 70)

def rabi_with_decay(y, t, Omega, delta, gamma):
    """
    Optical Bloch equations (simplified).

    y = [rho_gg, rho_ee, Re(rho_ge), Im(rho_ge)]
    """
    rho_gg, rho_ee, u, v = y

    # Bloch equations
    drho_gg = gamma * rho_ee + Omega * v
    drho_ee = -gamma * rho_ee - Omega * v
    du = -gamma/2 * u + delta * v
    dv = -gamma/2 * v - delta * u - Omega * (rho_ee - rho_gg)

    return [drho_gg, drho_ee, du, dv]

# Parameters
Omega = 10  # Rabi frequency
delta = 0   # On resonance
gamma_values = [0, 1, 5, 20]  # Decay rates

t = np.linspace(0, 2, 500)

fig, ax = plt.subplots(figsize=(10, 6))

for gamma in gamma_values:
    y0 = [1, 0, 0, 0]  # Start in ground state
    solution = odeint(rabi_with_decay, y0, t, args=(Omega, delta, gamma))
    rho_ee = solution[:, 1]
    label = f'γ = {gamma}' if gamma > 0 else 'No decay'
    ax.plot(t * Omega, rho_ee, linewidth=2, label=label)

ax.set_xlabel('Time (units of 1/Ω)', fontsize=12)
ax.set_ylabel('Excited State Population', fontsize=12)
ax.set_title('Rabi Oscillations with Spontaneous Emission', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('rabi_with_decay.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("MONTH 17 SUMMARY: WHEN TO USE EACH METHOD")
print("=" * 70)

summary = """
┌─────────────────────────────────────────────────────────────────────┐
│                    APPROXIMATION METHOD GUIDE                        │
├─────────────────────┬───────────────────────────────────────────────┤
│ METHOD              │ USE WHEN                                      │
├─────────────────────┼───────────────────────────────────────────────┤
│ Perturbation Theory │ • Small corrections to known systems          │
│                     │ • Energy shifts (Stark, Zeeman)               │
│                     │ • Fine/hyperfine structure                    │
│                     │ • Qubit frequency calibration                 │
├─────────────────────┼───────────────────────────────────────────────┤
│ Variational Method  │ • No exact solution exists                    │
│                     │ • Ground state estimation                     │
│                     │ • Molecular calculations                      │
│                     │ • VQE quantum computing                       │
├─────────────────────┼───────────────────────────────────────────────┤
│ WKB Approximation   │ • Semiclassical regime (large n)              │
│                     │ • Tunneling probabilities                     │
│                     │ • Quick energy estimates                      │
│                     │ • Flux qubit tunneling                        │
├─────────────────────┼───────────────────────────────────────────────┤
│ Time-Dependent PT   │ • Transition rates                            │
│                     │ • Decay lifetimes                             │
│                     │ • Light-matter interaction                    │
│                     │ • Qubit gate operations                       │
└─────────────────────┴───────────────────────────────────────────────┘

KEY INSIGHT: Real problems often require COMBINING methods!
• Perturbation + Time-dependent: Shifted transition frequencies
• Variational + WKB: Tunneling in complex potentials
• All methods: Error estimation through comparison
"""
print(summary)

print("\n" + "=" * 70)
print("CONGRATULATIONS! Month 17 Complete!")
print("=" * 70)
print("""
You have mastered:
✓ Time-independent perturbation theory (degenerate & non-degenerate)
✓ Variational method and VQE
✓ WKB semiclassical approximation
✓ Time-dependent perturbation theory
✓ Fermi's Golden Rule
✓ Selection rules and spontaneous emission

Next Month: Scattering Theory
""")
```

---

## Summary

### Month 17 Key Formulas

| Method | Core Formula |
|--------|--------------|
| Perturbation (1st) | $E_n^{(1)} = \langle n^{(0)}\|V\|n^{(0)}\rangle$ |
| Perturbation (2nd) | $E_n^{(2)} = \sum_{m \neq n}\frac{\|\langle m\|V\|n\rangle\|^2}{E_n^{(0)} - E_m^{(0)}}$ |
| Variational | $E_0 \leq \langle\psi_{trial}\|\hat{H}\|\psi_{trial}\rangle$ |
| WKB quantization | $\oint p\,dx = 2\pi\hbar(n + 1/2)$ |
| Fermi's Golden Rule | $W = \frac{2\pi}{\hbar}\|\langle f\|V\|i\rangle\|^2\rho(E_f)$ |
| Einstein A | $A = \frac{4\alpha\omega^3}{3c^2}\|\mathbf{d}\|^2$ |

### Key Insights

1. **Perturbation theory** works for small corrections; errors scale as λ^(n+1)
2. **Variational method** always gives upper bound; no small parameter needed
3. **WKB** bridges quantum and classical; breaks down at turning points
4. **Time-dependent PT** gives transition rates; leads to Fermi's Golden Rule
5. **Combine methods** for complex problems—each has complementary strengths

---

## Daily Checklist

- [ ] I can select the appropriate method for a given problem
- [ ] I can combine methods for complex systems
- [ ] I understand the error scaling of each approximation
- [ ] I can apply these methods to quantum computing scenarios
- [ ] I have completed the comprehensive problem set

---

## Month 17 Complete!

Congratulations on completing **Month 17: Approximation Methods**! You now have a powerful toolkit for solving realistic quantum mechanical problems.

**Next Month Preview:** Month 18 covers **Scattering Theory**—the quantum mechanics of collisions, phase shifts, Born approximation, and optical theorem.

---

**End of Month 17**
