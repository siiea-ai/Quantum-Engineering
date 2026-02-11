# Day 372: Infinite Square Well - Setup and Energy Quantization

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: ISW setup and boundary conditions |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Energy quantization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Eigenvalue visualization |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Define the infinite square well potential mathematically and physically
2. Apply boundary conditions to constrain wave function solutions
3. Derive the quantized energy spectrum from standing wave requirements
4. Calculate energy levels for particles in boxes of various sizes
5. Connect energy quantization to the uncertainty principle
6. Relate the ISW model to quantum dots and nanostructures

---

## Core Content

### 1. The Infinite Square Well Potential

The **infinite square well** (also called "particle in a box") is the simplest bound state problem in quantum mechanics. Despite its simplicity, it captures the essence of quantum confinement and serves as a starting point for understanding more complex systems.

#### Potential Definition

$$\boxed{V(x) = \begin{cases} 0 & \text{if } 0 < x < L \\ +\infty & \text{otherwise} \end{cases}}$$

This describes an impenetrable box: the particle is completely free inside the well ($0 < x < L$) but cannot exist outside due to infinite potential barriers.

#### Physical Realizations

While truly infinite potentials don't exist in nature, the ISW approximates:
- Electrons in very deep semiconductor quantum wells
- Nucleons confined within atomic nuclei
- Metallic nanoparticles with work function >> thermal energy
- Single-walled carbon nanotubes (in transverse direction)

### 2. Time-Independent Schrodinger Equation

Inside the well ($0 < x < L$), where $V = 0$:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$

Rearranging:

$$\frac{d^2\psi}{dx^2} = -\frac{2mE}{\hbar^2}\psi$$

Define the **wave number**:

$$k^2 = \frac{2mE}{\hbar^2} \implies k = \frac{\sqrt{2mE}}{\hbar}$$

The equation becomes:

$$\boxed{\frac{d^2\psi}{dx^2} = -k^2\psi}$$

This is the simple harmonic oscillator equation in $x$, with general solution:

$$\psi(x) = A\sin(kx) + B\cos(kx)$$

### 3. Boundary Conditions

The wave function must satisfy **boundary conditions** at the walls.

#### Why Does $\psi$ Vanish at the Walls?

At an infinite potential barrier, the particle cannot penetrate. Mathematically, for the probability current to be finite at $x = 0$ and $x = L$, we require:

$$\boxed{\psi(0) = 0 \quad \text{and} \quad \psi(L) = 0}$$

These are **Dirichlet boundary conditions**.

#### Rigorous Justification

Consider a finite well of depth $V_0$ and take $V_0 \to \infty$. Outside the well, the wave function decays as $e^{-\kappa x}$ with decay constant:

$$\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar} \xrightarrow{V_0 \to \infty} \infty$$

The penetration depth $\delta = 1/\kappa \to 0$, so the wave function is forced to zero at the boundary.

### 4. Applying the First Boundary Condition

At $x = 0$:

$$\psi(0) = A\sin(0) + B\cos(0) = B = 0$$

Therefore, the coefficient of cosine must vanish:

$$\psi(x) = A\sin(kx)$$

### 5. Applying the Second Boundary Condition

At $x = L$:

$$\psi(L) = A\sin(kL) = 0$$

For a non-trivial solution ($A \neq 0$), we require:

$$\sin(kL) = 0$$

This means:

$$kL = n\pi, \quad n = 1, 2, 3, \ldots$$

Note: $n = 0$ gives $\psi = 0$ everywhere (no particle), and negative $n$ just flips the sign of $\psi$ (same physical state).

### 6. Quantized Wave Numbers and Energies

The allowed wave numbers are:

$$\boxed{k_n = \frac{n\pi}{L}, \quad n = 1, 2, 3, \ldots}$$

Substituting back into $k^2 = 2mE/\hbar^2$:

$$E_n = \frac{\hbar^2 k_n^2}{2m} = \frac{\hbar^2}{2m}\left(\frac{n\pi}{L}\right)^2$$

$$\boxed{E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \quad n = 1, 2, 3, \ldots}$$

### 7. Properties of the Energy Spectrum

#### Ground State Energy

The lowest energy state ($n = 1$) has:

$$E_1 = \frac{\pi^2\hbar^2}{2mL^2}$$

This is **not zero**! The particle cannot be at rest - this is the **zero-point energy**, a purely quantum effect.

#### Energy Level Spacing

The energy levels grow quadratically with $n$:

$$E_n = n^2 E_1$$

| Level | Energy | Ratio to Ground |
|-------|--------|-----------------|
| $n = 1$ | $E_1$ | 1 |
| $n = 2$ | $4E_1$ | 4 |
| $n = 3$ | $9E_1$ | 9 |
| $n = 4$ | $16E_1$ | 16 |

The spacing between adjacent levels:

$$E_{n+1} - E_n = (2n + 1)E_1$$

Spacing **increases** with $n$ - very different from the harmonic oscillator!

#### Energy Level Diagram

```
Energy
  ^
  |  ═══════════════  n=4, E=16E₁
  |
  |  ═══════════════  n=3, E=9E₁
  |
  |  ═══════════════  n=2, E=4E₁
  |
  |  ═══════════════  n=1, E=E₁ (ground state)
  |
  |  - - - - - - - -  E=0 (classical minimum)
  +────────────────────────────> x
       0        L
```

### 8. Connection to the Uncertainty Principle

The zero-point energy can be understood from the uncertainty principle:

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

**Position uncertainty:** The particle is confined to width $L$:
$$\Delta x \sim L$$

**Momentum uncertainty:**
$$\Delta p \gtrsim \frac{\hbar}{2L}$$

**Minimum kinetic energy:**
$$E = \frac{(\Delta p)^2}{2m} \gtrsim \frac{\hbar^2}{8mL^2}$$

This gives the correct order of magnitude for $E_1$!

### 9. Standing Wave Interpretation

The quantization condition $kL = n\pi$ has a beautiful interpretation: **exactly $n$ half-wavelengths must fit in the box**.

The de Broglie wavelength is:

$$\lambda_n = \frac{2\pi}{k_n} = \frac{2L}{n}$$

| State | Half-wavelengths in box | Nodes (excluding boundaries) |
|-------|-------------------------|------------------------------|
| $n = 1$ | 1 | 0 |
| $n = 2$ | 2 | 1 |
| $n = 3$ | 3 | 2 |
| $n$ | $n$ | $n-1$ |

**General rule:** The $n$-th eigenfunction has $(n-1)$ nodes inside the well.

### 10. Classical Limit

For a classical particle with the same energy $E$:

$$E = \frac{1}{2}mv^2 \implies v = \sqrt{\frac{2E}{m}}$$

The particle bounces back and forth with period:

$$T_{\text{classical}} = \frac{2L}{v}$$

The quantum wave packet (superposition of many eigenstates) approximately follows this classical trajectory, but with quantum corrections that become negligible as $n \to \infty$.

---

## Physical Interpretation

### Why Quantization?

Energy quantization arises from the **wave nature of matter** combined with **boundary conditions**. Just as a guitar string can only vibrate at certain frequencies (harmonics), a quantum particle in a box can only exist in specific energy states.

### Confinement Energy Scaling

The energy scales inversely with the square of the box size:

$$E_n \propto \frac{1}{L^2}$$

This has profound implications:
- **Nanoscale confinement**: Shrinking $L$ from 1 m to 1 nm increases $E$ by a factor of $10^{18}$!
- **Quantum dots**: 10 nm semiconductor dots have visible-light-energy level spacing
- **Color tunability**: Smaller quantum dots emit higher energy (bluer) light

### The "Particle in a Box" in Technology

| System | Typical Size $L$ | Level Spacing |
|--------|------------------|---------------|
| Atom (electron) | $\sim 0.1$ nm | $\sim 10$ eV |
| Quantum dot | $\sim 10$ nm | $\sim 0.1$ eV |
| Metal nanoparticle | $\sim 100$ nm | $\sim 0.001$ eV |
| Macroscopic box | $\sim 1$ cm | $\sim 10^{-18}$ eV |

For macroscopic systems, the spacing is so small that energy appears continuous - the **correspondence principle** at work.

---

## Quantum Computing Connection

### Quantum Dots as Qubits

Semiconductor quantum dots are often called "artificial atoms" because their confined electrons exhibit discrete energy levels like real atoms. The ISW model provides the first approximation to their energy structure.

**Applications in quantum computing:**
- **Spin qubits**: Electron spin in a quantum dot encodes |0⟩ and |1⟩
- **Charge qubits**: Electron position in a double dot (left/right) encodes states
- **Exchange gates**: Tunable barriers between dots enable two-qubit operations

### Transmon Qubit Analogy

In superconducting qubits, the Josephson junction creates an effective potential well for the superconducting phase. The transmon qubit uses the lowest two levels of this "electromagnetic box" as computational states.

**Key insight:** The $n^2$ energy scaling of the infinite well is problematic for qubits because we want only two addressable levels. The transmon achieves this through **anharmonicity** - making the well shape deviate from a simple box.

---

## Worked Examples

### Example 1: Electron in a Nanowire

**Problem:** An electron is confined to a semiconductor nanowire segment of length $L = 10$ nm. Calculate:
(a) The ground state energy
(b) The wavelength of a photon emitted in the $n = 2 \to n = 1$ transition

**Solution:**

(a) Ground state energy:
$$E_1 = \frac{\pi^2\hbar^2}{2m_e L^2}$$

Using $\hbar = 1.055 \times 10^{-34}$ J·s, $m_e = 9.109 \times 10^{-31}$ kg, $L = 10^{-8}$ m:

$$E_1 = \frac{\pi^2 \times (1.055 \times 10^{-34})^2}{2 \times 9.109 \times 10^{-31} \times (10^{-8})^2}$$

$$E_1 = \frac{1.096 \times 10^{-67}}{1.822 \times 10^{-46}} = 6.02 \times 10^{-21} \text{ J}$$

Converting to eV ($1 \text{ eV} = 1.602 \times 10^{-19}$ J):

$$\boxed{E_1 = 0.0376 \text{ eV} = 37.6 \text{ meV}}$$

(b) Photon wavelength for $n = 2 \to n = 1$:

$$\Delta E = E_2 - E_1 = (4 - 1)E_1 = 3E_1 = 0.113 \text{ eV}$$

The photon wavelength:
$$\lambda = \frac{hc}{\Delta E} = \frac{1240 \text{ eV·nm}}{0.113 \text{ eV}} = \boxed{10.97 \text{ μm}}$$

This is in the **mid-infrared** region.

---

### Example 2: Nuclear Confinement

**Problem:** A proton is confined to a "box" representing a nucleus with $L = 2$ fm (femtometers). Estimate the ground state energy.

**Solution:**

Using $m_p = 1.673 \times 10^{-27}$ kg, $L = 2 \times 10^{-15}$ m:

$$E_1 = \frac{\pi^2\hbar^2}{2m_p L^2} = \frac{\pi^2 \times (1.055 \times 10^{-34})^2}{2 \times 1.673 \times 10^{-27} \times (2 \times 10^{-15})^2}$$

$$E_1 = \frac{1.096 \times 10^{-67}}{1.338 \times 10^{-56}} = 8.19 \times 10^{-12} \text{ J}$$

Converting to MeV ($1 \text{ MeV} = 1.602 \times 10^{-13}$ J):

$$\boxed{E_1 \approx 51 \text{ MeV}}$$

This is the correct order of magnitude for nuclear energy scales!

---

### Example 3: Quantum Number from Energy

**Problem:** A particle in a 1D box has energy $E = 50E_1$. Is this a valid quantum state? If so, what is $n$?

**Solution:**

Since $E_n = n^2 E_1$:

$$n^2 = \frac{E}{E_1} = 50$$

$$n = \sqrt{50} \approx 7.07$$

Since $n$ must be a **positive integer**, $E = 50E_1$ is **not** a valid energy eigenvalue.

The nearby allowed states are:
- $n = 7$: $E = 49E_1$
- $n = 8$: $E = 64E_1$

---

## Practice Problems

### Level 1: Direct Application

1. **Scaling exercise:** If the box width is doubled from $L$ to $2L$, by what factor do the energy levels change?

2. **Energy ratio:** Calculate $E_5/E_2$ for a particle in an infinite square well.

3. **Node counting:** How many nodes (excluding boundaries) does the $n = 7$ eigenfunction have?

4. **Wavelength in box:** If the ground state has $\lambda_1 = 2L$, what is $\lambda_3$?

### Level 2: Intermediate

5. **Transition energies:** An electron in a box absorbs a photon and transitions from $n = 1$ to $n = 3$. Express the photon energy in terms of $E_1$.

6. **Effective mass:** In GaAs quantum wells, electrons have effective mass $m^* = 0.067m_e$. How does this affect the energy levels compared to free electrons?

7. **Classical vs quantum:** For what value of $n$ does the energy level spacing become comparable to thermal energy at room temperature ($k_B T \approx 26$ meV) for an electron in a 10 nm box?

8. **Uncertainty verification:** For the ground state of the ISW, verify that $\Delta x \cdot \Delta p \geq \hbar/2$ using $\Delta x \approx L/2$ and estimate $\Delta p$ from $E_1$.

### Level 3: Challenging

9. **Two particles:** Two non-interacting electrons (ignoring spin) are placed in an infinite square well. What is the ground state energy of the system?

10. **Correspondence principle:** Show that for large $n$, the average probability density $|\psi_n(x)|^2$ (averaged over many oscillations) approaches the classical uniform distribution $1/L$.

11. **3D box:** Generalize to a 3D rectangular box with sides $L_x$, $L_y$, $L_z$. Derive the energy levels and discuss degeneracy when $L_x = L_y = L_z = L$.

12. **Relativistic correction:** The non-relativistic result fails when $E_n \sim m c^2$. For an electron, estimate the quantum number $n$ at which relativistic corrections become significant for $L = 1$ fm.

---

## Computational Lab

### Exercise 1: Visualizing Energy Levels

```python
"""
Day 372 Computational Lab: Infinite Square Well Energy Levels
Visualize the quantized energy spectrum and scaling behavior
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
hbar = 1.055e-34  # J·s
m_e = 9.109e-31   # kg (electron mass)
eV = 1.602e-19    # J per eV

def energy_levels_ISW(n_max, L, m):
    """
    Calculate energy levels for infinite square well

    Parameters:
    -----------
    n_max : int
        Maximum quantum number
    L : float
        Well width (meters)
    m : float
        Particle mass (kg)

    Returns:
    --------
    n : array
        Quantum numbers 1, 2, ..., n_max
    E : array
        Energy levels (Joules)
    """
    n = np.arange(1, n_max + 1)
    E = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    return n, E

# Parameters for a 10 nm quantum dot
L = 10e-9  # 10 nm in meters
n_max = 10

# Calculate energy levels
n_vals, E_vals = energy_levels_ISW(n_max, L, m_e)
E_eV = E_vals / eV  # Convert to eV

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Energy level diagram
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(0, E_eV[-1] * 1.1)

for i, (n, E) in enumerate(zip(n_vals, E_eV)):
    ax1.hlines(E, 0.2, 0.8, colors='blue', linewidth=2)
    ax1.text(0.85, E, f'n={n}', fontsize=10, va='center')
    ax1.text(0.1, E, f'{E:.4f} eV', fontsize=9, va='center', ha='right')

ax1.set_ylabel('Energy (eV)', fontsize=12)
ax1.set_title(f'Infinite Square Well Energy Levels\nL = {L*1e9:.0f} nm, electron', fontsize=12)
ax1.set_xticks([])
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='E=0 (classical minimum)')
ax1.legend(loc='upper left')

# Right plot: E_n vs n^2 (should be linear)
ax2.plot(n_vals**2, E_eV, 'bo-', markersize=8, linewidth=2)
ax2.set_xlabel('$n^2$', fontsize=12)
ax2.set_ylabel('Energy (eV)', fontsize=12)
ax2.set_title('Verification: $E_n = n^2 E_1$', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add linear fit
slope = E_eV[0]  # E_1
ax2.plot(n_vals**2, slope * n_vals**2, 'r--', label=f'Linear fit: slope = $E_1$ = {slope:.4f} eV')
ax2.legend()

plt.tight_layout()
plt.savefig('isw_energy_levels.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*50)
print("Infinite Square Well Energy Analysis")
print("="*50)
print(f"Well width: L = {L*1e9:.1f} nm")
print(f"Particle: electron (m = {m_e:.3e} kg)")
print(f"\nGround state energy: E_1 = {E_eV[0]:.4f} eV = {E_eV[0]*1000:.2f} meV")
print(f"\nEnergy levels:")
for n, E in zip(n_vals, E_eV):
    print(f"  n = {n:2d}: E_{n} = {E:8.4f} eV = {n**2:3d} × E_1")
```

### Exercise 2: Size Dependence of Energy Levels

```python
"""
Explore how energy levels depend on box size
This demonstrates quantum confinement effects in nanostructures
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.055e-34
m_e = 9.109e-31
eV = 1.602e-19

def ground_state_energy(L, m):
    """Calculate ground state energy in eV"""
    return (np.pi**2 * hbar**2) / (2 * m * L**2) / eV

# Range of box sizes from 1 nm to 100 nm
L_nm = np.logspace(0, 2, 100)  # 1 to 100 nm
L_m = L_nm * 1e-9

# Calculate ground state energies
E1 = ground_state_energy(L_m, m_e)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(L_nm, E1, 'b-', linewidth=2)
ax.set_xlabel('Box Width L (nm)', fontsize=12)
ax.set_ylabel('Ground State Energy $E_1$ (eV)', fontsize=12)
ax.set_title('Quantum Confinement: Ground State Energy vs Size', fontsize=14)
ax.grid(True, which='both', alpha=0.3)

# Mark important energy scales
ax.axhline(y=0.026, color='red', linestyle='--', alpha=0.7, label='Room temp kT = 26 meV')
ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Visible light ~ 1.5-3 eV')

# Mark typical quantum dot sizes
ax.axvline(x=10, color='purple', linestyle=':', alpha=0.7)
ax.text(11, 0.1, 'Typical QD\n(~10 nm)', fontsize=10, color='purple')

ax.legend(loc='upper right')
ax.set_xlim([1, 100])
ax.set_ylim([1e-3, 10])

plt.tight_layout()
plt.savefig('isw_size_dependence.png', dpi=150, bbox_inches='tight')
plt.show()

# Print specific values
print("\nGround State Energy for Various Box Sizes:")
print("-" * 40)
for L in [1, 2, 5, 10, 20, 50, 100]:
    E = ground_state_energy(L * 1e-9, m_e)
    print(f"L = {L:3d} nm:  E_1 = {E:8.4f} eV = {E*1000:8.2f} meV")
```

### Exercise 3: Transition Wavelengths

```python
"""
Calculate and visualize allowed optical transitions
between energy levels in a quantum dot
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.055e-34
m_e = 9.109e-31
c = 3e8
eV = 1.602e-19

def transition_wavelength(n_i, n_f, L, m):
    """
    Calculate wavelength of photon for transition n_i -> n_f

    Returns wavelength in nanometers
    """
    E_i = (n_i**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    E_f = (n_f**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    delta_E = abs(E_i - E_f)

    if delta_E == 0:
        return np.inf

    wavelength = (hbar * 2 * np.pi * c) / delta_E
    return wavelength * 1e9  # Convert to nm

# Quantum dot size
L = 5e-9  # 5 nm - produces visible wavelengths

print(f"Transition wavelengths for L = {L*1e9:.0f} nm quantum dot")
print("="*60)
print(f"{'Transition':<15} {'Delta E (eV)':<15} {'Wavelength (nm)':<15} {'Region'}")
print("-"*60)

# Calculate transitions
n_max = 6
visible_transitions = []

for n_i in range(1, n_max + 1):
    for n_f in range(1, n_i):
        lambda_nm = transition_wavelength(n_i, n_f, L, m_e)
        E_i = (n_i**2 * np.pi**2 * hbar**2) / (2 * m_e * L**2) / eV
        E_f = (n_f**2 * np.pi**2 * hbar**2) / (2 * m_e * L**2) / eV
        delta_E = E_i - E_f

        # Classify spectral region
        if lambda_nm < 400:
            region = "UV"
        elif lambda_nm < 700:
            region = "Visible"
            visible_transitions.append((n_i, n_f, lambda_nm))
        elif lambda_nm < 1000:
            region = "Near-IR"
        else:
            region = "IR"

        print(f"{n_i} -> {n_f:<10} {delta_E:<15.4f} {lambda_nm:<15.1f} {region}")

# Visualize visible transitions
if visible_transitions:
    print(f"\nVisible light transitions:")
    for n_i, n_f, lam in visible_transitions:
        print(f"  {n_i} -> {n_f}: {lam:.1f} nm")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Potential | $V(x) = 0$ inside, $+\infty$ outside |
| Boundary conditions | $\psi(0) = \psi(L) = 0$ |
| General solution | $\psi(x) = A\sin(kx)$ |
| Quantization | $k_n L = n\pi$ |
| Energy levels | $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$ |
| Wave number | $k_n = \frac{n\pi}{L}$ |
| Nodes | $n - 1$ interior nodes |

### Main Takeaways

1. **Boundary conditions quantize energy**: Requiring $\psi = 0$ at the walls restricts allowed $k$ values, leading to discrete energies.

2. **Energy scales as $n^2$**: Unlike equally-spaced harmonic oscillator levels, ISW levels spread apart with increasing $n$.

3. **Zero-point energy exists**: The ground state has $E_1 > 0$, a consequence of the uncertainty principle.

4. **Size matters inversely**: $E \propto L^{-2}$ means smaller boxes have dramatically higher energies.

5. **Standing waves interpretation**: Each eigenstate is a standing wave with $n$ half-wavelengths in the box.

---

## Daily Checklist

- [ ] I can write down the ISW potential and explain its physical meaning
- [ ] I can derive the energy levels from the TISE with boundary conditions
- [ ] I understand why $E_n \propto n^2$ and can calculate specific energy values
- [ ] I can explain zero-point energy using the uncertainty principle
- [ ] I know the relationship between quantum number $n$ and number of nodes
- [ ] I can calculate transition wavelengths between energy levels
- [ ] I understand why quantum confinement is important in nanostructures
- [ ] I completed the computational exercises and understand the visualizations

---

## Preview: Day 373

Tomorrow we will derive the **normalized eigenfunctions** of the infinite square well:

$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)$$

We will prove their **orthonormality**:
$$\langle\psi_m|\psi_n\rangle = \delta_{mn}$$

And demonstrate **completeness** - any function satisfying the boundary conditions can be expanded in this basis:
$$f(x) = \sum_{n=1}^{\infty} c_n \psi_n(x)$$

This sets the stage for time evolution and quantum dynamics on Day 374.

---

*Day 372 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*
