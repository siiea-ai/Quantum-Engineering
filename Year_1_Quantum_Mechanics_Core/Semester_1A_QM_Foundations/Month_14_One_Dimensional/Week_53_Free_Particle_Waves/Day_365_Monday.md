# Day 365: Free Particle - Time-Independent Schrödinger Equation

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Free particle TISE, plane wave solutions |
| Afternoon | 2.5 hours | Problem solving: Dispersion, momentum, energy |
| Evening | 2 hours | Computational lab: Plane wave visualization |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the time-independent Schrödinger equation** for a free particle (V=0)
2. **Solve the TISE** to obtain plane wave solutions
3. **Interpret the dispersion relation** E = ℏ²k²/2m physically
4. **Explain why the spectrum is continuous** rather than discrete
5. **Distinguish traveling waves from standing waves** in the free particle context
6. **Visualize plane waves** and understand their physical meaning

---

## Core Content

### 1. The Free Particle Hamiltonian

A **free particle** is one with no forces acting on it, meaning the potential energy vanishes everywhere:

$$V(x) = 0$$

The Hamiltonian is purely kinetic:

$$\boxed{\hat{H} = \frac{\hat{p}^2}{2m} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}}$$

This is the simplest possible quantum system, yet it reveals deep features of quantum mechanics.

### 2. Time-Independent Schrödinger Equation

The TISE is the eigenvalue equation for energy:

$$\hat{H}\psi = E\psi$$

For the free particle:

$$\boxed{-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi}$$

Rearranging:

$$\frac{d^2\psi}{dx^2} = -\frac{2mE}{\hbar^2}\psi$$

### 3. Solving the TISE

#### Case 1: E > 0 (Physical Solutions)

Define the **wave number**:

$$k = \frac{\sqrt{2mE}}{\hbar} \quad \Rightarrow \quad E = \frac{\hbar^2 k^2}{2m}$$

The differential equation becomes:

$$\frac{d^2\psi}{dx^2} = -k^2\psi$$

This is the harmonic oscillator equation with general solution:

$$\boxed{\psi(x) = Ae^{ikx} + Be^{-ikx}}$$

Or equivalently in trigonometric form:

$$\psi(x) = C\cos(kx) + D\sin(kx)$$

#### Case 2: E < 0 (Unphysical)

If E < 0, define κ² = -2mE/ℏ² > 0:

$$\frac{d^2\psi}{dx^2} = \kappa^2\psi$$

Solution: ψ(x) = Ae^{κx} + Be^{-κx}

This **diverges** as x → ±∞ and cannot be normalized. Therefore:

$$\boxed{\text{Free particle energies must satisfy } E \geq 0}$$

#### Case 3: E = 0 (Trivial)

If E = 0: d²ψ/dx² = 0 → ψ(x) = ax + b

This also diverges and gives ψ = 0 (trivial solution).

### 4. Plane Wave Solutions

The fundamental solutions are **plane waves**:

$$\boxed{\psi_k(x) = Ae^{ikx}}$$

where k can be **positive or negative**:
- k > 0: Wave traveling in +x direction
- k < 0: Wave traveling in -x direction

**Physical interpretation:**
- |ψ|² = |A|² is constant everywhere
- The particle is equally likely to be found anywhere
- This represents a particle with **definite momentum** p = ℏk

### 5. The Dispersion Relation

The relationship between energy and wave number:

$$\boxed{E = \frac{\hbar^2 k^2}{2m} = \frac{p^2}{2m}}$$

This is the **dispersion relation** for a free particle. Key features:

1. **Parabolic:** E ∝ k² (unlike light where E ∝ k)
2. **Symmetric in k:** States with ±k have the same energy (degeneracy)
3. **Classical correspondence:** Matches classical kinetic energy E = p²/2m

The angular frequency ω relates to energy via E = ℏω:

$$\omega(k) = \frac{\hbar k^2}{2m}$$

### 6. Continuous Spectrum

Unlike bound systems (harmonic oscillator, hydrogen atom), the free particle has a **continuous spectrum**:

- Any E ≥ 0 is allowed
- No boundary conditions to enforce quantization
- Energy can take any non-negative real value

**Why no quantization?**
- Quantization arises from boundary conditions
- Free particle: ψ must be finite, but no confinement
- No "box" means no standing wave conditions
- Result: continuous rather than discrete energies

### 7. Two-Fold Degeneracy

For each energy E > 0, there are **two independent solutions**:

$$\psi_{+k}(x) = e^{ikx} \quad \text{and} \quad \psi_{-k}(x) = e^{-ikx}$$

where k = √(2mE)/ℏ.

These correspond to:
- Right-moving wave (p = +ℏk)
- Left-moving wave (p = -ℏk)

This degeneracy reflects **parity symmetry**: the Hamiltonian is invariant under x → -x.

### 8. Momentum Eigenstates

Plane waves are eigenstates of momentum:

$$\hat{p}\psi_k = -i\hbar\frac{d}{dx}e^{ikx} = \hbar k \cdot e^{ikx}$$

$$\boxed{\hat{p}\psi_k = p\psi_k \quad \text{with} \quad p = \hbar k}$$

This is the **de Broglie relation**: momentum and wavelength are connected via p = ℏk = h/λ.

---

## Quantum Computing Connection

### Continuous-Variable Quantum Computing

In **continuous-variable (CV) quantum computing**, information is encoded in the position and momentum of quantum harmonic oscillators (often photonic modes). The free particle eigenstates serve as the basis:

1. **Momentum eigenstates |p⟩:** Logical basis for CV systems
2. **Position eigenstates |x⟩:** Complementary basis
3. **Gaussian states:** Practical approximations to these idealized states

### Quantum Communication

Free particle wave functions describe:
- Photon propagation in optical fibers
- Matter wave transmission in atom interferometers
- Electron beams in quantum electron microscopes

---

## Worked Examples

### Example 1: Verifying Plane Wave Solution

**Problem:** Verify that ψ(x) = Ae^{ikx} satisfies the TISE with E = ℏ²k²/2m.

**Solution:**

Step 1: Compute the first derivative:
$$\frac{d\psi}{dx} = ikAe^{ikx}$$

Step 2: Compute the second derivative:
$$\frac{d^2\psi}{dx^2} = (ik)^2 Ae^{ikx} = -k^2 Ae^{ikx} = -k^2\psi$$

Step 3: Substitute into TISE:
$$-\frac{\hbar^2}{2m}(-k^2\psi) = \frac{\hbar^2 k^2}{2m}\psi = E\psi \quad \checkmark$$

The plane wave satisfies the TISE with the dispersion relation E = ℏ²k²/2m.

### Example 2: Energy and Wavelength of an Electron

**Problem:** An electron has kinetic energy 1 eV. Find its (a) wave number k, (b) wavelength λ, and (c) momentum p.

**Solution:**

Given: E = 1 eV = 1.6 × 10⁻¹⁹ J, m = 9.11 × 10⁻³¹ kg, ℏ = 1.055 × 10⁻³⁴ J·s

(a) Wave number:
$$k = \frac{\sqrt{2mE}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$k = \frac{5.4 \times 10^{-25}}{1.055 \times 10^{-34}} = 5.12 \times 10^9 \text{ m}^{-1}$$

$$\boxed{k = 5.12 \times 10^9 \text{ m}^{-1}}$$

(b) Wavelength:
$$\lambda = \frac{2\pi}{k} = \frac{2\pi}{5.12 \times 10^9} = 1.23 \times 10^{-9} \text{ m} = 1.23 \text{ nm}$$

$$\boxed{\lambda = 1.23 \text{ nm}}$$

(c) Momentum:
$$p = \hbar k = 1.055 \times 10^{-34} \times 5.12 \times 10^9 = 5.4 \times 10^{-25} \text{ kg·m/s}$$

$$\boxed{p = 5.4 \times 10^{-25} \text{ kg·m/s}}$$

### Example 3: Superposition of Plane Waves

**Problem:** Show that ψ(x) = A cos(kx) can be written as a superposition of plane waves and find the momentum expectation value.

**Solution:**

Step 1: Express cosine as exponentials:
$$\cos(kx) = \frac{e^{ikx} + e^{-ikx}}{2}$$

So:
$$\psi(x) = \frac{A}{2}e^{ikx} + \frac{A}{2}e^{-ikx}$$

Step 2: This is an equal superposition of:
- Right-moving wave with momentum p = +ℏk
- Left-moving wave with momentum p = -ℏk

Step 3: Momentum expectation value:

By symmetry, ⟨p⟩ = 0.

Alternatively, compute directly (assuming proper normalization):
$$\langle p \rangle = \frac{1}{2}(+\hbar k) + \frac{1}{2}(-\hbar k) = 0$$

$$\boxed{\langle p \rangle = 0}$$

The particle has no net momentum—equal probability of moving left or right.

---

## Practice Problems

### Level 1: Direct Application

1. **Wave number calculation:** Find the wave number k for a neutron (m = 1.675 × 10⁻²⁷ kg) with energy E = 0.025 eV (thermal neutron).

2. **Momentum verification:** Show that ψ(x) = e^{-ikx} is a momentum eigenstate and find the eigenvalue.

3. **Energy from wavelength:** A particle has de Broglie wavelength λ = 0.5 nm. Express its energy in terms of its mass.

### Level 2: Intermediate

4. **Standing wave decomposition:** The wave function ψ(x) = A sin(kx) represents a standing wave.
   (a) Write it as a superposition of traveling waves.
   (b) What is ⟨p⟩? What is ⟨p²⟩?

5. **Relativistic corrections:** For what electron energy does the relativistic momentum p = γmv differ from the non-relativistic p = √(2mE) by 1%?

6. **Comparing particles:** An electron and a proton have the same kinetic energy. What is the ratio of their de Broglie wavelengths?

### Level 3: Challenging

7. **Time-dependent solution:** Write the full time-dependent wave function Ψ(x,t) for the plane wave and show that |Ψ|² is independent of both x and t.

8. **Probability current:** For ψ = Ae^{ikx}, calculate the probability current:
   $$j = \frac{\hbar}{2mi}\left(\psi^*\frac{d\psi}{dx} - \psi\frac{d\psi^*}{dx}\right)$$
   Interpret the result physically.

9. **Energy uncertainty:** A free particle is described by ψ(x) = A cos(kx). Calculate ⟨H⟩ and ⟨H²⟩, then find ΔE. Explain why ΔE = 0 even though this is a superposition.

---

## Computational Lab: Visualizing Plane Waves

```python
"""
Day 365 Computational Lab: Free Particle Plane Waves
=====================================================
Visualizing plane wave solutions to the free particle TISE
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical constants (atomic units for convenience)
hbar = 1.0
m = 1.0

def plane_wave(x, k):
    """
    Plane wave solution to free particle TISE.

    Parameters:
    -----------
    x : array-like
        Position coordinates
    k : float
        Wave number (can be positive or negative)

    Returns:
    --------
    psi : complex array
        Wave function values
    """
    return np.exp(1j * k * x)

def energy(k, m=1.0, hbar=1.0):
    """Dispersion relation: E = hbar^2 k^2 / 2m"""
    return hbar**2 * k**2 / (2 * m)

# Create spatial grid
x = np.linspace(-10, 10, 1000)

# Plot plane waves for different k values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Different wave numbers
k_values = [1.0, 2.0, 3.0, -2.0]
titles = ['k = 1 (slow, rightward)', 'k = 2 (medium, rightward)',
          'k = 3 (fast, rightward)', 'k = -2 (leftward)']

for ax, k, title in zip(axes.flatten(), k_values, titles):
    psi = plane_wave(x, k)

    ax.plot(x, np.real(psi), 'b-', label='Re(ψ)', linewidth=1.5)
    ax.plot(x, np.imag(psi), 'r-', label='Im(ψ)', linewidth=1.5)
    ax.plot(x, np.abs(psi)**2, 'k--', label='|ψ|²', linewidth=2)

    ax.set_xlabel('x')
    ax.set_ylabel('ψ(x)')
    ax.set_title(f'{title}\nE = {energy(k):.2f}, λ = {2*np.pi/abs(k):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('plane_waves.png', dpi=150)
plt.show()

print("="*60)
print("PLANE WAVE PROPERTIES")
print("="*60)
for k in k_values:
    E = energy(k)
    lam = 2 * np.pi / abs(k)
    p = hbar * k
    print(f"k = {k:5.1f} | E = {E:6.2f} | λ = {lam:5.2f} | p = {p:5.1f}")

# Dispersion relation plot
fig, ax = plt.subplots(figsize=(8, 6))

k_range = np.linspace(-5, 5, 500)
E_range = energy(k_range)

ax.plot(k_range, E_range, 'b-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Mark specific points
for k in [1, 2, 3, -2]:
    ax.plot(k, energy(k), 'ro', markersize=10)
    ax.annotate(f'k={k}', (k, energy(k)), textcoords="offset points",
                xytext=(10,10), fontsize=10)

ax.set_xlabel('Wave number k', fontsize=12)
ax.set_ylabel('Energy E = ℏ²k²/2m', fontsize=12)
ax.set_title('Dispersion Relation for Free Particle', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dispersion_relation.png', dpi=150)
plt.show()

# Animation: Traveling plane wave
fig, ax = plt.subplots(figsize=(10, 5))

k0 = 2.0
omega = hbar * k0**2 / (2 * m)  # ω = E/ℏ

line_real, = ax.plot([], [], 'b-', label='Re(Ψ)', linewidth=2)
line_imag, = ax.plot([], [], 'r-', label='Im(Ψ)', linewidth=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('Ψ(x,t)')
ax.set_title('Traveling Plane Wave: Ψ(x,t) = exp[i(kx - ωt)]')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

def init():
    line_real.set_data([], [])
    line_imag.set_data([], [])
    time_text.set_text('')
    return line_real, line_imag, time_text

def animate(frame):
    t = frame * 0.05
    psi = np.exp(1j * (k0 * x - omega * t))

    line_real.set_data(x, np.real(psi))
    line_imag.set_data(x, np.imag(psi))
    time_text.set_text(f't = {t:.2f}')

    return line_real, line_imag, time_text

anim = FuncAnimation(fig, animate, init_func=init, frames=100,
                     interval=50, blit=True)

# Save animation (uncomment to save)
# anim.save('plane_wave_animation.mp4', writer='ffmpeg', fps=20)

plt.show()

# Superposition of plane waves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Standing wave: cos(kx) = (e^{ikx} + e^{-ikx})/2
k = 2.0
psi_standing = np.cos(k * x)
psi_right = 0.5 * np.exp(1j * k * x)
psi_left = 0.5 * np.exp(-1j * k * x)

ax = axes[0]
ax.plot(x, psi_standing, 'k-', label='cos(kx)', linewidth=2)
ax.plot(x, np.real(psi_right), 'b--', label='Re(e^{ikx}/2)', alpha=0.7)
ax.plot(x, np.real(psi_left), 'r--', label='Re(e^{-ikx}/2)', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('ψ(x)')
ax.set_title('Standing Wave as Superposition')
ax.legend()
ax.grid(True, alpha=0.3)

# Probability density comparison
ax = axes[1]
ax.plot(x, np.abs(np.exp(1j * k * x))**2, 'b-',
        label='|e^{ikx}|² (traveling)', linewidth=2)
ax.plot(x, np.cos(k * x)**2, 'r-',
        label='cos²(kx) (standing)', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('|ψ|²')
ax.set_title('Probability Density Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('standing_vs_traveling.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)
print("""
1. Plane waves have CONSTANT probability density |ψ|² = 1
   - Particle equally likely everywhere (completely delocalized)

2. Real and imaginary parts oscillate with wavelength λ = 2π/k
   - Higher k → shorter wavelength → higher energy

3. Dispersion relation is PARABOLIC: E ∝ k²
   - Different from light where E ∝ k (linear)
   - This causes wave packet spreading (Day 369)

4. Standing waves (cos, sin) have VARYING probability density
   - Nodes where particle is never found
   - Equal superposition of left and right momentum
""")
```

---

## Summary

### Key Formulas Table

| Quantity | Formula | Notes |
|----------|---------|-------|
| TISE | $$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$ | V = 0 everywhere |
| Plane wave | $$\psi_k(x) = Ae^{ikx}$$ | Unnormalized |
| Dispersion | $$E = \frac{\hbar^2 k^2}{2m}$$ | Parabolic in k |
| de Broglie | $$p = \hbar k = \frac{h}{\lambda}$$ | Wave-particle duality |
| Wave number | $$k = \frac{\sqrt{2mE}}{\hbar}$$ | From energy |
| Wavelength | $$\lambda = \frac{2\pi}{k} = \frac{h}{p}$$ | de Broglie wavelength |

### Main Takeaways

1. **Free particle solutions are plane waves** ψ_k(x) = Ae^{ikx}
2. **Energy spectrum is continuous** (E ≥ 0, any value allowed)
3. **Two-fold degeneracy:** ±k have same energy (left/right moving)
4. **Plane waves are momentum eigenstates** with p = ℏk
5. **Probability density is uniform** — particle completely delocalized
6. **Dispersion relation** E = ℏ²k²/2m connects energy and wave number

---

## Daily Checklist

- [ ] I can derive the TISE for a free particle
- [ ] I understand why plane waves are solutions
- [ ] I can calculate k, λ, p, E relationships
- [ ] I understand the continuous spectrum concept
- [ ] I can explain the two-fold degeneracy
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 366

Tomorrow we address a crucial issue: **plane waves cannot be normalized** in the usual sense! We'll learn two approaches:
1. **Box normalization:** Confine to large box L, take L → ∞
2. **Delta-function normalization:** ⟨k|k'⟩ = δ(k - k')

These techniques are essential for working with continuous spectra and lead to the rigorous treatment of momentum eigenstates.

---

*"The plane wave is the quantum mechanical description of a particle with definite momentum but completely uncertain position—the extreme manifestation of the uncertainty principle."*
