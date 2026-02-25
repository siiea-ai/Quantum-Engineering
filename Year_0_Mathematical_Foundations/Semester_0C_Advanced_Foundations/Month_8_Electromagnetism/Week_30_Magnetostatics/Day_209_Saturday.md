# Day 209: Magnetization and Magnetic Materials

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Magnetization & Classification |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Ferromagnetism & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 209, you will be able to:

1. Define magnetization and relate it to bound currents
2. Classify materials as diamagnetic, paramagnetic, or ferromagnetic
3. Calculate susceptibility and relate $\mathbf{B}$, $\mathbf{H}$, and $\mathbf{M}$
4. Explain ferromagnetic hysteresis and domains
5. Understand the quantum origins of magnetism
6. Apply magnetic materials concepts to real-world applications

---

## Core Content

### 1. Magnetization

**Definition:** The magnetization $\mathbf{M}$ is the magnetic dipole moment per unit volume:

$$\boxed{\mathbf{M} = \frac{\sum\boldsymbol{\mu}_i}{\Delta V}}$$

**Units:** A/m (same as $\mathbf{H}$)

If the material has $n$ atoms per volume, each with moment $\boldsymbol{\mu}$:
$$\mathbf{M} = n\langle\boldsymbol{\mu}\rangle$$

### 2. Bound Currents

**A magnetized material acts like it contains currents:**

**Volume bound current density:**
$$\boxed{\mathbf{J}_b = \nabla \times \mathbf{M}}$$

**Surface bound current density:**
$$\boxed{\mathbf{K}_b = \mathbf{M} \times \hat{\mathbf{n}}}$$

Compare to electrostatics:
- $\rho_b = -\nabla \cdot \mathbf{P}$ (polarization charges)
- $\mathbf{J}_b = \nabla \times \mathbf{M}$ (magnetization currents)

### 3. The Auxiliary Field H

**Define the $\mathbf{H}$ field** to separate free and bound currents:

$$\boxed{\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}}$$

**Ampere's law for $\mathbf{H}$:**
$$\oint\mathbf{H}\cdot d\boldsymbol{\ell} = I_{f,\text{enc}}$$

$$\nabla \times \mathbf{H} = \mathbf{J}_f$$

where $\mathbf{J}_f$ is the free current density.

**Units of H:** A/m

**Relationship:**
$$\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$$

### 4. Linear Materials

**For most materials** (diamagnets and paramagnets):
$$\mathbf{M} = \chi_m\mathbf{H}$$

where $\chi_m$ is the **magnetic susceptibility**.

**Then:**
$$\mathbf{B} = \mu_0(1 + \chi_m)\mathbf{H} = \mu_0\mu_r\mathbf{H} = \mu\mathbf{H}$$

**Relative permeability:**
$$\boxed{\mu_r = 1 + \chi_m = \frac{\mu}{\mu_0}}$$

### 5. Classification of Magnetic Materials

| Type | $\chi_m$ | $\mu_r$ | Response |
|------|----------|---------|----------|
| Diamagnetic | $\sim -10^{-5}$ | $< 1$ | Opposes applied field |
| Paramagnetic | $\sim +10^{-5}$ to $+10^{-3}$ | $> 1$ | Aligns with field |
| Ferromagnetic | $\sim 10^2$ to $10^5$ | $\gg 1$ | Strong alignment, hysteresis |

### 6. Diamagnetism

**Origin:** Induced currents oppose changes in magnetic flux (Lenz's law at atomic scale).

**Classical model:** An applied field induces a change in orbital motion that creates a magnetic moment opposing the field.

**Key properties:**
- Present in ALL materials
- Weak effect ($\chi_m \sim -10^{-5}$)
- Independent of temperature
- Opposes the applied field ($\mathbf{M}$ anti-parallel to $\mathbf{B}$)

**Examples:**
| Material | $\chi_m$ |
|----------|----------|
| Copper | $-9.6 \times 10^{-6}$ |
| Water | $-9.0 \times 10^{-6}$ |
| Bismuth | $-1.7 \times 10^{-4}$ |
| Graphite | $-1.6 \times 10^{-4}$ |
| Superconductors | $-1$ (perfect diamagnet) |

### 7. Paramagnetism

**Origin:** Atoms/ions have permanent magnetic moments that align with the field.

**Classical model (Langevin):** Thermal energy randomizes moments; applied field creates partial alignment.

**Curie's Law:**
$$\boxed{\chi_m = \frac{C}{T}}$$

where $C$ is the Curie constant and $T$ is temperature.

**Key properties:**
- Requires unpaired electrons
- Weak effect ($\chi_m \sim 10^{-5}$ to $10^{-3}$)
- Temperature dependent (decreases with T)
- Enhances the applied field

**Examples:**
| Material | $\chi_m$ (room temp) |
|----------|---------------------|
| Aluminum | $2.3 \times 10^{-5}$ |
| Oxygen (O$_2$) | $1.9 \times 10^{-6}$ |
| Platinum | $2.6 \times 10^{-4}$ |
| Rare earth ions | $\sim 10^{-3}$ |

### 8. Ferromagnetism

**Origin:** Quantum mechanical exchange interaction aligns neighboring spins.

**Key properties:**
- Strong effect ($\chi_m \sim 10^2$ to $10^5$)
- Non-linear response (hysteresis)
- Spontaneous magnetization below Curie temperature $T_C$
- Forms magnetic domains

**Ferromagnetic elements:** Fe, Co, Ni (and some rare earths)

**Curie temperature (transition to paramagnetic):**
| Material | $T_C$ (K) |
|----------|-----------|
| Iron | 1043 |
| Cobalt | 1388 |
| Nickel | 627 |
| Gadolinium | 293 |

### 9. Hysteresis

**The B-H curve** for ferromagnets shows history dependence:

1. **Virgin curve:** Starting from demagnetized state
2. **Saturation:** $M$ reaches maximum $M_s$
3. **Remanence ($B_r$):** Magnetization remaining when $H = 0$
4. **Coercivity ($H_c$):** Field needed to reduce $B$ to zero
5. **Hysteresis loop:** Energy loss per cycle

**Hard vs soft magnetic materials:**
| Property | Soft (transformer cores) | Hard (permanent magnets) |
|----------|-------------------------|-------------------------|
| $H_c$ | Low ($\sim$ 1 A/m) | High ($\sim 10^5$ A/m) |
| Hysteresis loss | Small | Large |
| Applications | Transformers, motors | Permanent magnets |

### 10. Magnetic Domains

**Why domains form:** Minimizes total magnetic energy.

**Domain wall:** Boundary where magnetization direction changes

- **Bloch wall:** M rotates out of plane
- **Neel wall:** M rotates in plane

**Domain size:** Typically 0.1-1 mm

**Magnetization process:**
1. Domain wall motion (low H)
2. Domain rotation (high H)
3. Saturation (all domains aligned)

### 11. Boundary Conditions

**At interface between two magnetic materials:**

**Normal component:**
$$B_{1\perp} = B_{2\perp}$$

(No magnetic monopoles)

**Tangential component (no free surface current):**
$$H_{1\parallel} = H_{2\parallel}$$

---

## Quantum Mechanics Connection

### Quantum Origin of Diamagnetism: Landau Diamagnetism

**In quantum mechanics,** electrons in a magnetic field occupy Landau levels.

**Landau diamagnetic susceptibility:**
$$\boxed{\chi_{\text{Landau}} = -\frac{1}{3}\chi_{\text{Pauli}}}$$

where $\chi_{\text{Pauli}}$ is the paramagnetic susceptibility from spin.

**Key insight:** Landau diamagnetism is purely quantum — there is no classical diamagnetism from free electrons (Bohr-van Leeuwen theorem).

### Quantum Origin of Paramagnetism: Pauli Paramagnetism

**Free electrons have spin** that can align with the field:

$$\chi_{\text{Pauli}} = \mu_0\mu_B^2 D(E_F)$$

where $D(E_F)$ is the density of states at the Fermi energy.

**Result:** Pauli paramagnetism is temperature-independent (unlike Curie law).

### Exchange Interaction: Origin of Ferromagnetism

**The Heisenberg Hamiltonian:**
$$\hat{H} = -J\sum_{\langle ij\rangle}\mathbf{S}_i \cdot \mathbf{S}_j$$

where $J$ is the exchange constant.

- **$J > 0$:** Ferromagnetic (parallel spins favored)
- **$J < 0$:** Antiferromagnetic (anti-parallel spins favored)

**Origin:** The exchange interaction is purely quantum mechanical, arising from the Pauli exclusion principle and Coulomb interaction.

### Why is Iron Ferromagnetic?

**Hund's rule:** Electrons in partially filled d-orbitals maximize spin.

**Exchange energy** between neighboring atoms with overlapping d-orbitals:
- In Fe, Co, Ni: $J > 0$ due to favorable d-orbital overlap
- In most other elements: d-orbitals don't overlap favorably

**Stoner criterion for ferromagnetism:**
$$U \cdot D(E_F) > 1$$

where $U$ is the exchange energy.

### Superconductivity: Perfect Diamagnetism

**Below $T_c$,** superconductors expel all magnetic flux:
$$\mathbf{B}_{\text{inside}} = 0$$

**Meissner effect:** Surface currents create opposing field.

**London equation:**
$$\nabla^2\mathbf{B} = \frac{\mathbf{B}}{\lambda_L^2}$$

**London penetration depth:** $\lambda_L \sim 50$ nm

This is $\chi_m = -1$ (perfect diamagnet).

### Magnetic Resonance (NMR/MRI)

**Nuclear spins** precess in magnetic field at Larmor frequency:
$$\omega_L = \gamma B$$

**Application:** By applying RF pulses at resonance, we can:
- Flip spins (excitation)
- Measure relaxation (T1, T2 times)
- Create images (MRI)

---

## Worked Examples

### Example 1: Susceptibility Calculation

**Problem:** A paramagnetic salt has $n = 10^{28}$ ions/m$^3$, each with magnetic moment $\mu = 2\mu_B$. At room temperature (300 K), find:
(a) The Curie constant
(b) The susceptibility

**Solution:**

(a) Curie constant:
$$C = \frac{n\mu_0\mu^2}{3k_B}$$

$$C = \frac{(10^{28})(4\pi \times 10^{-7})(2 \times 9.27 \times 10^{-24})^2}{3(1.38 \times 10^{-23})}$$

$$C = 0.104\text{ K}$$

(b) Susceptibility:
$$\chi_m = \frac{C}{T} = \frac{0.104}{300} = 3.5 \times 10^{-4}$$

### Example 2: Ferromagnet in Solenoid

**Problem:** An iron core ($\mu_r = 5000$) fills a solenoid with $n = 1000$ turns/m and $I = 0.1$ A. Find $\mathbf{H}$, $\mathbf{B}$, and $\mathbf{M}$.

**Solution:**

$\mathbf{H}$ depends only on free current:
$$H = nI = 1000 \times 0.1 = 100\text{ A/m}$$

$$B = \mu_0\mu_r H = (4\pi \times 10^{-7})(5000)(100) = 0.628\text{ T}$$

$$M = \chi_m H = (\mu_r - 1)H = 4999 \times 100 = 5 \times 10^5\text{ A/m}$$

Note: $M \gg H$, so $B \approx \mu_0 M$

### Example 3: Boundary Conditions

**Problem:** A magnetic field in air ($\mu_r = 1$) makes angle 30 degrees with the surface of iron ($\mu_r = 1000$). If $B_{\text{air}} = 0.1$ T, find the angle in iron.

**Solution:**

Normal components: $B_{1\perp} = B_{2\perp}$
$$B_{\text{air}}\cos(30°) = B_{\text{iron}}\cos\theta_2$$

Tangential components: $H_{1\parallel} = H_{2\parallel}$
$$\frac{B_{\text{air}}\sin(30°)}{\mu_0} = \frac{B_{\text{iron}}\sin\theta_2}{\mu_0\mu_r}$$

Taking the ratio:
$$\frac{\tan\theta_2}{\tan(30°)} = \mu_r = 1000$$

$$\theta_2 = \arctan(1000 \times \tan 30°) = \arctan(577) \approx 89.9°$$

Field in iron is nearly parallel to surface!

---

## Practice Problems

### Problem 1: Direct Application
A paramagnetic material has susceptibility $\chi_m = 1.5 \times 10^{-4}$. If it is placed in a solenoid producing $H = 10^4$ A/m, find:
(a) The magnetization $M$
(b) The magnetic field $B$

**Answers:** (a) $M = 1.5$ A/m; (b) $B = 12.58$ mT

### Problem 2: Intermediate
A uniformly magnetized sphere of radius $R$ has magnetization $M_0$ along the z-axis. Find the bound surface current and show it equals a uniformly wound coil.

**Hint:** $K_b = M \times \hat{n} = M_0\sin\theta\,\hat{\phi}$

### Problem 3: Challenging
The Curie temperature of nickel is 627 K. Below this temperature, the spontaneous magnetization varies approximately as:
$$M(T) = M_0\left(1 - \frac{T}{T_C}\right)^{1/3}$$

Plot $M/M_0$ vs $T/T_C$ and explain why $M \to 0$ as $T \to T_C$.

### Problem 4: Hysteresis Loss
A transformer core has a hysteresis loop with area $400$ J/m$^3$ per cycle. If the core volume is $0.01$ m$^3$ and operates at 60 Hz, calculate:
(a) Energy lost per cycle
(b) Power dissipated

**Answers:** (a) $W = 4$ J; (b) $P = 240$ W

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
mu0 = 4 * np.pi * 1e-7
mu_B = 9.274e-24
k_B = 1.381e-23

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Susceptibility comparison ==========
ax1 = axes[0, 0]

materials = {
    'Superconductor': -1,
    'Bismuth': -1.7e-4,
    'Copper': -9.6e-6,
    'Air': 0,
    'Aluminum': 2.3e-5,
    'Platinum': 2.6e-4,
    'Soft Iron': 5000,
}

names = list(materials.keys())
chi_vals = list(materials.values())

# Use log scale for display
colors = ['blue' if x < 0 else 'green' if x < 1 else 'red' for x in chi_vals]

# Plot on log scale (handle negatives specially)
y_pos = np.arange(len(names))
ax1.barh(y_pos, [np.sign(x) * np.log10(abs(x) + 1e-10) for x in chi_vals],
         color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.set_xlabel('$\\log_{10}|\\chi_m|$ (sign preserved)')
ax1.set_title('Magnetic Susceptibility of Materials')
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3, axis='x')

# Add text labels
for i, (name, chi) in enumerate(materials.items()):
    ax1.text(np.sign(chi) * np.log10(abs(chi) + 1e-10) + 0.3,
             i, f'{chi:.2e}', va='center', fontsize=9)

# ========== Plot 2: Curie Law (paramagnetism) ==========
ax2 = axes[0, 1]

T = np.linspace(10, 500, 100)
C = 0.1  # Curie constant (K)

chi_curie = C / T

ax2.plot(T, chi_curie * 1e4, 'b-', linewidth=2)
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('$\\chi_m$ ($\\times 10^{-4}$)')
ax2.set_title("Curie's Law: $\\chi_m = C/T$")
ax2.grid(True, alpha=0.3)

# Inset: 1/chi vs T (should be linear)
ax2_inset = ax2.inset_axes([0.5, 0.5, 0.45, 0.4])
ax2_inset.plot(T, 1/chi_curie, 'r-', linewidth=2)
ax2_inset.set_xlabel('T (K)', fontsize=9)
ax2_inset.set_ylabel('$1/\\chi_m$', fontsize=9)
ax2_inset.set_title('$1/\\chi_m$ vs T', fontsize=9)
ax2_inset.grid(True, alpha=0.3)

# ========== Plot 3: Hysteresis loop ==========
ax3 = axes[1, 0]

# Model hysteresis with tanh
H_max = 1000  # A/m
H = np.linspace(-H_max, H_max, 500)

# Parameters
Ms = 1.7e6  # Saturation magnetization for iron (A/m)
Hc = 100    # Coercivity

# Simple hysteresis model (ascending and descending branches)
def M_branch(H, H_shift, Ms, Hc):
    return Ms * np.tanh((H - H_shift) / Hc)

# Ascending (from -Hmax)
H_up = np.linspace(-H_max, H_max, 250)
M_up = M_branch(H_up, -Hc, Ms, Hc*2)

# Descending (from +Hmax)
H_down = np.linspace(H_max, -H_max, 250)
M_down = M_branch(H_down, Hc, Ms, Hc*2)

ax3.plot(H_up, M_up/1e6, 'b-', linewidth=2, label='Ascending')
ax3.plot(H_down, M_down/1e6, 'r-', linewidth=2, label='Descending')

# Mark key points
ax3.plot([0, 0], [M_branch(0, -Hc, Ms, Hc*2)/1e6, M_branch(0, Hc, Ms, Hc*2)/1e6],
         'go', markersize=8, label='Remanence $\\pm M_r$')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

ax3.set_xlabel('H (A/m)')
ax3.set_ylabel('M ($\\times 10^6$ A/m)')
ax3.set_title('Ferromagnetic Hysteresis Loop')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add annotations
ax3.annotate('$M_s$ (saturation)', xy=(800, Ms/1e6), fontsize=10)
ax3.annotate('$H_c$ (coercivity)', xy=(Hc, 0.1), fontsize=10)

# ========== Plot 4: Temperature dependence of ferromagnet ==========
ax4 = axes[1, 1]

T_C = 1043  # Curie temperature of iron (K)
T = np.linspace(0, 1.2 * T_C, 200)

# Spontaneous magnetization (mean field model)
M0 = 1.7e6  # Saturation magnetization at T=0

# Below Tc: M = M0 * (1 - T/Tc)^beta, beta ~ 0.33
M = np.zeros_like(T)
mask = T < T_C
M[mask] = M0 * (1 - T[mask]/T_C)**0.33

# Plot
ax4.plot(T, M/1e6, 'b-', linewidth=2)
ax4.axvline(x=T_C, color='r', linestyle='--', label=f'$T_C$ = {T_C} K')
ax4.fill_between(T[mask], M[mask]/1e6, alpha=0.2)

ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel('Spontaneous Magnetization M ($\\times 10^6$ A/m)')
ax4.set_title('Ferromagnet: M(T) near Curie Point')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1.2*T_C)

# Add phase labels
ax4.text(T_C/2, M0*0.8/1e6, 'Ferromagnetic\nPhase', ha='center', fontsize=12)
ax4.text(T_C*1.1, M0*0.1/1e6, 'Paramagnetic\nPhase', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('day_209_magnetic_materials.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Additional plot: Langevin function ==========
fig2, ax = plt.subplots(figsize=(10, 6))

# Langevin function L(x) = coth(x) - 1/x
def langevin(x):
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-10
    large = ~small
    result[large] = 1/np.tanh(x[large]) - 1/x[large]
    result[small] = x[small]/3  # Taylor expansion for small x
    return result

x = np.linspace(-8, 8, 200)
L_x = langevin(x)

ax.plot(x, L_x, 'b-', linewidth=2, label='Langevin function $L(x)$')
ax.plot(x, x/3, 'r--', linewidth=1.5, label='Linear approx. $x/3$')
ax.axhline(y=1, color='g', linestyle=':', label='Saturation')
ax.axhline(y=-1, color='g', linestyle=':')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

ax.set_xlabel('$x = \\mu B / k_B T$')
ax.set_ylabel('$M/M_s = L(x)$')
ax.set_title('Langevin Function for Paramagnetism\n$M = nμL(μB/k_BT)$')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-8, 8)
ax.set_ylim(-1.2, 1.2)

plt.savefig('day_209_langevin.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Domain wall visualization ==========
fig3, ax = plt.subplots(figsize=(12, 4))

x = np.linspace(0, 10, 100)

# Domain structure (simplified)
# Domain 1: M up, Domain 2: M down, Domain 3: M up
domain_wall_1 = 3
domain_wall_2 = 7

M_z = np.tanh(5*(x - domain_wall_1)) - np.tanh(5*(x - domain_wall_2)) - 1

# Draw spins as arrows
x_arrows = np.linspace(0.5, 9.5, 20)
for xi in x_arrows:
    if xi < domain_wall_1 - 0.5 or xi > domain_wall_2 + 0.5:
        ax.annotate('', xy=(xi, 0.4), xytext=(xi, -0.4),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    elif domain_wall_1 + 0.5 < xi < domain_wall_2 - 0.5:
        ax.annotate('', xy=(xi, -0.4), xytext=(xi, 0.4),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Domain walls
ax.axvline(x=domain_wall_1, color='green', linewidth=3, linestyle='--',
           label='Domain wall')
ax.axvline(x=domain_wall_2, color='green', linewidth=3, linestyle='--')

# Shading
ax.fill_between([0, domain_wall_1], -0.5, 0.5, alpha=0.1, color='blue')
ax.fill_between([domain_wall_1, domain_wall_2], -0.5, 0.5, alpha=0.1, color='red')
ax.fill_between([domain_wall_2, 10], -0.5, 0.5, alpha=0.1, color='blue')

ax.text(domain_wall_1/2, 0.6, 'Domain 1\n(M↑)', ha='center', fontsize=11)
ax.text((domain_wall_1+domain_wall_2)/2, 0.6, 'Domain 2\n(M↓)', ha='center', fontsize=11)
ax.text((domain_wall_2+10)/2, 0.6, 'Domain 3\n(M↑)', ha='center', fontsize=11)

ax.set_xlim(0, 10)
ax.set_ylim(-0.7, 0.8)
ax.set_xlabel('Position')
ax.set_title('Magnetic Domains in Ferromagnet')
ax.legend(loc='lower right')
ax.set_yticks([])

plt.savefig('day_209_domains.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDay 209: Magnetization & Magnetic Materials Complete")
print("="*60)

print("\nSusceptibility classification:")
print(f"  Diamagnetic:    χ_m ~ -10^-5 (opposes field)")
print(f"  Paramagnetic:   χ_m ~ +10^-4 (enhances field)")
print(f"  Ferromagnetic:  χ_m ~ +10^4  (strong enhancement)")

print("\nCurie temperatures:")
curie_temps = {'Iron': 1043, 'Cobalt': 1388, 'Nickel': 627, 'Gadolinium': 293}
for mat, Tc in curie_temps.items():
    print(f"  {mat}: T_C = {Tc} K")

print("\nKey formulas:")
print("  B = μ₀(H + M)")
print("  M = χ_m H (linear materials)")
print("  χ_m = C/T (Curie law for paramagnets)")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\mathbf{M} = n\langle\boldsymbol{\mu}\rangle$ | Magnetization definition |
| $\mathbf{J}_b = \nabla \times \mathbf{M}$ | Bound current density |
| $\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}$ | H field definition |
| $\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$ | B-H-M relation |
| $\mathbf{M} = \chi_m\mathbf{H}$ | Linear material response |
| $\mu_r = 1 + \chi_m$ | Relative permeability |
| $\chi_m = C/T$ | Curie law (paramagnet) |

### Main Takeaways

1. **Magnetization** $\mathbf{M}$ is dipole moment per volume
2. **H field** responds to free currents only; $\mathbf{B}$ includes all sources
3. **Diamagnets** oppose the field (all materials); **paramagnets** enhance it
4. **Ferromagnets** show hysteresis due to domain dynamics
5. **Exchange interaction** (quantum!) causes ferromagnetism
6. **Superconductors** are perfect diamagnets ($\chi_m = -1$)

---

## Daily Checklist

- [ ] I can relate $\mathbf{B}$, $\mathbf{H}$, and $\mathbf{M}$
- [ ] I can classify materials by their magnetic response
- [ ] I understand the origin of diamagnetism and paramagnetism
- [ ] I can explain ferromagnetic hysteresis and domains
- [ ] I understand the quantum origins of magnetism
- [ ] I can apply boundary conditions at magnetic interfaces

---

## Preview: Day 210

Tomorrow is our **Week 30 Review** where we synthesize all magnetostatics topics: Lorentz force, Biot-Savart law, Ampere's law, vector potential, magnetic dipoles, and magnetic materials. Prepare for comprehensive problem sets!

---

*"Ferromagnetism cannot be explained classically — it is an inherently quantum phenomenon arising from the exchange interaction between electron spins."*

---

**Next:** Day 210 — Week 30 Review
