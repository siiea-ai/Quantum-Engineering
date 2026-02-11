# Day 474: Selection Rules

## Overview
**Day 474** | Year 1, Month 17, Week 68 | Symmetry Constraints on Transitions

Today we derive selection rules—the symmetry-based constraints that determine which quantum transitions are allowed and which are forbidden.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Electric dipole selection rules |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Higher multipoles and forbidden transitions |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Atomic spectra simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Derive selection rules from matrix element symmetry
2. Apply electric dipole selection rules to hydrogen
3. Understand magnetic dipole and electric quadrupole transitions
4. Calculate relative transition strengths
5. Explain forbidden transitions and when they occur
6. Connect selection rules to spectroscopic observations

---

## Core Content

### Origin of Selection Rules

Transition rates depend on matrix elements:
$$W_{i \to f} \propto |\langle f|V|i\rangle|^2$$

Selection rules arise when **symmetry** forces these matrix elements to vanish.

### Electric Dipole Transitions

The dominant interaction with electromagnetic radiation:
$$V = -\mathbf{d} \cdot \mathbf{E} = -e\mathbf{r} \cdot \mathbf{E}$$

For linearly polarized light (E along z):
$$V_{fi} = -eE_0\langle f|z|i\rangle$$

### Parity Selection Rule

The position operator **r** has **odd parity**:
$$P\mathbf{r}P^{-1} = -\mathbf{r}$$

For the matrix element ⟨f|z|i⟩ to be nonzero:
- If |i⟩ has parity π_i and |f⟩ has parity π_f
- Then π_f × (-1) × π_i = +1
- So: **π_f = -π_i** (parity must change)

$$\boxed{\Delta \ell = \text{odd} \quad \text{(parity selection rule)}}$$

### Angular Momentum Selection Rules

The dipole operator transforms as a **vector** (ℓ = 1 spherical tensor).

Using the Wigner-Eckart theorem:
$$\langle n', \ell', m'|r_q|n, \ell, m\rangle \propto \langle \ell', m'|\ell, 1; m, q\rangle$$

where q = 0, ±1 labels the spherical components.

**Selection rules from Clebsch-Gordan coefficients:**

$$\boxed{\Delta \ell = \pm 1}$$
$$\boxed{\Delta m = 0, \pm 1}$$

### Polarization and Δm

| Polarization | Δm | Physical meaning |
|--------------|-----|------------------|
| Linear (z) | 0 | π-transition |
| Circular (σ⁺) | +1 | Left-handed photon |
| Circular (σ⁻) | -1 | Right-handed photon |

### Summary: Electric Dipole (E1) Rules

For hydrogen-like atoms:
$$\boxed{\begin{aligned}
\Delta \ell &= \pm 1 \\
\Delta m_\ell &= 0, \pm 1 \\
\Delta m_s &= 0 \\
\Delta j &= 0, \pm 1 \text{ (but not } 0 \to 0\text{)}
\end{aligned}}$$

---

## Higher Multipole Transitions

### Multipole Expansion

The full interaction:
$$V = -\mathbf{d} \cdot \mathbf{E} - \boldsymbol{\mu} \cdot \mathbf{B} - Q_{ij}\nabla_i E_j + \cdots$$

| Type | Notation | Typical Rate | Selection Rules |
|------|----------|--------------|-----------------|
| Electric dipole | E1 | ~10⁸ s⁻¹ | Δℓ = ±1 |
| Magnetic dipole | M1 | ~10³ s⁻¹ | Δℓ = 0, Δm = 0, ±1 |
| Electric quadrupole | E2 | ~10³ s⁻¹ | Δℓ = 0, ±2 |

### Magnetic Dipole (M1) Selection Rules

The magnetic moment operator:
$$\boldsymbol{\mu} = -\frac{e}{2m_e}(\mathbf{L} + 2\mathbf{S})$$

This is an **axial vector** (even parity), so:
$$\boxed{\begin{aligned}
\Delta \ell &= 0 \\
\Delta j &= 0, \pm 1 \\
\text{Parity} &= \text{unchanged}
\end{aligned}}$$

### Electric Quadrupole (E2) Selection Rules

The quadrupole operator transforms as a rank-2 tensor:
$$\boxed{\begin{aligned}
\Delta \ell &= 0, \pm 2 \\
\Delta m &= 0, \pm 1, \pm 2 \\
\text{Parity} &= \text{unchanged}
\end{aligned}}$$

### Relative Transition Strengths

$$\frac{W_{M1}}{W_{E1}} \sim \left(\frac{v}{c}\right)^2 \sim \alpha^2 \approx 5 \times 10^{-5}$$

$$\frac{W_{E2}}{W_{E1}} \sim \left(\frac{a_0}{\lambda}\right)^2 \sim \alpha^2$$

---

## Forbidden Transitions

### Why "Forbidden"?

"Forbidden" transitions are those forbidden for E1 but allowed for higher multipoles:
- Much slower (10³-10⁵ times)
- Important in low-density environments (nebulae, corona)

### Metastable States

States that cannot decay via E1 are **metastable**:
- Hydrogen 2s: Cannot reach 1s (Δℓ = 0)
- Lifetime ~0.14 s (two-photon decay)

### The 21-cm Line

Hydrogen hyperfine transition (F = 1 → F = 0):
- Magnetic dipole (M1) transition
- Lifetime ~10⁷ years!
- Crucial for radio astronomy

---

## Quantum Computing Connection

### Selection Rules in Qubit Design

**Transmon qubits:** Transitions between adjacent levels only
- |0⟩ ↔ |1⟩: Allowed (E1)
- |0⟩ ↔ |2⟩: Suppressed (requires two-photon)

**Fluxonium:** Selection rules depend on flux bias
- Sweet spot: Enhanced isolation

### Qubit Leakage

Selection rule violations cause **leakage**:
$$|1\rangle \to |2\rangle \text{ (unwanted)}$$

DRAG pulses exploit selection rules to minimize leakage.

### Trapped Ion Qubits

Optical qubits use E2 or M1 transitions:
- Extremely narrow linewidths
- Long coherence times
- Examples: Ca⁺ (E2), Yb⁺ (E2)

---

## Worked Examples

### Example 1: Allowed Transitions in Hydrogen

**Problem:** Which transitions are allowed for the Balmer series (n → 2)?

**Solution:**

For electric dipole: Δℓ = ±1

n = 3: ℓ can be 0, 1, 2
- 3s → 2p: ✓ (Δℓ = +1)
- 3p → 2s: ✓ (Δℓ = -1)
- 3d → 2p: ✓ (Δℓ = -1)

All these contribute to Hα (656 nm).

**Forbidden:**
- 3s → 2s: ✗ (Δℓ = 0)
- 3d → 2s: ✗ (Δℓ = -2)

### Example 2: Zeeman Transitions

**Problem:** A hydrogen atom in a magnetic field B along z has the 2p state split into m = -1, 0, +1. Which transitions to 1s (m = 0) are allowed and with what polarization?

**Solution:**

Selection rule: Δm = 0, ±1

From 2p:
- m = +1 → m = 0: Δm = -1 (σ⁻, right circular)
- m = 0 → m = 0: Δm = 0 (π, linear)
- m = -1 → m = 0: Δm = +1 (σ⁺, left circular)

All three transitions allowed with different polarizations (normal Zeeman effect).

### Example 3: Metastable Helium

**Problem:** Why is He(2¹S₀) metastable but He(2¹P₁) is not?

**Solution:**

Ground state: He(1¹S₀) with ℓ = 0

For 2¹P₁ → 1¹S₀:
- Δℓ = 1 - 0 = -1 ✓
- E1 allowed, lifetime ~10⁻⁹ s

For 2¹S₀ → 1¹S₀:
- Δℓ = 0 - 0 = 0 ✗
- E1 forbidden
- Must decay via M1 or two-photon
- Lifetime ~20 ms (metastable)

---

## Practice Problems

### Problem Set 68.5

**Direct Application:**
1. For sodium (ground state 3s), list all allowed E1 transitions from the 3d state.

2. Which of these hydrogen transitions are E1 allowed?
   - 4f → 3d
   - 4d → 2s
   - 4p → 1s
   - 5g → 4f

3. In the Zeeman effect, how many spectral lines appear for the 2p → 1s transition? What are their polarizations?

**Intermediate:**
4. Show that the 2s → 1s transition in hydrogen is forbidden for all multipoles (E1, M1, E2). How does it decay?

5. Calculate the ratio of E2 to E1 transition rates for hydrogen. Use a₀/λ ≈ α.

6. For the calcium ion Ca⁺ (used in trapped ion qubits), the 4²S₁/₂ ↔ 3²D₅/₂ transition is used. What type of transition is this? Why is it useful?

**Challenging:**
7. Derive the electric dipole selection rules using the commutator [L_z, z] = 0 and [L², [L², z]].

8. In a magnetic field, the 2p state of hydrogen splits. Calculate the relative intensities of the three Zeeman components for unpolarized observation perpendicular to B.

9. The "forbidden" [O III] lines at 495.9 nm and 500.7 nm are prominent in planetary nebulae. Research and explain why these M1/E2 transitions are visible despite being "forbidden."

---

## Computational Lab

```python
"""
Day 474 Lab: Selection Rules and Atomic Spectra
Visualizes allowed transitions and spectral patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Hydrogen energy levels
def hydrogen_energy(n):
    """Energy in eV (relative to ionization)"""
    return -13.6 / n**2

# Generate allowed transitions
def is_e1_allowed(n1, l1, n2, l2):
    """Check if E1 transition is allowed"""
    return abs(l2 - l1) == 1

def generate_transitions(n_max):
    """Generate all allowed E1 transitions up to n_max"""
    transitions = []
    for n1 in range(1, n_max + 1):
        for l1 in range(n1):
            for n2 in range(n1 + 1, n_max + 1):
                for l2 in range(n2):
                    if is_e1_allowed(n1, l1, n2, l2):
                        E1 = hydrogen_energy(n1)
                        E2 = hydrogen_energy(n2)
                        wavelength = 1240 / abs(E2 - E1)  # nm
                        transitions.append({
                            'n1': n1, 'l1': l1,
                            'n2': n2, 'l2': l2,
                            'E1': E1, 'E2': E2,
                            'wavelength': wavelength,
                            'series': get_series_name(n1)
                        })
    return transitions

def get_series_name(n_lower):
    """Return series name based on lower level"""
    series = {1: 'Lyman', 2: 'Balmer', 3: 'Paschen', 4: 'Brackett'}
    return series.get(n_lower, f'n={n_lower}')

def l_to_letter(l):
    """Convert l quantum number to letter"""
    letters = ['s', 'p', 'd', 'f', 'g', 'h']
    return letters[l] if l < len(letters) else f'l={l}'

# Visualization 1: Energy level diagram with transitions
print("=" * 60)
print("HYDROGEN ENERGY LEVELS AND ALLOWED TRANSITIONS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 10))

n_max = 5
transitions = generate_transitions(n_max)

# Plot energy levels
for n in range(1, n_max + 1):
    E = hydrogen_energy(n)
    for l in range(n):
        x_pos = l + 0.1 * (n - 1)  # Slight offset by n
        ax.hlines(E, x_pos - 0.3, x_pos + 0.3, colors='black', linewidth=2)
        ax.text(x_pos, E + 0.3, f'{n}{l_to_letter(l)}', ha='center', fontsize=9)

# Plot some transitions (Lyman and Balmer only for clarity)
colors = {'Lyman': 'purple', 'Balmer': 'red', 'Paschen': 'brown'}
for t in transitions:
    if t['series'] in ['Lyman', 'Balmer'] and t['n2'] <= 4:
        x1 = t['l1'] + 0.1 * (t['n1'] - 1)
        x2 = t['l2'] + 0.1 * (t['n2'] - 1)
        color = colors.get(t['series'], 'gray')
        ax.annotate('', xy=(x1, t['E1']), xytext=(x2, t['E2']),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))

ax.set_xlabel('Orbital angular momentum ℓ', fontsize=12)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Hydrogen Energy Levels and E1 Transitions', fontsize=14)
ax.set_xlim(-0.5, 5)
ax.set_ylim(-15, 1)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='purple', label='Lyman (→ n=1)'),
                   Line2D([0], [0], color='red', label='Balmer (→ n=2)')]
ax.legend(handles=legend_elements, loc='lower right')

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hydrogen_transitions.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Selection rules table
print("\n" + "=" * 60)
print("SELECTION RULES SUMMARY")
print("=" * 60)

# Print allowed transitions for n=3 → n=2
print("\nTransitions n=3 → n=2 (Hα line):")
print("-" * 50)
print(f"{'Transition':<20} {'Δℓ':<8} {'Allowed?':<10}")
print("-" * 50)

for l2 in range(3):  # n=3: l=0,1,2
    for l1 in range(2):  # n=2: l=0,1
        delta_l = l1 - l2
        allowed = "Yes (E1)" if abs(delta_l) == 1 else "No"
        print(f"3{l_to_letter(l2)} → 2{l_to_letter(l1):<12} {delta_l:<8} {allowed:<10}")

# Visualization 3: Zeeman splitting pattern
print("\n" + "=" * 60)
print("ZEEMAN EFFECT: 2p → 1s TRANSITION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Energy levels
ax = axes[0]
B = 1  # Arbitrary field strength

# 1s state (m=0 only)
ax.hlines(-13.6, 0.7, 1.3, colors='blue', linewidth=3, label='1s (m=0)')

# 2p state (m=-1, 0, +1)
E_2p_base = -3.4
m_values = [-1, 0, 1]
zeeman_shift = 0.5  # Arbitrary units for visualization

for m in m_values:
    E = E_2p_base + m * zeeman_shift
    ax.hlines(E, 1.7, 2.3, colors='red', linewidth=3)
    ax.text(2.4, E, f'm={m}', va='center', fontsize=11)

# Transitions
transitions_zeeman = [
    (1, 0, 'σ⁻', 'blue'),   # m=+1 → m=0
    (0, 0, 'π', 'green'),   # m=0 → m=0
    (-1, 0, 'σ⁺', 'red')    # m=-1 → m=0
]

for m_upper, m_lower, pol, color in transitions_zeeman:
    y_upper = E_2p_base + m_upper * zeeman_shift
    y_lower = -13.6
    ax.annotate('', xy=(1.3, y_lower), xytext=(1.7, y_upper),
                arrowprops=dict(arrowstyle='->', color=color, linewidth=2))
    ax.text(1.5, (y_upper + y_lower)/2 + 0.5, pol, ha='center',
            fontsize=10, color=color)

ax.set_xlim(0.5, 3)
ax.set_ylim(-15, -2)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Zeeman Splitting of 2p → 1s', fontsize=14)
ax.set_xticks([1, 2])
ax.set_xticklabels(['1s', '2p'])
ax.grid(True, alpha=0.3, axis='y')

# Right: Spectral pattern
ax = axes[1]
freq_0 = 1  # Central frequency (normalized)
freq_shift = 0.1  # Zeeman shift

# No field
ax.axvline(freq_0, color='black', linewidth=2, linestyle='--', alpha=0.5)
ax.text(freq_0, 0.95, 'No field', ha='center', fontsize=10)

# With field
freqs = [freq_0 - freq_shift, freq_0, freq_0 + freq_shift]
colors = ['red', 'green', 'blue']
labels = ['σ⁺', 'π', 'σ⁻']

for f, c, l in zip(freqs, colors, labels):
    ax.axvline(f, color=c, linewidth=3, label=l)

ax.set_xlim(0.7, 1.3)
ax.set_ylim(0, 1)
ax.set_xlabel('Frequency (normalized)', fontsize=12)
ax.set_title('Zeeman Triplet Spectrum', fontsize=14)
ax.legend(loc='upper right')
ax.set_yticks([])

plt.tight_layout()
plt.savefig('zeeman_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 4: Transition strength comparison
print("\n" + "=" * 60)
print("MULTIPOLE TRANSITION STRENGTH COMPARISON")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

multipoles = ['E1', 'M1', 'E2', 'M2', 'E3']
# Typical rates for hydrogen-like transitions (relative to E1)
rates = [1, 5e-5, 5e-5, 2.5e-9, 2.5e-9]
lifetimes = [1e-8, 2e-4, 2e-4, 4, 4]  # seconds

x = np.arange(len(multipoles))
bars = ax.bar(x, rates, color=['blue', 'orange', 'green', 'red', 'purple'])

ax.set_yscale('log')
ax.set_ylabel('Relative Transition Rate', fontsize=12)
ax.set_xlabel('Multipole Type', fontsize=12)
ax.set_title('Relative Strengths of Multipole Transitions', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(multipoles)
ax.set_ylim(1e-10, 10)

# Add lifetime labels
for i, (bar, tau) in enumerate(zip(bars, lifetimes)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 2,
            f'τ ~ {tau:.0e} s', ha='center', fontsize=9)

ax.axhline(1, color='blue', linestyle='--', alpha=0.3)
ax.text(4.5, 1.5, 'E1 reference', fontsize=9)

plt.tight_layout()
plt.savefig('multipole_strengths.png', dpi=150, bbox_inches='tight')
plt.show()

# Print wavelength table
print("\n" + "=" * 60)
print("HYDROGEN SPECTRAL LINES (E1 allowed)")
print("=" * 60)
print(f"\n{'Series':<10} {'Transition':<15} {'Wavelength (nm)':<15} {'Region':<15}")
print("-" * 55)

for t in sorted(transitions, key=lambda x: x['wavelength']):
    if t['n2'] <= 5 and t['wavelength'] < 2000:
        trans_str = f"{t['n2']}{l_to_letter(t['l2'])} → {t['n1']}{l_to_letter(t['l1'])}"
        region = "UV" if t['wavelength'] < 400 else ("Visible" if t['wavelength'] < 700 else "IR")
        print(f"{t['series']:<10} {trans_str:<15} {t['wavelength']:<15.1f} {region:<15}")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. E1 selection rules: Δℓ = ±1, Δm = 0, ±1
2. Parity must change for E1 transitions
3. M1 and E2 have same order of magnitude (~α² × E1)
4. "Forbidden" transitions become important at low densities
5. Metastable states have no allowed E1 decay paths
6. Selection rules determine qubit transition addressability
""")
```

---

## Summary

### Key Formulas

| Selection Rule | E1 | M1 | E2 |
|----------------|----|----|-----|
| Δℓ | ±1 | 0 | 0, ±2 |
| Δm | 0, ±1 | 0, ±1 | 0, ±1, ±2 |
| Parity | Changes | Same | Same |
| Δj | 0, ±1 | 0, ±1 | 0, ±1, ±2 |

### Relative Strengths

$$\frac{W_{M1}}{W_{E1}} \sim \frac{W_{E2}}{W_{E1}} \sim \alpha^2 \approx 5 \times 10^{-5}$$

### Main Takeaways

1. **Selection rules** arise from symmetry of transition operators
2. **E1 dominates** for most atomic transitions
3. **Forbidden transitions** are allowed by higher multipoles
4. **Metastable states** cannot decay via E1
5. **Polarization** determines Δm selection

---

## Daily Checklist

- [ ] I can derive E1 selection rules from parity and angular momentum
- [ ] I know the selection rules for M1 and E2 transitions
- [ ] I can predict which spectral lines appear in atomic spectra
- [ ] I understand why some transitions are "forbidden"
- [ ] I can connect selection rules to qubit design

---

## Preview: Day 475

Tomorrow we study **spontaneous emission**—how atoms emit photons even without external fields, using Fermi's Golden Rule with quantized radiation.

---

**Next:** [Day_475_Saturday.md](Day_475_Saturday.md) — Spontaneous Emission
