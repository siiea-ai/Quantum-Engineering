# Day 430: Hydrogen Energy Spectrum

## Overview
**Day 430** | Year 1, Month 16, Week 62 | The Balmer Series and Beyond

Today we explore the hydrogen energy spectrum in detail, connecting to spectroscopy and deriving the famous series formulas that revolutionized atomic physics.

---

## Learning Objectives

By the end of today, you will be able to:
1. State and derive the hydrogen energy formula
2. Explain the spectral series (Lyman, Balmer, etc.)
3. Calculate transition wavelengths
4. Understand the ionization continuum
5. Connect to historical development of quantum mechanics
6. Apply to spectroscopic measurements

---

## Core Content

### The Energy Spectrum

$$\boxed{E_n = -\frac{13.6 \text{ eV}}{n^2} = -\frac{E_R}{n^2}}$$

where n = 1, 2, 3, ...

| n | E_n (eV) | E_n/E₁ |
|---|----------|--------|
| 1 | -13.60 | 1 |
| 2 | -3.40 | 1/4 |
| 3 | -1.51 | 1/9 |
| 4 | -0.85 | 1/16 |
| ∞ | 0 | 0 |

### Spectral Series

Transitions from n_i to n_f emit photons with:
$$\boxed{\frac{1}{\lambda} = R_H\left(\frac{1}{n_f^2} - \frac{1}{n_i^2}\right)}$$

**Rydberg constant:**
$$R_H = \frac{m_e e^4}{8\varepsilon_0^2 h^3 c} = 1.097 \times 10^7 \text{ m}^{-1}$$

### Named Series

| Series | Final state n_f | Region | Discovery |
|--------|-----------------|--------|-----------|
| Lyman | 1 | UV | 1906 |
| Balmer | 2 | Visible | 1885 |
| Paschen | 3 | IR | 1908 |
| Brackett | 4 | IR | 1922 |
| Pfund | 5 | Far IR | 1924 |

### The Balmer Series

Transitions to n = 2:
$$\frac{1}{\lambda} = R_H\left(\frac{1}{4} - \frac{1}{n^2}\right), \quad n = 3, 4, 5, ...$$

| Transition | Wavelength | Color |
|------------|------------|-------|
| 3 → 2 (Hα) | 656.3 nm | Red |
| 4 → 2 (Hβ) | 486.1 nm | Cyan |
| 5 → 2 (Hγ) | 434.0 nm | Blue-violet |
| 6 → 2 (Hδ) | 410.2 nm | Violet |
| ∞ → 2 | 364.6 nm | Series limit |

### Series Limits

The series limit corresponds to n_i → ∞:
$$\lambda_{\text{limit}} = \frac{n_f^2}{R_H}$$

Beyond this lies the ionization continuum.

### Ionization Energy

From ground state:
$$E_{\text{ion}} = |E_1| = 13.6 \text{ eV}$$

Corresponding wavelength: λ = 91.2 nm (Lyman limit)

### Fine Structure Preview

The actual spectrum shows fine structure splitting:
- Spin-orbit coupling
- Relativistic corrections
- Breaks the l-degeneracy

---

## Quantum Computing Connection

### Atomic Spectroscopy for Qubits

Precise energy levels enable:
- **Laser cooling:** Red-detuned from transition
- **State preparation:** Optical pumping
- **State readout:** Fluorescence detection

### Hydrogen as Benchmark

VQE calculations on hydrogen:
- Ground state energy: -0.5 Hartree
- First excited states: -0.125 Hartree
- Validates quantum algorithms

---

## Worked Examples

### Example 1: Hα Wavelength

**Problem:** Calculate the wavelength of the Hα (n=3→2) transition.

**Solution:**
$$\frac{1}{\lambda} = R_H\left(\frac{1}{4} - \frac{1}{9}\right) = R_H \cdot \frac{5}{36}$$

$$\lambda = \frac{36}{5 R_H} = \frac{36}{5 \times 1.097 \times 10^7} = 656.3 \text{ nm}$$

This is visible red light!

### Example 2: Ionization from n=2

**Problem:** What photon energy is needed to ionize hydrogen from n=2?

**Solution:**
$$E_{\text{ion}} = |E_2| = \frac{13.6}{4} = 3.4 \text{ eV}$$

$$\lambda = \frac{hc}{E} = \frac{1240 \text{ eV·nm}}{3.4 \text{ eV}} = 365 \text{ nm}$$

This is near-UV light (Balmer series limit).

### Example 3: Lyman α Photon Momentum

**Problem:** What is the recoil momentum when hydrogen emits Lyman α?

**Solution:**
Lyman α: n=2→1, E = 13.6(1 - 1/4) = 10.2 eV

Photon momentum: p = E/c = 10.2 eV/(3×10⁸ m/s)

Converting: p = (10.2 × 1.6×10⁻¹⁹)/(3×10⁸) = 5.4×10⁻²⁷ kg·m/s

Recoil velocity: v = p/m_p ≈ 3.2 m/s

---

## Practice Problems

### Direct Application
1. Calculate the Lyman β (n=3→1) wavelength.
2. What is the series limit for Paschen?
3. How many Balmer lines are in the visible range (400-700 nm)?

### Intermediate
4. Find the energy of the highest-energy photon in the Brackett series.
5. At what temperature does kT equal the ionization energy?
6. Calculate the reduced mass correction to R_H for deuterium.

### Challenging
7. Derive the Rydberg formula from the Bohr model.
8. Calculate the fine structure splitting for n=2 (preview).

---

## Computational Lab

```python
"""
Day 430: Hydrogen Energy Spectrum and Spectral Series
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R_H = 1.097e7  # Rydberg constant (m^-1)
E_R = 13.6     # Rydberg energy (eV)
hc = 1240      # eV·nm

# Energy levels
def E_n(n):
    return -E_R / n**2

# Transition wavelength
def wavelength_nm(n_i, n_f):
    delta_inv_n2 = 1/n_f**2 - 1/n_i**2
    return 1e9 / (R_H * delta_inv_n2)

# Plot energy level diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

ax = axes[0]
for n in range(1, 8):
    E = E_n(n)
    ax.hlines(E, 0.2, 0.8, colors='blue', linewidth=3)
    ax.text(0.85, E, f'n={n}', fontsize=11, va='center')
    ax.text(0.05, E, f'{E:.2f} eV', fontsize=10, va='center')

# Draw some transitions (Balmer series)
for n_i in [3, 4, 5, 6]:
    E_i, E_f = E_n(n_i), E_n(2)
    ax.annotate('', xy=(0.5, E_f), xytext=(0.5, E_i),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.axhline(y=0, color='k', linewidth=0.5)
ax.text(0.5, 0.3, 'Ionization\ncontinuum', ha='center', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(-15, 2)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Hydrogen Energy Levels\n(Balmer transitions shown)', fontsize=14)
ax.set_xticks([])

# Plot spectral series
ax = axes[1]

series_data = {
    'Lyman': (1, range(2, 8), 'purple'),
    'Balmer': (2, range(3, 10), 'blue'),
    'Paschen': (3, range(4, 12), 'green'),
    'Brackett': (4, range(5, 15), 'red'),
}

for name, (n_f, n_range, color) in series_data.items():
    wavelengths = [wavelength_nm(n_i, n_f) for n_i in n_range]
    y_pos = n_f
    for w in wavelengths:
        ax.axvline(x=w, ymin=(y_pos-0.3)/5, ymax=(y_pos+0.3)/5,
                   color=color, linewidth=2, alpha=0.7)
    ax.text(max(wavelengths)*1.1, y_pos, name, fontsize=11, va='center', color=color)

# Mark visible range
ax.axvspan(400, 700, alpha=0.2, color='yellow', label='Visible')

ax.set_xscale('log')
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Series (final n)', fontsize=12)
ax.set_title('Hydrogen Spectral Series', fontsize=14)
ax.set_xlim(50, 5000)
ax.set_ylim(0.5, 5)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('day430_spectrum.png', dpi=150)
plt.show()

# Print Balmer series details
print("=== Balmer Series (n → 2) ===\n")
print("Transition   λ (nm)    Color         Energy (eV)")
print("-" * 55)

colors = {
    (3,2): 'Red (Hα)',
    (4,2): 'Cyan (Hβ)',
    (5,2): 'Blue-violet (Hγ)',
    (6,2): 'Violet (Hδ)',
}

for n_i in range(3, 10):
    w = wavelength_nm(n_i, 2)
    E = hc / w
    color = colors.get((n_i, 2), 'UV' if w < 400 else 'IR')
    print(f"  {n_i} → 2      {w:7.1f}    {color:15s}  {E:.3f}")

# Series limits
print("\n\n=== Series Limits (Ionization from n_f) ===\n")
for n_f in range(1, 6):
    E_ion = E_R / n_f**2
    w_limit = hc / E_ion
    print(f"n = {n_f}: λ_limit = {w_limit:.1f} nm, E_ion = {E_ion:.2f} eV")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Energy | E_n = -13.6 eV/n² |
| Transition | 1/λ = R_H(1/n_f² - 1/n_i²) |
| Rydberg constant | R_H = 1.097×10⁷ m⁻¹ |
| Ionization | E_ion = 13.6/n² eV |

### Key Insights

1. **Discrete spectrum** from quantized energies
2. **Series structure** from common final states
3. **Visible light** only in Balmer series
4. **Ionization continuum** beyond series limit

---

## Daily Checklist

- [ ] I can derive the hydrogen energy formula
- [ ] I know the spectral series names and regions
- [ ] I can calculate transition wavelengths
- [ ] I understand series limits and ionization
- [ ] I see the historical significance

---

**Next:** [Day_431_Thursday.md](Day_431_Thursday.md) — Hydrogen Wavefunctions
