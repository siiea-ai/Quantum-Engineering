# Day 326: Capstone Project — Visualization and Analysis

## Overview

**Month 12, Week 47, Day 4 — Thursday**

Today you create publication-quality visualizations: wavefunctions, time evolution animations, and phase space representations.

## Learning Objectives

1. Create clear, informative figures
2. Add animations for time evolution
3. Apply visualization best practices
4. Analyze results quantitatively

---

## Visualization Best Practices

```python
"""
Day 326: Visualization and Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})


def create_eigenfunction_figure(qho, n_max=5):
    """Create figure showing eigenfunctions and energies."""
    x = np.linspace(-6, 6, 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Wavefunctions
    ax = axes[0]
    for n in range(n_max):
        psi = qho.eigenfunction(n, x)
        offset = qho.energy(n)
        ax.plot(x, psi + offset, label=f'n={n}')
        ax.axhline(y=offset, color='gray', linestyle='--', alpha=0.3)

    # Potential
    V = 0.5 * qho.m * qho.omega**2 * x**2
    ax.plot(x, V, 'k-', linewidth=2, label='V(x)')

    ax.set_xlabel('Position x')
    ax.set_ylabel('ψ_n(x) + E_n')
    ax.set_title('Quantum Harmonic Oscillator Eigenfunctions')
    ax.legend(loc='upper right')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, n_max + 1)

    # Right: Probability densities
    ax = axes[1]
    for n in range(n_max):
        psi = qho.eigenfunction(n, x)
        ax.fill_between(x, np.abs(psi)**2 + n, n, alpha=0.6, label=f'n={n}')

    ax.set_xlabel('Position x')
    ax.set_ylabel('|ψ_n(x)|² + n')
    ax.set_title('Probability Densities')
    ax.legend(loc='upper right')
    ax.set_xlim(-6, 6)

    plt.tight_layout()
    plt.savefig('eigenfunctions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: eigenfunctions.png")


def create_animation(qho, psi_0, x, t_max=10, fps=30):
    """Create animation of time evolution."""
    from matplotlib.animation import PillowWriter

    fig, ax = plt.subplots(figsize=(10, 6))

    line_real, = ax.plot([], [], 'b-', label='Re(ψ)', linewidth=2)
    line_imag, = ax.plot([], [], 'r-', label='Im(ψ)', linewidth=2)
    line_prob, = ax.plot([], [], 'k-', label='|ψ|²', linewidth=2)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    n_frames = int(t_max * fps)
    times = np.linspace(0, t_max, n_frames)

    def animate(frame):
        t = times[frame]
        psi_t = qho.time_evolve(psi_0, x, t)
        line_real.set_data(x, np.real(psi_t))
        line_imag.set_data(x, np.imag(psi_t))
        line_prob.set_data(x, np.abs(psi_t)**2)
        time_text.set_text(f't = {t:.2f}')
        return line_real, line_imag, line_prob, time_text

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save('time_evolution.gif', writer=writer)
    plt.close()
    print("Saved: time_evolution.gif")


def analyze_results(qho, n_max=10):
    """Quantitative analysis of results."""
    print("=" * 60)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 60)

    x = np.linspace(-10, 10, 1000)

    # Energy spacing
    print("\n1. Energy Spacing Analysis:")
    print(f"   ΔE = E_{1} - E_0 = {qho.energy(1) - qho.energy(0):.4f} ℏω")
    print(f"   (Should be exactly 1.0 ℏω)")

    # Position uncertainty
    print("\n2. Position Uncertainty ⟨x²⟩ - ⟨x⟩²:")
    for n in range(5):
        psi = qho.eigenfunction(n, x)
        x_avg = np.trapz(x * np.abs(psi)**2, x)
        x2_avg = np.trapz(x**2 * np.abs(psi)**2, x)
        delta_x = np.sqrt(x2_avg - x_avg**2)
        print(f"   n={n}: Δx = {delta_x:.4f} x₀ (theory: √{(2*n+1)/2:.2f} x₀)")

    # Completeness check
    print("\n3. Completeness Check:")
    total = np.zeros_like(x)
    for n in range(n_max):
        psi = qho.eigenfunction(n, x)
        total += np.abs(psi)**2

    # Should approach delta function behavior
    print(f"   Σ|ψ_n(0)|² (n=0 to {n_max-1}) = {total[len(x)//2]:.4f}")


if __name__ == "__main__":
    from day_324 import QuantumHarmonicOscillator

    qho = QuantumHarmonicOscillator()
    create_eigenfunction_figure(qho)
    analyze_results(qho)
```

---

## Today's Checklist

- [ ] Static figures created
- [ ] Animation working
- [ ] Results analyzed
- [ ] Figures saved in high quality
- [ ] Caption/labels complete

---

## Preview: Day 327

Tomorrow: **Testing and Validation** — ensure correctness.
