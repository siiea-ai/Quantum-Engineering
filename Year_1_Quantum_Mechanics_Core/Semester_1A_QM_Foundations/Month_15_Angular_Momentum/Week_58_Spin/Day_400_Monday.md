# Day 400: The Stern-Gerlach Experiment — Discovery of Spin

## Overview

**Day 400 of 2520 | Week 58, Day 1 | Month 15: Angular Momentum & Spin**

Today we encounter one of the most important experiments in quantum mechanics history: the Stern-Gerlach experiment of 1922. This experiment revealed that angular momentum is quantized in space, and more surprisingly, that electrons possess an intrinsic angular momentum—spin—that has no classical analog. The two-valued nature of electron spin makes it the perfect physical realization of a qubit.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Historical Context & Motivation | 60 min |
| 10:00 AM | Experimental Setup & Physics | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Observation: Two Spots | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Sequential Stern-Gerlach | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Quantum Measurement Foundations | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe** the historical context and motivation for the Stern-Gerlach experiment
2. **Explain** the experimental setup including the inhomogeneous magnetic field
3. **Analyze** why the observation of exactly two spots implies spin-1/2
4. **Predict** outcomes of sequential Stern-Gerlach experiments
5. **Connect** the Stern-Gerlach apparatus to quantum measurement theory
6. **Simulate** beam splitting computationally

---

## 1. Historical Context (1920-1922)

### The Old Quantum Theory Crisis

By 1920, physicists had accumulated evidence that challenged classical physics:

**Bohr Model Successes:**
- Hydrogen spectrum: $E_n = -13.6 \text{ eV}/n^2$
- Quantized orbits: $L = n\hbar$

**Unexplained Phenomena:**
- Fine structure of spectral lines
- Anomalous Zeeman effect (splitting patterns didn't fit)
- Why do some atoms have odd numbers of electrons but even quantum numbers?

### The Quantization Question

Bohr's model implied that angular momentum is quantized:

$$L = n\hbar, \quad n = 1, 2, 3, \ldots$$

But is the *direction* of angular momentum also quantized? Classically, $\vec{L}$ can point in any direction. Quantum mechanically, if $L_z$ can only take discrete values, this is called **space quantization** or **spatial quantization**.

### Stern's Proposal (1921)

Otto Stern proposed a direct test: Send atoms with magnetic moments through an inhomogeneous magnetic field. Classically, the atoms should spread continuously. If space quantization is real, discrete spots should appear.

---

## 2. Experimental Setup

### The Apparatus

```
         Oven                 Magnets              Screen
      ┌───────┐            ┌─────────┐
      │  Ag   │            │    N    │              │
      │ atoms │  =======>  │   ∇B    │  ========>   │ ●
      │       │            │    S    │              │ ●
      └───────┘            └─────────┘              │

      T~1000°C           Non-uniform B         Detection
```

**Components:**

1. **Oven:** Heated to ~1000°C to produce a beam of silver (Ag) atoms
2. **Collimators:** Slits to create a narrow beam
3. **Magnets:** Specially shaped to produce inhomogeneous field
4. **Screen:** Glass plate to detect deposited atoms

### The Magnetic Field

The key innovation was creating a **non-uniform** magnetic field:

$$\vec{B} = B_z(z)\hat{z}, \quad \frac{\partial B_z}{\partial z} \neq 0$$

The field is strongest near the pointed pole piece:

$$B_z \approx B_0 + \left(\frac{\partial B}{\partial z}\right)z$$

### Classical Prediction

An atom with magnetic moment $\vec{\mu}$ in this field experiences a force:

$$\vec{F} = \nabla(\vec{\mu} \cdot \vec{B})$$

For field primarily in z-direction:

$$F_z = \mu_z \frac{\partial B_z}{\partial z}$$

**Classical expectation:** Since $\mu_z = \mu \cos\theta$ where $\theta$ is random (thermal distribution), atoms should spread into a **continuous band** on the screen.

### Why Silver Atoms?

Silver (Ag) has atomic number 47:
- Electronic configuration: [Kr] 4d¹⁰ 5s¹
- The 4d shell is filled (contributes no net angular momentum)
- Single 5s electron: $\ell = 0$ (no orbital angular momentum!)
- Any magnetic moment must come from the electron itself

This was crucial: If silver shows quantization, it cannot be orbital angular momentum.

---

## 3. The Observation: Exactly Two Spots

### The Result (February 1922)

When Stern and Gerlach developed their detector plate, they saw:

```
    Classical Expectation          Actual Result

           │                            │
           │                            │  ●  (spin up)
    ═══════════════              ───────────────
           │                            │
           │                            │  ●  (spin down)
           │                            │

    Continuous spread             Two discrete spots
```

**The beam split into exactly TWO discrete spots!**

### What Does This Mean?

1. **Space quantization is real:** $\mu_z$ takes only discrete values
2. **Only two values:** $\mu_z = \pm \mu_s$ (not three, not a continuum)
3. **Since** $\mu \propto L$ and $L_z = m\hbar$ with $m = -\ell, \ldots, +\ell$, having exactly **two values** means $2\ell + 1 = 2$, so $\ell = 1/2$

### The Half-Integer Problem

For orbital angular momentum, $\ell = 0, 1, 2, \ldots$ (integers only).

$$\ell = \frac{1}{2} \text{ is impossible for orbital motion!}$$

**Conclusion:** Electrons must possess an **intrinsic** angular momentum, later called **spin**, with:

$$s = \frac{1}{2}$$

### Eigenvalues of Spin

For spin-1/2:

$$\boxed{S^2 |s,m_s\rangle = \hbar^2 s(s+1)|s,m_s\rangle = \frac{3\hbar^2}{4}|s,m_s\rangle}$$

$$\boxed{S_z |s,m_s\rangle = m_s\hbar|s,m_s\rangle = \pm\frac{\hbar}{2}|s,m_s\rangle}$$

---

## 4. Quantitative Analysis

### Force on the Atom

The force on a magnetic dipole in an inhomogeneous field:

$$F_z = \mu_z \frac{\partial B}{\partial z}$$

The magnetic moment of a spin-1/2 particle:

$$\mu_z = -g_s \frac{e}{2m_e} S_z = -g_s \frac{e\hbar}{4m_e}(\pm 1)$$

where $g_s \approx 2$ is the electron g-factor.

### Deflection Calculation

For an atom traveling with velocity $v_x$ through a field region of length $L$:

**Acceleration:**
$$a_z = \frac{F_z}{M} = \frac{\mu_z}{M}\frac{\partial B}{\partial z}$$

**Time in field:**
$$t = \frac{L}{v_x}$$

**Deflection in field region:**
$$z_1 = \frac{1}{2}a_z t^2 = \frac{\mu_z}{2M}\frac{\partial B}{\partial z}\left(\frac{L}{v_x}\right)^2$$

**Additional deflection to screen** (distance $D$ from magnet):
$$z_2 = v_z \cdot \frac{D}{v_x} = a_z t \cdot \frac{D}{v_x}$$

**Total deflection:**
$$\boxed{\Delta z = \frac{\mu_z}{Mv_x^2}\frac{\partial B}{\partial z}\left(\frac{L^2}{2} + LD\right)}$$

### Typical Values

- $\frac{\partial B}{\partial z} \approx 10^3$ T/m
- $L \approx 3$ cm, $D \approx 30$ cm
- $v_x \approx 500$ m/s (thermal velocity at 1000°C)
- $M_{Ag} = 1.8 \times 10^{-25}$ kg

This gives deflections of $\Delta z \approx \pm 0.1$ mm, easily observable.

---

## 5. Sequential Stern-Gerlach Experiments

### The Profound Implications

Sequential Stern-Gerlach experiments reveal the foundational features of quantum measurement.

### Experiment 1: SGz → SGz

```
                SGz                      SGz
    ●  ──────►  ● (z+)  ─────────────►   ● (z+, 100%)
                ● (z-)  blocked
```

**Result:** After selecting $|+z\rangle$ atoms, a second z-measurement gives $|+z\rangle$ with 100% probability.

**Interpretation:** Measurement prepares a definite state.

### Experiment 2: SGz → SGx

```
                SGz                      SGx
    ●  ──────►  ● (z+)  ─────────────►   ● (x+, 50%)
                ● (z-) blocked           ● (x-, 50%)
```

**Result:** Starting with $|+z\rangle$, measuring $S_x$ gives $|+x\rangle$ or $|-x\rangle$ with equal probability!

**Calculation:**
$$|+z\rangle = \frac{1}{\sqrt{2}}(|+x\rangle + |-x\rangle)$$

$$P(+x) = |\langle +x|+z\rangle|^2 = \frac{1}{2}$$

### Experiment 3: SGz → SGx → SGz

```
           SGz              SGx              SGz
    ●  ──►  ● (z+)  ──►     ● (x+)    ──►    ● (z+, 50%)
            ● blocked       ● blocked        ● (z-, 50%)
```

**Result:** Even though we started with $|+z\rangle$ atoms, after measuring $S_x$ and then $S_z$ again, we get **both** $|+z\rangle$ and $|-z\rangle$!

**Key Insight:** The $S_x$ measurement **disturbed** the $S_z$ information. This is because $[S_x, S_z] \neq 0$.

### Quantum Computing Connection: Measurement and Collapse

The sequential Stern-Gerlach experiment demonstrates:

1. **State preparation:** Measurement prepares definite states
2. **Superposition:** A state definite in z is a superposition in x
3. **Incompatible observables:** Measuring one disturbs the other
4. **Probabilistic outcomes:** Cannot predict individual results, only probabilities

This is exactly the physics behind qubit measurement in quantum computing!

---

## 6. The Stern-Gerlach as a Quantum Measurement Device

### Measurement Postulate in Action

The Stern-Gerlach apparatus is a **physical implementation** of the quantum measurement postulate:

**Before SG (state):**
$$|\psi\rangle = \alpha|+z\rangle + \beta|-z\rangle$$

**After SG (measurement):**
- Outcome $+z$ with probability $|\alpha|^2$, state becomes $|+z\rangle$
- Outcome $-z$ with probability $|\beta|^2$, state becomes $|-z\rangle$

### Ideal Projective Measurement

The SG apparatus performs a **projective measurement** of $S_z$:

$$\hat{P}_{+z} = |+z\rangle\langle+z|, \quad \hat{P}_{-z} = |-z\rangle\langle -z|$$

These are projection operators satisfying:
- $\hat{P}_{+z}^2 = \hat{P}_{+z}$ (idempotent)
- $\hat{P}_{+z}\hat{P}_{-z} = 0$ (orthogonal)
- $\hat{P}_{+z} + \hat{P}_{-z} = \hat{I}$ (complete)

### The Measurement Problem

The Stern-Gerlach experiment raises deep questions:
- When exactly does the "collapse" happen?
- Is it when the atom hits the screen? Enters the field?
- What constitutes an "observer"?

These questions remain debated in quantum foundations today.

---

## 7. Worked Examples

### Example 1: Deflection Calculation

**Problem:** Calculate the beam separation in a Stern-Gerlach experiment with:
- $\frac{\partial B}{\partial z} = 1000$ T/m
- Magnet length $L = 5$ cm
- Drift distance $D = 25$ cm
- Silver atom velocity $v = 600$ m/s

**Solution:**

Step 1: Calculate the magnetic moment.
$$\mu_z = \pm g_s \frac{e\hbar}{4m_e} = \pm \frac{2 \times 1.6 \times 10^{-19} \times 1.055 \times 10^{-34}}{4 \times 9.11 \times 10^{-31}}$$

$$\mu_z = \pm 9.27 \times 10^{-24} \text{ J/T} = \pm \mu_B$$ (Bohr magneton)

Step 2: Calculate the deflection.
$$\Delta z = \frac{\mu_B}{M_{Ag}v^2}\frac{\partial B}{\partial z}\left(\frac{L^2}{2} + LD\right)$$

With $M_{Ag} = 1.79 \times 10^{-25}$ kg:

$$\Delta z = \frac{9.27 \times 10^{-24}}{1.79 \times 10^{-25} \times (600)^2} \times 1000 \times \left(\frac{0.05^2}{2} + 0.05 \times 0.25\right)$$

$$\Delta z = \frac{9.27 \times 10^{-24}}{6.44 \times 10^{-20}} \times 1000 \times (0.00125 + 0.0125)$$

$$\Delta z = 1.44 \times 10^{-4} \times 1000 \times 0.01375$$

$$\boxed{\Delta z = \pm 1.98 \text{ mm}}$$

Total beam separation: $\approx 4$ mm.

---

### Example 2: Sequential Measurement Probabilities

**Problem:** A spin-1/2 particle is prepared in state $|+z\rangle$. It passes through:
1. An SGx apparatus (x-up beam selected)
2. An SGz apparatus

What is the probability of finding $|+z\rangle$ at the end?

**Solution:**

Step 1: Express $|+z\rangle$ in x-basis.
$$|+z\rangle = \frac{1}{\sqrt{2}}|+x\rangle + \frac{1}{\sqrt{2}}|-x\rangle$$

Step 2: After SGx selects $|+x\rangle$, the state is $|+x\rangle$.

Step 3: Express $|+x\rangle$ in z-basis.
$$|+x\rangle = \frac{1}{\sqrt{2}}|+z\rangle + \frac{1}{\sqrt{2}}|-z\rangle$$

Step 4: Probability of $|+z\rangle$ from final measurement:
$$P(+z) = |\langle +z|+x\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \boxed{\frac{1}{2}}$$

---

### Example 3: Arbitrary Measurement Direction

**Problem:** A particle in state $|+z\rangle$ passes through an SG apparatus oriented at angle $\theta$ from z-axis in the xz-plane. What is the probability of deflection in the positive direction?

**Solution:**

The eigenstate of $S_n$ where $\hat{n} = \sin\theta\,\hat{x} + \cos\theta\,\hat{z}$ with eigenvalue $+\hbar/2$ is:

$$|+n\rangle = \cos\frac{\theta}{2}|+z\rangle + \sin\frac{\theta}{2}|-z\rangle$$

The probability is:
$$P(+n) = |\langle +n|+z\rangle|^2 = \cos^2\frac{\theta}{2}$$

$$\boxed{P(+n) = \cos^2\frac{\theta}{2}}$$

**Check:**
- $\theta = 0$: $P = 1$ (same direction)
- $\theta = \pi/2$: $P = 1/2$ (perpendicular)
- $\theta = \pi$: $P = 0$ (opposite)

---

## 8. Practice Problems

### Problem Set

#### Level 1: Direct Application

**Problem 1.1:** A Stern-Gerlach experiment uses silver atoms at T = 1200 K. Estimate the average velocity of atoms using $\langle E \rangle = \frac{3}{2}k_BT$.

**Problem 1.2:** If the magnetic field gradient is $\frac{\partial B}{\partial z} = 800$ T/m and the effective interaction length is 10 cm, what is the force on a spin-up silver atom?

**Problem 1.3:** A beam of atoms in state $|\psi\rangle = \frac{1}{\sqrt{3}}|+z\rangle + \sqrt{\frac{2}{3}}|-z\rangle$ enters an SGz apparatus. What fraction of atoms emerge in each beam?

#### Level 2: Intermediate

**Problem 2.1:** Prove that for a spin-1/2 particle in state $|+z\rangle$, measuring $S_x$ gives outcomes with equal probability. Then show that the state $|+z\rangle$ can be written as a superposition of $S_x$ eigenstates.

**Problem 2.2:** A particle starts in $|+z\rangle$. Find the probability of measuring $+\hbar/2$ for $S_n$ where $\hat{n} = \frac{1}{\sqrt{2}}(\hat{x} + \hat{z})$.

**Problem 2.3:** Design a sequence of Stern-Gerlach apparatuses that would prepare the state $|+y\rangle$ starting from an unpolarized beam.

#### Level 3: Challenging

**Problem 3.1:** In a modified Stern-Gerlach experiment, atoms pass through two sequential SG apparatuses with the second rotated by angle $\theta$ from the first. If we only keep the $+$ deflected atoms from both, what fraction of the original beam survives? How does this depend on $\theta$?

**Problem 3.2:** Prove that no sequence of Stern-Gerlach measurements can determine both $S_x$ and $S_z$ simultaneously for a single particle, relating this to the uncertainty principle.

**Problem 3.3:** (Historical) When Stern and Gerlach first observed the splitting, they initially thought it confirmed Bohr's model with $\ell = 1$ (three spots expected: m = -1, 0, +1). Explain how their initial misinterpretation arose and why the observation of only two spots actually contradicted Bohr's model.

---

## 9. Computational Lab: Simulating Stern-Gerlach

### Python Implementation

```python
"""
Day 400 Computational Lab: Stern-Gerlach Experiment Simulation
Simulates beam trajectories and splitting in the SG apparatus.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
hbar = 1.055e-34  # J·s
mu_B = 9.274e-24  # Bohr magneton, J/T
m_e = 9.109e-31   # electron mass, kg
M_Ag = 1.79e-25   # silver atom mass, kg
g_s = 2.0         # electron g-factor

class SternGerlachApparatus:
    """
    Simulates a Stern-Gerlach apparatus for spin-1/2 particles.
    """

    def __init__(self, dB_dz=1000, L=0.05, D=0.25, orientation='z'):
        """
        Initialize SG apparatus.

        Parameters:
        -----------
        dB_dz : float
            Magnetic field gradient (T/m)
        L : float
            Length of magnetic field region (m)
        D : float
            Drift distance to detector (m)
        orientation : str
            Measurement axis: 'x', 'y', or 'z'
        """
        self.dB_dz = dB_dz
        self.L = L
        self.D = D
        self.orientation = orientation

    def calculate_deflection(self, v, spin_up=True):
        """
        Calculate deflection for a spin state.

        Parameters:
        -----------
        v : float
            Atom velocity (m/s)
        spin_up : bool
            True for spin-up, False for spin-down

        Returns:
        --------
        float : Total deflection (m)
        """
        # Magnetic moment
        mu_z = mu_B if spin_up else -mu_B

        # Force on atom
        F = mu_z * self.dB_dz

        # Acceleration
        a = F / M_Ag

        # Time in field
        t = self.L / v

        # Deflection in field region + drift region
        z1 = 0.5 * a * t**2  # In field
        v_z = a * t  # Acquired velocity
        z2 = v_z * (self.D / v)  # Drift

        return z1 + z2

    def simulate_beam(self, n_atoms, v_mean, v_std, initial_state=None):
        """
        Simulate a beam of atoms.

        Parameters:
        -----------
        n_atoms : int
            Number of atoms
        v_mean : float
            Mean velocity (m/s)
        v_std : float
            Velocity standard deviation (m/s)
        initial_state : tuple or None
            (alpha, beta) for state alpha|+z> + beta|-z>
            None for unpolarized beam

        Returns:
        --------
        tuple : (deflections, spin_states)
        """
        # Velocity distribution
        velocities = np.random.normal(v_mean, v_std, n_atoms)
        velocities = np.abs(velocities)  # Ensure positive

        # Determine spin states probabilistically
        if initial_state is None:
            # Unpolarized: 50-50 mixture
            spin_up = np.random.random(n_atoms) < 0.5
        else:
            alpha, beta = initial_state
            prob_up = np.abs(alpha)**2
            spin_up = np.random.random(n_atoms) < prob_up

        # Calculate deflections
        deflections = np.zeros(n_atoms)
        for i in range(n_atoms):
            deflections[i] = self.calculate_deflection(velocities[i], spin_up[i])

        return deflections, spin_up, velocities


def plot_sg_apparatus():
    """Create a schematic diagram of the Stern-Gerlach apparatus."""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Colors
    metal_color = '#708090'
    beam_color = '#FFD700'
    field_color = '#87CEEB'

    # Oven
    oven = FancyBboxPatch((0.5, 2), 1.5, 2,
                          boxstyle="round,pad=0.05",
                          facecolor='#CD853F', edgecolor='black', linewidth=2)
    ax.add_patch(oven)
    ax.text(1.25, 3, 'Oven\n(Ag atoms)', ha='center', va='center', fontsize=10)

    # Collimating slits
    ax.plot([2.5, 2.5], [2.2, 2.8], 'k-', linewidth=3)
    ax.plot([2.5, 2.5], [3.2, 3.8], 'k-', linewidth=3)
    ax.plot([3.5, 3.5], [2.3, 2.9], 'k-', linewidth=3)
    ax.plot([3.5, 3.5], [3.1, 3.7], 'k-', linewidth=3)
    ax.text(3, 4.3, 'Collimators', ha='center', fontsize=9)

    # Magnet (N pole)
    magnet_N = FancyBboxPatch((5, 3.5), 2.5, 1.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#FF6B6B', edgecolor='black', linewidth=2)
    ax.add_patch(magnet_N)
    ax.text(6.25, 4.25, 'N', ha='center', va='center', fontsize=14, fontweight='bold')

    # Magnet (S pole with pointed tip)
    ax.fill([5, 7.5, 7.5, 6.25, 5], [0.5, 0.5, 2, 2.8, 2],
            color='#4169E1', edgecolor='black', linewidth=2)
    ax.text(6.25, 1.2, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Field gradient arrows
    for y in np.linspace(2.9, 3.4, 4):
        length = 0.3 + 0.15 * (3.5 - y)
        ax.arrow(6.25, y, 0, -length, head_width=0.1, head_length=0.05, fc='blue', ec='blue', alpha=0.5)
    ax.text(5.5, 3.2, r'$\nabla B$', fontsize=12, color='blue')

    # Detector screen
    screen = Rectangle((10, 1), 0.2, 4, facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax.add_patch(screen)
    ax.text(10.1, 5.3, 'Detector', ha='center', fontsize=10)

    # Beam paths
    # Input beam (yellow)
    ax.annotate('', xy=(5, 3), xytext=(2, 3),
                arrowprops=dict(arrowstyle='->', color=beam_color, lw=3))

    # Split beams
    ax.annotate('', xy=(10, 4), xytext=(7.5, 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(10, 2), xytext=(7.5, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # Detection spots
    ax.plot(10.3, 4, 'ro', markersize=15, label=r'$|+z\rangle$ (spin up)')
    ax.plot(10.3, 2, 'bo', markersize=15, label=r'$|-z\rangle$ (spin down)')

    # Labels
    ax.text(3.5, 3, 'Atomic\nbeam', ha='center', fontsize=9, style='italic')
    ax.text(8.5, 4.2, r'$|\uparrow\rangle$', fontsize=12, color='red')
    ax.text(8.5, 1.8, r'$|\downarrow\rangle$', fontsize=12, color='blue')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stern-Gerlach Apparatus Schematic', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('sg_apparatus_schematic.png', dpi=150, bbox_inches='tight')
    plt.show()


def simulate_beam_splitting():
    """Simulate and visualize beam splitting in the SG experiment."""

    # Create apparatus
    sg = SternGerlachApparatus(dB_dz=1000, L=0.05, D=0.25)

    # Simulate beams
    n_atoms = 5000
    v_mean = 600  # m/s
    v_std = 50    # m/s

    # Unpolarized beam
    deflections, spin_up, velocities = sg.simulate_beam(n_atoms, v_mean, v_std)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Classical expectation (continuous)
    ax1 = axes[0]
    classical_deflections = np.random.normal(0, 0.001, n_atoms)
    ax1.hist(classical_deflections * 1000, bins=50, color='gray', alpha=0.7,
             density=True, label='Classical prediction')
    ax1.set_xlabel('Deflection (mm)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Classical Expectation:\nContinuous Spread', fontsize=12)
    ax1.axvline(0, color='red', linestyle='--', label='Beam center')
    ax1.legend()
    ax1.set_xlim(-4, 4)

    # Right: Quantum result (two peaks)
    ax2 = axes[1]
    deflections_mm = deflections * 1000

    # Separate by spin
    up_deflections = deflections_mm[spin_up]
    down_deflections = deflections_mm[~spin_up]

    ax2.hist(up_deflections, bins=30, color='red', alpha=0.6,
             density=True, label=r'$|+z\rangle$ (spin up)')
    ax2.hist(down_deflections, bins=30, color='blue', alpha=0.6,
             density=True, label=r'$|-z\rangle$ (spin down)')
    ax2.set_xlabel('Deflection (mm)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Quantum Result:\nTwo Discrete Spots', fontsize=12)
    ax2.legend()
    ax2.set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig('sg_beam_splitting.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("Stern-Gerlach Beam Splitting Simulation")
    print("=" * 50)
    print(f"Number of atoms: {n_atoms}")
    print(f"Mean velocity: {v_mean} m/s")
    print(f"Field gradient: {sg.dB_dz} T/m")
    print(f"\nSpin-up deflection: {np.mean(up_deflections):.3f} mm")
    print(f"Spin-down deflection: {np.mean(down_deflections):.3f} mm")
    print(f"Total separation: {np.mean(up_deflections) - np.mean(down_deflections):.3f} mm")


def sequential_sg_simulation():
    """
    Simulate sequential Stern-Gerlach experiments.
    Demonstrates quantum measurement and state preparation.
    """

    print("\n" + "=" * 60)
    print("SEQUENTIAL STERN-GERLACH SIMULATION")
    print("=" * 60)

    n_particles = 10000

    # Experiment 1: SGz -> SGz (same direction)
    print("\n--- Experiment 1: SGz → SGz ---")
    print("Initial state: |+z⟩ (prepared by first SGz, selecting +z)")

    # After first SGz selects +z, all are |+z⟩
    # Second SGz measures z: |+z⟩ → always +z
    results_1 = np.ones(n_particles)  # All measure +z

    print(f"Second SGz results:")
    print(f"  |+z⟩: {np.sum(results_1 == 1)/n_particles * 100:.1f}%")
    print(f"  |-z⟩: {np.sum(results_1 == 0)/n_particles * 100:.1f}%")

    # Experiment 2: SGz -> SGx
    print("\n--- Experiment 2: SGz → SGx ---")
    print("Initial state: |+z⟩")
    print("|+z⟩ = (1/√2)|+x⟩ + (1/√2)|-x⟩")

    # |+z⟩ has 50% chance of +x, 50% chance of -x
    results_2 = np.random.random(n_particles) < 0.5

    print(f"SGx results:")
    print(f"  |+x⟩: {np.sum(results_2)/n_particles * 100:.1f}%")
    print(f"  |-x⟩: {np.sum(~results_2)/n_particles * 100:.1f}%")

    # Experiment 3: SGz -> SGx -> SGz (select +x after SGx)
    print("\n--- Experiment 3: SGz → SGx → SGz ---")
    print("Initial state: |+z⟩")
    print("After SGx selects |+x⟩:")
    print("|+x⟩ = (1/√2)|+z⟩ + (1/√2)|-z⟩")

    # After selecting +x from SGx, state is |+x⟩
    # |+x⟩ has 50% +z, 50% -z
    results_3 = np.random.random(n_particles) < 0.5

    print(f"Final SGz results:")
    print(f"  |+z⟩: {np.sum(results_3)/n_particles * 100:.1f}%")
    print(f"  |-z⟩: {np.sum(~results_3)/n_particles * 100:.1f}%")
    print("\n⚠️  Information about initial |+z⟩ state is LOST!")
    print("    The SGx measurement 'disturbed' the Sz information.")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    experiments = ['SGz → SGz', 'SGz → SGx', 'SGz → SGx → SGz']
    results = [
        [100, 0],
        [np.sum(results_2)/n_particles * 100, np.sum(~results_2)/n_particles * 100],
        [np.sum(results_3)/n_particles * 100, np.sum(~results_3)/n_particles * 100]
    ]
    labels = [
        [r'$|+z\rangle$', r'$|-z\rangle$'],
        [r'$|+x\rangle$', r'$|-x\rangle$'],
        [r'$|+z\rangle$', r'$|-z\rangle$']
    ]
    colors = [['red', 'blue'], ['green', 'purple'], ['red', 'blue']]

    for i, ax in enumerate(axes):
        bars = ax.bar([0, 1], results[i], color=colors[i], alpha=0.7, width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels[i], fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_title(experiments[i], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)

        # Add value labels on bars
        for bar, val in zip(bars, results[i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('sequential_sg.png', dpi=150, bbox_inches='tight')
    plt.show()


def theta_dependence():
    """
    Plot measurement probability as a function of analyzer angle.
    P(+) = cos²(θ/2) where θ is angle from initial spin direction.
    """

    theta = np.linspace(0, np.pi, 100)
    prob_plus = np.cos(theta/2)**2
    prob_minus = np.sin(theta/2)**2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(np.degrees(theta), prob_plus, 'r-', linewidth=2,
            label=r'$P(+) = \cos^2(\theta/2)$')
    ax.plot(np.degrees(theta), prob_minus, 'b-', linewidth=2,
            label=r'$P(-) = \sin^2(\theta/2)$')

    # Mark key points
    key_angles = [0, 45, 90, 135, 180]
    for angle in key_angles:
        theta_rad = np.radians(angle)
        p_plus = np.cos(theta_rad/2)**2
        ax.plot(angle, p_plus, 'ro', markersize=8)
        ax.annotate(f'{p_plus:.2f}', (angle, p_plus + 0.05),
                   ha='center', fontsize=9, color='red')

    ax.set_xlabel(r'Angle $\theta$ (degrees)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Measurement Probability vs. Analyzer Angle\n' +
                 r'Initial state: $|+z\rangle$, measuring along $\hat{n}$ at angle $\theta$ from z',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 45, 90, 135, 180])

    plt.tight_layout()
    plt.savefig('sg_angle_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()


def quantum_computing_connection():
    """
    Demonstrate how SG measurements relate to qubit measurements.
    """

    print("\n" + "=" * 60)
    print("STERN-GERLACH AND QUANTUM COMPUTING")
    print("=" * 60)

    print("""
    The Stern-Gerlach apparatus is the physical prototype of a qubit!

    Mapping:
    --------
    Spin Physics          |  Quantum Computing
    ----------------------|----------------------
    |↑⟩ (spin up)         |  |0⟩ (computational 0)
    |↓⟩ (spin down)       |  |1⟩ (computational 1)
    SGz measurement       |  Z-basis measurement
    SGx measurement       |  X-basis measurement
    Sequential SG         |  Quantum circuits

    Key Insight: A qubit IS a spin-1/2 system!

    When you measure a qubit in a quantum computer, you're doing
    the exact same physics as Stern and Gerlach in 1922.

    The quantum computing revolution is built on understanding
    spin-1/2 quantum mechanics.
    """)


# Main execution
if __name__ == "__main__":
    print("Day 400: Stern-Gerlach Experiment Simulation")
    print("=" * 60)

    # Run all demonstrations
    plot_sg_apparatus()
    simulate_beam_splitting()
    sequential_sg_simulation()
    theta_dependence()
    quantum_computing_connection()
```

### Lab Exercises

1. **Modify the simulation** to include a third Stern-Gerlach apparatus at an arbitrary angle.

2. **Add thermal broadening:** Implement a Maxwell-Boltzmann velocity distribution and observe how it affects the spot widths.

3. **Multiple sequential measurements:** Extend the sequential simulation to track probability flow through many SG devices at different angles.

4. **Visualize in 3D:** Create a 3D plot showing atomic trajectories through the apparatus.

---

## 10. Summary

### Key Concepts Learned

| Concept | Description |
|---------|-------------|
| Spatial quantization | Angular momentum direction takes discrete values |
| Spin-1/2 | Intrinsic angular momentum with s = 1/2 |
| Two spots | Only two Sᵤ values possible: ±ℏ/2 |
| State preparation | SG measurement prepares definite spin states |
| Incompatibility | Measuring Sₓ disturbs Sᵤ information |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$s = \frac{1}{2}$$ | Electron spin quantum number |
| $$S_z = \pm\frac{\hbar}{2}$$ | Spin-z eigenvalues |
| $$P(+n) = \cos^2\frac{\theta}{2}$$ | Measurement probability at angle θ |
| $$\Delta z = \frac{\mu_B}{Mv^2}\frac{\partial B}{\partial z}\left(\frac{L^2}{2} + LD\right)$$ | Beam deflection |

### Historical Significance

- **1922:** Stern-Gerlach experiment performed
- **1925:** Uhlenbeck and Goudsmit propose electron spin
- **1927:** Pauli introduces spin matrices
- **Today:** Spin-1/2 is the physical basis of qubits

---

## 11. Daily Checklist

### Conceptual Understanding
- [ ] I can explain why classical physics predicts continuous spreading
- [ ] I understand why two spots implies half-integer spin
- [ ] I can describe what happens in sequential SG experiments
- [ ] I understand how SG measurement relates to quantum state preparation

### Mathematical Skills
- [ ] I can calculate beam deflection given apparatus parameters
- [ ] I can compute measurement probabilities using $\cos^2(\theta/2)$
- [ ] I can decompose spin states in different bases

### Computational Skills
- [ ] I ran the beam splitting simulation
- [ ] I explored the sequential SG demonstration
- [ ] I understand how the code models quantum measurement

### Quantum Computing Connection
- [ ] I see how |↑⟩, |↓⟩ map to |0⟩, |1⟩
- [ ] I understand that SG measurement = qubit measurement
- [ ] I recognize that sequential SG = quantum circuit

---

## 12. Preview: Day 401

Tomorrow we develop the **complete mathematical formalism for spin-1/2**:

- Two-dimensional Hilbert space structure
- Basis states and notation conventions
- General spinors: |χ⟩ = α|↑⟩ + β|↓⟩
- Matrix representations of spin operators
- Eigenvalue equations for S² and Sᵤ
- Expectation values and measurement

The Stern-Gerlach experiment showed us spin exists. Tomorrow we build the mathematical tools to work with it.

---

## References

1. Gerlach, W. & Stern, O. (1922). "Der experimentelle Nachweis der Richtungsquantelung im Magnetfeld." *Zeitschrift für Physik*, 9, 349-352.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 1.1.

3. Shankar, R. (2011). *Principles of Quantum Mechanics*, 2nd ed., Ch. 14.1.

4. Feynman, R.P. *Lectures on Physics*, Vol. III, Ch. 5.

---

*"I have done a beautiful experiment... which shows the directional quantization of atoms in a magnetic field."*
— Otto Stern, 1922

---

**Day 400 Complete.** Tomorrow: Spin-1/2 Formalism.
