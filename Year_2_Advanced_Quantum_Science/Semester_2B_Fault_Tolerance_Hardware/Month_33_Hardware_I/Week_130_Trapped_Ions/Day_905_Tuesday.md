# Day 905: Qubit Encoding Schemes in Trapped Ions

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Hyperfine, Zeeman, and optical qubit physics |
| Afternoon | 2 hours | Problem solving: coherence and transition calculations |
| Evening | 2 hours | Computational lab: coherence simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Explain hyperfine qubit encoding** and calculate hyperfine splitting
2. **Describe Zeeman qubit structure** and magnetic field sensitivity
3. **Analyze optical qubit transitions** using metastable states
4. **Compare coherence times** across different encoding schemes
5. **Evaluate ion species selection** for quantum computing applications
6. **Calculate transition frequencies** and magnetic field sensitivities

## Core Content

### 1. Overview of Qubit Encoding

In trapped ion quantum computing, qubits are encoded in internal electronic states of the ion. The choice of encoding significantly impacts:

- **Coherence time** ($T_2$): How long superpositions survive
- **Gate speed**: Faster gates reduce decoherence effects
- **Sensitivity to noise**: Magnetic field fluctuations, laser noise
- **Technical complexity**: Laser wavelengths, microwave requirements

Three main encoding schemes dominate:

| Scheme | States Used | Typical $T_2$ | Gate Speed | Example Ions |
|--------|-------------|---------------|------------|--------------|
| Hyperfine | Nuclear spin states | 1-1000 s | 10-100 μs | $^{171}$Yb$^+$, $^9$Be$^+$ |
| Zeeman | Magnetic sublevels | 1-100 ms | 1-10 μs | $^{40}$Ca$^+$, $^{88}$Sr$^+$ |
| Optical | Ground + metastable | 100 ms - 1 s | 10-100 μs | $^{40}$Ca$^+$ (729 nm) |

### 2. Hyperfine Qubits

Hyperfine structure arises from the interaction between electronic angular momentum ($\vec{J}$) and nuclear spin ($\vec{I}$):

$$\hat{H}_{hf} = A_{hf} \vec{I} \cdot \vec{J}$$

where $A_{hf}$ is the hyperfine constant.

#### Energy Level Structure

For an $S_{1/2}$ ground state with nuclear spin $I$:
- Total angular momentum: $F = I \pm 1/2$
- Number of states: $2(2I + 1)$

**Example: $^{171}$Yb$^+$** ($I = 1/2$):
- $F = 0$ (singlet): $|0\rangle \equiv |F=0, m_F=0\rangle$
- $F = 1$ (triplet): $|1\rangle \equiv |F=1, m_F=0\rangle$

$$\boxed{\Delta E_{hf} = A_{hf}(F(F+1) - I(I+1) - J(J+1))/2}$$

For $^{171}$Yb$^+$: $\Delta E_{hf}/h = 12.6$ GHz

#### Clock Transition

The $|F=0, m_F=0\rangle \leftrightarrow |F=1, m_F=0\rangle$ transition is called a **clock transition** because:

$$\boxed{\frac{\partial \omega}{\partial B}\bigg|_{B=0} = 0}$$

First-order magnetic field insensitivity leads to extremely long coherence times.

**Second-order sensitivity:**
$$\Delta \omega = \beta B^2, \quad \beta \approx 310 \text{ Hz/G}^2 \text{ for }^{171}\text{Yb}^+$$

#### Advantages of Hyperfine Qubits
- First-order magnetic field insensitivity
- Coherence times exceeding seconds (up to hours reported)
- Microwave control at GHz frequencies (simpler than optical)
- No spontaneous emission (ground state to ground state)

#### Disadvantages
- Slower gates (microwave Rabi frequencies limited)
- Requires isotopes with nuclear spin
- Hyperfine splitting varies by isotope

### 3. Zeeman Qubits

Zeeman qubits use magnetic sublevels $m_J$ of a single fine-structure level:

$$\hat{H}_Z = -\vec{\mu} \cdot \vec{B} = g_J \mu_B m_J B$$

#### Qubit States

For an $S_{1/2}$ state:
- $|0\rangle \equiv |m_J = -1/2\rangle$
- $|1\rangle \equiv |m_J = +1/2\rangle$

**Transition frequency:**
$$\boxed{\omega_Z = g_J \mu_B B / \hbar \approx 2.8 \text{ MHz/G} \cdot B}$$

#### Magnetic Field Sensitivity

Unlike clock transitions, Zeeman qubits have first-order sensitivity:

$$\frac{\partial \omega}{\partial B} = g_J \mu_B / \hbar \approx 2\pi \times 2.8 \text{ MHz/G}$$

This requires:
- Magnetic field stabilization to $\sim 1$ μG level
- Magnetic shielding ($\mu$-metal enclosures)
- Active field compensation

#### Advantages of Zeeman Qubits
- Works with any ion (no nuclear spin required)
- Simpler level structure
- Direct optical transitions for fast gates

#### Disadvantages
- Strong magnetic field sensitivity
- Shorter coherence times (ms scale without active stabilization)
- Requires careful B-field control

### 4. Optical Qubits

Optical qubits use transitions between the ground state and a long-lived metastable state.

#### Example: $^{40}$Ca$^+$ Optical Qubit

- $|0\rangle \equiv |S_{1/2}\rangle$ (ground state)
- $|1\rangle \equiv |D_{5/2}\rangle$ (metastable, $\tau \approx 1.17$ s)

**Transition wavelength:** 729 nm (quadrupole transition)

The $D_{5/2}$ state has natural lifetime:
$$\boxed{\Gamma^{-1} = \tau \approx 1.17 \text{ s}}$$

This limits the coherence time to $T_2 \leq 2\tau \approx 2.3$ s.

#### Selection Rules

The $S_{1/2} \leftrightarrow D_{5/2}$ transition is **electric quadrupole (E2)**:

$$\Delta L = 2, \quad \Delta J = 0, \pm 1, \pm 2$$

This gives weak coupling (slower gates) but also reduced sensitivity to stray fields.

#### Quadrupole Transition Rate

$$\Omega_{E2} = \frac{eQ\langle D|\hat{Q}|S\rangle}{\hbar} \cdot \nabla E$$

where $Q$ is the quadrupole matrix element.

#### Advantages of Optical Qubits
- Very narrow linewidth (sub-Hz possible)
- Long coherence times from metastable state lifetime
- Direct optical control
- Useful for optical clocks and precision measurement

#### Disadvantages
- Finite excited state lifetime limits $T_2$
- Requires ultra-stable lasers (Hz linewidth)
- More complex laser systems

### 5. Magnetic Field Effects and Sensitivity

#### General Zeeman Hamiltonian

$$\hat{H}_B = \mu_B B(g_J \hat{J}_z + g_I \hat{I}_z) = \mu_B B(g_J m_J + g_I m_I)$$

where $g_J \approx 2$ for $S$ states and $g_I \approx 10^{-3}$ (nuclear g-factor).

#### Sensitivity Comparison

| Transition Type | Sensitivity | Example |
|-----------------|-------------|---------|
| Clock ($m_F = 0 \leftrightarrow m_F = 0$) | $\propto B^2$ | 310 Hz/G$^2$ ($^{171}$Yb$^+$) |
| Zeeman ($\Delta m_J = 1$) | $\propto B$ | 2.8 MHz/G |
| Optical + Zeeman | $\propto B$ | 1-3 MHz/G (depends on states) |

#### Field-Insensitive Points

For some transitions, there exist "magic" magnetic field values where:

$$\frac{\partial \omega}{\partial B}\bigg|_{B=B_{magic}} = 0$$

These provide first-order insensitivity at non-zero field.

### 6. Ion Species Comparison

#### $^{171}$Yb$^+$ (Hyperfine Qubit)

| Property | Value |
|----------|-------|
| Nuclear spin | $I = 1/2$ |
| Hyperfine splitting | 12.6428 GHz |
| Clock transition | $|F=0,m_F=0\rangle \leftrightarrow |F=1,m_F=0\rangle$ |
| Cooling transition | 369.5 nm ($^2S_{1/2} \rightarrow ^2P_{1/2}$) |
| Coherence time | Up to 10 minutes demonstrated |
| Gate mechanism | Raman transitions or microwaves |

#### $^{40}$Ca$^+$ (Optical Qubit)

| Property | Value |
|----------|-------|
| Nuclear spin | $I = 0$ |
| Optical qubit transition | 729 nm ($S_{1/2} \rightarrow D_{5/2}$) |
| $D_{5/2}$ lifetime | 1.17 s |
| Cooling transition | 397 nm ($S_{1/2} \rightarrow P_{1/2}$) |
| Coherence time | ~1 s (lifetime limited) |
| Gate mechanism | Direct optical |

#### $^9$Be$^+$ (Hyperfine Qubit)

| Property | Value |
|----------|-------|
| Nuclear spin | $I = 3/2$ |
| Hyperfine splitting | 1.25 GHz |
| Mass | 9 amu (lightest commonly used) |
| Advantage | Fast gates (light ion = high $\omega$) |
| Challenge | UV lasers required (313 nm) |

### 7. Coherence and Decoherence Mechanisms

#### $T_1$ Processes (Population Decay)
- Spontaneous emission (optical qubits)
- Heating to thermal bath
- Inelastic collisions with background gas

#### $T_2$ Processes (Dephasing)
- Magnetic field fluctuations: $1/T_2 \propto \delta B \cdot (\partial\omega/\partial B)$
- Laser phase noise (for optical qubits)
- Motional decoherence
- Electric field noise

#### Coherence Time Hierarchy

$$T_2 \leq 2T_1$$

For trapped ions:
- Hyperfine qubits: $T_2$ can approach seconds to minutes
- Optical qubits: $T_2 \leq 2\tau_{metastable}$
- Zeeman qubits: $T_2 \sim$ ms (limited by B-field noise)

## Quantum Computing Applications

### Qubit Selection Criteria

| Criterion | Best Choice | Reason |
|-----------|-------------|--------|
| Long coherence | Hyperfine | Clock transition, B-field insensitive |
| Fast gates | Zeeman/Optical | Direct optical transitions |
| Simple control | Hyperfine | Microwave frequencies |
| Precision sensing | Optical | Narrow linewidth |
| Scalability | Hyperfine | Robust to field gradients |

### State-of-the-Art Performance (2024)

- **IonQ:** $^{171}$Yb$^+$ hyperfine qubits, 99.97% single-qubit fidelity
- **Quantinuum:** $^{171}$Yb$^+$ with 99.8% two-qubit fidelity
- **Oxford/Innsbruck:** $^{40}$Ca$^+$ optical qubits, 99.9% fidelity

## Worked Examples

### Example 1: Hyperfine Splitting Calculation

**Problem:** Calculate the hyperfine splitting for $^9$Be$^+$ in the $2s$ $^2S_{1/2}$ ground state given $A_{hf}/h = -625$ MHz.

**Solution:**

For $^9$Be$^+$: $I = 3/2$, $J = 1/2$

Possible $F$ values: $F = I + J = 2$ or $F = |I - J| = 1$

Energy levels:
$$E_F = \frac{A_{hf}}{2}[F(F+1) - I(I+1) - J(J+1)]$$

For $F = 2$:
$$E_2 = \frac{A_{hf}}{2}[2(3) - \frac{3}{2}(\frac{5}{2}) - \frac{1}{2}(\frac{3}{2})] = \frac{A_{hf}}{2}[6 - \frac{15}{4} - \frac{3}{4}] = \frac{A_{hf}}{2} \times \frac{3}{2}$$

For $F = 1$:
$$E_1 = \frac{A_{hf}}{2}[1(2) - \frac{15}{4} - \frac{3}{4}] = \frac{A_{hf}}{2}[2 - \frac{18}{4}] = \frac{A_{hf}}{2} \times (-\frac{5}{2})$$

Splitting:
$$\Delta E = E_2 - E_1 = \frac{A_{hf}}{2}\left(\frac{3}{2} + \frac{5}{2}\right) = 2A_{hf}$$

$$\boxed{\Delta E/h = 2 \times (-625 \text{ MHz}) = -1.25 \text{ GHz}}$$

(The sign indicates $F=1$ is higher in energy for negative $A_{hf}$.)

### Example 2: Magnetic Field Sensitivity

**Problem:** A Zeeman qubit in $^{40}$Ca$^+$ has $T_2 = 1$ ms. Estimate the magnetic field fluctuations.

**Solution:**

The dephasing rate from magnetic noise:
$$\frac{1}{T_2} = \gamma_\phi = \frac{\partial\omega}{\partial B} \cdot \delta B_{rms}$$

For a Zeeman qubit with $g_J = 2$:
$$\frac{\partial\omega}{\partial B} = \frac{g_J \mu_B}{\hbar} = 2 \times 2\pi \times 1.4 \text{ MHz/G}$$

From $T_2 = 1$ ms:
$$\gamma_\phi = 1000 \text{ s}^{-1}$$

$$\delta B_{rms} = \frac{\gamma_\phi}{\partial\omega/\partial B} = \frac{1000}{2\pi \times 2.8 \times 10^6}$$

$$\boxed{\delta B_{rms} \approx 57 \text{ μG}}$$

This sets the magnetic shielding requirement.

### Example 3: Optical Qubit Coherence Limit

**Problem:** For $^{40}$Ca$^+$, the $D_{5/2}$ state has lifetime $\tau = 1.17$ s. What is the maximum achievable coherence time?

**Solution:**

The population decay rate:
$$\Gamma = 1/\tau = 0.855 \text{ s}^{-1}$$

For an optical qubit, spontaneous emission causes both $T_1$ decay and dephasing:

$$T_1 = \tau = 1.17 \text{ s}$$

The coherence time is limited by:
$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

where $T_\phi$ is pure dephasing time.

**Upper bound** (no pure dephasing, $T_\phi \to \infty$):
$$\boxed{T_2^{max} = 2T_1 = 2.34 \text{ s}}$$

In practice, laser phase noise and other factors limit $T_2$ to 100 ms - 1 s.

## Practice Problems

### Level 1: Direct Application

1. Calculate the hyperfine splitting for $^{171}$Yb$^+$ given $A_{hf}/h = 12.6428$ GHz and $I = 1/2$, $J = 1/2$.

2. For a Zeeman qubit at $B = 5$ G, what is the qubit transition frequency?

3. If the $D_{5/2}$ state of $^{88}$Sr$^+$ has lifetime 0.39 s, what is the maximum optical qubit coherence time?

### Level 2: Intermediate

4. The clock transition in $^{171}$Yb$^+$ has second-order Zeeman shift $\beta = 310$ Hz/G$^2$. At what magnetic field does the shift equal 1 Hz?

5. Design a magnetic shield to achieve $T_2 = 100$ ms for a Zeeman qubit. What field fluctuation level is required?

6. Compare the gate speed (Rabi frequency) achievable with microwave vs. Raman transitions, given typical power constraints.

### Level 3: Challenging

7. For a stretched-state qubit $|F=2, m_F=2\rangle \leftrightarrow |F=1, m_F=1\rangle$ in $^{171}$Yb$^+$, calculate the first-order magnetic field sensitivity and compare to the clock transition.

8. Analyze the trade-off between coherence time and gate fidelity for optical vs. hyperfine qubits, considering realistic noise sources.

9. Derive the magic magnetic field for a specific transition in $^{43}$Ca$^+$ where the differential Zeeman shift vanishes.

## Computational Lab: Qubit Coherence Simulation

```python
"""
Day 905 Computational Lab: Qubit Encoding and Coherence
Simulating hyperfine structure, Zeeman splitting, and decoherence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import odeint

# Physical constants
mu_B = 9.274e-24  # Bohr magneton (J/T)
hbar = 1.055e-34  # Reduced Planck constant (J·s)
h = 2 * np.pi * hbar

class HyperfineQubit:
    """Hyperfine qubit model (e.g., Yb-171)"""

    def __init__(self, A_hf, I, J):
        """
        Initialize hyperfine qubit

        Parameters:
        -----------
        A_hf : float - Hyperfine constant (Hz)
        I : float - Nuclear spin
        J : float - Electronic angular momentum
        """
        self.A_hf = A_hf
        self.I = I
        self.J = J

        # Calculate F values
        self.F_values = [abs(I - J) + i for i in range(int(2 * min(I, J) + 1))]

        # Calculate energies
        self.energies = {}
        for F in self.F_values:
            E = (A_hf / 2) * (F * (F + 1) - I * (I + 1) - J * (J + 1))
            self.energies[F] = E

    def get_splitting(self):
        """Get hyperfine splitting frequency"""
        E_high = max(self.energies.values())
        E_low = min(self.energies.values())
        return E_high - E_low

    def zeeman_shift(self, F, mF, B, gJ=2.0, gI=0.001):
        """
        Calculate Zeeman shift for state |F, mF> at field B

        Returns shift in Hz
        """
        # For F = I ± 1/2 states
        if F == self.I + self.J:  # Stretched state
            g_F = gJ * (F * (F + 1) + self.J * (self.J + 1) - self.I * (self.I + 1)) / (2 * F * (F + 1))
        else:
            g_F = gJ * (F * (F + 1) + self.J * (self.J + 1) - self.I * (self.I + 1)) / (2 * F * (F + 1))

        # Zeeman shift
        return g_F * mu_B * B * mF / h

    def clock_transition_shift(self, B, beta=310):
        """Second-order Zeeman shift for clock transition (Hz)"""
        # B in Gauss, beta in Hz/G^2
        return beta * B**2


class ZeemanQubit:
    """Zeeman qubit model"""

    def __init__(self, gJ=2.0):
        self.gJ = gJ

    def transition_frequency(self, B):
        """Qubit frequency at magnetic field B (Gauss)"""
        # Convert to Tesla (1 G = 1e-4 T)
        B_tesla = B * 1e-4
        return self.gJ * mu_B * B_tesla / h

    def sensitivity(self):
        """Magnetic field sensitivity (Hz/G)"""
        return self.gJ * mu_B * 1e-4 / h


class OpticalQubit:
    """Optical qubit model (e.g., Ca-40 S-D transition)"""

    def __init__(self, wavelength, lifetime):
        """
        Parameters:
        -----------
        wavelength : float - Transition wavelength (m)
        lifetime : float - Metastable state lifetime (s)
        """
        self.wavelength = wavelength
        self.lifetime = lifetime
        self.frequency = 3e8 / wavelength  # Transition frequency (Hz)

    def decay_rate(self):
        """Population decay rate (1/s)"""
        return 1 / self.lifetime

    def max_coherence_time(self):
        """Maximum T2 from lifetime limit"""
        return 2 * self.lifetime


def simulate_ramsey_decay(T2, omega_detuning, t_array):
    """
    Simulate Ramsey fringe decay

    Parameters:
    -----------
    T2 : float - Coherence time (s)
    omega_detuning : float - Detuning from resonance (rad/s)
    t_array : array - Time points

    Returns:
    --------
    probability : array - Probability of measuring |1>
    """
    # Ramsey signal with exponential decay
    return 0.5 * (1 + np.cos(omega_detuning * t_array) * np.exp(-t_array / T2))


def simulate_dephasing(B_noise_rms, sensitivity, t_max, n_realizations=100):
    """
    Monte Carlo simulation of dephasing from magnetic noise

    Parameters:
    -----------
    B_noise_rms : float - RMS magnetic field fluctuation (G)
    sensitivity : float - Frequency sensitivity (Hz/G)
    t_max : float - Maximum time (s)
    n_realizations : int - Number of Monte Carlo realizations
    """
    t = np.linspace(0, t_max, 1000)
    dt = t[1] - t[0]

    coherence = np.zeros(len(t))

    for _ in range(n_realizations):
        # Generate random phase accumulation from noise
        # White noise model
        phase = np.cumsum(2 * np.pi * sensitivity * B_noise_rms *
                         np.random.randn(len(t)) * np.sqrt(dt))
        coherence += np.cos(phase)

    coherence /= n_realizations
    return t, coherence


def plot_energy_levels():
    """Plot energy level diagrams for different qubit types"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # Yb-171 Hyperfine levels
    ax1 = axes[0]
    yb = HyperfineQubit(A_hf=12.6428e9, I=0.5, J=0.5)

    # Plot levels at B=0 and B=5 G
    B_values = [0, 5]
    x_positions = [0.3, 0.7]

    for B, x in zip(B_values, x_positions):
        for F in yb.F_values:
            for mF in np.arange(-F, F + 1):
                E_base = yb.energies[F]
                E_zeeman = yb.zeeman_shift(F, mF, B)
                E_total = (E_base + E_zeeman) / 1e9  # Convert to GHz

                color = 'blue' if F == 0 else 'red'
                ax1.hlines(E_total, x - 0.1, x + 0.1, colors=color, linewidth=2)
                if B == 0:
                    ax1.text(x + 0.12, E_total, f'F={int(F)}', fontsize=10)

    ax1.set_xlim(0, 1)
    ax1.set_ylabel('Energy (GHz)', fontsize=12)
    ax1.set_title('$^{171}$Yb$^+$ Hyperfine Structure', fontsize=14)
    ax1.set_xticks([0.3, 0.7])
    ax1.set_xticklabels(['B = 0 G', 'B = 5 G'])

    # Ca-40 Zeeman levels
    ax2 = axes[1]
    zeeman = ZeemanQubit(gJ=2.0)

    B_range = np.linspace(0, 10, 100)
    freq_up = zeeman.transition_frequency(B_range) / 1e6  # MHz
    freq_down = -zeeman.transition_frequency(B_range) / 1e6

    ax2.plot(B_range, freq_up, 'b-', label=r'$m_J = +1/2$')
    ax2.plot(B_range, freq_down, 'r-', label=r'$m_J = -1/2$')
    ax2.set_xlabel('Magnetic Field (G)', fontsize=12)
    ax2.set_ylabel('Energy Shift (MHz)', fontsize=12)
    ax2.set_title('$^{40}$Ca$^+$ Zeeman Splitting', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Optical qubit energy diagram
    ax3 = axes[2]
    # Simplified Ca-40 levels
    levels = {
        'S_1/2': 0,
        'P_1/2': 755,  # in THz
        'P_3/2': 761,
        'D_3/2': 346,
        'D_5/2': 411
    }

    for name, energy in levels.items():
        if 'S' in name:
            color = 'blue'
        elif 'P' in name:
            color = 'red'
        else:
            color = 'green'
        ax3.hlines(energy, 0.2, 0.8, colors=color, linewidth=3)
        ax3.text(0.82, energy, name, fontsize=11, va='center')

    # Draw transitions
    ax3.annotate('', xy=(0.5, 411), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax3.text(0.52, 200, '729 nm\n(qubit)', fontsize=10, color='purple')

    ax3.annotate('', xy=(0.35, 755), xytext=(0.35, 0),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))
    ax3.text(0.15, 380, '397 nm\n(cooling)', fontsize=9, color='orange')

    ax3.set_xlim(0, 1.2)
    ax3.set_ylim(-50, 800)
    ax3.set_ylabel('Energy (THz)', fontsize=12)
    ax3.set_title('$^{40}$Ca$^+$ Level Structure', fontsize=14)
    ax3.set_xticks([])

    plt.tight_layout()
    plt.savefig('qubit_energy_levels.png', dpi=150)
    plt.show()


def compare_coherence_times():
    """Compare coherence for different qubit types under noise"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hyperfine clock transition coherence
    ax1 = axes[0, 0]
    T2_clock = 10.0  # 10 seconds for clock transition
    t = np.linspace(0, 30, 1000)
    signal = simulate_ramsey_decay(T2_clock, 2 * np.pi * 10, t)  # 10 Hz detuning
    ax1.plot(t, signal, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('P(|1⟩)', fontsize=12)
    ax1.set_title(f'Hyperfine Clock Qubit (T₂ = {T2_clock} s)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Zeeman qubit coherence
    ax2 = axes[0, 1]
    T2_zeeman = 0.001  # 1 ms for Zeeman qubit
    t = np.linspace(0, 5e-3, 1000)
    signal = simulate_ramsey_decay(T2_zeeman, 2 * np.pi * 1000, t)  # 1 kHz detuning
    ax2.plot(t * 1e3, signal, 'r-', linewidth=2)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('P(|1⟩)', fontsize=12)
    ax2.set_title(f'Zeeman Qubit (T₂ = {T2_zeeman*1e3} ms)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Optical qubit coherence (lifetime limited)
    ax3 = axes[1, 0]
    optical = OpticalQubit(wavelength=729e-9, lifetime=1.17)
    T2_optical = 0.5  # 500 ms (less than lifetime limit due to laser noise)
    t = np.linspace(0, 2, 1000)
    signal = simulate_ramsey_decay(T2_optical, 2 * np.pi * 5, t)  # 5 Hz detuning
    ax3.plot(t, signal, 'g-', linewidth=2, label='With laser noise')

    # Also show lifetime limit
    signal_ideal = simulate_ramsey_decay(optical.max_coherence_time(), 2 * np.pi * 5, t)
    ax3.plot(t, signal_ideal, 'g--', linewidth=1, alpha=0.5, label='Lifetime limit')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('P(|1⟩)', fontsize=12)
    ax3.set_title(f'Optical Qubit (T₂ = {T2_optical*1e3} ms)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Magnetic noise dephasing simulation
    ax4 = axes[1, 1]
    zeeman = ZeemanQubit()
    sens = zeeman.sensitivity()

    noise_levels = [10e-6, 50e-6, 100e-6]  # Different B noise levels in Gauss
    colors = ['blue', 'orange', 'red']

    for B_noise, color in zip(noise_levels, colors):
        t, coh = simulate_dephasing(B_noise, sens, 5e-3, n_realizations=200)
        ax4.plot(t * 1e3, coh, color=color, linewidth=2,
                label=f'δB = {B_noise*1e6:.0f} μG')

    ax4.set_xlabel('Time (ms)', fontsize=12)
    ax4.set_ylabel('Coherence ⟨cos(φ)⟩', fontsize=12)
    ax4.set_title('Dephasing from Magnetic Noise', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coherence_comparison.png', dpi=150)
    plt.show()


def clock_transition_analysis():
    """Analyze clock transition properties"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    yb = HyperfineQubit(A_hf=12.6428e9, I=0.5, J=0.5)

    # Second-order Zeeman shift
    ax1 = axes[0]
    B = np.linspace(0, 20, 200)  # Gauss
    shift = yb.clock_transition_shift(B, beta=310)  # Hz

    ax1.plot(B, shift / 1e3, 'b-', linewidth=2)
    ax1.set_xlabel('Magnetic Field (G)', fontsize=12)
    ax1.set_ylabel('Frequency Shift (kHz)', fontsize=12)
    ax1.set_title('Clock Transition Second-Order Zeeman Shift', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Mark typical operating point
    B_op = 5  # Gauss
    shift_op = yb.clock_transition_shift(B_op, beta=310)
    ax1.scatter([B_op], [shift_op/1e3], color='red', s=100, zorder=5)
    ax1.annotate(f'B = {B_op} G\nΔf = {shift_op:.1f} Hz',
                xy=(B_op, shift_op/1e3), xytext=(B_op+3, shift_op/1e3+2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    # Compare sensitivities
    ax2 = axes[1]
    qubit_types = ['Clock\n(Hyperfine)', 'Zeeman', 'Optical\n(Zeeman-like)']
    sensitivities = [310 * 5 / 1e6, 2.8, 1.5]  # MHz/G (clock at 5G for comparison)
    colors = ['blue', 'red', 'green']

    bars = ax2.bar(qubit_types, sensitivities, color=colors, alpha=0.7)
    ax2.set_ylabel('Sensitivity (MHz/G)', fontsize=12)
    ax2.set_title('Magnetic Field Sensitivity Comparison', fontsize=14)
    ax2.set_yscale('log')

    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{sens:.2f}',
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('clock_transition_analysis.png', dpi=150)
    plt.show()


def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 905: Qubit Encoding Schemes Simulation")
    print("=" * 60)

    # Initialize qubit models
    print("\n--- Hyperfine Qubit (Yb-171) ---")
    yb = HyperfineQubit(A_hf=12.6428e9, I=0.5, J=0.5)
    print(f"Hyperfine splitting: {yb.get_splitting()/1e9:.4f} GHz")
    print(f"Clock transition shift at B=5G: {yb.clock_transition_shift(5, 310):.1f} Hz")

    print("\n--- Zeeman Qubit ---")
    zeeman = ZeemanQubit()
    print(f"Transition frequency at B=5G: {zeeman.transition_frequency(5)/1e6:.2f} MHz")
    print(f"Sensitivity: {zeeman.sensitivity()/1e6:.2f} MHz/G")

    print("\n--- Optical Qubit (Ca-40) ---")
    optical = OpticalQubit(wavelength=729e-9, lifetime=1.17)
    print(f"Transition frequency: {optical.frequency/1e12:.1f} THz")
    print(f"Max coherence time: {optical.max_coherence_time():.2f} s")

    # Generate plots
    print("\nGenerating energy level diagrams...")
    plot_energy_levels()

    print("\nComparing coherence times...")
    compare_coherence_times()

    print("\nAnalyzing clock transition...")
    clock_transition_analysis()

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Hyperfine energy | $E_F = \frac{A_{hf}}{2}[F(F+1) - I(I+1) - J(J+1)]$ |
| Zeeman frequency | $\omega_Z = g_J \mu_B B / \hbar$ |
| Clock shift (2nd order) | $\Delta\omega = \beta B^2$ |
| Sensitivity | $\partial\omega/\partial B = g_F \mu_B m_F / \hbar$ |
| Coherence limit | $T_2 \leq 2T_1$ |

### Main Takeaways

1. **Hyperfine qubits** offer longest coherence (seconds to minutes) via clock transitions
2. **Zeeman qubits** are simpler but require excellent magnetic shielding
3. **Optical qubits** provide direct optical control with lifetime-limited coherence
4. **Clock transitions** have second-order field sensitivity, enabling record coherence
5. Ion species selection involves trade-offs between coherence, gate speed, and complexity
6. **Field sensitivity** determines shielding requirements and coherence limits

## Daily Checklist

- [ ] I can explain hyperfine structure and calculate splitting
- [ ] I understand Zeeman qubit encoding and field sensitivity
- [ ] I can analyze optical qubit coherence limits
- [ ] I can compare trade-offs between encoding schemes
- [ ] I understand clock transition properties
- [ ] I can calculate required shielding for target coherence
- [ ] I have run the computational lab and analyzed the results

## Preview of Day 906

Tomorrow we explore **Laser Cooling and State Preparation**:
- Doppler cooling physics and limits
- Resolved sideband cooling to motional ground state
- Optical pumping for state initialization
- Achieving near-unity state preparation fidelity

We will see how cooling enables the Lamb-Dicke regime essential for high-fidelity gates.

---

*Day 905 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
