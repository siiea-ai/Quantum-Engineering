# Day 906: Laser Cooling and State Preparation

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Doppler cooling, sideband cooling theory |
| Afternoon | 2 hours | Problem solving: cooling rates and limits |
| Evening | 2 hours | Computational lab: cooling dynamics simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the Doppler cooling limit** from photon recoil physics
2. **Explain resolved sideband cooling** and calculate cooling rates
3. **Analyze optical pumping** for state initialization
4. **Calculate mean phonon number** after cooling protocols
5. **Design cooling sequences** for quantum computing applications
6. **Understand Sisyphus and EIT cooling** as advanced techniques

## Core Content

### 1. Introduction to Laser Cooling

Laser cooling is essential for trapped ion quantum computing because:
- Reduces thermal motion to near ground state ($\bar{n} < 0.1$)
- Enables Lamb-Dicke regime for high-fidelity gates
- Provides state initialization through optical pumping
- Allows long ion storage times

**Cooling hierarchy:**
1. **Doppler cooling:** $T \sim 0.5$ mK, $\bar{n} \sim 10-20$
2. **Sideband cooling:** $T \sim 10$ μK, $\bar{n} < 0.1$
3. **Ground state preparation:** $\bar{n} \approx 0.01$

### 2. Doppler Cooling

#### Basic Mechanism

An ion moving toward a red-detuned laser absorbs photons preferentially when:
- Moving toward the laser (Doppler shift into resonance)
- Spontaneous emission is isotropic (no net momentum)

Net momentum transfer:
$$\langle \Delta p \rangle = \hbar k \cdot (\text{absorption rate} - 0) \propto -v$$

#### Scattering Force

The scattering rate from a two-level atom:

$$R_{sc} = \frac{\Gamma}{2} \cdot \frac{s}{1 + s + (2\delta/\Gamma)^2}$$

where:
- $\Gamma$ = natural linewidth
- $s = I/I_{sat}$ = saturation parameter
- $\delta = \omega_L - \omega_0$ = detuning

The cooling force:
$$F_{cool} = \hbar k R_{sc}(\delta - k \cdot v) \approx -\alpha v$$

**Damping coefficient:**
$$\boxed{\alpha = -\hbar k^2 \frac{8s\delta/\Gamma}{(1 + s + (2\delta/\Gamma)^2)^2}}$$

Maximum damping occurs at $\delta = -\Gamma/2$.

#### Doppler Limit

Cooling is balanced by heating from photon recoil:
$$E_{recoil} = \frac{(\hbar k)^2}{2m}$$

The **Doppler temperature limit:**

$$\boxed{T_D = \frac{\hbar\Gamma}{2k_B}}$$

| Ion | Transition | $\Gamma/2\pi$ | $T_D$ |
|-----|------------|---------------|-------|
| $^{40}$Ca$^+$ | 397 nm | 22 MHz | 0.5 mK |
| $^{171}$Yb$^+$ | 369 nm | 19 MHz | 0.5 mK |
| $^{9}$Be$^+$ | 313 nm | 19 MHz | 0.5 mK |

**Mean phonon number at Doppler limit:**
$$\boxed{\bar{n}_D = \frac{k_B T_D}{\hbar\omega} = \frac{\Gamma}{2\omega}}$$

For $\omega/2\pi = 1$ MHz and $\Gamma/2\pi = 20$ MHz: $\bar{n}_D \approx 10$.

### 3. Resolved Sideband Cooling

When $\omega > \Gamma$ (resolved sideband regime), we can cool beyond the Doppler limit.

#### Sideband Structure

The ion-laser interaction includes motional sidebands:

$$\hat{H}_{int} = \frac{\hbar\Omega}{2}(\hat{\sigma}_+ + \hat{\sigma}_-)(e^{i\eta(\hat{a} + \hat{a}^\dagger)} e^{-i\omega_L t} + h.c.)$$

Expanding in Lamb-Dicke regime ($\eta \ll 1$):

$$e^{i\eta(\hat{a} + \hat{a}^\dagger)} \approx 1 + i\eta(\hat{a} + \hat{a}^\dagger) - \frac{\eta^2}{2}(\hat{a} + \hat{a}^\dagger)^2 + ...$$

This gives transitions at:
- **Carrier:** $\omega_L = \omega_0$ (no phonon change)
- **Red sideband (RSB):** $\omega_L = \omega_0 - \omega$ (removes one phonon)
- **Blue sideband (BSB):** $\omega_L = \omega_0 + \omega$ (adds one phonon)

#### Sideband Cooling Protocol

1. Drive the **red sideband**: $|g, n\rangle \rightarrow |e, n-1\rangle$
2. Spontaneous emission: $|e, n-1\rangle \rightarrow |g, n-1\rangle$ (carrier)

Net effect: Remove one phonon per cycle.

**Red sideband Rabi frequency:**
$$\boxed{\Omega_{RSB} = \eta\sqrt{n}\Omega_0}$$

**Blue sideband Rabi frequency:**
$$\Omega_{BSB} = \eta\sqrt{n+1}\Omega_0$$

#### Cooling Rate and Limit

The cooling rate:
$$\dot{\bar{n}} = -W_{-}\bar{n} + W_{+}(\bar{n} + 1)$$

where:
- $W_{-} = \eta^2 \Omega^2/\Gamma$ (cooling rate per phonon)
- $W_{+}$ (heating rate from carrier spontaneous emission on sidebands)

**Sideband cooling limit:**
$$\boxed{\bar{n}_{min} = \frac{W_+}{W_- - W_+} \approx \left(\frac{\Gamma}{2\omega}\right)^2}$$

For $\Gamma/\omega = 0.1$: $\bar{n}_{min} \approx 0.0025$ (near ground state!).

#### Practical Considerations

| Parameter | Requirement | Reason |
|-----------|-------------|--------|
| $\omega/\Gamma$ | > 5-10 | Resolved sidebands |
| $\eta$ | 0.05-0.2 | Lamb-Dicke regime |
| Cooling cycles | 10-50 | Starting from $\bar{n} \sim 10$ |
| Pulse duration | $\pi/\Omega_{RSB}$ | Complete transfer |

### 4. Optical Pumping for State Initialization

Optical pumping prepares qubits in a specific internal state.

#### Protocol for Hyperfine Qubits

For $^{171}$Yb$^+$:
1. Apply resonant light on $|F=1\rangle \rightarrow |F'=0\rangle$ at 369 nm
2. Ions in $|F=1\rangle$ scatter photons and decay to $|F=0\rangle$ or $|F=1\rangle$
3. $|F=0\rangle$ is dark (no transition to $F'=0$)
4. Population accumulates in $|F=0, m_F=0\rangle$

**Initialization fidelity:**
$$F_{init} = 1 - e^{-N_{scatter}} \approx 1 - 10^{-4}$$

after ~10 scattering events.

#### Protocol for Zeeman/Optical Qubits

1. Apply $\sigma^+$ polarized light
2. Repeated absorption/emission pumps to $|m_J = +1/2\rangle$
3. Typical fidelity: 99.9% in a few μs

### 5. Complete Cooling Sequence

A typical initialization sequence for quantum computing:

```
1. Doppler cooling (1-2 ms)
   - 397 nm + 866 nm (Ca+) or 369 nm + 935 nm (Yb+)
   - Achieves T ~ 0.5 mK, n̄ ~ 10

2. Sideband cooling (1-5 ms per mode)
   - Drive red sideband at ω₀ - ω
   - Repump excited state
   - Repeat 10-50 cycles
   - Achieves n̄ < 0.1

3. Optical pumping (1-10 μs)
   - Polarized light to desired |0⟩ state
   - 99.9%+ state preparation

Total: 5-10 ms initialization time
```

### 6. Advanced Cooling Techniques

#### Electromagnetically Induced Transparency (EIT) Cooling

Uses quantum interference between transition pathways:
- Three-level system with two lasers
- Dark state at red detuning
- Can achieve $\bar{n} < 0.01$ faster than sideband cooling

**EIT cooling rate:**
$$W_{EIT} \propto \frac{\Omega_1^2 \Omega_2^2}{\Gamma \delta^2}$$

#### Sisyphus Cooling

Uses position-dependent light shifts:
- Ion climbs potential hill, loses kinetic energy
- Optical pumping at top resets to bottom
- Works in resolved sideband regime

#### Sympathetic Cooling

Cool one ion species to cool another:
- Laser-cooled "coolant" ion (e.g., $^{24}$Mg$^+$)
- Sympathetically cooled "qubit" ion (e.g., $^{27}$Al$^+$)
- Coulomb coupling transfers energy
- Essential for ion clocks and some QC architectures

### 7. Heating Mechanisms

Understanding heating is crucial for maintaining ground state:

#### Anomalous Heating

Electric field noise from electrode surfaces:
$$\dot{\bar{n}} = \frac{q^2}{4m\hbar\omega}S_E(\omega)$$

where $S_E(\omega)$ is the electric field noise spectral density.

**Empirical scaling:**
$$S_E \propto d^{-4} \cdot \omega^{-\alpha}$$

where $d$ is ion-electrode distance and $\alpha \approx 1-2$.

| Trap type | $d$ | $\dot{\bar{n}}$ |
|-----------|-----|-----------------|
| Macroscopic | 500 μm | 1-10 /s |
| Microfabricated | 50 μm | 100-10000 /s |

#### Mitigation Strategies

1. **Larger ion-electrode distance** (reduces noise)
2. **Cryogenic operation** (reduces surface noise)
3. **Electrode cleaning/treatment** (Ar ion bombardment)
4. **Fast gates** (minimize heating during operations)

## Quantum Computing Applications

### Initialization Requirements

| Application | $\bar{n}$ Required | Cooling Method |
|-------------|-------------------|----------------|
| Single-qubit gates | < 1 | Doppler |
| Two-qubit gates | < 0.1 | Sideband |
| Quantum simulation | < 0.05 | Ground state |
| Metrology | < 0.01 | EIT/multiple modes |

### Cooling All Motional Modes

For $N$ ions, there are $3N$ motional modes. Efficient cooling requires:
- Addressing all relevant modes (axial + radial)
- Pulsed or continuous sideband cooling
- Mode-specific cooling sequences

**Typical mode structure (2 ions):**
- Axial: COM ($\omega_z$), stretch ($\sqrt{3}\omega_z$)
- Radial: COM ($\omega_r$), rocking, etc.

## Worked Examples

### Example 1: Doppler Cooling Limit

**Problem:** Calculate the Doppler limit and mean phonon number for $^{171}$Yb$^+$ on the 369 nm transition ($\Gamma/2\pi = 19$ MHz, $\omega_z/2\pi = 1$ MHz).

**Solution:**

Doppler temperature:
$$T_D = \frac{\hbar\Gamma}{2k_B} = \frac{1.055 \times 10^{-34} \times 2\pi \times 19 \times 10^6}{2 \times 1.38 \times 10^{-23}}$$

$$\boxed{T_D = 0.45 \text{ mK}}$$

Mean phonon number:
$$\bar{n}_D = \frac{k_B T_D}{\hbar\omega_z} = \frac{\Gamma}{2\omega_z} = \frac{19 \times 10^6}{2 \times 10^6}$$

$$\boxed{\bar{n}_D = 9.5}$$

### Example 2: Sideband Cooling Time

**Problem:** Estimate the time to cool from $\bar{n} = 10$ to $\bar{n} = 0.1$ using sideband cooling with $\Omega_0/2\pi = 100$ kHz, $\eta = 0.1$, $\Gamma/2\pi = 20$ MHz.

**Solution:**

The cooling rate per phonon:
$$W_- = \frac{\eta^2 \Omega_0^2}{\Gamma} = \frac{0.01 \times (2\pi \times 10^5)^2}{2\pi \times 2 \times 10^7}$$

$$W_- = \frac{0.01 \times 4\pi^2 \times 10^{10}}{2\pi \times 2 \times 10^7} = 2\pi \times 500 \text{ Hz}$$

Time to remove one phonon (average):
$$\tau_1 = \frac{1}{W_-} = \frac{1}{2\pi \times 500} \approx 0.3 \text{ ms}$$

Number of phonons to remove: $\Delta n \approx 10$

$$\boxed{t_{cool} \approx 10 \times 0.3 \text{ ms} = 3 \text{ ms}}$$

(Note: This is approximate; actual cooling follows exponential decay.)

### Example 3: Sideband Cooling Limit

**Problem:** What is the minimum achievable $\bar{n}$ for $^{40}$Ca$^+$ with $\omega_z/2\pi = 1.2$ MHz and $\Gamma/2\pi = 22$ MHz on the S-P transition, using D-state repumping?

**Solution:**

For Raman sideband cooling through an excited state:
$$\bar{n}_{min} \approx \left(\frac{\Gamma}{2\omega_z}\right)^2 = \left(\frac{22}{2 \times 1.2}\right)^2 = \left(\frac{22}{2.4}\right)^2$$

$$\bar{n}_{min} = (9.17)^2 = 84$$

This is too high! The issue is that the P-state linewidth is much larger than the trap frequency.

**Solution:** Use the narrow 729 nm quadrupole transition ($\Gamma_{729}/2\pi \approx 0.14$ Hz):

$$\bar{n}_{min} \approx \left(\frac{\Gamma_{729}}{2\omega_z}\right)^2 = \left(\frac{0.14}{2.4 \times 10^6}\right)^2 \approx 10^{-15}$$

$$\boxed{\bar{n}_{min} \approx 0 \text{ (ground state)}}$$

In practice, $\bar{n} \approx 0.01-0.05$ is achievable, limited by heating during the cooling cycle.

## Practice Problems

### Level 1: Direct Application

1. Calculate the Doppler limit for $^9$Be$^+$ with $\Gamma/2\pi = 19$ MHz.

2. If $\bar{n}_D = 15$ after Doppler cooling, how many sideband cooling pulses are needed to reach $\bar{n} = 0.1$?

3. What is the recoil energy for a 397 nm photon absorbed by $^{40}$Ca$^+$?

### Level 2: Intermediate

4. Derive the optimal detuning for maximum Doppler cooling rate and verify it is $\delta = -\Gamma/2$.

5. For a two-ion crystal, design a cooling sequence that addresses both the COM and stretch modes. Estimate total cooling time.

6. Calculate the heating-limited gate fidelity for a trap with $\dot{\bar{n}} = 500$ quanta/s and gate time 50 μs.

### Level 3: Challenging

7. Derive the sideband cooling limit $\bar{n}_{min}$ by balancing cooling and heating rates from off-resonant carrier transitions.

8. Analyze EIT cooling: derive the cooling rate and dark state condition for a three-level system.

9. Design a sympathetic cooling protocol for $^{27}$Al$^+$ using $^{25}$Mg$^+$. Calculate the cooling timescale from mode coupling.

## Computational Lab: Cooling Dynamics Simulation

```python
"""
Day 906 Computational Lab: Laser Cooling Simulation
Simulating Doppler cooling, sideband cooling, and state preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm

# Physical constants
hbar = 1.055e-34
kB = 1.38e-23
amu = 1.661e-27

class DopplerCooling:
    """Simulate Doppler cooling dynamics"""

    def __init__(self, mass_amu, wavelength, gamma):
        """
        Parameters:
        -----------
        mass_amu : float - Ion mass in amu
        wavelength : float - Cooling transition wavelength (m)
        gamma : float - Natural linewidth (rad/s)
        """
        self.m = mass_amu * amu
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.gamma = gamma

        # Doppler limit
        self.T_D = hbar * gamma / (2 * kB)
        self.E_recoil = (hbar * self.k)**2 / (2 * self.m)

    def scattering_rate(self, detuning, saturation):
        """Calculate scattering rate"""
        s = saturation
        delta = detuning
        return (self.gamma / 2) * s / (1 + s + (2 * delta / self.gamma)**2)

    def cooling_rate(self, detuning, saturation):
        """Calculate cooling rate coefficient"""
        s = saturation
        delta = detuning
        numerator = 8 * s * delta / self.gamma
        denominator = (1 + s + (2 * delta / self.gamma)**2)**2
        return -hbar * self.k**2 * numerator / denominator

    def simulate(self, T_initial, detuning, saturation, t_max):
        """
        Simulate temperature evolution during Doppler cooling

        Returns time array and temperature array
        """
        def dT_dt(T, t):
            # Cooling power
            v_rms = np.sqrt(kB * T / self.m)
            P_cool = -self.cooling_rate(detuning, saturation) * v_rms**2

            # Heating power from recoil
            R = self.scattering_rate(detuning, saturation)
            P_heat = 2 * self.E_recoil * R / (3 * kB)  # Factor 1/3 for 3D

            return (P_heat - P_cool / kB)

        t = np.linspace(0, t_max, 1000)
        T = odeint(dT_dt, T_initial, t)
        return t, T.flatten()


class SidebandCooling:
    """Simulate resolved sideband cooling"""

    def __init__(self, omega_trap, gamma, eta, omega_rabi):
        """
        Parameters:
        -----------
        omega_trap : float - Trap frequency (rad/s)
        gamma : float - Decay rate of excited state (rad/s)
        eta : float - Lamb-Dicke parameter
        omega_rabi : float - Carrier Rabi frequency (rad/s)
        """
        self.omega = omega_trap
        self.gamma = gamma
        self.eta = eta
        self.Omega0 = omega_rabi

        # Cooling rate per phonon
        self.W_minus = eta**2 * omega_rabi**2 / gamma

        # Heating rate (from off-resonant carrier)
        self.W_plus = (eta**2 * gamma / 4) * (gamma / (2 * omega_trap))**2

    def n_min(self):
        """Minimum achievable phonon number"""
        return self.W_plus / (self.W_minus - self.W_plus)

    def simulate(self, n_initial, t_max):
        """Simulate phonon number evolution"""
        def dn_dt(n, t):
            return -self.W_minus * n + self.W_plus * (n + 1)

        t = np.linspace(0, t_max, 1000)
        n = odeint(dn_dt, n_initial, t)
        return t, n.flatten()

    def pulsed_cooling(self, n_initial, n_pulses):
        """Simulate pulsed sideband cooling"""
        n_values = [n_initial]
        n = n_initial

        for i in range(n_pulses):
            if n > 0:
                # Red sideband pi pulse: |g,n⟩ → |e,n-1⟩
                # Probability of successful removal
                p_success = 1 - np.exp(-self.eta**2 * n)

                # Average phonon removal
                n = n * (1 - p_success) + (n - 1) * p_success
                n = max(0, n)

            n_values.append(n)

        return np.arange(n_pulses + 1), np.array(n_values)


class OpticalPumping:
    """Simulate optical pumping for state preparation"""

    def __init__(self, pumping_rate, n_levels=3):
        """
        Parameters:
        -----------
        pumping_rate : float - Optical pumping rate (rad/s)
        n_levels : int - Number of ground state sublevels
        """
        self.R = pumping_rate
        self.n_levels = n_levels

    def simulate(self, initial_pop, t_max):
        """
        Simulate optical pumping dynamics

        initial_pop : array - Initial population distribution
        """
        # Simple model: pump to dark state (last level)
        n = self.n_levels

        # Rate matrix: all states pump to dark state
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i] = -self.R
            W[n - 1, i] = self.R

        def dpop_dt(pop, t):
            return W @ pop

        t = np.linspace(0, t_max, 500)
        pop = odeint(dpop_dt, initial_pop, t)
        return t, pop


def plot_doppler_cooling():
    """Plot Doppler cooling dynamics and optimization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create Yb-171 Doppler cooling system
    yb_cooling = DopplerCooling(
        mass_amu=171,
        wavelength=369e-9,
        gamma=2 * np.pi * 19e6
    )

    print(f"Doppler temperature limit: {yb_cooling.T_D * 1e3:.3f} mK")
    print(f"Recoil energy: {yb_cooling.E_recoil / kB * 1e6:.3f} μK")

    # Temperature vs time
    ax1 = axes[0, 0]
    T_init = 300  # Starting from room temperature (just for illustration)

    for s in [0.1, 1, 10]:
        t, T = yb_cooling.simulate(
            T_initial=10e-3,  # 10 mK
            detuning=-yb_cooling.gamma / 2,
            saturation=s,
            t_max=5e-3
        )
        ax1.semilogy(t * 1e3, T * 1e3, label=f's = {s}')

    ax1.axhline(y=yb_cooling.T_D * 1e3, color='red', linestyle='--', label=f'Doppler limit')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Temperature (mK)', fontsize=12)
    ax1.set_title('Doppler Cooling Dynamics', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cooling rate vs detuning
    ax2 = axes[0, 1]
    detuning = np.linspace(-3 * yb_cooling.gamma, 0.5 * yb_cooling.gamma, 200)

    for s in [0.1, 1, 10]:
        rate = yb_cooling.cooling_rate(detuning, s)
        ax2.plot(detuning / yb_cooling.gamma, -rate * 1e15, label=f's = {s}')

    ax2.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Detuning (Γ)', fontsize=12)
    ax2.set_ylabel('Cooling rate (arb. units)', fontsize=12)
    ax2.set_title('Cooling Rate vs Detuning', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Scattering rate
    ax3 = axes[1, 0]
    for s in [0.1, 1, 10]:
        R = yb_cooling.scattering_rate(detuning, s)
        ax3.plot(detuning / yb_cooling.gamma, R / yb_cooling.gamma, label=f's = {s}')

    ax3.set_xlabel('Detuning (Γ)', fontsize=12)
    ax3.set_ylabel('Scattering rate (Γ)', fontsize=12)
    ax3.set_title('Scattering Rate vs Detuning', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mean phonon number at Doppler limit
    ax4 = axes[1, 1]
    omega_trap = np.linspace(0.5e6, 5e6, 100) * 2 * np.pi
    n_bar = yb_cooling.gamma / (2 * omega_trap)

    ax4.plot(omega_trap / (2 * np.pi * 1e6), n_bar, 'b-', linewidth=2)
    ax4.axhline(y=10, color='red', linestyle='--', label='n̄ = 10')
    ax4.set_xlabel('Trap frequency (MHz)', fontsize=12)
    ax4.set_ylabel('Mean phonon number n̄', fontsize=12)
    ax4.set_title('Doppler Limit: n̄ vs Trap Frequency', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('doppler_cooling.png', dpi=150)
    plt.show()


def plot_sideband_cooling():
    """Plot sideband cooling dynamics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create sideband cooling system
    sbc = SidebandCooling(
        omega_trap=2 * np.pi * 1e6,  # 1 MHz trap
        gamma=2 * np.pi * 22e6,  # 22 MHz linewidth (Ca+)
        eta=0.1,
        omega_rabi=2 * np.pi * 100e3  # 100 kHz Rabi
    )

    print(f"\nSideband cooling:")
    print(f"Cooling rate W_-: {sbc.W_minus:.1f} /s")
    print(f"Minimum n̄: {sbc.n_min():.4f}")

    # Continuous sideband cooling
    ax1 = axes[0, 0]
    for n0 in [5, 10, 20]:
        t, n = sbc.simulate(n0, t_max=10e-3)
        ax1.semilogy(t * 1e3, n, label=f'n₀ = {n0}')

    ax1.axhline(y=sbc.n_min(), color='red', linestyle='--', label='n̄_min')
    ax1.axhline(y=0.1, color='green', linestyle=':', label='Target n̄ = 0.1')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Mean phonon number n̄', fontsize=12)
    ax1.set_title('Continuous Sideband Cooling', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-3, 50)

    # Pulsed sideband cooling
    ax2 = axes[0, 1]
    pulses, n_pulsed = sbc.pulsed_cooling(n_initial=10, n_pulses=50)
    ax2.semilogy(pulses, n_pulsed, 'bo-', markersize=3)
    ax2.axhline(y=0.1, color='green', linestyle=':', label='Target n̄ = 0.1')
    ax2.set_xlabel('Number of cooling pulses', fontsize=12)
    ax2.set_ylabel('Mean phonon number n̄', fontsize=12)
    ax2.set_title('Pulsed Sideband Cooling', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Sideband spectrum
    ax3 = axes[1, 0]
    n_bar = 5  # After some Doppler cooling

    # Detuning from carrier
    delta = np.linspace(-2.5, 2.5, 1000) * sbc.omega / (2 * np.pi)

    # Sideband transitions (simplified Lorentzian model)
    carrier = 1 / (1 + (delta / 1e5)**2)

    rsb = sbc.eta**2 * n_bar / (1 + ((delta + sbc.omega/(2*np.pi)) / 1e5)**2)
    bsb = sbc.eta**2 * (n_bar + 1) / (1 + ((delta - sbc.omega/(2*np.pi)) / 1e5)**2)

    ax3.plot(delta / 1e6, carrier, 'b-', label='Carrier', linewidth=2)
    ax3.plot(delta / 1e6, rsb * 10, 'r-', label='Red sideband (×10)', linewidth=2)
    ax3.plot(delta / 1e6, bsb * 10, 'g-', label='Blue sideband (×10)', linewidth=2)

    ax3.set_xlabel('Detuning from carrier (MHz)', fontsize=12)
    ax3.set_ylabel('Transition strength (arb.)', fontsize=12)
    ax3.set_title('Sideband Spectrum (n̄ = 5)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Ground state population vs cooling cycles
    ax4 = axes[1, 1]

    # Phonon distribution after cooling
    n_values = np.arange(0, 20)
    for n_bar_final in [5, 1, 0.1, 0.01]:
        # Thermal distribution
        p_n = (1 - np.exp(-1/n_bar_final)) * np.exp(-n_values / n_bar_final) if n_bar_final > 0 else np.zeros(20)
        p_n[0] = 1 / (1 + n_bar_final) if n_bar_final > 0 else 1
        ax4.bar(n_values + 0.2 * np.log10(n_bar_final + 0.01), p_n, width=0.15,
               label=f'n̄ = {n_bar_final}', alpha=0.7)

    ax4.set_xlabel('Phonon number n', fontsize=12)
    ax4.set_ylabel('Probability P(n)', fontsize=12)
    ax4.set_title('Phonon Distribution After Cooling', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.5, 10)

    plt.tight_layout()
    plt.savefig('sideband_cooling.png', dpi=150)
    plt.show()


def plot_optical_pumping():
    """Plot optical pumping dynamics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Three-level system (e.g., F=1 to F=0)
    op = OpticalPumping(pumping_rate=1e6, n_levels=3)

    # Initial equal population
    initial = np.array([1/3, 1/3, 1/3])

    t, pop = op.simulate(initial, t_max=10e-6)

    ax1 = axes[0]
    labels = ['|m=-1⟩', '|m=0⟩', '|m=+1⟩ (dark)']
    colors = ['blue', 'orange', 'green']

    for i in range(3):
        ax1.plot(t * 1e6, pop[:, i], label=labels[i], color=colors[i], linewidth=2)

    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Population', fontsize=12)
    ax1.set_title('Optical Pumping Dynamics', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fidelity vs pumping time
    ax2 = axes[1]
    pumping_times = np.linspace(0, 20e-6, 100)
    fidelities = []

    for t_pump in pumping_times:
        _, pop_final = op.simulate(initial, t_max=t_pump)
        fidelity = pop_final[-1, -1]  # Population in dark state
        fidelities.append(fidelity)

    ax2.plot(pumping_times * 1e6, fidelities, 'b-', linewidth=2)
    ax2.axhline(y=0.999, color='red', linestyle='--', label='99.9% fidelity')
    ax2.axhline(y=0.9999, color='green', linestyle=':', label='99.99% fidelity')

    ax2.set_xlabel('Pumping time (μs)', fontsize=12)
    ax2.set_ylabel('State preparation fidelity', fontsize=12)
    ax2.set_title('Optical Pumping Fidelity', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.99, 1.001)

    plt.tight_layout()
    plt.savefig('optical_pumping.png', dpi=150)
    plt.show()


def plot_complete_sequence():
    """Plot complete initialization sequence"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Simulated sequence timing
    t_doppler = 2e-3  # 2 ms
    t_sideband = 3e-3  # 3 ms
    t_pumping = 5e-6  # 5 μs

    # Generate data
    t1 = np.linspace(0, t_doppler, 200)
    n_doppler = 50 * np.exp(-t1 / 0.5e-3) + 10  # Cooling to n̄ ~ 10

    t2 = np.linspace(t_doppler, t_doppler + t_sideband, 200)
    n_sideband = 10 * np.exp(-(t2 - t_doppler) / 0.8e-3) + 0.05  # Cooling to n̄ ~ 0.05

    t3 = np.linspace(t_doppler + t_sideband, t_doppler + t_sideband + 50e-6, 50)
    n_final = np.ones(50) * 0.05  # Maintained during optical pumping

    # Combine
    t_total = np.concatenate([t1, t2, t3])
    n_total = np.concatenate([n_doppler, n_sideband, n_final])

    ax.semilogy(t_total * 1e3, n_total, 'b-', linewidth=2)

    # Mark phases
    ax.axvspan(0, t_doppler * 1e3, alpha=0.2, color='blue', label='Doppler cooling')
    ax.axvspan(t_doppler * 1e3, (t_doppler + t_sideband) * 1e3, alpha=0.2, color='green',
               label='Sideband cooling')
    ax.axvspan((t_doppler + t_sideband) * 1e3, (t_doppler + t_sideband + 50e-6) * 1e3,
               alpha=0.2, color='orange', label='Optical pumping')

    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='Target n̄ = 0.1')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Mean phonon number n̄', fontsize=12)
    ax.set_title('Complete Initialization Sequence', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 100)

    plt.tight_layout()
    plt.savefig('complete_initialization.png', dpi=150)
    plt.show()


def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 906: Laser Cooling Simulation")
    print("=" * 60)

    print("\nGenerating Doppler cooling plots...")
    plot_doppler_cooling()

    print("\nGenerating sideband cooling plots...")
    plot_sideband_cooling()

    print("\nGenerating optical pumping plots...")
    plot_optical_pumping()

    print("\nGenerating complete sequence plot...")
    plot_complete_sequence()

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
| Doppler limit | $T_D = \frac{\hbar\Gamma}{2k_B}$ |
| Mean phonon (Doppler) | $\bar{n}_D = \frac{\Gamma}{2\omega}$ |
| RSB Rabi frequency | $\Omega_{RSB} = \eta\sqrt{n}\Omega_0$ |
| Sideband cooling rate | $W_- = \frac{\eta^2\Omega^2}{\Gamma}$ |
| Sideband limit | $\bar{n}_{min} \approx \left(\frac{\Gamma}{2\omega}\right)^2$ |
| Recoil energy | $E_R = \frac{(\hbar k)^2}{2m}$ |

### Main Takeaways

1. **Doppler cooling** achieves $T \sim 0.5$ mK, $\bar{n} \sim 10$ in milliseconds
2. **Sideband cooling** reaches near ground state ($\bar{n} < 0.1$) in resolved sideband regime
3. **Optical pumping** initializes internal state with >99.9% fidelity in microseconds
4. **Heating rates** set limits on how long ground state can be maintained
5. Complete initialization takes 5-10 ms per ion
6. Advanced techniques (EIT, sympathetic) enable faster or deeper cooling

## Daily Checklist

- [ ] I can derive and calculate the Doppler cooling limit
- [ ] I understand sideband structure and resolved sideband regime
- [ ] I can calculate sideband cooling rates and limits
- [ ] I understand optical pumping for state preparation
- [ ] I can design a complete cooling and initialization sequence
- [ ] I understand heating mechanisms and mitigation strategies
- [ ] I have run the computational lab simulations

## Preview of Day 907

Tomorrow we explore **Single-Qubit Gates** in trapped ions:
- Stimulated Raman transitions for coherent control
- Microwave gates for hyperfine qubits
- Pulse shaping and composite pulses
- Gate fidelity characterization

We will learn how to rotate qubits on the Bloch sphere with high precision.

---

*Day 906 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
