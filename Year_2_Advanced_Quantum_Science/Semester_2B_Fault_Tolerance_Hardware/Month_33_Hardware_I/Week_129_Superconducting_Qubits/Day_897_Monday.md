# Day 897: Circuit QED Fundamentals

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Quantum LC circuits, transmission line resonators |
| Afternoon | 2 hours | Jaynes-Cummings model, problem solving |
| Evening | 2 hours | Computational lab: Circuit quantization simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Quantize** a classical LC circuit using canonical quantization
2. **Derive** the Hamiltonian of a transmission line resonator
3. **Explain** how Josephson junctions introduce nonlinearity essential for qubits
4. **Apply** the Jaynes-Cummings model to describe qubit-resonator coupling
5. **Calculate** vacuum Rabi splitting and strong coupling conditions
6. **Simulate** coupled qubit-resonator dynamics numerically

## Core Content

### 1. The LC Oscillator: A Quantum Harmonic Oscillator

The simplest superconducting circuit is the LC oscillator. We begin with the classical Lagrangian:

$$\mathcal{L} = \frac{1}{2}C\dot{\Phi}^2 - \frac{\Phi^2}{2L}$$

where $\Phi$ is the magnetic flux through the inductor. The conjugate variable to flux is charge:

$$Q = \frac{\partial \mathcal{L}}{\partial \dot{\Phi}} = C\dot{\Phi}$$

The classical Hamiltonian becomes:

$$H = \frac{Q^2}{2C} + \frac{\Phi^2}{2L}$$

**Canonical Quantization**: We promote $Q$ and $\Phi$ to operators satisfying:

$$[\hat{\Phi}, \hat{Q}] = i\hbar$$

This is analogous to $[x, p] = i\hbar$ in mechanical systems. The quantum Hamiltonian is:

$$\boxed{\hat{H}_{LC} = \frac{\hat{Q}^2}{2C} + \frac{\hat{\Phi}^2}{2L}}$$

### 2. Creation and Annihilation Operators

We introduce ladder operators:

$$\hat{a} = \sqrt{\frac{1}{2\hbar Z_r}}(\hat{\Phi} + iZ_r\hat{Q})$$

$$\hat{a}^\dagger = \sqrt{\frac{1}{2\hbar Z_r}}(\hat{\Phi} - iZ_r\hat{Q})$$

where $Z_r = \sqrt{L/C}$ is the characteristic impedance and $\omega_r = 1/\sqrt{LC}$ is the resonant frequency.

The Hamiltonian becomes:

$$\boxed{\hat{H}_{LC} = \hbar\omega_r\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right)}$$

with energy eigenvalues $E_n = \hbar\omega_r(n + 1/2)$. The equally spaced levels make a harmonic oscillator unsuitable as a qubit—we cannot selectively address a two-level subspace.

### 3. Transmission Line Resonators

In circuit QED, we typically use coplanar waveguide (CPW) resonators rather than lumped LC circuits. A transmission line of length $\ell$ supports standing wave modes.

For a $\lambda/2$ resonator with open boundary conditions at both ends:

$$\omega_n = \frac{n\pi v}{\ell}$$

where $v = 1/\sqrt{l c}$ is the phase velocity ($l$ and $c$ are inductance and capacitance per unit length).

The fundamental mode ($n=1$) is typically designed for $\omega_r/2\pi \approx 5-8$ GHz. Each mode acts as an independent quantum harmonic oscillator:

$$\hat{H}_{res} = \sum_n \hbar\omega_n\left(\hat{a}_n^\dagger\hat{a}_n + \frac{1}{2}\right)$$

**Why CPW resonators?**
- High quality factors ($Q > 10^6$)
- Easy fabrication with standard lithography
- Natural coupling to superconducting qubits
- Planar geometry compatible with scaling

### 4. The Josephson Junction: Nonlinearity

The critical ingredient for making qubits is the **Josephson junction**—a thin insulating barrier between two superconductors. The Josephson relations are:

$$I = I_c \sin\varphi$$
$$V = \frac{\hbar}{2e}\dot{\varphi} = \frac{\Phi_0}{2\pi}\dot{\varphi}$$

where $\varphi$ is the gauge-invariant phase difference, $I_c$ is the critical current, and $\Phi_0 = h/2e$ is the flux quantum.

The Josephson junction stores energy:

$$U_J = -E_J\cos\varphi$$

where the Josephson energy is:

$$\boxed{E_J = \frac{\Phi_0 I_c}{2\pi} = \frac{\hbar I_c}{2e}}$$

This cosine potential is **nonlinear**, unlike the parabolic potential of an LC circuit.

### 5. Cooper Pair Box: Simplest Superconducting Qubit

Consider a Josephson junction shunted by a capacitance $C$:

$$\hat{H}_{CPB} = 4E_C(\hat{n} - n_g)^2 - E_J\cos\hat{\varphi}$$

where:
- $E_C = e^2/2C$ is the charging energy
- $\hat{n}$ is the number of Cooper pairs on the island
- $n_g = C_g V_g / 2e$ is the gate-induced offset charge
- $[\hat{\varphi}, \hat{n}] = i$

The ratio $E_J/E_C$ determines the qubit character:
- $E_J/E_C \ll 1$: Charge qubit regime (sensitive to charge noise)
- $E_J/E_C \gg 1$: Transmon regime (charge-insensitive)

### 6. The Jaynes-Cummings Model

When a qubit is capacitively coupled to a resonator, the combined system is described by the **Jaynes-Cummings Hamiltonian**:

$$\boxed{\hat{H}_{JC} = \hbar\omega_r\hat{a}^\dagger\hat{a} + \frac{\hbar\omega_q}{2}\hat{\sigma}_z + \hbar g(\hat{a}^\dagger\hat{\sigma}^- + \hat{a}\hat{\sigma}^+)}$$

where:
- $\omega_r$: resonator frequency
- $\omega_q$: qubit frequency
- $g$: coupling strength
- $\hat{\sigma}^{\pm} = (\hat{\sigma}_x \pm i\hat{\sigma}_y)/2$

The interaction term describes:
- $\hat{a}^\dagger\hat{\sigma}^-$: photon creation + qubit de-excitation
- $\hat{a}\hat{\sigma}^+$: photon annihilation + qubit excitation

This conserves the total excitation number $\hat{N} = \hat{a}^\dagger\hat{a} + \hat{\sigma}^+\hat{\sigma}^-$.

### 7. Strong Coupling and Vacuum Rabi Splitting

The coupling strength is:

$$g = \frac{e\beta V_{rms}}{\hbar} = \frac{e\beta}{\hbar}\sqrt{\frac{\hbar\omega_r}{2C_r}}$$

where $\beta$ is the voltage division ratio and $V_{rms}$ is the zero-point voltage fluctuation.

**Strong coupling condition**: $g \gg \kappa, \gamma$

where $\kappa = \omega_r/Q_r$ is the resonator decay rate and $\gamma = 1/T_1$ is the qubit decay rate.

At resonance ($\omega_q = \omega_r$), the dressed states are:

$$|+, n\rangle = \frac{1}{\sqrt{2}}(|e, n\rangle + |g, n+1\rangle)$$
$$|-, n\rangle = \frac{1}{\sqrt{2}}(|e, n\rangle - |g, n+1\rangle)$$

with energies split by **vacuum Rabi splitting**:

$$\boxed{\Delta E = 2\hbar g\sqrt{n+1}}$$

For the vacuum state ($n=0$), this gives $\Delta E = 2\hbar g$, observable in spectroscopy.

### 8. Dispersive Regime

When qubit and resonator are detuned ($|\Delta| = |\omega_q - \omega_r| \gg g$), the Jaynes-Cummings Hamiltonian can be approximated via perturbation theory:

$$\hat{H}_{disp} \approx \hbar\omega_r\hat{a}^\dagger\hat{a} + \frac{\hbar\tilde{\omega}_q}{2}\hat{\sigma}_z + \hbar\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z$$

where the **dispersive shift** is:

$$\boxed{\chi = \frac{g^2}{\Delta}}$$

This qubit-state-dependent frequency shift enables:
- **Qubit readout**: Measure resonator frequency to infer qubit state
- **Qubit-qubit coupling**: Two qubits coupled to same resonator interact

## Quantum Computing Applications

### Cavity QED to Circuit QED

Circuit QED offers significant advantages over atomic cavity QED:

| Parameter | Atomic Cavity QED | Circuit QED |
|-----------|-------------------|-------------|
| Coupling $g/2\pi$ | ~100 kHz | ~100 MHz |
| Mode volume | $\sim\lambda^3$ | $\sim 10^{-6}\lambda^3$ |
| $g/\kappa$ | ~30 | ~1000 |
| Controllability | Limited | Highly tunable |

### Readout Via Dispersive Shift

The resonator frequency depends on qubit state:

$$\omega_r^{|g\rangle} = \omega_r - \chi, \quad \omega_r^{|e\rangle} = \omega_r + \chi$$

By probing the resonator at $\omega_r$, we can distinguish qubit states from the phase/amplitude of transmitted/reflected signal.

### Building Multi-Qubit Systems

Circuit QED provides a natural architecture for scaling:
- Multiple qubits coupled to shared bus resonator
- Individual readout resonators per qubit
- Frequency crowding managed by design

## Worked Examples

### Example 1: LC Circuit Quantization

**Problem**: An LC circuit has $L = 10$ nH and $C = 1$ pF. Calculate:
(a) Resonant frequency
(b) Characteristic impedance
(c) Zero-point fluctuations in flux and charge

**Solution**:

(a) Resonant frequency:
$$\omega_r = \frac{1}{\sqrt{LC}} = \frac{1}{\sqrt{10 \times 10^{-9} \times 10^{-12}}} = \frac{1}{\sqrt{10^{-20}}}$$
$$\omega_r = 10^{10} \text{ rad/s} \implies f_r = \frac{\omega_r}{2\pi} \approx 1.59 \text{ GHz}$$

(b) Characteristic impedance:
$$Z_r = \sqrt{\frac{L}{C}} = \sqrt{\frac{10^{-8}}{10^{-12}}} = \sqrt{10^4} = 100 \text{ }\Omega$$

(c) Zero-point fluctuations:
$$\Phi_{zp} = \sqrt{\frac{\hbar Z_r}{2}} = \sqrt{\frac{1.055 \times 10^{-34} \times 100}{2}} = \sqrt{5.27 \times 10^{-33}}$$
$$\Phi_{zp} \approx 2.3 \times 10^{-17} \text{ Wb} \approx 0.011 \Phi_0$$

$$Q_{zp} = \sqrt{\frac{\hbar}{2Z_r}} = \sqrt{\frac{1.055 \times 10^{-34}}{200}} \approx 2.3 \times 10^{-19} \text{ C}$$

### Example 2: Vacuum Rabi Splitting

**Problem**: A transmon qubit with frequency $\omega_q/2\pi = 5.5$ GHz is coupled to a resonator at $\omega_r/2\pi = 5.5$ GHz with coupling $g/2\pi = 100$ MHz. The resonator quality factor is $Q = 50,000$ and qubit $T_1 = 50$ $\mu$s.

(a) Is the system in strong coupling?
(b) What is the vacuum Rabi splitting?
(c) How many Rabi oscillations occur before decoherence?

**Solution**:

(a) Calculate decay rates:
$$\kappa = \frac{\omega_r}{Q} = \frac{2\pi \times 5.5 \times 10^9}{50000} = 2\pi \times 110 \text{ kHz}$$
$$\gamma = \frac{1}{T_1} = \frac{1}{50 \times 10^{-6}} = 2\pi \times 3.2 \text{ kHz}$$

Strong coupling condition: $g \gg \kappa, \gamma$
$$g/2\pi = 100 \text{ MHz} \gg \kappa/2\pi = 110 \text{ kHz}$$

**Yes, strongly coupled** with $g/\kappa \approx 900$.

(b) Vacuum Rabi splitting at resonance:
$$\Delta E = 2\hbar g = 2\hbar \times 2\pi \times 100 \text{ MHz}$$
$$\Delta f = 2g/2\pi = 200 \text{ MHz}$$

(c) Rabi oscillation period: $T_{Rabi} = 1/(2g/2\pi) = 5$ ns

Number of oscillations before decay:
$$N \approx \frac{1}{\kappa T_{Rabi}} = \frac{1}{2\pi \times 110 \times 10^3 \times 5 \times 10^{-9}} \approx 290$$

### Example 3: Dispersive Shift

**Problem**: A qubit at 5 GHz couples to a resonator at 7 GHz with $g/2\pi = 80$ MHz. Calculate the dispersive shift and resulting frequency separation for readout.

**Solution**:

Detuning:
$$\Delta = \omega_q - \omega_r = 2\pi(5 - 7) \text{ GHz} = -2\pi \times 2 \text{ GHz}$$

Dispersive shift:
$$\chi = \frac{g^2}{\Delta} = \frac{(2\pi \times 80 \times 10^6)^2}{-2\pi \times 2 \times 10^9}$$
$$\chi = \frac{(2\pi)^2 \times 6.4 \times 10^{15}}{-2\pi \times 2 \times 10^9} = -2\pi \times 3.2 \text{ MHz}$$

The resonator frequencies for qubit in $|g\rangle$ vs $|e\rangle$ differ by:
$$2|\chi|/2\pi = 6.4 \text{ MHz}$$

This separation is easily resolved with typical resonator linewidths of ~1 MHz.

## Practice Problems

### Level 1: Direct Application

1. A CPW resonator has fundamental frequency 6 GHz and internal quality factor $Q_i = 10^6$. Calculate the photon lifetime and the decay rate $\kappa$.

2. A Josephson junction has critical current $I_c = 30$ nA. Calculate the Josephson energy $E_J$ in GHz (i.e., $E_J/h$).

3. For a qubit-resonator system with $\omega_q/2\pi = 4.5$ GHz, $\omega_r/2\pi = 6.5$ GHz, and $g/2\pi = 50$ MHz, determine if the dispersive approximation is valid and calculate $\chi$.

### Level 2: Intermediate

4. Derive the transformation from flux-charge operators $(\hat{\Phi}, \hat{Q})$ to ladder operators $(\hat{a}, \hat{a}^\dagger)$ and verify the commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$.

5. A transmission line resonator of length 10 mm is made from niobium on silicon ($\epsilon_r = 11.7$). Calculate:
   (a) The phase velocity
   (b) The fundamental frequency
   (c) The first three mode frequencies

6. In the Jaynes-Cummings model at exact resonance, starting from state $|e, 0\rangle$, derive the time evolution and plot the probability of finding the system in $|g, 1\rangle$.

### Level 3: Challenging

7. **Beyond rotating-wave approximation**: The full qubit-resonator interaction includes counter-rotating terms: $\hat{H}_{int} = \hbar g(\hat{a} + \hat{a}^\dagger)(\hat{\sigma}^+ + \hat{\sigma}^-)$. Use perturbation theory to calculate the Bloch-Siegert shift—the correction to the qubit frequency from counter-rotating terms.

8. **Multi-mode effects**: A transmon at frequency 5 GHz couples to three resonator modes at 6, 7, and 8 GHz with coupling strengths 100, 70, and 50 MHz respectively. Calculate the total dispersive shift including Lamb shift contributions from all modes.

9. **Design problem**: Design a CPW resonator-transmon system achieving $\chi/2\pi = 2$ MHz with qubit frequency 5.5 GHz. Specify the required resonator frequency, coupling capacitance, and verify the dispersive approximation validity.

## Computational Lab: Circuit Quantization Simulation

```python
"""
Day 897 Computational Lab: Circuit QED Fundamentals
Simulating LC circuit quantization and Jaynes-Cummings dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Physical constants
hbar = 1.055e-34  # J·s
e_charge = 1.602e-19  # C
Phi_0 = 2.068e-15  # Wb, flux quantum

# =============================================================================
# Part 1: LC Circuit Quantization
# =============================================================================

def lc_circuit_analysis(L, C, n_levels=10):
    """
    Analyze a quantum LC circuit.

    Parameters:
    -----------
    L : float
        Inductance in Henries
    C : float
        Capacitance in Farads
    n_levels : int
        Number of energy levels to compute

    Returns:
    --------
    dict with circuit parameters and energy levels
    """
    # Classical parameters
    omega_r = 1 / np.sqrt(L * C)
    f_r = omega_r / (2 * np.pi)
    Z_r = np.sqrt(L / C)

    # Zero-point fluctuations
    Phi_zpf = np.sqrt(hbar * Z_r / 2)
    Q_zpf = np.sqrt(hbar / (2 * Z_r))

    # Energy levels (harmonic oscillator)
    n = np.arange(n_levels)
    E_n = hbar * omega_r * (n + 0.5)

    return {
        'omega_r': omega_r,
        'f_r': f_r,
        'Z_r': Z_r,
        'Phi_zpf': Phi_zpf,
        'Q_zpf': Q_zpf,
        'energies': E_n,
        'frequencies': E_n / hbar / (2 * np.pi)
    }

# Example: Typical CPW resonator parameters
L = 10e-9  # 10 nH
C = 0.1e-12  # 0.1 pF

result = lc_circuit_analysis(L, C)
print("=" * 60)
print("LC Circuit Quantization Results")
print("=" * 60)
print(f"Resonant frequency: {result['f_r']/1e9:.3f} GHz")
print(f"Characteristic impedance: {result['Z_r']:.1f} Ohm")
print(f"Flux ZPF: {result['Phi_zpf']/Phi_0:.4f} Phi_0")
print(f"Charge ZPF: {result['Q_zpf']/e_charge:.4f} e")
print(f"\nEnergy levels (GHz):")
for i, f in enumerate(result['frequencies'][:5]):
    print(f"  n={i}: {f/1e9:.3f} GHz")

# =============================================================================
# Part 2: Jaynes-Cummings Model Simulation
# =============================================================================

def create_operators(n_resonator, n_qubit=2):
    """
    Create operators for Jaynes-Cummings Hamiltonian.

    Parameters:
    -----------
    n_resonator : int
        Number of resonator Fock states
    n_qubit : int
        Number of qubit levels (2 for ideal qubit)

    Returns:
    --------
    dict with Hamiltonian components
    """
    # Identity matrices
    I_r = np.eye(n_resonator)
    I_q = np.eye(n_qubit)

    # Resonator operators
    a = np.diag(np.sqrt(np.arange(1, n_resonator)), k=1)  # annihilation
    a_dag = a.T  # creation
    n_op = np.diag(np.arange(n_resonator))  # number

    # Qubit operators (|g>=|0>, |e>=|1>)
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_plus = np.array([[0, 1], [0, 0]])
    sigma_minus = np.array([[0, 0], [1, 0]])

    # Tensor products (qubit x resonator)
    A = np.kron(I_q, a)  # a ⊗ I_q
    A_dag = np.kron(I_q, a_dag)
    N_r = np.kron(I_q, n_op)

    Sz = np.kron(sigma_z, I_r)
    Sp = np.kron(sigma_plus, I_r)
    Sm = np.kron(sigma_minus, I_r)

    return {
        'a': A, 'a_dag': A_dag, 'n_r': N_r,
        'sigma_z': Sz, 'sigma_plus': Sp, 'sigma_minus': Sm,
        'dim': n_resonator * n_qubit
    }

def jaynes_cummings_hamiltonian(omega_r, omega_q, g, ops):
    """
    Construct Jaynes-Cummings Hamiltonian.

    H = hbar * omega_r * a†a + (hbar * omega_q / 2) * sigma_z
        + hbar * g * (a† sigma_- + a sigma_+)

    Returns H in units where hbar = 1 (energies in rad/s)
    """
    H = (omega_r * ops['a_dag'] @ ops['a'] +
         (omega_q / 2) * ops['sigma_z'] +
         g * (ops['a_dag'] @ ops['sigma_minus'] +
              ops['a'] @ ops['sigma_plus']))
    return H

def simulate_dynamics(H, psi0, times):
    """
    Simulate unitary time evolution.

    Parameters:
    -----------
    H : ndarray
        Hamiltonian matrix
    psi0 : ndarray
        Initial state vector
    times : ndarray
        Time points

    Returns:
    --------
    states : list of state vectors at each time
    """
    states = []
    for t in times:
        U = expm(-1j * H * t)
        psi_t = U @ psi0
        states.append(psi_t)
    return states

# Set up Jaynes-Cummings system
n_fock = 10  # Number of Fock states
ops = create_operators(n_fock)

# Frequencies (in GHz, working in units of 2π GHz)
omega_r = 2 * np.pi * 6.0  # 6 GHz resonator
omega_q = 2 * np.pi * 6.0  # 6 GHz qubit (resonant)
g = 2 * np.pi * 0.1  # 100 MHz coupling

H_jc = jaynes_cummings_hamiltonian(omega_r, omega_q, g, ops)

# Find eigenvalues and eigenstates
eigenvalues, eigenstates = np.linalg.eigh(H_jc)

print("\n" + "=" * 60)
print("Jaynes-Cummings Spectrum (Resonant Case)")
print("=" * 60)
print(f"Qubit frequency: {omega_q/(2*np.pi):.2f} GHz")
print(f"Resonator frequency: {omega_r/(2*np.pi):.2f} GHz")
print(f"Coupling strength: {g/(2*np.pi)*1000:.1f} MHz")
print(f"\nLowest dressed state energies (GHz):")
for i in range(6):
    print(f"  |{i}>: {eigenvalues[i]/(2*np.pi):.4f} GHz")

# Calculate vacuum Rabi splitting
E_01 = eigenvalues[1] - eigenvalues[0]
E_02 = eigenvalues[2] - eigenvalues[0]
vacuum_rabi = E_02 - E_01
print(f"\nVacuum Rabi splitting: {vacuum_rabi/(2*np.pi)*1000:.2f} MHz")
print(f"Expected (2g): {2*g/(2*np.pi)*1000:.2f} MHz")

# =============================================================================
# Part 3: Vacuum Rabi Oscillations
# =============================================================================

# Initial state: |e, 0> (excited qubit, vacuum resonator)
# In our basis: |g,0>, |g,1>, ..., |e,0>, |e,1>, ...
# |e,0> is at index n_fock (first excited qubit state with 0 photons)
psi0 = np.zeros(2 * n_fock, dtype=complex)
psi0[n_fock] = 1.0  # |e, 0>

# Time evolution
t_max = 50e-9  # 50 ns
n_points = 500
times = np.linspace(0, t_max, n_points)

states = simulate_dynamics(H_jc, psi0, times)

# Calculate populations
P_e0 = np.array([np.abs(psi[n_fock])**2 for psi in states])  # |e, 0>
P_g1 = np.array([np.abs(psi[1])**2 for psi in states])  # |g, 1>
P_e = np.array([np.sum(np.abs(psi[n_fock:])**2) for psi in states])  # qubit excited

# Plot vacuum Rabi oscillations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Rabi oscillations
ax1 = axes[0, 0]
ax1.plot(times * 1e9, P_e0, 'b-', label=r'$|e, 0\rangle$', linewidth=2)
ax1.plot(times * 1e9, P_g1, 'r-', label=r'$|g, 1\rangle$', linewidth=2)
ax1.set_xlabel('Time (ns)', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Vacuum Rabi Oscillations', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, t_max * 1e9])

# Plot 2: Energy spectrum
ax2 = axes[0, 1]
n_levels_plot = 8
energies_plot = (eigenvalues[:n_levels_plot] - eigenvalues[0]) / (2 * np.pi)
ax2.barh(range(n_levels_plot), energies_plot, color='steelblue', alpha=0.7)
ax2.set_ylabel('Dressed State Index', fontsize=12)
ax2.set_xlabel('Energy (GHz)', fontsize=12)
ax2.set_title('Jaynes-Cummings Energy Spectrum', fontsize=14)
ax2.grid(True, alpha=0.3, axis='x')

# =============================================================================
# Part 4: Dispersive Regime
# =============================================================================

# Detuned case
omega_q_det = 2 * np.pi * 5.0  # 5 GHz qubit (detuned by 1 GHz)
Delta = omega_q_det - omega_r

H_jc_det = jaynes_cummings_hamiltonian(omega_r, omega_q_det, g, ops)
eigenvalues_det, eigenstates_det = np.linalg.eigh(H_jc_det)

# Calculate dispersive shift
chi_theory = g**2 / Delta
print("\n" + "=" * 60)
print("Dispersive Regime Analysis")
print("=" * 60)
print(f"Detuning Delta: {Delta/(2*np.pi):.2f} GHz")
print(f"g/Delta ratio: {g/abs(Delta):.3f}")
print(f"Dispersive shift (theory): {chi_theory/(2*np.pi)*1000:.2f} MHz")

# Plot 3: Dispersive spectrum
ax3 = axes[1, 0]

# Calculate resonator frequency for each qubit state
# Ground state: look at transitions from |g,0> to |g,1>, |g,2>, etc.
omega_r_g = eigenvalues_det[1] - eigenvalues_det[0]  # |g,0> to |g,1>
omega_r_e = eigenvalues_det[n_fock+1] - eigenvalues_det[n_fock]  # |e,0> to |e,1>

chi_numerical = (omega_r_e - omega_r_g) / 2

print(f"Dispersive shift (numerical): {chi_numerical/(2*np.pi)*1000:.2f} MHz")

# Transmission spectrum simulation
omega_probe = np.linspace(omega_r - 4*g, omega_r + 4*g, 500)
kappa = 2 * np.pi * 0.002  # 2 MHz linewidth

# Lorentzian response for each qubit state
S21_g = kappa**2 / ((omega_probe - omega_r + chi_numerical)**2 + kappa**2)
S21_e = kappa**2 / ((omega_probe - omega_r - chi_numerical)**2 + kappa**2)

ax3.plot((omega_probe - omega_r)/(2*np.pi)*1000, S21_g, 'b-',
         label=r'Qubit in $|g\rangle$', linewidth=2)
ax3.plot((omega_probe - omega_r)/(2*np.pi)*1000, S21_e, 'r-',
         label=r'Qubit in $|e\rangle$', linewidth=2)
ax3.axvline(chi_numerical/(2*np.pi)*1000, color='r', linestyle='--', alpha=0.5)
ax3.axvline(-chi_numerical/(2*np.pi)*1000, color='b', linestyle='--', alpha=0.5)
ax3.set_xlabel('Detuning from bare resonator (MHz)', fontsize=12)
ax3.set_ylabel('Transmission', fontsize=12)
ax3.set_title('Dispersive Readout: Qubit-State-Dependent Shift', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# =============================================================================
# Part 5: Coupling Strength vs Separation
# =============================================================================

ax4 = axes[1, 1]

# Vary coupling strength
g_values = np.linspace(0.01, 0.3, 50) * 2 * np.pi  # 10-300 MHz
chi_values = []
chi_theory_values = []

for g_val in g_values:
    H_temp = jaynes_cummings_hamiltonian(omega_r, omega_q_det, g_val, ops)
    evals = np.linalg.eigvalsh(H_temp)
    omega_r_g_temp = evals[1] - evals[0]
    omega_r_e_temp = evals[n_fock+1] - evals[n_fock]
    chi_values.append((omega_r_e_temp - omega_r_g_temp) / 2)
    chi_theory_values.append(g_val**2 / Delta)

ax4.plot(g_values/(2*np.pi)*1000, np.array(chi_values)/(2*np.pi)*1000,
         'b-', linewidth=2, label='Numerical')
ax4.plot(g_values/(2*np.pi)*1000, np.array(chi_theory_values)/(2*np.pi)*1000,
         'r--', linewidth=2, label=r'Theory: $\chi = g^2/\Delta$')
ax4.set_xlabel('Coupling g (MHz)', fontsize=12)
ax4.set_ylabel('Dispersive shift (MHz)', fontsize=12)
ax4.set_title('Dispersive Shift vs Coupling Strength', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('circuit_qed_fundamentals.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Strong Coupling Criterion
# =============================================================================

print("\n" + "=" * 60)
print("Strong Coupling Analysis")
print("=" * 60)

# Typical parameters for state-of-the-art superconducting circuits
params = {
    'g': 2 * np.pi * 100e6,  # 100 MHz coupling
    'kappa': 2 * np.pi * 1e6,  # 1 MHz resonator linewidth (Q ~ 6000)
    'gamma': 2 * np.pi * 0.02e6,  # 20 kHz qubit decay (T1 ~ 50 us)
}

cooperativity = params['g']**2 / (params['kappa'] * params['gamma'])
n_rabi = params['g'] / max(params['kappa'], params['gamma'])

print(f"Coupling g: {params['g']/(2*np.pi)/1e6:.1f} MHz")
print(f"Resonator decay kappa: {params['kappa']/(2*np.pi)/1e6:.2f} MHz")
print(f"Qubit decay gamma: {params['gamma']/(2*np.pi)/1e3:.1f} kHz")
print(f"\ng/kappa = {params['g']/params['kappa']:.1f}")
print(f"g/gamma = {params['g']/params['gamma']:.0f}")
print(f"Cooperativity C = g²/(kappa*gamma) = {cooperativity:.0f}")
print(f"Number of Rabi oscillations before decay: ~{n_rabi:.0f}")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| LC frequency | $\omega_r = 1/\sqrt{LC}$ |
| Impedance | $Z_r = \sqrt{L/C}$ |
| Flux ZPF | $\Phi_{zp} = \sqrt{\hbar Z_r/2}$ |
| Josephson energy | $E_J = \Phi_0 I_c / 2\pi$ |
| Charging energy | $E_C = e^2/2C$ |
| Jaynes-Cummings | $\hat{H} = \hbar\omega_r\hat{a}^\dagger\hat{a} + \frac{\hbar\omega_q}{2}\hat{\sigma}_z + \hbar g(\hat{a}^\dagger\hat{\sigma}^- + \hat{a}\hat{\sigma}^+)$ |
| Vacuum Rabi splitting | $2g\sqrt{n+1}$ |
| Dispersive shift | $\chi = g^2/\Delta$ |

### Main Takeaways

1. **Circuit quantization** treats flux $\Phi$ and charge $Q$ as conjugate quantum variables with $[\hat{\Phi}, \hat{Q}] = i\hbar$

2. **Josephson junctions** provide the essential nonlinearity through $U = -E_J\cos\varphi$, enabling non-degenerate energy levels

3. **Jaynes-Cummings model** describes qubit-resonator interaction in the rotating-wave approximation, conserving total excitation number

4. **Strong coupling** ($g \gg \kappa, \gamma$) enables coherent quantum dynamics; circuit QED typically achieves $g/\kappa > 100$

5. **Dispersive regime** ($|\Delta| \gg g$) creates qubit-state-dependent resonator frequency, enabling quantum non-demolition readout

## Daily Checklist

- [ ] I can derive the quantum Hamiltonian of an LC circuit
- [ ] I understand why Josephson junctions are essential for superconducting qubits
- [ ] I can write the Jaynes-Cummings Hamiltonian and explain each term
- [ ] I can calculate vacuum Rabi splitting and verify strong coupling
- [ ] I understand the dispersive approximation and its applications
- [ ] I have run the computational lab and can interpret the results
- [ ] I can solve problems involving circuit quantization

## Preview: Day 898

Tomorrow we explore the **transmon qubit**—the workhorse of superconducting quantum computing. We'll see how increasing $E_J/E_C$ exponentially suppresses charge noise sensitivity while maintaining sufficient anharmonicity for quantum control. Key topics include:

- Derivation of transmon energy levels
- The sweet spot at $E_J/E_C \approx 50$
- Anharmonicity and leakage to higher states
- Capacitive and inductive coupling schemes

---

*"The Jaynes-Cummings model is to circuit QED what the hydrogen atom is to atomic physics—the essential starting point for everything else."*
