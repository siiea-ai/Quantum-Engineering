# Day 901: Two-Qubit Gates

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Cross-resonance gates, CZ gates, coupling mechanisms |
| Afternoon | 2 hours | iSWAP, parametric gates, ZZ interaction, problem solving |
| Evening | 2 hours | Computational lab: Two-qubit gate simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the cross-resonance (CR) gate mechanism for fixed-frequency qubits
2. **Derive** the effective Hamiltonian for capacitively coupled transmons
3. **Describe** CZ gates implemented via flux tuning
4. **Analyze** iSWAP and parametric gate implementations
5. **Calculate** ZZ coupling strength and its impact on gate fidelity
6. **Compare** different two-qubit gate approaches and their tradeoffs

## Core Content

### 1. Capacitive Coupling Between Transmons

Two transmons coupled by a capacitance $C_c$ have interaction Hamiltonian:

$$\hat{H}_{int} = \frac{C_c}{C_1 C_2}\hat{Q}_1\hat{Q}_2 = 4E_c^{(c)}\hat{n}_1\hat{n}_2$$

where $E_c^{(c)} = e^2 C_c/(C_1 C_2)$.

In the charge-insensitive transmon regime, this becomes:

$$\hat{H}_{int} \approx g(\hat{a}_1^\dagger - \hat{a}_1)(\hat{a}_2^\dagger - \hat{a}_2)$$

Keeping only energy-conserving terms (rotating wave approximation):

$$\boxed{\hat{H}_{int} \approx g(\hat{a}_1^\dagger\hat{a}_2 + \hat{a}_1\hat{a}_2^\dagger)}$$

This is a "flip-flop" or exchange interaction—one qubit de-excites while the other excites.

### 2. Effective Qubit-Qubit Coupling

The full two-qubit Hamiltonian:

$$\hat{H} = \omega_1\hat{a}_1^\dagger\hat{a}_1 + \frac{\alpha_1}{2}\hat{a}_1^\dagger\hat{a}_1^\dagger\hat{a}_1\hat{a}_1 + (1 \leftrightarrow 2) + g(\hat{a}_1^\dagger\hat{a}_2 + h.c.)$$

In the dispersive limit ($|\Delta_{12}| = |\omega_1 - \omega_2| \gg g$):

$$\hat{H}_{eff} \approx \tilde{\omega}_1\hat{\sigma}_z^{(1)}/2 + \tilde{\omega}_2\hat{\sigma}_z^{(2)}/2 + J(\hat{\sigma}_+^{(1)}\hat{\sigma}_-^{(2)} + h.c.) + \zeta\hat{\sigma}_z^{(1)}\hat{\sigma}_z^{(2)}$$

where:
- $J = g^2\left(\frac{1}{\Delta_{12}} - \frac{1}{\Delta_{12} + \alpha_1} - \frac{1}{\Delta_{12} + \alpha_2}\right)$ is exchange coupling
- $\zeta$ is the always-on ZZ coupling (see Section 6)

### 3. Cross-Resonance Gate

The **cross-resonance (CR) gate** is the primary two-qubit gate for fixed-frequency transmons (IBM approach).

**Mechanism**: Drive qubit 1 (control) at the frequency of qubit 2 (target)

$$\hat{H}_{drive} = \Omega\cos(\omega_2 t)\hat{\sigma}_x^{(1)}$$

In the rotating frame, this creates effective Hamiltonian on qubit 2 that depends on qubit 1's state:

$$\boxed{\hat{H}_{CR} \approx \frac{\Omega_{ZX}}{2}\hat{\sigma}_z^{(1)}\hat{\sigma}_x^{(2)} + \frac{\Omega_{IX}}{2}\hat{\sigma}_x^{(2)} + \frac{\Omega_{ZI}}{2}\hat{\sigma}_z^{(1)}}$$

The desired $ZX$ term: When control is $|0\rangle$, target sees no drive; when control is $|1\rangle$, target is rotated.

**CR gate strength**:
$$\Omega_{ZX} \approx \frac{g\Omega}{\Delta_{12}}\cdot\frac{\alpha_2}{\Delta_{12} + \alpha_2}$$

**Echoed CR (ECR)**: Apply CR pulse, then X gate on control, then -CR pulse. This cancels unwanted $IX$ and $ZI$ terms.

### 4. Controlled-Z (CZ) Gate

For tunable transmons, the **CZ gate** uses flux pulses to bring the $|11\rangle$ state into resonance with $|20\rangle$ (or $|02\rangle$).

At the avoided crossing, the states hybridize:
$$|+\rangle = \frac{1}{\sqrt{2}}(|11\rangle + |20\rangle)$$
$$|-\rangle = \frac{1}{\sqrt{2}}(|11\rangle - |20\rangle)$$

Evolution for time $t = \pi/\sqrt{2}g_{12}$ accumulates phase:

$$|11\rangle \to e^{i\phi_{11}}|11\rangle$$

where $\phi_{11}$ differs from $\phi_{00} + \phi_{01} + \phi_{10}$ by $\pi$, implementing:

$$\boxed{CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}}$$

**Advantages**: Fast (20-40 ns), high fidelity (>99.5%)
**Disadvantages**: Requires flux-tunable qubits, flux noise sensitivity

### 5. iSWAP Gate

The iSWAP gate arises from direct exchange coupling at resonance:

When $\omega_1 = \omega_2$, the exchange Hamiltonian $g(\hat{\sigma}_+^{(1)}\hat{\sigma}_-^{(2)} + h.c.)$ generates:

$$\hat{U}_{iSWAP} = \exp(-i\frac{g t}{\hbar}(\hat{\sigma}_+^{(1)}\hat{\sigma}_-^{(2)} + h.c.))$$

At $gt/\hbar = \pi/2$:

$$\boxed{iSWAP = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}}$$

**Implementation**: Bring qubits into resonance via flux tuning, wait, detune.

**$\sqrt{iSWAP}$**: At $gt = \pi/4$, get the square root—a universal two-qubit gate.

### 6. Always-On ZZ Coupling

Even when qubits are detuned, there's a residual **ZZ interaction**:

$$\hat{H}_{ZZ} = \zeta\hat{\sigma}_z^{(1)}\hat{\sigma}_z^{(2)}/4$$

The ZZ coupling strength:

$$\boxed{\zeta = \frac{2g^2\alpha_1\alpha_2}{(\Delta_{12})(\Delta_{12} + \alpha_1)(\Delta_{12} + \alpha_2)}}$$

**Problems with ZZ coupling**:
- Causes conditional phase errors during idling
- Creates crosstalk during single-qubit gates
- Frequency-dependent: problematic for frequency crowding

**Mitigation strategies**:
- Careful frequency allocation
- Echo sequences to cancel ZZ
- Tunable couplers (see below)

### 7. Tunable Couplers

A **tunable coupler** is an additional element (often a transmon) between qubits that can modulate the effective coupling.

**Physical mechanism**:
- Direct capacitive coupling: $g_{direct}$
- Coupler-mediated coupling: $g_{mediated} = g_{1c}g_{2c}/\Delta$
- Total: $g_{eff} = g_{direct} + g_{mediated}$

By tuning the coupler frequency, $g_{eff}$ can be made:
- Large positive: strong coupling for gates
- Zero: qubits effectively decoupled
- Negative: possible for some designs

**Advantages**:
- Suppress ZZ coupling when idle
- Fast two-qubit gates when activated
- Reduced crosstalk

**State of the art**: Google's Sycamore, IBM's tunable coupler designs.

### 8. Parametric Gates

**Parametric modulation** uses AC flux drives to activate coupling:

Apply oscillating flux at frequency $\omega_m = |\omega_1 - \omega_2|$:

$$\Phi(t) = \Phi_{DC} + \delta\Phi\cos(\omega_m t)$$

This modulates the qubit frequency, creating sidebands that bridge the frequency gap:

$$\omega_q(t) \approx \omega_q^{(0)} + \epsilon\cos(\omega_m t)$$

**Parametric iSWAP**: Modulate at $\omega_m = \Delta_{12}$ to drive resonant exchange.

**Parametric CZ**: Modulate to bring $|11\rangle \leftrightarrow |20\rangle$ into resonance.

**Advantages**:
- Works with fixed-frequency qubits plus modulated coupler
- Selective activation (frequency-specific)
- Reduced static ZZ

### 9. Gate Fidelity Considerations

**Error sources for two-qubit gates**:

1. **Decoherence**: Gate time $T_{gate}$ vs $T_1$, $T_2$
   $$\epsilon_{decoh} \approx T_{gate}/T_1 + T_{gate}/T_2$$

2. **Leakage**: Population in $|20\rangle$, $|02\rangle$, $|21\rangle$, etc.

3. **State-dependent errors**: ZZ during gate causes different phases

4. **Crosstalk**: Drives affect non-target qubits

5. **Calibration errors**: Pulse amplitude, frequency, timing

**Typical fidelities (2025)**:
- CR gate: 99-99.5%
- CZ gate: 99.5-99.7%
- iSWAP: 99-99.5%

### 10. Gate Decomposition

Any two-qubit gate can be decomposed into single-qubit gates plus CNOT (or equivalent).

**CNOT from CR**:
$$CNOT = (I \otimes R_y(-\pi/2)) \cdot ZX(\pi/2) \cdot (I \otimes R_y(\pi/2))$$

**CNOT from CZ**:
$$CNOT = (I \otimes H) \cdot CZ \cdot (I \otimes H)$$

**CNOT from iSWAP**:
$$CNOT = (R_z \otimes I) \cdot iSWAP \cdot (R_z \otimes I) \cdot \sqrt{iSWAP} \cdot (single-qubit gates)$$

## Quantum Computing Applications

### Scaling Challenges

As qubit count increases:
- Frequency crowding: All pairs must have sufficient $\Delta_{12}$
- Crosstalk: More unwanted interactions
- Control complexity: More drive lines, more calibration

### Error Correction Requirements

Surface code threshold (~1% error) requires:
- Two-qubit gate fidelity > 99%
- Fast gates (reduce idle errors)
- Low crosstalk

### Current Architectures

| Company | Two-Qubit Gate | Typical Fidelity |
|---------|----------------|------------------|
| IBM | Cross-resonance | 99-99.5% |
| Google | CZ (tunable) | 99.5-99.7% |
| Rigetti | CZ/iSWAP | 99-99.5% |
| IQM | CZ (tunable coupler) | 99.5% |

## Worked Examples

### Example 1: Cross-Resonance Gate Strength

**Problem**: Two fixed-frequency transmons have $\omega_1/2\pi = 5.0$ GHz, $\omega_2/2\pi = 5.2$ GHz, coupling $g/2\pi = 3$ MHz, and anharmonicity $\alpha_2/2\pi = -250$ MHz. Calculate the ZX interaction strength for CR drive amplitude $\Omega/2\pi = 30$ MHz.

**Solution**:

Detuning:
$$\Delta_{12} = \omega_1 - \omega_2 = -2\pi \times 0.2 \text{ GHz} = -2\pi \times 200 \text{ MHz}$$

ZX strength:
$$\Omega_{ZX} = \frac{g\Omega}{\Delta_{12}} \cdot \frac{\alpha_2}{\Delta_{12} + \alpha_2}$$

$$= \frac{3 \times 30}{200} \times \frac{-250}{-200 + (-250)}$$

$$= \frac{90}{200} \times \frac{-250}{-450}$$

$$= 0.45 \times 0.556 = 0.25 \text{ MHz}$$

So $\Omega_{ZX}/2\pi = 250$ kHz.

Gate time for $\pi/4$ ZX rotation (equivalent to CNOT with single-qubit gates):
$$t_{gate} = \frac{\pi/2}{\Omega_{ZX}} = \frac{\pi/2}{2\pi \times 0.25 \times 10^6} = 1 \text{ }\mu\text{s}$$

This is rather slow; higher drive power or stronger coupling would be preferred.

### Example 2: CZ Gate Timing

**Problem**: Two transmons can be brought into the $|11\rangle \leftrightarrow |20\rangle$ resonance with effective coupling $g_{11-20}/2\pi = 25$ MHz. Calculate:
(a) The CZ gate time
(b) The required flux pulse precision

**Solution**:

(a) At the $|11\rangle - |20\rangle$ avoided crossing, the coupling creates oscillation at frequency $2g_{11-20}$.

For CZ, we need the $|11\rangle$ state to accumulate $\pi$ phase relative to other states. This occurs at:

$$t_{CZ} = \frac{\pi}{2g_{11-20}} = \frac{\pi}{2 \times 2\pi \times 25 \times 10^6} = \frac{1}{100 \times 10^6} = 10 \text{ ns}$$

Actually, for full population transfer and back: $t = 2\pi/2g = 20$ ns.
For $\pi$ phase accumulation: $t \approx \pi/2g = 10$ ns (but this is more subtle).

Standard CZ: gate time $\approx 20-40$ ns.

(b) Flux precision: The qubit frequency at sweet spot has second-derivative flux sensitivity. Near the operating point, typical requirement is:
$$\delta\Phi/\Phi_0 < 10^{-4}$$
to achieve 99.9% fidelity.

### Example 3: ZZ Coupling

**Problem**: Calculate the ZZ coupling for two transmons with $\omega_1/2\pi = 5.0$ GHz, $\omega_2/2\pi = 5.3$ GHz, $g/2\pi = 5$ MHz, and $\alpha_1 = \alpha_2 = -2\pi \times 250$ MHz.

**Solution**:

$$\zeta = \frac{2g^2\alpha_1\alpha_2}{\Delta_{12}(\Delta_{12} + \alpha_1)(\Delta_{12} + \alpha_2)}$$

With $\Delta_{12} = -2\pi \times 300$ MHz:

$$\zeta = \frac{2 \times (2\pi \times 5)^2 \times (-2\pi \times 250)^2}{(-2\pi \times 300)(-2\pi \times 550)(-2\pi \times 550)}$$

Let's compute in units of $2\pi$ MHz:
$$\zeta = \frac{2 \times 25 \times 62500}{300 \times 550 \times 550} \times 2\pi \text{ MHz}$$

$$= \frac{3.125 \times 10^6}{9.075 \times 10^7} \times 2\pi \text{ MHz} = 0.034 \times 2\pi \text{ MHz}$$

$$\zeta/2\pi \approx 34 \text{ kHz}$$

This means in 1 $\mu$s of idle time, the $|11\rangle$ state accumulates extra phase:
$$\phi_{ZZ} = \zeta \times 1 \text{ }\mu\text{s} = 2\pi \times 34 \times 10^3 \times 10^{-6} = 0.21 \text{ rad} \approx 12°$$

This is significant and must be calibrated out or echoed!

## Practice Problems

### Level 1: Direct Application

1. Calculate the exchange coupling $J$ for two transmons with $g/2\pi = 4$ MHz, $\Delta_{12}/2\pi = 400$ MHz, and $\alpha_1 = \alpha_2 = -2\pi \times 280$ MHz.

2. For an iSWAP gate with coupling $g/2\pi = 10$ MHz, how long should the qubits be held at resonance?

3. A CZ gate takes 30 ns. If $T_1 = 100$ $\mu$s and $T_2 = 80$ $\mu$s, estimate the decoherence-limited infidelity.

### Level 2: Intermediate

4. Derive the effective Hamiltonian for the cross-resonance drive by going to a doubly-rotating frame (at $\omega_1$ for qubit 1, $\omega_2$ for qubit 2) and applying perturbation theory.

5. For a tunable coupler design, the effective coupling is $g_{eff} = g_0(1 - \omega_c/\omega_{c0})$ where $\omega_c$ is the coupler frequency. At what coupler frequency is the coupling zero? How does this depend on the coupler anharmonicity?

6. Calculate the gate time for a parametric iSWAP with modulation amplitude $\epsilon/2\pi = 50$ MHz, given that the effective exchange rate is $J_{eff} \approx g \times J_1(2\epsilon/\omega_m)$ where $J_1$ is a Bessel function.

### Level 3: Challenging

7. **Leakage analysis**: For a CZ gate at the $|11\rangle - |20\rangle$ avoided crossing, calculate the leakage to $|20\rangle$ if the gate time is 5% too short. How does this compare to the phase error?

8. **Frequency collision**: In a 20-qubit processor, each qubit needs 300 MHz separation to avoid problematic ZZ coupling. Assuming qubit frequencies can be 4.5-5.5 GHz, design a frequency allocation strategy. What is the maximum sustainable qubit count?

9. **Crosstalk mitigation**: During a CR gate on qubits 1-2, qubit 3 (at 5.15 GHz, 50 MHz from qubit 2) sees the cross-resonance drive as an off-resonant pulse. Calculate the Stark shift and rotation angle on qubit 3, and propose a compensation strategy.

## Computational Lab: Two-Qubit Gate Simulation

```python
"""
Day 901 Computational Lab: Two-Qubit Gates
Simulating CR, CZ, and iSWAP gates for superconducting qubits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def tensor(A, B):
    """Tensor product of two matrices."""
    return np.kron(A, B)

# Two-qubit basis: |00>, |01>, |10>, |11>
II = tensor(I, I)
IX = tensor(I, X)
IY = tensor(I, Y)
IZ = tensor(I, Z)
XI = tensor(X, I)
YI = tensor(Y, I)
ZI = tensor(Z, I)
XX = tensor(X, X)
YY = tensor(Y, Y)
ZZ = tensor(Z, Z)
XY = tensor(X, Y)
YX = tensor(Y, X)
ZX = tensor(Z, X)
XZ = tensor(X, Z)

# =============================================================================
# Part 1: Cross-Resonance Gate
# =============================================================================

def cr_hamiltonian(omega_zx, omega_ix, omega_zi):
    """
    Cross-resonance effective Hamiltonian.

    H = (omega_ZX/2) * ZX + (omega_IX/2) * IX + (omega_ZI/2) * ZI
    """
    return (omega_zx/2) * ZX + (omega_ix/2) * IX + (omega_zi/2) * ZI

def evolve(H, t, psi0):
    """Evolve state under Hamiltonian for time t."""
    U = expm(-1j * H * t)
    return U @ psi0

print("=" * 60)
print("Cross-Resonance Gate Simulation")
print("=" * 60)

# CR parameters (in units where 2π = 1)
omega_zx = 0.5  # 500 kHz ZX
omega_ix = 0.3  # 300 kHz IX (unwanted)
omega_zi = 0.2  # 200 kHz ZI (unwanted)

H_cr = cr_hamiltonian(omega_zx, omega_ix, omega_zi)

# Gate time for ZX(π/2)
t_gate = np.pi / (2 * omega_zx)
print(f"ZX strength: {omega_zx} (arb units)")
print(f"Gate time for ZX(π/2): {t_gate:.2f} (arb units)")

# Initial states and evolution
states_init = {
    '|00>': np.array([1, 0, 0, 0], dtype=complex),
    '|01>': np.array([0, 1, 0, 0], dtype=complex),
    '|10>': np.array([0, 0, 1, 0], dtype=complex),
    '|11>': np.array([0, 0, 0, 1], dtype=complex),
}

t_points = np.linspace(0, 2 * t_gate, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (label, psi0) in enumerate(states_init.items()):
    ax = axes[idx // 2, idx % 2]
    probs = []
    for t in t_points:
        psi = evolve(H_cr, t, psi0)
        probs.append(np.abs(psi)**2)
    probs = np.array(probs)

    ax.plot(t_points / t_gate, probs[:, 0], 'b-', linewidth=2, label='|00⟩')
    ax.plot(t_points / t_gate, probs[:, 1], 'r-', linewidth=2, label='|01⟩')
    ax.plot(t_points / t_gate, probs[:, 2], 'g-', linewidth=2, label='|10⟩')
    ax.plot(t_points / t_gate, probs[:, 3], 'm-', linewidth=2, label='|11⟩')
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Time / $t_{gate}$', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(f'Initial state: {label}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Cross-Resonance Gate Dynamics', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('cr_gate_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 2: Echoed Cross-Resonance
# =============================================================================

print("\n" + "=" * 60)
print("Echoed Cross-Resonance (ECR)")
print("=" * 60)

def ecr_gate(omega_zx, omega_ix, omega_zi, t_half):
    """
    Echoed CR: CR(t/2) - X on control - CR(-t/2)
    This cancels IX and ZI terms.
    """
    H_cr = cr_hamiltonian(omega_zx, omega_ix, omega_zi)
    H_cr_neg = cr_hamiltonian(-omega_zx, -omega_ix, -omega_zi)

    U1 = expm(-1j * H_cr * t_half)
    U_X = tensor(X, I)  # X on control qubit
    U2 = expm(-1j * H_cr_neg * t_half)

    return U2 @ U_X @ U1

# Compare CR vs ECR
t_half = t_gate / 2

U_cr = expm(-1j * H_cr * t_gate)
U_ecr = ecr_gate(omega_zx, omega_ix, omega_zi, t_half)

# Ideal ZX(π/2)
U_zx_ideal = expm(-1j * (np.pi/4) * ZX)

def gate_fidelity(U_ideal, U_actual):
    """Average gate fidelity for two qubits."""
    d = 4
    trace = np.abs(np.trace(U_ideal.conj().T @ U_actual))**2
    return (trace / d + 1) / (d + 1)

# For ECR, need to account for the X gate
U_ecr_eff = U_ecr @ tensor(X, I)  # Remove the X we used for echo

print(f"CR gate fidelity (vs ideal ZX(π/2)): {gate_fidelity(U_zx_ideal, U_cr):.4f}")
print(f"ECR gate fidelity (approximate): {gate_fidelity(U_zx_ideal, U_ecr_eff):.4f}")

# =============================================================================
# Part 3: CZ Gate via Level Crossing
# =============================================================================

print("\n" + "=" * 60)
print("CZ Gate Simulation")
print("=" * 60)

def cz_hamiltonian_3level(omega1, omega2, alpha1, alpha2, g):
    """
    Two-qubit Hamiltonian including |2> states.
    Basis: |00>, |01>, |02>, |10>, |11>, |12>, |20>, |21>, |22>
    """
    dim = 9

    # Single qubit energies
    E = np.zeros(dim)
    E[0] = 0  # |00>
    E[1] = omega2  # |01>
    E[2] = 2*omega2 + alpha2  # |02>
    E[3] = omega1  # |10>
    E[4] = omega1 + omega2  # |11>
    E[5] = omega1 + 2*omega2 + alpha2  # |12>
    E[6] = 2*omega1 + alpha1  # |20>
    E[7] = 2*omega1 + alpha1 + omega2  # |21>
    E[8] = 2*omega1 + alpha1 + 2*omega2 + alpha2  # |22>

    H = np.diag(E)

    # Coupling terms (simplified: just |11> <-> |20> for CZ)
    # Full model would include all nearest-neighbor couplings
    g_11_20 = np.sqrt(2) * g  # Matrix element for |11> <-> |20>
    H[4, 6] = g_11_20
    H[6, 4] = g_11_20

    return H

# Parameters at the CZ operating point
omega1 = 5.0  # GHz
omega2 = 5.3  # GHz (but we'll tune omega1 to bring |11> ≈ |20>)
alpha1 = -0.25  # GHz
alpha2 = -0.25  # GHz
g = 0.02  # 20 MHz

# At CZ point: E(|11>) = E(|20>)
# omega1 + omega2 = 2*omega1 + alpha1
# omega2 = omega1 + alpha1
# omega1_cz = omega2 - alpha1

omega1_cz = omega2 - alpha1
print(f"CZ operating point: ω1 = {omega1_cz:.3f} GHz")
print(f"Detuning from |20>: {(omega1_cz + omega2) - (2*omega1_cz + alpha1):.4f} GHz")

H_cz = cz_hamiltonian_3level(omega1_cz, omega2, alpha1, alpha2, g)

# Gate time
g_eff = np.sqrt(2) * g
t_cz = np.pi / (2 * g_eff) * (1/1)  # In units where energies are in GHz, time in ns
print(f"Effective coupling at crossing: {g_eff*1000:.1f} MHz")
print(f"CZ gate time: {t_cz:.1f} ns")

# Evolution
t_points = np.linspace(0, 60, 300)

# Initial state |11> in 9-level space
psi0_11 = np.zeros(9, dtype=complex)
psi0_11[4] = 1.0  # |11>

populations = []
for t in t_points:
    U = expm(-1j * 2 * np.pi * H_cz * t)  # Factor of 2π to convert GHz to rad/ns
    psi = U @ psi0_11
    populations.append(np.abs(psi)**2)

populations = np.array(populations)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes2[0]
ax1.plot(t_points, populations[:, 4], 'b-', linewidth=2, label='|11⟩')
ax1.plot(t_points, populations[:, 6], 'r-', linewidth=2, label='|20⟩')
ax1.plot(t_points, populations[:, 4] + populations[:, 6], 'k--', linewidth=1, label='Sum')
ax1.set_xlabel('Time (ns)', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('CZ Gate: |11⟩ ↔ |20⟩ Oscillation', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Phase accumulation
phases = []
for t in t_points:
    U = expm(-1j * 2 * np.pi * H_cz * t)
    psi = U @ psi0_11
    # Phase of |11> component
    if np.abs(psi[4]) > 0.01:
        phases.append(np.angle(psi[4]))
    else:
        phases.append(np.nan)

ax2 = axes2[1]
ax2.plot(t_points, np.array(phases), 'g-', linewidth=2)
ax2.set_xlabel('Time (ns)', fontsize=12)
ax2.set_ylabel('Phase of |11⟩ (rad)', fontsize=12)
ax2.set_title('Phase Evolution During CZ Gate', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cz_gate_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: iSWAP Gate
# =============================================================================

print("\n" + "=" * 60)
print("iSWAP Gate Simulation")
print("=" * 60)

def iswap_hamiltonian(g):
    """
    Exchange Hamiltonian for iSWAP.
    H = g * (|01><10| + |10><01|)
    """
    H = np.zeros((4, 4), dtype=complex)
    H[1, 2] = g  # |01> <-> |10>
    H[2, 1] = g
    return H

g_iswap = 0.015  # 15 MHz
t_iswap = np.pi / (4 * g_iswap)  # For sqrt(iSWAP), π/2 for full iSWAP
print(f"Coupling strength: {g_iswap*1000:.0f} MHz")
print(f"Full iSWAP time: {np.pi/(2*g_iswap):.1f} ns")
print(f"sqrt(iSWAP) time: {t_iswap:.1f} ns")

H_iswap = iswap_hamiltonian(g_iswap)

# Evolution from |01>
psi0_01 = np.array([0, 1, 0, 0], dtype=complex)
t_points = np.linspace(0, 100, 300)

pops_iswap = []
for t in t_points:
    U = expm(-1j * 2 * np.pi * H_iswap * t)
    psi = U @ psi0_01
    pops_iswap.append(np.abs(psi)**2)

pops_iswap = np.array(pops_iswap)

fig3, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_points, pops_iswap[:, 0], 'b-', linewidth=2, label='|00⟩')
ax.plot(t_points, pops_iswap[:, 1], 'r-', linewidth=2, label='|01⟩')
ax.plot(t_points, pops_iswap[:, 2], 'g-', linewidth=2, label='|10⟩')
ax.plot(t_points, pops_iswap[:, 3], 'm-', linewidth=2, label='|11⟩')
ax.axvline(np.pi/(2*2*np.pi*g_iswap), color='gray', linestyle='--', alpha=0.5,
           label='iSWAP time')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Population', fontsize=12)
ax.set_title('iSWAP Gate: |01⟩ ↔ |10⟩ Exchange', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iswap_gate_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: ZZ Coupling Analysis
# =============================================================================

print("\n" + "=" * 60)
print("ZZ Coupling Analysis")
print("=" * 60)

def calculate_zz(g, delta, alpha1, alpha2):
    """Calculate ZZ coupling strength."""
    return 2 * g**2 * alpha1 * alpha2 / (delta * (delta + alpha1) * (delta + alpha2))

# Parameter sweep
delta_values = np.linspace(0.1, 0.5, 50)  # GHz detuning
g_values = [0.005, 0.010, 0.015]  # 5, 10, 15 MHz coupling
alpha = -0.25  # GHz

fig4, ax = plt.subplots(figsize=(8, 5))

for g in g_values:
    zz_vals = [calculate_zz(g, d, alpha, alpha) * 1000 for d in delta_values]  # Convert to MHz
    ax.plot(delta_values * 1000, zz_vals, linewidth=2, label=f'g = {g*1000:.0f} MHz')

ax.set_xlabel('Detuning Δ (MHz)', fontsize=12)
ax.set_ylabel('ZZ coupling ζ (kHz)', fontsize=12)
ax.set_title('Always-On ZZ Coupling vs Detuning', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zz_coupling.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate phase error from ZZ during idle
zz_example = calculate_zz(0.01, 0.3, alpha, alpha)
idle_time = 1000  # 1 μs
phase_error = 2 * np.pi * zz_example * idle_time
print(f"\nExample ZZ coupling: {zz_example*1e6:.1f} kHz")
print(f"Phase error after {idle_time} ns idle: {phase_error:.3f} rad = {np.degrees(phase_error):.1f}°")

# =============================================================================
# Part 6: Gate Fidelity vs Decoherence
# =============================================================================

print("\n" + "=" * 60)
print("Gate Fidelity Analysis")
print("=" * 60)

def decoherence_fidelity(t_gate, T1, T2):
    """Estimate fidelity loss from decoherence during gate."""
    # Simplified model: average over computational basis
    p_decay = 1 - np.exp(-t_gate / T1)
    p_dephase = 1 - np.exp(-t_gate / T2)
    return 1 - p_decay - p_dephase / 2

T1 = 100000  # 100 μs in ns
T2 = 80000   # 80 μs in ns

gate_times = np.linspace(10, 500, 50)  # ns
fidelities = [decoherence_fidelity(t, T1, T2) for t in gate_times]

fig5, ax = plt.subplots(figsize=(8, 5))
ax.plot(gate_times, np.array(fidelities) * 100, 'b-', linewidth=2)
ax.axhline(99.5, color='g', linestyle='--', label='99.5% threshold')
ax.axhline(99.0, color='r', linestyle='--', label='99.0% threshold')
ax.set_xlabel('Gate time (ns)', fontsize=12)
ax.set_ylabel('Fidelity (%)', fontsize=12)
ax.set_title('Decoherence-Limited Gate Fidelity', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([98, 100])

plt.tight_layout()
plt.savefig('gate_fidelity_decoherence.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"T1 = {T1/1000:.0f} μs, T2 = {T2/1000:.0f} μs")
for t in [30, 100, 300]:
    f = decoherence_fidelity(t, T1, T2)
    print(f"  {t} ns gate: {f*100:.3f}% fidelity")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Exchange coupling | $\hat{H} = g(\hat{a}_1^\dagger\hat{a}_2 + h.c.)$ |
| CR effective Hamiltonian | $\hat{H}_{CR} = (\Omega_{ZX}/2)\hat{\sigma}_z\hat{\sigma}_x + ...$ |
| CZ gate | At $\|11\rangle \leftrightarrow \|20\rangle$ crossing |
| iSWAP time | $t = \pi/(2g)$ |
| ZZ coupling | $\zeta = 2g^2\alpha_1\alpha_2/(\Delta(\Delta+\alpha_1)(\Delta+\alpha_2))$ |

### Main Takeaways

1. **Cross-resonance**: Drive control at target frequency creates ZX interaction; works with fixed-frequency qubits

2. **CZ gate**: Flux-tune to $|11\rangle - |20\rangle$ avoided crossing; fast but requires tunable qubits

3. **iSWAP**: Exchange interaction at resonance swaps $|01\rangle \leftrightarrow |10\rangle$ with phase

4. **ZZ coupling**: Always-on parasitic interaction; causes phase errors during idle

5. **Tunable couplers**: Enable turning coupling on/off, reducing idle errors

6. **State of the art**: 99-99.7% two-qubit gate fidelities with 20-200 ns gate times

## Daily Checklist

- [ ] I understand capacitive coupling between transmons
- [ ] I can explain the cross-resonance gate mechanism
- [ ] I understand CZ gates via avoided level crossings
- [ ] I can calculate ZZ coupling and its effects
- [ ] I understand the role of tunable couplers
- [ ] I have run the computational lab and interpreted the results
- [ ] I can compare different two-qubit gate approaches

## Preview: Day 902

Tomorrow we explore **readout mechanisms** for superconducting qubits:

- Dispersive readout theory and measurement
- Quantum non-demolition (QND) requirements
- Readout fidelity and errors
- Multiplexed readout for many qubits
- Purcell effect and filtering

---

*"The two-qubit gate is where quantum computing becomes more than just parallel classical bits. It's where entanglement is born."*
