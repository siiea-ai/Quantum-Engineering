# Day 915: Two-Qubit Rydberg Gates

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | CZ gates, Rydberg-dressed gates, native multi-qubit gates |
| **Afternoon** | 2 hours | Problem solving: gate pulse optimization |
| **Evening** | 2 hours | Computational lab: gate fidelity simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Implement CZ gates** using Rydberg blockade dynamics
2. **Analyze the controlled-phase mechanism** from blockade-induced phases
3. **Design Rydberg-dressed gates** for continuous two-qubit coupling
4. **Construct native CCZ gates** exploiting multi-atom blockade
5. **Calculate gate fidelities** limited by Rydberg decay and motional effects
6. **Optimize pulse sequences** for high-fidelity entangling operations

## Core Content

### 1. CZ Gate via Rydberg Blockade

#### Basic Mechanism

The controlled-Z (CZ) gate applies a π phase when both qubits are in state $|1\rangle$:
$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Implementation via Rydberg:**
1. Map computational $|1\rangle$ to Rydberg state $|r\rangle$
2. Blockade prevents double excitation $|11\rangle \to |rr\rangle$
3. Differential phase accumulation creates CZ

#### Pulse Sequence

**Three-pulse CZ protocol:**

1. **Global π pulse** on $|1\rangle \to |r\rangle$ transition
   - $|00\rangle \to |00\rangle$
   - $|01\rangle \to |0r\rangle$
   - $|10\rangle \to |r0\rangle$
   - $|11\rangle \to$ blocked (stays in ground manifold)

2. **Wait time** (optional, for phase accumulation)

3. **Global π pulse** on $|r\rangle \to |1\rangle$ transition
   - Returns excited atoms to $|1\rangle$

The $|11\rangle$ state never gets excited due to blockade, acquiring no phase. Other states with single excitations pick up phases that can be corrected with single-qubit operations.

#### Phase Analysis

Let's trace each computational basis state:

| Initial | After π | After π | Net Phase |
|---------|---------|---------|-----------|
| $|00\rangle$ | $|00\rangle$ | $|00\rangle$ | 0 |
| $|01\rangle$ | $-i|0r\rangle$ | $-|01\rangle$ | $\pi$ |
| $|10\rangle$ | $-i|r0\rangle$ | $-|10\rangle$ | $\pi$ |
| $|11\rangle$ | blocked | $|11\rangle$ | 0 |

This gives $\text{diag}(1, -1, -1, 1)$, equivalent to CZ up to single-qubit Z rotations.

### 2. Optimized CZ Gate Protocols

#### Time-Optimal Global CZ

Minimize gate time by using shaped pulses:

$$\Omega(t) = \Omega_0 f(t)$$

where $f(t)$ is optimized to maximize fidelity for a given gate time.

**Blackman pulse:**
$$f(t) = 0.42 - 0.5\cos(2\pi t/T) + 0.08\cos(4\pi t/T)$$

Reduces leakage to non-computational states.

#### Local Addressing CZ

Address only one of the two qubits:

1. π pulse on atom A: $|1\rangle_A \to |r\rangle_A$
2. 2π pulse on atom B (blocked if A is in $|r\rangle$)
3. π pulse on atom A: $|r\rangle_A \to |1\rangle_A$

The $|11\rangle$ state acquires phase from the blocked 2π pulse:
$$|11\rangle \xrightarrow{\pi} |r1\rangle \xrightarrow{2\pi_{blocked}} e^{i\phi}|r1\rangle \xrightarrow{\pi} e^{i\phi}|11\rangle$$

With proper detuning, $\phi = \pi$ gives exact CZ.

#### Gate Time and Fidelity

Minimum gate time set by Rydberg Rabi frequency:
$$t_{gate} \approx \frac{4\pi}{\Omega_r}$$

For $\Omega_r = 2\pi \times 5$ MHz: $t_{gate} \approx 400$ ns

**Fidelity limits:**
$$\boxed{F \approx 1 - \frac{\Gamma_r t_{gate}}{2} - \epsilon_{blockade}}$$

where:
- $\Gamma_r = 1/\tau_r$ is Rydberg decay rate
- $\epsilon_{blockade} = (\hbar\Omega/V)^2$ is blockade leakage

### 3. Rydberg-Dressed Gates

#### Dressed State Concept

Instead of fully exciting to Rydberg states, **weakly dress** the ground state:

$$|\tilde{1}\rangle = |1\rangle + \frac{\Omega}{2\Delta}|r\rangle$$

for large detuning $\Delta \gg \Omega$.

The dressed state inherits a fraction of Rydberg interactions:
$$\tilde{V} = \left(\frac{\Omega}{2\Delta}\right)^4 V_{rr}$$

#### Dressed State Hamiltonian

For two dressed atoms:
$$\hat{H}_{dressed} = \sum_i \frac{\Omega^2}{4\Delta}\hat{n}_i + \left(\frac{\Omega}{2\Delta}\right)^4 \frac{C_6}{r^6}\hat{n}_1\hat{n}_2$$

The first term is a light shift (correctable), the second is the effective interaction.

#### Gate via Continuous Interaction

The two-atom phase evolution:
$$\phi_{11}(t) = \tilde{V}t/\hbar = \left(\frac{\Omega}{2\Delta}\right)^4 \frac{C_6}{r^6\hbar}t$$

For a CZ gate, require $\phi_{11} = \pi$:
$$t_{CZ} = \pi\hbar \left(\frac{2\Delta}{\Omega}\right)^4 \frac{r^6}{C_6}$$

#### Advantages and Disadvantages

**Advantages:**
- Always-on gates (no Rydberg excitation pulses)
- Robust to Rydberg decay (only virtual excitation)
- Continuous evolution for analog simulation

**Disadvantages:**
- Slower gates (weaker interaction)
- Require precise intensity control
- Residual light shifts

### 4. Native Multi-Qubit Gates

#### CCZ Gate

The controlled-controlled-Z gate:
$$\text{CCZ}|ijk\rangle = (-1)^{ijk}|ijk\rangle$$

applies a sign flip only to $|111\rangle$.

**Native implementation:**
With three atoms in the blockade regime:
1. Global π pulse: $|1\rangle \to |r\rangle$
   - $|111\rangle$ is blockaded (no excitation)
   - $|011\rangle \to |0rr\rangle$? No, also blockaded if all three within $r_b$

Actually, the analysis is subtle. For perfect blockade of all pairs:
- $|111\rangle$: Stays blocked, acquires no phase
- Any state with two $|1\rangle$s: Also blocked
- States with one $|1\rangle$: Fully excited

This doesn't directly give CCZ. More sophisticated protocols exist.

#### True CCZ Protocol

**Using the super-atom picture:**
Three atoms within blockade radius act as a single super-atom with collective Rabi frequency $\sqrt{3}\Omega$.

Protocol:
1. Apply pulse that is a 2π rotation for single atom but different for collective state
2. Differential phase accumulation gives CCZ

Alternatively, use sequential controlled operations or specific pulse shapes.

#### Advantages of Native Multi-Qubit Gates

- Fewer operations than decomposition into two-qubit gates
- Lower error accumulation
- CCZ = key gate for Toffoli (with Hadamards)

### 5. Error Sources and Mitigation

#### Rydberg State Decay

Lifetime at n=70: $\tau \approx 300$ μs

Decay probability during gate:
$$P_{decay} = \frac{t_{gate}}{\tau}$$

For $t_{gate} = 500$ ns: $P_{decay} \approx 1.7 \times 10^{-3}$

**Mitigation:**
- Use higher Rabi frequencies (shorter gates)
- Work at lower n (shorter lifetime but stronger interaction)
- Cryogenic environment (reduce BBR)

#### Motional Errors

Position fluctuations cause interaction fluctuations:
$$\frac{\delta V}{V} = 6\frac{\delta r}{r}$$

For thermal atoms at temperature T:
$$\delta r = \sqrt{\frac{k_B T}{m\omega_{trap}^2}}$$

**Mitigation:**
- Sideband cooling to motional ground state
- Magic-wavelength traps
- Robust pulse sequences

#### Doppler Dephasing

Atomic motion during gate causes phase fluctuations:
$$\delta\phi = \mathbf{k}_{eff} \cdot \delta\mathbf{r}$$

For two-photon Rydberg excitation:
$$|\mathbf{k}_{eff}| \approx 2k_{blue} = \frac{4\pi}{\lambda_{blue}}$$

**Mitigation:**
- Counter-propagating beams
- Tight confinement
- Fast gates

#### Laser Phase Noise

Phase noise during gate contributes to infidelity:
$$\epsilon_{phase} \approx (\delta\phi_{rms})^2 \approx (\delta\nu \cdot t_{gate})^2$$

For 1 kHz linewidth and 500 ns gate:
$$\epsilon_{phase} \approx (10^3 \times 5 \times 10^{-7})^2 = 2.5 \times 10^{-7}$$

This is typically negligible.

### 6. State-of-the-Art Performance

#### Demonstrated Fidelities

| Group | Year | Gate | Fidelity |
|-------|------|------|----------|
| Lukin (Harvard) | 2022 | CZ | 99.5% |
| Browaeys (IOGS) | 2022 | CZ | 99.3% |
| Saffman (Wisconsin) | 2020 | CZ | 97% |

#### Limiting Factors

Current fidelity limits:
1. Rydberg decay: ~0.2%
2. Doppler/motional: ~0.1%
3. Laser noise: <0.1%
4. SPAM errors: ~0.2% (not gate error)

Theoretical limit with current technology: ~99.9%

## Worked Examples

### Example 1: CZ Gate Pulse Sequence Design

**Problem:** Design a CZ gate for two Rb-87 atoms separated by 5 μm using the 70S Rydberg state. Calculate gate time and expected fidelity.

**Solution:**

**Step 1: Check blockade condition**
$C_6(70S) \approx 2100$ GHz·μm⁶

$$V = \frac{C_6}{r^6} = \frac{2100}{5^6} = 134\,\text{MHz}$$

Choose $\Omega_r = 2\pi \times 5$ MHz for strong blockade ($V/\Omega = 27$).

**Step 2: Design pulse sequence**
Using global three-pulse protocol:
- π pulse duration: $t_\pi = \pi/\Omega_r = 100$ ns
- Total gate time: $t_{gate} = 3 \times t_\pi = 300$ ns

(Note: Can optimize to ~200 ns with shaped pulses)

**Step 3: Calculate decay error**
Rydberg lifetime at n=70: $\tau \approx 300$ μs

Time spent in Rydberg state (average): $\sim 100$ ns (one π pulse equivalent)

$$\epsilon_{decay} = \frac{100\,\text{ns}}{300\,\mu\text{s}} \approx 3 \times 10^{-4}$$

**Step 4: Calculate blockade error**
$$\epsilon_{blockade} = \left(\frac{\hbar\Omega}{V}\right)^2 = \left(\frac{5}{134}\right)^2 = 1.4 \times 10^{-3}$$

**Step 5: Total fidelity**
$$F \approx 1 - \epsilon_{decay} - \epsilon_{blockade} \approx 1 - 1.7 \times 10^{-3} = 99.83\%$$

---

### Example 2: Rydberg-Dressed Gate

**Problem:** Design a Rydberg-dressed CZ gate for atoms at 6 μm separation using n=50 states. Calculate the required dressing parameters and gate time.

**Solution:**

**Step 1: Rydberg parameters**
$C_6(50S) \approx 140$ GHz·μm⁶

Bare interaction at 6 μm:
$$V_{rr} = \frac{140}{6^6} = 3.0\,\text{MHz}$$

**Step 2: Choose dressing parameters**
For weak dressing, need $\Delta \gg \Omega$. Choose:
- Detuning: $\Delta = 2\pi \times 100$ MHz
- Rabi frequency: $\Omega = 2\pi \times 10$ MHz

Dressing ratio: $\Omega/(2\Delta) = 0.05$

**Step 3: Calculate dressed interaction**
$$\tilde{V} = \left(\frac{\Omega}{2\Delta}\right)^4 V_{rr} = (0.05)^4 \times 3.0\,\text{MHz}$$
$$\tilde{V} = 6.25 \times 10^{-6} \times 3.0 \times 10^6 = 19\,\text{Hz}$$

**Step 4: Gate time for π phase**
$$t_{CZ} = \frac{\pi}{\tilde{V}} = \frac{\pi}{2\pi \times 19} = 26\,\text{ms}$$

This is very slow!

**Step 5: Optimize**
Increase dressing by using smaller detuning:
- $\Delta = 2\pi \times 20$ MHz
- $\Omega = 2\pi \times 10$ MHz
- $\Omega/(2\Delta) = 0.25$

$$\tilde{V} = (0.25)^4 \times 3.0\,\text{MHz} = 11.7\,\text{kHz}$$
$$t_{CZ} = \frac{1}{2 \times 11.7\,\text{kHz}} = 43\,\mu\text{s}$$

Still slow but more practical for analog simulation.

---

### Example 3: Native CCZ Fidelity

**Problem:** Three atoms form an equilateral triangle with 4 μm sides. Using n=60 Rydberg states and $\Omega = 2\pi \times 3$ MHz, analyze whether native CCZ is feasible.

**Solution:**

**Step 1: Pair interactions**
$C_6(60S) \approx 600$ GHz·μm⁶

For all pairs at 4 μm:
$$V_{pair} = \frac{600}{4^6} = 146\,\text{MHz}$$

**Step 2: Check three-body blockade**
For perfect three-body blockade, need:
$$V_{pair} \gg \hbar\Omega$$

$$\frac{V_{pair}}{\hbar\Omega} = \frac{146}{3} = 49$$

Strong blockade condition satisfied.

**Step 3: Collective Rabi frequency**
For 3 atoms: $\Omega_3 = \sqrt{3} \times \Omega = 2\pi \times 5.2$ MHz

**Step 4: Analyze pulse**
A 2π pulse on single atom is a 2π/√3 pulse on collective state.

Phase difference: $2\pi - 2\pi/\sqrt{3} = 2\pi(1 - 1/\sqrt{3}) = 2.55$ rad

This isn't exactly π, so direct protocol doesn't give CCZ.

**Step 5: Modified protocol**
Use pulse duration $t$ such that:
- Single atom: $\Omega t = 2\pi$ → returns to ground
- Collective (3 atoms): $\sqrt{3}\Omega t = 2\pi\sqrt{3}$ → acquires phase $\pi$

Hmm, this gives the wrong sign. Need more sophisticated approach.

**Alternative:** Use sequential CZ gates:
$$CCZ = CZ_{12} \cdot CZ_{23} \cdot CZ_{13}$$

With 99.8% CZ fidelity, CCZ fidelity ≈ 99.4%.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the blockade error for a CZ gate with $V = 50\Omega$. What interaction strength is needed for 0.1% blockade error?

**Problem 1.2:** Design a CZ gate for atoms at 4 μm using n=80 Rydberg states. Calculate gate time for $\Omega = 2\pi \times 10$ MHz.

**Problem 1.3:** For a Rydberg-dressed gate with $\Omega/(2\Delta) = 0.1$ and bare interaction 10 MHz, what is the gate time for a π phase?

### Level 2: Intermediate Analysis

**Problem 2.1:** Compare the fidelity of CZ gates using n=50 vs n=80 Rydberg states for atoms at 5 μm. Consider both decay and blockade errors. Optimize the Rabi frequency for each case.

**Problem 2.2:** Design a CZ gate that is robust to 5% variation in Rabi frequency using composite pulses. Calculate the pulse sequence.

**Problem 2.3:** Analyze the motional error for a CZ gate with atoms at temperature T=10 μK and trap frequency 100 kHz. At what temperature does this error equal the Rydberg decay error?

### Level 3: Challenging Problems

**Problem 3.1:** Derive the optimal Rabi frequency that minimizes total gate error considering both blockade leakage ($\propto \Omega^2$) and Rydberg decay ($\propto 1/\Omega$).

**Problem 3.2:** Design a native CCZ gate using three addressing beams. Calculate the pulse sequence and expected fidelity.

**Problem 3.3:** Analyze the effect of electric field noise on CZ gate fidelity. For Rydberg state polarizability $\alpha = 10^{10}$ a.u. and field noise 1 mV/cm, what is the dephasing rate?

## Computational Lab: Rydberg Gate Simulations

### Lab 1: CZ Gate Dynamics

```python
"""
Day 915 Lab: Two-Qubit Rydberg Gate Simulations
CZ gates via blockade and dressed interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def two_qubit_rydberg_hamiltonian(Omega, Delta, V, basis='computational'):
    """
    Construct two-qubit Hamiltonian with Rydberg coupling.

    Basis: |00⟩, |01⟩, |0r⟩, |10⟩, |11⟩, |1r⟩, |r0⟩, |r1⟩, |rr⟩
    where |0⟩, |1⟩ are computational states and |r⟩ is Rydberg.
    """
    dim = 9
    H = np.zeros((dim, dim), dtype=complex)

    # State indices
    idx = {'00': 0, '01': 1, '0r': 2, '10': 3, '11': 4, '1r': 5,
           'r0': 6, 'r1': 7, 'rr': 8}

    # Rydberg coupling (|1⟩ ↔ |r⟩)
    # Atom 1
    H[idx['01'], idx['0r']] = Omega/2
    H[idx['0r'], idx['01']] = Omega/2
    H[idx['11'], idx['1r']] = Omega/2
    H[idx['1r'], idx['11']] = Omega/2

    # Atom 2
    H[idx['10'], idx['r0']] = Omega/2
    H[idx['r0'], idx['10']] = Omega/2
    H[idx['11'], idx['r1']] = Omega/2
    H[idx['r1'], idx['11']] = Omega/2

    # Coupled transitions
    H[idx['1r'], idx['rr']] = Omega/2
    H[idx['rr'], idx['1r']] = Omega/2
    H[idx['r1'], idx['rr']] = Omega/2
    H[idx['rr'], idx['r1']] = Omega/2

    # Detuning (Rydberg states)
    for state, i in idx.items():
        n_rydberg = state.count('r')
        H[i, i] = -Delta * n_rydberg

    # Interaction (|rr⟩ state)
    H[idx['rr'], idx['rr']] += V

    return H, idx

def simulate_cz_gate(Omega, V, t_gate, n_steps=500):
    """
    Simulate CZ gate via Rydberg blockade.

    Returns fidelity and final phases.
    """
    H, idx = two_qubit_rydberg_hamiltonian(Omega, 0, V)

    t = np.linspace(0, t_gate, n_steps)
    results = {}

    # Initial computational states
    init_states = {'00': 0, '01': 1, '10': 3, '11': 4}

    for name, init_idx in init_states.items():
        psi0 = np.zeros(9, dtype=complex)
        psi0[init_idx] = 1

        psi_t = np.zeros((n_steps, 9), dtype=complex)
        for i, ti in enumerate(t):
            U = expm(-1j * H * ti)
            psi_t[i] = U @ psi0

        results[name] = psi_t

    return t, results, idx

# Simulate CZ gate
Omega = 2 * np.pi * 5  # 5 MHz
V = 2 * np.pi * 100  # 100 MHz interaction (strong blockade)

# π pulse time
t_pi = np.pi / Omega
t_gate = 3 * t_pi  # Three-pulse protocol

print("=== CZ Gate Simulation ===")
print(f"Rabi frequency: {Omega/(2*np.pi):.1f} MHz")
print(f"Interaction: {V/(2*np.pi):.1f} MHz")
print(f"V/Ω ratio: {V/Omega:.1f}")
print(f"π-pulse time: {t_pi*1e9:.1f} ns")
print(f"Gate time: {t_gate*1e9:.1f} ns")

t, results, idx = simulate_cz_gate(Omega, V, t_gate)

# Plot dynamics for each initial state
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, psi_t) in zip(axes.flatten(), results.items()):
    # Computational state populations
    P_comp = np.abs(psi_t[:, idx['00']])**2 + np.abs(psi_t[:, idx['01']])**2 + \
             np.abs(psi_t[:, idx['10']])**2 + np.abs(psi_t[:, idx['11']])**2

    # Single Rydberg populations
    P_1r = np.abs(psi_t[:, idx['0r']])**2 + np.abs(psi_t[:, idx['1r']])**2 + \
           np.abs(psi_t[:, idx['r0']])**2 + np.abs(psi_t[:, idx['r1']])**2

    # Double Rydberg
    P_rr = np.abs(psi_t[:, idx['rr']])**2

    ax.plot(t*1e9, P_comp, 'b-', label='Computational', linewidth=2)
    ax.plot(t*1e9, P_1r, 'g-', label='Single Rydberg', linewidth=2)
    ax.plot(t*1e9, P_rr, 'r-', label='Double Rydberg', linewidth=2)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title(f'Initial: |{name}⟩')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig('cz_gate_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze final phases
print("\n=== Final State Analysis ===")
target_phases = {'00': 0, '01': np.pi, '10': np.pi, '11': 0}

for name, psi_t in results.items():
    psi_final = psi_t[-1]

    # Population in original computational state
    init_idx = {'00': 0, '01': 1, '10': 3, '11': 4}[name]
    pop = np.abs(psi_final[init_idx])**2
    phase = np.angle(psi_final[init_idx])

    print(f"|{name}⟩: P = {pop:.4f}, φ = {phase:.3f} rad ({phase/np.pi:.3f}π)")
```

### Lab 2: Fidelity Analysis

```python
"""
Lab 2: Gate fidelity analysis with error sources
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def cz_fidelity(Omega, V, Gamma_r, t_gate):
    """
    Calculate CZ gate fidelity including Rydberg decay.

    Uses simplified model:
    - Blockade error: (Omega/V)^2
    - Decay error: Gamma_r * t_gate * <n_r>
    """
    # Blockade error
    eps_blockade = (Omega / V)**2

    # Average Rydberg occupation during gate
    # For 3-pulse protocol: ~1/3 of time in Rydberg for single-excitation states
    n_r_avg = 0.3

    # Decay error
    eps_decay = Gamma_r * t_gate * n_r_avg

    # Total fidelity
    F = 1 - eps_blockade - eps_decay

    return F, eps_blockade, eps_decay

# Scan Rabi frequency
Omega_values = np.linspace(0.5, 20, 50) * 2 * np.pi  # 0.5-20 MHz
V = 2 * np.pi * 100  # 100 MHz interaction
tau_r = 300e-6  # 300 μs Rydberg lifetime
Gamma_r = 1 / tau_r

fidelities = []
eps_block = []
eps_dec = []
gate_times = []

for Omega in Omega_values:
    t_gate = 3 * np.pi / Omega  # 3 π-pulses
    gate_times.append(t_gate)

    F, eb, ed = cz_fidelity(Omega, V, Gamma_r, t_gate)
    fidelities.append(F)
    eps_block.append(eb)
    eps_dec.append(ed)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(Omega_values/(2*np.pi), np.array(fidelities)*100, 'b-', linewidth=2)
axes[0].set_xlabel('Rabi frequency (MHz)')
axes[0].set_ylabel('Fidelity (%)')
axes[0].set_title('CZ Gate Fidelity vs Rabi Frequency')
axes[0].grid(True, alpha=0.3)

# Find optimal
opt_idx = np.argmax(fidelities)
axes[0].axvline(x=Omega_values[opt_idx]/(2*np.pi), color='r', linestyle='--')
axes[0].text(Omega_values[opt_idx]/(2*np.pi), fidelities[opt_idx]*100 - 0.2,
             f'Optimal: {Omega_values[opt_idx]/(2*np.pi):.1f} MHz', ha='center')

# Error breakdown
axes[1].semilogy(Omega_values/(2*np.pi), eps_block, 'g-', label='Blockade', linewidth=2)
axes[1].semilogy(Omega_values/(2*np.pi), eps_dec, 'r-', label='Decay', linewidth=2)
axes[1].semilogy(Omega_values/(2*np.pi), np.array(eps_block)+np.array(eps_dec),
                 'b--', label='Total', linewidth=2)
axes[1].set_xlabel('Rabi frequency (MHz)')
axes[1].set_ylabel('Error probability')
axes[1].set_title('Error Budget')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Gate time
axes[2].plot(Omega_values/(2*np.pi), np.array(gate_times)*1e9, 'b-', linewidth=2)
axes[2].set_xlabel('Rabi frequency (MHz)')
axes[2].set_ylabel('Gate time (ns)')
axes[2].set_title('Gate Duration')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cz_fidelity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Optimal Rabi frequency: {Omega_values[opt_idx]/(2*np.pi):.2f} MHz")
print(f"Maximum fidelity: {fidelities[opt_idx]*100:.2f}%")
print(f"Gate time at optimum: {gate_times[opt_idx]*1e9:.1f} ns")

# Scan interaction strength
print("\n=== Fidelity vs Interaction Strength ===")

V_values = np.logspace(1, 3, 50) * 2 * np.pi  # 10-1000 MHz
Omega = 2 * np.pi * 5  # Fixed 5 MHz Rabi

fidelities_V = []
for V in V_values:
    t_gate = 3 * np.pi / Omega
    F, _, _ = cz_fidelity(Omega, V, Gamma_r, t_gate)
    fidelities_V.append(F)

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(V_values/(2*np.pi), np.array(fidelities_V)*100, 'b-', linewidth=2)
ax.axhline(y=99.9, color='gray', linestyle='--', label='99.9% threshold')
ax.set_xlabel('Interaction strength V (MHz)')
ax.set_ylabel('Fidelity (%)')
ax.set_title('CZ Fidelity vs Interaction Strength (Ω = 5 MHz)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cz_fidelity_vs_interaction.png', dpi=150, bbox_inches='tight')
plt.show()

# Find minimum V for 99.9% fidelity
threshold_idx = np.where(np.array(fidelities_V) > 0.999)[0]
if len(threshold_idx) > 0:
    V_min = V_values[threshold_idx[0]]
    print(f"Minimum V for 99.9% fidelity: {V_min/(2*np.pi):.1f} MHz")
    print(f"Required V/Ω ratio: {V_min/Omega:.1f}")
```

### Lab 3: Rydberg-Dressed Gates

```python
"""
Lab 3: Rydberg-dressed gate simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dressed_two_qubit_evolution(psi, t, Omega, Delta, V_rr):
    """
    Effective two-qubit dynamics with Rydberg dressing.

    States: |00⟩, |01⟩, |10⟩, |11⟩
    """
    # Dressed interaction
    V_dressed = (Omega / (2*Delta))**4 * V_rr

    # Light shift
    delta_LS = Omega**2 / (4*Delta)

    # Hamiltonian matrix (diagonal with interaction)
    # Energy shifts from light shift (affects |01⟩, |10⟩, |11⟩)
    E_01 = -delta_LS
    E_10 = -delta_LS
    E_11 = -2*delta_LS + V_dressed

    # Time evolution (diagonal, so just phases)
    dpsi = np.zeros(8, dtype=float)  # [Re, Im] for 4 states

    # |00⟩: no shift
    dpsi[0] = 0
    dpsi[1] = 0

    # |01⟩
    dpsi[2] = -E_01 * psi[3]
    dpsi[3] = E_01 * psi[2]

    # |10⟩
    dpsi[4] = -E_10 * psi[5]
    dpsi[5] = E_10 * psi[4]

    # |11⟩
    dpsi[6] = -E_11 * psi[7]
    dpsi[7] = E_11 * psi[6]

    return dpsi

# Parameters
Omega = 2 * np.pi * 10e6  # 10 MHz dressing Rabi
Delta = 2 * np.pi * 50e6  # 50 MHz detuning
V_rr = 2 * np.pi * 50e6   # 50 MHz bare Rydberg interaction

# Calculate dressed quantities
V_dressed = (Omega / (2*Delta))**4 * V_rr
delta_LS = Omega**2 / (4*Delta)

print("=== Rydberg-Dressed Gate ===")
print(f"Dressing Rabi: {Omega/(2*np.pi)/1e6:.1f} MHz")
print(f"Detuning: {Delta/(2*np.pi)/1e6:.1f} MHz")
print(f"Ω/(2Δ): {Omega/(2*Delta):.3f}")
print(f"Bare interaction: {V_rr/(2*np.pi)/1e6:.1f} MHz")
print(f"Dressed interaction: {V_dressed/(2*np.pi)/1e3:.2f} kHz")
print(f"Light shift: {delta_LS/(2*np.pi)/1e6:.2f} MHz")

# Time for CZ gate (π phase)
t_CZ = np.pi / V_dressed
print(f"CZ gate time: {t_CZ*1e6:.1f} μs")

# Simulate
t = np.linspace(0, 2*t_CZ, 500)

# Initial state: |++⟩ = (|0⟩+|1⟩)(|0⟩+|1⟩)/2
# = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
psi0 = np.array([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0])

psi_t = odeint(dressed_two_qubit_evolution, psi0, t, args=(Omega, Delta, V_rr))

# Calculate phases
phase_00 = np.arctan2(psi_t[:, 1], psi_t[:, 0])
phase_01 = np.arctan2(psi_t[:, 3], psi_t[:, 2])
phase_10 = np.arctan2(psi_t[:, 5], psi_t[:, 4])
phase_11 = np.arctan2(psi_t[:, 7], psi_t[:, 6])

# Relative phases (compensating single-qubit light shifts)
phase_relative = phase_11 - phase_01 - phase_10 + phase_00

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Phase evolution
axes[0].plot(t*1e6, phase_00/np.pi, label='|00⟩', linewidth=2)
axes[0].plot(t*1e6, phase_01/np.pi, label='|01⟩', linewidth=2)
axes[0].plot(t*1e6, phase_10/np.pi, label='|10⟩', linewidth=2)
axes[0].plot(t*1e6, phase_11/np.pi, label='|11⟩', linewidth=2)
axes[0].axvline(x=t_CZ*1e6, color='r', linestyle='--', label='CZ time')
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Phase (π)')
axes[0].set_title('Phase Evolution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative phase (should be π at CZ time)
axes[1].plot(t*1e6, phase_relative/np.pi, 'b-', linewidth=2)
axes[1].axhline(y=1, color='r', linestyle='--', label='π phase (CZ)')
axes[1].axvline(x=t_CZ*1e6, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Time (μs)')
axes[1].set_ylabel('Relative phase (π)')
axes[1].set_title('CZ Phase Accumulation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dressed_gate.png', dpi=150, bbox_inches='tight')
plt.show()

# Compare blockade vs dressed gates
print("\n=== Blockade vs Dressed Comparison ===")

# Blockade parameters for same atoms
Omega_block = 2 * np.pi * 5e6  # 5 MHz
t_CZ_block = 3 * np.pi / Omega_block

print(f"\nBlockade CZ:")
print(f"  Gate time: {t_CZ_block*1e9:.0f} ns")
print(f"  Rydberg occupation: ~30% during gate")

print(f"\nDressed CZ:")
print(f"  Gate time: {t_CZ*1e6:.1f} μs")
print(f"  Rydberg occupation: {(Omega/(2*Delta))**2*100:.2f}%")

# Scan dressing strength
dressing_ratios = np.linspace(0.05, 0.4, 50)
gate_times_dressed = []
rydberg_pop = []

for r in dressing_ratios:
    Omega_test = 2 * r * Delta
    V_d = r**4 * V_rr
    t_cz = np.pi / V_d

    gate_times_dressed.append(t_cz)
    rydberg_pop.append(r**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].semilogy(dressing_ratios, np.array(gate_times_dressed)*1e6, 'b-', linewidth=2)
axes[0].set_xlabel('Dressing ratio Ω/(2Δ)')
axes[0].set_ylabel('Gate time (μs)')
axes[0].set_title('Dressed Gate Time vs Dressing Strength')
axes[0].grid(True, alpha=0.3)

axes[1].plot(dressing_ratios, np.array(rydberg_pop)*100, 'r-', linewidth=2)
axes[1].set_xlabel('Dressing ratio Ω/(2Δ)')
axes[1].set_ylabel('Rydberg population (%)')
axes[1].set_title('Virtual Rydberg Excitation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dressed_gate_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| CZ gate time | $t_{CZ} \approx 3\pi/\Omega_r$ | 100-500 ns |
| Blockade error | $\epsilon_b = (\hbar\Omega/V)^2$ | 0.1-1% |
| Decay error | $\epsilon_d = \Gamma_r t_{gate}$ | 0.1-0.5% |
| Dressed interaction | $\tilde{V} = (\Omega/2\Delta)^4 V_{rr}$ | kHz-MHz |
| Dressed gate time | $t_{CZ} = \pi\hbar/\tilde{V}$ | 1-100 μs |

### Main Takeaways

1. **CZ gates via blockade** achieve >99% fidelity by preventing double Rydberg excitation; the three-pulse protocol is most common.

2. **Optimal Rabi frequency** balances blockade error ($\propto \Omega^2$) against decay error ($\propto 1/\Omega$).

3. **Rydberg-dressed gates** provide continuous interaction with reduced Rydberg occupation, suitable for analog simulation but slower.

4. **Native multi-qubit gates** like CCZ exploit collective blockade but require careful pulse engineering.

5. **Current limitations** are primarily from Rydberg decay (~0.2%) and motional effects (~0.1%), with demonstrated fidelities approaching 99.5%.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain CZ gate via blockade
- [ ] I understand dressed-state interactions
- [ ] I can describe native multi-qubit gates
- [ ] I know the main fidelity limitations

### Mathematical Skills
- [ ] I can calculate blockade and decay errors
- [ ] I can optimize Rabi frequency for fidelity
- [ ] I can derive dressed interaction strength

### Computational Skills
- [ ] I can simulate CZ gate dynamics
- [ ] I can analyze fidelity vs parameters
- [ ] I can model dressed-state evolution

## Preview: Day 916

Tomorrow we explore **Atom Sorting and Array Preparation**, where we will:
- Analyze defect detection via fluorescence imaging
- Implement real-time atom rearrangement algorithms
- Optimize loading efficiency with enhanced strategies
- Prepare defect-free arrays for quantum computing

Efficient array preparation is essential for scaling neutral atom quantum computers, as stochastic loading limits initial array occupancy to ~50%.
