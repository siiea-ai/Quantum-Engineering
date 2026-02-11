# Day 913: Rydberg Blockade Mechanism

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Blockade physics, collective enhancement, many-body dynamics |
| **Afternoon** | 2 hours | Problem solving: blockade radius calculations |
| **Evening** | 2 hours | Computational lab: blockade simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive the blockade radius** from the competition between interaction and Rabi frequency
2. **Analyze the blockade fidelity** as a function of experimental parameters
3. **Explain collective Rabi oscillations** in blockaded atomic ensembles
4. **Calculate the blockade shift** for multi-atom configurations
5. **Apply blockade physics** to quantum gates and simulation protocols
6. **Simulate blockade dynamics** for two-atom and many-atom systems

## Core Content

### 1. The Blockade Mechanism

#### Physical Picture

Consider two atoms separated by distance $r$, each driven by a laser to a Rydberg state with Rabi frequency $\Omega$. The Hamiltonian is:

$$\hat{H} = \frac{\hbar\Omega}{2}(\hat{\sigma}_1^{(+)} + \hat{\sigma}_1^{(-)} + \hat{\sigma}_2^{(+)} + \hat{\sigma}_2^{(-)}) + \frac{C_6}{r^6}\hat{n}_1\hat{n}_2$$

where $\hat{n}_i = |r\rangle\langle r|_i$ projects onto the Rydberg state.

The four basis states are:
- $|gg\rangle$: both in ground state
- $|gr\rangle$: atom 1 ground, atom 2 Rydberg
- $|rg\rangle$: atom 1 Rydberg, atom 2 ground
- $|rr\rangle$: both in Rydberg state

The interaction shifts only $|rr\rangle$ by energy $V = C_6/r^6$.

#### Blockade Condition

**Blockade occurs when the interaction energy exceeds the excitation linewidth:**
$$V(r) = \frac{C_6}{r^6} \gg \hbar\Omega$$

This prevents double excitation because the $|rr\rangle$ state is shifted out of resonance with the driving laser.

The **blockade radius** is defined where $V(r_b) = \hbar\Omega$:
$$\boxed{r_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}}$$

For typical parameters:
- $C_6 = 1000$ GHz·μm⁶ (n ≈ 70)
- $\Omega = 2\pi \times 1$ MHz

$$r_b = \left(\frac{1000 \times 10^9}{1 \times 10^6}\right)^{1/6}\,\mu\text{m} \approx 10\,\mu\text{m}$$

### 2. Two-Atom Blockade Dynamics

#### Hamiltonian in the Two-Atom Basis

In the rotating frame:
$$\hat{H} = \frac{\hbar\Omega}{2}\begin{pmatrix}
0 & 1 & 1 & 0 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
0 & 1 & 1 & V/\hbar
\end{pmatrix}$$

in the basis $\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$.

#### Perfect Blockade Limit ($V \gg \hbar\Omega$)

The $|rr\rangle$ state decouples, leaving an effective three-level system.

Introducing the symmetric entangled state:
$$|W\rangle = \frac{1}{\sqrt{2}}(|gr\rangle + |rg\rangle)$$

The effective Hamiltonian becomes:
$$\hat{H}_{eff} = \frac{\hbar\Omega\sqrt{2}}{2}(|gg\rangle\langle W| + |W\rangle\langle gg|)$$

This shows **collective enhancement**: the $|gg\rangle \leftrightarrow |W\rangle$ transition has Rabi frequency:
$$\boxed{\Omega_{\text{collective}} = \sqrt{2}\,\Omega}$$

#### Collective Rabi Oscillations

Starting from $|gg\rangle$, the system oscillates between $|gg\rangle$ and $|W\rangle$ with enhanced frequency $\sqrt{2}\Omega$.

The probability of finding exactly one Rydberg excitation:
$$P_1(t) = \sin^2\left(\frac{\sqrt{2}\Omega t}{2}\right)$$

A π-pulse transfers $|gg\rangle \to |W\rangle$:
$$t_\pi = \frac{\pi}{\sqrt{2}\Omega}$$

### 3. Blockade Fidelity Analysis

#### Blockade Error

Even with $V \gg \hbar\Omega$, there is finite leakage to $|rr\rangle$.

Using perturbation theory, the probability of double excitation during a π-pulse:
$$P_{rr} \approx \left(\frac{\hbar\Omega}{V}\right)^2 = \left(\frac{r}{r_b}\right)^{12}$$

For the blockade condition to give <1% error:
$$\frac{r}{r_b} < 0.68$$

Or equivalently: $V > 5\hbar\Omega$

#### Position Fluctuations

Atoms in the trap have finite temperature, causing position fluctuations:
$$\sigma_r \approx \sqrt{\frac{k_B T}{m\omega_{trap}^2}}$$

This leads to interaction fluctuations:
$$\frac{\delta V}{V} \approx 6\frac{\sigma_r}{r}$$

For $r = 5$ μm, $T = 10$ μK, $\omega_{trap} = 2\pi \times 100$ kHz:
$$\sigma_r \approx 30\,\text{nm} \Rightarrow \frac{\delta V}{V} \approx 4\%$$

### 4. Many-Body Blockade

#### N-Atom Collective Enhancement

For $N$ atoms within the blockade radius, all driven with Rabi frequency $\Omega$:

The collective ground state: $|G\rangle = |gg...g\rangle$

The single-excitation manifold (N states):
$$|W_k\rangle = |g...g\, r_k\, g...g\rangle$$

The symmetric superposition:
$$|W\rangle = \frac{1}{\sqrt{N}}\sum_{k=1}^N |W_k\rangle$$

The collective Rabi frequency:
$$\boxed{\Omega_N = \sqrt{N}\,\Omega}$$

This $\sqrt{N}$ enhancement enables:
- Faster gates with many atoms
- Collective quantum memory
- Superatom behavior

#### Antiblockade

When the detuning matches the interaction:
$$\delta = V/\hbar$$

The $|rr\rangle$ state comes back into resonance, and double excitation is enhanced. This **antiblockade** regime enables:
- Correlated excitation dynamics
- Rydberg aggregates
- Facilitated dynamics

### 5. Blockade in Different Geometries

#### 1D Chain

For a linear chain with spacing $a$, the nearest-neighbor interaction is $V_1 = C_6/a^6$.

Next-nearest-neighbor: $V_2 = C_6/(2a)^6 = V_1/64$

The blockade condition for NN while allowing NNN excitation:
$$\hbar\Omega < V_2 < \hbar\Omega' < V_1$$

This enables nearest-neighbor-only blockade for 1D quantum simulation.

#### 2D Square Lattice

In a square lattice:
- NN distance: $a$
- Diagonal distance: $\sqrt{2}a$
- NNN distance: $2a$

Interaction ratios:
$$V_{diag} = V_{NN}/2.83^6 \approx V_{NN}/511$$
$$V_{NNN} = V_{NN}/64$$

#### Triangular Lattice

For triangular geometry:
- 6 nearest neighbors at distance $a$
- 6 next-nearest neighbors at distance $\sqrt{3}a$

$$V_{NNN}/V_{NN} = 1/27$$

### 6. Applications of Rydberg Blockade

#### Quantum Gates

**CZ gate via blockade:**
The blockade prevents the $|11\rangle \to |rr\rangle$ transition, accumulating a conditional phase.

**Multi-qubit gates:**
The blockade naturally extends to CCZ and higher-order gates within the blockade volume.

#### Quantum Simulation

**Ising model:**
$$\hat{H} = \sum_i \hbar\Omega\hat{\sigma}_i^x + \sum_{i<j}\frac{C_6}{r_{ij}^6}\hat{n}_i\hat{n}_j$$

This is equivalent to a transverse-field Ising model with $1/r^6$ interactions.

**Quantum optimization:**
Maximum independent set problems map naturally to Rydberg blockade:
- Atoms = vertices
- Blockade = edges
- Maximize Rydberg excitations = find MIS

## Worked Examples

### Example 1: Blockade Radius Calculation

**Problem:** Calculate the blockade radius for Rb-87 atoms in the 60S state driven with $\Omega = 2\pi \times 5$ MHz.

**Solution:**

**Step 1: Find C₆ coefficient**
From scaling: $C_6 \approx 600$ GHz·μm⁶ for 60S+60S

**Step 2: Calculate blockade radius**
$$r_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}$$

Converting units: $\hbar\Omega = 5$ MHz = $5 \times 10^{-3}$ GHz

$$r_b = \left(\frac{600\,\text{GHz}\cdot\mu\text{m}^6}{5 \times 10^{-3}\,\text{GHz}}\right)^{1/6} = (1.2 \times 10^5)^{1/6}\,\mu\text{m}$$

$$r_b = 7.0\,\mu\text{m}$$

**Step 3: Verify blockade condition at typical spacing**
At $r = 5$ μm:
$$V = \frac{600}{5^6} = \frac{600}{15625} = 38.4\,\text{MHz}$$
$$V/(\hbar\Omega) = 38.4/5 = 7.7$$

This satisfies $V \gg \hbar\Omega$, giving blockade error $\approx (1/7.7)^2 = 1.7\%$.

---

### Example 2: Collective Rabi Oscillation Period

**Problem:** Five atoms are uniformly distributed within a blockade sphere of radius 8 μm. Calculate the collective Rabi oscillation frequency if each atom is driven with $\Omega = 2\pi \times 2$ MHz.

**Solution:**

**Step 1: Verify blockade condition**
All atoms within $r_b = 8$ μm means all pairs are blockaded.

**Step 2: Calculate collective frequency**
$$\Omega_5 = \sqrt{5} \times \Omega = \sqrt{5} \times 2\pi \times 2\,\text{MHz} = 2\pi \times 4.47\,\text{MHz}$$

**Step 3: π-pulse time**
$$t_\pi = \frac{\pi}{\Omega_5} = \frac{\pi}{2\pi \times 4.47 \times 10^6} = 112\,\text{ns}$$

**Step 4: Full oscillation period**
$$T = 2t_\pi = 224\,\text{ns}$$

Compare to single-atom: $T_1 = 2\pi/(2\pi \times 2\,\text{MHz}) = 500$ ns

The collective enhancement speeds up oscillations by factor $\sqrt{5} = 2.24$.

---

### Example 3: Blockade Gate Fidelity

**Problem:** Two atoms are separated by 4 μm in a trap with temperature 20 μK and trap frequency 80 kHz. Using n=70 Rydberg states ($C_6 = 2100$ GHz·μm⁶) and $\Omega = 2\pi \times 3$ MHz, calculate the expected gate fidelity limited by (a) finite blockade and (b) position fluctuations.

**Solution:**

**Step 1: Blockade error**
Interaction at 4 μm:
$$V = \frac{2100}{4^6} = \frac{2100}{4096} = 0.51\,\text{GHz} = 510\,\text{MHz}$$

Blockade error:
$$\epsilon_b = \left(\frac{\hbar\Omega}{V}\right)^2 = \left(\frac{3}{510}\right)^2 = 3.5 \times 10^{-5}$$

**Step 2: Position fluctuation error**
Position uncertainty:
$$\sigma_r = \sqrt{\frac{k_B T}{m\omega^2}} = \sqrt{\frac{1.38 \times 10^{-23} \times 20 \times 10^{-6}}{87 \times 1.66 \times 10^{-27} \times (2\pi \times 80000)^2}}$$
$$\sigma_r \approx 35\,\text{nm}$$

Relative interaction fluctuation:
$$\frac{\delta V}{V} = 6\frac{\sigma_r}{r} = 6 \times \frac{35\,\text{nm}}{4\,\mu\text{m}} = 5.3\%$$

This causes dephasing error:
$$\epsilon_p \approx \left(\frac{\delta V \cdot t_\pi}{\hbar}\right)^2$$

For $t_\pi \approx 100$ ns, $\delta V = 0.053 \times 510$ MHz = 27 MHz:
$$\epsilon_p \approx (27 \times 10^6 \times 100 \times 10^{-9})^2 = (2.7)^2 = 7.3$$

This exceeds 1, so the estimate breaks down. More carefully, the fidelity reduction is:
$$F \approx \exp(-(\delta V \cdot t/\hbar)^2/2) \approx 0.026$$

This shows position fluctuations are the dominant error source.

**Step 3: Mitigation**
Cooling to motional ground state ($\bar{n} = 0$) gives:
$$\sigma_r = \sqrt{\frac{\hbar}{2m\omega}} = 16\,\text{nm}$$

Then $\delta V/V = 2.4\%$, giving $\epsilon_p \approx 0.4$ — still significant but improved.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the blockade radius for Cs atoms in the 80S state ($C_6 = 5000$ GHz·μm⁶) with $\Omega = 2\pi \times 1$ MHz.

**Problem 1.2:** Three atoms within a blockade volume are driven with $\Omega = 2\pi \times 2$ MHz. What is the collective Rabi frequency, and how long is a π-pulse?

**Problem 1.3:** At what atom separation does the interaction equal 100 times the Rabi frequency for n=60 Rb atoms with $\Omega = 2\pi \times 5$ MHz?

### Level 2: Intermediate Analysis

**Problem 2.1:** Design a blockade gate for a 2D square lattice with 5 μm spacing. Choose an appropriate Rydberg state (n) and Rabi frequency to achieve:
a) Strong blockade for nearest neighbors (V > 20Ω)
b) Weak blockade for diagonals (V < 0.5Ω)

**Problem 2.2:** Analyze the blockade quality in a triangular lattice with 4 μm spacing for n=60 Rb atoms. Calculate the ratio V/Ω needed to achieve 99.9% blockade fidelity for all 6 nearest neighbors.

**Problem 2.3:** For the antiblockade condition $\delta = V/\hbar$, calculate the required detuning to achieve resonant double excitation at 6 μm for n=70 Rb.

### Level 3: Challenging Problems

**Problem 3.1:** Derive the collective Rabi frequency for N atoms in an arbitrary geometry where not all pairs are within the blockade radius. Consider a 1D chain where only nearest neighbors are blockaded.

**Problem 3.2:** Calculate the many-body spectrum for 4 atoms in a square geometry (spacing a) with Rydberg interactions. Identify the eigenstate energies and composition when $V_{NN} = 10\hbar\Omega$ and $V_{diag} = V_{NN}/2^6$.

**Problem 3.3:** Analyze the adiabatic preparation of a crystalline Rydberg phase by slowly ramping Ω while keeping the detuning fixed. Determine the conditions for adiabaticity in an 8-atom chain.

## Computational Lab: Blockade Dynamics

### Lab 1: Two-Atom Blockade Simulation

```python
"""
Day 913 Lab: Rydberg Blockade Dynamics
Simulating two-atom and many-atom blockaded systems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def two_atom_hamiltonian(Omega, V, delta=0):
    """
    Construct two-atom Hamiltonian in basis {|gg⟩, |gr⟩, |rg⟩, |rr⟩}.

    Parameters:
    -----------
    Omega : float
        Single-atom Rabi frequency
    V : float
        Interaction energy (same units as Omega)
    delta : float
        Detuning from Rydberg resonance
    """
    H = np.array([
        [0, Omega/2, Omega/2, 0],
        [Omega/2, -delta, 0, Omega/2],
        [Omega/2, 0, -delta, Omega/2],
        [0, Omega/2, Omega/2, -2*delta + V]
    ], dtype=complex)
    return H

def time_evolution(H, psi0, t_max, n_steps):
    """
    Compute time evolution under time-independent Hamiltonian.
    """
    t = np.linspace(0, t_max, n_steps)
    psi_t = np.zeros((n_steps, len(psi0)), dtype=complex)

    for i, ti in enumerate(t):
        U = expm(-1j * H * ti)
        psi_t[i] = U @ psi0

    return t, psi_t

# Parameters
Omega = 2 * np.pi * 1  # MHz (normalized)

# Different interaction strengths (in units of Omega)
V_values = [0.1, 1, 5, 20, 100]  # V/Omega ratios

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, V_ratio in enumerate(V_values):
    V = V_ratio * Omega

    H = two_atom_hamiltonian(Omega, V)

    # Initial state: |gg⟩
    psi0 = np.array([1, 0, 0, 0], dtype=complex)

    # Time evolution
    t_max = 4 * np.pi / Omega  # Multiple Rabi periods
    t, psi_t = time_evolution(H, psi0, t_max, 500)

    # Calculate populations
    P_gg = np.abs(psi_t[:, 0])**2
    P_gr = np.abs(psi_t[:, 1])**2
    P_rg = np.abs(psi_t[:, 2])**2
    P_rr = np.abs(psi_t[:, 3])**2
    P_single = P_gr + P_rg  # Single Rydberg excitation

    ax = axes.flatten()[idx]
    ax.plot(t * Omega / (2*np.pi), P_gg, 'b-', label='|gg⟩', linewidth=2)
    ax.plot(t * Omega / (2*np.pi), P_single, 'g-', label='|gr⟩+|rg⟩', linewidth=2)
    ax.plot(t * Omega / (2*np.pi), P_rr, 'r-', label='|rr⟩', linewidth=2)

    ax.set_xlabel('Time (1/Ω)')
    ax.set_ylabel('Population')
    ax.set_title(f'V/Ω = {V_ratio}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

# Collective enhancement visualization
ax = axes.flatten()[5]
# Perfect blockade: compare single atom vs two-atom collective
Omega_single = Omega
Omega_collective = np.sqrt(2) * Omega

t = np.linspace(0, 4 * np.pi / Omega_single, 200)
P_single_atom = np.sin(Omega_single * t / 2)**2
P_collective = np.sin(Omega_collective * t / 2)**2

ax.plot(t * Omega / (2*np.pi), P_single_atom, 'b--', label='Single atom', linewidth=2)
ax.plot(t * Omega / (2*np.pi), P_collective, 'g-', label='Collective (√2 Ω)', linewidth=2)
ax.set_xlabel('Time (1/Ω)')
ax.set_ylabel('Excitation probability')
ax.set_title('Collective Enhancement')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_atom_blockade.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze blockade fidelity
print("=== Blockade Fidelity Analysis ===\n")

V_ratios = np.logspace(-1, 3, 50)
max_P_rr = []

for V_ratio in V_ratios:
    V = V_ratio * Omega
    H = two_atom_hamiltonian(Omega, V)
    psi0 = np.array([1, 0, 0, 0], dtype=complex)

    t_max = 2 * np.pi / Omega
    t, psi_t = time_evolution(H, psi0, t_max, 200)

    P_rr = np.abs(psi_t[:, 3])**2
    max_P_rr.append(np.max(P_rr))

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(V_ratios, max_P_rr, 'b-', linewidth=2)
ax.loglog(V_ratios, (1/V_ratios)**2, 'r--', label='(Ω/V)² scaling')
ax.axhline(y=0.01, color='gray', linestyle=':', label='1% error threshold')

ax.set_xlabel('V/Ω (interaction/Rabi ratio)')
ax.set_ylabel('Maximum P(|rr⟩)')
ax.set_title('Blockade Leakage vs Interaction Strength')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('blockade_fidelity.png', dpi=150, bbox_inches='tight')
plt.show()

# Find threshold for 1% error
threshold_idx = np.where(np.array(max_P_rr) < 0.01)[0][0]
print(f"V/Ω > {V_ratios[threshold_idx]:.1f} needed for <1% double excitation")
```

### Lab 2: Blockade Radius and Spatial Dependence

```python
"""
Lab 2: Spatial dependence of blockade
"""

import numpy as np
import matplotlib.pyplot as plt

def blockade_radius(C6, Omega):
    """
    Calculate blockade radius.

    Parameters:
    -----------
    C6 : float
        C6 coefficient (GHz * um^6)
    Omega : float
        Rabi frequency (MHz)

    Returns:
    --------
    r_b : float
        Blockade radius (um)
    """
    # Convert to consistent units (GHz)
    Omega_GHz = Omega / 1000
    return (C6 / Omega_GHz)**(1/6)

def interaction_energy(r, C6):
    """Van der Waals interaction in GHz."""
    return C6 / r**6

# C6 values for different Rydberg states (GHz * um^6)
n_values = [50, 60, 70, 80, 100]
C6_values = {50: 140, 60: 600, 70: 2100, 80: 6500, 100: 40000}

# Typical Rabi frequencies (MHz)
Omega_values = [0.5, 1, 2, 5, 10]

print("=== Blockade Radius Table ===")
print(f"{'n':>4} | " + " | ".join([f"Ω={O} MHz" for O in Omega_values]))
print("-" * 60)

for n in n_values:
    C6 = C6_values[n]
    r_b_list = [blockade_radius(C6, O) for O in Omega_values]
    print(f"{n:4d} | " + " | ".join([f"{r_b:7.1f} μm" for r_b in r_b_list]))

# Visualize blockade region
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Interaction vs distance for different n
r = np.linspace(2, 15, 200)

for n in n_values:
    C6 = C6_values[n]
    V = interaction_energy(r, C6)
    axes[0].semilogy(r, V * 1000, label=f'n = {n}')  # Convert to MHz

# Add Rabi frequency lines
for Omega in [1, 5, 10]:
    axes[0].axhline(y=Omega, color='gray', linestyle='--', alpha=0.5)
    axes[0].text(14, Omega, f'Ω = {Omega} MHz', va='center')

axes[0].set_xlabel('Interatomic distance (μm)')
axes[0].set_ylabel('Interaction energy (MHz)')
axes[0].set_title('Rydberg-Rydberg Interaction')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(1e-1, 1e5)

# Blockade radius vs n for fixed Omega
Omega_fixed = 2  # MHz

n_range = np.arange(40, 110)
r_b_vs_n = []

for n in n_range:
    # C6 scaling: C6 ~ n^11
    C6 = 2.2e-7 * (n - 3.13)**11  # Approximate for Rb S states
    r_b_vs_n.append(blockade_radius(C6, Omega_fixed))

axes[1].plot(n_range, r_b_vs_n, 'b-', linewidth=2)
axes[1].fill_between(n_range, r_b_vs_n, alpha=0.3)

# Power law fit
log_n = np.log(n_range - 3.13)
log_rb = np.log(r_b_vs_n)
slope = np.polyfit(log_n, log_rb, 1)[0]
axes[1].set_xlabel('Principal quantum number n')
axes[1].set_ylabel('Blockade radius (μm)')
axes[1].set_title(f'Blockade Radius vs n (Ω = {Omega_fixed} MHz)\nScaling: r_b ~ n^{slope:.2f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('blockade_spatial.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBlockade radius scaling: r_b ∝ n^{slope:.2f}")
print("Expected: r_b ∝ (C6)^(1/6) ∝ n^(11/6) = n^1.83")
```

### Lab 3: Many-Body Blockade Dynamics

```python
"""
Lab 3: Many-body blockade in atomic arrays
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from itertools import product

def generate_basis(N):
    """
    Generate basis states for N two-level atoms.
    |0⟩ = ground, |1⟩ = Rydberg

    Returns list of tuples, e.g., (0,0,0) for |ggg⟩
    """
    return list(product([0, 1], repeat=N))

def many_body_hamiltonian(N, Omega, positions, C6, delta=0):
    """
    Construct many-body Hamiltonian for N atoms.

    Parameters:
    -----------
    N : int
        Number of atoms
    Omega : float
        Single-atom Rabi frequency
    positions : array (N, 2) or (N, 3)
        Atom positions
    C6 : float
        C6 coefficient
    delta : float
        Detuning
    """
    basis = generate_basis(N)
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    for i, state_i in enumerate(basis):
        # Diagonal: interaction + detuning
        n_rydberg = sum(state_i)
        H[i, i] = -delta * n_rydberg

        # Add interactions
        for a in range(N):
            for b in range(a+1, N):
                if state_i[a] == 1 and state_i[b] == 1:
                    r = np.linalg.norm(positions[a] - positions[b])
                    V = C6 / r**6
                    H[i, i] += V

        # Off-diagonal: Rabi coupling
        for j, state_j in enumerate(basis):
            diff = np.array(state_i) - np.array(state_j)
            if np.sum(np.abs(diff)) == 1:  # States differ by one excitation
                H[i, j] = Omega / 2

    return H, basis

def count_rydberg(state):
    """Count number of Rydberg excitations in state."""
    return sum(state)

# Simulate 1D chain
N = 4
spacing = 5.0  # μm
positions = np.array([[i * spacing, 0] for i in range(N)])

C6 = 600  # GHz * μm^6 (n ≈ 60)
Omega = 0.01  # GHz = 10 MHz

print(f"=== {N}-Atom Chain Simulation ===")
print(f"Spacing: {spacing} μm")
print(f"NN interaction: {C6/spacing**6:.2f} GHz = {C6/spacing**6*1000:.0f} MHz")
print(f"Rabi frequency: {Omega*1000:.0f} MHz")
print(f"V/Ω ratio: {C6/spacing**6/Omega:.1f}")

H, basis = many_body_hamiltonian(N, Omega, positions, C6)

# Diagonalize
eigenvalues, eigenvectors = np.linalg.eigh(H)

print(f"\nEnergy spectrum (GHz):")
for i, E in enumerate(eigenvalues[:8]):
    # Find dominant basis state
    probs = np.abs(eigenvectors[:, i])**2
    dominant = np.argmax(probs)
    print(f"  E_{i} = {E:.4f} GHz, dominant: {basis[dominant]}, prob = {probs[dominant]:.2f}")

# Time evolution
psi0 = np.zeros(len(basis), dtype=complex)
ground_idx = basis.index(tuple([0]*N))
psi0[ground_idx] = 1

t_max = 4 * np.pi / (np.sqrt(N) * Omega)  # Multiple collective periods
n_steps = 500
t = np.linspace(0, t_max, n_steps)

psi_t = np.zeros((n_steps, len(basis)), dtype=complex)
for i, ti in enumerate(t):
    U = expm(-1j * H * ti)
    psi_t[i] = U @ psi0

# Calculate populations by excitation number
P_n = np.zeros((n_steps, N+1))
for i, ti in enumerate(t):
    for j, state in enumerate(basis):
        n_exc = count_rydberg(state)
        P_n[i, n_exc] += np.abs(psi_t[i, j])**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Population dynamics
for n_exc in range(N+1):
    axes[0].plot(t * np.sqrt(N) * Omega / (2*np.pi), P_n[:, n_exc],
                label=f'{n_exc} excitations', linewidth=2)

axes[0].set_xlabel('Time (1/Ω_coll)')
axes[0].set_ylabel('Population')
axes[0].set_title(f'{N}-Atom Chain: Population by Excitation Number')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compare different chain lengths
N_values = [2, 3, 4, 5, 6]
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))

for N_test, color in zip(N_values, colors):
    positions = np.array([[i * spacing, 0] for i in range(N_test)])
    H, basis = many_body_hamiltonian(N_test, Omega, positions, C6)

    psi0 = np.zeros(len(basis), dtype=complex)
    psi0[basis.index(tuple([0]*N_test))] = 1

    t_max = 2 * np.pi / (np.sqrt(N_test) * Omega)
    t = np.linspace(0, t_max, 200)

    P_single = np.zeros(len(t))
    for i, ti in enumerate(t):
        U = expm(-1j * H * ti)
        psi = U @ psi0
        for j, state in enumerate(basis):
            if count_rydberg(state) == 1:
                P_single[i] += np.abs(psi[j])**2

    axes[1].plot(t * Omega / (2*np.pi), P_single, color=color,
                label=f'N = {N_test}', linewidth=2)

axes[1].set_xlabel('Time (1/Ω)')
axes[1].set_ylabel('Single-excitation probability')
axes[1].set_title('Collective Enhancement: √N Scaling')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('many_body_blockade.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify collective scaling
print("\n=== Collective Enhancement Verification ===")
for N_test in N_values:
    positions = np.array([[i * spacing, 0] for i in range(N_test)])
    H, basis = many_body_hamiltonian(N_test, Omega, positions, C6)

    psi0 = np.zeros(len(basis), dtype=complex)
    psi0[basis.index(tuple([0]*N_test))] = 1

    # Find time to first maximum
    t_search = np.linspace(0, np.pi / (np.sqrt(N_test) * Omega), 100)
    P_single_max = 0
    t_max_found = 0

    for ti in t_search:
        U = expm(-1j * H * ti)
        psi = U @ psi0
        P_single = sum(np.abs(psi[j])**2 for j, state in enumerate(basis) if count_rydberg(state) == 1)
        if P_single > P_single_max:
            P_single_max = P_single
            t_max_found = ti

    Omega_eff = np.pi / (2 * t_max_found)
    print(f"N = {N_test}: Ω_eff/Ω = {Omega_eff/Omega:.2f} (expected √{N_test} = {np.sqrt(N_test):.2f})")
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| Blockade radius | $r_b = (C_6/\hbar\Omega)^{1/6}$ | 5-15 μm |
| Blockade condition | $V \gg \hbar\Omega$ | V > 10Ω |
| Collective Rabi freq | $\Omega_N = \sqrt{N}\Omega$ | Enhanced by √N |
| Blockade error | $\epsilon \approx (\hbar\Omega/V)^2$ | <1% for V>10Ω |
| Symmetric state | $|W\rangle = \frac{1}{\sqrt{N}}\sum_k|r_k\rangle$ | Entangled |

### Main Takeaways

1. **Rydberg blockade** prevents double excitation when interaction energy exceeds Rabi coupling, creating a mesoscopic exclusion region around each Rydberg atom.

2. **Collective enhancement** speeds up dynamics by factor $\sqrt{N}$ for $N$ atoms within the blockade radius, enabling faster gates with larger atom numbers.

3. **Blockade fidelity** improves as $V/\Omega$ increases, with error scaling as $(Ω/V)^2$; typical experiments achieve V/Ω > 10 for sub-percent errors.

4. **Position fluctuations** cause interaction noise that limits gate fidelity; cooling to the motional ground state is essential for high-fidelity operations.

5. **Geometric considerations** matter: square lattice diagonals have 1/64 the NN interaction, enabling selective blockade in different directions.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain the physical origin of blockade
- [ ] I understand collective Rabi enhancement
- [ ] I can describe blockade limitations and error sources
- [ ] I know how geometry affects blockade

### Mathematical Skills
- [ ] I can calculate blockade radius from C₆ and Ω
- [ ] I can estimate blockade errors
- [ ] I can write the two-atom Hamiltonian

### Computational Skills
- [ ] I can simulate two-atom blockade dynamics
- [ ] I can verify √N collective enhancement
- [ ] I can analyze many-body spectra

## Preview: Day 914

Tomorrow we explore **Single-Qubit Gates** in neutral atom systems, including:
- Microwave transitions between hyperfine states
- Two-photon Raman transitions for fast gates
- Global vs local addressing strategies
- Gate fidelity optimization

These gates operate on the qubit states while the atoms remain in their ground state manifold, complementing the Rydberg-based entangling operations.
