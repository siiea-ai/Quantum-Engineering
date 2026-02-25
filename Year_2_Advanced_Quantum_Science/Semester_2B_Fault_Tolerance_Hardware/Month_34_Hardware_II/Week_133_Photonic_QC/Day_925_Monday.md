# Day 925: Linear Optical Quantum Computing Fundamentals

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Beam splitters, phase shifters, Hong-Ou-Mandel effect |
| Afternoon | 2.5 hours | Problem solving: Linear optical transformations |
| Evening | 1.5 hours | Computational lab: Fock state simulations |

## Learning Objectives

By the end of today, you will be able to:

1. Derive the quantum mechanical beam splitter transformation in Fock basis
2. Calculate the output state for arbitrary input states through linear optical networks
3. Explain the Hong-Ou-Mandel effect and its role in photonic quantum computing
4. Implement dual-rail qubit encoding and single-qubit gates
5. Analyze photon detection schemes and their quantum measurement properties
6. Simulate linear optical circuits with multi-photon states

## Core Content

### 1. Photons as Qubits: Encoding Schemes

In photonic quantum computing, quantum information can be encoded in several ways:

**Dual-Rail Encoding:**
$$|0_L\rangle = |1,0\rangle = \hat{a}_1^\dagger|0,0\rangle$$
$$|1_L\rangle = |0,1\rangle = \hat{a}_2^\dagger|0,0\rangle$$

A single photon in one of two spatial modes encodes a qubit. The general state:
$$|\psi\rangle = \alpha|1,0\rangle + \beta|0,1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

**Polarization Encoding:**
$$|0_L\rangle = |H\rangle, \quad |1_L\rangle = |V\rangle$$

These are equivalent via:
$$|H\rangle = \hat{a}_H^\dagger|0\rangle, \quad |V\rangle = \hat{a}_V^\dagger|0\rangle$$

**Time-Bin Encoding:**
$$|0_L\rangle = |early\rangle, \quad |1_L\rangle = |late\rangle$$

### 2. The Beam Splitter

The beam splitter is the fundamental building block of linear optical quantum computing.

**Classical Description:**
A beam splitter with reflectivity $R$ and transmissivity $T = 1-R$ relates input and output fields:
$$\begin{pmatrix} E_3 \\ E_4 \end{pmatrix} = \begin{pmatrix} t & r \\ r' & t' \end{pmatrix} \begin{pmatrix} E_1 \\ E_2 \end{pmatrix}$$

**Quantum Transformation:**
For mode operators $\hat{a}$ and $\hat{b}$, a 50:50 beam splitter:
$$\hat{a}_{out} = \frac{1}{\sqrt{2}}(\hat{a}_{in} + i\hat{b}_{in})$$
$$\hat{b}_{out} = \frac{1}{\sqrt{2}}(i\hat{a}_{in} + \hat{b}_{in})$$

**General Beam Splitter Unitary:**
$$\hat{U}_{BS}(\theta, \phi) = \exp\left[\theta(e^{i\phi}\hat{a}^\dagger\hat{b} - e^{-i\phi}\hat{a}\hat{b}^\dagger)\right]$$

The transformation matrix:
$$U_{BS}(\theta, \phi) = \begin{pmatrix} \cos\theta & e^{i\phi}\sin\theta \\ -e^{-i\phi}\sin\theta & \cos\theta \end{pmatrix}$$

For a 50:50 beam splitter ($\theta = \pi/4$, $\phi = 0$):
$$U_{BS} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

**Action on Fock States:**
For input $|1,0\rangle$:
$$\hat{U}_{BS}|1,0\rangle = \frac{1}{\sqrt{2}}(|1,0\rangle + i|0,1\rangle)$$

For input $|1,1\rangle$ (one photon in each mode):
$$\hat{U}_{BS}|1,1\rangle = \frac{i}{\sqrt{2}}(|2,0\rangle + |0,2\rangle)$$

This is the famous Hong-Ou-Mandel (HOM) effect: no $|1,1\rangle$ component in the output!

### 3. The Hong-Ou-Mandel Effect

**Physical Explanation:**
When two indistinguishable photons enter a 50:50 beam splitter from different input ports, they always exit together from the same output port.

**Derivation:**
Input state: $|1,1\rangle = \hat{a}^\dagger\hat{b}^\dagger|0,0\rangle$

Under beam splitter transformation:
$$\hat{a}^\dagger \to \frac{1}{\sqrt{2}}(\hat{c}^\dagger + i\hat{d}^\dagger)$$
$$\hat{b}^\dagger \to \frac{1}{\sqrt{2}}(i\hat{c}^\dagger + \hat{d}^\dagger)$$

Therefore:
$$\hat{a}^\dagger\hat{b}^\dagger|0,0\rangle \to \frac{1}{2}(\hat{c}^\dagger + i\hat{d}^\dagger)(i\hat{c}^\dagger + \hat{d}^\dagger)|0,0\rangle$$

Expanding:
$$= \frac{1}{2}(i\hat{c}^{\dagger 2} + \hat{c}^\dagger\hat{d}^\dagger - \hat{c}^\dagger\hat{d}^\dagger + i\hat{d}^{\dagger 2})|0,0\rangle$$

$$= \frac{i}{2}(\hat{c}^{\dagger 2} + \hat{d}^{\dagger 2})|0,0\rangle = \frac{i}{\sqrt{2}}(|2,0\rangle + |0,2\rangle)$$

$$\boxed{|1,1\rangle \xrightarrow{BS} \frac{i}{\sqrt{2}}(|2,0\rangle + |0,2\rangle)}$$

**HOM Dip:**
The probability of coincidence detection (one photon in each output):
$$P_{coincidence} = 0$$

This is the signature of quantum interference between indistinguishable photons.

### 4. Phase Shifters

A phase shifter applies a phase to a single mode:
$$\hat{U}_\phi = e^{i\phi \hat{n}} = e^{i\phi \hat{a}^\dagger\hat{a}}$$

Action on Fock states:
$$\hat{U}_\phi|n\rangle = e^{in\phi}|n\rangle$$

Combined with beam splitters, phase shifters enable universal single-mode unitaries.

### 5. Single-Qubit Gates in Dual-Rail Encoding

In dual-rail encoding, single-qubit gates are implemented with beam splitters and phase shifters.

**Hadamard Gate:**
A 50:50 beam splitter acts as a Hadamard on the dual-rail qubit:
$$H|0_L\rangle = H|1,0\rangle = \frac{1}{\sqrt{2}}(|1,0\rangle + |0,1\rangle) = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle)$$

**Phase Gate:**
A phase shifter on mode 2:
$$S|0_L\rangle = |1,0\rangle = |0_L\rangle$$
$$S|1_L\rangle = e^{i\pi/2}|0,1\rangle = i|1_L\rangle$$

**Arbitrary Single-Qubit Rotation:**
Any SU(2) operation can be decomposed as:
$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

Implemented with phase shifters and beam splitters in sequence.

### 6. Photon Detection

**Single-Photon Avalanche Diodes (SPADs):**
- Binary detection: "click" or "no click"
- Cannot resolve photon number for $n > 1$
- Detection efficiency $\eta$: typically 20-90%
- Dark count rate: spurious detection events

**Photon Number Resolving Detectors:**
- Transition Edge Sensors (TES): resolve photon number up to ~20
- Superconducting Nanowire SPDs (SNSPDs): high efficiency, fast timing
- Essential for KLM protocol and boson sampling verification

**POVM Description:**
For a SPAD with efficiency $\eta$:
$$\hat{\Pi}_0 = \sum_{n=0}^\infty (1-\eta)^n |n\rangle\langle n|$$ (no click)
$$\hat{\Pi}_1 = \hat{I} - \hat{\Pi}_0$$ (click)

### 7. Multi-Mode Interferometers

An $N$-mode linear optical network is described by an $N \times N$ unitary matrix $U$.

**Reck Decomposition:**
Any $N \times N$ unitary can be decomposed into $N(N-1)/2$ beam splitters and $N$ phase shifters.

For $N = 4$:
$$U_{4\times 4} = D \cdot T_{34} \cdot T_{24} \cdot T_{14} \cdot T_{23} \cdot T_{13} \cdot T_{12}$$

where $D$ is a diagonal phase matrix and $T_{ij}$ are beam splitter transformations.

**Clements Decomposition (2016):**
More practical for integrated photonics: rectangular mesh of beam splitters with alternating columns.

## Quantum Computing Applications

### Photonic Quantum Advantage

Linear optical systems have demonstrated "quantum advantage" through boson sampling (Day 927), but universal quantum computing requires additional resources.

**Advantages of Photonic Qubits:**
1. Room-temperature operation
2. Long coherence times (photons don't interact with environment)
3. Natural connectivity via optical fibers
4. High clock speeds (~GHz)

**Challenges:**
1. Photon loss (main error source)
2. Deterministic two-qubit gates require nonlinearity or measurement
3. Scalable single-photon sources
4. Photon detection efficiency

### Quantum Communication Interface

Photons are the natural carriers for quantum communication:
$$|\psi\rangle_{matter} \xrightarrow{transduction} |\psi\rangle_{photon} \xrightarrow{fiber} |\psi\rangle_{remote}$$

## Worked Examples

### Example 1: Three-Mode Interferometer

**Problem:** Calculate the output state when a single photon enters the first port of a symmetric 3-mode interferometer (tritter) with unitary:
$$U = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 & 1 \\ 1 & \omega & \omega^2 \\ 1 & \omega^2 & \omega \end{pmatrix}$$
where $\omega = e^{2\pi i/3}$.

**Solution:**
Input state: $|1,0,0\rangle = \hat{a}_1^\dagger|0,0,0\rangle$

The mode transformation:
$$\hat{a}_1^\dagger \to \frac{1}{\sqrt{3}}(\hat{b}_1^\dagger + \hat{b}_2^\dagger + \hat{b}_3^\dagger)$$

Output state:
$$|\psi_{out}\rangle = \frac{1}{\sqrt{3}}(|1,0,0\rangle + |0,1,0\rangle + |0,0,1\rangle)$$

This is an equal superposition across all three output modes - a photonic qutrit!

### Example 2: HOM with Distinguishable Photons

**Problem:** Two photons with partial overlap $\xi$ (where $\xi = 1$ means identical, $\xi = 0$ means completely distinguishable) enter a 50:50 beam splitter. Calculate the coincidence probability.

**Solution:**
For partial distinguishability, we model the two-photon state as:
$$|\psi\rangle = \sqrt{\xi}|\psi_{identical}\rangle + \sqrt{1-\xi}|\psi_{distinguishable}\rangle$$

For identical photons: $P_{coinc}^{id} = 0$ (HOM effect)
For distinguishable photons: $P_{coinc}^{dist} = 1/2$ (classical)

The coincidence probability:
$$P_{coinc} = \xi \cdot 0 + (1-\xi) \cdot \frac{1}{2} = \frac{1-\xi}{2}$$

$$\boxed{P_{coincidence} = \frac{1 - \xi}{2}}$$

When $\xi = 1$: $P = 0$ (perfect HOM dip)
When $\xi = 0$: $P = 1/2$ (classical behavior)

### Example 3: Dual-Rail Qubit Rotation

**Problem:** Implement a $R_y(\pi/4)$ rotation on a dual-rail qubit using beam splitters.

**Solution:**
The $R_y(\theta)$ gate:
$$R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

For $\theta = \pi/4$:
$$R_y(\pi/4) = \begin{pmatrix} \cos(\pi/8) & -\sin(\pi/8) \\ \sin(\pi/8) & \cos(\pi/8) \end{pmatrix}$$

This matches a beam splitter with $\theta_{BS} = \pi/8$ and appropriate phases:
$$U_{BS}(\pi/8, \pi) = \begin{pmatrix} \cos(\pi/8) & -\sin(\pi/8) \\ \sin(\pi/8) & \cos(\pi/8) \end{pmatrix}$$

The beam splitter reflectivity: $R = \sin^2(\pi/8) \approx 0.146$ (14.6% reflective).

## Practice Problems

### Level 1: Direct Application

1. **Beam Splitter Transformation**

   A 70:30 beam splitter has $|t|^2 = 0.7$ and $|r|^2 = 0.3$. If the input state is $|1,0\rangle$:
   a) Write the output state
   b) What are the probabilities of detecting the photon in each output port?

2. **Phase Shifter Action**

   Apply a phase shifter with $\phi = \pi/3$ to the state $|3\rangle$. What is the resulting state?

3. **Dual-Rail Encoding**

   Express the state $|\psi\rangle = (2|0_L\rangle + i|1_L\rangle)/\sqrt{5}$ in the Fock basis.

### Level 2: Intermediate

4. **HOM Visibility**

   In a HOM experiment, the measured coincidence rate drops from 1000 counts/s (with large time delay) to 50 counts/s (at zero delay). Calculate:
   a) The HOM visibility $V = (C_{max} - C_{min})/C_{max}$
   b) The photon indistinguishability $\xi$

5. **Mach-Zehnder Interferometer**

   A Mach-Zehnder interferometer consists of two 50:50 beam splitters with a phase $\phi$ in one arm. Calculate the output probabilities for input $|1,0\rangle$ as a function of $\phi$.

6. **Two-Photon State**

   The state $|2,0\rangle$ enters a 50:50 beam splitter. Calculate the output state and the probability of each measurement outcome.

### Level 3: Challenging

7. **N-Photon Bunching**

   Generalize the HOM effect: show that $N$ indistinguishable photons entering an $N$-port symmetric interferometer all exit from the same port with enhanced probability. Calculate this probability for $N = 3$.

8. **Arbitrary SU(2) Implementation**

   Design a linear optical circuit (beam splitters and phase shifters) to implement the gate:
   $$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & e^{i\pi/4} \\ e^{i\pi/4} & -1 \end{pmatrix}$$
   Specify all beam splitter angles and phase shifts.

9. **Photon Loss Channel**

   Model photon loss as a beam splitter with a vacuum mode. If the loss rate is $\gamma$ per unit length and the fiber length is $L$:
   a) Express the transmission $\eta(L)$
   b) For input $|\alpha\rangle$ (coherent state), show the output is $|\sqrt{\eta}\alpha\rangle$
   c) For input $|1\rangle$ (single photon), calculate the output density matrix

## Computational Lab: Fock State Simulations

```python
"""
Day 925 Computational Lab: Linear Optical Quantum Computing
Simulating Fock states, beam splitters, and the Hong-Ou-Mandel effect
"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product

# Set up Fock space dimension
N_max = 10  # Maximum photon number per mode

def fock_state(n, N=N_max):
    """Create a Fock state |n⟩ in a truncated Hilbert space."""
    state = np.zeros(N, dtype=complex)
    if n < N:
        state[n] = 1.0
    return state

def creation_operator(N=N_max):
    """Create the creation operator a† in Fock basis."""
    a_dag = np.zeros((N, N), dtype=complex)
    for n in range(N-1):
        a_dag[n+1, n] = np.sqrt(n + 1)
    return a_dag

def annihilation_operator(N=N_max):
    """Create the annihilation operator a in Fock basis."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

def number_operator(N=N_max):
    """Create the number operator n = a†a."""
    a = annihilation_operator(N)
    a_dag = creation_operator(N)
    return a_dag @ a

# Two-mode Fock states
def two_mode_fock(n1, n2, N=N_max):
    """Create two-mode Fock state |n1, n2⟩."""
    state1 = fock_state(n1, N)
    state2 = fock_state(n2, N)
    return np.outer(state1, state2).flatten()

def two_mode_basis(N=N_max):
    """Return basis labels for two-mode system."""
    return [(n1, n2) for n1 in range(N) for n2 in range(N)]

# Beam splitter implementation
def beam_splitter_matrix(theta, phi=0):
    """
    2x2 beam splitter transformation matrix.
    theta: mixing angle (θ=π/4 for 50:50)
    phi: phase shift
    """
    return np.array([
        [np.cos(theta), np.exp(1j*phi) * np.sin(theta)],
        [-np.exp(-1j*phi) * np.sin(theta), np.cos(theta)]
    ])

def beam_splitter_fock(theta, phi=0, N=N_max):
    """
    Beam splitter unitary in two-mode Fock basis.
    Uses the transformation of creation operators.
    """
    # Dimension of two-mode Hilbert space
    dim = N * N
    U = np.zeros((dim, dim), dtype=complex)

    # Beam splitter transformation
    bs = beam_splitter_matrix(theta, phi)
    t, r = bs[0, 0], bs[0, 1]

    # For each input Fock state |n1, n2⟩
    for n1 in range(N):
        for n2 in range(N):
            idx_in = n1 * N + n2

            # Output is superposition based on binomial distribution
            for k1 in range(n1 + n2 + 1):
                for k2 in range(n1 + n2 + 1):
                    if k1 + k2 != n1 + n2:  # Photon number conservation
                        continue
                    if k1 >= N or k2 >= N:
                        continue

                    idx_out = k1 * N + k2

                    # Calculate amplitude using permanent formula
                    amp = 0
                    for j1 in range(min(n1, k1) + 1):
                        j2 = k1 - j1
                        if j2 < 0 or j2 > n2:
                            continue

                        coeff = (np.math.comb(n1, j1) * np.math.comb(n2, j2) *
                                t**(j1 + (n2-j2)) *
                                r**(n1-j1) *
                                (-np.conj(r))**(j2))

                        amp += coeff * np.sqrt(factorial(k1) * factorial(k2) /
                                               (factorial(n1) * factorial(n2)))

                    U[idx_out, idx_in] = amp

    return U

def hong_ou_mandel_simulation():
    """
    Simulate the Hong-Ou-Mandel effect.
    """
    print("=" * 60)
    print("Hong-Ou-Mandel Effect Simulation")
    print("=" * 60)

    N = 5  # Smaller Fock space for HOM

    # Input: |1,1⟩
    psi_in = two_mode_fock(1, 1, N)

    # 50:50 beam splitter
    U_bs = beam_splitter_fock(np.pi/4, np.pi/2, N)  # Note phase for standard convention

    # Output state
    psi_out = U_bs @ psi_in

    print("\nInput state: |1,1⟩")
    print("\nOutput state amplitudes:")

    basis = two_mode_basis(N)
    for i, (n1, n2) in enumerate(basis):
        if np.abs(psi_out[i]) > 1e-10:
            print(f"  |{n1},{n2}⟩: {psi_out[i]:.4f}")

    # Coincidence probability (|1,1⟩ component)
    idx_11 = 1 * N + 1
    p_coinc = np.abs(psi_out[idx_11])**2
    print(f"\nCoincidence probability P(1,1): {p_coinc:.6f}")
    print("Expected: 0 (perfect HOM dip)")

    return psi_out

def hom_dip_scan():
    """
    Simulate HOM dip as function of time delay (distinguishability).
    """
    print("\n" + "=" * 60)
    print("HOM Dip Scan Simulation")
    print("=" * 60)

    # Model distinguishability via overlap parameter
    xi_values = np.linspace(0, 1, 50)
    p_coinc = []

    for xi in xi_values:
        # P_coinc = (1-xi)/2 for partial distinguishability
        p_coinc.append((1 - xi) / 2)

    plt.figure(figsize=(10, 6))
    plt.plot(xi_values, p_coinc, 'b-', linewidth=2)
    plt.xlabel('Indistinguishability ξ', fontsize=12)
    plt.ylabel('Coincidence Probability', fontsize=12)
    plt.title('Hong-Ou-Mandel Dip', fontsize=14)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Classical limit')
    plt.axhline(y=0, color='g', linestyle='--', label='Perfect HOM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(-0.05, 0.55)
    plt.savefig('hom_dip.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: hom_dip.png")

def mach_zehnder_interferometer():
    """
    Simulate a Mach-Zehnder interferometer with single photon.
    """
    print("\n" + "=" * 60)
    print("Mach-Zehnder Interferometer Simulation")
    print("=" * 60)

    N = 3  # Small Fock space sufficient

    # Scan phase
    phases = np.linspace(0, 2*np.pi, 100)
    p_port1 = []
    p_port2 = []

    for phi in phases:
        # Input: |1,0⟩
        psi = two_mode_fock(1, 0, N)

        # First 50:50 beam splitter
        U_bs1 = beam_splitter_fock(np.pi/4, 0, N)
        psi = U_bs1 @ psi

        # Phase shift in first mode
        phase_matrix = np.diag([np.exp(1j * phi * n) for n in range(N)])
        U_phase = np.kron(phase_matrix, np.eye(N))
        psi = U_phase @ psi

        # Second 50:50 beam splitter
        psi = U_bs1 @ psi

        # Detection probabilities
        p1 = np.abs(psi[1*N + 0])**2  # |1,0⟩
        p2 = np.abs(psi[0*N + 1])**2  # |0,1⟩
        p_port1.append(p1)
        p_port2.append(p2)

    plt.figure(figsize=(10, 6))
    plt.plot(phases/np.pi, p_port1, 'b-', linewidth=2, label='Port 1')
    plt.plot(phases/np.pi, p_port2, 'r-', linewidth=2, label='Port 2')
    plt.xlabel('Phase φ/π', fontsize=12)
    plt.ylabel('Detection Probability', fontsize=12)
    plt.title('Mach-Zehnder Interferometer: Single Photon', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(-0.05, 1.05)
    plt.savefig('mz_interferometer.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: mz_interferometer.png")

def photon_bunching_statistics():
    """
    Demonstrate photon bunching for N-photon Fock states.
    """
    print("\n" + "=" * 60)
    print("Photon Bunching Statistics")
    print("=" * 60)

    N_max_sim = 6

    for n_photons in [1, 2, 3]:
        print(f"\n--- {n_photons} photon(s) through 50:50 BS ---")

        N = n_photons + 3  # Truncation
        psi_in = two_mode_fock(n_photons, 0, N)

        U_bs = beam_splitter_fock(np.pi/4, 0, N)
        psi_out = U_bs @ psi_in

        print("Output distribution:")
        basis = two_mode_basis(N)
        for i, (n1, n2) in enumerate(basis):
            prob = np.abs(psi_out[i])**2
            if prob > 1e-10:
                print(f"  |{n1},{n2}⟩: P = {prob:.4f}")

def main():
    """Run all simulations."""
    print("\n" + "=" * 60)
    print("DAY 925: LINEAR OPTICAL QUANTUM COMPUTING")
    print("=" * 60)

    # Test basic Fock states and operators
    print("\n--- Testing Fock State Operators ---")
    a = annihilation_operator(5)
    a_dag = creation_operator(5)

    # Verify [a, a†] = 1
    commutator = a @ a_dag - a_dag @ a
    print(f"[a, a†] = I? {np.allclose(commutator, np.eye(5))}")

    # Run simulations
    hong_ou_mandel_simulation()
    hom_dip_scan()
    mach_zehnder_interferometer()
    photon_bunching_statistics()

    print("\n" + "=" * 60)
    print("Simulations Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Beam splitter unitary | $U_{BS} = \exp[\theta(e^{i\phi}a^\dagger b - e^{-i\phi}ab^\dagger)]$ |
| 50:50 BS matrix | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$ |
| Phase shifter | $U_\phi = e^{i\phi \hat{n}}$ |
| Dual-rail encoding | $\|0_L\rangle = \|1,0\rangle$, $\|1_L\rangle = \|0,1\rangle$ |
| HOM effect | $\|1,1\rangle \to \frac{i}{\sqrt{2}}(\|2,0\rangle + \|0,2\rangle)$ |
| Coincidence probability | $P_{coinc} = \frac{1-\xi}{2}$ |

### Key Takeaways

1. **Beam splitters and phase shifters** form a universal set for single-mode unitaries
2. The **Hong-Ou-Mandel effect** demonstrates quantum interference of indistinguishable photons
3. **Dual-rail encoding** maps qubit operations to linear optical transformations
4. Any $N \times N$ unitary can be decomposed into $O(N^2)$ beam splitters
5. **Photon detection** projects onto Fock states, enabling measurement-based protocols
6. Linear optics alone cannot generate entanglement deterministically (requires measurement)

## Daily Checklist

- [ ] I can derive the beam splitter transformation in Fock basis
- [ ] I understand why HOM gives zero coincidence probability
- [ ] I can implement single-qubit gates in dual-rail encoding
- [ ] I understand the role of photon detection in linear optical QC
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 926

Tomorrow we explore the **KLM (Knill-Laflamme-Milburn) Protocol**, which shows how to achieve universal quantum computation using only linear optics, single-photon sources, and photon detection. Key topics:
- Non-deterministic CNOT gate using measurement
- Gate teleportation to boost success probability
- Resource overhead for probabilistic gates
- The role of ancilla photons and conditional measurement
