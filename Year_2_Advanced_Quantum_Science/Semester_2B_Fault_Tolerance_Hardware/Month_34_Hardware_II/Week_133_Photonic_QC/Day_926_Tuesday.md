# Day 926: The KLM Protocol

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Non-deterministic gates, measurement-induced nonlinearity |
| Afternoon | 2.5 hours | Problem solving: Gate teleportation and success probabilities |
| Evening | 1.5 hours | Computational lab: KLM gate simulations |

## Learning Objectives

By the end of today, you will be able to:

1. Explain why linear optics alone cannot create deterministic entangling gates
2. Derive the conditional sign-flip gate using ancilla photons and measurement
3. Calculate success probabilities for non-deterministic CNOT implementations
4. Understand gate teleportation and its role in boosting success probability
5. Analyze the resource overhead for KLM-based quantum computation
6. Implement probabilistic gate simulations in Python

## Core Content

### 1. The Challenge of Two-Qubit Gates in Linear Optics

**Why Linear Optics is Limited:**
Consider two dual-rail qubits passing through a linear optical network. The total transformation is:
$$\hat{U}_{total} = \hat{U}^{(1)} \otimes \hat{U}^{(2)}$$

This is always a product of single-qubit unitaries - no entanglement can be created deterministically!

**The Problem:**
For universal quantum computing, we need an entangling gate like CNOT or CZ. In linear optics:
- Photons don't naturally interact
- Beam splitters and phase shifters preserve mode separability
- We need an effective nonlinearity

**The KLM Solution (2001):**
Knill, Laflamme, and Milburn showed that:
1. **Measurement** can induce effective nonlinearity
2. **Probabilistic gates** can be made near-deterministic via **teleportation**
3. Linear optics + single-photon sources + photon detection = universal QC

### 2. Nonlinear Sign Shift (NS Gate)

The fundamental building block is the Nonlinear Sign gate:
$$\hat{U}_{NS} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

In Fock basis $\{|0\rangle, |1\rangle, |2\rangle\}$:
$$\hat{U}_{NS}|0\rangle = |0\rangle, \quad \hat{U}_{NS}|1\rangle = |1\rangle, \quad \hat{U}_{NS}|2\rangle = -|2\rangle$$

This applies a $\pi$ phase to the two-photon component only.

**Implementation:**
The NS gate requires ancilla photons and conditional measurement.

**NS-1 Gate Circuit:**
- Input mode + one ancilla photon in mode 2
- Three beam splitters with specific reflectivities
- Conditional on detecting one photon in the ancilla output

The reflectivities: $R_1 = R_3 = 1 - 1/\sqrt{2}$, $R_2 = 1/2$

**Success Probability:**
$$P_{success} = \frac{1}{4}$$

### 3. Conditional Sign Flip (CZ Gate)

The CZ gate in dual-rail encoding:
$$CZ|00\rangle = |00\rangle, \quad CZ|01\rangle = |01\rangle$$
$$CZ|10\rangle = |10\rangle, \quad CZ|11\rangle = -|11\rangle$$

In the Fock basis for four modes (two dual-rail qubits):
$$|00\rangle = |1,0,1,0\rangle, \quad |01\rangle = |1,0,0,1\rangle$$
$$|10\rangle = |0,1,1,0\rangle, \quad |11\rangle = |0,1,0,1\rangle$$

**The KLM CZ Implementation:**
1. Mix the "1" rails of each qubit on a beam splitter
2. Apply NS gates to each output mode
3. Mix again on a beam splitter

The overlap of the two "1" rails can create a two-photon component that picks up the sign.

### 4. Non-Deterministic CNOT

**CNOT from CZ:**
$$CNOT = (I \otimes H) \cdot CZ \cdot (I \otimes H)$$

**Direct KLM CNOT:**
Ralph et al. (2002) gave a direct construction with:
- 2 control modes, 2 target modes
- 2 ancilla photons
- Success probability: $P = 1/16$

**Improved Constructions:**
With more ancillas, success probability can approach:
$$P_{success} \to \frac{1}{2}$$

But this requires $O(n^2)$ ancilla photons for $n$ gates.

### 5. Gate Teleportation

The key insight: **we can prepare entangled resource states offline**, then teleport the gate onto our data qubits.

**Standard Teleportation:**
$$|\psi\rangle|00\rangle \xrightarrow{CNOT}|\psi\rangle\frac{|00\rangle + |11\rangle}{\sqrt{2}} \xrightarrow{measure} \text{classical bits} \to \text{corrections}$$

**Gate Teleportation:**
Instead of teleporting a state, we teleport a gate:

1. Prepare entangled state $|\chi\rangle$ that encodes the desired gate
2. Teleport input qubit through $|\chi\rangle$
3. Apply Pauli corrections based on measurement outcomes

**Key Advantage:**
The entangled resource $|\chi\rangle$ can be prepared probabilistically offline. Once we have it, the gate succeeds with probability approaching 1!

**Resource State for CZ:**
$$|\chi_{CZ}\rangle = CZ \cdot (|+\rangle \otimes |+\rangle \otimes |+\rangle \otimes |+\rangle)$$

### 6. Boosting Success Probability

**Repeat-Until-Success:**
Since gates are probabilistic, we might fail. The solution:
1. Attempt the gate
2. If failed, reset and retry
3. Use teleportation to "save" successful preparations

**Hierarchical Teleportation:**
Build larger resource states from smaller ones:
- Level 0: Basic NS gates (P = 1/4)
- Level 1: CZ from NS gates (P = 1/16)
- Level 2: Larger cluster states
- Level n: Near-deterministic operations

**Polynomial Overhead:**
KLM showed that with $O(\log^c n)$ overhead, we can achieve near-deterministic gates, where $c \approx 5-7$.

### 7. Resource Analysis

**For a Single CNOT:**
- Original proposal: $~10^4$ optical elements
- Improved designs: $~10^2$ elements
- Modern cluster-state approach: more efficient

**Photon Sources Required:**
Each probabilistic attempt needs fresh single photons. For $n$ gates with success probability $p$:
$$\langle \text{photons} \rangle \sim \frac{n}{p} \cdot \log\left(\frac{1}{\epsilon}\right)$$

where $\epsilon$ is the target failure probability.

**Error Considerations:**
- Photon loss: most dominant error
- Mode mismatch: reduces HOM visibility
- Dark counts: false detection events
- Timing jitter: distinguishability effects

### 8. Modern Improvements

**One-Way Quantum Computing:**
Instead of building gates, prepare a large cluster state:
$$|cluster\rangle = \prod_{edges} CZ_{ij} |+\rangle^{\otimes N}$$

Computation proceeds by single-qubit measurements only!

**Percolation Approach (Kieling et al.):**
- Create small entangled clusters probabilistically
- Fuse them into larger clusters
- Above percolation threshold, can create arbitrarily large cluster

**Fusion Gates:**
Combine two entangled states:
$$|Bell\rangle \otimes |Bell\rangle \xrightarrow{fusion} |cluster_4\rangle$$

Success probability: 50% for type-I fusion, higher with ancillas.

## Quantum Computing Applications

### Fault-Tolerant Photonic QC

The KLM framework enables fault-tolerant quantum computing:
1. Encode logical qubits in error-correcting codes
2. Build fault-tolerant cluster states
3. Perform error-corrected computation via measurement

**PsiQuantum's Approach:**
- Silicon photonic chips for linear optics
- Integrated single-photon sources
- On-chip detection
- Target: 1 million physical qubits

### Comparison with Other Platforms

| Aspect | Photonic (KLM) | Superconducting | Trapped Ion |
|--------|----------------|-----------------|-------------|
| Operating temp | Room/cryogenic | mK | Room temp |
| Two-qubit gates | Probabilistic | Deterministic | Deterministic |
| Connectivity | Any-to-any | Limited | All-to-all |
| Coherence time | Very long | ~100 μs | ~1 s |
| Gate time | ~ns | ~100 ns | ~1 ms |

## Worked Examples

### Example 1: NS Gate Success Probability

**Problem:** Calculate the success probability for the NS gate using one ancilla photon.

**Solution:**
The NS gate circuit uses three beam splitters. The ancilla starts in $|1\rangle$ and we condition on measuring $|1\rangle$ at the output.

For input $|n\rangle|1\rangle_{anc}$ where $n \in \{0, 1, 2\}$:

After the circuit, the probability of getting the correct output AND measuring one ancilla photon:

For $|0\rangle|1\rangle$:
$$P(|0\rangle \text{ output}, |1\rangle \text{ detected}) = \frac{1}{4}$$

For $|1\rangle|1\rangle$:
$$P(|1\rangle \text{ output}, |1\rangle \text{ detected}) = \frac{1}{4}$$

For $|2\rangle|1\rangle$:
$$P(-|2\rangle \text{ output}, |1\rangle \text{ detected}) = \frac{1}{4}$$

The success probability is:
$$\boxed{P_{success} = \frac{1}{4}}$$

### Example 2: CZ Gate from NS Gates

**Problem:** Show how to construct a CZ gate from NS gates and beam splitters.

**Solution:**
Let the two qubits be in modes $(a_0, a_1)$ and $(b_0, b_1)$ with dual-rail encoding.

**Step 1:** Mix the "1" rails ($a_1$ and $b_1$) on a 50:50 beam splitter:
$$a_1 \to \frac{a_1 + b_1}{\sqrt{2}} = c, \quad b_1 \to \frac{a_1 - b_1}{\sqrt{2}} = d$$

**Step 2:** Apply NS to modes $c$ and $d$.

**Step 3:** Reverse the beam splitter:
$$c \to \frac{c + d}{\sqrt{2}} = a_1', \quad d \to \frac{c - d}{\sqrt{2}} = b_1'$$

**Analysis for $|11\rangle = |0,1,0,1\rangle$:**
After step 1: One photon goes to $c$ OR $d$ (superposition)
- $\frac{1}{\sqrt{2}}(|1\rangle_c|0\rangle_d + |0\rangle_c|1\rangle_d)$

If both photons happen to be in the same mode (requires two $|1\rangle$ rails):
- The $|2\rangle$ component gets a $-1$ from NS
- This creates the CZ phase!

The net effect: $|11\rangle \to -|11\rangle$, other states unchanged.

### Example 3: Teleportation Success

**Problem:** We prepare CZ resource states with success probability $p = 1/16$. How many attempts to get one successful resource state with 99% confidence?

**Solution:**
Probability of failure in one attempt: $q = 1 - p = 15/16$

Probability of all $n$ attempts failing: $q^n = (15/16)^n$

We want: $q^n < 0.01$

$$(15/16)^n < 0.01$$
$$n \log(15/16) < \log(0.01)$$
$$n > \frac{\log(0.01)}{\log(15/16)} = \frac{-4.605}{-0.0645} \approx 71.4$$

$$\boxed{n = 72 \text{ attempts}}$$

## Practice Problems

### Level 1: Direct Application

1. **NS Gate Action**

   Calculate the action of the NS gate on the superposition state:
   $$|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle + |1\rangle + |2\rangle)$$

2. **CZ in Dual-Rail**

   Express the CZ gate as a $16 \times 16$ matrix in the four-mode Fock basis for two dual-rail qubits (considering only the $\{|0\rangle, |1\rangle\}$ photon sectors).

3. **Gate Attempts**

   A probabilistic CNOT gate has success probability $p = 0.05$. On average, how many attempts are needed to get 10 successful gates?

### Level 2: Intermediate

4. **NS Gate Construction**

   The NS gate uses beam splitters with $R_1 = R_3 = 1 - 1/\sqrt{2} \approx 0.293$ and $R_2 = 0.5$. Calculate the corresponding mixing angles $\theta_i$ where $R = \sin^2\theta$.

5. **Fusion Gate Analysis**

   A type-I fusion gate takes two Bell pairs and, with probability 1/2, produces a 4-qubit linear cluster state. If we start with 100 Bell pairs and repeatedly apply fusion gates, estimate the expected size of the largest cluster.

6. **Error Propagation**

   If each NS gate has success probability 1/4 and the CZ requires 2 NS gates, what is the probability that the CZ succeeds? What if we use 4 NS gates in parallel with majority voting?

### Level 3: Challenging

7. **KLM Resource Counting**

   Estimate the total number of single photons needed to perform a 5-qubit Quantum Fourier Transform using KLM gates, assuming each CNOT has success probability 1/16 and we need 99.9% overall success probability.

8. **Cluster State Percolation**

   In a 2D square lattice of qubits, we attempt to create CZ bonds between neighbors with probability $p$. The percolation threshold is $p_c \approx 0.5$. If our CZ gates succeed with probability 0.4:
   a) Can we build an infinite cluster?
   b) How can boosted gates help?

9. **Fault-Tolerant Overhead**

   Using the surface code with distance $d = 7$ and physical error rate $p_{phys} = 10^{-3}$, estimate the logical error rate. Compare the overhead for photonic (probabilistic gates) vs. superconducting (deterministic gates) implementations.

## Computational Lab: KLM Gate Simulations

```python
"""
Day 926 Computational Lab: KLM Protocol Simulations
Implementing probabilistic gates and gate teleportation
"""

import numpy as np
from scipy.linalg import expm, block_diag
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Constants
SQRT2 = np.sqrt(2)

class FockSpace:
    """Truncated Fock space operations."""

    def __init__(self, n_max: int = 4):
        self.n_max = n_max
        self.dim = n_max

    def fock_state(self, n: int) -> np.ndarray:
        """Create Fock state |n⟩."""
        state = np.zeros(self.dim, dtype=complex)
        if n < self.dim:
            state[n] = 1.0
        return state

    def creation(self) -> np.ndarray:
        """Creation operator a†."""
        a_dag = np.zeros((self.dim, self.dim), dtype=complex)
        for n in range(self.dim - 1):
            a_dag[n + 1, n] = np.sqrt(n + 1)
        return a_dag

    def annihilation(self) -> np.ndarray:
        """Annihilation operator a."""
        return self.creation().T.conj()

    def number(self) -> np.ndarray:
        """Number operator."""
        return self.creation() @ self.annihilation()


def ns_gate_matrix(n_max: int = 4) -> np.ndarray:
    """
    Nonlinear sign gate: |0⟩→|0⟩, |1⟩→|1⟩, |2⟩→-|2⟩
    """
    ns = np.eye(n_max, dtype=complex)
    if n_max > 2:
        ns[2, 2] = -1
    return ns


def beam_splitter_2mode(theta: float, phi: float = 0, n_max: int = 4) -> np.ndarray:
    """
    Two-mode beam splitter in Fock basis.
    Works in the two-mode Hilbert space of dimension n_max^2.
    """
    dim = n_max
    # Build transformation using creation/annihilation operators
    fs = FockSpace(dim)
    a = fs.annihilation()
    a_dag = fs.creation()

    # Two-mode operators
    a1 = np.kron(a, np.eye(dim))
    a1_dag = np.kron(a_dag, np.eye(dim))
    a2 = np.kron(np.eye(dim), a)
    a2_dag = np.kron(np.eye(dim), a_dag)

    # Beam splitter Hamiltonian
    H = theta * (np.exp(1j * phi) * a1_dag @ a2 + np.exp(-1j * phi) * a1 @ a2_dag)

    return expm(-1j * H)


def simulate_ns_gate_probabilistic(input_state: np.ndarray,
                                   n_max: int = 4) -> Tuple[np.ndarray, float]:
    """
    Simulate probabilistic NS gate with ancilla photon.
    Returns (output_state, success_probability).
    """
    # Simplified model: NS succeeds with probability 1/4
    success_prob = 0.25

    # Apply NS gate
    ns = ns_gate_matrix(n_max)
    output_state = ns @ input_state
    output_state /= np.linalg.norm(output_state)

    return output_state, success_prob


def simulate_cz_gate(qubit1: np.ndarray, qubit2: np.ndarray,
                     n_max: int = 3) -> Tuple[np.ndarray, float]:
    """
    Simulate CZ gate on two dual-rail qubits.
    Each qubit is a 2-component vector in the logical basis.
    Returns (output_state, success_probability).
    """
    # Dual-rail encoding: |0_L⟩ = |1,0⟩, |1_L⟩ = |0,1⟩

    # Map to 4-qubit Fock state
    dim = 2  # Only |0⟩ and |1⟩ photon states for dual-rail
    state = np.kron(qubit1, qubit2)  # |q1_L, q2_L⟩

    # CZ flips sign of |11⟩ component
    cz = np.diag([1, 1, 1, -1])  # |00⟩, |01⟩, |10⟩, |11⟩

    output_state = cz @ state

    # Success probability from two NS gates
    success_prob = 0.25 * 0.25  # = 1/16

    return output_state, success_prob


def simulate_klm_cnot(control: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Simulate KLM CNOT gate.
    Returns (output_state, success_probability).
    """
    # H gate in computational basis
    H = np.array([[1, 1], [1, -1]]) / SQRT2

    # CNOT = (I ⊗ H) @ CZ @ (I ⊗ H)
    # First H on target
    target_h = H @ target

    # CZ
    combined = np.kron(control, target_h)
    cz = np.diag([1, 1, 1, -1])
    after_cz = cz @ combined

    # Reshape and apply H again
    after_cz_reshaped = after_cz.reshape(2, 2)
    output = np.zeros(4, dtype=complex)
    for i in range(2):
        h_applied = H @ after_cz_reshaped[i, :]
        output[i * 2:(i + 1) * 2] = h_applied

    success_prob = 1 / 16  # Probability of CZ success

    return output, success_prob


def teleportation_simulation():
    """
    Demonstrate gate teleportation protocol.
    """
    print("=" * 60)
    print("Gate Teleportation Simulation")
    print("=" * 60)

    # We want to teleport a CZ gate onto two data qubits

    # Step 1: Prepare data qubits
    data1 = np.array([1, 1]) / SQRT2  # |+⟩
    data2 = np.array([1, 0])  # |0⟩

    print(f"\nData qubit 1: |+⟩ = {data1}")
    print(f"Data qubit 2: |0⟩ = {data2}")

    # Step 2: Prepare resource state (offline, probabilistic)
    # For CZ teleportation, need entangled |χ⟩
    # Simplified: assume resource is prepared

    n_attempts = 0
    success = False
    while not success:
        n_attempts += 1
        # Try to prepare CZ resource with probability 1/16
        if np.random.random() < 1 / 16:
            success = True

    print(f"\nResource state prepared after {n_attempts} attempts")

    # Step 3: Apply CZ via teleportation (now deterministic!)
    output, _ = simulate_cz_gate(data1, data2)

    print(f"\nAfter CZ teleportation:")
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    for i, label in enumerate(labels):
        if np.abs(output[i]) > 1e-10:
            print(f"  {label}: {output[i]:.4f}")

    return n_attempts


def success_probability_analysis():
    """
    Analyze success probabilities for different gate constructions.
    """
    print("\n" + "=" * 60)
    print("Success Probability Analysis")
    print("=" * 60)

    # Different gate constructions and their success probabilities
    gates = {
        'NS (1 ancilla)': 0.25,
        'CZ (2 NS)': 0.25 ** 2,
        'CNOT (direct)': 1 / 16,
        'Boosted CZ (4 ancilla)': 0.25,
        'Fusion (type-I)': 0.5,
        'Fusion (type-II)': 0.75,
    }

    print("\nGate Success Probabilities:")
    for gate, prob in gates.items():
        print(f"  {gate}: {prob:.4f} ({1/prob:.1f} attempts avg)")

    # Plot attempts needed for N successful gates
    N_gates = np.arange(1, 21)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for gate, prob in gates.items():
        expected_attempts = N_gates / prob
        plt.plot(N_gates, expected_attempts, 'o-', label=gate, markersize=4)

    plt.xlabel('Number of Gates Needed', fontsize=12)
    plt.ylabel('Expected Attempts', fontsize=12)
    plt.title('Resource Scaling for Probabilistic Gates', fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    # Probability of getting all N gates in K attempts
    probs = [0.25, 1/16, 0.5]
    labels = ['NS (p=0.25)', 'CNOT (p=1/16)', 'Fusion (p=0.5)']

    N = 5  # Need 5 successful gates
    K_values = np.arange(N, 200)

    for prob, label in zip(probs, labels):
        # Binomial: P(at least N successes in K attempts)
        from scipy.stats import binom
        p_success = 1 - binom.cdf(N - 1, K_values, prob)
        plt.plot(K_values, p_success, '-', label=label, linewidth=2)

    plt.xlabel('Number of Attempts (K)', fontsize=12)
    plt.ylabel(f'P(at least {N} successes)', fontsize=12)
    plt.title(f'Cumulative Success Probability (need {N} gates)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% threshold')

    plt.tight_layout()
    plt.savefig('klm_success_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: klm_success_analysis.png")


def cluster_state_building():
    """
    Simulate building cluster states with probabilistic gates.
    """
    print("\n" + "=" * 60)
    print("Cluster State Building Simulation")
    print("=" * 60)

    # Start with N Bell pairs
    N_initial = 50
    fusion_success_prob = 0.5

    # Simulate fusion rounds
    current_clusters = [(2, 1)] * (N_initial // 2)  # List of (size, count)

    round_num = 0
    history = [(0, sum(c[1] for c in current_clusters))]

    while len(current_clusters) > 1:
        round_num += 1
        new_clusters = []
        i = 0

        while i < len(current_clusters) - 1:
            size1, count1 = current_clusters[i]
            size2, count2 = current_clusters[i + 1]

            # Attempt fusion
            n_fusions = min(count1, count2)
            n_success = np.random.binomial(n_fusions, fusion_success_prob)

            if n_success > 0:
                new_clusters.append((size1 + size2, n_success))

            # Leftover clusters
            leftover1 = count1 - n_fusions
            leftover2 = count2 - n_fusions
            if leftover1 > 0:
                new_clusters.append((size1, leftover1))
            if leftover2 > 0:
                new_clusters.append((size2, leftover2))

            i += 2

        if i == len(current_clusters) - 1:
            new_clusters.append(current_clusters[-1])

        current_clusters = new_clusters
        history.append((round_num, len(current_clusters)))

    print(f"\nStarting with {N_initial // 2} Bell pairs")
    print(f"After {round_num} fusion rounds:")
    for size, count in current_clusters:
        print(f"  {count} cluster(s) of size {size}")

    # Find largest cluster
    if current_clusters:
        largest = max(c[0] for c in current_clusters)
        print(f"\nLargest cluster size: {largest} qubits")


def error_analysis():
    """
    Analyze error sources in KLM protocol.
    """
    print("\n" + "=" * 60)
    print("Error Analysis for KLM Protocol")
    print("=" * 60)

    # Error sources
    photon_loss = 0.1  # 10% loss per component
    mode_mismatch = 0.02  # 2% mode mismatch
    dark_count_rate = 1e-5  # per detection window
    timing_jitter = 0.01  # 1% timing error

    # Total error per gate (simplified model)
    n_components = 10  # beam splitters, detectors per gate
    n_detections = 4

    p_loss_error = 1 - (1 - photon_loss) ** n_components
    p_mismatch_error = 1 - (1 - mode_mismatch) ** n_components
    p_dark_count = 1 - (1 - dark_count_rate) ** n_detections
    p_timing_error = timing_jitter

    total_error = p_loss_error + p_mismatch_error + p_dark_count + p_timing_error

    print("\nError Budget per Gate:")
    print(f"  Photon loss: {p_loss_error:.4f}")
    print(f"  Mode mismatch: {p_mismatch_error:.4f}")
    print(f"  Dark counts: {p_dark_count:.6f}")
    print(f"  Timing jitter: {p_timing_error:.4f}")
    print(f"  Total error: {total_error:.4f}")

    # Plot error vs loss rate
    loss_rates = np.linspace(0, 0.3, 50)
    errors = 1 - (1 - loss_rates) ** n_components

    plt.figure(figsize=(10, 6))
    plt.plot(loss_rates * 100, errors * 100, 'b-', linewidth=2)
    plt.xlabel('Component Loss Rate (%)', fontsize=12)
    plt.ylabel('Gate Error Rate (%)', fontsize=12)
    plt.title('Photon Loss Impact on KLM Gate Fidelity', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark threshold for fault-tolerance
    plt.axhline(y=1, color='r', linestyle='--', label='Typical FT threshold (~1%)')
    plt.legend()
    plt.savefig('klm_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: klm_error_analysis.png")


def main():
    """Run all KLM simulations."""
    print("\n" + "=" * 60)
    print("DAY 926: KLM PROTOCOL SIMULATIONS")
    print("=" * 60)

    # Test NS gate
    print("\n--- NS Gate Test ---")
    fs = FockSpace(4)
    test_state = (fs.fock_state(0) + fs.fock_state(1) + fs.fock_state(2)) / np.sqrt(3)
    ns = ns_gate_matrix(4)
    output = ns @ test_state

    print(f"Input: (|0⟩ + |1⟩ + |2⟩)/√3")
    print(f"Output after NS:")
    for n in range(4):
        if np.abs(output[n]) > 1e-10:
            print(f"  |{n}⟩: {output[n]:.4f}")

    # Test CZ gate
    print("\n--- CZ Gate Test ---")
    q1 = np.array([1, 1]) / SQRT2  # |+⟩
    q2 = np.array([1, 1]) / SQRT2  # |+⟩
    output, p = simulate_cz_gate(q1, q2)
    print(f"Input: |+⟩|+⟩")
    print(f"Output: {output}")
    print(f"Success probability: {p}")

    # Test CNOT
    print("\n--- CNOT Gate Test ---")
    control = np.array([1, 1]) / SQRT2  # |+⟩
    target = np.array([1, 0])  # |0⟩
    output, p = simulate_klm_cnot(control, target)
    print(f"Input: |+⟩|0⟩")
    print(f"Expected: (|00⟩ + |11⟩)/√2")
    print(f"Output: {output}")

    # Run detailed simulations
    teleportation_simulation()
    success_probability_analysis()
    cluster_state_building()
    error_analysis()

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
| NS gate | $\|2\rangle \to -\|2\rangle$, others unchanged |
| NS success probability | $P_{NS} = 1/4$ |
| CZ from NS | $P_{CZ} = P_{NS}^2 = 1/16$ |
| Gate teleportation | $P_{teleported} \to 1$ (with offline prep) |
| Attempts for confidence $c$ | $n = \log(1-c)/\log(1-p)$ |
| Fusion type-I | $P_{fusion} = 1/2$ |

### Key Takeaways

1. **Linear optics cannot create deterministic entanglement** - measurement provides the nonlinearity
2. The **NS gate** is the fundamental building block with $P_{success} = 1/4$
3. **Gate teleportation** converts probabilistic preparation to near-deterministic execution
4. **Resource overhead** scales polynomially with circuit size
5. Modern approaches use **cluster states** and **fusion** for better efficiency
6. **Photon loss** is the dominant error source in photonic quantum computing

## Daily Checklist

- [ ] I understand why linear optics needs measurement for entangling gates
- [ ] I can explain the NS gate operation and its role in CZ construction
- [ ] I can calculate success probabilities for KLM gates
- [ ] I understand gate teleportation and its advantages
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 927

Tomorrow we explore **Boson Sampling**, a non-universal quantum computing model that demonstrated "quantum advantage" before fault-tolerant universal QC. Key topics:
- Computational complexity of permanent calculation
- Aaronson-Arkhipov theorem
- Gaussian boson sampling and applications
- Xanadu's Borealis and experimental demonstrations
