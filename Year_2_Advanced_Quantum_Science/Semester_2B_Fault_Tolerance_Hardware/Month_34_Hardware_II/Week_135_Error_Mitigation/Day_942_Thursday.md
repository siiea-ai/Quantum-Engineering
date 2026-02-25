# Day 942: Symmetry Verification - Post-Selection and Symmetry Expansion

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Symmetry operators, conservation laws, post-selection theory |
| Afternoon | 2 hours | Symmetry expansion and error bounds |
| Evening | 2 hours | Computational lab - symmetry verification implementation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Identify exploitable symmetries** - Recognize conservation laws in quantum algorithms
2. **Implement post-selection** - Filter results based on symmetry violations
3. **Design symmetry verification circuits** - Add ancilla-based symmetry checks
4. **Apply symmetry expansion** - Use multiple symmetry sectors for bias reduction
5. **Calculate acceptance rates** - Estimate post-selection efficiency
6. **Combine with other techniques** - Integrate symmetry verification with ZNE and PEC

## Core Content

### 1. Symmetry in Quantum Systems

#### 1.1 Conservation Laws and Symmetries

Physical systems often possess symmetries that impose constraints on valid states:

**Particle Number Conservation**:
$$[\hat{N}, \hat{H}] = 0 \quad \Rightarrow \quad \langle N \rangle = \text{constant}$$

**Spin Conservation**:
$$[\hat{S}_z^{\text{total}}, \hat{H}] = 0 \quad \Rightarrow \quad \langle S_z \rangle = \text{constant}$$

**Parity**:
$$[\hat{P}, \hat{H}] = 0 \quad \Rightarrow \quad \text{states have definite parity}$$

If a computation should preserve a symmetry, any violation indicates an error occurred.

#### 1.2 Symmetry Operators in Qubit Systems

For $n$ qubits, common symmetry operators:

**Total Z Magnetization**:
$$\boxed{\hat{M}_z = \sum_{i=1}^n Z_i}$$

Eigenvalues: $m_z \in \{-n, -n+2, \ldots, n-2, n\}$

**Parity Operator**:
$$\boxed{\hat{\Pi} = \prod_{i=1}^n Z_i = Z_1 Z_2 \cdots Z_n}$$

Eigenvalues: $\pm 1$

**Number Parity** (for fermionic systems):
$$\hat{P}_N = (-1)^{\hat{N}} = \prod_i (1 - 2n_i)$$

### 2. Post-Selection Protocol

#### 2.1 Basic Post-Selection

If the ideal computation preserves symmetry $S$ with eigenvalue $s$:

1. Run the quantum circuit
2. Measure both the observable $O$ and symmetry $S$
3. Accept results where $S = s$, discard violations

**Post-selected expectation**:
$$\boxed{\langle O \rangle_{\text{PS}} = \frac{\langle P_s O P_s \rangle}{\langle P_s \rangle}}$$

where $P_s$ is the projector onto the symmetric subspace.

#### 2.2 Acceptance Rate

The acceptance probability depends on error rate:

$$\boxed{P_{\text{accept}} = \text{Tr}(P_s \rho_{\text{noisy}})}$$

For small error rate $\epsilon$ and circuit depth $d$:
$$P_{\text{accept}} \approx (1 - \epsilon_{\text{leak}})^d$$

where $\epsilon_{\text{leak}}$ is the probability per gate of leaving the symmetric subspace.

#### 2.3 Error Mitigation Effect

Post-selection reduces the effective error rate:

$$\boxed{\epsilon_{\text{eff}} \approx \frac{\epsilon - \epsilon_{\text{leak}}}{1 - \epsilon_{\text{leak}}}}$$

Errors that violate symmetry are detected and removed.

### 3. Ancilla-Based Symmetry Verification

#### 3.1 Indirect Measurement

To check symmetry without collapsing the state:

1. Prepare ancilla in $|+\rangle$
2. Apply controlled-$S$ operation
3. Measure ancilla in X basis

**Circuit**:
```
       ┌───┐      ┌───┐┌─┐
  |+⟩──┤ H ├──●───┤ H ├┤M├
       └───┘  │   └───┘└╥┘
              │         ║
|ψ⟩  ─────────S─────────╫──
                        ║
```

If $S|ψ\rangle = s|ψ\rangle$:
- Ancilla measures $|0\rangle$ with probability $\frac{1+s}{2}$
- Ancilla measures $|1\rangle$ with probability $\frac{1-s}{2}$

#### 3.2 Multi-Qubit Parity Check

For parity $\Pi = Z_1 Z_2 \cdots Z_n$:

```
     ┌───┐
|+⟩──┤ H ├──●──●──●── ... ──●──┤H├──┤M├
     └───┘  │  │  │         │
q1 ─────────Z──┼──┼── ... ──┼───────────
               │  │         │
q2 ─────────────Z──┼── ... ──┼───────────
                  │         │
q3 ─────────────────Z─ ... ──┼───────────
                             │
qn ──────────────────── ... ─Z───────────
```

Equivalent to applying controlled-$Z$ from ancilla to each qubit.

### 4. Symmetry Expansion

#### 4.1 Motivation

Post-selection introduces bias when error rates are high:
- Preferentially keeps states that "happened" to be in the symmetric sector
- Bias increases as $P_{\text{accept}}$ decreases

**Symmetry expansion** corrects this bias.

#### 4.2 Symmetry Expansion Formula

For a symmetry group $G = \{g_1, g_2, \ldots, g_k\}$:

$$\boxed{\langle O \rangle_{\text{SE}} = \frac{1}{|G|} \sum_{g \in G} \langle g O g^\dagger \rangle}$$

This averages over all symmetry sectors, removing bias from noise-induced sector leakage.

#### 4.3 Symmetry Expansion with Two Sectors

For parity symmetry with $G = \{I, \Pi\}$:

$$\langle O \rangle_{\text{SE}} = \frac{1}{2}\left[\langle O \rangle + \langle \Pi O \Pi \rangle\right]$$

If $[O, \Pi] = 0$ (observable respects parity):
$$\langle O \rangle_{\text{SE}} = \langle O \rangle$$

If $\{O, \Pi\} = 0$ (observable breaks parity):
$$\langle O \rangle_{\text{SE}} = 0$$

### 5. Combining Post-Selection with Symmetry Expansion

#### 5.1 Constrained Symmetry Expansion

First project onto symmetric subspace, then expand:

$$\boxed{\langle O \rangle_{\text{PSE}} = \frac{1}{|G|} \sum_{g \in G} \frac{\langle P_s g O g^\dagger P_s \rangle}{\langle P_s \rangle}}$$

This combines error detection (post-selection) with bias correction (expansion).

#### 5.2 McWeeny Purification Connection

Symmetry expansion is related to density matrix purification:

$$\rho_{\text{pure}} = \frac{P_s \rho P_s}{\text{Tr}(P_s \rho)}$$

For small errors, this removes off-diagonal coherences with symmetric subspace.

### 6. Practical Considerations

#### 6.1 Symmetry Selection

**Strong symmetries** (many constraints):
- Higher detection rate
- Lower acceptance rate
- Example: Full particle number conservation

**Weak symmetries** (fewer constraints):
- Lower detection rate
- Higher acceptance rate
- Example: Parity only

**Trade-off**: More symmetries detect more errors but discard more data.

#### 6.2 Non-Demolition Measurements

For repeated symmetry checks, use quantum non-demolition (QND) measurements:
- Symmetry measurement doesn't disturb observable
- Allows mid-circuit syndrome extraction

#### 6.3 Cost Analysis

| Method | Shot Overhead | Classical Cost |
|--------|---------------|----------------|
| Post-selection | $1/P_{\text{accept}}$ | Filtering |
| Ancilla verification | 1 (ancilla overhead) | Syndrome processing |
| Symmetry expansion | $|G|$ (group size) | Averaging |

## Quantum Computing Applications

### Quantum Chemistry

Electron number conservation in molecular simulations:
$$[\hat{N}_e, \hat{H}_{\text{mol}}] = 0$$

For VQE with Jordan-Wigner mapping:
- Check total occupation number
- Detect particle-hole errors

### Quantum Spin Systems

Magnetization conservation in Heisenberg models:
$$[\hat{S}_z^{\text{total}}, \hat{H}_{\text{XXZ}}] = 0$$

Post-select on correct magnetization sector.

### Variational Algorithms

Symmetry verification improves VQE accuracy:
- Enforce physical constraints
- Reduce search space
- Detect hardware errors

## Worked Examples

### Example 1: Parity Post-Selection

**Problem**: A 4-qubit state should have even parity ($\Pi = +1$). After noisy evolution, measurement outcomes are:
- 0000: 2500 counts
- 0011: 2000 counts
- 0101: 1500 counts
- 1111: 1000 counts
- 0001: 500 counts (odd parity)
- 0010: 500 counts (odd parity)

Calculate the post-selected $\langle Z_1 Z_2 \rangle$.

**Solution**:

First, identify even parity outcomes (even number of 1s):
- 0000 (even): 2500
- 0011 (even): 2000
- 0101 (even): 1500
- 1111 (even): 1000
- 0001 (odd): discard
- 0010 (odd): discard

Total accepted: $2500 + 2000 + 1500 + 1000 = 7000$

Calculate $\langle Z_1 Z_2 \rangle$ (first two qubits from the right):

For each bitstring $b_1 b_2 b_3 b_4$, $Z_1 Z_2 = (-1)^{b_1 + b_2}$:
- 0000: $(-1)^{0+0} = +1$, contribution: $+2500$
- 0011: $(-1)^{1+1} = +1$, contribution: $+2000$
- 0101: $(-1)^{0+1} = -1$, contribution: $-1500$
- 1111: $(-1)^{1+1} = +1$, contribution: $+1000$

$$\langle Z_1 Z_2 \rangle_{\text{PS}} = \frac{2500 + 2000 - 1500 + 1000}{7000} = \frac{4000}{7000}$$

$$\boxed{\langle Z_1 Z_2 \rangle_{\text{PS}} \approx 0.571}$$

Acceptance rate: $P_{\text{accept}} = 7000/8000 = 0.875$

### Example 2: Acceptance Rate Estimation

**Problem**: A 20-layer circuit has 5% probability per layer of causing a parity flip. What is the expected acceptance rate?

**Solution**:

If parity flip probability per layer is $p = 0.05$:

$$P_{\text{accept}} = (1 - p)^{20} = 0.95^{20}$$

$$= 0.3585$$

$$\boxed{P_{\text{accept}} \approx 35.8\%}$$

About 64% of shots are discarded, but remaining shots have higher fidelity.

### Example 3: Symmetry Expansion

**Problem**: For observable $O = X_1 X_2$ and parity symmetry $\Pi = Z_1 Z_2$, compute the symmetry-expanded expectation given $\langle X_1 X_2 \rangle_{\text{raw}} = 0.3$.

**Solution**:

Check commutation: $[X_1 X_2, Z_1 Z_2]$

$$X_1 X_2 \cdot Z_1 Z_2 = (X_1 Z_1)(X_2 Z_2) = (-iY_1)(-iY_2) = -Y_1 Y_2$$

$$Z_1 Z_2 \cdot X_1 X_2 = (Z_1 X_1)(Z_2 X_2) = (iY_1)(iY_2) = -Y_1 Y_2$$

So $[X_1 X_2, Z_1 Z_2] = 0$ (they commute).

Since they commute, $\Pi X_1 X_2 \Pi^\dagger = X_1 X_2$.

Symmetry expansion:
$$\langle X_1 X_2 \rangle_{\text{SE}} = \frac{1}{2}\left[\langle X_1 X_2 \rangle + \langle \Pi X_1 X_2 \Pi^\dagger \rangle\right]$$

$$= \frac{1}{2}\left[0.3 + 0.3\right] = 0.3$$

$$\boxed{\langle X_1 X_2 \rangle_{\text{SE}} = 0.3}$$

(No change because observable respects the symmetry.)

## Practice Problems

### Level 1: Direct Application

1. For a 3-qubit system with initial state $|000\rangle$, what symmetries should be preserved under $Z$-rotation gates?

2. Calculate the acceptance rate for post-selecting on $M_z = 0$ given measurement outcomes: $|00\rangle$: 4000, $|01\rangle$: 3000, $|10\rangle$: 2500, $|11\rangle$: 500.

3. Design a circuit to measure the parity $Z_1 Z_2 Z_3$ using an ancilla qubit.

### Level 2: Intermediate

4. For the XXZ Heisenberg model $H = \sum_{\langle i,j \rangle}(X_i X_j + Y_i Y_j + \Delta Z_i Z_j)$, what symmetry operators commute with $H$?

5. Derive the symmetry expansion formula for a group with 4 elements: $G = \{I, X^{\otimes n}, Y^{\otimes n}, Z^{\otimes n}\}$.

6. A VQE circuit has acceptance rate 40% with parity post-selection. If the non-post-selected error is 15%, estimate the post-selected error.

### Level 3: Challenging

7. Prove that for depolarizing noise, the acceptance rate for parity post-selection is $P_{\text{accept}} = \frac{1 + (1-2p)^n}{2}$ for an $n$-qubit system.

8. Design an optimal symmetry verification strategy for a molecular simulation that conserves both electron number and spin.

9. Derive conditions under which symmetry expansion is equivalent to post-selection (no bias correction needed).

## Computational Lab

```python
"""
Day 942: Symmetry Verification Implementation
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from collections import Counter

# ============================================================
# Part 1: Symmetry Operators
# ============================================================

def compute_parity(bitstring: str) -> int:
    """Compute parity of a bitstring (even=+1, odd=-1)."""
    return (-1) ** sum(int(b) for b in bitstring)

def compute_magnetization(bitstring: str) -> int:
    """Compute total Z magnetization: n_0 - n_1."""
    n_zeros = bitstring.count('0')
    n_ones = bitstring.count('1')
    return n_zeros - n_ones

def compute_sector(bitstring: str, symmetry: str) -> int:
    """Compute symmetry sector for a bitstring."""
    if symmetry == 'parity':
        return compute_parity(bitstring)
    elif symmetry == 'magnetization':
        return compute_magnetization(bitstring)
    else:
        raise ValueError(f"Unknown symmetry: {symmetry}")

# ============================================================
# Part 2: Post-Selection Implementation
# ============================================================

class SymmetryVerifier:
    """Class for symmetry-based error mitigation."""

    def __init__(self, n_qubits: int, symmetry: str, target_sector: int):
        """
        Args:
            n_qubits: Number of qubits in the system
            symmetry: Type of symmetry ('parity' or 'magnetization')
            target_sector: Expected symmetry eigenvalue
        """
        self.n_qubits = n_qubits
        self.symmetry = symmetry
        self.target_sector = target_sector

    def post_select(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Filter counts to keep only symmetric sector."""
        filtered = {}
        for bitstring, count in counts.items():
            # Remove any spaces and reverse for Qiskit convention
            bs = bitstring.replace(' ', '')
            if compute_sector(bs, self.symmetry) == self.target_sector:
                filtered[bitstring] = count
        return filtered

    def compute_acceptance_rate(self, counts: Dict[str, int]) -> float:
        """Compute the fraction of shots in the target sector."""
        total = sum(counts.values())
        accepted = sum(self.post_select(counts).values())
        return accepted / total if total > 0 else 0

    def compute_expectation(self, counts: Dict[str, int],
                           observable: Callable[[str], float]) -> Tuple[float, float]:
        """
        Compute expectation value of observable.

        Args:
            counts: Measurement counts
            observable: Function mapping bitstring to eigenvalue

        Returns:
            (expectation_value, statistical_error)
        """
        total = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            bs = bitstring.replace(' ', '')
            expectation += observable(bs) * count
        expectation /= total

        # Statistical error
        variance = 0
        for bitstring, count in counts.items():
            bs = bitstring.replace(' ', '')
            variance += (observable(bs) - expectation)**2 * count
        variance /= total
        std_error = np.sqrt(variance / total)

        return expectation, std_error

    def compute_expectation_postselected(self, counts: Dict[str, int],
                                         observable: Callable[[str], float]) -> Tuple[float, float, float]:
        """
        Compute post-selected expectation value.

        Returns:
            (expectation_value, statistical_error, acceptance_rate)
        """
        filtered = self.post_select(counts)
        if not filtered:
            return 0.0, float('inf'), 0.0

        exp_val, std_err = self.compute_expectation(filtered, observable)
        accept_rate = self.compute_acceptance_rate(counts)

        # Adjust error for post-selection
        std_err /= np.sqrt(accept_rate) if accept_rate > 0 else 1.0

        return exp_val, std_err, accept_rate

# ============================================================
# Part 3: Ancilla-Based Verification
# ============================================================

def create_parity_check_circuit(n_qubits: int) -> QuantumCircuit:
    """Create circuit with ancilla-based parity measurement."""
    # Main register + ancilla
    qr = QuantumRegister(n_qubits, 'q')
    anc = QuantumRegister(1, 'anc')
    cr_main = ClassicalRegister(n_qubits, 'c')
    cr_anc = ClassicalRegister(1, 'c_anc')

    qc = QuantumCircuit(qr, anc, cr_main, cr_anc)

    # Prepare ancilla in |+⟩
    qc.h(anc[0])

    # Apply controlled-Z from ancilla to each qubit
    for i in range(n_qubits):
        qc.cz(anc[0], qr[i])

    # Measure ancilla in X basis
    qc.h(anc[0])

    return qc, qr, anc, cr_main, cr_anc

def demonstrate_ancilla_verification():
    """Demonstrate ancilla-based parity verification."""

    print("="*60)
    print("Ancilla-Based Parity Verification")
    print("="*60)

    n_qubits = 3

    # Create base circuit with known parity
    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(0)  # Creates superposition
    qc_base.cx(0, 1)  # Bell state on qubits 0,1
    qc_base.cx(0, 2)  # GHZ state: (|000⟩ + |111⟩)/√2 - even parity

    # Add parity check
    qc_check, qr, anc, cr_main, cr_anc = create_parity_check_circuit(n_qubits)

    # Combine circuits
    full_qc = QuantumCircuit(n_qubits + 1, n_qubits + 1)
    full_qc.compose(qc_base, qubits=range(n_qubits), inplace=True)
    full_qc.compose(qc_check, qubits=list(range(n_qubits)) + [n_qubits],
                   clbits=list(range(n_qubits)) + [n_qubits], inplace=True)

    # Add measurements
    full_qc.measure(range(n_qubits), range(n_qubits))
    full_qc.measure(n_qubits, n_qubits)

    print("\nCircuit with parity check:")
    print(full_qc.draw())

    # Run without noise
    ideal_sim = AerSimulator()
    ideal_result = ideal_sim.run(full_qc, shots=10000).result()
    ideal_counts = ideal_result.get_counts()

    print("\nIdeal results (ancilla is rightmost bit):")
    for bs, count in sorted(ideal_counts.items(), key=lambda x: -x[1]):
        ancilla = bs[0]
        main = bs[2:]  # Skip space
        parity = "even" if compute_parity(main) == 1 else "odd"
        anc_status = "pass" if ancilla == '0' else "fail"
        print(f"  {bs}: {count:5d} (parity: {parity}, check: {anc_status})")

    # Run with noise
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.05, 2), ['cx', 'cz']
    )

    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_result = noisy_sim.run(full_qc, shots=10000).result()
    noisy_counts = noisy_result.get_counts()

    print("\nNoisy results:")
    passed = 0
    failed = 0
    for bs, count in noisy_counts.items():
        ancilla = bs[0]
        if ancilla == '0':
            passed += count
        else:
            failed += count

    print(f"  Passed parity check: {passed} ({100*passed/(passed+failed):.1f}%)")
    print(f"  Failed parity check: {failed} ({100*failed/(passed+failed):.1f}%)")

# ============================================================
# Part 4: Symmetry Expansion
# ============================================================

def symmetry_expansion(counts: Dict[str, int],
                       observable: Callable[[str], float],
                       symmetry_ops: List[Callable[[str], str]]) -> float:
    """
    Apply symmetry expansion to compute expectation value.

    Args:
        counts: Measurement counts
        observable: Function mapping bitstring to eigenvalue
        symmetry_ops: List of symmetry transformation functions

    Returns:
        Symmetry-expanded expectation value
    """
    total_counts = sum(counts.values())
    expanded_exp = 0

    for g in symmetry_ops:
        # Apply symmetry transformation to counts
        for bitstring, count in counts.items():
            bs = bitstring.replace(' ', '')
            transformed_bs = g(bs)
            expanded_exp += observable(transformed_bs) * count

    expanded_exp /= (len(symmetry_ops) * total_counts)

    return expanded_exp

def parity_flip(bitstring: str) -> str:
    """Apply parity flip (X on all qubits)."""
    return ''.join('1' if b == '0' else '0' for b in bitstring)

def identity_op(bitstring: str) -> str:
    """Identity operation."""
    return bitstring

# ============================================================
# Part 5: Full Demonstration
# ============================================================

def demonstrate_symmetry_verification():
    """Full demonstration of symmetry verification techniques."""

    print("\n" + "="*60)
    print("Symmetry Verification Demonstration")
    print("="*60)

    # Create a circuit that should preserve parity
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)

    # Start in |0000⟩ (even parity)
    # Apply gates that preserve parity
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.h(3)
    qc.cx(3, 0)

    # Note: CX preserves parity, H can change it
    # This circuit may not preserve parity perfectly

    qc.measure_all()

    print("\nTest circuit:")
    print(qc.draw())

    # Define observable: Z_0 Z_1
    def z0z1_observable(bs: str) -> float:
        # Qiskit convention: bs[-1] is qubit 0
        return (-1) ** (int(bs[-1]) + int(bs[-2]))

    # Run ideal
    ideal_sim = AerSimulator()
    ideal_result = ideal_sim.run(qc, shots=50000).result()
    ideal_counts = ideal_result.get_counts()

    # Run noisy
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01, 1), ['h']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.03, 2), ['cx']
    )

    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_result = noisy_sim.run(qc, shots=50000).result()
    noisy_counts = noisy_result.get_counts()

    # Analyze parity distribution
    print("\nParity distribution (ideal):")
    even_ideal = sum(c for bs, c in ideal_counts.items() if compute_parity(bs) == 1)
    odd_ideal = sum(c for bs, c in ideal_counts.items() if compute_parity(bs) == -1)
    print(f"  Even: {even_ideal} ({100*even_ideal/50000:.1f}%)")
    print(f"  Odd:  {odd_ideal} ({100*odd_ideal/50000:.1f}%)")

    print("\nParity distribution (noisy):")
    even_noisy = sum(c for bs, c in noisy_counts.items() if compute_parity(bs) == 1)
    odd_noisy = sum(c for bs, c in noisy_counts.items() if compute_parity(bs) == -1)
    print(f"  Even: {even_noisy} ({100*even_noisy/50000:.1f}%)")
    print(f"  Odd:  {odd_noisy} ({100*odd_noisy/50000:.1f}%)")

    # Compute expectation values
    verifier_even = SymmetryVerifier(n_qubits, 'parity', +1)
    verifier_odd = SymmetryVerifier(n_qubits, 'parity', -1)

    # Ideal expectation
    ideal_exp, ideal_err = verifier_even.compute_expectation(ideal_counts, z0z1_observable)
    print(f"\nIdeal <Z_0 Z_1>: {ideal_exp:.4f} +/- {ideal_err:.4f}")

    # Noisy expectation (no mitigation)
    noisy_exp, noisy_err = verifier_even.compute_expectation(noisy_counts, z0z1_observable)
    print(f"Noisy <Z_0 Z_1>: {noisy_exp:.4f} +/- {noisy_err:.4f}")

    # Post-selected expectation (even parity)
    ps_exp_even, ps_err_even, accept_even = verifier_even.compute_expectation_postselected(
        noisy_counts, z0z1_observable)
    print(f"Post-selected (even) <Z_0 Z_1>: {ps_exp_even:.4f} +/- {ps_err_even:.4f} "
          f"(accept: {100*accept_even:.1f}%)")

    # Symmetry expansion
    se_exp = symmetry_expansion(noisy_counts, z0z1_observable, [identity_op, parity_flip])
    print(f"Symmetry expanded <Z_0 Z_1>: {se_exp:.4f}")

    # Error comparison
    print("\n" + "-"*40)
    print("Error from ideal:")
    print(f"  Noisy:           {abs(noisy_exp - ideal_exp):.4f}")
    print(f"  Post-selected:   {abs(ps_exp_even - ideal_exp):.4f}")
    print(f"  Sym. expanded:   {abs(se_exp - ideal_exp):.4f}")

    return {
        'ideal': ideal_exp,
        'noisy': noisy_exp,
        'post_selected': ps_exp_even,
        'sym_expanded': se_exp
    }

# ============================================================
# Part 6: Acceptance Rate Analysis
# ============================================================

def analyze_acceptance_rate():
    """Analyze how acceptance rate varies with noise."""

    print("\n" + "="*60)
    print("Acceptance Rate vs Noise Analysis")
    print("="*60)

    n_qubits = 4
    error_rates = np.linspace(0.001, 0.1, 20)
    shots = 10000

    # Create GHZ circuit (preserves parity)
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    acceptance_rates = []
    expected_rates = []

    for p in error_rates:
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p, 2), ['cx']
        )

        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()

        # Compute acceptance rate
        verifier = SymmetryVerifier(n_qubits, 'parity', +1)
        accept = verifier.compute_acceptance_rate(counts)
        acceptance_rates.append(accept)

        # Expected rate: (1 - p_flip)^(n_gates)
        # For depolarizing, parity flip prob ~ p/2
        n_cx = n_qubits - 1
        expected = (1 - p/2) ** n_cx
        expected_rates.append(expected)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates * 100, acceptance_rates, 'bo-', label='Measured', markersize=8)
    plt.plot(error_rates * 100, expected_rates, 'r--', label='Expected $(1-p/2)^{n}$', linewidth=2)

    plt.xlabel('CX Error Rate (%)', fontsize=12)
    plt.ylabel('Acceptance Rate', fontsize=12)
    plt.title(f'Parity Post-Selection Acceptance Rate ({n_qubits}-qubit GHZ)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('acceptance_rate_analysis.png', dpi=150)
    plt.show()

    print("\nKey observations:")
    print(f"  At p=1%: Acceptance = {acceptance_rates[np.argmin(np.abs(error_rates - 0.01))]:.1%}")
    print(f"  At p=5%: Acceptance = {acceptance_rates[np.argmin(np.abs(error_rates - 0.05))]:.1%}")
    print(f"  At p=10%: Acceptance = {acceptance_rates[np.argmin(np.abs(error_rates - 0.10))]:.1%}")

# ============================================================
# Part 7: Magnetization Conservation
# ============================================================

def demonstrate_magnetization_conservation():
    """Demonstrate symmetry verification with magnetization conservation."""

    print("\n" + "="*60)
    print("Magnetization Conservation Demonstration")
    print("="*60)

    # Create circuit that should conserve magnetization
    # Start in |01⟩ (M_z = 0) and apply magnetization-preserving gates
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)

    # Initialize to |0101⟩ (M_z = 0)
    qc.x(1)
    qc.x(3)

    # Apply XX+YY interaction (preserves M_z)
    # Approximate with SWAP-like operations
    for _ in range(3):
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
            qc.cx(i+1, i)
            qc.cx(i, i+1)  # SWAP

    qc.measure_all()

    print("\nCircuit:")
    print(qc.draw())

    # Run with and without noise
    noise_model = NoiseModel()
    # Bit-flip noise can change magnetization
    noise_model.add_all_qubit_quantum_error(
        pauli_error([('X', 0.02), ('I', 0.98)], 2), ['cx']
    )

    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)

    ideal_counts = ideal_sim.run(qc, shots=20000).result().get_counts()
    noisy_counts = noisy_sim.run(qc, shots=20000).result().get_counts()

    # Analyze magnetization sectors
    print("\nMagnetization distribution:")

    def print_mz_dist(counts, label):
        mz_counts = Counter()
        for bs, count in counts.items():
            mz = compute_magnetization(bs)
            mz_counts[mz] += count

        total = sum(counts.values())
        print(f"\n{label}:")
        for mz in sorted(mz_counts.keys()):
            pct = 100 * mz_counts[mz] / total
            print(f"  M_z = {mz:+d}: {mz_counts[mz]:5d} ({pct:5.1f}%)")

    print_mz_dist(ideal_counts, "Ideal")
    print_mz_dist(noisy_counts, "Noisy")

    # Post-select on M_z = 0
    verifier = SymmetryVerifier(n_qubits, 'magnetization', 0)

    def z0_observable(bs):
        return (-1) ** int(bs[-1])

    ideal_exp, _ = verifier.compute_expectation(ideal_counts, z0_observable)
    noisy_exp, _ = verifier.compute_expectation(noisy_counts, z0_observable)
    ps_exp, _, accept = verifier.compute_expectation_postselected(noisy_counts, z0_observable)

    print(f"\n<Z_0> values:")
    print(f"  Ideal: {ideal_exp:.4f}")
    print(f"  Noisy: {noisy_exp:.4f}")
    print(f"  Post-selected (M_z=0): {ps_exp:.4f} (accept: {100*accept:.1f}%)")

# ============================================================
# Part 8: Combining with ZNE
# ============================================================

def combine_symmetry_with_zne():
    """Demonstrate combining symmetry verification with ZNE."""

    print("\n" + "="*60)
    print("Combining Symmetry Verification with ZNE")
    print("="*60)

    # Create circuit
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Observable
    def parity_obs(bs):
        return compute_parity(bs)

    # Noise model
    p_base = 0.02
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_base, 2), ['cx']
    )

    # Get ideal reference
    ideal_sim = AerSimulator()
    qc_meas = qc.copy()
    qc_meas.measure_all()
    ideal_counts = ideal_sim.run(qc_meas, shots=50000).result().get_counts()
    ideal_exp, _ = SymmetryVerifier(n_qubits, 'parity', 1).compute_expectation(
        ideal_counts, parity_obs)

    print(f"Ideal parity expectation: {ideal_exp:.4f}")

    # ZNE at multiple noise levels
    scale_factors = [1, 1.5, 2, 2.5]
    zne_raw_values = []
    zne_ps_values = []

    verifier = SymmetryVerifier(n_qubits, 'parity', 1)

    for scale in scale_factors:
        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(
            depolarizing_error(min(p_base * scale, 0.5), 2), ['cx']
        )

        sim = AerSimulator(noise_model=scaled_noise)
        counts = sim.run(qc_meas, shots=20000).result().get_counts()

        # Raw expectation
        raw_exp, _ = verifier.compute_expectation(counts, parity_obs)
        zne_raw_values.append(raw_exp)

        # Post-selected expectation
        ps_exp, _, _ = verifier.compute_expectation_postselected(counts, parity_obs)
        zne_ps_values.append(ps_exp)

    # Extrapolate
    coeffs_raw = np.polyfit(scale_factors, zne_raw_values, len(scale_factors) - 1)
    coeffs_ps = np.polyfit(scale_factors, zne_ps_values, len(scale_factors) - 1)

    zne_raw_extrap = np.polyval(coeffs_raw, 0)
    zne_ps_extrap = np.polyval(coeffs_ps, 0)

    print(f"\nNoisy (scale=1): {zne_raw_values[0]:.4f}")
    print(f"ZNE only: {zne_raw_extrap:.4f}")
    print(f"Post-selection only: {zne_ps_values[0]:.4f}")
    print(f"ZNE + Post-selection: {zne_ps_extrap:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    x_extrap = np.linspace(0, max(scale_factors), 100)

    plt.scatter(scale_factors, zne_raw_values, s=100, c='blue', label='Raw', zorder=5)
    plt.scatter(scale_factors, zne_ps_values, s=100, c='green', marker='s',
               label='Post-selected', zorder=5)

    plt.plot(x_extrap, np.polyval(coeffs_raw, x_extrap), 'b--', alpha=0.7)
    plt.plot(x_extrap, np.polyval(coeffs_ps, x_extrap), 'g--', alpha=0.7)

    plt.axhline(y=ideal_exp, color='black', linestyle='-', linewidth=2, label='Ideal')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Scale Factor', fontsize=12)
    plt.ylabel('Parity Expectation', fontsize=12)
    plt.title('ZNE with and without Symmetry Verification', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('symmetry_plus_zne.png', dpi=150)
    plt.show()

    # Error comparison
    print("\nErrors from ideal:")
    print(f"  Noisy: {abs(zne_raw_values[0] - ideal_exp):.4f}")
    print(f"  ZNE only: {abs(zne_raw_extrap - ideal_exp):.4f}")
    print(f"  PS only: {abs(zne_ps_values[0] - ideal_exp):.4f}")
    print(f"  ZNE+PS: {abs(zne_ps_extrap - ideal_exp):.4f}")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 942: Symmetry Verification Lab")
    print("="*60)

    # Part 1: Ancilla verification
    print("\n--- Part 1: Ancilla-Based Verification ---")
    demonstrate_ancilla_verification()

    # Part 2: Full demonstration
    print("\n--- Part 2: Symmetry Verification Demo ---")
    results = demonstrate_symmetry_verification()

    # Part 3: Acceptance rate analysis
    print("\n--- Part 3: Acceptance Rate Analysis ---")
    analyze_acceptance_rate()

    # Part 4: Magnetization conservation
    print("\n--- Part 4: Magnetization Conservation ---")
    demonstrate_magnetization_conservation()

    # Part 5: Combining with ZNE
    print("\n--- Part 5: Symmetry + ZNE ---")
    combine_symmetry_with_zne()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. Post-selection removes shots violating symmetry")
    print("  2. Acceptance rate decreases with noise/depth")
    print("  3. Symmetry expansion corrects bias from post-selection")
    print("  4. Best results from combining symmetry with ZNE")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Post-Selected Expectation | $\langle O \rangle_{\text{PS}} = \frac{\langle P_s O P_s \rangle}{\langle P_s \rangle}$ |
| Acceptance Rate | $P_{\text{accept}} \approx (1 - \epsilon_{\text{leak}})^d$ |
| Symmetry Expansion | $\langle O \rangle_{\text{SE}} = \frac{1}{|G|} \sum_{g \in G} \langle g O g^\dagger \rangle$ |
| Parity Operator | $\hat{\Pi} = \prod_{i=1}^n Z_i$ |
| Magnetization | $\hat{M}_z = \sum_{i=1}^n Z_i$ |

### Main Takeaways

1. **Symmetry violation indicates error**: Physical systems preserve certain symmetries; violations signal that errors occurred

2. **Post-selection removes errors**: Discarding results that violate known symmetries improves accuracy at the cost of reduced statistics

3. **Acceptance rate limits applicability**: As noise increases, acceptance rate drops, eventually making post-selection impractical

4. **Symmetry expansion removes bias**: Averaging over symmetry sectors corrects for bias introduced by error-induced sector leakage

5. **Combine with other techniques**: Symmetry verification works well alongside ZNE and measurement error mitigation

## Daily Checklist

- [ ] I can identify symmetries in quantum algorithms
- [ ] I can implement parity and magnetization verification
- [ ] I understand the trade-off between acceptance rate and error detection
- [ ] I can design ancilla-based symmetry measurement circuits
- [ ] I can apply symmetry expansion to remove bias
- [ ] I know how to combine symmetry verification with ZNE

## Preview of Day 943

Tomorrow we focus on **Measurement Error Mitigation**, addressing errors that occur during the readout process:

- Confusion matrix characterization
- Matrix inversion and regularization
- The M3 (Matrix-free Measurement Mitigation) method
- Scalable mitigation for many-qubit systems

Measurement errors often dominate total error budgets and are relatively easy to characterize and correct.
