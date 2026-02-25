# Day 624: Generalized Amplitude Amplification

## Overview
**Day 624** | Week 90, Day 1 | Year 1, Month 23 | Amplitude Amplification

Today we generalize Grover's algorithm to amplitude amplification, which works with arbitrary state preparation procedures rather than just uniform superposition.

---

## Learning Objectives

1. Understand the limitations of basic Grover's algorithm
2. Define the generalized amplitude amplification framework
3. Construct the Q operator for arbitrary preparations
4. Derive the amplification theorem
5. Analyze query complexity for general cases
6. Connect to practical algorithm design

---

## Core Content

### Beyond Grover: The Need for Generalization

Grover's algorithm assumes:
1. Uniform superposition as initial state
2. Access to a phase oracle

**Limitations:**
- What if we have a better initial state preparation?
- What if preparation is expensive and we want to reuse it?
- What if success probability varies with problem structure?

**Amplitude Amplification** addresses these by generalizing to arbitrary state preparations.

### The Amplitude Amplification Framework

**Setup:**
- Algorithm $A$ prepares a state from $|0\rangle$: $A|0\rangle = |\psi\rangle$
- State decomposes as: $|\psi\rangle = \sin\theta|good\rangle + \cos\theta|bad\rangle$
- "Good" states: $|good\rangle = \frac{1}{\sqrt{a}}\sum_{x: \chi(x)=1}\alpha_x|x\rangle$
- Success probability: $a = \sin^2\theta = \sum_{x: \chi(x)=1}|\alpha_x|^2$

**Goal:** Amplify the probability of measuring a "good" state.

### The Q Operator

The generalized Grover operator is:

$$\boxed{Q = -A S_0 A^{-1} S_\chi}$$

where:
- $S_\chi = I - 2\sum_{x \in good}|x\rangle\langle x|$ (phase flip good states)
- $S_0 = I - 2|0\rangle\langle 0|$ (phase flip $|0\rangle$)
- $A^{-1} = A^\dagger$ (assuming $A$ is unitary)

**Note:** The minus sign is conventional and affects only global phase.

### Decomposition of Q

Alternatively written as:
$$Q = (2A|0\rangle\langle 0|A^\dagger - I)(I - 2|good\rangle\langle good|)$$

This shows Q operates in the 2D subspace spanned by $|good\rangle$ and $|bad\rangle$, just like Grover's operator.

### The Amplification Theorem

**Theorem (Brassard, Hoyer, Mosca, Tapp 2002):**
Let $a = \sin^2\theta$ be the success probability of algorithm $A$. Then:

$$Q^k A|0\rangle = \sin((2k+1)\theta)|good\rangle + \cos((2k+1)\theta)|bad\rangle$$

**Corollary:** After $k = \lfloor\frac{\pi}{4\theta}\rfloor$ iterations:
$$P_{success} \geq 1 - a$$

**Query Complexity:** $O(1/\sqrt{a})$ uses of $A$ and $A^{-1}$ to boost success to near 1.

### Comparison: Standard Grover vs Amplitude Amplification

| Aspect | Grover | Amplitude Amplification |
|--------|--------|------------------------|
| Initial state | $H^{\otimes n}\|0\rangle$ | $A\|0\rangle$ |
| Success prob | $1/N$ | $a$ (arbitrary) |
| Iterations | $O(\sqrt{N})$ | $O(1/\sqrt{a})$ |
| Reflection 1 | Oracle $O_f$ | $S_\chi$ |
| Reflection 2 | Diffusion $D$ | $-AS_0A^{-1}$ |

### Why It Works: Geometric Picture

Just as in Grover, Q is a rotation in the 2D subspace:

$$Q = \begin{pmatrix} \cos 2\theta & -\sin 2\theta \\ \sin 2\theta & \cos 2\theta \end{pmatrix}$$

in the $\{|good\rangle, |bad\rangle\}$ basis.

Each application rotates by $2\theta$ toward $|good\rangle$.

### Circuit Structure

```
|0⟩──[  A  ]──[S_χ]──[ A† ]──[S_0]──[  A  ]──[S_χ]── ...
              └──────────Q────────┘  └────── ...
```

One Q iteration requires:
- 1 call to $A$
- 1 call to $A^\dagger$
- 1 call to $S_\chi$ (oracle)
- 1 call to $S_0$ (simple reflection)

---

## Worked Examples

### Example 1: Random Algorithm Amplification
An algorithm $A$ finds a solution with probability $a = 0.01$.

**How many amplitude amplification iterations to reach 99% success?**

**Solution:**
$\theta = \arcsin(\sqrt{0.01}) = \arcsin(0.1) = 0.1002$ rad

$k_{opt} = \lfloor\frac{\pi}{4 \times 0.1002}\rfloor = \lfloor 7.84 \rfloor = 7$

Check: $(2 \times 7 + 1) \times 0.1002 = 1.503$ rad

$P_{success} = \sin^2(1.503) = 0.9975$

**Speedup:** Classical repetition needs $1/0.01 = 100$ trials; quantum needs 7.

### Example 2: Recovering Grover
Show that standard Grover is a special case of amplitude amplification.

**Solution:**
For Grover: $A = H^{\otimes n}$, preparing uniform superposition.

Initial success probability: $a = M/N$

$\theta = \arcsin\sqrt{M/N}$

$k_{opt} = \lfloor\frac{\pi}{4}\sqrt{N/M}\rfloor$

This matches the Grover formula exactly.

The operators:
- $S_\chi$: Oracle marking M solutions ✓
- $-AS_0A^{-1} = -H^{\otimes n}(I-2|0\rangle\langle 0|)H^{\otimes n}$
  $= -(2H^{\otimes n}|0\rangle\langle 0|H^{\otimes n} - I)$
  $= I - 2|\psi_0\rangle\langle\psi_0|$...

Actually, with the sign, this gives the diffusion operator (up to global phase).

### Example 3: Better-than-Random Preparation
Suppose algorithm $A$ uses problem structure to achieve $a = 0.25$ success probability.

**Compare amplified vs non-amplified search:**

**Solution:**
Without amplification: Expected trials = $1/0.25 = 4$

With amplification:
$\theta = \arcsin(0.5) = \pi/6$
$k_{opt} = \lfloor\frac{\pi}{4 \times \pi/6}\rfloor = \lfloor 1.5 \rfloor = 1$

After 1 iteration: $(2 \times 1 + 1) \times \pi/6 = \pi/2$
$P_{success} = 1$

One Q iteration (2 uses of A) gives certainty, vs 4 classical trials.

---

## Practice Problems

### Problem 1: Amplification Iterations
An algorithm has success probability $a = 0.04$. Calculate:
a) The rotation angle $\theta$
b) Optimal number of Q iterations
c) Final success probability
d) Speedup over classical repetition

### Problem 2: Q Operator Construction
For a preparation $A$ and good states $\{|01\rangle, |10\rangle\}$ in a 2-qubit system:
a) Write the matrix for $S_\chi$
b) Write the matrix for $S_0$
c) Express Q in terms of A

### Problem 3: Inverse Algorithm
If $A$ is not its own inverse, how does using $A^\dagger \neq A$ affect the circuit? Describe the circuit for general unitary $A$.

---

## Computational Lab

```python
"""Day 624: Generalized Amplitude Amplification"""
import numpy as np
import matplotlib.pyplot as plt

class AmplitudeAmplification:
    """Generalized amplitude amplification framework."""

    def __init__(self, A, good_states, n_qubits):
        """
        Initialize amplitude amplification.

        Args:
            A: State preparation unitary (2^n x 2^n matrix)
            good_states: List of indices of "good" states
            n_qubits: Number of qubits
        """
        self.A = A
        self.A_inv = A.conj().T
        self.good_states = good_states
        self.n = n_qubits
        self.N = 2**n_qubits

        self._build_reflections()
        self._compute_parameters()

    def _build_reflections(self):
        """Build reflection operators."""
        # S_chi: Reflection about good states
        self.S_chi = np.eye(self.N)
        for g in self.good_states:
            self.S_chi[g, g] = -1

        # S_0: Reflection about |0⟩
        self.S_0 = np.eye(self.N)
        self.S_0[0, 0] = -1

        # Q operator: -A S_0 A^{-1} S_chi
        self.Q = -self.A @ self.S_0 @ self.A_inv @ self.S_chi

    def _compute_parameters(self):
        """Compute amplification parameters."""
        # Initial state
        zero_state = np.zeros(self.N)
        zero_state[0] = 1
        self.psi = self.A @ zero_state

        # Success probability
        self.a = sum(abs(self.psi[g])**2 for g in self.good_states)
        self.theta = np.arcsin(np.sqrt(self.a))

        # Optimal iterations
        if self.theta > 0:
            self.k_opt = int(np.floor(np.pi / (4 * self.theta)))
        else:
            self.k_opt = 0

    def amplify(self, k=None):
        """
        Apply k iterations of amplitude amplification.

        Args:
            k: Number of iterations (default: optimal)

        Returns:
            Final state and success probability
        """
        if k is None:
            k = self.k_opt

        state = self.psi.copy()
        for _ in range(k):
            state = self.Q @ state

        prob = sum(abs(state[g])**2 for g in self.good_states)
        return state, prob

    def theoretical_prob(self, k):
        """Theoretical success probability after k iterations."""
        return np.sin((2*k + 1) * self.theta)**2

    def summary(self):
        """Print summary of amplification parameters."""
        print(f"\nAmplitude Amplification Summary:")
        print(f"  Qubits: {self.n}, States: {self.N}")
        print(f"  Good states: {self.good_states}")
        print(f"  Initial success prob: a = {self.a:.6f}")
        print(f"  Rotation angle: θ = {np.degrees(self.theta):.4f}°")
        print(f"  Optimal iterations: k = {self.k_opt}")
        print(f"  Theoretical final prob: {self.theoretical_prob(self.k_opt):.6f}")

        # Classical comparison
        classical_trials = 1 / self.a if self.a > 0 else float('inf')
        print(f"  Classical expected trials: {classical_trials:.1f}")
        print(f"  Quantum Q iterations: {self.k_opt}")
        print(f"  Speedup: {classical_trials / max(1, 2*self.k_opt + 1):.1f}x")


def random_preparation_unitary(n):
    """Generate a random unitary for state preparation."""
    # Random complex matrix
    real = np.random.randn(2**n, 2**n)
    imag = np.random.randn(2**n, 2**n)
    random_matrix = real + 1j * imag

    # QR decomposition gives unitary
    Q, R = np.linalg.qr(random_matrix)
    return Q

def hadamard_preparation(n):
    """Standard Hadamard preparation (Grover case)."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    result = H
    for _ in range(n - 1):
        result = np.kron(result, H)
    return result

def compare_grover_vs_aa(n, marked_state):
    """Compare standard Grover with amplitude amplification framework."""
    N = 2**n

    # Grover (Hadamard preparation)
    A_grover = hadamard_preparation(n)
    aa_grover = AmplitudeAmplification(A_grover, [marked_state], n)

    print("\n" + "="*50)
    print("Standard Grover via Amplitude Amplification")
    aa_grover.summary()

    # Random preparation
    A_random = random_preparation_unitary(n)
    aa_random = AmplitudeAmplification(A_random, [marked_state], n)

    print("\n" + "="*50)
    print("Random Preparation Amplification")
    aa_random.summary()

    return aa_grover, aa_random

def visualize_amplification(aa, title="Amplitude Amplification"):
    """Visualize the amplification process."""
    k_max = max(10, 3 * aa.k_opt)

    # Track probability evolution
    probs_sim = []
    probs_theory = []

    state = aa.psi.copy()
    for k in range(k_max + 1):
        if k > 0:
            state = aa.Q @ state
        prob = sum(abs(state[g])**2 for g in aa.good_states)
        probs_sim.append(prob)
        probs_theory.append(aa.theoretical_prob(k))

    plt.figure(figsize=(10, 6))
    plt.plot(range(k_max + 1), probs_sim, 'bo-', label='Simulated', markersize=4)
    plt.plot(range(k_max + 1), probs_theory, 'r--', label='Theoretical', linewidth=2)
    plt.axvline(x=aa.k_opt, color='green', linestyle='--',
                label=f'Optimal k={aa.k_opt}')
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=aa.a, color='blue', linestyle=':', alpha=0.5,
                label=f'Initial a={aa.a:.4f}')

    plt.xlabel('Q Iterations', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig('amplitude_amplification.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_speedup_vs_initial_prob():
    """Analyze speedup as function of initial success probability."""
    a_values = np.linspace(0.001, 0.5, 100)

    classical_trials = 1 / a_values
    quantum_iters = []

    for a in a_values:
        theta = np.arcsin(np.sqrt(a))
        k = max(1, int(np.floor(np.pi / (4 * theta))))
        # Total operations: 2k+1 (including initial prep)
        quantum_iters.append(2*k + 1)

    quantum_iters = np.array(quantum_iters)
    speedup = classical_trials / quantum_iters

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(a_values, classical_trials, 'b-', label='Classical 1/a', linewidth=2)
    plt.semilogy(a_values, quantum_iters, 'r-', label='Quantum O(1/√a)', linewidth=2)
    plt.xlabel('Initial Success Probability a', fontsize=12)
    plt.ylabel('Expected Operations (log scale)', fontsize=12)
    plt.title('Classical vs Quantum Operations', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(a_values, speedup, 'g-', linewidth=2)
    plt.xlabel('Initial Success Probability a', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.title('Quantum Speedup = (1/a) / O(1/√a) = O(1/√a)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aa_speedup_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("="*60)
print("Generalized Amplitude Amplification")
print("="*60)

# Compare Grover vs general AA
aa_grover, aa_random = compare_grover_vs_aa(4, marked_state=7)

# Verify Q operator structure
print("\n" + "="*50)
print("Q Operator Verification")
print("="*50)

# Check Q is unitary
Q = aa_grover.Q
is_unitary = np.allclose(Q @ Q.conj().T, np.eye(Q.shape[0]))
print(f"Q is unitary: {is_unitary}")

# Check eigenvalues
eigvals = np.linalg.eigvals(Q)
print(f"Q eigenvalues (magnitude): {np.abs(eigvals[:4])}")
print(f"Q eigenvalues (phase): {np.angle(eigvals[:4]) / np.pi} × π")

# Visualize
visualize_amplification(aa_grover, "Amplitude Amplification (Grover Case)")

# Analyze speedup
analyze_speedup_vs_initial_prob()

# Example with specific initial probability
print("\n" + "="*60)
print("Example: Algorithm with a = 0.01 success probability")
print("="*60)

# Create custom preparation with ~1% success on state |0⟩
n = 4
A_custom = np.eye(2**n)
# Rotate state |0⟩ to have small overlap with |0⟩
theta_prep = np.arcsin(0.1)  # a = 0.01
A_custom[0:2, 0:2] = np.array([
    [np.cos(theta_prep), -np.sin(theta_prep)],
    [np.sin(theta_prep), np.cos(theta_prep)]
])

aa_custom = AmplitudeAmplification(A_custom, [0], n)
aa_custom.summary()

# Run amplification
state_final, prob_final = aa_custom.amplify()
print(f"\nAfter {aa_custom.k_opt} iterations:")
print(f"  Actual success probability: {prob_final:.6f}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Q operator | $Q = -A S_0 A^{-1} S_\chi$ |
| State after k iterations | $Q^k A\|0\rangle = \sin((2k+1)\theta)\|good\rangle + \cos((2k+1)\theta)\|bad\rangle$ |
| Optimal iterations | $k = \lfloor\frac{\pi}{4\theta}\rfloor = O(1/\sqrt{a})$ |
| Query complexity | $O(1/\sqrt{a})$ uses of A |

### Key Takeaways

1. **Amplitude amplification generalizes Grover** to arbitrary preparations
2. **Q operator** combines preparation inverse with reflections
3. **Quadratic speedup** over classical repetition
4. **Works for any initial success probability** $a > 0$
5. **Requires access to $A^{-1}$** (usually $A^\dagger$)
6. **Foundational for many quantum algorithms**

---

## Daily Checklist

- [ ] I understand the amplitude amplification framework
- [ ] I can construct the Q operator for a given preparation A
- [ ] I can derive the number of iterations needed
- [ ] I understand the connection to standard Grover
- [ ] I can analyze the speedup for arbitrary initial probabilities
- [ ] I ran the computational lab and verified the theory

---

*Next: Day 625 — Amplitude Estimation*
