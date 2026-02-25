# Day 630: Week 90 Review - Amplitude Amplification

## Overview
**Day 630** | Week 90, Day 7 | Year 1, Month 23 | Week Review and Synthesis

Today we consolidate our understanding of amplitude amplification, amplitude estimation, and their applications through review, practice problems, and synthesis.

---

## Week Summary

### Topics Covered

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 624 | Generalized AA | Q = -AS₀A⁻¹Sχ for arbitrary preparations |
| 625 | Amplitude Estimation | Phase estimation on Q extracts success probability |
| 626 | Fixed-Point | Variable phases avoid overshooting |
| 627 | Oblivious AA | Exponential search works without knowing a |
| 628 | Quantum Counting | Estimate M with O(√MN) queries |
| 629 | Applications | SAT, Monte Carlo, optimization speedups |

### Master Formula Sheet

**The Q Operator:**
$$Q = -AS_0A^{-1}S_\chi$$

**State Evolution:**
$$Q^k A|0\rangle = \sin((2k+1)\theta)|good\rangle + \cos((2k+1)\theta)|bad\rangle$$

**Optimal Iterations:**
$$k_{opt} = \left\lfloor\frac{\pi}{4\theta}\right\rfloor = O\left(\frac{1}{\sqrt{a}}\right)$$

**Amplitude Estimation:**
- Q eigenvalues: $e^{\pm 2i\theta}$ where $a = \sin^2\theta$
- Phase estimation extracts $\theta$
- Precision $\epsilon$ with $O(1/\epsilon)$ queries

**Quantum Counting:**
$$M = N\sin^2\theta$$
$$\text{Error: } O\left(\frac{\sqrt{MN}}{2^m}\right)$$

---

## Comprehensive Practice Problems

### Problem Set A: Fundamentals

**A1. Q Operator Properties**
Prove that the Q operator satisfies:
a) Q is unitary
b) Q acts within the 2D subspace {|good⟩, |bad⟩}
c) Q eigenvalues are $e^{\pm 2i\theta}$

**A2. Preparation Analysis**
An algorithm A prepares:
$$A|0\rangle = \frac{1}{2}|00\rangle + \frac{\sqrt{3}}{2}|11\rangle$$

If |00⟩ is the only "good" state:
a) What is the success probability $a$?
b) How many AA iterations for 99% success?
c) What is the total number of uses of A?

**A3. Comparing Methods**
For initial success probability $a = 0.01$:
a) Standard AA iterations needed
b) Fixed-point iterations for 99% success
c) Expected iterations with oblivious exponential search

### Problem Set B: Amplitude Estimation

**B1. Precision Calculation**
To estimate $a = 0.05$ to precision $\epsilon = 0.001$:
a) How many precision qubits needed?
b) Total controlled-Q operations?
c) Compare to classical sampling

**B2. Counting Application**
A graph coloring problem has $N = 2^{15}$ possible colorings. You want to count the valid colorings $M$ to within $\pm 100$.
a) What precision in $\theta$ is needed?
b) How many phase estimation qubits?
c) Total oracle queries?

**B3. Decision Problem**
Use quantum counting to decide if $M = 0$ or $M \geq 1$ for $N = 2^{20}$ with 99% confidence. How many queries?

### Problem Set C: Applications

**C1. SAT Problem**
A 3-SAT instance has 25 variables. Assuming random-like distribution of solutions:
a) Classical expected queries if $M = 100$ solutions
b) Quantum expected queries
c) If we first count M, then search, what's the total cost?

**C2. Monte Carlo Integration**
You need to estimate $\pi$ using Monte Carlo (random points in unit square, count those in quarter circle) to 6 decimal places.
a) Classical samples needed
b) Quantum queries needed
c) Practical considerations

**C3. Portfolio Optimization**
A portfolio has $N = 1000$ possible allocations. Using amplitude amplification to find the best one:
a) Classical queries for minimum finding
b) Quantum queries using Durr-Hoyer
c) If each query takes 1μs classically vs 100μs quantumly, which is faster?

---

## Solutions to Selected Problems

### Solution A2: Preparation Analysis

State: $A|0\rangle = \frac{1}{2}|00\rangle + \frac{\sqrt{3}}{2}|11\rangle$

a) Success probability: $a = |1/2|^2 = 1/4 = 0.25$

b) $\theta = \arcsin(\sqrt{0.25}) = \pi/6$
   $k_{opt} = \lfloor\pi/(4 \cdot \pi/6)\rfloor = \lfloor 3/2 \rfloor = 1$

   After 1 iteration: $(2\cdot 1 + 1) \cdot \pi/6 = \pi/2$
   $P = \sin^2(\pi/2) = 1$ (100%!)

c) Uses of A: Initial preparation (1) + k iterations × 2 = 1 + 2 = 3

### Solution B1: Precision Calculation

a) Need $2\pi/2^m \leq 0.001$
   $2^m \geq 6283$
   $m = 13$ qubits

b) Total controlled-Q: $\sum_{k=0}^{12} 2^k = 2^{13} - 1 = 8191$

c) Classical needs $(1/0.001)^2 = 10^6$ samples
   Speedup: $10^6 / 8191 \approx 122\times$

### Solution C3: Portfolio Optimization

a) Classical: Scan all $N = 1000$ → 1000 queries

b) Quantum: Durr-Hoyer gives $O(\sqrt{N}) \approx 32$ queries

c) Classical time: $1000 \times 1\mu s = 1ms$
   Quantum time: $32 \times 100\mu s = 3.2ms$

   Classical is faster! Quantum advantage requires larger N or slower classical queries.

---

## Computational Lab: Complete Implementation

```python
"""Day 630: Week 90 Comprehensive Review"""
import numpy as np
import matplotlib.pyplot as plt

class AmplitudeAmplificationSuite:
    """Complete amplitude amplification toolkit."""

    def __init__(self, A, good_states, n_qubits):
        self.A = A
        self.good = good_states
        self.n = n_qubits
        self.N = 2**n_qubits

        self._setup()

    def _setup(self):
        """Initialize all operators and parameters."""
        A_inv = self.A.conj().T

        # Reflections
        self.S_chi = np.eye(self.N, dtype=complex)
        for g in self.good:
            self.S_chi[g, g] = -1

        self.S_0 = np.eye(self.N, dtype=complex)
        self.S_0[0, 0] = -1

        # Q operator
        self.Q = -self.A @ self.S_0 @ A_inv @ self.S_chi

        # Initial state and parameters
        zero = np.zeros(self.N)
        zero[0] = 1
        self.psi_0 = self.A @ zero

        self.a = sum(abs(self.psi_0[g])**2 for g in self.good)
        self.theta = np.arcsin(np.sqrt(self.a))
        self.k_opt = int(np.floor(np.pi / (4 * self.theta))) if self.theta > 0 else 0

    def standard_amplification(self, k=None):
        """Standard amplitude amplification."""
        if k is None:
            k = self.k_opt

        state = self.psi_0.copy()
        for _ in range(k):
            state = self.Q @ state

        prob = sum(abs(state[g])**2 for g in self.good)
        return state, prob

    def amplitude_estimation(self, m_precision):
        """Simulate amplitude estimation."""
        true_phase = self.theta / np.pi
        k = int(np.round(true_phase * 2**m_precision)) % 2**m_precision
        est_phase = k / 2**m_precision
        theta_est = est_phase * np.pi
        a_est = np.sin(theta_est)**2
        return a_est, abs(a_est - self.a)

    def quantum_counting(self, m_precision):
        """Quantum counting."""
        a_est, _ = self.amplitude_estimation(m_precision)
        M_est = self.N * a_est
        M_true = len(self.good)
        return M_est, abs(M_est - M_true)

    def fixed_point_amplification(self, L):
        """Fixed-point amplification with L composite iterations."""
        def ylc_phases(L):
            return [2 * np.arctan(1 / np.tan(np.pi * j / (2*L + 1)))
                    for j in range(1, L + 1)]

        phases = ylc_phases(L)
        A_inv = self.A.conj().T
        state = self.psi_0.copy()
        probs = [sum(abs(state[g])**2 for g in self.good)]

        for phi in phases:
            S_chi_phi = np.eye(self.N, dtype=complex)
            for g in self.good:
                S_chi_phi[g, g] = np.exp(1j * phi)

            S_0_phi = np.eye(self.N, dtype=complex)
            S_0_phi[0, 0] = np.exp(1j * phi)

            Q_phi = -self.A @ S_0_phi @ A_inv @ S_chi_phi
            state = Q_phi @ state
            probs.append(sum(abs(state[g])**2 for g in self.good))

        return probs

    def oblivious_search(self, max_rounds=20):
        """Oblivious exponential search."""
        m = 1
        total_queries = 0

        for _ in range(max_rounds):
            k = np.random.randint(0, m)
            total_queries += k

            # Success probability at this k
            p = np.sin((2*k + 1) * self.theta)**2

            if np.random.random() < p:
                return True, total_queries

            m = min(2 * m, int(np.sqrt(self.N)))

        return False, total_queries

    def summary(self):
        """Print comprehensive summary."""
        print(f"\n{'='*50}")
        print("Amplitude Amplification Summary")
        print(f"{'='*50}")
        print(f"System: {self.n} qubits, {self.N} states")
        print(f"Good states: {self.good}")
        print(f"Initial probability: a = {self.a:.6f}")
        print(f"Rotation angle: θ = {self.theta:.6f} rad = {np.degrees(self.theta):.2f}°")
        print(f"Optimal iterations: k = {self.k_opt}")

        _, p_opt = self.standard_amplification()
        print(f"Success at k_opt: P = {p_opt:.6f}")

        print(f"\nComplexity Analysis:")
        print(f"  Classical expected: {1/self.a:.1f} trials")
        print(f"  Quantum AA: {self.k_opt} iterations ({2*self.k_opt + 1} uses of A)")
        print(f"  Speedup: {(1/self.a) / max(1, 2*self.k_opt + 1):.1f}x")


def comprehensive_comparison():
    """Compare all methods."""
    n = 5
    N = 2**n

    # Hadamard preparation
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    A = H
    for _ in range(n - 1):
        A = np.kron(A, H)

    good = [7, 15]  # Two solutions

    aa = AmplitudeAmplificationSuite(A, good, n)
    aa.summary()

    # Compare methods
    print(f"\n{'='*50}")
    print("Method Comparison")
    print(f"{'='*50}")

    # Standard
    _, p_std = aa.standard_amplification()
    print(f"Standard AA (k={aa.k_opt}): P = {p_std:.6f}")

    # Amplitude estimation
    for m in [4, 6, 8]:
        a_est, error = aa.amplitude_estimation(m)
        print(f"Amplitude Est (m={m}): â = {a_est:.6f}, error = {error:.6f}")

    # Quantum counting
    for m in [4, 6, 8]:
        M_est, error = aa.quantum_counting(m)
        print(f"Counting (m={m}): M̂ = {M_est:.2f}, error = {error:.2f}")

    # Fixed-point
    fp_probs = aa.fixed_point_amplification(10)
    print(f"Fixed-point (L=10): Final P = {fp_probs[-1]:.6f}")

    # Oblivious
    successes = sum(1 for _ in range(100) if aa.oblivious_search()[0])
    print(f"Oblivious (100 trials): Success rate = {successes/100:.2f}")

    return aa


def plot_method_comparison(aa):
    """Visualize different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Standard AA
    ax1 = axes[0, 0]
    k_max = 3 * aa.k_opt
    probs = [aa.standard_amplification(k)[1] for k in range(k_max + 1)]
    ax1.plot(range(k_max + 1), probs, 'b-o', markersize=4)
    ax1.axvline(x=aa.k_opt, color='red', linestyle='--', label=f'k_opt={aa.k_opt}')
    ax1.set_xlabel('Iterations k')
    ax1.set_ylabel('Success Probability')
    ax1.set_title('Standard Amplitude Amplification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Amplitude estimation
    ax2 = axes[0, 1]
    m_values = range(3, 12)
    errors = [aa.amplitude_estimation(m)[1] for m in m_values]
    ax2.semilogy(m_values, errors, 'ro-')
    ax2.set_xlabel('Precision Qubits m')
    ax2.set_ylabel('Estimation Error')
    ax2.set_title('Amplitude Estimation Precision')
    ax2.grid(True, alpha=0.3)

    # Fixed-point
    ax3 = axes[1, 0]
    for L in [3, 5, 10]:
        fp_probs = aa.fixed_point_amplification(L)
        ax3.plot(range(len(fp_probs)), fp_probs, 'o-', label=f'L={L}')
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Composite Iterations')
    ax3.set_ylabel('Success Probability')
    ax3.set_title('Fixed-Point Amplification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Counting
    ax4 = axes[1, 1]
    M_true = len(aa.good)
    m_values = range(3, 12)
    estimates = [aa.quantum_counting(m)[0] for m in m_values]
    ax4.plot(m_values, estimates, 'go-')
    ax4.axhline(y=M_true, color='red', linestyle='--', label=f'True M={M_true}')
    ax4.fill_between(m_values, M_true - 0.5, M_true + 0.5, alpha=0.2, color='red')
    ax4.set_xlabel('Precision Qubits m')
    ax4.set_ylabel('Estimated M')
    ax4.set_title('Quantum Counting')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aa_week_review.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
print("="*60)
print("WEEK 90 COMPREHENSIVE REVIEW")
print("Amplitude Amplification and Applications")
print("="*60)

aa = comprehensive_comparison()
plot_method_comparison(aa)

# Final summary table
print("\n" + "="*60)
print("WEEK 90 SUMMARY TABLE")
print("="*60)
print(f"\n{'Method':<25} | {'Complexity':<20} | {'Key Feature'}")
print("-"*70)
methods = [
    ("Standard AA", "O(1/√a)", "Optimal for known a"),
    ("Amplitude Estimation", "O(1/ε)", "Quadratic speedup in precision"),
    ("Fixed-Point AA", "O(log(1/δ)/√a)", "No overshooting"),
    ("Oblivious AA", "O(1/√a)", "Works for unknown a"),
    ("Quantum Counting", "O(√MN)", "Count solutions"),
]
for method, complexity, feature in methods:
    print(f"{method:<25} | {complexity:<20} | {feature}")
print("-"*70)
```

---

## Week 90 Assessment Checklist

### Conceptual Understanding
- [ ] I can construct the Q operator for any preparation A
- [ ] I understand how phase estimation extracts amplitude
- [ ] I know when and why fixed-point is preferred
- [ ] I can explain oblivious search strategies
- [ ] I understand quantum counting precision

### Technical Skills
- [ ] I can calculate optimal iteration counts
- [ ] I can design amplitude estimation circuits
- [ ] I can analyze query complexity for applications
- [ ] I can implement all AA variants in code

### Problem Solving
- [ ] I can apply AA to SAT problems
- [ ] I can design quantum Monte Carlo estimators
- [ ] I can evaluate practical quantum advantages
- [ ] I understand when quantum speedups apply

---

## Preview: Week 91

Next week we explore **Quantum Walks**, covering:
- Discrete-time quantum walks
- Continuous-time walks
- Quantum walk search algorithms
- Applications to graph problems

---

*End of Week 90 — Amplitude Amplification*
