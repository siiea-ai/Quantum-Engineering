# Week 164: Integration & Oral Practice - Problem Solutions

---

## Section A: Variational Algorithms Solutions

### Solution 1: VQE for Simple Hamiltonian

**Part a) Decomposition:**

$$H = Z_1 + Z_2 + 0.5 X_1 X_2 = 1 \cdot Z \otimes I + 1 \cdot I \otimes Z + 0.5 \cdot X \otimes X$$

Three Pauli strings: $\{ZI, IZ, XX\}$ with coefficients $\{1, 1, 0.5\}$.

**Part b) Hardware-efficient ansatz:**

```
|0⟩ --Ry(θ₁)--●--Ry(θ₃)--
              |
|0⟩ --Ry(θ₂)--⊕--Ry(θ₄)--
```

4 parameters for 2 qubits, 1 entangling layer.

**Part c) Measurement count:**

For precision $\epsilon = 0.01$ on $\langle H \rangle$:

Each Pauli term needs $O(c_i^2/\epsilon^2)$ shots.

Total variance: $\text{Var}(\hat{H}) = \sum_i c_i^2 \text{Var}(\hat{P}_i) \leq \sum_i c_i^2 = 1 + 1 + 0.25 = 2.25$

Shots needed: $\frac{2.25}{(0.01)^2} = 22,500$ shots total.

**Part d) Verification experiment:**

1. Initialize ansatz parameters
2. Optimize to find minimum $E(\theta^*)$
3. Verify $E(\theta^*) \approx -2.1$ within statistical error
4. Compare with classical exact diagonalization

---

### Solution 2: QAOA for Max-Cut (Triangle)

**Part a) Cost Hamiltonian:**

Triangle has edges: $(1,2), (2,3), (1,3)$.

$$H_C = \frac{1}{2}[(I - Z_1Z_2) + (I - Z_2Z_3) + (I - Z_1Z_3)]$$
$$= \frac{3}{2}I - \frac{1}{2}(Z_1Z_2 + Z_2Z_3 + Z_1Z_3)$$

**Part b) QAOA circuit ($p = 1$):**

```
|+⟩ --Rzz(γ)--Rzz(γ)--------Rx(2β)--
      |        |
|+⟩ --Rzz(γ)--+--Rzz(γ)-----Rx(2β)--
              |    |
|+⟩ ----------+----+---------Rx(2β)--
```

Where $R_{ZZ}(\gamma) = e^{-i\gamma Z_1 Z_2/2}$.

**Part c) Optimal parameters:**

For the triangle, by symmetry and analysis:
$$\gamma^* \approx 0.615, \quad \beta^* \approx 0.393$$

(Found by numerical optimization or analytic calculation.)

**Part d) Results:**

Maximum cut = 2 (any partition cuts 2 edges).
QAOA $p=1$ expected cut: approximately 1.73.
Approximation ratio: $1.73/2 = 0.865$.

---

### Solution 7: VQE vs. QPE Comparison

**Part a) Resource requirements:**

| Resource | VQE | QPE |
|----------|-----|-----|
| Qubits | $n$ (system) | $n + t$ (system + ancilla) |
| Depth | Short (ansatz) | Deep (controlled-$U^{2^k}$) |
| Measurements | Many ($O(1/\epsilon^2)$) | Few (one per run) |
| Classical processing | Optimizer loops | Post-processing |
| Precision | Limited by ansatz | Scales with $t$ |

**Part b) Near-term suitability:**

VQE is more suitable because:
1. **Shorter circuits:** Less decoherence
2. **Noise tolerance:** Variational optimization can adapt
3. **Flexibility:** Ansatz can be tailored to hardware
4. **No coherent phase estimation:** Avoids deep controlled operations

**Part c) Combination:**

Yes! **Variational Phase Estimation:**
1. Use VQE to prepare approximate ground state
2. Use QPE for precise energy measurement
3. VQE provides good initial state for QPE

**Part d) Precision limits:**

- VQE: Limited by ansatz expressibility and noise
- QPE: Limited by ancilla count ($t$ bits of precision)
- Combined: Can achieve high precision with good initial state

---

## Section B: Algorithm Synthesis Solutions

### Solution 9: Grover + Phase Estimation

**Part a) Grover for minimum:**

Key idea: Use Grover to search for $x$ such that $f(x) \leq$ threshold.

1. Binary search on threshold value
2. For each threshold, use Grover to find $x$ with $f(x) \leq$ threshold
3. Narrow down to minimum

**Part b) Complete algorithm:**

```
MinimumFinding(f, n, m):
    # Binary search on output value
    low = 0, high = 2^m - 1
    while low < high:
        mid = (low + high) / 2
        # Grover search for x with f(x) <= mid
        found = GroverSearch(oracle: f(x) <= mid)
        if found:
            high = mid
        else:
            low = mid + 1
    return low  # minimum value
```

**Part c) Query complexity:**

- Binary search: $O(m)$ iterations
- Each Grover: $O(\sqrt{N})$ queries
- Total: $O(m\sqrt{N}) = O(m\sqrt{2^n})$

**Part d) Classical comparison:**

Classical: $O(N) = O(2^n)$ evaluations needed.
Quantum: $O(m\sqrt{N})$ - quadratic speedup in $N$.

---

### Solution 10: Complete Algorithm Analysis

| Algorithm | Speedup | Key Technique | Qubits | Gates | NISQ? |
|-----------|---------|---------------|--------|-------|-------|
| Deutsch-Jozsa | Exponential | Interference | $n+1$ | $O(n)$ | Yes |
| Simon | Exponential | Fourier sampling | $2n$ | $O(n^2)$ | Partial |
| Shor | Exponential | Phase estimation | $O(n)$ | $O(n^3)$ | No |
| Grover | Quadratic | Amplitude amp. | $n$ | $O(\sqrt{N})$ | Partial |
| VQE | Heuristic | Variational | $n$ | $O(n)$ | Yes |
| QAOA | Heuristic | Variational | $n$ | $O(pn)$ | Yes |

---

### Solution 11: Collision Finding Algorithm

**Part a) Classical complexity:**

Birthday attack: $O(N^{1/2}) = O(2^{n/2})$ queries to find collision.

**Part b) Quantum algorithm:**

**BHT Algorithm (Brassard-Hoyer-Tapp):**
1. Query $f$ on $N^{1/3}$ random inputs, store in table
2. Use Grover to search for $x$ such that $f(x)$ is in table

**Part c) Query complexity:**

- Table building: $N^{1/3}$ queries
- Grover search: $O(\sqrt{N/N^{1/3}}) = O(N^{1/3})$ queries
- Total: $O(N^{1/3}) = O(2^{n/3})$

**Part d) Optimality:**

This is optimal! Proven lower bound of $\Omega(N^{1/3})$ for collision finding.

---

### Solution 14: Error Analysis Across Algorithms

**Part a) Deutsch-Jozsa:**

With error $\epsilon$ per gate and $O(n)$ gates:
- Total error: $O(n\epsilon)$
- For $n\epsilon \ll 1$: Still distinguishes constant/balanced
- **Relatively robust** (single-shot algorithm)

**Part b) Grover:**

With $O(\sqrt{N})$ iterations, each with $O(n)$ gates:
- Total error: $O(n\sqrt{N}\epsilon)$
- For large $N$, error accumulates significantly
- **Moderately sensitive** - may need error correction

**Part c) Shor:**

Phase estimation needs precision $O(1/N^2)$:
- Very deep circuits for modular exponentiation
- Gate error of $10^{-3}$ insufficient for large $N$
- **Highly sensitive** - requires fault tolerance

**Part d) Ranking (most to least sensitive):**

1. Shor (most sensitive - needs fault tolerance)
2. Grover (moderately sensitive)
3. VQE (somewhat resilient - variational)
4. Deutsch-Jozsa (least sensitive - single shot)

---

### Solution 16: Resource Estimation for 2048-bit RSA

**Part a) Qubits:**

Logical qubits: $\sim 3 \times 2048 = 6144$ for basic Shor
With optimizations: $\sim 4000$ logical qubits

Physical qubits (surface code, $10^{-3}$ error):
- Code distance $d \approx 20-30$
- Physical/logical ratio: $\sim 1000-3000$
- Total: $4 \times 10^6$ to $10^7$ physical qubits

**Part b) Gates:**

Logical gates: $O(n^3) = O((2048)^3) \approx 8 \times 10^9$
With fast arithmetic: $O(n^2 \log n) \approx 10^8$

**Part c) Run time:**

At 1 MHz gate speed (optimistic):
- $10^8$ gates $\times 10^{-6}$ s = 100 seconds per attempt
- With error correction overhead: $\times 10-100$
- Expected: 1-10 hours per factorization

**Part d) Timeline:**

Current (2024): $\sim 1000$ physical qubits
Need: $\sim 10^6$ physical qubits

Growth rate: approximately 2x per 2 years
Time to $10^6$: $\log_2(1000) \approx 10$ doublings = 20 years

**Estimate: 2040-2050** for practical RSA-2048 breaking.

---

## Section C: Oral Practice Solutions

### Solution 17: Grover Explanations

**For physics PhD (2 min):**

"Grover's algorithm solves search problems quadratically faster than classical. Imagine searching a database of N items for one marked item. Classically, you need N/2 checks on average.

Quantum mechanically, we start with a superposition of all states. The algorithm has two steps repeated $\sqrt{N}$ times: mark the solution with a phase flip, then 'reflect about the mean' amplitude. This causes constructive interference on the solution, amplifying its probability until it's nearly 1.

The geometric picture: we're rotating in a 2D plane from the uniform superposition toward the solution, by angle $2\theta \approx 2/\sqrt{N}$ per iteration."

**For CS professor (3 min):**

[Above + complexity details]

"The query complexity is $\Theta(\sqrt{N})$, proven optimal by BBBV using a hybrid argument. This gives quadratic speedup, which is the best possible for unstructured search - showing that quantum computers can't solve NP-complete problems via oracle methods."

**For qualifying exam (5 min):**

[Above + full derivation]

"Let me derive the iteration count. In the 2D subspace spanned by solution $|w\rangle$ and non-solutions $|s'\rangle$, the initial state is $|s\rangle = \sin\theta |w\rangle + \cos\theta |s'\rangle$ where $\sin\theta = 1/\sqrt{N}$...

[Complete derivation of $k = \pi\sqrt{N}/4$]"

---

### Solution 21: Identify the Error

**Part a)** "Grover gives exponential speedup because $\sqrt{N} = 2^{n/2}$"

**Error:** This confuses the input encoding. The speedup is from $N$ queries to $\sqrt{N}$ queries - a quadratic (not exponential) reduction. The fact that $N = 2^n$ doesn't make $\sqrt{N} = 2^{n/2}$ an exponential speedup; it's still quadratic in $N$.

**Part b)** "CNOT is universal because it can create any entanglement"

**Error:** CNOT alone is not universal. It can create entanglement, but cannot create arbitrary superpositions. Universality requires CNOT **plus** single-qubit gates (like H and T). The combination $\{H, T, \text{CNOT}\}$ is universal.

**Part c)** "VQE finds the ground state because it always converges to the minimum"

**Error:** VQE is not guaranteed to converge to the global minimum. Issues include:
- Ansatz may not be expressive enough
- Local minima in parameter landscape
- Barren plateaus preventing gradient-based optimization
- Noise affecting convergence

**Part d)** "BBBV proves quantum computers can't solve NP problems"

**Error:** BBBV only proves an oracle separation. It shows that treating NP problems as black-box search doesn't give polynomial quantum algorithms. It does NOT prove $\text{NP} \not\subseteq \text{BQP}$, as non-relativizing techniques might still work.

---

## Section D: Comprehensive Synthesis Solutions

### Solution 23: Eigenvalue Algorithm Portfolio

**Part a) Exact eigenvalues via QPE:**

Requirements:
- Prepare eigenstate (or superposition)
- Controlled-$U = e^{-iH}$ operations
- $t = O(\log(1/\epsilon))$ ancilla for precision $\epsilon$
- $O(2^t)$ controlled-U operations
- Inverse QFT

**Part b) Ground state via VQE:**

Ansatz design for $2^n \times 2^n$ matrix:
- System qubits: $n$
- Hardware-efficient: $O(n)$ layers with rotation and CNOT gates
- Chemistry-inspired: If $H$ has physical structure, use UCCSD-like ansatz

Steps:
1. Decompose $H = \sum_i c_i P_i$
2. Measure $\langle P_i \rangle$ for each term
3. Optimize parameters $\theta$ to minimize $\sum_i c_i \langle P_i(\theta) \rangle$

**Part c) Hybrid near-term approach:**

1. **VQE for initial state:** Find approximate ground state
2. **Iterative phase estimation:** Use VQE output as QPE input
3. **Error mitigation:** ZNE, PEC for noise reduction

**Part d) Trade-offs:**

| Method | Precision | Depth | Measurements | NISQ? |
|--------|-----------|-------|--------------|-------|
| Full QPE | High ($2^{-t}$) | Very deep | Few | No |
| VQE | Low-medium | Short | Many | Yes |
| Hybrid | Medium-high | Medium | Medium | Partial |

---

### Solution 25: Month 41 Capstone

**Part A: Gates**

1. **Universal gate set:** $\{R_z(\theta), \sqrt{X}, \text{CNOT}\}$ (IBM-native)

2. **Gates for $10^{-6}$ error:**

Using Solovay-Kitaev: $O(\log^{3.97}(10^6)) \approx O(20^4) \approx 160,000$ gates

Using Ross-Selinger for Clifford+T: $\approx 3\log_2(10^6) \approx 60$ T-gates per rotation

3. **T-gate overhead:**

For fault-tolerant T gates via magic state distillation:
- 15-to-1 distillation: 15 noisy T-states $\to$ 1 clean
- Multiple levels needed for high fidelity
- Overhead: $\sim 100-1000$ physical operations per logical T

**Part B: Algorithms**

4. **QAOA vs. Grover for optimization:**

QAOA:
- Directly optimizes cost function
- Heuristic, no proven speedup
- Works on NISQ devices

Grover-based:
- Search for optimal solution
- $O(\sqrt{N})$ speedup proven
- Needs depth $O(\sqrt{N})$

5. **Scaling comparison:**

- QAOA: Circuit depth $O(p)$, unclear speedup
- Grover: Depth $O(\sqrt{N})$, quadratic speedup

For $N = 2^{100}$, Grover needs depth $2^{50}$ - impractical!
QAOA with $p = 10$ is feasible but heuristic.

6. **Hybrid approach:**

Combine:
1. Use QAOA to find good approximate solutions
2. Use Grover to search neighborhood of good solutions
3. Iterate between exploration (QAOA) and exploitation (Grover)

**Part C: Implementation**

7. **Qubits for 100-variable Max-Cut:**

QAOA: 100 qubits (one per variable)
Plus ancilla for measurements: negligible
Total: ~100-120 qubits

8. **Circuit depth for $p = 5$:**

Per layer:
- Cost Hamiltonian: $O(|E|)$ RZZ gates
- Mixer: 100 RX gates
- For sparse graph: depth $O(5 \cdot 100) = O(500)$

9. **Error mitigation:**

- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Symmetry verification
- Post-selection on valid solutions

**Part D: Analysis**

10. **Quantum advantage achievable?**

For Max-Cut: **Uncertain**
- No proven quantum speedup for QAOA
- Best classical algorithms are very strong
- May achieve advantage for specific instances

11. **Main bottlenecks:**

- Gate fidelity (need $> 99.9\%$)
- Coherence time (need $> 1$ ms for depth 500)
- Measurement overhead (many shots)
- Optimizer convergence (barren plateaus)

12. **5-year roadmap:**

Year 1: Benchmark QAOA on small instances ($n < 50$)
Year 2: Develop better ansatze and optimizers
Year 3: Scale to $n = 100$ with error mitigation
Year 4: Demonstrate quantum advantage on selected instances
Year 5: Integrate with hybrid classical-quantum workflows

---

*This completes the solutions for Week 164 and Month 41.*
