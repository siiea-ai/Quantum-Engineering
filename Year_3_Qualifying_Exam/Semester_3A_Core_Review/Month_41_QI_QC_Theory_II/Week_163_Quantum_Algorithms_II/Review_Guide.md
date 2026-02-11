# Week 163: Quantum Algorithms II - Comprehensive Review Guide

## Shor's Factoring Algorithm and Grover's Search with Optimality Proofs

---

## 1. Shor's Algorithm: Overview

### 1.1 The Factoring Problem

**Problem:** Given composite integer $N$, find non-trivial factors $p, q$ with $N = pq$.

**Classical Complexity:**
- Best known: General Number Field Sieve, $\exp(O((\log N)^{1/3}(\log\log N)^{2/3}))$
- Sub-exponential but super-polynomial

**Cryptographic Importance:**
- RSA encryption relies on hardness of factoring
- Current RSA uses 2048-4096 bit keys
- Breaking RSA with Shor requires ~4000-8000 logical qubits

### 1.2 Reduction to Order Finding

**Order Finding Problem:** Given $a$ coprime to $N$, find the smallest $r > 0$ such that $a^r \equiv 1 \pmod{N}$.

**Key Theorem:** Factoring reduces to order finding.

**Proof:**
1. Choose random $a \in \{2, \ldots, N-1\}$
2. If $\gcd(a, N) > 1$, we found a factor
3. Otherwise, find $r = \text{ord}_N(a)$
4. If $r$ is even and $a^{r/2} \not\equiv -1 \pmod{N}$:
   - $a^r - 1 \equiv 0 \pmod{N}$
   - $(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$
   - At least one of $\gcd(a^{r/2} \pm 1, N)$ is a non-trivial factor
5. Probability of success: $\geq 1 - 1/2^{k-1}$ where $N$ has $k$ distinct odd prime factors

**Why This Works:** For a random $a$, with probability at least $1/2$:
- $r$ is even, AND
- $a^{r/2} \not\equiv -1 \pmod{N}$

---

## 2. Quantum Order Finding

### 2.1 The Modular Exponentiation Unitary

Define the unitary $U_a$ on $n$-qubit states:
$$U_a|x\rangle = |ax \mod N\rangle$$

for $x < N$, and $U_a|x\rangle = |x\rangle$ for $x \geq N$.

**Key Properties:**
- $U_a$ is unitary (permutation on $\{0, 1, \ldots, N-1\}$)
- $U_a^r = I$ where $r = \text{ord}_N(a)$
- Eigenvalues are $r$-th roots of unity

### 2.2 Eigenstates of $U_a$

For $s \in \{0, 1, \ldots, r-1\}$, define:
$$|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}\omega^{-sk}|a^k \mod N\rangle$$

where $\omega = e^{2\pi i/r}$.

**Eigenvalue Equation:**
$$U_a|u_s\rangle = e^{2\pi is/r}|u_s\rangle$$

**Important Observation:**
$$\frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}|u_s\rangle = |1\rangle$$

This means we can start with $|1\rangle$ without knowing the eigenstates!

### 2.3 Phase Estimation for Order Finding

**Algorithm:**
1. Prepare $|0\rangle^{\otimes t}|1\rangle$ where $t = 2n + O(\log n)$ for $n = \lceil\log_2 N\rceil$
2. Apply Hadamard to first register: $\frac{1}{\sqrt{2^t}}\sum_{j=0}^{2^t-1}|j\rangle|1\rangle$
3. Apply controlled-$U_a^{2^j}$: $\frac{1}{\sqrt{2^t}}\sum_{j}|j\rangle|a^j \mod N\rangle$
4. Apply inverse QFT to first register
5. Measure first register

**Analysis:**
The state $|1\rangle$ is a superposition of eigenstates:
$$|1\rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}|u_s\rangle$$

Phase estimation gives an approximation to $s/r$ for random $s$.

After measurement, we get $\tilde{\theta} \approx s/r$ for some $s \in \{0, \ldots, r-1\}$.

### 2.4 Continued Fractions

**Problem:** From $\tilde{\theta} \approx s/r$, extract $r$.

**Continued Fractions Algorithm:**
1. Write $\tilde{\theta} = [a_0; a_1, a_2, \ldots]$ as continued fraction
2. Compute convergents $p_k/q_k$
3. For each convergent, check if $q_k$ might be $r$ (verify $a^{q_k} \equiv 1 \pmod{N}$)

**Theorem:** If $|\tilde{\theta} - s/r| \leq 1/(2r^2)$, then $s/r$ is a convergent of $\tilde{\theta}$.

With $t = 2n$ qubits, we achieve precision $1/2^t < 1/(2N^2) < 1/(2r^2)$.

### 2.5 Complete Complexity Analysis

**Qubit Count:**
- First register (phase estimation): $2n + O(\log n)$ qubits
- Second register (modular arithmetic): $n$ qubits
- Ancilla for modular exponentiation: $O(n)$ qubits
- **Total:** $O(n)$ qubits where $n = \log_2 N$

**Gate Count:**
- Modular exponentiation: $O(n^2)$ multiplications
- Each multiplication: $O(n \log n \log\log n)$ using FFT-based methods
- QFT: $O(n^2)$ gates
- **Total:** $O(n^3)$ gates (using fast arithmetic), or $O(n^2 \cdot n^2) = O(n^4)$ with schoolbook multiplication

**Success Probability:**
- Phase estimation success: $\Omega(1)$ per run
- Order $r$ is usable: $\geq 1/2$
- **Expected runs:** $O(1)$ to factor

$$\boxed{\text{Shor: } O((\log N)^3) \text{ gates, } O(\log N) \text{ qubits, } O(1) \text{ runs}}$$

---

## 3. Grover's Search Algorithm

### 3.1 Problem Statement

**Unstructured Search:** Given oracle access to $f:\{0,1\}^n \to \{0,1\}$, find $x$ with $f(x) = 1$.

**Promise:** Exactly one such $x$ exists (generalized to $t$ solutions later).

**Classical Complexity:** $\Omega(N)$ queries required (must check most elements).

### 3.2 Algorithm Components

**Oracle Operator:**
$$O_f|x\rangle = (-1)^{f(x)}|x\rangle$$

This can be constructed from standard oracle $U_f|x\rangle|b\rangle = |x\rangle|b \oplus f(x)\rangle$ using ancilla $|{-}\rangle$.

**Diffusion Operator:**
$$D = 2|s\rangle\langle s| - I$$

where $|s\rangle = H^{\otimes n}|0\rangle = \frac{1}{\sqrt{N}}\sum_x |x\rangle$.

**Implementation:**
$$D = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$$

The middle term can be implemented with multi-controlled Z gate.

**Grover Iterator:**
$$G = D \cdot O_f$$

### 3.3 Geometric Analysis

Define:
- $|w\rangle$ = solution state (assuming unique solution)
- $|s'\rangle = \frac{1}{\sqrt{N-1}}\sum_{x: f(x)=0}|x\rangle$ = uniform superposition of non-solutions

The initial state:
$$|s\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$$

where $\sin\theta = 1/\sqrt{N}$ (for large $N$, $\theta \approx 1/\sqrt{N}$).

**Effect of $G$:**
In the 2D subspace spanned by $\{|w\rangle, |s'\rangle\}$:
- $O_f$: Reflects about $|s'\rangle$ (flips $|w\rangle$ component)
- $D$: Reflects about $|s\rangle$

Net effect: Rotation by $2\theta$ toward $|w\rangle$.

**After $k$ iterations:**
$$G^k|s\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$$

### 3.4 Optimal Iteration Count

**Goal:** Maximize $\sin^2((2k+1)\theta)$

**Optimal $k$:** $(2k+1)\theta = \pi/2$, giving:
$$k = \frac{\pi/2 - \theta}{2\theta} \approx \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{N}$$

**Success Probability:**
$$P(\text{find } w) = \sin^2((2k+1)\theta) \geq 1 - O(1/N)$$

**Critical Point:** Too many iterations overshoots! After $k_{\text{opt}}$, probability decreases.

### 3.5 Multiple Solutions

If there are $t$ solutions among $N$ items:
- $\sin\theta = \sqrt{t/N}$
- Optimal iterations: $k = \frac{\pi}{4}\sqrt{N/t}$

**Problem:** If $t$ is unknown, we might overshoot or undershoot.

**Solution:** Quantum Counting
1. Use phase estimation on $G$
2. Eigenvalues of $G$ are $e^{\pm 2i\theta}$
3. Estimate $\theta$, deduce $t = N\sin^2\theta$

---

## 4. Grover Optimality: The BBBV Theorem

### 4.1 Statement of the Theorem

**Theorem (Bennett, Bernstein, Brassard, Vazirani, 1997):**
Any quantum algorithm that solves the unstructured search problem requires $\Omega(\sqrt{N})$ queries.

### 4.2 Proof Outline

**Setup:**
- Consider distinguishing $f \equiv 0$ (no solution) from $f$ with unique solution at $w$
- Any algorithm makes $T$ queries
- We show $T = \Omega(\sqrt{N})$ is necessary

**Key Idea:** Track how much the algorithm's state changes as we modify the oracle.

**The Hybrid Argument:**

Let $|\psi^{(f)}_t\rangle$ = state after $t$ queries with oracle $f$.

For the zero function $f_0 \equiv 0$ and function $f_w$ with $f_w(w) = 1$:

$$\||\psi^{(f_w)}_T\rangle - |\psi^{(f_0)}_T\rangle\| \leq 2T/\sqrt{N}$$

**Why?** Each query can change the state by at most $O(1/\sqrt{N})$ on average over the choice of $w$.

**Detailed Argument:**

Define $D_t = \mathbb{E}_w[\||\psi^{(f_w)}_t\rangle - |\psi^{(f_0)}_t\rangle\|^2]$

Using properties of the query model:
$$D_t \leq D_{t-1} + \frac{4}{N} + \frac{4\sqrt{D_{t-1}}}{\sqrt{N}}$$

Solving this recurrence: $D_T \leq O(T^2/N)$.

**Conclusion:**
To distinguish $f_0$ from random $f_w$ with high probability:
$$\sqrt{D_T} = \Omega(1) \Rightarrow T = \Omega(\sqrt{N})$$

### 4.3 Implications

1. **Grover is optimal:** $O(\sqrt{N})$ is the best possible
2. **No NP in BQP from oracle:** Can't solve NP-complete problems in polynomial time using oracle model
3. **Quadratic is the limit:** For unstructured problems, quantum gives at most quadratic speedup

---

## 5. Amplitude Amplification

### 5.1 General Framework

**Setup:**
- Algorithm $\mathcal{A}$ prepares state $|\psi\rangle = \sin\theta|good\rangle + \cos\theta|bad\rangle$
- We can check "good" vs "bad" (have oracle $S_\chi$ that marks good states)

**Theorem (Brassard, Hoyer, Mosca, Tapp):**
Using $O(1/\sin\theta)$ applications of $\mathcal{A}$ and $\mathcal{A}^{-1}$, we can prepare a state with $\Omega(1)$ probability on $|good\rangle$.

### 5.2 The Amplification Operator

$$Q = -\mathcal{A}S_0\mathcal{A}^{-1}S_\chi$$

where $S_0 = I - 2|0\rangle\langle 0|$ and $S_\chi$ marks good states.

This is the generalization of Grover's $G = DO$.

### 5.3 Applications

1. **Amplitude Estimation:** Estimate $\sin^2\theta$ to precision $\epsilon$ with $O(1/\epsilon)$ queries
2. **Quantum Counting:** Count solutions
3. **Element Distinctness:** $O(N^{2/3})$ for finding collision
4. **Triangle Finding:** $O(N^{5/4})$ for finding triangle in graph

---

## 6. Comparison: Shor vs. Grover

| Aspect | Shor's Algorithm | Grover's Algorithm |
|--------|------------------|-------------------|
| Problem | Structured (number theory) | Unstructured (search) |
| Speedup | Exponential | Quadratic |
| Technique | Phase estimation, QFT | Amplitude amplification |
| Key insight | Hidden period structure | Interference enhancement |
| Optimal? | Unknown (maybe faster?) | Yes (BBBV proves) |
| Practical impact | Breaks RSA | Modest speedup |
| Near-term feasibility | Requires error correction | Possible with NISQ |

### 6.1 Why the Different Speedups?

**Shor's exponential speedup:**
- Exploits algebraic structure of $\mathbb{Z}_N^*$
- Period finding is a "structured" promise problem
- Classical algorithms don't exploit structure efficiently

**Grover's quadratic speedup:**
- No structure to exploit (unstructured search)
- BBBV shows quadratic is fundamental limit
- Interference can only "amplify" by $\sqrt{N}$

---

## 7. Summary: Key Results for Qualifying Exams

### Essential Theorems

**Theorem (Shor):** Factoring $N$ can be solved in $O((\log N)^3)$ quantum gates.

**Theorem (Grover):** Search among $N$ items requires $\Theta(\sqrt{N})$ quantum queries.

**Theorem (BBBV):** Any quantum search algorithm needs $\Omega(\sqrt{N})$ queries.

### Essential Formulas

$$\boxed{\text{Shor qubits: } O(\log N), \text{ gates: } O((\log N)^3)}$$

$$\boxed{\text{Grover iterations: } k = \frac{\pi}{4}\sqrt{N/t} \text{ for } t \text{ solutions}}$$

$$\boxed{\text{Amplitude after } k \text{ Grover iterations: } \sin((2k+1)\theta)}$$

### Proof Techniques to Master

1. **Reduction arguments:** Factoring to order-finding
2. **Geometric analysis:** Grover in 2D subspace
3. **Hybrid/adversary method:** BBBV lower bound
4. **Continued fractions:** Extracting period from phase

---

*This review guide covers the essential theory of Shor's and Grover's algorithms. For problem practice, see Problem_Set.md.*
