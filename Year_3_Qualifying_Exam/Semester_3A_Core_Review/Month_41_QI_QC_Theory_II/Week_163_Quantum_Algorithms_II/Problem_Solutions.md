# Week 163: Quantum Algorithms II - Complete Problem Solutions

---

## Section A: Shor's Algorithm Solutions

### Solution 1: Reduction to Order Finding

**Part a) Precise statement:**

**Reduction:** Given $N$ to factor:
1. Choose random $a \in \{2, \ldots, N-2\}$
2. If $\gcd(a, N) > 1$, return this factor
3. Find $r = \text{ord}_N(a)$ (smallest $r > 0$ with $a^r \equiv 1 \pmod N$)
4. If $r$ is odd, restart
5. If $a^{r/2} \equiv -1 \pmod N$, restart
6. Return $\gcd(a^{r/2} - 1, N)$ or $\gcd(a^{r/2} + 1, N)$

**Part b) Order of 7 mod 15:**

$7^1 \equiv 7 \pmod{15}$
$7^2 \equiv 49 \equiv 4 \pmod{15}$
$7^3 \equiv 28 \equiv 13 \pmod{15}$
$7^4 \equiv 91 \equiv 1 \pmod{15}$

$\boxed{r = 4}$

**Part c) Computing factors:**

$a^{r/2} = 7^2 = 49 \equiv 4 \pmod{15}$

$\gcd(4 - 1, 15) = \gcd(3, 15) = 3$
$\gcd(4 + 1, 15) = \gcd(5, 15) = 5$

$\boxed{15 = 3 \times 5}$ $\checkmark$

**Part d) Values where reduction fails:**

- $a = 1$: $r = 1$ (odd)
- $a = 4$: $4^2 \equiv 1$, $r = 2$, but $4^1 \equiv 4 \equiv -11 \not\equiv -1$, gives $\gcd(3,15) = 3$ (works)
- $a = 11$: $11^2 \equiv 1$, $11^1 \equiv 11 \equiv -4 \not\equiv -1$, gives $\gcd(10,15) = 5$ (works)
- $a = 14 \equiv -1$: $14^2 \equiv 1$, $14^1 \equiv -1$, fails condition 5

Fails for: $a \in \{1, 14\}$ and any $a$ with $\gcd(a, 15) > 1$ (i.e., $a = 3, 5, 6, 9, 10, 12$)

---

### Solution 3: Modular Exponentiation Unitary

**Part a) Proving unitarity:**

For $\gcd(a, N) = 1$, the map $x \mapsto ax \mod N$ is a bijection on $\{0, 1, \ldots, N-1\}$.

Therefore, $U_a$ permutes basis states, making it a unitary (permutation matrices are unitary).

Formally: $U_a^\dagger U_a = \sum_x |a^{-1}x\rangle\langle x| \cdot \sum_y |ay\rangle\langle y| = \sum_x |x\rangle\langle x| = I$

**Part b) $U_a^r = I$:**

$U_a^r|x\rangle = |a^r x \mod N\rangle = |x\rangle$ since $a^r \equiv 1 \pmod N$.

**Part c) Eigenvalues:**

Since $U_a^r = I$, eigenvalues satisfy $\lambda^r = 1$, so $\lambda \in \{e^{2\pi ik/r} : k = 0, \ldots, r-1\}$.

To show all are achieved: The eigenstate $|u_k\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1}e^{-2\pi ijk/r}|a^j\rangle$ has eigenvalue $e^{2\pi ik/r}$.

These $r$ eigenstates span the subspace $\text{span}\{|1\rangle, |a\rangle, \ldots, |a^{r-1}\rangle\}$, confirming all eigenvalues are realized.

---

### Solution 5: Phase Estimation in Shor's

**Part a) State after controlled modular exponentiation:**

Initial: $\frac{1}{\sqrt{2^t}}\sum_{j=0}^{2^t-1}|j\rangle|1\rangle$

After controlled-$U_a^j$:
$$\frac{1}{\sqrt{2^t}}\sum_{j=0}^{2^t-1}|j\rangle|a^j \mod N\rangle$$

**Part b) Why random eigenvalue:**

The state $|1\rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}|u_s\rangle$ is a uniform superposition of eigenstates.

After phase estimation and measurement, we collapse to one eigenstate, with equal probability $1/r$ for each $s$.

Thus we get a random $s/r$.

**Part c) Precision needed:**

Need to distinguish $s/r$ from $(s \pm 1)/r$, requiring precision $< 1/(2r)$.

Since $r < N$, use $t = 2\log_2 N + O(1)$ bits, giving precision $< 1/(2N) < 1/(2r)$.

$\boxed{t = 2n + O(\log n) \text{ bits where } n = \lceil\log_2 N\rceil}$

---

### Solution 6: Continued Fractions

**Part a) Theorem:**

If $|p/q - \theta| < 1/(2q^2)$ with $\gcd(p, q) = 1$, then $p/q$ is a convergent of the continued fraction expansion of $\theta$.

**Part b) Continued fraction of 0.428571:**

$0.428571 \approx 3/7$

$0.428571 = 0 + 1/(2 + 1/(3))$ approximately

$[0; 2, 3] = 0 + 1/(2 + 1/3) = 0 + 1/(7/3) = 3/7$

**Part c) If approximating $s/r$ with $r = 7$:**

$0.428571 \approx 3/7$, so $s = 3$.

Other convergents might give $s \in \{0, 1, 2, 3, 4, 5, 6\}$ depending on measurement.

**Part d) Pseudocode:**

```python
def extract_period(theta, N):
    # Compute continued fraction expansion
    cf = continued_fraction(theta)

    for convergent p/q in convergents(cf):
        if q < N and pow(a, q, N) == 1:
            return q

    return None  # Need to retry
```

---

### Solution 8: Shor's on N = 21

**Part a) Order of 2 mod 21:**

$2^1 = 2$, $2^2 = 4$, $2^3 = 8$, $2^4 = 16$, $2^5 = 32 \equiv 11$, $2^6 = 64 \equiv 1$

$\boxed{r = 6}$

**Part b) Usability check:**

- $r = 6$ is even $\checkmark$
- $2^{r/2} = 2^3 = 8 \not\equiv -1 \pmod{21}$ $\checkmark$

Order is usable.

**Part c) Computing factors:**

$\gcd(8 - 1, 21) = \gcd(7, 21) = 7$
$\gcd(8 + 1, 21) = \gcd(9, 21) = 3$

$\boxed{21 = 3 \times 7}$ $\checkmark$

**Part d) Quantum circuit output:**

Phase estimation would measure $\tilde{\theta} \approx s/6$ for random $s \in \{0, 1, 2, 3, 4, 5\}$.

For $s = 1$: $\tilde{\theta} \approx 0.1667$, continued fraction gives $1/6$, so $r = 6$.

---

## Section B: Grover's Algorithm Solutions

### Solution 11: Grover Operators

For $n = 2$, $N = 4$, with solution at $|11\rangle$:

**Part a) Oracle matrix:**

$$O_f = \text{diag}(1, 1, 1, -1) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Part b) Diffusion operator:**

$$D = 2|s\rangle\langle s| - I = \frac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{pmatrix} - I = \frac{1}{2}\begin{pmatrix} -1 & 1 & 1 & 1 \\ 1 & -1 & 1 & 1 \\ 1 & 1 & -1 & 1 \\ 1 & 1 & 1 & -1 \end{pmatrix}$$

**Part c) Grover iterator $G = DO_f$:**

$$G = \frac{1}{2}\begin{pmatrix} -1 & 1 & 1 & -1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & 1 & 1 & 1 \end{pmatrix}$$

**Part d) Eigenvalues:**

$G$ has eigenvalues $e^{\pm 2i\theta}$ where $\sin\theta = 1/\sqrt{4} = 1/2$, so $\theta = \pi/6$.

$\boxed{\lambda = e^{\pm i\pi/3} = \frac{1}{2} \pm i\frac{\sqrt{3}}{2}}$

---

### Solution 12: Geometric Analysis

**Part a) Expressing $|s\rangle$:**

$|w\rangle = |11\rangle$, $|s'\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |01\rangle + |10\rangle)$

$|s\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) = \frac{\sqrt{3}}{2}|s'\rangle + \frac{1}{2}|w\rangle$

So $\sin\theta = 1/2$, $\cos\theta = \sqrt{3}/2$, $\theta = \pi/6$.

**Part b) $O_f$ as reflection about $|s'\rangle$:**

$O_f|w\rangle = -|w\rangle$, $O_f|s'\rangle = |s'\rangle$

In the $\{|w\rangle, |s'\rangle\}$ basis:
$$O_f = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix} = I - 2|w\rangle\langle w|$$

This is reflection about $|s'\rangle$. $\checkmark$

**Part c) $D$ as reflection about $|s\rangle$:**

$D = 2|s\rangle\langle s| - I$

This is exactly the reflection operator about $|s\rangle$. $\checkmark$

**Part d) $G$ is rotation by $2\theta$:**

Composition of two reflections (about $|s'\rangle$ then $|s\rangle$) with angle $\theta$ between them gives rotation by $2\theta$.

$$G|s\rangle = \sin(3\theta)|w\rangle + \cos(3\theta)|s'\rangle$$

After $k$ iterations:
$$G^k|s\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$$

---

### Solution 13: Optimal Iterations

**Part a) Derivation:**

Want $(2k+1)\theta = \pi/2$, so $k = \frac{\pi/2 - \theta}{2\theta}$.

For $\theta \approx 1/\sqrt{N}$ (when $N$ large):
$$k \approx \frac{\pi/2}{2/\sqrt{N}} = \frac{\pi\sqrt{N}}{4}$$

$\boxed{k_{\text{opt}} = \frac{\pi}{4}\sqrt{N}}$

**Part b) For $N = 256$:**

$k_{\text{opt}} = \frac{\pi}{4}\sqrt{256} = \frac{\pi}{4} \times 16 = 4\pi \approx 12.57$

Use $k = 13$ (rounding).

$\theta = \arcsin(1/16) \approx 0.0625$

Success probability: $\sin^2((2 \times 13 + 1) \times 0.0625) = \sin^2(1.6875) \approx 0.996$

$\boxed{k = 13, P \approx 99.6\%}$

**Part c) With $k + 1$ iterations:**

$(2(k+1)+1)\theta = (2k+3)\theta \approx \pi/2 + 2\theta$

Success probability decreases: $\sin^2(\pi/2 + 2\theta) = \cos^2(2\theta) < 1$

Overshooting reduces probability!

**Part d) Graph for $N = 16$:**

$\theta = \arcsin(1/4) \approx 0.253$

| $k$ | $(2k+1)\theta$ | $\sin^2$ |
|-----|----------------|----------|
| 0 | 0.253 | 0.0625 |
| 1 | 0.758 | 0.473 |
| 2 | 1.264 | 0.920 |
| 3 | 1.769 | 0.969 |
| 4 | 2.274 | 0.596 |

Optimal at $k = 3$, then decreases.

---

### Solution 14: Multiple Solutions

**Part a) Optimal iterations for $t$ solutions:**

$\sin\theta = \sqrt{t/N}$

$k_{\text{opt}} = \frac{\pi/2 - \theta}{2\theta} \approx \frac{\pi}{4\theta} = \frac{\pi}{4}\sqrt{N/t}$

**Part b) Success probability:**

$P = \sin^2((2k+1)\theta) \geq 1 - O(t/N)$

For large $N/t$, probability approaches 1.

**Part c) If $t = N/4$:**

$\sin\theta = 1/2$, $\theta = \pi/6$

$k = \frac{\pi/2 - \pi/6}{2\pi/6} = \frac{\pi/3}{\pi/3} = 1$

$\boxed{k = 1}$, success probability $= \sin^2(\pi/2) = 1$

**Part d) If $t > N/2$:**

$\sin\theta > 1/\sqrt{2}$, $\theta > \pi/4$

Already $(2 \times 0 + 1)\theta > \pi/4$, so even $k = 0$ overshoots!

Optimal is to not iterate at all (just measure initial superposition).

---

## Section C: BBBV Theorem Solutions

### Solution 20: BBBV Theorem Statement

**Part a) Precise statement:**

**Theorem (BBBV):** Any quantum algorithm that determines whether $f:\{0,1\}^n \to \{0,1\}$ has zero or exactly one satisfying input requires $\Omega(\sqrt{N})$ queries, where $N = 2^n$.

**Part b) Key assumptions:**

1. Oracle access only (no structure)
2. Promise problem (0 or 1 solution)
3. Bounded error probability
4. Counting queries to oracle

**Part c) Why not Shor's:**

Shor's algorithm exploits algebraic structure (group structure of $\mathbb{Z}_N^*$).
BBBV applies to unstructured search with black-box oracle.
Factoring/period-finding is NOT an unstructured search problem.

---

### Solution 21: BBBV Proof Outline

**Part a) Hybrid states:**

$|\psi^{(f_0)}_t\rangle$ = state after $t$ queries when $f \equiv 0$ (no solutions)
$|\psi^{(f_w)}_t\rangle$ = state after $t$ queries when $f_w$ has solution at $w$

**Part b) Single query effect:**

For a single oracle query, the states differ only on terms involving $|w\rangle$.

The amplitude on $|w\rangle$ is at most $1/\sqrt{N}$ in expectation (uniform superposition).

Therefore:
$$\mathbb{E}_w[\||\psi^{(f_w)}_{t+1}\rangle - |\psi^{(f_0)}_{t+1}\rangle\|] \leq \mathbb{E}_w[\||\psi^{(f_w)}_t\rangle - |\psi^{(f_0)}_t\rangle\|] + O(1/\sqrt{N})$$

**Part c) Lower bound:**

By induction: $\mathbb{E}_w[\||\psi^{(f_w)}_T\rangle - |\psi^{(f_0)}_T\rangle\|] \leq O(T/\sqrt{N})$

To distinguish with constant probability, need this $\geq \Omega(1)$.

Therefore: $T = \Omega(\sqrt{N})$.

$\boxed{\text{Grover's algorithm is optimal}}$

---

### Solution 23: Implications of BBBV

**Part a) Cannot solve in $O(N^{0.49})$:**

BBBV gives $\Omega(N^{0.5})$. Since $0.49 < 0.5$, any $O(N^{0.49})$ algorithm would violate BBBV.

**Part b) Implications for NP in BQP:**

BBBV shows that treating NP problems as black-box search doesn't give polynomial speedup.

However, this is a relativized result. It doesn't rule out polynomial quantum algorithms that exploit problem structure.

**Part c) Does BBBV rule out poly-time for NP-complete?**

No! BBBV is an oracle separation. It says:
- There exists oracle $A$ with $\text{NP}^A \not\subseteq \text{BQP}^A$

This doesn't prove $\text{NP} \not\subseteq \text{BQP}$ in the real world, as non-relativizing techniques might work.

However, it strongly suggests quantum computers can't solve NP-complete problems in polynomial time.

---

## Section D: Amplitude Amplification Solutions

### Solution 25: General Amplitude Amplification

**Part a) Theorem statement:**

**Theorem:** Let $\mathcal{A}$ be an algorithm with $\mathcal{A}|0\rangle = \sin\theta|good\rangle + \cos\theta|bad\rangle$.

Then $O(1/\sin\theta)$ applications of $\mathcal{A}$, $\mathcal{A}^{-1}$, and marking oracle transform $|0\rangle$ to a state with $\Omega(1)$ probability on $|good\rangle$.

**Part b) Proof:**

The amplification operator $Q = -\mathcal{A}S_0\mathcal{A}^{-1}S_{good}$ rotates by $2\theta$ in the good-bad plane.

After $k = O(1/\theta)$ applications, angle becomes $\approx \pi/2$.

Since $\sin\theta \approx \theta$ for small $\theta$: $k = O(1/\sin\theta)$.

**Part c) Requirements on $\mathcal{A}$:**

- Must be able to implement $\mathcal{A}$ and $\mathcal{A}^{-1}$
- Must have oracle to mark "good" states
- Initial success probability $\sin^2\theta > 0$

---

### Solution 30: Comprehensive Analysis

**Part A: Shor's on 1024-bit RSA**

1. **Qubits:** $n = 1024$, need $\approx 3n = 3072$ qubits for basic algorithm. With error correction: $\approx 4000-6000$ logical qubits.

2. **Gates:** $O(n^3) = O(10^9)$ logical gates for one run.

3. **Run time:** At 1MHz: $10^9$ gates $\times 10^{-6}$s = $10^3$s $\approx$ 17 minutes per attempt.

4. **Error correction:** Need $\sim 1000$ physical qubits per logical qubit, so $\sim 4 \times 10^6$ physical qubits. Current systems have $\sim 1000$ physical qubits.

**Part B: Grover's on $2^{40}$ items**

1. **Optimal iterations:** $k = \frac{\pi}{4}\sqrt{2^{40}} = \frac{\pi}{4} \times 2^{20} \approx 8.2 \times 10^5$

2. **Gate count:** Each iteration: $O(n) = O(40)$ gates. Total: $\approx 3.3 \times 10^7$ gates.

3. **Classical comparison:** Classical needs $O(2^{40}) \approx 10^{12}$ checks. At 1GHz: $\sim 1000$ seconds.
   Quantum: $8 \times 10^5$ iterations $\times$ 40 gates $\times 1\mu$s = 32 seconds.
   Speedup factor: $\sim 30\times$

4. **Practical limitations:** Quadratic speedup often insufficient to offset quantum overhead.

**Part C: Comparison**

1. **Fundamental difference:**
   - Shor: Exponential speedup from exploiting algebraic structure
   - Grover: Quadratic speedup from amplitude amplification (no structure)

2. **Stronger evidence:**
   Shor provides stronger evidence because:
   - Exponential > quadratic
   - Suggests hidden structure in number theory problems
   - BBBV shows quadratic is limit for unstructured

3. **Near-term relevance:**
   Grover is more practically relevant:
   - Shorter circuits
   - Potential for NISQ implementation
   - Doesn't require full error correction
   - But speedup is modest

---

*This completes the solutions for Week 163. For oral practice, see Oral_Practice.md.*
