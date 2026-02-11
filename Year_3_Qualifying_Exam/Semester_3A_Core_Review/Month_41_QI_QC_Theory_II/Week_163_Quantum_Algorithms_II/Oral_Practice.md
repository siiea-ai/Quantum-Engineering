# Week 163: Quantum Algorithms II - Oral Examination Practice

## Shor's Algorithm, Grover's Algorithm, and Optimality

---

## Part I: Shor's Algorithm Discussion Questions

### Question 1: The Big Picture
**Examiner:** "Explain Shor's algorithm at a high level. What problem does it solve and why is it important?"

**Expected Response:**

**Problem:** Factor large integers into prime factors.

**Why important:**
- RSA cryptography relies on factoring being hard
- Classical algorithms are exponential
- Shor's is polynomial: $O((\log N)^3)$
- Breaks RSA, Diffie-Hellman, elliptic curve crypto

**Algorithm overview:**
1. Reduce factoring to order-finding (classical number theory)
2. Use quantum phase estimation to find order
3. Use continued fractions to extract period
4. Compute factors using $\gcd$

**Follow-up:** "Why is order-finding easier than factoring?"
- Order-finding has algebraic structure we can exploit
- Eigenstates of modular exponentiation encode the period
- Phase estimation extracts this efficiently

---

### Question 2: The Reduction
**Examiner:** "Walk me through why finding the order of $a$ modulo $N$ helps factor $N$."

**Expected Derivation:**

Given $r = \text{ord}_N(a)$:
$$a^r \equiv 1 \pmod{N}$$
$$a^r - 1 \equiv 0 \pmod{N}$$

If $r$ is even:
$$(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$$

If $a^{r/2} \not\equiv \pm 1 \pmod{N}$:
- Neither factor is divisible by $N$
- But product is divisible by $N$
- So $\gcd(a^{r/2} \pm 1, N)$ gives non-trivial factor

**Follow-up:** "What's the probability this works?"
- Probability $\geq 1/2$ for random $a$
- Fails if $r$ is odd or $a^{r/2} \equiv -1$
- Expected 2 attempts to factor

---

### Question 3: Quantum Order Finding
**Examiner:** "How does quantum phase estimation find the order?"

**Expected Response:**

Define $U_a|x\rangle = |ax \mod N\rangle$.

**Key insight:** $U_a$ has eigenvalues $e^{2\pi ik/r}$ for $k = 0, \ldots, r-1$.

**Eigenstates:**
$$|u_k\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1}e^{-2\pi ijk/r}|a^j \mod N\rangle$$

**Crucial property:** $|1\rangle = \frac{1}{\sqrt{r}}\sum_k |u_k\rangle$

**Algorithm:**
1. Start with $|1\rangle$ (superposition of all eigenstates)
2. Phase estimation gives random $k/r$
3. Continued fractions extracts $r$

**Follow-up:** "Why can we start with $|1\rangle$ without knowing the eigenstates?"
- Uniform superposition of eigenstates
- Phase estimation samples random eigenvalue
- Each $k/r$ gives same $r$ via continued fractions

---

### Question 4: Complexity Analysis
**Examiner:** "Give a complete complexity analysis of Shor's algorithm."

**Expected Analysis:**

**Qubits:**
- Phase estimation: $2n$ qubits for precision
- Modular arithmetic: $n$ qubits
- Ancilla: $O(n)$ for multiplication
- **Total:** $O(n)$ where $n = \log_2 N$

**Gates:**
- Controlled modular exponentiation: $O(n)$ multiplications of $n$-bit numbers
- Each multiplication: $O(n^2)$ gates (schoolbook) or $O(n \log n \log\log n)$ (FFT)
- QFT: $O(n^2)$ gates
- **Total:** $O(n^3)$ with fast arithmetic

**Success probability:**
- Phase estimation: $\Omega(1)$
- Usable order: $\geq 1/2$
- **Expected runs:** $O(1)$

$$\boxed{O(n^3) \text{ gates, } O(n) \text{ qubits}}$$

---

## Part II: Grover's Algorithm Discussion Questions

### Question 5: Algorithm Overview
**Examiner:** "Explain Grover's search algorithm and its speedup."

**Expected Response:**

**Problem:** Find $x$ with $f(x) = 1$ among $N$ items.

**Classical:** Must check $\Omega(N)$ items.

**Quantum:** Only $O(\sqrt{N})$ oracle calls!

**Algorithm:**
1. Initialize $|s\rangle = \frac{1}{\sqrt{N}}\sum_x |x\rangle$
2. Repeat $\frac{\pi}{4}\sqrt{N}$ times:
   - Apply oracle: $O_f|x\rangle = (-1)^{f(x)}|x\rangle$
   - Apply diffusion: $D = 2|s\rangle\langle s| - I$
3. Measure to find solution

**Speedup:** Quadratic (from $N$ to $\sqrt{N}$)

**Follow-up:** "Why quadratic and not exponential like Shor?"
- No structure to exploit
- BBBV proves $\sqrt{N}$ is optimal
- Unstructured search fundamentally limited

---

### Question 6: Geometric Interpretation
**Examiner:** "Give the geometric interpretation of Grover's algorithm."

**Expected Response:**

Consider 2D plane spanned by $|w\rangle$ (solution) and $|s'\rangle$ (non-solutions).

**Initial state:**
$$|s\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$$
where $\sin\theta = 1/\sqrt{N}$.

**Oracle $O_f$:** Reflection about $|s'\rangle$
$$O_f: |w\rangle \to -|w\rangle, \quad |s'\rangle \to |s'\rangle$$

**Diffusion $D$:** Reflection about $|s\rangle$

**Combined $G = DO_f$:** Rotation by $2\theta$ toward $|w\rangle$

**After $k$ iterations:**
$$G^k|s\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$$

**Optimal:** When $(2k+1)\theta = \pi/2$, giving $k = \frac{\pi}{4\theta} \approx \frac{\pi}{4}\sqrt{N}$

---

### Question 7: Why Not More Iterations?
**Examiner:** "A student asks: why not just run more iterations to be extra sure? What's wrong with this?"

**Expected Response:**

**The problem:** Grover's algorithm oscillates!

After optimal $k$, further iterations rotate past $|w\rangle$.

At $k = 2k_{\text{opt}}$: State is approximately $|s\rangle$ again.
At $k = 3k_{\text{opt}}$: Near optimal again.

**Visualization:** Amplitude oscillates like $\sin((2k+1)\theta)$.

**Solution:** Stop at optimal $k$, or use amplitude estimation to find unknown $t$.

**Key insight:** This is fundamentally different from classical - more work can hurt!

---

### Question 8: BBBV Lower Bound
**Examiner:** "Prove that Grover's algorithm is optimal."

**Expected Proof Outline:**

**BBBV Theorem:** Any quantum algorithm needs $\Omega(\sqrt{N})$ queries.

**Proof idea:**
1. Consider distinguishing $f \equiv 0$ from $f$ with one solution at unknown $w$
2. Track $D_t = \mathbb{E}_w[\||\psi^{(f_w)}_t\rangle - |\psi^{(f_0)}_t\rangle\|^2]$
3. Each query increases $D_t$ by at most $O(1/N)$ on average
4. After $T$ queries: $D_T \leq O(T^2/N)$
5. To distinguish with constant probability: $\sqrt{D_T} = \Omega(1)$
6. Therefore: $T^2/N = \Omega(1)$, so $T = \Omega(\sqrt{N})$

**Conclusion:** Grover matches lower bound - it's optimal!

---

## Part III: Comparison and Synthesis

### Question 9: Shor vs. Grover Comparison
**Examiner:** "Compare Shor's and Grover's algorithms. What fundamentally differs?"

**Expected Analysis:**

| Aspect | Shor | Grover |
|--------|------|--------|
| Speedup | Exponential | Quadratic |
| Problem type | Structured | Unstructured |
| Key technique | QFT/Phase estimation | Amplitude amplification |
| Optimal? | Unknown | Yes (BBBV) |
| Exploits | Algebraic structure | Nothing |

**Fundamental difference:**
- Shor: Number-theoretic structure enables superposition to "find" period
- Grover: No structure, just interference enhancement

**Why different speedups:**
- Structured problems may hide exponential shortcuts
- Unstructured search has fundamental quantum limit

---

### Question 10: Which is More Important?
**Examiner:** "Which algorithm provides stronger evidence for quantum computational advantage?"

**Expected Response:**

**Case for Shor:**
- Exponential speedup is dramatic
- Breaks real cryptography
- Suggests hidden structure in mathematics
- No known classical algorithm comes close

**Case for Grover:**
- Proved optimal (clean theoretical result)
- Generalizes to amplitude amplification
- Works for any search problem
- Quadratic speedup is guaranteed

**My answer:** Shor provides stronger evidence because:
1. Exponential > quadratic
2. Solves a problem believed hard classically
3. Implies quantum computers access fundamentally different resources

However, Grover is more theoretically clean (proven optimal).

---

## Part IV: Problem-Solving Scenarios

### Scenario 1: Complexity Question
**Examiner:** "How many qubits would Shor's algorithm need to break 2048-bit RSA?"

**Expected Calculation:**

$n = 2048$ bits for $N$.

Basic algorithm:
- Phase estimation: $2n = 4096$ qubits
- Arithmetic register: $n = 2048$ qubits
- Ancilla: $O(n) \approx 2000$ qubits
- **Total:** $\approx 8000$ logical qubits

With error correction:
- Assume 1000:1 physical to logical ratio
- **Physical qubits:** $\approx 8 \times 10^6$

Current state-of-art: $\sim 1000$ physical qubits.

**Timeline estimate:** Need $\sim 10,000\times$ more qubits, likely 10-20 years.

---

### Scenario 2: Algorithm Design
**Examiner:** "How would you modify Grover's algorithm if you don't know how many solutions exist?"

**Expected Response:**

**Problem:** Unknown number $t$ of solutions.

**Solution 1: Quantum Counting**
1. Use phase estimation on Grover operator $G$
2. $G$ has eigenvalues $e^{\pm 2i\theta}$ where $\sin\theta = \sqrt{t/N}$
3. Estimate $\theta$, deduce $t$
4. Run Grover with correct iteration count

**Solution 2: Exponential Search**
1. Try $k = 1, 2, 4, 8, \ldots$ iterations
2. Check if solution found
3. Expected overhead: $O(\log N)$ factor

**Solution 3: Fixed-point Amplitude Amplification**
- Modified algorithm that converges to solution monotonically
- No oscillation, works for any $t$

---

### Scenario 3: Error Analysis
**Examiner:** "What happens to Shor's algorithm if the phase estimation has some error?"

**Expected Analysis:**

Phase estimation error $\epsilon$ means we get $\tilde{\theta}$ with $|\tilde{\theta} - s/r| < \epsilon$.

**Continued fractions recovery:**
- Works if $\epsilon < 1/(2r^2)$
- For $r < N$, need $\epsilon < 1/(2N^2)$
- This requires $2\log_2 N$ bits of precision

**With $t = 2n$ ancilla qubits:**
- Precision is $1/2^{2n} < 1/(2N^2)$
- Continued fractions succeeds

**If precision is insufficient:**
- May get wrong period $r'$
- Check: $a^{r'} \stackrel{?}{\equiv} 1 \pmod{N}$
- If fails, retry

**Error tolerance:** Algorithm is robust to moderate phase estimation errors.

---

## Part V: Mock Oral Exam (25 minutes)

### Complete Exam Script

**Opening (3 min):**
"Today I want to examine your understanding of quantum algorithms. Let's start with Shor's algorithm. What problem does it solve and why does that matter?"

**Shor Deep Dive (8 min):**
1. "Explain the reduction from factoring to order finding."
2. "How does phase estimation find the order?"
3. "Walk me through the continued fractions step."
4. "What's the complete complexity?"

**Grover Deep Dive (8 min):**
1. "Explain Grover's algorithm geometrically."
2. "Derive the optimal number of iterations."
3. "Prove that Grover is optimal." (BBBV)
4. "What if there are multiple solutions?"

**Synthesis (4 min):**
1. "Compare the speedups in Shor vs. Grover."
2. "Which provides stronger evidence for quantum advantage?"

**Closing (2 min):**
"Any aspects you'd like to clarify or expand on?"

---

## Evaluation Rubric

### Excellent (A)
- Derives Shor's reduction correctly
- Explains phase estimation clearly
- Proves BBBV lower bound
- Makes insightful comparisons

### Good (B)
- Understands both algorithms
- Minor gaps in derivations
- Can outline BBBV proof
- Reasonable comparisons

### Adequate (C)
- Knows algorithm structure
- Cannot derive key results
- BBBV proof incomplete
- Limited synthesis

### Needs Work (D/F)
- Confuses algorithm components
- Major technical errors
- Cannot explain optimality
- No comparative understanding

---

## Self-Preparation Checklist

Before the exam, ensure you can:

- [ ] Derive factoring to order-finding reduction
- [ ] Explain eigenstates of modular exponentiation
- [ ] Calculate Shor's complexity from first principles
- [ ] Draw Grover's geometric picture
- [ ] Prove optimal iteration count
- [ ] Outline BBBV proof
- [ ] Compare Shor and Grover speedup mechanisms
- [ ] Discuss practical implementation challenges

---

*This oral practice guide covers Week 163 topics. For written problems, see Problem_Set.md.*
