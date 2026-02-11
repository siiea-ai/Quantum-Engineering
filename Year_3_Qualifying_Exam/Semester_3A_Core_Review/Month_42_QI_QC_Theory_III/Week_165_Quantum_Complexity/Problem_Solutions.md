# Week 165: Quantum Complexity Theory - Problem Solutions

## Section A: BQP Fundamentals

### Solution 1: BQP Definition

**Formal Definition:**

A language $L \subseteq \{0,1\}^*$ is in BQP if there exists a uniform family of polynomial-size quantum circuits $\{C_n\}_{n \in \mathbb{N}}$ such that for all $x$ with $|x| = n$:

$$x \in L \Rightarrow \Pr[\text{measuring first qubit of } C_n|x\rangle|0^{m(n)}\rangle \text{ gives } 1] \geq \frac{2}{3}$$

$$x \notin L \Rightarrow \Pr[\text{measuring first qubit of } C_n|x\rangle|0^{m(n)}\rangle \text{ gives } 1] \leq \frac{1}{3}$$

where $m(n) = \text{poly}(n)$ is the number of ancilla qubits.

**Parameters:**
- Completeness: $c = 2/3$
- Soundness: $s = 1/3$
- Gap: $c - s = 1/3$

**"Bounded error"** means the algorithm may err with probability at most 1/3. This error is two-sided: both false positives and false negatives can occur.

---

### Solution 2: P ⊆ BQP

**Proof:**

Let $L \in P$. Then there exists a polynomial-time deterministic Turing machine $M$ deciding $L$.

**Step 1:** Convert $M$ to a classical Boolean circuit family $\{D_n\}$ of size $\text{poly}(n)$ (by the standard TM-to-circuit simulation).

**Step 2:** The classical circuit uses AND, OR, NOT gates. Replace these with reversible Toffoli gates:
- NOT: $|x\rangle \mapsto |\neg x\rangle$ (just use X gate)
- AND: $|a,b,0\rangle \mapsto |a,b,a \wedge b\rangle$ (Toffoli gate)
- OR: Use De Morgan: $a \vee b = \neg(\neg a \wedge \neg b)$

**Step 3:** The resulting reversible circuit is a valid quantum circuit. Since Toffoli can be decomposed into Clifford+T gates, we have a polynomial-size quantum circuit.

**Step 4:** The quantum circuit computes deterministically (no superposition needed), so:
$$x \in L \Rightarrow \Pr[\text{output } 1] = 1 \geq 2/3$$
$$x \notin L \Rightarrow \Pr[\text{output } 1] = 0 \leq 1/3$$

Therefore $L \in BQP$. $\square$

---

### Solution 3: Error Reduction in BQP

**Claim:** $\text{BQP}_{1/3, 2/3} = \text{BQP}_{2^{-n}, 1-2^{-n}}$

**Proof:**

Let $L \in BQP$ with circuit $C_n$ having error probability $\leq 1/3$.

**Amplification procedure:**
1. Run $C_n$ independently $k = O(n)$ times
2. Take the majority vote of the $k$ outcomes

**Analysis using Chernoff bound:**

Let $X_i$ be the indicator for the $i$-th run giving the correct answer. Then $\mathbb{E}[X_i] \geq 2/3$.

Let $S = \sum_{i=1}^{k} X_i$. By Chernoff:

$$\Pr[S \leq k/2] \leq e^{-2k(\frac{2}{3} - \frac{1}{2})^2} = e^{-k/18}$$

Setting $k = 18n$ gives error $\leq e^{-n} < 2^{-n}$.

**Importance:** Error reduction shows BQP is robust—the specific error bounds don't matter as long as there's a constant gap.

---

### Solution 4: Gate Set Independence

**Claim:** BQP is independent of the universal gate set.

**Explanation:**

The Solovay-Kitaev theorem states that any single-qubit unitary $U$ can be approximated to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates from any universal set (where $c \approx 3.97$).

For a circuit with $\text{poly}(n)$ gates:
- Each gate needs precision $\epsilon = 1/\text{poly}(n)$
- This requires $O(\log^c(\text{poly}(n))) = O(\log^c(n))$ gates per original gate
- Total overhead: $\text{poly}(n) \cdot O(\log^c(n)) = \text{poly}(n)$

The accumulated error is at most $\text{poly}(n) \cdot \epsilon = O(1)$ by choosing $\epsilon$ small enough.

---

### Solution 5: BQP ⊆ PSPACE

**Proof:**

Let $L \in BQP$ with quantum circuit $C$ of size $s = \text{poly}(n)$ on $m = \text{poly}(n)$ qubits.

**Goal:** Compute $\Pr[C \text{ outputs } 1]$ using polynomial space.

**Path-sum formula:**

$$\langle y | C | 0^m \rangle = \sum_{\text{paths } p} \prod_{i=1}^{s} \langle p_{i} | G_i | p_{i-1} \rangle$$

where the sum is over all $2^{ms}$ paths through intermediate computational basis states.

**PSPACE algorithm:**
1. For each output $y$ with $y_1 = 1$:
   - Enumerate all paths $p$ (exponentially many, but done sequentially)
   - For each path, compute the product of matrix elements
   - Keep running sum (polynomial space for the amplitude)
2. Square the amplitude and sum over all accepting $y$

**Space analysis:**
- Current path: $O(ms) = O(\text{poly}(n))$ bits
- Each matrix element: $O(\text{poly}(n))$ bits
- Running sum: $O(\text{poly}(n))$ bits

Total: $O(\text{poly}(n))$ space. $\square$

---

### Solution 6: BQP ⊆ PP

**Proof Sketch (Adleman-DeMarrais-Huang):**

PP decides languages by probabilistic polynomial-time machines where acceptance requires probability $> 1/2$ (not $\geq 2/3$).

**Key insight:** The acceptance probability of a quantum circuit can be written as:

$$\Pr[\text{accept}] = \frac{N_+}{N_+ + N_-}$$

where $N_+$ and $N_-$ are sums of weights (possibly negative) over computational paths.

**PP simulation:**
1. Create a polynomial-time probabilistic TM that samples a random path
2. The probability of accepting path minus rejecting path encodes the quantum acceptance probability
3. The "gap" between $>1/2$ and $\leq 1/2$ can encode whether quantum probability is $\geq 2/3$ or $\leq 1/3$

The technical details involve careful accounting of positive and negative contributions. $\square$

---

### Solution 7: BQP vs BPP

**Relationship:** $\text{BPP} \subseteq \text{BQP}$ (proven, since random coin flips can be simulated quantumly)

**Open questions:**
- Is $\text{BQP} = \text{BPP}$? Unknown and widely believed to be NO
- Is $\text{BQP} \supsetneq \text{BPP}$? Not proven

**Evidence for $\text{BQP} \neq \text{BPP}$:**
1. Oracle separations (Simon's problem)
2. Problems like factoring with quantum speedups
3. Quantum sampling advantages (BosonSampling)

**Why it's not proven:** Proving circuit lower bounds is notoriously hard (relates to P vs NP).

---

### Solution 8: Oracle Separations

**Definition:** An oracle separation between classes A and B means there exists an oracle O such that $A^O \neq B^O$.

**Example:** Simon's oracle gives $\text{BQP}^O \neq \text{BPP}^O$.

**Why it doesn't prove BQP ≠ BPP:**

The Baker-Gill-Solovay theorem shows oracles can both separate and collapse complexity classes. Oracle results prove the techniques currently used cannot resolve the question.

Specifically: if $P = NP$, it would require non-relativizing proof techniques (proofs that don't work for all oracles). Similarly for BQP vs BPP.

---

### Solution 9: Factoring in BQP

**Shor's Algorithm:**
- Input: $N$ to factor
- Quantum subroutine: Find period of $f(x) = a^x \mod N$ using QFT
- Classical post-processing: Use period to find factors

**Complexity:** $O((\log N)^3)$ quantum operations = polynomial in input size.

**Why believed not in BPP:**
- Best classical algorithm: General Number Field Sieve with $\exp(O(n^{1/3} \log^{2/3} n))$
- Sub-exponential but not polynomial
- No polynomial classical algorithm known despite decades of research

**Implication:** If factoring is hard classically (widely believed), then $\text{BQP} \neq \text{BPP}$.

---

### Solution 10: Quantum Sampling

**Definition:** A sampling problem asks to output samples from a distribution $D$ over $\{0,1\}^n$ (not decide membership).

**Quantum sampling problems:**
- BosonSampling: Sample from output distribution of linear optical network
- Random circuit sampling: Sample from output of random quantum circuit

**Why evidence for quantum advantage:**

These problems are in "SampBQP" (quantum polynomial-time sampling). Under complexity assumptions (like non-collapse of polynomial hierarchy), these cannot be efficiently simulated classically.

**Key theorem (Aaronson-Arkhipov):** If BosonSampling can be classically simulated, then $P^{\#P} = BPP^{NP}$, collapsing PH.

---

## Section B: QMA and QMA-Completeness

### Solution 11: QMA Definition

**Formal Definition:**

A language $L$ is in QMA if there exists a polynomial-time quantum verifier $V$ such that:

**Completeness:** $x \in L \Rightarrow \exists |w\rangle \in (\mathbb{C}^2)^{\otimes p(n)}: \Pr[V(x,|w\rangle) = 1] \geq 2/3$

**Soundness:** $x \notin L \Rightarrow \forall |w\rangle: \Pr[V(x,|w\rangle) = 1] \leq 1/3$

**Comparison with NP:**

| Aspect | NP | QMA |
|--------|----|----|
| Witness | Classical string $w \in \{0,1\}^{p(n)}$ | Quantum state $\|w\rangle \in (\mathbb{C}^2)^{\otimes p(n)}$ |
| Verifier | Deterministic poly-time | Quantum poly-time |
| Error | None (perfect soundness) | Two-sided bounded error |
| Witness encoding | $p(n)$ bits | $2^{p(n)}$ amplitudes |

**Key difference:** Quantum witnesses encode exponentially more information, but only polynomial information is extractable via measurement.

---

### Solution 12: QMA Error Reduction

**Challenge:** Cannot simply repeat with same witness (witness might be entangled across runs).

**Solution (Marriott-Watrous):**

Use in-place amplification within a single verification:

1. Start with witness $|w\rangle$
2. Apply phase estimation to verifier's "accept" projector $\Pi_{\text{acc}}$
3. This estimates $\|\Pi_{\text{acc}}|w,0\rangle\|^2$ which equals acceptance probability

**Analysis:**
- If $x \in L$: good witness gives acceptance prob $\geq 2/3$, phase estimation detects this
- If $x \notin L$: all witnesses give acceptance prob $\leq 1/3$, phase estimation detects this

Amplification to exponentially small error uses polynomial overhead. $\square$

---

### Solution 13: QMA ⊆ PP

**Proof:**

Let $L \in QMA$ with verifier $V$.

**PP algorithm:**
1. Enumerate over all possible witnesses $|w\rangle$ (continuous, handled by discretization)
2. For each discretized witness, run the BQP ⊆ PP simulation
3. Accept if any witness leads to high acceptance

**Technical details:**
- Discretize witness space to precision $1/\text{poly}(n)$ (finite set)
- Sum acceptance probabilities over all witnesses
- Use PP's $> 1/2$ threshold to determine if any good witness exists

This works because the gap between $2/3$ and $1/3$ is constant. $\square$

---

### Solution 14: Local Hamiltonian Problem

**Definition:**

A $k$-local Hamiltonian on $n$ qubits is $H = \sum_{i=1}^{m} H_i$ where:
- Each $H_i$ acts non-trivially on at most $k$ qubits
- Each $H_i$ is Hermitian with $\|H_i\| \leq \text{poly}(n)$
- $m = \text{poly}(n)$

**Problem Statement:**

*Input:* $k$-local Hamiltonian $H$, thresholds $a < b$ with $b - a \geq 1/\text{poly}(n)$

*Promise:* Either $\lambda_{\min}(H) \leq a$ (YES) or $\lambda_{\min}(H) \geq b$ (NO)

*Output:* Determine which case

**Why gap is necessary:**
Computing $\lambda_{\min}(H)$ exactly is undecidable (equivalent to Halting problem for infinite systems). The gap makes the problem decidable and in QMA.

---

### Solution 15: Local Hamiltonian in QMA

**Proof:**

**Witness:** Ground state $|\psi_0\rangle$ of $H$ (when $x \in L$)

**Verification:**
1. Use phase estimation on $e^{-iHt}$ to estimate $\langle\psi|H|\psi\rangle$
2. Accept if estimated energy $\leq (a+b)/2$

**Completeness:** If $\lambda_{\min}(H) \leq a$, ground state gives energy $\leq a < (a+b)/2$. Accept with high probability.

**Soundness:** If $\lambda_{\min}(H) \geq b$, all states give energy $\geq b > (a+b)/2$. Reject with high probability.

**Technical issue:** Phase estimation on $e^{-iHt}$ requires simulating $H$-evolution, which is efficient for local Hamiltonians (Hamiltonian simulation algorithms). $\square$

---

### Solution 16: QMA-Hardness Intuition

**History State Construction:**

Given QMA verifier circuit $C = U_T \cdots U_1$ with witness $|w\rangle$:

**Define history state:**
$$|\psi_{\text{hist}}\rangle = \frac{1}{\sqrt{T+1}} \sum_{t=0}^{T} |t\rangle \otimes U_t \cdots U_1 |w,0^a\rangle$$

This encodes the entire computation history.

**Hamiltonian construction:**
$$H = H_{\text{in}} + H_{\text{out}} + H_{\text{prop}} + H_{\text{clock}}$$

- $H_{\text{in}}$: Penalizes wrong initial state
- $H_{\text{out}}$: Penalizes rejection at end
- $H_{\text{prop}}$: Penalizes incorrect propagation (not applying $U_t$ correctly)
- $H_{\text{clock}}$: Enforces valid clock states

**Key insight:** Low energy ⟺ correct computation with accepting output.

---

### Solution 17: 2-Local vs 5-Local

**Significance of locality reduction:**

Physical Hamiltonians are typically 2-local (pairwise interactions). Showing 2-local Hamiltonian is QMA-complete means:
- Ground state problems for physical systems are hard
- No polynomial-time algorithm for general 2-local ground states (unless QMA = P)

**Perturbation theory approach (Kempe-Kitaev-Regev):**

1. Start with 5-local Hamiltonian from circuit construction
2. Use "gadgets" - perturbative constructions that simulate higher-locality terms with lower-locality Hamiltonians
3. In low-energy subspace, effective Hamiltonian matches the 5-local one

The construction preserves the spectral gap, maintaining QMA-completeness. $\square$

---

## Section C: Query Complexity

### Solution 18: Query Model Definition

**Query Oracle:**

For function $f: \{0,1\}^n \to \{0,1\}$, the query oracle is the unitary:
$$O_f |i, b\rangle = |i, b \oplus f(x_i)\rangle$$

where $x_i$ is the $i$-th bit of input $x$.

**Quantum Query Algorithm:**
1. Start with fixed initial state $|0\rangle^{\otimes m}$
2. Alternate between query oracles $O_f$ and fixed unitaries $U_0, U_1, \ldots, U_T$
3. Measure and output

**Query Complexity $Q(f)$:**
Minimum $T$ such that some algorithm computes $f$ with probability $\geq 2/3$ using at most $T$ queries.

---

### Solution 19: Polynomial Method Statement

**Theorem (Beals et al., 2001):**

Let $A$ be a quantum algorithm making $T$ queries to input $x \in \{0,1\}^n$. Then:

1. The amplitude of any basis state $|b\rangle$ after the algorithm is a multilinear polynomial in $(x_1, \ldots, x_n)$ of degree at most $T$.

2. The probability of any measurement outcome is a polynomial of degree at most $2T$.

**Corollary:**
$$Q(f) \geq \frac{\widetilde{\deg}(f)}{2}$$

where $\widetilde{\deg}(f)$ is the minimum degree of a polynomial $p$ satisfying:
- $|p(x) - f(x)| \leq 1/3$ for all $x$

---

### Solution 20: Polynomial Method Proof Sketch

**Proof by induction on number of queries:**

**Base case (T=0):** No queries made. State is $U_0|0\rangle^{\otimes m}$, which is independent of $x$. Amplitudes are degree-0 polynomials. ✓

**Inductive step:** Assume after $t$ queries, each amplitude is degree-$t$ polynomial.

After $t$-th query, apply $U_{t+1}$ (query-free): linear combination of amplitudes, still degree $\leq t$.

Apply $(t+1)$-th query $O_f$:
$$O_f|i,b\rangle = |i, b \oplus x_i\rangle$$

The amplitude of $|i,b\rangle$ becomes:
$$\alpha_{i,b \oplus x_i} = x_i \cdot \alpha_{i,1-b} + (1-x_i) \cdot \alpha_{i,b}$$

This is linear in $x_i$ times degree-$t$ polynomial = degree-$(t+1)$ polynomial.

**Probability:** Probability = $|\text{amplitude}|^2$, so degree at most $2T$. $\square$

---

### Solution 21: Approximate Degree

**Definition:**

The $\epsilon$-approximate degree of $f: \{0,1\}^n \to \{0,1\}$ is:

$$\widetilde{\deg}_\epsilon(f) = \min\{\deg(p) : |p(x) - f(x)| \leq \epsilon \text{ for all } x\}$$

The approximate degree is $\widetilde{\deg}(f) = \widetilde{\deg}_{1/3}(f)$.

**Relationship to query complexity:**
$$Q(f) \geq \frac{\widetilde{\deg}(f)}{2}$$

This is because any $T$-query algorithm produces acceptance probability that is a degree-$2T$ polynomial approximating $f$.

---

### Solution 22: Grover Lower Bound

**Theorem:** $Q(\text{OR}_n) = \Omega(\sqrt{n})$

**Proof:**

**Step 1:** By symmetry, we can assume the approximating polynomial is symmetric:
$$p(x_1, \ldots, x_n) = q(|x|)$$
where $|x| = \sum_i x_i$ is the Hamming weight.

**Step 2:** Requirements for $q$:
- $q(0) \leq 1/3$ (no 1's means OR = 0)
- $q(k) \geq 2/3$ for $k \geq 1$ (any 1 means OR = 1)

**Step 3:** Consider the polynomial $r(t) = q(t) - 1/2$. Then:
- $r(0) \leq -1/6$
- $r(k) \geq 1/6$ for $k \geq 1$

**Step 4:** The polynomial $r$ must cross from negative to positive between 0 and 1, and stay positive. By Chebyshev polynomial theory, any polynomial that achieves this oscillation must have degree $\Omega(\sqrt{n})$.

**Formal argument:** The polynomial $r(t)$ restricted to $\{0, 1, \ldots, n\}$ must satisfy:
- $|r(0) - (-1/6)| \leq 0$
- $|r(k) - 1/6| \leq 0$ for $k \geq 1$

Using the Markov brothers' inequality or Chebyshev polynomial bounds:
$$\deg(r) \geq c\sqrt{n}$$

for some constant $c > 0$.

**Step 5:** Therefore $\deg(q) = \Omega(\sqrt{n})$, so $\widetilde{\deg}(\text{OR}) = \Omega(\sqrt{n})$.

**Step 6:** By the polynomial method:
$$Q(\text{OR}_n) \geq \frac{\widetilde{\deg}(\text{OR}_n)}{2} = \Omega(\sqrt{n})$$

Combined with Grover's $O(\sqrt{n})$ upper bound: $Q(\text{OR}_n) = \Theta(\sqrt{n})$. $\square$

---

### Solution 23: AND Lower Bound

**Proof:**

Note that $\text{AND}(x_1, \ldots, x_n) = \neg\text{OR}(\neg x_1, \ldots, \neg x_n)$.

**Claim:** $\widetilde{\deg}(\text{AND}) = \widetilde{\deg}(\text{OR})$

**Proof of claim:** If $p$ approximates OR, then $1 - p(1-x_1, \ldots, 1-x_n)$ approximates AND with the same degree.

Therefore:
$$Q(\text{AND}_n) \geq \frac{\widetilde{\deg}(\text{AND}_n)}{2} = \frac{\widetilde{\deg}(\text{OR}_n)}{2} = \Omega(\sqrt{n})$$

$\square$

---

### Solution 24: Parity Lower Bound

**Theorem:** $Q(\text{PARITY}_n) = n$ (exact)

**Proof:**

**Upper bound:** Query all $n$ bits and compute XOR classically. Uses exactly $n$ queries.

**Lower bound:** Consider input $x$ vs. $x \oplus e_i$ (flipping bit $i$). These have opposite parity.

After $T < n$ queries, there exists a bit $x_j$ never queried. The algorithm's state is identical for inputs differing only in bit $j$.

But PARITY differs on these inputs, so the algorithm cannot distinguish them. Contradiction.

**Formal polynomial argument:**

$\text{PARITY}(x) = x_1 \oplus \ldots \oplus x_n = \prod_{i=1}^n (-1)^{x_i}$ (over $\{-1,+1\}$)

In real polynomial form: $\text{PARITY}(x) = \prod_{i=1}^n (1 - 2x_i) \pmod{2}$

This polynomial has exact degree $n$, and no lower-degree polynomial can represent parity exactly or approximately. $\square$

---

### Solution 25: Element Distinctness

**a) Classical complexity: $\Theta(n)$**

Lower bound: Must read almost all elements (adversary can hide collision in unread elements).
Upper bound: Sort in $O(n \log n)$ comparisons, but comparison model allows $O(n)$ with hashing.

**b) Quantum algorithm achieving $O(n^{2/3})$ (Ambainis):**

Uses quantum walk on Johnson graph:
1. Maintain a set $S$ of $r = n^{2/3}$ sampled elements
2. Quantum walk checks for collision within $S$
3. Walk spreads over subsets, finding collision if one exists

Query complexity: $O(n^{2/3})$ for setup + walk iterations.

**c) Why intermediate hardness:**

- Easier than full search: structure (pairs must match) can be exploited
- Harder than OR: checking specific pair requires querying both elements
- The $n^{2/3}$ bound is tight (proven by adversary method)

---

## Section D: Oracle Problems and Separations

### Solution 26: Deutsch-Jozsa Analysis

**Problem:** Given $f: \{0,1\}^n \to \{0,1\}$ promised constant or balanced, determine which.

**a) Quantum query complexity: O(1)**

Single query suffices:
1. Prepare $H^{\otimes n}|0^n\rangle \otimes H|1\rangle$
2. Query $f$ via phase oracle
3. Apply $H^{\otimes n}$ to first register
4. Measure: all zeros ⟺ constant

**b) Deterministic classical: $\Omega(2^{n-1} + 1)$**

Must query $2^{n-1} + 1$ points to distinguish (adversary could make any subset of $\leq 2^{n-1}$ consistent with either case).

**c) Randomized classical: O(1) with high probability**

Query 2 random points: if different outputs, definitely balanced. Repeat $O(1)$ times for high confidence. (Note: this gives BPP separation but not exact.)

**d) Quantum algorithm:**

$$|0^n\rangle|1\rangle \xrightarrow{H^{\otimes(n+1)}} \frac{1}{\sqrt{2^n}}\sum_x |x\rangle \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

$$\xrightarrow{O_f} \frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

$$\xrightarrow{H^{\otimes n}} \sum_y c_y |y\rangle$$

where $c_{0^n} = \frac{1}{2^n}\sum_x (-1)^{f(x)}$.

If constant: $c_{0^n} = \pm 1$, so measure $0^n$ with certainty.
If balanced: $c_{0^n} = 0$, so never measure $0^n$.

---

### Solution 27: Simon's Problem

**Problem:** Given $f: \{0,1\}^n \to \{0,1\}^n$ with $f(x) = f(y) \Leftrightarrow x \oplus y \in \{0^n, s\}$, find $s$.

**a) Quantum query complexity: O(n)**

Algorithm:
1. Prepare $|0^n\rangle|0^n\rangle$, apply Hadamards to first register
2. Query $f$: $\sum_x |x\rangle|f(x)\rangle$
3. Measure second register (collapses to uniform superposition over $\{x, x \oplus s\}$)
4. Apply Hadamard to first register, measure
5. Get vector $y$ orthogonal to $s$: $y \cdot s = 0 \pmod{2}$
6. Repeat $O(n)$ times, solve linear system for $s$

**b) Classical complexity:**

Deterministic: $\Omega(2^{n/2})$ (birthday paradox bound)
Randomized: $\Omega(2^{n/2})$ (must find collision)

**c) Significance:**

First problem with exponential quantum-classical separation.
Inspired Shor's algorithm (both use period-finding structure).
Provides oracle evidence that $\text{BQP} \neq \text{BPP}$.

---

### Solution 28: Forrelation

**a) Definition:**

Given $f, g: \{0,1\}^n \to \{-1, +1\}$, compute whether they are "forrelated":

$$\Phi(f,g) = \frac{1}{2^n} \sum_x f(x) \tilde{g}(x)$$

where $\tilde{g}$ is the Fourier transform of $g$.

Promise: Either $|\Phi(f,g)| \geq 3/5$ or $|\Phi(f,g)| \leq 1/100$.

**b) Quantum query complexity: O(1)**

Single query: Apply Hadamard, query $f$, apply Hadamard, query $g$, measure.

**c) Classical lower bound:**

Any classical algorithm needs $\Omega(2^{n/2})$ queries (Aaronson-Ambainis).

**d) Implications for BQP vs PH:**

Forrelation provides evidence that $\text{BQP} \not\subseteq \text{PH}$ (polynomial hierarchy). Combined with relativized results (Raz-Tal 2019), there exists oracle $O$ with $\text{BQP}^O \not\subseteq \text{PH}^O$.

---

## Section E: Advanced Problems

### Solution 29: Complexity Zoo Navigation

**a) P ⊆ NP:** Known and proven (deterministic TM is special case of NTM).

**b) P = BQP?:** Open. Widely believed to be NO but not proven.

**c) BQP ⊆ NP?:** Open. Believed to be NO (factoring is in BQP but not known in NP).

**d) NP ⊆ BQP?:** Open. Believed to be NO (unstructured search requires $\sqrt{N}$ quantum queries but NP has polynomial witnesses).

**e) BQP = QMA?:** Open. Believed to be NO (analogous to P vs NP).

**f) QMA ⊆ NEXP?:** Known. QMA ⊆ PP ⊆ PSPACE ⊆ NEXP.

**g) BQP ⊆ PH?:** Open. Recent evidence (Raz-Tal oracle separation) suggests NO.

---

### Solution 30: Research Frontier

**a) Meaning of BQP ∩ NP-hard ≠ ∅:**

Would mean quantum computers solve NP-hard problems efficiently, implying:
- NP ⊆ BQP (major breakthrough)
- Cryptography based on NP-hardness would be broken

**b) Evidence:**

*Against:* Grover's lower bound suggests no super-quadratic speedup for NP problems. NP-complete problems like 3-SAT show no exponential quantum speedup.

*For:* Some structured problems might be in both BQP and NP-hard. No proven separation.

**c) Quantum PCP conjecture:**

Classical PCP theorem: NP = PCP[O(log n), O(1)] (verification with few random bits and queries).

Quantum PCP: Does QMA have similar structure? Would imply approximating Local Hamiltonian is QMA-hard, connecting to condensed matter physics.

**d) If factoring is NP-hard:**

Would imply BQP ∩ NP-hard ≠ ∅ (since factoring ∈ BQP).
This is unexpected: factoring is believed in NP ∩ coNP, suggesting not NP-hard (unless NP = coNP).

---

**Solutions Complete**

**Created:** February 9, 2026
