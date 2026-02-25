# Week 163: Quantum Algorithms II - Problem Set

## 30 Problems on Shor's Algorithm, Grover's Algorithm, and Optimality

---

## Section A: Shor's Algorithm - Foundations (Problems 1-10)

### Problem 1: Reduction to Order Finding
a) State the reduction from factoring to order finding precisely.

b) For $N = 15$ and $a = 7$, compute the order $r$ of $a$ modulo $N$.

c) Verify that $\gcd(7^{r/2} - 1, 15)$ and $\gcd(7^{r/2} + 1, 15)$ give the factors of 15.

d) For which values of $a$ would the reduction fail for $N = 15$?

---

### Problem 2: Success Probability
a) Prove that for $N = pq$ (product of distinct odd primes), the probability that a random $a$ gives a usable order is at least $1/2$.

b) What is this probability for $N = p^2$ (prime power)?

c) How many repetitions are needed to factor $N$ with probability $\geq 0.99$?

---

### Problem 3: Modular Exponentiation Unitary
Define $U_a|x\rangle = |ax \mod N\rangle$ for $0 \leq x < N$.

a) Prove that $U_a$ is unitary.

b) Show that $U_a^r = I$ where $r = \text{ord}_N(a)$.

c) Prove that the eigenvalues of $U_a$ are exactly $\{e^{2\pi ik/r} : k = 0, 1, \ldots, r-1\}$.

---

### Problem 4: Eigenstates of Modular Exponentiation
For the eigenstate $|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}e^{-2\pi isk/r}|a^k \mod N\rangle$:

a) Verify that $U_a|u_s\rangle = e^{2\pi is/r}|u_s\rangle$.

b) Prove that $\sum_{s=0}^{r-1}|u_s\rangle = \sqrt{r}|1\rangle$.

c) Explain why this sum property is crucial for the algorithm.

---

### Problem 5: Phase Estimation in Shor's
a) After controlled modular exponentiation with input $|1\rangle$, what is the state?

b) Why does phase estimation give a random $s/r$ rather than a specific eigenvalue?

c) How many bits of precision are needed for the phase estimation?

---

### Problem 6: Continued Fractions
a) State the theorem about continued fraction convergents approximating rationals.

b) For $\tilde{\theta} = 0.428571$, find the continued fraction expansion.

c) If this approximates $s/r$ with $r = 7$, what are the possible values of $s$?

d) Write pseudocode for extracting $r$ from $\tilde{\theta}$.

---

### Problem 7: Complete Complexity Analysis
a) Count the qubits needed for Shor's algorithm on $N$ with $n = \lceil\log_2 N\rceil$ bits.

b) Analyze the gate complexity for modular exponentiation.

c) What is the total gate count using fast integer multiplication?

d) What is the circuit depth?

---

### Problem 8: Shor's on Small Examples
Apply Shor's algorithm to factor $N = 21$.

a) Choose $a = 2$ and find $\text{ord}_{21}(2)$.

b) Check if this order is usable.

c) Compute the factors using the algorithm.

d) Trace through what the quantum circuit would produce.

---

### Problem 9: Modular Exponentiation Circuit
a) Design a circuit for computing $|x\rangle|y\rangle \to |x\rangle|y \cdot a^x \mod N\rangle$ using repeated squaring.

b) How many modular multiplications are needed?

c) Analyze the depth of this circuit.

---

### Problem 10: Error Analysis
a) What happens if the phase estimation has error $\epsilon$ in the measured phase?

b) Derive the precision requirement to correctly identify $r$ with high probability.

c) How does the success probability depend on the precision?

---

## Section B: Grover's Algorithm - Foundations (Problems 11-18)

### Problem 11: Grover Operators
a) Write the matrix representation of the oracle $O_f$ for $n = 2$ qubits with $f(11) = 1$.

b) Write the matrix representation of the diffusion operator $D$ for $n = 2$.

c) Compute the Grover iterator $G = DO_f$.

d) Find the eigenvalues of $G$.

---

### Problem 12: Geometric Analysis
Consider the 2D subspace spanned by $|w\rangle$ and $|s'\rangle$.

a) Express $|s\rangle$ in terms of $|w\rangle$ and $|s'\rangle$.

b) Show that $O_f$ is a reflection about $|s'\rangle$ in this subspace.

c) Show that $D$ is a reflection about $|s\rangle$ in this subspace.

d) Prove that $G = DO_f$ is a rotation by angle $2\theta$.

---

### Problem 13: Optimal Iterations
a) Derive the formula for optimal number of iterations $k = \frac{\pi}{4}\sqrt{N}$.

b) For $N = 256$, what is the optimal $k$? What is the success probability?

c) What happens if we use $k + 1$ iterations instead of optimal $k$?

d) Graph the success probability as a function of $k$ for $N = 16$.

---

### Problem 14: Multiple Solutions
For $t$ solutions among $N$ items:

a) Derive the optimal number of iterations.

b) What is the success probability after optimal iterations?

c) If $t = N/4$, how many iterations are needed?

d) What goes wrong if $t > N/2$?

---

### Problem 15: Implementation Details
a) Implement the diffusion operator using only Hadamard, X, and multi-controlled Z gates.

b) What is the gate complexity of one Grover iteration?

c) What is the total gate complexity for optimal search?

---

### Problem 16: Quantum Counting
a) Explain how phase estimation on $G$ can estimate the number of solutions.

b) What precision in phase estimation is needed to estimate $t$ to within $\pm 1$?

c) Design an algorithm to find a solution when $t$ is unknown.

---

### Problem 17: Grover with Imperfect Oracle
Suppose the oracle has error: with probability $p$, it fails to mark a solution.

a) How does this affect the success probability after $k$ iterations?

b) Derive the modified optimal iteration count.

c) For what range of $p$ does Grover still give advantage over classical?

---

### Problem 18: Partial Search
Instead of searching for exact solution, suppose we want any $x$ in a "good" set $S$.

a) How does Grover's algorithm change?

b) What if $|S| = N/2$?

c) What if we can only approximately check membership in $S$?

---

## Section C: BBBV Theorem and Optimality (Problems 19-24)

### Problem 19: Query Complexity Model
a) Define the query complexity model precisely.

b) Why is query complexity the right measure for comparing quantum and classical search?

c) Give an example where gate complexity differs significantly from query complexity.

---

### Problem 20: BBBV Theorem Statement
a) State the BBBV theorem precisely.

b) What are the key assumptions?

c) Why doesn't this theorem apply to Shor's algorithm?

---

### Problem 21: BBBV Proof Outline
a) Define the hybrid states $|\psi^{(f_w)}_t\rangle$ and $|\psi^{(f_0)}_t\rangle$.

b) Show that a single query can change the state norm by at most $O(1/\sqrt{N})$ on average.

c) Use this to prove the $\Omega(\sqrt{N})$ lower bound.

---

### Problem 22: Adversary Method
The adversary method is an alternative proof technique for quantum lower bounds.

a) State the adversary lemma.

b) Apply it to prove $\Omega(\sqrt{N})$ for unstructured search.

c) What advantages does adversary method have over hybrid argument?

---

### Problem 23: Implications of BBBV
a) Prove that unstructured search cannot be solved in $O(N^{0.49})$ queries.

b) What does BBBV imply about $\text{NP} \subseteq \text{BQP}$?

c) Does BBBV rule out polynomial-time quantum algorithms for NP-complete problems?

---

### Problem 24: Tight Bounds
a) Show that Grover's algorithm matches the BBBV lower bound up to constants.

b) What is the exact constant in the lower bound?

c) Can Grover be improved by a constant factor?

---

## Section D: Amplitude Amplification and Applications (Problems 25-30)

### Problem 25: General Amplitude Amplification
a) State the general amplitude amplification theorem.

b) Prove that $O(1/\sin\theta)$ iterations suffice to amplify from $\sin^2\theta$ to $\Omega(1)$ success probability.

c) What are the requirements on the initial algorithm $\mathcal{A}$?

---

### Problem 26: Amplitude Estimation
a) Explain how phase estimation on the amplification operator estimates $\sin^2\theta$.

b) What precision is needed to estimate $\theta$ to additive error $\epsilon$?

c) Derive the query complexity of amplitude estimation.

---

### Problem 27: Element Distinctness
The element distinctness problem: given list of $N$ elements, determine if all are distinct.

a) What is the classical query complexity?

b) Using amplitude amplification with a collision-checking subroutine, derive a quantum algorithm.

c) The optimal quantum complexity is $\Theta(N^{2/3})$. Explain why naive Grover doesn't achieve this.

---

### Problem 28: Quantum Walk Speedup
a) Briefly explain how quantum walks differ from classical random walks.

b) How do quantum walks achieve the $O(N^{2/3})$ bound for element distinctness?

c) Give another example where quantum walks provide speedup.

---

### Problem 29: Synthesis Problem - Algorithm Design
Design a quantum algorithm for the following problem:

Given oracle access to $f:\{0,1\}^n \to \{0,1\}^n$, find $x \neq y$ such that $f(x) = f(y)$ (assuming one exists).

a) What is the classical query complexity?

b) Design an algorithm using amplitude amplification.

c) What is your algorithm's query complexity?

d) Is your algorithm optimal?

---

### Problem 30: Comprehensive Analysis
This is a qualifying exam-style comprehensive problem.

**Part A:** For Shor's algorithm on a 1024-bit RSA modulus:
1. Estimate the number of qubits needed.
2. Estimate the number of gates.
3. Estimate the run time assuming 1MHz gate speed.
4. Discuss error correction overhead.

**Part B:** For Grover's algorithm on a database of $2^{40}$ items:
1. Calculate optimal iterations.
2. Estimate total gate count.
3. Compare to classical time for searching.
4. Discuss practical limitations.

**Part C:** Compare Shor's and Grover's speedups:
1. Explain the fundamental difference in speedup type.
2. Which provides stronger evidence for quantum advantage?
3. Which is more practically relevant in the near term?

---

## Problem Difficulty Summary

| Difficulty | Problems | Focus |
|------------|----------|-------|
| Foundational | 1-3, 11-13 | Algorithm basics |
| Intermediate | 4-7, 14-17 | Analysis and derivations |
| Advanced | 8-10, 18-24 | Proofs and optimality |
| Research-Level | 25-30 | Synthesis and applications |

---

*Complete solutions are available in Problem_Solutions.md*
