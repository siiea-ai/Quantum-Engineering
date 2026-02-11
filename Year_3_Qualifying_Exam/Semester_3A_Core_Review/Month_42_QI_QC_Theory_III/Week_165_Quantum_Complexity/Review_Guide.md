# Week 165: Quantum Complexity Theory - Review Guide

## Introduction

Quantum complexity theory studies the computational power of quantum computers relative to classical computers. This review covers the fundamental complexity classes BQP and QMA, their relationships to classical classes, and quantum query complexity—the study of how many oracle queries are needed to solve problems.

This material is essential for PhD qualifying examinations as it provides the theoretical foundations for understanding quantum computational advantage and its limitations.

---

## 1. Classical Complexity Background

### 1.1 Key Classical Complexity Classes

Before studying quantum complexity, recall the classical landscape:

**P (Polynomial Time):**
$$\text{P} = \bigcup_{k \geq 1} \text{DTIME}(n^k)$$

Languages decidable by a deterministic Turing machine in polynomial time.

**NP (Nondeterministic Polynomial Time):**
Languages with polynomial-time verifiable witnesses. Equivalently, languages decidable by a nondeterministic TM in polynomial time.

**BPP (Bounded-Error Probabilistic Polynomial Time):**
$$\text{BPP} = \{L : \exists \text{ probabilistic poly-time TM } M \text{ s.t.}$$
$$\Pr[M(x) = L(x)] \geq 2/3 \text{ for all } x\}$$

**PSPACE (Polynomial Space):**
Languages decidable using polynomial space (no time restriction).

**PP (Probabilistic Polynomial Time):**
Languages decidable by probabilistic TM with probability > 1/2 (arbitrarily close to 1/2).

**Key Containments:**
$$\text{P} \subseteq \text{NP} \subseteq \text{PP} \subseteq \text{PSPACE}$$
$$\text{P} \subseteq \text{BPP} \subseteq \text{PP} \subseteq \text{PSPACE}$$

### 1.2 Computational Models

**Turing Machine Model:**
- Deterministic, nondeterministic, probabilistic variants
- Polynomial-time Church-Turing thesis for classical computation

**Circuit Model:**
- Uniform circuit families {Cₙ}
- Size (number of gates), depth (longest path)
- Equivalence to TM model for uniform families

---

## 2. BQP: Bounded-Error Quantum Polynomial Time

### 2.1 Definition via Quantum Circuits

**Definition (BQP via Circuits):**

A language L is in BQP if there exists a uniform family of quantum circuits {Cₙ} such that:

1. Each Cₙ uses poly(n) gates from a finite universal gate set
2. For all x of length n:
   - If x ∈ L: Pr[Cₙ outputs 1 on input x] ≥ 2/3
   - If x ∉ L: Pr[Cₙ outputs 1 on input x] ≤ 1/3

**Formal Statement:**
$$\text{BQP} = \left\{L \subseteq \{0,1\}^* : \begin{array}{l} \exists \text{ uniform poly-size quantum circuit family } \{C_n\} \\ x \in L \Rightarrow \Pr[\text{measure } C_n|x\rangle|0^{m}\rangle = 1] \geq 2/3 \\ x \notin L \Rightarrow \Pr[\text{measure } C_n|x\rangle|0^{m}\rangle = 1] \leq 1/3 \end{array}\right\}$$

### 2.2 Robustness of Definition

**Error Reduction:**
The 2/3 and 1/3 bounds can be amplified to 1 - 2^{-poly(n)} and 2^{-poly(n)} by majority voting over poly(n) repetitions.

**Gate Set Independence:**
Any universal gate set yields the same class (Solovay-Kitaev theorem provides efficient approximation).

**Equivalence to Quantum Turing Machines:**
BQP defined via quantum Turing machines equals circuit-defined BQP.

### 2.3 Relationship to Classical Classes

**Theorem: P ⊆ BQP**

*Proof Sketch:*
Classical computation is a special case of quantum computation. Any classical circuit can be implemented using Toffoli gates, which are quantum gates. The Toffoli gate, together with preparation of |0⟩ and measurement in the computational basis, is universal for classical computation.

**Theorem: BQP ⊆ PSPACE**

*Proof Sketch:*
A quantum circuit on n qubits has a state vector of dimension 2ⁿ. Each amplitude can be computed to polynomial precision using polynomial space via path summation:

$$\langle y|C|x\rangle = \sum_{\text{paths } p} \prod_{\text{gates } g \in p} \langle \cdot|g|\cdot\rangle$$

Each path has polynomially many factors, each computable in polynomial space.

**Theorem: BQP ⊆ PP**

*Proof Sketch (Adleman-DeMarrais-Huang):*
Use the path-sum representation. The probability of outputting 1 is:

$$\Pr[1] = \sum_{y: y_1 = 1} |\langle y|C|0^n\rangle|^2$$

This can be expressed as a gap problem: count the number of accepting paths minus rejecting paths, weighted appropriately. This is exactly what PP computes.

### 2.4 Evidence for Separation from Classical Classes

**Oracle Separations:**

**Theorem (Bernstein-Vazirani):** There exists an oracle O such that BQP^O ⊄ BPP^O.

*Proof:* Simon's problem provides an oracle for which quantum algorithms need O(n) queries but classical algorithms need Ω(2^{n/2}) queries.

**Theorem:** There exists an oracle O such that NP^O ⊄ BQP^O.

*Proof:* The unstructured search problem (OR function) requires Ω(√N) quantum queries but has a polynomial-sized NP witness.

**Important Note:** These are oracle separations, not absolute separations. They provide evidence but not proof of P ≠ BQP.

### 2.5 Problems in BQP

**Complete Problems:**
Unlike P, no natural BQP-complete problems are known (would require proof that BQP ≠ BPP).

**Problems believed in BQP but not BPP:**
- Integer factoring
- Discrete logarithm
- Simulation of quantum systems (with promise on structure)

**Sampling Problems:**
Recent work focuses on sampling problems (BosonSampling, IQP, random circuit sampling) where quantum advantage is proven under plausible assumptions.

---

## 3. QMA: Quantum Merlin-Arthur

### 3.1 Definition

**Definition (QMA):**

A language L is in QMA if there exists a polynomial-time quantum verifier V such that:

**Completeness:** If x ∈ L, there exists a quantum witness |w⟩ on poly(|x|) qubits such that:
$$\Pr[V(x, |w\rangle) = 1] \geq 2/3$$

**Soundness:** If x ∉ L, for all quantum witnesses |w⟩:
$$\Pr[V(x, |w\rangle) = 1] \leq 1/3$$

### 3.2 Comparison with NP

| Aspect | NP | QMA |
|--------|----|----|
| Witness | Classical bit string | Quantum state |
| Verifier | Classical poly-time | Quantum poly-time |
| Witness size | Polynomial bits | Polynomial qubits |
| Verification | Deterministic | Probabilistic |

**Key Insight:** A quantum witness on n qubits encodes 2ⁿ complex amplitudes, potentially much richer than an n-bit classical string. However, the verifier can only extract polynomial information through measurement.

### 3.3 Properties of QMA

**Error Reduction:**
Like BQP, QMA errors can be reduced exponentially with polynomial overhead.

**Containments:**
$$\text{NP} \subseteq \text{QMA} \subseteq \text{PP} \subseteq \text{PSPACE}$$

**QMA vs. QCMA:**
QCMA uses quantum verifier but classical witness. It is unknown whether QMA = QCMA.

### 3.4 QMA-Complete Problems

**The Local Hamiltonian Problem:**

**Definition:** A k-local Hamiltonian on n qubits is:
$$H = \sum_{i=1}^{m} H_i$$

where each Hᵢ acts non-trivially on at most k qubits and ∥Hᵢ∥ ≤ poly(n).

**k-Local Hamiltonian Problem:**

*Input:* k-local Hamiltonian H, thresholds a, b with b - a ≥ 1/poly(n)

*Promise:* Either λ_min(H) ≤ a or λ_min(H) ≥ b

*Output:* Determine which case holds

**Theorem (Kitaev, 1999):** 5-local Hamiltonian is QMA-complete.

**Theorem (Kempe-Kitaev-Regev, 2006):** 2-local Hamiltonian is QMA-complete.

*Proof Sketch:*

1. **QMA-hardness:** Encode a QMA verification circuit into a Hamiltonian. The ground state encodes the "history" of the computation. Use perturbation theory to reduce locality.

2. **Containment in QMA:** The ground state serves as a witness. Phase estimation can estimate the ground state energy.

### 3.5 Other QMA-Complete Problems

- **Consistency of Local Density Matrices**
- **Quantum k-SAT** (for k ≥ 3)
- **N-representability problem** (quantum chemistry)
- **Quantum clique problem**

---

## 4. Query Complexity

### 4.1 The Oracle Model

**Setup:**
- Function f: {0,1}ⁿ → {0,1}ᵐ given as a black box
- Query: Apply unitary Oₓ|i,b⟩ = |i, b ⊕ xᵢ⟩
- Goal: Compute some property of f using minimum queries

**Query Complexity Q(f):**
Minimum number of queries needed by any quantum algorithm to compute f with probability ≥ 2/3.

**Classical Query Complexity D(f):**
Minimum queries for deterministic algorithms.

**Randomized Query Complexity R(f):**
Minimum queries for randomized algorithms with bounded error.

### 4.2 The Polynomial Method

**Theorem (Beals, Buhrman, Cleve, Mosca, de Wolf, 2001):**

Let A be a quantum algorithm making T queries to input x ∈ {0,1}ⁿ. Then:

1. The amplitude of any basis state is a multilinear polynomial in (x₁, ..., xₙ) of degree at most T.

2. The probability of any outcome is a polynomial of degree at most 2T.

*Proof Idea:*

Start with |0⟩⊗ᵐ. Each query is linear in the input variables:

$$O_f|i,b\rangle = |i, b \oplus f(i)\rangle$$

If we track amplitudes as polynomials in the input, each query increases degree by at most 1. Non-query operations don't increase degree (they're linear combinations).

**Corollary:**
$$Q(f) \geq \frac{\widetilde{\text{deg}}(f)}{2}$$

where deg̃(f) is the minimum degree of a polynomial p such that:
- p(x) ≥ 2/3 if f(x) = 1
- p(x) ≤ 1/3 if f(x) = 0

### 4.3 Grover Lower Bound

**Theorem:** Q(OR_n) = Ω(√n)

*Proof using Polynomial Method:*

1. Let ORₙ(x) = 1 iff at least one xᵢ = 1.

2. Any polynomial p approximating OR must satisfy:
   - p(0,0,...,0) ≤ 1/3
   - p(eᵢ) ≥ 2/3 for each standard basis vector eᵢ

3. By symmetry, we can assume p is symmetric: p(x) = q(|x|) for some univariate polynomial q.

4. For OR: q(0) ≤ 1/3, q(k) ≥ 2/3 for k ≥ 1.

5. The polynomial q must "jump" from ≤ 1/3 to ≥ 2/3 between 0 and 1, but then stay ≥ 2/3 for larger inputs.

6. Using Chebyshev polynomial analysis: deg(q) = Ω(√n).

7. Therefore: Q(ORₙ) ≥ deg̃(ORₙ)/2 = Ω(√n).

**Conclusion:** Grover's algorithm is optimal up to constants.

### 4.4 The Adversary Method

An alternative technique for lower bounds, often giving tight results.

**Negative Weights Adversary (Spalek-Szegedy):**

$$\text{ADV}^{\pm}(f) = \max_{\Gamma} \frac{\|\Gamma\|}{\max_i \|\Gamma \circ D_i\|}$$

where Γ is a matrix indexed by (x,y) pairs with f(x) ≠ f(y), and Dᵢ is the indicator for xᵢ ≠ yᵢ.

**Theorem:** Q(f) = Θ(ADV±(f)) for total functions.

### 4.5 Important Oracle Problems

| Problem | Quantum | Classical | Separation |
|---------|---------|-----------|------------|
| Deutsch-Jozsa | O(1) | Ω(2^{n-1}+1) | Exponential |
| Bernstein-Vazirani | O(1) | Ω(n) | Polynomial |
| Simon | O(n) | Ω(2^{n/2}) | Exponential |
| Grover | O(√N) | Ω(N) | Polynomial |
| Ordered Search | O(log N) | O(log N) | None |

---

## 5. Information-Theoretic Limits

### 5.1 Holevo's Bound and Query Complexity

**Connection:** The Holevo bound limits classical information extractable from quantum states, which impacts what can be learned from oracle queries.

### 5.2 Lower Bounds from Information Theory

**One-Way Communication Complexity:**
Query complexity lower bounds can be derived from communication complexity lower bounds.

**Information vs. Query Trade-offs:**
For many problems, information-theoretic arguments give matching upper and lower bounds.

---

## 6. Recent Developments

### 6.1 Quantum Supremacy/Advantage

**Random Circuit Sampling (Google, 2019):**
Demonstrated quantum sampling task taking ~200 seconds on quantum device vs. estimated 10,000 years classically.

**BosonSampling:**
Photonic experiments demonstrating quantum advantage in sampling from photon distributions.

### 6.2 BQP vs. PH

**Theorem (Raz-Tal, 2019):** There exists an oracle O such that BQP^O ⊄ PH^O.

This strengthens evidence that quantum computing offers advantages beyond the polynomial hierarchy.

### 6.3 Quantum Algorithms for Classical Problems

- **Quantum machine learning algorithms**
- **Quantum simulation of physical systems**
- **Quantum optimization heuristics (QAOA, VQE)**

---

## 7. Exam Preparation Tips

### Key Definitions to Know Cold

1. BQP definition (circuit model)
2. QMA definition (verifier + witness)
3. Local Hamiltonian problem statement
4. Query complexity model
5. Polynomial method statement

### Key Proofs to Master

1. P ⊆ BQP ⊆ PSPACE
2. Grover lower bound via polynomial method
3. Local Hamiltonian is in QMA (phase estimation argument)

### Common Exam Questions

1. "Define BQP and explain why BQP ⊆ PSPACE"
2. "What is the Local Hamiltonian problem? Why is it important?"
3. "Prove that any quantum algorithm needs Ω(√N) queries for unstructured search"
4. "Compare and contrast NP and QMA"

### Pitfalls to Avoid

- Confusing oracle separations with absolute separations
- Forgetting that QMA-completeness requires both hardness and containment
- Misapplying the polynomial method (degree of probability, not amplitude)

---

## 8. Summary

### Core Results

| Result | Statement |
|--------|-----------|
| BQP containment | P ⊆ BPP ⊆ BQP ⊆ PP ⊆ PSPACE |
| QMA containment | NP ⊆ QMA ⊆ PP ⊆ PSPACE |
| Local Hamiltonian | k-local Hamiltonian is QMA-complete for k ≥ 2 |
| Grover optimality | Q(OR_n) = Θ(√n) |
| Polynomial method | Q(f) ≥ deg̃(f)/2 |

### Open Problems

1. Is BQP = BPP? (Is quantum faster than classical?)
2. Is QMA = QCMA? (Do quantum witnesses help?)
3. What is the exact relationship between BQP and NP?

---

## References

1. Watrous, J. "Quantum Computational Complexity." *Encyclopedia of Complexity and Systems Science* (2009).

2. Kempe, J., Kitaev, A., and Regev, O. "The Complexity of the Local Hamiltonian Problem." *SIAM Journal on Computing* 35(5), 2006.

3. Beals, R., et al. "Quantum Lower Bounds by Polynomials." *Journal of the ACM* 48(4), 2001.

4. Nielsen, M.A. and Chuang, I.L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2010.

5. Aaronson, S. "Quantum Computing Since Democritus." Cambridge University Press, 2013.

---

**Word Count:** ~2500 words
**Created:** February 9, 2026
