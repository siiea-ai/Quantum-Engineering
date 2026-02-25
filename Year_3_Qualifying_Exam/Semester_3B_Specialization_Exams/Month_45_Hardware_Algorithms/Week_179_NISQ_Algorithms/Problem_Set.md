# Week 179: Problem Set - NISQ Algorithms

## Instructions

This problem set contains 28 problems covering VQE, QAOA, and error mitigation techniques. Problems are organized by topic and difficulty.

---

## Section A: VQE Fundamentals (Problems 1-8)

### Problem 1 [Level 1]
Consider a simple 2-qubit Hamiltonian:

$$\hat{H} = -\hat{Z}_1 - \hat{Z}_2 + 0.5\hat{X}_1\hat{X}_2$$

(a) What is the ground state energy? (Hint: diagonalize the 4×4 matrix)

(b) Propose a simple parameterized ansatz using RY gates and one CNOT.

(c) At what parameter values does your ansatz achieve the ground state?

---

### Problem 2 [Level 1]
The variational principle states $$E(\vec{\theta}) \geq E_0$$.

(a) Prove this inequality for any normalized trial state.

(b) Under what conditions does equality hold?

(c) Why is this principle useful even if we don't reach equality?

---

### Problem 3 [Level 2]
The H₂ molecule Hamiltonian in the STO-3G basis after Jordan-Wigner transformation can be written as:

$$\hat{H} = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0Z_1 + g_4 X_0X_1 + g_5 Y_0Y_1$$

(a) How many Pauli terms need to be measured?

(b) Group the terms by qubit-wise commuting sets.

(c) What is the minimum number of measurement bases required?

---

### Problem 4 [Level 2]
A hardware-efficient ansatz for 4 qubits has the form:

$$\hat{U}(\vec{\theta}) = \prod_{l=1}^{L}\left[\prod_{i=1}^{4}R_Y(\theta_{l,i})\cdot \text{CNOT}_{1,2}\cdot\text{CNOT}_{2,3}\cdot\text{CNOT}_{3,4}\right]$$

(a) How many parameters does this ansatz have for $$L$$ layers?

(b) Estimate the circuit depth in terms of $$L$$.

(c) For a device with T2 = 50 μs and gate time of 50 ns, what is the maximum practical $$L$$?

---

### Problem 5 [Level 2]
The gradient of the VQE energy with respect to parameter $$\theta_j$$ can be computed using the parameter-shift rule:

$$\frac{\partial E}{\partial\theta_j} = \frac{E(\theta_j + \pi/2) - E(\theta_j - \pi/2)}{2}$$

(a) Derive this rule for a gate $$e^{-i\theta G/2}$$ where $$G^2 = I$$.

(b) How many circuit evaluations are needed to compute the full gradient for 20 parameters?

(c) Compare to finite differences: which is preferable on noisy hardware?

---

### Problem 6 [Level 3]
UCCSD (Unitary Coupled Cluster Singles and Doubles) for a 4-electron system has excitation operators:

$$\hat{T}_1 = \sum_{i\in occ, a\in virt} t_i^a (\hat{a}_a^\dagger\hat{a}_i - \hat{a}_i^\dagger\hat{a}_a)$$

(a) For 4 spatial orbitals (8 spin-orbitals) with 4 electrons, how many single excitation amplitudes $$t_i^a$$ are there?

(b) How many double excitation amplitudes?

(c) Estimate the circuit depth for implementing UCCSD on this system.

---

### Problem 7 [Level 3]
Barren plateaus occur when the variance of the gradient vanishes exponentially with system size.

(a) For a random hardware-efficient ansatz on $$n$$ qubits, the variance of $$\partial_\theta E$$ scales as $$O(1/2^n)$$. How many shots are needed to estimate a gradient of $$10^{-5}$$ for $$n = 20$$?

(b) Propose three strategies to mitigate barren plateaus.

(c) Why do problem-inspired ansatze often avoid barren plateaus?

---

### Problem 8 [Level 3]
ADAPT-VQE builds the ansatz iteratively by adding operators from a pool.

(a) Describe the algorithm in pseudocode.

(b) For a molecular problem with $$M$$ operators in the pool, what is the computational cost per iteration?

(c) How does ADAPT-VQE compare to UCCSD in terms of circuit depth and accuracy?

---

## Section B: QAOA (Problems 9-16)

### Problem 9 [Level 1]
Consider MaxCut on a triangle graph (3 vertices, 3 edges).

(a) Write the cost Hamiltonian $$\hat{H}_C$$.

(b) What is the maximum cut value?

(c) What is the optimal solution(s)?

---

### Problem 10 [Level 1]
The QAOA mixing Hamiltonian is $$\hat{H}_M = \sum_i \hat{X}_i$$.

(a) What is the effect of $$e^{-i\beta\hat{H}_M}$$ on the computational basis state $$|z\rangle$$?

(b) For $$\beta = \pi/4$$, what is the state after applying the mixer to $$|000\rangle$$?

(c) Why is this called a "mixing" Hamiltonian?

---

### Problem 11 [Level 2]
For MaxCut QAOA at $$p=1$$ on a 2-node, 1-edge graph:

(a) Write the QAOA state $$|\gamma, \beta\rangle$$ explicitly.

(b) Compute $$\langle\gamma, \beta|\hat{H}_C|\gamma, \beta\rangle$$.

(c) Find the optimal $$\gamma^*, \beta^*$$ analytically.

---

### Problem 12 [Level 2]
The cost Hamiltonian for a QUBO problem $$x^TQx$$ is:

$$\hat{H}_C = \sum_{i<j} Q_{ij}\frac{(1-\hat{Z}_i)(1-\hat{Z}_j)}{4} + \sum_i Q_{ii}\frac{1-\hat{Z}_i}{2}$$

(a) Verify this encoding maps $$x_i \in \{0,1\}$$ to $$Z_i = \pm 1$$ correctly.

(b) For $$Q = \begin{pmatrix} 1 & -2 \\ -2 & 1 \end{pmatrix}$$, write the Hamiltonian explicitly.

(c) What is the ground state of this Hamiltonian?

---

### Problem 13 [Level 2]
QAOA approximation ratios depend on the problem structure.

(a) For MaxCut on 3-regular graphs at $$p=1$$, the approximation ratio is $$\geq 0.6924$$. What does this mean?

(b) The Goemans-Williamson classical algorithm achieves 0.878. Is QAOA competitive at $$p=1$$?

(c) How does the approximation ratio improve with $$p$$?

---

### Problem 14 [Level 3]
Implement QAOA for portfolio optimization with 3 assets.

(a) Given expected returns $$\mu = (0.1, 0.2, 0.15)$$ and covariance matrix:
$$\Sigma = \begin{pmatrix} 0.01 & 0.005 & 0.002 \\ 0.005 & 0.02 & 0.01 \\ 0.002 & 0.01 & 0.015 \end{pmatrix}$$

Write the objective function for minimizing risk with target return $$R = 0.15$$.

(b) Convert to QUBO form with a penalty for the return constraint.

(c) How many QAOA layers would you expect to need for good solutions?

---

### Problem 15 [Level 3]
Warm-start QAOA initializes from a classical solution.

(a) If a classical algorithm provides solution $$z^*$$, how should the initial state be modified?

(b) How should the mixer Hamiltonian be adapted?

(c) What are the advantages and disadvantages of warm-starting?

---

### Problem 16 [Level 3]
Analyze the QAOA parameter landscape.

(a) For MaxCut on a specific graph, explain why optimal parameters often exhibit patterns (periodicity, symmetry).

(b) How can concentration of optimal parameters be exploited for larger instances?

(c) Discuss the role of overparameterization ($$p$$ too large) in QAOA.

---

## Section C: Error Mitigation (Problems 17-24)

### Problem 17 [Level 1]
Zero-noise extrapolation measures the expectation value at different noise levels.

(a) If $$E(1) = 0.45$$, $$E(2) = 0.40$$, $$E(3) = 0.35$$, use linear extrapolation to estimate $$E(0)$$.

(b) If the true value is $$E_{ideal} = 0.52$$, what is the error of your estimate?

(c) Why might a nonlinear extrapolation be more accurate?

---

### Problem 18 [Level 1]
Gate folding increases effective noise by replacing $$G$$ with $$GG^\dagger G$$.

(a) Why does this increase the noise level?

(b) If the base circuit has 10 CNOT gates, how many CNOTs are in the 3× noise-scaled circuit?

(c) What error model does gate folding assume?

---

### Problem 19 [Level 2]
Probabilistic error cancellation (PEC) decomposes the noisy inverse.

For a depolarizing channel $$\mathcal{D}_p(\rho) = (1-p)\rho + p \cdot I/2$$:

(a) Write the inverse map as a linear combination of Pauli channels.

(b) Calculate the quasi-probability overhead $$\gamma$$ for $$p = 0.01$$.

(c) For a circuit with 50 gates, what is the sampling overhead?

---

### Problem 20 [Level 2]
Symmetry verification post-selects on correct quantum numbers.

(a) For an $$N$$-electron molecular simulation, what symmetry can be verified by measuring $$\sum_i Z_i$$?

(b) If 30% of shots give incorrect particle number, what is the effective post-selection rate?

(c) How does this affect the statistical uncertainty?

---

### Problem 21 [Level 2]
Virtual distillation uses multiple copies to suppress errors.

(a) For a noisy state $$\rho = (1-p)|\psi\rangle\langle\psi| + p\frac{I}{2}$$, show that $$\text{Tr}(\rho^2) = 1 - p + p^2/2$$.

(b) How does $$\langle O\rangle = \text{Tr}(O\rho^2)/\text{Tr}(\rho^2)$$ suppress errors compared to $$\text{Tr}(O\rho)$$?

(c) What circuit is needed to implement two-copy virtual distillation?

---

### Problem 22 [Level 3]
Design an error mitigation strategy for VQE on a 4-qubit Hamiltonian.

(a) The device has $$T_1 = 100$$ μs, $$T_2 = 50$$ μs, and single-qubit gate error $$\epsilon_1 = 0.1\%$$, two-qubit gate error $$\epsilon_2 = 1\%$$.

(b) The VQE circuit has 20 single-qubit gates and 15 CNOT gates. Estimate the total error.

(c) Propose a combination of ZNE and symmetry verification. Estimate the improvement.

---

### Problem 23 [Level 3]
Compare the resource requirements of different error mitigation techniques.

For a circuit with $$L$$ two-qubit gates and physical error rate $$p$$:

(a) ZNE with 3 noise levels: how many circuit runs per energy evaluation?

(b) PEC: what is the sampling overhead $$\gamma^L$$ for $$p = 0.01$$?

(c) For what circuit depths is each technique practical?

---

### Problem 24 [Level 3]
Error mitigation has limits.

(a) For what error rates does ZNE extrapolation become unreliable?

(b) PEC overhead grows exponentially with circuit depth. At what depth does it become impractical (overhead > 1000)?

(c) How do these limits compare to the thresholds for error correction?

---

## Section D: Integration (Problems 25-28)

### Problem 25 [Level 2]
Match algorithms to hardware platforms.

| Algorithm | Best Platform | Reasoning |
|-----------|---------------|-----------|
| VQE for H₂O | | |
| QAOA for 100-node MaxCut | | |
| VQE-UCCSD for Fe-porphyrin | | |
| QAOA p=1 for portfolio | | |

Platforms: Superconducting (IBM), Trapped Ion (Quantinuum), Neutral Atom (QuEra)

---

### Problem 26 [Level 3]
Design a complete VQE experiment for LiH at equilibrium geometry.

(a) Active space: 4 electrons in 4 orbitals. How many qubits (after symmetry reduction)?

(b) Choose an ansatz. Justify your choice.

(c) Estimate the number of shots needed for chemical accuracy (1 mHa).

(d) Propose error mitigation strategy.

---

### Problem 27 [Level 3]
Analyze the quantum utility demonstration (IBM, 2023).

(a) The experiment ran 127-qubit circuits with 60 layers. Estimate the total gate count.

(b) Error mitigation (ZNE + other techniques) was essential. Why wasn't error correction used?

(c) What does "utility" mean in this context? Is it the same as "advantage"?

---

### Problem 28 [Level 3]
Propose an experiment to demonstrate quantum advantage for a chemistry problem.

(a) What molecular system would you choose? Why?

(b) What ansatz and error mitigation would you use?

(c) How would you verify that classical simulation cannot match the quantum result?

(d) What are the main technical challenges?

---

*Solutions are provided in Problem_Solutions.md*
