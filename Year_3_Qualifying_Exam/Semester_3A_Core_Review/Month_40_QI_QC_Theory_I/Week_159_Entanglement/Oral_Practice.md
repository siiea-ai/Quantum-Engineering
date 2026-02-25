# Week 159: Entanglement - Oral Examination Practice

## Conceptual Questions

### Q1: What is entanglement and why is it important?

**Model Answer:**

Entanglement is a quantum correlation between particles that has no classical analog. A bipartite pure state is entangled if it cannot be written as a product state: $$|\psi\rangle_{AB} \neq |a\rangle_A \otimes |b\rangle_B$$.

**Key features:**
1. **Non-local correlations**: Measurement outcomes are correlated in ways that cannot be explained by local hidden variables
2. **Reduced state mixedness**: For a pure entangled state, the reduced density matrices are mixed
3. **No classical communication**: Entanglement alone cannot send information

**Importance in quantum information:**
1. **Quantum teleportation**: Transmit quantum states using entanglement + classical communication
2. **Superdense coding**: Send 2 classical bits using 1 qubit + shared entanglement
3. **Quantum cryptography**: Security guaranteed by Bell inequality violations
4. **Quantum computing**: Entanglement is necessary for quantum speedup

---

### Q2: Explain the CHSH inequality and its significance.

**Model Answer:**

The CHSH inequality tests whether nature obeys local hidden variable theories.

**Setup:**
- Two parties, Alice and Bob, share particles
- Each makes one of two possible measurements (outcomes $$\pm 1$$)
- They compute correlations $$E(a,b)$$

**The inequality:**
For local hidden variables: $$|S| \leq 2$$

where $$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

**Quantum violation:**
Quantum mechanics predicts $$|S|_{\max} = 2\sqrt{2} \approx 2.83$$

This is achieved with maximally entangled states and optimal measurement angles.

**Significance:**
1. **Foundational**: Rules out local hidden variable explanations of quantum mechanics
2. **Experimental**: Loophole-free Bell tests confirm quantum predictions
3. **Practical**: Basis for device-independent quantum cryptography
4. **Conceptual**: Nature is either non-local or non-realistic (or both)

---

### Q3: What is the PPT criterion and when is it useful?

**Model Answer:**

The PPT (Positive Partial Transpose) criterion is a necessary condition for separability.

**Definition:**
The partial transpose $$\rho^{T_B}$$ is obtained by transposing only the B subsystem indices.

**Criterion:**
- If $$\rho$$ is separable, then $$\rho^{T_B} \geq 0$$ (all eigenvalues non-negative)
- Contrapositive: If $$\rho^{T_B}$$ has negative eigenvalues, $$\rho$$ is entangled

**When it's useful:**
1. **Entanglement detection**: Negative eigenvalue proves entanglement
2. **Computable**: Easy to implement numerically
3. **Quantification**: Negativity is based on PPT

**Limitations:**
- For $$2 \times 2$$ and $$2 \times 3$$: PPT $$\Leftrightarrow$$ separable (complete)
- For larger systems: PPT is necessary but not sufficient
- **Bound entangled states** exist: PPT but not separable

---

### Q4: Compare different entanglement measures.

**Model Answer:**

**Entanglement Entropy (pure states):**
$$E = S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$
- Unique measure for pure bipartite states
- Equals Schmidt entropy
- Maximum: $$\log_2(\min(d_A, d_B))$$

**Concurrence (two qubits):**
$$C = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$
- Analytically computable for two-qubit mixed states (Wootters formula)
- Ranges from 0 (separable) to 1 (Bell states)
- Related to entanglement of formation

**Negativity:**
$$\mathcal{N} = \sum_i (|\mu_i| - \mu_i)/2$$
- Based on partial transpose eigenvalues
- Computable for any dimension
- Zero for PPT states (misses bound entanglement)

**Comparison:**
| Measure | Domain | Computability | Meaning |
|---------|--------|---------------|---------|
| Entropy | Pure states | Easy | Mixedness of parts |
| Concurrence | Two qubits | Moderate | Formation cost |
| Negativity | Any state | Easy | PPT violation |

All good measures satisfy: zero for separable, positive for entangled, non-increasing under LOCC.

---

### Q5: Explain the Bell states and their properties.

**Model Answer:**

The four Bell states form a complete orthonormal basis for two-qubit systems:

$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)$$
$$|\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

**Properties:**

1. **Maximally entangled:**
   - Schmidt coefficients: $$\lambda_1 = \lambda_2 = 1/\sqrt{2}$$
   - Entanglement entropy: 1 ebit (maximum)
   - Concurrence: 1

2. **Maximally mixed reduced states:**
   $$\rho_A = \rho_B = I/2$$

3. **Local unitary equivalence:**
   Related by single-qubit Pauli operations on one qubit

4. **Perfect correlations:**
   - $$|\Phi^+\rangle$$: Same outcomes in $$Z$$ basis, same in $$X$$ basis
   - $$|\Psi^-\rangle$$: Opposite outcomes in any basis (singlet)

5. **Applications:**
   - Quantum teleportation
   - Superdense coding
   - Bell inequality tests
   - Entanglement swapping

---

## Technical Questions

### Q6: Derive the quantum prediction for CHSH violation with the singlet state.

**Model Answer:**

**Singlet state:** $$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

**Measurement:** $$\vec{a} \cdot \vec{\sigma}$$ on A, $$\vec{b} \cdot \vec{\sigma}$$ on B

**Correlation function:**
$$E(a,b) = \langle\Psi^-|(\vec{a}\cdot\vec{\sigma}) \otimes (\vec{b}\cdot\vec{\sigma})|\Psi^-\rangle$$

For the singlet:
$$E(a,b) = -\vec{a} \cdot \vec{b} = -\cos\theta_{ab}$$

**Optimal angles:**
- $$\vec{a}$$: 0°
- $$\vec{a}'$$: 90°
- $$\vec{b}$$: 45°
- $$\vec{b}'$$: 135°

**CHSH value:**
$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$
$$= -\cos 45° + \cos 135° - \cos 45° - \cos 45°$$
$$= -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} = -2\sqrt{2}$$

$$|S| = 2\sqrt{2} > 2$$ (violates classical bound)

---

### Q7: Calculate concurrence for a pure two-qubit state.

**Model Answer:**

For $$|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$$:

**Step 1:** Compute spin-flipped state
$$|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)|\psi^*\rangle$$

$$\sigma_y|0\rangle = i|1\rangle, \quad \sigma_y|1\rangle = -i|0\rangle$$

$$|\tilde{\psi}\rangle = -a^*|11\rangle + b^*|10\rangle + c^*|01\rangle - d^*|00\rangle$$

**Step 2:** Compute overlap
$$\langle\psi|\tilde{\psi}\rangle = -ad^* - d^*a + bc^* + cb^* = 2(bc - ad)$$

**Step 3:** Concurrence
$$C = |\langle\psi|\tilde{\psi}\rangle| = 2|ad - bc|$$

**Examples:**
- Product state $$|00\rangle$$: $$C = 0$$
- Bell state: $$C = 2 \times \frac{1}{2} = 1$$

---

### Q8: How do you compute negativity from a density matrix?

**Model Answer:**

**Step 1:** Compute partial transpose $$\rho^{T_B}$$

For $$\rho_{AB}$$ in basis $$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$:
$$(\rho^{T_B})_{ij,kl} = \rho_{il,kj}$$

In matrix form: swap columns 2,3 and rows 2,3 appropriately.

**Step 2:** Find eigenvalues $$\{\mu_i\}$$ of $$\rho^{T_B}$$

**Step 3:** Compute negativity
$$\mathcal{N} = \sum_{i: \mu_i < 0} |\mu_i|$$

Equivalently:
$$\mathcal{N} = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

**Step 4:** Logarithmic negativity
$$E_N = \log_2(1 + 2\mathcal{N}) = \log_2\|\rho^{T_B}\|_1$$

**Example:** Bell state
Eigenvalues of $$\rho^{T_B}$$: $$\{1/2, 1/2, 1/2, -1/2\}$$
$$\mathcal{N} = 1/2$$, $$E_N = 1$$

---

## Oral Exam Tips

1. **Start with definitions**: Define entanglement precisely before discussing properties

2. **Use Bell states as examples**: They illustrate all key concepts

3. **Know the numbers**: $$2\sqrt{2} \approx 2.83$$ for Tsirelson bound

4. **Connect theory to applications**: Entanglement enables teleportation, cryptography

5. **Understand limitations**: PPT doesn't detect all entanglement; concurrence is for two qubits only

---

*Practice until you can explain these concepts clearly without notes.*
