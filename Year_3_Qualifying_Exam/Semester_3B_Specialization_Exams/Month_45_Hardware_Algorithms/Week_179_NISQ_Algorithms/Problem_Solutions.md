# Week 179: Problem Solutions - NISQ Algorithms

## Section A: VQE Fundamentals

### Solution 1

**(a) Ground State Energy:**

Matrix form of $$\hat{H} = -Z_1 - Z_2 + 0.5 X_1X_2$$:

$$H = \begin{pmatrix} -2 & 0 & 0 & 0.5 \\ 0 & 0 & 0.5 & 0 \\ 0 & 0.5 & 0 & 0 \\ 0.5 & 0 & 0 & 2 \end{pmatrix}$$

Eigenvalues: $$\{-2.06, -0.5, 0.5, 2.06\}$$

$$\boxed{E_0 = -\sqrt{4 + 0.25} \approx -2.06}$$

**(b) Simple Ansatz:**

$$|\psi(\theta)\rangle = \text{CNOT}_{12} \cdot R_Y(\theta)|00\rangle$$

$$= \text{CNOT}_{12}[\cos(\theta/2)|00\rangle + \sin(\theta/2)|10\rangle]$$

$$= \cos(\theta/2)|00\rangle + \sin(\theta/2)|11\rangle$$

**(c) Optimal Parameters:**

Ground state is approximately $$|00\rangle$$ with small admixture of $$|11\rangle$$.

At $$\theta \approx 0.24$$ (close to 0), the ansatz approximates the ground state.

$$\boxed{\theta \approx 0.24 \text{ rad}}$$

---

### Solution 2

**(a) Proof of Variational Principle:**

Let $$|\psi\rangle = \sum_n c_n|E_n\rangle$$ where $$|E_n\rangle$$ are eigenstates with energies $$E_n$$.

$$\langle\psi|\hat{H}|\psi\rangle = \sum_n |c_n|^2 E_n \geq E_0\sum_n |c_n|^2 = E_0$$

since $$E_n \geq E_0$$ and $$\sum_n|c_n|^2 = 1$$.

$$\boxed{E(\vec{\theta}) \geq E_0 \text{ for any normalized state}}$$

**(b) Equality Condition:**

Equality holds iff $$|\psi\rangle = |E_0\rangle$$ (ground state exactly).

**(c) Practical Utility:**

Even without reaching equality:
- We get an upper bound on $$E_0$$
- Monotonically approaching true value
- Useful for relative comparisons and chemistry

---

### Solution 3

**(a) Number of Pauli Terms:**

$$\hat{H} = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0Z_1 + g_4 X_0X_1 + g_5 Y_0Y_1$$

$$\boxed{6 \text{ terms}}$$

**(b) QWC Grouping:**

Group 1 (Z basis): $$\{I, Z_0, Z_1, Z_0Z_1\}$$ - measure both qubits in Z

Group 2 (X basis): $$\{X_0X_1\}$$ - measure both in X

Group 3 (Y basis): $$\{Y_0Y_1\}$$ - measure both in Y

**(c) Minimum Bases:**

$$\boxed{3 \text{ measurement bases}}$$

---

### Solution 4

**(a) Number of Parameters:**

$$4 \times L$$ parameters (one RY per qubit per layer).

$$\boxed{4L \text{ parameters}}$$

**(b) Circuit Depth:**

Each layer: 4 RY gates (depth 1) + 3 CNOTs (depth 3)

Total depth $$\approx 4L$$

$$\boxed{\text{Depth} \approx 4L}$$

**(c) Maximum Practical L:**

Circuit time = $$4L \times 50$$ ns = $$200L$$ ns

For coherence-limited operation: time $$< T_2 = 50$$ μs

$$200L < 50000$$ ns → $$L < 250$$

Practical limit (fidelity considerations): $$L \sim 10-50$$

$$\boxed{L_{max} \sim 50 \text{ (practical)}}$$

---

### Solution 5

**(a) Parameter-Shift Rule Derivation:**

For gate $$U(\theta) = e^{-i\theta G/2}$$ with $$G^2 = I$$:

$$\langle\hat{O}\rangle(\theta) = \langle\psi|U^\dagger(\theta)\hat{O}U(\theta)|\psi\rangle$$

Taking derivative and using $$e^{\pm i\pi G/4} = \frac{1}{\sqrt{2}}(I \pm iG)$$:

$$\frac{\partial\langle O\rangle}{\partial\theta} = \frac{\langle O\rangle(\theta + \pi/2) - \langle O\rangle(\theta - \pi/2)}{2}$$

$$\boxed{\frac{\partial E}{\partial\theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}}$$

**(b) Circuit Evaluations:**

For 20 parameters: $$2 \times 20 = 40$$ evaluations

$$\boxed{40 \text{ circuit evaluations}}$$

**(c) Comparison:**

Parameter-shift gives exact gradients (for ideal circuits).

Finite differences: $$\partial_\theta E \approx \frac{E(\theta + \epsilon) - E(\theta)}{\epsilon}$$

On noisy hardware, parameter-shift is more robust (larger $$\epsilon$$ equivalent).

$$\boxed{\text{Parameter-shift is preferable: exact formula, less noise sensitivity}}$$

---

### Solution 6

**(a) Single Excitations:**

Occupied: 4 electrons in 4 spin-orbitals (2 spatial × 2 spin)

Virtual: 4 spin-orbitals

Singles: $$4 \times 4 = 16$$ amplitudes

$$\boxed{16 \text{ single excitation amplitudes}}$$

**(b) Double Excitations:**

Doubles: $$\binom{4}{2} \times \binom{4}{2} = 6 \times 6 = 36$$ amplitudes

$$\boxed{36 \text{ double excitation amplitudes}}$$

**(c) Circuit Depth:**

Each excitation requires O(n) CNOT gates after JW transformation.

52 excitations × ~10 CNOTs each = ~500 CNOTs

$$\boxed{\text{Depth} \sim 500-1000 \text{ gates}}$$

---

### Solution 7

**(a) Shot Requirements:**

To estimate gradient $$g = 10^{-5}$$ with variance $$\text{Var}[g] \sim 1/2^{20} \approx 10^{-6}$$:

Standard error: $$\sigma = 1/2^{10} \approx 10^{-3}$$

Shots needed: $$N = (σ/g)^2 = (10^{-3}/10^{-5})^2 = 10^{10}$$

$$\boxed{N \sim 10^{10} \text{ shots (impractical)}}$$

**(b) Barren Plateau Mitigation:**

1. **Layerwise training:** Train shallow circuits first, then add layers
2. **Problem-inspired ansatze:** Use structure that avoids random unitaries
3. **Local cost functions:** Measure local observables instead of global

**(c) Problem-Inspired Ansatze:**

They don't explore the full Hilbert space uniformly—they stay near physically relevant subspaces where gradients don't vanish.

---

### Solution 8

**(a) ADAPT-VQE Pseudocode:**

```
Initialize: |ψ⟩ = |HF⟩, ansatz = identity
Repeat:
  For each operator A in pool:
    Compute gradient: g_A = d⟨H⟩/dθ |_{θ=0} for e^{iθA}
  Select A* with largest |g_A|
  Add A* to ansatz
  Optimize all parameters in ansatz
Until: max|g_A| < threshold
Return: optimized ansatz and energy
```

**(b) Computational Cost:**

Per iteration: $$O(M)$$ gradient evaluations, each requiring 2 circuit runs

Plus re-optimization: $$O(k^2)$$ evaluations for $$k$$ parameters

Total per iteration: $$O(M + k^2)$$

$$\boxed{O(M + k^2) \text{ per iteration}}$$

**(c) Comparison:**

- ADAPT: typically 10-30% of UCCSD depth for similar accuracy
- Problem-adapted rather than fixed structure
- May require many iterations for complex systems

---

## Section B: QAOA

### Solution 9

**(a) Cost Hamiltonian:**

For triangle with edges $$(1,2), (2,3), (1,3)$$:

$$\hat{H}_C = \frac{1}{2}(1 - Z_1Z_2) + \frac{1}{2}(1 - Z_2Z_3) + \frac{1}{2}(1 - Z_1Z_3)$$

$$= \frac{3}{2} - \frac{1}{2}(Z_1Z_2 + Z_2Z_3 + Z_1Z_3)$$

$$\boxed{\hat{H}_C = \frac{3}{2} - \frac{1}{2}(Z_1Z_2 + Z_2Z_3 + Z_1Z_3)}$$

**(b) Maximum Cut:**

$$C_{max} = 2$$ (can cut at most 2 edges in a triangle)

$$\boxed{C_{max} = 2}$$

**(c) Optimal Solutions:**

$$|001\rangle, |010\rangle, |100\rangle, |110\rangle, |101\rangle, |011\rangle$$

(Any partition with one vertex on one side, two on the other)

---

### Solution 10

**(a) Mixer Effect:**

$$e^{-i\beta H_M}|z\rangle = \prod_i e^{-i\beta X_i}|z_i\rangle$$

$$= \prod_i[\cos\beta|z_i\rangle - i\sin\beta|1-z_i\rangle]$$

Mixes each bit with its complement.

**(b) State After Mixer:**

$$e^{-i\pi X/4}|0\rangle = \cos(\pi/4)|0\rangle - i\sin(\pi/4)|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$$

$$e^{-i\pi H_M/4}|000\rangle = \frac{1}{2\sqrt{2}}(|0\rangle - i|1\rangle)^{\otimes 3}$$

$$\boxed{\frac{1}{2\sqrt{2}}(|0\rangle - i|1\rangle)^{\otimes 3}}$$

**(c) Mixing:**

It creates superpositions between computational basis states, enabling quantum exploration of the solution space.

---

### Solution 11

**(a) QAOA State:**

$$|+\rangle^{\otimes 2} = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

$$e^{-i\gamma H_C}|+\rangle^{\otimes 2} = \frac{1}{2}(e^{-i\gamma/2}|00\rangle + e^{i\gamma/2}|01\rangle + e^{i\gamma/2}|10\rangle + e^{-i\gamma/2}|11\rangle)$$

After mixer... (complex expression)

**(b) Expectation Value:**

$$F(\gamma,\beta) = \frac{1}{2}[1 + \sin(2\beta)\sin(2\gamma)]$$

**(c) Optimal Parameters:**

Maximum at $$\sin(2\beta) = 1$$ and $$\sin(2\gamma) = 1$$:

$$\boxed{\gamma^* = \pi/4, \quad \beta^* = \pi/4}$$

---

### Solution 12

**(a) Encoding Verification:**

$$x_i = 0$$ maps to $$Z_i = +1$$: $$(1-Z_i)/2 = 0$$

$$x_i = 1$$ maps to $$Z_i = -1$$: $$(1-Z_i)/2 = 1$$

Correct encoding: $$x_i = (1-Z_i)/2$$ ✓

**(b) Explicit Hamiltonian:**

$$Q = \begin{pmatrix} 1 & -2 \\ -2 & 1 \end{pmatrix}$$

$$\hat{H}_C = \frac{(1-Z_1)(1-Z_2)}{4} \cdot (-2) + \frac{1-Z_1}{2} \cdot 1 + \frac{1-Z_2}{2} \cdot 1$$

$$= -\frac{1}{2}(1 - Z_1 - Z_2 + Z_1Z_2) + \frac{1}{2}(1 - Z_1) + \frac{1}{2}(1 - Z_2)$$

$$= \frac{1}{2} - \frac{Z_1Z_2}{2}$$

$$\boxed{\hat{H}_C = \frac{1}{2}(1 - Z_1Z_2)}$$

**(c) Ground State:**

Minimum when $$Z_1Z_2 = 1$$: $$|00\rangle$$ or $$|11\rangle$$

$$\boxed{|00\rangle \text{ or } |11\rangle}$$

---

### Solution 13

**(a) Approximation Ratio:**

Ratio $$r \geq 0.6924$$ means QAOA output $$C_{QAOA} \geq 0.6924 \cdot C_{max}$$

For any 3-regular graph, QAOA at $$p=1$$ achieves at least 69.24% of optimal.

**(b) Comparison:**

GW: 87.8% vs QAOA $$p=1$$: 69.2%

QAOA at $$p=1$$ is not competitive with best classical algorithms.

$$\boxed{\text{No, QAOA } p=1 \text{ underperforms GW}}$$

**(c) Improvement with p:**

- $$p=2$$: ratio improves to ~0.75
- $$p \rightarrow \infty$$: approaches 1 (adiabatic limit)
- Improvement is gradual and problem-dependent

---

### Solution 14

**(a) Objective Function:**

Risk: $$\frac{1}{2}x^T\Sigma x$$

With Lagrange multiplier for return constraint:

$$\min \frac{1}{2}x^T\Sigma x + \lambda(R - \mu^Tx)^2$$

**(b) QUBO Form:**

Binary variables: $$x_i \in \{0,1\}$$ (invest or not in asset $$i$$)

$$Q_{ij} = \frac{1}{2}\Sigma_{ij} + \lambda\mu_i\mu_j \text{ for } i \neq j$$

$$Q_{ii} = \frac{1}{2}\Sigma_{ii} + \lambda(\mu_i^2 - 2R\mu_i)$$

**(c) QAOA Layers:**

For 3 assets: $$p = 2-3$$ layers should give reasonable solutions.

$$\boxed{p \sim 2-3}$$

---

### Solution 15

**(a) Initial State Modification:**

Instead of $$|+\rangle^{\otimes n}$$, start with:

$$|\psi_0\rangle = \bigotimes_i[\cos\theta_i|0\rangle + \sin\theta_i|1\rangle]$$

where $$\theta_i$$ biases toward $$z_i^*$$.

**(b) Mixer Adaptation:**

Use XY-mixer to preserve constraint structure:

$$H_M = \sum_{i<j}(X_iX_j + Y_iY_j)$$

Or problem-specific constraint-preserving mixer.

**(c) Advantages/Disadvantages:**

**Pros:** Faster convergence, better local optima

**Cons:** Dependent on classical solution quality, may miss global optima

---

### Solution 16

**(a) Parameter Patterns:**

Periodicity: $$\gamma$$ has period $$2\pi$$ in $$H_C$$ eigenvalues

Symmetry: $$(\gamma, \beta) \equiv (-\gamma, \pi-\beta)$$ often

Regularity: Optimal $$\gamma_p$$ often increase with $$p$$

**(b) Concentration Exploitation:**

Optimal parameters cluster for graphs of similar structure:
- Train on small instances
- Transfer to larger instances
- Reduces optimization overhead

**(c) Overparameterization:**

Too many layers:
- Increases noise and depth
- Marginal improvement in approximation
- Risk of barren plateaus
- Generally $$p \sim O(\sqrt{n})$$ is sufficient

---

## Section C: Error Mitigation

### Solution 17

**(a) Linear Extrapolation:**

Points: $$(1, 0.45), (2, 0.40), (3, 0.35)$$

Slope: $$m = -0.05$$

Intercept: $$E(0) = 0.45 + 0.05 = 0.50$$

$$\boxed{E(0) = 0.50}$$

**(b) Error:**

$$|0.50 - 0.52| = 0.02$$

$$\boxed{\text{Error} = 0.02}$$

**(c) Nonlinear Improvement:**

Exponential decay model $$E(\lambda) = E_0 + ae^{-b\lambda}$$ may capture saturation effects better for low noise.

---

### Solution 18

**(a) Noise Increase:**

$$GG^\dagger G = G$$ ideally, but each gate application accumulates noise.

3 gates → 3× the noise of 1 gate.

**(b) CNOT Count:**

Base: 10 CNOTs

Folded (3×): Each CNOT → CNOT + CNOT† + CNOT = 3 CNOTs

Total: $$10 \times 3 = 30$$ CNOTs

$$\boxed{30 \text{ CNOTs}}$$

**(c) Error Model:**

Assumes depolarizing or amplitude damping noise proportional to number of gate applications.

---

### Solution 19

**(a) Inverse Decomposition:**

$$\mathcal{D}_p^{-1} = \frac{1}{1-p}\mathcal{I} - \frac{p/3}{1-p}(\mathcal{X} + \mathcal{Y} + \mathcal{Z})$$

where $$\mathcal{X}[\rho] = X\rho X$$, etc.

**(b) Overhead $$\gamma$$:**

$$\gamma = \frac{1}{1-p} + \frac{p}{1-p} = \frac{1+p/3 \cdot 3}{1-p} = \frac{1}{1-p}$$

For $$p = 0.01$$: $$\gamma = 1/(1-0.01) = 1.01$$

$$\boxed{\gamma \approx 1.01}$$

**(c) 50 Gates:**

$$\gamma^{50} = (1.01)^{50} \approx 1.64$$

$$\boxed{\text{Overhead} \approx 1.64 \times}$$

---

### Solution 20

**(a) Particle Number:**

$$\hat{N} = \frac{n - \sum_i Z_i}{2}$$ (after JW transformation)

Measuring $$\sum_i Z_i$$ gives particle number.

**(b) Post-Selection Rate:**

70% of shots in correct sector.

$$\boxed{70\% \text{ post-selection rate}}$$

**(c) Statistical Uncertainty:**

Effective shots reduced by 1/0.7.

Uncertainty increases by $$\sqrt{1/0.7} \approx 1.2$$

$$\boxed{\sim 20\% \text{ increase in uncertainty}}$$

---

### Solution 21

**(a) Purity Calculation:**

$$\text{Tr}(\rho^2) = (1-p)^2 \cdot 1 + 2(1-p)p \cdot \text{Tr}(|\psi\rangle\langle\psi| \cdot I/2) + p^2 \cdot \text{Tr}(I/2 \cdot I/2)$$

$$= (1-p)^2 + 2(1-p)p \cdot 1/2 + p^2 \cdot 1/2$$

$$= (1-p)^2 + (1-p)p + p^2/2 = 1 - p + p^2/2$$

$$\boxed{\text{Tr}(\rho^2) = 1 - p + p^2/2}$$

**(b) Error Suppression:**

$$\text{Tr}(O\rho) = (1-p)\langle O\rangle + p \cdot \text{Tr}(O)/2$$

$$\frac{\text{Tr}(O\rho^2)}{\text{Tr}(\rho^2)} = \langle O\rangle + O(p^2)$$

Error reduced from $$O(p)$$ to $$O(p^2)$$!

**(c) Circuit for 2-Copy:**

SWAP test or controlled-SWAP between two copies, followed by measurement.

---

### Solution 22

**(a) Device Parameters:**

Given in problem statement.

**(b) Error Estimate:**

Depolarizing error: $$\epsilon \approx 20 \times 0.001 + 15 \times 0.01 = 0.02 + 0.15 = 0.17$$

Coherence error (rough): circuit depth ~35 gates × 50 ns = 1.75 μs

$$T_2$$ error: $$1 - e^{-1.75/50} \approx 0.03$$

Total: $$\sim 20\%$$ error

$$\boxed{\sim 20\% \text{ total error}}$$

**(c) Mitigation Strategy:**

- ZNE: 3 noise levels, expect ~3× improvement
- Symmetry: particle number check, expect 30% rejection
- Combined: ~10× improvement possible

Final error ~2-5%

---

### Solution 23

**(a) ZNE Runs:**

3 noise levels × shots per level = 3× circuit evaluations

$$\boxed{3 \times \text{ baseline runs}}$$

**(b) PEC Overhead:**

$$\gamma = (1/(1-p))^L$$ for depolarizing

For $$p = 0.01$$, $$L$$ gates: $$\gamma = (1.01)^L$$

$$\boxed{\gamma = 1.01^L}$$

**(c) Practical Depths:**

ZNE: works up to $$\sim 100$$ gates

PEC: For $$\gamma < 1000$$: $$L < \ln(1000)/\ln(1.01) \approx 693$$

But variance scaling makes $$L \sim 50-100$$ practical.

---

### Solution 24

**(a) ZNE Unreliability:**

When $$E(\lambda)$$ becomes non-monotonic or oscillatory.

Typically for $$p \cdot L > 0.5$$ (total error > 50%)

**(b) PEC Impractical Depth:**

$$\gamma^L > 1000$$ when $$L > 690$$ (for $$p = 0.01$$)

Practically, $$L \sim 50-100$$ is the limit due to variance.

**(c) Comparison to QEC:**

Error correction threshold: $$p \sim 1\%$$

Error mitigation works above threshold but has exponential overhead.

Error correction works below threshold with polynomial overhead.

$$\boxed{\text{QEC is sustainable; mitigation is transitional}}$$

---

## Section D: Integration

### Solution 25

| Algorithm | Best Platform | Reasoning |
|-----------|---------------|-----------|
| VQE for H₂O | Trapped Ion | High fidelity needed for chemistry accuracy |
| QAOA for 100-node MaxCut | Neutral Atom | Large qubit count, native ZZ interactions |
| VQE-UCCSD for Fe-porphyrin | Superconducting | Deep circuits benefit from fast gates |
| QAOA p=1 for portfolio | Superconducting | Shallow circuit, moderate fidelity OK |

---

### Solution 26

**(a) Qubit Count:**

4 electrons in 4 orbitals = 8 spin-orbitals (JW)

Symmetry reduction (particle number + spin): 4-6 qubits

$$\boxed{4-6 \text{ qubits}}$$

**(b) Ansatz:**

UCCSD: chemically motivated, moderate depth (~50 CNOTs)

Or ADAPT-VQE: problem-adapted, potentially shorter

$$\boxed{\text{ADAPT-VQE or UCCSD}}$$

**(c) Shot Estimate:**

For 1 mHa = 0.001 Ha precision:

Variance per measurement: $$\sigma^2 \sim 1$$ Ha²

Shots: $$N \sim (\sigma/0.001)^2 = 10^6$$

$$\boxed{N \sim 10^6 \text{ shots per energy evaluation}}$$

**(d) Error Mitigation:**

ZNE + symmetry verification (particle number)

---

### Solution 27

**(a) Gate Count:**

127 qubits, 60 layers, ~2 CNOTs per layer per qubit pair

Rough estimate: $$127 \times 60 \times 2 \approx 15000$$ two-qubit gates

$$\boxed{\sim 15000 \text{ two-qubit gates}}$$

**(b) Why Not QEC:**

- QEC requires overhead (many physical per logical qubit)
- Would reduce effective problem size
- Error mitigation sufficient for this demonstration
- Focus was on "utility" not full fault tolerance

**(c) Utility vs Advantage:**

**Utility:** Quantum computer produces useful, verifiable results that inform classical computation.

**Advantage:** Quantum outperforms all classical methods definitively.

$$\boxed{\text{Utility} \neq \text{Advantage; utility is weaker claim}}$$

---

### Solution 28

**(a) Molecular System:**

Transition metal complex with strong correlation (e.g., Fe(II) center)

- Classically hard: strong correlation
- Chemically relevant: catalysis
- Right size: 10-30 qubits after active space

**(b) Ansatz and Mitigation:**

- ADAPT-VQE for compact circuit
- ZNE + symmetry verification
- Active space carefully chosen

**(c) Classical Verification:**

- Compare to best classical methods (DMRG, CCSD(T))
- Show quantum result is beyond classical reach
- Check consistency with experimental data if available

**(d) Technical Challenges:**

- Active space selection
- Sufficient shot counts for precision
- Error mitigation effectiveness
- Classical post-processing

$$\boxed{\text{Main challenge: achieving chemical accuracy with current noise levels}}$$
