# Mathematical Framework Checklist

## Purpose

This checklist ensures you've covered all mathematical foundations necessary for your thesis. Check off each item as you address it in your background chapter.

---

## 1. Hilbert Space Fundamentals

### Basic Structure
- [ ] **Hilbert space definition** - Complex vector space with inner product
- [ ] **Dirac notation** - Bras, kets, brakets defined
- [ ] **Orthonormality** - $\langle i | j \rangle = \delta_{ij}$
- [ ] **Completeness relation** - $\sum_i |i\rangle\langle i| = I$
- [ ] **Dimension** - Finite vs. infinite dimensional cases

### Operators
- [ ] **Operator definition** - Linear maps on Hilbert space
- [ ] **Adjoint operator** - $\langle \phi | A | \psi \rangle = \langle A^\dagger \phi | \psi \rangle$
- [ ] **Hermitian operators** - $A = A^\dagger$, real eigenvalues
- [ ] **Unitary operators** - $U^\dagger U = UU^\dagger = I$
- [ ] **Positive operators** - $\langle \psi | A | \psi \rangle \geq 0$ for all $|\psi\rangle$
- [ ] **Trace** - $\text{Tr}(A) = \sum_i \langle i | A | i \rangle$
- [ ] **Spectral decomposition** - $A = \sum_i \lambda_i |i\rangle\langle i|$

---

## 2. Quantum States

### Pure States
- [ ] **State vector** - Unit vector in Hilbert space
- [ ] **Global phase** - Physical equivalence of $|\psi\rangle$ and $e^{i\phi}|\psi\rangle$
- [ ] **Superposition** - $|\psi\rangle = \sum_i \alpha_i |i\rangle$
- [ ] **Normalization** - $\sum_i |\alpha_i|^2 = 1$

### Mixed States
- [ ] **Density operator** - $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$
- [ ] **Properties** - $\rho \geq 0$, $\text{Tr}(\rho) = 1$
- [ ] **Purity** - $\text{Tr}(\rho^2) = 1$ iff pure
- [ ] **Ensemble interpretation** - Probabilistic mixture
- [ ] **Purification** - Every mixed state has a pure state extension

### Composite Systems
- [ ] **Tensor product** - $\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$
- [ ] **Product states** - $|\psi\rangle_A \otimes |\phi\rangle_B$
- [ ] **Entangled states** - Cannot be written as product
- [ ] **Partial trace** - $\rho_A = \text{Tr}_B(\rho_{AB})$
- [ ] **Schmidt decomposition** - Bipartite pure state form

---

## 3. Quantum Evolution

### Closed Systems
- [ ] **Schrödinger equation** - $i\hbar \frac{d}{dt}|\psi\rangle = H|\psi\rangle$
- [ ] **Unitary evolution** - $|\psi(t)\rangle = U(t)|\psi(0)\rangle$
- [ ] **Time-independent case** - $U(t) = e^{-iHt/\hbar}$

### Open Systems
- [ ] **Quantum channels** - CPTP maps
- [ ] **Complete positivity** - $(I \otimes \mathcal{E})(\rho) \geq 0$
- [ ] **Trace preservation** - $\text{Tr}(\mathcal{E}(\rho)) = \text{Tr}(\rho)$
- [ ] **Kraus representation** - $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$
- [ ] **Choi-Jamiołkowski** - Channel-state duality (if used)
- [ ] **Stinespring dilation** - Unitary extension (brief mention)

### Common Channels
- [ ] **Identity** - $\mathcal{I}(\rho) = \rho$
- [ ] **Depolarizing** - Defined with explicit Kraus operators
- [ ] **Dephasing** - Phase noise model
- [ ] **Bit-flip** - $X$ errors
- [ ] **Amplitude damping** - Energy relaxation
- [ ] **Biased noise** - Your specific model if applicable

---

## 4. Quantum Measurement

### Projective Measurements
- [ ] **Observable** - Hermitian operator
- [ ] **Eigenvalue spectrum** - Possible outcomes
- [ ] **Born rule** - $p(m) = \text{Tr}(P_m \rho)$
- [ ] **State update** - $\rho \to P_m \rho P_m / p(m)$
- [ ] **Measurement bases** - Computational, Hadamard, etc.

### Generalized Measurements
- [ ] **POVM definition** - $\{E_m\}$, $\sum_m E_m = I$, $E_m \geq 0$
- [ ] **Outcome probability** - $p(m) = \text{Tr}(E_m \rho)$
- [ ] **Neumark extension** - POVMs as projective in larger space
- [ ] **Weak measurements** - (if applicable to thesis)

---

## 5. Entanglement

### Fundamentals
- [ ] **Separability** - Definition for pure and mixed states
- [ ] **Bell states** - All four defined explicitly
- [ ] **Maximally entangled states** - General definition
- [ ] **No-cloning theorem** - (brief mention)

### Quantification
- [ ] **Entanglement entropy** - $S(\rho_A)$ for pure bipartite states
- [ ] **Entropy properties** - Non-negative, zero for product states
- [ ] **Schmidt rank** - Number of non-zero Schmidt coefficients
- [ ] **Monogamy** - (if relevant to thesis)

### Operations
- [ ] **LOCC** - Local operations and classical communication
- [ ] **Entanglement distillation** - (if relevant)
- [ ] **Entanglement as resource** - Connection to QEC/QI

---

## 6. Information Theory

### Classical Information
- [ ] **Shannon entropy** - $H(X) = -\sum_x p(x) \log p(x)$
- [ ] **Joint entropy** - $H(X,Y)$
- [ ] **Conditional entropy** - $H(X|Y)$
- [ ] **Mutual information** - $I(X:Y) = H(X) + H(Y) - H(X,Y)$

### Quantum Information
- [ ] **Von Neumann entropy** - $S(\rho) = -\text{Tr}(\rho \log \rho)$
- [ ] **Properties** - Concavity, subadditivity
- [ ] **Quantum conditional entropy** - $S(A|B)$ can be negative
- [ ] **Quantum mutual information** - $I(A:B) = S(A) + S(B) - S(AB)$
- [ ] **Relative entropy** - $S(\rho \| \sigma)$ (if used)
- [ ] **Holevo bound** - (if relevant)

### Channel Capacity
- [ ] **Classical capacity** - (brief, if relevant)
- [ ] **Quantum capacity** - (brief, if relevant)
- [ ] **Connection to QEC** - Capacity and threshold relationship

---

## 7. Specific to Your Research

Add items specific to your thesis topic:

### [Your Topic 1]
- [ ]
- [ ]
- [ ]

### [Your Topic 2]
- [ ]
- [ ]
- [ ]

---

## Notation Consistency Check

Verify these are defined and used consistently:

| Symbol | Definition | First Introduced |
|--------|------------|------------------|
| $\|\psi\rangle$ | Pure state | Section ___ |
| $\rho$ | Density operator | Section ___ |
| $\mathcal{H}$ | Hilbert space | Section ___ |
| $X, Y, Z$ | Pauli operators | Section ___ |
| $\mathcal{E}$ | Quantum channel | Section ___ |
| $S(\rho)$ | Von Neumann entropy | Section ___ |
| $\text{Tr}$ | Trace | Section ___ |
| | | |
| | | |

---

## Cross-Reference Verification

For each concept used in later chapters, verify it's defined in Background:

| Concept | Used in Chapter | Defined in Background? |
|---------|-----------------|----------------------|
| | | |
| | | |
| | | |
| | | |
| | | |

---

## References Required

Check that you've cited appropriate references for:

- [ ] Standard QM formalism (Nielsen & Chuang, Preskill)
- [ ] Quantum channels (Wilde)
- [ ] Entanglement theory (Horodecki et al. review)
- [ ] Information theory (Cover & Thomas for classical)
- [ ] Any specific results you invoke

---

## Final Verification

Before moving to QEC background:

- [ ] All checked items above are addressed in the text
- [ ] Items marked N/A are genuinely not needed for your thesis
- [ ] Notation table is complete and consistent
- [ ] All equations are numbered for future reference
- [ ] All key results are attributed with citations
