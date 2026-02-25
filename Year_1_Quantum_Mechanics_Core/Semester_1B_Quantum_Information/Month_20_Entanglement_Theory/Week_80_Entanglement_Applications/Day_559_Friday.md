# Day 559: LOCC Operations

## Overview
**Day 559** | Week 80, Day 6 | Year 1, Month 20 | Entanglement Applications

Today we study Local Operations and Classical Communication (LOCC), the fundamental operational framework for distributed quantum information processing. LOCC defines what spatially separated parties can accomplish when they share entanglement but cannot exchange quantum systems.

---

## Learning Objectives
1. Define LOCC operations precisely
2. Understand the LOCC hierarchy and its structure
3. Learn what LOCC can and cannot accomplish
4. Study entanglement monotones and their properties
5. Master Nielsen's majorization theorem for pure state conversion
6. Analyze state convertibility under LOCC

---

## Core Content

### What is LOCC?

**LOCC = Local Operations and Classical Communication**

In a distributed quantum scenario:
- Alice holds system A
- Bob holds system B
- They can perform local quantum operations
- They can communicate classically (phone, internet, etc.)
- They CANNOT send quantum states

```
Alice (Lab A)                                Bob (Lab B)
┌──────────────────┐                        ┌──────────────────┐
│  Quantum system A │      Classical        │  Quantum system B │
│    Local ops      │◄────────────────────► │    Local ops      │
│    Measurements   │      Channel          │    Measurements   │
└──────────────────┘                        └──────────────────┘
```

### Formal Definition

An **LOCC operation** is a quantum channel that can be implemented as:

1. **One-way LOCC (→):**
   - Alice performs measurement, sends result to Bob
   - Bob applies operation conditioned on result

2. **Two-way LOCC (↔):**
   - Multiple rounds of back-and-forth communication
   - Each party conditions operations on previous results

Mathematically, LOCC ⊆ Separable operations ⊆ All operations

### LOCC Protocol Structure

```
Round 1:
Alice: A₁ = {M_i^A} → outcome i → sends to Bob
Bob:   B₁ = {N_j^B|i} → outcome j → sends to Alice

Round 2:
Alice: A₂ = {M_k^A|i,j} → outcome k → sends to Bob
Bob:   B₂ = {N_l^B|i,j,k} → outcome l

... continue ...
```

**Key property:** Finite rounds suffice for any LOCC protocol.

### What LOCC CAN Do

1. **Local unitary operations**
   $$\rho \mapsto (U_A \otimes U_B) \rho (U_A^\dagger \otimes U_B^\dagger)$$

2. **Local measurements and post-selection**

3. **Classical communication of measurement outcomes**

4. **Probabilistic state transformations**

5. **Teleportation** (with pre-shared entanglement)

6. **Superdense coding** (with pre-shared entanglement)

7. **Entanglement distillation** (probabilistically)

### What LOCC CANNOT Do

1. **Create entanglement from scratch**
   $$\rho_{sep} \xrightarrow{LOCC} \rho_{ent} \quad \text{IMPOSSIBLE!}$$

2. **Increase entanglement on average**
   - Entanglement is a resource that can only decrease

3. **Deterministically convert less-entangled to more-entangled states**

4. **Distinguish all orthogonal product states** (surprising!)

5. **Perfectly distinguish non-orthogonal states**

### Entanglement Monotones

An **entanglement monotone** $E$ satisfies:

$$\boxed{E(\rho) \geq \sum_i p_i E(\rho_i) \quad \text{for any LOCC operation}}$$

where $\{p_i, \rho_i\}$ are the outcomes.

**Key monotones:**

| Monotone | Formula | Properties |
|----------|---------|------------|
| Entropy of entanglement | $E(|\psi\rangle) = S(\rho_A)$ | Pure states only |
| Entanglement of formation | $E_F(\rho) = \min \sum_i p_i E(|\psi_i\rangle)$ | Mixed states |
| Distillable entanglement | $E_D(\rho)$ = max Bell pair rate | Operational |
| Relative entropy of ent. | $E_R(\rho) = \min_{\sigma \in SEP} S(\rho \| \sigma)$ | Distance-based |
| Negativity | $N(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2}$ | Computable |
| Concurrence | $C(\rho)$ | 2-qubit states |

### Nielsen's Majorization Theorem

For pure state transformations under LOCC:

$$\boxed{|\psi\rangle \xrightarrow{LOCC} |\phi\rangle \text{ iff } \lambda_\psi \prec \lambda_\phi}$$

where $\lambda_\psi$ and $\lambda_\phi$ are the Schmidt coefficient vectors, and $\prec$ denotes **majorization**.

#### Majorization Definition

Vector $x$ is **majorized by** $y$ (written $x \prec y$) if:

$$\sum_{i=1}^{k} x_i^\downarrow \leq \sum_{i=1}^{k} y_i^\downarrow \quad \forall k$$

with equality for $k = n$ (full sum).

Here $x^\downarrow$ means sorted in descending order.

#### Interpretation

- More entangled states have "flatter" Schmidt coefficients
- Maximally entangled: $(1/d, 1/d, ..., 1/d)$
- Product state: $(1, 0, 0, ..., 0)$

The transformation $|\psi\rangle \to |\phi\rangle$ is possible iff $|\phi\rangle$ is "less entangled" than $|\psi\rangle$ in the majorization sense.

### LOCC vs Separable Operations

**Separable operations** have Kraus form:
$$\mathcal{E}(\rho) = \sum_i (A_i \otimes B_i) \rho (A_i^\dagger \otimes B_i^\dagger)$$

**Theorem:** LOCC ⊂ Separable (strict inclusion!)

There exist separable operations that cannot be implemented by LOCC.

### Entanglement Catalysis

Surprisingly, sometimes a transformation impossible under LOCC becomes possible with an **entanglement catalyst**:

$$|\psi\rangle \otimes |\chi\rangle \xrightarrow{LOCC} |\phi\rangle \otimes |\chi\rangle$$

even when $|\psi\rangle \xrightarrow{LOCC} |\phi\rangle$ is impossible!

The catalyst $|\chi\rangle$ is returned unchanged but enables the transformation.

### State Discrimination Under LOCC

**Local distinguishability problem:** Given one of $\{|\psi_1\rangle, ..., |\psi_n\rangle\}$, can Alice and Bob determine which?

**Product states:** Even orthogonal product states may not be perfectly distinguishable by LOCC!

**Example (Bennett et al.):** The "nonlocality without entanglement" set:
$$\{|0\rangle|0\rangle, |1\rangle|+\rangle, |+\rangle|1\rangle, |-\rangle|-\rangle, ...\}$$

These are orthogonal product states that cannot be perfectly distinguished by LOCC.

---

## Worked Examples

### Example 1: Majorization Check
Can $|\psi\rangle = \sqrt{0.6}|00\rangle + \sqrt{0.4}|11\rangle$ be converted to $|\phi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$?

**Solution:**

Schmidt coefficients (squared):
- $\lambda_\psi = (0.6, 0.4)$
- $\lambda_\phi = (0.8, 0.2)$

Check majorization $\lambda_\psi \prec \lambda_\phi$:
- $k = 1$: $0.6 \leq 0.8$ ✓
- $k = 2$: $0.6 + 0.4 = 1.0 = 0.8 + 0.2$ ✓

Yes, $\lambda_\psi \prec \lambda_\phi$, so $|\psi\rangle \xrightarrow{LOCC} |\phi\rangle$ is possible!

**Interpretation:** $|\psi\rangle$ is more entangled than $|\phi\rangle$, so we can "decrease" entanglement via LOCC. ∎

### Example 2: Impossible Transformation
Can $|\phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ be converted to $|00\rangle$?

**Solution:**

Schmidt coefficients:
- $\lambda_{\Phi^+} = (0.5, 0.5)$
- $\lambda_{00} = (1, 0)$

Check majorization $\lambda_{\Phi^+} \prec \lambda_{00}$:
- $k = 1$: $0.5 \leq 1$ ✓
- $k = 2$: $0.5 + 0.5 = 1 = 1 + 0$ ✓

Yes! So $|\Phi^+\rangle \xrightarrow{LOCC} |00\rangle$ is possible.

Wait, but this destroys entanglement—is that allowed? YES! LOCC can always decrease or destroy entanglement. We just can't increase it. ∎

### Example 3: Incomparable States
Are $|\psi\rangle = \sqrt{0.5}|00\rangle + \sqrt{0.3}|11\rangle + \sqrt{0.2}|22\rangle$ and $|\phi\rangle = \sqrt{0.6}|00\rangle + \sqrt{0.25}|11\rangle + \sqrt{0.15}|22\rangle$ interconvertible?

**Solution:**

Schmidt coefficients (sorted descending):
- $\lambda_\psi = (0.5, 0.3, 0.2)$
- $\lambda_\phi = (0.6, 0.25, 0.15)$

Check $\lambda_\psi \prec \lambda_\phi$:
- $k = 1$: $0.5 \leq 0.6$ ✓
- $k = 2$: $0.5 + 0.3 = 0.8 \leq 0.6 + 0.25 = 0.85$ ✓
- $k = 3$: $1 = 1$ ✓

So $|\psi\rangle \xrightarrow{LOCC} |\phi\rangle$ is possible.

Check $\lambda_\phi \prec \lambda_\psi$:
- $k = 1$: $0.6 \leq 0.5$? NO! ✗

So $|\phi\rangle \xrightarrow{LOCC} |\psi\rangle$ is NOT possible.

These states are **not** interconvertible: we can go $\psi \to \phi$ but not $\phi \to \psi$. ∎

---

## Practice Problems

### Problem 1: Three-Qubit LOCC
Can GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ be converted to W state $\frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$ by LOCC?

### Problem 2: Entanglement Monotone
Verify that entropy of entanglement $E(|\psi\rangle) = S(\rho_A)$ is an entanglement monotone.

### Problem 3: Catalysis
Show that the transformation $|\psi\rangle = \sqrt{0.4}|00\rangle + \sqrt{0.4}|11\rangle + \sqrt{0.2}|22\rangle$ to $|\phi\rangle = \sqrt{0.5}|00\rangle + \sqrt{0.25}|11\rangle + \sqrt{0.25}|22\rangle$ is impossible, but find a catalyst that enables it.

### Problem 4: Nonlocality Without Entanglement
Construct an explicit LOCC protocol that distinguishes $\{|00\rangle, |11\rangle\}$ perfectly.

---

## Computational Lab

```python
"""Day 559: LOCC Operations Analysis"""
import numpy as np
from itertools import permutations
from scipy.linalg import svd

def majorization_leq(x, y, tol=1e-10):
    """
    Check if x is majorized by y: x ≺ y

    Args:
        x, y: Probability vectors (must sum to same value)

    Returns:
        True if x ≺ y, False otherwise
    """
    # Sort in descending order
    x_sorted = np.sort(x)[::-1]
    y_sorted = np.sort(y)[::-1]

    n = len(x_sorted)

    # Check partial sums
    for k in range(n):
        if np.sum(x_sorted[:k+1]) > np.sum(y_sorted[:k+1]) + tol:
            return False

    return True

def schmidt_coefficients(psi, dim_A, dim_B):
    """
    Compute Schmidt coefficients (squared) from state vector

    Args:
        psi: State vector
        dim_A, dim_B: Dimensions of subsystems

    Returns:
        Squared Schmidt coefficients (eigenvalues of reduced state)
    """
    # Reshape to matrix
    C = psi.reshape(dim_A, dim_B)

    # SVD gives Schmidt decomposition
    _, S, _ = svd(C)

    # Return squared coefficients
    return S**2

def can_convert_locc(psi, phi, dim_A, dim_B):
    """
    Check if |ψ⟩ → |φ⟩ is possible under LOCC

    Uses Nielsen's majorization theorem
    """
    lambda_psi = schmidt_coefficients(psi, dim_A, dim_B)
    lambda_phi = schmidt_coefficients(phi, dim_A, dim_B)

    # Pad with zeros if needed
    max_len = max(len(lambda_psi), len(lambda_phi))
    lambda_psi = np.pad(lambda_psi, (0, max_len - len(lambda_psi)))
    lambda_phi = np.pad(lambda_phi, (0, max_len - len(lambda_phi)))

    return majorization_leq(lambda_psi, lambda_phi)

def entropy_of_entanglement(psi, dim_A, dim_B):
    """
    Compute entropy of entanglement E(|ψ⟩) = S(ρ_A)
    """
    lambdas = schmidt_coefficients(psi, dim_A, dim_B)

    # Von Neumann entropy
    entropy = 0
    for l in lambdas:
        if l > 1e-10:
            entropy -= l * np.log2(l)

    return entropy

def negativity(rho, dim_A, dim_B):
    """
    Compute negativity: N(ρ) = (||ρ^TB||_1 - 1) / 2
    """
    # Partial transpose on B
    rho_pt = partial_transpose(rho, dim_A, dim_B)

    # Trace norm (sum of absolute eigenvalues)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    trace_norm = np.sum(np.abs(eigenvalues))

    return (trace_norm - 1) / 2

def partial_transpose(rho, dim_A, dim_B):
    """
    Compute partial transpose ρ^TB
    """
    # Reshape to 4-index tensor
    rho_tensor = rho.reshape(dim_A, dim_B, dim_A, dim_B)

    # Transpose B indices
    rho_pt_tensor = np.transpose(rho_tensor, (0, 3, 2, 1))

    # Reshape back
    return rho_pt_tensor.reshape(dim_A * dim_B, dim_A * dim_B)

def generate_random_state(dim_A, dim_B):
    """Generate random pure state"""
    psi = np.random.randn(dim_A * dim_B) + 1j * np.random.randn(dim_A * dim_B)
    return psi / np.linalg.norm(psi)

# Demonstration
print("LOCC OPERATIONS ANALYSIS")
print("="*60)

# 1. Majorization examples
print("\n1. MAJORIZATION AND STATE CONVERTIBILITY")
print("-"*50)

# Define some states
# |ψ₁⟩ = √0.6|00⟩ + √0.4|11⟩
psi1 = np.array([np.sqrt(0.6), 0, 0, np.sqrt(0.4)])

# |ψ₂⟩ = √0.8|00⟩ + √0.2|11⟩
psi2 = np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)])

# |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

# |00⟩
zero_zero = np.array([1, 0, 0, 0])

states = [
    (psi1, "ψ₁ (0.6, 0.4)"),
    (psi2, "ψ₂ (0.8, 0.2)"),
    (phi_plus, "Φ⁺ (0.5, 0.5)"),
    (zero_zero, "|00⟩ (1, 0)")
]

print("\nSchmidt coefficients (squared):")
for psi, name in states:
    lambdas = schmidt_coefficients(psi, 2, 2)
    E = entropy_of_entanglement(psi, 2, 2)
    print(f"  {name:20s}: λ = {lambdas}, E = {E:.4f}")

print("\nConvertibility matrix (row → col possible?):")
print("         ", end="")
for _, name in states:
    print(f"{name[:8]:10s}", end="")
print()

for psi_from, name_from in states:
    print(f"{name_from[:8]:10s}", end="")
    for psi_to, name_to in states:
        can_convert = can_convert_locc(psi_from, psi_to, 2, 2)
        print(f"{'✓':^10s}" if can_convert else f"{'✗':^10s}", end="")
    print()

# 2. Three-level system
print("\n\n2. THREE-LEVEL SYSTEM (QUTRIT)")
print("-"*50)

# |ψ⟩ = √0.5|00⟩ + √0.3|11⟩ + √0.2|22⟩
psi_qutrit1 = np.zeros(9)
psi_qutrit1[0] = np.sqrt(0.5)
psi_qutrit1[4] = np.sqrt(0.3)
psi_qutrit1[8] = np.sqrt(0.2)

# |φ⟩ = √0.6|00⟩ + √0.25|11⟩ + √0.15|22⟩
psi_qutrit2 = np.zeros(9)
psi_qutrit2[0] = np.sqrt(0.6)
psi_qutrit2[4] = np.sqrt(0.25)
psi_qutrit2[8] = np.sqrt(0.15)

print("State ψ: coefficients (0.5, 0.3, 0.2)")
print("State φ: coefficients (0.6, 0.25, 0.15)")

print(f"\nψ → φ possible: {can_convert_locc(psi_qutrit1, psi_qutrit2, 3, 3)}")
print(f"φ → ψ possible: {can_convert_locc(psi_qutrit2, psi_qutrit1, 3, 3)}")

print("\nInterpretation: ψ is more entangled (flatter distribution)")
print("Can decrease entanglement (ψ→φ) but not increase (φ→ψ)")

# 3. Entanglement monotones
print("\n\n3. ENTANGLEMENT MONOTONES")
print("-"*50)

print("\nEntropy of entanglement for various states:")
test_states = [
    (np.array([1, 0, 0, 0]), "|00⟩ (product)"),
    (np.array([1, 0, 0, 1])/np.sqrt(2), "|Φ⁺⟩ (max ent.)"),
    (np.array([np.sqrt(0.9), 0, 0, np.sqrt(0.1)]), "√0.9|00⟩+√0.1|11⟩"),
    (np.array([np.sqrt(0.7), 0, 0, np.sqrt(0.3)]), "√0.7|00⟩+√0.3|11⟩"),
]

for psi, name in test_states:
    E = entropy_of_entanglement(psi, 2, 2)
    print(f"  {name:30s}: E = {E:.4f} ebits")

# 4. Negativity for mixed states
print("\n\n4. NEGATIVITY (MIXED STATE MONOTONE)")
print("-"*50)

# Werner states
def werner_state(F):
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_phi = np.outer(phi_plus, phi_plus.conj())
    rho_mixed = (np.eye(4) - rho_phi) / 3
    return F * rho_phi + (1 - F) * rho_mixed

print("\nNegativity of Werner states:")
print("  F     | Negativity | Entangled?")
print("-"*40)
for F in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rho = werner_state(F)
    N = negativity(rho, 2, 2)
    ent = "Yes" if N > 1e-10 else "No"
    print(f" {F:.2f}   |   {N:.4f}   |    {ent}")

# 5. LOCC protocol example
print("\n\n5. EXAMPLE LOCC PROTOCOL: STATE DISCRIMINATION")
print("-"*50)
print("""
Task: Distinguish |00⟩ from |11⟩ using LOCC

Protocol:
1. Alice measures in computational basis {|0⟩, |1⟩}
2. Alice sends result to Bob (1 classical bit)
3. Bob's measurement confirms Alice's result

This is LOCC because:
- Only local measurements performed
- Only classical information exchanged
- Perfect discrimination achieved

Note: Distinguishing |0+⟩ = |0⟩|+⟩ from |1-⟩ = |1⟩|-⟩ is harder!
""")

# 6. Catalysis example
print("\n6. ENTANGLEMENT CATALYSIS")
print("-"*50)
print("""
Sometimes |ψ⟩ → |φ⟩ is impossible, but
|ψ⟩ ⊗ |χ⟩ → |φ⟩ ⊗ |χ⟩ is possible with catalyst |χ⟩!

Example:
  ψ: (0.4, 0.4, 0.1, 0.1)
  φ: (0.5, 0.25, 0.25, 0)

Direct conversion fails (check majorization).
But with catalyst χ: (0.6, 0.4), it becomes possible!
""")

# Check the example
lambda_psi = np.array([0.4, 0.4, 0.1, 0.1])
lambda_phi = np.array([0.5, 0.25, 0.25, 0])

print(f"ψ ≺ φ (direct): {majorization_leq(lambda_psi, lambda_phi)}")

# With catalyst (0.6, 0.4)
# Combined Schmidt coefficients
lambda_psi_cat = np.sort(np.outer(lambda_psi, [0.6, 0.4]).flatten())[::-1]
lambda_phi_cat = np.sort(np.outer(lambda_phi, [0.6, 0.4]).flatten())[::-1]

print(f"ψ⊗χ ≺ φ⊗χ (catalyzed): {majorization_leq(lambda_psi_cat, lambda_phi_cat)}")

# 7. Summary diagram
print("\n\n7. LOCC HIERARCHY")
print("-"*60)
print("""
           All quantum operations
                    ⊃
           Separable operations
                    ⊃
              LOCC operations
                    ⊃
         One-way LOCC (Alice→Bob)
                    ⊃
            Local operations only

Key constraints under LOCC:
┌────────────────────────────────────────────────────────────┐
│  ✓ CAN DO:                    │  ✗ CANNOT DO:              │
│  - Local unitaries            │  - Create entanglement     │
│  - Local measurements         │  - Increase entanglement   │
│  - Classical communication    │  - Clone quantum states    │
│  - Teleportation (with ent.)  │  - Some state conversions  │
│  - Distillation (probabilistic│  - Distinguish some        │
│  - Decrease entanglement      │    orthogonal product sets │
└────────────────────────────────────────────────────────────┘

Nielsen's Theorem: |ψ⟩ → |φ⟩ possible iff λ_ψ ≺ λ_φ
""")

# 8. Random state analysis
print("\n8. RANDOM STATE CONVERTIBILITY")
print("-"*50)

np.random.seed(42)
n_tests = 100
conversions = {'both': 0, 'forward': 0, 'backward': 0, 'neither': 0}

for _ in range(n_tests):
    psi = generate_random_state(2, 2)
    phi = generate_random_state(2, 2)

    fwd = can_convert_locc(psi, phi, 2, 2)
    bwd = can_convert_locc(phi, psi, 2, 2)

    if fwd and bwd:
        conversions['both'] += 1
    elif fwd:
        conversions['forward'] += 1
    elif bwd:
        conversions['backward'] += 1
    else:
        conversions['neither'] += 1

print(f"\nOut of {n_tests} random state pairs:")
print(f"  Both directions possible: {conversions['both']}%")
print(f"  Only ψ→φ possible:        {conversions['forward']}%")
print(f"  Only φ→ψ possible:        {conversions['backward']}%")
print(f"  Neither possible:         {conversions['neither']}%")

print("\nConclusion: Most random pairs are incomparable under LOCC!")
```

**Expected Output:**
```
LOCC OPERATIONS ANALYSIS
============================================================

1. MAJORIZATION AND STATE CONVERTIBILITY
--------------------------------------------------

Schmidt coefficients (squared):
  ψ₁ (0.6, 0.4)        : λ = [0.6 0.4], E = 0.9710
  ψ₂ (0.8, 0.2)        : λ = [0.8 0.2], E = 0.7219
  Φ⁺ (0.5, 0.5)        : λ = [0.5 0.5], E = 1.0000
  |00⟩ (1, 0)          : λ = [1. 0.], E = 0.0000

Convertibility matrix (row → col possible?):
         ψ₁ (0.6,  ψ₂ (0.8,  Φ⁺ (0.5,  |00⟩ (1,
ψ₁ (0.6,      ✓         ✓         ✗         ✓
ψ₂ (0.8,      ✗         ✓         ✗         ✓
Φ⁺ (0.5,      ✓         ✓         ✓         ✓
|00⟩ (1,      ✗         ✗         ✗         ✓


3. ENTANGLEMENT MONOTONES
--------------------------------------------------

Entropy of entanglement for various states:
  |00⟩ (product)                : E = 0.0000 ebits
  |Φ⁺⟩ (max ent.)               : E = 1.0000 ebits
  √0.9|00⟩+√0.1|11⟩             : E = 0.4690 ebits
  √0.7|00⟩+√0.3|11⟩             : E = 0.8813 ebits
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| LOCC constraint | Cannot create or increase entanglement |
| Monotone property | $E(\rho) \geq \sum_i p_i E(\rho_i)$ under LOCC |
| Nielsen's theorem | $\|\psi\rangle \to \|\phi\rangle$ iff $\lambda_\psi \prec \lambda_\phi$ |
| Majorization | $x \prec y$ iff $\sum_{i=1}^k x_i^\downarrow \leq \sum_{i=1}^k y_i^\downarrow$ |
| Entropy of entanglement | $E(\|\psi\rangle) = S(\rho_A) = -\sum_i \lambda_i \log \lambda_i$ |
| Negativity | $N(\rho) = (\|\rho^{T_B}\|_1 - 1)/2$ |

### Key Takeaways
1. **LOCC = Local Operations + Classical Communication**
2. **Entanglement cannot be created or increased** by LOCC
3. **Entanglement monotones** quantify entanglement decrease
4. **Nielsen's theorem** characterizes pure state conversion via majorization
5. **Catalysis** can enable otherwise impossible transformations
6. **Not all orthogonal states** can be distinguished by LOCC

---

## Daily Checklist

- [ ] I can define LOCC operations precisely
- [ ] I understand what LOCC can and cannot do
- [ ] I can check majorization conditions
- [ ] I can apply Nielsen's theorem for pure states
- [ ] I understand entanglement monotones
- [ ] I ran the simulation and verified convertibility results

---

*Next: Day 560 — Month 20 Review*
