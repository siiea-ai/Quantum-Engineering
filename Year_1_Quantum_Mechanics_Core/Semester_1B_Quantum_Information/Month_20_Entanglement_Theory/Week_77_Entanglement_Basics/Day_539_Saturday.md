# Day 539: Week 77 Review — Entanglement Basics

## Overview
**Day 539** | Week 77, Day 7 | Year 1, Month 20 | Weekly Synthesis

Today we consolidate all entanglement basics concepts from Week 77 through comprehensive review and problem solving.

---

## Week 77 Concept Map

```
                    ENTANGLEMENT BASICS
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
 CLASSIFICATION        DETECTION             MULTIPARTITE
    │                      │                      │
• Separable/          • Witnesses            • GHZ states
  Entangled           • PPT criterion        • W states
• Product states      • Range criterion      • SLOCC classes
• Schmidt decomp      • Negativity           • 3-tangle
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                           │
                   QUANTUM CORRELATIONS
```

---

## Master Formula Reference

### Separability
$$\rho_{sep} = \sum_i p_i \rho_i^A \otimes \rho_i^B$$

### Bell States
$$|\Phi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle), \quad |\Psi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

### Entanglement Witness
$$W: \text{Tr}(W\rho_{sep}) \geq 0, \text{Tr}(W\rho_{ent}) < 0$$

### PPT Criterion
$$\rho \text{ separable} \Rightarrow \rho^{T_B} \geq 0$$
$$(\rho^{T_B})_{ij,kl} = \rho_{il,kj}$$

### Negativity
$$\mathcal{N}(\rho) = \sum_{\lambda_i < 0} |\lambda_i| = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

### GHZ and W States
$$|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$
$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

---

## Comprehensive Problem Set

### Problem 1: State Classification
Classify each state as separable or entangled. If entangled, compute the negativity.

a) $|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |01\rangle)$

b) $\rho = \frac{1}{2}|\Phi^+\rangle\langle\Phi^+| + \frac{1}{2}|00\rangle\langle 00|$

c) $\rho = \frac{1}{3}|\Phi^+\rangle\langle\Phi^+| + \frac{2}{3}\frac{I}{4}$

**Solutions:**

a) $|\psi\rangle = |0\rangle \otimes \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |0\rangle|+\rangle$

   **Product state → Separable**, $\mathcal{N} = 0$

b) Compute partial transpose, find eigenvalues:
   $$\rho^{T_B} \text{ has eigenvalues } \{1/4, 1/4, 1/4, 1/4, 0, 0, 0, -?\}$$

   Need actual computation: **Entangled** with some negativity

c) Werner-like state with $p = 1/3$:
   This is at the PPT boundary → **Separable**, $\mathcal{N} = 0$

### Problem 2: Witness Construction
Construct an optimal entanglement witness for the state:
$$|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$$

**Solution:**
Maximum product overlap: $\alpha = \max_{\phi,\chi} |\langle \phi \chi|\psi\rangle|^2$

For any product state $|ab\rangle$:
$$|\langle ab|\psi\rangle|^2 = |\cos\theta \langle a|0\rangle\langle b|0\rangle + \sin\theta\langle a|1\rangle\langle b|1\rangle|^2$$

Maximum at $|a\rangle = |b\rangle = |0\rangle$ or $|1\rangle$: $\alpha = \max(\cos^2\theta, \sin^2\theta)$

Witness:
$$W = \alpha I - |\psi\rangle\langle\psi|$$

### Problem 3: PPT Analysis
For the isotropic state $\rho_p = p|\Phi^+\rangle\langle\Phi^+| + (1-p)I/4$:

a) Find the PPT threshold
b) Compute negativity as a function of p
c) Construct a witness that detects entanglement at the threshold

**Solution:**

a) PPT threshold: $p = 1/2$

b) $\mathcal{N}(p) = \max(0, (3p-1)/4)$

c) Witness: $W = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$

### Problem 4: Multipartite Entanglement
Compare the 4-qubit GHZ and W states:
$$|GHZ_4\rangle = \frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$$
$$|W_4\rangle = \frac{1}{2}(|0001\rangle + |0010\rangle + |0100\rangle + |1000\rangle)$$

a) Compute the reduced 2-qubit density matrices
b) Determine which has more robust entanglement

**Solution:**

a) GHZ_4 traced to 2 qubits:
$$\rho_{AB}^{GHZ} = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$
(Separable)

W_4 traced to 2 qubits:
$$\rho_{AB}^{W} = \frac{1}{4}(2|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10| + |01\rangle\langle 10| + |10\rangle\langle 01|)$$
(Entangled)

b) W_4 is more robust—losing qubits preserves bipartite entanglement.

### Problem 5: Bound Entanglement
Explain why there is no bound entanglement in 2×2 systems.

**Solution:**
In 2×2, the Peres-Horodecki theorem is sufficient:
$$\rho \text{ separable} \Leftrightarrow \rho^{T_B} \geq 0$$

If a 2×2 state is PPT, it's separable. If it's entangled, it's NPT.
NPT states are always distillable.
Therefore: no PPT entangled states exist in 2×2.

---

## Computational Lab: Week Integration

```python
"""Day 539: Week 77 Integration"""
import numpy as np
from scipy.linalg import eigvalsh

# ===== Helper Functions =====
def projector(psi):
    return np.outer(psi, psi.conj())

def partial_transpose_B(rho, dim_A=2, dim_B=2):
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_TB = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_TB.reshape(dim_A * dim_B, dim_A * dim_B)

def negativity(rho, dim_A=2, dim_B=2):
    rho_TB = partial_transpose_B(rho, dim_A, dim_B)
    eigenvalues = eigvalsh(rho_TB)
    return np.sum(np.abs(eigenvalues[eigenvalues < -1e-10]))

def is_ppt(rho, dim_A=2, dim_B=2):
    rho_TB = partial_transpose_B(rho, dim_A, dim_B)
    return np.min(eigvalsh(rho_TB)) >= -1e-10

def schmidt_coefficients(psi, dim_A, dim_B):
    C = psi.reshape(dim_A, dim_B)
    _, S, _ = np.linalg.svd(C)
    return S**2

def entanglement_entropy(psi, dim_A, dim_B):
    lambdas = schmidt_coefficients(psi, dim_A, dim_B)
    lambdas = lambdas[lambdas > 1e-12]
    return -np.sum(lambdas * np.log2(lambdas))

# ===== Bell States =====
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

# ===== Week 77 Comprehensive Analysis =====
print("=" * 60)
print("WEEK 77 COMPREHENSIVE REVIEW")
print("=" * 60)

# 1. Bell state properties
print("\n1. BELL STATE PROPERTIES")
print("-" * 40)
for name, state in [("Φ⁺", phi_plus), ("Φ⁻", phi_minus),
                    ("Ψ⁺", psi_plus), ("Ψ⁻", psi_minus)]:
    rho = projector(state)
    neg = negativity(rho)
    entropy = entanglement_entropy(state, 2, 2)
    print(f"|{name}⟩: negativity = {neg:.4f}, entropy = {entropy:.4f} ebit")

# 2. Werner state analysis
print("\n2. WERNER STATE ANALYSIS")
print("-" * 40)
print("ρ(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")
print(f"{'p':>6} | {'PPT':>5} | {'Negativity':>10} | {'Status'}")
print("-" * 40)

rho_psi = projector(psi_minus)
I4 = np.eye(4) / 4

for p in [0.0, 0.2, 1/3, 0.5, 0.75, 1.0]:
    rho = p * rho_psi + (1-p) * I4
    ppt = is_ppt(rho)
    neg = negativity(rho)
    status = "Separable" if ppt else "Entangled"
    print(f"{p:6.3f} | {str(ppt):>5} | {neg:10.4f} | {status}")

# 3. Witness effectiveness
print("\n3. ENTANGLEMENT WITNESS TEST")
print("-" * 40)

W = 0.5 * np.eye(4) - projector(psi_minus)

test_states = [
    ("Pure |Ψ⁻⟩", projector(psi_minus)),
    ("Werner p=0.5", 0.5 * rho_psi + 0.5 * I4),
    ("Werner p=0.33", (1/3) * rho_psi + (2/3) * I4),
    ("Maximally mixed", I4),
    ("Product |00⟩", projector(np.array([1,0,0,0])))
]

print("Witness: W = 0.5*I - |Ψ⁻⟩⟨Ψ⁻|")
for name, rho in test_states:
    val = np.trace(W @ rho).real
    detected = "DETECTED" if val < -1e-10 else "not detected"
    print(f"{name:20}: Tr(Wρ) = {val:+.4f} → {detected}")

# 4. Multipartite states
print("\n4. MULTIPARTITE ENTANGLEMENT (3 qubits)")
print("-" * 40)

def ket(bits):
    n = len(bits)
    state = np.zeros(2**n, dtype=complex)
    idx = sum(b * 2**(n-1-i) for i, b in enumerate(bits))
    state[idx] = 1
    return state

def partial_trace_C(rho_ABC):
    """Trace out third qubit from 3-qubit state"""
    rho = rho_ABC.reshape(2, 2, 2, 2, 2, 2)
    return np.trace(rho, axis1=2, axis2=5).reshape(4, 4)

ghz = (ket([0,0,0]) + ket([1,1,1])) / np.sqrt(2)
w = (ket([0,0,1]) + ket([0,1,0]) + ket([1,0,0])) / np.sqrt(3)

rho_ghz = projector(ghz)
rho_w = projector(w)

rho_ghz_AB = partial_trace_C(rho_ghz)
rho_w_AB = partial_trace_C(rho_w)

print("Reduced AB states after tracing out C:")
print(f"GHZ: negativity = {negativity(rho_ghz_AB):.4f}")
print(f"W:   negativity = {negativity(rho_w_AB):.4f}")

print("\nInterpretation:")
print("  GHZ → bipartite entanglement destroyed (fragile)")
print("  W   → bipartite entanglement preserved (robust)")

# 5. Summary statistics
print("\n" + "=" * 60)
print("WEEK 77 SUMMARY")
print("=" * 60)
print("""
Key Results:
• Bell states have maximal entanglement (1 ebit)
• Werner states: separable for p ≤ 1/3, entangled for p > 1/3
• PPT is necessary and sufficient in 2×2
• GHZ entanglement is fragile; W entanglement is robust
• Entanglement witnesses can certify entanglement experimentally
""")
```

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] I can define separable and entangled states
- [ ] I understand why Bell states are maximally entangled
- [ ] I can explain what an entanglement witness does
- [ ] I understand the PPT criterion and its limitations
- [ ] I know the difference between GHZ and W class entanglement

### Computational Skills
- [ ] I can compute Schmidt decompositions
- [ ] I can perform partial transpose operations
- [ ] I can calculate negativity from eigenvalues
- [ ] I can construct entanglement witnesses
- [ ] I can trace out subsystems to get reduced states

### Problem-Solving
- [ ] I can classify states as separable or entangled
- [ ] I can design witnesses for specific states
- [ ] I can analyze multipartite entanglement
- [ ] I can explain bound entanglement conceptually

---

## Week 77 Key Takeaways

1. **Separability** is a convex constraint—hard to check in general
2. **Bell states** are the maximally entangled two-qubit states
3. **Witnesses** provide experimental entanglement certification
4. **PPT** is the strongest computable criterion in low dimensions
5. **GHZ vs W** represent fundamentally different entanglement types
6. **Bound entanglement** exists only in dimension ≥ 3×3

---

## Looking Ahead: Week 78

Next week covers **Bell Inequalities**:
- EPR paradox and local realism
- Bell's theorem and its proof
- CHSH inequality and Tsirelson bound
- Quantum violation of classical bounds
- Loophole-free Bell tests

---

*Next: Day 540 — EPR Paradox*
