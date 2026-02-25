# Day 669: Month 23 Review - Quantum Channels

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Review Scope

**Month 23: Quantum Channels (Days 617-644)**
- Week 89: CPTP Maps
- Week 90: Kraus Representation
- Week 91: Choi-Jamiolkowski Isomorphism
- Week 92: Channel Properties and Capacities

---

## Core Concepts: CPTP Maps

### 1. Quantum Channel Definition

A **quantum channel** is a completely positive, trace-preserving (CPTP) linear map:
$$\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$$

**Trace-preserving (TP):**
$$\text{Tr}[\mathcal{E}(\rho)] = \text{Tr}[\rho] \quad \forall \rho$$

**Completely positive (CP):**
$$(\mathcal{E} \otimes \mathcal{I}_R)(\rho_{AR}) \geq 0 \quad \forall \rho_{AR} \geq 0$$

Not just positive, but positive even when tensored with identity on any reference system.

### 2. Why Complete Positivity?

**Positive but not CP:** The transpose map $\mathcal{T}(\rho) = \rho^T$

**Problem:** If $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
$$(\mathcal{T} \otimes \mathcal{I})|\Phi^+\rangle\langle\Phi^+|$$
has negative eigenvalue!

**Physical channels must be CP** because system may be entangled with reference.

---

## Core Concepts: Channel Representations

### 3. Kraus (Operator-Sum) Representation

$$\boxed{\mathcal{E}(\rho) = \sum_{k=1}^r K_k \rho K_k^\dagger}$$

**Trace preservation:** $\sum_k K_k^\dagger K_k = I$

**Kraus rank:** Minimum number of operators needed (≤ $d_A d_B$)

### 4. Stinespring Dilation

$$\mathcal{E}(\rho) = \text{Tr}_E[V \rho V^\dagger]$$

where $V: \mathcal{H}_A \to \mathcal{H}_B \otimes \mathcal{H}_E$ is an isometry ($V^\dagger V = I_A$).

**Physical interpretation:**
1. System interacts with environment
2. Trace out environment
3. Kraus operators: $K_k = (\langle k|_E \otimes I_B)V$

### 5. Choi-Jamiolkowski Isomorphism

**Choi matrix:**
$$\boxed{J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|)}$$

where $|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|ii\rangle$.

**Key theorem:** $\mathcal{E}$ is CPTP ⟺ $J(\mathcal{E}) \geq 0$ and $\text{Tr}_B[J(\mathcal{E})] = I_A/d$

**Channel from Choi:**
$$\mathcal{E}(\rho) = d \cdot \text{Tr}_A[({\rho^T \otimes I_B}) J(\mathcal{E})]$$

### 6. Unitary Freedom in Kraus Representation

If $\{K_k\}$ and $\{L_l\}$ both represent $\mathcal{E}$, then:
$$K_k = \sum_l U_{kl} L_l$$

for some unitary matrix $U$.

**Physical meaning:** Different environment decompositions give same channel.

---

## Core Concepts: Important Channels

### 7. Common Single-Qubit Channels

**Identity:** $\mathcal{I}(\rho) = \rho$

**Bit-flip:** $\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X$

**Phase-flip:** $\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$

**Depolarizing:** $\mathcal{E}_{dep}(\rho) = (1-p)\rho + p\frac{I}{2}$

**Amplitude damping:**
$$K_0 = \begin{pmatrix}1 & 0\\0 & \sqrt{1-\gamma}\end{pmatrix}, \quad K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\0 & 0\end{pmatrix}$$

### 8. Channel Properties

| Property | Definition | Example |
|----------|------------|---------|
| Unital | $\mathcal{E}(I) = I$ | Pauli channels |
| Trace-preserving | $\text{Tr}[\mathcal{E}(\rho)] = \text{Tr}[\rho]$ | All physical channels |
| Unitary | $\mathcal{E}(\rho) = U\rho U^\dagger$ | Ideal gates |
| Entanglement-breaking | $(\mathcal{E} \otimes \mathcal{I})(\rho)$ always separable | Measurement |

### 9. Channel Composition

**Sequential:** $(\mathcal{E}_2 \circ \mathcal{E}_1)(\rho) = \mathcal{E}_2(\mathcal{E}_1(\rho))$
- Kraus: $K_{jk} = L_j M_k$
- Choi: NOT simple product!

**Parallel:** $(\mathcal{E}_1 \otimes \mathcal{E}_2)(\rho)$
- Kraus: $K_{jk} = L_j \otimes M_k$
- Choi: tensor product

---

## Core Concepts: Channel Capacities

### 10. Classical Capacity

Maximum rate of classical information through quantum channel:
$$C(\mathcal{E}) = \lim_{n\to\infty} \frac{1}{n} \chi(\mathcal{E}^{\otimes n})$$

**Holevo quantity:**
$$\chi(\mathcal{E}) = \max_{\{p_i, \rho_i\}} S\left(\sum_i p_i \mathcal{E}(\rho_i)\right) - \sum_i p_i S(\mathcal{E}(\rho_i))$$

### 11. Quantum Capacity

Maximum rate of quantum information:
$$Q(\mathcal{E}) = \lim_{n\to\infty} \frac{1}{n} I_c(\mathcal{E}^{\otimes n})$$

**Coherent information:**
$$I_c(\mathcal{E}, \rho) = S(\mathcal{E}(\rho)) - S((\mathcal{E} \otimes \mathcal{I})|\psi\rangle\langle\psi|)$$

where $|\psi\rangle$ purifies $\rho$.

---

## Integration Summary

### 12. Connection to Error Correction

**Error channels** (Month 24) are quantum channels!

**Error correction** reverses channel effects:
$$\mathcal{R} \circ \mathcal{E}(\rho) = \rho \quad \text{for } \rho \in \mathcal{C}$$

**Knill-Laflamme** = condition for recovery channel $\mathcal{R}$ to exist.

### 13. Connection to Open Systems

**Lindblad evolution** generates channels:
$$\mathcal{E}_t = e^{\mathcal{L}t}$$

**Kraus operators** encode environment measurement outcomes.

**Choi matrix** gives entanglement structure of channel.

---

## Practice Problems

### Problem 1: Kraus to Choi

Find the Choi matrix for the bit-flip channel with error probability $p$.

**Solution:**
$$\mathcal{E}(\rho) = (1-p)\rho + pX\rho X$$

Kraus operators: $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}X$

$$J = (K_0 \otimes I)|\Phi^+\rangle\langle\Phi^+|(K_0^\dagger \otimes I) + (K_1 \otimes I)|\Phi^+\rangle\langle\Phi^+|(K_1^\dagger \otimes I)$$

$$= (1-p)|\Phi^+\rangle\langle\Phi^+| + p(X\otimes I)|\Phi^+\rangle\langle\Phi^+|(X\otimes I)$$

$$= (1-p)|\Phi^+\rangle\langle\Phi^+| + p|\Psi^+\rangle\langle\Psi^+|$$

### Problem 2: Verify CPTP

Show the depolarizing channel is CPTP.

**Solution:**
Kraus operators: $K_0 = \sqrt{1-3p/4}I$, $K_1 = \sqrt{p/4}X$, $K_2 = \sqrt{p/4}Y$, $K_3 = \sqrt{p/4}Z$

TP check:
$$\sum_k K_k^\dagger K_k = (1-3p/4)I + (p/4)(I + I + I) = I \checkmark$$

CP: By construction (Kraus form is always CP).

### Problem 3: Channel Composition

Find Kraus operators for two consecutive bit-flip channels with probabilities $p$ and $q$.

**Solution:**
First channel: $\{M_0 = \sqrt{1-p}I, M_1 = \sqrt{p}X\}$
Second channel: $\{L_0 = \sqrt{1-q}I, L_1 = \sqrt{q}X\}$

Composed:
- $K_{00} = L_0 M_0 = \sqrt{(1-p)(1-q)}I$
- $K_{01} = L_0 M_1 = \sqrt{p(1-q)}X$
- $K_{10} = L_1 M_0 = \sqrt{q(1-p)}X$
- $K_{11} = L_1 M_1 = \sqrt{pq}I$

Effective single bit-flip with $p_{eff} = p + q - 2pq$.

---

## Computational Lab

```python
"""Day 669: Month 23 Review - Quantum Channels"""

import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(A, B):
    return np.kron(A, B)

print("Month 23 Review: Quantum Channels")
print("=" * 60)

# ============================================
# Part 1: Kraus Representation
# ============================================
print("\nPART 1: Kraus Representation")
print("-" * 40)

def apply_channel(rho, kraus_ops):
    """Apply quantum channel defined by Kraus operators."""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result

def check_tp(kraus_ops):
    """Check trace-preserving condition."""
    sum_kdk = sum(K.conj().T @ K for K in kraus_ops)
    return np.allclose(sum_kdk, np.eye(len(kraus_ops[0])))

# Bit-flip channel
p = 0.1
K_bf = [np.sqrt(1-p) * I, np.sqrt(p) * X]
print(f"Bit-flip (p={p}): TP = {check_tp(K_bf)}")

# Depolarizing channel
K_dep = [np.sqrt(1-3*p/4) * I, np.sqrt(p/4) * X, np.sqrt(p/4) * Y, np.sqrt(p/4) * Z]
print(f"Depolarizing (p={p}): TP = {check_tp(K_dep)}")

# Amplitude damping
gamma = 0.2
K_ad = [
    np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex),
    np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
]
print(f"Amplitude damping (γ={gamma}): TP = {check_tp(K_ad)}")

# Test on |+⟩ state
rho_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

print("\nChannel effects on |+⟩ state:")
for name, K in [("Bit-flip", K_bf), ("Depolarizing", K_dep), ("Amp. damp.", K_ad)]:
    rho_out = apply_channel(rho_plus, K)
    purity = np.real(np.trace(rho_out @ rho_out))
    print(f"  {name}: purity = {purity:.4f}")

# ============================================
# Part 2: Choi Matrix
# ============================================
print("\n" + "=" * 60)
print("PART 2: Choi-Jamiolkowski Isomorphism")
print("-" * 40)

# Maximally entangled state |Φ+⟩
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
Phi_plus = (tensor(ket_0, ket_0) + tensor(ket_1, ket_1)) / np.sqrt(2)
Phi_plus_dm = Phi_plus @ Phi_plus.conj().T

def compute_choi(kraus_ops):
    """Compute Choi matrix from Kraus operators."""
    d = len(kraus_ops[0])
    J = np.zeros((d**2, d**2), dtype=complex)
    for K in kraus_ops:
        # (K ⊗ I)|Φ+⟩
        K_I = tensor(K, np.eye(d, dtype=complex))
        vec = K_I @ Phi_plus.flatten()
        J += np.outer(vec, vec.conj())
    return J

def check_choi_cptp(J):
    """Check CPTP conditions from Choi matrix."""
    d = int(np.sqrt(len(J)))

    # CP: J ≥ 0
    eigenvalues = np.linalg.eigvalsh(J)
    is_positive = np.all(eigenvalues >= -1e-10)

    # TP: Tr_B(J) = I/d
    J_reshaped = J.reshape(d, d, d, d)
    partial_trace = np.trace(J_reshaped, axis1=1, axis2=3)
    is_tp = np.allclose(partial_trace, np.eye(d) / d)

    return is_positive, is_tp

# Compute and check Choi matrices
for name, K in [("Bit-flip", K_bf), ("Depolarizing", K_dep), ("Amp. damp.", K_ad)]:
    J = compute_choi(K)
    is_cp, is_tp = check_choi_cptp(J)
    print(f"{name}: CP = {is_cp}, TP = {is_tp}")

# ============================================
# Part 3: Channel Properties
# ============================================
print("\n" + "=" * 60)
print("PART 3: Channel Properties")
print("-" * 40)

def is_unital(kraus_ops):
    """Check if channel is unital: E(I) = I."""
    d = len(kraus_ops[0])
    output = apply_channel(np.eye(d, dtype=complex), kraus_ops)
    return np.allclose(output, np.eye(d))

def channel_contraction(kraus_ops):
    """Measure how much Bloch sphere contracts."""
    # Apply to |+⟩, |+i⟩, |0⟩
    rho_x = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    rho_y = 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
    rho_z = np.array([[1, 0], [0, 0]], dtype=complex)

    out_x = apply_channel(rho_x, kraus_ops)
    out_y = apply_channel(rho_y, kraus_ops)
    out_z = apply_channel(rho_z, kraus_ops)

    # Bloch vectors
    r_x = 2*np.real(out_x[0,1])
    r_y = 2*np.real(out_y[0,1])
    r_z = np.real(out_z[0,0] - out_z[1,1])

    return np.abs(r_x), np.abs(r_y), np.abs(r_z)

print("\nChannel properties:")
for name, K in [("Bit-flip", K_bf), ("Depolarizing", K_dep), ("Amp. damp.", K_ad)]:
    unital = is_unital(K)
    contractions = channel_contraction(K)
    print(f"  {name}:")
    print(f"    Unital: {unital}")
    print(f"    Bloch contractions (x,y,z): ({contractions[0]:.3f}, {contractions[1]:.3f}, {contractions[2]:.3f})")

# ============================================
# Part 4: Channel Composition
# ============================================
print("\n" + "=" * 60)
print("PART 4: Channel Composition")
print("-" * 40)

def compose_kraus(K1, K2):
    """Compose two channels: K2 ∘ K1."""
    composed = []
    for L in K2:
        for M in K1:
            composed.append(L @ M)
    return composed

# Two bit-flips with different probabilities
p1, p2 = 0.1, 0.2
K1 = [np.sqrt(1-p1) * I, np.sqrt(p1) * X]
K2 = [np.sqrt(1-p2) * I, np.sqrt(p2) * X]

K_composed = compose_kraus(K1, K2)
print(f"Bit-flip p1={p1} followed by p2={p2}:")

# Find effective p
# The composed channel should be bit-flip with p_eff = p1 + p2 - 2*p1*p2
p_eff = p1 + p2 - 2*p1*p2
print(f"  Expected p_eff = {p_eff:.4f}")

# Verify by action
rho_test = np.array([[1, 0.5], [0.5, 0]], dtype=complex)  # Arbitrary state
out_composed = apply_channel(rho_test, K_composed)

K_eff = [np.sqrt(1-p_eff) * I, np.sqrt(p_eff) * X]
out_eff = apply_channel(rho_test, K_eff)

print(f"  Composed and effective match: {np.allclose(out_composed, out_eff)}")

# ============================================
# Part 5: Stinespring Dilation
# ============================================
print("\n" + "=" * 60)
print("PART 5: Stinespring Dilation")
print("-" * 40)

def stinespring_dilation(kraus_ops):
    """Construct Stinespring isometry from Kraus operators."""
    d = len(kraus_ops[0])
    r = len(kraus_ops)  # Number of Kraus operators = environment dimension

    # V: H_S -> H_S ⊗ H_E
    # V|ψ⟩ = Σ_k K_k|ψ⟩ ⊗ |k⟩
    V = np.zeros((d * r, d), dtype=complex)
    for k, K in enumerate(kraus_ops):
        V[k*d:(k+1)*d, :] = K

    return V

# Check V†V = I for amplitude damping
V_ad = stinespring_dilation(K_ad)
VdV = V_ad.conj().T @ V_ad
print(f"Amplitude damping V†V = I: {np.allclose(VdV, np.eye(2))}")

# Verify channel action: E(ρ) = Tr_E[VρV†]
rho_test = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
VrhoVd = V_ad @ rho_test @ V_ad.conj().T

# Partial trace over environment (2D)
d_E = len(K_ad)
d_S = 2
VrhoVd_reshaped = VrhoVd.reshape(d_E, d_S, d_E, d_S)
rho_out_stine = np.trace(VrhoVd_reshaped, axis1=0, axis2=2)

rho_out_kraus = apply_channel(rho_test, K_ad)
print(f"Stinespring matches Kraus: {np.allclose(rho_out_stine, rho_out_kraus)}")

print("\n" + "=" * 60)
print("Review Complete!")
```

---

## Summary

### Channel Representations

| Representation | Definition | Use |
|----------------|------------|-----|
| Kraus | $\mathcal{E}(\rho) = \sum_k K_k\rho K_k^\dagger$ | Computation, simulation |
| Stinespring | $\mathcal{E}(\rho) = \text{Tr}_E[V\rho V^\dagger]$ | Physical interpretation |
| Choi | $J = (\mathcal{E} \otimes \mathcal{I})\|\Phi^+\rangle\langle\Phi^+\|$ | Mathematical analysis |

### Key Properties

- **CPTP:** Physical validity requirement
- **Unital:** Preserves maximally mixed state
- **Kraus rank:** Number of environment dimensions needed

### Connections

- Open systems → Kraus from Lindblad
- Error correction → Recovery channels
- Information theory → Channel capacities

---

## Preview: Day 670

Tomorrow: **Month 24 Review** - Error channels and quantum error correction fundamentals!
