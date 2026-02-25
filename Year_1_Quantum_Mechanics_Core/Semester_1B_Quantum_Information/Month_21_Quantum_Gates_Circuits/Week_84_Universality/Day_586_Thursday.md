# Day 586: Native Gate Sets

## Overview
**Day 586** | Week 84, Day 5 | Year 1, Month 21 | Hardware-Specific Gate Sets

Today we explore native gate sets—the gates physically implemented by different quantum hardware platforms. Understanding native gates is essential for efficient quantum circuit compilation.

---

## Learning Objectives

1. Identify native gate sets for superconducting, ion trap, and photonic systems
2. Understand why different hardware uses different gates
3. Convert between abstract and native gate sets
4. Evaluate trade-offs in gate set choices
5. Connect physical implementation to gate fidelity
6. Appreciate the diversity of quantum computing platforms

---

## Core Content

### What Are Native Gates?

**Native gates** are the quantum operations directly implementable by hardware:
- Determined by physical system and control mechanisms
- Typically a small set (1-5 gate types)
- All other gates must be decomposed into natives

**Goal:** Minimize native gate count for any given circuit.

### Superconducting Qubits (IBM, Google, Rigetti)

**Physical system:** Josephson junction circuits (transmons)

**Common native gate sets:**

**IBM Quantum:**
$$\{R_z(\theta), \sqrt{X}, X, CNOT\}$$

Recent IBM systems use:
$$\{R_z(\theta), \sqrt{X}, ECR\}$$

where ECR (echoed cross-resonance) is the native two-qubit gate.

**Google Sycamore:**
$$\{R_z(\theta), \sqrt{W}, fSim(\theta, \phi)\}$$

where $\sqrt{W} = R_{\hat{n}}(\pi/2)$ for $\hat{n} = (X+Y)/\sqrt{2}$

**fSim gate:**
$$fSim(\theta, \phi) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -i\sin\theta & 0 \\ 0 & -i\sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & e^{-i\phi} \end{pmatrix}$$

**Rigetti:**
$$\{R_z(\theta), R_x(\theta), CZ\}$$

### Trapped Ion Qubits (IonQ, Honeywell/Quantinuum)

**Physical system:** Individual atomic ions in electromagnetic traps

**Native gate set:**
$$\{R(\theta, \phi), XX(\theta)\}$$

where $R(\theta, \phi)$ is arbitrary single-qubit rotation:
$$R(\theta, \phi) = \cos(\theta/2)I - i\sin(\theta/2)(\cos\phi \cdot X + \sin\phi \cdot Y)$$

**XX (Molmer-Sorensen) gate:**
$$XX(\theta) = \exp\left(-i\frac{\theta}{2} X \otimes X\right)$$

**Key advantage:** All-to-all connectivity (any qubit can interact with any other).

### Photonic Qubits (Xanadu, PsiQuantum)

**Physical system:** Photons with polarization or path encoding

**Gate implementations:**
- Single-qubit: Waveplates, beam splitters
- Two-qubit: Measurement-based (KLM scheme)

**Native operations:**
$$\{R(\theta), PBS, \text{Measurement}\}$$

PBS = Polarizing Beam Splitter

**Key feature:** Measurement-induced nonlinearity (probabilistic two-qubit gates).

### Neutral Atom Qubits (QuEra, Pasqal)

**Physical system:** Neutral atoms in optical tweezers

**Native gates:**
$$\{R_z(\theta), R_x(\theta), CZ_{\text{Rydberg}}\}$$

**Rydberg gate:** Uses strong interactions when atoms are excited to Rydberg states.

**Key feature:** Reconfigurable connectivity via atom movement.

### Comparison Table

| Platform | Single-Qubit | Two-Qubit | Connectivity | Speed |
|----------|--------------|-----------|--------------|-------|
| Superconducting | $R_z$, $\sqrt{X}$ | CNOT, CZ, ECR | Fixed (sparse) | ~ns |
| Ion Trap | $R(\theta,\phi)$ | XX | All-to-all | ~μs |
| Photonic | Waveplates | Probabilistic | Configurable | ~ps |
| Neutral Atom | $R_z$, $R_x$ | CZ (Rydberg) | Reconfigurable | ~μs |

### Virtual vs Physical Gates

**Virtual gates:** Implemented in software by adjusting phases.

**Example:** $R_z(\theta)$ on superconducting qubits
- Often implemented by adjusting the phase of subsequent pulses
- Zero error, zero time!

**Physical gates:** Require actual control pulses.

**Example:** $\sqrt{X}$ requires calibrated microwave pulse.

### Gate Fidelity Considerations

**Typical fidelities (2024):**

| Platform | 1Q Fidelity | 2Q Fidelity |
|----------|-------------|-------------|
| Superconducting | 99.9%+ | 99.5%+ |
| Ion Trap | 99.99%+ | 99.9%+ |
| Photonic | 99.9%+ | ~99% (probabilistic) |
| Neutral Atom | 99.9%+ | 99%+ |

**Key insight:** Two-qubit gates are typically 10-100× more error-prone than single-qubit gates.

### Decomposition Examples

**CNOT from CZ:**
```
   ──●──    =    ───────●───────
     │                  │
   ──⊕──    =    ──[H]──●──[H]──
```

**CZ from CNOT:**
```
   ──●──    =    ──[H]──⊕──[H]──
     │                  │
   ──●──    =    ───────●───────
```

**SWAP from CNOT:**
```
   ──×──    =    ──●──⊕──●──
     │             │  │  │
   ──×──    =    ──⊕──●──⊕──
```

### Hardware Topology

**Connectivity constraints:**
- Superconducting: Fixed nearest-neighbor (heavy hex, grid)
- Ion trap: All-to-all (but slower with more ions)
- Neutral atoms: Programmable (can move atoms)

**Impact:** Non-adjacent qubits require SWAP operations.

### Calibration and Variability

**Native gates must be calibrated:**
- Daily calibration typical for superconducting
- Gate parameters can drift
- Error rates vary across qubits

**Consequence:** Optimal compilation depends on current calibration data.

---

## Worked Examples

### Example 1: Hadamard on IBM

Express $H$ using IBM native gates $\{R_z, \sqrt{X}\}$.

**Solution:**

$$H = R_z(\pi) \cdot \sqrt{X} \cdot R_z(\pi/2)$$

Verification:
$$R_z(\pi) = \begin{pmatrix} e^{-i\pi/2} & 0 \\ 0 & e^{i\pi/2} \end{pmatrix} = -i\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = -iZ$$

$$R_z(\pi/2) = \begin{pmatrix} e^{-i\pi/4} & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

$$\sqrt{X} = \frac{1}{2}\begin{pmatrix} 1+i & 1-i \\ 1-i & 1+i \end{pmatrix}$$

The product equals $H$ up to global phase.

### Example 2: CNOT from Ion Trap XX Gate

Express CNOT using $XX(\pi/4)$ and single-qubit rotations.

**Solution:**

$$CNOT = (I \otimes R_y(\pi/2)) \cdot XX(\pi/4) \cdot (R_z(-\pi/2) \otimes R_x(-\pi/2))$$

This decomposition uses:
- 1 XX gate
- 3 single-qubit rotations

### Example 3: S Gate from Native Gates

Express $S$ gate using IBM natives.

**Solution:**

$S = R_z(\pi/2)$

On IBM: This is directly a native gate!

Even better: $R_z$ gates are virtual (frame changes), so $S$ has zero error.

---

## Practice Problems

### Problem 1: T Gate Decomposition
Express $T = R_z(\pi/4)$ using IBM native gates.

### Problem 2: Two-Qubit Count
How many native two-qubit gates are needed to implement SWAP on:
a) IBM (CNOT native)
b) Rigetti (CZ native)

### Problem 3: Ion Trap
Express $R_z(\theta)$ using the ion trap native $R(\theta, \phi)$.

### Problem 4: Connectivity Overhead
On a linear chain of 5 qubits, how many SWAPs are needed to execute CNOT(0, 4)?

---

## Computational Lab

```python
"""Day 586: Native Gate Sets"""
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(A, B):
    return np.kron(A, B)

def Rx(theta):
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * X

def Ry(theta):
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * Y

def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)

# ===== IBM Native Gates =====
print("=" * 60)
print("IBM Native Gate Set")
print("=" * 60)

def sqrt_X():
    """IBM's sqrt(X) gate"""
    return (1/2) * np.array([
        [1 + 1j, 1 - 1j],
        [1 - 1j, 1 + 1j]
    ], dtype=complex)

def ECR():
    """Echoed Cross-Resonance gate"""
    return (1/np.sqrt(2)) * np.array([
        [0, 1, 0, 1j],
        [1, 0, -1j, 0],
        [0, 1j, 0, 1],
        [-1j, 0, 1, 0]
    ], dtype=complex)

print("\nIBM native gates: {Rz(θ), sqrt(X), X, CNOT}")
print("\nNote: Rz is a 'virtual' gate (phase adjustment, zero error)")

# Verify Hadamard decomposition
print("\nHadamard decomposition:")
print("  H = Rz(π) · sqrt(X) · Rz(π/2)")
H_decomp = Rz(np.pi) @ sqrt_X() @ Rz(np.pi/2)
print(f"  Correct: {np.allclose(H_decomp / H_decomp[0,0] * H[0,0], H)}")

# ===== Google Native Gates =====
print("\n" + "=" * 60)
print("Google Sycamore Native Gate Set")
print("=" * 60)

def fSim(theta, phi):
    """Google's fSim gate"""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)]
    ], dtype=complex)

print("\nGoogle native gates: {Rz(θ), sqrt(W), fSim(θ, φ)}")
print("\nfSim gate family:")
print("  fSim(π/2, 0) = iSWAP")
print("  fSim(π/2, π) = SWAP + CZ")

iSWAP = fSim(np.pi/2, 0)
print(f"\niSWAP:\n{np.round(iSWAP, 3)}")

# ===== Ion Trap Native Gates =====
print("\n" + "=" * 60)
print("Ion Trap Native Gate Set")
print("=" * 60)

def R_ion(theta, phi):
    """Ion trap single-qubit rotation R(θ, φ)"""
    axis = np.cos(phi) * X + np.sin(phi) * Y
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * axis

def XX(theta):
    """Molmer-Sorensen XX gate"""
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    return np.array([
        [c, 0, 0, -1j*s],
        [0, c, -1j*s, 0],
        [0, -1j*s, c, 0],
        [-1j*s, 0, 0, c]
    ], dtype=complex)

print("\nIon trap native gates: {R(θ, φ), XX(θ)}")
print("\nXX gate at θ=π/4 (maximally entangling):")
XX_max = XX(np.pi/4)
print(np.round(XX_max, 3))

# Verify: Rz from R(θ, φ)
# Rz(θ) = R(θ, -π/2) · R(θ, π/2) · some phase
print("\nExpressing Rz using R(θ, φ):")
print("  Rz(θ) can be done via R(-π/2, π/2) · R(θ, 0) · R(π/2, π/2)")

# ===== Gate Decompositions =====
print("\n" + "=" * 60)
print("Common Gate Decompositions")
print("=" * 60)

# CNOT from CZ
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

# CNOT = (I ⊗ H) CZ (I ⊗ H)
CNOT_from_CZ = tensor(I, H) @ CZ @ tensor(I, H)
print(f"\nCNOT from CZ: CNOT = (I⊗H) CZ (I⊗H)")
print(f"  Correct: {np.allclose(CNOT, CNOT_from_CZ)}")

# SWAP from CNOTs
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# CNOT(0,1)
CNOT_01 = CNOT

# CNOT(1,0)
CNOT_10 = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

SWAP_from_CNOT = CNOT_01 @ CNOT_10 @ CNOT_01
print(f"\nSWAP from CNOTs: SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)")
print(f"  Correct: {np.allclose(SWAP, SWAP_from_CNOT)}")

# ===== Gate Costs =====
print("\n" + "=" * 60)
print("Two-Qubit Gate Costs for Common Operations")
print("=" * 60)

decompositions = {
    'CNOT': {'CNOT': 1, 'CZ': 1, 'XX': 1},
    'CZ': {'CNOT': 1, 'CZ': 1, 'XX': 1},
    'SWAP': {'CNOT': 3, 'CZ': 3, 'XX': 3},
    'iSWAP': {'CNOT': 2, 'CZ': 2, 'XX': 1},
    'Toffoli': {'CNOT': 6, 'CZ': 6, 'XX': 6},
}

print(f"\n{'Operation':<12} {'CNOT-based':<12} {'CZ-based':<12} {'XX-based':<12}")
print("-" * 50)
for op, costs in decompositions.items():
    print(f"{op:<12} {costs['CNOT']:<12} {costs['CZ']:<12} {costs['XX']:<12}")

# ===== Connectivity Overhead =====
print("\n" + "=" * 60)
print("Connectivity and SWAP Overhead")
print("=" * 60)

print("""
Linear chain connectivity: 0 - 1 - 2 - 3 - 4

To execute CNOT(0, 4):
  - Need to route qubits to be adjacent
  - Options:
    a) SWAP chain: 0↔1, 1↔2, 2↔3, then CNOT(3,4)
       Cost: 3 SWAPs × 3 CNOTs/SWAP = 9 CNOTs + 1 CNOT = 10 CNOTs
    b) Use intermediate qubits (if algorithm allows)

Grid connectivity (Heavy Hex - IBM):
  - Average path length: ~O(√n)
  - SWAP overhead can be significant

All-to-all connectivity (Ion Trap):
  - Any CNOT: 1 two-qubit gate
  - No SWAP overhead!
  - But gate time increases with qubit count
""")

# ===== Platform Comparison =====
print("\n" + "=" * 60)
print("Platform Comparison Summary")
print("=" * 60)
print("""
                    Superconducting    Ion Trap    Photonic    Neutral Atom
---------------------------------------------------------------------------
Native 1Q           Rz, √X             R(θ,φ)      Waveplates  Rz, Rx
Native 2Q           CNOT/CZ/ECR        XX          PBS         CZ
Connectivity        Fixed (sparse)     All-to-all  Flexible    Reconfigurable
Gate speed          ~10-100 ns         ~10-100 μs  ~ps         ~1-10 μs
1Q Fidelity         99.9%+             99.99%+     99.9%+      99.9%+
2Q Fidelity         99.5%+             99.9%+      ~99%        99%+
Scalability         100+ qubits        ~30 qubits  Growing     ~100 atoms
T1 (coherence)      ~100 μs            ~1 s        N/A         ~seconds

Best for:
- Superconducting: Fast gates, established technology
- Ion Trap: High fidelity, full connectivity
- Photonic: Quantum networking, room temperature
- Neutral Atom: Scalability, reconfigurability
""")
```

---

## Summary

### Native Gate Sets by Platform

| Platform | 1-Qubit | 2-Qubit | Key Feature |
|----------|---------|---------|-------------|
| IBM | $R_z$, $\sqrt{X}$ | CNOT, ECR | Virtual $R_z$ |
| Google | $R_z$, $\sqrt{W}$ | fSim | Tunable 2Q |
| Ion Trap | $R(\theta,\phi)$ | XX | All-to-all |
| Neutral Atom | $R_z$, $R_x$ | CZ | Reconfigurable |

### Key Formulas

| Decomposition | Formula |
|---------------|---------|
| CNOT from CZ | $(I \otimes H) \cdot CZ \cdot (I \otimes H)$ |
| CZ from CNOT | $(I \otimes H) \cdot CNOT \cdot (I \otimes H)$ |
| SWAP | 3 CNOTs |
| H from IBM | $R_z(\pi) \cdot \sqrt{X} \cdot R_z(\pi/2)$ |

### Key Takeaways

1. **Native gates** are hardware-specific
2. **Virtual gates** ($R_z$) have zero error on some platforms
3. **Two-qubit gates** are the bottleneck (10-100x more error)
4. **Connectivity** strongly affects SWAP overhead
5. **Trade-offs** exist between speed, fidelity, and connectivity
6. **Compilation** must account for native gate sets

---

## Daily Checklist

- [ ] I can identify native gates for major quantum platforms
- [ ] I understand why different platforms use different gates
- [ ] I can decompose common gates into native gates
- [ ] I understand the impact of connectivity on circuit cost
- [ ] I can evaluate trade-offs between platforms
- [ ] I ran the computational lab and verified decompositions

---

*Next: Day 587 — Compiling to Hardware*
