# Day 537: Bound Entanglement

## Overview
**Day 537** | Week 77, Day 5 | Year 1, Month 20 | Undistillable Entanglement

Today we study bound entanglement—states that are entangled but cannot be distilled into pure entanglement using LOCC operations.

---

## Learning Objectives
1. Define distillable vs bound entanglement
2. Understand the PPT entangled state phenomenon
3. Analyze the first bound entangled states
4. Explore the role of bound entanglement in quantum information
5. Connect to activation and unlocking
6. Understand current open problems

---

## Core Content

### Entanglement Distillation

**Distillation:** Transform n copies of a noisy entangled state into m < n copies of near-perfect Bell pairs using LOCC.

$$\rho^{\otimes n} \xrightarrow{LOCC} |\Phi^+\rangle^{\otimes m}$$

**Distillable entanglement:** Rate $E_D(\rho) = \lim_{n\to\infty} m/n$

### Types of Entangled States

```
                 ENTANGLED STATES
                       │
           ┌───────────┴───────────┐
           │                       │
      NPT States              PPT States
    (always distillable)    (can be entangled!)
           │                       │
           ▼                       ▼
    FREE ENTANGLEMENT        BOUND ENTANGLEMENT
    E_D > 0                  E_D = 0, but E_F > 0
```

### NPT vs PPT Entanglement

**NPT (Negative Partial Transpose):** $\rho^{T_B}$ has negative eigenvalues
- Always distillable: $E_D(\rho) > 0$
- Called "free" entanglement

**PPT Entangled:** $\rho^{T_B} \geq 0$ but ρ entangled
- **Bound entanglement:** Cannot distill any Bell pairs!
- Requires dimension ≥ 3×3

### The First Bound Entangled State

**Horodecki (1998)** constructed the first example in 3×3:

$$\rho_{BE} = \frac{1}{8a+1}\begin{pmatrix} a & 0 & 0 & 0 & a & 0 & 0 & 0 & a \\ 0 & a & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & a & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & a & 0 & 0 & 0 & 0 & 0 \\ a & 0 & 0 & 0 & a & 0 & 0 & 0 & a \\ 0 & 0 & 0 & 0 & 0 & a & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & \frac{1+a}{2} & 0 & \frac{\sqrt{1-a^2}}{2} \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & a & 0 \\ a & 0 & 0 & 0 & a & 0 & \frac{\sqrt{1-a^2}}{2} & 0 & \frac{1+a}{2} \end{pmatrix}$$

For $0 < a < 1$:
- ρ is PPT (positive partial transpose)
- ρ is entangled (detected by range criterion)
- ρ is NOT distillable

### Range Criterion Detection

**Range criterion:** If ρ is separable, then Range(ρ) is spanned by product vectors.

For the Horodecki state: Range(ρ) contains unextendible product basis (UPB) → entangled!

### Why PPT States are Undistillable

**Key theorem:** Any distillation protocol must produce an NPT output state.

Since distillation uses LOCC and:
$$(\rho^{\otimes n})^{T_B} = (\rho^{T_B})^{\otimes n} \geq 0$$

LOCC cannot create negative partial transpose from PPT input!

### Activation of Bound Entanglement

**Activation:** Bound entanglement can help distill other entanglement.

$$\rho_{BE} \otimes \rho_{free} \xrightarrow{LOCC} \text{more Bell pairs than } \rho_{free} \text{ alone}$$

This shows bound entanglement has "hidden" quantum correlations.

### Bound Entanglement Properties

| Property | Free (NPT) | Bound (PPT) |
|----------|------------|-------------|
| Distillable | Yes | No |
| Teleportation | Yes | No (directly) |
| Dense coding | Yes | No (directly) |
| Key distillation | Yes | Sometimes! |
| Activation | — | Can assist |

### Open Problems

1. **NPT Bound Entanglement?** Do NPT states exist with $E_D = 0$?
2. **Distillability Problem:** Given ρ, determine if $E_D(\rho) > 0$
3. **Quantifying bound entanglement**

---

## Worked Examples

### Example 1: UPB States
The UPB (Unextendible Product Basis) method constructs bound entangled states.

In 3×3, the 5-state UPB:
$$|0\rangle|0-1\rangle, |2\rangle|1-2\rangle, |0-1\rangle|2\rangle, |1-2\rangle|0\rangle, |0+1+2\rangle|0+1+2\rangle$$

where $|a-b\rangle = (|a\rangle - |b\rangle)/\sqrt{2}$.

The projector onto the orthogonal complement:
$$\rho = \frac{1}{4}(I_9 - P_{UPB})$$

This is PPT but entangled (no product states in range).

### Example 2: Werner State in Higher Dimensions
In d×d, the isotropic state:
$$\rho_p = p|\Phi^+_d\rangle\langle\Phi^+_d| + (1-p)\frac{I}{d^2}$$

where $|\Phi^+_d\rangle = \sum_i |ii\rangle/\sqrt{d}$.

For d ≥ 3:
- Separable: $p \leq 1/(d+1)$
- PPT entangled (bound): $1/(d+1) < p \leq 1/d$ [possible range]
- NPT entangled (free): $p > $ some threshold

### Example 3: Checking Range Criterion
For a 2-qubit state, the range criterion is:
- If Range(ρ) ⊆ span of product states → may be separable
- If Range(ρ) contains non-product vectors only → separable still possible
- Combined with PPT → determines separability in 2×2

---

## Practice Problems

### Problem 1: PPT Check
Show that any separable state must be PPT.

### Problem 2: No Bound Entanglement in 2×2
Prove there is no bound entanglement in 2×2 systems.

### Problem 3: UPB Construction
Find a UPB in 2×4 Hilbert space.

---

## Computational Lab

```python
"""Day 537: Bound Entanglement"""
import numpy as np
from scipy.linalg import eigvalsh, null_space

def partial_transpose_B(rho, dim_A, dim_B):
    """Partial transpose over B"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_TB = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_TB.reshape(dim_A * dim_B, dim_A * dim_B)

def is_ppt(rho, dim_A, dim_B, tol=1e-10):
    """Check PPT condition"""
    rho_TB = partial_transpose_B(rho, dim_A, dim_B)
    return np.min(eigvalsh(rho_TB)) >= -tol

def horodecki_bound_entangled(a):
    """
    Construct Horodecki bound entangled state in 3x3.
    Valid for 0 < a < 1.
    """
    rho = np.zeros((9, 9), dtype=complex)

    # Fill according to the formula
    norm = 1 / (8*a + 1)

    # a on certain diagonals
    for i in [0, 1, 2, 3, 5, 7]:
        rho[i, i] = a

    # Specific entries
    rho[4, 4] = a
    rho[6, 6] = (1 + a) / 2
    rho[8, 8] = (1 + a) / 2

    # Off-diagonal entries
    rho[0, 4] = rho[4, 0] = a
    rho[0, 8] = rho[8, 0] = a
    rho[4, 8] = rho[8, 4] = a

    b = np.sqrt(1 - a**2) / 2
    rho[6, 8] = rho[8, 6] = b

    return norm * rho

def check_positive_semidefinite(rho, tol=1e-10):
    """Check if matrix is positive semidefinite"""
    eigenvalues = eigvalsh(rho)
    return np.all(eigenvalues >= -tol)

def check_range_for_products(rho, dim_A, dim_B, num_checks=1000):
    """
    Heuristic check if range contains product vectors.
    Returns fraction of random product vectors in range.
    """
    # Get range (column space)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    range_basis = eigenvectors[:, eigenvalues > 1e-10]

    if range_basis.shape[1] == 0:
        return 0.0

    # Project random product vectors onto range
    overlaps = []
    for _ in range(num_checks):
        # Random product state
        a = np.random.randn(dim_A) + 1j * np.random.randn(dim_A)
        a /= np.linalg.norm(a)
        b = np.random.randn(dim_B) + 1j * np.random.randn(dim_B)
        b /= np.linalg.norm(b)
        prod = np.kron(a, b)

        # Project onto range
        proj = range_basis @ range_basis.conj().T @ prod
        overlap = np.abs(np.vdot(prod, proj))
        overlaps.append(overlap)

    return np.mean(overlaps)

# Analyze Horodecki bound entangled state
print("=== Horodecki Bound Entangled State (3×3) ===\n")

for a in [0.1, 0.5, 0.9]:
    rho = horodecki_bound_entangled(a)

    # Check valid density matrix
    is_pos = check_positive_semidefinite(rho)
    trace = np.trace(rho).real

    # Check PPT
    ppt = is_ppt(rho, 3, 3)

    # Check eigenvalues of partial transpose
    rho_TB = partial_transpose_B(rho, 3, 3)
    min_eig_TB = np.min(eigvalsh(rho_TB))

    print(f"a = {a}:")
    print(f"  Valid density matrix: {is_pos}, Tr(ρ) = {trace:.4f}")
    print(f"  PPT: {ppt}, min eigenvalue of ρ^TB: {min_eig_TB:.6f}")

    # Range analysis (heuristic)
    overlap = check_range_for_products(rho, 3, 3)
    print(f"  Range product overlap (heuristic): {overlap:.4f}")
    print()

# Compare with 2×2 (no bound entanglement possible)
print("=== 2×2 Systems: No Bound Entanglement ===\n")

# Werner state
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho_psi = np.outer(psi_minus, psi_minus.conj())
I4 = np.eye(4) / 4

print("Werner state ρ_W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4:")
print(f"{'p':<6} {'PPT':<6} {'Status'}")
print("-" * 30)

for p in [0.2, 0.33, 0.34, 0.5, 0.8]:
    rho = p * rho_psi + (1-p) * I4
    ppt = is_ppt(rho, 2, 2)
    if ppt:
        status = "Separable (PPT ↔ Sep in 2×2)"
    else:
        status = "Entangled (NPT → Distillable)"
    print(f"{p:<6.2f} {str(ppt):<6} {status}")

# Isotropic state in 3×3
print("\n=== Isotropic State in 3×3 ===\n")

def isotropic_state(F, d):
    """Create isotropic state ρ = F|Φ⁺⟩⟨Φ⁺| + (1-F)(I - |Φ⁺⟩⟨Φ⁺|)/(d²-1)"""
    phi_plus = np.zeros(d**2, dtype=complex)
    for i in range(d):
        phi_plus[i*d + i] = 1/np.sqrt(d)
    rho_phi = np.outer(phi_plus, phi_plus.conj())
    return F * rho_phi + (1-F) * (np.eye(d**2) - rho_phi) / (d**2 - 1)

d = 3
print(f"d = {d}:")
print(f"Separable threshold: F ≤ 1/(d+1) = {1/(d+1):.4f}")
print(f"PPT threshold: F ≤ 1/d = {1/d:.4f}")
print()

for F in [0.2, 0.25, 0.3, 0.33, 0.4, 0.5]:
    rho = isotropic_state(F, d)
    ppt = is_ppt(rho, d, d)
    print(f"F = {F:.2f}: PPT = {ppt}")
```

**Expected Output:**
```
=== Horodecki Bound Entangled State (3×3) ===

a = 0.1:
  Valid density matrix: True, Tr(ρ) = 1.0000
  PPT: True, min eigenvalue of ρ^TB: 0.000XXX
  Range product overlap (heuristic): 0.XXXX

=== 2×2 Systems: No Bound Entanglement ===

Werner state ρ_W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4:
p      PPT    Status
------------------------------
0.20   True   Separable (PPT ↔ Sep in 2×2)
0.33   True   Separable (PPT ↔ Sep in 2×2)
0.34   False  Entangled (NPT → Distillable)
```

---

## Summary

### Key Concepts

| Type | PPT | Distillable | Example |
|------|-----|-------------|---------|
| Separable | Yes | N/A | Product states |
| Free entangled | No | Yes | Bell states |
| Bound entangled | Yes | No | Horodecki state |

### Key Takeaways
1. **Bound entanglement** is PPT but entangled
2. **Exists only** in dimensions ≥ 3×3
3. **Cannot distill** Bell pairs from bound entangled states
4. **Detection** requires range criterion or other methods
5. **Activation** can make bound entanglement useful

---

## Daily Checklist

- [ ] I understand the difference between free and bound entanglement
- [ ] I know why PPT states cannot be distilled
- [ ] I understand the dimension requirements
- [ ] I can identify the role of bound entanglement in QI
- [ ] I know the key open problems

---

*Next: Day 538 — GHZ and W States*
