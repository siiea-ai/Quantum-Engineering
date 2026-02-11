# Day 441: Week 63 Review — Fine Structure

## Overview
**Day 441** | Year 1, Month 16, Week 63 | Comprehensive Review

Today we consolidate our understanding of fine structure corrections before moving to hyperfine structure and external fields.

---

## Week 63 Summary

### Day 435: Relativistic Correction
- H'_rel = -p⁴/(8m³c²)
- First relativistic correction to kinetic energy
- Lifts l-degeneracy

### Day 436: Spin-Orbit Coupling
- H'_SO = (1/2m²c²)(1/r)(dV/dr)L·S
- Interaction of spin magnetic moment with orbital field
- Thomas precession factor of 1/2

### Day 437: Darwin Term
- H'_D = (πℏ²e²/2m²c²)δ³(r)
- Contact interaction from zitterbewegung
- Affects only s-orbitals (l = 0)

### Day 438: Total Fine Structure
- E_FS = E_n(α²/n²)[n/(j+1/2) - 3/4]
- Remarkable simplification: depends only on j
- Order α² ≈ 5×10⁻⁵ of binding energy

### Day 439: Good Quantum Numbers
- n, l, j, m_j (not m_l, m_s separately)
- Coupled basis |n,l,j,m_j⟩
- J = L + S is conserved

### Day 440: Spectroscopic Notation
- Term symbols: ²S+¹L_J
- Selection rules: Δl = ±1, Δj = 0, ±1

---

## Master Formula Sheet

### Fine Structure Formula
$$E_{nj} = E_n\left[1 + \frac{\alpha^2}{n^2}\left(\frac{n}{j+1/2} - \frac{3}{4}\right)\right]$$

### L·S Eigenvalue
$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

### Fine Structure Constant
$$\alpha = \frac{e^2}{4\pi\varepsilon_0\hbar c} \approx \frac{1}{137}$$

---

## Energy Level Diagram (n = 2)

```
Without FS:      With FS:

  2P ——————     2P_{3/2} ——————
                        ↑ FS splitting
  2S ——————     2P_{1/2} ——————
                2S_{1/2} ——————  (same as 2P_{1/2}!)
```

---

## Quantum Computing Connections

| Physics | QC Application |
|---------|----------------|
| Fine structure | Qubit level spacing |
| Good quantum numbers | State labeling |
| Selection rules | Allowed transitions |
| Term symbols | Spectroscopic addressing |

---

## Week 63 Checklist

- [ ] I can derive each fine structure correction
- [ ] I understand why they combine to depend only on j
- [ ] I know the good quantum numbers with spin-orbit
- [ ] I can use spectroscopic notation
- [ ] I understand selection rules

---

## Preview: Week 64 — Hyperfine & External Fields

Next week:
- Nuclear spin and hyperfine structure
- 21 cm line of hydrogen
- Zeeman effect (weak and strong field)
- Stark effect (electric fields)
- Atomic qubits

---

**Congratulations on completing Week 63!**

**Next:** [Week_64_Hyperfine_Zeeman](../Week_64_Hyperfine_Zeeman/README.md)
