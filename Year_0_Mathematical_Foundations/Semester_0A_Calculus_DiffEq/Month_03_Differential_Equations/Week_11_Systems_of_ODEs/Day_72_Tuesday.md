# Day 72: The Eigenvalue Method

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Eigenvalue Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Solution Construction |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Find eigenvalues of the coefficient matrix
2. Find eigenvectors corresponding to each eigenvalue
3. Construct the general solution using eigenvectors
4. Handle the case of distinct real eigenvalues
5. Apply initial conditions to find particular solutions

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 7.3**: Systems of Linear Algebraic Equations (pp. 386-398)
- **Section 7.5**: Homogeneous Linear Systems with Constant Coefficients (pp. 410-425)

---

## 📖 Core Content: The Eigenvalue Method

### 1. The Key Insight

For $\mathbf{x}' = A\mathbf{x}$, try a solution of the form:
$$\mathbf{x}(t) = \mathbf{v}e^{\lambda t}$$

where $\mathbf{v}$ is a constant vector and $\lambda$ is a constant.

Substituting:
$$\lambda \mathbf{v} e^{\lambda t} = A \mathbf{v} e^{\lambda t}$$

Since $e^{\lambda t} \neq 0$:
$$A\mathbf{v} = \lambda \mathbf{v}$$

This is the **eigenvalue equation**!

### 2. Eigenvalues and Eigenvectors

> **Definition:** $\lambda$ is an **eigenvalue** of $A$ if there exists a nonzero vector $\mathbf{v}$ such that $A\mathbf{v} = \lambda\mathbf{v}$. The vector $\mathbf{v}$ is the corresponding **eigenvector**.

### 3. Finding Eigenvalues

$(A - \lambda I)\mathbf{v} = \mathbf{0}$ has nonzero solutions when:
$$\det(A - \lambda I) = 0$$

This is the **characteristic equation**.

For a 2×2 matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:
$$\det(A - \lambda I) = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad-bc)$$
$$= \lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$$

### 4. General Solution (Distinct Real Eigenvalues)

If $A$ has $n$ distinct real eigenvalues $\lambda_1, \ldots, \lambda_n$ with corresponding eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$, the general solution is:

$$\mathbf{x}(t) = c_1 \mathbf{v}_1 e^{\lambda_1 t} + c_2 \mathbf{v}_2 e^{\lambda_2 t} + \cdots + c_n \mathbf{v}_n e^{\lambda_n t}$$

---

## ✏️ Worked Examples

### Example 1: 2×2 System with Distinct Real Eigenvalues
Solve $\mathbf{x}' = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix}\mathbf{x}$

**Step 1: Find eigenvalues**
$$\det(A - \lambda I) = \det\begin{pmatrix} 4-\lambda & -2 \\ 1 & 1-\lambda \end{pmatrix}$$
$$= (4-\lambda)(1-\lambda) + 2 = \lambda^2 - 5\lambda + 6 = (\lambda-2)(\lambda-3) = 0$$

Eigenvalues: $\lambda_1 = 2$, $\lambda_2 = 3$

**Step 2: Find eigenvectors**

For $\lambda_1 = 2$:
$$(A - 2I)\mathbf{v} = \begin{pmatrix} 2 & -2 \\ 1 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$$

Row reduce: both rows give $v_1 = v_2$

$$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

For $\lambda_2 = 3$:
$$(A - 3I)\mathbf{v} = \begin{pmatrix} 1 & -2 \\ 1 & -2 \end{pmatrix}\mathbf{v} = \mathbf{0}$$

Row gives $v_1 = 2v_2$

$$\mathbf{v}_2 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

**Step 3: General solution**
$$\mathbf{x}(t) = c_1 \begin{pmatrix} 1 \\ 1 \end{pmatrix} e^{2t} + c_2 \begin{pmatrix} 2 \\ 1 \end{pmatrix} e^{3t}$$

---

### Example 2: With Initial Conditions
Solve the system from Example 1 with $\mathbf{x}(0) = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$

**Apply IC:**
$$\mathbf{x}(0) = c_1 \begin{pmatrix} 1 \\ 1 \end{pmatrix} + c_2 \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$$

System:
$$c_1 + 2c_2 = 3$$
$$c_1 + c_2 = 2$$

Solving: $c_2 = 1$, $c_1 = 1$

**Particular solution:**
$$\mathbf{x}(t) = \begin{pmatrix} 1 \\ 1 \end{pmatrix} e^{2t} + \begin{pmatrix} 2 \\ 1 \end{pmatrix} e^{3t} = \begin{pmatrix} e^{2t} + 2e^{3t} \\ e^{2t} + e^{3t} \end{pmatrix}$$

---

### Example 3: 3×3 System
Solve $\mathbf{x}' = \begin{pmatrix} 2 & 0 & 0 \\ 1 & 2 & -1 \\ 1 & 0 & 1 \end{pmatrix}\mathbf{x}$

**Characteristic equation:**
$$\det(A - \lambda I) = (2-\lambda)[(2-\lambda)(1-\lambda) + 0] = (2-\lambda)^2(1-\lambda) = 0$$

Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = 2$ (multiplicity 2)

**Eigenvectors:**

For $\lambda_1 = 1$: Solve $(A - I)\mathbf{v} = 0$
$$\mathbf{v}_1 = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}$$

For $\lambda_2 = 2$: Solve $(A - 2I)\mathbf{v} = 0$
$$\mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \mathbf{v}_3 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$$

**General solution:**
$$\mathbf{x}(t) = c_1 \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} e^{t} + c_2 \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} e^{2t} + c_3 \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} e^{2t}$$

---

## 📋 Summary: Eigenvalue Method

| Step | Action |
|------|--------|
| 1 | Write the system as $\mathbf{x}' = A\mathbf{x}$ |
| 2 | Find eigenvalues: $\det(A - \lambda I) = 0$ |
| 3 | For each $\lambda_i$, find eigenvector: $(A - \lambda_i I)\mathbf{v}_i = 0$ |
| 4 | General solution: $\mathbf{x} = \sum c_i \mathbf{v}_i e^{\lambda_i t}$ |
| 5 | Apply initial conditions to find $c_i$ |

---

## 📝 Practice Problems

### Level 1: Finding Eigenvalues/Eigenvectors
1. Find eigenvalues and eigenvectors: $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$
2. Find eigenvalues and eigenvectors: $A = \begin{pmatrix} 5 & -1 \\ 3 & 1 \end{pmatrix}$
3. Find eigenvalues and eigenvectors: $A = \begin{pmatrix} 1 & 4 \\ 2 & 3 \end{pmatrix}$

### Level 2: Solving Systems
4. Solve $\mathbf{x}' = \begin{pmatrix} 3 & -2 \\ 2 & -2 \end{pmatrix}\mathbf{x}$
5. Solve $\mathbf{x}' = \begin{pmatrix} 1 & 1 \\ 4 & 1 \end{pmatrix}\mathbf{x}$
6. Solve $\mathbf{x}' = \begin{pmatrix} 5 & 3 \\ -6 & -4 \end{pmatrix}\mathbf{x}$

### Level 3: With Initial Conditions
7. Solve $\mathbf{x}' = \begin{pmatrix} 1 & 2 \\ 3 & 2 \end{pmatrix}\mathbf{x}$, $\mathbf{x}(0) = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$
8. Solve $\mathbf{x}' = \begin{pmatrix} -1 & 1 \\ 0 & -2 \end{pmatrix}\mathbf{x}$, $\mathbf{x}(0) = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$

### Level 4: 3×3 Systems
9. Solve $\mathbf{x}' = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 3 & -2 \\ 0 & 1 & 0 \end{pmatrix}\mathbf{x}$
10. Find the general solution if eigenvalues are $\lambda = 1, 2, 3$ with eigenvectors $\mathbf{v}_1 = (1,0,1)^T$, $\mathbf{v}_2 = (1,1,0)^T$, $\mathbf{v}_3 = (0,1,1)^T$

### Level 5: Analysis
11. For what values of $a$ does $A = \begin{pmatrix} 2 & a \\ 0 & 3 \end{pmatrix}$ have distinct eigenvalues?
12. If $\lambda$ is an eigenvalue of $A$, show $\lambda^2$ is an eigenvalue of $A^2$

---

## 📊 Answers

1. $\lambda_1 = 3$, $\mathbf{v}_1 = (1, 0)^T$; $\lambda_2 = 2$, $\mathbf{v}_2 = (1, -1)^T$
2. $\lambda = 4, 2$; $\mathbf{v}_1 = (1, 1)^T$, $\mathbf{v}_2 = (1, 3)^T$
3. $\lambda = 5, -1$; $\mathbf{v}_1 = (1, 1)^T$, $\mathbf{v}_2 = (2, -1)^T$
4. $\mathbf{x} = c_1 (2, 1)^T e^{-t} + c_2 (1, 2)^T e^{2t}$
5. $\mathbf{x} = c_1 (1, 2)^T e^{3t} + c_2 (1, -2)^T e^{-t}$
6. $\mathbf{x} = c_1 (1, -1)^T e^{2t} + c_2 (1, -2)^T e^{-t}$
7. $\mathbf{x} = 2(1, 1)^T e^{4t} + 4(2, -3)^T e^{-t}$
8. $\mathbf{x} = (2e^{-t} + e^{-2t}, e^{-2t})^T$
9. Three distinct eigenvalues
10. $\mathbf{x} = c_1(1,0,1)^T e^t + c_2(1,1,0)^T e^{2t} + c_3(0,1,1)^T e^{3t}$
11. All values of $a$ (eigenvalues are always 2 and 3)
12. If $A\mathbf{v} = \lambda\mathbf{v}$, then $A^2\mathbf{v} = A(A\mathbf{v}) = A(\lambda\mathbf{v}) = \lambda(A\mathbf{v}) = \lambda^2\mathbf{v}$

---

## 🔬 Quantum Mechanics Connection

### Diagonalizing the Hamiltonian

Finding eigenvalues of the Hamiltonian matrix gives **energy levels**:
$$H|\psi_n\rangle = E_n|\psi_n\rangle$$

The eigenvectors are the **stationary states**!

### Two-Level System (Qubit)

For $H = \begin{pmatrix} E_1 & V \\ V & E_2 \end{pmatrix}$:

Eigenvalues give the **dressed energies** when two levels couple.

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Sections 7.3, 7.5
- [ ] Review eigenvalue/eigenvector computation
- [ ] Solve 2×2 systems completely
- [ ] Practice with initial conditions
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 73: Complex and Repeated Eigenvalues**
- Complex eigenvalues give oscillations
- Repeated eigenvalues need generalized eigenvectors
- Connection to second-order ODEs

---

*"Eigenvalues unlock the natural modes of a system—the frequencies at which it wants to vibrate."*
