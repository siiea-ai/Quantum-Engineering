# Day 73: Complex and Repeated Eigenvalues

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Complex Eigenvalues |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Repeated Eigenvalues |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Handle systems with complex conjugate eigenvalues
2. Extract real solutions from complex eigenvectors
3. Recognize oscillatory behavior in phase portraits
4. Solve systems with repeated eigenvalues
5. Find generalized eigenvectors when needed

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 7.6**: Complex Eigenvalues (pp. 426-438)
- **Section 7.8**: Repeated Eigenvalues (pp. 453-463)

---

## 📖 Part I: Complex Eigenvalues

### 1. When Complex Roots Arise

For real matrix $A$, complex eigenvalues always come in **conjugate pairs**:
$$\lambda = \alpha + i\beta, \quad \bar{\lambda} = \alpha - i\beta$$

### 2. The Complex Solution

If $\lambda = \alpha + i\beta$ with eigenvector $\mathbf{v} = \mathbf{a} + i\mathbf{b}$, the complex solution is:
$$\mathbf{x}(t) = e^{(\alpha + i\beta)t}(\mathbf{a} + i\mathbf{b})$$

### 3. Extracting Real Solutions

Using Euler's formula: $e^{i\beta t} = \cos\beta t + i\sin\beta t$

$$\mathbf{x}(t) = e^{\alpha t}[(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + i(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)]$$

The **two real solutions** are:
$$\mathbf{x}_1(t) = e^{\alpha t}(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t)$$
$$\mathbf{x}_2(t) = e^{\alpha t}(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)$$

### 4. General Solution (Complex Case)

$$\mathbf{x}(t) = c_1 e^{\alpha t}(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + c_2 e^{\alpha t}(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)$$

---

## ✏️ Example: Complex Eigenvalues

### Example 1
Solve $\mathbf{x}' = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\mathbf{x}$

**Step 1: Eigenvalues**
$$\det(A - \lambda I) = \lambda^2 + 1 = 0$$
$$\lambda = \pm i$$

**Step 2: Eigenvector for $\lambda = i$**
$$(A - iI)\mathbf{v} = \begin{pmatrix} -i & 1 \\ -1 & -i \end{pmatrix}\mathbf{v} = 0$$

From row 1: $-iv_1 + v_2 = 0 \Rightarrow v_2 = iv_1$

$$\mathbf{v} = \begin{pmatrix} 1 \\ i \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} + i\begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

So $\mathbf{a} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$, $\alpha = 0$, $\beta = 1$

**Step 3: Real solutions**
$$\mathbf{x}_1 = \begin{pmatrix} \cos t \\ -\sin t \end{pmatrix}, \quad \mathbf{x}_2 = \begin{pmatrix} \sin t \\ \cos t \end{pmatrix}$$

**General solution:**
$$\mathbf{x}(t) = c_1\begin{pmatrix} \cos t \\ -\sin t \end{pmatrix} + c_2\begin{pmatrix} \sin t \\ \cos t \end{pmatrix}$$

This is **circular motion**! (Center at origin)

---

### Example 2: Damped Oscillation
Solve $\mathbf{x}' = \begin{pmatrix} -1 & 2 \\ -1 & -1 \end{pmatrix}\mathbf{x}$

**Eigenvalues:**
$$\lambda^2 + 2\lambda + 2 = 0 \Rightarrow \lambda = -1 \pm i$$

So $\alpha = -1$, $\beta = 1$.

**Eigenvector for $\lambda = -1 + i$:**
After calculation: $\mathbf{v} = \begin{pmatrix} 2 \\ i \end{pmatrix}$

So $\mathbf{a} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

**General solution:**
$$\mathbf{x}(t) = e^{-t}\left[c_1\begin{pmatrix} 2\cos t \\ -\sin t \end{pmatrix} + c_2\begin{pmatrix} 2\sin t \\ \cos t \end{pmatrix}\right]$$

This is a **spiral into the origin** (damped oscillation).

---

## 📖 Part II: Repeated Eigenvalues

### 5. The Problem

If $\lambda$ has multiplicity $k$, we need $k$ linearly independent solutions.

**Two subcases:**
- Enough eigenvectors (complete): Use them directly
- Not enough eigenvectors (deficient): Need **generalized eigenvectors**

### 6. Generalized Eigenvectors

If $\lambda$ is repeated but has only one eigenvector $\mathbf{v}$, find $\mathbf{w}$ such that:
$$(A - \lambda I)\mathbf{w} = \mathbf{v}$$

The second solution is:
$$\mathbf{x}_2(t) = (t\mathbf{v} + \mathbf{w})e^{\lambda t}$$

---

### Example 3: Repeated Eigenvalue (Complete)
Solve $\mathbf{x}' = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}\mathbf{x}$

$\lambda = 2$ (multiplicity 2), but two independent eigenvectors exist:
$$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**General solution:**
$$\mathbf{x}(t) = c_1\begin{pmatrix} 1 \\ 0 \end{pmatrix}e^{2t} + c_2\begin{pmatrix} 0 \\ 1 \end{pmatrix}e^{2t}$$

---

### Example 4: Repeated Eigenvalue (Deficient)
Solve $\mathbf{x}' = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}\mathbf{x}$

**Eigenvalues:** $\lambda = 3$ (multiplicity 2)

**Eigenvectors:**
$$(A - 3I)\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = 0$$

Only one eigenvector: $\mathbf{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

**Generalized eigenvector:**
$$(A - 3I)\mathbf{w} = \mathbf{v}$$
$$\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{w} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

From row 1: $w_2 = 1$. Choose $w_1 = 0$: $\mathbf{w} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

**General solution:**
$$\mathbf{x}(t) = c_1\begin{pmatrix} 1 \\ 0 \end{pmatrix}e^{3t} + c_2\left[t\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix}\right]e^{3t}$$
$$= \begin{pmatrix} c_1 + c_2 t \\ c_2 \end{pmatrix}e^{3t}$$

---

## 📋 Summary: Three Cases

| Eigenvalue Type | Solution Form |
|-----------------|---------------|
| Real distinct | $\sum c_i \mathbf{v}_i e^{\lambda_i t}$ |
| Complex $\alpha \pm i\beta$ | $e^{\alpha t}[c_1(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + c_2(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)]$ |
| Repeated (deficient) | $c_1\mathbf{v}e^{\lambda t} + c_2(t\mathbf{v} + \mathbf{w})e^{\lambda t}$ |

---

## 📝 Practice Problems

### Level 1-3: Complex Eigenvalues
1. Solve $\mathbf{x}' = \begin{pmatrix} 0 & 2 \\ -2 & 0 \end{pmatrix}\mathbf{x}$
2. Solve $\mathbf{x}' = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}\mathbf{x}$
3. Solve $\mathbf{x}' = \begin{pmatrix} -3 & 2 \\ -1 & -1 \end{pmatrix}\mathbf{x}$

### Level 4-6: Repeated Eigenvalues
4. Solve $\mathbf{x}' = \begin{pmatrix} 5 & -3 \\ 3 & -1 \end{pmatrix}\mathbf{x}$
5. Solve $\mathbf{x}' = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\mathbf{x}$
6. Solve $\mathbf{x}' = \begin{pmatrix} -1 & 1 \\ -1 & -1 \end{pmatrix}\mathbf{x}$

### Level 7-8: With ICs
7. Solve $\mathbf{x}' = \begin{pmatrix} 3 & -2 \\ 4 & -1 \end{pmatrix}\mathbf{x}$, $\mathbf{x}(0) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$
8. Classify the origin for each system above (stable/unstable, node/spiral/center)

---

## 📊 Answers

1. $\mathbf{x} = c_1(\cos 2t, -\sin 2t)^T + c_2(\sin 2t, \cos 2t)^T$ (center)
2. $\mathbf{x} = e^t[c_1(\cos t, \sin t)^T + c_2(\sin t, -\cos t)^T]$ (unstable spiral)
3. $\mathbf{x} = e^{-2t}[c_1(2\cos t, \sin t)^T + c_2(2\sin t, -\cos t)^T]$ (stable spiral)
4. $\lambda = 2$ (repeated); use generalized eigenvector
5. $\mathbf{x} = c_1(1,0)^T e^t + c_2[(t,1)^T]e^t$ (deficient node)
6. Complex: $\lambda = -1 \pm i$ (stable spiral)
7. Computed from ICs
8. Classifications based on eigenvalues

---

## 🔬 Quantum Mechanics Connection

### Rabi Oscillations

A two-level quantum system driven by an oscillating field:
$$H = \begin{pmatrix} 0 & \Omega e^{-i\omega t} \\ \Omega e^{i\omega t} & \Delta \end{pmatrix}$$

The eigenvalues determine the **Rabi frequency** of oscillation between states!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Sections 7.6, 7.8
- [ ] Extract real solutions from complex eigenvalues
- [ ] Find generalized eigenvectors
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 74: Phase Portraits and Stability**
- Classify equilibrium points
- Draw phase portraits
- Stability analysis

---

*"Complex eigenvalues bring oscillation; repeated eigenvalues bring algebraic growth—nature's two paths beyond simple exponentials."*
