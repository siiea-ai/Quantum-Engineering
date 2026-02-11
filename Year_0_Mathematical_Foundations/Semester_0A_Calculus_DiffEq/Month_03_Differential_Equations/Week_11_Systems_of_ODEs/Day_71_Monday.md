# Day 71: Introduction to Systems of ODEs

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Systems Fundamentals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Matrix Formulation |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Convert higher-order ODEs to first-order systems
2. Write systems in matrix-vector form
3. Understand the existence and uniqueness theorem for systems
4. Recognize linear vs. nonlinear systems
5. Solve simple 2×2 systems by elimination

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 7.1**: Introduction to Systems of First Order Linear Equations (pp. 359-370)
- **Section 7.2**: Review of Matrices (pp. 371-385)

---

## 📖 Core Content: Why Systems?

### 1. Motivation

Many physical systems involve **multiple interacting quantities**:

**Coupled Springs:**
$$m_1 x_1'' = -k_1 x_1 + k_2(x_2 - x_1)$$
$$m_2 x_2'' = -k_2(x_2 - x_1)$$

**Predator-Prey (Lotka-Volterra):**
$$\frac{dx}{dt} = ax - bxy$$
$$\frac{dy}{dt} = -cy + dxy$$

**RLC Network:**
Multiple coupled circuits share currents.

### 2. Converting Higher-Order to Systems

Any $n$th-order ODE can be written as a system of $n$ first-order ODEs.

**Example:** Convert $y'' + 3y' + 2y = 0$

Let $x_1 = y$ and $x_2 = y'$. Then:
$$x_1' = x_2$$
$$x_2' = y'' = -3y' - 2y = -2x_1 - 3x_2$$

**As a system:**
$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

---

## 📖 Matrix-Vector Form

### 3. General Linear System

A first-order linear system:
$$\mathbf{x}' = A(t)\mathbf{x} + \mathbf{f}(t)$$

where:
- $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$ is the state vector
- $A(t)$ is an $n \times n$ coefficient matrix
- $\mathbf{f}(t)$ is the forcing vector

**Homogeneous:** $\mathbf{f}(t) = \mathbf{0}$
**Constant coefficients:** $A(t) = A$ (constant matrix)

### 4. Example: Writing in Matrix Form

System:
$$x_1' = 3x_1 - 2x_2$$
$$x_2' = x_1 + x_2$$

Matrix form:
$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} 3 & -2 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

or simply: $\mathbf{x}' = A\mathbf{x}$ where $A = \begin{pmatrix} 3 & -2 \\ 1 & 1 \end{pmatrix}$

---

## 📖 Existence and Uniqueness

### 5. Theorem

> **Theorem:** If $A(t)$ and $\mathbf{f}(t)$ are continuous on an interval $I$ containing $t_0$, then the IVP:
> $$\mathbf{x}' = A(t)\mathbf{x} + \mathbf{f}(t), \quad \mathbf{x}(t_0) = \mathbf{x}_0$$
> has a **unique** solution on the entire interval $I$.

### 6. Superposition Principle

For homogeneous systems $\mathbf{x}' = A\mathbf{x}$:

> If $\mathbf{x}_1$ and $\mathbf{x}_2$ are solutions, then so is $c_1\mathbf{x}_1 + c_2\mathbf{x}_2$ for any constants $c_1, c_2$.

---

## ✏️ Worked Examples

### Example 1: Convert to System
Convert $y''' - 2y'' + y' = e^t$ to a system.

Let $x_1 = y$, $x_2 = y'$, $x_3 = y''$. Then:
$$x_1' = x_2$$
$$x_2' = x_3$$
$$x_3' = y''' = 2y'' - y' + e^t = -x_2 + 2x_3 + e^t$$

**Matrix form:**
$$\mathbf{x}' = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 2 \end{pmatrix}\mathbf{x} + \begin{pmatrix} 0 \\ 0 \\ e^t \end{pmatrix}$$

---

### Example 2: Verify a Solution
Show that $\mathbf{x}(t) = \begin{pmatrix} e^{2t} \\ e^{2t} \end{pmatrix}$ solves $\mathbf{x}' = \begin{pmatrix} 3 & -1 \\ 1 & 1 \end{pmatrix}\mathbf{x}$

**Check:**
$$\mathbf{x}' = \begin{pmatrix} 2e^{2t} \\ 2e^{2t} \end{pmatrix}$$

$$A\mathbf{x} = \begin{pmatrix} 3 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} e^{2t} \\ e^{2t} \end{pmatrix} = \begin{pmatrix} 3e^{2t} - e^{2t} \\ e^{2t} + e^{2t} \end{pmatrix} = \begin{pmatrix} 2e^{2t} \\ 2e^{2t} \end{pmatrix}$$ ✓

---

### Example 3: Solve by Elimination
Solve the system:
$$x' = x + 2y$$
$$y' = 3x + 2y$$

**Method:** Differentiate the first equation and substitute.

From equation 1: $y = (x' - x)/2$

Differentiate: $y' = (x'' - x')/2$

Substitute into equation 2:
$$\frac{x'' - x'}{2} = 3x + 2 \cdot \frac{x' - x}{2}$$
$$x'' - x' = 6x + 2x' - 2x$$
$$x'' - 3x' - 4x = 0$$

Characteristic equation: $r^2 - 3r - 4 = (r-4)(r+1) = 0$

So $r = 4, -1$ and $x = c_1 e^{4t} + c_2 e^{-t}$

Back-substitute: $y = (x' - x)/2 = \frac{3}{2}c_1 e^{4t} - c_2 e^{-t}$

**Solution:**
$$\mathbf{x}(t) = c_1 \begin{pmatrix} 1 \\ 3/2 \end{pmatrix} e^{4t} + c_2 \begin{pmatrix} 1 \\ -1 \end{pmatrix} e^{-t}$$

---

## 📋 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| State vector | $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$ |
| System matrix | $A$ in $\mathbf{x}' = A\mathbf{x}$ |
| Homogeneous | $\mathbf{x}' = A\mathbf{x}$ (no forcing) |
| Fundamental set | $n$ linearly independent solutions |
| General solution | Linear combination of fundamental set |

---

## 📝 Practice Problems

### Level 1: Conversion
1. Convert $y'' + 4y' + 3y = 0$ to a system
2. Convert $y''' + y' - y = \sin t$ to a system
3. Write in matrix form: $x' = 2x - y$, $y' = x + 3y$

### Level 2: Verification
4. Verify $\mathbf{x} = \begin{pmatrix} e^t \\ 2e^t \end{pmatrix}$ solves $\mathbf{x}' = \begin{pmatrix} -1 & 1 \\ -2 & 2 \end{pmatrix}\mathbf{x}$
5. Verify $\mathbf{x} = \begin{pmatrix} \cos t \\ -\sin t \end{pmatrix}$ solves $\mathbf{x}' = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\mathbf{x}$

### Level 3: Elimination Method
6. Solve: $x' = x - 2y$, $y' = 3x - 4y$
7. Solve: $x' = 4x - 3y$, $y' = 2x - y$
8. Solve: $x' = y$, $y' = -x$

### Level 4: Initial Value Problems
9. Solve $\mathbf{x}' = \begin{pmatrix} 1 & 2 \\ 3 & 2 \end{pmatrix}\mathbf{x}$, $\mathbf{x}(0) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$
10. Solve $x' = x + y$, $y' = -2x + 4y$, $x(0) = 1$, $y(0) = 0$

### Level 5: Applications
11. Two tanks are connected. Write the system for salt concentrations.
12. A coupled spring system has $m_1 x_1'' = -2x_1 + (x_2 - x_1)$, $m_2 x_2'' = -(x_2 - x_1)$. Convert to first-order system.

---

## 📊 Answers

1. $\mathbf{x}' = \begin{pmatrix} 0 & 1 \\ -3 & -4 \end{pmatrix}\mathbf{x}$
2. $\mathbf{x}' = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & -1 & 0 \end{pmatrix}\mathbf{x} + \begin{pmatrix} 0 \\ 0 \\ \sin t \end{pmatrix}$
3. $A = \begin{pmatrix} 2 & -1 \\ 1 & 3 \end{pmatrix}$
4. Direct verification
5. Direct verification
6. $x = c_1 e^{-2t} + c_2 e^{-t}$, $y = \frac{3}{2}c_1 e^{-2t} + c_2 e^{-t}$
7. $x = c_1 e^{2t} + 3c_2 e^t$, $y = c_1 e^{2t} + 2c_2 e^t$
8. $x = c_1 \cos t + c_2 \sin t$, $y = -c_1 \sin t + c_2 \cos t$
9. Use eigenvalue method (Day 72)
10. Use eigenvalue method (Day 72)
11. System of first-order linear ODEs
12. 4×4 system

---

## 🔬 Quantum Mechanics Connection

### Multi-Level Systems

A quantum system with $n$ energy levels evolves according to:
$$i\hbar \frac{d}{dt}\begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix} = H \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix}$$

where $H$ is the Hamiltonian matrix and $c_i$ are probability amplitudes.

### Spin-1/2 Systems

The simplest quantum system (qubit) satisfies:
$$i\hbar \frac{d}{dt}\begin{pmatrix} c_\uparrow \\ c_\downarrow \end{pmatrix} = \begin{pmatrix} E_\uparrow & V \\ V^* & E_\downarrow \end{pmatrix}\begin{pmatrix} c_\uparrow \\ c_\downarrow \end{pmatrix}$$

This is a 2×2 system of ODEs!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Sections 7.1-7.2
- [ ] Practice converting higher-order to systems
- [ ] Write systems in matrix form
- [ ] Solve simple systems by elimination
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 72: The Eigenvalue Method**
- Finding eigenvalues and eigenvectors of the system matrix
- Building the general solution from eigenvectors
- The elegant connection between linear algebra and ODEs

---

*"Systems of equations reveal interactions—the mathematics of connection and coupling."*
