# Day 103: Applications to Differential Equations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: ODEs and Stability |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

1. Solve systems of linear ODEs using eigenvalue methods
2. Understand phase portraits and stability
3. Classify equilibrium points by eigenvalue structure
4. Apply matrix exponentials to initial value problems
5. Connect to quantum time evolution (Schrödinger equation)

---

## 📖 Core Content

### 1. Linear Systems of ODEs

A first-order linear system:
$$\frac{d\mathbf{x}}{dt} = A\mathbf{x}$$

where x(t) ∈ ℝⁿ and A is an n×n constant matrix.

**Solution:** $\mathbf{x}(t) = e^{At}\mathbf{x}(0)$

### 2. Solution via Diagonalization

If $A = PDP^{-1}$:
$$e^{At} = Pe^{Dt}P^{-1}$$

For diagonal D = diag(λ₁, ..., λₙ):
$$e^{Dt} = \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})$$

**General solution:**
$$\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{v}_1 + c_2 e^{\lambda_2 t}\mathbf{v}_2 + \cdots + c_n e^{\lambda_n t}\mathbf{v}_n$$

### 3. Stability Analysis

The equilibrium point x = 0 is:

| Eigenvalue Condition | Stability | Phase Portrait |
|---------------------|-----------|----------------|
| All Re(λᵢ) < 0 | Asymptotically stable | Sink (attractor) |
| All Re(λᵢ) > 0 | Unstable | Source (repeller) |
| Mixed signs | Unstable | Saddle point |
| Re(λᵢ) = 0, distinct | Stable (not asymptotic) | Center |

### 4. 2D Classification

For 2×2 system with eigenvalues λ₁, λ₂:

| Type | Eigenvalues | tr(A) | det(A) | Portrait |
|------|-------------|-------|--------|----------|
| Stable node | λ₁, λ₂ < 0 real | < 0 | > 0 | Converging |
| Unstable node | λ₁, λ₂ > 0 real | > 0 | > 0 | Diverging |
| Saddle | λ₁ < 0 < λ₂ real | any | < 0 | Hyperbolic |
| Stable spiral | α ± iβ, α < 0 | < 0 | > 0 | Inward spiral |
| Unstable spiral | α ± iβ, α > 0 | > 0 | > 0 | Outward spiral |
| Center | ±iβ | = 0 | > 0 | Closed orbits |

### 5. Quantum Connection: Schrödinger Equation

The time-dependent Schrödinger equation:
$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

**Solution:** $|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$

For energy eigenstates: $|\psi_n(t)\rangle = e^{-iE_n t/\hbar}|n\rangle$

---

## ✏️ Worked Example

**Solve:** x' = -x + 2y, y' = 2x - y with x(0) = 1, y(0) = 0

**Matrix form:** $\mathbf{x}' = A\mathbf{x}$ where $A = \begin{pmatrix} -1 & 2 \\ 2 & -1 \end{pmatrix}$

**Eigenvalues:** λ = 1, -3
**Eigenvectors:** v₁ = (1,1), v₂ = (1,-1)

**Solution:**
$$\mathbf{x}(t) = c_1 e^{t}\begin{pmatrix}1\\1\end{pmatrix} + c_2 e^{-3t}\begin{pmatrix}1\\-1\end{pmatrix}$$

From initial conditions: c₁ = c₂ = 1/2

$$x(t) = \frac{1}{2}(e^t + e^{-3t}), \quad y(t) = \frac{1}{2}(e^t - e^{-3t})$$

**Stability:** One positive eigenvalue → unstable (saddle point).

---

## 📝 Practice Problems

1. Solve x' = 2x - y, y' = x with x(0) = 1, y(0) = 0.

2. Classify the equilibrium for A = [[-1, 2], [-2, -1]].

3. For what values of k is x' = x + y, y' = kx + y stable?

4. Solve the quantum harmonic oscillator in the ground state.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm

def plot_phase_portrait(A, title="Phase Portrait"):
    """Plot phase portrait for dx/dt = Ax"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Vector field
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    U = A[0,0]*X + A[0,1]*Y
    V = A[1,0]*X + A[1,1]*Y
    
    ax.streamplot(X, Y, U, V, density=1.5, color='blue', linewidth=0.5)
    
    # Eigenspaces
    eigenvalues, eigenvectors = np.linalg.eig(A)
    colors = ['red', 'green']
    
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(lam):
            ax.plot([-3*v[0], 3*v[0]], [-3*v[1], 3*v[1]], 
                   colors[i], linewidth=2, label=f'λ={lam.real:.2f}')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}\nλ = {eigenvalues}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig(f'phase_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

# Test different cases
cases = [
    (np.array([[-1, 0], [0, -2]]), "Stable Node"),
    (np.array([[1, 0], [0, 2]]), "Unstable Node"),
    (np.array([[1, 0], [0, -1]]), "Saddle"),
    (np.array([[-1, 2], [-2, -1]]), "Stable Spiral"),
    (np.array([[0, 1], [-1, 0]]), "Center"),
]

for A, name in cases:
    plot_phase_portrait(A, name)
```

---

## ✅ Daily Checklist

- [ ] Solve linear ODE systems via eigenvalues
- [ ] Classify equilibrium stability
- [ ] Draw phase portraits
- [ ] Connect to quantum time evolution
- [ ] Complete computational lab

---

*"Differential equations are the language of physics."*
— Richard Feynman
