# Day 966: Variational Quantum Simulation and Week Synthesis

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3.5 hours | Variational quantum simulation methods |
| Afternoon | 2.5 hours | Week synthesis and comparison of approaches |
| Evening | 1 hour | Computational lab: VQS and final project |

## Learning Objectives

By the end of today, you will be able to:

1. Implement Variational Quantum Simulation (VQS) for time dynamics
2. Apply variational imaginary time evolution for ground state preparation
3. Compare NISQ and fault-tolerant simulation approaches
4. Synthesize the complete landscape of quantum simulation methods
5. Identify appropriate methods for different problem types
6. Discuss open problems and future directions in quantum simulation

## Core Content

### 1. Variational Quantum Simulation (VQS)

VQS extends VQE concepts to **time-dependent simulation**:

$$\boxed{|\psi(t)\rangle \approx |\psi(\vec{\theta}(t))\rangle}$$

Instead of finding fixed optimal parameters, we track how $\vec{\theta}(t)$ should evolve to approximate real time dynamics.

**The Variational Principle for Dynamics:**

Minimize the residual:

$$\mathcal{L} = \left\| \frac{d|\psi(\vec{\theta})\rangle}{dt} + iH|\psi(\vec{\theta})\rangle \right\|^2$$

This leads to the **McLachlan variational principle**:

$$\boxed{\sum_j M_{ij} \dot{\theta}_j = V_i}$$

where:
- $M_{ij} = \text{Re}\left[\langle\partial_i \psi | \partial_j \psi\rangle\right]$ (metric tensor)
- $V_i = -\text{Im}\left[\langle\partial_i \psi | H | \psi\rangle\right]$

---

### 2. Variational Imaginary Time Evolution (VarQITE)

**Imaginary time evolution** prepares ground states:

$$|\psi(\tau)\rangle = \frac{e^{-H\tau}|\psi_0\rangle}{\|e^{-H\tau}|\psi_0\rangle\|}$$

As $\tau \to \infty$, this converges to the ground state.

**VarQITE** approximates this variationally:

$$\boxed{\sum_j A_{ij} \dot{\theta}_j = C_i}$$

where:
- $A_{ij} = \text{Re}\left[\langle\partial_i \psi | \partial_j \psi\rangle\right]$
- $C_i = -\text{Re}\left[\langle\partial_i \psi | H | \psi\rangle\right]$

**Advantages:**
- Natural ground state preparation
- Avoids barren plateaus (gradient flows downhill)
- More robust than gradient descent VQE

---

### 3. Quantum Natural Gradient

The **quantum natural gradient** accounts for the geometry of the parameter space:

$$\boxed{\dot{\theta}_j = -\sum_k (F^{-1})_{jk} \frac{\partial E}{\partial \theta_k}}$$

where $F_{jk}$ is the **quantum Fisher information matrix**:

$$F_{jk} = 4 \text{Re}\left[\langle\partial_j \psi | \partial_k \psi\rangle - \langle\partial_j \psi | \psi\rangle\langle\psi | \partial_k \psi\rangle\right]$$

This is equivalent to VarQITE evolution!

---

### 4. ADAPT-VQE: Adaptive Ansatz Construction

**ADAPT-VQE** grows the ansatz adaptively:

1. Start with reference state $|\phi_0\rangle$
2. Compute gradient $\partial E/\partial \theta$ for each operator in a pool
3. Add the operator with largest gradient to the ansatz
4. Optimize all parameters
5. Repeat until convergence

**Operator Pool Examples:**
- Generalized singles and doubles (GSD)
- Qubit excitations
- Hardware-efficient operators

**Advantages:**
- Avoids unnecessary parameters
- Adapts to problem structure
- Often finds shorter circuits than fixed UCCSD

---

### 5. Week 138 Synthesis: The Simulation Landscape

| Day | Topic | Key Method | Complexity |
|-----|-------|------------|------------|
| 960 | Hamiltonian Simulation | Problem definition | BQP-complete |
| 961 | First-Order Trotter | $\prod_j e^{-iH_j t/n}$ | $O(t^2/\epsilon)$ |
| 962 | Higher-Order Trotter | Suzuki formulas | $O(t^{1+1/2k}/\epsilon^{1/2k})$ |
| 963 | QSP | Polynomial transformations | Foundation for optimal |
| 964 | Block Encoding | LCU + Qubitization | $O(\lambda t + \log(1/\epsilon))$ |
| 965 | Chemistry | VQE + UCCSD | NISQ approach |
| 966 | VQS | Variational dynamics | NISQ dynamics |

---

### 6. Choosing the Right Method

**Decision Tree:**

```
Is fault tolerance available?
├── YES → Use QSVT/Qubitization
│   └── Query complexity: O(λt + log(1/ε))
│
└── NO (NISQ) → What's the goal?
    ├── Ground state? → VQE / VarQITE / ADAPT-VQE
    ├── Dynamics? → VQS / Trotter (short time)
    └── Thermal state? → Variational methods / QMETTS
```

**NISQ Considerations:**
- Circuit depth must be << coherence limit
- Error mitigation essential
- Variational methods more noise-tolerant

**Fault-Tolerant Considerations:**
- Can run arbitrarily long circuits
- Focus on query/gate complexity
- Block encoding overhead acceptable

---

### 7. Comparison of Methods

| Method | Precision | Depth | Ancillas | Best For |
|--------|-----------|-------|----------|----------|
| Trotter-1 | $O(t^2/n)$ | $O(Ln)$ | 0 | Simple, short time |
| Trotter-2k | $O(t^{2k+1}/n^{2k})$ | $O(5^k Ln)$ | 0 | Moderate precision |
| qDRIFT | $O(\lambda^2 t^2/N)$ | $O(N)$ | 0 | Many terms, NISQ |
| QSVT | $O(\epsilon)$ | $O(\lambda t)$ | $O(\log L)$ | High precision, FT |
| VQE | Variational | $O(n_p)$ | 0 | Ground states, NISQ |
| VQS | Variational | $O(n_p)$ | 0 | Dynamics, NISQ |

---

### 8. Open Problems and Future Directions

**Theoretical Questions:**
1. Can we prove quantum advantage for specific chemistry problems?
2. What is the optimal fermion-to-qubit mapping?
3. How do we handle strong correlation efficiently?

**Algorithmic Challenges:**
1. Better classical-quantum hybrid protocols
2. Adaptive methods for circuit construction
3. Error mitigation at scale

**Application Frontiers:**
1. Nitrogen fixation catalyst (FeMoCo)
2. High-temperature superconductors
3. Quantum dynamics in biological systems
4. Lattice gauge theories (particle physics)

---

## Worked Examples

### Example 1: VQS for Single-Qubit Evolution

**Problem:** Use VQS to simulate $H = X$ evolution starting from $|0\rangle$.

**Solution:**

Step 1: Ansatz.
Use $|\psi(\theta)\rangle = R_y(\theta)|0\rangle$.

$$|\psi(\theta)\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$$

Step 2: Compute the metric tensor $M$.

$$|\partial_\theta \psi\rangle = \frac{1}{2}\left(-\sin(\theta/2)|0\rangle + \cos(\theta/2)|1\rangle\right)$$

$$M = \langle\partial_\theta \psi | \partial_\theta \psi\rangle = \frac{1}{4}$$

Step 3: Compute $V$.

$$V = -\text{Im}[\langle\partial_\theta \psi | X | \psi\rangle]$$

$$\langle\partial_\theta \psi | X | \psi\rangle = \frac{1}{2}\left(-\sin(\theta/2)\cos(\theta/2) + \cos(\theta/2)\sin(\theta/2)\right) = 0$$

Hmm, this is zero! Let's recalculate.

Actually:
$$X|\psi\rangle = \cos(\theta/2)|1\rangle + \sin(\theta/2)|0\rangle$$

$$\langle\partial_\theta \psi | X | \psi\rangle = \frac{1}{2}\left(-\sin(\theta/2)\sin(\theta/2) + \cos(\theta/2)\cos(\theta/2)\right)$$
$$= \frac{1}{2}\cos(\theta)$$

So $V = -\text{Im}[\frac{1}{2}\cos(\theta)] = 0$ since it's real.

Step 4: The equation $M\dot{\theta} = V$ gives $\dot{\theta} = 0$!

This means $R_y(\theta)|0\rangle$ cannot capture the dynamics of $e^{-iXt}|0\rangle$.

Step 5: Need richer ansatz.
Try $|\psi(\theta_1, \theta_2)\rangle = R_z(\theta_1)R_y(\theta_2)|0\rangle$.

This can capture any single-qubit state and will give non-trivial dynamics.

**Lesson:** The ansatz must be expressive enough for the dynamics.

$\square$

---

### Example 2: VarQITE for Ground State

**Problem:** Use VarQITE to find the ground state of $H = Z + 0.5X$.

**Solution:**

Step 1: Ansatz.
$$|\psi(\theta)\rangle = R_y(\theta)|0\rangle$$

Step 2: Energy.
$$E(\theta) = \langle\psi|Z|\psi\rangle + 0.5\langle\psi|X|\psi\rangle$$
$$= \cos^2(\theta/2) - \sin^2(\theta/2) + 0.5 \cdot 2\cos(\theta/2)\sin(\theta/2)$$
$$= \cos(\theta) + 0.5\sin(\theta)$$

Step 3: VarQITE equation.
$$A\dot{\theta} = C$$

where:
$$A = \frac{1}{4}$$ (same as before)
$$C = -\text{Re}[\langle\partial_\theta \psi | H | \psi\rangle]$$

$$\langle\partial_\theta \psi | H | \psi\rangle = \frac{1}{2}(-\sin(\theta) + 0.5\cos(\theta))$$

$$C = -\frac{1}{2}(-\sin(\theta) + 0.5\cos(\theta))$$

Step 4: Evolution equation.
$$\dot{\theta} = 4C = 2\sin(\theta) - \cos(\theta)$$

Step 5: Find fixed point.
Set $\dot{\theta} = 0$:
$$2\sin(\theta) = \cos(\theta)$$
$$\tan(\theta) = 0.5$$
$$\theta^* = \arctan(0.5) \approx 0.464 \text{ rad}$$

Step 6: Verify.
At $\theta^* = 0.464$:
$$E = \cos(0.464) + 0.5\sin(0.464) = 0.894 + 0.5(0.447) = 1.118$$

Compare to exact ground state energy:
$$E_0 = -\sqrt{1 + 0.25} = -\sqrt{1.25} \approx -1.118$$

Wait, the signs are wrong. Let me recalculate...

The ground state should have $E_0 = -\sqrt{1.25} \approx -1.118$.

For $H = Z + 0.5X$, the minimum of $\cos\theta + 0.5\sin\theta$ occurs at:
$$\frac{d}{d\theta}(\cos\theta + 0.5\sin\theta) = -\sin\theta + 0.5\cos\theta = 0$$
$$\tan\theta = 0.5$$
$$\theta = \arctan(0.5) + \pi \approx 3.61$$ (for minimum)

At this $\theta$:
$$E = -\cos(0.464) - 0.5\sin(0.464) \approx -1.118$$ ✓

$\square$

---

### Example 3: Resource Comparison

**Problem:** Compare Trotter-2 and QSVT for simulating a 50-qubit Heisenberg chain for time $t = 100$ with error $\epsilon = 10^{-6}$.

**Solution:**

Step 1: Hamiltonian parameters.
- $L = 3 \times 49 = 147$ terms (XX + YY + ZZ per bond)
- $\lambda = \sum_j |h_j| = 3 \times 49 = 147$ (assuming unit couplings)

Step 2: Trotter-2 resources.
Error scaling: $O(\lambda^3 t^3 / n^2)$

Required steps:
$$n \geq \left(\frac{\lambda^3 t^3}{\epsilon}\right)^{1/2} = \left(\frac{147^3 \times 100^3}{10^{-6}}\right)^{1/2}$$

Let me compute:
$147^3 \approx 3.2 \times 10^6$
$100^3 = 10^6$
Numerator: $3.2 \times 10^{12}$
$n \geq (3.2 \times 10^{18})^{1/2} \approx 1.8 \times 10^9$

Gates per step: $2L - 1 \approx 293$
Total gates: $\approx 5 \times 10^{11}$

Step 3: QSVT resources.
Query complexity:
$$Q = O(\lambda t + \log(1/\epsilon)) = O(147 \times 100 + 20) \approx 14,700$$

Each query uses the block encoding once.
SELECT cost: $O(L) \approx 147$ gates
Total gates: $\approx 14,700 \times 500 \approx 7 \times 10^6$

Step 4: Comparison.

| Method | Trotter Steps/Queries | Total Gates |
|--------|----------------------|-------------|
| Trotter-2 | $1.8 \times 10^9$ | $5 \times 10^{11}$ |
| QSVT | $1.5 \times 10^4$ | $7 \times 10^6$ |

**QSVT wins by factor of ~70,000!**

But QSVT requires:
- $O(\log L) \approx 8$ ancilla qubits
- Block encoding construction
- QSP phase computation

$\square$

---

## Practice Problems

### Level 1: Direct Application

1. **VQS setup:** For ansatz $|\psi(\theta)\rangle = R_x(\theta)|0\rangle$ and $H = Y$, compute the metric tensor $M$ and vector $V$.

2. **VarQITE:** For $H = Z$, show that VarQITE with $R_y(\theta)|0\rangle$ converges to $|0\rangle$ (ground state).

3. **Method selection:** Given a 10-qubit system, $t = 10$, $\epsilon = 0.01$, which method would you recommend for NISQ hardware?

### Level 2: Intermediate Analysis

4. **Quantum Fisher information:** Derive the QFI matrix for the 2-parameter ansatz $R_z(\theta_1)R_y(\theta_2)|0\rangle$.

5. **ADAPT-VQE:** For H2, identify which operator from the pool {$X_0, Y_0, Z_0, X_0X_1, ...$} would be selected first.

6. **Barren plateaus:** Explain why VarQITE is less susceptible to barren plateaus than standard VQE.

### Level 3: Challenging Problems

7. **Hybrid algorithms:** Design a hybrid classical-quantum algorithm that uses VQE for initialization and Trotter for dynamics.

8. **Error analysis:** Derive how shot noise in quantum measurements affects VQS trajectory accuracy.

9. **Quantum advantage:** For what problem sizes and precisions would quantum simulation outperform the best classical methods?

---

## Computational Lab: Variational Quantum Simulation

### Lab Objective

Implement VQS and VarQITE for small systems and compare with exact dynamics.

```python
"""
Day 966 Lab: Variational Quantum Simulation and Week Synthesis
Week 138: Quantum Simulation
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================
# Part 1: VQS Implementation
# =============================================================

print("=" * 60)
print("Part 1: Variational Quantum Simulation (VQS)")
print("=" * 60)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def parametrized_state(params: np.ndarray) -> np.ndarray:
    """
    Two-qubit parametrized state:
    |psi(theta)> = Ry(theta[0]) x Ry(theta[1]) * CNOT * Ry(theta[2]) x Ry(theta[3]) |00>
    """
    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # Build circuit
    layer1 = np.kron(Ry(params[0]), Ry(params[1]))
    layer2 = np.kron(Ry(params[2]), Ry(params[3]))
    U = layer2 @ CNOT @ layer1

    # Apply to |00>
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    return U @ psi0

def parameter_gradient_states(params: np.ndarray, idx: int,
                              delta: float = 1e-5) -> np.ndarray:
    """Compute d|psi>/d(theta_idx) numerically."""
    params_plus = params.copy()
    params_plus[idx] += delta
    params_minus = params.copy()
    params_minus[idx] -= delta

    psi_plus = parametrized_state(params_plus)
    psi_minus = parametrized_state(params_minus)

    return (psi_plus - psi_minus) / (2 * delta)

def vqs_matrices(params: np.ndarray, H: np.ndarray) -> tuple:
    """
    Compute M and V for VQS: M @ theta_dot = V
    """
    n_params = len(params)
    psi = parametrized_state(params)

    # Compute gradient states
    grad_psi = [parameter_gradient_states(params, i) for i in range(n_params)]

    # Metric tensor M_ij = Re[<d_i psi | d_j psi>]
    M = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            M[i, j] = np.real(np.vdot(grad_psi[i], grad_psi[j]))

    # Vector V_i = -Im[<d_i psi | H | psi>]
    V = np.zeros(n_params)
    H_psi = H @ psi
    for i in range(n_params):
        V[i] = -np.imag(np.vdot(grad_psi[i], H_psi))

    return M, V

def vqs_derivative(t: float, params: np.ndarray, H: np.ndarray) -> np.ndarray:
    """RHS for VQS ODE: theta_dot = M^(-1) @ V"""
    M, V = vqs_matrices(params, H)

    # Regularize M for numerical stability
    M += 1e-8 * np.eye(len(params))

    try:
        theta_dot = np.linalg.solve(M, V)
    except np.linalg.LinAlgError:
        theta_dot = np.linalg.lstsq(M, V, rcond=None)[0]

    return theta_dot

# Test VQS on 2-qubit Ising model
H_ising = 0.5 * np.kron(Z, Z) + 0.25 * (np.kron(X, I) + np.kron(I, X))

# Initial parameters (random)
params_init = np.array([0.1, 0.2, 0.1, 0.2])

# Exact dynamics
psi0 = parametrized_state(params_init)
def exact_dynamics(t):
    return expm(-1j * H_ising * t) @ psi0

# VQS dynamics
t_span = (0, 5)
t_eval = np.linspace(0, 5, 50)

solution = solve_ivp(
    lambda t, p: vqs_derivative(t, p, H_ising),
    t_span,
    params_init,
    t_eval=t_eval,
    method='RK45'
)

# Compare observables
obs_Z = np.kron(Z, I)  # Z on first qubit

exact_Z = []
vqs_Z = []

for i, t in enumerate(t_eval):
    # Exact
    psi_exact = exact_dynamics(t)
    exact_Z.append(np.real(np.vdot(psi_exact, obs_Z @ psi_exact)))

    # VQS
    if i < len(solution.y[0]):
        params_t = solution.y[:, i]
        psi_vqs = parametrized_state(params_t)
        vqs_Z.append(np.real(np.vdot(psi_vqs, obs_Z @ psi_vqs)))
    else:
        vqs_Z.append(vqs_Z[-1])

print(f"\nVQS simulation complete")
print(f"Initial <Z>: {exact_Z[0]:.4f}")
print(f"Final <Z> (exact): {exact_Z[-1]:.4f}")
print(f"Final <Z> (VQS): {vqs_Z[-1]:.4f}")

# =============================================================
# Part 2: VarQITE Implementation
# =============================================================

print("\n" + "=" * 60)
print("Part 2: Variational Imaginary Time Evolution (VarQITE)")
print("=" * 60)

def varqite_matrices(params: np.ndarray, H: np.ndarray) -> tuple:
    """
    Compute A and C for VarQITE: A @ theta_dot = C
    """
    n_params = len(params)
    psi = parametrized_state(params)

    # Compute gradient states
    grad_psi = [parameter_gradient_states(params, i) for i in range(n_params)]

    # A_ij = Re[<d_i psi | d_j psi>]
    A = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            A[i, j] = np.real(np.vdot(grad_psi[i], grad_psi[j]))

    # C_i = -Re[<d_i psi | H | psi>]
    C = np.zeros(n_params)
    H_psi = H @ psi
    for i in range(n_params):
        C[i] = -np.real(np.vdot(grad_psi[i], H_psi))

    return A, C

def varqite_derivative(tau: float, params: np.ndarray, H: np.ndarray) -> np.ndarray:
    """RHS for VarQITE ODE"""
    A, C = varqite_matrices(params, H)
    A += 1e-8 * np.eye(len(params))

    try:
        theta_dot = np.linalg.solve(A, C)
    except np.linalg.LinAlgError:
        theta_dot = np.linalg.lstsq(A, C, rcond=None)[0]

    return theta_dot

# VarQITE for ground state
params_init_qite = np.random.randn(4) * 0.5
tau_span = (0, 10)
tau_eval = np.linspace(0, 10, 100)

solution_qite = solve_ivp(
    lambda tau, p: varqite_derivative(tau, p, H_ising),
    tau_span,
    params_init_qite,
    t_eval=tau_eval,
    method='RK45'
)

# Track energy during VarQITE
energies_qite = []
for i in range(len(tau_eval)):
    if i < solution_qite.y.shape[1]:
        params_tau = solution_qite.y[:, i]
        psi_tau = parametrized_state(params_tau)
        E = np.real(np.vdot(psi_tau, H_ising @ psi_tau))
        energies_qite.append(E)
    else:
        energies_qite.append(energies_qite[-1])

# Exact ground state energy
E_gs = np.min(np.linalg.eigvalsh(H_ising))

print(f"\nVarQITE ground state preparation:")
print(f"  Exact ground state energy: {E_gs:.6f}")
print(f"  VarQITE final energy: {energies_qite[-1]:.6f}")
print(f"  Energy error: {abs(energies_qite[-1] - E_gs):.6f}")

# =============================================================
# Part 3: Visualization
# =============================================================

print("\n" + "=" * 60)
print("Part 3: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: VQS dynamics
ax = axes[0, 0]
ax.plot(t_eval, exact_Z, 'b-', linewidth=2, label='Exact')
ax.plot(t_eval[:len(vqs_Z)], vqs_Z, 'ro', markersize=4, label='VQS')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel(r'$\langle Z_1 \rangle$', fontsize=12)
ax.set_title('VQS: Real-Time Dynamics', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: VQS error
ax = axes[0, 1]
vqs_error = np.abs(np.array(exact_Z[:len(vqs_Z)]) - np.array(vqs_Z))
ax.semilogy(t_eval[:len(vqs_Z)], vqs_error, 'g-', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('|Exact - VQS|', fontsize=12)
ax.set_title('VQS Error Growth', fontsize=14)
ax.grid(True, alpha=0.3)

# Plot 3: VarQITE energy
ax = axes[1, 0]
ax.plot(tau_eval[:len(energies_qite)], energies_qite, 'b-', linewidth=2, label='VarQITE')
ax.axhline(y=E_gs, color='r', linestyle='--', linewidth=2, label=f'Exact $E_0$ = {E_gs:.4f}')
ax.set_xlabel('Imaginary Time $\\tau$', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('VarQITE: Ground State Preparation', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Method comparison summary
ax = axes[1, 1]
methods = ['Trotter-1', 'Trotter-2', 'Trotter-4', 'QSVT', 'VQE', 'VQS']
nisq_compatible = [True, True, True, False, True, True]
ft_required = [False, False, False, True, False, False]
optimal_scaling = [False, False, False, True, False, False]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, nisq_compatible, width, label='NISQ Compatible', color='green', alpha=0.7)
bars2 = ax.bar(x, ft_required, width, label='Requires FT', color='blue', alpha=0.7)
bars3 = ax.bar(x + width, optimal_scaling, width, label='Optimal Scaling', color='orange', alpha=0.7)

ax.set_ylabel('Yes/No', fontsize=12)
ax.set_title('Method Characteristics', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1.3])

plt.tight_layout()
plt.savefig('day_966_vqs_varqite.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 4: Week Synthesis - Algorithm Comparison
# =============================================================

print("\n" + "=" * 60)
print("Part 4: Week 138 Synthesis")
print("=" * 60)

def complexity_comparison(L: int, t: float, epsilon: float, lambda_1: float):
    """Compare complexities of different methods."""

    results = {}

    # Trotter-1: O(lambda^2 t^2 / epsilon)
    results['Trotter-1'] = lambda_1**2 * t**2 / epsilon

    # Trotter-2: O((lambda t)^{3/2} / sqrt(epsilon))
    results['Trotter-2'] = (lambda_1 * t)**1.5 / np.sqrt(epsilon)

    # Trotter-4: O((lambda t)^{5/4} / epsilon^{1/4})
    results['Trotter-4'] = (lambda_1 * t)**1.25 / epsilon**0.25

    # QSVT: O(lambda t + log(1/epsilon))
    results['QSVT'] = lambda_1 * t + np.log(1/epsilon)

    return results

# Compare for typical chemistry problem
L = 100  # Terms
t = 100  # Evolution time
epsilon = 1e-6  # Precision
lambda_1 = L  # 1-norm

comparison = complexity_comparison(L, t, epsilon, lambda_1)

print("\nComplexity comparison for L=100, t=100, ε=10^-6:")
print("-" * 40)
for method, complexity in sorted(comparison.items(), key=lambda x: x[1]):
    print(f"  {method:<12}: {complexity:>12.2e}")

# =============================================================
# Part 5: Final Summary Table
# =============================================================

print("\n" + "=" * 60)
print("Part 5: Week 138 Summary")
print("=" * 60)

summary_table = """
+---------------+------------------+-------------------+---------------+
| Day           | Topic            | Key Equation      | Main Insight  |
+---------------+------------------+-------------------+---------------+
| 960 (Mon)     | Ham. Simulation  | U ≈ e^{-iHt}      | BQP-complete  |
| 961 (Tue)     | Trotter-1        | (e^A e^B)^n       | O(t²/n) error |
| 962 (Wed)     | Higher-Order     | Suzuki recursion  | O(t^{2k+1})   |
| 963 (Thu)     | QSP              | Polynomial trans. | Foundation    |
| 964 (Fri)     | Block Encoding   | LCU + Qubitize    | Optimal query |
| 965 (Sat)     | Chemistry        | VQE + UCCSD       | NISQ approach |
| 966 (Sun)     | VQS + Synthesis  | McLachlan var.    | Dynamics NISQ |
+---------------+------------------+-------------------+---------------+
"""
print(summary_table)

print("\nKey Takeaways from Week 138:")
print("-" * 40)
print("1. Hamiltonian simulation is BQP-complete and captures quantum advantage")
print("2. Product formulas trade simplicity for suboptimal complexity")
print("3. QSP/QSVT achieve optimal complexity with sophisticated structure")
print("4. Block encoding enables polynomial transformations")
print("5. Chemistry applications drive near-term quantum computing")
print("6. Variational methods bridge NISQ and fault-tolerant eras")

print("\nLab complete!")
print("Figure saved: day_966_vqs_varqite.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| VQS (McLachlan) | $\sum_j M_{ij}\dot{\theta}_j = V_i$ |
| VarQITE | $\sum_j A_{ij}\dot{\theta}_j = C_i$ |
| Quantum Natural Gradient | $\dot{\theta} = -F^{-1}\nabla E$ |
| Fisher Information | $F_{jk} = 4\text{Re}[\langle\partial_j\psi|\partial_k\psi\rangle - ...]$ |

### Week 138 Summary

| Method | Best Use Case | Complexity | Era |
|--------|---------------|------------|-----|
| Trotter | Short time, low precision | $O(t^2/\epsilon)$ | Both |
| High-order Trotter | Moderate precision | $O(t^{1+o(1)})$ | Both |
| QSVT | High precision, long time | $O(t + \log(1/\epsilon))$ | FT |
| VQE | Ground states | Variational | NISQ |
| VQS | Dynamics | Variational | NISQ |

### The Complete Picture

1. **Feynman's insight** launched the field: use quantum to simulate quantum.

2. **Product formulas** are simple but have polynomial precision dependence.

3. **QSP/QSVT** achieve optimal scaling through polynomial transformations.

4. **Block encoding** provides the interface between matrices and unitaries.

5. **Chemistry** is the killer application driving practical development.

6. **Variational methods** enable NISQ-era simulations despite noise.

---

## Week 138 Checklist

- [ ] I understand the Hamiltonian simulation problem and its complexity
- [ ] I can implement and analyze Trotter formulas of various orders
- [ ] I understand QSP and its connection to polynomial transformations
- [ ] I can construct block encodings using LCU
- [ ] I know how to map fermions to qubits using Jordan-Wigner
- [ ] I can implement VQE for molecular ground states
- [ ] I understand VQS and VarQITE for variational dynamics
- [ ] I can choose appropriate simulation methods for different scenarios

---

## Looking Ahead: Week 139

Next week we explore **Quantum Machine Learning Foundations**, including:

- Quantum data encoding strategies
- Variational quantum classifiers
- Quantum kernel methods
- Barren plateaus and trainability
- Applications in pattern recognition

Quantum simulation and quantum machine learning share many techniques, especially variational approaches.

---

*"The purpose of computing is insight, not numbers."*
*— Richard Hamming*

---

**Week 138 Complete!**

---

*"Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical."*
*— Richard Feynman, 1982*
