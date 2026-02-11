# Day 632: Discrete-Time Quantum Walks

## Overview
**Day 632** | Week 91, Day 2 | Year 1, Month 23 | Quantum Walks

Today we introduce discrete-time quantum walks, which use a "coin" degree of freedom to achieve quantum interference effects that lead to faster spreading than classical random walks.

---

## Learning Objectives

1. Define the discrete-time quantum walk model
2. Understand the coin and position spaces
3. Construct the walk operator
4. Analyze the walk on a line
5. Compare to classical random walk
6. Observe the quadratic speedup in spreading

---

## Core Content

### Quantum Walk on a Line

**Classical random walk:** At each step, move left or right with probability 1/2.

**Quantum walk:** We need to make this unitary (reversible)!

**Solution:** Add a "coin" qubit that determines direction.

### State Space

The state lives in:
$$\mathcal{H} = \mathcal{H}_{coin} \otimes \mathcal{H}_{position}$$

**Coin space:** $\mathcal{H}_{coin} = \text{span}\{|L\rangle, |R\rangle\}$ (2-dimensional)

**Position space:** $\mathcal{H}_{position} = \text{span}\{|x\rangle : x \in \mathbb{Z}\}$

**General state:**
$$|\psi\rangle = \sum_x (\alpha_x|L\rangle + \beta_x|R\rangle)|x\rangle$$

### Walk Operator

One step of the quantum walk:

$$\boxed{U = S \cdot (C \otimes I)}$$

**Coin operator $C$:** Acts on coin space (typically Hadamard)
$$C = H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Shift operator $S$:** Moves position based on coin state
$$S = |L\rangle\langle L| \otimes \sum_x |x-1\rangle\langle x| + |R\rangle\langle R| \otimes \sum_x |x+1\rangle\langle x|$$

### Walk Dynamics

**Initial state:** $|\psi_0\rangle = |R\rangle|0\rangle$ (or symmetric superposition)

**After t steps:** $|\psi_t\rangle = U^t|\psi_0\rangle$

### Classical vs Quantum Spreading

**Classical:** Position distribution is Gaussian
$$P(x, t) \approx \frac{1}{\sqrt{2\pi t}}e^{-x^2/(2t)}$$

Standard deviation: $\sigma \propto \sqrt{t}$

**Quantum:** Distribution is very different!
- Two peaks moving outward
- Standard deviation: $\sigma \propto t$

**Quadratic speedup** in spreading!

### The Asymmetry Issue

Starting from $|R\rangle|0\rangle$:
- Distribution is asymmetric
- Biased toward positive positions

Starting from $\frac{1}{\sqrt{2}}(|L\rangle + i|R\rangle)|0\rangle$:
- Symmetric distribution
- The phase matters!

### Mathematical Analysis

The walk operator can be analyzed via Fourier transform.

For momentum $k$: $|k\rangle = \sum_x e^{ikx}|x\rangle$

The effective Hamiltonian:
$$H_{eff}(k) = \omega(k)\vec{n}(k) \cdot \vec{\sigma}$$

where $\cos\omega(k) = \cos k / \sqrt{2}$.

**Group velocity:** $v_g = d\omega/dk \leq 1/\sqrt{2}$

The peaks travel at velocity $1/\sqrt{2}$.

---

## Worked Examples

### Example 1: First Step
Starting from $|R\rangle|0\rangle$, compute the state after one step.

**Solution:**
Apply coin: $C|R\rangle = \frac{1}{\sqrt{2}}(|L\rangle - |R\rangle)$

State: $\frac{1}{\sqrt{2}}(|L\rangle - |R\rangle)|0\rangle$

Apply shift: $S$ moves $|L\rangle$ left and $|R\rangle$ right

$|\psi_1\rangle = \frac{1}{\sqrt{2}}(|L\rangle|-1\rangle - |R\rangle|+1\rangle)$

### Example 2: Second Step
Continue from Example 1.

**Solution:**
Apply coin to each term:
- $C|L\rangle = \frac{1}{\sqrt{2}}(|L\rangle + |R\rangle)$
- $C|R\rangle = \frac{1}{\sqrt{2}}(|L\rangle - |R\rangle)$

After coin:
$$\frac{1}{2}[(|L\rangle + |R\rangle)|-1\rangle - (|L\rangle - |R\rangle)|+1\rangle]$$
$$= \frac{1}{2}[|L\rangle|-1\rangle + |R\rangle|-1\rangle - |L\rangle|+1\rangle + |R\rangle|+1\rangle]$$

After shift:
$$|\psi_2\rangle = \frac{1}{2}[|L\rangle|-2\rangle + |R\rangle|0\rangle - |L\rangle|0\rangle + |R\rangle|+2\rangle]$$

Position probabilities:
- $P(-2) = 1/4$
- $P(0) = 1/4 + 1/4 = 1/2$ (interference!)
- $P(+2) = 1/4$

Wait, let me recalculate. At position 0: amplitude from $|R\rangle|0\rangle$ is $1/2$, from $-|L\rangle|0\rangle$ is $-1/2$.

$P(0) = |1/2 - 1/2|^2 = 0$ (destructive interference!)

### Example 3: Variance Growth
Compare variance for classical and quantum walks.

**Classical:** $\langle x^2 \rangle = t$, so $\sigma = \sqrt{t}$

**Quantum:** $\langle x^2 \rangle \approx t^2/2$, so $\sigma \approx t/\sqrt{2}$

After 100 steps:
- Classical: $\sigma \approx 10$
- Quantum: $\sigma \approx 70$

---

## Practice Problems

### Problem 1: Walk Operator
Write out the 6×6 matrix for U on a 3-position system with periodic boundaries.

### Problem 2: Symmetric Initial State
Starting from $\frac{1}{\sqrt{2}}(|L\rangle + |R\rangle)|0\rangle$, compute the state after 1 step.

### Problem 3: Momentum Analysis
For momentum $k = 0$ and $k = \pi$, find the eigenstates of the walk operator.

---

## Computational Lab

```python
"""Day 632: Discrete-Time Quantum Walks"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def hadamard_coin():
    """Hadamard coin operator."""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def shift_operator(n_positions):
    """
    Shift operator for n positions.
    Assumes coin states are |0⟩ = |L⟩, |1⟩ = |R⟩.
    """
    dim = 2 * n_positions
    S = np.zeros((dim, dim))

    for x in range(n_positions):
        # |L⟩|x⟩ → |L⟩|x-1⟩
        x_left = (x - 1) % n_positions
        S[2*x_left, 2*x] = 1  # |L⟩ at x_left from |L⟩ at x

        # |R⟩|x⟩ → |R⟩|x+1⟩
        x_right = (x + 1) % n_positions
        S[2*x_right + 1, 2*x + 1] = 1  # |R⟩ at x_right from |R⟩ at x

    return S

def walk_operator(n_positions, coin=None):
    """Complete walk operator U = S @ (C ⊗ I)."""
    if coin is None:
        coin = hadamard_coin()

    # Coin operator on full space: C ⊗ I_position
    C_full = np.kron(coin, np.eye(n_positions))

    # Shift operator
    S = shift_operator(n_positions)

    # Walk operator
    U = S @ C_full
    return U

def quantum_walk_simulation(n_positions, n_steps, initial_coin, initial_pos):
    """
    Simulate discrete-time quantum walk.

    Returns probability distribution at each step.
    """
    U = walk_operator(n_positions)

    # Initial state: |coin⟩|pos⟩
    state = np.zeros(2 * n_positions, dtype=complex)
    idx = 2 * initial_pos
    state[idx] = initial_coin[0]      # |L⟩ component
    state[idx + 1] = initial_coin[1]  # |R⟩ component

    distributions = [get_position_distribution(state, n_positions)]

    for _ in range(n_steps):
        state = U @ state
        distributions.append(get_position_distribution(state, n_positions))

    return np.array(distributions)

def get_position_distribution(state, n_positions):
    """Extract position probability distribution from state."""
    probs = np.zeros(n_positions)
    for x in range(n_positions):
        # Sum over coin states
        probs[x] = abs(state[2*x])**2 + abs(state[2*x + 1])**2
    return probs

def classical_walk_simulation(n_positions, n_steps, initial_pos, n_trials=10000):
    """Simulate classical random walk."""
    positions = np.zeros((n_steps + 1, n_positions))

    for _ in range(n_trials):
        pos = initial_pos
        positions[0, pos] += 1

        for t in range(n_steps):
            # Random step
            if np.random.random() < 0.5:
                pos = (pos - 1) % n_positions
            else:
                pos = (pos + 1) % n_positions
            positions[t + 1, pos] += 1

    return positions / n_trials

def compare_classical_quantum():
    """Compare classical and quantum walk distributions."""
    n_pos = 101  # Odd for symmetry
    n_steps = 50
    center = n_pos // 2

    # Quantum walk with symmetric initial coin
    initial_coin = np.array([1, 1j]) / np.sqrt(2)  # Symmetric
    quantum_dist = quantum_walk_simulation(n_pos, n_steps, initial_coin, center)

    # Classical walk
    classical_dist = classical_walk_simulation(n_pos, n_steps, center)

    # Shift positions for plotting
    x = np.arange(n_pos) - center

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Final distributions
    ax1 = axes[0]
    ax1.bar(x, quantum_dist[-1], width=1, alpha=0.7, label='Quantum')
    ax1.plot(x, classical_dist[-1], 'r-', linewidth=2, label='Classical')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Distribution after {n_steps} steps')
    ax1.legend()
    ax1.set_xlim(-n_steps, n_steps)

    # Variance comparison
    ax2 = axes[1]
    quantum_var = []
    classical_var = []

    for t in range(n_steps + 1):
        q_var = np.sum(x**2 * quantum_dist[t])
        c_var = np.sum(x**2 * classical_dist[t])
        quantum_var.append(q_var)
        classical_var.append(c_var)

    t_axis = np.arange(n_steps + 1)
    ax2.plot(t_axis, quantum_var, 'b-', label='Quantum', linewidth=2)
    ax2.plot(t_axis, classical_var, 'r-', label='Classical', linewidth=2)
    ax2.plot(t_axis, t_axis, 'r--', alpha=0.5, label='O(t)')
    ax2.plot(t_axis, t_axis**2 / 2, 'b--', alpha=0.5, label='O(t²/2)')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Variance ⟨x²⟩')
    ax2.set_title('Variance Growth')
    ax2.legend()

    # Spacetime diagram
    ax3 = axes[2]
    im = ax3.imshow(quantum_dist.T, aspect='auto', origin='lower',
                     extent=[0, n_steps, -center, center], cmap='viridis')
    ax3.set_xlabel('Time steps')
    ax3.set_ylabel('Position')
    ax3.set_title('Quantum Walk Spacetime Diagram')
    plt.colorbar(im, ax=ax3, label='Probability')

    plt.tight_layout()
    plt.savefig('quantum_walk_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return quantum_dist, classical_dist

def analyze_coin_dependence():
    """Analyze how initial coin state affects walk."""
    n_pos = 101
    n_steps = 50
    center = n_pos // 2

    coins = [
        (np.array([1, 0]), '|L⟩'),
        (np.array([0, 1]), '|R⟩'),
        (np.array([1, 1]) / np.sqrt(2), '|+⟩'),
        (np.array([1, 1j]) / np.sqrt(2), 'Symmetric'),
    ]

    x = np.arange(n_pos) - center

    plt.figure(figsize=(12, 4))

    for coin, label in coins:
        dist = quantum_walk_simulation(n_pos, n_steps, coin, center)
        plt.plot(x, dist[-1], label=label, linewidth=1.5)

    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Quantum Walk After {n_steps} Steps (Different Initial Coins)', fontsize=12)
    plt.legend(fontsize=10)
    plt.xlim(-n_steps, n_steps)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coin_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("="*60)
print("Discrete-Time Quantum Walks")
print("="*60)

# Compare walks
print("\n1. CLASSICAL VS QUANTUM COMPARISON")
print("-"*50)
q_dist, c_dist = compare_classical_quantum()

# Statistics
n_steps = 50
x = np.arange(101) - 50
q_std = np.sqrt(np.sum(x**2 * q_dist[-1]))
c_std = np.sqrt(np.sum(x**2 * c_dist[-1]))

print(f"After {n_steps} steps:")
print(f"  Classical std dev: {c_std:.2f} ≈ √{n_steps} = {np.sqrt(n_steps):.2f}")
print(f"  Quantum std dev: {q_std:.2f} ≈ t/√2 = {n_steps/np.sqrt(2):.2f}")
print(f"  Quantum/Classical ratio: {q_std/c_std:.2f}")

# Coin dependence
print("\n2. COIN DEPENDENCE")
print("-"*50)
analyze_coin_dependence()

# Verify unitarity
print("\n3. OPERATOR VERIFICATION")
print("-"*50)
n = 21
U = walk_operator(n)
is_unitary = np.allclose(U @ U.conj().T, np.eye(U.shape[0]))
print(f"Walk operator is unitary: {is_unitary}")

# Eigenvalue analysis
eigvals = np.linalg.eigvals(U)
print(f"All eigenvalues have |λ| = 1: {np.allclose(np.abs(eigvals), 1)}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Walk operator | $U = S \cdot (C \otimes I)$ |
| Hadamard coin | $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ |
| Classical spreading | $\sigma \propto \sqrt{t}$ |
| Quantum spreading | $\sigma \propto t$ |
| Peak velocity | $v = 1/\sqrt{2}$ |

### Key Takeaways

1. **Discrete-time walk** uses coin and shift operators
2. **Coin creates superposition** of directions
3. **Quantum walk spreads faster** (linear vs sqrt)
4. **Initial coin state** affects symmetry
5. **Interference effects** create distinctive pattern
6. **Quadratic speedup** in spreading

---

## Daily Checklist

- [ ] I understand the coin and position spaces
- [ ] I can construct the walk operator
- [ ] I see how interference leads to faster spreading
- [ ] I know the effect of initial coin state
- [ ] I can compare to classical random walks
- [ ] I ran the computational lab and visualized the walks

---

*Next: Day 633 — Coin and Shift Operators*
