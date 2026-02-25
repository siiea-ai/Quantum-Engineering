# Day 795: Mutual Statistics and Braiding

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 114: Anyons & Topological Order

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | e-m braiding phase and Berry connection |
| Afternoon | 2.5 hours | Problem solving: braiding operations |
| Evening | 1.5 hours | Computational lab: visualizing braiding |

---

## Learning Objectives

By the end of today, you will be able to:

1. Calculate the phase acquired when e circles m: $e^{i\pi} = -1$
2. Derive the mutual statistics using string operator commutation
3. Understand the Berry phase interpretation of braiding
4. Construct explicit braiding matrices for toric code anyons
5. Explain why e and m are "mutual semions"
6. Connect braiding to the Aharonov-Bohm effect

---

## Core Content

### 1. The Key Question: What Happens When e Circles m?

We've established:
- e-particles have bosonic self-statistics: $\theta_{ee} = 0$
- m-particles have bosonic self-statistics: $\theta_{mm} = 0$

But what about **mutual statistics**—what happens when an e-particle makes a complete loop around an m-particle?

$$\boxed{R_{em} = e^{i\pi} = -1}$$

This remarkable result means e and m are **mutual semions**: neither one is a semion itself, but together they exhibit nontrivial mutual statistics.

### 2. Deriving the Braiding Phase

#### Method 1: String Operator Commutation

Consider the commutation of Z-strings and X-strings:

For a single edge $e$:
$$Z_e X_e = -X_e Z_e$$

This anticommutation is the source of the nontrivial mutual statistics.

**Setup**:
- m-particle at plaquette $p_0$, created by X-string $S_X$
- e-particle at vertex $v_0$, created by Z-string $S_Z$

To braid e around m, we extend the Z-string in a loop encircling $p_0$.

**The Calculation**:

Let $\gamma$ be a Z-string that forms a closed loop encircling exactly one plaquette $p_0$.

This Z-loop equals the plaquette operator:
$$S_Z(\gamma_{\text{loop around } p}) = B_p$$

If there's an m-particle at $p_0$ (meaning $B_{p_0} = -1$), then:
$$S_Z(\gamma_{\text{loop}}) |\psi_m\rangle = B_{p_0} |\psi_m\rangle = -|\psi_m\rangle$$

The e-particle, after completing a loop around the m-particle, acquires a phase of $-1$.

#### Method 2: Counting Edge Crossings

An alternative derivation uses explicit string crossings.

Consider:
- X-string from $p_1$ to $p_2$ passing through edges $\{e_1, e_2, \ldots, e_n\}$
- Z-string forming a loop that crosses this X-string exactly once at edge $e_k$

The commutator:
$$S_Z S_X = (-1)^{\text{crossings}} S_X S_Z$$

Since there's exactly one crossing:
$$S_Z S_X = -S_X S_Z$$

This minus sign is the braiding phase!

### 3. Berry Phase Interpretation

#### Adiabatic Transport

Consider adiabatically transporting an e-particle around an m-particle:

1. **Initial state**: e at position $v_0$, m at plaquette $p_0$
2. **Transport**: Move e along path $\gamma$ that encircles $p_0$
3. **Final state**: e returns to $v_0$

The wave function acquires a **geometric phase** (Berry phase):
$$|\psi_{\text{final}}\rangle = e^{i\phi_{\text{Berry}}} |\psi_{\text{initial}}\rangle$$

For the toric code:
$$\phi_{\text{Berry}} = \pi$$

#### Berry Connection Formalism

The Berry phase is:
$$\phi = \oint_\gamma \mathcal{A} \cdot d\ell + \phi_{\text{dynamical}}$$

where $\mathcal{A}$ is the Berry connection. In the presence of an m-particle, the Berry connection has a singularity creating a $\pi$ flux.

This is the **Aharonov-Bohm effect** for anyons:
- e-particle = charged particle
- m-particle = flux tube carrying $\pi$ flux
- Braiding = Aharonov-Bohm phase

### 4. The Mutual Semion

#### Definition

A **semion** is an anyon with exchange phase $\theta = \pi/2$. A full exchange (one particle going around another and returning) gives phase $2\theta = \pi$.

For self-statistics:
- True semion: $\theta_{aa} = \pi/2$

For mutual statistics of e and m:
- $\theta_{em} = \pi/2$ (half-braiding gives $e^{i\pi/2}$)
- Full braiding: $e^{2i\theta_{em}} = e^{i\pi} = -1$

e and m are **mutual semions**: their mutual statistics equals that of a semion, even though neither is a semion individually.

#### Comparison of Statistics

| Anyon Pair | Self-Statistics | Mutual Statistics |
|------------|-----------------|-------------------|
| e with e | $\theta_{ee} = 0$ (boson) | — |
| m with m | $\theta_{mm} = 0$ (boson) | — |
| e with m | — | $\theta_{em} = \pi/2$ (mutual semion) |

### 5. Braiding Matrices

#### Full Braid Matrix for Two Anyons

For anyons $a$ and $b$, the braiding matrix $R_{ab}$ gives the phase when $a$ goes counterclockwise around $b$:

$$\boxed{R^{ab} = e^{i\theta_{ab}}}$$

For the toric code:

$$R^{11} = 1, \quad R^{1e} = R^{e1} = 1, \quad R^{1m} = R^{m1} = 1$$
$$R^{ee} = 1, \quad R^{mm} = 1$$
$$R^{em} = R^{me} = -1$$

#### The R-Matrix

In the more general framework of modular tensor categories, the R-matrix encodes all braiding data:

$$R = \begin{pmatrix} R^{11} & R^{1e} & R^{1m} & R^{1\varepsilon} \\ R^{e1} & R^{ee} & R^{em} & R^{e\varepsilon} \\ R^{m1} & R^{me} & R^{mm} & R^{m\varepsilon} \\ R^{\varepsilon 1} & R^{\varepsilon e} & R^{\varepsilon m} & R^{\varepsilon\varepsilon} \end{pmatrix}$$

For the toric code:
$$R = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & 1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}$$

### 6. Modular S-Matrix

The **S-matrix** encodes the mutual statistics information:

$$S_{ab} = \frac{1}{D} \sum_c N_{ab}^c d_c \theta_c / (\theta_a \theta_b)$$

For Abelian anyons, this simplifies to:
$$S_{ab} = \frac{1}{D} e^{2\pi i \theta_{ab}}$$

For the toric code ($D = 2$):
$$S = \frac{1}{2} \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & 1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}$$

This matrix is:
- Symmetric: $S = S^T$
- Unitary: $S^\dagger S = I$
- Determines the ground state degeneracy on torus

### 7. Braiding as Gauge-Invariant Observable

#### Path Independence of Braiding

The braiding phase depends only on:
- The topological class of the path (how many times e winds around m)
- The anyon types involved

It does **not** depend on:
- The exact path taken
- The speed of transport
- Local details of the Hamiltonian

This is because the braiding phase is a **topological invariant**.

#### Gauge Transformations

String operators are not unique—we can modify them by stabilizers:
$$S'_Z = S_Z \cdot \prod_p B_p^{n_p}$$

But the braiding phase is unchanged because stabilizers have eigenvalue +1 in the code space.

### 8. Physical Realizations

#### Fractional Quantum Hall Effect

In the ν = 1/3 Laughlin state:
- Quasiparticles carry fractional charge $e/3$
- Braiding phase: $\theta = \pi/3$ (true anyons!)

#### Topological Superconductors

Majorana bound states exhibit non-Abelian braiding—even more exotic than the Abelian case we study here.

#### Quantum Simulation

Cold atoms in optical lattices and superconducting qubits can simulate toric code anyons and verify braiding statistics.

---

## Quantum Computing Connection

### Braiding for Computation

In topological quantum computing:
- **Qubits**: Encoded in anyon positions/fusion channels
- **Gates**: Implemented by braiding anyons
- **Readout**: Fusion outcomes

For the toric code's Abelian anyons:
- Braiding gives only phases (diagonal gates)
- Cannot perform universal quantum computation by braiding alone
- Need additional operations (magic state injection)

### Non-Abelian Extension

For universal topological quantum computing, we need non-Abelian anyons where:
- Braiding implements non-diagonal unitary matrices
- The braid group representation is dense in SU(N)

Examples: Fibonacci anyons, Ising anyons (Majorana-based).

### Error Detection via Braiding

The braiding phase can detect the presence of anyons:
- Prepare test e-particle
- Braid around suspected m-particle location
- Measure phase: $-1$ confirms m-particle present

This is the theoretical basis for topological syndrome measurement.

---

## Worked Examples

### Example 1: Explicit Braiding Calculation

**Problem**: On a 4×4 toric code, explicitly compute the braiding phase when an e-particle at vertex $(0,0)$ circles an m-particle at plaquette $(1,1)$.

**Solution**:

**Step 1**: Create initial configuration.
- Z-string from $(0,0)$ to far away creates e at $(0,0)$
- X-string creates m at plaquette $(1,1)$

**Step 2**: Identify the braiding path.
A Z-loop encircling plaquette $(1,1)$:
$$\gamma: (1,1) \to (2,1) \to (2,2) \to (1,2) \to (1,1)$$

The edges in this loop are exactly the boundary of plaquette $(1,1)$.

**Step 3**: Compute the Z-loop operator.
$$S_Z(\gamma) = Z_{(1,1,h)} Z_{(2,1,v)} Z_{(1,2,h)} Z_{(1,1,v)} = B_{(1,1)}$$

Wait, this equals $B_p$, but with Z operators, not X!

Let me reconsider. A Z-loop around the boundary of plaquette $p$ is:
$$\prod_{e \in \partial p} Z_e$$

This is **not** the same as $B_p = \prod_{e \in \partial p} X_e$.

**Correct analysis**:

The Z-loop creates an e-particle worldline that encircles the m-particle. The key is the commutation when we "pull" the Z-loop through the X-string.

Let the X-string creating the m-particle at $(1,1)$ cross edge $e_0$.

The Z-loop contains edge $e_0$ (it goes around the plaquette).

$$S_Z(\text{loop}) \cdot S_X(\text{string}) = (-1)^{\text{crossings}} S_X(\text{string}) \cdot S_Z(\text{loop})$$

If the loop crosses the X-string once:
$$S_Z S_X |\Omega\rangle = -S_X S_Z |\Omega\rangle = -S_X |\Omega\rangle$$

since $S_Z(\text{closed loop})|\Omega\rangle = |\Omega\rangle$ (it's a product of stabilizers... wait, is it?).

Actually, a closed Z-loop is a product of **star operators** if it's contractible. Let's be more careful.

**Careful Analysis**:

A closed Z-loop on the original lattice that bounds a region $R$:
$$S_Z(\partial R) = \prod_{v \in R} A_v$$

This is because each edge interior to $R$ appears in exactly 2 star operators, canceling, while boundary edges appear once.

So $S_Z(\partial R)|\Omega\rangle = |\Omega\rangle$ since $A_v|\Omega\rangle = |\Omega\rangle$.

But the **phase** comes from commuting past the X-string:

$$S_Z(\partial R) S_X(\gamma^*) = (-1)^{\text{mod 2 crossings}} S_X(\gamma^*) S_Z(\partial R)$$

If the X-string enters the region $R$ and exits (or passes through entirely), the number of crossings is even → no phase.

If the X-string starts or ends inside $R$, the number of crossings is odd → phase $-1$.

**For m-particle at plaquette $(1,1)$**:

The m-particle is inside the Z-loop (which encircles plaquette $(1,1)$). The X-string endpoint is at $(1,1)$.

Crossings: 1 (the X-string ends inside the loop).

$$\boxed{\text{Braiding phase} = -1}$$

### Example 2: Half-Braiding

**Problem**: Calculate the phase for a half-braiding (e goes halfway around m, exchanges positions with another e).

**Solution**:

A half-braiding is not well-defined for distinguishable particles. For indistinguishable particles or for the abstract braiding operation:

Full braiding phase: $R_{em} = -1 = e^{i\pi}$

Half-braiding: $\sqrt{R_{em}} = e^{i\pi/2} = i$

This interpretation requires care—the square root has a sign ambiguity. The convention is set by consistency with fusion rules (Day 796).

$$\boxed{\text{Half-braiding phase} = e^{i\pi/2} = i}$$

### Example 3: Braiding e Around Two m-Particles

**Problem**: What phase does e acquire when braiding around two m-particles?

**Solution**:

Phases from Abelian anyons are additive:

If e circles m₁ and m₂ (both inside the loop):
$$\text{Phase} = R_{em_1} \cdot R_{em_2} = (-1)(-1) = +1$$

Alternatively, two m-particles can fuse to vacuum: $m \times m = 1$.
The braiding phase of e around vacuum is trivial.

$$\boxed{\text{Phase around two m-particles} = +1}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Triple Braiding
An e-particle goes around an m-particle 3 times. What is the total accumulated phase?

**Problem 2**: String Crossing Count
On a 5×5 toric code, an X-string goes from plaquette $(0,0)$ to $(4,4)$ along a diagonal path. A Z-loop encircles plaquette $(2,2)$. How many times does the Z-loop cross the X-string?

**Problem 3**: Braiding Order
Does the braiding phase depend on whether e moves clockwise or counterclockwise around m? Explain.

### Intermediate

**Problem 4**: Aharonov-Bohm Analogy
In the Aharonov-Bohm effect, a charged particle circling a flux tube of strength $\Phi$ acquires phase $e^{ie\Phi/\hbar}$. What "effective flux" does an m-particle represent for an e-particle?

**Problem 5**: S-Matrix Properties
Verify that the toric code S-matrix satisfies $S^2 = C$ where $C$ is the charge conjugation matrix (swapping particles with antiparticles).

**Problem 6**: Braiding Non-Commutativity
For Abelian anyons, braiding is commutative: braiding $a$ around $b$ and then $c$ around $d$ gives the same result as braiding $c$ around $d$ first. Prove this for the toric code.

### Challenging

**Problem 7**: Braiding Worldlines
Draw the worldline diagram in (2+1)D spacetime for an e-particle braiding around an m-particle. Label the initial and final states.

**Problem 8**: Non-Abelian Preview
For Ising anyons, the braiding matrices are:
$$R^{\sigma\sigma} = e^{-i\pi/8} \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$
Compute $R^{\sigma\sigma} \cdot R^{\sigma\sigma}$ and interpret the result.

---

## Computational Lab: Visualizing Braiding

```python
"""
Day 795 Computational Lab: Mutual Statistics and Braiding
Visualizing the e-m braiding phase in the toric code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

class ToricCodeBraiding:
    """
    Toric code simulator for demonstrating braiding operations
    """

    def __init__(self, L):
        """Initialize L x L toric code"""
        self.L = L
        self.n_edges = 2 * L * L

        self.edge_list = []
        for x in range(L):
            for y in range(L):
                self.edge_list.append((x, y, 'h'))
                self.edge_list.append((x, y, 'v'))
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}

        self.reset_state()

    def reset_state(self):
        """Reset to ground state"""
        self.x_errors = np.zeros(self.n_edges, dtype=int)
        self.z_errors = np.zeros(self.n_edges, dtype=int)
        self.phase = 1.0  # Track accumulated phase

    def get_plaquette_edges(self, p):
        x, y = p
        L = self.L
        return [(x, y, 'h'), (x, (y+1) % L, 'h'),
                (x, y, 'v'), ((x+1) % L, y, 'v')]

    def get_star_edges(self, v):
        x, y = v
        L = self.L
        return [(x, y, 'h'), ((x-1) % L, y, 'h'),
                (x, y, 'v'), (x, (y-1) % L, 'v')]

    def apply_x_error(self, edge):
        idx = self.edge_to_idx[edge]
        self.x_errors[idx] = 1 - self.x_errors[idx]

    def apply_z_error(self, edge):
        idx = self.edge_to_idx[edge]
        self.z_errors[idx] = 1 - self.z_errors[idx]

    def compute_e_particles(self):
        """Find all e-particle locations"""
        e_particles = set()
        for x in range(self.L):
            for y in range(self.L):
                edges = self.get_star_edges((x, y))
                parity = sum(self.z_errors[self.edge_to_idx[e]] for e in edges)
                if parity % 2 == 1:
                    e_particles.add((x, y))
        return e_particles

    def compute_m_particles(self):
        """Find all m-particle locations"""
        m_particles = set()
        for x in range(self.L):
            for y in range(self.L):
                edges = self.get_plaquette_edges((x, y))
                parity = sum(self.x_errors[self.edge_to_idx[e]] for e in edges)
                if parity % 2 == 1:
                    m_particles.add((x, y))
        return m_particles

    def count_string_crossings(self, z_edges, x_edges):
        """Count how many edges are in both Z and X strings"""
        z_set = set(z_edges)
        x_set = set(x_edges)
        return len(z_set & x_set)

    def compute_braiding_phase(self, z_path, x_string_edges):
        """
        Compute phase from Z-path crossing X-string
        z_path: list of edges in the Z-string
        x_string_edges: set of edges in the X-string
        """
        crossings = self.count_string_crossings(z_path, x_string_edges)
        phase = (-1) ** crossings
        return phase


def visualize_em_braiding():
    """Visualize e-particle braiding around m-particle"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    L = 6
    tc = ToricCodeBraiding(L)

    # Setup: m-particle at plaquette (2, 2)
    m_pos = (2, 2)

    # Create m-particle with X-string
    x_string_edges = [(2, 2, 'v'), (3, 2, 'v')]  # Minimal X-string
    for e in x_string_edges:
        tc.apply_x_error(e)

    # e-particle starting position
    e_start = (1, 2)

    # Braiding path: e goes around m
    braiding_path = [
        (1, 2), (2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)
    ]

    # Convert vertex path to edge path
    def get_edge_between(v1, v2):
        x1, y1 = v1
        x2, y2 = v2
        L = tc.L
        if y1 == y2:
            if (x1 + 1) % L == x2:
                return (x1, y1, 'h')
            elif (x2 + 1) % L == x1:
                return (x2, y2, 'h')
        if x1 == x2:
            if (y1 + 1) % L == y2:
                return (x1, y1, 'v')
            elif (y2 + 1) % L == y1:
                return (x2, y2, 'v')
        return None

    z_path_edges = []
    for i in range(len(braiding_path) - 1):
        edge = get_edge_between(braiding_path[i], braiding_path[i+1])
        if edge:
            z_path_edges.append(edge)

    # Visualize stages
    stages = [0, 1, 2, 3, 4, 5]

    for idx, stage in enumerate(stages):
        ax = axes[idx // 3, idx % 3]

        # Draw lattice
        for x in range(L):
            for y in range(L):
                ax.plot([x, x+1], [y, y], 'lightgray', linewidth=1)
                ax.plot([x, x], [y, y+1], 'lightgray', linewidth=1)

        # Draw X-string (creating m-particle)
        for e in x_string_edges:
            x, y, d = e
            if d == 'h':
                ax.plot([x, x+1], [y, y], 'red', linewidth=3, alpha=0.5)
            else:
                ax.plot([x, x], [y, y+1], 'red', linewidth=3, alpha=0.5)

        # Draw m-particle
        rect = Rectangle((m_pos[0]+0.2, m_pos[1]+0.2), 0.6, 0.6,
                         color='red', alpha=0.7, zorder=4)
        ax.add_patch(rect)
        ax.annotate('m', (m_pos[0]+0.5, m_pos[1]+0.5), ha='center', va='center',
                   fontsize=14, color='white', fontweight='bold', zorder=5)

        # Draw braiding path up to current stage
        edges_so_far = z_path_edges[:stage+1] if stage < len(z_path_edges) else z_path_edges
        for e in edges_so_far:
            x, y, d = e
            if d == 'h':
                ax.plot([x, x+1], [y, y], 'blue', linewidth=4, alpha=0.6)
            else:
                ax.plot([x, x], [y, y+1], 'blue', linewidth=4, alpha=0.6)

        # Draw e-particle at current position
        if stage < len(braiding_path):
            e_pos = braiding_path[stage]
        else:
            e_pos = braiding_path[-1]

        circle = Circle((e_pos[0], e_pos[1]), 0.15, color='blue', zorder=5)
        ax.add_patch(circle)
        ax.annotate('e', (e_pos[0], e_pos[1]), ha='center', va='center',
                   fontsize=12, color='white', fontweight='bold', zorder=6)

        # Compute phase
        crossings = tc.count_string_crossings(edges_so_far, x_string_edges)
        phase = (-1) ** crossings

        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-0.5, L + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {stage+1}: Phase = {phase:+d}\nCrossings: {crossings}",
                    fontsize=12)

    fig.suptitle("e-Particle Braiding Around m-Particle\nFinal phase = -1 (semionic mutual statistics)",
                fontsize=16)
    plt.tight_layout()
    plt.savefig('em_braiding.png', dpi=150, bbox_inches='tight')
    plt.show()

    return tc.compute_braiding_phase(z_path_edges, x_string_edges)


def visualize_braiding_matrix():
    """Visualize the R-matrix for toric code anyons"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R-matrix values
    anyons = ['1', 'e', 'm', 'ε']
    R = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ])

    # Plot R-matrix
    ax = axes[0]
    im = ax.imshow(R, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(anyons)
    ax.set_yticklabels(anyons)
    ax.set_xlabel('Anyon b (encircled)', fontsize=12)
    ax.set_ylabel('Anyon a (braiding)', fontsize=12)
    ax.set_title('Braiding Matrix R^{ab}\n(phase when a circles b)', fontsize=14)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            color = 'white' if abs(R[i, j]) > 0.5 else 'black'
            ax.annotate(f'{R[i,j]:+d}', (j, i), ha='center', va='center',
                       fontsize=14, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Phase')

    # S-matrix (normalized)
    ax = axes[1]
    S = R / 2
    im = ax.imshow(S, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(anyons)
    ax.set_yticklabels(anyons)
    ax.set_xlabel('Anyon b', fontsize=12)
    ax.set_ylabel('Anyon a', fontsize=12)
    ax.set_title('Modular S-Matrix\nS = R/D (D=2 is total quantum dimension)', fontsize=14)

    for i in range(4):
        for j in range(4):
            color = 'white' if abs(S[i, j]) > 0.25 else 'black'
            ax.annotate(f'{S[i,j]:+.1f}', (j, i), ha='center', va='center',
                       fontsize=14, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Matrix element')

    plt.tight_layout()
    plt.savefig('braiding_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_worldlines():
    """Visualize braiding in (2+1)D spacetime"""
    fig = plt.figure(figsize=(14, 6))

    # 2D view (spatial snapshot)
    ax1 = fig.add_subplot(121)

    # Draw braiding path
    theta = np.linspace(0, 2*np.pi, 100)
    x = 1.5 * np.cos(theta)
    y = 1.5 * np.sin(theta)

    ax1.plot(x, y, 'b-', linewidth=2, label='e-particle path')
    ax1.arrow(x[25], y[25], x[26]-x[25], y[26]-y[25],
             head_width=0.2, head_length=0.1, fc='blue', ec='blue')

    # m-particle at center
    circle = Circle((0, 0), 0.3, color='red', zorder=5)
    ax1.add_patch(circle)
    ax1.annotate('m', (0, 0), ha='center', va='center',
                fontsize=16, color='white', fontweight='bold', zorder=6)

    # e-particle
    circle = Circle((1.5, 0), 0.2, color='blue', zorder=5)
    ax1.add_patch(circle)
    ax1.annotate('e', (1.5, 0), ha='center', va='center',
                fontsize=14, color='white', fontweight='bold', zorder=6)

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Spatial View: e circles m', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2+1)D spacetime view
    ax2 = fig.add_subplot(122, projection='3d')

    # Time axis
    t = np.linspace(0, 2*np.pi, 100)

    # m-particle worldline (stationary)
    ax2.plot([0]*100, [0]*100, t, 'r-', linewidth=4, label='m worldline')

    # e-particle worldline (braiding)
    x_e = 1.5 * np.cos(t)
    y_e = 1.5 * np.sin(t)
    ax2.plot(x_e, y_e, t, 'b-', linewidth=3, label='e worldline')

    # Mark start and end points
    ax2.scatter([1.5, 1.5], [0, 0], [0, 2*np.pi], c=['blue', 'blue'],
               s=100, marker='o', zorder=5)
    ax2.scatter([0, 0], [0, 0], [0, 2*np.pi], c=['red', 'red'],
               s=100, marker='s', zorder=5)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('time', fontsize=12)
    ax2.set_title('Spacetime View: Braiding = Worldline Linking\nLinking number = 1 → Phase = -1', fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('worldlines.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_multiple_braidings():
    """Show how phase accumulates with multiple braidings"""
    print("=" * 60)
    print("Phase Accumulation with Multiple Braidings")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 6))

    n_braidings = np.arange(0, 9)
    phases = [(-1)**n for n in n_braidings]
    phase_angles = [n * np.pi for n in n_braidings]

    # Plot phase values
    colors = ['green' if p == 1 else 'red' for p in phases]
    bars = ax.bar(n_braidings, phases, color=colors, edgecolor='black', linewidth=2)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Number of Complete Braidings', fontsize=14)
    ax.set_ylabel('Phase Factor', fontsize=14)
    ax.set_title('Accumulated Phase: e Circling m Multiple Times\n'
                'Phase = (-1)^n (Abelian, Z₂ periodicity)', fontsize=16)
    ax.set_xticks(n_braidings)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1', '0', '+1'])

    # Add phase angle labels
    for i, (n, p) in enumerate(zip(n_braidings, phases)):
        angle = n * 180  # degrees
        ax.annotate(f'{angle}°', (n, p + 0.1 * np.sign(p)), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('multiple_braidings.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nResults:")
    for n, p in zip(n_braidings, phases):
        print(f"  {n} braidings: phase = {p:+d} = e^{{i·{n}π}}")


def verify_smatrix_properties():
    """Verify mathematical properties of the S-matrix"""
    print("\n" + "=" * 60)
    print("Verifying S-Matrix Properties")
    print("=" * 60)

    S = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ]) / 2

    # Property 1: Unitarity
    print("\n1. Unitarity: S†S = I")
    SdagS = S.conj().T @ S
    print(f"   S†S =\n{np.round(SdagS, 10)}")
    print(f"   Is unitary: {np.allclose(SdagS, np.eye(4))}")

    # Property 2: Symmetry
    print("\n2. Symmetry: S = S^T")
    print(f"   Is symmetric: {np.allclose(S, S.T)}")

    # Property 3: S^2 = C (charge conjugation)
    print("\n3. S² = C (charge conjugation)")
    S2 = S @ S
    print(f"   S² =\n{np.round(S2, 10)}")

    # Charge conjugation for Z2xZ2: each anyon is its own antiparticle
    C = np.eye(4)  # For toric code
    print(f"   C =\n{C}")
    print(f"   S² = C: {np.allclose(S2, C)}")

    # Property 4: Verlinde formula test
    print("\n4. Fusion rules from S-matrix (Verlinde formula):")
    print("   N^c_{ab} = Σ_x (S_ax S_bx S*_cx) / S_1x")

    anyons = ['1', 'e', 'm', 'ε']
    for a in range(4):
        for b in range(4):
            # Compute fusion coefficient
            for c in range(4):
                N = 0
                for x in range(4):
                    N += S[a,x] * S[b,x] * np.conj(S[c,x]) / S[0,x]
                N = np.real(N)
                if np.abs(N) > 0.1:
                    print(f"   {anyons[a]} × {anyons[b]} → {anyons[c]}: N = {N:.0f}")


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Day 795: Mutual Statistics and Braiding - Computational Lab")
    print("=" * 60)

    print("\n1. Visualizing e-m braiding...")
    phase = visualize_em_braiding()
    print(f"\nFinal braiding phase: {phase}")

    print("\n2. Braiding and S-matrices...")
    visualize_braiding_matrix()

    print("\n3. Spacetime worldlines...")
    visualize_worldlines()

    print("\n4. Multiple braidings...")
    demonstrate_multiple_braidings()

    print("\n5. S-matrix properties...")
    verify_smatrix_properties()

    print("\n" + "=" * 60)
    print("Key Insights from Lab:")
    print("=" * 60)
    print("1. e circling m once gives phase -1 (semionic)")
    print("2. The phase comes from Z crossing X: ZX = -XZ")
    print("3. R-matrix encodes all braiding information")
    print("4. S-matrix is unitary and symmetric")
    print("5. Verlinde formula derives fusion from braiding")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| e-m braiding phase | $R_{em} = e^{i\pi} = -1$ |
| Half-braiding | $\sqrt{R_{em}} = e^{i\pi/2} = i$ |
| String crossing | $Z_e X_e = -X_e Z_e$ |
| Multiple braidings | $R_{em}^n = (-1)^n$ |
| S-matrix | $S_{ab} = \frac{1}{D} R_{ab}$ |
| Aharonov-Bohm flux | $\Phi_m = \pi \hbar / e$ |

### Main Takeaways

1. **Mutual semions**: e and m are bosons individually but have semionic mutual statistics
2. **Origin of phase**: The $ZX = -XZ$ anticommutation at string crossings
3. **Berry phase**: Braiding phase = geometric phase from adiabatic transport
4. **Topological invariant**: The phase depends only on winding number, not path details
5. **Abelian statistics**: All braiding phases commute; limited computational power
6. **Foundation for TQC**: Braiding phases are robust to local noise

---

## Daily Checklist

### Morning Theory (3 hours)
- [ ] Derive the e-m braiding phase from string commutation
- [ ] Understand the Berry phase interpretation
- [ ] Construct the R-matrix for toric code anyons
- [ ] Connect to the Aharonov-Bohm effect

### Afternoon Problems (2.5 hours)
- [ ] Complete all Direct Application problems
- [ ] Work through at least 2 Intermediate problems
- [ ] Attempt at least 1 Challenging problem

### Evening Lab (1.5 hours)
- [ ] Run all visualization code
- [ ] Experiment with different braiding paths
- [ ] Verify the S-matrix properties

### Self-Assessment Questions
1. Why is the e-m braiding phase exactly $-1$?
2. What physical effect is analogous to anyon braiding?
3. Why can't toric code anyons perform universal quantum computation?

---

## Preview: Day 796

Tomorrow we complete our picture of toric code anyons with **fusion rules**. We'll see how combining anyons produces new particles: $e \times e = 1$, $m \times m = 1$, and the remarkable $e \times m = \varepsilon$—the epsilon fermion. This bound state has fermionic self-statistics, emerging from two bosons!

---

*Day 795 of 2184 | Year 2, Month 29, Week 114 | Quantum Engineering PhD Curriculum*
