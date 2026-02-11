# Day 793: Electric Charges (e-particles)

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 114: Anyons & Topological Order

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Star operator violations and Z-string theory |
| Afternoon | 2.5 hours | Problem solving: e-particle manipulation |
| Evening | 1.5 hours | Computational lab: e-particle dynamics |

---

## Learning Objectives

By the end of today, you will be able to:

1. Identify e-particles as violations of star operators ($A_v = -1$)
2. Construct Z-string operators that create e-particle pairs
3. Demonstrate that e-particles always appear in pairs
4. Calculate the energy cost of e-particle excitations
5. Prove the bosonic self-statistics of e-particles
6. Transport e-particles through the lattice using string operators

---

## Core Content

### 1. Review: The Star Operator

The star operator $A_v$ acts on all edges incident to vertex $v$:

$$\boxed{A_v = \prod_{e \ni v} Z_e}$$

On a square lattice, each vertex has 4 incident edges, so:
$$A_v = Z_{\text{up}} Z_{\text{down}} Z_{\text{left}} Z_{\text{right}}$$

**Properties**:
- $A_v^2 = I$ (involutory)
- $A_v$ has eigenvalues $\pm 1$
- $[A_v, A_{v'}] = 0$ for all vertices
- $[A_v, B_p] = 0$ for all vertices and plaquettes

### 2. Defining e-Particles

An **e-particle** (electric charge) is localized at a vertex $v$ where:

$$\boxed{A_v |\psi\rangle = -|\psi\rangle}$$

The name "electric charge" comes from the analogy with $\mathbb{Z}_2$ lattice gauge theory, where these excitations correspond to violations of Gauss's law.

#### Physical Interpretation

In the gauge theory picture:
- Edges carry $\mathbb{Z}_2$ gauge field configurations
- Vertices are subject to the Gauss law constraint $A_v = +1$
- An e-particle is a source of $\mathbb{Z}_2$ electric field

#### Energy Cost

From the Hamiltonian $H = -\sum_v A_v - \sum_p B_p$:
- Ground state: $A_v = +1$ contributes $-1$ to energy
- Excited state: $A_v = -1$ contributes $+1$ to energy
- **Energy cost per e-particle**: $\Delta E = 2$

### 3. Creating e-Particles with Z-Strings

#### The Fundamental Creation Operator

A **Z-string operator** is a product of $Z$ operators along a path $\gamma$:

$$\boxed{S_Z(\gamma) = \prod_{e \in \gamma} Z_e}$$

**Theorem**: $S_Z(\gamma)$ anticommutes with star operators at the endpoints of $\gamma$ and commutes with all other star operators.

**Proof**:

Consider a vertex $v$ and the star operator $A_v = \prod_{e \ni v} Z_e$.

For any edge $e'$:
$$Z_{e'} A_v = \begin{cases} -A_v Z_{e'} & \text{if } e' \ni v \\ +A_v Z_{e'} & \text{if } e' \not\ni v \end{cases}$$

If $\gamma$ passes through $v$, it uses exactly 2 edges incident to $v$ (entering and leaving). But for an endpoint, only 1 edge is incident.

At an endpoint $v_{\text{end}}$: $S_Z(\gamma)$ contains exactly one $Z_e$ with $e \ni v_{\text{end}}$, so:
$$S_Z(\gamma) A_{v_{\text{end}}} = -A_{v_{\text{end}}} S_Z(\gamma)$$

At intermediate vertex $v_{\text{mid}}$: $S_Z(\gamma)$ contains exactly two $Z_e$'s with $e \ni v_{\text{mid}}$:
$$S_Z(\gamma) A_{v_{\text{mid}}} = (-1)^2 A_{v_{\text{mid}}} S_Z(\gamma) = +A_{v_{\text{mid}}} S_Z(\gamma)$$

$\square$

#### Creating a Pair of e-Particles

Starting from the ground state $|\Omega\rangle$ with $A_v |\Omega\rangle = +|\Omega\rangle$ for all $v$:

$$|\psi_{e,e}\rangle = S_Z(\gamma) |\Omega\rangle$$

At the endpoints $v_1, v_2$ of $\gamma$:
$$A_{v_1} |\psi_{e,e}\rangle = A_{v_1} S_Z(\gamma) |\Omega\rangle = -S_Z(\gamma) A_{v_1} |\Omega\rangle = -S_Z(\gamma) |\Omega\rangle = -|\psi_{e,e}\rangle$$

Similarly for $v_2$. This state has e-particles at $v_1$ and $v_2$.

### 4. Pair Creation Is Mandatory

**Theorem**: It is impossible to create a single e-particle; they always come in pairs.

**Proof 1** (Algebraic):

Consider the global constraint:
$$\prod_{v} A_v = I$$

This is because each edge contributes to exactly 2 vertices, so each $Z_e$ appears twice in the product.

If $|\psi\rangle$ is any state:
$$\prod_v A_v |\psi\rangle = |\psi\rangle$$

Let $n_e$ be the number of vertices with $A_v = -1$:
$$\prod_v A_v |\psi\rangle = (-1)^{n_e} |\psi\rangle = |\psi\rangle$$

Therefore $(-1)^{n_e} = 1$, so $n_e$ is even.

$\square$

**Proof 2** (Topological):

The boundary of a 1-chain (path) always has an even number of points. The Z-string is a 1-chain, and e-particles live at its boundary.

### 5. Moving e-Particles

#### Transport via String Extension

To move an e-particle from $v_1$ to $v_3$, extend the Z-string:

Initial state: $|\psi_1\rangle = S_Z(\gamma_{12}) |\Omega\rangle$ with e-particles at $v_1, v_2$

Apply: $Z_e$ where $e$ connects $v_1$ to $v_3$

Result: e-particle moves from $v_1$ to $v_3$

$$|\psi_2\rangle = Z_e |\psi_1\rangle = S_Z(\gamma_{32}) |\Omega\rangle$$

Now e-particles are at $v_2, v_3$.

#### String Independence

**Key Insight**: The positions of e-particles depend only on the endpoints of the Z-string, not the path taken.

Two Z-strings $\gamma, \gamma'$ with the same endpoints differ by a closed loop. A closed Z-loop is a product of plaquette operators $B_p$:

$$S_Z(\gamma) S_Z(\gamma')^\dagger = \prod_{p \in \text{interior}} B_p$$

Acting on the ground state: $B_p |\Omega\rangle = |\Omega\rangle$, so:
$$S_Z(\gamma) |\Omega\rangle = S_Z(\gamma') |\Omega\rangle$$

The e-particle positions are **topologically determined**.

### 6. Self-Statistics of e-Particles

#### Exchange of Two e-Particles

Consider exchanging two e-particles. We need to compute the Berry phase acquired when:
1. e-particle at $v_1$ moves to $v_2$
2. e-particle at $v_2$ moves to $v_1$

**Method**: Use the string picture.

Initial configuration: Z-string from $v_1$ to $v_2$ creates e-particles at both locations.

After exchange: The string is deformed but endpoints return to original positions.

**Key calculation**: The exchange operation is equivalent to extending the Z-string in a loop. For e-particles only, this loop doesn't enclose any m-particles.

The phase acquired is:
$$\theta_{ee} = 0$$

**e-particles are bosons** with respect to self-exchange.

#### Formal Proof Using Ribbon Operators

The full proof uses the **ribbon operator** formalism:

A ribbon $R$ is a path with a framing (thickness). The e-particle ribbon operator is:
$$F_e(R) = S_Z(\gamma_R)$$

where $\gamma_R$ is the central path of the ribbon.

Exchange of two e-particles corresponds to a half-twist of a ribbon connecting them. For e-ribbons:
$$\text{Half-twist} = (+1)$$

because $Z \cdot Z = I$ contributes no phase.

### 7. e-Particles and Error Correction

#### Connection to Bit-Flip Errors

In the error correction picture:
- **Z error on edge $e$**: $Z_e$ acting on a codeword
- **Syndrome**: The vertices where $A_v = -1$

A Z-error string creates e-particles at its endpoints. The syndrome reveals e-particle locations, but **not** the error path!

#### Error Equivalence Classes

Different Z-error strings with the same endpoints are equivalent up to stabilizers:
$$Z_{\gamma_1} \sim Z_{\gamma_2} \iff Z_{\gamma_1} Z_{\gamma_2} \in \langle B_p \rangle$$

This is why we only need to find **any** correction connecting the syndrome points.

#### Logical Errors

A Z-string wrapping around a non-trivial cycle of the torus:
- Creates no e-particles (no endpoints)
- Is not a product of stabilizers
- Implements a **logical operation**

$$\bar{Z}_1 = \prod_{e \in \gamma_{\text{horizontal}}} Z_e$$

This is undetectable but changes the encoded information.

---

## Quantum Computing Connection

### Syndrome-Based Decoding

The positions of e-particles form the **Z-error syndrome**:

1. **Measure** all star operators $A_v$
2. **Identify** vertices with $A_v = -1$ (e-particle locations)
3. **Match** e-particles in pairs
4. **Apply** Z-corrections along paths connecting pairs

The decoder must avoid creating strings that wrap non-trivially around the torus, which would cause logical errors.

### Minimum Weight Perfect Matching

The optimal decoder finds the minimum-weight matching of e-particles:
- Weight = distance between e-particles
- Matching = pairing of all e-particles
- Minimum weight = minimize total correction string length

This is solved efficiently by Edmonds' blossom algorithm in $O(n^3)$ time.

---

## Worked Examples

### Example 1: Computing a Z-String

**Problem**: On a 3×3 toric code, write out the explicit Z-string operator that creates e-particles at vertices $(0,0)$ and $(2,1)$.

**Solution**:

We need to find a path from $(0,0)$ to $(2,1)$ on the lattice. One choice:

Path: $(0,0) \to (1,0) \to (2,0) \to (2,1)$

Edges traversed:
- Horizontal edge from $(0,0)$ to $(1,0)$: call it $e_1$
- Horizontal edge from $(1,0)$ to $(2,0)$: call it $e_2$
- Vertical edge from $(2,0)$ to $(2,1)$: call it $e_3$

$$\boxed{S_Z = Z_{e_1} Z_{e_2} Z_{e_3}}$$

We can verify: this anticommutes with $A_{(0,0)}$ and $A_{(2,1)}$, and commutes with all other star operators.

### Example 2: Verifying the Constraint

**Problem**: On a $4 \times 4$ toric code, verify that $\prod_v A_v = I$ by counting Z operators.

**Solution**:

The lattice has:
- $4 \times 4 = 16$ vertices
- $2 \times 4 \times 4 = 32$ edges (horizontal and vertical)

In $\prod_v A_v$, each $Z_e$ appears for every vertex incident to $e$. Each edge has exactly 2 incident vertices, so:

$$\prod_v A_v = \prod_e Z_e^2 = \prod_e I = I$$

$$\boxed{\prod_v A_v = I \quad \checkmark}$$

### Example 3: Equivalent Z-Strings

**Problem**: Show that two Z-strings from $v_1$ to $v_2$ differ by a product of plaquette operators.

**Solution**:

Let $\gamma_1, \gamma_2$ be two paths from $v_1$ to $v_2$. The concatenation $\gamma_1 \cup \bar{\gamma}_2$ (where $\bar{\gamma}_2$ is $\gamma_2$ reversed) forms a closed loop.

A closed loop on the lattice bounds a region $R$ consisting of plaquettes. For each plaquette $p \in R$:
$$B_p = \prod_{e \in \partial p} X_e$$

But we have Z-strings! The key is that closed Z-loops **are** products of $B_p$'s when acting on states with no m-particles:

On the ground state:
$$S_Z(\gamma_1) S_Z(\gamma_2)^\dagger |\Omega\rangle = S_Z(\gamma_1 \cup \bar{\gamma}_2) |\Omega\rangle$$

The closed loop $\gamma_1 \cup \bar{\gamma}_2$ can be contracted to a point on the torus (if it's homologically trivial). This means:
$$S_Z(\gamma_1 \cup \bar{\gamma}_2) = \prod_{p \in R} B_p^{\alpha_p}$$

for some coefficients, and this acts as identity on $|\Omega\rangle$.

Therefore:
$$\boxed{S_Z(\gamma_1) |\Omega\rangle = S_Z(\gamma_2) |\Omega\rangle}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Edge Counting
On an $L \times L$ toric code, how many edges are there? Verify that $\sum_v (\text{degree of } v) = 2 \times (\text{number of edges})$.

**Problem 2**: Single-Edge String
Apply a single $Z_e$ operator to the ground state. At which vertices do e-particles appear? What is the energy of this state?

**Problem 3**: Syndrome Measurement
Given that measuring star operators yields $A_{(0,0)} = -1$, $A_{(2,2)} = -1$, and all others $+1$, what is the minimum-weight Z-string that corrects this error?

### Intermediate

**Problem 4**: String Commutation
Let $\gamma$ be a Z-string from $v_1$ to $v_2$. Let $\gamma'$ be another Z-string from $v_3$ to $v_4$, where all four vertices are distinct.
(a) Does $S_Z(\gamma)$ commute with $S_Z(\gamma')$?
(b) What if the paths share some edges?

**Problem 5**: Multiple Pairs
Create a state with 4 e-particles at vertices $v_1, v_2, v_3, v_4$.
(a) How many distinct Z-string configurations achieve this?
(b) Are all these states identical?

**Problem 6**: Logical Operator
On a $3 \times 3$ toric code with periodic boundaries, write out the explicit logical $\bar{Z}_1$ operator as a product of Pauli Z's. Verify it commutes with all stabilizers.

### Challenging

**Problem 7**: Degenerate Ground States
Show that on a torus, there are 4 degenerate ground states distinguished by the eigenvalues of $\bar{Z}_1$ and $\bar{Z}_2$. How do Z-strings distinguish these states?

**Problem 8**: Finite Temperature
At temperature $T$, the probability of having $n$ e-particle pairs is proportional to $e^{-2n/k_B T}$ times a degeneracy factor. Estimate the degeneracy for $n$ pairs on an $L \times L$ torus.

---

## Computational Lab: e-Particle Dynamics

```python
"""
Day 793 Computational Lab: Electric Charges (e-Particles)
Simulating Z-string operators and e-particle creation/manipulation
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.collections import PatchCollection

class ToricCodeEParticles:
    """
    Toric code simulator focused on e-particle (star violation) dynamics
    """

    def __init__(self, L):
        """
        Initialize L x L toric code
        Edges indexed as (x, y, direction) where direction is 'h' or 'v'
        """
        self.L = L
        self.n_edges = 2 * L * L
        self.n_vertices = L * L
        self.n_plaquettes = L * L

        # Edge indexing: (x, y, 'h') for horizontal, (x, y, 'v') for vertical
        self.edge_list = []
        for x in range(L):
            for y in range(L):
                self.edge_list.append((x, y, 'h'))
                self.edge_list.append((x, y, 'v'))
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}

        # State vector (computational basis coefficients)
        # Start in ground state (all +1 eigenstate of stabilizers)
        self.reset_to_ground_state()

    def reset_to_ground_state(self):
        """Reset to the ground state |+⟩^⊗n projected to code space"""
        # For simplicity, we work in the Z-basis and track stabilizers
        self.z_config = np.zeros(self.n_edges, dtype=int)  # 0 = |0⟩, 1 = |1⟩
        self.e_particles = set()  # Set of vertices with A_v = -1

    def get_incident_edges(self, vertex):
        """Get edges incident to vertex (x, y)"""
        x, y = vertex
        L = self.L
        edges = [
            (x, y, 'h'),           # right horizontal
            ((x-1) % L, y, 'h'),   # left horizontal
            (x, y, 'v'),           # up vertical
            (x, (y-1) % L, 'v'),   # down vertical
        ]
        return edges

    def compute_star_operator(self, vertex):
        """Compute A_v eigenvalue at vertex"""
        edges = self.get_incident_edges(vertex)
        product = 0
        for e in edges:
            idx = self.edge_to_idx[e]
            product += self.z_config[idx]
        return (-1) ** (product % 2)

    def apply_z_edge(self, edge):
        """Apply Z operator to a single edge"""
        idx = self.edge_to_idx[edge]
        self.z_config[idx] = 1 - self.z_config[idx]
        self._update_e_particles()

    def apply_z_string(self, path):
        """
        Apply Z-string operator along path
        path: list of vertices the string passes through
        """
        if len(path) < 2:
            return

        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i+1]
            edge = self._get_edge_between(v1, v2)
            if edge:
                self.apply_z_edge(edge)

    def _get_edge_between(self, v1, v2):
        """Get edge connecting two adjacent vertices"""
        x1, y1 = v1
        x2, y2 = v2
        L = self.L

        # Check horizontal adjacency
        if y1 == y2:
            if (x1 + 1) % L == x2:
                return (x1, y1, 'h')
            elif (x2 + 1) % L == x1:
                return (x2, y2, 'h')

        # Check vertical adjacency
        if x1 == x2:
            if (y1 + 1) % L == y2:
                return (x1, y1, 'v')
            elif (y2 + 1) % L == y1:
                return (x2, y2, 'v')

        return None

    def _update_e_particles(self):
        """Update set of e-particle locations"""
        self.e_particles = set()
        for x in range(self.L):
            for y in range(self.L):
                if self.compute_star_operator((x, y)) == -1:
                    self.e_particles.add((x, y))

    def get_energy(self):
        """Compute total energy relative to ground state"""
        # Each e-particle costs energy 2
        return 2 * len(self.e_particles)

    def visualize(self, title="Toric Code State", show_strings=None):
        """Visualize the current state with e-particle locations"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        L = self.L

        # Draw edges
        for e in self.edge_list:
            x, y, d = e
            idx = self.edge_to_idx[e]
            color = 'red' if self.z_config[idx] == 1 else 'lightgray'
            linewidth = 3 if self.z_config[idx] == 1 else 1

            if d == 'h':
                ax.plot([x, x+1], [y, y], color=color, linewidth=linewidth)
            else:
                ax.plot([x, x], [y, y+1], color=color, linewidth=linewidth)

        # Draw vertices
        for x in range(L):
            for y in range(L):
                if (x, y) in self.e_particles:
                    circle = Circle((x, y), 0.15, color='blue', zorder=5)
                    ax.add_patch(circle)
                    ax.annotate('e', (x, y), ha='center', va='center',
                               fontsize=12, color='white', fontweight='bold', zorder=6)
                else:
                    circle = Circle((x, y), 0.08, color='black', zorder=5)
                    ax.add_patch(circle)

        # Draw string path if provided
        if show_strings:
            for path, color in show_strings:
                xs = [v[0] for v in path]
                ys = [v[1] for v in path]
                ax.plot(xs, ys, color=color, linewidth=4, alpha=0.5,
                       linestyle='--', zorder=3)

        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-0.5, L + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nEnergy = {self.get_energy()}, e-particles = {len(self.e_particles)}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=3, label='Z error (flipped)'),
            plt.Line2D([0], [0], marker='o', color='blue', markersize=15,
                      label='e-particle', linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig, ax


def demonstrate_e_particle_creation():
    """Show how Z-strings create e-particle pairs"""
    print("=" * 60)
    print("Demonstration: e-Particle Creation with Z-Strings")
    print("=" * 60)

    tc = ToricCodeEParticles(L=5)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # State 1: Ground state
    ax = axes[0, 0]
    tc.reset_to_ground_state()
    tc.visualize("Ground State")
    plt.sca(ax)
    ax.set_title("1. Ground State\n(No e-particles)")
    for x in range(tc.L):
        for y in range(tc.L):
            circle = Circle((x, y), 0.08, color='black')
            ax.add_patch(circle)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')

    # State 2: Single Z operator
    ax = axes[0, 1]
    tc.reset_to_ground_state()
    tc.apply_z_edge((2, 2, 'h'))
    print(f"\nAfter Z on edge (2,2,'h'):")
    print(f"  e-particles at: {tc.e_particles}")
    print(f"  Energy: {tc.get_energy()}")
    tc.visualize("Single Z Error")

    # State 3: Z-string creating separated pair
    ax = axes[1, 0]
    tc.reset_to_ground_state()
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    tc.apply_z_string(path)
    print(f"\nAfter Z-string from (0,0) to (4,0):")
    print(f"  e-particles at: {tc.e_particles}")
    print(f"  Energy: {tc.get_energy()}")
    tc.visualize("Z-String (Horizontal)")

    # State 4: Two separate pairs
    ax = axes[1, 1]
    tc.reset_to_ground_state()
    path1 = [(1, 1), (1, 2), (1, 3)]
    path2 = [(3, 2), (4, 2)]
    tc.apply_z_string(path1)
    tc.apply_z_string(path2)
    print(f"\nAfter two Z-strings:")
    print(f"  e-particles at: {tc.e_particles}")
    print(f"  Energy: {tc.get_energy()}")
    tc.visualize("Two e-Particle Pairs")

    plt.savefig('e_particle_creation.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_e_particle_movement():
    """Show how extending Z-strings moves e-particles"""
    print("\n" + "=" * 60)
    print("Demonstration: e-Particle Movement")
    print("=" * 60)

    tc = ToricCodeEParticles(L=6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    positions = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]

    for i, end_pos in enumerate(positions[:5]):
        ax = axes[i // 3, i % 3]
        tc.reset_to_ground_state()

        # Create initial pair at (0, 2) and move one to end_pos
        path = [(0, 2)]
        for x in range(1, end_pos[0] + 1):
            path.append((x, 2))

        tc.apply_z_string(path)

        plt.sca(ax)
        for e in tc.edge_list:
            x, y, d = e
            idx = tc.edge_to_idx[e]
            color = 'red' if tc.z_config[idx] == 1 else 'lightgray'
            lw = 3 if tc.z_config[idx] == 1 else 1
            if d == 'h':
                ax.plot([x, x+1], [y, y], color=color, linewidth=lw)
            else:
                ax.plot([x, x], [y, y+1], color=color, linewidth=lw)

        for x in range(tc.L):
            for y in range(tc.L):
                if (x, y) in tc.e_particles:
                    circle = Circle((x, y), 0.2, color='blue', zorder=5)
                    ax.add_patch(circle)
                else:
                    circle = Circle((x, y), 0.08, color='black', zorder=5)
                    ax.add_patch(circle)

        ax.set_xlim(-0.5, tc.L + 0.5)
        ax.set_ylim(-0.5, tc.L + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {i+1}: e-particle at {end_pos}")

    # Last panel: show that different paths give same e-positions
    ax = axes[1, 2]
    tc.reset_to_ground_state()

    # Path 1: straight
    path1 = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    # Path 2: detour
    path2 = [(0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (4, 2)]

    # Apply path 1
    tc.apply_z_string(path1)
    e_particles_1 = tc.e_particles.copy()

    # Reset and apply path 2
    tc.reset_to_ground_state()
    tc.apply_z_string(path2)
    e_particles_2 = tc.e_particles.copy()

    plt.sca(ax)
    ax.text(0.5, 0.7, f"Path 1 (straight): e at {e_particles_1}",
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.5, f"Path 2 (detour): e at {e_particles_2}",
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.3, f"Same endpoints = same physics!",
            transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Path Independence")

    plt.tight_layout()
    plt.savefig('e_particle_movement.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nKey insight: Different paths with same endpoints create")
    print("e-particles at the same locations!")


def demonstrate_pair_constraint():
    """Show that e-particles always come in pairs"""
    print("\n" + "=" * 60)
    print("Demonstration: e-Particles Always Come in Pairs")
    print("=" * 60)

    tc = ToricCodeEParticles(L=4)

    # Count e-particles for random Z-string configurations
    pair_counts = []

    np.random.seed(42)
    for trial in range(100):
        tc.reset_to_ground_state()

        # Apply random Z operators
        n_errors = np.random.randint(1, 10)
        for _ in range(n_errors):
            edge = tc.edge_list[np.random.randint(tc.n_edges)]
            tc.apply_z_edge(edge)

        pair_counts.append(len(tc.e_particles))

    # All should be even
    all_even = all(n % 2 == 0 for n in pair_counts)

    print(f"Tested {len(pair_counts)} random configurations")
    print(f"All have even number of e-particles: {all_even}")
    print(f"Distribution of e-particle counts: {np.bincount(pair_counts)}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pair_counts, bins=range(max(pair_counts)+2), align='left',
            color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of e-particles', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('e-Particle Count Distribution (All Even!)', fontsize=16)
    ax.set_xticks(range(0, max(pair_counts)+1, 2))

    plt.tight_layout()
    plt.savefig('e_particle_pairs.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_syndrome_measurement():
    """Simulate syndrome measurement and error correction"""
    print("\n" + "=" * 60)
    print("Demonstration: Syndrome Measurement and Correction")
    print("=" * 60)

    tc = ToricCodeEParticles(L=5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Step 1: Create error
    ax = axes[0]
    tc.reset_to_ground_state()
    error_path = [(1, 1), (2, 1), (2, 2), (3, 2)]
    tc.apply_z_string(error_path)

    plt.sca(ax)
    for e in tc.edge_list:
        x, y, d = e
        idx = tc.edge_to_idx[e]
        color = 'red' if tc.z_config[idx] == 1 else 'lightgray'
        lw = 3 if tc.z_config[idx] == 1 else 1
        if d == 'h':
            ax.plot([x, x+1], [y, y], color=color, linewidth=lw)
        else:
            ax.plot([x, x], [y, y+1], color=color, linewidth=lw)

    for x in range(tc.L):
        for y in range(tc.L):
            if (x, y) in tc.e_particles:
                circle = Circle((x, y), 0.2, color='blue', zorder=5)
                ax.add_patch(circle)
            else:
                circle = Circle((x, y), 0.08, color='black', zorder=5)
                ax.add_patch(circle)

    ax.set_xlim(-0.5, tc.L + 0.5)
    ax.set_ylim(-0.5, tc.L + 0.5)
    ax.set_aspect('equal')
    ax.set_title("1. Error Occurs\n(Unknown to us)")

    error_endpoints = list(tc.e_particles)
    print(f"Error created e-particles at: {error_endpoints}")

    # Step 2: Syndrome measurement
    ax = axes[1]
    syndrome = tc.e_particles.copy()

    plt.sca(ax)
    for e in tc.edge_list:
        x, y, d = e
        color = 'lightgray'
        if d == 'h':
            ax.plot([x, x+1], [y, y], color=color, linewidth=1)
        else:
            ax.plot([x, x], [y, y+1], color=color, linewidth=1)

    for x in range(tc.L):
        for y in range(tc.L):
            if (x, y) in syndrome:
                circle = Circle((x, y), 0.2, color='blue', zorder=5)
                ax.add_patch(circle)
                ax.annotate('!', (x, y), ha='center', va='center',
                           fontsize=14, color='white', fontweight='bold', zorder=6)
            else:
                circle = Circle((x, y), 0.08, color='black', zorder=5)
                ax.add_patch(circle)

    ax.set_xlim(-0.5, tc.L + 0.5)
    ax.set_ylim(-0.5, tc.L + 0.5)
    ax.set_aspect('equal')
    ax.set_title("2. Syndrome Measurement\n(e-particle locations revealed)")

    # Step 3: Apply correction
    ax = axes[2]

    # Simple correction: connect e-particles with shortest path
    if len(error_endpoints) == 2:
        v1, v2 = error_endpoints
        correction_path = []
        # Move horizontally then vertically
        x, y = v1
        while x != v2[0]:
            correction_path.append((x, y))
            x = (x + 1) % tc.L if (v2[0] - v1[0]) % tc.L <= tc.L // 2 else (x - 1) % tc.L
        while y != v2[1]:
            correction_path.append((x, y))
            y = (y + 1) % tc.L if (v2[1] - v1[1]) % tc.L <= tc.L // 2 else (y - 1) % tc.L
        correction_path.append(v2)

        tc.apply_z_string(correction_path)

    plt.sca(ax)
    for e in tc.edge_list:
        x, y, d = e
        idx = tc.edge_to_idx[e]
        color = 'green' if tc.z_config[idx] == 1 else 'lightgray'
        lw = 3 if tc.z_config[idx] == 1 else 1
        if d == 'h':
            ax.plot([x, x+1], [y, y], color=color, linewidth=lw)
        else:
            ax.plot([x, x], [y, y+1], color=color, linewidth=lw)

    for x in range(tc.L):
        for y in range(tc.L):
            if (x, y) in tc.e_particles:
                circle = Circle((x, y), 0.2, color='blue', zorder=5)
                ax.add_patch(circle)
            else:
                circle = Circle((x, y), 0.08, color='black', zorder=5)
                ax.add_patch(circle)

    ax.set_xlim(-0.5, tc.L + 0.5)
    ax.set_ylim(-0.5, tc.L + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"3. Correction Applied\n(Energy = {tc.get_energy()}, e-particles = {len(tc.e_particles)})")

    plt.tight_layout()
    plt.savefig('syndrome_correction.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"After correction: {len(tc.e_particles)} e-particles remain")


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Day 793: Electric Charges (e-Particles) - Computational Lab")
    print("=" * 60)

    demonstrate_e_particle_creation()
    demonstrate_e_particle_movement()
    demonstrate_pair_constraint()
    demonstrate_syndrome_measurement()

    print("\n" + "=" * 60)
    print("Key Insights from Lab:")
    print("=" * 60)
    print("1. Z-string operators create e-particle pairs at endpoints")
    print("2. e-particles can be moved by extending Z-strings")
    print("3. Different paths with same endpoints = same e-particle locations")
    print("4. e-particles ALWAYS come in pairs (parity constraint)")
    print("5. Syndrome measurement reveals e-locations but not error path")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Star operator | $A_v = \prod_{e \ni v} Z_e$ |
| e-particle definition | $A_v \|\psi\rangle = -\|\psi\rangle$ |
| Z-string operator | $S_Z(\gamma) = \prod_{e \in \gamma} Z_e$ |
| Energy per e-particle | $\Delta E = 2$ |
| Parity constraint | $\prod_v A_v = I \Rightarrow n_e \in 2\mathbb{Z}$ |
| Self-statistics | $\theta_{ee} = 0$ (bosonic) |

### Main Takeaways

1. **e-particles are star violations**: Vertices where $A_v = -1$ host e-particles
2. **Created by Z-strings**: The operator $S_Z(\gamma)$ creates e-particles at the endpoints of path $\gamma$
3. **Pair creation is mandatory**: The constraint $\prod_v A_v = I$ forces even parity
4. **Path independence**: Only endpoints matter; the string path is gauge-dependent
5. **Bosonic self-statistics**: Exchanging two e-particles gives phase +1
6. **Error correction connection**: e-particles form the Z-error syndrome

---

## Daily Checklist

### Morning Theory (3 hours)
- [ ] Master the star operator $A_v$ and its eigenvalues
- [ ] Derive how Z-strings create e-particle pairs
- [ ] Prove the pair parity constraint
- [ ] Understand path independence of e-particle positions

### Afternoon Problems (2.5 hours)
- [ ] Complete all Direct Application problems
- [ ] Work through at least 2 Intermediate problems
- [ ] Attempt at least 1 Challenging problem

### Evening Lab (1.5 hours)
- [ ] Run all simulation code
- [ ] Experiment with different Z-string configurations
- [ ] Implement your own error correction decoder

### Self-Assessment Questions
1. Why do e-particles always come in pairs?
2. How would you move an e-particle from vertex $v_1$ to $v_2$?
3. What is the energy cost of creating $n$ e-particle pairs?

---

## Preview: Day 794

Tomorrow we study **magnetic fluxes (m-particles)**: the dual excitations created by plaquette operator violations. We'll see how X-string operators create m-particles, explore their bosonic self-statistics, and understand how m-particles behave on the torus topology. This parallel treatment sets up Thursday's study of e-m mutual statistics.

---

*Day 793 of 2184 | Year 2, Month 29, Week 114 | Quantum Engineering PhD Curriculum*
