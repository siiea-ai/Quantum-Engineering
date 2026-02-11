# Day 275: Electromagnetism Visualizations

## Schedule Overview
**Date**: Week 40, Day 2 (Tuesday)
**Duration**: 7 hours
**Theme**: Numerical Computation and Visualization of Electromagnetic Fields

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Electric field computation, Coulomb's law |
| Afternoon | 2.5 hours | Magnetic fields, Biot-Savart law |
| Evening | 1.5 hours | Computational lab: Electromagnetic wave propagation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Compute electric fields from discrete and continuous charge distributions
2. Visualize vector fields using streamlines and quiver plots
3. Calculate magnetic fields using the Biot-Savart law
4. Simulate electromagnetic wave propagation
5. Connect classical electromagnetism to quantum electrodynamics

---

## Core Content

### 1. Electric Field Visualization

For point charges, the electric field is:
$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_i \frac{q_i(\mathbf{r} - \mathbf{r}_i)}{|\mathbf{r} - \mathbf{r}_i|^3}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class ElectricField:
    """Compute and visualize electric fields from point charges."""

    def __init__(self, k=1.0):
        """Initialize with Coulomb constant k = 1/(4πε₀)."""
        self.k = k
        self.charges = []

    def add_charge(self, q, position):
        """Add a point charge."""
        self.charges.append({'q': q, 'pos': np.array(position)})

    def field_at(self, r):
        """Compute electric field at position r."""
        r = np.array(r)
        E = np.zeros(2)

        for charge in self.charges:
            r_vec = r - charge['pos']
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 1e-10:  # Avoid singularity
                E += self.k * charge['q'] * r_vec / r_mag**3

        return E

    def compute_field_grid(self, x_range, y_range, n_points=50):
        """Compute field on a 2D grid."""
        x = np.linspace(*x_range, n_points)
        y = np.linspace(*y_range, n_points)
        X, Y = np.meshgrid(x, y)

        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

        for i in range(n_points):
            for j in range(n_points):
                E = self.field_at([X[i, j], Y[i, j]])
                Ex[i, j] = E[0]
                Ey[i, j] = E[1]

        return X, Y, Ex, Ey

    def potential_at(self, r):
        """Compute electric potential at position r."""
        r = np.array(r)
        V = 0

        for charge in self.charges:
            r_mag = np.linalg.norm(r - charge['pos'])
            if r_mag > 1e-10:
                V += self.k * charge['q'] / r_mag

        return V

    def visualize(self, x_range=(-5, 5), y_range=(-5, 5), n_points=30):
        """Create comprehensive field visualization."""
        X, Y, Ex, Ey = self.compute_field_grid(x_range, y_range, n_points)
        E_mag = np.sqrt(Ex**2 + Ey**2)

        # Normalize for quiver plot
        Ex_norm = Ex / (E_mag + 1e-10)
        Ey_norm = Ey / (E_mag + 1e-10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Vector field with streamlines
        ax1 = axes[0]
        ax1.streamplot(X, Y, Ex, Ey, color=np.log10(E_mag + 1e-10),
                      cmap='plasma', density=1.5, linewidth=1)

        # Add charge markers
        for charge in self.charges:
            color = 'red' if charge['q'] > 0 else 'blue'
            size = min(abs(charge['q']) * 50, 300)
            ax1.scatter(*charge['pos'], c=color, s=size, zorder=5, edgecolor='black')
            ax1.annotate(f"{'+' if charge['q'] > 0 else ''}{charge['q']:.1f}",
                        charge['pos'], textcoords="offset points",
                        xytext=(10, 10), fontsize=10)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Electric Field Lines')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_aspect('equal')

        # Potential contours
        ax2 = axes[1]
        x_fine = np.linspace(*x_range, 100)
        y_fine = np.linspace(*y_range, 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        V = np.zeros_like(X_fine)

        for i in range(len(x_fine)):
            for j in range(len(y_fine)):
                V[j, i] = self.potential_at([x_fine[i], y_fine[j]])

        # Clip potential for visualization
        V_clipped = np.clip(V, -10, 10)

        cs = ax2.contourf(X_fine, Y_fine, V_clipped, levels=30, cmap='RdBu_r')
        ax2.contour(X_fine, Y_fine, V_clipped, levels=15, colors='black',
                   linewidths=0.5, alpha=0.5)
        plt.colorbar(cs, ax=ax2, label='Potential V')

        for charge in self.charges:
            color = 'red' if charge['q'] > 0 else 'blue'
            ax2.scatter(*charge['pos'], c=color, s=100, zorder=5, edgecolor='black')

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Electric Potential')
        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_aspect('equal')

        plt.tight_layout()
        return fig

# Example: Dipole
field = ElectricField()
field.add_charge(1.0, [-1, 0])
field.add_charge(-1.0, [1, 0])
fig = field.visualize()
plt.savefig('electric_dipole.png', dpi=150, bbox_inches='tight')
plt.show()

# Quadrupole
field2 = ElectricField()
field2.add_charge(1.0, [1, 1])
field2.add_charge(1.0, [-1, -1])
field2.add_charge(-1.0, [1, -1])
field2.add_charge(-1.0, [-1, 1])
fig2 = field2.visualize()
plt.savefig('electric_quadrupole.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2. Magnetic Field Visualization

The Biot-Savart law for current elements:
$$d\mathbf{B} = \frac{\mu_0}{4\pi} \frac{I\,d\mathbf{l} \times \hat{\mathbf{r}}}{r^2}$$

```python
class MagneticField:
    """Compute magnetic fields from current distributions."""

    def __init__(self, mu_0=1.0):
        self.mu_0 = mu_0
        self.current_elements = []

    def add_wire_segment(self, I, start, end, n_segments=10):
        """Add a straight wire segment."""
        start, end = np.array(start), np.array(end)
        dl = (end - start) / n_segments

        for i in range(n_segments):
            pos = start + (i + 0.5) * dl
            self.current_elements.append({'I': I, 'dl': dl, 'pos': pos})

    def add_circular_loop(self, I, center, radius, normal='z', n_segments=50):
        """Add a circular current loop."""
        center = np.array(center)
        theta = np.linspace(0, 2*np.pi, n_segments + 1)

        for i in range(n_segments):
            t_mid = (theta[i] + theta[i+1]) / 2
            if normal == 'z':
                pos = center + radius * np.array([np.cos(t_mid), np.sin(t_mid), 0])
                dl = radius * (theta[i+1] - theta[i]) * np.array([-np.sin(t_mid), np.cos(t_mid), 0])
            elif normal == 'x':
                pos = center + radius * np.array([0, np.cos(t_mid), np.sin(t_mid)])
                dl = radius * (theta[i+1] - theta[i]) * np.array([0, -np.sin(t_mid), np.cos(t_mid)])
            elif normal == 'y':
                pos = center + radius * np.array([np.cos(t_mid), 0, np.sin(t_mid)])
                dl = radius * (theta[i+1] - theta[i]) * np.array([-np.sin(t_mid), 0, np.cos(t_mid)])

            self.current_elements.append({'I': I, 'dl': dl, 'pos': pos})

    def field_at(self, r):
        """Compute magnetic field at position r using Biot-Savart."""
        r = np.array(r)
        B = np.zeros(3)

        for element in self.current_elements:
            r_vec = r - element['pos']
            r_mag = np.linalg.norm(r_vec)

            if r_mag > 1e-10:
                dB = (self.mu_0 / (4 * np.pi)) * element['I'] * \
                     np.cross(element['dl'], r_vec) / r_mag**3
                B += dB

        return B

    def visualize_2d(self, plane='xz', slice_val=0, extent=(-3, 3, -3, 3), n_points=30):
        """Visualize field in a 2D plane."""
        x = np.linspace(extent[0], extent[1], n_points)
        y = np.linspace(extent[2], extent[3], n_points)
        X, Y = np.meshgrid(x, y)

        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        B_mag = np.zeros_like(X)

        for i in range(n_points):
            for j in range(n_points):
                if plane == 'xz':
                    r = np.array([X[i, j], slice_val, Y[i, j]])
                elif plane == 'xy':
                    r = np.array([X[i, j], Y[i, j], slice_val])
                elif plane == 'yz':
                    r = np.array([slice_val, X[i, j], Y[i, j]])

                B = self.field_at(r)

                if plane == 'xz':
                    Bx[i, j], By[i, j] = B[0], B[2]
                elif plane == 'xy':
                    Bx[i, j], By[i, j] = B[0], B[1]
                elif plane == 'yz':
                    Bx[i, j], By[i, j] = B[1], B[2]

                B_mag[i, j] = np.linalg.norm([Bx[i, j], By[i, j]])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Streamplot
        ax.streamplot(X, Y, Bx, By, color=np.log10(B_mag + 1e-10),
                     cmap='viridis', density=1.5, linewidth=1.5)

        # Show current direction
        for element in self.current_elements:
            if plane == 'xz' and abs(element['pos'][1] - slice_val) < 0.5:
                ax.plot(element['pos'][0], element['pos'][2], 'ro', markersize=3)

        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_title(f'Magnetic Field in {plane.upper()} Plane')
        ax.set_aspect('equal')

        return fig

# Magnetic dipole (current loop)
mag_field = MagneticField()
mag_field.add_circular_loop(I=1.0, center=[0, 0, 0], radius=1.0, normal='y', n_segments=50)
fig = mag_field.visualize_2d(plane='xz', slice_val=0, extent=(-4, 4, -4, 4))
plt.savefig('magnetic_dipole.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3. Electromagnetic Wave Propagation

The wave equation from Maxwell's equations:
$$\frac{\partial^2 E}{\partial t^2} = c^2 \frac{\partial^2 E}{\partial x^2}$$

```python
def simulate_em_wave():
    """Simulate 1D electromagnetic wave propagation."""

    # Grid
    L = 10.0
    N = 500
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    c = 1.0  # Speed of light
    dt = 0.5 * dx / c  # CFL condition
    n_steps = 500

    # Fields: E_y and B_z
    E = np.zeros(N)
    B = np.zeros(N)

    # Initial condition: Gaussian pulse
    E = np.exp(-(x - L/4)**2 / 0.5)

    # Storage for animation
    E_history = [E.copy()]

    # FDTD update (Yee scheme)
    for step in range(n_steps):
        # Update B (half-step behind E)
        B[:-1] += (c * dt / dx) * (E[1:] - E[:-1])

        # Boundary conditions (absorbing)
        B[-1] = B[-2]

        # Update E
        E[1:] += (c * dt / dx) * (B[1:] - B[:-1])

        # Source or boundary
        E[0] = 0  # Fixed boundary

        if step % 5 == 0:
            E_history.append(E.copy())

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Final state
    axes[0].plot(x, E_history[-1], 'b-', linewidth=2, label='E field')
    axes[0].plot(x, E_history[0], 'r--', alpha=0.5, label='Initial')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('E')
    axes[0].legend()
    axes[0].set_title('Electromagnetic Wave Propagation')
    axes[0].grid(True, alpha=0.3)

    # Space-time diagram
    E_array = np.array(E_history)
    im = axes[1].imshow(E_array, aspect='auto', origin='lower',
                        extent=[0, L, 0, len(E_history)*5*dt],
                        cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Time')
    axes[1].set_title('Space-Time Diagram')
    plt.colorbar(im, ax=axes[1], label='E field')

    plt.tight_layout()
    plt.savefig('em_wave_propagation.png', dpi=150)
    plt.show()

    return E_history

E_history = simulate_em_wave()
```

---

## Quantum Mechanics Connection

### From Classical E&M to Quantum Electrodynamics

| Classical EM | Quantum Electrodynamics |
|--------------|------------------------|
| Electric field E | Photon annihilation/creation operators |
| Magnetic field B | Quantized vector potential |
| EM wave | Photon (quantum of light) |
| Coulomb potential | Virtual photon exchange |

```python
def classical_to_quantum_em():
    """Illustrate classical-quantum correspondence in E&M."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Classical wave packet
    x = np.linspace(0, 20, 1000)
    omega = 5
    k = omega  # c = 1
    envelope = np.exp(-(x-10)**2/4)
    E_classical = envelope * np.sin(k*x)

    axes[0, 0].plot(x, E_classical, 'b-', linewidth=1.5)
    axes[0, 0].fill_between(x, E_classical, alpha=0.3)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('E(x)')
    axes[0, 0].set_title('Classical Wave Packet')

    # Panel 2: Quantized energy levels (photon number states)
    n_photons = np.arange(0, 6)
    E_n = (n_photons + 0.5)  # ℏω units

    axes[0, 1].barh(n_photons, E_n, height=0.6, color='orange')
    for n, E in zip(n_photons, E_n):
        axes[0, 1].text(E + 0.1, n, f'|{n}⟩: E={E:.1f}ℏω', va='center')
    axes[0, 1].set_xlabel('Energy (ℏω)')
    axes[0, 1].set_ylabel('Photon number n')
    axes[0, 1].set_title('Quantized EM Field Energy')

    # Panel 3: Coherent state |α⟩ ~ classical
    theta = np.linspace(0, 2*np.pi, 100)
    alpha = 3  # Coherent amplitude

    # Phase space representation
    axes[1, 0].plot(alpha + 0.5*np.cos(theta), 0.5*np.sin(theta), 'b-', linewidth=2)
    axes[1, 0].scatter([alpha], [0], s=100, c='red', zorder=5)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Re(α) ~ E')
    axes[1, 0].set_ylabel('Im(α) ~ B')
    axes[1, 0].set_title('Coherent State in Phase Space')
    axes[1, 0].set_xlim(-1, 5)
    axes[1, 0].set_ylim(-2, 2)
    axes[1, 0].set_aspect('equal')

    # Panel 4: Photon number distribution for coherent state
    n = np.arange(0, 15)
    P_n = np.exp(-alpha**2) * alpha**(2*n) / np.array([np.math.factorial(k) for k in n])

    axes[1, 1].bar(n, P_n, color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Photon number n')
    axes[1, 1].set_ylabel('P(n)')
    axes[1, 1].set_title(f'Coherent State |α={alpha}⟩: Poisson Distribution')
    axes[1, 1].axvline(alpha**2, color='red', linestyle='--', label=f'⟨n⟩ = |α|² = {alpha**2}')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('classical_quantum_em.png', dpi=150)
    plt.show()

classical_to_quantum_em()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Point Charges**: Visualize the field of three charges at vertices of an equilateral triangle.

2. **Parallel Wires**: Compute and plot the magnetic field between two parallel current-carrying wires.

3. **Standing Wave**: Simulate an EM wave with reflecting boundaries to create a standing wave pattern.

### Level 2: Intermediate

4. **Charged Ring**: Compute the electric field along the axis of a uniformly charged ring.

5. **Solenoid**: Model a solenoid using multiple circular current loops and visualize the interior field.

6. **Wave Reflection**: Simulate wave reflection at a dielectric interface (different wave speeds).

### Level 3: Challenging

7. **Radiation Pattern**: Compute and visualize the radiation pattern of an oscillating dipole.

8. **Helmholtz Coils**: Design Helmholtz coils for uniform magnetic field and verify numerically.

9. **Casimir Effect Analogy**: Relate the quantized modes between conducting plates to the classical boundary problem.

---

## Summary

### Key Equations

$$\boxed{\mathbf{E} = \frac{1}{4\pi\epsilon_0} \sum_i \frac{q_i \hat{\mathbf{r}}_i}{r_i^2}}$$

$$\boxed{\mathbf{B} = \frac{\mu_0}{4\pi} \int \frac{I\, d\mathbf{l} \times \hat{\mathbf{r}}}{r^2}}$$

$$\boxed{\frac{\partial^2 E}{\partial t^2} = c^2 \nabla^2 E}$$

---

## Daily Checklist

- [ ] Computed electric fields from charge distributions
- [ ] Created field line and potential visualizations
- [ ] Implemented Biot-Savart law for magnetic fields
- [ ] Simulated electromagnetic wave propagation
- [ ] Connected to quantum electrodynamics concepts
- [ ] Completed practice problems

---

## Preview of Day 276

Tomorrow: **Wave Equation Simulations**
- Numerical schemes for wave equation (FTCS, Lax)
- Stability analysis (CFL condition)
- Standing waves and normal modes
- Connection to quantum wave mechanics
