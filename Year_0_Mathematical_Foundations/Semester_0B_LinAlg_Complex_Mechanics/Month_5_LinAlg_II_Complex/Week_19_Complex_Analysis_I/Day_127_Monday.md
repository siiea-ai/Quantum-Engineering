# Day 127: Complex Numbers — The Foundation of Quantum Amplitudes

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Complex Number Foundations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Work fluently with complex arithmetic
2. Represent complex numbers in Cartesian and polar forms
3. Understand the geometry of the complex plane
4. Compute powers and roots of complex numbers
5. Connect complex numbers to quantum amplitudes
6. Visualize complex functions

---

## 📚 Required Reading

### Primary Text: Churchill & Brown, "Complex Variables and Applications"
- **Chapter 1**: Complex Numbers (pp. 1-30)

### Alternative: Needham, "Visual Complex Analysis"
- **Chapter 1**: Geometry and Complex Arithmetic

### Physics Connection
- **Sakurai, Chapter 1.2**: Ket space and bra space (complex amplitudes)

---

## 🎬 Video Resources

### 3Blue1Brown
- "What is Euler's formula actually saying?"
- "e^iπ in 3.14 minutes"

### Welch Labs
- "Imaginary Numbers are Real" series

---

## 📖 Core Content: Theory and Concepts

### 1. The Need for Complex Numbers

**Algebraic motivation:** Solve x² + 1 = 0
- No real solution exists
- Define i = √(-1), so i² = -1
- Now x = ±i are solutions

**Physical motivation (QM):**
- Quantum states are vectors in complex vector spaces
- Amplitudes ⟨ψ|φ⟩ are complex numbers
- Interference requires phase information

### 2. Definition and Arithmetic

**Definition:** A complex number z is:
$$z = x + iy \quad \text{where } x, y \in \mathbb{R}$$

- **x** = Re(z) = real part
- **y** = Im(z) = imaginary part

**Arithmetic:**
| Operation | Formula |
|-----------|---------|
| Addition | (a+bi) + (c+di) = (a+c) + (b+d)i |
| Subtraction | (a+bi) - (c+di) = (a-c) + (b-d)i |
| Multiplication | (a+bi)(c+di) = (ac-bd) + (ad+bc)i |
| Division | $\frac{a+bi}{c+di} = \frac{(a+bi)(c-di)}{c²+d²}$ |

### 3. Complex Conjugate

**Definition:**
$$\bar{z} = z^* = x - iy$$

**Properties:**
| Property | Formula |
|----------|---------|
| Double conjugate | $\overline{\bar{z}} = z$ |
| Sum | $\overline{z+w} = \bar{z} + \bar{w}$ |
| Product | $\overline{zw} = \bar{z}\bar{w}$ |
| Quotient | $\overline{z/w} = \bar{z}/\bar{w}$ |
| Real part | Re(z) = (z + z̄)/2 |
| Imaginary part | Im(z) = (z - z̄)/(2i) |

### 4. Modulus (Absolute Value)

**Definition:**
$$|z| = \sqrt{x^2 + y^2} = \sqrt{z\bar{z}}$$

**Properties:**
| Property | Formula |
|----------|---------|
| Non-negative | \|z\| ≥ 0, with \|z\| = 0 ⟺ z = 0 |
| Product | \|zw\| = \|z\|\|w\| |
| Quotient | \|z/w\| = \|z\|/\|w\| |
| Triangle inequality | \|z + w\| ≤ \|z\| + \|w\| |
| Reverse triangle | \|\|z\| - \|w\|\| ≤ \|z - w\| |

### 5. The Complex Plane (Argand Diagram)

**Representation:**
- Horizontal axis: Real part (Re)
- Vertical axis: Imaginary part (Im)
- Point z = x + iy located at (x, y)

```
        Im
         ↑
         │    • z = x + iy
         │   /|
         │  / |
         │ /  | y
         │/θ  |
    ─────┼────┼───→ Re
         │    x
```

### 6. Polar Form

**Conversion:**
$$z = x + iy = r(\cos\theta + i\sin\theta) = re^{i\theta}$$

where:
- **r** = |z| = √(x² + y²) (modulus)
- **θ** = arg(z) = arctan(y/x) (argument, with quadrant adjustment)

**Euler's Formula:**
$$\boxed{e^{i\theta} = \cos\theta + i\sin\theta}$$

**Key special cases:**
| θ | e^(iθ) |
|---|--------|
| 0 | 1 |
| π/2 | i |
| π | -1 |
| 3π/2 | -i |
| 2π | 1 |

### 7. Multiplication in Polar Form

$$z_1 z_2 = r_1 e^{i\theta_1} \cdot r_2 e^{i\theta_2} = r_1 r_2 e^{i(\theta_1 + \theta_2)}$$

**Geometric interpretation:**
- Multiply moduli
- Add arguments (angles)

### 8. Powers and Roots

**De Moivre's Theorem:**
$$z^n = r^n e^{in\theta} = r^n(\cos n\theta + i\sin n\theta)$$

**nth Roots:**
$$z^{1/n} = r^{1/n} e^{i(\theta + 2\pi k)/n}, \quad k = 0, 1, ..., n-1$$

There are exactly n distinct nth roots, equally spaced on a circle of radius r^(1/n).

---

## 🔬 Quantum Mechanics Connection

### Complex Amplitudes

**Quantum states:** |ψ⟩ = Σᵢ cᵢ|i⟩ where cᵢ ∈ ℂ

**Probability:** P(i) = |cᵢ|² = cᵢc̄ᵢ

**Phase matters for interference:**
|ψ⟩ = (|0⟩ + e^(iφ)|1⟩)/√2

Measurement probabilities don't depend on φ, but interference does!

### Inner Products

**Bra-ket:** ⟨φ|ψ⟩ ∈ ℂ

**Properties:**
- ⟨φ|ψ⟩ = ⟨ψ|φ⟩* (conjugate symmetry)
- ⟨ψ|ψ⟩ ≥ 0 (positive definite)

### Unitary Operations

**Unitary matrix:** U†U = I

**Preserves norms:** |Uz|² = |z|²

The complex structure is essential — real matrices cannot represent all unitary operations!

### Phase in Quantum Mechanics

**Global phase:** |ψ⟩ and e^(iφ)|ψ⟩ represent the same physical state
**Relative phase:** |0⟩ + |1⟩ vs |0⟩ + e^(iφ)|1⟩ are different states!

This is why ℂ (not ℝ) is fundamental to quantum mechanics.

---

## ✏️ Worked Examples

### Example 1: Complex Arithmetic

Compute (3 + 2i)(1 - 4i) and express in Cartesian form.

**Solution:**
$$(3 + 2i)(1 - 4i) = 3(1) + 3(-4i) + 2i(1) + 2i(-4i)$$
$$= 3 - 12i + 2i - 8i^2 = 3 - 10i - 8(-1) = 3 - 10i + 8 = 11 - 10i$$

### Example 2: Division

Compute (2 + 3i)/(1 - i).

**Solution:**
$$\frac{2 + 3i}{1 - i} = \frac{(2 + 3i)(1 + i)}{(1 - i)(1 + i)} = \frac{2 + 2i + 3i + 3i^2}{1 - i^2}$$
$$= \frac{2 + 5i - 3}{1 + 1} = \frac{-1 + 5i}{2} = -\frac{1}{2} + \frac{5}{2}i$$

### Example 3: Polar Form

Convert z = -1 + i to polar form.

**Solution:**
$$r = |z| = \sqrt{(-1)^2 + 1^2} = \sqrt{2}$$
$$\theta = \arctan\left(\frac{1}{-1}\right) = \arctan(-1)$$

Since z is in second quadrant: θ = π - π/4 = 3π/4

$$z = \sqrt{2}e^{i(3\pi/4)} = \sqrt{2}\left(\cos\frac{3\pi}{4} + i\sin\frac{3\pi}{4}\right)$$

### Example 4: Powers Using De Moivre

Compute (1 + i)^8.

**Solution:**
First convert to polar: 1 + i = √2 e^(iπ/4)

$$(1 + i)^8 = (\sqrt{2})^8 e^{i(8 \cdot \pi/4)} = 16 e^{i(2\pi)} = 16 \cdot 1 = 16$$

### Example 5: Finding Roots

Find all cube roots of 8.

**Solution:**
8 = 8e^(i·0) = 8e^(i·2πk) for any integer k

$$8^{1/3} = 2e^{i(2\pi k/3)}, \quad k = 0, 1, 2$$

- k = 0: 2e^0 = 2
- k = 1: 2e^(i·2π/3) = 2(-1/2 + i√3/2) = -1 + i√3
- k = 2: 2e^(i·4π/3) = 2(-1/2 - i√3/2) = -1 - i√3

**Check:** (-1 + i√3)³ = ... = 8 ✓

### Example 6: Quantum Amplitude

A qubit is in state |ψ⟩ = (1 + i)|0⟩/√2 + (1 - i)|1⟩/√2. Find P(0) and P(1).

**Solution:**
Normalize: |ψ⟩ = [(1+i)/√2]|0⟩/√2 + [(1-i)/√2]|1⟩/√2

Wait, let me recalculate normalization:
|1+i|² = 2, |1-i|² = 2
Total: 2/2 + 2/2 = 2 ≠ 1

Need to normalize: N² = 2, so N = √2

|ψ⟩ = (1+i)|0⟩/2 + (1-i)|1⟩/2

P(0) = |(1+i)/2|² = 2/4 = 1/2
P(1) = |(1-i)/2|² = 2/4 = 1/2

---

## 📝 Practice Problems

### Level 1: Basic Arithmetic
1. Compute (2 - 3i) + (4 + 5i).
2. Compute (1 + i)(1 - i).
3. Find the conjugate and modulus of z = 3 - 4i.
4. Compute |2 + 2i|.

### Level 2: Division and Polar Form
5. Compute (3 + 4i)/(1 + 2i).
6. Convert z = -2 - 2i to polar form.
7. Convert z = 3e^(iπ/3) to Cartesian form.
8. Verify Euler's formula for θ = π/4.

### Level 3: Powers and Roots
9. Compute (1 - i)^6 using De Moivre's theorem.
10. Find all fourth roots of -16.
11. Find all solutions to z³ = -8i.
12. Compute i^i (hint: i = e^(iπ/2)).

### Level 4: Proofs and Theory
13. Prove |z₁z₂| = |z₁||z₂| using z = re^(iθ).
14. Prove the triangle inequality |z + w| ≤ |z| + |w|.
15. Show that the nth roots of unity sum to zero.

---

## 💻 Evening Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Complex Number Operations
# ============================================

def complex_info(z, name="z"):
    """Display information about a complex number"""
    print(f"{name} = {z}")
    print(f"  Real part: {z.real}")
    print(f"  Imaginary part: {z.imag}")
    print(f"  Conjugate: {np.conj(z)}")
    print(f"  Modulus: {np.abs(z):.4f}")
    print(f"  Argument: {np.angle(z):.4f} rad = {np.degrees(np.angle(z)):.2f}°")
    print(f"  Polar form: {np.abs(z):.4f} * exp({np.angle(z):.4f}i)")
    print()

# Basic examples
z1 = 3 + 4j
z2 = 1 - 2j

print("=== Complex Number Basics ===\n")
complex_info(z1, "z1")
complex_info(z2, "z2")

print(f"z1 + z2 = {z1 + z2}")
print(f"z1 * z2 = {z1 * z2}")
print(f"z1 / z2 = {z1 / z2}")
print(f"|z1 * z2| = {np.abs(z1 * z2):.4f}")
print(f"|z1| * |z2| = {np.abs(z1) * np.abs(z2):.4f}")

# ============================================
# Visualization: Complex Plane
# ============================================

def plot_complex_numbers(numbers, labels=None, title="Complex Plane"):
    """Plot complex numbers on the Argand diagram"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    
    # Plot axes
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot numbers
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(numbers)))
    for i, z in enumerate(numbers):
        label = labels[i] if labels else f"z{i+1}"
        ax.plot([0, z.real], [0, z.imag], color=colors[i], linewidth=2)
        ax.scatter([z.real], [z.imag], color=colors[i], s=100, zorder=5)
        ax.annotate(label, (z.real, z.imag), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12)
    
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax

# Plot various complex numbers
numbers = [1+0j, 0+1j, -1+0j, 0-1j, 1+1j, 2-1j, -1+2j]
labels = ['1', 'i', '-1', '-i', '1+i', '2-i', '-1+2i']

fig, ax = plot_complex_numbers(numbers, labels)
plt.savefig('complex_plane_basics.png', dpi=150)
plt.show()

# ============================================
# Euler's Formula Visualization
# ============================================

def visualize_euler():
    """Visualize e^(iθ) on unit circle"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Unit circle with key points
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    
    # Mark special angles
    special_angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    special_labels = ['1', 'e^{iπ/4}', 'i', '-1', '-i']
    
    for angle, label in zip(special_angles, special_labels):
        z = np.exp(1j * angle)
        ax.scatter([z.real], [z.imag], s=100, zorder=5)
        ax.annotate(f'${label}$', (z.real, z.imag), xytext=(10, 10),
                   textcoords='offset points', fontsize=14)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title('Unit Circle: $e^{i\\theta}$', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Right: Real and imaginary parts vs theta
    ax = axes[1]
    theta = np.linspace(0, 4*np.pi, 200)
    z = np.exp(1j * theta)
    
    ax.plot(theta, z.real, 'b-', linewidth=2, label='cos(θ) = Re($e^{iθ}$)')
    ax.plot(theta, z.imag, 'r-', linewidth=2, label='sin(θ) = Im($e^{iθ}$)')
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Euler\'s Formula: $e^{i\\theta} = \\cos\\theta + i\\sin\\theta$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi, 7*np.pi/2, 4*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π', '5π/2', '3π', '7π/2', '4π'])
    
    plt.tight_layout()
    plt.savefig('euler_formula.png', dpi=150)
    plt.show()

visualize_euler()

# ============================================
# Roots of Unity
# ============================================

def plot_roots_of_unity(n):
    """Plot the nth roots of unity"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Roots
    roots = [np.exp(2j * np.pi * k / n) for k in range(n)]
    
    for k, root in enumerate(roots):
        ax.scatter([root.real], [root.imag], s=100, c='red', zorder=5)
        ax.plot([0, root.real], [0, root.imag], 'b-', alpha=0.5)
        ax.annotate(f'$\\omega_{k}$', (root.real, root.imag), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12)
    
    # Connect roots to form polygon
    roots_closed = roots + [roots[0]]
    ax.plot([r.real for r in roots_closed], [r.imag for r in roots_closed], 
            'r-', linewidth=1.5)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title(f'{n}th Roots of Unity: $\\omega_k = e^{{2\\pi i k / {n}}}$', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    return fig, ax, roots

# Plot roots for n = 3, 4, 5, 6
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, n in zip(axes.flatten(), [3, 4, 5, 6]):
    plt.sca(ax)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    roots = [np.exp(2j * np.pi * k / n) for k in range(n)]
    for k, root in enumerate(roots):
        ax.scatter([root.real], [root.imag], s=80, c='red', zorder=5)
        ax.plot([0, root.real], [0, root.imag], 'b-', alpha=0.5)
    
    roots_closed = roots + [roots[0]]
    ax.plot([r.real for r in roots_closed], [r.imag for r in roots_closed], 'r-', linewidth=1.5)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'n = {n}', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('roots_of_unity.png', dpi=150)
plt.show()

# Verify sum of roots = 0
print("\n=== Sum of nth Roots of Unity ===")
for n in [3, 4, 5, 6, 7]:
    roots = [np.exp(2j * np.pi * k / n) for k in range(n)]
    total = sum(roots)
    print(f"n = {n}: Sum = {total:.6f} (should be 0)")

# ============================================
# Complex Multiplication Visualization
# ============================================

def visualize_multiplication():
    """Show how multiplication rotates and scales"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    z = 2 + 1j  # Original number
    w = 1 + 1j  # Multiplier
    
    # Original
    ax = axes[0]
    ax.scatter([z.real], [z.imag], s=100, c='blue', label=f'z = {z}')
    ax.plot([0, z.real], [0, z.imag], 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'z = {z}', fontsize=14)
    ax.legend()
    
    # Multiplier
    ax = axes[1]
    ax.scatter([w.real], [w.imag], s=100, c='green', label=f'w = {w}')
    ax.plot([0, w.real], [0, w.imag], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'w = {w}\n|w| = {np.abs(w):.2f}, arg(w) = {np.degrees(np.angle(w)):.1f}°', fontsize=14)
    ax.legend()
    
    # Product
    ax = axes[2]
    zw = z * w
    ax.scatter([z.real], [z.imag], s=100, c='blue', alpha=0.5, label=f'z = {z}')
    ax.plot([0, z.real], [0, z.imag], 'b--', linewidth=1, alpha=0.5)
    ax.scatter([zw.real], [zw.imag], s=100, c='red', label=f'zw = {zw}')
    ax.plot([0, zw.real], [0, zw.imag], 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'zw = {zw}\nScaled by {np.abs(w):.2f}, rotated {np.degrees(np.angle(w)):.1f}°', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('complex_multiplication.png', dpi=150)
    plt.show()

visualize_multiplication()

# ============================================
# Quantum Mechanics Connection
# ============================================

print("\n=== Quantum Mechanics: Complex Amplitudes ===")

# Qubit state with complex amplitudes
alpha = (1 + 1j) / 2
beta = (1 - 1j) / 2

print(f"State: |ψ⟩ = α|0⟩ + β|1⟩")
print(f"  α = {alpha}")
print(f"  β = {beta}")
print(f"  |α|² = {np.abs(alpha)**2:.4f}")
print(f"  |β|² = {np.abs(beta)**2:.4f}")
print(f"  Normalization: |α|² + |β|² = {np.abs(alpha)**2 + np.abs(beta)**2:.4f}")

# Relative phase
print(f"\nRelative phase:")
print(f"  arg(α) = {np.degrees(np.angle(alpha)):.2f}°")
print(f"  arg(β) = {np.degrees(np.angle(beta)):.2f}°")
print(f"  Relative phase = {np.degrees(np.angle(beta) - np.angle(alpha)):.2f}°")
```

---

## ✅ Daily Checklist

- [ ] Review complex number arithmetic
- [ ] Practice polar form conversions
- [ ] Master Euler's formula
- [ ] Compute powers using De Moivre's theorem
- [ ] Find all nth roots
- [ ] Connect to quantum amplitudes
- [ ] Complete computational lab
- [ ] Solve at least 8 practice problems

---

## 📓 Reflection Questions

1. Why do quantum mechanics require complex numbers, not just real numbers?

2. What is the geometric meaning of complex multiplication?

3. How does Euler's formula connect trigonometry to exponentials?

---

## 🔜 Preview: Tomorrow

**Day 128: The Complex Plane — Topology and Geometry**
- Extended complex plane (Riemann sphere)
- Neighborhoods and open sets
- Limits and continuity
- Paths and curves
- QM Connection: Path integrals preview

---

*"The shortest path between two truths in the real domain passes through the complex domain."*
— Jacques Hadamard
