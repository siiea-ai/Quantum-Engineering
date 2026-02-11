# Day 218: Einstein's Postulates and Lorentz Transformations

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Foundations of Special Relativity |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Lorentz Transformations and Consequences |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 218, you will be able to:

1. State Einstein's two postulates and explain their physical significance
2. Derive the Lorentz transformation equations from first principles
3. Calculate time dilation and length contraction effects
4. Apply the relativistic velocity addition formula
5. Understand the relativity of simultaneity
6. Connect to the incompatibility of Maxwell's equations with Galilean relativity

---

## Core Content

### 1. The Crisis in Classical Physics

**The problem with Maxwell's equations:** Maxwell's equations predict electromagnetic waves traveling at speed $c = 1/\sqrt{\mu_0\epsilon_0}$. But velocity relative to what?

**The luminiferous ether:** 19th-century physicists proposed a medium (ether) through which light propagates. The Michelson-Morley experiment (1887) found no evidence for Earth's motion through this ether.

**The incompatibility:** Maxwell's equations are not invariant under Galilean transformations:
$$x' = x - vt, \quad y' = y, \quad z' = z, \quad t' = t$$

If we transform the wave equation $\nabla^2\phi - \frac{1}{c^2}\frac{\partial^2\phi}{\partial t^2} = 0$, we get additional terms involving $v$.

### 2. Einstein's Postulates (1905)

**Postulate 1 (Principle of Relativity):**
> *The laws of physics are the same in all inertial reference frames.*

This is an extension of Galilean relativity to include electromagnetism.

**Postulate 2 (Constancy of the Speed of Light):**
> *The speed of light in vacuum has the same value $c$ in all inertial reference frames, independent of the motion of the source or observer.*

$$\boxed{c = 299,792,458 \text{ m/s (exact, by definition)}}$$

These two innocent-seeming postulates revolutionize our understanding of space and time.

### 3. Derivation of Lorentz Transformations

**Setup:** Frame $S'$ moves with velocity $v$ along the positive $x$-axis relative to frame $S$. At $t = t' = 0$, the origins coincide.

**Assumption:** The transformation must be linear (to preserve uniformity of space and time):
$$x' = Ax + Bt, \quad t' = Cx + Dt$$

**Condition 1:** Origin of $S'$ (where $x' = 0$) moves at $x = vt$ in $S$:
$$0 = A(vt) + Bt \Rightarrow B = -Av$$

**Condition 2:** Origin of $S$ (where $x = 0$) moves at $x' = -vt'$ in $S'$:
$$-vt' = 0 + B \cdot t \Rightarrow B = -v \cdot (Dt') / t'$$

**Condition 3:** Speed of light is $c$ in both frames. A light pulse emitted at origin:
- In $S$: $x = ct$
- In $S'$: $x' = ct'$

Substituting:
$$ct' = A(ct) - Avt = Act - Avt$$
$$t' = Ct + Dt \cdot \frac{ct - vt}{ct} = Ct + D(t - vt/c)$$

Wait, let me use a cleaner derivation using the invariance of the speed of light directly.

**From $x = ct$ and $x' = ct'$:**
$$x'^2 - c^2t'^2 = x^2 - c^2t^2 = 0$$

More generally, the spacetime interval must be preserved:
$$s^2 = -c^2t^2 + x^2 = -c^2t'^2 + x'^2 = s'^2$$

This leads to:

$$\boxed{x' = \gamma(x - vt)}$$
$$\boxed{t' = \gamma\left(t - \frac{vx}{c^2}\right)}$$

where the **Lorentz factor** is:

$$\boxed{\gamma = \frac{1}{\sqrt{1 - v^2/c^2}} = \frac{1}{\sqrt{1 - \beta^2}}}$$

with $\beta = v/c$.

**Properties of $\gamma$:**
- $\gamma \geq 1$ always
- $\gamma = 1$ when $v = 0$
- $\gamma \to \infty$ as $v \to c$
- $\gamma \approx 1 + \frac{1}{2}\beta^2$ for $v \ll c$

### 4. The Full Lorentz Transformation

For motion along the $x$-axis:

$$\begin{pmatrix} ct' \\ x' \\ y' \\ z' \end{pmatrix} = \begin{pmatrix} \gamma & -\beta\gamma & 0 & 0 \\ -\beta\gamma & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} ct \\ x \\ y \\ z \end{pmatrix}$$

**Inverse transformation** (replace $v \to -v$):
$$x = \gamma(x' + vt'), \quad t = \gamma\left(t' + \frac{vx'}{c^2}\right)$$

### 5. Time Dilation

**Proper time:** Time measured in the rest frame of a clock.

Consider a clock at rest in $S'$ at position $x' = 0$. Two events:
- Event 1: $t'_1$ at $x' = 0$
- Event 2: $t'_2$ at $x' = 0$

Proper time interval: $\Delta\tau = t'_2 - t'_1$

In frame $S$:
$$\Delta t = \gamma\left(\Delta t' + \frac{v \cdot 0}{c^2}\right) = \gamma\Delta\tau$$

$$\boxed{\Delta t = \gamma\Delta\tau}$$

**Moving clocks run slow!** The coordinate time $\Delta t$ is always longer than the proper time $\Delta\tau$.

**Example:** Muons created in the upper atmosphere have a lifetime of $\tau = 2.2$ μs at rest. Traveling at $v = 0.998c$:
$$\gamma = \frac{1}{\sqrt{1 - 0.998^2}} \approx 15.8$$

Laboratory lifetime: $\gamma\tau = 15.8 \times 2.2 = 34.8$ μs

They travel: $d = v \cdot \gamma\tau = 0.998c \times 34.8 \text{ μs} = 10.4$ km (enough to reach Earth's surface!)

### 6. Length Contraction

**Proper length:** Length measured in the rest frame of the object.

Consider a rod at rest in $S'$ with proper length $L_0 = x'_2 - x'_1$.

To measure length in $S$, we must determine both endpoints **simultaneously** in $S$ (at the same $t$).

Using the Lorentz transformation:
$$x'_2 - x'_1 = \gamma(x_2 - x_1) - \gamma v(t_2 - t_1) = \gamma(x_2 - x_1)$$

since $t_2 = t_1$ (simultaneous in $S$).

$$\boxed{L = \frac{L_0}{\gamma} = L_0\sqrt{1 - v^2/c^2}}$$

**Moving objects are contracted along the direction of motion!**

### 7. Relativity of Simultaneity

Events simultaneous in one frame are **not** simultaneous in another frame moving relative to it.

Two events at different positions $x_1$ and $x_2$ that are simultaneous in $S$ (i.e., $t_1 = t_2$):
$$t'_1 - t'_2 = \gamma\left(-\frac{v}{c^2}\right)(x_1 - x_2)$$

This is non-zero unless $x_1 = x_2$ or $v = 0$.

### 8. Relativistic Velocity Addition

If an object moves with velocity $u'$ in frame $S'$, what is its velocity $u$ in frame $S$?

Using $u = dx/dt$ and $u' = dx'/dt'$:

$$\boxed{u = \frac{u' + v}{1 + u'v/c^2}}$$

**Properties:**
- If $u' = c$, then $u = c$ (speed of light is invariant)
- If $u', v \ll c$, then $u \approx u' + v$ (Galilean addition recovered)
- $u < c$ if both $u', v < c$ (nothing exceeds $c$)

---

## Quantum Mechanics Connection

### Relativistic Energy and the Klein-Gordon Equation

In quantum mechanics, energy becomes an operator: $\hat{E} \to i\hbar\frac{\partial}{\partial t}$, $\hat{p} \to -i\hbar\nabla$.

The relativistic energy-momentum relation $E^2 = (pc)^2 + (m_0c^2)^2$ leads to:

$$\left(-\hbar^2\frac{\partial^2}{\partial t^2}\right)\psi = \left(-\hbar^2 c^2\nabla^2 + m_0^2c^4\right)\psi$$

This is the **Klein-Gordon equation**:

$$\boxed{\left(\frac{1}{c^2}\frac{\partial^2}{\partial t^2} - \nabla^2 + \frac{m_0^2c^2}{\hbar^2}\right)\psi = 0}$$

Or in covariant notation: $(\Box + m^2c^2/\hbar^2)\psi = 0$

### Problems with Klein-Gordon

1. **Second-order in time:** Requires two initial conditions, unlike Schrödinger equation
2. **Negative probability densities:** $\rho$ can be negative
3. **Negative energy solutions:** Leads to instabilities

These issues motivated Dirac to develop his first-order relativistic equation, which predicted antimatter.

### Proper Time and Quantum Phase

In relativistic quantum mechanics, the phase of a wave function evolves according to proper time:
$$\psi \propto e^{-imc^2\tau/\hbar}$$

This connects time dilation to quantum phase accumulation, important for interferometry with clocks.

---

## Worked Examples

### Example 1: Time Dilation in Particle Physics

**Problem:** A pion has a rest lifetime of $\tau_0 = 26$ ns. If it travels at $v = 0.99c$, how far does it travel in the lab frame before decaying (on average)?

**Solution:**

Step 1: Calculate Lorentz factor.
$$\gamma = \frac{1}{\sqrt{1 - 0.99^2}} = \frac{1}{\sqrt{0.0199}} = 7.09$$

Step 2: Calculate lab frame lifetime.
$$\tau_{lab} = \gamma\tau_0 = 7.09 \times 26 \text{ ns} = 184 \text{ ns}$$

Step 3: Calculate distance traveled.
$$d = v\tau_{lab} = 0.99 \times 3 \times 10^8 \times 184 \times 10^{-9} = 54.6 \text{ m}$$

$$\boxed{d = 54.6 \text{ m}}$$

### Example 2: Length Contraction and Ladder Paradox

**Problem:** A 10-meter ladder moves at $v = 0.8c$ toward a 6-meter garage. In the garage frame, the ladder appears contracted. Can it fit inside?

**Solution:**

In garage frame:
$$\gamma = \frac{1}{\sqrt{1 - 0.64}} = \frac{1}{0.6} = 1.67$$

Contracted length: $L = 10/1.67 = 6$ m

The ladder just fits! But in the ladder's frame, the garage is contracted to $6/1.67 = 3.6$ m. How can this be?

**Resolution:** The relativity of simultaneity! "Fitting inside" means both ends are in the garage **simultaneously**. But simultaneity is relative. Different frames disagree on whether both ends are ever inside at the same moment. Neither frame's conclusion is "wrong" - they simply measure different things.

### Example 3: Velocity Addition

**Problem:** A spaceship moves at $0.8c$ relative to Earth. It launches a probe at $0.6c$ (relative to the ship) in the same direction. What is the probe's speed relative to Earth?

**Solution:**

Using relativistic velocity addition:
$$u = \frac{0.6c + 0.8c}{1 + (0.6)(0.8)} = \frac{1.4c}{1.48} = 0.946c$$

Not $1.4c$ as Galilean addition would give!

$$\boxed{u = 0.946c}$$

---

## Practice Problems

### Problem 1: Direct Application
An astronaut travels to a star 20 light-years away at $v = 0.95c$.
(a) How long does the trip take in Earth's frame?
(b) How long does the trip take in the astronaut's frame?
(c) How far is the star in the astronaut's frame?

**Answers:** (a) 21.1 years; (b) 6.57 years; (c) 6.24 light-years

### Problem 2: Intermediate
Two events occur at $(x_1, t_1) = (0, 0)$ and $(x_2, t_2) = (4 \text{ m}, 3 \text{ ns})$ in frame $S$.
(a) What is the spacetime interval?
(b) Is the interval timelike, spacelike, or lightlike?
(c) Find a frame where the events are simultaneous (if possible).

**Answer:** (a) $s^2 = 15.19$ m²; (b) Spacelike (events are not causally connected); (c) Events occur at $t' = 0$ in frame with $v = 0.75c$

### Problem 3: Challenging
A photon of frequency $\nu$ is emitted by a source moving at velocity $v$ toward an observer. Derive the relativistic Doppler formula:
$$\nu_{obs} = \nu_{source}\sqrt{\frac{1 + v/c}{1 - v/c}}$$

**Hint:** Consider time dilation and the change in distance during one wave period.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
c = 299792458  # m/s (exact)

def gamma(v):
    """Calculate Lorentz factor"""
    beta = v / c
    return 1 / np.sqrt(1 - beta**2)

def lorentz_transform(x, t, v):
    """Transform (x, t) from S to S' moving with velocity v"""
    g = gamma(v)
    x_prime = g * (x - v * t)
    t_prime = g * (t - v * x / c**2)
    return x_prime, t_prime

def velocity_addition(u_prime, v):
    """Relativistic velocity addition: u' in S' frame, v is frame velocity"""
    return (u_prime + v) / (1 + u_prime * v / c**2)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Lorentz factor vs velocity ==========
ax1 = axes[0, 0]

v_values = np.linspace(0, 0.999*c, 1000)
gamma_values = gamma(v_values)

ax1.plot(v_values / c, gamma_values, 'b-', linewidth=2)
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)

# Mark specific velocities
special_v = [0.5, 0.8, 0.9, 0.95, 0.99]
for sv in special_v:
    g = gamma(sv * c)
    ax1.plot(sv, g, 'ro', markersize=8)
    ax1.annotate(f'γ={g:.2f}', (sv, g), textcoords='offset points',
                 xytext=(10, 5), fontsize=9)

ax1.set_xlabel('v/c', fontsize=12)
ax1.set_ylabel('γ (Lorentz factor)', fontsize=12)
ax1.set_title('Lorentz Factor vs Velocity', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 15)
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Time dilation ==========
ax2 = axes[0, 1]

# Proper time
tau_0 = 1  # normalized
velocities = [0, 0.5*c, 0.8*c, 0.9*c, 0.95*c]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(velocities)))

t_proper = np.linspace(0, 5, 100)

for v, color in zip(velocities, colors):
    g = gamma(v) if v > 0 else 1
    t_coord = g * t_proper
    ax2.plot(t_proper, t_coord, color=color, linewidth=2,
             label=f'v = {v/c:.2f}c (γ = {g:.2f})')

ax2.plot([0, 5], [0, 5], 'k--', alpha=0.3, label='No dilation')
ax2.set_xlabel('Proper time τ (normalized)', fontsize=12)
ax2.set_ylabel('Coordinate time t', fontsize=12)
ax2.set_title('Time Dilation: Moving Clocks Run Slow', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# ========== Plot 3: Spacetime diagram ==========
ax3 = axes[1, 0]

# Light cones
t = np.linspace(-3, 3, 100)
ax3.plot(t, t, 'y-', linewidth=2, label='Light (x = ct)')
ax3.plot(t, -t, 'y-', linewidth=2)
ax3.fill_between(t, t, 3, alpha=0.1, color='yellow')
ax3.fill_between(t, -t, 3, alpha=0.1, color='yellow')

# World lines
ax3.axvline(x=0, color='blue', linewidth=2, label='Observer at rest')
ax3.plot(t * 0.5, t, 'r-', linewidth=2, label='v = 0.5c')
ax3.plot(t * 0.8, t, 'g-', linewidth=2, label='v = 0.8c')

# Events
events = [(0, 0), (1, 1.5), (-0.5, 2), (2, 1)]
for i, (x, ct) in enumerate(events):
    ax3.plot(x, ct, 'ko', markersize=10)
    ax3.annotate(f'E{i+1}', (x, ct), textcoords='offset points',
                 xytext=(5, 5), fontsize=10)

ax3.set_xlabel('x (light-seconds)', fontsize=12)
ax3.set_ylabel('ct (light-seconds)', fontsize=12)
ax3.set_title('Spacetime Diagram', fontsize=14)
ax3.legend(loc='lower right', fontsize=9)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-1, 3)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# ========== Plot 4: Relativistic velocity addition ==========
ax4 = axes[1, 1]

# Frame velocity
frame_v = np.array([0.3, 0.5, 0.7, 0.9]) * c

u_prime = np.linspace(0, 0.99*c, 100)

for fv in frame_v:
    u_relativistic = velocity_addition(u_prime, fv)
    ax4.plot(u_prime / c, u_relativistic / c, linewidth=2,
             label=f'Frame v = {fv/c:.1f}c')

# Galilean addition for comparison (v = 0.5c)
u_galilean = (u_prime + 0.5*c)
ax4.plot(u_prime / c, u_galilean / c, 'k--', linewidth=1,
         label='Galilean (v = 0.5c)', alpha=0.5)

ax4.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Speed of light')
ax4.set_xlabel("u'/c (velocity in moving frame)", fontsize=12)
ax4.set_ylabel("u/c (velocity in rest frame)", fontsize=12)
ax4.set_title('Relativistic Velocity Addition', fontsize=14)
ax4.legend(fontsize=9)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_218_lorentz_transformations.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Muon lifetime demonstration ==========
print("=" * 60)
print("MUON LIFETIME DEMONSTRATION")
print("=" * 60)

# Muon properties
tau_muon = 2.2e-6  # seconds (rest lifetime)
v_muon = 0.998 * c  # typical cosmic ray muon velocity

gamma_muon = gamma(v_muon)
tau_lab = gamma_muon * tau_muon
distance = v_muon * tau_lab

print(f"\nMuon rest lifetime: τ₀ = {tau_muon*1e6:.1f} μs")
print(f"Muon velocity: v = {v_muon/c:.4f}c")
print(f"Lorentz factor: γ = {gamma_muon:.2f}")
print(f"Lab frame lifetime: τ = {tau_lab*1e6:.1f} μs")
print(f"Distance traveled: d = {distance/1000:.1f} km")
print(f"\nWithout relativity, would travel: {(v_muon * tau_muon)/1000:.2f} km")
print(f"Atmospheric depth: ~10-20 km")
print(f"Conclusion: Muons reach Earth's surface thanks to time dilation!")

# ========== Lorentz transformation animation data ==========
print("\n" + "=" * 60)
print("LORENTZ TRANSFORMATION EXAMPLE")
print("=" * 60)

# Event in frame S
x_S, t_S = 1e9, 5  # 1 billion meters, 5 seconds

# Transform to frame moving at 0.6c
v_frame = 0.6 * c
x_S_prime, t_S_prime = lorentz_transform(x_S, t_S, v_frame)

print(f"\nEvent in frame S: x = {x_S/1e9:.1f} Gm, t = {t_S:.1f} s")
print(f"Frame S' moves at v = {v_frame/c:.1f}c relative to S")
print(f"Lorentz factor: γ = {gamma(v_frame):.3f}")
print(f"Event in frame S': x' = {x_S_prime/1e9:.3f} Gm, t' = {t_S_prime:.3f} s")

# Verify invariance of spacetime interval
s2_S = -(c * t_S)**2 + x_S**2
s2_S_prime = -(c * t_S_prime)**2 + x_S_prime**2
print(f"\nSpacetime interval in S: s² = {s2_S:.3e} m²")
print(f"Spacetime interval in S': s² = {s2_S_prime:.3e} m²")
print(f"Difference: {abs(s2_S - s2_S_prime):.3e} m² (should be ~0)")

print("\n" + "=" * 60)
print("Day 218: Lorentz Transformations Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\gamma = 1/\sqrt{1 - v^2/c^2}$ | Lorentz factor |
| $x' = \gamma(x - vt)$ | Lorentz transformation (position) |
| $t' = \gamma(t - vx/c^2)$ | Lorentz transformation (time) |
| $\Delta t = \gamma\Delta\tau$ | Time dilation |
| $L = L_0/\gamma$ | Length contraction |
| $u = (u' + v)/(1 + u'v/c^2)$ | Velocity addition |

### Main Takeaways

1. **Einstein's postulates** revolutionize our understanding of space and time
2. **Lorentz transformations** preserve the spacetime interval $s^2 = -c^2t^2 + x^2$
3. **Time dilation:** Moving clocks run slow by factor $\gamma$
4. **Length contraction:** Moving objects are shortened by factor $\gamma$
5. **Simultaneity is relative:** Events simultaneous in one frame are not in another
6. **Nothing travels faster than light** - velocity addition ensures $u < c$

---

## Daily Checklist

- [ ] I can state Einstein's two postulates
- [ ] I can derive the Lorentz transformations
- [ ] I can calculate time dilation and length contraction
- [ ] I understand the relativity of simultaneity
- [ ] I can apply relativistic velocity addition
- [ ] I understand the connection to the Klein-Gordon equation

---

## Preview: Day 219

Tomorrow we formalize spacetime using **4-vectors and the Minkowski metric**. We'll see how position, velocity, and momentum become unified spacetime objects that transform simply under Lorentz transformations.

---

*"When you sit with a nice girl for two hours, it seems like two minutes. When you sit on a hot stove for two minutes, it seems like two hours. That's relativity."*
— Albert Einstein (popular explanation)

---

**Next:** Day 219 — Spacetime and 4-Vectors
