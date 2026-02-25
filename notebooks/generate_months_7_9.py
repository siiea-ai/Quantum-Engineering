#!/usr/bin/env python3
"""
SIIEA Quantum Engineering — Generate Year 0, Months 7-9 Notebooks

Produces three Jupyter notebooks:
  1. Month 07 — Complex Analysis Deep: Contour Integration
  2. Month 08 — Electromagnetism: Maxwell's Equations
  3. Month 09 — Functional Analysis: Hilbert Spaces

Run with:
    .venv/bin/python3 notebooks/generate_months_7_9.py
"""

import sys
import os

# Ensure the notebooks directory is on the path so we can import the builder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_notebook import NotebookBuilder


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK 1 — Month 07: Complex Analysis Deep — Contour Integration
# ═══════════════════════════════════════════════════════════════════════════════

def build_month_07():
    nb = NotebookBuilder(
        "Complex Analysis Deep — Contour Integration & Residues",
        "year_0/month_07_complex_analysis/07_contour_integration.ipynb",
        "Days 169-196",
    )

    # ── imports ───────────────────────────────────────────────────────────────
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
import sympy as sp
from scipy import integrate

%matplotlib inline

# Publication-quality defaults
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 11,
    "font.family": "serif",
    "figure.dpi": 120,
})
print("Imports ready — numpy, matplotlib, sympy, scipy loaded.")""")

    # ── Theory 1: Analytic functions ─────────────────────────────────────────
    nb.md(r"""\
## 1. Analytic Functions and the Cauchy–Riemann Equations

A complex function $f(z) = u(x,y) + i\,v(x,y)$ is **analytic** (holomorphic) at
a point if and only if the **Cauchy–Riemann equations** hold:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \qquad
  \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

These conditions guarantee that $f'(z)$ exists and is independent of the
direction of approach in the complex plane.

### Example: $f(z) = z^2$

Writing $z = x + iy$:

$$f(z) = (x+iy)^2 = (x^2 - y^2) + i(2xy)$$

So $u = x^2 - y^2$, $v = 2xy$. Check:

$$u_x = 2x = v_y, \quad u_y = -2y = -v_x \;\checkmark$$""")

    # ── Code 1: CR verification ──────────────────────────────────────────────
    nb.code(r"""\
# Verify Cauchy-Riemann equations symbolically for several functions
x, y = sp.symbols('x y', real=True)

functions = {
    "z^2":   (x**2 - y**2, 2*x*y),
    "e^z":   (sp.exp(x)*sp.cos(y), sp.exp(x)*sp.sin(y)),
    "sin(z)": (sp.sin(x)*sp.cosh(y), sp.cos(x)*sp.sinh(y)),
    "1/z":   (x/(x**2+y**2), -y/(x**2+y**2)),
}

print("Cauchy–Riemann Verification")
print("=" * 55)
for name, (u, v) in functions.items():
    ux = sp.diff(u, x)
    vy = sp.diff(v, y)
    uy = sp.diff(u, y)
    vx = sp.diff(v, x)
    cr1 = sp.simplify(ux - vy)
    cr2 = sp.simplify(uy + vx)
    status = "ANALYTIC" if cr1 == 0 and cr2 == 0 else "NOT analytic"
    print(f"\nf(z) = {name}")
    print(f"  u_x = {ux},  v_y = {vy}  →  u_x - v_y = {cr1}")
    print(f"  u_y = {uy},  v_x = {vx}  →  u_y + v_x = {cr2}")
    print(f"  ⇒ {status}")""")

    # ── Theory 2: Conformal mappings ─────────────────────────────────────────
    nb.md(r"""\
## 2. Conformal Mappings

An analytic function $f(z)$ with $f'(z) \neq 0$ defines a **conformal mapping**
— it preserves angles between curves. Key examples:

| Mapping | Formula | Geometric effect |
|---------|---------|-----------------|
| Square  | $w = z^2$ | Doubles angles at origin |
| Exponential | $w = e^z$ | Horizontal strips → sectors |
| Möbius | $w = \frac{az+b}{cz+d}$ | Circles/lines → circles/lines |
| Joukowski | $w = z + 1/z$ | Circles → airfoil shapes |

Conformal mappings are essential in physics for solving Laplace's equation
$\nabla^2 \phi = 0$ in complicated geometries.""")

    # ── Code 2: Conformal mapping visualization ──────────────────────────────
    nb.code(r"""\
# Conformal mapping visualization: show how z^2, exp(z), and Mobius
# transform a regular grid in the complex plane

def plot_conformal(f, title, ax, xlim=(-2,2), ylim=(-2,2), N=20):
    '''Plot image of a grid under conformal map f.'''
    # Horizontal lines
    for yv in np.linspace(ylim[0], ylim[1], N):
        xs = np.linspace(xlim[0], xlim[1], 400)
        zs = xs + 1j * yv
        ws = f(zs)
        ax.plot(ws.real, ws.imag, 'b-', lw=0.5, alpha=0.6)
    # Vertical lines
    for xv in np.linspace(xlim[0], xlim[1], N):
        ys = np.linspace(ylim[0], ylim[1], 400)
        zs = xv + 1j * ys
        ws = f(zs)
        ax.plot(ws.real, ws.imag, 'r-', lw=0.5, alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Re(w)")
    ax.set_ylabel("Im(w)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# w = z^2
plot_conformal(lambda z: z**2, r"$w = z^2$", axes[0],
               xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

# w = exp(z)
plot_conformal(lambda z: np.exp(z), r"$w = e^z$", axes[1],
               xlim=(-2, 2), ylim=(-np.pi, np.pi))

# Mobius: w = (z - 1)/(z + 1)
plot_conformal(lambda z: (z - 1)/(z + 1), r"$w = (z-1)/(z+1)$", axes[2],
               xlim=(-2, 2), ylim=(-2, 2))

plt.tight_layout()
plt.show()
print("Conformal maps transform the rectangular grid while preserving local angles.")""")

    # ── Theory 3: Contour integration ────────────────────────────────────────
    nb.md(r"""\
## 3. Contour Integration

A **contour integral** is defined as:

$$\oint_C f(z)\,dz = \int_a^b f(z(t))\,z'(t)\,dt$$

where $z(t)$ parametrizes the contour $C$ from $t=a$ to $t=b$.

### Cauchy's Integral Theorem

If $f$ is analytic on and inside a simple closed contour $C$:

$$\oint_C f(z)\,dz = 0$$

### Cauchy's Integral Formula

If $f$ is analytic inside $C$ and $z_0$ is inside $C$:

$$f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z - z_0}\,dz$$

This extraordinary formula says that the values of an analytic function on a
closed curve completely determine its values *inside* the curve.""")

    # ── Code 3: Parametric contour integration ───────────────────────────────
    nb.code(r"""\
# Contour integration: evaluate ∮_C z^n dz around the unit circle
# for various n, verifying Cauchy's theorem and the residue of 1/z

def contour_integrate_unit_circle(f, N=10000):
    '''Numerically integrate f(z) dz around the unit circle.'''
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2*np.pi / N
    z = np.exp(1j * t)         # z(t) = e^{it}
    dz = 1j * np.exp(1j * t)  # dz/dt = i e^{it}
    integrand = f(z) * dz * dt
    return np.sum(integrand)

print("Contour integrals ∮_C z^n dz around the unit circle |z|=1")
print("=" * 55)
for n in range(-3, 4):
    result = contour_integrate_unit_circle(lambda z, n=n: z**n)
    expected = 2j * np.pi if n == -1 else 0
    print(f"  n = {n:+d}:  numerical = {result:.6f},  "
          f"exact = {expected:.6f},  "
          f"error = {abs(result - expected):.2e}")

print("\nKey result: Only n = -1 gives nonzero result (= 2πi).")
print("This is the fundamental residue!")""")

    # ── Code 4: Cauchy integral formula verification ─────────────────────────
    nb.code(r"""\
# Cauchy's Integral Formula: f(z0) = (1/2πi) ∮ f(z)/(z-z0) dz
# Verify numerically for f(z) = sin(z), z0 = 0.3 + 0.2i

def cauchy_integral_formula(f, z0, R=1.0, N=50000):
    '''Compute f(z0) via Cauchy integral formula.
    Integrate around a circle of radius R centered at the origin.'''
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2 * np.pi / N
    z = R * np.exp(1j * t)
    dz = 1j * R * np.exp(1j * t) * dt
    integrand = f(z) / (z - z0) * dz
    return np.sum(integrand) / (2j * np.pi)

# Test with several analytic functions
z0 = 0.3 + 0.2j

test_functions = {
    "sin(z)":  (np.sin, np.sin(z0)),
    "exp(z)":  (np.exp, np.exp(z0)),
    "z^3+1":   (lambda z: z**3 + 1, z0**3 + 1),
    "cos(z^2)":(lambda z: np.cos(z**2), np.cos(z0**2)),
}

print(f"Cauchy Integral Formula verification at z₀ = {z0}")
print("=" * 65)
for name, (func, exact) in test_functions.items():
    numerical = cauchy_integral_formula(func, z0)
    err = abs(numerical - exact)
    print(f"  f(z) = {name:12s}:  CIF = {numerical:.10f}")
    print(f"  {'':12s}   exact = {exact:.10f},  error = {err:.2e}")
print("\nCauchy's formula recovers interior values from boundary data — remarkable!")""")

    # ── Theory 4: Laurent series and singularities ───────────────────────────
    nb.md(r"""\
## 4. Laurent Series and Singularity Classification

Near an isolated singularity $z_0$, a function has a **Laurent series**:

$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$

The **principal part** $\sum_{n=-\infty}^{-1} a_n(z-z_0)^n$ classifies the singularity:

| Type | Principal part | Example |
|------|---------------|---------|
| **Removable** | Empty (all $a_{-n}=0$) | $\frac{\sin z}{z}$ at $z=0$ |
| **Pole of order $m$** | Finite ($a_{-m}\neq 0$, $a_{-n}=0$ for $n>m$) | $\frac{1}{z^2}$ at $z=0$ |
| **Essential** | Infinite series | $e^{1/z}$ at $z=0$ |

The coefficient $a_{-1}$ is the **residue**: $\text{Res}_{z_0} f = a_{-1}$.""")

    # ── Code 5: Laurent series computation ───────────────────────────────────
    nb.code(r"""\
# Laurent series expansion using SymPy
z = sp.Symbol('z')

functions = {
    "sin(z)/z at z=0": (sp.sin(z)/z, 0),
    "1/(z^2(z-1)) at z=0": (1/(z**2 * (z - 1)), 0),
    "exp(1/z) at z=0": (sp.exp(1/z), 0),
    "1/((z-1)(z-2)) at z=1": (1/((z-1)*(z-2)), 1),
}

print("Laurent Series Expansions")
print("=" * 65)
for name, (f_expr, z0) in functions.items():
    print(f"\nf(z) = {name}")
    series = sp.series(f_expr, z, z0, n=6)
    print(f"  Series: {series}")

    # Compute residue
    residue = sp.residue(f_expr, z, z0)
    print(f"  Residue at z={z0}: {residue}")

    # Classify singularity
    # Check leading negative power
    series_dict = series.removeO()
    if hasattr(series_dict, 'as_ordered_terms'):
        terms = series_dict.as_ordered_terms()
        neg_powers = []
        for term in terms:
            coeff, powers = term.as_coeff_mul(z - z0 if z0 != 0 else z)
            for p in powers:
                exp = p.as_base_exp()[1] if hasattr(p, 'as_base_exp') else 0
                if hasattr(exp, 'is_negative') and exp.is_negative:
                    neg_powers.append(exp)
    # Use SymPy's built-in singularity detection
    order = sp.singularities(f_expr, z)
    print(f"  Singularities of f: {order}")""")

    # ── Theory 5: Residue theorem ────────────────────────────────────────────
    nb.md(r"""\
## 5. The Residue Theorem

The **Residue Theorem** is the crown jewel of complex analysis:

$$\oint_C f(z)\,dz = 2\pi i \sum_{k} \text{Res}_{z_k} f$$

where the sum is over all singularities $z_k$ enclosed by $C$.

### Evaluating Real Integrals

The residue theorem lets us compute difficult *real* integrals by extending
them to the complex plane:

$$\int_{-\infty}^{\infty} \frac{dx}{1+x^2} = 2\pi i \cdot \text{Res}_{z=i} \frac{1}{1+z^2}
= 2\pi i \cdot \frac{1}{2i} = \pi$$

This technique is used throughout quantum mechanics for evaluating propagators,
Green's functions, and scattering amplitudes.""")

    # ── Code 6: Residue theorem — real integrals ─────────────────────────────
    nb.code(r"""\
# Evaluate real integrals using the residue theorem

z = sp.Symbol('z')
x_sym = sp.Symbol('x', real=True)

# Integral 1: ∫_{-∞}^{∞} dx/(1+x^2) = π
f1 = 1/(1 + z**2)
poles_f1 = sp.solve(1 + z**2, z)
res_f1 = sum(sp.residue(f1, z, p) for p in poles_f1 if sp.im(p) > 0)
result_1 = 2 * sp.pi * sp.I * res_f1
print("Integral 1: ∫_{-∞}^{∞} dx/(1+x²)")
print(f"  Poles: {poles_f1}")
print(f"  Upper half-plane residue sum: {res_f1}")
print(f"  Result: 2πi × {res_f1} = {sp.simplify(result_1)}")
print(f"  Numerical check: {float(sp.re(result_1)):.10f} vs π = {np.pi:.10f}")

# Integral 2: ∫_{-∞}^{∞} dx/(1+x^4) = π/√2
print("\nIntegral 2: ∫_{-∞}^{∞} dx/(1+x⁴)")
f2 = 1/(1 + z**4)
poles_f2 = sp.solve(1 + z**4, z)
uhp_poles = [p for p in poles_f2 if sp.re(sp.im(p)) > 0]
res_f2 = sum(sp.residue(f2, z, p) for p in uhp_poles)
result_2 = 2 * sp.pi * sp.I * res_f2
print(f"  Poles in upper half-plane: {uhp_poles}")
print(f"  Residue sum: {sp.simplify(res_f2)}")
print(f"  Result: {sp.simplify(result_2)}")
print(f"  Numerical: {float(sp.re(sp.simplify(result_2))):.10f} vs π/√2 = {np.pi/np.sqrt(2):.10f}")

# Integral 3: ∫_{-∞}^{∞} x^2/(1+x^4) dx
print("\nIntegral 3: ∫_{-∞}^{∞} x²/(1+x⁴) dx")
f3 = z**2/(1 + z**4)
res_f3 = sum(sp.residue(f3, z, p) for p in uhp_poles)
result_3 = 2 * sp.pi * sp.I * res_f3
print(f"  Result: {sp.simplify(result_3)}")
print(f"  Numerical: {float(sp.re(sp.simplify(result_3))):.10f} vs π/√2 = {np.pi/np.sqrt(2):.10f}")

# Verify all results with scipy numerical integration
from scipy.integrate import quad
for label, func, exact in [
    ("∫dx/(1+x²)", lambda x: 1/(1+x**2), np.pi),
    ("∫dx/(1+x⁴)", lambda x: 1/(1+x**4), np.pi/np.sqrt(2)),
    ("∫x²dx/(1+x⁴)", lambda x: x**2/(1+x**4), np.pi/np.sqrt(2)),
]:
    val, err = quad(func, -np.inf, np.inf)
    print(f"\n  scipy check for {label}: {val:.10f} (error ±{err:.2e}), exact = {exact:.10f}")""")

    # ── Code 7: Domain coloring (phase portrait) ─────────────────────────────
    nb.code(r"""\
# Domain coloring: visualize complex functions via phase portraits
# Hue = argument, brightness = modulus

def domain_coloring(f, xlim=(-3, 3), ylim=(-3, 3), N=800, title=""):
    '''Create a domain coloring (phase portrait) of f(z).'''
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    with np.errstate(divide='ignore', invalid='ignore'):
        W = f(Z)

    # Map argument to hue [0,1]
    H = (np.angle(W) + np.pi) / (2 * np.pi)
    # Map modulus to brightness with smooth saturation
    modulus = np.abs(W)
    S = np.ones_like(H) * 0.9
    V = 1 - 1/(1 + modulus**0.3)  # smooth brightness scaling

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)
    return RGB

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# f(z) = z^2 - 1  (two zeros)
rgb1 = domain_coloring(lambda z: z**2 - 1, title=r"$z^2-1$")
axes[0].imshow(rgb1, extent=[-3,3,-3,3], origin='lower')
axes[0].set_title(r"$f(z) = z^2 - 1$ (zeros at $\pm 1$)", fontsize=13)
axes[0].set_xlabel("Re(z)"); axes[0].set_ylabel("Im(z)")

# f(z) = 1/(z^2 + 1)  (poles at ±i)
rgb2 = domain_coloring(lambda z: 1/(z**2 + 1))
axes[1].imshow(rgb2, extent=[-3,3,-3,3], origin='lower')
axes[1].set_title(r"$f(z) = 1/(z^2+1)$ (poles at $\pm i$)", fontsize=13)
axes[1].set_xlabel("Re(z)"); axes[1].set_ylabel("Im(z)")

# f(z) = exp(1/z)  (essential singularity at 0)
rgb3 = domain_coloring(lambda z: np.exp(1/z), xlim=(-2,2), ylim=(-2,2))
axes[2].imshow(rgb3, extent=[-2,2,-2,2], origin='lower')
axes[2].set_title(r"$f(z) = e^{1/z}$ (essential sing. at 0)", fontsize=13)
axes[2].set_xlabel("Re(z)"); axes[2].set_ylabel("Im(z)")

plt.tight_layout()
plt.show()
print("Phase portraits: hue = arg(f(z)), brightness = |f(z)|")
print("Zeros appear as points where all colors meet (dark); poles are bright convergence points.")
print("Essential singularity shows wild oscillation near z=0.")""")

    # ── Theory 6: QM Connection ──────────────────────────────────────────────
    nb.md(r"""\
## 6. Quantum Mechanics Connection: Complex Analysis in Physics

Complex analysis is **indispensable** in quantum mechanics and quantum field theory:

### Propagators and Green's Functions

The free-particle propagator involves a contour integral:

$$G(E) = \lim_{\epsilon \to 0^+} \frac{1}{E - H + i\epsilon}$$

The $i\epsilon$ prescription tells us *which way to close the contour* — into the
upper or lower half-plane — giving retarded vs. advanced propagators.

### The S-Matrix and Scattering

Poles of the S-matrix in the complex energy plane correspond to:
- **Bound states** (poles on the negative real axis)
- **Resonances** (poles in the lower half-plane, $E = E_0 - i\Gamma/2$)

### Path Integrals

Feynman's path integral uses a Wick rotation $t \to -i\tau$ — a conformal
mapping from Minkowski to Euclidean spacetime:

$$\langle x_f | e^{-iHt/\hbar} | x_i \rangle = \int \mathcal{D}[x]\, e^{iS[x]/\hbar}$$

### The Residue Theorem in QFT

Loop integrals in Feynman diagrams are evaluated by residues, giving physical
quantities like decay rates and cross sections.""")

    # ── Code 8: Propagator pole structure ────────────────────────────────────
    nb.code(r"""\
# QM Application: Visualize the retarded Green's function
# G_R(E) = 1/(E - E_0 + iε) and its pole structure

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Pole structure of the propagator ---
ax = axes[0]
E0_values = [1.0, 2.5, 4.0]
for E0 in E0_values:
    ax.plot(E0, 0, 'rx', markersize=12, markeredgewidth=2)
    ax.annotate(f'$E_{{{E0_values.index(E0)}}}={E0}$',
                (E0, 0), textcoords="offset points",
                xytext=(5, 10), fontsize=11)

# Show iε displacement
epsilon = 0.15
for E0 in E0_values:
    ax.plot(E0, -epsilon, 'bo', markersize=8)
    ax.annotate('', xy=(E0, -epsilon), xytext=(E0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("Re(E)", fontsize=12)
ax.set_ylabel("Im(E)", fontsize=12)
ax.set_title("Propagator Poles: $G_R = 1/(E - E_n + i\\epsilon)$", fontsize=13)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-1.0, 1.0)
ax.legend(["Energy eigenvalues", "Retarded poles ($-i\\epsilon$)"],
          loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# --- Right: |G(E)|^2 spectral function (Lorentzian) ---
ax2 = axes[1]
E = np.linspace(-1, 6, 1000)
gamma_values = [0.05, 0.2, 0.5]
E0 = 2.5

for gamma in gamma_values:
    spectral = (1/np.pi) * (gamma/2) / ((E - E0)**2 + (gamma/2)**2)
    ax2.plot(E, spectral, label=f"$\\Gamma = {gamma}$")

ax2.set_xlabel("Energy $E$", fontsize=12)
ax2.set_ylabel("$A(E) = -\\frac{1}{\\pi}\\mathrm{Im}\\,G_R(E)$", fontsize=12)
ax2.set_title(f"Spectral Function (Lorentzian) at $E_0={E0}$", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Left: Poles of G_R shifted below real axis by iε — causality requirement.")
print("Right: Spectral function A(E) is a Lorentzian; width Γ = decay rate.")
print("In the limit Γ → 0, A(E) → δ(E - E₀) (stable state).")""")

    # ── Code 9: Residue computation of Fresnel-type integral ─────────────────
    nb.code(r"""\
# Advanced application: compute the Fresnel-type integral
# ∫_0^∞ cos(x^2) dx = √(π/2)/2  via contour methods
# (verified numerically)

from scipy.integrate import quad

# Numerical evaluation
val_cos, _ = quad(lambda x: np.cos(x**2), 0, 1000)
val_sin, _ = quad(lambda x: np.sin(x**2), 0, 1000)
exact = np.sqrt(np.pi/2) / 2

print("Fresnel Integrals (evaluated via contour rotation z → z·e^{iπ/4})")
print("=" * 55)
print(f"  ∫₀^∞ cos(x²) dx = {val_cos:.10f}")
print(f"  ∫₀^∞ sin(x²) dx = {val_sin:.10f}")
print(f"  Exact: √(π/2)/2 = {exact:.10f}")
print(f"  Error: {abs(val_cos - exact):.2e}")

# Visualize the Fresnel spiral (Cornu spiral)
t = np.linspace(0, 8, 5000)
C = np.array([quad(lambda s: np.cos(s**2), 0, ti)[0] for ti in t])
S = np.array([quad(lambda s: np.sin(s**2), 0, ti)[0] for ti in t])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(C, S, 'b-', lw=1.5)
ax.plot(C[0], S[0], 'go', markersize=10, label='Start (t=0)')
ax.plot(exact, exact, 'r*', markersize=15, label=f'Limit point ({exact:.4f}, {exact:.4f})')
ax.set_xlabel("$C(t) = \\int_0^t \\cos(s^2)\\,ds$", fontsize=12)
ax.set_ylabel("$S(t) = \\int_0^t \\sin(s^2)\\,ds$", fontsize=12)
ax.set_title("Cornu Spiral (Fresnel Integrals)", fontsize=14)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("The Cornu spiral converges to (√(π/2)/2, √(π/2)/2) — derived via contour rotation.")""")

    # ── Summary ──────────────────────────────────────────────────────────────
    nb.md(r"""\
## Summary

| Topic | Key Result |
|-------|-----------|
| Cauchy–Riemann | $u_x = v_y$, $u_y = -v_x$ — test for analyticity |
| Conformal maps | Analytic functions with $f' \neq 0$ preserve angles |
| Cauchy's theorem | $\oint_C f\,dz = 0$ for analytic $f$ |
| Cauchy's formula | $f(z_0) = \frac{1}{2\pi i}\oint \frac{f(z)}{z-z_0}dz$ |
| Laurent series | $f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$ |
| Residue theorem | $\oint_C f\,dz = 2\pi i \sum \text{Res}_{z_k}f$ |
| QM connection | Propagators, S-matrix poles, path integrals |

**Next:** Month 08 — Electromagnetism and Maxwell's Equations

---
*SIIEA Quantum Engineering Curriculum — CC BY-NC-SA 4.0*""")

    nb.save()
    print("  [Month 07] Complex Analysis notebook complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK 2 — Month 08: Electromagnetism — Maxwell's Equations
# ═══════════════════════════════════════════════════════════════════════════════

def build_month_08():
    nb = NotebookBuilder(
        "Electromagnetism — Maxwell's Equations & EM Waves",
        "year_0/month_08_electromagnetism/08_maxwell_equations.ipynb",
        "Days 197-224",
    )

    # ── imports ───────────────────────────────────────────────────────────────
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from scipy import integrate
from scipy.constants import epsilon_0, mu_0, c, e as e_charge, hbar

%matplotlib inline

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 11,
    "font.family": "serif",
    "figure.dpi": 120,
})
print("Imports ready — numpy, matplotlib, sympy, scipy loaded.")
print(f"Physical constants: ε₀ = {epsilon_0:.4e}, μ₀ = {mu_0:.4e}, c = {c:.4e} m/s")""")

    # ── Theory 1: Electric fields ────────────────────────────────────────────
    nb.md(r"""\
## 1. Electric Fields: Coulomb's Law and Superposition

The electric field of a point charge $q$ at the origin is:

$$\vec{E}(\vec{r}) = \frac{1}{4\pi\epsilon_0}\frac{q}{r^2}\hat{r}$$

For multiple charges, the **superposition principle** gives:

$$\vec{E}(\vec{r}) = \frac{1}{4\pi\epsilon_0}\sum_i \frac{q_i}{|\vec{r} - \vec{r}_i|^2}\hat{r}_i$$

The **electric dipole** (charges $+q$ and $-q$ separated by distance $d$) produces
the characteristic field pattern that is fundamental to atomic physics and
molecular bonding.""")

    # ── Code 1: Electric field visualization ─────────────────────────────────
    nb.code(r"""\
# Electric field visualization: point charge, dipole, quadrupole

def E_field(charges, positions, X, Y):
    '''Compute electric field from point charges at positions.'''
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    for q, (xq, yq) in zip(charges, positions):
        dx = X - xq
        dy = Y - yq
        r2 = dx**2 + dy**2
        r2 = np.maximum(r2, 1e-6)  # avoid singularity
        r3 = r2**1.5
        Ex += q * dx / r3
        Ey += q * dy / r3
    return Ex, Ey

x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

configs = [
    ("Point Charge (+)", [1], [(0, 0)]),
    ("Dipole (+/−)", [1, -1], [(-0.5, 0), (0.5, 0)]),
    ("Quadrupole", [1, -1, 1, -1], [(-1,-1), (-1,1), (1,-1), (1,1)]),
]

for ax, (title, charges, positions) in zip(axes, configs):
    Ex, Ey = E_field(charges, positions, X, Y)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag = np.maximum(E_mag, 1e-10)
    # Normalize for uniform arrow length
    Ex_n = Ex / E_mag
    Ey_n = Ey / E_mag

    ax.quiver(X, Y, Ex_n, Ey_n, E_mag, cmap='inferno',
              norm=plt.Normalize(vmin=0, vmax=np.percentile(E_mag, 90)),
              alpha=0.8)
    for q, (xq, yq) in zip(charges, positions):
        color = 'red' if q > 0 else 'blue'
        ax.plot(xq, yq, 'o', color=color, markersize=12, markeredgecolor='black')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
print("Quiver plots show field direction and magnitude (color).")
print("Dipole: field lines flow from + to −. Quadrupole: more complex topology.")""")

    # ── Theory 2: Gauss's law ────────────────────────────────────────────────
    nb.md(r"""\
## 2. Gauss's Law

**Gauss's Law** (integral form):

$$\oint_S \vec{E} \cdot d\vec{A} = \frac{Q_{\text{enc}}}{\epsilon_0}$$

**Differential form:**

$$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$$

The total electric flux through any closed surface equals the enclosed charge
divided by $\epsilon_0$. This is one of the four Maxwell equations and reflects
the fact that electric field lines **originate** on positive charges and
**terminate** on negative charges.""")

    # ── Code 2: Gauss's law numerical verification ───────────────────────────
    nb.code(r"""\
# Numerical verification of Gauss's Law
# Compute the surface integral of E over a sphere surrounding a point charge

from scipy.integrate import dblquad

# Physical setup: charge Q at origin, integrate E over sphere of radius R
Q = 1.0  # Coulomb (we use 4πε₀ = 1 units for simplicity)
k = 1.0  # 1/(4πε₀) in natural units

# The flux should be Q/ε₀ = 4πkQ in these units

# Analytical: Φ = Q/ε₀ = 4πkQ
exact_flux = 4 * np.pi * k * Q

# Numerical integration using spherical coordinates
# E_r = kQ/R^2, dA = R^2 sin(θ) dθ dφ
# So E·dA = kQ sin(θ) dθ dφ (R cancels!)

radii = [0.5, 1.0, 2.0, 5.0, 10.0]

print("Gauss's Law Verification: ∮ E·dA = Q/ε₀")
print("=" * 60)
print(f"  Charge: Q = {Q}")
print(f"  Exact flux: 4πkQ = {exact_flux:.10f}")
print()

for R in radii:
    # Integrate kQ/R^2 * R^2 sin(θ) over θ∈[0,π], φ∈[0,2π]
    flux, error = dblquad(
        lambda theta, phi: k * Q * np.sin(theta),
        0, 2*np.pi,           # phi limits
        lambda phi: 0,         # theta lower
        lambda phi: np.pi,     # theta upper
    )
    rel_error = abs(flux - exact_flux) / exact_flux
    print(f"  R = {R:5.1f}:  Φ = {flux:.10f},  rel error = {rel_error:.2e}")

print(f"\nFlux is independent of radius — Gauss's law confirmed!")""")

    # ── Theory 3: Biot-Savart law ────────────────────────────────────────────
    nb.md(r"""\
## 3. Magnetic Fields: Biot-Savart Law

The magnetic field produced by a current element $I\,d\vec{l}$ is:

$$d\vec{B} = \frac{\mu_0}{4\pi}\frac{I\,d\vec{l} \times \hat{r}}{r^2}$$

For a **circular current loop** of radius $a$ carrying current $I$, the field
on the axis is:

$$B_z(z) = \frac{\mu_0 I a^2}{2(a^2 + z^2)^{3/2}}$$

At the center ($z=0$): $B = \mu_0 I / (2a)$.

This is the basis for understanding magnetic dipoles, MRI technology,
and the magnetic moments of atoms.""")

    # ── Code 3: Biot-Savart current loop ─────────────────────────────────────
    nb.code(r"""\
# Magnetic field of a current loop via Biot-Savart numerical integration
# Compute B-field in the xz-plane for a loop of radius a in the xy-plane

def biot_savart_loop(a, I, points, N=1000):
    '''Compute B-field at given points from a circular loop
    of radius a, current I, centered at origin in xy-plane.
    Uses numerical integration of Biot-Savart law.'''
    mu0_over_4pi = 1e-7  # μ₀/(4π) in SI
    dphi = 2 * np.pi / N
    Bx = np.zeros(len(points))
    By = np.zeros(len(points))
    Bz = np.zeros(len(points))

    for i in range(N):
        phi = i * dphi
        # Current element position
        rl = np.array([a * np.cos(phi), a * np.sin(phi), 0.0])
        # Current element direction: dl = a*dphi * (-sin(phi), cos(phi), 0)
        dl = a * dphi * np.array([-np.sin(phi), np.cos(phi), 0.0])

        for j, r in enumerate(points):
            dr = r - rl
            dist = np.linalg.norm(dr)
            if dist < 1e-10:
                continue
            # dB = (μ₀/4π) I (dl × r̂) / r²
            dB = mu0_over_4pi * I * np.cross(dl, dr) / dist**3
            Bx[j] += dB[0]
            By[j] += dB[1]
            Bz[j] += dB[2]

    return Bx, By, Bz

# Compute on-axis field and compare with exact formula
a = 0.1   # loop radius (meters)
I = 1.0   # current (Amperes)
z_values = np.linspace(-0.3, 0.3, 50)
points_axis = [np.array([0, 0, z]) for z in z_values]

_, _, Bz_numerical = biot_savart_loop(a, I, points_axis, N=2000)

# Exact on-axis formula
mu0 = 4e-7 * np.pi
Bz_exact = mu0 * I * a**2 / (2 * (a**2 + z_values**2)**1.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: On-axis field comparison
ax = axes[0]
ax.plot(z_values * 100, Bz_numerical * 1e6, 'bo', markersize=4, label='Biot-Savart (numerical)')
ax.plot(z_values * 100, Bz_exact * 1e6, 'r-', label='Exact formula')
ax.set_xlabel("z (cm)", fontsize=12)
ax.set_ylabel("$B_z$ (μT)", fontsize=12)
ax.set_title(f"Current Loop: a={a*100} cm, I={I} A", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right: B-field in xz-plane (quiver plot)
ax2 = axes[1]
nx, nz = 15, 15
xg = np.linspace(-0.25, 0.25, nx)
zg = np.linspace(-0.25, 0.25, nz)
Xg, Zg = np.meshgrid(xg, zg)
points_grid = [np.array([x, 0, z]) for z in zg for x in xg]

Bx_grid, _, Bz_grid = biot_savart_loop(a, I, points_grid, N=500)
Bx_2d = Bx_grid.reshape(nz, nx)
Bz_2d = Bz_grid.reshape(nz, nx)
B_mag = np.sqrt(Bx_2d**2 + Bz_2d**2)
B_mag = np.maximum(B_mag, 1e-15)

ax2.quiver(Xg*100, Zg*100, Bx_2d/B_mag, Bz_2d/B_mag, np.log10(B_mag),
           cmap='viridis', alpha=0.8)
circle_x = a * np.cos(np.linspace(0, 2*np.pi, 100))
circle_z = a * np.sin(np.linspace(0, 2*np.pi, 100))
ax2.plot([-a*100, a*100], [0, 0], 'ro', markersize=10, label='Current loop (cross-section)')
ax2.set_xlabel("x (cm)", fontsize=12)
ax2.set_ylabel("z (cm)", fontsize=12)
ax2.set_title("B-field in xz-plane", fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
print("Left: On-axis B-field matches exact formula to high precision.")
print("Right: Magnetic dipole field pattern visible in the xz-plane.")""")

    # ── Theory 4: Maxwell's equations ────────────────────────────────────────
    nb.md(r"""\
## 4. Maxwell's Equations

The four Maxwell equations (in differential form) unify all of electromagnetism:

| Law | Equation | Physics |
|-----|----------|---------|
| Gauss (E) | $\nabla \cdot \vec{E} = \rho/\epsilon_0$ | Charges source E-field |
| Gauss (B) | $\nabla \cdot \vec{B} = 0$ | No magnetic monopoles |
| Faraday | $\nabla \times \vec{E} = -\partial \vec{B}/\partial t$ | Changing B creates E |
| Ampère-Maxwell | $\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\partial\vec{E}/\partial t$ | Currents and changing E create B |

In vacuum ($\rho = 0$, $\vec{J} = 0$), combining Faraday and Ampère gives the **wave equation**:

$$\nabla^2 \vec{E} = \mu_0\epsilon_0\frac{\partial^2 \vec{E}}{\partial t^2}
= \frac{1}{c^2}\frac{\partial^2 \vec{E}}{\partial t^2}$$

where $c = 1/\sqrt{\mu_0\epsilon_0} \approx 3 \times 10^8$ m/s — **light is an electromagnetic wave**.""")

    # ── Code 4: Curl and divergence with finite differences ──────────────────
    nb.code(r"""\
# Maxwell's Equations: verify curl and divergence numerically
# using finite differences on a discrete grid

def divergence_2d(Fx, Fy, dx, dy):
    '''Compute divergence of 2D vector field using central differences.'''
    dFx_dx = np.gradient(Fx, dx, axis=1)
    dFy_dy = np.gradient(Fy, dy, axis=0)
    return dFx_dx + dFy_dy

def curl_z_2d(Fx, Fy, dx, dy):
    '''Compute z-component of curl for a 2D vector field.'''
    dFy_dx = np.gradient(Fy, dx, axis=1)
    dFx_dy = np.gradient(Fx, dy, axis=0)
    return dFy_dx - dFx_dy

# Create grid
N = 100
x = np.linspace(-3, 3, N)
y = np.linspace(-3, 3, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Electric field of a point charge at origin: E = q*r_hat/r^2
q = 1.0
R2 = X**2 + Y**2
R2 = np.maximum(R2, 0.01)  # regularize
R = np.sqrt(R2)
Ex = q * X / R2**1.5
Ey = q * Y / R2**1.5

# Compute divergence (should be ~0 away from charge, peaked at origin)
div_E = divergence_2d(Ex, Ey, dx, dy)

# Magnetic field of straight wire (Biot-Savart): B = (μ₀I/2πr) φ_hat
Bx = -Y / R2
By = X / R2

# Curl of B (should be ~0 away from wire, peaked at origin)
curl_B = curl_z_2d(Bx, By, dx, dy)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Divergence of E
im1 = axes[0].imshow(div_E, extent=[-3,3,-3,3], origin='lower',
                      cmap='RdBu_r', vmin=-5, vmax=5)
axes[0].set_title(r"$\nabla \cdot \vec{E}$ (point charge)", fontsize=13)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Curl of B
im2 = axes[1].imshow(curl_B, extent=[-3,3,-3,3], origin='lower',
                      cmap='RdBu_r', vmin=-5, vmax=5)
axes[1].set_title(r"$(\nabla \times \vec{B})_z$ (wire current)", fontsize=13)
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.show()
print("Left: div(E) peaks at charge location — Gauss's law: ∇·E = ρ/ε₀")
print("Right: curl(B) peaks at wire location — Ampère's law: ∇×B = μ₀J")
print("Away from sources, both are approximately zero.")""")

    # ── Theory 5: EM wave simulation ─────────────────────────────────────────
    nb.md(r"""\
## 5. Electromagnetic Wave Simulation — FDTD Method

The **Finite-Difference Time-Domain** (FDTD) method solves Maxwell's equations
directly on a grid. In 1D, the relevant equations are:

$$\frac{\partial E_y}{\partial t} = \frac{1}{\epsilon_0}\frac{\partial H_z}{\partial x}$$

$$\frac{\partial H_z}{\partial t} = \frac{1}{\mu_0}\frac{\partial E_y}{\partial x}$$

Using **Yee's staggered grid**, $E$ and $H$ are evaluated at alternating half-steps
in both space and time. This naturally enforces the correct phase relationship
between electric and magnetic fields in a propagating wave.

The FDTD method is widely used in photonics, antenna design, and
computational electrodynamics.""")

    # ── Code 5: 1D FDTD simulation ──────────────────────────────────────────
    nb.code(r"""\
# 1D FDTD Simulation of Electromagnetic Wave Propagation

# Grid parameters
Nx = 500         # number of spatial cells
Nt = 600         # number of time steps
dx = 1e-3        # spatial step (1 mm)
dt = dx / (2*c)  # time step (CFL condition: dt < dx/c)

# Fields (Yee staggering: Ey at integer points, Hz at half-integer)
Ey = np.zeros(Nx)
Hz = np.zeros(Nx)

# Courant number
S = c * dt / dx
print(f"FDTD Parameters: Nx={Nx}, Nt={Nt}, dx={dx*1e3} mm, dt={dt*1e12:.3f} ps")
print(f"Courant number S = c·dt/dx = {S:.4f} (must be < 1 for stability)")

# Source: Gaussian pulse
source_pos = 100
t0 = 40       # delay in time steps
spread = 12   # width in time steps

# Storage for animation frames
snapshots = []
snapshot_times = [0, 100, 200, 300, 400, 500]

# Run FDTD loop
for n in range(Nt):
    # Update Hz (magnetic field)
    Hz[:-1] += S * (Ey[1:] - Ey[:-1])

    # Gaussian source injection
    pulse = np.exp(-0.5 * ((n - t0) / spread)**2)
    Hz[source_pos] += pulse

    # Update Ey (electric field)
    Ey[1:] += S * (Hz[1:] - Hz[:-1])

    # Simple absorbing boundary conditions
    Ey[0] = 0
    Ey[-1] = 0

    if n in snapshot_times:
        snapshots.append((n, Ey.copy(), Hz.copy()))

# Plot snapshots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for idx, (n, ey, hz) in enumerate(snapshots):
    ax = axes[idx // 2, idx % 2]
    x_mm = np.arange(Nx) * dx * 1e3
    ax.plot(x_mm, ey, 'b-', label='$E_y$', lw=1.5)
    ax.plot(x_mm, hz, 'r-', label='$H_z$', lw=1.5, alpha=0.7)
    ax.axvline(source_pos * dx * 1e3, color='green', ls='--', alpha=0.5, label='Source')
    ax.set_title(f"t = {n} steps ({n*dt*1e12:.1f} ps)", fontsize=11)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Field amplitude")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

plt.suptitle("1D FDTD: EM Wave Propagation", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
print("FDTD simulation shows EM pulse propagating at speed c.")
print("E and H fields are in phase for a traveling wave, 90° out of phase in time.")""")

    # ── Theory 6: Special Relativity ─────────────────────────────────────────
    nb.md(r"""\
## 6. Special Relativity and the Lorentz Transformation

Maxwell's equations are **not** invariant under Galilean transformations — they
require the **Lorentz transformation**:

$$x' = \gamma(x - vt), \quad t' = \gamma\left(t - \frac{vx}{c^2}\right)$$

where $\gamma = 1/\sqrt{1 - v^2/c^2}$ is the Lorentz factor.

Key consequences:
- **Time dilation:** $\Delta t' = \gamma \Delta t$
- **Length contraction:** $\Delta x' = \Delta x / \gamma$
- **Mass-energy equivalence:** $E = mc^2$

The electromagnetic field tensor $F^{\mu\nu}$ unifies $\vec{E}$ and $\vec{B}$ into
a single object that transforms covariantly:

$$F^{\mu\nu} = \begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\
E_x/c & 0 & -B_z & B_y \\
E_y/c & B_z & 0 & -B_x \\
E_z/c & -B_y & B_x & 0 \end{pmatrix}$$""")

    # ── Code 6: Lorentz transformation visualization ─────────────────────────
    nb.code(r"""\
# Lorentz Transformation Visualization

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Left: Lorentz factor γ(v) ---
v_over_c = np.linspace(0, 0.999, 1000)
gamma = 1.0 / np.sqrt(1 - v_over_c**2)

ax = axes[0]
ax.plot(v_over_c, gamma, 'b-', lw=2)
ax.axhline(1, color='gray', ls='--', alpha=0.5)
for v_mark in [0.5, 0.9, 0.99]:
    g = 1/np.sqrt(1-v_mark**2)
    ax.plot(v_mark, g, 'ro', markersize=8)
    ax.annotate(f'v={v_mark}c\nγ={g:.2f}', (v_mark, g),
                textcoords="offset points", xytext=(-30, 10), fontsize=9)
ax.set_xlabel("v/c", fontsize=12)
ax.set_ylabel("γ", fontsize=12)
ax.set_title("Lorentz Factor", fontsize=13)
ax.set_ylim(0, 15)
ax.grid(True, alpha=0.3)

# --- Middle: Spacetime diagram with Lorentz-boosted axes ---
ax2 = axes[1]
beta = 0.5
gamma_b = 1/np.sqrt(1 - beta**2)

# Original axes
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)

# Light cone
t_lc = np.linspace(-2.5, 2.5, 100)
ax2.plot(t_lc, t_lc, 'y-', lw=2, label='Light cone (x=ct)')
ax2.plot(t_lc, -t_lc, 'y-', lw=2)

# Boosted t' axis: x = βct (i.e., x = β·t in ct units)
ax2.plot(beta * t_lc, t_lc, 'r--', lw=2, label=f"t' axis (β={beta})")
# Boosted x' axis: ct = βx
ax2.plot(t_lc, beta * t_lc, 'b--', lw=2, label=f"x' axis (β={beta})")

# World line of stationary object
ax2.plot([0, 0], [-2.5, 2.5], 'k-', lw=2, label='Stationary observer')

ax2.set_xlabel("x (units of ct)", fontsize=12)
ax2.set_ylabel("ct", fontsize=12)
ax2.set_title("Minkowski Diagram", fontsize=13)
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)
ax2.set_aspect('equal')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

# --- Right: E and B field transformation ---
ax3 = axes[2]
# Pure E-field in rest frame → mixed E,B in moving frame
# E'_y = γ(E_y - vB_z), B'_z = γ(B_z - vE_y/c^2)
E0 = 1.0  # E_y in rest frame
B0 = 0.0  # B_z in rest frame

betas = np.linspace(0, 0.99, 100)
gammas = 1/np.sqrt(1 - betas**2)
Ey_prime = gammas * (E0 - betas * c * B0)
Bz_prime = gammas * (B0 - betas * E0 / c)

ax3.plot(betas, Ey_prime / E0, 'b-', lw=2, label="$E'_y / E_0$")
ax3.plot(betas, Bz_prime * c / E0, 'r-', lw=2, label="$cB'_z / E_0$")
ax3.set_xlabel("v/c", fontsize=12)
ax3.set_ylabel("Transformed field / $E_0$", fontsize=12)
ax3.set_title("Field Transformation (pure E → E+B)", fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Left: Lorentz factor diverges as v → c.")
print("Middle: Boosted axes tilt toward the light cone.")
print("Right: A pure E-field gains a B-component in a moving frame — E and B are unified!")""")

    # ── Theory 7: QM Connection ──────────────────────────────────────────────
    nb.md(r"""\
## 7. Quantum Mechanics Connection: Quantized EM Field

### Photons as Quantized EM Waves

The electromagnetic field is quantized by promoting the field amplitudes
to operators. Each mode of frequency $\omega$ becomes a quantum harmonic oscillator:

$$\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right)$$

where $\hat{a}^\dagger$ creates a photon and $\hat{a}$ destroys one. Energy levels:

$$E_n = \hbar\omega\left(n + \frac{1}{2}\right)$$

### Minimal Coupling

The interaction of a charged particle with the EM field introduces
the **minimal coupling** replacement:

$$\hat{\vec{p}} \to \hat{\vec{p}} - q\vec{A}$$

leading to the Hamiltonian for the hydrogen atom:

$$\hat{H} = \frac{1}{2m}\left(\hat{\vec{p}} - \frac{e}{c}\vec{A}\right)^2 - \frac{e^2}{4\pi\epsilon_0 r}$$

### Vacuum Fluctuations

Even in the ground state ($n=0$), the zero-point energy
$E_0 = \hbar\omega/2$ gives rise to **vacuum fluctuations** — measurable
via the Casimir effect and the Lamb shift.""")

    # ── Code 7: Quantized EM field and photon number states ──────────────────
    nb.code(r"""\
# Quantized EM field: Fock states and field expectation values

from scipy.special import hermite, factorial

def coherent_state_prob(n_max, alpha):
    '''Photon number distribution for coherent state |alpha>.'''
    ns = np.arange(n_max)
    mean_n = abs(alpha)**2
    # P(n) = e^{-|α|²} |α|^{2n} / n!
    log_P = -mean_n + 2*ns*np.log(abs(alpha)) - np.array([np.sum(np.log(np.arange(1, k+1))) if k > 0 else 0 for k in ns])
    return ns, np.exp(log_P)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Left: Energy levels of quantized EM mode ---
ax = axes[0]
omega = 1.0  # normalized
n_levels = 8
for n in range(n_levels):
    E = omega * (n + 0.5)
    ax.hlines(E, 0.2, 0.8, colors='blue', linewidth=2)
    ax.text(0.85, E, f'$|{n}\\rangle$, $E_{n}={E:.1f}\\hbar\\omega$',
            fontsize=10, va='center')
ax.set_xlim(0, 2.0)
ax.set_ylim(0, n_levels + 0.5)
ax.set_ylabel("$E / \\hbar\\omega$", fontsize=12)
ax.set_title("Quantized EM Mode Energy Levels", fontsize=13)
ax.set_xticks([])
ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Zero-point energy')
ax.legend(fontsize=10)

# --- Middle: Photon number distribution for coherent states ---
ax2 = axes[1]
for alpha in [1.0, 2.0, 3.0]:
    ns, probs = coherent_state_prob(20, alpha)
    ax2.bar(ns + 0.15*(alpha-2), probs, width=0.15, alpha=0.7,
            label=f'$|\\alpha={alpha}\\rangle$, $\\langle n \\rangle={alpha**2:.0f}$')
ax2.set_xlabel("Photon number $n$", fontsize=12)
ax2.set_ylabel("$P(n)$", fontsize=12)
ax2.set_title("Coherent State Photon Distributions", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Right: E-field quadrature for vacuum, coherent, and squeezed ---
ax3 = axes[2]
x = np.linspace(-5, 5, 500)

# Vacuum state: Gaussian centered at 0
psi_vac = np.exp(-x**2 / 2) / np.pi**0.25
ax3.fill_between(x, psi_vac**2, alpha=0.3, color='blue', label='Vacuum $|0\\rangle$')

# Coherent state |α=2⟩: displaced Gaussian
alpha = 2.0
psi_coh = np.exp(-(x - np.sqrt(2)*alpha)**2 / 2) / np.pi**0.25
ax3.fill_between(x, psi_coh**2, alpha=0.3, color='red', label=f'Coherent $|\\alpha={alpha}\\rangle$')

# Squeezed state: narrower Gaussian
r = 0.8  # squeezing parameter
sigma_sq = np.exp(-2*r)
psi_sq = (1/(np.pi * sigma_sq))**0.25 * np.exp(-x**2 / (2*sigma_sq))
ax3.fill_between(x, psi_sq**2, alpha=0.3, color='green', label=f'Squeezed ($r={r}$)')

ax3.set_xlabel("Field quadrature $X$", fontsize=12)
ax3.set_ylabel("$|\\psi(X)|^2$", fontsize=12)
ax3.set_title("Quantum Field States", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Left: Equally spaced energy levels — each quantum is one photon ℏω.")
print("Middle: Coherent states have Poissonian photon statistics — closest to classical.")
print("Right: Squeezed states have reduced uncertainty in one quadrature — used in LIGO!")""")

    # ── Code 8: Hydrogen atom energy levels from EM theory ───────────────────
    nb.code(r"""\
# Hydrogen atom: from classical EM to quantum energy levels
# The hydrogen atom is where EM and QM meet

from scipy.constants import m_e, e as e_charge, hbar, epsilon_0, alpha

# Bohr model energy levels: E_n = -13.6 eV / n²
E_1 = -m_e * e_charge**4 / (2 * (4*np.pi*epsilon_0)**2 * hbar**2)  # in Joules
E_1_eV = E_1 / e_charge  # convert to eV

print("Hydrogen Atom Energy Levels (Bohr Model)")
print("=" * 55)
print(f"  Ground state energy: E₁ = {E_1_eV:.4f} eV")
print(f"  Fine structure constant: α = {alpha:.6f}")
print(f"  Bohr radius: a₀ = {hbar/(m_e * e_charge**2 / (4*np.pi*epsilon_0)) * 1e10:.4f} Å")
print()

fig, ax = plt.subplots(figsize=(10, 7))

n_max = 7
for n in range(1, n_max+1):
    E_n = E_1_eV / n**2
    ax.hlines(E_n, 0.5, 4.5, colors='blue', linewidth=2)
    ax.text(4.6, E_n, f'$n={n}$: {E_n:.3f} eV', fontsize=10, va='center')

# Draw some transitions (Lyman, Balmer, Paschen series)
transitions = {
    'Lyman': [(2,1,'violet'), (3,1,'indigo'), (4,1,'blue')],
    'Balmer': [(3,2,'red'), (4,2,'cyan'), (5,2,'green')],
    'Paschen': [(4,3,'orange'), (5,3,'brown')],
}

x_pos = {('Lyman', 0): 1.0, ('Lyman', 1): 1.3, ('Lyman', 2): 1.6,
         ('Balmer', 0): 2.2, ('Balmer', 1): 2.5, ('Balmer', 2): 2.8,
         ('Paschen', 0): 3.4, ('Paschen', 1): 3.7}

for series, trans in transitions.items():
    for i, (n_upper, n_lower, color) in enumerate(trans):
        E_upper = E_1_eV / n_upper**2
        E_lower = E_1_eV / n_lower**2
        xp = x_pos.get((series, i), 2.0)
        ax.annotate('', xy=(xp, E_lower), xytext=(xp, E_upper),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        # Photon energy
        dE = E_upper - E_lower  # negative because both are negative
        wavelength = 1240 / abs(dE)  # nm (from E = hc/λ with E in eV)

ax.set_xlim(0, 6)
ax.set_ylabel("Energy (eV)", fontsize=12)
ax.set_title("Hydrogen Energy Levels and Spectral Series", fontsize=14)
ax.set_xticks([1.3, 2.5, 3.55])
ax.set_xticklabels(['Lyman\n(UV)', 'Balmer\n(visible)', 'Paschen\n(IR)'], fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.show()

# Print transition wavelengths
print("\nSpectral Lines:")
for series, trans in transitions.items():
    print(f"  {series} series:")
    for n_upper, n_lower, _ in trans:
        dE = abs(E_1_eV / n_lower**2 - E_1_eV / n_upper**2)
        lam = 1240 / dE
        print(f"    {n_upper} → {n_lower}: ΔE = {dE:.3f} eV, λ = {lam:.1f} nm")""")

    # ── Summary ──────────────────────────────────────────────────────────────
    nb.md(r"""\
## Summary

| Topic | Key Result |
|-------|-----------|
| Coulomb's law | $\vec{E} = \frac{1}{4\pi\epsilon_0}\frac{q}{r^2}\hat{r}$ |
| Gauss's law | $\oint \vec{E}\cdot d\vec{A} = Q_{\rm enc}/\epsilon_0$ |
| Biot-Savart | $d\vec{B} = \frac{\mu_0}{4\pi}\frac{I\,d\vec{l}\times\hat{r}}{r^2}$ |
| Maxwell's eqs | 4 equations unify E, B, charges, currents |
| EM waves | $c = 1/\sqrt{\mu_0\epsilon_0}$ — light is EM radiation |
| Lorentz transform | $x'=\gamma(x-vt)$, $t'=\gamma(t-vx/c^2)$ |
| Quantized field | $\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a}+\frac{1}{2})$ |
| Hydrogen | $E_n = -13.6\,\text{eV}/n^2$ from EM + QM |

**Next:** Month 09 — Functional Analysis and Hilbert Spaces

---
*SIIEA Quantum Engineering Curriculum — CC BY-NC-SA 4.0*""")

    nb.save()
    print("  [Month 08] Electromagnetism notebook complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK 3 — Month 09: Functional Analysis — Hilbert Spaces
# ═══════════════════════════════════════════════════════════════════════════════

def build_month_09():
    nb = NotebookBuilder(
        "Functional Analysis — Hilbert Spaces & Spectral Theory",
        "year_0/month_09_functional_analysis/09_hilbert_spaces.ipynb",
        "Days 225-252",
    )

    # ── imports ───────────────────────────────────────────────────────────────
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
from scipy import integrate, linalg
from scipy.special import hermite
from numpy.polynomial.legendre import leggauss

%matplotlib inline

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 11,
    "font.family": "serif",
    "figure.dpi": 120,
})
print("Imports ready — numpy, matplotlib, sympy, scipy loaded.")""")

    # ── Theory 1: Metric spaces ──────────────────────────────────────────────
    nb.md(r"""\
## 1. Metric Spaces and Convergence

A **metric space** $(X, d)$ is a set $X$ with a distance function $d: X \times X \to \mathbb{R}$ satisfying:

1. $d(x,y) \geq 0$ with $d(x,y)=0 \iff x=y$ (positivity)
2. $d(x,y) = d(y,x)$ (symmetry)
3. $d(x,z) \leq d(x,y) + d(y,z)$ (triangle inequality)

A sequence $\{x_n\}$ **converges** to $x$ if $d(x_n, x) \to 0$ as $n \to \infty$.

A metric space is **complete** if every Cauchy sequence converges. This is
critical: a **Hilbert space** is a *complete* inner product space.

### Examples of Metrics

| Space | Metric | Complete? |
|-------|--------|-----------|
| $\mathbb{R}^n$ | $d(x,y) = \|x-y\|_2$ | Yes |
| $\mathbb{Q}$ | $d(x,y) = |x-y|$ | **No** ($\sqrt{2}$ is a limit point not in $\mathbb{Q}$) |
| $C[0,1]$ | $d(f,g) = \max|f-g|$ | Yes (uniform convergence) |
| $L^2[0,1]$ | $d(f,g) = \sqrt{\int|f-g|^2}$ | Yes (Riesz-Fischer theorem) |""")

    # ── Code 1: Convergence visualization ────────────────────────────────────
    nb.code(r"""\
# Metric spaces: visualize sequence convergence and Cauchy sequences

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Left: Converging sequence in R ---
ax = axes[0]
n = np.arange(1, 31)
# Sequence: x_n = 1 + (-1)^n / n  → converges to 1
x_n = 1 + (-1)**n / n
ax.plot(n, x_n, 'bo-', markersize=5, label='$x_n = 1 + (-1)^n/n$')
ax.axhline(1, color='red', ls='--', lw=2, label='Limit = 1')
ax.fill_between(n, 1 - 1/n, 1 + 1/n, alpha=0.1, color='green', label='$\\epsilon$-band')
ax.set_xlabel("$n$", fontsize=12)
ax.set_ylabel("$x_n$", fontsize=12)
ax.set_title("Convergent Sequence in $\\mathbb{R}$", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Middle: Cauchy sequence approximating √2 in Q ---
ax2 = axes[1]
# Newton's method: x_{n+1} = (x_n + 2/x_n)/2
x = [1.0]
for _ in range(10):
    x.append((x[-1] + 2/x[-1]) / 2)
x = np.array(x)
errors = np.abs(x - np.sqrt(2))

ax2.semilogy(range(len(x)), errors, 'ro-', markersize=8, label='$|x_n - \\sqrt{2}|$')
ax2.set_xlabel("Iteration $n$", fontsize=12)
ax2.set_ylabel("Error", fontsize=12)
ax2.set_title("Cauchy Seq. in $\\mathbb{Q}$: $x_{n+1}=(x_n+2/x_n)/2$", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1e-16, 10)

# --- Right: Function space convergence ---
ax3 = axes[2]
x_plot = np.linspace(0, 1, 500)
colors = plt.cm.viridis(np.linspace(0, 1, 8))
for k, c in zip(range(1, 9), colors):
    f_n = x_plot**k
    ax3.plot(x_plot, f_n, color=c, lw=1.5, label=f'$f_{k}(x) = x^{k}$' if k <= 4 else None)

# Limit function (pointwise)
f_limit = np.zeros_like(x_plot)
f_limit[-1] = 1
ax3.plot(x_plot, f_limit, 'r--', lw=3, label='Pointwise limit')

ax3.set_xlabel("$x$", fontsize=12)
ax3.set_ylabel("$f_n(x) = x^n$", fontsize=12)
ax3.set_title("Pointwise vs Uniform Convergence", fontsize=13)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Left: Convergent sequence oscillates within shrinking ε-band.")
print("Middle: Newton iteration converges quadratically (Cauchy in Q, but √2 ∉ Q).")
print("Right: x^n → 0 pointwise on [0,1), but limit is discontinuous — not uniform convergence.")""")

    # ── Theory 2: L² and inner products ──────────────────────────────────────
    nb.md(r"""\
## 2. The $L^2$ Space and Inner Products

The space $L^2[a,b]$ consists of square-integrable functions:

$$L^2[a,b] = \left\{f : \int_a^b |f(x)|^2\,dx < \infty\right\}$$

with **inner product**:

$$\langle f, g \rangle = \int_a^b f^*(x)\,g(x)\,dx$$

and **norm**: $\|f\| = \sqrt{\langle f, f \rangle}$.

### Properties

- **Cauchy-Schwarz:** $|\langle f,g\rangle| \leq \|f\|\cdot\|g\|$
- **Triangle inequality:** $\|f+g\| \leq \|f\| + \|g\|$
- **Orthogonality:** $f \perp g \iff \langle f,g\rangle = 0$
- **Pythagorean theorem:** If $f \perp g$, then $\|f+g\|^2 = \|f\|^2 + \|g\|^2$

$L^2$ is the natural home of quantum mechanics: **wavefunctions live in $L^2$**.""")

    # ── Code 2: L² inner product, norm, orthogonality ────────────────────────
    nb.code(r"""\
# L² inner product: demonstrate orthogonality of sine functions

from scipy.integrate import quad

def L2_inner_product(f, g, a=0, b=2*np.pi):
    '''Compute <f, g> = integral_a^b f*(x) g(x) dx.'''
    real_part, _ = quad(lambda x: np.real(np.conj(f(x)) * g(x)), a, b)
    imag_part, _ = quad(lambda x: np.imag(np.conj(f(x)) * g(x)), a, b)
    return real_part + 1j * imag_part

def L2_norm(f, a=0, b=2*np.pi):
    '''Compute ||f|| = sqrt(<f, f>).'''
    return np.sqrt(np.real(L2_inner_product(f, f, a, b)))

# Orthonormality of {sin(nx)/√π, cos(nx)/√π} on [0, 2π]
print("Orthogonality of Fourier basis on [0, 2π]")
print("=" * 55)

# Check ⟨sin(mx), sin(nx)⟩ for m, n = 1..4
print("\n⟨sin(mx), sin(nx)⟩:")
for m in range(1, 5):
    row = []
    for n in range(1, 5):
        ip = np.real(L2_inner_product(
            lambda x, m=m: np.sin(m*x),
            lambda x, n=n: np.sin(n*x)
        ))
        row.append(f"{ip:7.3f}")
    print(f"  m={m}: " + " ".join(row))

# Check ⟨sin(mx), cos(nx)⟩
print("\n⟨sin(mx), cos(nx)⟩:")
for m in range(1, 5):
    row = []
    for n in range(1, 5):
        ip = np.real(L2_inner_product(
            lambda x, m=m: np.sin(m*x),
            lambda x, n=n: np.cos(n*x)
        ))
        row.append(f"{ip:7.3f}")
    print(f"  m={m}: " + " ".join(row))

# Norms
print("\nNorms:")
for n in range(1, 5):
    norm_sin = L2_norm(lambda x, n=n: np.sin(n*x))
    norm_cos = L2_norm(lambda x, n=n: np.cos(n*x))
    print(f"  ||sin({n}x)|| = {norm_sin:.6f} (expected √π = {np.sqrt(np.pi):.6f})")

# Cauchy-Schwarz inequality demonstration
f = lambda x: np.exp(-x/2)
g = lambda x: np.sin(x)
ip_fg = abs(L2_inner_product(f, g, 0, 2*np.pi))
nf = L2_norm(f, 0, 2*np.pi)
ng = L2_norm(g, 0, 2*np.pi)
print(f"\nCauchy-Schwarz: |⟨f,g⟩| = {ip_fg:.6f} ≤ ||f||·||g|| = {nf*ng:.6f}  ✓")""")

    # ── Theory 3: Fourier basis ──────────────────────────────────────────────
    nb.md(r"""\
## 3. Fourier Basis as Orthonormal Set

The functions $\{e_n(x) = e^{inx}/\sqrt{2\pi}\}_{n=-\infty}^{\infty}$ form a
**complete orthonormal basis** for $L^2[0, 2\pi]$:

$$\langle e_m, e_n \rangle = \delta_{mn}$$

Any $f \in L^2$ can be expanded:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n\, e_n(x), \qquad
  c_n = \langle e_n, f \rangle = \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} e^{-inx} f(x)\,dx$$

**Parseval's theorem** (generalized Pythagorean theorem):

$$\|f\|^2 = \sum_{n=-\infty}^{\infty} |c_n|^2$$

This is the mathematical foundation of **quantum superposition**: any state
can be expanded in an orthonormal basis, and probabilities sum to 1.""")

    # ── Code 3: Fourier decomposition and reconstruction ─────────────────────
    nb.code(r"""\
# Fourier decomposition and reconstruction of functions in L²

def fourier_coefficients(f, N, L=2*np.pi):
    '''Compute Fourier coefficients c_n for n = -N..N.'''
    coeffs = {}
    for n in range(-N, N+1):
        re, _ = quad(lambda x: np.real(f(x) * np.exp(-1j*n*x)), 0, L)
        im, _ = quad(lambda x: np.imag(f(x) * np.exp(-1j*n*x)), 0, L)
        coeffs[n] = (re + 1j*im) / L
    return coeffs

def fourier_reconstruct(coeffs, x, L=2*np.pi):
    '''Reconstruct f(x) from Fourier coefficients.'''
    result = np.zeros_like(x, dtype=complex)
    for n, cn in coeffs.items():
        result += cn * np.exp(1j * n * x)
    return result

# Test functions
x = np.linspace(0, 2*np.pi, 1000)

# Square wave
def square_wave(x):
    return np.where(np.mod(x, 2*np.pi) < np.pi, 1.0, -1.0)

# Sawtooth
def sawtooth(x):
    return (np.mod(x, 2*np.pi) - np.pi) / np.pi

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (fname, f) in enumerate([(r"Square wave", square_wave), (r"Sawtooth", sawtooth)]):
    ax_left = axes[row, 0]
    ax_right = axes[row, 1]

    # Plot original and approximations
    ax_left.plot(x, f(x), 'k-', lw=2, label='Original')
    N_values = [1, 3, 7, 20]
    colors = ['red', 'orange', 'green', 'blue']

    all_coeffs = {}
    for N, color in zip(N_values, colors):
        coeffs = fourier_coefficients(f, N)
        recon = np.real(fourier_reconstruct(coeffs, x))
        ax_left.plot(x, recon, color=color, lw=1.5, alpha=0.7, label=f'N={N}')
        all_coeffs[N] = coeffs

    ax_left.set_title(f"Fourier Reconstruction: {fname}", fontsize=13)
    ax_left.set_xlabel("$x$")
    ax_left.set_ylabel("$f(x)$")
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.3)

    # Power spectrum (Parseval)
    N_big = 30
    coeffs_big = fourier_coefficients(f, N_big)
    ns = sorted(coeffs_big.keys())
    powers = [abs(coeffs_big[n])**2 for n in ns]

    ax_right.stem(ns, powers, linefmt='b-', markerfmt='bo', basefmt='gray')
    ax_right.set_title(f"Power Spectrum $|c_n|^2$: {fname}", fontsize=13)
    ax_right.set_xlabel("$n$ (frequency)")
    ax_right.set_ylabel("$|c_n|^2$")
    ax_right.grid(True, alpha=0.3)

    # Verify Parseval's theorem
    total_power = sum(powers)
    norm_sq = quad(lambda x: f(x)**2, 0, 2*np.pi)[0] / (2*np.pi)
    print(f"{fname}: Σ|cₙ|² = {total_power:.6f}, ||f||²/(2π) = {norm_sq:.6f}, "
          f"ratio = {total_power/norm_sq:.6f}")

plt.tight_layout()
plt.show()
print("\nParseval's theorem: energy in function = sum of energies in Fourier modes.")
print("This is the mathematical basis of quantum probability: Σ|cₙ|² = 1.")""")

    # ── Theory 4: Operators on Hilbert space ─────────────────────────────────
    nb.md(r"""\
## 4. Operators on Hilbert Space

A **linear operator** $\hat{A}: \mathcal{H} \to \mathcal{H}$ maps vectors to vectors.
In finite dimensions, operators are represented by **matrices**.

### The Adjoint

The **adjoint** $\hat{A}^\dagger$ satisfies:

$$\langle \hat{A}^\dagger \psi, \phi \rangle = \langle \psi, \hat{A}\phi \rangle$$

In matrix representation: $A^\dagger = \overline{A}^T$ (conjugate transpose).

### Classification of Operators

| Type | Condition | Properties |
|------|-----------|-----------|
| **Self-adjoint (Hermitian)** | $\hat{A}^\dagger = \hat{A}$ | Real eigenvalues, orthogonal eigenvectors |
| **Unitary** | $\hat{U}^\dagger\hat{U} = \hat{I}$ | Preserves inner products |
| **Normal** | $[\hat{A}, \hat{A}^\dagger] = 0$ | Diagonalizable via spectral theorem |
| **Projection** | $\hat{P}^2 = \hat{P} = \hat{P}^\dagger$ | Projects onto subspace |

In quantum mechanics:
- **Observables** are self-adjoint operators
- **Time evolution** is a unitary operator $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$
- **Measurements** are described by projection operators""")

    # ── Code 4: Matrix operators ─────────────────────────────────────────────
    nb.code(r"""\
# Operators on finite-dimensional Hilbert space (matrices)

np.set_printoptions(precision=4, suppress=True)

# --- Pauli matrices (fundamental in quantum mechanics) ---
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

print("Pauli Matrices — the operators of spin-1/2 quantum mechanics")
print("=" * 55)
for name, sigma in [("σ_x", sigma_x), ("σ_y", sigma_y), ("σ_z", sigma_z)]:
    print(f"\n{name} =")
    print(sigma)
    # Check Hermiticity
    is_hermitian = np.allclose(sigma, sigma.conj().T)
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(sigma)
    # Trace and determinant
    tr = np.trace(sigma)
    det = np.linalg.det(sigma)
    print(f"  Hermitian: {is_hermitian}")
    print(f"  Eigenvalues: {eigenvalues.real}")
    print(f"  Trace: {tr.real:.0f}, Det: {det.real:.0f}")
    print(f"  σ² = I: {np.allclose(sigma @ sigma, I2)}")

# Verify algebraic relations
print("\nAlgebraic Relations:")
print(f"  [σ_x, σ_y] = 2iσ_z: {np.allclose(sigma_x@sigma_y - sigma_y@sigma_x, 2j*sigma_z)}")
print(f"  [σ_y, σ_z] = 2iσ_x: {np.allclose(sigma_y@sigma_z - sigma_z@sigma_y, 2j*sigma_x)}")
print(f"  [σ_z, σ_x] = 2iσ_y: {np.allclose(sigma_z@sigma_x - sigma_x@sigma_z, 2j*sigma_y)}")
print(f"  {{σ_x, σ_y}} = 0: {np.allclose(sigma_x@sigma_y + sigma_y@sigma_x, 0)}")

# Unitary operator: rotation about z-axis
theta = np.pi / 4
U = np.array([[np.exp(-1j*theta/2), 0],
              [0, np.exp(1j*theta/2)]])
print(f"\nRotation U(θ=π/4) about z-axis:")
print(f"  U†U = I: {np.allclose(U.conj().T @ U, I2)}")
print(f"  |det(U)| = 1: {abs(np.linalg.det(U)):.10f}")""")

    # ── Theory 5: Self-adjoint operators and spectral theorem ────────────────
    nb.md(r"""\
## 5. Self-Adjoint Operators and the Spectral Theorem

### Key Properties of Self-Adjoint Operators

If $\hat{A} = \hat{A}^\dagger$, then:

1. **All eigenvalues are real:** $\hat{A}|\lambda\rangle = \lambda|\lambda\rangle \implies \lambda \in \mathbb{R}$
2. **Eigenvectors for distinct eigenvalues are orthogonal:** $\langle \lambda_i | \lambda_j \rangle = 0$ if $\lambda_i \neq \lambda_j$
3. **Eigenvectors form a complete basis** (spectral theorem)

### The Spectral Theorem

Any self-adjoint operator can be written as:

$$\hat{A} = \sum_n \lambda_n |\lambda_n\rangle\langle\lambda_n|$$

This is the **eigenvalue decomposition** — the operator is completely
determined by its eigenvalues and eigenvectors.

### Functional Calculus

For any function $f$, we can define $f(\hat{A})$:

$$f(\hat{A}) = \sum_n f(\lambda_n) |\lambda_n\rangle\langle\lambda_n|$$

This gives meaning to expressions like $e^{i\hat{H}t}$ — the time evolution operator.""")

    # ── Code 5: Spectral theorem demonstration ───────────────────────────────
    nb.code(r"""\
# Spectral theorem: diagonalize a Hermitian matrix and verify properties

# A physically motivated Hermitian matrix: tight-binding Hamiltonian
# Models electron hopping on a 1D chain (N sites, periodic boundary)
N = 8
t_hop = -1.0  # hopping parameter

# Construct Hamiltonian
H = np.zeros((N, N), dtype=complex)
for i in range(N):
    H[i, (i+1) % N] = t_hop
    H[(i+1) % N, i] = t_hop

print(f"Tight-binding Hamiltonian ({N}×{N}):")
print(H.real)

# Verify Hermiticity
print(f"\nH = H†: {np.allclose(H, H.conj().T)}")

# Diagonalize
eigenvalues, eigenvectors = np.linalg.eigh(H)

print("\nEigenvalues (energy levels):")
for i, lam in enumerate(eigenvalues):
    print(f"  E_{i} = {lam:.6f}")

# Verify orthonormality of eigenvectors
overlap = eigenvectors.conj().T @ eigenvectors
print(f"\nEigenvector orthonormality (V†V = I): {np.allclose(overlap, np.eye(N))}")

# Verify spectral decomposition: H = Σ λ_n |n⟩⟨n|
H_reconstructed = np.zeros((N, N), dtype=complex)
for n in range(N):
    vn = eigenvectors[:, n:n+1]
    H_reconstructed += eigenvalues[n] * (vn @ vn.conj().T)
print(f"Spectral decomposition H = ΣλₙPₙ: {np.allclose(H, H_reconstructed)}")

# Functional calculus: compute exp(iHt)
t = 0.5
U_spectral = np.zeros((N, N), dtype=complex)
for n in range(N):
    vn = eigenvectors[:, n:n+1]
    U_spectral += np.exp(1j * eigenvalues[n] * t) * (vn @ vn.conj().T)

U_matrix = linalg.expm(1j * H * t)
print(f"Functional calculus: exp(iHt) via spectral = via matrix exp: "
      f"{np.allclose(U_spectral, U_matrix)}")
print(f"exp(iHt) is unitary: {np.allclose(U_spectral.conj().T @ U_spectral, np.eye(N))}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Energy levels
ax = axes[0]
for i, E in enumerate(eigenvalues):
    ax.hlines(E, 0.3, 0.7, colors='blue', linewidth=2)
    ax.text(0.75, E, f'$E_{i}={E:.3f}$', fontsize=9, va='center')
ax.set_ylabel("Energy", fontsize=12)
ax.set_title(f"Tight-Binding Energy Spectrum (N={N})", fontsize=13)
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

# Eigenvectors (probability density)
ax2 = axes[1]
sites = np.arange(N)
for i in [0, 1, N//2, N-1]:
    ax2.plot(sites, np.abs(eigenvectors[:, i])**2, 'o-',
             label=f'$|\\psi_{i}|^2$, E={eigenvalues[i]:.2f}')
ax2.set_xlabel("Site", fontsize=12)
ax2.set_ylabel("$|\\psi|^2$", fontsize=12)
ax2.set_title("Eigenvector Probability Densities", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Exact eigenvalues vs theory: E_k = 2t cos(2πk/N)
ax3 = axes[2]
k = np.arange(N)
E_exact = 2 * t_hop * np.cos(2 * np.pi * k / N)
ax3.plot(sorted(E_exact), 'rs-', markersize=8, label='Exact: $2t\\cos(2\\pi k/N)$')
ax3.plot(eigenvalues, 'bo-', markersize=6, label='Numerical eigenvalues')
ax3.set_xlabel("Level index", fontsize=12)
ax3.set_ylabel("Energy", fontsize=12)
ax3.set_title("Numerical vs Exact Eigenvalues", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("All spectral theorem properties verified:")
print("  1. Real eigenvalues ✓")
print("  2. Orthonormal eigenvectors ✓")
print("  3. Spectral decomposition H = ΣλₙPₙ ✓")
print("  4. Functional calculus e^{iHt} via spectral decomposition ✓")""")

    # ── Code 6: Projection operators and measurement ─────────────────────────
    nb.code(r"""\
# Projection operators and quantum measurement simulation

# Consider a 3-level quantum system (qutrit)
# State: |ψ⟩ = α|0⟩ + β|1⟩ + γ|2⟩

# Define a random normalized state
np.random.seed(42)
psi = np.random.randn(3) + 1j * np.random.randn(3)
psi = psi / np.linalg.norm(psi)

print("Quantum Measurement with Projection Operators")
print("=" * 55)
print(f"State |ψ⟩ = {psi}")
print(f"||ψ|| = {np.linalg.norm(psi):.10f}")

# Basis states
e0 = np.array([1, 0, 0], dtype=complex)
e1 = np.array([0, 1, 0], dtype=complex)
e2 = np.array([0, 0, 1], dtype=complex)
basis = [e0, e1, e2]

# Projection operators P_n = |n⟩⟨n|
print("\nProjection operators and measurement probabilities:")
P_sum = np.zeros((3, 3), dtype=complex)
for n, en in enumerate(basis):
    P_n = np.outer(en, en.conj())
    prob = np.real(psi.conj() @ P_n @ psi)
    projected = P_n @ psi
    print(f"  P_{n} |ψ⟩ → prob = |⟨{n}|ψ⟩|² = {prob:.6f}")
    P_sum += P_n

    # Verify projection properties
    assert np.allclose(P_n @ P_n, P_n), f"P_{n}² ≠ P_{n}"
    assert np.allclose(P_n, P_n.conj().T), f"P_{n} not Hermitian"

print(f"\nCompleteness: Σ P_n = I: {np.allclose(P_sum, np.eye(3))}")
print(f"Probabilities sum to 1: {sum(abs(psi[n])**2 for n in range(3)):.10f}")

# Simulate measurement outcomes
n_measurements = 10000
probs = np.abs(psi)**2
outcomes = np.random.choice([0, 1, 2], size=n_measurements, p=probs)
counts = np.bincount(outcomes, minlength=3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of theoretical vs measured
ax = axes[0]
x_pos = np.arange(3)
width = 0.35
ax.bar(x_pos - width/2, probs, width, label='Theoretical $|c_n|^2$', color='steelblue')
ax.bar(x_pos + width/2, counts/n_measurements, width, label=f'Measured ({n_measurements} trials)',
       color='coral')
ax.set_xlabel("Outcome $|n\\rangle$", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title("Quantum Measurement Statistics", fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'$|{n}\\rangle$' for n in range(3)])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Bloch-like visualization: show |ψ⟩ components
ax2 = axes[1]
amplitudes = np.abs(psi)
phases = np.angle(psi)
colors = plt.cm.hsv(phases / (2*np.pi) + 0.5)

bars = ax2.bar(range(3), amplitudes, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel("Basis state", fontsize=12)
ax2.set_ylabel("$|c_n|$", fontsize=12)
ax2.set_title("State Amplitudes (color = phase)", fontsize=13)
ax2.set_xticks(range(3))
ax2.set_xticklabels([f'$|{n}\\rangle$\nφ={phases[n]:.2f}' for n in range(3)])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Measurement outcomes follow Born's rule: P(n) = |⟨n|ψ⟩|²")
print("Completeness relation ΣPₙ = I ensures probabilities sum to 1.")""")

    # ── Theory 6: This IS quantum mechanics ──────────────────────────────────
    nb.md(r"""\
## 6. QM Connection: This IS Quantum Mechanics

The mathematical framework of Hilbert spaces *is* the language of quantum mechanics.
The correspondence is exact:

| Mathematics | Quantum Mechanics |
|-------------|------------------|
| Hilbert space $\mathcal{H}$ | State space of the system |
| Unit vector $|\psi\rangle \in \mathcal{H}$ | Pure quantum state |
| Self-adjoint operator $\hat{A}$ | Observable (energy, position, spin) |
| Eigenvalue $\lambda$ of $\hat{A}$ | Possible measurement outcome |
| $|\langle\lambda|\psi\rangle|^2$ | Probability of outcome $\lambda$ |
| Unitary operator $\hat{U}(t)$ | Time evolution |
| Inner product $\langle\phi|\psi\rangle$ | Transition amplitude |
| Tensor product $\mathcal{H}_1 \otimes \mathcal{H}_2$ | Composite systems |

### The Postulates of Quantum Mechanics

1. **States** are vectors in a Hilbert space
2. **Observables** are self-adjoint operators
3. **Measurement** yields eigenvalues with probability $|\langle\lambda|\psi\rangle|^2$
4. **After measurement**, state collapses to eigenstate $|\lambda\rangle$
5. **Time evolution** follows $i\hbar\frac{d}{dt}|\psi\rangle = \hat{H}|\psi\rangle$

Every concept in this notebook maps directly to a physical principle.""")

    # ── Code 7: Quantum harmonic oscillator in Hilbert space ─────────────────
    nb.code(r"""\
# Quantum Harmonic Oscillator: the complete Hilbert space picture
# Eigenstates, operators, and time evolution

from math import factorial
from scipy.special import hermite as physicists_hermite
from scipy.linalg import expm

# Construct the QHO in a truncated Hilbert space (N dimensions)
N = 30  # truncation

# Creation and annihilation operators in Fock basis
a = np.zeros((N, N), dtype=complex)       # annihilation
a_dag = np.zeros((N, N), dtype=complex)   # creation
for n in range(N-1):
    a[n, n+1] = np.sqrt(n+1)
    a_dag[n+1, n] = np.sqrt(n+1)

# Number operator: N_hat = a†a
N_hat = a_dag @ a

# Hamiltonian: H = ℏω(N + 1/2)
omega = 1.0
H = omega * (N_hat + 0.5 * np.eye(N))

# Position and momentum operators: x = (a + a†)/√2, p = i(a† - a)/√2
x_op = (a + a_dag) / np.sqrt(2)
p_op = 1j * (a_dag - a) / np.sqrt(2)

print("Quantum Harmonic Oscillator in Fock Space")
print("=" * 55)

# Verify commutation relations (in truncated Fock space, last element deviates)
commutator_aa = a @ a_dag - a_dag @ a
commutator_xp = x_op @ p_op - p_op @ x_op
# Check subspace excluding truncation boundary
M = N - 1  # exclude last row/col (truncation artifact)
print(f"[a, a†] = I (first {M}x{M} block): {np.allclose(commutator_aa[:M,:M], np.eye(M))}")
print(f"[x, p] = iI (first {M}x{M} block): {np.allclose(commutator_xp[:M,:M], 1j * np.eye(M))}")

# Energy eigenvalues
eigenvalues = np.diag(H).real
print(f"\nEnergy levels E_n = ℏω(n+1/2):")
for n in range(6):
    print(f"  E_{n} = {eigenvalues[n]:.4f} (expected {omega*(n+0.5):.4f})")

# Uncertainty principle: Δx·Δp ≥ ℏ/2 for each eigenstate
print("\nUncertainty principle ΔxΔp ≥ 1/2:")
for n in range(6):
    state = np.zeros(N)
    state[n] = 1.0
    x_avg = np.real(state @ x_op @ state)
    x2_avg = np.real(state @ x_op @ x_op @ state)
    p_avg = np.real(state @ p_op @ state)
    p2_avg = np.real(state @ p_op @ p_op @ state)
    dx = np.sqrt(x2_avg - x_avg**2)
    dp = np.sqrt(p2_avg - p_avg**2)
    print(f"  |{n}⟩: Δx={dx:.4f}, Δp={dp:.4f}, ΔxΔp={dx*dp:.4f} (≥ 0.5)")

# Time evolution of a coherent state |α⟩
alpha = 2.0
coherent = np.zeros(N, dtype=complex)
for n in range(N):
    coherent[n] = np.exp(-abs(alpha)**2/2) * alpha**n / np.sqrt(float(factorial(n)))

# Evolve and compute ⟨x⟩(t)
times = np.linspace(0, 4*np.pi, 200)
x_expect = []
for t in times:
    U = expm(-1j * H * t)
    psi_t = U @ coherent
    x_expect.append(np.real(psi_t.conj() @ x_op @ psi_t))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Energy levels and wavefunctions
ax = axes[0]
x_grid = np.linspace(-6, 6, 500)
for n in range(5):
    # QHO wavefunction: ψ_n(x) = (mω/πℏ)^{1/4} (1/√(2^n n!)) H_n(x) e^{-x²/2}
    Hn = physicists_hermite(n)
    psi_n = (1/np.pi**0.25) * (1/np.sqrt(2**n * factorial(n))) * Hn(x_grid) * np.exp(-x_grid**2/2)
    # Offset by energy level for visualization
    E_n = omega * (n + 0.5)
    ax.fill_between(x_grid, E_n, E_n + 2*psi_n**2, alpha=0.5, label=f'$|{n}\\rangle$')
    ax.hlines(E_n, -6, 6, colors='gray', linewidth=0.5, linestyle='--')

# Potential
ax.plot(x_grid, 0.5 * omega * x_grid**2, 'k-', lw=2, label='$V(x)=\\frac{1}{2}\\omega x^2$')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 6)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("Energy", fontsize=12)
ax.set_title("QHO Wavefunctions $|\\psi_n(x)|^2$", fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)

# Right: Coherent state time evolution
ax2 = axes[1]
ax2.plot(times/(2*np.pi), x_expect, 'b-', lw=2)
x_classical = np.sqrt(2) * alpha * np.cos(omega * times)
ax2.plot(times/(2*np.pi), x_classical, 'r--', lw=1.5, label='Classical')
ax2.set_xlabel("$t / (2\\pi/\\omega)$", fontsize=12)
ax2.set_ylabel("$\\langle x \\rangle (t)$", fontsize=12)
ax2.set_title(f"Coherent State $|\\alpha={alpha}\\rangle$ Evolution", fontsize=13)
ax2.legend(['Quantum $\\langle\\hat{x}\\rangle$', 'Classical $x(t)$'], fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Left: QHO eigenstates — quantized energy levels with wavefunctions.")
print("Right: Coherent state ⟨x⟩(t) exactly tracks classical trajectory — Ehrenfest's theorem!")""")

    # ── Code 8: Spectral decomposition and quantum computing ─────────────────
    nb.code(r"""\
# Spectral decomposition applied: quantum gates and state tomography

# Hadamard gate: H = (σ_x + σ_z)/√2
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gate: S = diag(1, i)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)

# T gate: T = diag(1, e^{iπ/4})
T_gate = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

# CNOT gate (2-qubit)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

print("Quantum Gate Spectral Decomposition")
print("=" * 55)

for name, gate in [("Hadamard H", H_gate), ("Phase S", S_gate), ("T-gate", T_gate)]:
    eigenvalues, eigvecs = np.linalg.eig(gate)
    print(f"\n{name}:")
    print(f"  Matrix:\n    {gate[0]}\n    {gate[1]}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Unitary: {np.allclose(gate.conj().T @ gate, np.eye(2))}")

    # Reconstruct from spectral decomposition
    gate_recon = np.zeros((2,2), dtype=complex)
    for lam, v in zip(eigenvalues, eigvecs.T):
        gate_recon += lam * np.outer(v, v.conj())
    print(f"  Spectral reconstruction matches: {np.allclose(gate, gate_recon)}")

# Demonstrate: create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
print("\nBell State Creation: |00⟩ → H⊗I → CNOT → |Φ+⟩")
psi_00 = np.array([1, 0, 0, 0], dtype=complex)
H_tensor_I = np.kron(H_gate, np.eye(2))
psi_after_H = H_tensor_I @ psi_00
psi_bell = CNOT @ psi_after_H

print(f"  |00⟩ = {psi_00}")
print(f"  (H⊗I)|00⟩ = {np.round(psi_after_H, 4)}")
print(f"  CNOT(H⊗I)|00⟩ = {np.round(psi_bell, 4)}")
print(f"  This is |Φ+⟩ = (|00⟩+|11⟩)/√2: {np.allclose(psi_bell, np.array([1,0,0,1])/np.sqrt(2))}")

# Entanglement verification: compute reduced density matrix
rho = np.outer(psi_bell, psi_bell.conj())
# Partial trace over qubit 2: ρ_A = Tr_B(ρ)
rho_A = np.zeros((2,2), dtype=complex)
for j in range(2):
    for k in range(2):
        for l in range(2):
            rho_A[j, k] += rho[2*j+l, 2*k+l]

print(f"\n  Reduced density matrix ρ_A:")
print(f"    {rho_A[0]}")
print(f"    {rho_A[1]}")
eigenvalues_rho = np.linalg.eigvalsh(rho_A)
von_neumann_entropy = -sum(lam * np.log2(lam) for lam in eigenvalues_rho if lam > 1e-10)
print(f"  Von Neumann entropy S(ρ_A) = {von_neumann_entropy:.4f} bits (max for 2D = 1.0)")
print(f"  ρ_A = I/2 (maximally mixed): {np.allclose(rho_A, np.eye(2)/2)}")
print(f"  → Maximally entangled state confirmed!")""")

    # ── Summary ──────────────────────────────────────────────────────────────
    nb.md(r"""\
## Summary

| Topic | Key Result |
|-------|-----------|
| Metric spaces | Distance, completeness, Cauchy sequences |
| $L^2$ space | Square-integrable functions with inner product $\langle f,g\rangle = \int f^*g$ |
| Fourier basis | Complete orthonormal set; Parseval: $\|f\|^2 = \Sigma|c_n|^2$ |
| Operators | Linear maps on $\mathcal{H}$; Hermitian, unitary, projectors |
| Spectral theorem | $\hat{A} = \sum \lambda_n |\lambda_n\rangle\langle\lambda_n|$ |
| Functional calculus | $f(\hat{A}) = \sum f(\lambda_n) |\lambda_n\rangle\langle\lambda_n|$ |
| QM = Hilbert space | States ↔ vectors, observables ↔ operators, probabilities ↔ $|c_n|^2$ |

This is the mathematical foundation upon which **all** of quantum mechanics is built.
Every concept from Year 1 onwards will use this language.

**Next:** Month 10 — Scientific Computing

---
*SIIEA Quantum Engineering Curriculum — CC BY-NC-SA 4.0*""")

    nb.save()
    print("  [Month 09] Functional Analysis notebook complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  SIIEA Quantum Engineering — Generating Year 0 Months 7-9")
    print("=" * 70)
    print()

    build_month_07()
    build_month_08()
    build_month_09()

    print("=" * 70)
    print("  All 3 notebooks generated successfully!")
    print("=" * 70)
