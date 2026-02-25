# Day 6: Review and Computational Lab — Python for Limits

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Python Setup & Basics |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Numerical Limits Lab |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Visualization |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Set up a Python environment for mathematical computing
2. Use NumPy for numerical limit computation
3. Visualize limits graphically with Matplotlib
4. Understand numerical vs. analytical limits
5. Explore the epsilon-delta definition computationally

---

## 🖥️ Environment Setup

### Required Software
1. **Python 3.9+** - Download from python.org
2. **Jupyter Notebook** - For interactive coding
3. **NumPy** - Numerical computing
4. **Matplotlib** - Visualization
5. **SymPy** - Symbolic mathematics

### Installation Commands
```bash
# Create virtual environment (recommended)
python -m venv quantum_env
source quantum_env/bin/activate  # On Windows: quantum_env\Scripts\activate

# Install packages
pip install numpy matplotlib sympy jupyter scipy

# Start Jupyter
jupyter notebook
```

---

## 📖 Lab 1: Numerical Exploration of Limits

### 1.1 Creating Tables of Values

```python
import numpy as np

def explore_limit(f, c, approaches='both', num_points=10):
    """
    Explore a limit numerically by creating tables of values.
    
    Parameters:
    - f: function to evaluate
    - c: point to approach
    - approaches: 'left', 'right', or 'both'
    - num_points: number of points in table
    """
    
    if approaches in ['left', 'both']:
        print(f"\nApproaching {c} from the LEFT:")
        print("-" * 40)
        print(f"{'x':^15} | {'f(x)':^20}")
        print("-" * 40)
        
        for i in range(num_points, 0, -1):
            x = c - 10**(-i)
            try:
                fx = f(x)
                print(f"{x:^15.10f} | {fx:^20.10f}")
            except:
                print(f"{x:^15.10f} | {'undefined':^20}")
    
    if approaches in ['right', 'both']:
        print(f"\nApproaching {c} from the RIGHT:")
        print("-" * 40)
        print(f"{'x':^15} | {'f(x)':^20}")
        print("-" * 40)
        
        for i in range(num_points, 0, -1):
            x = c + 10**(-i)
            try:
                fx = f(x)
                print(f"{x:^15.10f} | {fx:^20.10f}")
            except:
                print(f"{x:^15.10f} | {'undefined':^20}")

# Example 1: lim(x→2) (x²-4)/(x-2)
f1 = lambda x: (x**2 - 4) / (x - 2)
explore_limit(f1, 2)

# Example 2: lim(x→0) sin(x)/x
f2 = lambda x: np.sin(x) / x
explore_limit(f2, 0)

# Example 3: lim(x→0) sin(1/x) - limit does not exist
f3 = lambda x: np.sin(1/x)
explore_limit(f3, 0)
```

### 1.2 Expected Output Analysis

For Example 1, you should see values approaching 4 from both sides.
For Example 2, values approach 1 from both sides.
For Example 3, values oscillate wildly — demonstrating no limit exists!

---

## 📖 Lab 2: Visualizing Limits

### 2.1 Basic Limit Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_limit(f, c, L, xrange=(-2, 6), title="Limit Visualization"):
    """
    Visualize a limit with the target point highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x values, excluding the point c
    x_left = np.linspace(xrange[0], c - 0.01, 1000)
    x_right = np.linspace(c + 0.01, xrange[1], 1000)
    
    # Plot function
    try:
        y_left = f(x_left)
        y_right = f(x_right)
        ax.plot(x_left, y_left, 'b-', linewidth=2, label='f(x)')
        ax.plot(x_right, y_right, 'b-', linewidth=2)
    except:
        pass
    
    # Mark the limit point
    ax.plot(c, L, 'ro', markersize=10, markerfacecolor='white', 
            markeredgewidth=2, label=f'Limit = {L}')
    
    # Add reference lines
    ax.axhline(y=L, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=c, color='g', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'limit_visualization_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

# Example: Removable discontinuity
f = lambda x: (x**2 - 4) / (x - 2)
visualize_limit(f, c=2, L=4, xrange=(0, 4), 
                title="Removable Discontinuity at x=2")
```

### 2.2 Epsilon-Delta Visualization

```python
def visualize_epsilon_delta(f, c, L, epsilon, delta, xrange=(-1, 5)):
    """
    Visualize the epsilon-delta definition of a limit.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot function
    x = np.linspace(xrange[0], xrange[1], 1000)
    x = x[x != c]  # Remove point c
    y = f(x)
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    
    # Epsilon band (horizontal)
    ax.axhspan(L - epsilon, L + epsilon, alpha=0.2, color='red', 
               label=f'ε-band: ({L-epsilon:.2f}, {L+epsilon:.2f})')
    ax.axhline(y=L, color='red', linestyle='-', linewidth=1)
    ax.axhline(y=L+epsilon, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=L-epsilon, color='red', linestyle='--', linewidth=1)
    
    # Delta band (vertical)
    ax.axvspan(c - delta, c + delta, alpha=0.2, color='green',
               label=f'δ-band: ({c-delta:.2f}, {c+delta:.2f})')
    ax.axvline(x=c, color='green', linestyle='-', linewidth=1)
    ax.axvline(x=c-delta, color='green', linestyle='--', linewidth=1)
    ax.axvline(x=c+delta, color='green', linestyle='--', linewidth=1)
    
    # Mark the limit point
    ax.plot(c, L, 'ko', markersize=10, markerfacecolor='white',
            markeredgewidth=2, zorder=5)
    
    # Annotations
    ax.annotate(f'L = {L}', xy=(xrange[1]-0.5, L), fontsize=10)
    ax.annotate(f'c = {c}', xy=(c, xrange[0]+0.2), fontsize=10, ha='center')
    ax.annotate(f'ε = {epsilon}', xy=(xrange[1]-0.5, L+epsilon/2), fontsize=10)
    ax.annotate(f'δ = {delta}', xy=(c+delta/2, xrange[0]+0.5), fontsize=10)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'Epsilon-Delta Visualization: ε={epsilon}, δ={delta}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xrange)
    
    plt.tight_layout()
    plt.savefig('epsilon_delta_visualization.png', dpi=150)
    plt.show()

# Example: f(x) = 2x + 1, limit as x→2 is 5
f = lambda x: 2*x + 1
visualize_epsilon_delta(f, c=2, L=5, epsilon=0.5, delta=0.25, xrange=(0, 4))
```

---

## 📖 Lab 3: Symbolic Computation with SymPy

### 3.1 Analytical Limits

```python
from sympy import *

# Define symbolic variable
x = Symbol('x')

# Compute limits symbolically
print("Symbolic Limit Computation")
print("=" * 50)

# Example 1: Polynomial
expr1 = x**3 - 2*x + 1
limit1 = limit(expr1, x, 2)
print(f"\nlim(x→2) x³ - 2x + 1 = {limit1}")

# Example 2: Rational with indeterminate form
expr2 = (x**2 - 4) / (x - 2)
limit2 = limit(expr2, x, 2)
print(f"lim(x→2) (x² - 4)/(x - 2) = {limit2}")

# Example 3: Trigonometric
expr3 = sin(x) / x
limit3 = limit(expr3, x, 0)
print(f"lim(x→0) sin(x)/x = {limit3}")

# Example 4: One-sided limits
expr4 = 1 / x
limit4_left = limit(expr4, x, 0, '-')
limit4_right = limit(expr4, x, 0, '+')
print(f"\nlim(x→0⁻) 1/x = {limit4_left}")
print(f"lim(x→0⁺) 1/x = {limit4_right}")

# Example 5: Limit at infinity
expr5 = (3*x**2 + 2*x - 1) / (5*x**2 - x + 4)
limit5 = limit(expr5, x, oo)
print(f"\nlim(x→∞) (3x² + 2x - 1)/(5x² - x + 4) = {limit5}")

# Example 6: L'Hôpital's Rule (preview)
expr6 = (exp(x) - 1) / x
limit6 = limit(expr6, x, 0)
print(f"lim(x→0) (eˣ - 1)/x = {limit6}")
```

### 3.2 Continuity Analysis

```python
from sympy import *

x = Symbol('x')

def analyze_continuity(f_expr, point):
    """
    Analyze continuity of a function at a point.
    """
    print(f"\nAnalyzing continuity of f(x) = {f_expr} at x = {point}")
    print("-" * 50)
    
    # Check if defined
    try:
        f_at_point = f_expr.subs(x, point)
        if f_at_point.is_finite:
            print(f"1. f({point}) = {f_at_point} ✓ (defined)")
        else:
            print(f"1. f({point}) = {f_at_point} ✗ (undefined or infinite)")
            return False
    except:
        print(f"1. f({point}) is undefined ✗")
        return False
    
    # Check limit exists
    left_limit = limit(f_expr, x, point, '-')
    right_limit = limit(f_expr, x, point, '+')
    
    if left_limit == right_limit:
        two_sided_limit = left_limit
        print(f"2. lim(x→{point}) f(x) = {two_sided_limit} ✓ (limit exists)")
    else:
        print(f"2. Left limit = {left_limit}, Right limit = {right_limit} ✗ (limit DNE)")
        return False
    
    # Check limit equals function value
    if two_sided_limit == f_at_point:
        print(f"3. lim(x→{point}) f(x) = f({point}) ✓")
        print(f"\nConclusion: f is CONTINUOUS at x = {point}")
        return True
    else:
        print(f"3. lim(x→{point}) f(x) ≠ f({point}) ✗")
        print(f"\nConclusion: f has a REMOVABLE DISCONTINUITY at x = {point}")
        return False

# Test cases
f1 = x**2 + 1
analyze_continuity(f1, 2)

f2 = (x**2 - 1) / (x - 1)
analyze_continuity(f2, 1)

f3 = 1 / x
analyze_continuity(f3, 0)
```

---

## 📖 Lab 4: Exploring the Squeeze Theorem

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_squeeze_theorem():
    """
    Visualize the squeeze theorem with x²sin(1/x).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: The squeeze
    ax1 = axes[0]
    x = np.linspace(-0.5, 0.5, 10000)
    x = x[x != 0]
    
    y = x**2 * np.sin(1/x)
    y_upper = x**2
    y_lower = -x**2
    
    ax1.plot(x, y, 'b-', linewidth=0.5, label='$x^2 \sin(1/x)$')
    ax1.plot(x, y_upper, 'r--', linewidth=2, label='$x^2$ (upper bound)')
    ax1.plot(x, y_lower, 'g--', linewidth=2, label='$-x^2$ (lower bound)')
    ax1.plot(0, 0, 'ko', markersize=10, label='Limit = 0')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Squeeze Theorem: $x^2 \sin(1/x)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.3, 0.3)
    
    # Right plot: Zoom in
    ax2 = axes[1]
    x_zoom = np.linspace(-0.1, 0.1, 10000)
    x_zoom = x_zoom[x_zoom != 0]
    
    y_zoom = x_zoom**2 * np.sin(1/x_zoom)
    y_upper_zoom = x_zoom**2
    y_lower_zoom = -x_zoom**2
    
    ax2.plot(x_zoom, y_zoom, 'b-', linewidth=0.5)
    ax2.plot(x_zoom, y_upper_zoom, 'r--', linewidth=2)
    ax2.plot(x_zoom, y_lower_zoom, 'g--', linewidth=2)
    ax2.plot(0, 0, 'ko', markersize=10)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Zoomed View Near x = 0')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('squeeze_theorem_visualization.png', dpi=150)
    plt.show()

visualize_squeeze_theorem()
```

---

## 📖 Lab 5: Interactive Epsilon-Delta Explorer

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_epsilon_delta():
    """
    Create an interactive epsilon-delta demonstration.
    Run this in a local Jupyter notebook with %matplotlib notebook
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Function: f(x) = 2x + 1, limit as x→2 is 5
    f = lambda x: 2*x + 1
    c, L = 2, 5
    
    x = np.linspace(0, 4, 1000)
    line, = ax.plot(x, f(x), 'b-', linewidth=2)
    
    # Initial epsilon and delta
    epsilon_init = 0.5
    delta_init = 0.25
    
    # Create shaded regions
    eps_band = ax.axhspan(L - epsilon_init, L + epsilon_init, alpha=0.2, color='red')
    delta_band = ax.axvspan(c - delta_init, c + delta_init, alpha=0.2, color='green')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Interactive ε-δ Definition')
    ax.grid(True, alpha=0.3)
    
    # Add sliders
    ax_eps = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_delta = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    s_eps = Slider(ax_eps, 'Epsilon', 0.1, 2.0, valinit=epsilon_init)
    s_delta = Slider(ax_delta, 'Delta', 0.05, 1.0, valinit=delta_init)
    
    def update(val):
        eps = s_eps.val
        delta = s_delta.val
        
        # Update shaded regions (recreate them)
        ax.collections.clear()
        ax.axhspan(L - eps, L + eps, alpha=0.2, color='red')
        ax.axvspan(c - delta, c + delta, alpha=0.2, color='green')
        
        # Check if current delta works for current epsilon
        # For f(x) = 2x + 1, we need delta = epsilon/2
        required_delta = eps / 2
        if delta <= required_delta:
            ax.set_title(f'ε = {eps:.2f}, δ = {delta:.2f} ✓ (δ ≤ ε/2 = {required_delta:.2f})', 
                        color='green')
        else:
            ax.set_title(f'ε = {eps:.2f}, δ = {delta:.2f} ✗ (need δ ≤ {required_delta:.2f})', 
                        color='red')
        
        fig.canvas.draw_idle()
    
    s_eps.on_changed(update)
    s_delta.on_changed(update)
    
    plt.show()

# Note: Run this in an interactive environment
# interactive_epsilon_delta()
```

---

## ✅ Lab Completion Checklist

- [ ] Python environment set up and working
- [ ] Completed Lab 1: Numerical exploration
- [ ] Completed Lab 2: Basic visualization
- [ ] Completed Lab 3: Symbolic computation
- [ ] Completed Lab 4: Squeeze theorem visualization
- [ ] Saved all generated images
- [ ] Experimented with additional functions

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Explores the limit $\lim_{x \to 0} \frac{e^x - 1}{x}$ numerically and symbolically
2. Visualizes a function with a jump discontinuity
3. Creates an epsilon-delta visualization for $\lim_{x \to 3} (x + 1) = 4$
4. Uses the squeeze theorem to prove $\lim_{x \to 0} x \cos(1/x) = 0$

Save your notebook as `Week1_Lab_Report.ipynb`

---

## 🔜 Tomorrow: Rest and Review

Day 7 is a lighter day for consolidation and preparation for Week 2.

---

*"The purpose of computing is insight, not numbers."*
— Richard Hamming
