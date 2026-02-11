# Day 270: Interactive Visualization with Plotly

## Schedule Overview
**Date**: Week 39, Day 4 (Thursday)
**Duration**: 7 hours
**Theme**: Building Interactive Quantum State Explorers

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Plotly fundamentals, graph objects, express API |
| Afternoon | 2.5 hours | 3D plots, animations, widgets |
| Evening | 1.5 hours | Computational lab: Interactive quantum dashboard |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Create interactive plots using Plotly's Python interface
2. Build 3D visualizations with rotation and zoom
3. Add hover information displaying physical quantities
4. Create animated parameter sweeps
5. Build interactive dashboards for quantum state exploration
6. Export interactive visualizations as HTML files

---

## Core Content

### 1. Introduction to Plotly

Plotly provides interactive, publication-quality visualizations that work in browsers, Jupyter notebooks, and can be embedded in web applications.

#### Installation and Import

```python
# Installation (if needed)
# pip install plotly

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
```

#### Two APIs: Express vs Graph Objects

**Plotly Express** - Quick, high-level API:
```python
import plotly.express as px
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = px.line(x=x, y=y, title='Sine Wave')
fig.show()
```

**Graph Objects** - Full control:
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(title='Sine Wave', xaxis_title='x', yaxis_title='y')
fig.show()
```

### 2. Basic Interactive Plots

#### Line Plots with Hover Information

```python
import numpy as np
import plotly.graph_objects as go

# Wave function data
x = np.linspace(-5, 5, 200)
psi = np.exp(-x**2/2) / np.pi**0.25
prob = np.abs(psi)**2

fig = go.Figure()

# Wave function
fig.add_trace(go.Scatter(
    x=x, y=psi,
    mode='lines',
    name='ψ(x)',
    line=dict(color='blue', width=2),
    hovertemplate='x: %{x:.3f}<br>ψ: %{y:.4f}<extra></extra>'
))

# Probability density
fig.add_trace(go.Scatter(
    x=x, y=prob,
    mode='lines',
    name='|ψ|²',
    line=dict(color='red', width=2, dash='dash'),
    hovertemplate='x: %{x:.3f}<br>P: %{y:.4f}<extra></extra>'
))

fig.update_layout(
    title='Gaussian Wave Packet',
    xaxis_title='Position x',
    yaxis_title='Amplitude',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified'
)

fig.show()
# Save as HTML
fig.write_html('wavefunction_interactive.html')
```

#### Scatter Plots with Color Mapping

```python
# Phase space visualization
np.random.seed(42)
n_points = 1000
x = np.random.normal(0, 1, n_points)
p = np.random.normal(0, 1, n_points)
energy = 0.5 * (x**2 + p**2)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=p,
    mode='markers',
    marker=dict(
        size=6,
        color=energy,
        colorscale='Viridis',
        colorbar=dict(title='Energy'),
        showscale=True
    ),
    hovertemplate=(
        'x: %{x:.3f}<br>'
        'p: %{y:.3f}<br>'
        'E: %{marker.color:.3f}<extra></extra>'
    )
))

fig.update_layout(
    title='Quantum Phase Space Distribution',
    xaxis_title='Position x',
    yaxis_title='Momentum p',
    xaxis=dict(scaleanchor='y', scaleratio=1),
    height=600, width=600
)

fig.show()
```

### 3. 3D Interactive Plots

Plotly's 3D plots support rotation, zoom, and pan with mouse interaction.

#### 3D Surface Plots

```python
import numpy as np
import plotly.graph_objects as go

# 2D harmonic oscillator potential
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
V = 0.5 * (X**2 + Y**2)

fig = go.Figure(data=[
    go.Surface(
        x=X, y=Y, z=V,
        colorscale='Viridis',
        colorbar=dict(title='V(x,y)'),
        hovertemplate=(
            'x: %{x:.2f}<br>'
            'y: %{y:.2f}<br>'
            'V: %{z:.3f}<extra></extra>'
        )
    )
])

fig.update_layout(
    title='2D Harmonic Oscillator Potential',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='V(x,y)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    width=800, height=700
)

fig.show()
```

#### 3D Scatter Plots for Orbitals

```python
from scipy.special import sph_harm

def create_orbital_cloud(n, l, m, n_points=5000, r_max=15):
    """Generate point cloud for hydrogen orbital."""
    # Sample positions
    r = r_max * np.random.rand(n_points * 10)
    theta = np.arccos(2 * np.random.rand(n_points * 10) - 1)
    phi = 2 * np.pi * np.random.rand(n_points * 10)

    # Simplified radial part (for visualization)
    R = (r / n)**l * np.exp(-r / n)
    Y = sph_harm(m, l, phi, theta)
    psi = R * Y
    prob = np.abs(psi)**2 * r**2

    # Rejection sampling
    accept = np.random.rand(len(prob)) < prob / prob.max()
    r_acc = r[accept][:n_points]
    theta_acc = theta[accept][:n_points]
    phi_acc = phi[accept][:n_points]

    # Cartesian coordinates
    x = r_acc * np.sin(theta_acc) * np.cos(phi_acc)
    y = r_acc * np.sin(theta_acc) * np.sin(phi_acc)
    z = r_acc * np.cos(theta_acc)

    return x, y, z, r_acc

# Create 2p orbital
x, y, z, r = create_orbital_cloud(2, 1, 0, n_points=3000)

fig = go.Figure(data=[
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,  # Color by z to show orientation
            colorscale='RdBu',
            opacity=0.6
        ),
        hovertemplate=(
            'x: %{x:.2f}<br>'
            'y: %{y:.2f}<br>'
            'z: %{z:.2f}<extra></extra>'
        )
    )
])

fig.update_layout(
    title='Hydrogen 2p Orbital (m=0)',
    scene=dict(
        xaxis_title='x (a₀)',
        yaxis_title='y (a₀)',
        zaxis_title='z (a₀)',
        aspectmode='cube'
    ),
    width=800, height=700
)

fig.show()
```

### 4. Animations

Plotly supports frame-based animations for visualizing time evolution.

#### Basic Animation

```python
import numpy as np
import plotly.graph_objects as go

# Wave packet time evolution
x = np.linspace(-10, 10, 200)

# Create frames
frames = []
times = np.linspace(0, 4, 50)

for t in times:
    # Gaussian wave packet with momentum k0
    k0 = 2
    sigma = 1
    x0 = k0 * t  # Classical trajectory

    psi_real = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.cos(k0 * x - 0.5 * k0**2 * t)
    psi_imag = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.sin(k0 * x - 0.5 * k0**2 * t)
    prob = psi_real**2 + psi_imag**2

    frame = go.Frame(
        data=[
            go.Scatter(x=x, y=psi_real, name='Re[ψ]'),
            go.Scatter(x=x, y=psi_imag, name='Im[ψ]'),
            go.Scatter(x=x, y=prob, name='|ψ|²')
        ],
        name=f't={t:.2f}'
    )
    frames.append(frame)

# Initial frame
fig = go.Figure(
    data=[
        go.Scatter(x=x, y=frames[0].data[0].y, name='Re[ψ]',
                  line=dict(color='blue')),
        go.Scatter(x=x, y=frames[0].data[1].y, name='Im[ψ]',
                  line=dict(color='red')),
        go.Scatter(x=x, y=frames[0].data[2].y, name='|ψ|²',
                  line=dict(color='purple', width=2))
    ],
    frames=frames
)

# Add animation controls
fig.update_layout(
    title='Wave Packet Time Evolution',
    xaxis_title='Position x',
    yaxis_title='Amplitude',
    yaxis_range=[-1.2, 1.2],
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor='center',
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=50, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate'
                    )]
                )
            ]
        )
    ],
    sliders=[dict(
        active=0,
        yanchor='top',
        xanchor='left',
        currentvalue=dict(
            font=dict(size=12),
            prefix='t = ',
            visible=True,
            xanchor='right'
        ),
        steps=[
            dict(args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode='immediate'
            )], label=f'{t:.1f}', method='animate')
            for f, t in zip(frames, times)
        ]
    )]
)

fig.show()
fig.write_html('wave_packet_animation.html')
```

### 5. Subplots and Linked Views

```python
from plotly.subplots import make_subplots
import numpy as np

# Create linked subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Wave Function', 'Probability Density',
        'Position Distribution', 'Momentum Distribution'
    )
)

x = np.linspace(-5, 5, 200)
k = np.linspace(-5, 5, 200)

# Ground state
psi = np.exp(-x**2/2) / np.pi**0.25
prob = np.abs(psi)**2

# Momentum space (Fourier transform of Gaussian)
phi = np.exp(-k**2/2) / np.pi**0.25
prob_k = np.abs(phi)**2

# Add traces
fig.add_trace(
    go.Scatter(x=x, y=psi, name='ψ(x)', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=x, y=prob, name='|ψ(x)|²', line=dict(color='red'),
              fill='tozeroy'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=x, y=prob, name='P(x)', line=dict(color='green'),
              fill='tozeroy'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=k, y=prob_k, name='P(k)', line=dict(color='purple'),
              fill='tozeroy'),
    row=2, col=2
)

fig.update_layout(
    title='Harmonic Oscillator Ground State Analysis',
    height=700, width=900,
    showlegend=True
)

fig.update_xaxes(title_text='x', row=1, col=1)
fig.update_xaxes(title_text='x', row=1, col=2)
fig.update_xaxes(title_text='x', row=2, col=1)
fig.update_xaxes(title_text='k', row=2, col=2)

fig.show()
```

### 6. Interactive Widgets with Dash

For true interactivity with sliders and callbacks, use Plotly Dash:

```python
# Note: This requires running as a web application
# pip install dash

# Simple example structure (run in separate file):
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Quantum Harmonic Oscillator'),

    html.Label('Quantum Number n:'),
    dcc.Slider(id='n-slider', min=0, max=5, value=0, step=1,
               marks={i: str(i) for i in range(6)}),

    dcc.Graph(id='wavefunction-plot')
])

@app.callback(
    Output('wavefunction-plot', 'figure'),
    Input('n-slider', 'value')
)
def update_plot(n):
    from scipy.special import hermite
    from scipy.misc import factorial

    x = np.linspace(-5, 5, 500)
    xi = x
    prefactor = (1/np.pi)**0.25
    norm = 1 / np.sqrt(2**n * factorial(n))
    H = hermite(n)
    psi = prefactor * norm * H(xi) * np.exp(-xi**2/2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=psi, name=f'ψ_{n}(x)'))
    fig.add_trace(go.Scatter(x=x, y=np.abs(psi)**2, name=f'|ψ_{n}|²'))
    fig.update_layout(title=f'Quantum Number n = {n}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
"""
```

### 7. Alternative: ipywidgets in Jupyter

For Jupyter notebook interactivity:

```python
# In Jupyter notebook:
"""
from ipywidgets import interact, FloatSlider, IntSlider
import plotly.graph_objects as go
from IPython.display import display
import numpy as np

def plot_wavepacket(k0=2.0, sigma=1.0, t=0.0):
    x = np.linspace(-15, 15, 500)

    # Wave packet
    x0 = k0 * t
    psi_real = np.exp(-(x-x0)**2/(2*sigma**2)) * np.cos(k0*x - 0.5*k0**2*t)
    psi_imag = np.exp(-(x-x0)**2/(2*sigma**2)) * np.sin(k0*x - 0.5*k0**2*t)

    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=x, y=psi_real, name='Re[ψ]'))
    fig.add_trace(go.Scatter(x=x, y=psi_imag, name='Im[ψ]'))
    fig.add_trace(go.Scatter(x=x, y=psi_real**2 + psi_imag**2, name='|ψ|²'))
    fig.update_layout(yaxis_range=[-1.2, 1.2])
    fig.show()

interact(plot_wavepacket,
         k0=FloatSlider(min=0, max=5, step=0.1, value=2),
         sigma=FloatSlider(min=0.5, max=3, step=0.1, value=1),
         t=FloatSlider(min=0, max=5, step=0.1, value=0))
"""
```

---

## Quantum Mechanics Connection

### Interactive Bloch Sphere

The Bloch sphere represents single qubit states:
$$|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$$

```python
import numpy as np
import plotly.graph_objects as go

def create_bloch_sphere(theta=0, phi=0, show_state=True):
    """
    Create interactive Bloch sphere visualization.

    Parameters
    ----------
    theta : float
        Polar angle (0 to π)
    phi : float
        Azimuthal angle (0 to 2π)
    show_state : bool
        Whether to show the state vector
    """
    fig = go.Figure()

    # Sphere wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.3,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        hoverinfo='skip'
    ))

    # Axes
    for axis, color, name in [
        ([0, 0], [0, 0], [-1.3, 1.3], 'blue', '|0⟩↔|1⟩'),
        ([0, 0], [-1.3, 1.3], [0, 0], 'green', '|+⟩↔|-⟩'),
        ([-1.3, 1.3], [0, 0], [0, 0], 'red', '|+i⟩↔|-i⟩')
    ]:
        fig.add_trace(go.Scatter3d(
            x=axis[0] if isinstance(axis[0], list) else [axis[0], axis[0]],
            y=axis[1] if isinstance(axis[1], list) else [axis[1], axis[1]],
            z=axis[2] if isinstance(axis[2], list) else [axis[2], axis[2]],
            mode='lines',
            line=dict(color=color, width=2),
            name=name,
            showlegend=False
        ))

    # Basis state markers
    states = [
        (0, 0, 1, '|0⟩'),
        (0, 0, -1, '|1⟩'),
        (1, 0, 0, '|+⟩'),
        (-1, 0, 0, '|-⟩'),
        (0, 1, 0, '|+i⟩'),
        (0, -1, 0, '|-i⟩')
    ]
    for x, y, z, label in states:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=5, color='black'),
            text=[label],
            textposition='top center',
            showlegend=False
        ))

    # State vector
    if show_state:
        x_state = np.sin(theta) * np.cos(phi)
        y_state = np.sin(theta) * np.sin(phi)
        z_state = np.cos(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_state], y=[0, y_state], z=[0, z_state],
            mode='lines+markers',
            line=dict(color='purple', width=6),
            marker=dict(size=[0, 10], color='purple'),
            name='|ψ⟩'
        ))

        # State info
        alpha = np.cos(theta/2)
        beta = np.sin(theta/2) * np.exp(1j * phi)
        state_text = f'|ψ⟩ = {alpha:.3f}|0⟩ + ({beta.real:.3f}+{beta.imag:.3f}i)|1⟩'

        fig.add_annotation(
            x=0.5, y=1.1,
            xref='paper', yref='paper',
            text=state_text,
            showarrow=False,
            font=dict(size=14)
        )

    fig.update_layout(
        title='Bloch Sphere',
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title='x'),
            yaxis=dict(range=[-1.5, 1.5], title='y'),
            zaxis=dict(range=[-1.5, 1.5], title='z'),
            aspectmode='cube'
        ),
        width=700, height=700
    )

    return fig

# Example: Superposition state
fig = create_bloch_sphere(theta=np.pi/3, phi=np.pi/4)
fig.show()
fig.write_html('bloch_sphere.html')
```

### Energy Level Explorer

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_energy_level_explorer():
    """Interactive energy level diagram for different potentials."""

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Harmonic Oscillator',
            'Particle in Box',
            'Hydrogen Atom'
        )
    )

    # Number of levels to show
    n_levels = 8

    # Harmonic oscillator: E_n = (n + 1/2)ℏω
    E_ho = [(n + 0.5) for n in range(n_levels)]

    # Particle in box: E_n = n²E₁
    E_box = [n**2 for n in range(1, n_levels + 1)]

    # Hydrogen atom: E_n = -13.6/n² eV
    E_H = [-13.6/n**2 for n in range(1, n_levels + 1)]

    # Plot energy levels
    for i, (energies, col, color) in enumerate([
        (E_ho, 1, 'blue'),
        (E_box, 2, 'green'),
        (E_H, 3, 'red')
    ]):
        for j, E in enumerate(energies):
            fig.add_trace(
                go.Scatter(
                    x=[0.2, 0.8],
                    y=[E, E],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'n={j}' if col == 1 else f'n={j+1}',
                    hovertemplate=f'E = {E:.3f}<extra>n={j if col==1 else j+1}</extra>',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

    fig.update_layout(
        title='Quantum Energy Level Comparison',
        height=600, width=1000,
        showlegend=True
    )

    fig.update_yaxes(title_text='E (ℏω)', row=1, col=1)
    fig.update_yaxes(title_text='E (E₁)', row=1, col=2)
    fig.update_yaxes(title_text='E (eV)', row=1, col=3)

    for col in [1, 2, 3]:
        fig.update_xaxes(showticklabels=False, row=1, col=col)

    return fig

fig = create_energy_level_explorer()
fig.show()
```

---

## Worked Examples

### Example 1: Interactive Wave Function Comparison

**Problem**: Create an interactive comparison of harmonic oscillator eigenstates.

**Solution**:
```python
import numpy as np
import plotly.graph_objects as go
from scipy.special import hermite
from math import factorial

def harmonic_oscillator_state(n, x):
    """Compute n-th harmonic oscillator eigenstate."""
    xi = x
    prefactor = (1/np.pi)**0.25
    norm = 1 / np.sqrt(2**n * factorial(n))
    H = hermite(n)
    return prefactor * norm * H(xi) * np.exp(-xi**2/2)

x = np.linspace(-6, 6, 500)

# Create figure with all states
fig = go.Figure()

colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

for n in range(6):
    psi = harmonic_oscillator_state(n, x)
    visible = (n == 0)  # Only show n=0 initially

    fig.add_trace(go.Scatter(
        x=x, y=psi,
        mode='lines',
        name=f'ψ_{n}(x)',
        line=dict(color=colors[n], width=2),
        visible=visible,
        hovertemplate=f'n={n}<br>x: %{{x:.3f}}<br>ψ: %{{y:.4f}}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x, y=np.abs(psi)**2,
        mode='lines',
        name=f'|ψ_{n}|²',
        line=dict(color=colors[n], width=2, dash='dash'),
        visible=visible,
        fill='tozeroy',
        fillcolor=f'rgba({int(colors[n] != "blue")*255}, 0, {int(colors[n] == "blue")*255}, 0.1)'
    ))

# Create buttons for each state
buttons = []
for n in range(6):
    visibility = [False] * 12
    visibility[2*n] = True
    visibility[2*n + 1] = True

    buttons.append(dict(
        label=f'n = {n}',
        method='update',
        args=[{'visible': visibility},
              {'title': f'Harmonic Oscillator: n = {n}, E = {n + 0.5}ℏω'}]
    ))

# Add "Show All" button
buttons.append(dict(
    label='Show All',
    method='update',
    args=[{'visible': [True] * 12},
          {'title': 'Harmonic Oscillator: All States'}]
))

fig.update_layout(
    title='Harmonic Oscillator: n = 0',
    xaxis_title='Position x (√(ℏ/mω))',
    yaxis_title='Amplitude',
    updatemenus=[dict(
        type='buttons',
        direction='right',
        x=0.5, y=1.15,
        xanchor='center',
        buttons=buttons
    )],
    height=600, width=900
)

fig.show()
fig.write_html('ho_interactive.html')
```

### Example 2: 3D Orbital Viewer

**Problem**: Create an interactive 3D viewer for hydrogen orbitals.

**Solution**:
```python
import numpy as np
import plotly.graph_objects as go
from scipy.special import sph_harm

def generate_orbital_data(l, m, n_points=40):
    """Generate surface data for spherical harmonic."""
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    THETA, PHI = np.meshgrid(theta, phi)

    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y_cart = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Color by real part (sign)
    color = Y.real

    return X, Y_cart, Z, color

# Create figure with multiple orbitals
fig = go.Figure()

orbitals = [
    (0, 0, 's orbital'),
    (1, 0, 'p_z orbital'),
    (1, 1, 'p_x + ip_y'),
    (2, 0, 'd_z² orbital'),
    (2, 1, 'd_xz + id_yz'),
    (2, 2, 'd_x²-y² + id_xy')
]

for i, (l, m, name) in enumerate(orbitals):
    X, Y, Z, color = generate_orbital_data(l, m)

    visible = (i == 0)

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=color,
        colorscale='RdBu',
        showscale=False,
        visible=visible,
        name=name,
        hovertemplate=(
            f'{name}<br>'
            'x: %{x:.3f}<br>'
            'y: %{y:.3f}<br>'
            'z: %{z:.3f}<extra></extra>'
        )
    ))

# Buttons
buttons = []
for i, (l, m, name) in enumerate(orbitals):
    visibility = [False] * len(orbitals)
    visibility[i] = True
    buttons.append(dict(
        label=name,
        method='update',
        args=[{'visible': visibility},
              {'title': f'Orbital: {name} (l={l}, m={m})'}]
    ))

fig.update_layout(
    title='Orbital: s orbital (l=0, m=0)',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    updatemenus=[dict(
        type='dropdown',
        x=0.1, y=1.1,
        buttons=buttons
    )],
    width=800, height=700
)

fig.show()
fig.write_html('orbital_viewer.html')
```

### Example 3: Animated Quantum Tunneling

**Problem**: Animate a wave packet tunneling through a potential barrier.

**Solution**:
```python
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import expm

def create_tunneling_animation():
    """Create animation of quantum tunneling."""

    # Grid
    N = 200
    L = 20
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # Potential barrier
    barrier_width = 1.0
    barrier_height = 5.0
    V = np.where(np.abs(x) < barrier_width/2, barrier_height, 0)

    # Initial wave packet
    x0 = -4
    k0 = 3
    sigma = 0.8
    psi0 = (1/(2*np.pi*sigma**2))**0.25 * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)
    psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

    # Hamiltonian (finite difference)
    hbar = 1
    m = 1
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = hbar**2/(m*dx**2) + V[i]
        if i > 0:
            H[i, i-1] = -hbar**2/(2*m*dx**2)
        if i < N-1:
            H[i, i+1] = -hbar**2/(2*m*dx**2)

    # Time evolution
    dt = 0.02
    n_frames = 100
    U = expm(-1j * H * dt / hbar)

    # Create frames
    frames = []
    psi = psi0.copy()

    for frame_idx in range(n_frames):
        prob = np.abs(psi)**2

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x, y=V/barrier_height * 0.5, name='V(x)',
                          fill='tozeroy', line=dict(color='gray')),
                go.Scatter(x=x, y=prob, name='|ψ|²',
                          line=dict(color='blue', width=2))
            ],
            name=f'{frame_idx}'
        ))

        psi = U @ psi

    # Initial figure
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=V/barrier_height * 0.5, name='V(x)',
                      fill='tozeroy', line=dict(color='gray')),
            go.Scatter(x=x, y=np.abs(psi0)**2, name='|ψ|²',
                      line=dict(color='blue', width=2))
        ],
        frames=frames
    )

    fig.update_layout(
        title='Quantum Tunneling Through Barrier',
        xaxis_title='Position x',
        yaxis_title='Probability Density',
        yaxis_range=[0, 0.6],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.1, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=30, redraw=True),
                                     fromcurrent=True)]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0),
                                       mode='immediate')])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0),
                                            mode='immediate')],
                       method='animate')
                  for f in frames]
        )]
    )

    return fig

fig = create_tunneling_animation()
fig.show()
fig.write_html('tunneling_animation.html')
```

---

## Practice Problems

### Level 1: Direct Application

1. **Basic Interactive Plot**: Create an interactive line plot of $$\sin(nx)$$ for $$x \in [0, 2\pi]$$ with buttons to select $$n = 1, 2, 3, 4$$.

2. **3D Surface**: Create an interactive 3D surface plot of $$z = x^2 - y^2$$ (saddle point) with proper axis labels.

3. **Hover Information**: Create a scatter plot of 100 random points with hover text showing the point index and exact coordinates.

### Level 2: Intermediate

4. **Multi-Trace Animation**: Animate both the real and imaginary parts of a rotating phasor $$e^{i\omega t}$$ as it evolves in time.

5. **Linked Subplots**: Create a 2×2 subplot figure where clicking on an energy level in panel 1 highlights the corresponding wave function in panel 2.

6. **3D Probability Distribution**: Create an interactive 3D scatter plot representing the probability distribution of a 2D harmonic oscillator ground state.

### Level 3: Challenging

7. **Bloch Sphere Trajectory**: Animate the state vector on the Bloch sphere as it undergoes Rabi oscillations (rotation about the x-axis).

8. **Interactive Parameter Explorer**: Create a visualization where sliders control the width and height of a rectangular potential barrier, and the transmission coefficient updates in real time.

9. **Density Matrix Visualization**: Create an interactive heatmap of a density matrix with sliders controlling the purity of the state (interpolating between pure and maximally mixed).

---

## Computational Lab

### Project: Complete Interactive Quantum Dashboard

```python
"""
Interactive Quantum Visualization Dashboard
Day 270: Plotly Interactive Visualization
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import hermite, sph_harm
from math import factorial


class QuantumDashboard:
    """
    Interactive dashboard for exploring quantum mechanical concepts.
    """

    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'background': '#f0f0f0'
        }

    def harmonic_oscillator_state(self, n, x):
        """Compute HO eigenstate."""
        xi = x
        prefactor = (1/np.pi)**0.25
        norm = 1 / np.sqrt(2**n * factorial(n))
        H = hermite(n)
        return prefactor * norm * H(xi) * np.exp(-xi**2/2)

    def create_state_explorer(self, max_n=5):
        """Create interactive harmonic oscillator state explorer."""

        x = np.linspace(-6, 6, 500)
        V = 0.5 * x**2

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Wave Function', 'Probability Density',
                'In Potential', 'Energy Spectrum'
            ),
            specs=[[{}, {}], [{}, {}]]
        )

        # Generate all states
        states = {}
        for n in range(max_n + 1):
            psi = self.harmonic_oscillator_state(n, x)
            states[n] = psi

        # Add traces (initially show n=0)
        n = 0
        psi = states[0]

        # Wave function
        fig.add_trace(
            go.Scatter(x=x, y=psi, name='ψ(x)',
                      line=dict(color=self.colors['primary'], width=2)),
            row=1, col=1
        )

        # Probability
        fig.add_trace(
            go.Scatter(x=x, y=np.abs(psi)**2, name='|ψ|²',
                      line=dict(color=self.colors['secondary'], width=2),
                      fill='tozeroy'),
            row=1, col=2
        )

        # In potential
        fig.add_trace(
            go.Scatter(x=x, y=V, name='V(x)',
                      line=dict(color='gray', width=1)),
            row=2, col=1
        )
        scale = 0.5
        fig.add_trace(
            go.Scatter(x=x, y=0.5 + scale * np.abs(psi)**2, name='|ψ|² + E',
                      line=dict(color=self.colors['primary'], width=2),
                      fill='tonexty'),
            row=2, col=1
        )

        # Energy levels
        for i in range(max_n + 1):
            E = i + 0.5
            color = self.colors['primary'] if i == n else 'lightgray'
            fig.add_trace(
                go.Scatter(x=[0.2, 0.8], y=[E, E], mode='lines',
                          line=dict(color=color, width=3),
                          name=f'n={i}', showlegend=False),
                row=2, col=2
            )

        # Create buttons for each n
        buttons = []
        for n_val in range(max_n + 1):
            psi_n = states[n_val]

            button = dict(
                label=f'n={n_val}',
                method='update',
                args=[
                    {
                        'y': [
                            psi_n,  # Wave function
                            np.abs(psi_n)**2,  # Probability
                            V,  # Potential
                            (n_val + 0.5) + scale * np.abs(psi_n)**2  # In potential
                        ] + [[n_val + 0.5, n_val + 0.5] if i == n_val else None
                             for i in range(max_n + 1)]
                    },
                    {'title': f'Harmonic Oscillator: n={n_val}, E={(n_val+0.5):.1f}ℏω'}
                ]
            )
            buttons.append(button)

        fig.update_layout(
            title='Harmonic Oscillator: n=0, E=0.5ℏω',
            height=700, width=1000,
            updatemenus=[dict(
                type='buttons',
                direction='right',
                x=0.5, y=1.12,
                xanchor='center',
                buttons=buttons
            )]
        )

        return fig

    def create_bloch_sphere(self, theta=np.pi/4, phi=np.pi/4):
        """Create interactive Bloch sphere."""

        fig = go.Figure()

        # Sphere surface
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_s, y=y_s, z=z_s,
            opacity=0.2,
            colorscale=[[0, 'lightcyan'], [1, 'lightcyan']],
            showscale=False
        ))

        # State vector
        x_state = np.sin(theta) * np.cos(phi)
        y_state = np.sin(theta) * np.sin(phi)
        z_state = np.cos(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_state], y=[0, y_state], z=[0, z_state],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=[0, 12], color='red'),
            name='|ψ⟩'
        ))

        # Coordinate axes
        for coords, color, name in [
            ([[0, 0, -1.3, 1.3], [0, 0, 0, 0], [0, 0, 0, 0]], 'red', 'x'),
            ([[0, 0, 0, 0], [0, 0, -1.3, 1.3], [0, 0, 0, 0]], 'green', 'y'),
            ([[0, 0, 0, 0], [0, 0, 0, 0], [-1.3, 1.3, 0, 0]], 'blue', 'z')
        ]:
            fig.add_trace(go.Scatter3d(
                x=[coords[0][2], coords[0][3]],
                y=[coords[1][2], coords[1][3]],
                z=[coords[2][0], coords[2][1]] if name == 'z' else [0, 0],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))

        fig.update_layout(
            title=f'Bloch Sphere: θ={theta:.2f}, φ={phi:.2f}',
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5]),
                yaxis=dict(range=[-1.5, 1.5]),
                zaxis=dict(range=[-1.5, 1.5]),
                aspectmode='cube'
            ),
            width=700, height=700
        )

        return fig

    def create_uncertainty_explorer(self):
        """Visualize position-momentum uncertainty."""

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Position Space', 'Momentum Space'))

        sigmas = [0.5, 1.0, 2.0, 3.0]

        for sigma in sigmas:
            x = np.linspace(-10, 10, 500)
            k = np.linspace(-5, 5, 500)

            # Position space
            psi_x = (1/(2*np.pi*sigma**2))**0.25 * np.exp(-x**2/(4*sigma**2))
            prob_x = np.abs(psi_x)**2

            # Momentum space (Gaussian Fourier transform)
            sigma_k = 1/(2*sigma)
            psi_k = (2*np.pi*sigma_k**2)**0.25 * np.exp(-k**2/(4*sigma_k**2))
            prob_k = np.abs(psi_k)**2

            visible = (sigma == 1.0)

            fig.add_trace(
                go.Scatter(x=x, y=prob_x, name=f'σ_x={sigma}',
                          visible=visible, line=dict(width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=k, y=prob_k, name=f'σ_k={sigma_k:.2f}',
                          visible=visible, line=dict(width=2)),
                row=1, col=2
            )

        # Buttons
        buttons = []
        for i, sigma in enumerate(sigmas):
            visibility = [False] * (2 * len(sigmas))
            visibility[2*i] = True
            visibility[2*i + 1] = True
            sigma_k = 1/(2*sigma)

            buttons.append(dict(
                label=f'σ_x = {sigma}',
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': f'Uncertainty: Δx·Δp = {sigma * sigma_k:.3f} ≥ 1/2'}
                ]
            ))

        fig.update_layout(
            title='Uncertainty: Δx·Δp = 0.500 ≥ 1/2',
            updatemenus=[dict(
                type='buttons',
                direction='right',
                x=0.5, y=1.15,
                xanchor='center',
                buttons=buttons
            )],
            height=500, width=1000
        )

        fig.update_xaxes(title_text='x', row=1, col=1)
        fig.update_xaxes(title_text='k', row=1, col=2)
        fig.update_yaxes(title_text='|ψ(x)|²', row=1, col=1)
        fig.update_yaxes(title_text='|φ(k)|²', row=1, col=2)

        return fig


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Interactive Quantum Dashboard")
    print("Day 270: Plotly Visualization")
    print("=" * 60)

    dashboard = QuantumDashboard()

    # 1. State explorer
    print("\n1. Creating harmonic oscillator state explorer...")
    fig1 = dashboard.create_state_explorer(max_n=5)
    fig1.write_html('ho_explorer.html')
    print("   Saved: ho_explorer.html")

    # 2. Bloch sphere
    print("\n2. Creating Bloch sphere visualization...")
    fig2 = dashboard.create_bloch_sphere(theta=np.pi/3, phi=np.pi/4)
    fig2.write_html('bloch_sphere.html')
    print("   Saved: bloch_sphere.html")

    # 3. Uncertainty explorer
    print("\n3. Creating uncertainty principle explorer...")
    fig3 = dashboard.create_uncertainty_explorer()
    fig3.write_html('uncertainty_explorer.html')
    print("   Saved: uncertainty_explorer.html")

    # Show all figures
    print("\n" + "=" * 60)
    print("Interactive HTML files created!")
    print("Open in browser for full interactivity.")
    print("=" * 60)

    fig1.show()
    fig2.show()
    fig3.show()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| `plotly.graph_objects` | Full-control API for complex visualizations |
| `plotly.express` | Quick, high-level plotting API |
| `go.Scatter`, `go.Surface` | Basic trace types |
| `go.Figure(frames=...)` | Animation support |
| `updatemenus` | Buttons and dropdowns for interactivity |
| `hovertemplate` | Custom hover text formatting |
| `.write_html()` | Export to standalone HTML |

### Key Plotly Patterns

```python
# Basic structure
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
fig.update_layout(title='Title', xaxis_title='X', yaxis_title='Y')
fig.show()

# Animation
fig = go.Figure(data=[...], frames=[go.Frame(data=[...]) for ...])

# Interactivity
fig.update_layout(updatemenus=[dict(type='buttons', buttons=[...])])
```

---

## Daily Checklist

- [ ] Created basic interactive plots with hover info
- [ ] Built 3D visualizations with rotation/zoom
- [ ] Added dropdown menus and buttons
- [ ] Created frame-based animations
- [ ] Built linked subplots
- [ ] Exported visualizations to HTML
- [ ] Completed computational lab exercises

---

## Preview of Day 271

Tomorrow we focus on **Animation** with Matplotlib:
- `FuncAnimation` for frame-by-frame animation
- Time evolution of wave packets
- Bloch sphere dynamics for qubit states
- Exporting animations as MP4 and GIF
- Animation optimization for smooth playback

Animation brings quantum dynamics to life, showing how wave functions evolve in time.
