# Week 39: Visualization & Scientific Plotting

## Overview

This week provides comprehensive coverage of scientific visualization in Python, the essential skill for communicating quantum mechanical results effectively. Building on the NumPy and SciPy foundations from previous weeks, we master Matplotlib's powerful plotting capabilities, learn 3D visualization techniques, explore interactive plotting with Plotly, create animations of quantum dynamics, and develop publication-quality figures that meet journal standards.

## Weekly Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **Day 267** | Matplotlib Fundamentals | `pyplot`, `Figure`, `Axes`, subplots, basic plot types, styling |
| **Day 268** | Scientific Plotting | Error bars, log scales, colormaps, contour plots, heatmaps |
| **Day 269** | 3D Visualization | `mplot3d`, surface plots, wireframes, 3D wave functions |
| **Day 270** | Interactive Visualization | Plotly, widgets, callbacks, interactive quantum state exploration |
| **Day 271** | Animation | `FuncAnimation`, time evolution videos, Bloch sphere dynamics |
| **Day 272** | Publication Quality | LaTeX labels, journal styles, vector exports, multi-panel figures |
| **Day 273** | Week Review | Comprehensive visualization project, style guide creation |

## Learning Objectives

By the end of this week, you will be able to:

1. **Matplotlib Mastery**
   - Create any standard plot type with full customization
   - Use the object-oriented interface for complex layouts
   - Apply consistent styling across all figures

2. **Scientific Visualization**
   - Plot data with proper error representation
   - Use appropriate scales (linear, log, symlog) for different data
   - Create effective colormapped visualizations

3. **3D Graphics**
   - Visualize 3D wave functions and probability densities
   - Create surface plots of potential energy surfaces
   - Render orbital shapes and electron densities

4. **Interactive Exploration**
   - Build interactive dashboards for parameter exploration
   - Create linked views for multi-dimensional data
   - Design intuitive controls for quantum state manipulation

5. **Animation**
   - Animate time-dependent quantum dynamics
   - Create publication-quality videos of wave packet evolution
   - Visualize Bloch sphere trajectories for qubit evolution

6. **Publication Standards**
   - Format figures for journal submission
   - Use LaTeX for mathematical labels
   - Export in vector formats (PDF, SVG, EPS)

## Quantum Mechanics Connections

| Visualization Type | Quantum Application |
|--------------------|---------------------|
| Line plots | Wave function $$\psi(x)$$, probability density $$|\psi(x)|^2$$ |
| Colormaps | 2D probability densities, quantum dot states |
| 3D surfaces | Orbital shapes $$|Y_l^m(\theta, \phi)|^2$$, potential landscapes |
| Contour plots | Energy level diagrams, phase space portraits |
| Animations | Time evolution $$|\psi(x,t)|^2$$, wave packet dynamics |
| Bloch sphere | Qubit state visualization, gate operations |
| Interactive | Parameter sweeps, eigenvalue exploration |

## Prerequisites

- Week 37: Python & NumPy Essentials
- Week 38: SciPy & Numerical Methods
- Linear Algebra (for understanding transformations)
- Complex Analysis (for phase visualization)

## Key Resources

### Documentation
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book)

### Textbooks
- Rougier, *Scientific Visualization: Python + Matplotlib*
- VanderPlas, *Python Data Science Handbook* (Chapter 4)
- Johansson, *Numerical Python* (Visualization chapters)

### Style Guides
- [Nature Figure Guidelines](https://www.nature.com/documents/nature-final-artwork.pdf)
- [APS Physical Review Style](https://journals.aps.org/authors/preparing-figures)
- [IEEE Visualization Best Practices](https://ieeevis.org/)

## Required Software

```python
# Core visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import Normalize, LogNorm
from matplotlib import cm

# Interactive visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Scientific computing (from previous weeks)
import numpy as np
from scipy import special, integrate

# Jupyter widgets (optional)
# from ipywidgets import interact, FloatSlider

# Version requirements
# matplotlib >= 3.5
# plotly >= 5.0
# numpy >= 1.20
```

## Weekly Project: Quantum Visualization Dashboard

Throughout this week, you will build a comprehensive visualization toolkit for quantum mechanics:

1. **Monday**: Create basic wave function plots with proper styling
2. **Tuesday**: Add uncertainty visualization and energy level diagrams
3. **Wednesday**: Implement 3D orbital visualization
4. **Thursday**: Build interactive state explorer with Plotly
5. **Friday**: Animate time evolution and Bloch sphere dynamics
6. **Saturday**: Polish all figures to publication quality
7. **Sunday**: Integrate into a complete visualization package

## Figure Quality Standards

### Resolution Requirements
| Output Type | DPI | Format |
|-------------|-----|--------|
| Screen/Presentation | 100-150 | PNG |
| Print (journal) | 300-600 | PDF, EPS |
| Web interactive | N/A | HTML (Plotly) |
| Animation | 100 | MP4, GIF |

### Color Accessibility
- Use colorblind-safe palettes (viridis, cividis)
- Avoid red-green contrasts
- Ensure sufficient contrast ratios
- Include redundant encoding (shape + color)

### Typography
- Use LaTeX for all mathematical expressions
- Consistent font sizes across panels
- Axis labels: 10-12pt
- Title: 12-14pt
- Tick labels: 8-10pt

## Assessment Criteria

| Component | Weight | Description |
|-----------|--------|-------------|
| Daily Labs | 40% | Working visualizations for each day's exercises |
| Code Quality | 20% | Clean, reusable plotting functions |
| Aesthetic Quality | 20% | Professional, publication-ready figures |
| QM Interpretation | 20% | Correct physical interpretation of visualizations |

## Tips for Success

1. **Start with the data** - Understand your arrays before plotting
2. **Iterate on style** - Good figures require refinement
3. **Use the gallery** - Matplotlib gallery has examples for everything
4. **Think about your audience** - Different styles for papers vs. talks
5. **Save figure code** - Keep plotting scripts with your data analysis

## Navigation

- **Previous**: [Week 38: SciPy & Numerical Methods](../Week_38_SciPy_ODEs/README.md)
- **Next**: [Week 40: Physics Simulations](../Week_40_Physics_Simulations/README.md)
- **Month Overview**: [Month 10: Scientific Computing](../README.md)
