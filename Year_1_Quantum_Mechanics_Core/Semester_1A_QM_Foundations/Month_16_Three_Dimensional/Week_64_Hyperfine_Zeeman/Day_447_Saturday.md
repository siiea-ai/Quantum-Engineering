# Day 447: Atomic Qubits

## Overview
**Day 447** | Year 1, Month 16, Week 64 | Trapped Ion Quantum Computing

Today we connect our atomic physics knowledge to trapped ion quantum computing — one of the leading qubit platforms.

---

## Learning Objectives

1. Understand trapped ion qubit implementations
2. Connect atomic structure to qubit states
3. Analyze qubit manipulation techniques
4. Recognize key ion species used
5. See the physics-engineering connection

---

## Core Content

### Trapped Ion Qubits

Ions are trapped using electromagnetic fields (Paul trap) and cooled to their motional ground state.

### Qubit Encodings

**1. Hyperfine qubit** (e.g., ¹⁷¹Yb⁺, ⁹Be⁺):
- |0⟩ = |F=0, m_F=0⟩
- |1⟩ = |F=1, m_F=0⟩
- Transition: microwave (~GHz)
- Long coherence times (seconds)

**2. Optical qubit** (e.g., ⁴⁰Ca⁺):
- |0⟩ = S₁/₂
- |1⟩ = D₅/₂ (metastable)
- Transition: optical (~THz)
- Fast gates

**3. Zeeman qubit:**
- |0⟩ = |m_J = -1/2⟩
- |1⟩ = |m_J = +1/2⟩
- Magnetic field sensitive

### Key Ion Species

| Ion | Type | Transition |
|-----|------|------------|
| ⁴⁰Ca⁺ | Optical | S₁/₂ ↔ D₅/₂ |
| ¹⁷¹Yb⁺ | Hyperfine | F=0 ↔ F=1 |
| ⁹Be⁺ | Hyperfine | F=1 ↔ F=2 |
| Sr⁺ | Optical | S₁/₂ ↔ D₅/₂ |

### Gate Operations

**Single-qubit:** Laser/microwave Rabi oscillations
$$|\psi(t)\rangle = \cos(\Omega t/2)|0\rangle - i\sin(\Omega t/2)|1\rangle$$

**Two-qubit:** Mølmer-Sørensen, Cirac-Zoller gates
- Use shared motional modes as bus
- Entangling operations via phonons

### State Readout

- Electron shelving technique
- Fluorescence detection
- State-dependent scattering

---

## Quantum Computing Connection

This IS the quantum computing connection! Trapped ions offer:
- Highest fidelity gates (>99.9%)
- Long coherence times
- All-to-all connectivity
- Demonstrated quantum advantage

---

## Practice Problems

1. Why are hyperfine qubits less sensitive to magnetic noise?
2. What is the advantage of optical qubits?
3. How does Doppler cooling work for ions?

---

## Summary

| Qubit Type | Example | Key Feature |
|------------|---------|-------------|
| Hyperfine | ¹⁷¹Yb⁺ | Long T₂ |
| Optical | ⁴⁰Ca⁺ | Fast gates |
| Zeeman | Various | Simple control |

---

**Next:** [Day_448_Sunday.md](Day_448_Sunday.md) — Month 16 Capstone
