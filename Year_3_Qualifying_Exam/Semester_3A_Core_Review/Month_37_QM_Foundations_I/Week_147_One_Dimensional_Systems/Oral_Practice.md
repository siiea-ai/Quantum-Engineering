# Week 147: One-Dimensional Systems — Oral Practice Questions

## Conceptual Questions

### Question 1: Why Does the Particle in a Box Have Non-Zero Ground State Energy?

**Answer Framework:**

1. **Zero-Point Energy:** $$E_1 = \frac{\pi^2\hbar^2}{2mL^2} \neq 0$$

2. **Uncertainty Principle Explanation:**
   - Confined to region $$\Delta x \sim L$$
   - Must have $$\Delta p \geq \hbar/(2L)$$
   - Kinetic energy: $$E \sim (\Delta p)^2/(2m) \sim \hbar^2/(mL^2)$$

3. **Classical Contrast:**
   - Classical particle can sit still at bottom
   - Quantum particle always has motion

4. **Significance:**
   - Quantum systems cannot have zero energy
   - Explains stability of atoms
   - Essential for quantum phenomena

---

### Question 2: Compare Infinite and Finite Square Wells

**Answer Framework:**

| Feature | Infinite Well | Finite Well |
|---------|---------------|-------------|
| Bound states | Infinite | Finite |
| $$\psi$$ at boundary | Zero | Continuous |
| Wavefunction | Confined | Leaks outside |
| Lowest energy | $$E_1 > 0$$ | $$E_1$$ lower |

**Key Point:** Finite well always has at least one bound state.

**Physical Applications:**
- Quantum dots (finite wells)
- Nuclear potentials
- Heterostructures

---

### Question 3: Explain Ladder Operators for the Harmonic Oscillator

**Answer Framework:**

1. **Definition:**
   - $$\hat{a}$$ (lowering): $$\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$$
   - $$\hat{a}^\dagger$$ (raising): $$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

2. **Why "Ladder":**
   - Step between energy levels
   - $$\hat{a}^\dagger\hat{a} = \hat{n}$$ counts quanta

3. **Advantages Over Series Solution:**
   - Algebraic, not differential
   - Easier matrix elements
   - Generalizes to other systems

4. **Applications:**
   - Quantum optics (photons)
   - Phonons in solids
   - Quantum field theory

---

### Question 4: What is a Coherent State?

**Answer Framework:**

1. **Definition:** Eigenstate of $$\hat{a}$$: $$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$

2. **Properties:**
   - Minimum uncertainty: $$\Delta x \Delta p = \hbar/2$$
   - Poissonian photon distribution
   - Most classical-like quantum state

3. **Time Evolution:**
   - $$\langle x(t)\rangle$$ oscillates classically
   - Shape preserved (Gaussian)
   - No spreading

4. **Physical Realization:**
   - Laser light
   - Driven oscillator

---

### Question 5: Explain the Delta Function Potential

**Answer Framework:**

1. **Model:** $$V(x) = -\alpha\delta(x)$$ — infinitely deep, infinitely narrow well

2. **Bound State:**
   - Exactly one bound state
   - Energy: $$E = -m\alpha^2/(2\hbar^2)$$
   - Wavefunction: cusp at origin

3. **Discontinuity Condition:**
   - $$\psi'$$ has jump: $$\Delta\psi' = -\frac{2m\alpha}{\hbar^2}\psi(0)$$

4. **Why Useful:**
   - Exactly solvable
   - Model for impurities
   - Simplest tunneling problem

---

## Technical Questions

### Question 6: Derive the Energy Levels of the Infinite Square Well

**Answer Framework:**

1. **Schrödinger Equation:**
   $$-\frac{\hbar^2}{2m}\psi'' = E\psi$$ inside well

2. **General Solution:**
   $$\psi = A\sin(kx) + B\cos(kx)$$

3. **Boundary Conditions:**
   - $$\psi(0) = 0 \Rightarrow B = 0$$
   - $$\psi(L) = 0 \Rightarrow kL = n\pi$$

4. **Energy Quantization:**
   $$E_n = \frac{\hbar^2 k_n^2}{2m} = \frac{n^2\pi^2\hbar^2}{2mL^2}$$

---

### Question 7: Explain the Algebraic Solution of the Harmonic Oscillator

**Answer Framework:**

1. **Rewrite Hamiltonian:**
   $$\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a} + 1/2)$$

2. **Commutation:**
   $$[\hat{a}, \hat{a}^\dagger] = 1$$

3. **Ground State:**
   - $$\hat{a}|0\rangle = 0$$
   - $$E_0 = \hbar\omega/2$$

4. **Excited States:**
   - $$|n\rangle = (\hat{a}^\dagger)^n/\sqrt{n!}|0\rangle$$
   - $$E_n = \hbar\omega(n + 1/2)$$

5. **Why Elegant:**
   - No differential equations
   - Shows structure of quantum mechanics
   - Generalizes to many systems

---

### Question 8: How Do You Determine the Number of Bound States in a Finite Well?

**Answer Framework:**

1. **Define:** $$z_0 = a\sqrt{2mV_0}/\hbar$$

2. **Graphical Method:**
   - Plot $$\tan z$$ vs. $$\sqrt{(z_0/z)^2 - 1}$$
   - Count intersections

3. **Estimate:**
   $$N \approx \lfloor z_0/(\pi/2) \rfloor + 1$$

4. **Key Result:**
   - Always at least one bound state
   - More states as $$V_0$$ or $$a$$ increases

---

### Question 9: Describe Wave Packet Spreading

**Answer Framework:**

1. **Cause:**
   - Different momentum components have different velocities
   - $$v = p/m = \hbar k/m$$

2. **Gaussian Packet:**
   - Initial width $$\sigma_0$$
   - Spreading: $$\sigma(t) = \sigma_0\sqrt{1 + (t/\tau)^2}$$
   - Time scale: $$\tau = 2m\sigma_0^2/\hbar$$

3. **Implications:**
   - Narrower packets spread faster
   - Heavier particles spread slower
   - Cannot localize forever

---

### Question 10: What are Selection Rules for the Harmonic Oscillator?

**Answer Framework:**

1. **For $$\hat{x}$$:** $$\Delta n = \pm 1$$
   - $$\langle n|\hat{x}|m\rangle \neq 0$$ only if $$m = n \pm 1$$

2. **For $$\hat{x}^2$$:** $$\Delta n = 0, \pm 2$$

3. **Physical Meaning:**
   - Electric dipole transitions: $$\Delta n = \pm 1$$
   - Photon carries one quantum

4. **Derivation:**
   - $$\hat{x} \propto (\hat{a} + \hat{a}^\dagger)$$
   - Raises or lowers by one

---

## "Explain to Non-Expert" Questions

### Question 11: Why Are Energy Levels Quantized?

**Answer:**
- Bound particles are like standing waves
- Only certain wavelengths "fit"
- This restricts allowed energies
- Like guitar string: only certain notes

---

### Question 12: What is Zero-Point Energy?

**Answer:**
- Lowest possible energy, but not zero
- Uncertainty principle prevents rest
- Particle must always have some motion
- Example: helium stays liquid at T = 0

---

### Question 13: Why is the Harmonic Oscillator Important?

**Answer:**
- Appears everywhere: atoms, molecules, light
- Any potential is approximately quadratic near minimum
- Exactly solvable
- Building block for quantum field theory

---

### Question 14: What Happens When a Well Suddenly Changes?

**Answer:**
- State doesn't instantly change (sudden approximation)
- But it's now a superposition of new eigenstates
- Energy expectation may change
- Interesting quantum dynamics follow

---

### Question 15: How Does a Quantum Particle "Know" About a Barrier?

**Answer:**
- Wave nature: penetrates into barrier
- Evanescent wave in classically forbidden region
- Tunneling if barrier is thin enough
- Fundamentally different from classical bouncing

---

*Oral Practice for Week 147 — One-Dimensional Systems*
