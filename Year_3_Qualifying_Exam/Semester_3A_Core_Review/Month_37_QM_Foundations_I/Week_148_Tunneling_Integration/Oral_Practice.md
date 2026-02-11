# Week 148: Tunneling and WKB — Oral Practice Questions

## Conceptual Questions

### Question 1: Explain Quantum Tunneling

**Answer Framework:**

1. **Definition:** Particle passes through barrier it classically cannot surmount

2. **Why It Happens:**
   - Wave nature of particles
   - Wavefunction penetrates into barrier (evanescent)
   - Non-zero amplitude on other side

3. **Key Factors:**
   - Barrier height above particle energy
   - Barrier width
   - Particle mass

4. **Formula:** $$\mathcal{T} \approx e^{-2\kappa a}$$ where $$\kappa = \sqrt{2m(V-E)}/\hbar$$

5. **Applications:** STM, alpha decay, tunnel diodes

---

### Question 2: What is the WKB Approximation?

**Answer Framework:**

1. **Name:** Wentzel-Kramers-Brillouin (semiclassical approximation)

2. **Key Idea:**
   - Valid when potential varies slowly
   - Wavelength $$\ll$$ scale of potential variation

3. **Result:**
   $$\psi \approx \frac{C}{\sqrt{p(x)}}e^{\pm i\int p\,dx/\hbar}$$

4. **Validity Condition:**
   $$\frac{\hbar}{p^2}\left|\frac{dp}{dx}\right| \ll 1$$

5. **Failure Points:**
   - Classical turning points ($$p = 0$$)
   - Need connection formulas

---

### Question 3: What are Connection Formulas?

**Answer Framework:**

1. **Problem:** WKB fails at turning points

2. **Solution:**
   - Solve exactly near turning point (Airy equation)
   - Match to WKB solutions on both sides

3. **Physical Picture:**
   - Smooth transition from oscillatory to decaying

4. **Key Result:**
   - Phase shift of $$\pi/4$$ at turning point
   - Leads to $$n + 1/2$$ in quantization

---

### Question 4: Explain Alpha Decay

**Answer Framework:**

1. **Process:** Alpha particle tunnels through Coulomb barrier

2. **Barrier:**
   - Inside nucleus: alpha is bound
   - Outside: repulsive Coulomb potential

3. **Gamow Factor:**
   $$\mathcal{T} = e^{-2\gamma}$$
   where $$\gamma \propto Z/\sqrt{E}$$

4. **Result:**
   - Wide range of half-lives
   - Geiger-Nuttall law: $$\log t_{1/2} \propto Z/\sqrt{E}$$

---

### Question 5: Why is Transmission Non-Zero Above Barrier?

**Answer Framework:**

1. **Classical Expectation:**
   $$E > V_0$$ should give $$\mathcal{T} = 1$$

2. **Quantum Reality:**
   $$\mathcal{R} > 0$$ even for $$E > V_0$$

3. **Reason:**
   - Momentum changes at interface
   - Wave impedance mismatch
   - Partial reflection like light at glass

4. **Formula:**
   $$\mathcal{R} = \left(\frac{k_1-k_2}{k_1+k_2}\right)^2$$

---

## Technical Questions

### Question 6: Derive Transmission for Step Potential

**Framework:**

1. Write wavefunctions in each region
2. Apply matching conditions at interface
3. Solve for R and T
4. Use current for probability coefficients

---

### Question 7: Apply WKB to Harmonic Oscillator

**Framework:**

1. Find turning points: $$x_0 = \sqrt{2E/(m\omega^2)}$$
2. Evaluate $$\oint p\,dx = \pi E/(\omega/2)$$
3. Apply quantization: $$= 2\pi\hbar(n+1/2)$$
4. Result: $$E_n = \hbar\omega(n+1/2)$$

---

### Question 8: What Causes Transmission Resonances?

**Framework:**

1. Occur when $$E > V_0$$ for barriers
2. Perfect transmission when $$k'a = n\pi$$
3. Constructive interference inside barrier
4. Like Fabry-Perot cavity

---

### Question 9: Explain STM Sensitivity

**Framework:**

1. Current $$\propto e^{-2\kappa d}$$
2. $$\kappa \approx 10$$ nm$$^{-1}$$ for typical work functions
3. 0.1 nm change → factor of 8 in current
4. Enables atomic resolution

---

### Question 10: What is the Gamow Factor?

**Framework:**

1. $$\gamma = \int \kappa\,dx$$ through barrier
2. $$\mathcal{T} = e^{-2\gamma}$$
3. Determines tunneling probability
4. Named after George Gamow (alpha decay theory)

---

## "Explain to Non-Expert" Questions

### Question 11: How Can a Particle Go Through a Wall?

**Answer:**
- Particles are waves, not billiard balls
- Wave "leaks" into the wall
- If wall is thin enough, some probability on other side
- Like sound through a thin wall

---

### Question 12: Why Does the Sun Shine?

**Answer:**
- Nuclear fusion requires protons to overcome Coulomb repulsion
- Thermal energy too low classically
- But quantum tunneling allows fusion
- Without tunneling, Sun would be cold

---

### Question 13: What is an Evanescent Wave?

**Answer:**
- Wave that decays instead of propagates
- Exists in classically forbidden region
- Carries no energy (standing wave)
- Enables tunneling if thin barrier

---

### Question 14: Why are Some Nuclei Stable?

**Answer:**
- Alpha particles are bound inside
- Barrier too thick to tunnel effectively
- $$\mathcal{T}$$ is incredibly small
- Half-life longer than universe age

---

### Question 15: How Does a Tunnel Diode Work?

**Answer:**
- Electrons tunnel between bands
- Sharp junction creates thin barrier
- Current increases with voltage... then decreases!
- Negative resistance enables oscillators

---

## Month 37 Integration Questions

### Question 16: Connect All Month's Topics

**Answer:**
1. Mathematical framework (Week 145) provides operators
2. Measurement (Week 146) gives probabilities
3. 1D systems (Week 147) are standard examples
4. Tunneling (Week 148) shows quantum effects

---

### Question 17: Why is This Material on Qualifying Exams?

**Answer:**
1. Foundations of all quantum physics
2. Mathematical techniques apply everywhere
3. Physical intuition essential for research
4. Problem-solving skills transfer

---

*Oral Practice for Week 148 — Tunneling and WKB*
