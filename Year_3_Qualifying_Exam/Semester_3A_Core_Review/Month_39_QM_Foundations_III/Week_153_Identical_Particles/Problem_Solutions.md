# Week 153: Identical Particles - Problem Solutions

## Level A Solutions

### Solution 1: Exchange Operator Properties

**(a) Show that $\hat{P}_{12}$ is Hermitian:**

For any two-particle states $|\psi\rangle$ and $|\phi\rangle$:
$$\langle\phi|\hat{P}_{12}|\psi\rangle = \int\int \phi^*(\mathbf{r}_2, \mathbf{r}_1)\psi(\mathbf{r}_1, \mathbf{r}_2)d^3r_1 d^3r_2$$

Relabeling dummy integration variables $\mathbf{r}_1 \leftrightarrow \mathbf{r}_2$:
$$= \int\int \phi^*(\mathbf{r}_1, \mathbf{r}_2)\psi(\mathbf{r}_2, \mathbf{r}_1)d^3r_1 d^3r_2 = \langle\hat{P}_{12}\phi|\psi\rangle$$

Therefore $\hat{P}_{12}^\dagger = \hat{P}_{12}$. $\square$

**(b) Prove $\hat{P}_{12}^2 = \mathbf{1}$:**

$$\hat{P}_{12}^2\psi(\mathbf{r}_1, \mathbf{r}_2) = \hat{P}_{12}[\psi(\mathbf{r}_2, \mathbf{r}_1)] = \psi(\mathbf{r}_1, \mathbf{r}_2)$$

Exchanging twice returns the original function. $\square$

**(c) Eigenvalues of $\hat{P}_{12}$:**

If $\hat{P}_{12}|\psi\rangle = \lambda|\psi\rangle$, then:
$$\hat{P}_{12}^2|\psi\rangle = \lambda^2|\psi\rangle = |\psi\rangle$$

Therefore $\lambda^2 = 1$, giving:
$$\boxed{\lambda = +1 \text{ (symmetric) or } \lambda = -1 \text{ (antisymmetric)}}$$

---

### Solution 2: Two-Particle Wavefunctions

**(a) Symmetric wavefunction:**

$$\boxed{\Psi_S(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\phi_a(\mathbf{r}_1)\phi_b(\mathbf{r}_2) + \phi_a(\mathbf{r}_2)\phi_b(\mathbf{r}_1)]}$$

The $1/\sqrt{2}$ ensures normalization when $\phi_a$ and $\phi_b$ are orthonormal.

**(b) Antisymmetric wavefunction:**

$$\boxed{\Psi_A(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\phi_a(\mathbf{r}_1)\phi_b(\mathbf{r}_2) - \phi_a(\mathbf{r}_2)\phi_b(\mathbf{r}_1)]}$$

**(c) Probability at $\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}$:**

For symmetric:
$$|\Psi_S(\mathbf{r}, \mathbf{r})|^2 = \frac{1}{2}|2\phi_a(\mathbf{r})\phi_b(\mathbf{r})|^2 = 2|\phi_a(\mathbf{r})|^2|\phi_b(\mathbf{r})|^2$$

For antisymmetric:
$$|\Psi_A(\mathbf{r}, \mathbf{r})|^2 = \frac{1}{2}|0|^2 = 0$$

$$\boxed{\text{Bosons: enhanced probability; Fermions: zero probability (exchange hole)}}$$

---

### Solution 3: Pauli Exclusion

**(a) Both fermions in state $\phi$:**

$$\Psi_A = \frac{1}{\sqrt{2}}[\phi(\mathbf{r}_1)\phi(\mathbf{r}_2) - \phi(\mathbf{r}_2)\phi(\mathbf{r}_1)] = \frac{1}{\sqrt{2}}[0] = 0$$

The wavefunction vanishes identically. $\square$

**(b) Two electrons in 1D infinite square well:**

Single-particle energies: $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$

**Same spin state:** Must occupy different spatial states.
$$E_{\min} = E_1 + E_2 = \frac{\pi^2\hbar^2}{2mL^2}(1 + 4) = \boxed{\frac{5\pi^2\hbar^2}{2mL^2}}$$

**Different spin states:** Both can be in ground spatial state.
$$E_{\min} = 2E_1 = \boxed{\frac{\pi^2\hbar^2}{mL^2}}$$

**(c) Helium ground state:**

Both electrons are in the 1s orbital (same spatial state). For the total wavefunction to be antisymmetric:
- Spatial part: symmetric $\phi_{1s}(\mathbf{r}_1)\phi_{1s}(\mathbf{r}_2)$
- Spin part: must be antisymmetric = singlet

$$\boxed{\text{Ground state is spin singlet: } \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)}$$

---

### Solution 4: Slater Determinant - Two Electrons

For electrons in states $\psi_0(x)|\uparrow\rangle$ and $\psi_1(x)|\uparrow\rangle$:

$$\Psi(x_1, s_1; x_2, s_2) = \frac{1}{\sqrt{2}}\begin{vmatrix}
\psi_0(x_1)|\uparrow\rangle_1 & \psi_0(x_2)|\uparrow\rangle_2 \\
\psi_1(x_1)|\uparrow\rangle_1 & \psi_1(x_2)|\uparrow\rangle_2
\end{vmatrix}$$

$$= \frac{1}{\sqrt{2}}[\psi_0(x_1)\psi_1(x_2) - \psi_0(x_2)\psi_1(x_1)]|\uparrow\uparrow\rangle$$

Using harmonic oscillator functions:
$$\boxed{\Psi = \frac{1}{\sqrt{2}}[\psi_0(x_1)\psi_1(x_2) - \psi_0(x_2)\psi_1(x_1)]|\uparrow\uparrow\rangle}$$

---

### Solution 5: Slater Determinant - Three Electrons (Lithium)

Configuration: $1s^2 2s^1$. One possible assignment:
- $\phi_1 = \psi_{1s}|\uparrow\rangle$
- $\phi_2 = \psi_{1s}|\downarrow\rangle$
- $\phi_3 = \psi_{2s}|\uparrow\rangle$

$$\boxed{\Psi = \frac{1}{\sqrt{6}}\begin{vmatrix}
\psi_{1s}(1)\alpha(1) & \psi_{1s}(2)\alpha(2) & \psi_{1s}(3)\alpha(3) \\
\psi_{1s}(1)\beta(1) & \psi_{1s}(2)\beta(2) & \psi_{1s}(3)\beta(3) \\
\psi_{2s}(1)\alpha(1) & \psi_{2s}(2)\alpha(2) & \psi_{2s}(3)\alpha(3)
\end{vmatrix}}$$

where $\alpha = |\uparrow\rangle$ and $\beta = |\downarrow\rangle$.

---

### Solution 6: Bosonic Commutation Relations

**(a) Verify $[a, a^\dagger] = 1$:**

$$aa^\dagger|n\rangle = a\sqrt{n+1}|n+1\rangle = \sqrt{n+1}\sqrt{n+1}|n\rangle = (n+1)|n\rangle$$

$$a^\dagger a|n\rangle = a^\dagger\sqrt{n}|n-1\rangle = \sqrt{n}\sqrt{n}|n\rangle = n|n\rangle$$

Therefore:
$$[a, a^\dagger]|n\rangle = (n+1-n)|n\rangle = |n\rangle$$

$$\boxed{[a, a^\dagger] = 1}$$

**(b) $[a, (a^\dagger)^n] = n(a^\dagger)^{n-1}$:**

By induction. Base case $n=1$: $[a, a^\dagger] = 1$. ✓

Assume true for $n-1$. Then:
$$[a, (a^\dagger)^n] = [a, a^\dagger(a^\dagger)^{n-1}] = a^\dagger[a, (a^\dagger)^{n-1}] + [a, a^\dagger](a^\dagger)^{n-1}$$
$$= a^\dagger(n-1)(a^\dagger)^{n-2} + (a^\dagger)^{n-1} = (n-1)(a^\dagger)^{n-1} + (a^\dagger)^{n-1}$$
$$\boxed{= n(a^\dagger)^{n-1}}$$

**(c) $\langle n|a^\dagger a|n\rangle$:**

$$\langle n|a^\dagger a|n\rangle = \langle n|n|n\rangle = \boxed{n}$$

---

### Solution 7: Fermionic Anticommutators

**(a) $(c^\dagger)^2 = 0$:**

From $\{c^\dagger, c^\dagger\} = 2(c^\dagger)^2 = 0$:
$$\boxed{(c^\dagger)^2 = 0}$$

Similarly, $\{c, c\} = 2c^2 = 0$:
$$\boxed{c^2 = 0}$$

**(b) $c^\dagger c + cc^\dagger = 1$:**

This is exactly the definition of $\{c, c^\dagger\} = 1$. $\square$

**(c) Eigenvalues of $\hat{n} = c^\dagger c$:**

From $\{c, c^\dagger\} = 1$: $cc^\dagger = 1 - c^\dagger c$

$$\hat{n}^2 = c^\dagger c c^\dagger c = c^\dagger(1 - c^\dagger c)c = c^\dagger c - c^\dagger c^\dagger cc = c^\dagger c = \hat{n}$$

Since $\hat{n}^2 = \hat{n}$, the eigenvalues satisfy $\lambda^2 = \lambda$, so $\lambda = 0$ or $\lambda = 1$.

$$\boxed{\text{Eigenvalues: } n = 0 \text{ or } n = 1}$$

---

### Solution 8: Number Operator

**(a) Total number operator:**

$$\boxed{\hat{N} = a_1^\dagger a_1 + a_2^\dagger a_2 = \hat{n}_1 + \hat{n}_2}$$

**(b) Commutators:**

$$[\hat{N}, a_i^\dagger] = [a_i^\dagger a_i, a_i^\dagger] = a_i^\dagger[a_i, a_i^\dagger] = \boxed{a_i^\dagger}$$

$$[\hat{N}, a_i] = [a_i^\dagger a_i, a_i] = [a_i^\dagger, a_i]a_i = \boxed{-a_i}$$

**(c) Interpretation:**

$[\hat{N}, a_i^\dagger] = +a_i^\dagger$ means $a_i^\dagger$ **increases** particle number by 1.

$[\hat{N}, a_i] = -a_i$ means $a_i$ **decreases** particle number by 1.

---

### Solution 9: Vacuum State

**(a) Normalization of $|1_i\rangle = a_i^\dagger|0\rangle$:**

$$\langle 1_i|1_i\rangle = \langle 0|a_i a_i^\dagger|0\rangle = \langle 0|(a_i^\dagger a_i + 1)|0\rangle = \langle 0|0\rangle = \boxed{1}$$

**(b) Bosonic two-particle state:**

$$|1_1, 1_2\rangle = a_1^\dagger a_2^\dagger|0\rangle$$

This is automatically normalized:
$$\langle 1_1, 1_2|1_1, 1_2\rangle = \langle 0|a_2 a_1 a_1^\dagger a_2^\dagger|0\rangle = \langle 0|a_2(1 + a_1^\dagger a_1)a_2^\dagger|0\rangle = \langle 0|a_2 a_2^\dagger|0\rangle = 1$$

**(c) Fermionic two-particle state:**

$$|1_1, 1_2\rangle = c_1^\dagger c_2^\dagger|0\rangle$$

Antisymmetry: $c_1^\dagger c_2^\dagger = -c_2^\dagger c_1^\dagger$ (from anticommutation)

$$\boxed{|1_1, 1_2\rangle = -|1_2, 1_1\rangle}$$

---

### Solution 10: Helium Ground State Energy

**(a) Why spin singlet?**

Both electrons occupy the same spatial orbital (1s), so the spatial wavefunction is symmetric. Total antisymmetry requires antisymmetric spin: the singlet.

**(b) First-order perturbation energy:**

Zeroth order: $E^{(0)} = 2 \times (-Z^2 \cdot 13.6 \text{ eV}) = 2 \times (-4 \times 13.6) = -108.8$ eV

First-order correction:
$$E^{(1)} = \langle V_{ee}\rangle = \frac{e^2}{4\pi\epsilon_0}\langle 1/r_{12}\rangle = \frac{5Z}{8}\frac{e^2}{4\pi\epsilon_0 a_0} = \frac{5 \times 2}{8} \times 27.2 \text{ eV} = 34 \text{ eV}$$

$$\boxed{E_0 \approx -108.8 + 34 = -74.8 \text{ eV}}$$

**(c) Comparison:**

Experimental: $-78.98$ eV

Error: $\frac{78.98 - 74.8}{78.98} \approx 5.3\%$

The perturbation estimate is too high (not negative enough) because first-order perturbation overestimates the effect of electron-electron repulsion.

---

## Level B Solutions

### Solution 11: Exchange Integral Calculation

**(a) Direct integral:**

$$J = \lambda\int_0^L |\psi_1(x)|^2|\psi_2(x)|^2 dx = \lambda \cdot \frac{4}{L^2}\int_0^L \sin^2\left(\frac{\pi x}{L}\right)\sin^2\left(\frac{2\pi x}{L}\right)dx$$

Using $\sin^2\theta = (1-\cos 2\theta)/2$:

$$J = \lambda \cdot \frac{4}{L^2} \cdot \frac{L}{4} = \boxed{\frac{\lambda}{L}}$$

**(b) Exchange integral:**

$$K = \lambda\int_0^L \psi_1^*(x)\psi_2^*(x)\psi_2(x)\psi_1(x)dx = \lambda\int_0^L |\psi_1(x)|^2|\psi_2(x)|^2 dx = J$$

For this specific interaction, $\boxed{K = J = \lambda/L}$.

**(c) Lower energy state:**

Singlet: $E_S = E_0 + J + K$
Triplet: $E_T = E_0 + J - K$

Since $K > 0$: $E_T < E_S$

$$\boxed{\text{Triplet state has lower energy}}$$

---

### Solution 12: Second Quantization of Kinetic Energy

**(a) General form:**

$$\hat{T} = \sum_{\alpha,\beta}\langle\alpha|\frac{\mathbf{p}^2}{2m}|\beta\rangle a_\alpha^\dagger a_\beta$$

**(b) For plane waves:**

$$\langle\mathbf{k}|\frac{\mathbf{p}^2}{2m}|\mathbf{k}'\rangle = \frac{\hbar^2 k^2}{2m}\delta_{\mathbf{k},\mathbf{k}'}$$

$$\boxed{\hat{T} = \sum_{\mathbf{k},\sigma}\frac{\hbar^2 k^2}{2m}c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma}}$$

**(c) Ground state energy of N fermions:**

Fill Fermi sea up to $k_F = (3\pi^2 n)^{1/3}$:

$$E_0 = 2 \times \frac{V}{(2\pi)^3}\int_{k<k_F}\frac{\hbar^2 k^2}{2m}d^3k = \frac{V}{\pi^2}\frac{\hbar^2}{2m}\int_0^{k_F}k^4 dk = \frac{V\hbar^2 k_F^5}{10\pi^2 m}$$

Per particle:
$$\boxed{\frac{E_0}{N} = \frac{3}{5}\epsilon_F = \frac{3\hbar^2 k_F^2}{10m}}$$

---

### Solution 13: Two-Body Interaction

**(a) In terms of field operators:**

$$\boxed{\hat{V} = \frac{1}{2}\int\int \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}^\dagger(\mathbf{r}')v(|\mathbf{r}-\mathbf{r}'|)\hat{\psi}(\mathbf{r}')\hat{\psi}(\mathbf{r})d^3r\,d^3r'}$$

**(b) Momentum space:**

$$\hat{V} = \frac{1}{2V}\sum_{\mathbf{k},\mathbf{k}',\mathbf{q}}v_\mathbf{q}c_{\mathbf{k}+\mathbf{q}}^\dagger c_{\mathbf{k}'-\mathbf{q}}^\dagger c_{\mathbf{k}'}c_{\mathbf{k}}$$

where $v_\mathbf{q} = \int v(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$.

**(c) Direct and exchange terms:**

The direct term has $\mathbf{q} = 0$: scattering without momentum transfer.

The exchange term has $\mathbf{q} = \mathbf{k} - \mathbf{k}'$: particles exchange momenta.

For fermions with same spin, these terms have opposite signs due to antisymmetry.

---

### Solution 14: Fermi Gas Ground State

**(a) Ground state:**

$$\boxed{|GS\rangle = \prod_{|\mathbf{k}| < k_F, \sigma}c_{\mathbf{k}\sigma}^\dagger|0\rangle}$$

All states with $k < k_F$ occupied, spin up and down.

**(b) Fermi energy:**

Number of states: $N = 2 \times \frac{V}{(2\pi)^3}\frac{4\pi k_F^3}{3}$

Solving: $k_F = (3\pi^2 n)^{1/3}$

$$\boxed{\epsilon_F = \frac{\hbar^2 k_F^2}{2m} = \frac{\hbar^2(3\pi^2 n)^{2/3}}{2m}}$$

**(c) Energy per particle:**

$$\boxed{\frac{E_0}{N} = \frac{3}{5}\epsilon_F}$$

---

### Solution 15: Bosonic Ground State

**(a) Ground state energy:**

All N bosons in ground state ($\hbar\omega/2$ each... but they share the same state!):

$$\boxed{E_0 = N \cdot \frac{3\hbar\omega}{2}}$$

(For 3D oscillator, ground state has energy $3\hbar\omega/2$.)

**(b) Second quantization:**

$$\boxed{|GS\rangle = \frac{(a_0^\dagger)^N}{\sqrt{N!}}|0\rangle}$$

where $a_0^\dagger$ creates a particle in the ground state.

**(c) Comparison with fermions:**

Fermions must occupy different states. For spin-polarized fermions:
$$E_{\text{fermion}} = \sum_{n_x,n_y,n_z}^{N}\hbar\omega\left(n_x + n_y + n_z + \frac{3}{2}\right) \gg N\frac{3\hbar\omega}{2}$$

$$\boxed{\text{Fermions have much higher ground state energy due to Pauli exclusion}}$$

---

### Solution 17: Helium Excited States

**(a) Singlet spatial wavefunction:**

$$\boxed{\psi_S(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\phi_{1s}(\mathbf{r}_1)\phi_{2s}(\mathbf{r}_2) + \phi_{1s}(\mathbf{r}_2)\phi_{2s}(\mathbf{r}_1)]}$$

**(b) Triplet spatial wavefunction:**

$$\boxed{\psi_T(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\phi_{1s}(\mathbf{r}_1)\phi_{2s}(\mathbf{r}_2) - \phi_{1s}(\mathbf{r}_2)\phi_{2s}(\mathbf{r}_1)]}$$

**(c) Energy ordering:**

$E_{\text{singlet}} = E_0 + J + K$ (parahelium)
$E_{\text{triplet}} = E_0 + J - K$ (orthohelium)

Since $K > 0$: $\boxed{E_{\text{triplet}} < E_{\text{singlet}}}$

Orthohelium is lower in energy (Hund's rule).

---

## Level C Solutions

### Solution 21: Variational Treatment of Helium

**(a) Kinetic energy:**

For hydrogen-like 1s with effective charge $Z_{\text{eff}}$:
$$\langle T_1\rangle = \frac{Z_{\text{eff}}^2 e^2}{2a_0}$$

Total: $\boxed{\langle T\rangle = \frac{Z_{\text{eff}}^2 e^2}{a_0}}$

**(b) Electron-nucleus potential:**

$$\langle V_{en}\rangle = -\frac{2Ze^2}{a_0}Z_{\text{eff}} \cdot Z = \boxed{-\frac{2Z \cdot Z_{\text{eff}} e^2}{a_0}}$$

(Factor of 2 from two electrons, and the actual nuclear charge is Z=2.)

**(c) Optimal $Z_{\text{eff}}$:**

$$E(Z_{\text{eff}}) = \frac{Z_{\text{eff}}^2 e^2}{a_0} - \frac{4Z_{\text{eff}} e^2}{a_0} + \frac{5Z_{\text{eff}} e^2}{8a_0}$$

$$\frac{dE}{dZ_{\text{eff}}} = \frac{2Z_{\text{eff}} e^2}{a_0} - \frac{4e^2}{a_0} + \frac{5e^2}{8a_0} = 0$$

$$Z_{\text{eff}} = 2 - \frac{5}{16} = \boxed{\frac{27}{16} \approx 1.69}$$

$$E_{\text{min}} = -\left(\frac{27}{16}\right)^2 \times 27.2 \text{ eV} = \boxed{-77.5 \text{ eV}}$$

**(d) Physical interpretation:**

$Z_{\text{eff}} < 2$ because each electron "screens" the nuclear charge from the other. The inner electron reduces the effective attraction felt by the outer electron.

Comparison: Experimental $-78.98$ eV, variational $-77.5$ eV (error $\approx 2\%$).

---

### Solution 22: Three-Fermion System

**(a) Ground state configuration:**

Harmonic oscillator: $E_n = (n + 1/2)\hbar\omega$ for each dimension.

1D: $E_0 = \hbar\omega/2$, $E_1 = 3\hbar\omega/2$, ...

Three spin-1/2 fermions: at most 2 in each spatial state.

Ground state: Two in $n=0$ (one $\uparrow$, one $\downarrow$), one in $n=1$.

$$\boxed{(n=0,\uparrow), (n=0,\downarrow), (n=1,\uparrow) \text{ or } (n=1,\downarrow)}$$

**(b) Ground state energy:**

$$E_0 = 2 \times \frac{\hbar\omega}{2} + \frac{3\hbar\omega}{2} = \boxed{\frac{5\hbar\omega}{2}}$$

**(c) Slater determinant:**

$$\Psi = \frac{1}{\sqrt{6}}\begin{vmatrix}
\psi_0(x_1)\alpha_1 & \psi_0(x_2)\alpha_2 & \psi_0(x_3)\alpha_3 \\
\psi_0(x_1)\beta_1 & \psi_0(x_2)\beta_2 & \psi_0(x_3)\beta_3 \\
\psi_1(x_1)\alpha_1 & \psi_1(x_2)\alpha_2 & \psi_1(x_3)\alpha_3
\end{vmatrix}$$

**(d) Effect of repulsive interaction:**

Repulsive delta interaction: $V = \lambda\sum_{i<j}\delta(x_i - x_j)$ with $\lambda > 0$.

First-order correction vanishes for fermions with same spin (can't be at same position).

For opposite spins: small positive energy shift.

$$\boxed{\text{Energy increases slightly due to electron-electron repulsion}}$$

---

### Solution 23: Exchange Hole

**(a) Same spin - exchange hole:**

For two electrons with same spin, the spatial wavefunction is antisymmetric:
$$\psi(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\phi_k(\mathbf{r}_1)\phi_{k'}(\mathbf{r}_2) - \phi_k(\mathbf{r}_2)\phi_{k'}(\mathbf{r}_1)]$$

At $\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}$:
$$|\psi(\mathbf{r}, \mathbf{r})|^2 = 0$$

$$\boxed{g(0) = 0 \text{ (exchange hole)}}$$

**(b) Opposite spin:**

No exchange symmetry requirement between different spin states.

$$\boxed{g(0) = 1 \text{ (no correlation)}}$$

**(c) Size of exchange hole:**

The exchange hole extends over a distance $\sim 1/k_F$ (Fermi wavelength).

$$\boxed{r_{\text{hole}} \sim \lambda_F = \frac{2\pi}{k_F}}$$

**(d) Effect on Coulomb energy:**

The exchange hole reduces the probability of finding same-spin electrons close together, reducing Coulomb repulsion. This is the exchange energy, which lowers the total energy.

$$\boxed{\text{Exchange energy is negative, stabilizing the system}}$$

---

### Solution 24: Second Quantized Tight-Binding

**(a) Second quantized Hamiltonian:**

First quantization: electron can hop from site $i$ to $i\pm 1$ with amplitude $-t$.

$$\boxed{H = -t\sum_{i,\sigma}(c_{i,\sigma}^\dagger c_{i+1,\sigma} + c_{i+1,\sigma}^\dagger c_{i,\sigma})}$$

**(b) Momentum space transformation:**

$$c_{i,\sigma} = \frac{1}{\sqrt{N}}\sum_k e^{ika\cdot i}c_{k,\sigma}$$

$$H = \sum_{k,\sigma}\epsilon_k c_{k,\sigma}^\dagger c_{k,\sigma}$$

**(c) Dispersion relation:**

$$\boxed{\epsilon(k) = -2t\cos(ka)}$$

Bandwidth: $W = 4t$ (from $-2t$ to $+2t$).

**(d) Mott insulator at half-filling:**

At half-filling: one electron per site on average.

For $U \gg t$: double occupancy costs energy $U$, becomes unfavorable.

Each site has exactly one electron. Hopping is suppressed because it creates double occupancy.

$$\boxed{\text{System becomes a Mott insulator: localized electrons, insulating despite half-filled band}}$$

---

### Solution 25: Bosons in Double Well

**(a) $U = 0$ ground state:**

Non-interacting: all bosons in symmetric superposition.

Single-particle ground state: $|+\rangle = \frac{1}{\sqrt{2}}(|L\rangle + |R\rangle)$ with energy $-J$.

N-particle ground state:
$$|GS\rangle = \frac{(a_+^\dagger)^N}{\sqrt{N!}}|0\rangle$$

$$\boxed{E_0 = -NJ}$$

**(b) $U \gg NJ$ ground state:**

Strong interactions favor equal distribution to minimize $n(n-1)$ terms.

For even N: $N/2$ particles in each well.
$$|GS\rangle \approx |N/2, N/2\rangle$$

$$\boxed{E_0 = U\frac{N}{2}\left(\frac{N}{2}-1\right)}$$

**(c) Quantum phase transition:**

At $U \sim NJ$: transition from superfluid (delocalized) to Mott insulator (localized) phase.

This is the **superfluid-Mott insulator transition**.

**(d) Experimental system:**

$$\boxed{\text{Ultracold atoms in optical lattices (Bose-Hubbard model)}}$$

---

## Summary

These solutions demonstrate the key techniques needed for qualifying exams on identical particles:

1. **Exchange symmetry** - Constructing symmetric/antisymmetric states
2. **Slater determinants** - Systematic antisymmetrization
3. **Second quantization** - Operator formalism for many-body systems
4. **Helium atom** - Paradigmatic two-electron system
5. **Exchange energy** - Origin of Hund's rules

Practice deriving these results from scratch, as oral exams often ask for derivations.
