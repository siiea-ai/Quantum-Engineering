# Week 155: Scattering Theory - Problem Solutions

## Part I: Scattering Fundamentals and Born Approximation

### Solution 1: Cross Section Basics

**(a) Differential cross sections:**

$$\frac{d\sigma}{d\Omega} = \frac{\text{detected rate}}{(\text{incident flux})(\text{solid angle})}$$

At $\theta = 30°$:
$$\frac{d\sigma}{d\Omega}(30°) = \frac{100}{10^6 \times 10^{-3}} = \boxed{0.1 \text{ m}^2/\text{sr}}$$

At $\theta = 60°$:
$$\frac{d\sigma}{d\Omega}(60°) = \frac{25}{10^6 \times 10^{-3}} = \boxed{0.025 \text{ m}^2/\text{sr}}$$

**(b) Isotropic scattering:**

$$\sigma_{\text{tot}} = 4\pi \frac{d\sigma}{d\Omega} = 4\pi \times 0.1 = \boxed{1.26 \text{ m}^2}$$

(Using the 30° value; the 60° value would give a different answer, indicating non-isotropic scattering.)

**(c) $\cos^2\theta$ distribution:**

Check: $\frac{d\sigma/d\Omega(30°)}{d\sigma/d\Omega(60°)} = \frac{\cos^2 30°}{\cos^2 60°} = \frac{3/4}{1/4} = 3 \neq 4$

Actually, let's use the data: $0.1/0.025 = 4$

And $\cos^2(30°)/\cos^2(60°) = (0.866)^2/(0.5)^2 = 0.75/0.25 = 3$

This doesn't match exactly, so let's find $A$ from the data:

$A\cos^2(30°) = A(0.75) = 0.1 \Rightarrow A = 0.133$ m$^2$/sr

$$\sigma_{\text{tot}} = \int A\cos^2\theta \,d\Omega = 2\pi A\int_0^\pi \cos^2\theta\sin\theta\,d\theta = 2\pi A \times \frac{2}{3}$$

$$\boxed{\sigma_{\text{tot}} = \frac{4\pi A}{3} \approx 0.56 \text{ m}^2}$$

---

### Solution 3: Yukawa Potential

**(a) Fourier transform:**

$$\tilde{V}(\mathbf{q}) = \int V_0\frac{e^{-\mu r}}{\mu r}e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$$

Using spherical coordinates with $\mathbf{q}$ along $z$-axis:
$$= \frac{V_0}{\mu}\int_0^\infty e^{-\mu r}r\,dr \int_0^\pi e^{-iqr\cos\theta}\sin\theta\,d\theta \times 2\pi$$

$$= \frac{2\pi V_0}{\mu}\int_0^\infty e^{-\mu r}r \cdot \frac{2\sin(qr)}{qr}dr = \frac{4\pi V_0}{\mu q}\int_0^\infty e^{-\mu r}\sin(qr)\,dr$$

Using $\int_0^\infty e^{-\mu r}\sin(qr)dr = \frac{q}{\mu^2 + q^2}$:

$$\boxed{\tilde{V}(q) = \frac{4\pi V_0}{\mu(q^2 + \mu^2)}}$$

**(b) Born scattering amplitude:**

$$f^{(1)} = -\frac{m}{2\pi\hbar^2}\tilde{V}(q) = \boxed{-\frac{2mV_0}{\hbar^2\mu(q^2 + \mu^2)}}$$

**(c) Differential cross section:**

$$\frac{d\sigma}{d\Omega} = |f|^2 = \boxed{\left(\frac{2mV_0}{\hbar^2\mu}\right)^2\frac{1}{(q^2 + \mu^2)^2}}$$

With $q = 2k\sin(\theta/2)$:
$$\frac{d\sigma}{d\Omega} = \left(\frac{2mV_0}{\hbar^2\mu}\right)^2\frac{1}{(4k^2\sin^2(\theta/2) + \mu^2)^2}$$

**(d) Coulomb limit ($\mu \to 0$):**

For Coulomb: $V_0/\mu \to Ze^2/(4\pi\epsilon_0)$

$$\frac{d\sigma}{d\Omega} \to \left(\frac{Ze^2}{8\pi\epsilon_0\hbar^2 k^2\sin^2(\theta/2)}\right)^2 = \left(\frac{Ze^2}{16\pi\epsilon_0 E}\right)^2\frac{1}{\sin^4(\theta/2)}$$

This is the **Rutherford formula**! $\square$

---

### Solution 5: Square Well Born

**(a) Fourier transform:**

$$\tilde{V}(q) = -V_0\int_0^a 4\pi r^2 \cdot \frac{\sin(qr)}{qr}dr = -\frac{4\pi V_0}{q}\int_0^a r\sin(qr)dr$$

Using integration by parts:
$$\int r\sin(qr)dr = -\frac{r\cos(qr)}{q} + \frac{\sin(qr)}{q^2}$$

$$\boxed{\tilde{V}(q) = \frac{4\pi V_0}{q^3}[\sin(qa) - qa\cos(qa)]}$$

**(b) Born amplitude:**

$$f^{(1)} = -\frac{m}{2\pi\hbar^2}\tilde{V}(q) = -\frac{2mV_0}{\hbar^2 q^3}[\sin(qa) - qa\cos(qa)]$$

**(c) Low-energy limit ($ka \ll 1$, hence $qa \ll 1$):**

Taylor expand: $\sin(qa) \approx qa - (qa)^3/6$, $\cos(qa) \approx 1 - (qa)^2/2$

$$\sin(qa) - qa\cos(qa) \approx qa - \frac{(qa)^3}{6} - qa + \frac{(qa)^3}{2} = \frac{(qa)^3}{3}$$

$$f \approx -\frac{2mV_0}{\hbar^2}\frac{a^3}{3} = -\frac{2mV_0 a^3}{3\hbar^2}$$

This is independent of $q$ (and hence $\theta$).

$$\boxed{\text{Cross section is isotropic at low energy}}$$

---

## Part II: Partial Wave Analysis

### Solution 9: Phase Shift Definition

**(a) Asymptotic behavior:**

For a free particle: $R_\ell(r) \xrightarrow{r\to\infty} \frac{A}{kr}\sin(kr - \ell\pi/2)$

With potential: $R_\ell(r) \xrightarrow{r\to\infty} \frac{A'}{kr}\sin(kr - \ell\pi/2 + \delta_\ell)$

The **phase shift** $\delta_\ell$ is the extra phase acquired due to the potential.

**(b) Partial wave amplitude:**

The scattering amplitude contributes the difference between scattered and incident waves.

In partial waves: $f_\ell = \frac{e^{2i\delta_\ell} - 1}{2ik}$

**(c) Alternative form:**

$$f_\ell = \frac{e^{2i\delta_\ell} - 1}{2ik} = \frac{e^{i\delta_\ell}(e^{i\delta_\ell} - e^{-i\delta_\ell})}{2ik} = \frac{e^{i\delta_\ell} \cdot 2i\sin\delta_\ell}{2ik}$$

$$\boxed{f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}}$$

---

### Solution 11: Hard Sphere Phase Shift

**(a) Boundary condition:**

The wavefunction must vanish at $r = a$: $\psi(a) = 0$

**(b) Calculate $\delta_0$:**

For s-wave, the radial function outside is:
$$u(r) = A\sin(kr + \delta_0)$$

Boundary condition: $u(a) = A\sin(ka + \delta_0) = 0$

$$ka + \delta_0 = n\pi$$

Taking the principal value:
$$\boxed{\delta_0 = -ka}$$

(The phase shift is negative for a repulsive potential.)

**(c) Limits:**

**Low energy ($ka \ll 1$):**
$$\delta_0 \approx -ka$$
$$\sigma_0 = \frac{4\pi}{k^2}\sin^2\delta_0 \approx \frac{4\pi}{k^2}(ka)^2 = 4\pi a^2$$

**High energy ($ka \gg 1$):**
Many partial waves contribute. The total cross section approaches $2\pi a^2$ (geometric plus diffraction).

---

### Solution 12: Square Well Phase Shift

**(a) Radial equations:**

Inside ($r < a$): $u'' + K^2 u = 0$ where $K = \sqrt{2m(E+V_0)}/\hbar$

Outside ($r > a$): $u'' + k^2 u = 0$ where $k = \sqrt{2mE}/\hbar$

**(b) Matching:**

Inside: $u = A\sin(Kr)$
Outside: $u = B\sin(kr + \delta_0)$

Continuity of $u$ and $u'$ at $r = a$:
$$A\sin(Ka) = B\sin(ka + \delta_0)$$
$$AK\cos(Ka) = Bk\cos(ka + \delta_0)$$

Dividing:
$$K\cot(Ka) = k\cot(ka + \delta_0)$$

$$\boxed{\tan\delta_0 = \frac{k\tan(Ka) - K\tan(ka)}{K + k\tan(Ka)\tan(ka)}}$$

**(c) Resonance condition ($\delta_0 = \pi/2$):**

At resonance, $\cot\delta_0 = 0$, so:
$$K\cot(Ka) = k\cot(ka + \pi/2) = -k\tan(ka)$$

For $ka \ll 1$: $\tan(ka) \approx ka$
$$K\cot(Ka) \approx -k^2 a$$

This occurs when $Ka \approx (n+1/2)\pi$, i.e., near a virtual bound state.

---

### Solution 15: Optical Theorem Proof

**(a) Proof:**

Forward scattering: $P_\ell(1) = 1$ for all $\ell$

$$f(0) = \sum_\ell(2\ell+1)f_\ell = \sum_\ell(2\ell+1)\frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$

$$\text{Im}[f(0)] = \sum_\ell(2\ell+1)\frac{\sin^2\delta_\ell}{k}$$

Total cross section from partial waves:
$$\sigma_{\text{tot}} = \sum_\ell\frac{4\pi(2\ell+1)}{k^2}\sin^2\delta_\ell$$

Therefore:
$$\boxed{\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]}$$

**(b) Physical interpretation:**

Forward scattering interferes with the incident wave, creating a "shadow" behind the target. The flux removed from the forward direction equals the total scattered flux.

**(c) Conservation of probability:**

The optical theorem is a direct consequence of unitarity (probability conservation). If particles scatter, they must be removed from the incident beam.

---

### Solution 19: Breit-Wigner Formula

**(a) Near resonance:**

Define the resonance energy $E_R$ where $\delta_\ell(E_R) = \pi/2$. Near this energy:
$$\delta_\ell(E) \approx \frac{\pi}{2} + (E - E_R)\left.\frac{d\delta_\ell}{dE}\right|_{E_R}$$

Define $\Gamma = 2\left(\frac{d\delta_\ell}{dE}\right)^{-1}_{E_R}$:
$$\delta_\ell \approx \frac{\pi}{2} + \frac{2(E-E_R)}{\Gamma}$$

$$\cot\delta_\ell = -\tan\left(\frac{2(E-E_R)}{\Gamma}\right) \approx -\frac{2(E-E_R)}{\Gamma} = \frac{2(E_R - E)}{\Gamma}$$

**(b) Breit-Wigner amplitude:**

$$f_\ell = \frac{1}{k(\cot\delta_\ell - i)} = \frac{1}{k\left(\frac{2(E_R-E)}{\Gamma} - i\right)}$$

$$= \frac{\Gamma/2k}{(E_R - E) - i\Gamma/2} = \frac{-\Gamma/2k}{(E - E_R) + i\Gamma/2}$$

For the cross section, we care about $|f_\ell|^2$:
$$\boxed{|f_\ell|^2 = \frac{(\Gamma/2)^2/k^2}{(E-E_R)^2 + (\Gamma/2)^2}}$$

**(c) Cross section values:**

At resonance ($E = E_R$):
$$\sigma_\ell^{\max} = \frac{4\pi(2\ell+1)}{k^2}$$
This is the **unitarity limit**.

At $E = E_R \pm \Gamma/2$:
$$\sigma_\ell = \frac{4\pi(2\ell+1)}{k^2} \times \frac{1}{2} = \frac{\sigma_\ell^{\max}}{2}$$

Hence $\Gamma$ is the **full width at half maximum**.

---

### Solution 22: Coulomb Scattering

**(a) Difficulties:**

The Coulomb potential $V \sim 1/r$ falls off too slowly. The asymptotic wavefunction is not a simple plane wave plus spherical wave:
$$\psi \sim e^{ikz + i\eta\ln[k(r-z)]}$$

The logarithmic phase never settles to a constant. Standard partial wave analysis doesn't work directly.

**(b) Classical Rutherford derivation:**

Impact parameter $b$ related to scattering angle:
$$b = \frac{Z_1Z_2e^2}{8\pi\epsilon_0 E}\cot(\theta/2)$$

Cross section:
$$d\sigma = 2\pi b|db| = \pi b^2 \cdot \frac{2|db|}{b}$$

Computing:
$$\frac{d\sigma}{d\Omega} = \frac{b}{\sin\theta}\left|\frac{db}{d\theta}\right| = \left(\frac{Z_1Z_2e^2}{16\pi\epsilon_0 E}\right)^2\frac{1}{\sin^4(\theta/2)}$$

**(c) Why quantum = classical:**

The Coulomb potential is special because:
1. The Born series sums exactly to the classical result
2. There's no natural length scale (no screening)
3. The potential couples all partial waves equally

This is related to the $1/r^2$ force law and hidden symmetry (Runge-Lenz vector).

---

### Solution 24: Neutron-Proton Scattering

**(a) Total cross section:**

At low energy, only s-wave contributes:
$$\sigma = 4\pi a^2$$

But neutron and proton have spin 1/2 each, giving total spin $S = 0$ (singlet) or $S = 1$ (triplet).

Statistical weights: singlet 1/4, triplet 3/4.

$$\sigma = \frac{1}{4}(4\pi a_s^2) + \frac{3}{4}(4\pi a_t^2) = \pi(a_s^2 + 3a_t^2)$$

$$= \pi((23.7)^2 + 3(5.4)^2) \text{ fm}^2 = \pi(562 + 87.5) \text{ fm}^2$$

$$\boxed{\sigma \approx 20.4 \text{ barns}}$$

**(b) Negative scattering length:**

Negative $a_s$ means there's NO bound state in the singlet channel, but there's a "virtual state" - a pole of the S-matrix on the unphysical sheet.

The large magnitude indicates a near-threshold resonance.

**(c) Deuteron connection:**

For a shallow bound state with binding energy $B$:
$$a \approx \frac{\hbar}{\sqrt{2\mu B}}$$

where $\mu$ is the reduced mass.

For deuteron: $B = 2.2$ MeV, $\mu \approx m_p/2$:
$$a \approx \frac{197 \text{ MeV}\cdot\text{fm}}{\sqrt{2 \times 470 \text{ MeV} \times 2.2 \text{ MeV}}} \approx 4.3 \text{ fm}$$

Close to experimental $a_t = 5.4$ fm. $\checkmark$

---

## Summary of Key Results

### Born Approximation
- $f_{\text{Born}} \propto$ Fourier transform of potential
- Yukawa: $f \propto 1/(q^2 + \mu^2)$
- Valid for weak/high-energy scattering

### Partial Waves
- $f_\ell = e^{i\delta_\ell}\sin\delta_\ell/k$
- Hard sphere: $\delta_0 = -ka$
- Resonance: $\delta_\ell = \pi/2$

### Optical Theorem
- $\sigma_{\text{tot}} = (4\pi/k)\text{Im}[f(0)]$
- Consequence of unitarity

### Resonances
- Breit-Wigner: Lorentzian shape
- Width = lifetime$^{-1}$
- Peak cross section = unitarity limit
