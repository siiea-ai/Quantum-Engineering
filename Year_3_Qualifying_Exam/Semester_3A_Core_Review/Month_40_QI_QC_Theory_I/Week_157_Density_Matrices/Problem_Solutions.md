# Week 157: Density Matrices - Problem Solutions

## Section A: Pure States and Basic Properties

### Solution 1

Given $$|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$$

**Density matrix:**
$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} 1/\sqrt{3} \\ \sqrt{2/3} \end{pmatrix}\begin{pmatrix} 1/\sqrt{3} & \sqrt{2/3} \end{pmatrix}$$

$$\rho = \begin{pmatrix} 1/3 & \sqrt{2}/3 \\ \sqrt{2}/3 & 2/3 \end{pmatrix}$$

**Verification:**

1. **Hermiticity**: $$\rho = \rho^\dagger$$ ✓ (matrix is real symmetric)

2. **Unit trace**: $$\text{Tr}(\rho) = 1/3 + 2/3 = 1$$ ✓

3. **Positivity**: Eigenvalues are $$\lambda = 1, 0$$ (since $$\rho^2 = \rho$$ for pure states) ✓

---

### Solution 2

For $$|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$:

**(a) Density matrix:**
$$\rho = |-\rangle\langle-| = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

**(b) Traces:**
$$\text{Tr}(\rho) = \frac{1}{2}(1 + 1) = 1$$

$$\rho^2 = \frac{1}{4}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{4}\begin{pmatrix} 2 & -2 \\ -2 & 2 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} = \rho$$

So $$\text{Tr}(\rho^2) = 1$$ (confirming pure state).

**(c) Eigenvalues:**
$$\det(\rho - \lambda I) = \left(\frac{1}{2} - \lambda\right)^2 - \frac{1}{4} = \lambda^2 - \lambda = \lambda(\lambda - 1) = 0$$

Eigenvalues: $$\lambda_1 = 1$$, $$\lambda_2 = 0$$

---

### Solution 3

For $$|\psi\rangle = \cos\theta|0\rangle + e^{i\phi}\sin\theta|1\rangle$$:

**(a) General density matrix:**
$$\rho = \begin{pmatrix} \cos^2\theta & \cos\theta\sin\theta \cdot e^{-i\phi} \\ \cos\theta\sin\theta \cdot e^{i\phi} & \sin^2\theta \end{pmatrix}$$

**(b) Off-diagonal elements:**
$$\rho_{01} = \cos\theta\sin\theta \cdot e^{-i\phi} = \frac{1}{2}\sin(2\theta)e^{-i\phi}$$
$$\rho_{10} = \cos\theta\sin\theta \cdot e^{i\phi} = \frac{1}{2}\sin(2\theta)e^{i\phi}$$

**(c) Diagonal conditions:**
$$\rho$$ is diagonal when $$\rho_{01} = \rho_{10} = 0$$, which requires:
- $$\sin(2\theta) = 0$$, i.e., $$\theta = 0, \pi/2, \pi, \ldots$$

This means the state is $$|0\rangle$$ or $$|1\rangle$$ (up to global phase).

---

### Solution 4

**(a) Algebraic proof:**
For $$\rho = |\psi\rangle\langle\psi|$$:
$$\rho^2 = |\psi\rangle\langle\psi|\psi\rangle\langle\psi| = |\psi\rangle \cdot 1 \cdot \langle\psi| = |\psi\rangle\langle\psi| = \rho$$

since $$\langle\psi|\psi\rangle = 1$$.

**(b) Eigenvalue implication:**
If $$\rho|v\rangle = \lambda|v\rangle$$, then $$\rho^2|v\rangle = \lambda^2|v\rangle$$.
But $$\rho^2 = \rho$$ implies $$\lambda^2|v\rangle = \lambda|v\rangle$$, so $$\lambda^2 = \lambda$$.
Therefore $$\lambda(\lambda - 1) = 0$$, giving $$\lambda = 0$$ or $$\lambda = 1$$.

**(c) Exactly one eigenvalue equals 1:**
From $$\text{Tr}(\rho) = 1$$ and eigenvalues being only 0 or 1, we need exactly one eigenvalue equal to 1. A pure state has rank 1, so one eigenvalue is 1 and the rest are 0.

---

### Solution 5

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}$$

**Expectation value of $$\sigma_z$$:**
$$\langle\sigma_z\rangle = \text{Tr}(\rho\sigma_z) = \text{Tr}\left(\frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\right)$$

$$= \text{Tr}\left(\frac{1}{2}\begin{pmatrix} 1 & i \\ i & -1 \end{pmatrix}\right) = \frac{1}{2}(1 - 1) = 0$$

**Pure or mixed?**
$$\text{Tr}(\rho^2) = \text{Tr}\left(\frac{1}{4}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}\right) = \text{Tr}\left(\frac{1}{4}\begin{pmatrix} 2 & -2i \\ 2i & 2 \end{pmatrix}\right) = \frac{1}{4}(2+2) = 1$$

This is a **pure state** (can verify: this is $$|+i\rangle\langle+i|$$ where $$|+i\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$$).

---

### Solution 6

**(a) Density matrix:**
$$\rho = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|+\rangle\langle+|$$

$$= \frac{2}{3}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \frac{1}{3}\cdot\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

$$= \begin{pmatrix} 2/3 + 1/6 & 1/6 \\ 1/6 & 1/6 \end{pmatrix} = \begin{pmatrix} 5/6 & 1/6 \\ 1/6 & 1/6 \end{pmatrix}$$

**(b) Purity:**
$$\rho^2 = \begin{pmatrix} 5/6 & 1/6 \\ 1/6 & 1/6 \end{pmatrix}^2 = \begin{pmatrix} 26/36 & 6/36 \\ 6/36 & 2/36 \end{pmatrix} = \begin{pmatrix} 13/18 & 1/6 \\ 1/6 & 1/18 \end{pmatrix}$$

$$\gamma = \text{Tr}(\rho^2) = \frac{13}{18} + \frac{1}{18} = \frac{14}{18} = \frac{7}{9} \approx 0.778$$

Since $$\gamma < 1$$, this is a **mixed state**.

**(c) Probability of measuring $$|0\rangle$$:**
$$P(0) = \langle 0|\rho|0\rangle = \rho_{00} = \frac{5}{6}$$

---

### Solution 7

**Proof:**

Any density matrix $$\rho$$ has spectral decomposition:
$$\rho = \sum_i \lambda_i |i\rangle\langle i|$$

where $$\lambda_i \geq 0$$ (positivity), $$\sum_i \lambda_i = 1$$ (unit trace), and $$\{|i\rangle\}$$ are orthonormal eigenvectors.

Setting $$p_i = \lambda_i$$ and $$|\psi_i\rangle = |i\rangle$$, we have the required decomposition.

For non-orthogonal decompositions, consider that any ensemble $$\{p_i, |\psi_i\rangle\}$$ with $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$ is valid. The spectral decomposition is the unique orthogonal one.

---

### Solution 8

For a $$2 \times 2$$ matrix:
$$\det(\rho) = \lambda_1 \lambda_2$$

Since $$\text{Tr}(\rho) = \lambda_1 + \lambda_2 = 1$$, we have:
$$\text{Tr}(\rho^2) = \lambda_1^2 + \lambda_2^2 = (\lambda_1 + \lambda_2)^2 - 2\lambda_1\lambda_2 = 1 - 2\det(\rho)$$

Therefore:
$$\boxed{\gamma = 1 - 2\det(\rho)}$$

---

## Section B: Mixed States and Ensembles

### Solution 9

**(a) Matrix form:**
$$\rho = \frac{I}{2} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**(b) Verification:**
- Hermiticity: $$\rho = \rho^\dagger$$ ✓
- Unit trace: $$\text{Tr}(\rho) = 1/2 + 1/2 = 1$$ ✓
- Positivity: Eigenvalues are $$1/2, 1/2$$ (both positive) ✓

**(c) Von Neumann entropy:**
$$S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i = -2 \cdot \frac{1}{2}\log_2\frac{1}{2} = -\log_2\frac{1}{2} = 1 \text{ bit}$$

---

### Solution 10

For any orthonormal basis $$\{|i\rangle\}_{i=1}^d$$:

$$\frac{1}{d}\sum_{i=1}^d |i\rangle\langle i| = \frac{1}{d}\sum_{i=1}^d P_i$$

where $$P_i = |i\rangle\langle i|$$ are orthogonal projectors.

By completeness: $$\sum_{i=1}^d |i\rangle\langle i| = I$$

Therefore: $$\frac{1}{d}\sum_{i=1}^d |i\rangle\langle i| = \frac{I}{d}$$

This holds for ANY orthonormal basis, proving the result.

---

### Solution 11

**(a) Alice's ensemble:**
$$\rho_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \frac{1}{2}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Bob's ensemble:**
$$\rho_B = \frac{1}{2}|+\rangle\langle+| + \frac{1}{2}|-\rangle\langle-|$$

$$= \frac{1}{2} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + \frac{1}{2} \cdot \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Both give $$\rho = I/2$$.

**(b) Distinguishability:**
No measurement can distinguish the preparations. The expectation value of any observable $$A$$ is:
$$\langle A \rangle = \text{Tr}(\rho A)$$

which depends only on $$\rho$$, not on the specific ensemble.

**(c) Classical difference:**
In classical probability, knowing the probability distribution uniquely determines the ensemble. In quantum mechanics, infinitely many ensembles produce the same density matrix. This is a genuinely quantum feature related to the impossibility of determining an unknown quantum state.

---

### Solution 12

$$\rho = \frac{1}{4}|0\rangle\langle 0| + \frac{3}{4}|1\rangle\langle 1| = \begin{pmatrix} 1/4 & 0 \\ 0 & 3/4 \end{pmatrix}$$

**(a) Purity:**
$$\gamma = \text{Tr}(\rho^2) = (1/4)^2 + (3/4)^2 = 1/16 + 9/16 = 10/16 = 5/8$$

**(b) Von Neumann entropy:**
$$S = -\frac{1}{4}\log_2\frac{1}{4} - \frac{3}{4}\log_2\frac{3}{4}$$
$$= \frac{1}{4} \cdot 2 + \frac{3}{4}(\log_2 4 - \log_2 3) = \frac{1}{2} + \frac{3}{4}(2 - 1.585)$$
$$= 0.5 + 0.75 \times 0.415 \approx 0.811 \text{ bits}$$

**(c) Bloch vector:**
$$r_x = \text{Tr}(\rho\sigma_x) = 0$$
$$r_y = \text{Tr}(\rho\sigma_y) = 0$$
$$r_z = \text{Tr}(\rho\sigma_z) = 1/4 - 3/4 = -1/2$$

$$\vec{r} = (0, 0, -1/2)$$

---

### Solution 13

**(a) Eigenvalues:**
$$S = -\lambda\log_2\lambda - (1-\lambda)\log_2(1-\lambda) = 0.5$$

Numerical solution (or using binary entropy function $$H_2(\lambda) = 0.5$$):
$$\lambda \approx 0.110$$ or $$\lambda \approx 0.890$$

**(b) General form:**
$$\rho = \lambda|u\rangle\langle u| + (1-\lambda)|v\rangle\langle v|$$

where $$\{|u\rangle, |v\rangle\}$$ is any orthonormal basis and $$\lambda \approx 0.11$$.

**(c) Purity:**
$$\gamma = \lambda^2 + (1-\lambda)^2 = 0.11^2 + 0.89^2 \approx 0.012 + 0.792 = 0.804$$

---

### Solution 14

**Proof:**

Let $$\rho$$ have rank $$r$$, with eigenvalues $$\lambda_1, \ldots, \lambda_r > 0$$ and $$\sum_i \lambda_i = 1$$.

$$S(\rho) = -\sum_{i=1}^r \lambda_i \log_2 \lambda_i$$

By the maximum entropy principle (Lagrange multipliers or Jensen's inequality):
$$S \leq \log_2 r$$

with equality when $$\lambda_i = 1/r$$ for all $$i$$, i.e., $$\rho = P/r$$ where $$P$$ is the projector onto the support of $$\rho$$.

---

## Section C: Bloch Sphere Representation

### Solution 15

**(a) $$|0\rangle$$:**
$$\rho = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$,
$$r_x = 0, r_y = 0, r_z = 1$$
$$\vec{r} = (0, 0, 1)$$ (north pole)

**(b) $$|1\rangle$$:**
$$\rho = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$,
$$r_x = 0, r_y = 0, r_z = -1$$
$$\vec{r} = (0, 0, -1)$$ (south pole)

**(c) $$|+\rangle$$:**
$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$,
$$r_x = 1, r_y = 0, r_z = 0$$
$$\vec{r} = (1, 0, 0)$$ (+x axis)

**(d) $$|+i\rangle$$:**
$$\rho = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}$$,
$$r_x = 0, r_y = 1, r_z = 0$$
$$\vec{r} = (0, 1, 0)$$ (+y axis)

---

### Solution 16

$$\vec{r} = (1/2, 0, \sqrt{3}/2)$$

**(a) Density matrix:**
$$\rho = \frac{1}{2}\left(I + \frac{1}{2}\sigma_x + \frac{\sqrt{3}}{2}\sigma_z\right)$$

$$= \frac{1}{2}\begin{pmatrix} 1 + \sqrt{3}/2 & 1/2 \\ 1/2 & 1 - \sqrt{3}/2 \end{pmatrix}$$

**(b) Pure or mixed?**
$$|\vec{r}|^2 = 1/4 + 0 + 3/4 = 1$$

Since $$|\vec{r}| = 1$$, this is a **pure state**.

**(c) Probability of $$|0\rangle$$:**
$$P(0) = \rho_{00} = \frac{1}{2}\left(1 + \frac{\sqrt{3}}{2}\right) = \frac{2 + \sqrt{3}}{4} \approx 0.933$$

**(d) Eigenvalues:**
For a pure state: $$\lambda_1 = 1$$, $$\lambda_2 = 0$$

---

### Solution 17

$$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$

$$\rho^2 = \frac{1}{4}(I + \vec{r}\cdot\vec{\sigma})^2$$

Using $$(\vec{r}\cdot\vec{\sigma})^2 = |\vec{r}|^2 I$$ (from Pauli algebra):

$$\rho^2 = \frac{1}{4}(I + 2\vec{r}\cdot\vec{\sigma} + |\vec{r}|^2 I) = \frac{1}{4}((1+|\vec{r}|^2)I + 2\vec{r}\cdot\vec{\sigma})$$

$$\text{Tr}(\rho^2) = \frac{1}{4}(2(1+|\vec{r}|^2)) = \frac{1}{2}(1+|\vec{r}|^2)$$

$$\boxed{\gamma = \frac{1}{2}(1 + |\vec{r}|^2)}$$

---

### Solution 18

$$\vec{r} = (0.6, 0, 0.8)$$

**(a) Purity:**
$$|\vec{r}|^2 = 0.36 + 0 + 0.64 = 1$$
$$\gamma = \frac{1}{2}(1 + 1) = 1$$

**(b) Pure state form:**
Since $$|\vec{r}| = 1$$, this is pure. The Bloch vector corresponds to:
$$|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$$

where $$r_z = \cos\theta = 0.8$$ and $$r_x = \sin\theta\cos\phi = 0.6$$.

$$\cos\theta = 0.8 \Rightarrow \theta = \arccos(0.8) \approx 36.87°$$
$$\sin\theta = 0.6$$, $$\cos\phi = 1 \Rightarrow \phi = 0$$

$$|\psi\rangle = \cos(18.43°)|0\rangle + \sin(18.43°)|1\rangle \approx 0.949|0\rangle + 0.316|1\rangle$$

**(c) Probability of spin-up along x:**
$$P(+x) = \langle+|\rho|+\rangle = \frac{1}{2}(1 + r_x) = \frac{1}{2}(1 + 0.6) = 0.8$$

---

### Solution 19

**(a) Bloch vector transformation:**
The Hadamard gate $$H$$ acts as:
$$H\sigma_x H = \sigma_z, \quad H\sigma_y H = -\sigma_y, \quad H\sigma_z H = \sigma_x$$

For the Bloch vector:
$$\vec{r}' = (r_z, -r_y, r_x)$$

This is reflection through the xz-plane followed by exchange of x and z components.

**(b) Verification for $$\vec{r} = (0,0,1)$$:**
$$\vec{r}' = (1, 0, 0)$$

Indeed, $$H|0\rangle = |+\rangle$$, which has Bloch vector $$(1,0,0)$$. ✓

**(c) Rotation axis:**
$$H$$ corresponds to a 180° rotation about the axis $$\hat{n} = \frac{1}{\sqrt{2}}(1, 0, 1)$$ (the direction between +x and +z).

---

### Solution 20

**(a) Global phase effect:**
$$\rho' = U\rho U^\dagger = e^{i\alpha}e^{-i\theta(\hat{n}\cdot\vec{\sigma})/2}\rho e^{i\theta(\hat{n}\cdot\vec{\sigma})/2}e^{-i\alpha}$$

The $$e^{i\alpha}$$ and $$e^{-i\alpha}$$ cancel, so the Bloch vector is unaffected.

**(b) Rotation proof:**
Using the Bloch representation $$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$ and the identity:
$$e^{-i\theta(\hat{n}\cdot\vec{\sigma})/2}(\vec{r}\cdot\vec{\sigma})e^{i\theta(\hat{n}\cdot\vec{\sigma})/2} = (\vec{r}'\cdot\vec{\sigma})$$

where $$\vec{r}'$$ is $$\vec{r}$$ rotated by angle $$\theta$$ about $$\hat{n}$$.

**(c) For the given U:**
$$U = \begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix} = e^{-i(\pi/8)\sigma_z}$$

Rotation axis: $$\hat{n} = (0, 0, 1)$$ (z-axis)
Rotation angle: $$\theta = \pi/4$$ (45°)

---

## Section D: Trace Distance and Fidelity

### Solution 21

**(a)** $$\rho = |0\rangle\langle 0|$$, $$\sigma = |1\rangle\langle 1|$$

$$\rho - \sigma = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

Eigenvalues: $$+1, -1$$

$$D = \frac{1}{2}(|1| + |-1|) = 1$$

**(b)** $$\rho = |0\rangle\langle 0|$$, $$\sigma = |+\rangle\langle+|$$

Bloch vectors: $$\vec{r}_\rho = (0,0,1)$$, $$\vec{r}_\sigma = (1,0,0)$$

$$D = \frac{1}{2}|\vec{r}_\rho - \vec{r}_\sigma| = \frac{1}{2}\sqrt{1 + 1} = \frac{1}{\sqrt{2}} \approx 0.707$$

**(c)** $$\rho = |0\rangle\langle 0|$$, $$\sigma = I/2$$

Bloch vectors: $$\vec{r}_\rho = (0,0,1)$$, $$\vec{r}_\sigma = (0,0,0)$$

$$D = \frac{1}{2}|1| = \frac{1}{2}$$

---

### Solution 22

**(a)** Identical states:
$$F(|\psi\rangle, |\psi\rangle) = |\langle\psi|\psi\rangle|^2 = 1$$

**(b)** $$|0\rangle$$ and $$|+\rangle$$:
$$F = |\langle 0|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**(c)** $$|0\rangle$$ and $$I/2$$:
$$F = \langle 0|\frac{I}{2}|0\rangle = \frac{1}{2}$$

---

### Solution 23

**(a) Trace distance formula:**

$$\rho - \sigma = \frac{1}{2}(\vec{r}_1 - \vec{r}_2)\cdot\vec{\sigma}$$

Let $$\vec{d} = \vec{r}_1 - \vec{r}_2$$. The eigenvalues of $$\vec{d}\cdot\vec{\sigma}$$ are $$\pm|\vec{d}|$$.

$$D = \frac{1}{2} \cdot \frac{1}{2}(|\vec{d}| + |-\vec{d}|) = \frac{1}{2}|\vec{d}| = \frac{1}{2}|\vec{r}_1 - \vec{r}_2|$$

**(b) Maximum distance from maximally mixed:**
$$D_{\max} = \frac{1}{2}|\vec{r}_{\text{pure}} - \vec{0}| = \frac{1}{2} \cdot 1 = \frac{1}{2}$$

**(c) States maximizing distance:**
Any two pure states that are orthogonal (antipodal on Bloch sphere) give $$D = 1$$.

---

### Solution 24

**Fuchs-van de Graaf proof sketch:**

The inequalities relate trace distance and fidelity:
$$1 - \sqrt{F} \leq D \leq \sqrt{1-F}$$

Lower bound saturation: Both pure and orthogonal ($$F=0, D=1$$)
Upper bound saturation: Both pure (any overlap)

For mixed states, the bounds are generally not tight.

---

## Section E: Partial Trace

### Solution 25

**(a) Density matrix:**
$$|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$\rho_{AB} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

**(b) Partial trace over B:**
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \frac{1}{2}\begin{pmatrix} 1+0 & 0+0 \\ 0+0 & 0+1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{I}{2}$$

**(c) Partial trace over A:**
By symmetry: $$\rho_B = \frac{I}{2}$$

**(d) Product test:**
$$\rho_A \otimes \rho_B = \frac{I}{2} \otimes \frac{I}{2} = \frac{I_4}{4} \neq \rho_{AB}$$

This inequality proves the state is **entangled**. A product state would satisfy $$\rho_{AB} = \rho_A \otimes \rho_B$$.

---

### Solution 26

**(a) Full density matrix:**
$$|\psi\rangle = |+\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

$$\rho_{AB} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**(b) Reduced density matrices:**
$$\rho_A = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = |+\rangle\langle+|$$

$$\rho_B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = |0\rangle\langle 0|$$

**(c) Verification:**
$$\rho_A \otimes \rho_B = |+\rangle\langle+| \otimes |0\rangle\langle 0| = |\psi\rangle\langle\psi| = \rho_{AB}$$ ✓

---

### Solution 27

**(a) Reduced density matrix:**
$$\rho_A = \begin{pmatrix} |\alpha|^2 + |\beta|^2 & \alpha\gamma^* + \beta\delta^* \\ \gamma\alpha^* + \delta\beta^* & |\gamma|^2 + |\delta|^2 \end{pmatrix}$$

**(b) Purity condition for $$\rho_A$$:**
$$\rho_A$$ is pure iff $$\text{Tr}(\rho_A^2) = 1$$.

This requires $$\det(\rho_A) = 0$$:
$$(|\alpha|^2 + |\beta|^2)(|\gamma|^2 + |\delta|^2) = |\alpha\gamma^* + \beta\delta^*|^2$$

**(c) Product state equivalence:**
The condition $$\det(\rho_A) = 0$$ is equivalent to the state being writable as:
$$|\psi\rangle = (a|0\rangle + b|1\rangle) \otimes (c|0\rangle + d|1\rangle)$$

i.e., $$\alpha\delta = \beta\gamma$$ (product state condition).

---

### Solution 28

$$|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

**(a) Two-party reduced state:**
$$\rho_{AB} = \text{Tr}_C(|\text{GHZ}\rangle\langle\text{GHZ}|)$$

$$= \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$

This is a classical mixture, not entangled!

**(b) Single-party reduced state:**
$$\rho_A = \text{Tr}_{BC}(|\text{GHZ}\rangle\langle\text{GHZ}|) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

**(c) Entropies:**
$$S(\rho_A) = 1$$ bit (maximally mixed qubit)

$$S(\rho_{AB}) = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1$$ bit

Interestingly, $$S(\rho_A) = S(\rho_{AB})$$ for GHZ states.

---

## Section F: Advanced Topics

### Solution 29

**(a) Purification:**
$$\rho_A = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1|$$

Using the spectral decomposition:
$$|\Psi\rangle_{AB} = \sqrt{\frac{2}{3}}|0\rangle_A|0\rangle_B + \sqrt{\frac{1}{3}}|1\rangle_A|1\rangle_B$$

**(b) Uniqueness:**
Purifications are not unique. Any purification of the form:
$$|\Psi'\rangle_{AB} = \sqrt{\frac{2}{3}}|0\rangle_A|u_0\rangle_B + \sqrt{\frac{1}{3}}|1\rangle_A|u_1\rangle_B$$

where $$\{|u_0\rangle, |u_1\rangle\}$$ are orthonormal, works.

More generally, all purifications are related by unitaries on B:
$$|\Psi'\rangle = (I_A \otimes U_B)|\Psi\rangle$$

**(c) Schmidt decomposition:**
The purification above IS already in Schmidt form:
- Schmidt coefficients: $$\sqrt{2/3}$$, $$\sqrt{1/3}$$
- Schmidt rank: 2

---

### Solution 30

**(a) Extremality of pure states:**

A density matrix $$\rho$$ is extremal iff it cannot be written as:
$$\rho = p\rho_1 + (1-p)\rho_2$$

for distinct $$\rho_1, \rho_2$$ and $$0 < p < 1$$.

**Proof that pure states are extremal:**
Suppose $$|\psi\rangle\langle\psi| = p\rho_1 + (1-p)\rho_2$$.

Taking trace with $$|\psi\rangle\langle\psi|$$:
$$1 = p\langle\psi|\rho_1|\psi\rangle + (1-p)\langle\psi|\rho_2|\psi\rangle$$

Since $$\langle\psi|\rho_i|\psi\rangle \leq 1$$, equality requires $$\langle\psi|\rho_i|\psi\rangle = 1$$ for both.

This means $$\rho_i = |\psi\rangle\langle\psi|$$ (the only state with unit overlap), so $$\rho_1 = \rho_2 = \rho$$.

**Proof that mixed states are not extremal:**
Any mixed state has $$\rho = \sum_i \lambda_i |i\rangle\langle i|$$ with at least two nonzero $$\lambda_i$$.
This is explicitly a non-trivial convex combination.

**(b) Caratheodory bound:**
A density matrix on $$\mathbb{C}^d$$ is an element of a $$(d^2-1)$$-dimensional real vector space (Hermitian, traceless part).

By Caratheodory's theorem, any point in a convex hull can be written as a combination of at most $$d^2$$ extremal points (pure states).

**(c) Geometric picture for qubits:**
The Bloch ball is a 3D ball. Any interior point (mixed state) lies on a line segment connecting two surface points (pure states).

Explicitly: for Bloch vector $$\vec{r}$$ with $$|\vec{r}| < 1$$:
$$\rho = \frac{1+|\vec{r}|}{2}|\hat{r}\rangle\langle\hat{r}| + \frac{1-|\vec{r}|}{2}|-\hat{r}\rangle\langle-\hat{r}|$$

where $$|\pm\hat{r}\rangle$$ are the pure states at $$\pm\hat{r}$$ on the Bloch sphere.

---

## Bonus Solutions

### Bonus 1

**(a) Klein's inequality:**
$$S(\rho||\sigma) = \text{Tr}(\rho\log\rho - \rho\log\sigma) \geq 0$$

Proof uses the operator convexity of $$-\log$$ and the Peierls-Bogoliubov inequality.

**(b) Asymmetry:**
$$S(\rho||\sigma) \neq S(\sigma||\rho)$$ in general. It's not symmetric.

**(c) Triangle inequality:**
Does NOT hold. Relative entropy is not a metric.

---

### Bonus 2

**(a) Non-negativity:**
By strong subadditivity of von Neumann entropy:
$$S(AB) + S(B) \leq S(A) + S(AB)$$

This implies $$S(A) + S(B) \geq S(AB)$$, so $$I(A:B) \geq 0$$.

**(b) Bell state:**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$S(\rho_{AB}) = 0$$ (pure state)
$$S(\rho_A) = S(\rho_B) = 1$$ (maximally mixed)

$$I(A:B) = 1 + 1 - 0 = 2$$ bits

**(c) Product state:**
For $$\rho_{AB} = \rho_A \otimes \rho_B$$:
$$S(\rho_{AB}) = S(\rho_A) + S(\rho_B)$$

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_A) - S(\rho_B) = 0$$

---

*Solutions complete. Review any problems you found challenging before the oral examination.*
