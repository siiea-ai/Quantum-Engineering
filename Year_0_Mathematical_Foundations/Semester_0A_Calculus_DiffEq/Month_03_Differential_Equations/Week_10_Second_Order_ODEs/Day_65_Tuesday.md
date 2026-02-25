# Day 65: Nonhomogeneous Equations — Undetermined Coefficients

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory & Method |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Examples & Special Cases |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand the structure of nonhomogeneous solutions
2. Apply the method of undetermined coefficients
3. Choose correct trial solutions for various forcing functions
4. Handle special cases when trial solutions overlap homogeneous solutions
5. Solve complete IVPs for nonhomogeneous equations

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 3.5**: Nonhomogeneous Equations; Method of Undetermined Coefficients (pp. 181-194)

---

## 📖 Core Content: Nonhomogeneous Equations

### 1. The General Problem

Solve $ay'' + by' + cy = f(x)$ where $f(x) \neq 0$.

### 2. Solution Structure

> **Theorem:** The general solution to the nonhomogeneous equation is:
> $$y = y_h + y_p$$
> where:
> - $y_h$ = general solution to the **homogeneous** equation ($f = 0$)
> - $y_p$ = any **particular** solution to the nonhomogeneous equation

### 3. Why This Works

If $y_h$ solves $ay'' + by' + cy = 0$ and $y_p$ solves $ay'' + by' + cy = f(x)$, then:
$$a(y_h + y_p)'' + b(y_h + y_p)' + c(y_h + y_p) = 0 + f(x) = f(x)$$ ✓

---

## 📖 Method of Undetermined Coefficients

### 4. The Idea

**Guess** the form of $y_p$ based on the form of $f(x)$, then determine the coefficients.

### 5. Standard Trial Solutions

| $f(x)$ | Trial $y_p$ |
|--------|-------------|
| $ke^{\alpha x}$ | $Ae^{\alpha x}$ |
| $kx^n$ | $A_n x^n + A_{n-1}x^{n-1} + \cdots + A_0$ |
| $k\cos(\beta x)$ or $k\sin(\beta x)$ | $A\cos(\beta x) + B\sin(\beta x)$ |
| $ke^{\alpha x}\cos(\beta x)$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |
| Products/sums | Products/sums of above |

### 6. The Procedure

1. Find $y_h$ (solve homogeneous equation)
2. Guess $y_p$ based on $f(x)$
3. Substitute $y_p$ into the equation
4. Solve for undetermined coefficients
5. General solution: $y = y_h + y_p$

---

## ✏️ Worked Examples

### Example 1: Exponential Forcing
Solve $y'' - 3y' + 2y = e^{3x}$.

**Step 1: Find $y_h$**
$$r^2 - 3r + 2 = (r-1)(r-2) = 0 \Rightarrow r = 1, 2$$
$$y_h = c_1 e^x + c_2 e^{2x}$$

**Step 2: Guess $y_p$**
Since $f(x) = e^{3x}$, try $y_p = Ae^{3x}$

**Step 3: Substitute**
$$y_p' = 3Ae^{3x}, \quad y_p'' = 9Ae^{3x}$$
$$9Ae^{3x} - 3(3Ae^{3x}) + 2(Ae^{3x}) = e^{3x}$$
$$9A - 9A + 2A = 1$$
$$2A = 1 \Rightarrow A = 1/2$$

**Solution:**
$$y = c_1 e^x + c_2 e^{2x} + \frac{1}{2}e^{3x}$$

---

### Example 2: Polynomial Forcing
Solve $y'' + 4y = x^2$.

**Step 1: Find $y_h$**
$$r^2 + 4 = 0 \Rightarrow r = \pm 2i$$
$$y_h = c_1 \cos 2x + c_2 \sin 2x$$

**Step 2: Guess $y_p$**
Since $f(x) = x^2$, try $y_p = Ax^2 + Bx + C$

**Step 3: Substitute**
$$y_p' = 2Ax + B, \quad y_p'' = 2A$$
$$2A + 4(Ax^2 + Bx + C) = x^2$$
$$4Ax^2 + 4Bx + (2A + 4C) = x^2$$

**Match coefficients:**
$$4A = 1 \Rightarrow A = 1/4$$
$$4B = 0 \Rightarrow B = 0$$
$$2A + 4C = 0 \Rightarrow C = -1/8$$

**Solution:**
$$y = c_1 \cos 2x + c_2 \sin 2x + \frac{1}{4}x^2 - \frac{1}{8}$$

---

### Example 3: Trigonometric Forcing
Solve $y'' + y' - 2y = \sin x$.

**Step 1: Find $y_h$**
$$r^2 + r - 2 = (r+2)(r-1) = 0 \Rightarrow r = -2, 1$$
$$y_h = c_1 e^{-2x} + c_2 e^x$$

**Step 2: Guess $y_p$**
Try $y_p = A\cos x + B\sin x$

**Step 3: Substitute**
$$y_p' = -A\sin x + B\cos x$$
$$y_p'' = -A\cos x - B\sin x$$

$$(-A\cos x - B\sin x) + (-A\sin x + B\cos x) - 2(A\cos x + B\sin x) = \sin x$$

**Collect terms:**
- cos x: $-A + B - 2A = -3A + B = 0$
- sin x: $-B - A - 2B = -A - 3B = 1$

**Solve:** $B = 3A$ and $-A - 9A = 1$, so $A = -1/10$, $B = -3/10$

**Solution:**
$$y = c_1 e^{-2x} + c_2 e^x - \frac{1}{10}\cos x - \frac{3}{10}\sin x$$

---

## 📖 Special Case: Overlap with Homogeneous Solution

### 7. The Modification Rule

If the standard trial solution (or any part of it) is already in $y_h$, multiply by $x$ (or $x^2$ if needed).

### Example 4: Exponential Overlap
Solve $y'' - 3y' + 2y = e^{2x}$.

**Step 1: Find $y_h$**
$$y_h = c_1 e^x + c_2 e^{2x}$$

**Step 2: Standard guess would be $Ae^{2x}$, but $e^{2x}$ is in $y_h$!**

Modified guess: $y_p = Axe^{2x}$

**Step 3: Substitute**
$$y_p' = Ae^{2x} + 2Axe^{2x} = A(1 + 2x)e^{2x}$$
$$y_p'' = 2Ae^{2x} + 2A(1 + 2x)e^{2x} = A(4 + 4x)e^{2x}$$

$$A(4 + 4x)e^{2x} - 3A(1 + 2x)e^{2x} + 2Axe^{2x} = e^{2x}$$
$$A[(4 + 4x) - 3(1 + 2x) + 2x]e^{2x} = e^{2x}$$
$$A[4 + 4x - 3 - 6x + 2x] = 1$$
$$A[1] = 1 \Rightarrow A = 1$$

**Solution:**
$$y = c_1 e^x + c_2 e^{2x} + xe^{2x}$$

---

### Example 5: Trig Overlap (Resonance!)
Solve $y'' + 4y = \cos 2x$.

**Step 1: Find $y_h$**
$$y_h = c_1 \cos 2x + c_2 \sin 2x$$

**Step 2:** Standard guess $A\cos 2x + B\sin 2x$ overlaps with $y_h$!

Modified guess: $y_p = x(A\cos 2x + B\sin 2x)$

**Step 3: Substitute** (lengthy but straightforward)

After calculation: $A = 0$, $B = 1/4$

**Solution:**
$$y = c_1 \cos 2x + c_2 \sin 2x + \frac{x}{4}\sin 2x$$

Note: The amplitude grows linearly with x! This is **resonance**.

---

### Example 6: Double Overlap (Repeated Root)
Solve $y'' - 4y' + 4y = e^{2x}$.

**Step 1: Find $y_h$**
$$r^2 - 4r + 4 = (r-2)^2 = 0 \Rightarrow r = 2 \text{ (repeated)}$$
$$y_h = (c_1 + c_2 x)e^{2x}$$

**Step 2:** Both $e^{2x}$ and $xe^{2x}$ are in $y_h$!

Modified guess: $y_p = Ax^2 e^{2x}$

**Step 3:** After substitution: $A = 1/2$

**Solution:**
$$y = (c_1 + c_2 x)e^{2x} + \frac{1}{2}x^2 e^{2x}$$

---

## 📋 Summary: Trial Solutions

| $f(x)$ | Standard Trial | If overlap: multiply by |
|--------|----------------|------------------------|
| $e^{\alpha x}$ | $Ae^{\alpha x}$ | $x$ (or $x^2$) |
| $x^n$ | Polynomial of degree n | $x$ |
| $\cos\beta x$ or $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ | $x$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ | $x$ |

---

## 📝 Practice Problems

### Level 1: Basic
1. Solve $y'' - y = e^{2x}$
2. Solve $y'' + 4y = 8$
3. Solve $y'' - y' - 2y = 4x$

### Level 2: Trigonometric
4. Solve $y'' + 9y = \cos 3x$ (overlap!)
5. Solve $y'' - 4y = \sin x$
6. Solve $y'' + y' = \sin 2x$

### Level 3: With Initial Conditions
7. Solve $y'' + y = x$, $y(0) = 1$, $y'(0) = 0$
8. Solve $y'' - 4y = e^{2x}$, $y(0) = 1$, $y'(0) = 0$ (overlap!)
9. Solve $y'' + 4y' + 4y = e^{-2x}$, $y(0) = 0$, $y'(0) = 1$ (double overlap!)

### Level 4: Combined Forcing
10. Solve $y'' + y = x + \cos x$ (overlap in trig part!)
11. Solve $y'' - y = e^x + e^{-x}$ (overlap in $e^x$ part)
12. Solve $y'' + 4y = x^2 + \sin 2x$

### Level 5: Challenging
13. Solve $y'' - 2y' + y = xe^x$ (all terms overlap!)
14. Find a differential equation $ay'' + by' + cy = f(x)$ whose general solution is $y = c_1 e^x + c_2 e^{-x} + x^2$
15. Explain physically why overlap causes resonance in $y'' + \omega^2 y = \cos\omega x$

---

## 📊 Answers

1. $y = c_1 e^x + c_2 e^{-x} + \frac{1}{3}e^{2x}$
2. $y = c_1 \cos 2x + c_2 \sin 2x + 2$
3. $y = c_1 e^{2x} + c_2 e^{-x} - 2x + 1$
4. $y = c_1 \cos 3x + c_2 \sin 3x + \frac{x}{6}\sin 3x$
5. $y = c_1 e^{2x} + c_2 e^{-2x} - \frac{1}{5}\sin x$
6. $y = c_1 + c_2 e^{-x} - \frac{1}{5}\cos 2x + \frac{2}{5}\sin 2x$
7. $y = \cos x + x$
8. $y = \frac{5}{8}e^{2x} + \frac{3}{8}e^{-2x} + \frac{x}{4}e^{2x}$
9. $y = (x + \frac{1}{2}x^2)e^{-2x}$
10. $y = c_1 \cos x + c_2 \sin x + x + \frac{x}{2}\sin x$
11. $y = c_1 e^x + c_2 e^{-x} + \frac{x}{2}e^x - \frac{1}{2}e^{-x}$
12. $y = c_1 \cos 2x + c_2 \sin 2x + \frac{x^2}{4} - \frac{1}{8} - \frac{x}{4}\sin 2x$
13. $y = (c_1 + c_2 x + \frac{x^3}{6})e^x$
14. $y'' - y = -2$ (verify!)
15. The driving frequency equals the natural frequency

---

## 🔬 Quantum Mechanics Connection

### Driven Quantum Systems

In quantum mechanics, time-dependent perturbations give nonhomogeneous equations:
$$i\hbar\frac{d c_n}{dt} = E_n c_n + V_{nm}(t) c_m$$

When $V_{nm} \propto \cos\omega t$, resonance occurs when $\omega = (E_m - E_n)/\hbar$!

### Fermi's Golden Rule

The transition rate between quantum states due to a periodic perturbation:
$$\Gamma_{n\to m} = \frac{2\pi}{\hbar}|V_{nm}|^2 \delta(E_m - E_n - \hbar\omega)$$

The delta function enforces the resonance condition!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Section 3.5
- [ ] Understand solution structure ($y = y_h + y_p$)
- [ ] Master undetermined coefficients method
- [ ] Recognize and handle overlap cases
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 66: Variation of Parameters**
- A general method that works for ANY forcing function
- No guessing required!
- More powerful but more work

---

*"The particular solution responds to the forcing function—it's the system's answer to external influence."*
