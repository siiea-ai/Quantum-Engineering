# Day 79: Inverse Laplace Transforms

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Inverse Transform Methods |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Partial Fractions |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Find inverse Laplace transforms using tables
2. Decompose rational functions using partial fractions
3. Handle repeated and complex roots
4. Use completing the square for irreducible quadratics
5. Apply the first shifting theorem in reverse

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 6.2**: Solution of Initial Value Problems (pp. 304-315)
- Review: Partial fractions from calculus

---

## 📖 Core Content: Inverse Laplace Transform

### 1. Definition

> **Inverse Transform:** If $\mathcal{L}\{f(t)\} = F(s)$, then:
> $$f(t) = \mathcal{L}^{-1}\{F(s)\}$$

In practice, we use **tables** and **algebraic manipulation** rather than the complex integral formula.

### 2. Key Principle

Break $F(s)$ into pieces that match known transforms:
$$\mathcal{L}^{-1}\{F(s) + G(s)\} = f(t) + g(t)$$

---

## 📖 Partial Fraction Decomposition

### 3. Why Partial Fractions?

Transform rational functions like:
$$F(s) = \frac{s+3}{s^2-5s+6} = \frac{s+3}{(s-2)(s-3)}$$

into simpler pieces we can invert.

### 4. Case 1: Distinct Linear Factors

$$\frac{P(s)}{(s-a_1)(s-a_2)\cdots(s-a_n)} = \frac{A_1}{s-a_1} + \frac{A_2}{s-a_2} + \cdots + \frac{A_n}{s-a_n}$$

**Cover-up method:** To find $A_i$, multiply both sides by $(s-a_i)$ and set $s = a_i$.

---

### Example 1: Distinct Roots
Find $\mathcal{L}^{-1}\left\{\frac{s+3}{(s-2)(s-3)}\right\}$

**Partial fractions:**
$$\frac{s+3}{(s-2)(s-3)} = \frac{A}{s-2} + \frac{B}{s-3}$$

**Cover-up:**
- $A = \frac{(2)+3}{(2)-3} = \frac{5}{-1} = -5$
- $B = \frac{(3)+3}{(3)-2} = \frac{6}{1} = 6$

**Inverse:**
$$f(t) = -5e^{2t} + 6e^{3t}$$

---

### 5. Case 2: Repeated Linear Factors

$$\frac{P(s)}{(s-a)^n} = \frac{A_1}{s-a} + \frac{A_2}{(s-a)^2} + \cdots + \frac{A_n}{(s-a)^n}$$

---

### Example 2: Repeated Roots
Find $\mathcal{L}^{-1}\left\{\frac{2s+1}{(s-1)^3}\right\}$

**Partial fractions:**
$$\frac{2s+1}{(s-1)^3} = \frac{A}{s-1} + \frac{B}{(s-1)^2} + \frac{C}{(s-1)^3}$$

Multiply by $(s-1)^3$:
$$2s+1 = A(s-1)^2 + B(s-1) + C$$

Set $s = 1$: $C = 3$

Expand and compare:
$$2s+1 = As^2 - 2As + A + Bs - B + 3$$
$$= As^2 + (-2A+B)s + (A-B+3)$$

Comparing: $A = 0$, $-2A + B = 2 \Rightarrow B = 2$

**Inverse:**
$$f(t) = \mathcal{L}^{-1}\left\{\frac{2}{(s-1)^2} + \frac{3}{(s-1)^3}\right\} = 2te^t + \frac{3t^2}{2}e^t$$

---

### 6. Case 3: Irreducible Quadratic Factors

$$\frac{P(s)}{s^2 + bs + c} = \frac{As + B}{s^2 + bs + c}$$

**Strategy:** Complete the square to match $\sin$ and $\cos$ transforms.

---

### Example 3: Completing the Square
Find $\mathcal{L}^{-1}\left\{\frac{s+5}{s^2+4s+13}\right\}$

**Complete the square:**
$$s^2 + 4s + 13 = (s+2)^2 + 9 = (s+2)^2 + 3^2$$

**Rewrite numerator:**
$$\frac{s+5}{(s+2)^2+9} = \frac{(s+2)+3}{(s+2)^2+9} = \frac{s+2}{(s+2)^2+9} + \frac{3}{(s+2)^2+9}$$

**Inverse using shifting:**
$$f(t) = e^{-2t}\cos(3t) + e^{-2t}\sin(3t) = e^{-2t}[\cos(3t) + \sin(3t)]$$

---

### 7. Case 4: Complex Roots (Alternative)

If $F(s)$ has complex roots $\alpha \pm i\beta$:
$$\frac{As + B}{(s-\alpha)^2 + \beta^2}$$

Match to:
- $\mathcal{L}^{-1}\left\{\frac{s-\alpha}{(s-\alpha)^2+\beta^2}\right\} = e^{\alpha t}\cos(\beta t)$
- $\mathcal{L}^{-1}\left\{\frac{\beta}{(s-\alpha)^2+\beta^2}\right\} = e^{\alpha t}\sin(\beta t)$

---

## 📋 Summary: Partial Fraction Strategy

| Denominator Factor | Partial Fraction Form |
|-------------------|----------------------|
| $(s-a)$ | $\frac{A}{s-a}$ |
| $(s-a)^n$ | $\frac{A_1}{s-a} + \frac{A_2}{(s-a)^2} + \cdots + \frac{A_n}{(s-a)^n}$ |
| $(s^2+bs+c)$ irreducible | $\frac{As+B}{s^2+bs+c}$ |
| $(s^2+bs+c)^n$ | Sum of $\frac{A_is+B_i}{(s^2+bs+c)^i}$ for $i=1$ to $n$ |

---

## 📝 Practice Problems

### Level 1: Simple Inversions
1. $\mathcal{L}^{-1}\left\{\frac{3}{s-4}\right\}$
2. $\mathcal{L}^{-1}\left\{\frac{5}{s^2}\right\}$
3. $\mathcal{L}^{-1}\left\{\frac{2}{s^2+9}\right\}$
4. $\mathcal{L}^{-1}\left\{\frac{s}{s^2+4}\right\}$

### Level 2: Partial Fractions (Distinct Roots)
5. $\mathcal{L}^{-1}\left\{\frac{1}{s^2-4}\right\}$
6. $\mathcal{L}^{-1}\left\{\frac{2s-1}{s^2-s-2}\right\}$
7. $\mathcal{L}^{-1}\left\{\frac{s}{(s+1)(s+2)(s+3)}\right\}$

### Level 3: Repeated Roots
8. $\mathcal{L}^{-1}\left\{\frac{1}{(s+2)^3}\right\}$
9. $\mathcal{L}^{-1}\left\{\frac{s+1}{(s-1)^2}\right\}$
10. $\mathcal{L}^{-1}\left\{\frac{3s+2}{s^2(s-1)}\right\}$

### Level 4: Completing the Square
11. $\mathcal{L}^{-1}\left\{\frac{1}{s^2+2s+5}\right\}$
12. $\mathcal{L}^{-1}\left\{\frac{s}{s^2+6s+13}\right\}$
13. $\mathcal{L}^{-1}\left\{\frac{s+3}{s^2-4s+8}\right\}$

### Level 5: Mixed
14. $\mathcal{L}^{-1}\left\{\frac{s^2+2s+3}{(s-1)(s^2+4)}\right\}$
15. $\mathcal{L}^{-1}\left\{\frac{s+2}{(s+1)^2(s+3)}\right\}$

---

## 📊 Answers

1. $3e^{4t}$
2. $5t$
3. $\frac{2}{3}\sin(3t)$
4. $\cos(2t)$
5. $\frac{1}{4}(e^{2t} - e^{-2t}) = \frac{1}{2}\sinh(2t)$
6. $e^{2t} + e^{-t}$
7. $\frac{1}{2}e^{-t} - 2e^{-2t} + \frac{3}{2}e^{-3t}$
8. $\frac{t^2}{2}e^{-2t}$
9. $e^t + 2te^t$
10. $-2 - 5t + 2e^t$
11. $\frac{1}{2}e^{-t}\sin(2t)$
12. $e^{-3t}\cos(2t) - \frac{3}{2}e^{-3t}\sin(2t)$
13. $e^{2t}\cos(2t) + \frac{5}{2}e^{2t}\sin(2t)$
14. $\frac{6}{5}e^t + \frac{-1}{5}\cos(2t) + \frac{3}{10}\sin(2t)$
15. Partial fractions, then invert

---

## 🔬 Quantum Mechanics Connection

### Spectral Decomposition

When finding inverse transforms, we decompose into simple poles—this mirrors **spectral decomposition** in QM:

$$G(E) = \sum_n \frac{|n\rangle\langle n|}{E - E_n}$$

Each pole corresponds to an energy eigenstate!

---

## ✅ Daily Checklist

- [ ] Master partial fraction decomposition
- [ ] Practice completing the square
- [ ] Handle all cases (distinct, repeated, complex roots)
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 80: Solving ODEs with Laplace Transforms**
- The complete method
- IVPs become algebraic equations
- Applications

---

*"Inverse transforms reveal the time-domain story hidden in the frequency-domain algebra."*
