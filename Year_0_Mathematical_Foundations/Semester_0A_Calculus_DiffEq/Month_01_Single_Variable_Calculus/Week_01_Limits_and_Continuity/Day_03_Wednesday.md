# Day 3: Limit Laws and Evaluation Techniques

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Limit Laws Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Squeeze Theorem |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and apply all fundamental limit laws
2. Evaluate limits of polynomials and rational functions
3. Apply the Squeeze Theorem to find limits
4. Handle indeterminate forms (0/0, ∞/∞)
5. Evaluate one-sided limits systematically

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 2.3**: Calculating Limits Using the Limit Laws (pp. 97-109)

### Focus Areas
- Theorem statements with their conditions
- When each law applies (and when it fails)
- Connection to ε-δ proofs from yesterday

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture 2: Limits** - Focus on limit computation techniques

### Professor Leonard
**Calculus 1: Limits (Finding Limits Algebraically)**
- Comprehensive worked examples
- Excellent for building computational fluency

### Organic Chemistry Tutor
**Evaluating Limits** - Many practice problems with solutions

---

## 📖 Core Content: The Limit Laws

### 1. Fundamental Limit Laws

Let $\lim_{x \to c} f(x) = L$ and $\lim_{x \to c} g(x) = M$, where L and M are real numbers.

#### **Law 1: Constant Law**
$$\lim_{x \to c} k = k$$
where k is any constant.

#### **Law 2: Identity Law**
$$\lim_{x \to c} x = c$$

#### **Law 3: Sum/Difference Law**
$$\lim_{x \to c} [f(x) \pm g(x)] = L \pm M$$

#### **Law 4: Constant Multiple Law**
$$\lim_{x \to c} [k \cdot f(x)] = k \cdot L$$

#### **Law 5: Product Law**
$$\lim_{x \to c} [f(x) \cdot g(x)] = L \cdot M$$

#### **Law 6: Quotient Law**
$$\lim_{x \to c} \frac{f(x)}{g(x)} = \frac{L}{M}, \quad \text{provided } M \neq 0$$

#### **Law 7: Power Law**
$$\lim_{x \to c} [f(x)]^n = L^n$$
for any positive integer n.

#### **Law 8: Root Law**
$$\lim_{x \to c} \sqrt[n]{f(x)} = \sqrt[n]{L}$$
(For even n, require L ≥ 0 and f(x) ≥ 0 near c)

---

### 2. Direct Substitution Property

**Theorem:** If f is a polynomial or rational function and c is in the domain of f, then:
$$\lim_{x \to c} f(x) = f(c)$$

**Proof Sketch:** Polynomials are built from constants and x using addition and multiplication. By the limit laws, we can evaluate the limit by substitution.

**Example:**
$$\lim_{x \to 2} (x^3 - 4x + 1) = 2^3 - 4(2) + 1 = 8 - 8 + 1 = 1$$

---

### 3. Indeterminate Forms and Algebraic Techniques

When direct substitution gives an indeterminate form (0/0, ∞/∞, etc.), we need algebraic manipulation.

#### **Technique 1: Factoring**

**Example:** $\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$

Direct substitution: $\frac{0}{0}$ (indeterminate)

Factor:
$$\frac{x^2 - 4}{x - 2} = \frac{(x-2)(x+2)}{x-2} = x + 2 \quad (x \neq 2)$$

Therefore:
$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = \lim_{x \to 2} (x + 2) = 4$$

#### **Technique 2: Rationalizing**

**Example:** $\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x}$

Direct substitution: $\frac{0}{0}$ (indeterminate)

Rationalize the numerator:
$$\frac{\sqrt{x+4} - 2}{x} \cdot \frac{\sqrt{x+4} + 2}{\sqrt{x+4} + 2} = \frac{(x+4) - 4}{x(\sqrt{x+4} + 2)} = \frac{x}{x(\sqrt{x+4} + 2)}$$

Simplify:
$$= \frac{1}{\sqrt{x+4} + 2}$$

Therefore:
$$\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x} = \frac{1}{\sqrt{4} + 2} = \frac{1}{4}$$

#### **Technique 3: Common Denominators**

**Example:** $\lim_{x \to 0} \frac{\frac{1}{x+1} - 1}{x}$

Combine fractions in numerator:
$$\frac{1}{x+1} - 1 = \frac{1 - (x+1)}{x+1} = \frac{-x}{x+1}$$

So:
$$\frac{\frac{1}{x+1} - 1}{x} = \frac{-x}{x(x+1)} = \frac{-1}{x+1}$$

Therefore:
$$\lim_{x \to 0} \frac{\frac{1}{x+1} - 1}{x} = \frac{-1}{1} = -1$$

---

### 4. The Squeeze Theorem (Sandwich Theorem)

> **Theorem (Squeeze/Sandwich):** If $g(x) \leq f(x) \leq h(x)$ for all x near c (except possibly at c), and
> $$\lim_{x \to c} g(x) = \lim_{x \to c} h(x) = L$$
> then
> $$\lim_{x \to c} f(x) = L$$

**Geometric Intuition:** If f is "squeezed" between g and h, and both g and h approach L, then f must also approach L.

#### **Classic Application:**

**Prove:** $\lim_{x \to 0} x^2 \sin\left(\frac{1}{x}\right) = 0$

**Solution:**
We know $-1 \leq \sin\left(\frac{1}{x}\right) \leq 1$ for all x ≠ 0.

Multiply by x² (which is positive for x ≠ 0):
$$-x^2 \leq x^2 \sin\left(\frac{1}{x}\right) \leq x^2$$

Now:
- $\lim_{x \to 0} (-x^2) = 0$
- $\lim_{x \to 0} x^2 = 0$

By the Squeeze Theorem:
$$\lim_{x \to 0} x^2 \sin\left(\frac{1}{x}\right) = 0$$

---

### 5. Special Limits to Memorize

#### **The Fundamental Trigonometric Limit**
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

*Proof requires geometric argument (area of sectors) - see Stewart.*

#### **Related Limit**
$$\lim_{x \to 0} \frac{1 - \cos x}{x} = 0$$

#### **Exponential Limit**
$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

#### **Logarithmic Limit**
$$\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1$$

---

### 6. Limits at Infinity

**Definition:** $\lim_{x \to \infty} f(x) = L$ means f(x) approaches L as x grows without bound.

#### **Key Results:**

For any positive integer n:
$$\lim_{x \to \infty} \frac{1}{x^n} = 0$$

**Technique for Rational Functions:** Divide by highest power of x.

**Example:** $\lim_{x \to \infty} \frac{3x^2 - 2x + 1}{5x^2 + 4}$

Divide numerator and denominator by x²:
$$= \lim_{x \to \infty} \frac{3 - \frac{2}{x} + \frac{1}{x^2}}{5 + \frac{4}{x^2}} = \frac{3 - 0 + 0}{5 + 0} = \frac{3}{5}$$

**Rule of Thumb for Rational Functions:**
- Same degree: ratio of leading coefficients
- Higher degree in numerator: ±∞
- Higher degree in denominator: 0

---

## ✏️ Worked Examples

### Example 1: Polynomial
$$\lim_{x \to -1} (2x^4 - 3x^2 + x - 7)$$

By direct substitution:
$$= 2(-1)^4 - 3(-1)^2 + (-1) - 7 = 2 - 3 - 1 - 7 = -9$$

### Example 2: Rational with Indeterminate Form
$$\lim_{x \to 3} \frac{x^2 - 5x + 6}{x^2 - 9}$$

Direct substitution: 0/0

Factor:
$$= \lim_{x \to 3} \frac{(x-2)(x-3)}{(x-3)(x+3)} = \lim_{x \to 3} \frac{x-2}{x+3} = \frac{1}{6}$$

### Example 3: Rationalizing
$$\lim_{x \to 9} \frac{x - 9}{\sqrt{x} - 3}$$

Rationalize denominator:
$$= \lim_{x \to 9} \frac{(x-9)(\sqrt{x}+3)}{(\sqrt{x}-3)(\sqrt{x}+3)} = \lim_{x \to 9} \frac{(x-9)(\sqrt{x}+3)}{x-9}$$
$$= \lim_{x \to 9} (\sqrt{x} + 3) = 3 + 3 = 6$$

### Example 4: Squeeze Theorem
$$\lim_{x \to 0} x^4 \cos\left(\frac{2}{x}\right)$$

Since $-1 \leq \cos\left(\frac{2}{x}\right) \leq 1$:
$$-x^4 \leq x^4 \cos\left(\frac{2}{x}\right) \leq x^4$$

Both bounds → 0 as x → 0.

By Squeeze Theorem: limit = 0.

### Example 5: Limit at Infinity
$$\lim_{x \to \infty} \frac{2x^3 - x}{4x^3 + 5x^2 - 1}$$

Divide by x³:
$$= \lim_{x \to \infty} \frac{2 - \frac{1}{x^2}}{4 + \frac{5}{x} - \frac{1}{x^3}} = \frac{2}{4} = \frac{1}{2}$$

---

## 📝 Practice Problems

### Level 1: Direct Substitution
1. $\lim_{x \to 2} (x^3 + 2x - 1)$
2. $\lim_{x \to -3} \frac{x^2 + 1}{x - 1}$
3. $\lim_{x \to 4} \sqrt{3x + 4}$

### Level 2: Factoring
4. $\lim_{x \to 1} \frac{x^2 - 1}{x - 1}$
5. $\lim_{x \to -2} \frac{x^3 + 8}{x + 2}$
6. $\lim_{x \to 5} \frac{x^2 - 25}{x^2 - 4x - 5}$

### Level 3: Rationalizing
7. $\lim_{x \to 0} \frac{\sqrt{x+1} - 1}{x}$
8. $\lim_{x \to 4} \frac{x - 4}{\sqrt{x} - 2}$
9. $\lim_{x \to 0} \frac{\sqrt{2+x} - \sqrt{2-x}}{x}$

### Level 4: Squeeze Theorem
10. $\lim_{x \to 0} x \sin\left(\frac{1}{x}\right)$
11. $\lim_{x \to 0^+} \sqrt{x} \sin\left(\frac{1}{x}\right)$
12. $\lim_{x \to \infty} \frac{\sin x}{x}$

### Level 5: Limits at Infinity
13. $\lim_{x \to \infty} \frac{5x^2 - 3x + 2}{2x^2 + 1}$
14. $\lim_{x \to -\infty} \frac{x^3 + 1}{x^2 + 1}$
15. $\lim_{x \to \infty} \frac{\sqrt{4x^2 + 1}}{x + 1}$

---

## 📊 Answers

1. 11
2. 5/2
3. 4
4. 2
5. 12
6. 5/3
7. 1/2
8. 4
9. 1/√2
10. 0
11. 0
12. 0
13. 5/2
14. -∞
15. 2

---

## 🔬 Physics Application: Instantaneous Velocity

The concept of limit directly gives us instantaneous velocity:

$$v(t) = \lim_{\Delta t \to 0} \frac{s(t + \Delta t) - s(t)}{\Delta t}$$

This is the birth of the derivative, which we'll study soon!

---

## ✅ Daily Checklist

- [ ] Read Stewart 2.3
- [ ] Memorize all 8 limit laws
- [ ] Master the three algebraic techniques
- [ ] Understand and apply Squeeze Theorem
- [ ] Complete Level 1-3 problems
- [ ] Attempt Level 4-5 problems
- [ ] Memorize special limits (sin x/x, etc.)
- [ ] Create summary sheet of techniques

---

## 📓 Reflection Questions

1. When does the Quotient Law fail? Give an example.
2. Why is factoring the first technique to try for 0/0?
3. How do you decide whether to rationalize numerator or denominator?
4. Create your own Squeeze Theorem problem and solve it.

---

## 🔜 Preview: Tomorrow

**Day 4: Continuity**

- Definition of continuity at a point
- Types of discontinuities
- Intermediate Value Theorem
- Continuous functions on closed intervals

---

*"The art of doing mathematics consists in finding that special case which contains all the germs of generality."*
— David Hilbert
