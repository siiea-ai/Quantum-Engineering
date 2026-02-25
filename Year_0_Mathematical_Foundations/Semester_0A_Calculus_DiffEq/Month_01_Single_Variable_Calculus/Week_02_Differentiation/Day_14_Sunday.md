# Day 14: Rest, Review, and Week 3 Preparation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 11:30 AM | 1.5 hours | Concept Review |
| Afternoon | 2:00 PM - 3:00 PM | 1 hour | Flashcard Review |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Week 3 Preview |

**Total Study Time: 3.5 hours (REST DAY)**

---

## 🧘 Rest Day Importance

After an intensive week learning differentiation, your brain needs time to:
- Consolidate new neural pathways
- Process the relationships between concepts
- Recover from cognitive load

**Honor this rest day—it's part of the learning process.**

---

## 📝 Week 2 Summary Sheet

### The Derivative Definition
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Basic Differentiation Rules

| Rule | Formula |
|------|---------|
| Constant | d/dx[c] = 0 |
| Power | d/dx[xⁿ] = nxⁿ⁻¹ |
| Constant Multiple | d/dx[cf] = cf' |
| Sum | d/dx[f ± g] = f' ± g' |
| Product | d/dx[fg] = f'g + fg' |
| Quotient | d/dx[f/g] = (f'g - fg')/g² |
| Chain | d/dx[f(g(x))] = f'(g(x))·g'(x) |

### Special Derivatives

| Function | Derivative |
|----------|------------|
| eˣ | eˣ |
| aˣ | aˣ ln(a) |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |
| tan(x) | sec²(x) |

### Implicit Differentiation Process
1. Differentiate both sides with respect to x
2. Apply chain rule to y terms (multiply by dy/dx)
3. Collect dy/dx terms
4. Solve for dy/dx

---

## 🔄 Self-Assessment Quiz

Answer without notes, then check:

**Q1:** What is the derivative of f(x) = x³·eˣ?

<details>
<summary>Answer</summary>
f'(x) = 3x²·eˣ + x³·eˣ = eˣ(x³ + 3x²) = x²eˣ(x + 3)
</details>

**Q2:** Find dy/dx for x² + y² = 9.

<details>
<summary>Answer</summary>
dy/dx = -x/y
</details>

**Q3:** What is d/dx[sin(x²)]?

<details>
<summary>Answer</summary>
d/dx[sin(x²)] = cos(x²)·2x = 2x cos(x²)
</details>

**Q4:** Find the equation of the tangent line to y = x² at x = 3.

<details>
<summary>Answer</summary>
f(3) = 9, f'(3) = 6
Tangent: y - 9 = 6(x - 3), or y = 6x - 9
</details>

**Q5:** If f(x) = ln(x³ + 1), what is f'(x)?

<details>
<summary>Answer</summary>
f'(x) = 3x²/(x³ + 1)
</details>

---

## 📊 Concept Connections Map

```
                    LIMITS
                      │
                      ▼
              ┌───────────────┐
              │   DERIVATIVE  │
              │   Definition  │
              └───────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ Power   │ │ Product │ │ Chain   │
     │ Rule    │ │ Rule    │ │ Rule    │
     └─────────┘ └─────────┘ └─────────┘
          │           │           │
          └───────────┼───────────┘
                      ▼
              ┌───────────────┐
              │   IMPLICIT    │
              │ Differentiation│
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ APPLICATIONS  │
              │ (Week 3)      │
              └───────────────┘
```

---

## 📚 Flashcard Review

Review these key cards:

**Card 1:**
- Front: Power Rule
- Back: d/dx[xⁿ] = nxⁿ⁻¹

**Card 2:**
- Front: Product Rule
- Back: (fg)' = f'g + fg'

**Card 3:**
- Front: Quotient Rule
- Back: (f/g)' = (f'g - fg')/g²

**Card 4:**
- Front: Chain Rule
- Back: d/dx[f(g(x))] = f'(g(x))·g'(x)

**Card 5:**
- Front: d/dx[eˣ]
- Back: eˣ (only function equal to its derivative!)

**Card 6:**
- Front: d/dx[ln(x)]
- Back: 1/x

**Card 7:**
- Front: Implicit differentiation key step
- Back: When differentiating y, multiply by dy/dx

---

## 🔜 Week 3 Preview: Applications of Derivatives

### Topics Coming Up

**Day 15:** Related Rates
- How quantities change together
- Setting up and solving word problems

**Day 16:** Linear Approximation and Differentials
- Using tangent lines to approximate
- The differential dy = f'(x)dx

**Day 17:** Maximum and Minimum Values
- Critical points
- First and second derivative tests

**Day 18:** Optimization Problems
- Setting up objective functions
- Finding optimal solutions

**Day 19:** Week 3 Problem Set

**Day 20:** L'Hôpital's Rule and Newton's Method Lab

**Day 21:** Rest and Review

### Preview Reading (Optional)

Skim Stewart Sections 3.9-3.10:
- Just read the examples
- Don't worry about details yet

### Key Concepts to Prepare For

1. **Related Rates:** Multiple quantities changing with time
   - dx/dt, dy/dt connected through an equation
   - Chain rule is essential!

2. **Extrema:** Finding peaks and valleys
   - Where f'(x) = 0 or undefined
   - Local vs. global extrema

3. **Optimization:** Real-world max/min problems
   - "Find the dimensions that maximize volume"
   - "Minimize the cost given constraints"

---

## 📓 Reflection Questions

Take time to write thoughtful answers:

1. **What was the most challenging concept this week?**

2. **Which differentiation rule do you need more practice with?**

3. **How does the derivative connect geometry (tangent lines) to rates of change?**

4. **Give a real-world example where you'd want to find a derivative.**

5. **What questions do you still have about differentiation?**

---

## 🎯 Week 2 Competency Checklist

Before moving to Week 3, ensure you can:

- [ ] Compute derivatives from the limit definition
- [ ] Apply the power rule fluently
- [ ] Use product and quotient rules correctly
- [ ] Apply chain rule to nested functions
- [ ] Perform implicit differentiation
- [ ] Find tangent line equations
- [ ] Compute higher derivatives
- [ ] Use SymPy for symbolic differentiation
- [ ] Implement numerical differentiation in Python

**If any box is unchecked, spend extra time reviewing that topic.**

---

## 📈 Progress Check

Rate your confidence (1-5):

| Topic | Confidence |
|-------|------------|
| Derivative definition | /5 |
| Power rule | /5 |
| Product rule | /5 |
| Quotient rule | /5 |
| Chain rule | /5 |
| Implicit differentiation | /5 |
| Tangent lines | /5 |
| Numerical methods | /5 |

**Average score ≥ 4:** Ready for Week 3
**Average score 3-4:** Quick review recommended
**Average score < 3:** Extended review needed

---

## 💤 Rest Well

Tonight:
- Get 7-8 hours of sleep
- Avoid heavy studying
- Let your brain consolidate

Tomorrow begins **Applications of Derivatives**—the payoff for all your hard work!

---

## 📌 Week 2 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════╗
║                    DIFFERENTIATION RULES                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Power:    (xⁿ)' = nxⁿ⁻¹                                      ║
║  Product:  (fg)' = f'g + fg'                                   ║
║  Quotient: (f/g)' = (f'g - fg')/g²                            ║
║  Chain:    (f∘g)' = f'(g)·g'                                  ║
╠═══════════════════════════════════════════════════════════════╣
║  SPECIAL DERIVATIVES                                           ║
║  (eˣ)' = eˣ       (aˣ)' = aˣ ln a     (ln x)' = 1/x          ║
║  (sin x)' = cos x     (cos x)' = -sin x                       ║
╠═══════════════════════════════════════════════════════════════╣
║  IMPLICIT: Differentiate both sides, dy/dx on y terms         ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Week 2 Complete! 🎉**

You've mastered the fundamental techniques of differentiation. Week 3 shows you how to apply these tools to solve real problems.

*"The mathematician does not study pure mathematics because it is useful; he studies it because he delights in it and he delights in it because it is beautiful."*
— Henri Poincaré
