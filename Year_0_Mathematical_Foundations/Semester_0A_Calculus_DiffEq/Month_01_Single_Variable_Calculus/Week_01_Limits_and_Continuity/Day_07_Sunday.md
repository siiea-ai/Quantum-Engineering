# Day 7: Rest, Review, and Preparation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 11:30 AM | 1.5 hours | Light Review |
| Afternoon | 2:00 PM - 3:00 PM | 1 hour | Spaced Repetition |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Week 2 Preview |

**Total Study Time: 3.5 hours (REST DAY)**

---

## 🧘 Philosophy of Rest Days

Rest is not optional—it's essential for learning. During sleep and rest:
- Memory consolidation occurs
- Neural pathways strengthen
- Mental fatigue recovers
- Creativity and insight emerge

**Do not skip rest days.** The temptation to "push through" leads to burnout and reduced retention.

---

## 📝 Morning: Light Review (1.5 hours)

### Activity 1: Summary Sheet Creation (45 min)

Create a single-page summary of Week 1 containing:

#### Key Definitions
```
LIMIT: lim(x→c) f(x) = L means...
  ∀ε > 0, ∃δ > 0 such that 0 < |x-c| < δ ⟹ |f(x) - L| < ε

CONTINUITY at c requires:
  1. f(c) is defined
  2. lim(x→c) f(x) exists
  3. lim(x→c) f(x) = f(c)
```

#### Limit Laws (Quick Reference)
| Law | Formula |
|-----|---------|
| Sum | lim(f + g) = lim f + lim g |
| Product | lim(fg) = (lim f)(lim g) |
| Quotient | lim(f/g) = (lim f)/(lim g), if lim g ≠ 0 |
| Power | lim(f^n) = (lim f)^n |

#### Special Limits
- lim(x→0) sin(x)/x = 1
- lim(x→0) (1-cos(x))/x = 0
- lim(x→∞) (1 + 1/n)^n = e

#### Discontinuity Types
1. **Removable:** Limit exists, but f(c) ≠ limit or undefined
2. **Jump:** Left and right limits exist but differ
3. **Infinite:** One or both one-sided limits are ±∞
4. **Oscillating:** No limiting behavior

#### IVT Statement
If f is continuous on [a,b] and N is between f(a) and f(b), 
then ∃c ∈ (a,b) such that f(c) = N.

### Activity 2: Self-Assessment (45 min)

Answer these questions without notes:

1. Write the ε-δ definition of a limit.
2. What are the three conditions for continuity?
3. State the Intermediate Value Theorem.
4. How do you handle the indeterminate form 0/0?
5. What is the Squeeze Theorem?

**Check your answers against your notes. Note any gaps.**

---

## 🔄 Afternoon: Spaced Repetition (1 hour)

### Anki Flashcard Recommendations

Create or review flashcards for:

**Card 1 (Front/Back):**
- Front: What is the ε-δ definition of lim(x→c) f(x) = L?
- Back: ∀ε > 0, ∃δ > 0 such that 0 < |x-c| < δ ⟹ |f(x) - L| < ε

**Card 2:**
- Front: When does lim(x→c) f(x) NOT exist?
- Back: Jump (one-sided limits differ), Infinite (limits ±∞), Oscillation (no limit)

**Card 3:**
- Front: lim(x→0) sin(x)/x = ?
- Back: 1

**Card 4:**
- Front: Three conditions for f continuous at c?
- Back: f(c) defined, lim exists, lim = f(c)

**Card 5:**
- Front: For f(x) = mx + b, if lim(x→c) f(x) = L, what is δ in terms of ε?
- Back: δ = ε/|m|

**Card 6:**
- Front: What is a removable discontinuity?
- Back: The limit exists but f(c) is undefined or doesn't equal the limit

**Card 7:**
- Front: IVT: If f continuous on [a,b] and f(a) < N < f(b), then...
- Back: ∃c ∈ (a,b) with f(c) = N

**Card 8:**
- Front: Squeeze Theorem statement
- Back: If g(x) ≤ f(x) ≤ h(x) near c and lim g = lim h = L, then lim f = L

### Spaced Repetition Schedule

Follow this pattern for maximum retention:
- Review new cards: Today, Tomorrow, Day 4, Day 7, Day 14, Day 30
- Difficult cards: More frequent review
- Easy cards: Less frequent review

---

## 📚 Evening: Week 2 Preview (1 hour)

### What's Coming: Differentiation

Week 2 introduces the **derivative**, which is defined as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Everything you learned this week directly applies!

### Preview Reading (Optional)

Skim Stewart Section 2.7 (Derivatives and Rates of Change):
- Just read the introduction and definitions
- Don't worry about computation techniques yet
- Notice how limits appear everywhere

### Conceptual Preview

The derivative answers: **"How fast is f changing at point x?"**

- Geometrically: Slope of tangent line
- Physically: Instantaneous velocity, acceleration
- In quantum mechanics: Time evolution of wave functions

### Key Connection

The derivative IS a limit:
- If you understand limits well, derivatives will make sense
- Every derivative rule comes from limit laws
- The chain rule comes from properties of limits of compositions

---

## 📊 Week 1 Progress Tracker

### Complete This Self-Assessment

Rate your confidence (1-5) on each topic:

| Topic | Confidence | Notes |
|-------|------------|-------|
| Intuitive limits | /5 | |
| ε-δ definition | /5 | |
| Limit laws | /5 | |
| Factoring 0/0 | /5 | |
| Rationalizing | /5 | |
| Squeeze theorem | /5 | |
| One-sided limits | /5 | |
| Continuity definition | /5 | |
| Types of discontinuities | /5 | |
| IVT statement | /5 | |
| IVT applications | /5 | |
| Python/NumPy basics | /5 | |

### Interpretation
- **40-60:** Excellent foundation, ready to proceed
- **30-39:** Good understanding, review weak areas
- **Below 30:** Consider extra review before Week 2

---

## 🧠 Reflection Prompts

Write a paragraph for each:

1. **What concept was most challenging this week? How did you overcome it?**

2. **What surprised you about limits and continuity?**

3. **How do you see limits connecting to physics or other sciences?**

4. **What study strategies worked best for you?**

5. **What would you do differently next week?**

---

## 🎯 Week 1 Accomplishments

Congratulations! You have learned:

✅ The formal definition of a limit (ε-δ)
✅ How to evaluate limits algebraically
✅ The Squeeze Theorem
✅ What continuity means precisely
✅ Four types of discontinuities
✅ The Intermediate Value Theorem
✅ Basic Python for mathematical visualization

This is serious mathematical content—be proud of your progress!

---

## 🗓️ Week 2 Schedule Preview

| Day | Topic |
|-----|-------|
| Monday | Definition of Derivative, Rates of Change |
| Tuesday | Differentiation Rules (Power, Sum, Product) |
| Wednesday | Chain Rule |
| Thursday | Implicit Differentiation |
| Friday | Problem Set |
| Saturday | Applications Lab |
| Sunday | Review |

---

## 📖 Recommended Light Reading

If you want something mathematically interesting but less intense:

- **"What Is Mathematics?"** by Courant & Robbins — Chapter 8 on Calculus
- **3Blue1Brown's "Essence of Calculus"** — Watch episodes 2-4 on derivatives
- **Popular math articles** about the history of calculus (Newton vs. Leibniz)

---

## 💤 Rest Well

Tonight:
- Get 7-8 hours of sleep
- Avoid screens 1 hour before bed
- Let your brain consolidate the week's learning

Tomorrow, you begin the exciting journey into differentiation!

---

*"In mathematics, the art of proposing a question must be held of higher value than solving it."*
— Georg Cantor

---

## 📌 Quick Reference Card

Print this and keep at your desk:

```
╔══════════════════════════════════════════════════════════════╗
║                    LIMITS & CONTINUITY                        ║
╠══════════════════════════════════════════════════════════════╣
║  LIMIT: lim(x→c) f(x) = L                                    ║
║  ∀ε>0, ∃δ>0: 0<|x-c|<δ ⟹ |f(x)-L|<ε                        ║
╠══════════════════════════════════════════════════════════════╣
║  CONTINUITY at c:                                             ║
║  1. f(c) defined  2. lim exists  3. lim = f(c)               ║
╠══════════════════════════════════════════════════════════════╣
║  SPECIAL LIMITS:                                              ║
║  sin(x)/x → 1    (1-cos(x))/x → 0    (eˣ-1)/x → 1           ║
╠══════════════════════════════════════════════════════════════╣
║  IVT: f continuous on [a,b], f(a)<N<f(b) ⟹ ∃c: f(c)=N       ║
╚══════════════════════════════════════════════════════════════╝
```

---

**Week 1 Complete! 🎉**

You've taken your first steps toward quantum mechanics. The mathematical maturity you're building now will pay dividends throughout the program.

See you in Week 2!
