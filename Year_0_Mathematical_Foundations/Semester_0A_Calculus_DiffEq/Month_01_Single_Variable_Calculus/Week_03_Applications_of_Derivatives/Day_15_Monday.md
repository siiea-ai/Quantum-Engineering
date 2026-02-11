# Day 15: Related Rates

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Related Rates Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand the concept of related rates
2. Set up related rates problems systematically
3. Use implicit differentiation with respect to time
4. Solve classic related rates problems
5. Interpret results in physical contexts

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 3.9**: Related Rates (pp. 245-253)

### Key Focus
- Identifying the relationship between variables
- Differentiating with respect to time
- Careful problem setup

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture on Related Rates**

### Professor Leonard
**Related Rates in Calculus**
- Excellent systematic approach

### Organic Chemistry Tutor
**Related Rates Problems**

---

## 📖 Core Content: Related Rates

### 1. The Big Idea

**Related rates** problems involve finding how fast one quantity is changing given information about how fast another related quantity is changing.

**Key insight:** When quantities are related by an equation, their rates of change are also related (through the chain rule).

### 2. The Setup

If x and y are both functions of time t, and they're related by some equation, then we can differentiate that equation with respect to t to find how dx/dt and dy/dt are related.

**Example:** If x² + y² = 25, then:
$$\frac{d}{dt}[x^2 + y^2] = \frac{d}{dt}[25]$$
$$2x\frac{dx}{dt} + 2y\frac{dy}{dt} = 0$$

### 3. Problem-Solving Strategy

> **Step 1: Draw a diagram** 
> - Label all quantities with variables
> - Identify what's constant and what changes

> **Step 2: Write the relationship**
> - Find an equation relating the variables
> - Don't substitute numbers yet!

> **Step 3: Differentiate with respect to t**
> - Use implicit differentiation
> - Every variable gets a d/dt term

> **Step 4: Substitute known values**
> - Plug in specific values AFTER differentiating
> - Include rates (dx/dt, etc.) and positions (x, y, etc.)

> **Step 5: Solve for the unknown rate**
> - Algebraically isolate the desired rate

---

## ✏️ Worked Examples

### Example 1: Expanding Circle

**Problem:** A stone is dropped into a pond, creating circular ripples. The radius expands at 2 ft/s. How fast is the area increasing when the radius is 5 ft?

**Solution:**

**Step 1:** Variables
- r = radius (changing)
- A = area (changing)
- Given: dr/dt = 2 ft/s
- Find: dA/dt when r = 5 ft

**Step 2:** Relationship
$$A = \pi r^2$$

**Step 3:** Differentiate with respect to t
$$\frac{dA}{dt} = 2\pi r \frac{dr}{dt}$$

**Step 4:** Substitute
$$\frac{dA}{dt} = 2\pi (5)(2) = 20\pi$$

**Answer:** The area is increasing at **20π ft²/s** (≈ 62.8 ft²/s).

---

### Example 2: Sliding Ladder

**Problem:** A 10-foot ladder rests against a wall. The bottom slides away at 1 ft/s. How fast is the top sliding down when the bottom is 6 ft from the wall?

**Solution:**

**Step 1:** Set up coordinates
- x = distance from wall to bottom
- y = height of top on wall
- Given: dx/dt = 1 ft/s
- Find: dy/dt when x = 6 ft

**Step 2:** Relationship (Pythagorean theorem)
$$x^2 + y^2 = 100$$

**Step 3:** Differentiate
$$2x\frac{dx}{dt} + 2y\frac{dy}{dt} = 0$$

**Step 4:** Find y when x = 6
$$36 + y^2 = 100 \implies y = 8$$

Substitute:
$$2(6)(1) + 2(8)\frac{dy}{dt} = 0$$
$$12 + 16\frac{dy}{dt} = 0$$
$$\frac{dy}{dt} = -\frac{12}{16} = -\frac{3}{4}$$

**Answer:** The top is sliding down at **3/4 ft/s** (negative indicates downward).

---

### Example 3: Filling a Cone

**Problem:** Water is poured into a conical tank at 2 m³/min. The tank has height 6m and radius 3m at the top. How fast is the water level rising when the water is 4m deep?

**Solution:**

**Step 1:** Variables
- h = height of water
- r = radius of water surface
- V = volume of water
- Given: dV/dt = 2 m³/min
- Find: dh/dt when h = 4m

**Step 2:** Relationships
Volume of cone: $V = \frac{1}{3}\pi r^2 h$

By similar triangles (cone has r/h = 3/6 = 1/2):
$$\frac{r}{h} = \frac{3}{6} = \frac{1}{2} \implies r = \frac{h}{2}$$

Substitute to eliminate r:
$$V = \frac{1}{3}\pi \left(\frac{h}{2}\right)^2 h = \frac{\pi h^3}{12}$$

**Step 3:** Differentiate
$$\frac{dV}{dt} = \frac{\pi}{12} \cdot 3h^2 \frac{dh}{dt} = \frac{\pi h^2}{4}\frac{dh}{dt}$$

**Step 4:** Substitute h = 4, dV/dt = 2
$$2 = \frac{\pi (16)}{4}\frac{dh}{dt} = 4\pi\frac{dh}{dt}$$
$$\frac{dh}{dt} = \frac{2}{4\pi} = \frac{1}{2\pi}$$

**Answer:** The water level is rising at **1/(2π) m/min** ≈ 0.159 m/min.

---

### Example 4: Moving Shadow

**Problem:** A 6-foot person walks away from a 15-foot lamppost at 4 ft/s. How fast is the tip of the person's shadow moving?

**Solution:**

**Step 1:** Variables
- x = distance from lamppost to person
- s = length of shadow
- Given: dx/dt = 4 ft/s
- Find: d(x+s)/dt (rate of shadow tip moving)

**Step 2:** Use similar triangles
$$\frac{15}{x+s} = \frac{6}{s}$$
$$15s = 6(x+s) = 6x + 6s$$
$$9s = 6x$$
$$s = \frac{2x}{3}$$

**Step 3:** Differentiate
$$\frac{ds}{dt} = \frac{2}{3}\frac{dx}{dt} = \frac{2}{3}(4) = \frac{8}{3}$$

**Step 4:** Find rate of shadow tip
$$\frac{d(x+s)}{dt} = \frac{dx}{dt} + \frac{ds}{dt} = 4 + \frac{8}{3} = \frac{20}{3}$$

**Answer:** The shadow tip moves at **20/3 ft/s** ≈ 6.67 ft/s.

---

### Example 5: Two Ships

**Problem:** Ship A is 100 km north of Ship B. Ship A sails south at 20 km/h, Ship B sails east at 15 km/h. How fast is the distance between them changing after 2 hours?

**Solution:**

**Step 1:** Set up coordinates
At time t hours:
- Ship A: starts at (0, 100), moves south → position (0, 100-20t)
- Ship B: starts at (0, 0), moves east → position (15t, 0)
- Let D = distance between ships

**Step 2:** Distance formula
$$D^2 = (15t)^2 + (100-20t)^2$$

**Step 3:** Differentiate
$$2D\frac{dD}{dt} = 2(15t)(15) + 2(100-20t)(-20)$$
$$D\frac{dD}{dt} = 225t - 20(100-20t) = 225t - 2000 + 400t = 625t - 2000$$

**Step 4:** At t = 2:
- Ship A at (0, 60), Ship B at (30, 0)
- D = √(30² + 60²) = √(900 + 3600) = √4500 = 30√5

$$30\sqrt{5} \cdot \frac{dD}{dt} = 625(2) - 2000 = -750$$
$$\frac{dD}{dt} = \frac{-750}{30\sqrt{5}} = \frac{-25}{\sqrt{5}} = -5\sqrt{5} \approx -11.18$$

**Answer:** Distance is decreasing at **5√5 km/h** ≈ 11.18 km/h.

---

## 📝 Practice Problems

### Level 1: Basic
1. A circle's radius increases at 3 cm/s. Find dA/dt when r = 10 cm.
2. A square's side length increases at 2 cm/s. Find dA/dt when s = 5 cm.
3. Air is pumped into a balloon at 4 cm³/s. How fast is the radius increasing when r = 2 cm?

### Level 2: Geometric
4. A 25-foot ladder slides down a wall. The bottom moves at 2 ft/s. How fast is the top moving when the bottom is 7 ft from the wall?
5. A kite is 100 ft high moving horizontally at 8 ft/s. How fast is the string being released when 200 ft of string is out?
6. A cone-shaped container (height 12 ft, radius 4 ft) is being filled at 3 ft³/min. How fast is the water rising when h = 6 ft?

### Level 3: Motion
7. Two cars leave an intersection. One goes north at 40 mph, one goes east at 30 mph. How fast is the distance between them increasing after 2 hours?
8. A plane flies at altitude 3 km toward an observer at 500 km/h. How fast is the distance changing when the plane is 5 km away from the observer?
9. A point moves along y = x². At what point is dy/dt three times dx/dt?

### Level 4: Challenge
10. Water leaks from a conical tank at 0.5 m³/min while being filled at 1 m³/min. The tank has height 10m and radius 5m. Find dh/dt when h = 4m.
11. A searchlight is 100m from a straight wall. It rotates at 2 rad/s. How fast does the light move along the wall when the angle from perpendicular is 45°?

---

## 📊 Answers

1. dA/dt = 60π cm²/s
2. dA/dt = 20 cm²/s
3. dr/dt = 1/(4π) cm/s
4. dy/dt = -7/12 ft/s
5. ds/dt = 4√3 ft/s ≈ 6.93 ft/s
6. dh/dt = 3/(π) ft/min
7. dD/dt = 50 mph
8. dD/dt = -400 km/h
9. At point (3/2, 9/4)
10. dh/dt = 1/(2π) m/min
11. dx/dt = 400 m/s

---

## 🔬 Physics Connection

Related rates appear everywhere in physics:
- **Thermodynamics:** PV = nRT, how pressure changes as volume changes
- **Circuit theory:** Q = CV, relating charge and voltage changes
- **Orbital mechanics:** Changes in orbital parameters

### Quantum Mechanics Preview
The time-dependent Schrödinger equation relates the rate of change of the wave function to its spatial derivatives:
$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V\psi$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 3.9
- [ ] Master the 5-step problem-solving strategy
- [ ] Complete Examples 1-5 independently
- [ ] Solve Level 1-2 practice problems
- [ ] Attempt Level 3-4 problems
- [ ] Draw diagrams for every problem
- [ ] Check units in all answers

---

## 📓 Reflection Questions

1. Why must we differentiate BEFORE substituting values?
2. How does the chain rule connect to related rates?
3. What's the most common mistake in related rates problems?
4. Give a real-world example of related rates from your experience.

---

## 🔜 Preview: Tomorrow

**Day 16: Linear Approximation and Differentials**
- Using tangent lines to approximate function values
- The differential dy = f'(x)dx
- Error analysis

---

*"The rate at which things change is often more important than the things themselves."*
— Anonymous
