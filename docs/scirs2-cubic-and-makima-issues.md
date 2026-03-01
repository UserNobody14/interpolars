# scirs2-interpolate: Cubic Spline Mismatch & Missing Makima

Follow-up to [scirs2-interpolate-issues.md](./scirs2-interpolate-issues.md). This
document covers two remaining gaps between `scirs2-interpolate` and
`scipy.interpolate` that forced us to retain custom implementations in
`interpolars`: (1) cubic spline interpolation and (2) the Makima algorithm.

---

## 1. `Interp1d::Cubic` uses Catmull-Rom, not a cubic spline

### Symptom

`Interp1d::new(&x, &y, InterpolationMethod::Cubic, ...)` does not match
`scipy.interpolate.interp1d(x, y, kind='cubic')`. In particular, it fails to
reproduce linear data exactly and gives different values on curved data.

### Root cause

scipy's `kind='cubic'` is a **not-a-knot cubic spline**: a global method that
solves an (n × n) tridiagonal system for the second-derivative "moments" M_i,
with the boundary condition that the third derivative is continuous at the 2nd
and penultimate knots.

scirs2's `cubic_interp` (called by `Interp1d` for the `Cubic` variant) is a
**Catmull-Rom spline**: a local method that uses a sliding 4-point stencil with
uniform parameterization.

```rust
// scirs2-interpolate/src/interp1d/mod.rs, line 381
fn cubic_interp<F: Float + FromPrimitive>(
    x: &ArrayView1<F>, y: &ArrayView1<F>, idx: usize, xnew: F,
) -> InterpolateResult<F> {
    // 4-point stencil with boundary duplication
    let (i0, i1, i2, i3) = if idx == 0 {
        (0, 0, 1, 2)        // duplicates the first point
    } else if idx == x.len() - 2 {
        (idx - 1, idx, idx + 1, idx + 1) // duplicates the last point
    } else {
        (idx - 1, idx, idx + 1, idx + 2)
    };
    // ...
    // Catmull-Rom formula (assumes uniform spacing between stencil points):
    // p(t) = 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t² + (-p0+3*p1-3*p2+p3)*t³)
}
```

### Key differences

| Property | scipy (`kind='cubic'`) | scirs2 `Interp1d::Cubic` |
|----------|----------------------|--------------------------|
| Algorithm | Not-a-knot cubic spline | Catmull-Rom spline |
| Solve scope | Global tridiagonal system | Local 4-point stencil |
| Boundary handling | 3rd-derivative continuity at 2nd/penultimate knots | Duplicates boundary points (`i0=i1` or `i2=i3`) |
| Spacing assumption | Handles non-uniform spacing correctly | Catmull-Rom formula assumes uniform spacing; applies it to non-uniform data via `t = (x-x1)/(x2-x1)` normalization but ignores the x-distances to the outer two stencil points |
| Linear reproduction | Exact — a cubic spline through collinear points produces zero curvature | **Not exact** — boundary point duplication introduces artificial curvature |
| C² continuity | Guaranteed globally | Not guaranteed — second derivatives can be discontinuous at knots |
| Minimum points | 4 (for not-a-knot conditions) | 3 (degrades to using duplicated stencil points) |

### Concrete example

```python
import numpy as np
from scipy.interpolate import interp1d

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])   # y = x²

f = interp1d(x, y, kind='cubic')
print(f(1.5))  # 2.25 exactly (cubic spline reproduces quadratics exactly)
```

scirs2's Catmull-Rom will give a different value at `x=1.5` because:
- At the left boundary (`idx=0`), the stencil is `(0,0,1,2)` — point 0 is
  duplicated, which skews the Catmull-Rom tangent estimate.
- The formula ignores the actual spacing between the outer stencil points.

### scirs2 already has a not-a-knot cubic spline — it's just not wired up

`scirs2-interpolate` contains `CubicSpline::new_not_a_knot()` in the
`spline_modules` subsystem (`spline_modules/core.rs`, line 306), backed by
`compute_not_a_knot_cubic_spline()` in `spline_modules/algorithms.rs` (line
208). This implementation uses a proper tridiagonal system with not-a-knot
boundary conditions.

However, `Interp1d::Cubic` does **not** use it. The dispatch in
`interp1d/mod.rs` (line 241) calls the Catmull-Rom `cubic_interp` function
instead.

### Suggested fix

**Option A (minimal):** Change the `Interp1d::Cubic` dispatch to create a
`CubicSpline::new_not_a_knot()` and evaluate through it, mirroring how
`Interp1d::Pchip` already delegates to `PchipInterpolator`:

```rust
// In interp1d/mod.rs, evaluate_at()
InterpolationMethod::Cubic => {
    let spline = CubicSpline::new_not_a_knot(&self.x.view(), &self.y.view())?;
    Ok(spline.evaluate(xnew)?)
}
```

This requires verifying that `CubicSpline::new_not_a_knot()` correctly handles:
- Extrapolation modes (currently `CubicSpline` has its own extrapolation logic
  that may need alignment with `Interp1d`'s `ExtrapolateMode`).
- The n=3 case (not-a-knot requires n >= 4; for n=3 scipy falls back to a
  quadratic fit; `CubicSpline::new_not_a_knot` currently rejects n < 4).

**Option B (thorough):** Additionally audit `compute_not_a_knot_cubic_spline`
against scipy's implementation to ensure the boundary condition formulation is
correct. The current implementation solves for second derivatives `sigma[i]`
via Thomas algorithm, but the boundary rows should satisfy the not-a-knot
condition:

```
h₁·σ₀ − (h₀+h₁)·σ₁ + h₀·σ₂ = 0       (left)
h_{n-2}·σ_{n-3} − (h_{n-3}+h_{n-2})·σ_{n-2} + h_{n-3}·σ_{n-1} = 0  (right)
```

The existing code sets up the first row as
`b[0]=h1, c[0]=h0+h1, d[0]=<derivative estimate>`, which does not encode the
above relation (it omits the σ₂ term and uses a non-zero RHS). This should be
verified against scipy's `CubicSpline` source to confirm correctness.

### Why we retained a custom implementation

Our `CubicSplineInterp::compute_moments` uses the second approach: eliminate
M₀ and M_{n−1} via the not-a-knot relations to produce a reduced (n−2)×(n−2)
tridiagonal system. This has been validated against scipy's output for linear,
quadratic, cubic, and oscillatory test data across uniform and non-uniform
grids.

---

## 2. `MonotonicMethod::ModifiedAkima` is not scipy's Makima

### Background: what is Makima?

scipy (≥ 1.12) provides `Akima1DInterpolator(x, y, method='makima')`, which
implements the **Modified Akima** algorithm. Both standard Akima and Makima are
Hermite splines — they compute a slope d_i at each knot and build piecewise
cubic Hermite polynomials. The difference is in how the weights for the slope
calculation are computed.

Given interval slopes `δ_k = (y_{k+1} − y_k) / (x_{k+1} − x_k)`, and extended
slopes at the boundaries via `δ_{-1} = 2δ₀ − δ₁`, `δ_{-2} = 2δ_{-1} − δ₀`,
etc., the slope at knot i is:

```
d_i = (w₁·δ_{i-1} + w₂·δ_i) / (w₁ + w₂)
```

where the weights differ between methods:

| Method | w₁ | w₂ |
|--------|----|----|
| **Standard Akima (1970)** | \|δ_{i+1} − δ_i\| | \|δ_{i-1} − δ_{i-2}\| |
| **Makima (scipy)** | \|δ_{i+1} − δ_i\| + ½\|δ_{i+1} + δ_i\| | \|δ_{i-1} − δ_{i-2}\| + ½\|δ_{i-1} + δ_{i-2}\| |

The additional `½|δ_a + δ_b|` term in Makima biases the weights toward segments
with larger absolute slopes. This provides significantly better resistance to
outliers and flat regions compared to standard Akima, without enforcing strict
monotonicity.

Crucially, **Makima does not enforce monotonicity**. It is a general-purpose
smooth interpolant that simply has improved weighting relative to standard
Akima.

### What scirs2's `ModifiedAkima` actually does

`scirs2-interpolate` provides `MonotonicMethod::ModifiedAkima` in the
`MonotonicInterpolator` (file `interp1d/monotonic.rs`, line 557). Despite the
similar name, this algorithm differs from scipy's Makima in three fundamental
ways:

**1. Different weight formula**

```rust
// scirs2: find_modified_akima_derivatives (line 618)
let w1 = (s3 - s2).abs();   // |δ_i − δ_{i-1}|
let w2 = (s1 - s4).abs();   // |δ_{i-2} − δ_{i+1}|
```

These are the standard Akima weights — there is no `½|δ_a + δ_b|` bias term.
Worse, `w2` uses `|s1 - s4|` which computes `|δ_{i-2} − δ_{i+1}|` — a
4-interval span instead of the correct `|δ_{i-1} − δ_{i-2}|` (2-interval
span). This is not the standard Akima formula either.

**2. Enforced monotonicity**

After computing the weighted slope, scirs2 applies a monotonicity filter:

```rust
// scirs2: line 632
if s2 * s3 <= F::zero() {
    derivatives[i] = F::zero();   // zero derivative at local extrema
} else {
    let max_slope = three * F::min(s2.abs(), s3.abs());
    if derivatives[i].abs() > max_slope {
        derivatives[i] = max_slope * derivatives[i].signum(); // clamp
    }
}
```

This forces the interpolant to be monotone between each pair of adjacent knots.
scipy's Makima does **not** do this — it allows overshooting. The monotonicity
enforcement is appropriate for a `MonotonicInterpolator` (that's its purpose),
but makes the algorithm unsuitable as a scipy-compatible Makima.

**3. Harmonic-mean endpoint derivatives**

```rust
// scirs2: line 652
derivatives[0] = (two * s1 * s2) / (s1 + s2); // harmonic mean
```

scipy's Makima uses the same extended-slope boundary formulas as standard
Akima:
```
δ_{-1} = 2·δ₀ − δ₁
δ_{-2} = 2·δ_{-1} − δ₀
```
and then applies the Makima weight formula uniformly at all knots including
endpoints. The harmonic mean used by scirs2 gives different boundary slopes.

### Summary of algorithmic differences

| Aspect | scipy Makima | scirs2 `ModifiedAkima` |
|--------|-------------|----------------------|
| Weight formula | Standard Akima + ½\|δ_a+δ_b\| bias | Standard Akima (incorrect stencil for w₂) |
| Monotonicity | **Not enforced** — general-purpose spline | **Strictly enforced** — d_i = 0 at extrema, clamped to 3·min(\|δ\|) |
| Boundary slopes | Extended-slope formula + Makima weights | Harmonic mean of adjacent slopes |
| Located in | General interpolation (alongside Akima) | `MonotonicInterpolator` — a monotonicity-preserving tool |
| Use case | Drop-in replacement for Akima with better outlier handling | Shape-preserving interpolation for monotonic data |

### Suggested fix: implement scipy-compatible Makima in scirs2

The cleanest approach is to add Makima as a sibling to the existing
`AkimaSpline` in `scirs2-interpolate/src/advanced/akima.rs`, since they share
~90% of the same code.

**Step 1: Add a variant flag**

```rust
pub enum AkimaMethod {
    Original,   // Akima 1970
    Makima,     // Modified Akima (scipy-compatible)
}

impl AkimaSpline {
    pub fn new_makima(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> InterpolateResult<Self> {
        Self::with_method(x, y, AkimaMethod::Makima)
    }
}
```

**Step 2: Modify the slope computation**

The slope calculation is identical to standard Akima except for the weight
formula. In the existing slope loop, add a branch:

```rust
// For each interior knot i:
let (w1, w2) = match method {
    AkimaMethod::Original => (
        (delta[i+1] - delta[i]).abs(),
        (delta[i-1] - delta[i-2]).abs(),
    ),
    AkimaMethod::Makima => (
        (delta[i+1] - delta[i]).abs() + 0.5 * (delta[i+1] + delta[i]).abs(),
        (delta[i-1] - delta[i-2]).abs() + 0.5 * (delta[i-1] + delta[i-2]).abs(),
    ),
};
```

The boundary extension (`δ_{-1} = 2δ₀ − δ₁`, etc.), the weighted-average slope
formula, and the fallback to arithmetic mean when `w₁ + w₂ ≈ 0` are all
identical for both methods.

**Step 3: Expose through `Interp1d`**

Add `InterpolationMethod::Makima` to the `Interp1d` enum, dispatching to
`AkimaSpline::new_makima()`:

```rust
InterpolationMethod::Makima => {
    let spline = AkimaSpline::new_makima(&self.x.view(), &self.y.view())?
        .with_extrapolation(self.extrapolate);
    Ok(spline.evaluate(xnew)?)
}
```

### Why we retained a custom implementation

Our `makima_slopes` function (`interpolation.rs`, line 320) implements the
exact scipy Makima weight formula with proper boundary extension:

```rust
w1[i] = dm[i + 2] + 0.5 * (m[i + 3] + m[i + 2]).abs();  // |Δδ| + ½|δ+δ|
w2[i] = dm[i] + 0.5 * (m[i + 1] + m[i]).abs();
```

This has been validated against scipy's `Akima1DInterpolator(method='makima')`
across multiple test datasets including flat regions, outliers, and
non-uniform spacing.

---

## Summary

| # | Component | Issue | Custom impl retained? |
|---|-----------|-------|-----------------------|
| 1 | `Interp1d::Cubic` | Uses Catmull-Rom instead of not-a-knot cubic spline | Yes — `CubicSplineInterp` with correct not-a-knot moments |
| 2 | `MonotonicMethod::ModifiedAkima` | Different algorithm from scipy's Makima (wrong weights, enforced monotonicity, different endpoints) | Yes — `MakimaInterp` with exact scipy weight formula |

Both custom implementations match scipy's output. The scirs2 fixes are
straightforward: wire `Interp1d::Cubic` to `CubicSpline::new_not_a_knot()`,
and add a proper Makima variant alongside `AkimaSpline`.
