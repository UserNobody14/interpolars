# scirs2-interpolate: Issues & Proposed Fixes

We attempted to adopt `scirs2-interpolate` (v0.1.5) as a drop-in replacement for
custom interpolation math in our Polars plugin. While the library covers the right
surface area, we hit several correctness and API issues that forced us to fall back
to custom implementations for 4 of 6 methods and abandon `RegularGridInterpolator`
entirely.

This document describes each issue, demonstrates the incorrect behavior, and
sketches what a fix inside `scirs2-interpolate` would look like.

---

## 1. `RegularGridInterpolator` does not extrapolate with `Linear` method

### Symptom

Constructing a `RegularGridInterpolator` with `ExtrapolateMode::Extrapolate` and
`InterpolationMethod::Linear`, then evaluating at a point outside the grid, returns
the boundary value (nearest-neighbor clamp) instead of a linearly extrapolated value.

```rust
use scirs2_interpolate::interpnd::*;
use ndarray::{Array, Array1, Array2, IxDyn};

let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
let interp = RegularGridInterpolator::new(
    vec![x],
    values,
    InterpolationMethod::Linear,
    ExtrapolateMode::Extrapolate, // should extrapolate
).unwrap();

let xi = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
let result = interp.__call__(&xi.view()).unwrap();
// ACTUAL:   result[0] == 20.0  (boundary clamp)
// EXPECTED: result[0] == 30.0  (linear extrapolation: slope=10, 20 + 10*(3-2))
```

### Expected behavior

`ExtrapolateMode::Extrapolate` with `Linear` should continue the slope of the
boundary cell beyond the grid, matching `scipy.RegularGridInterpolator(...,
bounds_error=False, fill_value=None)`.

### Suggested fix

In the `Linear` evaluation path of `RegularGridInterpolator::__call__`, when a
coordinate falls outside the grid, the code currently clamps the index. Instead it
should:

1. Detect that the query point is out-of-bounds on dimension `d`.
2. Use the boundary cell's slope `(values[i+1] - values[i]) / (points[d][i+1] - points[d][i])`
   where `i` is 0 (below) or `n-2` (above).
3. Linearly extrapolate from the boundary point using that slope.

This mirrors how `Interp1d` with `ExtrapolateMode::Extrapolate` and `Linear`
correctly extrapolates in 1D — the same logic just needs to be applied per-dimension
in the ND case.

---

## 2. `RegularGridInterpolator` requires >= 2 points per dimension

### Symptom

```rust
RegularGridInterpolator::new(
    vec![Array1::from_vec(vec![5.0])], // 1 point
    Array::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap(),
    InterpolationMethod::Linear,
    ExtrapolateMode::Extrapolate,
); // => Err: "any dimension has less than 2 points"
```

### Expected behavior

A single-point axis should be valid. Interpolation along that dimension is trivially
the constant value. This is important in practice: a 3D grid where one dimension has
been sliced to a single level should still interpolate over the remaining dimensions.

scipy handles this gracefully — `RegularGridInterpolator` accepts length-1 axes and
treats them as constant along that dimension.

### Suggested fix

In `RegularGridInterpolator::new`, relax the validation from `>= 2` to `>= 1`. In
the per-dimension interpolation logic, when `points[d].len() == 1`, skip
interpolation on that dimension and pass the single value through. No slope
calculation is needed.

---

## 3. `RegularGridInterpolator` has no `Nearest`-clamp extrapolation mode

### Symptom

The `ExtrapolateMode` enum for `interpnd` has three variants:

```rust
pub enum ExtrapolateMode {
    Nan,        // return NaN
    Error,      // return Err
    Extrapolate // continue beyond boundary
}
```

There is no equivalent of `interp1d::ExtrapolateMode::Nearest`, which clamps
out-of-range queries to the boundary value. This is the most common "no
extrapolation" behavior (matching scipy's `bounds_error=False,
fill_value=(y[0], y[-1])`).

### Expected behavior

Add a `Nearest` variant to `interpnd::ExtrapolateMode` that clamps query
coordinates to the grid bounds before interpolation. This is distinct from `Nan`
(which returns NaN) and `Error` (which fails).

### Suggested fix

```rust
pub enum ExtrapolateMode {
    Nan,
    Error,
    Nearest,     // <-- new: clamp to boundary
    Extrapolate,
}
```

When `Nearest` is active, clamp each query coordinate to
`[points[d][0], points[d][last]]` before performing interpolation. This is trivial
to implement and eliminates the need for callers to manually clamp.

---

## 4. `AkimaSpline` requires >= 5 data points

### Symptom

```rust
use scirs2_interpolate::advanced::akima::AkimaSpline;
use ndarray::ArrayView1;

let x = [0.0, 1.0, 2.0, 3.0]; // 4 points
let y = [0.0, 1.0, 0.0, 1.0];
AkimaSpline::new(
    &ArrayView1::from(&x),
    &ArrayView1::from(&y),
); // => Err: "at least 5 points are required for Akima spline"
```

### Expected behavior

The Akima algorithm extends the slope array with 2 phantom points on each side,
which naturally handles `n >= 2`. scipy's `Akima1DInterpolator` accepts any `n >= 2`.
Requiring 5 is an unnecessary restriction.

### Suggested fix

The boundary extension formulas (`m[1] = 2*m[2] - m[3]`, etc.) work for any
`n >= 3`. For `n == 2`, the spline degenerates to linear interpolation (single
interval, slope = Δy/Δx). Change the validation from `n >= 5` to `n >= 2` and
ensure the boundary formulas handle the `n == 2` case (where there's only one actual
slope).

---

## 5. `AkimaSpline::evaluate` panics on out-of-bounds input

### Symptom

```rust
let spline = AkimaSpline::new(&x_view, &y_view).unwrap();
spline.evaluate(x_outside_range); // panics with OutOfBounds error
```

The `evaluate` method returns `Result`, but with `OutOfBounds` as the error variant,
which in practice causes panics when unwrapped. There's no way to configure the
spline for extrapolation or clamping at construction time.

### Expected behavior

`AkimaSpline` should support the same `ExtrapolateMode` as `Interp1d`:

- `Nearest`: clamp `x` to `[x[0], x[n-1]]` and evaluate the boundary polynomial
- `Extrapolate`: evaluate the boundary polynomial at the out-of-range `x`
  (polynomial continuation, matching scipy's `Akima1DInterpolator(extrapolate=True)`)
- `Error` / `Nan`: current behavior

### Suggested fix

Add an `extrapolate` field to `AkimaSpline` (set at construction or via a builder
method). In `evaluate`, instead of returning `OutOfBounds`:

- **Nearest**: clamp `x` to `[x_min, x_max]`, evaluate at the clamped value.
- **Extrapolate**: when `x < x[0]`, evaluate the polynomial for interval 0 at `x`.
  When `x > x[n-1]`, evaluate the polynomial for interval `n-2` at `x`. The
  polynomial coefficients are already computed for each interval, so this is just
  removing the bounds check.

---

## 6. `Interp1d::Pchip` extrapolation does not match scipy

### Symptom

```rust
use scirs2_interpolate::interp1d::*;
use ndarray::ArrayView1;

let x = ArrayView1::from(&[0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]);
let y = ArrayView1::from(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]);
let interp = Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

interp.evaluate(-1.0)  // => -1.0  (linear extrapolation from boundary interval)
// scipy PchipInterpolator: -3.0  (polynomial continuation using PCHIP derivative)

interp.evaluate(11.0)  // => 9.75  (linear extrapolation)
// scipy PchipInterpolator: 8.841346...  (polynomial continuation)
```

### Expected behavior

`ExtrapolateMode::Extrapolate` for PCHIP should continue the boundary Hermite
polynomial beyond the grid, not fall back to linear extrapolation. This matches
scipy's `PchipInterpolator(extrapolate=True)`.

The boundary polynomial is a cubic Hermite defined by `(x[i], y[i], d[i])` and
`(x[i+1], y[i+1], d[i+1])` where `d` are the PCHIP slopes. For `x < x[0]`, the
polynomial for interval 0 should be evaluated. For `x > x[n-1]`, the polynomial for
interval `n-2` should be evaluated.

### Suggested fix

In the `Pchip` evaluation path within `Interp1d`, when `x` is out of bounds and
`ExtrapolateMode::Extrapolate` is set:

1. Select the boundary interval (interval 0 for below, interval `n-2` for above).
2. Evaluate the Hermite polynomial for that interval at `x` (no clamping).

The Hermite coefficients are already computed during construction. The fix is to
remove the bounds-clamping logic and let the polynomial evaluate naturally at the
out-of-range `x`. The Hermite basis functions `H00, H10, H01, H11` work for any `t`
value (not just `t ∈ [0, 1]`), producing the natural polynomial continuation.

---

## Summary

| # | Component | Issue | Severity |
|---|-----------|-------|----------|
| 1 | `RegularGridInterpolator` | Linear extrapolation returns clamped values | High |
| 2 | `RegularGridInterpolator` | Rejects single-point axes | Medium |
| 3 | `RegularGridInterpolator` | No `Nearest` clamp mode in `ExtrapolateMode` | Low |
| 4 | `AkimaSpline` | Requires >= 5 points (scipy needs only 2) | Medium |
| 5 | `AkimaSpline` | No extrapolation/clamp support, panics on OOB | High |
| 6 | `Interp1d::Pchip` | Extrapolation uses linear instead of polynomial continuation | Medium |

Issues 1, 5, and 6 are correctness bugs that prevent matching scipy's behavior.
Issues 2 and 4 are unnecessary restrictions. Issue 3 is a missing convenience that
forces callers to manually clamp coordinates.
