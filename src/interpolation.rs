//! Pure interpolation math -- no Polars dependency.
//!
//! Provides a trait-driven architecture for 1D interpolation methods and an N-D
//! engine that decomposes multi-dimensional interpolation into successive 1D passes
//! (mathematically equivalent to tensor-product interpolation on rectilinear grids).

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Trait & method enum
// ---------------------------------------------------------------------------

/// 1D interpolation of a function defined by sorted knots `(xs, ys)` at target `x`.
pub trait Interpolate1D: Sync + Send {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64;
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    Linear,
    Nearest,
    Cubic,
    Pchip,
    Akima,
    Makima,
}

static LINEAR: LinearInterp = LinearInterp;
static NEAREST: NearestInterp = NearestInterp;
static CUBIC: CubicSplineInterp = CubicSplineInterp;
static PCHIP: PchipInterp = PchipInterp;
static AKIMA: AkimaInterp = AkimaInterp { makima: false };
static MAKIMA: AkimaInterp = AkimaInterp { makima: true };

impl InterpolationMethod {
    pub fn as_interpolator(&self) -> &'static dyn Interpolate1D {
        match self {
            Self::Linear => &LINEAR,
            Self::Nearest => &NEAREST,
            Self::Cubic => &CUBIC,
            Self::Pchip => &PCHIP,
            Self::Akima => &AKIMA,
            Self::Makima => &MAKIMA,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Return `i` such that `xs[i] <= x < xs[i+1]`, clamped to `[0, n-2]`.
pub(crate) fn find_interval(xs: &[f64], x: f64) -> usize {
    debug_assert!(xs.len() >= 2);
    let n = xs.len();
    if x <= xs[0] {
        return 0;
    }
    if x >= xs[n - 1] {
        return n - 2;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn sign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Evaluate the cubic Hermite basis on interval `[x0, x1]`.
fn hermite_eval(x0: f64, x1: f64, y0: f64, y1: f64, d0: f64, d1: f64, x: f64) -> f64 {
    let h = x1 - x0;
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * d1
}

/// Lagrange quadratic through exactly 3 points (fallback for cubic with n=3).
fn eval_quadratic(xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
    debug_assert_eq!(xs.len(), 3);
    if !extrapolate {
        if x <= xs[0] {
            return ys[0];
        }
        if x >= xs[2] {
            return ys[2];
        }
    }
    let l0 = (x - xs[1]) * (x - xs[2]) / ((xs[0] - xs[1]) * (xs[0] - xs[2]));
    let l1 = (x - xs[0]) * (x - xs[2]) / ((xs[1] - xs[0]) * (xs[1] - xs[2]));
    let l2 = (x - xs[0]) * (x - xs[1]) / ((xs[2] - xs[0]) * (xs[2] - xs[1]));
    ys[0] * l0 + ys[1] * l1 + ys[2] * l2
}

/// Generic Hermite-spline evaluator used by PCHIP, Akima, and Makima.
fn eval_hermite_spline(
    xs: &[f64],
    ys: &[f64],
    slopes: &[f64],
    x: f64,
    extrapolate: bool,
) -> f64 {
    let n = xs.len();
    if !extrapolate {
        if x <= xs[0] {
            return ys[0];
        }
        if x >= xs[n - 1] {
            return ys[n - 1];
        }
    }
    let i = find_interval(xs, x);
    hermite_eval(xs[i], xs[i + 1], ys[i], ys[i + 1], slopes[i], slopes[i + 1], x)
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

pub struct LinearInterp;

impl Interpolate1D for LinearInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        if !extrapolate {
            if x <= xs[0] {
                return ys[0];
            }
            if x >= xs[n - 1] {
                return ys[n - 1];
            }
        }
        let i = find_interval(xs, x);
        let h = xs[i + 1] - xs[i];
        if h == 0.0 {
            return ys[i];
        }
        let t = (x - xs[i]) / h;
        ys[i] + t * (ys[i + 1] - ys[i])
    }
}

// ---------------------------------------------------------------------------
// Nearest
// ---------------------------------------------------------------------------

pub struct NearestInterp;

impl Interpolate1D for NearestInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, _extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        let i = find_interval(xs, x);
        let d_lo = (x - xs[i]).abs();
        let d_hi = (xs[i + 1] - x).abs();
        if d_lo <= d_hi {
            ys[i]
        } else {
            ys[i + 1]
        }
    }
}

// ---------------------------------------------------------------------------
// Cubic spline (not-a-knot, matching scipy interp1d(kind='cubic'))
// ---------------------------------------------------------------------------

pub struct CubicSplineInterp;

impl CubicSplineInterp {
    /// Compute second-derivative "moments" M_i at each knot via the not-a-knot
    /// tridiagonal system. Requires `n >= 4`.
    fn compute_moments(xs: &[f64], ys: &[f64]) -> Vec<f64> {
        let n = xs.len();
        debug_assert!(n >= 4);

        let h: Vec<f64> = (0..n - 1).map(|i| xs[i + 1] - xs[i]).collect();
        let delta: Vec<f64> = (1..n - 1)
            .map(|i| 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]))
            .collect();

        let m = n - 2; // interior unknowns M_1 .. M_{n-2}

        // Not-a-knot: M_0 = alpha_l*M_1 + beta_l*M_2
        let alpha_l = (h[0] + h[1]) / h[1];
        let beta_l = -h[0] / h[1];
        // Not-a-knot: M_{n-1} = alpha_r*M_{n-2} + beta_r*M_{n-3}
        let alpha_r = (h[n - 3] + h[n - 2]) / h[n - 3];
        let beta_r = -h[n - 2] / h[n - 3];

        // Tridiagonal coefficients (lower, diag, upper, rhs)
        let mut a = vec![0.0; m];
        let mut b = vec![0.0; m];
        let mut c = vec![0.0; m];
        let mut rhs = delta; // already length m

        // First row: substitute M_0
        b[0] = h[0] * alpha_l + 2.0 * (h[0] + h[1]);
        c[0] = h[0] * beta_l + h[1];

        // Interior rows (unchanged)
        for j in 1..m.saturating_sub(1) {
            let i = j + 1; // original point index
            a[j] = h[i - 1];
            b[j] = 2.0 * (h[i - 1] + h[i]);
            c[j] = h[i];
        }

        // Last row: substitute M_{n-1}
        if m >= 2 {
            a[m - 1] = h[n - 3] + h[n - 2] * beta_r;
            b[m - 1] = 2.0 * (h[n - 3] + h[n - 2]) + h[n - 2] * alpha_r;
        }

        // Thomas algorithm -- forward sweep
        for i in 1..m {
            let w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }

        // Back-substitution
        let mut mi = vec![0.0; m];
        mi[m - 1] = rhs[m - 1] / b[m - 1];
        for i in (0..m - 1).rev() {
            mi[i] = (rhs[i] - c[i] * mi[i + 1]) / b[i];
        }

        // Full moments vector
        let mut moments = vec![0.0; n];
        for j in 0..m {
            moments[j + 1] = mi[j];
        }
        moments[0] = alpha_l * moments[1] + beta_l * moments[2];
        moments[n - 1] = alpha_r * moments[n - 2] + beta_r * moments[n - 3];
        moments
    }
}

impl Interpolate1D for CubicSplineInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        if n == 2 {
            return LinearInterp.eval(xs, ys, x, extrapolate);
        }
        if n == 3 {
            return eval_quadratic(xs, ys, x, extrapolate);
        }

        if !extrapolate {
            if x <= xs[0] {
                return ys[0];
            }
            if x >= xs[n - 1] {
                return ys[n - 1];
            }
        }

        let moments = Self::compute_moments(xs, ys);
        let i = find_interval(xs, x);
        let hi = xs[i + 1] - xs[i];
        let a = xs[i + 1] - x;
        let b = x - xs[i];

        moments[i] * a * a * a / (6.0 * hi)
            + moments[i + 1] * b * b * b / (6.0 * hi)
            + (ys[i] / hi - moments[i] * hi / 6.0) * a
            + (ys[i + 1] / hi - moments[i + 1] * hi / 6.0) * b
    }
}

// ---------------------------------------------------------------------------
// PCHIP  (Fritsch-Carlson monotone cubic Hermite, matches scipy PchipInterpolator)
// ---------------------------------------------------------------------------

pub struct PchipInterp;

/// One-sided three-point derivative estimate with monotonicity constraints,
/// matching `scipy.interpolate.PchipInterpolator._edge_case`.
fn pchip_edge_case(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0) {
        0.0
    } else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs() {
        3.0 * m0
    } else {
        d
    }
}

fn pchip_slopes(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    debug_assert!(n >= 2);

    let h: Vec<f64> = (0..n - 1).map(|i| xs[i + 1] - xs[i]).collect();
    let m: Vec<f64> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    if n == 2 {
        return vec![m[0]; 2];
    }

    let mut dk = vec![0.0; n];

    // Interior points: weighted harmonic mean of adjacent secants
    for k in 1..n - 1 {
        if m[k - 1] * m[k] <= 0.0 {
            dk[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            dk[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k]);
        }
    }

    // Endpoints
    dk[0] = pchip_edge_case(h[0], h[1], m[0], m[1]);
    dk[n - 1] = pchip_edge_case(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
    dk
}

impl Interpolate1D for PchipInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        let slopes = pchip_slopes(xs, ys);
        eval_hermite_spline(xs, ys, &slopes, x, extrapolate)
    }
}

// ---------------------------------------------------------------------------
// Akima / Makima  (matches scipy.interpolate.Akima1DInterpolator)
// ---------------------------------------------------------------------------

pub struct AkimaInterp {
    makima: bool,
}

fn akima_slopes(xs: &[f64], ys: &[f64], makima: bool) -> Vec<f64> {
    let n = xs.len();
    debug_assert!(n >= 2);

    let ns = n - 1; // number of secant segments

    if ns == 1 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s; n];
    }

    // Padded secant array: size ns + 4 = n + 3
    // m[2..2+ns] hold the actual secants; m[0..2] and m[2+ns..] are extrapolated.
    let total = ns + 4;
    let mut m = vec![0.0; total];
    for i in 0..ns {
        m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[total - 2] = 2.0 * m[total - 3] - m[total - 4];
    m[total - 1] = 2.0 * m[total - 2] - m[total - 3];

    // dm[j] = |m[j+1] - m[j]|
    let dm: Vec<f64> = (0..total - 1).map(|i| (m[i + 1] - m[i]).abs()).collect();

    // Compute per-point weights.  For makima, each weight gets its own
    // absolute-sum-of-secants term (pm), matching scipy exactly:
    //   f1[i] = dm[i+2] + 0.5*|m[i+3]+m[i+2]|
    //   f2[i] = dm[i]   + 0.5*|m[i+1]+m[i]|
    let mut w1 = vec![0.0; n];
    let mut w2 = vec![0.0; n];
    for i in 0..n {
        if makima {
            w1[i] = dm[i + 2] + 0.5 * (m[i + 3] + m[i + 2]).abs();
            w2[i] = dm[i] + 0.5 * (m[i + 1] + m[i]).abs();
        } else {
            w1[i] = dm[i + 2];
            w2[i] = dm[i];
        }
    }

    let f12_max: f64 = (0..n).map(|i| w1[i] + w2[i]).fold(0.0_f64, f64::max);
    let threshold = 1e-9 * f12_max;

    let mut dk = vec![0.0; n];
    for i in 0..n {
        let f12 = w1[i] + w2[i];
        if f12 > threshold {
            dk[i] = (w1[i] * m[i + 1] + w2[i] * m[i + 2]) / f12;
        } else {
            dk[i] = (m[i + 1] + m[i + 2]) / 2.0;
        }
    }
    dk
}

impl Interpolate1D for AkimaInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        let slopes = akima_slopes(xs, ys, self.makima);
        eval_hermite_spline(xs, ys, &slopes, x, extrapolate)
    }
}

// ---------------------------------------------------------------------------
// N-D successive 1D interpolation engine
// ---------------------------------------------------------------------------

/// Interpolate on a rectilinear grid via successive 1D reduction.
///
/// * `axes`   – one sorted slice per dimension
/// * `values` – flat row-major array of shape `[s0, s1, …, s_{d-1}]`
///              where `s_i = axes[i].len()`
/// * `target` – one target coordinate per dimension
pub fn interpolate_grid(
    axes: &[&[f64]],
    values: &[f64],
    target: &[f64],
    method: &dyn Interpolate1D,
    extrapolate: bool,
) -> f64 {
    let n_dims = axes.len();
    debug_assert_eq!(n_dims, target.len());

    if n_dims == 0 {
        return values[0];
    }

    let mut current = values.to_vec();

    // Reduce from the last dimension to the first.  After each pass the
    // innermost dimension disappears and `current` shrinks accordingly.
    for dim in (0..n_dims).rev() {
        let axis = axes[dim];
        let t = target[dim];
        let n_last = axis.len();
        let n_outer = current.len() / n_last;

        let mut next = Vec::with_capacity(n_outer);
        for i in 0..n_outer {
            let slice = &current[i * n_last..(i + 1) * n_last];
            next.push(method.eval(axis, slice, t, extrapolate));
        }
        current = next;
    }

    debug_assert_eq!(current.len(), 1);
    current[0]
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    // --- 1D interpolation tests ---

    #[test]
    fn linear_basic() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 2.0, 1.0];
        approx_eq(LinearInterp.eval(&xs, &ys, 0.5, false), 1.0, 1e-12);
        approx_eq(LinearInterp.eval(&xs, &ys, 1.5, false), 1.5, 1e-12);
        // clamp
        approx_eq(LinearInterp.eval(&xs, &ys, -1.0, false), 0.0, 1e-12);
        approx_eq(LinearInterp.eval(&xs, &ys, 3.0, false), 1.0, 1e-12);
        // extrapolate
        approx_eq(LinearInterp.eval(&xs, &ys, -1.0, true), -2.0, 1e-12);
    }

    #[test]
    fn nearest_basic() {
        let xs = [0.0, 1.0, 3.0];
        let ys = [10.0, 20.0, 30.0];
        approx_eq(NearestInterp.eval(&xs, &ys, 0.4, false), 10.0, 1e-12);
        approx_eq(NearestInterp.eval(&xs, &ys, 0.6, false), 20.0, 1e-12);
        approx_eq(NearestInterp.eval(&xs, &ys, 2.5, false), 30.0, 1e-12);
    }

    #[test]
    fn cubic_linear_data() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 1.0, 2.0, 3.0];
        for &t in &[0.5, 1.0, 1.5, 2.5] {
            approx_eq(CubicSplineInterp.eval(&xs, &ys, t, false), t, 1e-10);
        }
    }

    #[test]
    fn pchip_monotone() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 2.0, 3.0, 4.0];
        for &t in &[0.5, 1.5, 2.5, 3.5] {
            approx_eq(PchipInterp.eval(&xs, &ys, t, false), t, 1e-10);
        }
    }

    #[test]
    fn akima_linear_data() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 2.0, 4.0, 6.0, 8.0];
        for &t in &[0.5, 1.5, 2.5, 3.5] {
            approx_eq(AKIMA.eval(&xs, &ys, t, false), 2.0 * t, 1e-10);
        }
    }

    #[test]
    fn nd_linear_2d() {
        let ax = [0.0, 1.0, 2.0];
        let ay = [0.0, 1.0];
        let mut vals = Vec::new();
        for &x in &ax {
            for &y in &ay {
                vals.push(x + 2.0 * y);
            }
        }
        let axes: Vec<&[f64]> = vec![&ax, &ay];
        let r = interpolate_grid(&axes, &vals, &[0.5, 0.5], &LINEAR, false);
        approx_eq(r, 0.5 + 1.0, 1e-12);
    }

}
