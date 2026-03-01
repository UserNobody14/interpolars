//! Pure interpolation math -- no Polars dependency.
//!
//! Delegates to `scirs2-interpolate`:
//! - 1D: Linear, Nearest, Cubic, Pchip, Makima via `Interp1d`;
//!        Akima via `AkimaSpline`
//! - N-D: Linear, Nearest via `RegularGridInterpolator`
//! - N-D successive 1D fallback for Cubic/Pchip/Akima/Makima

use ndarray::{Array, Array2, ArrayView1, IxDyn};
use scirs2_interpolate::advanced::akima::AkimaSpline;
use scirs2_interpolate::interp1d::{
    ExtrapolateMode, Interp1d, InterpolationMethod as Scirs2Method1D,
};
use scirs2_interpolate::interpnd::{
    ExtrapolateMode as NdExtrapolateMode, InterpolationMethod as Scirs2MethodNd,
    RegularGridInterpolator,
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Method enum
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 1D trait (still needed for the N-D fallback successive-1D engine)
// ---------------------------------------------------------------------------

pub trait Interpolate1D: Sync + Send {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64;
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

fn extrapolate_mode(extrapolate: bool) -> ExtrapolateMode {
    if extrapolate {
        ExtrapolateMode::Extrapolate
    } else {
        ExtrapolateMode::Nearest
    }
}

/// Delegate a 1D evaluation to `scirs2_interpolate::interp1d::Interp1d`.
fn eval_via_interp1d(
    xs: &[f64],
    ys: &[f64],
    x: f64,
    extrapolate: bool,
    method: Scirs2Method1D,
) -> f64 {
    let n = xs.len();
    debug_assert_eq!(n, ys.len());
    if n == 1 {
        return ys[0];
    }
    let x_view = ArrayView1::from(xs);
    let y_view = ArrayView1::from(ys);
    let mode = extrapolate_mode(extrapolate);
    let interp =
        Interp1d::new(&x_view, &y_view, method, mode).expect("Interp1d construction failed");
    interp.evaluate(x).expect("Interp1d evaluation failed")
}

// ---------------------------------------------------------------------------
// Linear  (delegates to scirs2)
// ---------------------------------------------------------------------------

pub struct LinearInterp;

impl Interpolate1D for LinearInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        eval_via_interp1d(xs, ys, x, extrapolate, Scirs2Method1D::Linear)
    }
}

// ---------------------------------------------------------------------------
// Nearest  (delegates to scirs2)
// ---------------------------------------------------------------------------

pub struct NearestInterp;

impl Interpolate1D for NearestInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        eval_via_interp1d(xs, ys, x, extrapolate, Scirs2Method1D::Nearest)
    }
}

// ---------------------------------------------------------------------------
// Cubic spline  (delegates to scirs2 not-a-knot cubic spline)
// ---------------------------------------------------------------------------

pub struct CubicSplineInterp;

impl Interpolate1D for CubicSplineInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        eval_via_interp1d(xs, ys, x, extrapolate, Scirs2Method1D::Cubic)
    }
}

// ---------------------------------------------------------------------------
// PCHIP  (delegates to scirs2)
// ---------------------------------------------------------------------------

pub struct PchipInterp;

impl Interpolate1D for PchipInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        eval_via_interp1d(xs, ys, x, extrapolate, Scirs2Method1D::Pchip)
    }
}

// ---------------------------------------------------------------------------
// Akima  (delegates to scirs2 AkimaSpline)
// ---------------------------------------------------------------------------

struct AkimaInterp;

impl Interpolate1D for AkimaInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        let n = xs.len();
        debug_assert_eq!(n, ys.len());
        if n == 1 {
            return ys[0];
        }
        let x_view = ArrayView1::from(xs);
        let y_view = ArrayView1::from(ys);
        let mode = extrapolate_mode(extrapolate);
        let spline = AkimaSpline::new(&x_view, &y_view)
            .expect("AkimaSpline construction failed")
            .with_extrapolation(mode);
        spline.evaluate(x).expect("AkimaSpline evaluation failed")
    }
}

// ---------------------------------------------------------------------------
// Makima  (delegates to scirs2 Modified Akima)
// ---------------------------------------------------------------------------

struct MakimaInterp;

impl Interpolate1D for MakimaInterp {
    fn eval(&self, xs: &[f64], ys: &[f64], x: f64, extrapolate: bool) -> f64 {
        eval_via_interp1d(xs, ys, x, extrapolate, Scirs2Method1D::Makima)
    }
}

// ---------------------------------------------------------------------------
// Static interpolator instances (for the N-D fallback path)
// ---------------------------------------------------------------------------

static LINEAR: LinearInterp = LinearInterp;
static NEAREST: NearestInterp = NearestInterp;
static CUBIC: CubicSplineInterp = CubicSplineInterp;
static PCHIP: PchipInterp = PchipInterp;
static AKIMA_1D: AkimaInterp = AkimaInterp;
static MAKIMA: MakimaInterp = MakimaInterp;

impl InterpolationMethod {
    fn as_1d_interpolator(&self) -> &'static dyn Interpolate1D {
        match self {
            Self::Linear => &LINEAR,
            Self::Nearest => &NEAREST,
            Self::Cubic => &CUBIC,
            Self::Pchip => &PCHIP,
            Self::Akima => &AKIMA_1D,
            Self::Makima => &MAKIMA,
        }
    }
}

// ---------------------------------------------------------------------------
// N-D interpolation
// ---------------------------------------------------------------------------

/// N-D interpolation via successive 1D reduction (fallback for methods not
/// supported by `RegularGridInterpolator`).
fn interpolate_grid_successive_1d(
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

/// Interpolate on a rectilinear grid, dispatching to scirs2's
/// `RegularGridInterpolator` for Linear/Nearest and falling back
/// to successive 1D reduction for Cubic/Pchip/Akima/Makima.
pub fn interpolate_grid(
    axes: &[&[f64]],
    values: &[f64],
    target: &[f64],
    method: InterpolationMethod,
    extrapolate: bool,
) -> f64 {
    let n_dims = axes.len();
    debug_assert_eq!(n_dims, target.len());

    if n_dims == 0 {
        return values[0];
    }

    let nd_method = match method {
        InterpolationMethod::Linear => Some(Scirs2MethodNd::Linear),
        InterpolationMethod::Nearest => Some(Scirs2MethodNd::Nearest),
        _ => None,
    };

    if let Some(scirs2_method) = nd_method {
        let shape: Vec<usize> = axes.iter().map(|a| a.len()).collect();
        let points: Vec<ndarray::Array1<f64>> = axes
            .iter()
            .map(|a| ndarray::Array1::from(a.to_vec()))
            .collect();

        let nd_values = Array::from_shape_vec(IxDyn(&shape), values.to_vec())
            .expect("shape mismatch in interpolate_grid");

        let nd_extrap = if extrapolate {
            NdExtrapolateMode::Extrapolate
        } else {
            NdExtrapolateMode::Nearest
        };

        let interp = RegularGridInterpolator::new(points, nd_values, scirs2_method, nd_extrap)
            .expect("RegularGridInterpolator construction failed");

        let xi = Array2::from_shape_vec((1, n_dims), target.to_vec())
            .expect("target shape mismatch");

        let result = interp
            .__call__(&xi.view())
            .expect("RegularGridInterpolator eval failed");
        result[0]
    } else {
        interpolate_grid_successive_1d(
            axes,
            values,
            target,
            method.as_1d_interpolator(),
            extrapolate,
        )
    }
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
            approx_eq(AKIMA_1D.eval(&xs, &ys, t, false), 2.0 * t, 1e-10);
        }
    }

    #[test]
    fn makima_linear_data() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 2.0, 4.0, 6.0, 8.0];
        for &t in &[0.5, 1.5, 2.5, 3.5] {
            approx_eq(MAKIMA.eval(&xs, &ys, t, false), 2.0 * t, 1e-10);
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
        let r = interpolate_grid(
            &axes,
            &vals,
            &[0.5, 0.5],
            InterpolationMethod::Linear,
            false,
        );
        approx_eq(r, 0.5 + 1.0, 1e-12);
    }
}
