"""Tests for all 6 interpolation methods, validated against scipy equivalents."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.interpolate import (
    Akima1DInterpolator,
    PchipInterpolator,
    RegularGridInterpolator,
    interp1d,
)

from interpolars import interpolate_nd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interp_1d(source_x, source_y, target_x, method, extrapolate=False):
    """Run interpolars on 1D data and return the interpolated values as a list."""
    source = pl.DataFrame({"x": source_x, "v": source_y})
    target = pl.DataFrame({"x": target_x})
    result = (
        source.lazy()
        .select(
            interpolate_nd(
                ["x"], ["v"], target, method=method, extrapolate=extrapolate
            )
        )
        .collect()
    )
    return result.unnest("interpolated")["v"].to_list()


def _interp_2d(sx, sy, values, tx, ty, method, extrapolate=False):
    """Run interpolars on a 2D cartesian grid and return interpolated values."""
    rows_x, rows_y, rows_v = [], [], []
    for xi in sx:
        for yi in sy:
            rows_x.append(xi)
            rows_y.append(yi)
            rows_v.append(values[(xi, yi)])
    source = pl.DataFrame({"x": rows_x, "y": rows_y, "v": rows_v})
    target = pl.DataFrame({"x": tx, "y": ty})
    result = (
        source.lazy()
        .select(
            interpolate_nd(
                ["x", "y"], ["v"], target, method=method, extrapolate=extrapolate
            )
        )
        .collect()
    )
    return result.unnest("interpolated")["v"].to_list()


def _scipy_1d(sx, sy, tx, method, extrapolate=False):
    """Reference 1D interpolation via scipy."""
    sx = np.array(sx, dtype=float)
    sy = np.array(sy, dtype=float)
    tx = np.array(tx, dtype=float)

    if method == "linear":
        if extrapolate:
            f = interp1d(sx, sy, kind="linear", fill_value="extrapolate")
        else:
            f = interp1d(sx, sy, kind="linear", bounds_error=False, fill_value=(sy[0], sy[-1]))
        return f(tx).tolist()
    if method == "nearest":
        f = interp1d(sx, sy, kind="nearest", bounds_error=False, fill_value=(sy[0], sy[-1]))
        return f(tx).tolist()
    if method == "cubic":
        if len(sx) < 4:
            if len(sx) == 3:
                f = interp1d(sx, sy, kind="quadratic", bounds_error=False,
                             fill_value=(sy[0], sy[-1]))
            else:
                f = interp1d(sx, sy, kind="linear", bounds_error=False,
                             fill_value=(sy[0], sy[-1]))
            return f(tx).tolist()
        if extrapolate:
            f = interp1d(sx, sy, kind="cubic", fill_value="extrapolate")
        else:
            f = interp1d(sx, sy, kind="cubic", bounds_error=False, fill_value=(sy[0], sy[-1]))
        return f(tx).tolist()
    if method == "pchip":
        f = PchipInterpolator(sx, sy, extrapolate=extrapolate)
        out = f(tx)
        if not extrapolate:
            out = np.where(tx < sx[0], sy[0], out)
            out = np.where(tx > sx[-1], sy[-1], out)
        return out.tolist()
    if method in ("akima", "makima"):
        use_makima = method == "makima"
        if len(sx) < 2:
            return [sy[0]] * len(tx)
        f = Akima1DInterpolator(sx, sy, method="makima" if use_makima else "akima")
        out = f(tx, extrapolate=extrapolate)
        if not extrapolate:
            out = np.where(tx < sx[0], sy[0], out)
            out = np.where(tx > sx[-1], sy[-1], out)
        return out.tolist()

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# 1D data fixtures
# ---------------------------------------------------------------------------

SRC_X = [0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]
SRC_Y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]
TGT_X = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 4.5, 6.0, 7.5, 8.0, 9.0, 10.0]

SRC_X_SHORT = [0.0, 1.0, 2.0, 3.0]
SRC_Y_SHORT = [0.0, 1.0, 0.0, 1.0]
TGT_X_SHORT = [0.5, 1.5, 2.5]


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_1d_interior(method):
    """Interior-point interpolation matches scipy for all methods."""
    actual = _interp_1d(SRC_X, SRC_Y, TGT_X, method)
    expected = _scipy_1d(SRC_X, SRC_Y, TGT_X, method)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_1d_exact_hits(method):
    """Interpolation at grid points returns exact source values."""
    actual = _interp_1d(SRC_X, SRC_Y, SRC_X, method)
    np.testing.assert_allclose(actual, SRC_Y, atol=1e-12)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "akima", "makima"])
def test_1d_clamp(method):
    """Out-of-range points clamp to boundary values when extrapolate=False."""
    target_x = [-1.0, 11.0]
    actual = _interp_1d(SRC_X, SRC_Y, target_x, method, extrapolate=False)
    assert actual[0] == pytest.approx(SRC_Y[0], abs=1e-12)
    assert actual[1] == pytest.approx(SRC_Y[-1], abs=1e-12)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "akima", "makima"])
def test_1d_extrapolate(method):
    """Extrapolation matches scipy."""
    target_x = [-1.0, 11.0]
    actual = _interp_1d(SRC_X, SRC_Y, target_x, method, extrapolate=True)
    expected = _scipy_1d(SRC_X, SRC_Y, target_x, method, extrapolate=True)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "akima", "makima"])
def test_1d_linear_data_reproduces_line(method):
    """Continuous methods should reproduce a line exactly on co-linear data."""
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [2.0, 4.0, 6.0, 8.0, 10.0]
    tx = [0.5, 1.5, 2.5, 3.5]
    actual = _interp_1d(sx, sy, tx, method)
    expected = [3.0, 5.0, 7.0, 9.0]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------

def _build_2d_grid():
    """f(x,y) = x^2 + 2*y for a simple nonlinear test surface."""
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [0.0, 1.0, 2.0, 3.0]
    values = {}
    for x in sx:
        for y in sy:
            values[(x, y)] = x * x + 2.0 * y
    return sx, sy, values


def _scipy_2d_successive(sx, sy, values, tx, ty, method):
    """
    Reference 2D interpolation via successive 1D scipy calls.
    For each target point, interpolate along y for each source x,
    then interpolate those results along x.
    """
    results = []
    for target_x, target_y in zip(tx, ty):
        intermediate = []
        for x in sx:
            ys_at_x = [values[(x, y)] for y in sy]
            val = _scipy_1d(sy, ys_at_x, [target_y], method)[0]
            intermediate.append(val)
        result = _scipy_1d(sx, intermediate, [target_x], method)[0]
        results.append(result)
    return results


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_2d_interior(method):
    """2D interior interpolation matches successive scipy 1D calls."""
    sx, sy, values = _build_2d_grid()
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    actual = _interp_2d(sx, sy, values, tx, ty, method)
    expected = _scipy_2d_successive(sx, sy, values, tx, ty, method)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_2d_exact_grid_hits(method):
    """2D interpolation at grid points returns exact values."""
    sx, sy, values = _build_2d_grid()
    tx = [0.0, 1.0, 2.0, 3.0, 4.0]
    ty = [0.0, 1.0, 2.0, 3.0, 0.0]
    actual = _interp_2d(sx, sy, values, tx, ty, method)
    expected = [values[(x, y)] for x, y in zip(tx, ty)]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "akima", "makima"])
def test_2d_bilinear_surface(method):
    """All methods reproduce a bilinear surface f(x,y) = 3x + 5y + 1 exactly."""
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [0.0, 1.0, 2.0, 3.0]
    values = {(x, y): 3.0 * x + 5.0 * y + 1.0 for x in sx for y in sy}
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    actual = _interp_2d(sx, sy, values, tx, ty, method)
    expected = [3.0 * x + 5.0 * y + 1.0 for x, y in zip(tx, ty)]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_point_axis():
    """1D interpolation with a single source point returns that value."""
    for method in ["linear", "nearest", "cubic", "pchip", "akima", "makima"]:
        actual = _interp_1d([5.0], [42.0], [3.0, 5.0, 7.0], method)
        assert actual == [pytest.approx(42.0)] * 3


def test_two_point_axis():
    """1D interpolation with exactly 2 source points (all methods degenerate to linear)."""
    for method in ["linear", "nearest", "cubic", "pchip", "akima", "makima"]:
        actual = _interp_1d([0.0, 1.0], [0.0, 10.0], [0.5], method)
        if method == "nearest":
            assert actual[0] == pytest.approx(0.0, abs=1e-12) or actual[0] == pytest.approx(10.0, abs=1e-12)
        else:
            assert actual[0] == pytest.approx(5.0, abs=1e-10)


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_method_kwarg_passthrough(method):
    """Smoke test: method kwarg reaches the Rust plugin without errors."""
    source = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "v": [0.0, 1.0, 4.0, 9.0, 16.0]})
    target = pl.DataFrame({"x": [0.5, 1.5, 2.5, 3.5]})
    result = (
        source.lazy()
        .select(interpolate_nd(["x"], ["v"], target, method=method))
        .collect()
    )
    vals = result.unnest("interpolated")["v"].to_list()
    assert len(vals) == 4
    assert all(isinstance(v, float) for v in vals)


# ---------------------------------------------------------------------------
# Successive 1D vs tensor product (RegularGridInterpolator) comparison
# ---------------------------------------------------------------------------
#
# interpolars decomposes N-D interpolation into successive 1D passes.
#
# For methods where 1D interpolation is a LINEAR operator on the data values
# (linear, cubic splines), this is mathematically equivalent to the tensor
# product formulation.  The proof: each 1D pass computes result = w^T @ data
# where weights w depend only on the knot positions and target coordinate
# (not the data values).  Matrix-multiplication associativity guarantees that
# the order of axis reduction doesn't matter.
#
# For methods whose slope computation is NONLINEAR in the data values (PCHIP,
# Akima, Makima -- which use harmonic means, absolute differences, and
# sign-dependent clamping), the successive 1D decomposition is NOT equivalent
# to any unique tensor product: the result depends on the order in which axes
# are reduced.

def _build_nonseparable_surface():
    """f(x,y) = sin(x)*cos(y) + 0.5*x*y on a non-uniform rectilinear grid."""
    sx = np.array([0.0, 0.3, 1.0, 1.5, 2.2, 3.0, 3.5])
    sy = np.array([0.0, 0.4, 1.0, 1.8, 2.5, 3.0])
    xx, yy = np.meshgrid(sx, sy, indexing="ij")
    values = np.sin(xx) * np.cos(yy) + 0.5 * xx * yy
    return sx, sy, values


def _build_peaked_surface():
    """Surface with a sharp central peak that stresses nonlinear slope methods.

    f(x,y) = 1 / (1 + 5*((x-1.5)^2 + (y-1.5)^2)) + 0.3*x*sin(2y)
    """
    sx = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    sy = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    xx, yy = np.meshgrid(sx, sy, indexing="ij")
    values = 1.0 / (1.0 + 5 * ((xx - 1.5) ** 2 + (yy - 1.5) ** 2)) + 0.3 * xx * np.sin(2 * yy)
    return sx, sy, values


_TP_TX = np.array([0.15, 0.65, 1.25, 1.85, 2.6, 3.25])
_TP_TY = np.array([0.2, 0.7, 1.4, 2.15, 2.75, 0.5])

_PEAK_TX = np.array([0.3, 0.7, 1.2, 1.8, 2.3, 2.7])
_PEAK_TY = np.array([0.2, 0.8, 1.3, 1.7, 2.2, 2.8])


def _interpolars_2d_grid(sx, sy, values_grid, tx, ty, method, axes_order=("x", "y")):
    """Run interpolars on a 2D grid defined by numpy arrays.

    axes_order controls the axis reduction order passed to interpolate_nd.
    ("x","y") reduces y (last) first then x; ("y","x") reduces x first then y.
    """
    rows_x, rows_y, rows_v = [], [], []
    for i, x in enumerate(sx):
        for j, y in enumerate(sy):
            rows_x.append(float(x))
            rows_y.append(float(y))
            rows_v.append(float(values_grid[i, j]))
    source = pl.DataFrame({"x": rows_x, "y": rows_y, "v": rows_v})
    target = pl.DataFrame({"x": tx.tolist(), "y": ty.tolist()})
    result = (
        source.lazy()
        .select(interpolate_nd(list(axes_order), ["v"], target, method=method))
        .collect()
    )
    return np.array(result.unnest("interpolated")["v"].to_list())


def _rgi_eval(sx, sy, values_grid, tx, ty, method, **kwargs):
    """Evaluate via scipy RegularGridInterpolator (tensor product)."""
    rgi = RegularGridInterpolator(
        (sx, sy), values_grid, method=method, bounds_error=False, **kwargs,
    )
    return rgi(np.column_stack([tx, ty]))


class TestSuccessive1DVsTensorProduct:
    """Compare interpolars' successive-1D approach against scipy's tensor
    product (RegularGridInterpolator) on non-separable 2D surfaces.

    Key findings documented by these tests:

    1. Linear & cubic: successive 1D IS equivalent to tensor product
       (both are linear operators on the data values).
    2. The default RGI "cubic" appears to differ by ~1e-5, but this is
       entirely due to its iterative sparse solver (gcrotmk, atol=1e-6);
       a direct solver yields machine-epsilon agreement.
    3. RGI's "pchip" method uses successive 1D internally, so it matches.
    4. For PCHIP / Akima / Makima, the successive 1D result is
       axis-order-dependent because the slope computation is nonlinear.
    """

    # ------------------------------------------------------------------
    # Linear & cubic: successive 1D == tensor product
    # ------------------------------------------------------------------

    def test_linear_matches_rgi(self):
        """Successive 1D linear == tensor product linear (bilinear)."""
        sx, sy, vals = _build_nonseparable_surface()
        actual = _interpolars_2d_grid(sx, sy, vals, _TP_TX, _TP_TY, "linear")
        expected = _rgi_eval(sx, sy, vals, _TP_TX, _TP_TY, "linear")
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_cubic_matches_rgi_direct_solver(self):
        """Successive 1D cubic == tensor product cubic (direct solver).

        With a direct sparse solver the Kronecker-product system is solved
        exactly, confirming that successive 1D and tensor product cubic are
        mathematically identical (both not-a-knot, both linear in the data).
        """
        from scipy.sparse.linalg import spsolve

        sx, sy, vals = _build_nonseparable_surface()
        actual = _interpolars_2d_grid(sx, sy, vals, _TP_TX, _TP_TY, "cubic")
        expected = _rgi_eval(sx, sy, vals, _TP_TX, _TP_TY, "cubic", solver=spsolve)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_cubic_rgi_default_solver_noise(self):
        """Default RGI cubic uses gcrotmk with atol=1e-6, creating ~1e-5
        discrepancy that is solver noise, not a mathematical difference."""
        from scipy.sparse.linalg import spsolve

        sx, sy, vals = _build_nonseparable_surface()
        r_iterative = _rgi_eval(sx, sy, vals, _TP_TX, _TP_TY, "cubic")
        r_direct = _rgi_eval(sx, sy, vals, _TP_TX, _TP_TY, "cubic", solver=spsolve)
        r_successive = _interpolars_2d_grid(sx, sy, vals, _TP_TX, _TP_TY, "cubic")

        iterative_vs_successive = np.max(np.abs(r_iterative - r_successive))
        direct_vs_successive = np.max(np.abs(r_direct - r_successive))

        assert iterative_vs_successive > 1e-6, (
            "Expected iterative solver to introduce >1e-6 noise"
        )
        assert direct_vs_successive < 1e-12, (
            f"Direct solver should match successive 1D to machine eps, "
            f"got {direct_vs_successive:.2e}"
        )

    # ------------------------------------------------------------------
    # PCHIP: RGI also uses successive 1D, so they agree
    # ------------------------------------------------------------------

    def test_pchip_matches_rgi(self):
        """PCHIP matches RGI because scipy's RegularGridInterpolator also
        uses successive 1D evaluation for method='pchip' (via
        _evaluate_spline / _do_pchip), not a tensor product NdBSpline."""
        sx, sy, vals = _build_nonseparable_surface()
        actual = _interpolars_2d_grid(sx, sy, vals, _TP_TX, _TP_TY, "pchip")
        expected = _rgi_eval(sx, sy, vals, _TP_TX, _TP_TY, "pchip")
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    # ------------------------------------------------------------------
    # Axis-order dependence: linear operator methods are invariant,
    # nonlinear slope methods are not
    # ------------------------------------------------------------------

    def test_linear_axis_order_invariant(self):
        """Linear interpolation result is independent of axis reduction order."""
        sx, sy, vals = _build_peaked_surface()
        r_xy = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "linear", ("x", "y"))
        r_yx = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "linear", ("y", "x"))
        np.testing.assert_allclose(r_xy, r_yx, atol=1e-10)

    def test_cubic_axis_order_invariant(self):
        """Cubic spline result is independent of axis reduction order."""
        sx, sy, vals = _build_peaked_surface()
        r_xy = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "cubic", ("x", "y"))
        r_yx = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "cubic", ("y", "x"))
        np.testing.assert_allclose(r_xy, r_yx, atol=1e-10)

    def test_pchip_axis_order_dependent(self):
        """Successive 1D PCHIP gives different results depending on which
        axis is reduced first, because slope computation (weighted harmonic
        mean with monotonicity constraints) is nonlinear in the data."""
        sx, sy, vals = _build_peaked_surface()
        r_xy = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "pchip", ("x", "y"))
        r_yx = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "pchip", ("y", "x"))
        max_diff = np.max(np.abs(r_xy - r_yx))
        assert max_diff > 1e-3, (
            f"Expected axis-order dependence for PCHIP on peaked surface, "
            f"but max diff was only {max_diff:.2e}"
        )

    def test_akima_axis_order_dependent(self):
        """Successive 1D Akima gives different results depending on axis
        reduction order, because slope weights use |m_{i+1} - m_i| which
        is nonlinear in the data."""
        sx, sy, vals = _build_peaked_surface()
        r_xy = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "akima", ("x", "y"))
        r_yx = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "akima", ("y", "x"))
        max_diff = np.max(np.abs(r_xy - r_yx))
        assert max_diff > 1e-3, (
            f"Expected axis-order dependence for Akima on peaked surface, "
            f"but max diff was only {max_diff:.2e}"
        )

    def test_makima_axis_order_dependent(self):
        """Successive 1D Makima gives different results depending on axis
        reduction order (same nonlinearity as Akima)."""
        sx, sy, vals = _build_peaked_surface()
        r_xy = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "makima", ("x", "y"))
        r_yx = _interpolars_2d_grid(sx, sy, vals, _PEAK_TX, _PEAK_TY, "makima", ("y", "x"))
        max_diff = np.max(np.abs(r_xy - r_yx))
        assert max_diff > 1e-3, (
            f"Expected axis-order dependence for Makima on peaked surface, "
            f"but max diff was only {max_diff:.2e}"
        )
