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
    interpn,
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
# N-D interpolation: comparison against scipy.interpolate.interpn
# (true tensor-product interpolation via RegularGridInterpolator)
# ---------------------------------------------------------------------------

def _scipy_interpn(axes, values_dict, target_points, method):
    """
    Reference N-D interpolation via scipy.interpolate.interpn, which uses
    RegularGridInterpolator internally (tensor-product, NOT successive 1D).

    `method` is mapped to the scipy interpn method name.
    """
    method_map = {
        "linear": "linear",
        "nearest": "nearest",
        "cubic": "cubic",
        "pchip": "pchip",
    }
    scipy_method = method_map.get(method)
    if scipy_method is None:
        raise ValueError(f"interpn does not support method={method!r}")

    grid_arrays = [np.array(a) for a in axes]
    shape = tuple(len(a) for a in axes)
    values = np.zeros(shape)
    for idx in np.ndindex(shape):
        key = tuple(axes[dim][idx[dim]] for dim in range(len(axes)))
        values[idx] = values_dict[key]

    xi = np.array(target_points)
    return interpn(grid_arrays, values, xi, method=scipy_method,
                   bounds_error=False, fill_value=None).tolist()


def _interp_3d(sx, sy, sz, values, tx, ty, tz, method, extrapolate=False):
    """Run interpolars on a 3D cartesian grid and return interpolated values."""
    rows_x, rows_y, rows_z, rows_v = [], [], [], []
    for xi in sx:
        for yi in sy:
            for zi in sz:
                rows_x.append(xi)
                rows_y.append(yi)
                rows_z.append(zi)
                rows_v.append(values[(xi, yi, zi)])
    source = pl.DataFrame({"x": rows_x, "y": rows_y, "z": rows_z, "v": rows_v})
    target = pl.DataFrame({"x": tx, "y": ty, "z": tz})
    result = (
        source.lazy()
        .select(
            interpolate_nd(
                ["x", "y", "z"], ["v"], target, method=method, extrapolate=extrapolate
            )
        )
        .collect()
    )
    return result.unnest("interpolated")["v"].to_list()


def _build_3d_grid():
    """f(x,y,z) = x^2 + 2*y + 3*z for a nonlinear 3D test volume."""
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [0.0, 1.0, 2.0, 3.0]
    sz = [0.0, 1.0, 2.0]
    values = {}
    for x in sx:
        for y in sy:
            for z in sz:
                values[(x, y, z)] = x * x + 2.0 * y + 3.0 * z
    return sx, sy, sz, values


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_2d_matches_interpn(method):
    """
    Linear and Nearest 2D interpolation uses RegularGridInterpolator and
    should match scipy.interpolate.interpn exactly.
    """
    sx, sy, values = _build_2d_grid()
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    actual = _interp_2d(sx, sy, values, tx, ty, method)
    target_points = list(zip(tx, ty))
    expected = _scipy_interpn([sx, sy], values, target_points, method)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["cubic", "pchip"])
def test_2d_successive_vs_interpn_nonlinear(method):
    """
    For cubic/pchip on a nonlinear surface, successive 1D interpolation may
    differ from scipy's true tensor-product interpn. This test documents
    the expected behavior: our implementation uses successive 1D, so we
    compare against successive 1D scipy, NOT interpn.

    We also compute the interpn result for reference and verify the magnitude
    of the difference is reasonable.
    """
    sx, sy, values = _build_2d_grid()
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]

    actual = _interp_2d(sx, sy, values, tx, ty, method)
    successive = _scipy_2d_successive(sx, sy, values, tx, ty, method)
    target_points = list(zip(tx, ty))
    interpn_result = _scipy_interpn([sx, sy], values, target_points, method)

    # Our code should match successive 1D scipy
    np.testing.assert_allclose(actual, successive, atol=1e-10, rtol=1e-10)

    # Document the difference between successive 1D and tensor-product interpn.
    # For f(x,y) = x^2 + 2y, the nonlinearity is only in x, so the difference
    # should be small but potentially nonzero.
    diff = np.abs(np.array(actual) - np.array(interpn_result))
    assert np.max(diff) < 1.0, (
        f"Successive 1D vs interpn diff too large for {method}: max={np.max(diff):.6f}"
    )


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_2d_interpn_exact_grid_hits(method):
    """N-D interpn-compatible methods return exact values at grid points."""
    sx, sy, values = _build_2d_grid()
    tx = [0.0, 1.0, 2.0, 3.0, 4.0]
    ty = [0.0, 1.0, 2.0, 3.0, 0.0]
    actual = _interp_2d(sx, sy, values, tx, ty, method)
    target_points = list(zip(tx, ty))
    expected = _scipy_interpn([sx, sy], values, target_points, method)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3D tests: verify multi-dimensional interpolation across 3 axes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_3d_matches_interpn(method):
    """
    3D interpolation for linear/nearest should match scipy interpn
    (both use RegularGridInterpolator / true tensor-product).
    """
    sx, sy, sz, values = _build_3d_grid()
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    tz = [0.5, 0.5, 1.5, 1.0]

    actual = _interp_3d(sx, sy, sz, values, tx, ty, tz, method)
    target_points = list(zip(tx, ty, tz))
    expected = _scipy_interpn([sx, sy, sz], values, target_points, method)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"])
def test_3d_exact_grid_hits(method):
    """3D interpolation at grid points returns exact values for all methods."""
    sx, sy, sz, values = _build_3d_grid()
    tx = [0.0, 1.0, 2.0, 3.0, 4.0]
    ty = [0.0, 1.0, 2.0, 3.0, 0.0]
    tz = [0.0, 1.0, 2.0, 0.0, 1.0]

    actual = _interp_3d(sx, sy, sz, values, tx, ty, tz, method)
    expected = [values[(x, y, z)] for x, y, z in zip(tx, ty, tz)]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "akima", "makima"])
def test_3d_trilinear_surface(method):
    """All methods reproduce a trilinear surface f(x,y,z) = 2x + 3y + 5z + 1 exactly."""
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [0.0, 1.0, 2.0, 3.0]
    sz = [0.0, 1.0, 2.0]
    values = {(x, y, z): 2.0 * x + 3.0 * y + 5.0 * z + 1.0
              for x in sx for y in sy for z in sz}

    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    tz = [0.5, 0.5, 1.5, 1.0]

    actual = _interp_3d(sx, sy, sz, values, tx, ty, tz, method)
    expected = [2.0 * x + 3.0 * y + 5.0 * z + 1.0 for x, y, z in zip(tx, ty, tz)]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("method", ["cubic", "pchip"])
def test_3d_successive_1d_consistency(method):
    """
    For cubic/pchip on 3D data, our implementation uses successive 1D.
    Verify it matches the equivalent successive 1D scipy computation.
    """
    sx, sy, sz, values = _build_3d_grid()
    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 0.5, 2.5]
    tz = [0.5, 0.5, 1.5, 1.0]

    actual = _interp_3d(sx, sy, sz, values, tx, ty, tz, method)

    # Build reference via successive 1D scipy calls (z -> y -> x)
    expected = []
    for target_x, target_y, target_z in zip(tx, ty, tz):
        # First reduce along z for each (x, y) pair
        after_z = {}
        for x in sx:
            for y in sy:
                zs_at_xy = [values[(x, y, z)] for z in sz]
                after_z[(x, y)] = _scipy_1d(sz, zs_at_xy, [target_z], method)[0]

        # Then reduce along y for each x
        after_y = []
        for x in sx:
            ys_at_x = [after_z[(x, y)] for y in sy]
            after_y.append(_scipy_1d(sy, ys_at_x, [target_y], method)[0])

        # Finally reduce along x
        result = _scipy_1d(sx, after_y, [target_x], method)[0]
        expected.append(result)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["cubic", "pchip"])
def test_2d_nonlinear_cross_term_shows_difference(method):
    """
    For a surface with cross-terms f(x,y) = x*y, successive 1D and
    tensor-product interpn may differ for cubic/pchip. This test
    characterizes the difference.
    """
    sx = [0.0, 1.0, 2.0, 3.0, 4.0]
    sy = [0.0, 1.0, 2.0, 3.0, 4.0]
    values = {(x, y): x * y for x in sx for y in sy}

    tx = [0.5, 1.5, 2.5, 3.5]
    ty = [0.5, 1.5, 2.5, 3.5]

    actual = _interp_2d(sx, sy, values, tx, ty, method)

    # True values of the function
    true_vals = [x * y for x, y in zip(tx, ty)]

    # Successive 1D scipy
    successive = _scipy_2d_successive(sx, sy, values, tx, ty, method)

    # Tensor-product interpn
    target_points = list(zip(tx, ty))
    interpn_result = _scipy_interpn([sx, sy], values, target_points, method)

    # Our results should match successive 1D
    np.testing.assert_allclose(actual, successive, atol=1e-10, rtol=1e-10)

    # Both should be reasonable approximations of the true function
    np.testing.assert_allclose(actual, true_vals, atol=0.5)
    np.testing.assert_allclose(interpn_result, true_vals, atol=0.5)


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
