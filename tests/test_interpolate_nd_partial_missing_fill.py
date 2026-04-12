"""interpolars.interpolate_nd behaviour when the source field has partial NaN/Null.

The public API uses ``handle_missing`` (not a boolean) and ``fill_value``. After filling
missing values (including with NaN), the grid must still be recognised as a full Cartesian
product so linear tensor-product interpolation can run; results match
``scipy.interpolate.interpn`` on the same rectilinear grid (including NaN propagation).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.interpolate import interpn

from interpolars import interpolate_nd


def _grid_3x3_partial_missing() -> pl.DataFrame:
    """3×3 (y, x) grid with NaNs at (y=0, x=2) and (y=1, x=1)."""
    return pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "value": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0],
        }
    )


def _run_interp(
    source: pl.DataFrame,
    target: pl.DataFrame,
    *,
    handle_missing: str = "error",
    fill_value: float | None = None,
) -> pl.DataFrame:
    kwargs: dict[str, object] = {"handle_missing": handle_missing}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    out = (
        source.lazy()
        .select(interpolate_nd(["y", "x"], ["value"], target, **kwargs))
        .collect()
    )
    return out.unnest("interpolated")


def _scipy_linear_after_fill(
    y: np.ndarray,
    x: np.ndarray,
    values: np.ndarray,
    fill: float,
    yq: float,
    xq: float,
) -> float:
    filled = values.copy()
    filled[np.isnan(filled)] = fill
    return float(interpn((y, x), filled, np.array([[yq, xq]]), method="linear")[0])


def _assert_close_or_both_nan(got: float, want: float) -> None:
    if math.isnan(want):
        assert math.isnan(got)
    else:
        assert float(got) == pytest.approx(want, abs=1e-10)


@pytest.mark.parametrize("fill", [0.0, -999.0, np.pi, float("nan")])
def test_interpolate_nd_fill_matches_scipy_after_nan_replace(fill: float) -> None:
    """Linear interpolation matches scipy on the grid after replacing NaNs with ``fill``."""
    source = _grid_3x3_partial_missing()
    yq, xq = 0.5, 0.5
    target = pl.DataFrame({"y": [yq], "x": [xq]})
    got = _run_interp(source, target, handle_missing="fill", fill_value=fill)["value"][0]

    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    vals = np.array(
        [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    want = _scipy_linear_after_fill(y, x, vals, fill, yq, xq)
    _assert_close_or_both_nan(float(got), want)


def test_interpolate_nd_fill_nan_ok_when_no_source_missing() -> None:
    """If there is nothing to fill, fill_value=NaN is a no-op and interpolation works."""
    source = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    out = _run_interp(
        source,
        target,
        handle_missing="fill",
        fill_value=float("nan"),
    )
    assert float(out["value"][0]) == pytest.approx(3.0, abs=1e-10)


def test_interpolate_nd_drop_partial_missing_raises_compute_error() -> None:
    """Dropping rows with NaNs breaks the full grid; the plugin rejects it."""
    source = _grid_3x3_partial_missing()
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    with pytest.raises(pl.exceptions.ComputeError, match="source grid missing points"):
        _run_interp(source, target, handle_missing="drop")


def test_interpolate_nd_error_on_any_missing() -> None:
    source = _grid_3x3_partial_missing()
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    with pytest.raises(pl.exceptions.ComputeError):
        _run_interp(source, target, handle_missing="error")


def test_interpolate_nd_nearest_partial_missing_returns_finite() -> None:
    """Nearest mode should tolerate partial NaNs and return a finite scalar."""
    source = _grid_3x3_partial_missing()
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    out = _run_interp(source, target, handle_missing="nearest")
    v = float(out["value"][0])
    assert np.isfinite(v)


def test_interpolate_nd_fill_nan_multi_column_partial_missing_matches_scipy() -> None:
    """fill_value=NaN with holes in one column: compare each column to scipy."""
    source = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "a": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0],
            "b": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        }
    )
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    out = (
        source.lazy()
        .select(
            interpolate_nd(
                ["y", "x"],
                ["a", "b"],
                target,
                handle_missing="fill",
                fill_value=float("nan"),
            )
        )
        .collect()
        .unnest("interpolated")
    )

    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    a = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    b = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    want_a = _scipy_linear_after_fill(y, x, a, float("nan"), 0.5, 0.5)
    want_b = _scipy_linear_after_fill(y, x, b, float("nan"), 0.5, 0.5)
    _assert_close_or_both_nan(float(out["a"][0]), want_a)
    _assert_close_or_both_nan(float(out["b"][0]), want_b)


def test_interpolate_nd_fill_zero_multi_column_partial_missing_matches_scipy() -> None:
    """Numeric fill still works for the affected column; the other is unchanged."""
    source = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "a": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0],
            "b": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        }
    )
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    out = (
        source.lazy()
        .select(
            interpolate_nd(
                ["y", "x"],
                ["a", "b"],
                target,
                handle_missing="fill",
                fill_value=0.0,
            )
        )
        .collect()
        .unnest("interpolated")
    )

    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    a = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    b = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    want_a = _scipy_linear_after_fill(y, x, a, 0.0, 0.5, 0.5)
    want_b = _scipy_linear_after_fill(y, x, b, 0.0, 0.5, 0.5)
    assert float(out["a"][0]) == pytest.approx(want_a, abs=1e-10)
    assert float(out["b"][0]) == pytest.approx(want_b, abs=1e-10)


def test_interpolate_nd_null_source_fill_nan_matches_scipy() -> None:
    """Polars Null in the value column matches NaN source after fill-then-interpolate."""
    source = _grid_3x3_partial_missing().with_columns(
        pl.when(pl.col("value").is_nan())
        .then(None)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    got = _run_interp(
        source,
        target,
        handle_missing="fill",
        fill_value=float("nan"),
    )["value"][0]

    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    vals = np.array(
        [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    want = _scipy_linear_after_fill(y, x, vals, float("nan"), 0.5, 0.5)
    _assert_close_or_both_nan(float(got), want)


def test_interpolate_nd_incomplete_grid_still_raises() -> None:
    """A genuinely missing (y, x) pair is still reported (not confused with NaN values)."""
    # Full 3×3 coordinates but one row removed so the product is incomplete.
    source = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1, 2, 2],  # missing (2, 2)
            "x": [0, 1, 2, 0, 1, 2, 0, 1],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})
    with pytest.raises(pl.exceptions.ComputeError, match="source grid missing points"):
        _run_interp(source, target, handle_missing="fill", fill_value=0.0)
