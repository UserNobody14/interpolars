import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_df(xs: list[float], vals: list[float]) -> pl.DataFrame:
    """Build the expected single-column struct DataFrame that interpolate_nd returns."""
    return (
        pl.DataFrame({"xfield": xs, "valuefield": vals})
        .with_columns(
            pl.struct([pl.col("xfield"), pl.col("valuefield")]).alias("interpolated")
        )
        .select(["interpolated"])
    )


def _run_1d(
    source: pl.DataFrame,
    target: pl.DataFrame,
    **kwargs,
) -> pl.DataFrame:
    return (
        source.lazy()
        .select(interpolate_nd(["xfield"], ["valuefield"], target, **kwargs))
        .collect()
    )


# ---------------------------------------------------------------------------
# Source / target fixtures
# ---------------------------------------------------------------------------


def simple_source():
    """5-point 1D grid: x in {0, 1, 2, 3, 4}, value = 10*x."""
    return pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valuefield": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


def simple_target():
    return pl.DataFrame({"xfield": [0.5, 1.5, 2.5, 3.5]})


# ---------------------------------------------------------------------------
# Error mode (default)
# ---------------------------------------------------------------------------


def test_error_mode_with_clean_data():
    """No NaN/Null -> should work identically to before."""
    result = _run_1d(simple_source(), simple_target(), handle_missing="error")
    expected = _make_result_df([0.5, 1.5, 2.5, 3.5], [5.0, 15.0, 25.0, 35.0])
    assert_frame_equal(result, expected)


def test_error_mode_nan_in_value():
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    with pytest.raises(Exception, match="NaN or Null"):
        _run_1d(source, simple_target(), handle_missing="error")


def test_error_mode_null_in_value():
    source = pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valuefield": [0.0, 10.0, None, 30.0, 40.0],
        }
    )
    with pytest.raises(Exception, match="NaN or Null|null"):
        _run_1d(source, simple_target(), handle_missing="error")


def test_error_mode_nan_in_coord():
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("xfield"))
        .alias("xfield")
    )
    with pytest.raises(Exception, match="NaN or Null"):
        _run_1d(source, simple_target(), handle_missing="error")


# ---------------------------------------------------------------------------
# Drop mode
# ---------------------------------------------------------------------------


def test_drop_nan_in_value():
    """Drop x=2 row (NaN value). Remaining grid: {0,1,3,4}."""
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    target = pl.DataFrame({"xfield": [0.5, 2.0]})
    result = _run_1d(source, target, handle_missing="drop")
    expected = _make_result_df(
        [0.5, 2.0],
        [
            5.0,  # between (0,0) and (1,10)
            10.0 + (2.0 - 1.0) * (30.0 - 10.0) / (3.0 - 1.0),  # between (1,10) and (3,30)
        ],
    )
    assert_frame_equal(result, expected)


def test_drop_null_in_value():
    source = pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valuefield": [0.0, 10.0, None, 30.0, 40.0],
        }
    )
    target = pl.DataFrame({"xfield": [0.5]})
    result = _run_1d(source, target, handle_missing="drop")
    expected = _make_result_df([0.5], [5.0])
    assert_frame_equal(result, expected)


def test_drop_nan_in_coord():
    """Drop a row with NaN coordinate; remaining grid still works."""
    source = pl.DataFrame(
        {
            "xfield": [0.0, float("nan"), 2.0, 3.0, 4.0],
            "valuefield": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    target = pl.DataFrame({"xfield": [1.0]})
    result = _run_1d(source, target, handle_missing="drop")
    # x=1 is between (0,0) and (2,20) -> 10
    expected = _make_result_df([1.0], [10.0])
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Fill mode
# ---------------------------------------------------------------------------


def test_fill_nan_in_value():
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    target = pl.DataFrame({"xfield": [2.0]})
    result = _run_1d(source, target, handle_missing="fill", fill_value=99.0)
    # The NaN at x=2 is replaced with 99.0; querying exactly x=2 gives 99.0.
    expected = _make_result_df([2.0], [99.0])
    assert_frame_equal(result, expected)


def test_fill_null_in_value():
    source = pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valuefield": [0.0, 10.0, None, 30.0, 40.0],
        }
    )
    target = pl.DataFrame({"xfield": [2.0]})
    result = _run_1d(source, target, handle_missing="fill", fill_value=0.0)
    expected = _make_result_df([2.0], [0.0])
    assert_frame_equal(result, expected)


def test_fill_drops_nan_coord():
    """NaN coordinate rows are dropped even in fill mode."""
    source = pl.DataFrame(
        {
            "xfield": [0.0, float("nan"), 2.0],
            "valuefield": [0.0, 10.0, 20.0],
        }
    )
    target = pl.DataFrame({"xfield": [1.0]})
    result = _run_1d(source, target, handle_missing="fill", fill_value=0.0)
    expected = _make_result_df([1.0], [10.0])
    assert_frame_equal(result, expected)


def test_fill_requires_fill_value():
    with pytest.raises(Exception, match="fill_value"):
        _run_1d(simple_source(), simple_target(), handle_missing="fill")


# ---------------------------------------------------------------------------
# Nearest mode
# ---------------------------------------------------------------------------


def test_nearest_nan_in_value():
    """NaN at x=2 is replaced by nearest valid value (x=1 or x=3, both distance 1)."""
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    target = pl.DataFrame({"xfield": [2.0]})
    result = _run_1d(source, target, handle_missing="nearest")
    val = result.unnest("interpolated")["valuefield"][0]
    # Nearest neighbor: x=1 (val=10) or x=3 (val=30), both at distance 1.
    assert val in (10.0, 30.0)


def test_nearest_null_in_value():
    source = pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valuefield": [0.0, None, 20.0, 30.0, 40.0],
        }
    )
    target = pl.DataFrame({"xfield": [1.0]})
    result = _run_1d(source, target, handle_missing="nearest")
    val = result.unnest("interpolated")["valuefield"][0]
    # Nearest valid to x=1: x=0 (dist=1, val=0) or x=2 (dist=1, val=20)
    assert val in (0.0, 20.0)


def test_nearest_edge_nan():
    """NaN at the edge: nearest is the adjacent interior point."""
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 0.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    target = pl.DataFrame({"xfield": [0.0]})
    result = _run_1d(source, target, handle_missing="nearest")
    val = result.unnest("interpolated")["valuefield"][0]
    # Nearest valid to x=0 is x=1, val=10
    assert val == 10.0


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def test_extrapolate_below_grid():
    """Target x=-1 below grid [0..4]: linear extrapolation from (0,0)-(1,10) -> -10."""
    source = simple_source()
    target = pl.DataFrame({"xfield": [-1.0]})
    result = _run_1d(source, target, extrapolate=True)
    expected = _make_result_df([-1.0], [-10.0])
    assert_frame_equal(result, expected)


def test_extrapolate_above_grid():
    """Target x=5 above grid [0..4]: linear extrapolation from (3,30)-(4,40) -> 50."""
    source = simple_source()
    target = pl.DataFrame({"xfield": [5.0]})
    result = _run_1d(source, target, extrapolate=True)
    expected = _make_result_df([5.0], [50.0])
    assert_frame_equal(result, expected)


def test_clamp_without_extrapolate():
    """Without extrapolate, out-of-bounds should clamp to boundary values."""
    source = simple_source()
    target = pl.DataFrame({"xfield": [-1.0, 5.0]})
    result = _run_1d(source, target, extrapolate=False)
    expected = _make_result_df([-1.0, 5.0], [0.0, 40.0])
    assert_frame_equal(result, expected)


def test_extrapolate_interior_unchanged():
    """Interior points should be unchanged by the extrapolate flag."""
    result_no = _run_1d(simple_source(), simple_target(), extrapolate=False)
    result_yes = _run_1d(simple_source(), simple_target(), extrapolate=True)
    assert_frame_equal(result_no, result_yes)


# ---------------------------------------------------------------------------
# 2D extrapolation
# ---------------------------------------------------------------------------


def test_extrapolate_2d():
    """Extrapolate outside a 2D grid."""
    source = pl.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "val": [0.0, 10.0, 10.0, 20.0],
        }
    )
    # Target: x=2, y=0.5 -> extrapolate along x from (0,5)-(1,15) -> 25
    target = pl.DataFrame({"x": [2.0], "y": [0.5]})
    result = (
        source.lazy()
        .select(interpolate_nd(["x", "y"], ["val"], target, extrapolate=True))
        .collect()
    )
    val = result.unnest("interpolated")["val"][0]
    assert math.isclose(val, 25.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Composition: handle_missing + extrapolate
# ---------------------------------------------------------------------------


def test_drop_and_extrapolate():
    """Combine drop with extrapolate."""
    source = simple_source().with_columns(
        pl.when(pl.col("xfield") == 2.0)
        .then(float("nan"))
        .otherwise(pl.col("valuefield"))
        .alias("valuefield")
    )
    target = pl.DataFrame({"xfield": [5.0]})
    result = _run_1d(source, target, handle_missing="drop", extrapolate=True)
    # Grid after drop: {0,1,3,4}. Extrapolate from (3,30)-(4,40) -> 50
    expected = _make_result_df([5.0], [50.0])
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# All-valid data passes through unchanged
# ---------------------------------------------------------------------------


def test_all_modes_clean_data():
    """All modes produce identical results on clean data."""
    target = simple_target()
    expected = _make_result_df([0.5, 1.5, 2.5, 3.5], [5.0, 15.0, 25.0, 35.0])
    for mode in ("error", "drop", "nearest"):
        result = _run_1d(simple_source(), target, handle_missing=mode)
        assert_frame_equal(result, expected, check_exact=False)
    result = _run_1d(simple_source(), target, handle_missing="fill", fill_value=0.0)
    assert_frame_equal(result, expected, check_exact=False)
