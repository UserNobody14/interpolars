from __future__ import annotations

from datetime import date, datetime

import polars as pl
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd


def test_interpolation_with_date_coordinate_dim():
    """
    Date is a non-float dtype; it should be accepted as an interpolation coordinate.
    Output should preserve the Date dtype from the target.
    """
    source_df = pl.DataFrame(
        {
            "d": [date(2020, 1, 1), date(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("d").cast(pl.Date))

    target_df = pl.DataFrame({"d": [date(2020, 1, 2)], "label": ["mid"]}).with_columns(
        pl.col("d").cast(pl.Date)
    )

    result = (
        source_df.lazy().select(interpolate_nd(["d"], ["value"], target_df)).collect()
    )

    expected_df = (
        pl.DataFrame({"d": [date(2020, 1, 2)], "label": ["mid"], "value": [1.0]})
        .with_columns(pl.col("d").cast(pl.Date))
        .with_columns(pl.struct([pl.col("d"), pl.col("label"), pl.col("value")]).alias("interpolated"))
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_interpolation_with_duration_coordinate_dim():
    """
    Duration is a non-float dtype; it should be accepted as an interpolation coordinate.
    Output should preserve the Duration dtype from the target.
    """
    # Construct durations as a proper Duration dtype (avoid Object).
    source_df = pl.DataFrame(
        {
            "dt": pl.Series("dt", [0, 10_000], dtype=pl.Duration("ms")),
            "value": [0.0, 10.0],
        }
    )

    target_df = pl.DataFrame(
        {
            "dt": pl.Series("dt", [5_000], dtype=pl.Duration("ms")),
            "label": ["half"],
        }
    )

    result = (
        source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()
    )

    expected_df = (
        pl.DataFrame(
            {
                "dt": pl.Series("dt", [5_000], dtype=pl.Duration("ms")),
                "label": ["half"],
                "value": [5.0],
            }
        )
        .with_columns(pl.struct([pl.col("dt"), pl.col("label"), pl.col("value")]).alias("interpolated"))
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


# ── Precision mismatch tests ──────────────────────────────────────────


def test_datetime_precision_us_source_ns_target():
    """Source Datetime(us) + target Datetime(ns) should reconcile to ns."""
    source_df = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 1), datetime(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("us")))

    target_df = pl.DataFrame({"dt": [datetime(2020, 1, 2)]}).with_columns(
        pl.col("dt").cast(pl.Datetime("ns"))
    )

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame({"dt": [datetime(2020, 1, 2)], "value": [1.0]})
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")))
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_datetime_precision_ns_source_us_target():
    """Source Datetime(ns) + target Datetime(us) should reconcile to ns."""
    source_df = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 1), datetime(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns")))

    target_df = pl.DataFrame({"dt": [datetime(2020, 1, 2)]}).with_columns(
        pl.col("dt").cast(pl.Datetime("us"))
    )

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame({"dt": [datetime(2020, 1, 2)], "value": [1.0]})
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")))
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_datetime_precision_ms_source_ns_target():
    """Source Datetime(ms) + target Datetime(ns): largest gap reconciles to ns."""
    source_df = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 1), datetime(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("ms")))

    target_df = pl.DataFrame({"dt": [datetime(2020, 1, 2)]}).with_columns(
        pl.col("dt").cast(pl.Datetime("ns"))
    )

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame({"dt": [datetime(2020, 1, 2)], "value": [1.0]})
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")))
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_duration_precision_ms_source_us_target():
    """Source Duration(ms) + target Duration(us) should reconcile to us."""
    source_df = pl.DataFrame(
        {
            "dt": pl.Series("dt", [0, 10_000], dtype=pl.Duration("ms")),
            "value": [0.0, 10.0],
        }
    )

    target_df = pl.DataFrame(
        {"dt": pl.Series("dt", [5_000_000], dtype=pl.Duration("us"))}
    )

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame(
            {
                "dt": pl.Series("dt", [5_000_000], dtype=pl.Duration("us")),
                "value": [5.0],
            }
        )
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_duration_precision_ns_source_ms_target():
    """Source Duration(ns) + target Duration(ms) should reconcile to ns."""
    source_df = pl.DataFrame(
        {
            "dt": pl.Series("dt", [0, 10_000_000_000], dtype=pl.Duration("ns")),
            "value": [0.0, 10.0],
        }
    )

    target_df = pl.DataFrame(
        {"dt": pl.Series("dt", [5_000], dtype=pl.Duration("ms"))}
    )

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame(
            {
                "dt": pl.Series("dt", [5_000_000_000], dtype=pl.Duration("ns")),
                "value": [5.0],
            }
        )
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_date_source_datetime_target():
    """Source Date + target Datetime(us) should reconcile to Datetime(us)."""
    source_df = pl.DataFrame(
        {
            "d": [date(2020, 1, 1), date(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("d").cast(pl.Date))

    target_df = pl.DataFrame({"d": [datetime(2020, 1, 2)]}).with_columns(
        pl.col("d").cast(pl.Datetime("us"))
    )

    result = source_df.lazy().select(interpolate_nd(["d"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame({"d": [datetime(2020, 1, 2)], "value": [1.0]})
        .with_columns(pl.col("d").cast(pl.Datetime("us")))
        .with_columns(
            pl.struct([pl.col("d"), pl.col("value")]).alias("interpolated")
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_datetime_precision_mismatch_with_passthrough_columns():
    """Precision reconciliation should not affect passthrough (non-coord) columns."""
    source_df = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 1), datetime(2020, 1, 3)],
            "value": [0.0, 2.0],
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("us")))

    target_df = pl.DataFrame(
        {"dt": [datetime(2020, 1, 2)], "label": ["mid"]}
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns")))

    result = source_df.lazy().select(interpolate_nd(["dt"], ["value"], target_df)).collect()

    expected_df = (
        pl.DataFrame({"dt": [datetime(2020, 1, 2)], "label": ["mid"], "value": [1.0]})
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")))
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("label"), pl.col("value")]).alias(
                "interpolated"
            )
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)


def test_datetime_precision_mismatch_2d_interpolation():
    """2D interpolation where one coord dim has a precision mismatch."""
    source_df = pl.DataFrame(
        {
            "dt": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                datetime(2020, 1, 3),
                datetime(2020, 1, 3),
            ],
            "x": [0.0, 1.0, 0.0, 1.0],
            "value": [0.0, 1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("us")))

    target_df = pl.DataFrame(
        {"dt": [datetime(2020, 1, 2)], "x": [0.5]}
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns")))

    result = source_df.lazy().select(
        interpolate_nd(["dt", "x"], ["value"], target_df)
    ).collect()

    expected_df = (
        pl.DataFrame({"dt": [datetime(2020, 1, 2)], "x": [0.5], "value": [1.5]})
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")))
        .with_columns(
            pl.struct([pl.col("dt"), pl.col("x"), pl.col("value")]).alias(
                "interpolated"
            )
        )
        .select(["interpolated"])
    )
    assert_frame_equal(result, expected_df)

