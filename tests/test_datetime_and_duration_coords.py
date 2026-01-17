from __future__ import annotations

from datetime import date

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

