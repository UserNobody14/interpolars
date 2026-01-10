import polars as pl
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd


def get_target_df():
    """
    Get a target DataFrame for 1D interpolation.
    Everything is in a single field struct called "x" with a single field "xfield"
    Everything on the source df is between -1 and 10
    """
    return pl.DataFrame(
        {
            "xfield": [0.1, 1, 2.5, 3.5, 4.5, 5.5, 8, 8.5, 8.75, 10],
        }
        # Make x column a struct with a single field "xfield"
    ).with_columns(pl.struct([pl.col("xfield")]).alias("x"))


def get_source_df():
    """
    Get a source DataFrame for 1D interpolation.
    Source df has two field:
    - x: a struct with a single field "xfield"
    - value: a struct with a single field "valuefield"
    Everything on the source df's xfield is between -1 and 10
    """
    return pl.DataFrame(
        {
            "xfield": [0.0, 7, 8, 9, 10],
            "valuefield": [100.0, 200, 300, 400, 500],
        }
        # Make x column a struct with a single field "xfield"
    ).with_columns(
        **{
            "x": pl.struct([pl.col("xfield")]),
            "value": pl.struct([pl.col("valuefield")]),
        }
    )


def test_1d_interpolation():
    """
    Test 1D interpolation.
    """
    target_df = get_target_df()
    source_df = get_source_df()
    interpolated_df = (
        source_df.lazy()
        .select(
            interpolate_nd(pl.col("x"), pl.col("value"), target_df).alias(
                "interpolated"
            )
        )
        .collect()
    )
    result = target_df.hstack(interpolated_df).select(["xfield", "x", "interpolated"])
    xs = [0.1, 1, 2.5, 3.5, 4.5, 5.5, 8, 8.5, 8.75, 10]
    expected_values = [
        # Between (0, 100) and (7, 200)
        100 + (0.1 - 0) * (200 - 100) / (7 - 0),
        100 + (1 - 0) * (200 - 100) / (7 - 0),
        100 + (2.5 - 0) * (200 - 100) / (7 - 0),
        100 + (3.5 - 0) * (200 - 100) / (7 - 0),
        100 + (4.5 - 0) * (200 - 100) / (7 - 0),
        100 + (5.5 - 0) * (200 - 100) / (7 - 0),
        # Between (8, 300) and (9, 400)
        300,  # Precisely 300 (on 8)
        300 + (8.5 - 8) * (400 - 300) / (9 - 8),
        300 + (8.75 - 8) * (400 - 300) / (9 - 8),
        # Precisely on 10
        500,
    ]

    expected_df = (
        pl.DataFrame({"xfield": xs, "interpolated": expected_values})
        .with_columns(pl.struct([pl.col("xfield")]).alias("x"))
        .with_columns(
            pl.struct([pl.col("interpolated").alias("valuefield")]).alias(
                "interpolated"
            )
        )
        .select(["xfield", "x", "interpolated"])
    )
    assert_frame_equal(result, expected_df)
