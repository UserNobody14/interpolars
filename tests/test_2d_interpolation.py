import polars as pl
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd


def get_target_df():
    """
    Get a target DataFrame for 1D interpolation.
    Everything is in a 2 field struct called "x" with the fields:
    - xfield: a float
    - yfield: a float
    Everything on the source df is between 0 and 2
    """
    return pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 1.25, 1.5],
            "yfield": [1.0, 0.0, 0.5, 2.0],
            # Imaginary labels
            # "labels": ["a", "b", "c", "d"],
        }
        # Make x column a struct with a single field "xfield"
    ).with_columns(pl.struct([pl.col("xfield"), pl.col("yfield")]).alias("x"))


# ----------
# 2d visualization of where xfield and yfield ooints are:
# ----------

# ------ 0.00 -- 0.25 -- 0.50 -- 0.75 -- 1.00 -- 1.25 -- 1.50 -- 1.75 -- 2.00 --
# - 0.00 -------------------------------- a ------------------------------------
# ------ -----------------------------------------------------------------------
# - 0.25 -----------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 0.50 -----------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 0.75 -----------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 1.00 b ---------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 1.25 ---------------- c ----------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 1.50 -----------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 1.75 -----------------------------------------------------------------------
# ------ -----------------------------------------------------------------------
# - 2.00 ---------------------------------------------- d ----------------------
# ------ -----------------------------------------------------------------------


def get_source_df():
    """
    Get a source DataFrame for 1D interpolation.
    Source df has two field:
    - x: a struct with the fields "xfield" and "yfield"
    - value: a struct with a single field "valuefield"
    Everything on the source df's xfield and yfield is between 0 and 2
    """
    return pl.DataFrame(
        {
            "xfield": [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
            ],
            "yfield": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "valuefield": [
                0.0,
                100.0,
                200.0,
                100.0,
                200.0,
                300.0,
                200.0,
                300.0,
                400.0,
            ],
        }
        # Make x column a struct with a single field "xfield"
    ).with_columns(
        **{
            "x": pl.struct([pl.col("xfield"), pl.col("yfield")]),
            "value": pl.struct([pl.col("valuefield")]),
        }
    )


def test_2d_interpolation():
    """
    Test 2D interpolation.
    Performs 2d linear interpolation
    """
    target_df = get_target_df()
    source_df = get_source_df()
    result = source_df.with_columns(
        **{"interpolated": interpolate_nd(pl.col("x"), pl.col("value"), target_df)}
    )
    result = result.select(["xfield", "yfield", "x", "interpolated"])
    expected_values = [
        # (0, 1) is exactly on a source grid point
        100.0,
        # (1, 0) is exactly on a source grid point
        100,
        # Bilinear interpolation inside the unit square:
        # z = 100*(x + y) on the provided grid, so expected is 100*(1.25 + 0.5) = 175
        175,
        # Linear along y=2 between x=1 (300) and x=2 (400): at x=1.5 => 350
        350,
    ]

    expected_df = (
        pl.DataFrame(
            {
                "xfield": [0.0, 1.0, 1.25, 1.5],
                "yfield": [1.0, 0.0, 0.5, 2.0],
                # Imaginary labels
                # "labels": ["a", "b", "c", "d"],
                "interpolated": expected_values,
            }
        )
        .with_columns(pl.struct([pl.col("xfield"), pl.col("yfield")]).alias("x"))
        .with_columns(
            pl.struct([pl.col("interpolated").alias("valuefield")]).alias(
                "interpolated"
            )
        )
        .select(["xfield", "yfield", "x", "interpolated"])
    )
    assert_frame_equal(result, expected_df)
