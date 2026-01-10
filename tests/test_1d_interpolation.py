import polars as pl

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
            "xfield": [0, 7, 8, 9, 10],
            "valuefield": [100, 200, 300, 400, 500],
        }
        # Make x column a struct with a single field "xfield"
    ).with_columns(
        {
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
    result = target_df.with_columns(
        {"interpolated": interpolate_nd(pl.col("x"), source_df)}
    )
    result_df = pl.DataFrame(
        {
            "x": [0.1, 1, 2.5, 3.5, 4.5, 5.5, 8, 8.5, 8.75, 10],
            "interpolated": [
                # First few between 0 and 7
                (0.1 / 7) * 100,
                (1 / 7) * 100,
                (2.5 / 7) * 100,
                (3.5 / 7) * 100,
                (4.5 / 7) * 100,
                (5.5 / 7) * 100,
                # Last few between the other points
                300,  # Precisely 300 (on 8)
                350,  # 300 + (8.5 - 8) * (400 - 300) / (8.5 - 8)
                375,  # 300 + (8.75 - 8) * (400 - 300) / (8.75 - 8)
                500,  # Precisely 500
            ],
        }
    ).with_columns(pl.struct([pl.col("interpolated")]).alias("interpolated"))
    pl.testing.assert_frame_equal(result, result_df)
