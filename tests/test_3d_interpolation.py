import itertools

import polars as pl
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd


def get_target_df():
    """
    Get a target DataFrame for 3D interpolation.
    Everything is in a 3 field struct called "x" with the fields:
    - xfield: a float
    - yfield: a float
    - zfield: a float
    Target points are all inside the unit cube [0, 1]^3 (including corners).
    """
    return pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 0.0, 0.0, 0.5, 0.25],
            "yfield": [0.0, 0.0, 1.0, 0.0, 0.5, 0.75],
            "zfield": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5],
            # Imaginary labels
            # "labels": ["a", "b", "c", "d", "e", "f"],
        }
    ).with_columns(
        pl.struct([pl.col("xfield"), pl.col("yfield"), pl.col("zfield")]).alias("x")
    )


def get_source_df():
    """
    Get a source DataFrame for 3D interpolation.
    Source df has two fields:
    - x: a struct with the fields "xfield", "yfield", and "zfield"
    - value: a struct with a single field "valuefield"

    The source points are all 8 corners of the unit cube [0, 1]^3.
    The underlying function is affine:
        f(x, y, z) = 100 + 10*x + 20*y + 30*z
    Multilinear interpolation over the cube reproduces affine functions exactly.
    """
    coords = list(itertools.product([0.0, 1.0], repeat=3))
    xf = [c[0] for c in coords]
    yf = [c[1] for c in coords]
    zf = [c[2] for c in coords]
    vf = [100 + 10 * x + 20 * y + 30 * z for x, y, z in coords]

    return pl.DataFrame(
        {"xfield": xf, "yfield": yf, "zfield": zf, "valuefield": vf}
    ).with_columns(
        **{
            "x": pl.struct([pl.col("xfield"), pl.col("yfield"), pl.col("zfield")]),
            "value": pl.struct([pl.col("valuefield")]),
        }
    )


def test_3d_interpolation():
    """
    Test 3D interpolation.
    Performs 3D linear (multilinear) interpolation.
    """
    target_df = get_target_df()
    source_df = get_source_df()
    result = source_df.with_columns(
        **{"interpolated": interpolate_nd(pl.col("x"), pl.col("value"), target_df)}
    )
    result = result.select(["xfield", "yfield", "zfield", "x", "interpolated"])
    expected_values = [
        100.0,  # (0, 0, 0)
        110.0,  # (1, 0, 0)
        120.0,  # (0, 1, 0)
        130.0,  # (0, 0, 1)
        130.0,  # (0.5, 0.5, 0.5) => 100 + 5 + 10 + 15
        132.5,  # (0.25, 0.75, 0.5) => 100 + 2.5 + 15 + 15
    ]

    expected_df = (
        pl.DataFrame(
            {
                "xfield": [0.0, 1.0, 0.0, 0.0, 0.5, 0.25],
                "yfield": [0.0, 0.0, 1.0, 0.0, 0.5, 0.75],
                "zfield": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5],
                # Imaginary labels
                # "labels": ["a", "b", "c", "d", "e", "f"],
                "interpolated": expected_values,
            }
        )
        .with_columns(
            pl.struct([pl.col("xfield"), pl.col("yfield"), pl.col("zfield")]).alias("x")
        )
        .with_columns(
            pl.struct([pl.col("interpolated").alias("valuefield")]).alias(
                "interpolated"
            )
        )
        .select(["xfield", "yfield", "zfield", "x", "interpolated"])
    )

    assert_frame_equal(result, expected_df)
