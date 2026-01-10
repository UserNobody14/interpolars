import itertools

import polars as pl

from interpolars import interpolate_nd


def get_target_df():
    """
    Get a target DataFrame for 4D interpolation.
    Everything is in a 4 field struct called "x" with the fields:
    - xfield: a float
    - yfield: a float
    - zfield: a float
    - wfield: a float
    Target points are all inside the unit hypercube [0, 1]^4 (including corners).
    """
    return pl.DataFrame(
        {
            "xfield": [0.0, 1.0, 0.5, 0.25, 1.0],
            "yfield": [0.0, 1.0, 0.5, 0.75, 0.0],
            "zfield": [0.0, 1.0, 0.5, 0.5, 0.5],
            "wfield": [0.0, 1.0, 0.5, 0.125, 1.0],
            "labels": ["a", "b", "c", "d", "e"],
        }
    ).with_columns(
        pl.struct(
            [pl.col("xfield"), pl.col("yfield"), pl.col("zfield"), pl.col("wfield")]
        ).alias("x")
    )


def get_source_df():
    """
    Get a source DataFrame for 4D interpolation.
    Source df has two fields:
    - x: a struct with the fields "xfield", "yfield", "zfield", and "wfield"
    - value: a struct with a single field "valuefield"

    The source points are all 16 corners of the unit hypercube [0, 1]^4.
    The underlying function is affine:
        f(x, y, z, w) = 100 + 10*x + 20*y + 30*z + 40*w
    Multilinear interpolation over the hypercube reproduces affine functions exactly.
    """
    coords = list(itertools.product([0.0, 1.0], repeat=4))
    xf = [c[0] for c in coords]
    yf = [c[1] for c in coords]
    zf = [c[2] for c in coords]
    wf = [c[3] for c in coords]
    vf = [100 + 10 * x + 20 * y + 30 * z + 40 * w for x, y, z, w in coords]

    return pl.DataFrame(
        {"xfield": xf, "yfield": yf, "zfield": zf, "wfield": wf, "valuefield": vf}
    ).with_columns(
        {
            "x": pl.struct(
                [pl.col("xfield"), pl.col("yfield"), pl.col("zfield"), pl.col("wfield")]
            ),
            "value": pl.struct([pl.col("valuefield")]),
        }
    )


def test_4d_interpolation():
    """
    Test 4D interpolation.
    Performs 4D linear (multilinear) interpolation.
    """
    target_df = get_target_df()
    source_df = get_source_df()

    result = target_df.with_columns({"interpolated": interpolate_nd(pl.col("x"), source_df)})

    expected_values = [
        100.0,  # (0, 0, 0, 0)
        200.0,  # (1, 1, 1, 1)
        150.0,  # (0.5, 0.5, 0.5, 0.5) => 100 + 5 + 10 + 15 + 20
        137.5,  # (0.25, 0.75, 0.5, 0.125) => 100 + 2.5 + 15 + 15 + 5
        165.0,  # (1, 0, 0.5, 1) => 100 + 10 + 0 + 15 + 40
    ]

    expected_df = (
        pl.DataFrame(
            {
                "xfield": [0.0, 1.0, 0.5, 0.25, 1.0],
                "yfield": [0.0, 1.0, 0.5, 0.75, 0.0],
                "zfield": [0.0, 1.0, 0.5, 0.5, 0.5],
                "wfield": [0.0, 1.0, 0.5, 0.125, 1.0],
                "labels": ["a", "b", "c", "d", "e"],
                "interpolated": expected_values,
            }
        )
        .with_columns(
            pl.struct(
                [pl.col("xfield"), pl.col("yfield"), pl.col("zfield"), pl.col("wfield")]
            ).alias("x")
        )
        .with_columns(
            pl.struct([pl.col("interpolated").alias("valuefield")]).alias("interpolated")
        )
        .select(["xfield", "yfield", "zfield", "wfield", "labels", "x", "interpolated"])
    )

    pl.testing.assert_frame_equal(result, expected_df)

