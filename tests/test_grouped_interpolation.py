import polars as pl
from polars.testing import assert_frame_equal

from interpolars import interpolate_nd


def test_grouped_interpolation_extra_coord_dims():
    """
    If the source has extra coordinate dimensions that are NOT present on the target,
    we should interpolate per unique value of those extra dims and concatenate results.

    Example: source coords = (latitude, longitude, time)
             target coords = (latitude, longitude) + metadata (label)
             -> group by time, interpolate over (latitude, longitude) per time.
    """

    # 2x2 grid over (latitude, longitude) for each time value.
    times = [0.0, 1.0]
    lats = [0.0, 1.0]
    lons = [0.0, 1.0]

    rows: list[dict[str, float]] = []
    for t in times:
        for lat in lats:
            for lon in lons:
                rows.append(
                    {
                        "time": t,
                        "latitude": lat,
                        "longitude": lon,
                        # Affine functions in (lat, lon, time); multilinear interpolation should
                        # reproduce them exactly.
                        "2m_temp": 10.0 * t + 100.0 * lat + 1000.0 * lon,
                        "precipitation": 1.0 * t + 2.0 * lat + 3.0 * lon,
                    }
                )

    source_df = pl.DataFrame(rows)

    target_df = pl.DataFrame(
        {
            "latitude": [0.25, 0.75],
            "longitude": [0.50, 0.25],
            "label": ["a", "b"],
        }
    )

    result = (
        source_df.lazy()
        .select(
            interpolate_nd(
                ["latitude", "longitude", "time"],
                ["2m_temp", "precipitation"],
                target_df,
            )
        )
        .collect()
    )

    expected_df = (
        pl.DataFrame(
            {
                "latitude": [0.25, 0.75, 0.25, 0.75],
                "longitude": [0.50, 0.25, 0.50, 0.25],
                "label": ["a", "b", "a", "b"],
                "time": [0.0, 0.0, 1.0, 1.0],
                "2m_temp": [525.0, 325.0, 535.0, 335.0],
                "precipitation": [2.0, 2.25, 3.0, 3.25],
            }
        )
        .with_columns(
            pl.struct(
                [
                    pl.col("latitude"),
                    pl.col("longitude"),
                    pl.col("label"),
                    pl.col("time"),
                    pl.col("2m_temp"),
                    pl.col("precipitation"),
                ]
            ).alias("interpolated")
        )
        .select(["interpolated"])
    )

    assert_frame_equal(result, expected_df)

