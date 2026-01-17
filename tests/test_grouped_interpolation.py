from datetime import date

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

def test_grouped_interpolation_extra_coord_dims_date_group_key():
    """
    Grouping dims should preserve dtype.

    Here, `time` is a Date and is not present in the target, so it becomes a group dim.
    The output should include `time` as a Date (not float) and the interpolated values
    should match per-date groups.
    """
    times = [date(2020, 1, 1), date(2020, 1, 2)]
    lats = [0.0, 1.0]
    lons = [0.0, 1.0]

    rows: list[dict[str, object]] = []
    for t_idx, t in enumerate(times):
        for lat in lats:
            for lon in lons:
                rows.append(
                    {
                        "time": t,
                        "latitude": lat,
                        "longitude": lon,
                        "value": 100.0 * lat + 1000.0 * lon + 10.0 * t_idx,
                    }
                )

    # Ensure `time` is a proper Date dtype (avoid Object).
    source_df = pl.DataFrame(
        {
            "time": pl.Series("time", [r["time"] for r in rows], dtype=pl.Date),
            "latitude": [r["latitude"] for r in rows],
            "longitude": [r["longitude"] for r in rows],
            "value": [r["value"] for r in rows],
        }
    )

    target_df = pl.DataFrame(
        {
            "latitude": [0.25, 0.75],
            "longitude": [0.50, 0.25],
            "label": ["a", "b"],
        }
    )

    result = (
        source_df.lazy()
        .select(interpolate_nd(["latitude", "longitude", "time"], ["value"], target_df))
        .collect()
    )

    expected_values = [
        # date 2020-01-01 (t_idx=0)
        525.0,
        325.0,
        # date 2020-01-02 (t_idx=1)
        535.0,
        335.0,
    ]
    expected_df = (
        pl.DataFrame(
            {
                "latitude": [0.25, 0.75, 0.25, 0.75],
                "longitude": [0.50, 0.25, 0.50, 0.25],
                "label": ["a", "b", "a", "b"],
                "time": [times[0], times[0], times[1], times[1]],
                "value": expected_values,
            }
        )
        .with_columns(pl.col("time").cast(pl.Date))
        .with_columns(
            pl.struct(
                [
                    pl.col("latitude"),
                    pl.col("longitude"),
                    pl.col("label"),
                    pl.col("time"),
                    pl.col("value"),
                ]
            ).alias("interpolated")
        )
        .select(["interpolated"])
    )

    assert_frame_equal(result, expected_df)


def test_grouped_interpolation_extra_coord_dims_arr():
    """
    If the source has extra coordinate dimensions that are NOT present on the target,
    we should interpolate per unique value of those extra dims and concatenate results.

    Example: source coords = (latitude, longitude, time)
             target coords = (latitude, longitude) + metadata (label)
             -> group by time, interpolate over (latitude, longitude) per time.
    """

    # 2x2 grid over (latitude, longitude) for each time value.
    times = [0.0, 0.0, 1.0, 1.0]
    lats = [0.0, 1.0, 0.0, 1.0]
    lons = [0.0, 1.0, 0.0, 1.0]

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


def test_grouped_interpolation_extra_coord_dims_arr2():
    """
    If the source has extra coordinate dimensions that are NOT present on the target,
    we should interpolate per unique value of those extra dims and concatenate results.

    Example: source coords = (latitude, longitude, time)
             target coords = (latitude, longitude) + metadata (label)
             -> group by time, interpolate over (latitude, longitude) per time.
    """

    # 2x2 grid over (latitude, longitude) for each time value.
    times = [0.0, 0.0, 1.0, 1.0]
    lats = [0.0, 1.0, 0.0, 1.0]
    lons = [0.0, 1.0, 0.0, 1.0]
    tmptest = 0.0
    prectest = 0.0
    rows: list[dict[str, float]] = []
    for t in times:
        for lat in lats:
            for lon in lons:
                tmptest += 1.0
                prectest += 0.5
                rows.append(
                    {
                        "time": t,
                        "latitude": lat,
                        "longitude": lon,
                        # Affine functions in (lat, lon, time); multilinear interpolation should
                        # reproduce them exactly.
                        "2m_temp": tmptest,
                        "precipitation": prectest,
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
                "2m_temp": [28.5, 30.25, 60.5, 62.25],
                "precipitation": [14.25, 15.125, 30.25, 31.125],
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


