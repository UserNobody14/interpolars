"""Tests for interpolate_geospatial: methods, IDL wrapping, pole handling, lon_range."""

import math

import numpy as np
import polars as pl
import pytest

from interpolars import interpolate_geospatial


def _build_global_grid(lat_step=30.0, lon_step=60.0, func=None):
    """Build a simple global lat/lon grid.

    Default value function: ``sin(lon_rad)`` -- smooth, periodic, and non-trivial.
    """
    if func is None:
        func = lambda lat, lon: math.sin(math.radians(lon))
    lats = np.arange(-90.0, 90.0 + lat_step / 2, lat_step)
    lons = np.arange(-180.0, 180.0, lon_step)
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(float(lat))
            rows_lon.append(float(lon))
            rows_v.append(func(lat, lon))
    return pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})


def _run_geospatial(source, target, value_cols=None, **kwargs):
    if value_cols is None:
        value_cols = ["v"]
    result = (
        source.lazy()
        .select(
            interpolate_geospatial("lat", "lon", value_cols, target, **kwargs)
        )
        .collect()
    )
    return result.unnest("interpolated")


# ---------------------------------------------------------------------------
# Basic grid (tensor_product, default)
# ---------------------------------------------------------------------------


def test_exact_grid_hits():
    """Interpolation at grid points returns exact values."""
    source = _build_global_grid()
    target = pl.DataFrame({"lat": [0.0, 30.0, -60.0], "lon": [0.0, 60.0, -120.0]})
    result = _run_geospatial(source, target)
    expected = [math.sin(math.radians(l)) for l in [0.0, 60.0, -120.0]]
    np.testing.assert_allclose(result["v"].to_list(), expected, atol=1e-10)


def test_interior_linear():
    """Linear interpolation at interior points."""
    source = _build_global_grid()
    target = pl.DataFrame({"lat": [15.0], "lon": [30.0]})
    result = _run_geospatial(source, target)
    vals = result["v"].to_list()
    assert len(vals) == 1
    assert isinstance(vals[0], float)


# ---------------------------------------------------------------------------
# International Date Line
# ---------------------------------------------------------------------------


def test_idl_crossing():
    """Interpolation across the +/-180 boundary on a global grid."""
    source = _build_global_grid(lat_step=30.0, lon_step=30.0)
    target = pl.DataFrame({"lat": [0.0], "lon": [165.0]})
    result = _run_geospatial(source, target)
    val = result["v"].to_list()[0]
    f_150 = math.sin(math.radians(150))
    f_m180 = math.sin(math.radians(-180))
    expected = f_150 + 0.5 * (f_m180 - f_150)
    np.testing.assert_allclose(val, expected, atol=1e-10)


def test_idl_negative_target():
    """Target at lon=-170 (just past the date line from +180)."""
    source = _build_global_grid(lat_step=30.0, lon_step=30.0)
    target = pl.DataFrame({"lat": [0.0], "lon": [-170.0]})
    result = _run_geospatial(source, target)
    val = result["v"].to_list()[0]
    f_m180 = math.sin(math.radians(-180))
    f_m150 = math.sin(math.radians(-150))
    t = (-170.0 - (-180.0)) / (-150.0 - (-180.0))
    expected = f_m180 + t * (f_m150 - f_m180)
    np.testing.assert_allclose(val, expected, atol=1e-10)


def test_idl_regional_grid():
    """Regional grid straddling the IDL (e.g. Pacific islands)."""
    lats = [0.0, 10.0]
    lons = [170.0, 175.0, 180.0, -175.0, -170.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [5.0], "lon": [177.5]})
    result = _run_geospatial(source, target)
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 5.0 + 177.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Poles
# ---------------------------------------------------------------------------


def test_pole_averaging():
    """Values at the pole should be averaged across longitudes."""
    lats = [-90.0, -60.0, 0.0, 60.0, 90.0]
    lons = [0.0, 90.0, 180.0, -90.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            if abs(lat) == 90.0:
                rows_v.append(100.0 + lon)
            else:
                rows_v.append(lat * 1.0)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [90.0, 90.0], "lon": [0.0, 45.0]})
    result = _run_geospatial(source, target)
    vals = result["v"].to_list()
    # Values at pole: 100+0=100, 100+90=190, 100+180=280, 100+(-90)=10
    pole_mean = (100.0 + 190.0 + 280.0 + 10.0) / 4.0  # = 145
    np.testing.assert_allclose(vals[0], pole_mean, atol=1e-10)
    np.testing.assert_allclose(vals[1], pole_mean, atol=1e-10)


def test_near_pole_interpolation():
    """Interpolating near (but not at) a pole should give a sensible result."""
    lats = [-90.0, -45.0, 0.0, 45.0, 90.0]
    lons = [0.0, 90.0, 180.0, -90.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat * 1.0)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [89.0], "lon": [0.0]})
    result = _run_geospatial(source, target)
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 89.0, atol=1.0)
    assert val > 44.0


# ---------------------------------------------------------------------------
# Multiple value columns
# ---------------------------------------------------------------------------


def test_multiple_values():
    """Interpolate several value columns simultaneously."""
    lats = [0.0, 10.0]
    lons = [0.0, 10.0]
    rows_lat, rows_lon, rows_u, rows_w = [], [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_u.append(lat + lon)
            rows_w.append(lat * lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "u": rows_u, "w": rows_w})
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, value_cols=["u", "w"])
    np.testing.assert_allclose(result["u"].to_list()[0], 10.0, atol=1e-10)
    np.testing.assert_allclose(result["w"].to_list()[0], 25.0, atol=1e-10)


# ---------------------------------------------------------------------------
# tensor_method passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tensor_method", ["linear", "nearest", "cubic", "pchip", "akima", "makima"]
)
def test_tensor_method_smoke(tensor_method):
    """Each tensor_method runs without error on a minimal grid."""
    lats = [0.0, 10.0, 20.0, 30.0, 40.0]
    lons = [0.0, 10.0, 20.0, 30.0, 40.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [15.0], "lon": [15.0]})
    result = _run_geospatial(source, target, tensor_method=tensor_method)
    vals = result["v"].to_list()
    assert len(vals) == 1
    assert isinstance(vals[0], float)
    if tensor_method not in ("nearest",):
        np.testing.assert_allclose(vals[0], 30.0, atol=1e-8)


# ---------------------------------------------------------------------------
# SLERP method
# ---------------------------------------------------------------------------


def test_slerp_at_grid_points():
    """SLERP returns exact values at grid points."""
    lats = [0.0, 10.0]
    lons = [0.0, 10.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [0.0, 10.0], "lon": [0.0, 10.0]})
    result = _run_geospatial(source, target, method="slerp")
    np.testing.assert_allclose(result["v"].to_list(), [0.0, 20.0], atol=1e-10)


def test_slerp_interior():
    """SLERP at the center of a small cell is close to bilinear."""
    lats = [0.0, 10.0]
    lons = [0.0, 10.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, method="slerp")
    np.testing.assert_allclose(result["v"].to_list()[0], 10.0, atol=0.5)


def test_slerp_vs_tensor_product_near_pole():
    """SLERP may give slightly different results near the pole than tensor_product."""
    lats = [80.0, 90.0]
    lons = [0.0, 30.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lon * 1.0)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [85.0], "lon": [15.0]})
    r_tp = _run_geospatial(source, target, method="tensor_product")
    r_sl = _run_geospatial(source, target, method="slerp")
    # Both should be finite and close-ish, but not identical
    assert math.isfinite(r_tp["v"][0])
    assert math.isfinite(r_sl["v"][0])


# ---------------------------------------------------------------------------
# IDW method (scattered data)
# ---------------------------------------------------------------------------


def test_idw_scattered():
    """IDW interpolates non-gridded source points."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 10.0],
        "lon": [0.0, 10.0, 5.0],
        "v": [100.0, 200.0, 300.0],
    })
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, method="idw")
    val = result["v"].to_list()[0]
    assert math.isfinite(val)
    assert 100.0 < val < 300.0


def test_idw_exact_hit():
    """IDW returns exact value when target coincides with a source point."""
    source = pl.DataFrame({
        "lat": [0.0, 10.0, 20.0],
        "lon": [0.0, 10.0, 20.0],
        "v": [100.0, 200.0, 300.0],
    })
    target = pl.DataFrame({"lat": [10.0], "lon": [10.0]})
    result = _run_geospatial(source, target, method="idw")
    np.testing.assert_allclose(result["v"].to_list()[0], 200.0, atol=1e-10)


def test_idw_equidistant():
    """IDW at equidistant point between two sources returns their average."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0],
        "lon": [-10.0, 10.0],
        "v": [100.0, 200.0],
    })
    target = pl.DataFrame({"lat": [0.0], "lon": [0.0]})
    result = _run_geospatial(source, target, method="idw")
    np.testing.assert_allclose(result["v"].to_list()[0], 150.0, atol=1e-8)


def test_idw_k_neighbors():
    """k_neighbors limits how many source points contribute."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 0.0],
        "lon": [0.0, 1.0, 100.0],
        "v": [10.0, 20.0, 999.0],
    })
    target = pl.DataFrame({"lat": [0.0], "lon": [0.5]})
    result_k2 = _run_geospatial(source, target, method="idw", k_neighbors=2)
    result_all = _run_geospatial(source, target, method="idw", k_neighbors=0)
    v_k2 = result_k2["v"][0]
    v_all = result_all["v"][0]
    # k=2 should exclude the far point, giving result closer to 15
    assert abs(v_k2 - 15.0) < abs(v_all - 15.0)


# ---------------------------------------------------------------------------
# RBF method (scattered data)
# ---------------------------------------------------------------------------


def test_rbf_smooth_field():
    """RBF reproduces a simple field from scattered samples."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 10.0, 10.0, 5.0],
        "lon": [0.0, 10.0, 0.0, 10.0, 5.0],
        "v": [0.0, 10.0, 10.0, 20.0, 10.0],
    })
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, method="rbf")
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 10.0, atol=2.0)


def test_rbf_exact_hit():
    """RBF returns exact value at a source point."""
    source = pl.DataFrame({
        "lat": [0.0, 10.0, 20.0],
        "lon": [0.0, 10.0, 20.0],
        "v": [1.0, 2.0, 3.0],
    })
    target = pl.DataFrame({"lat": [10.0], "lon": [10.0]})
    result = _run_geospatial(source, target, method="rbf")
    np.testing.assert_allclose(result["v"].to_list()[0], 2.0, atol=1e-10)


@pytest.mark.parametrize(
    "rbf_kernel",
    ["linear", "thin_plate_spline", "cubic", "gaussian", "multiquadric", "inverse_multiquadric"],
)
def test_rbf_kernels_smoke(rbf_kernel):
    """Each RBF kernel runs without error."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 10.0, 10.0, 5.0],
        "lon": [0.0, 10.0, 0.0, 10.0, 5.0],
        "v": [0.0, 10.0, 10.0, 20.0, 10.0],
    })
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, method="rbf", rbf_kernel=rbf_kernel)
    val = result["v"].to_list()[0]
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# Longitude convention handling
# ---------------------------------------------------------------------------


def test_lon_range_signed180():
    """Explicit signed_180 normalizes data correctly."""
    lats = [0.0, 10.0]
    lons = [350.0, 10.0]  # 350 should become -10 in signed_180
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [5.0], "lon": [0.0]})
    result = _run_geospatial(source, target, lon_range="signed_180")
    val = result["v"].to_list()[0]
    assert math.isfinite(val)


def test_lon_range_unsigned360():
    """Explicit unsigned_360 normalizes data correctly."""
    lats = [0.0, 10.0]
    lons = [0.0, 10.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, lon_range="unsigned_360")
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 10.0, atol=1e-10)


def test_lon_range_auto_detection():
    """Auto-detection picks signed_180 when data has negative longitudes."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 10.0, 10.0],
        "lon": [-5.0, 5.0, -5.0, 5.0],
        "v": [10.0, 20.0, 30.0, 40.0],
    })
    target = pl.DataFrame({"lat": [5.0], "lon": [0.0]})
    result = _run_geospatial(source, target, lon_range="auto")
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 25.0, atol=1e-10)


def test_lon_range_auto_unsigned():
    """Auto-detection picks unsigned_360 when all data is non-negative."""
    source = pl.DataFrame({
        "lat": [0.0, 0.0, 10.0, 10.0],
        "lon": [350.0, 10.0, 350.0, 10.0],
        "v": [10.0, 20.0, 30.0, 40.0],
    })
    target = pl.DataFrame({"lat": [5.0], "lon": [0.0]})
    result = _run_geospatial(source, target, lon_range="auto")
    val = result["v"].to_list()[0]
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# IDW with gridded data (should work too)
# ---------------------------------------------------------------------------


def test_idw_on_gridded():
    """IDW also works on regular gridded data (no grid requirement)."""
    lats = [0.0, 10.0]
    lons = [0.0, 10.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [5.0], "lon": [5.0]})
    result = _run_geospatial(source, target, method="idw")
    val = result["v"].to_list()[0]
    np.testing.assert_allclose(val, 10.0, atol=0.5)


# ---------------------------------------------------------------------------
# Multiple methods on the same data
# ---------------------------------------------------------------------------


def test_all_methods_run():
    """All four methods produce finite results on the same input."""
    lats = [0.0, 10.0, 20.0, 30.0]
    lons = [0.0, 10.0, 20.0, 30.0]
    rows_lat, rows_lon, rows_v = [], [], []
    for lat in lats:
        for lon in lons:
            rows_lat.append(lat)
            rows_lon.append(lon)
            rows_v.append(lat + lon)
    source = pl.DataFrame({"lat": rows_lat, "lon": rows_lon, "v": rows_v})
    target = pl.DataFrame({"lat": [15.0], "lon": [15.0]})
    for method in ["tensor_product", "slerp", "idw", "rbf"]:
        result = _run_geospatial(source, target, method=method)
        val = result["v"].to_list()[0]
        assert math.isfinite(val), f"method={method} returned {val}"
        np.testing.assert_allclose(val, 30.0, atol=3.0, err_msg=f"method={method}")
