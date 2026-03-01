from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn
else:
    IntoExprColumn = Any

PLUGIN_PATH = Path(__file__).parent


InterpolationMethod = Literal[
    "linear", "nearest", "cubic", "pchip", "akima", "makima"
]

GeospatialMethod = Literal["tensor_product", "slerp", "idw", "rbf"]

LonRange = Literal["signed_180", "unsigned_360", "auto"]

RbfKernel = Literal[
    "linear", "thin_plate_spline", "cubic",
    "gaussian", "multiquadric", "inverse_multiquadric",
]


def interpolate_nd(
    expr_cols_or_exprs: IntoExprColumn | Sequence[IntoExprColumn],
    value_cols_or_exprs: IntoExprColumn | Sequence[IntoExprColumn],
    interp_target: pl.DataFrame,
    handle_missing: Literal["error", "drop", "fill", "nearest"] = "error",
    fill_value: float | None = None,
    extrapolate: bool = False,
    method: InterpolationMethod = "linear",
) -> pl.Expr:
    """
    Interpolate from a source "grid" (the calling DataFrame) to an explicit target DataFrame.

    - **Source coords**: `expr_cols_or_exprs` (column name(s) or Polars expr(s))
    - **Source values**: `value_cols_or_exprs` (column name(s) or Polars expr(s))
    - **Target coords**: `interp_target` must contain *plain columns* matching the source coord
      field names (e.g. `xfield`, `yfield`, ...). Struct columns are not considered.

      If the **source** has additional coordinate columns that are **missing** from `interp_target`
      (e.g. `time`), the plugin will treat those as **grouping dimensions**: it will group the
      source rows by those extra coordinate columns and run interpolation independently per group.

    For multi-dimensional data the interpolation is decomposed into successive
    independent 1-D interpolations along each axis (tensor-product interpolation),
    which is mathematically equivalent to ``scipy.interpolate.interpn`` on
    rectilinear grids.

    Args:
        handle_missing: How to handle NaN/Null in source data.
            - ``"error"`` (default): raise on any NaN or Null.
            - ``"drop"``: silently drop source rows containing NaN/Null.
            - ``"fill"``: replace NaN/Null values with *fill_value* (coords are dropped).
            - ``"nearest"``: replace NaN/Null values with the nearest valid grid point's
              value by Euclidean distance in coordinate space (coords are dropped).
        fill_value: Constant used when ``handle_missing="fill"``. Required in that mode.
        extrapolate: When ``True``, extrapolate for target points outside the source grid
            instead of clamping to the boundary value.
        method: Interpolation method. Each method matches the corresponding
            scipy/xarray implementation:

            - ``"linear"`` (default): piecewise linear (``numpy.interp``).
            - ``"nearest"``: snap to closest grid point.
            - ``"cubic"``: not-a-knot cubic spline
              (``scipy.interpolate.interp1d(kind='cubic')``).
            - ``"pchip"``: monotone Piecewise Cubic Hermite Interpolating
              Polynomial (``scipy.interpolate.PchipInterpolator``).
            - ``"akima"``: Akima 1D interpolator
              (``scipy.interpolate.Akima1DInterpolator``).
            - ``"makima"``: Modified Akima with adjusted weights
              (``scipy.interpolate.Akima1DInterpolator(method="makima")``).
    """

    if isinstance(expr_cols_or_exprs, (list, tuple)):
        coord_struct = pl.struct(list(expr_cols_or_exprs))
    else:
        coord_struct = pl.struct([expr_cols_or_exprs])

    if isinstance(value_cols_or_exprs, (list, tuple)):
        value_struct = pl.struct(list(value_cols_or_exprs))
    else:
        value_struct = pl.struct([value_cols_or_exprs])

    # Pass the full target as a *literal* struct Series.
    # Important: `pl.lit(Series)` preserves the Series' own length, so this literal can drive
    # the plugin's output length (changes_length=True).
    target_struct_series = (
        interp_target.select(pl.struct(pl.all()).alias("__interp_target__")).to_series()
    )
    target_struct_lit = pl.lit(target_struct_series)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="interpolate_nd",
        args=[coord_struct, value_struct, target_struct_lit],
        kwargs={
            "handle_missing": handle_missing,
            "fill_value": fill_value,
            "extrapolate": extrapolate,
            "method": method,
        },
        is_elementwise=False,
        changes_length=True,
    ).alias("interpolated")


def interpolate_geospatial(
    source_lat: IntoExprColumn,
    source_lon: IntoExprColumn,
    value_cols_or_exprs: IntoExprColumn | Sequence[IntoExprColumn],
    interp_target: pl.DataFrame,
    handle_missing: Literal["error", "drop", "fill", "nearest"] = "error",
    fill_value: float | None = None,
    extrapolate: bool = False,
    method: GeospatialMethod = "tensor_product",
    *,
    tensor_method: InterpolationMethod = "linear",
    power: float = 2.0,
    k_neighbors: int = 0,
    rbf_kernel: RbfKernel = "thin_plate_spline",
    rbf_epsilon: float | None = None,
    lon_range: LonRange = "auto",
) -> pl.Expr:
    """
    Interpolate a scalar field defined on latitude/longitude coordinates,
    with spherical-geometry awareness.

    Four methods are available, selected via the ``method`` parameter:

    - ``"tensor_product"`` (default): Tensor-product interpolation on a
      rectilinear lat/lon grid (longitude wrapping, pole averaging, ghost
      points for periodic grids).  Supports all 1-D sub-methods via
      ``tensor_method``.  **Requires gridded (full Cartesian product) input.**

    - ``"slerp"``: Bilinear interpolation using SLERP-derived angular
      fraction weights along parallels.  More accurate than standard
      bilinear near the poles and for large grid cells.  **Requires gridded
      input.  Linear only.**

    - ``"idw"``: Inverse Distance Weighting using Haversine (great-circle)
      distance.  **Works on scattered (non-gridded) source data.**  Tune
      via ``power`` and ``k_neighbors``.

    - ``"rbf"``: Local Radial Basis Function interpolation using Haversine
      distance.  Solves a k×k linear system per target point.
      **Works on scattered source data.**  Tune via ``rbf_kernel``,
      ``rbf_epsilon``, and ``k_neighbors``.

    Args:
        source_lat: Column name or expression for source latitude (degrees).
        source_lon: Column name or expression for source longitude (degrees).
        value_cols_or_exprs: Column name(s) or expression(s) for the value
            field(s) to interpolate.
        interp_target: DataFrame with ``lat`` and ``lon`` columns (matching
            the source column names) specifying the target points.
        handle_missing: How to handle NaN/Null in source data (same semantics
            as ``interpolate_nd``).
        fill_value: Constant for ``handle_missing="fill"``.
        extrapolate: Extrapolate outside the source extent instead of clamping.
        method: Interpolation method (see above).
        tensor_method: 1-D sub-method for ``tensor_product`` mode.
        power: Distance exponent for ``idw`` (default 2.0).
        k_neighbors: Number of nearest neighbors for ``idw`` / ``rbf``
            (0 = use all sources for ``idw``; default 20 for ``rbf``).
        rbf_kernel: Kernel function for ``rbf``.
        rbf_epsilon: Shape parameter for ``rbf`` (``None`` = auto from
            median pairwise distance).
        lon_range: Longitude convention:
            - ``"signed_180"``: normalize to [-180, 180).
            - ``"unsigned_360"``: normalize to [0, 360).
            - ``"auto"`` (default): detect from source data.
    """
    coord_struct = pl.struct([source_lat, source_lon])

    if isinstance(value_cols_or_exprs, (list, tuple)):
        value_struct = pl.struct(list(value_cols_or_exprs))
    else:
        value_struct = pl.struct([value_cols_or_exprs])

    target_struct_series = (
        interp_target.select(pl.struct(pl.all()).alias("__interp_target__")).to_series()
    )
    target_struct_lit = pl.lit(target_struct_series)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="interpolate_geospatial",
        args=[coord_struct, value_struct, target_struct_lit],
        kwargs={
            "handle_missing": handle_missing,
            "fill_value": fill_value,
            "extrapolate": extrapolate,
            "method": method,
            "tensor_method": tensor_method,
            "power": power,
            "k_neighbors": k_neighbors,
            "rbf_kernel": rbf_kernel,
            "rbf_epsilon": rbf_epsilon,
            "lon_range": lon_range,
        },
        is_elementwise=False,
        changes_length=True,
    ).alias("interpolated")
