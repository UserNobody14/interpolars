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
