from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn
else:
    IntoExprColumn = Any

PLUGIN_PATH = Path(__file__).parent


def interpolate_nd(
    expr_cols_or_exprs: IntoExprColumn | Sequence[IntoExprColumn],
    value_cols_or_exprs: IntoExprColumn | Sequence[IntoExprColumn],
    interp_target: pl.DataFrame,
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

    Notes:
    - The adaptor wraps the provided coords/values into structs internally:
      `pl.struct(expr_cols_or_exprs)` and `pl.struct(value_cols_or_exprs)`.
    - The returned expression evaluates to a **single struct** that contains:
      - all columns from `interp_target` (coords + metadata)
      - any extra coordinate columns from the source that are missing from `interp_target`
      - all interpolated value fields
    - This plugin **changes length**:
      - if there are no extra source coordinate dims, output length equals `interp_target.height()`
      - otherwise output length equals `interp_target.height() * number_of_groups`
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
        is_elementwise=False,
        changes_length=True,
    ).alias("interpolated")
