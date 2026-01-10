from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn
else:
    IntoExprColumn = Any
# from interpolars._core import interpolate_nd as interpolate_nd_core

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

    Notes:
    - The adaptor wraps the provided coords/values into structs internally:
      `pl.struct(expr_cols_or_exprs)` and `pl.struct(value_cols_or_exprs)`.
    - The returned expression evaluates to a **single struct** that contains:
      - all columns from `interp_target` (coords + metadata)
      - all interpolated value fields
    - This plugin **changes length**: the output length equals `interp_target.height()`.
    """

    if isinstance(expr_cols_or_exprs, (list, tuple)):
        coord_struct = pl.struct(list(expr_cols_or_exprs))
    else:
        coord_struct = pl.struct([expr_cols_or_exprs])

    if isinstance(value_cols_or_exprs, (list, tuple)):
        value_struct = pl.struct(list(value_cols_or_exprs))
    else:
        value_struct = pl.struct([value_cols_or_exprs])

    # Provide a dummy struct with the schema of `interp_target` so the Rust plugin can
    # compute a stable output dtype (including passthrough metadata columns).
    target_schema_exprs: list[pl.Expr] = []
    for name in interp_target.columns:
        dtype = interp_target.schema[name]
        target_schema_exprs.append(pl.lit(None).cast(dtype).alias(name))
    target_schema_struct = pl.struct(target_schema_exprs)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="interpolate_nd",
        args=[coord_struct, value_struct, target_schema_struct],
        kwargs={
            "interp_target": interp_target,
        },
        is_elementwise=False,
        changes_length=True,
    ).alias("interpolated")
