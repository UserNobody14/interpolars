import polars as pl
from polars.expr import IntoExprColumn

def print_extension_info() -> str: ...
def interpolate_nd(
    expr_cols_or_exprs: IntoExprColumn
    | list[IntoExprColumn]
    | tuple[IntoExprColumn, ...],
    value_cols_or_exprs: IntoExprColumn
    | list[IntoExprColumn]
    | tuple[IntoExprColumn, ...],
    interp_target: pl.DataFrame,
) -> pl.Expr: ...
