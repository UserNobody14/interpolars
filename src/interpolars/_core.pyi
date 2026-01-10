import polars as pl
from polars.expr import IntoExprColumn

def hello_from_bin() -> str: ...
def interpolate_nd(
    expr: IntoExprColumn,
    interp_target: pl.DataFrame,
) -> pl.Expr: ...
