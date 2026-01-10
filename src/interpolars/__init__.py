from pathlib import Path

from interpolars._core import interpolate_nd as interpolate_nd_core

PLUGIN_PATH = Path(__file__).parent

import polars as pl
from polars.expr import IntoExpr, IntoExprColumn
from polars.plugin import register_plugin_function


def interpolate_nd(
    expr: IntoExprColumn,
    interp_target: pl.DataFrame,
) -> pl.Expr:
    """
    Interpolate a given expression across a target DataFrame.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="interpolate_nd",
        args=[expr],
        kwargs={
            "interp_target": interp_target,
        },
        is_elementwise=False,
    )
