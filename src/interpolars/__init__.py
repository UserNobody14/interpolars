from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polars import Expr
    from polars._typing import IntoExpr, IntoExprColumn
else:
    IntoExprColumn = Any
# from interpolars._core import interpolate_nd as interpolate_nd_core

PLUGIN_PATH = Path(__file__).parent


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
        changes_length=True,
    )
