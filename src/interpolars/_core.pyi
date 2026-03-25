from typing import Literal, Sequence

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
    handle_missing: Literal["error", "drop", "fill", "nearest"] = ...,
    fill_value: float | None = ...,
    extrapolate: bool = ...,
    method: Literal["linear", "nearest", "cubic", "pchip", "akima", "makima"] = ...,
) -> pl.Expr: ...
def interpolate_geospatial(
    source_lat: IntoExprColumn,
    source_lon: IntoExprColumn,
    value_cols_or_exprs: IntoExprColumn
    | list[IntoExprColumn]
    | tuple[IntoExprColumn, ...],
    interp_target: pl.DataFrame,
    handle_missing: Literal["error", "drop", "fill", "nearest"] = ...,
    fill_value: float | None = ...,
    extrapolate: bool = ...,
    method: Literal["tensor_product", "slerp", "idw", "rbf"] = ...,
    *,
    tensor_method: Literal[
        "linear", "nearest", "cubic", "pchip", "akima", "makima"
    ] = ...,
    power: float = ...,
    k_neighbors: int = ...,
    rbf_kernel: Literal[
        "linear", "thin_plate_spline", "cubic",
        "gaussian", "multiquadric", "inverse_multiquadric",
    ] = ...,
    rbf_epsilon: float | None = ...,
    lon_range: Literal["signed_180", "unsigned_360", "auto"] = ...,
) -> pl.Expr: ...
