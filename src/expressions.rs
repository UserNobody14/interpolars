#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use serde::Deserialize;

#[derive(Deserialize)]
struct InterpolateNdArgs {
    interp_target: DataFrame,
}

/// Returns a struct of each field present in the inputs and all the fields present in the interp source
fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> { 
       // The first input field is the interp_source struct
       let field = &input_fields[1];
       match field.dtype() {
           DataType::Struct(fields) => {
               Ok(Field::new("interpolated".into(), DataType::Struct(fields.clone())))
           }
           dtype => polars_bail!(InvalidOperation: "expected Struct dtype, got {}", dtype),
       }
}

#[polars_expr(output_type_func=same_output_type)]
fn interpolate_nd(inputs: &[Series], kwargs: InterpolateNdArgs) -> PolarsResult<Series> {
    // Source will have values like lat, lon, etc that we will be using as the "grid"
    // let interp_source = inputs[0].clone();
    // Target will be variables that are a value at a given nd grid point
    let target_values = inputs[1].clone();
    // let interp_target = kwargs.interp_target;
    // // Get the fields of the interp_source
    // let interp_source_fields = interp_source.struct_()?.fields();
    // // Get the fields of the interp_target
    // let interp_target_fields = interp_target.fields();
    // // Create a new struct with the fields of the interp_source and the interp_target
    // let new_fields = interp_source_fields.iter().chain(interp_target_fields.iter()).collect();
    // Ok(interp_source.struct_(new_fields).into_series())
    Ok(target_values)
}