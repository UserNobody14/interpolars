use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct InterpolateNdArgs {
    interp_target: DataFrame,
}

/// Output type is a concatenation of:
/// - the passthrough target struct schema (3rd input)
/// - the value-struct input (2nd input)
fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if input_fields.len() < 3 {
        polars_bail!(
            InvalidOperation:
            "expected 3 inputs (coords struct, values struct, target schema struct), got {}",
            input_fields.len()
        );
    }

    let value_field = &input_fields[1];
    let target_schema_field = &input_fields[2];

    let DataType::Struct(value_fields) = value_field.dtype() else {
        polars_bail!(
            InvalidOperation:
            "expected Struct dtype for values input, got {}",
            value_field.dtype()
        );
    };
    let DataType::Struct(target_fields) = target_schema_field.dtype() else {
        polars_bail!(
            InvalidOperation:
            "expected Struct dtype for target schema input, got {}",
            target_schema_field.dtype()
        );
    };

    let mut out_fields: Vec<Field> = Vec::with_capacity(target_fields.len() + value_fields.len());
    out_fields.extend(target_fields.iter().cloned());
    for f in value_fields {
        if out_fields.iter().any(|existing| existing.name == f.name) {
            polars_bail!(
                InvalidOperation:
                "duplicate output field name {} (present in interp_target and values)",
                f.name
            );
        }
        out_fields.push(f.clone());
    }

    Ok(Field::new("interpolated".into(), DataType::Struct(out_fields)))
}

fn series_to_f64_vec(s: &Series) -> PolarsResult<Vec<f64>> {
    let s = s.cast(&DataType::Float64)?;
    let ca = s.f64()?;
    ca.into_iter()
        .map(|opt| {
            opt.ok_or_else(|| polars_err!(ComputeError: "null not supported in interpolation inputs"))
        })
        .collect()
}

fn lower_upper_indices(axis: &[f64], t: f64) -> (usize, usize) {
    debug_assert!(!axis.is_empty());
    if t <= axis[0] {
        return (0, 0);
    }
    let last = axis.len() - 1;
    if t >= axis[last] {
        return (last, last);
    }

    // Find first index with axis[i] >= t
    let mut lo = 0usize;
    let mut hi = last;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if axis[mid] < t {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let idx = lo;
    if axis[idx] == t {
        (idx, idx)
    } else {
        (idx - 1, idx)
    }
}

#[polars_expr(output_type_func=same_output_type)]
fn interpolate_nd(inputs: &[Series], kwargs: InterpolateNdArgs) -> PolarsResult<Series> {
    // inputs[0]: source coordinates as a Struct (e.g. {xfield, yfield, ...})
    // inputs[1]: source values as a Struct (e.g. {valuefield, ...})
    // inputs[2]: dummy target schema struct (names + dtypes of all columns to passthrough)
    // kwargs.interp_target: target DataFrame containing target coordinates + metadata columns.

    let source_coords = inputs[0].struct_()?;
    let source_values = inputs[1].struct_()?;
    let target_schema = inputs[2].struct_()?;
    let interp_target = kwargs.interp_target;

    let coord_fields = source_coords.fields_as_series();
    if coord_fields.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 coordinate field for interpolation");
    }

    let dim_names: Vec<String> = coord_fields.iter().map(|s| s.name().to_string()).collect();
    let dims = dim_names.len();

    // Build axes (unique sorted coordinate values per dimension)
    let mut axes: Vec<Vec<f64>> = Vec::with_capacity(dims);
    for s in &coord_fields {
        let mut axis = series_to_f64_vec(s)?;
        axis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        axis.dedup();
        if axis.is_empty() {
            polars_bail!(ComputeError: "empty axis after dedup; cannot interpolate");
        }
        axes.push(axis);
    }

    // Map coordinate tuple -> row index in source
    let mut coord_map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(source_coords.len());
    let coord_cols: Vec<Vec<f64>> = coord_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;
    for row_idx in 0..source_coords.len() {
        let mut key: Vec<u64> = Vec::with_capacity(dims);
        for d in 0..dims {
            key.push(coord_cols[d][row_idx].to_bits());
        }
        coord_map.insert(key, row_idx);
    }

    // Extract value fields
    let value_fields = source_values.fields_as_series();
    if value_fields.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 value field for interpolation");
    }
    let value_names: Vec<String> = value_fields.iter().map(|s| s.name().to_string()).collect();
    let value_cols: Vec<Vec<f64>> = value_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    // Extract target coordinates from individual columns on interp_target.
    let target_n = interp_target.height();
    let mut target_coord_cols: Vec<Vec<f64>> = Vec::with_capacity(dims);
    for name in &dim_names {
        let s = interp_target.column(name).map_err(|_| {
            polars_err!(InvalidOperation: "interp_target missing column {}", name)
        })?;
        target_coord_cols.push(series_to_f64_vec(s.as_materialized_series())?);
    }

    // Interpolate each target row
    let mut out_value_cols: Vec<Vec<f64>> = value_names
        .iter()
        .map(|_| Vec::with_capacity(target_n))
        .collect();

    for row in 0..target_n {
        let mut lo_idx: Vec<usize> = Vec::with_capacity(dims);
        let mut hi_idx: Vec<usize> = Vec::with_capacity(dims);
        let mut tvals: Vec<f64> = Vec::with_capacity(dims);

        for d in 0..dims {
            let t = target_coord_cols[d][row];
            tvals.push(t);
            let (lo, hi) = lower_upper_indices(&axes[d], t);
            lo_idx.push(lo);
            hi_idx.push(hi);
        }

        let corners = 1usize << dims;
        let mut sums: Vec<f64> = vec![0.0; value_names.len()];

        for corner in 0..corners {
            let mut weight = 1.0f64;
            let mut key: Vec<u64> = Vec::with_capacity(dims);

            for d in 0..dims {
                let lo = lo_idx[d];
                let hi = hi_idx[d];
                let t = tvals[d];
                let use_hi = (corner >> d) & 1 == 1;

                if lo == hi {
                    // clamped or exact hit
                    key.push(axes[d][lo].to_bits());
                    // Avoid overcounting duplicate corners when a dimension isn't interpolating.
                    // Only the "lo" branch should contribute.
                    if use_hi {
                        weight = 0.0;
                    }
                    continue;
                }

                let x0 = axes[d][lo];
                let x1 = axes[d][hi];
                let denom = x1 - x0;
                if denom == 0.0 {
                    polars_bail!(ComputeError: "zero grid spacing for axis {}", dim_names[d]);
                }

                if use_hi {
                    weight *= (t - x0) / denom;
                    key.push(x1.to_bits());
                } else {
                    weight *= (x1 - t) / denom;
                    key.push(x0.to_bits());
                }
            }

            if weight == 0.0 {
                continue;
            }

            let src_row = coord_map.get(&key).copied().ok_or_else(|| {
                polars_err!(
                    ComputeError:
                    "source grid missing corner point for key {:?}; ensure source is a full cartesian grid",
                    key
                )
            })?;

            for (vi, col) in value_cols.iter().enumerate() {
                sums[vi] += weight * col[src_row];
            }
        }

        for (vi, s) in sums.into_iter().enumerate() {
            out_value_cols[vi].push(s);
        }
    }

    // Build output struct series:
    // - passthrough columns from interp_target (in the order described by `target_schema`)
    // - interpolated value columns
    let mut out_fields: Vec<Series> = Vec::with_capacity(target_schema.fields_as_series().len() + value_names.len());

    let passthrough_fields = target_schema.fields_as_series();
    for s in &passthrough_fields {
        let name = s.name();
        if value_names.iter().any(|v| v == name) {
            polars_bail!(
                InvalidOperation:
                "duplicate output field name {} (present in interp_target and values)",
                name
            );
        }
        let col = interp_target.column(name).map_err(|_| {
            polars_err!(InvalidOperation: "interp_target missing column {}", name)
        })?;
        out_fields.push(col.as_materialized_series().clone());
    }

    for (name, vals) in value_names.iter().zip(out_value_cols.into_iter()) {
        out_fields.push(Series::new(name.as_str().into(), vals));
    }
    Ok(
        StructChunked::from_series("interpolated".into(), target_n, out_fields.iter())?
            .into_series(),
    )
}

