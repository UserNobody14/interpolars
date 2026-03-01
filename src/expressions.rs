use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::HashMap;

use crate::geospatial::{self, GeospatialMethod, LonRange, RbfKernel};
use crate::interpolation::{self, InterpolationMethod};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum MissingHandling {
    Error,
    Drop,
    Fill,
    Nearest,
}

#[derive(Deserialize)]
struct InterpolateKwargs {
    handle_missing: MissingHandling,
    fill_value: Option<f64>,
    extrapolate: bool,
    #[serde(default = "default_method")]
    method: InterpolationMethod,
}

fn default_method() -> InterpolationMethod {
    InterpolationMethod::Linear
}

/// Output type is a concatenation of:
/// - the passthrough target struct schema (3rd input)
/// - any "group" coordinate fields that are present in the source coords (1st input)
///   but missing from the target schema (3rd input)
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
    let source_coords_field = &input_fields[0];

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
    let DataType::Struct(source_coord_fields) = source_coords_field.dtype() else {
        polars_bail!(
            InvalidOperation:
            "expected Struct dtype for coords input, got {}",
            source_coords_field.dtype()
        );
    };

    let target_names: std::collections::HashSet<PlSmallStr> =
        target_fields.iter().map(|f| f.name.clone()).collect();

    let group_fields: Vec<Field> = source_coord_fields
        .iter()
        .filter(|f| !target_names.contains(&f.name))
        .cloned()
        .collect();

    let source_coord_dtype_map: HashMap<&PlSmallStr, &DataType> = source_coord_fields
        .iter()
        .map(|f| (&f.name, f.dtype()))
        .collect();

    let mut out_fields: Vec<Field> =
        Vec::with_capacity(target_fields.len() + group_fields.len() + value_fields.len());
    for f in target_fields {
        if let Some(&src_dt) = source_coord_dtype_map.get(&f.name) {
            if let Some(reconciled) = reconcile_temporal_dtype(src_dt, f.dtype()) {
                out_fields.push(Field::new(f.name.clone(), reconciled));
                continue;
            }
        }
        out_fields.push(f.clone());
    }
    for f in &group_fields {
        if out_fields.iter().any(|existing| existing.name == f.name) {
            polars_bail!(
                InvalidOperation:
                "duplicate output field name {} (present multiple times)",
                f.name
            );
        }
        out_fields.push(f.clone());
    }
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

    Ok(Field::new(
        "interpolated".into(),
        DataType::Struct(out_fields),
    ))
}

fn series_to_f64_vec(s: &Series) -> PolarsResult<Vec<f64>> {
    let s = s.cast(&DataType::Float64)?;
    let ca = s.f64()?;
    ca.into_iter()
        .map(|opt| {
            opt.ok_or_else(
                || polars_err!(ComputeError: "null not supported in interpolation inputs"),
            )
        })
        .collect()
}

fn is_missing(v: Option<f64>) -> bool {
    match v {
        None => true,
        Some(x) => x.is_nan(),
    }
}

fn finest_time_unit(tu1: TimeUnit, tu2: TimeUnit) -> TimeUnit {
    match (tu1, tu2) {
        (TimeUnit::Nanoseconds, _) | (_, TimeUnit::Nanoseconds) => TimeUnit::Nanoseconds,
        (TimeUnit::Microseconds, _) | (_, TimeUnit::Microseconds) => TimeUnit::Microseconds,
        _ => TimeUnit::Milliseconds,
    }
}

fn reconcile_temporal_dtype(dt1: &DataType, dt2: &DataType) -> Option<DataType> {
    if dt1 == dt2 {
        return None;
    }
    match (dt1, dt2) {
        (DataType::Datetime(tu1, tz1), DataType::Datetime(tu2, tz2)) => {
            let tu = finest_time_unit(*tu1, *tu2);
            let tz = tz1.as_ref().or(tz2.as_ref()).cloned();
            Some(DataType::Datetime(tu, tz))
        }
        (DataType::Duration(tu1), DataType::Duration(tu2)) => {
            Some(DataType::Duration(finest_time_unit(*tu1, *tu2)))
        }
        (DataType::Date, DataType::Datetime(tu, tz))
        | (DataType::Datetime(tu, tz), DataType::Date) => Some(DataType::Datetime(*tu, tz.clone())),
        _ => None,
    }
}

/// Pre-process source coordinate and value fields to handle NaN/Null according to `handle_missing`.
fn preprocess_sources(
    coord_fields: Vec<Series>,
    value_fields: Vec<Series>,
    handle_missing: &MissingHandling,
    fill_value: Option<f64>,
) -> PolarsResult<(Vec<Series>, Vec<Series>)> {
    let n = coord_fields.first().map_or(0, |s| s.len());

    let mut coord_valid = vec![true; n];
    for s in &coord_fields {
        let s_f64 = s.cast(&DataType::Float64)?;
        let ca = s_f64.f64()?;
        for (i, v) in ca.into_iter().enumerate() {
            if is_missing(v) {
                coord_valid[i] = false;
            }
        }
    }

    match handle_missing {
        MissingHandling::Error => {
            if let Some(i) = coord_valid.iter().position(|&v| !v) {
                polars_bail!(
                    ComputeError:
                    "NaN or Null found in source coordinate at row {}",
                    i
                );
            }
            for s in &value_fields {
                let s_f64 = s.cast(&DataType::Float64)?;
                let ca = s_f64.f64()?;
                for (i, v) in ca.into_iter().enumerate() {
                    if is_missing(v) {
                        polars_bail!(
                            ComputeError:
                            "NaN or Null found in source value '{}' at row {}",
                            s.name(),
                            i
                        );
                    }
                }
            }
            Ok((coord_fields, value_fields))
        }
        MissingHandling::Drop => {
            let mut valid = coord_valid;
            for s in &value_fields {
                let s_f64 = s.cast(&DataType::Float64)?;
                let ca = s_f64.f64()?;
                for (i, v) in ca.into_iter().enumerate() {
                    if is_missing(v) {
                        valid[i] = false;
                    }
                }
            }
            let mask = BooleanChunked::from_slice("mask".into(), &valid);
            let coords = coord_fields
                .iter()
                .map(|s| s.filter(&mask))
                .collect::<PolarsResult<Vec<_>>>()?;
            let values = value_fields
                .iter()
                .map(|s| s.filter(&mask))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok((coords, values))
        }
        MissingHandling::Fill => {
            let fv = fill_value.ok_or_else(|| {
                polars_err!(
                    ComputeError:
                    "fill_value is required when handle_missing='fill'"
                )
            })?;
            let mask = BooleanChunked::from_slice("mask".into(), &coord_valid);
            let coords = coord_fields
                .iter()
                .map(|s| s.filter(&mask))
                .collect::<PolarsResult<Vec<_>>>()?;
            let values = value_fields
                .iter()
                .map(|s| {
                    let filtered = s.filter(&mask)?;
                    let f64_s = filtered.cast(&DataType::Float64)?;
                    let ca = f64_s.f64()?;
                    let mut filled: Float64Chunked = ca
                        .into_iter()
                        .map(|opt| if is_missing(opt) { Some(fv) } else { opt })
                        .collect();
                    filled.rename(s.name().clone());
                    Ok(filled.into_series())
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok((coords, values))
        }
        MissingHandling::Nearest => {
            let mask = BooleanChunked::from_slice("mask".into(), &coord_valid);
            let coords = coord_fields
                .iter()
                .map(|s| s.filter(&mask))
                .collect::<PolarsResult<Vec<_>>>()?;
            let filtered_values = value_fields
                .iter()
                .map(|s| s.filter(&mask))
                .collect::<PolarsResult<Vec<_>>>()?;

            let coord_vecs: Vec<Vec<f64>> = coords
                .iter()
                .map(series_to_f64_vec)
                .collect::<PolarsResult<_>>()?;
            let filtered_n = coords.first().map_or(0, |s| s.len());

            let values = filtered_values
                .iter()
                .map(|s| {
                    let f64_s = s.cast(&DataType::Float64)?;
                    let ca = f64_s.f64()?;
                    let vals: Vec<Option<f64>> = ca.into_iter().collect();

                    let valid_indices: Vec<usize> = vals
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| !is_missing(**v))
                        .map(|(i, _)| i)
                        .collect();

                    if valid_indices.is_empty() {
                        polars_bail!(
                            ComputeError:
                            "no valid values in column '{}' for nearest-neighbor fill",
                            s.name()
                        );
                    }

                    let filled: Vec<Option<f64>> = (0..filtered_n)
                        .map(|i| {
                            if !is_missing(vals[i]) {
                                return vals[i];
                            }
                            let mut best_dist = f64::INFINITY;
                            let mut best_val = None;
                            for &j in &valid_indices {
                                let dist: f64 = coord_vecs
                                    .iter()
                                    .map(|cv| {
                                        let d = cv[i] - cv[j];
                                        d * d
                                    })
                                    .sum();
                                if dist < best_dist {
                                    best_dist = dist;
                                    best_val = vals[j];
                                }
                            }
                            best_val
                        })
                        .collect();

                    let mut result: Float64Chunked = filled.into_iter().collect();
                    result.rename(s.name().clone());
                    Ok(result.into_series())
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok((coords, values))
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for both interpolation expression functions
// ---------------------------------------------------------------------------

struct GroupedRows {
    groups: HashMap<Vec<u64>, Vec<usize>>,
    keys: Vec<Vec<u64>>,
    first_row: HashMap<Vec<u64>, usize>,
}

/// Group source rows by the specified dimension indices.
fn build_groups(
    coord_cols: &[Vec<f64>],
    group_dim_indices: &[usize],
    source_n: usize,
) -> PolarsResult<GroupedRows> {
    let mut groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    if group_dim_indices.is_empty() {
        groups.insert(Vec::new(), (0..source_n).collect());
    } else {
        for row_idx in 0..source_n {
            let key: Vec<u64> = group_dim_indices
                .iter()
                .map(|&d| coord_cols[d][row_idx].to_bits())
                .collect();
            groups.entry(key).or_default().push(row_idx);
        }
    }

    let mut keys: Vec<Vec<u64>> = groups.keys().cloned().collect();
    keys.sort();

    let mut first_row: HashMap<Vec<u64>, usize> = HashMap::with_capacity(groups.len());
    for (k, rows) in &groups {
        let first = *rows
            .first()
            .ok_or_else(|| polars_err!(ComputeError: "empty group"))?;
        first_row.insert(k.clone(), first);
    }

    Ok(GroupedRows {
        groups,
        keys,
        first_row,
    })
}

/// Build output struct series from computed value columns.
fn build_output_struct(
    target_fields: &[Series],
    coord_fields: &[Series],
    group_dim_names: &[String],
    group_dim_indices: &[usize],
    value_names: &[String],
    out_value_cols: Vec<Vec<f64>>,
    grouped: &GroupedRows,
    target_n: usize,
) -> PolarsResult<Series> {
    let group_count = grouped.keys.len();
    let out_n = target_n * group_count;
    let mut out_fields: Vec<Series> =
        Vec::with_capacity(target_fields.len() + group_dim_names.len() + value_names.len());

    for s in target_fields {
        let name = s.name();
        if group_dim_names.iter().any(|g| g == name) || value_names.iter().any(|v| v == name) {
            polars_bail!(
                InvalidOperation:
                "duplicate output field name {} (present in interp_target and group/value fields)",
                name
            );
        }
        let mut out = s.clone();
        for _ in 1..group_count {
            out.append(s)?;
        }
        out_fields.push(out);
    }

    for (pos, name) in group_dim_names.iter().enumerate() {
        if value_names.iter().any(|v| v == name) {
            polars_bail!(
                InvalidOperation:
                "duplicate output field name {} (present in coords group field and values)",
                name
            );
        }
        let src_series = &coord_fields[group_dim_indices[pos]];
        let dtype = src_series.dtype().clone();

        let mut vals: Vec<AnyValue<'static>> = Vec::with_capacity(out_n);
        for gkey in &grouped.keys {
            let first_row = *grouped
                .first_row
                .get(gkey)
                .ok_or_else(|| polars_err!(ComputeError: "missing group representative row"))?;
            let av = src_series.get(first_row)?.into_static();
            vals.extend(std::iter::repeat_n(av, target_n));
        }
        out_fields.push(Series::from_any_values_and_dtype(
            name.as_str().into(),
            &vals,
            &dtype,
            true,
        )?);
    }

    for (name, vals) in value_names.iter().zip(out_value_cols.into_iter()) {
        out_fields.push(Series::new(name.as_str().into(), vals));
    }
    Ok(StructChunked::from_series("interpolated".into(), out_n, out_fields.iter())?.into_series())
}

// ---------------------------------------------------------------------------
// interpolate_nd
// ---------------------------------------------------------------------------

#[polars_expr(output_type_func=same_output_type)]
fn interpolate_nd(inputs: &[Series], kwargs: InterpolateKwargs) -> PolarsResult<Series> {
    let source_coords = inputs[0].struct_()?;
    let source_values = inputs[1].struct_()?;
    let interp_target = inputs[2].struct_()?;

    let coord_fields_raw = source_coords.fields_as_series();
    if coord_fields_raw.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 coordinate field for interpolation");
    }
    let value_fields_raw = source_values.fields_as_series();
    if value_fields_raw.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 value field for interpolation");
    }

    let (mut coord_fields, value_fields) = preprocess_sources(
        coord_fields_raw,
        value_fields_raw,
        &kwargs.handle_missing,
        kwargs.fill_value,
    )?;

    let mut target_fields = interp_target.fields_as_series();
    let target_name_set: std::collections::HashSet<PlSmallStr> =
        target_fields.iter().map(|s| s.name().clone()).collect();

    let mut interp_dim_names: Vec<String> = Vec::new();
    let mut interp_dim_indices: Vec<usize> = Vec::new();
    let mut group_dim_names: Vec<String> = Vec::new();
    let mut group_dim_indices: Vec<usize> = Vec::new();

    for (idx, s) in coord_fields.iter().enumerate() {
        if target_name_set.contains(s.name()) {
            interp_dim_indices.push(idx);
            interp_dim_names.push(s.name().to_string());
        } else {
            group_dim_indices.push(idx);
            group_dim_names.push(s.name().to_string());
        }
    }

    if interp_dim_indices.is_empty() {
        polars_bail!(
            InvalidOperation:
            "no interpolation dimensions found: none of the source coord fields are present in interp_target"
        );
    }

    // Reconcile temporal precision between source and target coordinate dimensions.
    {
        let mut reconcile_map: HashMap<String, DataType> = HashMap::new();
        for (pos, &d) in interp_dim_indices.iter().enumerate() {
            let target_s = target_fields
                .iter()
                .find(|s| s.name() == interp_dim_names[pos].as_str())
                .ok_or_else(|| {
                    polars_err!(
                        InvalidOperation: "interp_target missing field {}",
                        interp_dim_names[pos]
                    )
                })?;
            if let Some(reconciled) =
                reconcile_temporal_dtype(coord_fields[d].dtype(), target_s.dtype())
            {
                reconcile_map.insert(interp_dim_names[pos].clone(), reconciled);
            }
        }
        if !reconcile_map.is_empty() {
            for (pos, &d) in interp_dim_indices.iter().enumerate() {
                if let Some(dt) = reconcile_map.get(&interp_dim_names[pos]) {
                    coord_fields[d] = coord_fields[d].cast(dt)?;
                }
            }
            for s in &mut target_fields {
                if let Some(dt) = reconcile_map.get(s.name().as_str()) {
                    *s = s.cast(dt)?;
                }
            }
        }
    }

    let coord_cols_all: Vec<Vec<f64>> = coord_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    let source_n = coord_fields.first().map_or(0, |s| s.len());
    let grouped = build_groups(&coord_cols_all, &group_dim_indices, source_n)?;

    let value_names: Vec<String> = value_fields.iter().map(|s| s.name().to_string()).collect();
    let value_cols: Vec<Vec<f64>> = value_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    let target_n = interp_target.len();
    let mut target_coord_cols: Vec<Vec<f64>> = Vec::with_capacity(interp_dim_indices.len());
    for name in &interp_dim_names {
        let s = target_fields
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| polars_err!(InvalidOperation: "interp_target missing field {}", name))?;
        target_coord_cols.push(series_to_f64_vec(s)?);
    }

    let mut out_value_cols: Vec<Vec<f64>> = value_names
        .iter()
        .map(|_| Vec::with_capacity(target_n * grouped.keys.len()))
        .collect();

    let interp_dims = interp_dim_indices.len();

    for group_key in &grouped.keys {
        let rows = grouped.groups.get(group_key).expect("group key missing");

        let mut axes: Vec<Vec<f64>> = Vec::with_capacity(interp_dims);
        for &d in &interp_dim_indices {
            let mut axis: Vec<f64> = rows.iter().map(|&ri| coord_cols_all[d][ri]).collect();
            axis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            axis.dedup();
            if axis.is_empty() {
                polars_bail!(ComputeError: "empty axis after dedup; cannot interpolate");
            }
            axes.push(axis);
        }

        let mut coord_map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(rows.len());
        for &row_idx in rows {
            let key: Vec<u64> = interp_dim_indices
                .iter()
                .map(|&d| coord_cols_all[d][row_idx].to_bits())
                .collect();
            coord_map.insert(key, row_idx);
        }

        let total_grid: usize = axes.iter().map(|a| a.len()).product();

        let axis_idx_maps: Vec<HashMap<u64, usize>> = axes
            .iter()
            .map(|axis| {
                axis.iter()
                    .enumerate()
                    .map(|(i, &v)| (v.to_bits(), i))
                    .collect()
            })
            .collect();

        let mut strides = vec![1usize; interp_dims];
        for d in (0..interp_dims.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * axes[d + 1].len();
        }

        let grids: Vec<Vec<f64>> = value_cols
            .iter()
            .map(|col| {
                let mut grid = vec![f64::NAN; total_grid];
                for (key, &src_row) in &coord_map {
                    let flat_idx: usize = (0..interp_dims)
                        .map(|d| axis_idx_maps[d][&key[d]] * strides[d])
                        .sum();
                    grid[flat_idx] = col[src_row];
                }
                grid
            })
            .collect();

        for (vi, grid) in grids.iter().enumerate() {
            if grid.iter().any(|v| v.is_nan()) {
                polars_bail!(
                    ComputeError:
                    "source grid missing points for value '{}'; ensure source is a full cartesian grid within each group",
                    value_names[vi]
                );
            }
        }

        let axis_refs: Vec<&[f64]> = axes.iter().map(|a| a.as_slice()).collect();

        for row in 0..target_n {
            let target_pt: Vec<f64> = (0..interp_dims)
                .map(|d| target_coord_cols[d][row])
                .collect();

            for (vi, grid) in grids.iter().enumerate() {
                let val = interpolation::interpolate_grid(
                    &axis_refs,
                    grid,
                    &target_pt,
                    kwargs.method,
                    kwargs.extrapolate,
                );
                out_value_cols[vi].push(val);
            }
        }
    }

    build_output_struct(
        &target_fields,
        &coord_fields,
        &group_dim_names,
        &group_dim_indices,
        &value_names,
        out_value_cols,
        &grouped,
        target_n,
    )
}

// ---------------------------------------------------------------------------
// Geospatial interpolation
// ---------------------------------------------------------------------------

fn default_geospatial_method() -> GeospatialMethod {
    GeospatialMethod::TensorProduct
}

fn default_lon_range() -> LonRange {
    LonRange::Auto
}

#[derive(Deserialize)]
struct InterpGeospatialKwargs {
    handle_missing: MissingHandling,
    fill_value: Option<f64>,
    extrapolate: bool,
    #[serde(default = "default_geospatial_method")]
    method: GeospatialMethod,
    #[serde(default = "default_method")]
    tensor_method: InterpolationMethod,
    power: Option<f64>,
    k_neighbors: Option<usize>,
    rbf_kernel: Option<RbfKernel>,
    rbf_epsilon: Option<f64>,
    #[serde(default = "default_lon_range")]
    lon_range: LonRange,
}

/// Build a lat/lon grid from grouped source rows.  Shared by `tensor_product`
/// and `slerp` paths.
fn build_latlon_grid(
    rows: &[usize],
    coord_cols: &[Vec<f64>],
    value_cols: &[Vec<f64>],
    value_names: &[String],
    lon_range: &LonRange,
) -> PolarsResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
    let mut lat_axis: Vec<f64> = rows.iter().map(|&ri| coord_cols[0][ri]).collect();
    lat_axis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    lat_axis.dedup();
    if lat_axis.is_empty() {
        polars_bail!(ComputeError: "empty latitude axis");
    }

    let raw_lons: Vec<f64> = rows.iter().map(|&ri| coord_cols[1][ri]).collect();
    let (lon_axis, _lon_shift) = geospatial::normalize_longitudes(&raw_lons, lon_range);
    if lon_axis.is_empty() {
        polars_bail!(ComputeError: "empty longitude axis");
    }

    let n_lat = lat_axis.len();
    let n_lon = lon_axis.len();

    let lon_base = lon_axis[0];
    let mut coord_map: HashMap<(u64, u64), usize> = HashMap::with_capacity(rows.len());
    for &row_idx in rows {
        let lat_val = coord_cols[0][row_idx];
        let lon_val = geospatial::normalize_target_lon(
            geospatial::normalize_lon(coord_cols[1][row_idx], lon_range),
            lon_base,
        );
        coord_map.insert((lat_val.to_bits(), lon_val.to_bits()), row_idx);
    }

    let lat_idx_map: HashMap<u64, usize> = lat_axis
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.to_bits(), i))
        .collect();
    let lon_idx_map: HashMap<u64, usize> = lon_axis
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.to_bits(), i))
        .collect();

    let total_grid = n_lat * n_lon;
    let grids: Vec<Vec<f64>> = value_cols
        .iter()
        .enumerate()
        .map(|(vi, col)| {
            let mut grid = vec![f64::NAN; total_grid];
            for (&(lat_b, lon_b), &src_row) in &coord_map {
                if let (Some(&li), Some(&loi)) =
                    (lat_idx_map.get(&lat_b), lon_idx_map.get(&lon_b))
                {
                    grid[li * n_lon + loi] = col[src_row];
                }
            }
            if grid.iter().any(|v| v.is_nan()) {
                polars_bail!(
                    ComputeError:
                    "source grid missing points for value '{}'; ensure source is a full cartesian grid within each group",
                    value_names[vi]
                );
            }
            Ok(grid)
        })
        .collect::<PolarsResult<_>>()?;

    Ok((lat_axis, lon_axis, grids))
}

#[polars_expr(output_type_func=same_output_type)]
fn interpolate_geospatial(
    inputs: &[Series],
    kwargs: InterpGeospatialKwargs,
) -> PolarsResult<Series> {
    let source_coords = inputs[0].struct_()?;
    let source_values = inputs[1].struct_()?;
    let interp_target = inputs[2].struct_()?;

    let coord_fields_raw = source_coords.fields_as_series();
    if coord_fields_raw.len() < 2 {
        polars_bail!(
            InvalidOperation:
            "interpolate_geospatial requires at least 2 coordinate fields (lat, lon), got {}",
            coord_fields_raw.len()
        );
    }
    let value_fields_raw = source_values.fields_as_series();
    if value_fields_raw.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 value field for interpolation");
    }

    let (coord_fields, value_fields) = preprocess_sources(
        coord_fields_raw,
        value_fields_raw,
        &kwargs.handle_missing,
        kwargs.fill_value,
    )?;

    let lat_name = coord_fields[0].name().to_string();
    let lon_name = coord_fields[1].name().to_string();

    let target_fields = interp_target.fields_as_series();
    let target_name_set: std::collections::HashSet<PlSmallStr> =
        target_fields.iter().map(|s| s.name().clone()).collect();

    if !target_name_set.contains(lat_name.as_str()) {
        polars_bail!(
            InvalidOperation: "interp_target missing latitude field '{}'", lat_name
        );
    }
    if !target_name_set.contains(lon_name.as_str()) {
        polars_bail!(
            InvalidOperation: "interp_target missing longitude field '{}'", lon_name
        );
    }

    let mut group_dim_names: Vec<String> = Vec::new();
    let mut group_dim_indices: Vec<usize> = Vec::new();
    for (idx, s) in coord_fields.iter().enumerate().skip(2) {
        if !target_name_set.contains(s.name()) {
            group_dim_indices.push(idx);
            group_dim_names.push(s.name().to_string());
        }
    }

    let coord_cols_all: Vec<Vec<f64>> = coord_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    let resolved_lon_range = geospatial::resolve_lon_range(&coord_cols_all[1], &kwargs.lon_range);

    let source_n = coord_fields.first().map_or(0, |s| s.len());
    let grouped = build_groups(&coord_cols_all, &group_dim_indices, source_n)?;

    let value_names: Vec<String> = value_fields.iter().map(|s| s.name().to_string()).collect();
    let value_cols: Vec<Vec<f64>> = value_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    let target_n = interp_target.len();
    let target_lat_vec = {
        let s = target_fields
            .iter()
            .find(|s| s.name() == lat_name.as_str())
            .unwrap();
        series_to_f64_vec(s)?
    };
    let target_lon_vec = {
        let s = target_fields
            .iter()
            .find(|s| s.name() == lon_name.as_str())
            .unwrap();
        series_to_f64_vec(s)?
    };

    let mut out_value_cols: Vec<Vec<f64>> = value_names
        .iter()
        .map(|_| Vec::with_capacity(target_n * grouped.keys.len()))
        .collect();

    match kwargs.method {
        GeospatialMethod::TensorProduct | GeospatialMethod::Slerp => {
            let is_tensor = matches!(kwargs.method, GeospatialMethod::TensorProduct);

            for group_key in &grouped.keys {
                let rows = grouped.groups.get(group_key).expect("group key missing");
                let (lat_axis, lon_axis, grids) = build_latlon_grid(
                    rows,
                    &coord_cols_all,
                    &value_cols,
                    &value_names,
                    &resolved_lon_range,
                )?;

                let (final_lon_axis, final_grids) =
                    geospatial::preprocess_geo_grids(&lat_axis, &lon_axis, grids);

                let lon_norm_base = lon_axis[0];
                for row in 0..target_n {
                    let target_lat = target_lat_vec[row];
                    let target_lon = geospatial::normalize_target_lon(
                        geospatial::normalize_lon(target_lon_vec[row], &resolved_lon_range),
                        lon_norm_base,
                    );

                    for (vi, grid) in final_grids.iter().enumerate() {
                        let val = if is_tensor {
                            let axis_refs: Vec<&[f64]> =
                                vec![lat_axis.as_slice(), final_lon_axis.as_slice()];
                            interpolation::interpolate_grid(
                                &axis_refs,
                                grid,
                                &[target_lat, target_lon],
                                kwargs.tensor_method,
                                kwargs.extrapolate,
                            )
                        } else {
                            geospatial::slerp_bilinear(
                                &lat_axis,
                                &final_lon_axis,
                                grid,
                                target_lat,
                                target_lon,
                                kwargs.extrapolate,
                            )
                        };
                        out_value_cols[vi].push(val);
                    }
                }
            }
        }

        GeospatialMethod::Idw | GeospatialMethod::Rbf => {
            let power = kwargs.power.unwrap_or(2.0);
            let k =
                kwargs
                    .k_neighbors
                    .unwrap_or(if matches!(kwargs.method, GeospatialMethod::Rbf) {
                        20
                    } else {
                        0
                    });
            let kernel = kwargs.rbf_kernel.unwrap_or(RbfKernel::ThinPlateSpline);
            let epsilon = kwargs.rbf_epsilon.unwrap_or(f64::NAN);
            let is_idw = matches!(kwargs.method, GeospatialMethod::Idw);

            for group_key in &grouped.keys {
                let rows = grouped.groups.get(group_key).expect("group key missing");

                let src_lats: Vec<f64> = rows.iter().map(|&ri| coord_cols_all[0][ri]).collect();
                let src_lons: Vec<f64> = rows
                    .iter()
                    .map(|&ri| {
                        geospatial::normalize_lon(coord_cols_all[1][ri], &resolved_lon_range)
                    })
                    .collect();

                for row in 0..target_n {
                    let target_lat = target_lat_vec[row];
                    let target_lon =
                        geospatial::normalize_lon(target_lon_vec[row], &resolved_lon_range);

                    for (vi, col) in value_cols.iter().enumerate() {
                        let src_vals: Vec<f64> = rows.iter().map(|&ri| col[ri]).collect();
                        let val = if is_idw {
                            geospatial::idw_haversine(
                                &src_lats, &src_lons, &src_vals, target_lat, target_lon, power, k,
                            )
                        } else {
                            geospatial::rbf_haversine(
                                &src_lats, &src_lons, &src_vals, target_lat, target_lon, kernel,
                                epsilon, k,
                            )
                        };
                        out_value_cols[vi].push(val);
                    }
                }
            }
        }
    }

    build_output_struct(
        &target_fields,
        &coord_fields,
        &group_dim_names,
        &group_dim_indices,
        &value_names,
        out_value_cols,
        &grouped,
        target_n,
    )
}
