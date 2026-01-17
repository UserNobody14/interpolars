use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::collections::HashMap;

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

    // Coordinates to interpolate over are the intersection between source coord fields
    // and the target schema fields. Any remaining source coord fields are treated as
    // "group dims" (e.g. time) and are appended to the output.
    let target_names: std::collections::HashSet<PlSmallStr> =
        target_fields.iter().map(|f| f.name.clone()).collect();

    let group_fields: Vec<Field> = source_coord_fields
        .iter()
        .filter(|f| !target_names.contains(&f.name))
        .cloned()
        .collect();

    let mut out_fields: Vec<Field> = Vec::with_capacity(
        target_fields.len() + group_fields.len() + value_fields.len(),
    );
    out_fields.extend(target_fields.iter().cloned());
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
fn interpolate_nd(inputs: &[Series]) -> PolarsResult<Series> {
    // inputs[0]: source coordinates as a Struct (e.g. {xfield, yfield, ...})
    // inputs[1]: source values as a Struct (e.g. {valuefield, ...})
    // inputs[2]: target rows as a Struct Series where each field is a column from interp_target
    //          (coordinates + any passthrough metadata columns).

    let source_coords = inputs[0].struct_()?;
    let source_values = inputs[1].struct_()?;
    let interp_target = inputs[2].struct_()?;

    let coord_fields = source_coords.fields_as_series();
    if coord_fields.is_empty() {
        polars_bail!(InvalidOperation: "expected at least 1 coordinate field for interpolation");
    }

    // Split source coord fields into:
    // - interp dims: coord fields present in target (we interpolate across these)
    // - group dims: coord fields missing from target (we group by these)
    let target_fields = interp_target.fields_as_series();
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

    // Materialize all coord columns as f64 so we can do grouping + interpolation math.
    // Note: nulls are not supported (see `series_to_f64_vec`).
    let coord_cols_all: Vec<Vec<f64>> = coord_fields
        .iter()
        .map(series_to_f64_vec)
        .collect::<PolarsResult<_>>()?;

    // Group source rows by the "group dims" (extra coord fields missing from target).
    let source_n = source_coords.len();
    let mut groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    if group_dim_indices.is_empty() {
        groups.insert(Vec::new(), (0..source_n).collect());
    } else {
        for row_idx in 0..source_n {
            let mut key: Vec<u64> = Vec::with_capacity(group_dim_indices.len());
            for &d in &group_dim_indices {
                key.push(coord_cols_all[d][row_idx].to_bits());
            }
            groups.entry(key).or_default().push(row_idx);
        }
    }

    // Deterministic output ordering: sort groups by their group key (lexicographic).
    let mut group_keys: Vec<Vec<u64>> = groups.keys().cloned().collect();
    group_keys.sort();

    // For dtype-preserving output of group dims, capture one representative row index per group.
    // (All rows in a group share the same group-dim values by construction.)
    let mut group_first_row: HashMap<Vec<u64>, usize> = HashMap::with_capacity(groups.len());
    for (k, rows) in &groups {
        let first = *rows.first().ok_or_else(|| polars_err!(ComputeError: "empty group"))?;
        group_first_row.insert(k.clone(), first);
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

    // Extract target coordinates from fields on interp_target struct series.
    let target_n = interp_target.len();
    let mut target_coord_cols: Vec<Vec<f64>> = Vec::with_capacity(interp_dim_indices.len());
    for name in &interp_dim_names {
        let s = target_fields.iter().find(|s| s.name() == name).ok_or_else(|| {
            polars_err!(InvalidOperation: "interp_target missing field {}", name)
        })?;
        target_coord_cols.push(series_to_f64_vec(s)?);
    }

    // Interpolate each target row for every group.
    let mut out_value_cols: Vec<Vec<f64>> = value_names
        .iter()
        .map(|_| Vec::with_capacity(target_n * group_keys.len()))
        .collect();

    let interp_dims = interp_dim_indices.len();
    for group_key in &group_keys {
        let rows = groups.get(group_key).expect("group key missing");

        // Build axes (unique sorted coordinate values per interpolation dimension) within this group.
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

        // Map coordinate tuple -> row index in source (within this group).
        let mut coord_map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(rows.len());
        for &row_idx in rows {
            let mut key: Vec<u64> = Vec::with_capacity(interp_dims);
            for (pos, &d) in interp_dim_indices.iter().enumerate() {
                let _ = pos; // positional alignment with axes
                key.push(coord_cols_all[d][row_idx].to_bits());
            }
            coord_map.insert(key, row_idx);
        }

        for row in 0..target_n {
            let mut lo_idx: Vec<usize> = Vec::with_capacity(interp_dims);
            let mut hi_idx: Vec<usize> = Vec::with_capacity(interp_dims);
            let mut tvals: Vec<f64> = Vec::with_capacity(interp_dims);

            for d in 0..interp_dims {
                let t = target_coord_cols[d][row];
                tvals.push(t);
                let (lo, hi) = lower_upper_indices(&axes[d], t);
                lo_idx.push(lo);
                hi_idx.push(hi);
            }

            let corners = 1usize << interp_dims;
            let mut sums: Vec<f64> = vec![0.0; value_names.len()];

            for corner in 0..corners {
                let mut weight = 1.0f64;
                let mut key: Vec<u64> = Vec::with_capacity(interp_dims);

                for d in 0..interp_dims {
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
                        polars_bail!(
                            ComputeError:
                            "zero grid spacing for axis {}",
                            interp_dim_names[d]
                        );
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
                        "source grid missing corner point for key {:?}; ensure source is a full cartesian grid within each group",
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
    }

    // Build output struct series:
    // - passthrough columns from interp_target (repeated once per group)
    // - group columns (one per extra coord dim, repeated per target row)
    // - interpolated value columns
    let group_count = group_keys.len();
    let out_n = target_n * group_count;
    let mut out_fields: Vec<Series> =
        Vec::with_capacity(target_fields.len() + group_dim_names.len() + value_names.len());

    // Repeat target fields for each group (preserve dtype).
    for s in &target_fields {
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

    // Add group fields (preserve dtype) repeated for each target row, per group.
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
        for gkey in &group_keys {
            let first_row = *group_first_row
                .get(gkey)
                .ok_or_else(|| polars_err!(ComputeError: "missing group representative row"))?;
            let av = src_series
                .get(first_row)?
                .into_static();
            vals.extend(std::iter::repeat_n(av, target_n));
        }
        out_fields.push(Series::from_any_values_and_dtype(
            name.as_str().into(),
            &vals,
            &dtype,
            true,
        )?);
    }

    // Add interpolated value columns.
    for (name, vals) in value_names.iter().zip(out_value_cols.into_iter()) {
        out_fields.push(Series::new(name.as_str().into(), vals));
    }
    Ok(
        StructChunked::from_series("interpolated".into(), out_n, out_fields.iter())?
            .into_series(),
    )
}

