//! Geospatial interpolation: spherically-aware types and algorithms.
//!
//! Provides longitude normalization, Haversine distance, SLERP bilinear
//! interpolation on lat/lon grids, IDW and RBF interpolation using
//! great-circle distance, and lat/lon grid preprocessing helpers
//! (wrapping detection, ghost-point extension, pole averaging).

use serde::Deserialize;

use crate::interpolation::find_interval;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum GeospatialMethod {
    TensorProduct,
    Slerp,
    Idw,
    Rbf,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum LonRange {
    #[serde(rename = "signed_180")]
    Signed180,
    #[serde(rename = "unsigned_360")]
    Unsigned360,
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum RbfKernel {
    Linear,
    ThinPlateSpline,
    Cubic,
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
}

impl RbfKernel {
    pub fn eval(self, d: f64, epsilon: f64) -> f64 {
        match self {
            Self::Linear => d,
            Self::ThinPlateSpline => {
                if d < 1e-18 {
                    0.0
                } else {
                    d * d * d.ln()
                }
            }
            Self::Cubic => d * d * d,
            Self::Gaussian => (-(d / epsilon).powi(2)).exp(),
            Self::Multiquadric => (1.0 + (d / epsilon).powi(2)).sqrt(),
            Self::InverseMultiquadric => 1.0 / (1.0 + (d / epsilon).powi(2)).sqrt(),
        }
    }
}

// ---------------------------------------------------------------------------
// Longitude normalization
// ---------------------------------------------------------------------------

fn normalize_lon_signed(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

fn normalize_lon_unsigned(lon: f64) -> f64 {
    lon.rem_euclid(360.0)
}

/// Normalize a longitude value to the canonical range specified by `range`.
pub fn normalize_lon(lon: f64, range: &LonRange) -> f64 {
    match range {
        LonRange::Signed180 | LonRange::Auto => normalize_lon_signed(lon),
        LonRange::Unsigned360 => normalize_lon_unsigned(lon),
    }
}

/// Resolve `Auto` to a concrete `LonRange` based on the source data.
pub fn resolve_lon_range(lons: &[f64], range: &LonRange) -> LonRange {
    match range {
        LonRange::Signed180 | LonRange::Unsigned360 => *range,
        LonRange::Auto => {
            if lons.iter().any(|&v| v < 0.0) {
                LonRange::Signed180
            } else {
                LonRange::Unsigned360
            }
        }
    }
}

/// Normalize a target longitude into `[base, base + 360)`.
pub fn normalize_target_lon(lon: f64, base: f64) -> f64 {
    base + (lon - base).rem_euclid(360.0)
}

// ---------------------------------------------------------------------------
// Lat/lon grid helpers
// ---------------------------------------------------------------------------

/// Normalize source longitudes to form a contiguous sorted axis.
///
/// Returns `(sorted_unique_axis, shift_offset)`.  `shift_offset` is the value
/// added to longitudes that were on the left side of an IDL-crossing gap so
/// callers can apply the same shift to raw source data.
pub fn normalize_longitudes(lons: &[f64], lon_range: &LonRange) -> (Vec<f64>, f64) {
    let mut normed: Vec<f64> = lons.iter().map(|&l| normalize_lon(l, lon_range)).collect();
    normed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    normed.dedup();

    let n = normed.len();
    if n <= 1 {
        return (normed, 0.0);
    }

    let mut max_gap = 0.0_f64;
    let mut max_gap_idx: usize = n;
    for i in 0..n - 1 {
        let gap = normed[i + 1] - normed[i];
        if gap > max_gap {
            max_gap = gap;
            max_gap_idx = i;
        }
    }
    let wrap_gap = (normed[0] + 360.0) - normed[n - 1];
    if wrap_gap >= max_gap {
        max_gap_idx = n;
    }

    if max_gap_idx < n {
        let split_threshold = normed[max_gap_idx] + max_gap / 2.0;
        let shift = 360.0;
        for v in &mut normed {
            if *v < split_threshold {
                *v += shift;
            }
        }
        normed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        normed.dedup();
        return (normed, shift);
    }

    (normed, 0.0)
}

/// Returns `true` when the sorted longitude axis covers a full 360° cycle.
pub fn detect_wrapping(lon_axis: &[f64]) -> bool {
    let n = lon_axis.len();
    if n < 3 {
        return false;
    }
    let mut gaps: Vec<f64> = (0..n - 1).map(|i| lon_axis[i + 1] - lon_axis[i]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_gap = gaps[gaps.len() / 2];

    let wrap_gap = (lon_axis[0] + 360.0) - lon_axis[n - 1];
    wrap_gap <= 1.5 * median_gap
}

/// Extend a 2-D grid's longitude dimension with ghost points for periodic
/// wrapping.  The grid is in row-major `[n_lat, n_lon]` layout.
pub fn extend_lon_grid(n_lat: usize, lon_axis: &[f64], grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n_lon = lon_axis.len();
    let k = n_lon.saturating_sub(1).min(3);
    if k == 0 {
        return (lon_axis.to_vec(), grid.to_vec());
    }

    let n_ext = n_lon + 2 * k;
    let mut ext_axis = Vec::with_capacity(n_ext);
    for i in (n_lon - k)..n_lon {
        ext_axis.push(lon_axis[i] - 360.0);
    }
    ext_axis.extend_from_slice(lon_axis);
    for i in 0..k {
        ext_axis.push(lon_axis[i] + 360.0);
    }

    let mut ext_grid = Vec::with_capacity(n_lat * n_ext);
    for lat_i in 0..n_lat {
        let row = &grid[lat_i * n_lon..(lat_i + 1) * n_lon];
        for i in (n_lon - k)..n_lon {
            ext_grid.push(row[i]);
        }
        ext_grid.extend_from_slice(row);
        for i in 0..k {
            ext_grid.push(row[i]);
        }
    }

    (ext_axis, ext_grid)
}

/// If the first or last latitude is exactly ±90, replace that entire row with
/// the mean across all longitudes (physically a single point at the pole).
pub fn average_pole_rows(lat_axis: &[f64], n_lon: usize, grid: &mut [f64]) {
    let n_lat = lat_axis.len();
    if n_lat == 0 || n_lon == 0 {
        return;
    }
    let pole_tolerance = 1e-10;

    if (lat_axis[0].abs() - 90.0).abs() < pole_tolerance {
        let mean = grid[..n_lon].iter().sum::<f64>() / n_lon as f64;
        for v in &mut grid[..n_lon] {
            *v = mean;
        }
    }
    if (lat_axis[n_lat - 1].abs() - 90.0).abs() < pole_tolerance {
        let start = (n_lat - 1) * n_lon;
        let mean = grid[start..start + n_lon].iter().sum::<f64>() / n_lon as f64;
        for v in &mut grid[start..start + n_lon] {
            *v = mean;
        }
    }
}

/// Apply pole averaging and (if global) longitude wrapping extension to a set
/// of grids.  Returns `(final_lon_axis, processed_grids)`.
pub fn preprocess_geo_grids(
    lat_axis: &[f64],
    lon_axis: &[f64],
    mut grids: Vec<Vec<f64>>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n_lat = lat_axis.len();
    let n_lon = lon_axis.len();

    for grid in &mut grids {
        average_pole_rows(lat_axis, n_lon, grid);
    }

    if detect_wrapping(lon_axis) {
        let mut ext_grids = Vec::with_capacity(grids.len());
        let mut final_lon = Vec::new();
        for grid in &grids {
            let (el, eg) = extend_lon_grid(n_lat, lon_axis, grid);
            final_lon = el;
            ext_grids.push(eg);
        }
        (final_lon, ext_grids)
    } else {
        (lon_axis.to_vec(), grids)
    }
}

// ---------------------------------------------------------------------------
// Haversine distance
// ---------------------------------------------------------------------------

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Central angle in radians between two points on the unit sphere.
/// All inputs in degrees.
pub fn haversine_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = lat1 * DEG_TO_RAD;
    let phi2 = lat2 * DEG_TO_RAD;
    let dphi = (lat2 - lat1) * DEG_TO_RAD * 0.5;
    let dlam = (lon2 - lon1) * DEG_TO_RAD * 0.5;
    let a = dphi.sin().powi(2) + phi1.cos() * phi2.cos() * dlam.sin().powi(2);
    2.0 * a.sqrt().asin()
}

// ---------------------------------------------------------------------------
// SLERP bilinear interpolation
// ---------------------------------------------------------------------------

/// Angular distance along a parallel at latitude `lat_deg` between two
/// longitudes, in radians.
fn angular_distance_on_parallel(lat_deg: f64, dlon_deg: f64) -> f64 {
    let phi = lat_deg * DEG_TO_RAD;
    let dl = dlon_deg * DEG_TO_RAD;
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let arg = sin_phi * sin_phi + cos_phi * cos_phi * dl.cos();
    arg.clamp(-1.0, 1.0).acos()
}

/// Bilinear interpolation on a lat/lon grid using SLERP-derived angular
/// fraction weights.  The grid is row-major `[n_lat, n_lon]`.
pub fn slerp_bilinear(
    lat_axis: &[f64],
    lon_axis: &[f64],
    grid: &[f64],
    target_lat: f64,
    target_lon: f64,
    extrapolate: bool,
) -> f64 {
    let n_lat = lat_axis.len();
    let n_lon = lon_axis.len();

    if n_lat == 1 && n_lon == 1 {
        return grid[0];
    }

    let i_lat = if n_lat >= 2 {
        find_interval(lat_axis, target_lat)
    } else {
        0
    };
    let i_lon = if n_lon >= 2 {
        find_interval(lon_axis, target_lon)
    } else {
        0
    };

    let lat_lo = lat_axis[i_lat];
    let lat_hi = if n_lat >= 2 {
        lat_axis[i_lat + 1]
    } else {
        lat_lo
    };
    let s = if (lat_hi - lat_lo).abs() < 1e-15 {
        0.0
    } else {
        let raw = (target_lat - lat_lo) / (lat_hi - lat_lo);
        if extrapolate {
            raw
        } else {
            raw.clamp(0.0, 1.0)
        }
    };

    let gv = |li: usize, loi: usize| grid[li * n_lon + loi];

    if n_lon < 2 {
        let v_lo = gv(i_lat, 0);
        let v_hi = if n_lat >= 2 { gv(i_lat + 1, 0) } else { v_lo };
        return (1.0 - s) * v_lo + s * v_hi;
    }

    let lon_lo = lon_axis[i_lon];
    let lon_hi = lon_axis[i_lon + 1];
    let dlon_target = target_lon - lon_lo;
    let dlon_cell = lon_hi - lon_lo;

    let slerp_lon_frac = |lat: f64| -> f64 {
        if dlon_cell.abs() < 1e-15 {
            return 0.0;
        }
        let d_full = angular_distance_on_parallel(lat, dlon_cell);
        if d_full < 1e-18 {
            return dlon_target / dlon_cell;
        }
        let d_t = angular_distance_on_parallel(lat, dlon_target);
        let raw = d_t / d_full;
        if extrapolate {
            raw
        } else {
            raw.clamp(0.0, 1.0)
        }
    };

    let t_bot = slerp_lon_frac(lat_lo);
    let t_top = if n_lat < 2 {
        t_bot
    } else {
        slerp_lon_frac(lat_hi)
    };

    let v00 = gv(i_lat, i_lon);
    let v01 = gv(i_lat, i_lon + 1);
    let v_bot = (1.0 - t_bot) * v00 + t_bot * v01;

    let (v10, v11) = if n_lat >= 2 {
        (gv(i_lat + 1, i_lon), gv(i_lat + 1, i_lon + 1))
    } else {
        (v00, v01)
    };
    let v_top = (1.0 - t_top) * v10 + t_top * v11;

    (1.0 - s) * v_bot + s * v_top
}

// ---------------------------------------------------------------------------
// IDW with Haversine distance
// ---------------------------------------------------------------------------

/// Inverse Distance Weighting using Haversine (great-circle) distance.
pub fn idw_haversine(
    src_lats: &[f64],
    src_lons: &[f64],
    src_vals: &[f64],
    target_lat: f64,
    target_lon: f64,
    power: f64,
    k: usize,
) -> f64 {
    let n = src_lats.len();
    debug_assert_eq!(n, src_lons.len());
    debug_assert_eq!(n, src_vals.len());

    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return src_vals[0];
    }

    let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let d = haversine_rad(target_lat, target_lon, src_lats[i], src_lons[i]);
        if d < 1e-15 {
            return src_vals[i];
        }
        dists.push((d, i));
    }

    let use_k = if k == 0 || k >= n { n } else { k };
    if use_k < n {
        dists.select_nth_unstable_by(use_k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(use_k);
    }

    let mut w_sum = 0.0;
    let mut v_sum = 0.0;
    for &(d, idx) in &dists {
        let w = 1.0 / d.powf(power);
        w_sum += w;
        v_sum += w * src_vals[idx];
    }
    v_sum / w_sum
}

// ---------------------------------------------------------------------------
// RBF with Haversine distance
// ---------------------------------------------------------------------------

/// Solve a dense N×N linear system `A x = b` in-place using Gaussian
/// elimination with partial pivoting.
pub fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            a[row * n + col] = 0.0;
            for j in (col + 1)..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}

/// Local RBF interpolation using Haversine distance.
pub fn rbf_haversine(
    src_lats: &[f64],
    src_lons: &[f64],
    src_vals: &[f64],
    target_lat: f64,
    target_lon: f64,
    kernel: RbfKernel,
    epsilon: f64,
    k: usize,
) -> f64 {
    let n = src_lats.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return src_vals[0];
    }

    let use_k = if k == 0 || k > n { n } else { k };

    let mut dists: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            (
                haversine_rad(target_lat, target_lon, src_lats[i], src_lons[i]),
                i,
            )
        })
        .collect();

    if use_k < n {
        dists.select_nth_unstable_by(use_k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(use_k);
    }

    for &(d, idx) in &dists {
        if d < 1e-15 {
            return src_vals[idx];
        }
    }

    let kk = dists.len();
    let indices: Vec<usize> = dists.iter().map(|&(_, i)| i).collect();
    let target_dists: Vec<f64> = dists.iter().map(|&(d, _)| d).collect();

    let eps = if epsilon.is_finite() && epsilon > 0.0 {
        epsilon
    } else {
        let mut pair_dists: Vec<f64> = Vec::with_capacity(kk * (kk - 1) / 2);
        for i in 0..kk {
            for j in (i + 1)..kk {
                pair_dists.push(haversine_rad(
                    src_lats[indices[i]],
                    src_lons[indices[i]],
                    src_lats[indices[j]],
                    src_lons[indices[j]],
                ));
            }
        }
        if pair_dists.is_empty() {
            1.0
        } else {
            pair_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            pair_dists[pair_dists.len() / 2]
        }
    };

    let mut phi = vec![0.0; kk * kk];
    for i in 0..kk {
        phi[i * kk + i] = kernel.eval(0.0, eps);
        for j in (i + 1)..kk {
            let d = haversine_rad(
                src_lats[indices[i]],
                src_lons[indices[i]],
                src_lats[indices[j]],
                src_lons[indices[j]],
            );
            let v = kernel.eval(d, eps);
            phi[i * kk + j] = v;
            phi[j * kk + i] = v;
        }
    }

    let mut rhs: Vec<f64> = indices.iter().map(|&i| src_vals[i]).collect();

    let weights = match solve_linear_system(&mut phi, &mut rhs, kk) {
        Some(w) => w,
        None => return idw_haversine(src_lats, src_lons, src_vals, target_lat, target_lon, 2.0, k),
    };

    let mut result = 0.0;
    for i in 0..kk {
        result += weights[i] * kernel.eval(target_dists[i], eps);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    // --- Longitude normalization ---

    #[test]
    fn normalize_lon_signed_test() {
        approx_eq(normalize_lon(190.0, &LonRange::Signed180), -170.0, 1e-12);
        approx_eq(normalize_lon(-190.0, &LonRange::Signed180), 170.0, 1e-12);
        approx_eq(normalize_lon(0.0, &LonRange::Signed180), 0.0, 1e-12);
        approx_eq(normalize_lon(360.0, &LonRange::Signed180), 0.0, 1e-12);
    }

    #[test]
    fn normalize_lon_unsigned_test() {
        approx_eq(normalize_lon(-10.0, &LonRange::Unsigned360), 350.0, 1e-12);
        approx_eq(normalize_lon(370.0, &LonRange::Unsigned360), 10.0, 1e-12);
        approx_eq(normalize_lon(0.0, &LonRange::Unsigned360), 0.0, 1e-12);
    }

    #[test]
    fn resolve_lon_range_auto() {
        assert_eq!(
            resolve_lon_range(&[-10.0, 20.0], &LonRange::Auto),
            LonRange::Signed180
        );
        assert_eq!(
            resolve_lon_range(&[10.0, 350.0], &LonRange::Auto),
            LonRange::Unsigned360
        );
    }

    #[test]
    fn normalize_lons_no_idl() {
        let (axis, shift) = normalize_longitudes(&[0.0, 10.0, 20.0, 30.0], &LonRange::Signed180);
        assert_eq!(shift, 0.0);
        assert_eq!(axis, vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn normalize_lons_idl_crossing() {
        let (axis, shift) =
            normalize_longitudes(&[170.0, 175.0, 180.0, -175.0, -170.0], &LonRange::Signed180);
        assert!(shift > 0.0);
        for w in axis.windows(2) {
            assert!(w[1] > w[0], "axis should be sorted ascending: {axis:?}");
        }
        let span = axis.last().unwrap() - axis.first().unwrap();
        assert!(span < 180.0, "span should be small, got {span}");
    }

    #[test]
    fn detect_wrapping_global() {
        let axis: Vec<f64> = (0..72).map(|i| i as f64 * 5.0).collect();
        let (normed, _) = normalize_longitudes(&axis, &LonRange::Signed180);
        assert!(detect_wrapping(&normed));
    }

    #[test]
    fn detect_wrapping_regional() {
        let axis: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
        assert!(!detect_wrapping(&axis));
    }

    #[test]
    fn extend_lon_grid_basic() {
        let lon = vec![0.0, 90.0, 180.0, 270.0];
        let grid = vec![1.0, 2.0, 3.0, 4.0];
        let (ext_lon, ext_grid) = extend_lon_grid(1, &lon, &grid);
        assert_eq!(ext_lon.len(), 4 + 6);
        assert_eq!(ext_grid.len(), ext_lon.len());
        assert_eq!(ext_grid[0], 2.0);
        assert_eq!(ext_grid[1], 3.0);
        assert_eq!(ext_grid[2], 4.0);
    }

    #[test]
    fn average_pole_rows_basic() {
        let lat = vec![-90.0, 0.0, 90.0];
        let n_lon = 4;
        let mut grid = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0,
        ];
        average_pole_rows(&lat, n_lon, &mut grid);
        for i in 0..n_lon {
            approx_eq(grid[i], 2.5, 1e-12);
        }
        for i in 0..n_lon {
            approx_eq(grid[2 * n_lon + i], 25.0, 1e-12);
        }
        assert_eq!(grid[4..8], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn normalize_target_lon_basic() {
        approx_eq(normalize_target_lon(370.0, 0.0), 10.0, 1e-12);
        approx_eq(normalize_target_lon(-10.0, 0.0), 350.0, 1e-12);
        approx_eq(normalize_target_lon(5.0, 0.0), 5.0, 1e-12);
    }

    // --- Haversine ---

    #[test]
    fn haversine_same_point() {
        approx_eq(haversine_rad(45.0, 90.0, 45.0, 90.0), 0.0, 1e-14);
    }

    #[test]
    fn haversine_antipodal() {
        approx_eq(
            haversine_rad(0.0, 0.0, 0.0, 180.0),
            std::f64::consts::PI,
            1e-10,
        );
    }

    #[test]
    fn haversine_quarter_circle() {
        approx_eq(
            haversine_rad(0.0, 0.0, 90.0, 0.0),
            std::f64::consts::FRAC_PI_2,
            1e-10,
        );
    }

    #[test]
    fn haversine_known_distance() {
        let d = haversine_rad(51.5074, -0.1278, 48.8566, 2.3522);
        approx_eq(d, 0.05392, 0.001);
    }

    // --- SLERP bilinear ---

    #[test]
    fn slerp_at_grid_points() {
        let lat = vec![0.0, 10.0];
        let lon = vec![0.0, 10.0];
        let grid = vec![1.0, 2.0, 3.0, 4.0];
        approx_eq(
            slerp_bilinear(&lat, &lon, &grid, 0.0, 0.0, false),
            1.0,
            1e-12,
        );
        approx_eq(
            slerp_bilinear(&lat, &lon, &grid, 0.0, 10.0, false),
            2.0,
            1e-12,
        );
        approx_eq(
            slerp_bilinear(&lat, &lon, &grid, 10.0, 0.0, false),
            3.0,
            1e-12,
        );
        approx_eq(
            slerp_bilinear(&lat, &lon, &grid, 10.0, 10.0, false),
            4.0,
            1e-12,
        );
    }

    #[test]
    fn slerp_center_equator() {
        let lat = vec![0.0, 10.0];
        let lon = vec![0.0, 10.0];
        let grid = vec![0.0, 10.0, 10.0, 20.0];
        approx_eq(
            slerp_bilinear(&lat, &lon, &grid, 5.0, 5.0, false),
            10.0,
            0.1,
        );
    }

    #[test]
    fn slerp_near_pole_differs_from_bilinear() {
        let lat = vec![80.0, 90.0];
        let lon = vec![0.0, 30.0];
        let grid = vec![0.0, 30.0, 0.0, 30.0];
        let v_slerp = slerp_bilinear(&lat, &lon, &grid, 85.0, 15.0, false);
        assert!((v_slerp - 15.0).abs() < 1.0);
    }

    // --- IDW ---

    #[test]
    fn idw_exact_hit() {
        let lats = vec![0.0, 10.0, 20.0];
        let lons = vec![0.0, 10.0, 20.0];
        let vals = vec![100.0, 200.0, 300.0];
        approx_eq(
            idw_haversine(&lats, &lons, &vals, 10.0, 10.0, 2.0, 0),
            200.0,
            1e-10,
        );
    }

    #[test]
    fn idw_midpoint() {
        let lats = vec![0.0, 0.0];
        let lons = vec![-10.0, 10.0];
        let vals = vec![100.0, 200.0];
        approx_eq(
            idw_haversine(&lats, &lons, &vals, 0.0, 0.0, 2.0, 0),
            150.0,
            1e-10,
        );
    }

    #[test]
    fn idw_k_neighbors() {
        let lats = vec![0.0, 0.0, 0.0];
        let lons = vec![0.0, 1.0, 100.0];
        let vals = vec![10.0, 20.0, 999.0];
        let v2 = idw_haversine(&lats, &lons, &vals, 0.0, 0.5, 2.0, 2);
        let v_all = idw_haversine(&lats, &lons, &vals, 0.0, 0.5, 2.0, 0);
        assert!((v2 - 15.0).abs() < (v_all - 15.0).abs());
    }

    // --- RBF ---

    #[test]
    fn rbf_exact_hit() {
        let lats = vec![0.0, 10.0, 20.0];
        let lons = vec![0.0, 10.0, 20.0];
        let vals = vec![1.0, 2.0, 3.0];
        approx_eq(
            rbf_haversine(
                &lats,
                &lons,
                &vals,
                10.0,
                10.0,
                RbfKernel::ThinPlateSpline,
                f64::NAN,
                0,
            ),
            2.0,
            1e-10,
        );
    }

    #[test]
    fn rbf_gaussian_smooth() {
        let lats = vec![0.0, 0.0, 10.0, 10.0];
        let lons = vec![0.0, 10.0, 0.0, 10.0];
        let vals = vec![0.0, 10.0, 10.0, 20.0];
        approx_eq(
            rbf_haversine(
                &lats,
                &lons,
                &vals,
                5.0,
                5.0,
                RbfKernel::Gaussian,
                f64::NAN,
                0,
            ),
            10.0,
            1.0,
        );
    }

    #[test]
    fn rbf_kernels_smoke() {
        let lats = vec![0.0, 0.0, 10.0, 10.0, 5.0];
        let lons = vec![0.0, 10.0, 0.0, 10.0, 5.0];
        let vals = vec![0.0, 10.0, 10.0, 20.0, 10.0];
        for kernel in [
            RbfKernel::Linear,
            RbfKernel::ThinPlateSpline,
            RbfKernel::Cubic,
            RbfKernel::Gaussian,
            RbfKernel::Multiquadric,
            RbfKernel::InverseMultiquadric,
        ] {
            let v = rbf_haversine(&lats, &lons, &vals, 5.0, 5.0, kernel, f64::NAN, 0);
            assert!(v.is_finite(), "kernel {kernel:?} returned non-finite: {v}");
        }
    }

    #[test]
    fn solve_linear_system_basic() {
        let mut a = vec![2.0, 1.0, 5.0, 3.0];
        let mut b = vec![4.0, 7.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        approx_eq(x[0], 5.0, 1e-10);
        approx_eq(x[1], -6.0, 1e-10);
    }

    #[test]
    fn solve_linear_system_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![3.0, 6.0];
        assert!(solve_linear_system(&mut a, &mut b, 2).is_none());
    }
}
