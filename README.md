### interpolars

`interpolars` is a small Polars plugin that does **N-dimensional linear interpolation** from a
source "grid" (your DataFrame) onto an explicit **target** DataFrame.

It supports:

- **1D/2D/3D/... multilinear interpolation**
- **Multiple value columns** in one call
- **Target passthrough columns** (e.g. labels/metadata)
- **Grouped interpolation over "extra" coordinate dims** (e.g. group by `time` and interpolate
  over `latitude/longitude` for each time slice)
- **Non-float coordinate dtypes** such as **Date** and **Duration** (they are cast internally for
  interpolation math; group keys preserve dtype in output)
- **Configurable NaN/Null handling** (`handle_missing`): error, drop, fill with a constant, or
  nearest-neighbor fill
- **Boundary extrapolation** (`extrapolate`): linearly project beyond the source grid instead of
  clamping

---

### Installation

This repo is built with `maturin` and managed with `uv`.

- **As a local editable/dev install (recommended for hacking on it)**:

```bash
cd /path/to/interpolars
uv sync --dev
```

- **Run tests**:

```bash
cd /path/to/interpolars
uv run pytest
```

Notes:

- **Python**: `>= 3.12` (see `pyproject.toml`)
- **Polars**: pinned to `polars==1.37.1`

---

### The API: `interpolate_nd`

The only public function is `interpolate_nd`, which returns a **Polars expression** (`pl.Expr`).

```python
from interpolars import interpolate_nd

expr = interpolate_nd(
    expr_cols_or_exprs=["x", "y"],            # source coordinate columns/exprs
    value_cols_or_exprs=["value_a", "value_b"],  # source value columns/exprs
    interp_target=target_df,                  # DataFrame with target coordinates (+ metadata)
    handle_missing="error",                   # "error" | "drop" | "fill" | "nearest"
    fill_value=None,                          # required when handle_missing="fill"
    extrapolate=False,                        # True → linear extrapolation at boundaries
)
```

You typically use it inside `LazyFrame.select`:

```python
import polars as pl
from interpolars import interpolate_nd

out = (
    source_df.lazy()
    .select(interpolate_nd(["x", "y"], ["value"], target_df))
    .collect()
)
```

---

### Output shape and how to consume it

`interpolate_nd(...)` produces a single **struct column** named `"interpolated"`.

That struct contains, in order:

- all columns from `interp_target` (including metadata like `label`)
- any "extra/group" coordinate dims from the source (see next section)
- all interpolated value fields

To "flatten" the result into normal columns, use `unnest`:

```python
flat = (
    source_df.lazy()
    .select(interpolate_nd(["x", "y"], ["value"], target_df))
    .unnest("interpolated")
    .collect()
)
```

Or access fields directly:

```python
only_value = (
    source_df.lazy()
    .select(interpolate_nd(["x"], ["value"], target_df).struct.field("value").alias("value"))
    .collect()
)
```

---

### Grouped interpolation over extra coordinate dims (e.g. time slices)

If the **source** coordinate columns include fields that do **not** exist in `interp_target`,
those fields are treated as **grouping dimensions**.

Example:

- source coords: `["latitude", "longitude", "time"]`
- target df columns: `["latitude", "longitude", "label"]`
- values: `["2m_temp", "precipitation"]`

Then `time` is a group key:

1. The source rows are grouped by unique `time`
2. Interpolation runs over (`latitude`, `longitude`) **within each time group**
3. Results are concatenated, producing `len(target_df) * n_times` rows

```python
import polars as pl
from interpolars import interpolate_nd

target = pl.DataFrame(
    {
        "latitude": [0.25, 0.75],
        "longitude": [0.50, 0.25],
        "label": ["a", "b"],
    }
)

out = (
    source_df.lazy()
    .select(
        interpolate_nd(
            ["latitude", "longitude", "time"],
            ["2m_temp", "precipitation"],
            target,
        )
    )
    .unnest("interpolated")
    .collect()
)
```

Output order is deterministic:

- target rows are repeated per group
- groups are ordered by ascending group key (e.g. ascending `time`)

---

### Date and Duration coordinates

Coordinate columns can be `pl.Date`, `pl.Datetime`, and `pl.Duration` (and other numeric-like
dtypes). The plugin will cast coordinates internally for interpolation computations.

Example (Date as an interpolation axis):

```python
from datetime import date
import polars as pl
from interpolars import interpolate_nd

source = pl.DataFrame(
    {
        "d": pl.Series("d", [date(2020, 1, 1), date(2020, 1, 3)], dtype=pl.Date),
        "value": [0.0, 2.0],
    }
)
target = pl.DataFrame(
    {
        "d": pl.Series("d", [date(2020, 1, 2)], dtype=pl.Date),
        "label": ["mid"],
    }
)

out = (
    source.lazy()
    .select(interpolate_nd(["d"], ["value"], target))
    .unnest("interpolated")
    .collect()
)
```

Example (Duration as an interpolation axis):

```python
import polars as pl
from interpolars import interpolate_nd

source = pl.DataFrame(
    {
        "dt": pl.Series("dt", [0, 10_000], dtype=pl.Duration("ms")),
        "value": [0.0, 10.0],
    }
)
target = pl.DataFrame(
    {
        "dt": pl.Series("dt", [5_000], dtype=pl.Duration("ms")),
        "label": ["half"],
    }
)

out = (
    source.lazy()
    .select(interpolate_nd(["dt"], ["value"], target))
    .unnest("interpolated")
    .collect()
)
```

---

### Handling NaN and Null values (`handle_missing`)

By default, any `NaN` or `Null` in source coordinates or values will raise an error. You can
change this with the `handle_missing` parameter:

| Mode | Coords with NaN/Null | Values with NaN/Null |
|------|---------------------|---------------------|
| `"error"` (default) | Error | Error |
| `"drop"` | Drop row | Drop row |
| `"fill"` | Drop row | Replace with `fill_value` |
| `"nearest"` | Drop row | Replace with nearest valid grid point's value |

- Rows with `NaN`/`Null` in **coordinate** columns are always dropped (except in `"error"` mode,
  which raises). A grid point with no location cannot be meaningfully filled.
- `fill_value` is required when `handle_missing="fill"` and ignored otherwise.
- `"nearest"` finds the closest valid grid point by Euclidean distance in coordinate space.

```python
# Drop any source rows that have NaN or Null in coords or values
out = (
    source_df.lazy()
    .select(interpolate_nd(["x", "y"], ["value"], target_df, handle_missing="drop"))
    .collect()
)

# Replace NaN/Null values with 0.0 (NaN coords are dropped)
out = (
    source_df.lazy()
    .select(
        interpolate_nd(
            ["x", "y"], ["value"], target_df,
            handle_missing="fill", fill_value=0.0,
        )
    )
    .collect()
)

# Replace NaN/Null values with the nearest valid grid point's value
out = (
    source_df.lazy()
    .select(interpolate_nd(["x", "y"], ["value"], target_df, handle_missing="nearest"))
    .collect()
)
```

> **Note:** `"drop"` can cause "missing corner point" errors if the remaining grid is no longer a
> full cartesian product after removing rows. `"fill"` and `"nearest"` preserve the grid structure.

---

### Boundary extrapolation (`extrapolate`)

By default, target points outside the source grid are clamped to the nearest boundary value. Set
`extrapolate=True` to linearly project from the two nearest grid points along each axis instead:

```python
import polars as pl
from interpolars import interpolate_nd

source = pl.DataFrame({"x": [0.0, 1.0, 2.0], "value": [0.0, 10.0, 20.0]})

# x=3.0 is outside [0, 2]; extrapolate from slope of (1,10)→(2,20)
target = pl.DataFrame({"x": [3.0]})

out = (
    source.lazy()
    .select(interpolate_nd(["x"], ["value"], target, extrapolate=True))
    .unnest("interpolated")
    .collect()
)
# value = 30.0  (linear projection)
```

Without `extrapolate=True`, the same query would clamp to the boundary and return `20.0`.

`handle_missing` and `extrapolate` compose freely -- for example,
`handle_missing="nearest", extrapolate=True` fills NaN values with the nearest neighbor **and**
extrapolates at boundaries.

---

### Important constraints / behavior

- **NaN/Null handling is configurable** via `handle_missing` (see above). The default (`"error"`)
  raises on any NaN or Null.
- **Source must be a full cartesian grid (per group)** for the interpolation dimensions:
  every "corner" required for multilinear interpolation must exist, otherwise you'll get an error.
- **Out-of-bounds targets**: clamped by default; set `extrapolate=True` for linear extrapolation.
- **Duplicate names**: value field names cannot collide with `interp_target` columns (and group
  fields cannot collide either); collisions error.

---

### Project layout

- `src/interpolars/__init__.py`: Python API wrapper (registers the Polars plugin function)
- `src/expressions.rs`: the Polars expression implementation (Rust)
- `tests/`: pytest suite with examples (including grouped + Date/Duration coverage)
