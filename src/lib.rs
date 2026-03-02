use pyo3::prelude::*;
mod expressions;
mod geospatial;
mod interpolation;


/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use pyo3::prelude::*;
    use pyo3_polars::PolarsAllocator;

    #[pyfunction]
    fn print_extension_info() -> String {
        "Interpolars extension module loaded successfully".to_string()
    }

    #[global_allocator]
    static ALLOC: PolarsAllocator = PolarsAllocator::new();
}
