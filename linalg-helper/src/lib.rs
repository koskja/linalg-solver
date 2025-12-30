//! Linear algebra helper library with fast Rust implementations.
//!
//! This crate provides optimized algorithms for sparse matrix operations,
//! exposed to Python via PyO3.
#![allow(clippy::needless_range_loop)]
use pyo3::prelude::*;

mod adjacency;
mod bitlist;
mod canonical;
mod determinant;
mod dm;
mod hopcroft_karp;
mod nonzeros;
mod permutation;
mod tarjan;

#[cfg(test)]
mod tests;

pub use adjacency::{AdjacencyMatrix, Matching};
pub use canonical::{CanonicalForm, are_permutation_equivalent, canonicalize};
pub use determinant::{
    AddRow, BlockTriangular, ColExpansion, Cost, Direct, Process, RawProcess, RowExpansion,
    find_optimal_process,
};
pub use dm::{DMResult, dulmage_mendelsohn};
pub use hopcroft_karp::hopcroft_karp;
pub use nonzeros::Nonzeros;
pub use permutation::{Permutation, RowColPermutation};
pub use tarjan::tarjan_scc;

pub type MatrixIndex = usize;
pub const INLINE_PERM_CAPACITY: usize = 4;

/// Compute the Dulmage-Mendelsohn decomposition of a matrix.
///
/// Args:
///     matrix: A list of lists of bools representing the nonzero pattern.
///             matrix[i][j] = True means the entry is nonzero.
///
/// Returns:
///     DMResult with row_perm, col_perm, and block_sizes.
#[pyfunction]
fn dm_decomposition(matrix: Vec<Vec<bool>>) -> PyResult<DMResult> {
    let graph = AdjacencyMatrix::from_vec(matrix);
    Ok(dulmage_mendelsohn(&graph))
}

/// Compute the canonical form of a matrix under row/column permutation.
///
/// Given any matrix X, returns permutations P, Q such that C = PXQ is the
/// canonical representative. If Y = P2 * X * Q2 for some permutations P2, Q2,
/// then canonicalize(Y) will produce the same canonical form C.
///
/// Args:
///     matrix: A list of lists of bools representing the nonzero pattern.
///
/// Returns:
///     CanonicalForm with row_perm, col_perm, and canonical_hash.
#[pyfunction]
fn canonicalize_matrix(matrix: Vec<Vec<bool>>) -> PyResult<CanonicalForm> {
    let graph = AdjacencyMatrix::from_vec(matrix);
    Ok(canonicalize(&graph))
}

/// Check if two matrices are permutation equivalent.
///
/// Returns True if there exist permutation matrices P, Q such that A = P * B * Q.
#[pyfunction]
fn check_permutation_equivalent(a: Vec<Vec<bool>>, b: Vec<Vec<bool>>) -> PyResult<bool> {
    let graph_a = AdjacencyMatrix::from_vec(a);
    let graph_b = AdjacencyMatrix::from_vec(b);
    Ok(are_permutation_equivalent(&graph_a, &graph_b))
}

/// Result of finding the optimal determinant computation process
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct OptimalProcessResult {
    #[pyo3(get)]
    pub cost: Cost,
    #[pyo3(get)]
    pub process: Process,
}

#[pymethods]
impl OptimalProcessResult {
    fn __repr__(&self) -> String {
        format!(
            "OptimalProcessResult(cost=Cost(multiplications={}, additions={}, total={}), process={:?})",
            self.cost.multiplications,
            self.cost.additions,
            self.cost.total(),
            self.process
        )
    }
}

/// Find the optimal process for computing the determinant of a sparse matrix.
///
/// Given the sparsity pattern of a matrix (which entries are non-zero),
/// finds the optimal strategy for computing its determinant using:
/// - Minor expansion (row/column)
/// - Block triangular decomposition
/// - Row operations (AddRow, SwapRows)
///
/// Args:
///     matrix: A list of lists of bools representing the nonzero pattern.
///             matrix[i][j] = True means the entry is nonzero.
///
/// Returns:
///     OptimalProcessResult with cost and process.
#[pyfunction]
fn find_optimal_determinant_process(matrix: Vec<Vec<bool>>) -> PyResult<OptimalProcessResult> {
    let graph = AdjacencyMatrix::from_vec(matrix);
    let (cost, process) = find_optimal_process(&graph);
    Ok(OptimalProcessResult { cost, process })
}

/// Python module for linear algebra helpers
#[pymodule]
fn linalg_helper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dm_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(check_permutation_equivalent, m)?)?;
    m.add_function(wrap_pyfunction!(find_optimal_determinant_process, m)?)?;
    m.add_class::<DMResult>()?;
    m.add_class::<CanonicalForm>()?;
    m.add_class::<Cost>()?;
    m.add_class::<RawProcess>()?;
    m.add_class::<Process>()?;
    m.add_class::<Direct>()?;
    m.add_class::<RowExpansion>()?;
    m.add_class::<ColExpansion>()?;
    m.add_class::<BlockTriangular>()?;
    m.add_class::<AddRow>()?;
    m.add_class::<Nonzeros>()?;
    m.add_class::<OptimalProcessResult>()?;
    m.add_class::<Permutation>()?;
    m.add_class::<RowColPermutation>()?;
    Ok(())
}
