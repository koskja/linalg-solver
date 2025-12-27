//! Linear algebra helper library with fast Rust implementations.
//!
//! This crate provides optimized algorithms for sparse matrix operations,
//! exposed to Python via PyO3.

use pyo3::prelude::*;
use smallvec::SmallVec;

mod adjacency;
mod bitlist;
mod canonical;
mod determinant;
mod dm;
mod hopcroft_karp;
mod nonzeros;
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
pub use tarjan::tarjan_scc;

pub type MatrixIndex = usize;
pub const INLINE_PERM_CAPACITY: usize = 16;
pub type Permutation = SmallVec<[MatrixIndex; INLINE_PERM_CAPACITY]>;

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

/// Python-accessible wrapper for the determinant computation cost
#[pyclass]
#[derive(Clone)]
pub struct PyCost {
    #[pyo3(get)]
    pub multiplications: usize,
    #[pyo3(get)]
    pub additions: usize,
}

#[pymethods]
impl PyCost {
    fn __repr__(&self) -> String {
        format!(
            "Cost(multiplications={}, additions={}, total={})",
            self.multiplications,
            self.additions,
            self.multiplications + self.additions
        )
    }

    #[getter]
    fn total(&self) -> usize {
        self.multiplications + self.additions
    }
}

/// Direct computation process (for matrices of size <= 2)
#[pyclass]
#[derive(Clone)]
pub struct ProcessDirect {
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
}

/// Laplace expansion along a row
#[pyclass]
#[derive(Clone)]
pub struct ProcessRowExpansion {
    #[pyo3(get)]
    pub row: usize,
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
    /// (column_index, subprocess) for each non-zero entry
    pub minors_internal: Vec<(usize, PyProcess)>,
}

#[pymethods]
impl ProcessRowExpansion {
    #[getter]
    fn minors(&self) -> Vec<(usize, PyProcess)> {
        self.minors_internal.clone()
    }
}

/// Laplace expansion along a column
#[pyclass]
#[derive(Clone)]
pub struct ProcessColExpansion {
    #[pyo3(get)]
    pub col: usize,
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
    /// (row_index, subprocess) for each non-zero entry
    pub minors_internal: Vec<(usize, PyProcess)>,
}

#[pymethods]
impl ProcessColExpansion {
    #[getter]
    fn minors(&self) -> Vec<(usize, PyProcess)> {
        self.minors_internal.clone()
    }
}

/// Block triangular decomposition (det = product of block dets)
#[pyclass]
#[derive(Clone)]
pub struct ProcessBlockTriangular {
    #[pyo3(get)]
    pub row_perm: Vec<usize>,
    #[pyo3(get)]
    pub col_perm: Vec<usize>,
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
    /// Processes for each diagonal block
    pub blocks_internal: Vec<PyProcess>,
}

#[pymethods]
impl ProcessBlockTriangular {
    #[getter]
    fn blocks(&self) -> Vec<PyProcess> {
        self.blocks_internal.clone()
    }
}

/// Single row operation: add row `src` (scaled) to row `dst` to zero out (dst, pivot_col)
#[pyclass]
#[derive(Clone)]
pub struct ProcessAddRow {
    #[pyo3(get)]
    pub src: usize,
    #[pyo3(get)]
    pub dst: usize,
    #[pyo3(get)]
    pub pivot_col: usize,
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
    /// Process for the resulting matrix
    pub result_internal: Box<PyProcess>,
}

#[pymethods]
impl ProcessAddRow {
    #[getter]
    fn result(&self) -> PyProcess {
        (*self.result_internal).clone()
    }
}

/// Python-accessible wrapper for the determinant computation process
#[pyclass]
#[derive(Clone)]
pub enum PyProcess {
    Direct(ProcessDirect),
    RowExpansion(ProcessRowExpansion),
    ColExpansion(ProcessColExpansion),
    BlockTriangular(ProcessBlockTriangular),
    AddRow(ProcessAddRow),
}

impl From<&Process> for PyProcess {
    fn from(process: &Process) -> Self {
        let expected_nonzeros = process.expected_nonzeros.to_vec();
        match &process.raw {
            RawProcess::Direct(Direct { size }) => PyProcess::Direct(ProcessDirect {
                size: *size,
                expected_nonzeros,
            }),
            RawProcess::RowExpansion(RowExpansion { row, minors }) => {
                PyProcess::RowExpansion(ProcessRowExpansion {
                    row: *row,
                    expected_nonzeros,
                    minors_internal: minors
                        .iter()
                        .map(|(col, p)| (*col, PyProcess::from(p.as_ref())))
                        .collect(),
                })
            }
            RawProcess::ColExpansion(ColExpansion { col, minors }) => {
                PyProcess::ColExpansion(ProcessColExpansion {
                    col: *col,
                    expected_nonzeros,
                    minors_internal: minors
                        .iter()
                        .map(|(row, p)| (*row, PyProcess::from(p.as_ref())))
                        .collect(),
                })
            }
            RawProcess::BlockTriangular(BlockTriangular {
                blocks,
                row_perm,
                col_perm,
            }) => PyProcess::BlockTriangular(ProcessBlockTriangular {
                row_perm: row_perm.clone().into_vec(),
                col_perm: col_perm.clone().into_vec(),
                expected_nonzeros,
                blocks_internal: blocks.iter().map(|p| PyProcess::from(p.as_ref())).collect(),
            }),
            RawProcess::AddRow(AddRow {
                src,
                dst,
                pivot_col,
                result,
            }) => PyProcess::AddRow(ProcessAddRow {
                src: *src,
                dst: *dst,
                pivot_col: *pivot_col,
                expected_nonzeros,
                result_internal: Box::new(PyProcess::from(result.as_ref())),
            }),
        }
    }
}

#[pymethods]
impl PyProcess {
    fn __repr__(&self) -> String {
        match self {
            PyProcess::Direct(d) => format!("Process::Direct(size={})", d.size),
            PyProcess::RowExpansion(r) => format!(
                "Process::RowExpansion(row={}, minors={})",
                r.row,
                r.minors_internal.len()
            ),
            PyProcess::ColExpansion(c) => format!(
                "Process::ColExpansion(col={}, minors={})",
                c.col,
                c.minors_internal.len()
            ),
            PyProcess::BlockTriangular(b) => format!(
                "Process::BlockTriangular(blocks={})",
                b.blocks_internal.len()
            ),
            PyProcess::AddRow(a) => format!(
                "Process::AddRow(src={}, dst={}, pivot_col={})",
                a.src, a.dst, a.pivot_col
            ),
        }
    }

    fn __str__(&self) -> String {
        self.format_tree(0)
    }

    /// Get the inner data for Direct variant
    #[getter]
    fn direct(&self) -> Option<ProcessDirect> {
        match self {
            PyProcess::Direct(d) => Some(d.clone()),
            _ => None,
        }
    }

    /// Get the inner data for RowExpansion variant
    #[getter]
    fn row_expansion(&self) -> Option<ProcessRowExpansion> {
        match self {
            PyProcess::RowExpansion(r) => Some(r.clone()),
            _ => None,
        }
    }

    /// Get the inner data for ColExpansion variant
    #[getter]
    fn col_expansion(&self) -> Option<ProcessColExpansion> {
        match self {
            PyProcess::ColExpansion(c) => Some(c.clone()),
            _ => None,
        }
    }

    /// Get the inner data for BlockTriangular variant
    #[getter]
    fn block_triangular(&self) -> Option<ProcessBlockTriangular> {
        match self {
            PyProcess::BlockTriangular(b) => Some(b.clone()),
            _ => None,
        }
    }

    /// Get the inner data for AddRow variant
    #[getter]
    fn add_row(&self) -> Option<ProcessAddRow> {
        match self {
            PyProcess::AddRow(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Get the matrix size that this process operates on
    #[getter]
    fn size(&self) -> usize {
        self.compute_size()
    }
}

impl PyProcess {
    fn format_tree(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        match self {
            PyProcess::Direct(d) => format!("{}Direct(size={})", prefix, d.size),
            PyProcess::RowExpansion(r) => {
                let mut result = format!("{}RowExpansion(row={}):", prefix, r.row);
                for (col, subprocess) in &r.minors_internal {
                    result.push_str(&format!("\n{}  col={} =>", prefix, col));
                    result.push_str(&format!("\n{}", subprocess.format_tree(indent + 2)));
                }
                result
            }
            PyProcess::ColExpansion(c) => {
                let mut result = format!("{}ColExpansion(col={}):", prefix, c.col);
                for (row, subprocess) in &c.minors_internal {
                    result.push_str(&format!("\n{}  row={} =>", prefix, row));
                    result.push_str(&format!("\n{}", subprocess.format_tree(indent + 2)));
                }
                result
            }
            PyProcess::BlockTriangular(b) => {
                let row_cycles = perm_to_cycles(&b.row_perm);
                let col_cycles = perm_to_cycles(&b.col_perm);
                let mut result = format!(
                    "{}BlockTriangular(row_perm={}, col_perm={}):",
                    prefix, row_cycles, col_cycles
                );
                for (i, block) in b.blocks_internal.iter().enumerate() {
                    result.push_str(&format!("\n{}  block[{}] =>", prefix, i));
                    result.push_str(&format!("\n{}", block.format_tree(indent + 2)));
                }
                result
            }
            PyProcess::AddRow(a) => {
                let mut result = format!(
                    "{}AddRow(src={}, dst={}, pivot_col={}):",
                    prefix, a.src, a.dst, a.pivot_col
                );
                result.push_str(&format!("\n{}", a.result_internal.format_tree(indent + 1)));
                result
            }
        }
    }

    /// Compute the matrix size that this process operates on
    fn compute_size(&self) -> usize {
        match self {
            PyProcess::Direct(d) => d.size,
            PyProcess::RowExpansion(r) => {
                if let Some((_, first_minor)) = r.minors_internal.first() {
                    1 + first_minor.compute_size()
                } else {
                    1
                }
            }
            PyProcess::ColExpansion(c) => {
                if let Some((_, first_minor)) = c.minors_internal.first() {
                    1 + first_minor.compute_size()
                } else {
                    1
                }
            }
            PyProcess::BlockTriangular(b) => b
                .blocks_internal
                .iter()
                .map(|block| block.compute_size())
                .sum(),
            PyProcess::AddRow(a) => a.result_internal.compute_size(),
        }
    }
}

/// Convert a permutation array to cycle notation string.
/// The permutation maps position i -> perm[i] (original index).
/// E.g., [2, 0, 1] means position 0 gets original index 2, etc.
fn perm_to_cycles(perm: &[usize]) -> String {
    if perm.is_empty() {
        return "()".to_string();
    }

    let n = perm.len();

    // Build inverse: inv[original_idx] = new_position (if it exists in perm)
    let max_val = perm.iter().copied().max().unwrap_or(0);
    let mut inv = vec![None; max_val + 1];
    for (new_pos, &orig_idx) in perm.iter().enumerate() {
        if orig_idx < inv.len() {
            inv[orig_idx] = Some(new_pos);
        }
    }

    // Check if this is a proper permutation of 0..n
    let is_proper_perm = perm.iter().all(|&x| x < n) && {
        let mut seen = vec![false; n];
        perm.iter().all(|&x| {
            if seen[x] {
                false
            } else {
                seen[x] = true;
                true
            }
        })
    };

    if !is_proper_perm {
        // Not a standard permutation, just show the mapping
        let mappings: Vec<String> = perm
            .iter()
            .enumerate()
            .filter(|&(i, &v)| i != v)
            .map(|(i, v)| format!("{}<-{}", i, v))
            .collect();
        if mappings.is_empty() {
            return "id".to_string();
        }
        return format!("[{}]", mappings.join(", "));
    }

    // Standard permutation - compute cycles
    let mut visited = vec![false; n];
    let mut cycles: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }

        let mut cycle = Vec::new();
        let mut current = start;

        while !visited[current] {
            visited[current] = true;
            cycle.push(perm[current]); // Show original indices in cycle
            current = perm[current];
        }

        if cycle.len() > 1 {
            cycles.push(cycle);
        }
    }

    if cycles.is_empty() {
        return "id".to_string();
    }

    cycles
        .iter()
        .map(|cycle| {
            format!(
                "({})",
                cycle
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Result of finding the optimal determinant computation process
#[pyclass]
#[derive(Clone)]
pub struct OptimalProcessResult {
    #[pyo3(get)]
    pub cost: PyCost,
    #[pyo3(get)]
    pub process: PyProcess,
}

#[pymethods]
impl OptimalProcessResult {
    fn __repr__(&self) -> String {
        format!(
            "OptimalProcessResult(cost={}, process={})",
            self.cost.__repr__(),
            self.process.__repr__()
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
    Ok(OptimalProcessResult {
        cost: PyCost {
            multiplications: cost.multiplications,
            additions: cost.additions,
        },
        process: PyProcess::from(&process),
    })
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
    m.add_class::<PyCost>()?;
    m.add_class::<PyProcess>()?;
    m.add_class::<ProcessDirect>()?;
    m.add_class::<ProcessRowExpansion>()?;
    m.add_class::<ProcessColExpansion>()?;
    m.add_class::<ProcessBlockTriangular>()?;
    m.add_class::<ProcessAddRow>()?;
    m.add_class::<OptimalProcessResult>()?;
    Ok(())
}
