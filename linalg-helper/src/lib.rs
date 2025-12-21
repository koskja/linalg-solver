//! Linear algebra helper library with fast Rust implementations.
//!
//! This crate provides optimized algorithms for sparse matrix operations,
//! exposed to Python via PyO3.

use pyo3::prelude::*;
use smallvec::SmallVec;

mod adjacency;
mod canonical;
mod determinant;
mod dm;
mod hopcroft_karp;
mod tarjan;

#[cfg(test)]
mod tests;

pub use adjacency::{AdjacencyMatrix, Matching};
pub use canonical::{CanonicalForm, are_permutation_equivalent, canonicalize};
pub use determinant::{Cost, Nonzeros, Process, find_optimal_process};
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

/// Python-accessible wrapper for the determinant computation process
#[pyclass]
#[derive(Clone)]
pub struct PyProcess {
    #[pyo3(get)]
    pub process_type: String,
    #[pyo3(get)]
    pub size: Option<usize>,
    #[pyo3(get)]
    pub row: Option<usize>,
    #[pyo3(get)]
    pub col: Option<usize>,
    #[pyo3(get)]
    pub src: Option<usize>,
    #[pyo3(get)]
    pub dst: Option<usize>,
    #[pyo3(get)]
    pub r1: Option<usize>,
    #[pyo3(get)]
    pub r2: Option<usize>,
    #[pyo3(get)]
    pub pivot_col: Option<usize>,
    #[pyo3(get)]
    pub row_perm: Option<Vec<usize>>,
    #[pyo3(get)]
    pub col_perm: Option<Vec<usize>>,
    /// Expected non-zero positions (row, col) in the matrix for this process step
    #[pyo3(get)]
    pub expected_nonzeros: Vec<(usize, usize)>,
    // These fields use custom getters due to recursive structure
    pub minors: Option<Vec<(usize, PyProcess)>>,
    pub blocks: Option<Vec<PyProcess>>,
    pub result: Option<Box<PyProcess>>,
}

impl From<&Process> for PyProcess {
    fn from(process: &Process) -> Self {
        match process {
            Process::Direct {
                size,
                expected_nonzeros,
            } => PyProcess {
                process_type: "Direct".to_string(),
                size: Some(*size),
                row: None,
                col: None,
                src: None,
                dst: None,
                r1: None,
                r2: None,
                pivot_col: None,
                row_perm: None,
                col_perm: None,
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: None,
                blocks: None,
                result: None,
            },
            Process::RowExpansion {
                row,
                minors,
                expected_nonzeros,
            } => PyProcess {
                process_type: "RowExpansion".to_string(),
                size: None,
                row: Some(*row),
                col: None,
                src: None,
                dst: None,
                r1: None,
                r2: None,
                pivot_col: None,
                row_perm: None,
                col_perm: None,
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: Some(
                    minors
                        .iter()
                        .map(|(col, p)| (*col, PyProcess::from(p.as_ref())))
                        .collect(),
                ),
                blocks: None,
                result: None,
            },
            Process::ColExpansion {
                col,
                minors,
                expected_nonzeros,
            } => PyProcess {
                process_type: "ColExpansion".to_string(),
                size: None,
                row: None,
                col: Some(*col),
                src: None,
                dst: None,
                r1: None,
                r2: None,
                pivot_col: None,
                row_perm: None,
                col_perm: None,
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: Some(
                    minors
                        .iter()
                        .map(|(row, p)| (*row, PyProcess::from(p.as_ref())))
                        .collect(),
                ),
                blocks: None,
                result: None,
            },
            Process::BlockTriangular {
                blocks,
                row_perm,
                col_perm,
                expected_nonzeros,
            } => PyProcess {
                process_type: "BlockTriangular".to_string(),
                size: None,
                row: None,
                col: None,
                src: None,
                dst: None,
                r1: None,
                r2: None,
                pivot_col: None,
                row_perm: Some(row_perm.clone().into_vec()),
                col_perm: Some(col_perm.clone().into_vec()),
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: None,
                blocks: Some(blocks.iter().map(|p| PyProcess::from(p.as_ref())).collect()),
                result: None,
            },
            Process::AddRow {
                src,
                dst,
                pivot_col,
                result,
                expected_nonzeros,
            } => PyProcess {
                process_type: "AddRow".to_string(),
                size: None,
                row: None,
                col: None,
                src: Some(*src),
                dst: Some(*dst),
                r1: None,
                r2: None,
                pivot_col: Some(*pivot_col),
                row_perm: None,
                col_perm: None,
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: None,
                blocks: None,
                result: Some(Box::new(PyProcess::from(result.as_ref()))),
            },
            Process::SwapRows {
                r1,
                r2,
                result,
                expected_nonzeros,
            } => PyProcess {
                process_type: "SwapRows".to_string(),
                size: None,
                row: None,
                col: None,
                src: None,
                dst: None,
                r1: Some(*r1),
                r2: Some(*r2),
                pivot_col: None,
                row_perm: None,
                col_perm: None,
                expected_nonzeros: expected_nonzeros.to_vec(),
                minors: None,
                blocks: None,
                result: Some(Box::new(PyProcess::from(result.as_ref()))),
            },
        }
    }
}

#[pymethods]
impl PyProcess {
    fn __repr__(&self) -> String {
        match self.process_type.as_str() {
            "Direct" => format!("Process::Direct(size={})", self.size.unwrap_or(0)),
            "RowExpansion" => format!(
                "Process::RowExpansion(row={}, minors={})",
                self.row.unwrap_or(0),
                self.minors.as_ref().map(|m| m.len()).unwrap_or(0)
            ),
            "ColExpansion" => format!(
                "Process::ColExpansion(col={}, minors={})",
                self.col.unwrap_or(0),
                self.minors.as_ref().map(|m| m.len()).unwrap_or(0)
            ),
            "BlockTriangular" => format!(
                "Process::BlockTriangular(blocks={})",
                self.blocks.as_ref().map(|b| b.len()).unwrap_or(0)
            ),
            "AddRow" => format!(
                "Process::AddRow(src={}, dst={}, pivot_col={})",
                self.src.unwrap_or(0),
                self.dst.unwrap_or(0),
                self.pivot_col.unwrap_or(0)
            ),
            "SwapRows" => format!(
                "Process::SwapRows(r1={}, r2={})",
                self.r1.unwrap_or(0),
                self.r2.unwrap_or(0)
            ),
            _ => format!("Process::Unknown({})", self.process_type),
        }
    }

    fn __str__(&self) -> String {
        self.format_tree(0)
    }

    /// Get minors for RowExpansion/ColExpansion
    /// Returns list of (index, subprocess) tuples
    #[getter]
    fn minors(&self) -> Option<Vec<(usize, PyProcess)>> {
        self.minors.clone()
    }

    /// Get blocks for BlockTriangular
    #[getter]
    fn blocks(&self) -> Option<Vec<PyProcess>> {
        self.blocks.clone()
    }

    /// Get result subprocess for AddRow/SwapRows
    #[getter]
    fn result(&self) -> Option<PyProcess> {
        self.result.as_ref().map(|r| (**r).clone())
    }
}

impl PyProcess {
    fn format_tree(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        match self.process_type.as_str() {
            "Direct" => format!("{}Direct(size={})", prefix, self.size.unwrap_or(0)),
            "RowExpansion" => {
                let mut result = format!("{}RowExpansion(row={}):", prefix, self.row.unwrap_or(0));
                if let Some(ref minors) = self.minors {
                    for (col, subprocess) in minors {
                        result.push_str(&format!("\n{}  col={} =>", prefix, col));
                        result.push_str(&format!("\n{}", subprocess.format_tree(indent + 2)));
                    }
                }
                result
            }
            "ColExpansion" => {
                let mut result = format!("{}ColExpansion(col={}):", prefix, self.col.unwrap_or(0));
                if let Some(ref minors) = self.minors {
                    for (row, subprocess) in minors {
                        result.push_str(&format!("\n{}  row={} =>", prefix, row));
                        result.push_str(&format!("\n{}", subprocess.format_tree(indent + 2)));
                    }
                }
                result
            }
            "BlockTriangular" => {
                let row_cycles = perm_to_cycles(self.row_perm.as_ref().unwrap_or(&vec![]));
                let col_cycles = perm_to_cycles(self.col_perm.as_ref().unwrap_or(&vec![]));
                let mut result = format!(
                    "{}BlockTriangular(row_perm={}, col_perm={}):",
                    prefix, row_cycles, col_cycles
                );
                if let Some(ref blocks) = self.blocks {
                    for (i, block) in blocks.iter().enumerate() {
                        result.push_str(&format!("\n{}  block[{}] =>", prefix, i));
                        result.push_str(&format!("\n{}", block.format_tree(indent + 2)));
                    }
                }
                result
            }
            "AddRow" => {
                let mut result = format!(
                    "{}AddRow(src={}, dst={}, pivot_col={}):",
                    prefix,
                    self.src.unwrap_or(0),
                    self.dst.unwrap_or(0),
                    self.pivot_col.unwrap_or(0)
                );
                if let Some(ref subprocess) = self.result {
                    result.push_str(&format!("\n{}", subprocess.format_tree(indent + 1)));
                }
                result
            }
            "SwapRows" => {
                let mut result = format!(
                    "{}SwapRows(r1={}, r2={}):",
                    prefix,
                    self.r1.unwrap_or(0),
                    self.r2.unwrap_or(0)
                );
                if let Some(ref subprocess) = self.result {
                    result.push_str(&format!("\n{}", subprocess.format_tree(indent + 1)));
                }
                result
            }
            _ => format!("{}Unknown({})", prefix, self.process_type),
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
    m.add_class::<OptimalProcessResult>()?;
    Ok(())
}
