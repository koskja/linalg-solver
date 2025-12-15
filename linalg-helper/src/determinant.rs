//! Optimal Determinant Computation Strategy Finder
//!
//! Finds the optimal way to compute the determinant of a sparse matrix
//! using DFS with canonical caching, supporting:
//! - Minor expansion (row/column)
//! - Block triangular decomposition
//! - Atomic row operations (AddRow, SwapRows)

use std::collections::HashMap;

use crate::adjacency::AdjacencyMatrix;
use crate::canonical::canonicalize;
use crate::dm::dulmage_mendelsohn;

/// Represents a computation strategy for calculating a determinant
#[derive(Clone, Debug)]
pub enum Process {
    /// Base case: direct computation (n <= 2)
    Direct {
        size: usize,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },

    /// Laplace expansion along a row
    RowExpansion {
        row: usize,
        /// (column_index, subprocess) for each non-zero entry
        minors: Vec<(usize, Box<Process>)>,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },

    /// Laplace expansion along a column
    ColExpansion {
        col: usize,
        /// (row_index, subprocess) for each non-zero entry
        minors: Vec<(usize, Box<Process>)>,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },

    /// Block triangular decomposition (det = product of block dets)
    BlockTriangular {
        /// Processes for each diagonal block
        blocks: Vec<Box<Process>>,
        /// Row permutation to achieve block form
        row_perm: Vec<usize>,
        /// Column permutation to achieve block form
        col_perm: Vec<usize>,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },

    /// Single row operation: add row `src` (scaled) to row `dst` to zero out (dst, pivot_col)
    /// det(original) = det(after_operation), so cost is just the additions + subprocess
    AddRow {
        src: usize,
        dst: usize,
        /// Column where dst has a non-zero that we're eliminating
        pivot_col: usize,
        /// Process for the resulting matrix (with one more zero)
        result: Box<Process>,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },

    /// Single row operation: swap two rows
    /// det(original) = -det(after_swap), sign flip is free
    SwapRows {
        r1: usize,
        r2: usize,
        /// Process for the resulting matrix
        result: Box<Process>,
        /// Expected non-zero positions (row, col) in the matrix
        expected_nonzeros: Vec<(usize, usize)>,
    },
}

/// Cost of a computation strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cost {
    /// Number of multiplications (excluding ×(-1))
    pub multiplications: usize,
    /// Number of additions (excluding +0)
    pub additions: usize,
}

impl Cost {
    pub fn new(multiplications: usize, additions: usize) -> Self {
        Self {
            multiplications,
            additions,
        }
    }

    pub fn zero() -> Self {
        Self {
            multiplications: 0,
            additions: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.multiplications + self.additions
    }

    pub fn add(&self, other: &Cost) -> Self {
        Self {
            multiplications: self.multiplications + other.multiplications,
            additions: self.additions + other.additions,
        }
    }

    pub fn add_mults(&self, n: usize) -> Self {
        Self {
            multiplications: self.multiplications + n,
            additions: self.additions,
        }
    }

    pub fn add_adds(&self, n: usize) -> Self {
        Self {
            multiplications: self.multiplications,
            additions: self.additions + n,
        }
    }
}

/// Cost of direct computation for a matrix of given size
fn direct_cost(size: usize) -> Cost {
    match size {
        0 | 1 => Cost::zero(),
        2 => Cost::new(2, 1), // a*d - b*c = 2 mults, 1 add
        _ => {
            // n! terms, each with n-1 multiplications, summed together
            let n_fact = (1..=size).product::<usize>();
            Cost::new(n_fact * (size - 1), n_fact - 1)
        }
    }
}

/// Extract all non-zero positions from an adjacency matrix
fn get_nonzeros(matrix: &AdjacencyMatrix) -> Vec<(usize, usize)> {
    let mut nonzeros = Vec::new();
    for r in 0..matrix.rows {
        for c in 0..matrix.cols {
            if matrix.get(r, c) {
                nonzeros.push((r, c));
            }
        }
    }
    nonzeros
}

/// Cache for storing optimal processes by canonical hash
type ProcessCache = HashMap<u64, (Cost, Process)>;

/// Find the optimal process for computing the determinant of a matrix
pub fn find_optimal_process(matrix: &AdjacencyMatrix) -> (Cost, Process) {
    let mut cache = ProcessCache::new();
    find_optimal_process_cached(matrix, &mut cache)
}

/// Internal recursive implementation with caching
fn find_optimal_process_cached(
    matrix: &AdjacencyMatrix,
    cache: &mut ProcessCache,
) -> (Cost, Process) {
    let n = matrix.rows;

    // Must be square
    assert_eq!(n, matrix.cols, "Matrix must be square");

    // Base cases
    if n <= 2 {
        return (
            direct_cost(n),
            Process::Direct {
                size: n,
                expected_nonzeros: get_nonzeros(matrix),
            },
        );
    }

    // Check canonical cache
    let canon = canonicalize(matrix);
    if let Some(cached) = cache.get(&canon.canonical_hash) {
        // Remap the cached process to use original indices
        let remapped = remap_process(&cached.1, &canon.row_perm, &canon.col_perm);
        return (cached.0, remapped);
    }

    // Insert a sentinel to prevent infinite recursion
    // This uses direct cost as a fallback if we encounter a cycle
    // Note: We store canonical nonzeros (indices 0..n) as this is for the cached canonical form
    let canonical_nonzeros: Vec<(usize, usize)> = {
        let mut nz = Vec::new();
        for new_r in 0..n {
            for new_c in 0..n {
                let old_r = canon.row_perm[new_r];
                let old_c = canon.col_perm[new_c];
                if matrix.get(old_r, old_c) {
                    nz.push((new_r, new_c));
                }
            }
        }
        nz
    };
    cache.insert(
        canon.canonical_hash,
        (
            direct_cost(n),
            Process::Direct {
                size: n,
                expected_nonzeros: canonical_nonzeros,
            },
        ),
    );

    let mut best: Option<(Cost, Process)> = None;

    // Helper to update best
    let mut update_best = |cost: Cost, process: Process| {
        if best.is_none() || cost.total() < best.as_ref().unwrap().0.total() {
            best = Some((cost, process));
        }
    };

    // Strategy 1: Block triangular decomposition via DM
    try_block_triangular(matrix, cache, &mut update_best);

    // Strategy 2: Row expansion (Laplace expansion along each row)
    for row in 0..n {
        try_row_expansion(matrix, row, cache, &mut update_best);
    }

    // Strategy 3: Column expansion (Laplace expansion along each column)
    for col in 0..n {
        try_col_expansion(matrix, col, cache, &mut update_best);
    }

    // Strategy 4: AddRow operations
    try_add_row_operations(matrix, cache, &mut update_best);

    // Note: SwapRows is NOT a useful strategy because:
    // - Swapped matrices are permutation-equivalent to the original
    // - They have the same canonical hash
    // - The optimal strategy is the same (just with remapped indices)
    // So we don't try SwapRows as a standalone strategy.

    // Fall back to direct if nothing else worked
    let result = best.unwrap_or_else(|| {
        (
            direct_cost(n),
            Process::Direct {
                size: n,
                expected_nonzeros: get_nonzeros(matrix),
            },
        )
    });

    // Cache the result (in canonical form)
    let canonical_process = canonicalize_process(&result.1, &canon.row_perm, &canon.col_perm);
    cache.insert(canon.canonical_hash, (result.0, canonical_process));

    result
}

/// Try block triangular decomposition
fn try_block_triangular<F>(matrix: &AdjacencyMatrix, cache: &mut ProcessCache, update_best: &mut F)
where
    F: FnMut(Cost, Process),
{
    let dm = dulmage_mendelsohn(matrix);

    // Only useful if we have multiple blocks
    if dm.block_sizes.len() <= 1 {
        return;
    }

    // Verify that the DM decomposition is valid for our purposes
    // (block sizes should sum to n, and permutations should be valid)
    let n = matrix.rows;
    let total_block_size: usize = dm.block_sizes.iter().sum();
    if total_block_size != n || dm.row_perm.len() != n || dm.col_perm.len() != n {
        // Invalid decomposition for a square matrix, skip
        return;
    }

    let mut total_cost = Cost::zero();
    let mut blocks = Vec::new();
    let mut offset = 0;

    for &block_size in &dm.block_sizes {
        if offset + block_size > n {
            // Safety check - shouldn't happen with valid DM result
            return;
        }

        // Extract block rows and columns
        let block_rows: Vec<usize> = dm.row_perm[offset..offset + block_size].to_vec();
        let block_cols: Vec<usize> = dm.col_perm[offset..offset + block_size].to_vec();

        let block_matrix = matrix.submatrix(&block_rows, &block_cols);
        let (block_cost, block_proc) = find_optimal_process_cached(&block_matrix, cache);

        total_cost = total_cost.add(&block_cost);
        blocks.push(Box::new(block_proc));
        offset += block_size;
    }

    // Add cost for multiplying block determinants together
    if dm.block_sizes.len() > 1 {
        total_cost = total_cost.add_mults(dm.block_sizes.len() - 1);
    }

    update_best(
        total_cost,
        Process::BlockTriangular {
            blocks,
            row_perm: dm.row_perm.clone(),
            col_perm: dm.col_perm.clone(),
            expected_nonzeros: get_nonzeros(matrix),
        },
    );
}

/// Try row expansion along a specific row
fn try_row_expansion<F>(
    matrix: &AdjacencyMatrix,
    row: usize,
    cache: &mut ProcessCache,
    update_best: &mut F,
) where
    F: FnMut(Cost, Process),
{
    let n = matrix.rows;
    let nonzero_cols = matrix.row_neighbors(row);

    // If row has no non-zeros, determinant is zero (trivial)
    if nonzero_cols.is_empty() {
        update_best(
            Cost::zero(),
            Process::Direct {
                size: n,
                expected_nonzeros: get_nonzeros(matrix),
            },
        );
        return;
    }

    let mut total_cost = Cost::zero();
    let mut minors = Vec::new();

    // Build list of rows/cols excluding the expansion row
    let remaining_rows: Vec<usize> = (0..n).filter(|&r| r != row).collect();

    for &col in &nonzero_cols {
        let remaining_cols: Vec<usize> = (0..n).filter(|&c| c != col).collect();
        let minor_matrix = matrix.submatrix(&remaining_rows, &remaining_cols);
        let (minor_cost, minor_proc) = find_optimal_process_cached(&minor_matrix, cache);

        total_cost = total_cost.add(&minor_cost);
        minors.push((col, Box::new(minor_proc)));
    }

    // Cost: k multiplications (element × minor) + (k-1) additions
    let k = nonzero_cols.len();
    total_cost = total_cost.add_mults(k);
    if k > 1 {
        total_cost = total_cost.add_adds(k - 1);
    }

    update_best(
        total_cost,
        Process::RowExpansion {
            row,
            minors,
            expected_nonzeros: get_nonzeros(matrix),
        },
    );
}

/// Try column expansion along a specific column
fn try_col_expansion<F>(
    matrix: &AdjacencyMatrix,
    col: usize,
    cache: &mut ProcessCache,
    update_best: &mut F,
) where
    F: FnMut(Cost, Process),
{
    let n = matrix.rows;
    let nonzero_rows = matrix.col_neighbors(col);

    // If column has no non-zeros, determinant is zero (trivial)
    if nonzero_rows.is_empty() {
        update_best(
            Cost::zero(),
            Process::Direct {
                size: n,
                expected_nonzeros: get_nonzeros(matrix),
            },
        );
        return;
    }

    let mut total_cost = Cost::zero();
    let mut minors = Vec::new();

    // Build list of cols excluding the expansion col
    let remaining_cols: Vec<usize> = (0..n).filter(|&c| c != col).collect();

    for &row in &nonzero_rows {
        let remaining_rows: Vec<usize> = (0..n).filter(|&r| r != row).collect();
        let minor_matrix = matrix.submatrix(&remaining_rows, &remaining_cols);
        let (minor_cost, minor_proc) = find_optimal_process_cached(&minor_matrix, cache);

        total_cost = total_cost.add(&minor_cost);
        minors.push((row, Box::new(minor_proc)));
    }

    // Cost: k multiplications (element × minor) + (k-1) additions
    let k = nonzero_rows.len();
    total_cost = total_cost.add_mults(k);
    if k > 1 {
        total_cost = total_cost.add_adds(k - 1);
    }

    update_best(
        total_cost,
        Process::ColExpansion {
            col,
            minors,
            expected_nonzeros: get_nonzeros(matrix),
        },
    );
}

/// Try all possible AddRow operations
fn try_add_row_operations<F>(
    matrix: &AdjacencyMatrix,
    cache: &mut ProcessCache,
    update_best: &mut F,
) where
    F: FnMut(Cost, Process),
{
    let n = matrix.rows;

    for src in 0..n {
        for dst in 0..n {
            if src == dst {
                continue;
            }

            // Find columns where dst has a non-zero that we could eliminate
            // (src must also have a non-zero there to use as pivot)
            for pivot_col in 0..matrix.cols {
                if !matrix.get(dst, pivot_col) || !matrix.get(src, pivot_col) {
                    continue;
                }

                // Compute the resulting matrix
                let modified = matrix.with_add_row(src, dst, pivot_col);

                // Only proceed if we actually reduced the non-zero count
                // (otherwise this operation doesn't help)
                if modified.total_nnz() >= matrix.total_nnz() {
                    continue;
                }

                let (sub_cost, sub_proc) = find_optimal_process_cached(&modified, cache);

                // Cost of AddRow:
                // - Multiplications: (src_nnz - 1) for scaling src row by -dst[pivot]/src[pivot]
                // - Additions: count of columns where both src and dst have non-zeros (excluding pivot)
                let src_nnz = matrix.row_nnz(src);
                let overlapping: usize = (0..matrix.cols)
                    .filter(|&c| c != pivot_col && matrix.get(src, c) && matrix.get(dst, c))
                    .count();

                let op_cost = Cost::new(src_nnz.saturating_sub(1), overlapping);
                let total_cost = op_cost.add(&sub_cost);

                update_best(
                    total_cost,
                    Process::AddRow {
                        src,
                        dst,
                        pivot_col,
                        result: Box::new(sub_proc),
                        expected_nonzeros: get_nonzeros(matrix),
                    },
                );
            }
        }
    }
}

// Note: try_swap_row_operations is intentionally not implemented because
// swapped matrices are permutation-equivalent to the original, meaning they
// have the same canonical form and optimal strategy. SwapRows as a standalone
// strategy would only add overhead without benefit. The SwapRows variant is
// kept in the Process enum for potential use in execution or future extensions.

/// Remap a process from canonical indices back to original indices
fn remap_process(process: &Process, row_perm: &[usize], col_perm: &[usize]) -> Process {
    // Build inverse permutations
    let mut inv_row_perm = vec![0; row_perm.len()];
    let mut inv_col_perm = vec![0; col_perm.len()];
    for (new_idx, &old_idx) in row_perm.iter().enumerate() {
        inv_row_perm[old_idx] = new_idx;
    }
    for (new_idx, &old_idx) in col_perm.iter().enumerate() {
        inv_col_perm[old_idx] = new_idx;
    }

    remap_process_with_inv(process, &inv_row_perm, &inv_col_perm)
}

fn remap_nonzeros(
    nonzeros: &[(usize, usize)],
    inv_row: &[usize],
    inv_col: &[usize],
) -> Vec<(usize, usize)> {
    nonzeros
        .iter()
        .map(|&(r, c)| {
            (
                inv_row.get(r).copied().unwrap_or(r),
                inv_col.get(c).copied().unwrap_or(c),
            )
        })
        .collect()
}

fn remap_process_with_inv(process: &Process, inv_row: &[usize], inv_col: &[usize]) -> Process {
    match process {
        Process::Direct {
            size,
            expected_nonzeros,
        } => Process::Direct {
            size: *size,
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
        Process::RowExpansion {
            row,
            minors,
            expected_nonzeros,
        } => Process::RowExpansion {
            row: inv_row.get(*row).copied().unwrap_or(*row),
            minors: minors
                .iter()
                .map(|(col, p)| {
                    (
                        inv_col.get(*col).copied().unwrap_or(*col),
                        Box::new(remap_process_with_inv(p, inv_row, inv_col)),
                    )
                })
                .collect(),
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
        Process::ColExpansion {
            col,
            minors,
            expected_nonzeros,
        } => Process::ColExpansion {
            col: inv_col.get(*col).copied().unwrap_or(*col),
            minors: minors
                .iter()
                .map(|(row, p)| {
                    (
                        inv_row.get(*row).copied().unwrap_or(*row),
                        Box::new(remap_process_with_inv(p, inv_row, inv_col)),
                    )
                })
                .collect(),
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
        Process::BlockTriangular {
            blocks,
            row_perm,
            col_perm,
            expected_nonzeros,
        } => Process::BlockTriangular {
            blocks: blocks
                .iter()
                .map(|p| Box::new(remap_process_with_inv(p, inv_row, inv_col)))
                .collect(),
            row_perm: row_perm
                .iter()
                .map(|&r| inv_row.get(r).copied().unwrap_or(r))
                .collect(),
            col_perm: col_perm
                .iter()
                .map(|&c| inv_col.get(c).copied().unwrap_or(c))
                .collect(),
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
        Process::AddRow {
            src,
            dst,
            pivot_col,
            result,
            expected_nonzeros,
        } => Process::AddRow {
            src: inv_row.get(*src).copied().unwrap_or(*src),
            dst: inv_row.get(*dst).copied().unwrap_or(*dst),
            pivot_col: inv_col.get(*pivot_col).copied().unwrap_or(*pivot_col),
            result: Box::new(remap_process_with_inv(result, inv_row, inv_col)),
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
        Process::SwapRows {
            r1,
            r2,
            result,
            expected_nonzeros,
        } => Process::SwapRows {
            r1: inv_row.get(*r1).copied().unwrap_or(*r1),
            r2: inv_row.get(*r2).copied().unwrap_or(*r2),
            result: Box::new(remap_process_with_inv(result, inv_row, inv_col)),
            expected_nonzeros: remap_nonzeros(expected_nonzeros, inv_row, inv_col),
        },
    }
}

fn canonicalize_nonzeros(
    nonzeros: &[(usize, usize)],
    row_perm: &[usize],
    col_perm: &[usize],
) -> Vec<(usize, usize)> {
    nonzeros
        .iter()
        .map(|&(r, c)| {
            let canon_r = row_perm.iter().position(|&pr| pr == r).unwrap_or(r);
            let canon_c = col_perm.iter().position(|&pc| pc == c).unwrap_or(c);
            (canon_r, canon_c)
        })
        .collect()
}

/// Convert a process to canonical form for caching
fn canonicalize_process(process: &Process, row_perm: &[usize], col_perm: &[usize]) -> Process {
    // row_perm[canonical_idx] = original_idx
    // So to go from original to canonical, we use the permutation directly
    match process {
        Process::Direct {
            size,
            expected_nonzeros,
        } => Process::Direct {
            size: *size,
            expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
        },
        Process::RowExpansion {
            row,
            minors,
            expected_nonzeros,
        } => {
            // Find canonical index for this row
            let canon_row = row_perm.iter().position(|&r| r == *row).unwrap_or(*row);
            Process::RowExpansion {
                row: canon_row,
                minors: minors
                    .iter()
                    .map(|(col, p)| {
                        let canon_col = col_perm.iter().position(|&c| c == *col).unwrap_or(*col);
                        (
                            canon_col,
                            Box::new(canonicalize_process(p, row_perm, col_perm)),
                        )
                    })
                    .collect(),
                expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
            }
        }
        Process::ColExpansion {
            col,
            minors,
            expected_nonzeros,
        } => {
            let canon_col = col_perm.iter().position(|&c| c == *col).unwrap_or(*col);
            Process::ColExpansion {
                col: canon_col,
                minors: minors
                    .iter()
                    .map(|(row, p)| {
                        let canon_row = row_perm.iter().position(|&r| r == *row).unwrap_or(*row);
                        (
                            canon_row,
                            Box::new(canonicalize_process(p, row_perm, col_perm)),
                        )
                    })
                    .collect(),
                expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
            }
        }
        Process::BlockTriangular {
            blocks,
            row_perm: block_row_perm,
            col_perm: block_col_perm,
            expected_nonzeros,
        } => Process::BlockTriangular {
            blocks: blocks
                .iter()
                .map(|p| Box::new(canonicalize_process(p, row_perm, col_perm)))
                .collect(),
            row_perm: block_row_perm
                .iter()
                .map(|&r| row_perm.iter().position(|&pr| pr == r).unwrap_or(r))
                .collect(),
            col_perm: block_col_perm
                .iter()
                .map(|&c| col_perm.iter().position(|&pc| pc == c).unwrap_or(c))
                .collect(),
            expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
        },
        Process::AddRow {
            src,
            dst,
            pivot_col,
            result,
            expected_nonzeros,
        } => {
            let canon_src = row_perm.iter().position(|&r| r == *src).unwrap_or(*src);
            let canon_dst = row_perm.iter().position(|&r| r == *dst).unwrap_or(*dst);
            let canon_pivot = col_perm
                .iter()
                .position(|&c| c == *pivot_col)
                .unwrap_or(*pivot_col);
            Process::AddRow {
                src: canon_src,
                dst: canon_dst,
                pivot_col: canon_pivot,
                result: Box::new(canonicalize_process(result, row_perm, col_perm)),
                expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
            }
        }
        Process::SwapRows {
            r1,
            r2,
            result,
            expected_nonzeros,
        } => {
            let canon_r1 = row_perm.iter().position(|&r| r == *r1).unwrap_or(*r1);
            let canon_r2 = row_perm.iter().position(|&r| r == *r2).unwrap_or(*r2);
            Process::SwapRows {
                r1: canon_r1,
                r2: canon_r2,
                result: Box::new(canonicalize_process(result, row_perm, col_perm)),
                expected_nonzeros: canonicalize_nonzeros(expected_nonzeros, row_perm, col_perm),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix(data: Vec<Vec<bool>>) -> AdjacencyMatrix {
        AdjacencyMatrix::from_vec(data)
    }

    #[test]
    fn test_direct_1x1() {
        let m = make_matrix(vec![vec![true]]);
        let (cost, proc) = find_optimal_process(&m);
        assert_eq!(cost.total(), 0);
        assert!(matches!(proc, Process::Direct { size: 1, .. }));
    }

    #[test]
    fn test_direct_2x2() {
        let m = make_matrix(vec![vec![true, true], vec![true, true]]);
        let (cost, proc) = find_optimal_process(&m);
        assert_eq!(cost.total(), 3); // 2 mults + 1 add
        assert!(matches!(proc, Process::Direct { size: 2, .. }));
    }

    #[test]
    fn test_sparse_3x3_diagonal() {
        // Diagonal matrix - should use block triangular (3 blocks of size 1)
        let m = make_matrix(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]);
        let (cost, proc) = find_optimal_process(&m);
        // Diagonal: det = a * b * c = 2 mults, 0 adds
        assert_eq!(cost.multiplications, 2);
        assert_eq!(cost.additions, 0);
        assert!(matches!(proc, Process::BlockTriangular { .. }));
    }

    #[test]
    fn test_row_with_single_nonzero() {
        // Lower triangular matrix - row 0 has only one non-zero
        let m = make_matrix(vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![true, true, true],
        ]);
        let (cost, proc) = find_optimal_process(&m);
        // Should use expansion along sparse row/col or block triangular
        // Expected: block triangular with 3 blocks of size 1, cost = 2 mults
        assert!(cost.total() <= 5);
        // Either block triangular or row expansion along row 0
        match proc {
            Process::BlockTriangular { .. } => {}
            Process::RowExpansion { row, .. } if row == 0 => {}
            Process::ColExpansion { col, .. } if col == 2 => {}
            _ => {} // Other valid strategies are acceptable
        }
    }

    #[test]
    fn test_block_diagonal_2x2_blocks() {
        // 4x4 matrix with two 2x2 blocks on diagonal
        let m = make_matrix(vec![
            vec![true, true, false, false],
            vec![true, true, false, false],
            vec![false, false, true, true],
            vec![false, false, true, true],
        ]);
        let (cost, proc) = find_optimal_process(&m);
        // Should find block triangular: 2 blocks of 2x2
        // Each 2x2 costs 3, plus 1 mult to combine = 7
        assert!(matches!(proc, Process::BlockTriangular { .. }));
        assert_eq!(cost.total(), 7);
    }

    #[test]
    fn test_full_3x3() {
        // Full 3x3 matrix - no sparse structure to exploit
        let m = make_matrix(vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ]);
        let (cost, proc) = find_optimal_process(&m);
        // Best strategy is row/col expansion: 3 minors of 2x2 (cost 3 each = 9)
        // Plus 3 mults + 2 adds = 14
        // Or with AddRow we might do better
        assert!(cost.total() <= 14);
        // Should use some expansion strategy
        match proc {
            Process::RowExpansion { .. }
            | Process::ColExpansion { .. }
            | Process::AddRow { .. } => {}
            _ => panic!("Expected expansion or row operation for full matrix"),
        }
    }

    #[test]
    fn test_add_row_beneficial() {
        // Matrix where AddRow can help: sparse with one dense row
        // After elimination, we get a sparser matrix
        let m = make_matrix(vec![
            vec![true, true, true, true],
            vec![true, false, false, false],
            vec![false, true, false, false],
            vec![false, false, true, false],
        ]);
        let (cost, _proc) = find_optimal_process(&m);
        // Row 0 is dense, but rows 1-3 are very sparse
        // Should find an efficient strategy
        assert!(cost.total() < 50);
    }

    #[test]
    fn test_zero_determinant_sparse_row() {
        // Matrix with a zero row - determinant is trivially 0
        let m = make_matrix(vec![
            vec![false, false, false],
            vec![true, true, true],
            vec![true, true, true],
        ]);
        let (cost, _proc) = find_optimal_process(&m);
        // Should recognize immediately that det = 0
        assert_eq!(cost.total(), 0);
    }

    #[test]
    fn test_zero_determinant_sparse_col() {
        // Matrix with a zero column - determinant is trivially 0
        let m = make_matrix(vec![
            vec![false, true, true],
            vec![false, true, true],
            vec![false, true, true],
        ]);
        let (cost, _proc) = find_optimal_process(&m);
        // Should recognize immediately that det = 0
        assert_eq!(cost.total(), 0);
    }

    #[test]
    fn test_larger_sparse_matrix() {
        // 5x5 sparse matrix with specific structure
        let m = make_matrix(vec![
            vec![true, true, false, false, false],
            vec![true, true, true, false, false],
            vec![false, true, true, true, false],
            vec![false, false, true, true, true],
            vec![false, false, false, true, true],
        ]);
        let (cost, _proc) = find_optimal_process(&m);
        // Tridiagonal-like structure should be efficient
        // Much cheaper than 5! = 120
        assert!(cost.total() < 50);
    }

    #[test]
    fn test_permuted_matrix_same_cost() {
        // Two permutation-equivalent matrices should have same optimal cost
        let m1 = make_matrix(vec![
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, false],
        ]);
        let m2 = make_matrix(vec![
            vec![true, false, false],
            vec![false, true, true],
            vec![true, true, false],
        ]);

        let (cost1, _) = find_optimal_process(&m1);
        let (cost2, _) = find_optimal_process(&m2);
        assert_eq!(cost1.total(), cost2.total());
    }

    #[test]
    fn test_cost_model_consistency() {
        // Verify cost calculations are consistent
        let cost = Cost::new(5, 3);
        assert_eq!(cost.total(), 8);
        assert_eq!(cost.add_mults(2).multiplications, 7);
        assert_eq!(cost.add_adds(2).additions, 5);
        assert_eq!(cost.add(&Cost::new(1, 1)).total(), 10);
    }
}
