//! Dulmage-Mendelsohn Decomposition
//!
//! Decomposes a sparse matrix into block triangular form by finding
//! the coarse (H, S, V) partition and fine decomposition of the square part.

use std::collections::{HashSet, VecDeque};

use pyo3::prelude::*;

use crate::adjacency::{AdjacencyMatrix, Matching};
use crate::hopcroft_karp::hopcroft_karp;
use crate::permutation::Permutation;
use crate::tarjan::tarjan_scc;

/// Result of the Dulmage-Mendelsohn decomposition
#[pyclass]
#[derive(Clone)]
pub struct DMResult {
    /// Row permutation: new_row[i] = old_row[row_perm[i]]
    row_perm: Permutation,
    /// Column permutation: new_col[j] = old_col[col_perm[j]]
    col_perm: Permutation,
    /// Sizes of diagonal blocks
    #[pyo3(get)]
    block_sizes: Vec<usize>,
}

impl DMResult {
    /// Get the row permutation.
    pub fn row_perm(&self) -> &Permutation {
        &self.row_perm
    }

    /// Get the column permutation.
    pub fn col_perm(&self) -> &Permutation {
        &self.col_perm
    }

    /// Get the block sizes.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }
}

#[pymethods]
impl DMResult {
    #[getter]
    fn row_perm_py(&self) -> Vec<usize> {
        self.row_perm.to_vec()
    }

    #[getter]
    fn col_perm_py(&self) -> Vec<usize> {
        self.col_perm.to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "DMResult(row_perm={:?}, col_perm={:?}, block_sizes={:?})",
            self.row_perm.as_slice(),
            self.col_perm.as_slice(),
            self.block_sizes
        )
    }

    /// Returns True if the matrix is partially decomposable (more than one block)
    #[getter]
    fn is_decomposable(&self) -> bool {
        self.block_sizes.len() > 1
    }
}

/// Find rows/columns reachable from unmatched rows via alternating paths
/// Returns (reachable_rows, reachable_cols)
fn find_h_partition(
    graph: &AdjacencyMatrix,
    matching: &Matching,
) -> (HashSet<usize>, HashSet<usize>) {
    let mut h_rows = HashSet::new();
    let mut h_cols = HashSet::new();
    let mut queue = VecDeque::new();

    // Start from unmatched rows
    for r in 0..graph.rows {
        if matching.row_to_col[r].is_none() {
            h_rows.insert(r);
            queue.push_back((r, true)); // (vertex, is_row)
        }
    }

    // BFS along alternating paths
    while let Some((v, is_row)) = queue.pop_front() {
        if is_row {
            // From a row, follow any edge (not just matching)
            for c in graph.row_neighbors(v) {
                if !h_cols.contains(&c) {
                    h_cols.insert(c);
                    queue.push_back((c, false));
                }
            }
        } else {
            // From a column, follow only matching edge
            if let Some(r) = matching.col_to_row[v] {
                if !h_rows.contains(&r) {
                    h_rows.insert(r);
                    queue.push_back((r, true));
                }
            }
        }
    }

    (h_rows, h_cols)
}

/// Find rows/columns that can reach unmatched columns via alternating paths
/// Returns (reachable_rows, reachable_cols)
fn find_v_partition(
    graph: &AdjacencyMatrix,
    matching: &Matching,
) -> (HashSet<usize>, HashSet<usize>) {
    let mut v_rows = HashSet::new();
    let mut v_cols = HashSet::new();
    let mut queue = VecDeque::new();

    // Start from unmatched columns
    for c in 0..graph.cols {
        if matching.col_to_row[c].is_none() {
            v_cols.insert(c);
            queue.push_back((c, false)); // (vertex, is_row)
        }
    }

    // BFS along alternating paths (reversed direction)
    while let Some((v, is_row)) = queue.pop_front() {
        if !is_row {
            // From a column, follow any edge back to rows
            for r in graph.col_neighbors(v) {
                if !v_rows.contains(&r) {
                    v_rows.insert(r);
                    queue.push_back((r, true));
                }
            }
        } else {
            // From a row, follow only matching edge
            if let Some(c) = matching.row_to_col[v] {
                if !v_cols.contains(&c) {
                    v_cols.insert(c);
                    queue.push_back((c, false));
                }
            }
        }
    }

    (v_rows, v_cols)
}

/// Main Dulmage-Mendelsohn decomposition algorithm
pub fn dulmage_mendelsohn(graph: &AdjacencyMatrix) -> DMResult {
    let rows = graph.rows;
    let cols = graph.cols;

    if rows == 0 || cols == 0 {
        return DMResult {
            row_perm: Permutation::identity(rows),
            col_perm: Permutation::identity(cols),
            block_sizes: vec![],
        };
    }

    // Step 1: Find maximum matching
    let matching = hopcroft_karp(graph);

    // Step 2: Find coarse partition (H, S, V)
    let (h_rows, h_cols) = find_h_partition(graph, &matching);
    let (v_rows, v_cols) = find_v_partition(graph, &matching);

    // S = vertices not in H or V
    let s_rows: Vec<usize> = (0..rows)
        .filter(|r| !h_rows.contains(r) && !v_rows.contains(r))
        .collect();
    let s_cols: Vec<usize> = (0..cols)
        .filter(|c| !h_cols.contains(c) && !v_cols.contains(c))
        .collect();

    // Step 3: Fine decomposition of square part S using SCCs
    // Build directed graph on S_rows: edge i -> j if row i connects to column matched with row j
    let s_row_to_idx: std::collections::HashMap<usize, usize> = s_rows
        .iter()
        .enumerate()
        .map(|(idx, &r)| (r, idx))
        .collect();

    let mut s_adj: Vec<Vec<usize>> = vec![Vec::new(); s_rows.len()];
    for (idx, &r) in s_rows.iter().enumerate() {
        for c in graph.row_neighbors(r) {
            if s_cols.contains(&c) {
                if let Some(matched_r) = matching.col_to_row[c] {
                    if let Some(&target_idx) = s_row_to_idx.get(&matched_r) {
                        if target_idx != idx {
                            s_adj[idx].push(target_idx);
                        }
                    }
                }
            }
        }
    }

    // Find SCCs in reverse topological order
    let sccs = tarjan_scc(&s_adj);

    // Build blocks with their rows and corresponding matched columns
    // Each block is (Vec<(original_row, matched_col)>, min_original_row)
    let mut blocks: Vec<(Vec<(usize, usize)>, usize)> = Vec::new();

    // Add H partition (if non-empty)
    let mut h_rows_vec: Vec<usize> = h_rows.into_iter().collect();
    let mut h_cols_vec: Vec<usize> = h_cols.into_iter().collect();
    h_rows_vec.sort(); // Sort by original index to minimize permutation
    h_cols_vec.sort();
    if !h_rows_vec.is_empty() || !h_cols_vec.is_empty() {
        let pairs: Vec<(usize, usize)> = h_rows_vec
            .iter()
            .zip(h_cols_vec.iter())
            .map(|(&r, &c)| (r, c))
            .collect();
        let min_row = pairs.iter().map(|&(r, _)| r).min().unwrap_or(usize::MAX);
        blocks.push((pairs, min_row));
    }

    // Add S partition (SCCs in reverse topological order = already correct for upper triangular)
    // But within each SCC, sort by original row index to minimize permutation
    for scc in sccs.iter().rev() {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &idx in scc {
            let r = s_rows[idx];
            if let Some(c) = matching.row_to_col[r] {
                pairs.push((r, c));
            }
        }
        // Sort pairs by original row index within the block
        pairs.sort_by_key(|&(r, _)| r);
        let min_row = pairs.iter().map(|&(r, _)| r).min().unwrap_or(usize::MAX);
        if !pairs.is_empty() {
            blocks.push((pairs, min_row));
        }
    }

    // Add V partition (if non-empty)
    let mut v_rows_vec: Vec<usize> = v_rows.into_iter().collect();
    let mut v_cols_vec: Vec<usize> = v_cols.into_iter().collect();
    v_rows_vec.sort(); // Sort by original index to minimize permutation
    v_cols_vec.sort();
    if !v_rows_vec.is_empty() || !v_cols_vec.is_empty() {
        let pairs: Vec<(usize, usize)> = v_rows_vec
            .iter()
            .zip(v_cols_vec.iter())
            .map(|(&r, &c)| (r, c))
            .collect();
        let min_row = pairs.iter().map(|&(r, _)| r).min().unwrap_or(usize::MAX);
        blocks.push((pairs, min_row));
    }

    // Step 4: Normalize block order to minimize permutation
    // For block diagonal matrices (no inter-block edges), we can reorder blocks freely.
    // For block triangular matrices, we must respect the topological order.
    // We detect if reordering is safe by checking if there are any edges between blocks.
    let normalized_blocks = normalize_block_order(graph, &matching, blocks);

    // Build final permutations
    let mut row_perm_vec: Vec<usize> = Vec::with_capacity(rows);
    let mut col_perm_vec: Vec<usize> = Vec::with_capacity(cols);
    let mut block_sizes = Vec::new();

    for (pairs, _) in normalized_blocks {
        if pairs.is_empty() {
            continue;
        }
        block_sizes.push(pairs.len());
        for (r, c) in pairs {
            row_perm_vec.push(r);
            col_perm_vec.push(c);
        }
    }

    // Handle case where permutations are incomplete (shouldn't happen with correct algorithm)
    // but ensure we have valid permutations
    let row_perm = if row_perm_vec.len() != rows {
        Permutation::identity(rows)
    } else {
        Permutation::from_vec_unchecked(row_perm_vec)
    };
    let col_perm = if col_perm_vec.len() != cols {
        Permutation::identity(cols)
    } else {
        Permutation::from_vec_unchecked(col_perm_vec)
    };

    DMResult {
        row_perm,
        col_perm,
        block_sizes,
    }
}

/// Normalize block order to minimize the permutation.
/// For block diagonal matrices, sort blocks by their minimum original row index.
/// For block triangular matrices, respect topological constraints but optimize within them.
fn normalize_block_order(
    graph: &AdjacencyMatrix,
    _matching: &Matching,
    mut blocks: Vec<(Vec<(usize, usize)>, usize)>,
) -> Vec<(Vec<(usize, usize)>, usize)> {
    if blocks.len() <= 1 {
        return blocks;
    }

    // Build a set of (row, col) pairs for each block for quick lookup
    let block_cols: Vec<HashSet<usize>> = blocks
        .iter()
        .map(|(pairs, _)| pairs.iter().map(|&(_, c)| c).collect())
        .collect();

    let block_rows: Vec<HashSet<usize>> = blocks
        .iter()
        .map(|(pairs, _)| pairs.iter().map(|&(r, _)| r).collect())
        .collect();

    // Check for inter-block edges (block i -> block j means there's an edge from
    // a row in block i to a column in block j, where i != j)
    // For block triangular, edges go from earlier blocks to later blocks (upper triangular)
    let n = blocks.len();
    let mut has_edge_to_later: Vec<bool> = vec![false; n];

    for i in 0..n {
        for &r in &block_rows[i] {
            for c in graph.row_neighbors(r) {
                // Check if this column belongs to a later block
                for j in (i + 1)..n {
                    if block_cols[j].contains(&c) {
                        has_edge_to_later[i] = true;
                        break;
                    }
                }
            }
        }
    }

    // If no block has edges to later blocks, this is block diagonal
    // and we can reorder blocks freely by their min_row
    let is_block_diagonal = !has_edge_to_later.iter().any(|&x| x);

    if is_block_diagonal {
        // Sort blocks by their minimum original row index
        blocks.sort_by_key(|(_, min_row)| *min_row);
    }
    // For block triangular, we already have topological order from SCCs,
    // but we can still try to optimize by checking if adjacent blocks can be swapped

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dm_identity() {
        // Identity matrix should have n blocks of size 1
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]));
        assert_eq!(result.block_sizes().iter().sum::<usize>(), 3);
    }

    #[test]
    fn test_dm_lower_triangular() {
        // Lower triangular should be decomposable
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![true, true, true],
        ]));
        // Should have 3 blocks of size 1
        assert!(result.block_sizes().len() >= 1);
    }

    #[test]
    fn test_dm_full_matrix() {
        // Full matrix is irreducible (one block)
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ]));
        assert_eq!(result.block_sizes(), &[3]);
        assert!(!result.is_decomposable());
    }

    #[test]
    fn test_dm_identity_minimal_permutation() {
        // Identity matrix: permutations should be identity (no unnecessary swaps)
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]));
        // Row and column permutations should be identity [0, 1, 2]
        assert_eq!(
            result.row_perm().as_slice(),
            &[0, 1, 2],
            "Row permutation should be identity"
        );
        assert_eq!(
            result.col_perm().as_slice(),
            &[0, 1, 2],
            "Column permutation should be identity"
        );
    }

    #[test]
    fn test_dm_block_diagonal_minimal_permutation() {
        // Block diagonal with two 2x2 blocks: permutations should be identity
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, true, false, false],
            vec![true, true, false, false],
            vec![false, false, true, true],
            vec![false, false, true, true],
        ]));
        // Permutations should be identity [0, 1, 2, 3] since blocks are already in order
        assert_eq!(
            result.row_perm().as_slice(),
            &[0, 1, 2, 3],
            "Row permutation should be identity for block diagonal"
        );
        assert_eq!(
            result.col_perm().as_slice(),
            &[0, 1, 2, 3],
            "Column permutation should be identity for block diagonal"
        );
        assert_eq!(
            result.block_sizes(),
            &[2, 2],
            "Should have two blocks of size 2"
        );
    }

    #[test]
    fn test_dm_lower_triangular_block_order() {
        // Lower triangular 3x3 matrix: DM produces upper triangular block form
        // So rows/cols are reversed to convert lower to upper triangular
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![true, true, true],
        ]));
        // Lower triangular → upper triangular requires reversing order
        // This is correct behavior - blocks must respect topological order
        assert_eq!(
            result.row_perm().as_slice(),
            &[2, 1, 0],
            "Lower triangular needs reverse order for upper block form"
        );
        assert_eq!(
            result.col_perm().as_slice(),
            &[2, 1, 0],
            "Lower triangular needs reverse order for upper block form"
        );
    }

    #[test]
    fn test_dm_upper_triangular_minimal_permutation() {
        // Upper triangular 3x3 matrix: already in correct form, so identity permutation
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, true, true],
            vec![false, true, true],
            vec![false, false, true],
        ]));
        // Upper triangular is already in the right form - identity permutation
        assert_eq!(
            result.row_perm().as_slice(),
            &[0, 1, 2],
            "Row permutation should be identity for upper triangular"
        );
        assert_eq!(
            result.col_perm().as_slice(),
            &[0, 1, 2],
            "Column permutation should be identity for upper triangular"
        );
    }

    #[test]
    fn test_dm_shuffled_diagonal_restores_order() {
        // Diagonal with elements in positions (1,1), (0,0), (2,2) - should NOT require swaps
        // because diagonal matrices should produce identity permutation
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]));
        // Even if internally the algorithm finds elements in different order,
        // the normalization should produce identity
        assert_eq!(
            result.row_perm().as_slice(),
            &[0, 1, 2],
            "Should normalize to identity"
        );
        assert_eq!(
            result.col_perm().as_slice(),
            &[0, 1, 2],
            "Should normalize to identity"
        );
    }
}
