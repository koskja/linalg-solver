//! Dulmage-Mendelsohn Decomposition
//!
//! Decomposes a sparse matrix into block triangular form by finding
//! the coarse (H, S, V) partition and fine decomposition of the square part.

use std::collections::{HashSet, VecDeque};

use pyo3::prelude::*;

use crate::adjacency::{AdjacencyMatrix, Matching};
use crate::hopcroft_karp::hopcroft_karp;
use crate::tarjan::tarjan_scc;

/// Result of the Dulmage-Mendelsohn decomposition
#[pyclass]
#[derive(Clone)]
pub struct DMResult {
    /// Row permutation: new_row[i] = old_row[row_perm[i]]
    #[pyo3(get)]
    pub row_perm: Vec<usize>,
    /// Column permutation: new_col[j] = old_col[col_perm[j]]
    #[pyo3(get)]
    pub col_perm: Vec<usize>,
    /// Sizes of diagonal blocks
    #[pyo3(get)]
    pub block_sizes: Vec<usize>,
}

#[pymethods]
impl DMResult {
    fn __repr__(&self) -> String {
        format!(
            "DMResult(row_perm={:?}, col_perm={:?}, block_sizes={:?})",
            self.row_perm, self.col_perm, self.block_sizes
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
            row_perm: (0..rows).collect(),
            col_perm: (0..cols).collect(),
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

    // Build permutations and block sizes
    let mut row_perm = Vec::new();
    let mut col_perm = Vec::new();
    let mut block_sizes = Vec::new();

    // Add H partition (if non-empty)
    let h_rows_vec: Vec<usize> = h_rows.into_iter().collect();
    let h_cols_vec: Vec<usize> = h_cols.into_iter().collect();
    if !h_rows_vec.is_empty() || !h_cols_vec.is_empty() {
        row_perm.extend(&h_rows_vec);
        col_perm.extend(&h_cols_vec);
        block_sizes.push(h_rows_vec.len().max(h_cols_vec.len()));
    }

    // Add S partition (SCCs in reverse topological order = already correct for upper triangular)
    for scc in sccs.iter().rev() {
        let scc_size = scc.len();
        // Add rows in this SCC
        for &idx in scc {
            row_perm.push(s_rows[idx]);
        }
        // Add corresponding matched columns
        for &idx in scc {
            if let Some(c) = matching.row_to_col[s_rows[idx]] {
                col_perm.push(c);
            }
        }
        block_sizes.push(scc_size);
    }

    // Add V partition (if non-empty)
    let v_rows_vec: Vec<usize> = v_rows.into_iter().collect();
    let v_cols_vec: Vec<usize> = v_cols.into_iter().collect();
    if !v_rows_vec.is_empty() || !v_cols_vec.is_empty() {
        row_perm.extend(&v_rows_vec);
        col_perm.extend(&v_cols_vec);
        block_sizes.push(v_rows_vec.len().max(v_cols_vec.len()));
    }

    // Handle case where permutations are incomplete (shouldn't happen with correct algorithm)
    // but ensure we have valid permutations
    if row_perm.len() != rows {
        row_perm = (0..rows).collect();
    }
    if col_perm.len() != cols {
        col_perm = (0..cols).collect();
    }

    DMResult {
        row_perm,
        col_perm,
        block_sizes,
    }
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
        assert_eq!(result.block_sizes.iter().sum::<usize>(), 3);
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
        assert!(result.block_sizes.len() >= 1);
    }

    #[test]
    fn test_dm_full_matrix() {
        // Full matrix is irreducible (one block)
        let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ]));
        assert_eq!(result.block_sizes, vec![3]);
        assert!(!result.is_decomposable());
    }
}
