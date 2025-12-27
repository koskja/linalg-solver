//! Matrix Canonicalization
//!
//! Finds a canonical form C = PXQ for any matrix X such that all
//! permutation-equivalent matrices map to the same canonical form.
//!
//! Uses Weisfeiler-Lehman color refinement followed by lexicographic
//! tie-breaking for a deterministic canonical representative.

use std::collections::{BTreeMap, HashMap};

use pyo3::prelude::*;

use crate::adjacency::AdjacencyMatrix;
use crate::bitlist::BitList;
use crate::permutation::Permutation;

/// Result of matrix canonicalization
#[pyclass]
#[derive(Clone, Debug)]
pub struct CanonicalForm {
    /// Row permutation: canonical_row[i] = original_row[row_perm[i]]
    pub row_perm: Permutation,
    /// Column permutation: canonical_col[j] = original_col[col_perm[j]]
    pub col_perm: Permutation,
    /// A hash of the canonical form for quick equality checks
    #[pyo3(get)]
    pub canonical_hash: u64,
}

impl CanonicalForm {
    /// Get the row permutation.
    pub fn row_perm(&self) -> &Permutation {
        &self.row_perm
    }

    /// Get the column permutation.
    pub fn col_perm(&self) -> &Permutation {
        &self.col_perm
    }
}

#[pymethods]
impl CanonicalForm {
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
            "CanonicalForm(row_perm={:?}, col_perm={:?}, hash={:#x})",
            self.row_perm.as_slice(),
            self.col_perm.as_slice(),
            self.canonical_hash
        )
    }
}

/// Color/label for a vertex during WL refinement
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Color(Vec<usize>);

impl Color {
    fn new(label: usize) -> Self {
        Color(vec![label])
    }

    fn from_multiset(own_color: usize, neighbor_colors: &mut Vec<usize>) -> Self {
        neighbor_colors.sort();
        let mut v = vec![own_color];
        v.extend(neighbor_colors.iter());
        Color(v)
    }
}

/// Weisfeiler-Lehman refinement for bipartite graphs (matrices)
/// Returns stable partitions of rows and columns
fn wl_refine(graph: &AdjacencyMatrix) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_rows = graph.rows;
    let n_cols = graph.cols;

    if n_rows == 0 || n_cols == 0 {
        return (vec![], vec![]);
    }

    // Initialize colors based on degree
    let mut row_colors: Vec<Color> = (0..n_rows).map(|r| Color::new(graph.row_nnz(r))).collect();
    let mut col_colors: Vec<Color> = (0..n_cols).map(|c| Color::new(graph.col_nnz(c))).collect();

    // Iterate until stable
    for _ in 0..n_rows + n_cols {
        let old_row_colors = row_colors.clone();
        let old_col_colors = col_colors.clone();

        // Map colors to integers for efficiency (use old colors for consistent mapping)
        let row_color_map: HashMap<Color, usize> = old_row_colors
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        let col_color_map: HashMap<Color, usize> = old_col_colors
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();

        // Update row colors based on neighbor column colors
        for r in 0..n_rows {
            let mut neighbor_cols = Vec::with_capacity(graph.row_nnz(r));
            for c in 0..n_cols {
                if graph.get(r, c) {
                    neighbor_cols.push(col_color_map[&old_col_colors[c]]);
                }
            }
            row_colors[r] =
                Color::from_multiset(row_color_map[&old_row_colors[r]], &mut neighbor_cols);
        }

        // Update column colors based on neighbor row colors
        for c in 0..n_cols {
            let mut neighbor_rows = Vec::with_capacity(graph.col_nnz(c));
            for r in 0..n_rows {
                if graph.get(r, c) {
                    neighbor_rows.push(row_color_map[&old_row_colors[r]]);
                }
            }
            col_colors[c] =
                Color::from_multiset(col_color_map[&old_col_colors[c]], &mut neighbor_rows);
        }

        // Check for stability
        if row_colors == old_row_colors && col_colors == old_col_colors {
            break;
        }
    }

    // Group by colors
    let row_partitions = group_by_color(&row_colors);
    let col_partitions = group_by_color(&col_colors);

    (row_partitions, col_partitions)
}

/// Group indices by their color, returning groups sorted by color
fn group_by_color(colors: &[Color]) -> Vec<Vec<usize>> {
    let mut groups: BTreeMap<&Color, Vec<usize>> = BTreeMap::new();
    for (idx, color) in colors.iter().enumerate() {
        groups.entry(color).or_default().push(idx);
    }
    groups.into_values().collect()
}

/// Compute the lexicographic signature of a vertex relative to an ordering
fn row_signature(graph: &AdjacencyMatrix, row: usize, col_order: &[usize]) -> BitList {
    BitList::from_bools(col_order.iter().map(|&c| graph.get(row, c)))
}

fn col_signature(graph: &AdjacencyMatrix, col: usize, row_order: &[usize]) -> BitList {
    BitList::from_bools(row_order.iter().map(|&r| graph.get(r, col)))
}

/// Order indices within a partition lexicographically by their signatures
fn order_partition_lex<F, S>(partition: &[usize], mut signature_fn: F) -> Vec<usize>
where
    F: FnMut(usize) -> S,
    S: Ord,
{
    let mut indexed: Vec<(usize, S)> = partition.iter().map(|&i| (i, signature_fn(i))).collect();
    indexed.sort_by(|a, b| a.1.cmp(&b.1));
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Find canonical form using WL refinement + lexicographic ordering
pub fn canonicalize(graph: &AdjacencyMatrix) -> CanonicalForm {
    let n_rows = graph.rows;
    let n_cols = graph.cols;

    if n_rows == 0 || n_cols == 0 {
        return CanonicalForm {
            row_perm: Permutation::identity(n_rows),
            col_perm: Permutation::identity(n_cols),
            canonical_hash: 0,
        };
    }

    // Step 1: WL refinement to get stable partitions
    let (row_partitions, col_partitions) = wl_refine(graph);

    // Step 2: Build canonical ordering by processing partitions
    // Start with a preliminary column order (indices sorted by partition then by index)
    let mut col_perm_vec: Vec<usize> = col_partitions
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();

    // Step 3: Iteratively refine row and column orderings
    // Order rows lexicographically within each partition based on current column order
    let mut row_perm_vec: Vec<usize> = Vec::with_capacity(n_rows);
    for partition in &row_partitions {
        let ordered = order_partition_lex(partition, |r| row_signature(graph, r, &col_perm_vec));
        row_perm_vec.extend(ordered);
    }

    // Re-order columns based on the new row order
    col_perm_vec.clear();
    for partition in &col_partitions {
        let ordered = order_partition_lex(partition, |c| col_signature(graph, c, &row_perm_vec));
        col_perm_vec.extend(ordered);
    }

    // One more pass to stabilize
    row_perm_vec.clear();
    for partition in &row_partitions {
        let ordered = order_partition_lex(partition, |r| row_signature(graph, r, &col_perm_vec));
        row_perm_vec.extend(ordered);
    }

    // Step 4: Compute canonical hash
    let canonical_hash = compute_hash(graph, &row_perm_vec, &col_perm_vec);

    CanonicalForm {
        row_perm: Permutation::from_vec_unchecked(row_perm_vec),
        col_perm: Permutation::from_vec_unchecked(col_perm_vec),
        canonical_hash,
    }
}

/// Compute a hash of the matrix in canonical form
fn compute_hash(graph: &AdjacencyMatrix, row_perm: &[usize], col_perm: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Hash dimensions
    graph.rows.hash(&mut hasher);
    graph.cols.hash(&mut hasher);

    // Hash the adjacency pattern in canonical order
    for &r in row_perm {
        for &c in col_perm {
            graph.get(r, c).hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Verify that two matrices have the same canonical form
pub fn are_permutation_equivalent(a: &AdjacencyMatrix, b: &AdjacencyMatrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }

    let canon_a = canonicalize(a);
    let canon_b = canonicalize(b);

    if canon_a.canonical_hash != canon_b.canonical_hash {
        return false;
    }

    // Verify by comparing actual canonical forms (in case of hash collision)
    for (i, &ra) in canon_a.row_perm().iter().enumerate() {
        for (j, &ca) in canon_a.col_perm().iter().enumerate() {
            let rb = canon_b.row_perm()[i];
            let cb = canon_b.col_perm()[j];
            if a.get(ra, ca) != b.get(rb, cb) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix(data: Vec<Vec<bool>>) -> AdjacencyMatrix {
        AdjacencyMatrix::from_vec(data)
    }

    #[test]
    fn test_identity_canonical() {
        let m = make_matrix(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]);
        let canon = canonicalize(&m);
        assert_eq!(canon.row_perm().len(), 3);
        assert_eq!(canon.col_perm().len(), 3);
    }

    #[test]
    fn test_permuted_matrices_same_canonical() {
        // Original matrix
        let m1 = make_matrix(vec![
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, false],
        ]);

        // Same matrix with rows permuted (swap rows 0 and 2)
        let m2 = make_matrix(vec![
            vec![true, false, false],
            vec![false, true, true],
            vec![true, true, false],
        ]);

        assert!(are_permutation_equivalent(&m1, &m2));
    }

    #[test]
    fn test_different_matrices_different_canonical() {
        let m1 = make_matrix(vec![
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, false],
        ]);

        let m2 = make_matrix(vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ]);

        assert!(!are_permutation_equivalent(&m1, &m2));
    }

    #[test]
    fn test_row_and_col_permutation() {
        // Original
        let m1 = make_matrix(vec![vec![true, false], vec![false, true]]);

        // Swap both rows and columns
        let m2 = make_matrix(vec![vec![true, false], vec![false, true]]);

        assert!(are_permutation_equivalent(&m1, &m2));
    }

    #[test]
    fn test_canonical_hash_stability() {
        let m = make_matrix(vec![
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, true],
        ]);

        let c1 = canonicalize(&m);
        let c2 = canonicalize(&m);

        assert_eq!(c1.canonical_hash, c2.canonical_hash);
        assert_eq!(c1.row_perm(), c2.row_perm());
        assert_eq!(c1.col_perm(), c2.col_perm());
    }
}
