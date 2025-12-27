//! Permutation types for matrix row/column operations.
//!
//! This module provides `Permutation` and `RowColPermutation` types
//! that mirror the Python implementations in `linalg_solver/permutation.py`.

use std::ops::Index;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use smallvec::SmallVec;

use crate::{INLINE_PERM_CAPACITY, MatrixIndex};

/// A permutation of 0..n-1.
///
/// Internally stored as a vector where perm[i] = j means i maps to j.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    perm: SmallVec<[MatrixIndex; INLINE_PERM_CAPACITY]>,
}

impl Permutation {
    /// Create a new permutation from a vector, without validation.
    /// Use `new` for validated construction.
    pub fn from_vec_unchecked(perm: Vec<MatrixIndex>) -> Self {
        Self {
            perm: SmallVec::from_vec(perm),
        }
    }

    /// Create a new permutation from a SmallVec, without validation.
    pub fn from_smallvec_unchecked(perm: SmallVec<[MatrixIndex; INLINE_PERM_CAPACITY]>) -> Self {
        Self { perm }
    }

    /// Get the underlying permutation as a slice.
    pub fn as_slice(&self) -> &[MatrixIndex] {
        &self.perm
    }

    /// Get the underlying permutation as a vector.
    pub fn to_vec(&self) -> Vec<MatrixIndex> {
        self.perm.to_vec()
    }

    /// Get the length of the permutation.
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Check if the permutation is empty.
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Iterate over the permutation values.
    pub fn iter(&self) -> impl Iterator<Item = &MatrixIndex> {
        self.perm.iter()
    }

    /// Apply the permutation to an index.
    pub fn apply(&self, i: MatrixIndex) -> MatrixIndex {
        self.perm[i]
    }

    /// Compose two permutations: (self * other)(i) = self(other(i))
    pub fn compose(&self, other: &Permutation) -> Permutation {
        debug_assert_eq!(self.perm.len(), other.perm.len());
        let composed: SmallVec<[MatrixIndex; INLINE_PERM_CAPACITY]> = (0..self.perm.len())
            .map(|i| self.perm[other.perm[i]])
            .collect();
        Permutation { perm: composed }
    }

    /// Create a new identity permutation of size n.
    pub fn identity(n: usize) -> Permutation {
        Permutation {
            perm: (0..n).collect(),
        }
    }
}

/// Index trait allows using perm[i] syntax.
impl Index<usize> for Permutation {
    type Output = MatrixIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &self.perm[index]
    }
}

/// Allow collecting an iterator into a Permutation (unchecked).
impl FromIterator<MatrixIndex> for Permutation {
    fn from_iter<T: IntoIterator<Item = MatrixIndex>>(iter: T) -> Self {
        Permutation {
            perm: iter.into_iter().collect(),
        }
    }
}

#[pymethods]
impl Permutation {
    /// Create a new permutation from a list of indices.
    ///
    /// The list must be a valid permutation of 0..n-1.
    #[new]
    pub fn new(perm: Vec<MatrixIndex>) -> PyResult<Self> {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in &perm {
            if p >= n {
                return Err(PyValueError::new_err(
                    "Input list is not a valid permutation of 0..n-1",
                ));
            }
            if seen[p] {
                return Err(PyValueError::new_err(
                    "Input list is not a valid permutation of 0..n-1",
                ));
            }
            seen[p] = true;
        }
        Ok(Self {
            perm: SmallVec::from_vec(perm),
        })
    }

    /// Apply the permutation to an index.
    pub fn __call__(&self, i: MatrixIndex) -> MatrixIndex {
        self.perm[i]
    }

    /// Get the length of the permutation.
    pub fn __len__(&self) -> usize {
        self.perm.len()
    }

    /// Check equality with another permutation.
    pub fn __eq__(&self, other: &Permutation) -> bool {
        self.perm == other.perm
    }

    /// Compose two permutations: (self * other)(i) = self(other(i))
    pub fn __mul__(&self, other: &Permutation) -> PyResult<Permutation> {
        if self.perm.len() != other.perm.len() {
            return Err(PyValueError::new_err("Permutations must have same length"));
        }
        let n = self.perm.len();
        let composed: SmallVec<[MatrixIndex; INLINE_PERM_CAPACITY]> =
            (0..n).map(|i| self.perm[other.perm[i]]).collect();
        Ok(Permutation { perm: composed })
    }

    /// Create the identity permutation of size n.
    #[staticmethod]
    pub fn id(n: usize) -> Permutation {
        Permutation {
            perm: (0..n).collect(),
        }
    }

    /// Check if this is the identity permutation.
    pub fn is_id(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| i == p)
    }

    /// Get the cycle decomposition (only cycles of length > 1).
    pub fn cycle_decomposition(&self) -> Vec<Vec<MatrixIndex>> {
        let (cycles, _) = self.get_cycle_decomposition_and_count();
        cycles.into_iter().filter(|c| c.len() > 1).collect()
    }

    /// Compute the sign (parity) of the permutation.
    /// Returns 1 for even permutations, -1 for odd permutations.
    pub fn sign(&self) -> i32 {
        let n = self.perm.len();
        if n == 0 {
            return 1;
        }
        let (_, num_cycles) = self.get_cycle_decomposition_and_count();
        if (n - num_cycles) % 2 == 0 { 1 } else { -1 }
    }

    /// Format the permutation in cycle notation for LaTeX.
    ///
    /// If the permutation is the identity, returns r"\text{id}".
    /// Otherwise returns cycles like "(1 2 3)(4 5)".
    ///
    /// The `arg_of` parameter is accepted for compatibility with Python's
    /// `cformat(val, arg_of)` protocol but is ignored since cycle notation
    /// doesn't require parenthesization based on context.
    #[pyo3(signature = (arg_of=None))]
    pub fn cformat(&self, arg_of: Option<&str>) -> String {
        let _ = arg_of; // Unused but required for Python compatibility
        let cycles = self.cycle_decomposition();
        if cycles.is_empty() {
            return r"\text{id}".to_string();
        }
        cycles
            .iter()
            .map(|cycle| {
                let elements: Vec<String> = cycle.iter().map(|&x| (x + 1).to_string()).collect();
                format!("({})", elements.join(" "))
            })
            .collect()
    }

    /// Compute the cost of the permutation (sum of cycle lengths - 1).
    /// This represents the minimum number of transpositions needed.
    pub fn cost(&self) -> usize {
        self.cycle_decomposition().iter().map(|c| c.len() - 1).sum()
    }

    /// If the permutation is exactly one transposition, return the pair.
    /// Otherwise return None.
    pub fn try_get_one_transpose(&self) -> Option<(MatrixIndex, MatrixIndex)> {
        let cd = self.cycle_decomposition();
        let c1: Vec<_> = cd.iter().filter(|c| c.len() == 2).collect();
        let c2: Vec<_> = cd.iter().filter(|c| c.len() > 2).collect();
        if c1.len() == 1 && c2.is_empty() {
            Some((c1[0][0], c1[0][1]))
        } else {
            None
        }
    }

    /// Get the permutation as a list.
    #[getter]
    pub fn perm(&self) -> Vec<MatrixIndex> {
        self.perm.to_vec()
    }

    fn __repr__(&self) -> String {
        format!("Permutation({:?})", self.perm.as_slice())
    }

    fn __str__(&self) -> String {
        self.cformat(None)
    }

    pub fn inverse(&self) -> Permutation {
        let n = self.perm.len();
        let mut inverse = vec![0; n];
        for (i, &p) in self.perm.iter().enumerate() {
            inverse[p] = i;
        }
        Permutation::from_vec_unchecked(inverse)
    }
}

impl Permutation {
    /// Get cycle decomposition and count of all cycles (including fixed points).
    fn get_cycle_decomposition_and_count(&self) -> (Vec<Vec<MatrixIndex>>, usize) {
        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();
        let mut num_cycles = 0;

        for i in 0..n {
            if !visited[i] {
                num_cycles += 1;
                let mut cycle = Vec::new();
                let mut j = i;
                while !visited[j] {
                    visited[j] = true;
                    cycle.push(j);
                    j = self.perm[j];
                }
                cycles.push(cycle);
            }
        }
        (cycles, num_cycles)
    }
}

/// Permutation of rows and columns of a matrix.
///
/// Represents permutation matrices P and Q, applied to A in the fashion of PAQ.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowColPermutation {
    row_perm: Permutation,
    col_perm: Permutation,
}

impl RowColPermutation {
    /// Create a new RowColPermutation from Permutation objects.
    pub fn from_perms(row_perm: Permutation, col_perm: Permutation) -> Self {
        Self { row_perm, col_perm }
    }

    /// Get the row permutation.
    pub fn row(&self) -> &Permutation {
        &self.row_perm
    }

    /// Get the column permutation.
    pub fn col(&self) -> &Permutation {
        &self.col_perm
    }

    /// Get the size (length of the row permutation).
    pub fn len(&self) -> usize {
        self.row_perm.len()
    }

    /// Check if the permutation is empty.
    pub fn is_empty(&self) -> bool {
        self.row_perm.is_empty()
    }

    /// Create a new identity RowColPermutation of size n.
    pub fn identity(n: usize) -> RowColPermutation {
        RowColPermutation {
            row_perm: Permutation::identity(n),
            col_perm: Permutation::identity(n),
        }
    }

    /// Apply the permutation to (i, j) -> (row_perm(i), col_perm(j)).
    pub fn apply(&self, i: MatrixIndex, j: MatrixIndex) -> (MatrixIndex, MatrixIndex) {
        (self.row_perm[i], self.col_perm[j])
    }

    /// Compose two RowColPermutations.
    /// For PAQ * P'AQ' = (P*P')A(Q'*Q)
    pub fn compose(&self, other: &RowColPermutation) -> RowColPermutation {
        RowColPermutation {
            row_perm: self.row_perm.compose(&other.row_perm),
            col_perm: other.col_perm.compose(&self.col_perm),
        }
    }
}

#[pymethods]
impl RowColPermutation {
    /// Create a new RowColPermutation from row and column permutation lists.
    #[new]
    pub fn new(row_perm: Vec<MatrixIndex>, col_perm: Vec<MatrixIndex>) -> PyResult<Self> {
        Ok(Self {
            row_perm: Permutation::new(row_perm)?,
            col_perm: Permutation::new(col_perm)?,
        })
    }

    /// Get the size (based on row permutation length).
    pub fn __len__(&self) -> usize {
        self.row_perm.perm.len()
    }

    /// Apply the permutation to (i, j) -> (row_perm(i), col_perm(j)).
    pub fn __call__(&self, i: MatrixIndex, j: MatrixIndex) -> (MatrixIndex, MatrixIndex) {
        (self.row_perm.perm[i], self.col_perm.perm[j])
    }

    /// Compose two RowColPermutations.
    /// For PAQ * P'AQ' = (P*P')A(Q'*Q)
    pub fn __mul__(&self, other: &RowColPermutation) -> PyResult<RowColPermutation> {
        Ok(RowColPermutation {
            row_perm: self.row_perm.__mul__(&other.row_perm)?,
            col_perm: other.col_perm.__mul__(&self.col_perm)?,
        })
    }

    /// Check equality with another RowColPermutation.
    pub fn __eq__(&self, other: &RowColPermutation) -> bool {
        self.row_perm == other.row_perm && self.col_perm == other.col_perm
    }

    /// Create the identity RowColPermutation of size n.
    #[staticmethod]
    pub fn id(n: usize) -> RowColPermutation {
        RowColPermutation {
            row_perm: Permutation::id(n),
            col_perm: Permutation::id(n),
        }
    }

    /// Check if this is the identity.
    pub fn is_id(&self) -> bool {
        self.row_perm.is_id() && self.col_perm.is_id()
    }

    /// Construct a permutation that transposes a matrix of size n x n.
    /// This reverses both row and column indices.
    #[staticmethod]
    pub fn matrix_transpose(n: usize) -> RowColPermutation {
        let reversed: Vec<MatrixIndex> = (0..n).rev().collect();
        RowColPermutation {
            row_perm: Permutation::from_vec_unchecked(reversed.clone()),
            col_perm: Permutation::from_vec_unchecked(reversed),
        }
    }

    /// Compose with matrix transpose.
    pub fn with_transpose(&self) -> PyResult<RowColPermutation> {
        let n = self.row_perm.perm.len();
        self.__mul__(&RowColPermutation::matrix_transpose(n))
    }

    /// Compute the total cost (row cost + column cost).
    pub fn cost(&self) -> usize {
        self.row_perm.cost() + self.col_perm.cost()
    }

    /// Try applying transpose if it reduces cost.
    /// Returns (result, did_transpose).
    pub fn try_transpose(&self) -> PyResult<(RowColPermutation, bool)> {
        let new = self.with_transpose()?;
        let old_cost = self.cost();
        let new_cost = new.cost() + 1; // +1 for the transpose operation
        if new_cost < old_cost {
            Ok((new, true))
        } else {
            Ok((self.clone(), false))
        }
    }

    /// Get the row and column permutations as a tuple.
    pub fn to_rows_cols_permutations(&self) -> (Permutation, Permutation) {
        (self.row_perm.clone(), self.col_perm.clone())
    }

    /// Get the row permutation.
    #[getter]
    pub fn row_perm(&self) -> Permutation {
        self.row_perm.clone()
    }

    /// Get the column permutation.
    #[getter]
    pub fn col_perm(&self) -> Permutation {
        self.col_perm.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RowColPermutation(row={:?}, col={:?})",
            self.row_perm.perm.as_slice(),
            self.col_perm.perm.as_slice()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "RowColPermutation(row={}, col={})",
            self.row_perm.cformat(None),
            self.col_perm.cformat(None)
        )
    }

    pub fn inverse(&self) -> RowColPermutation {
        RowColPermutation {
            row_perm: self.row_perm.inverse(),
            col_perm: self.col_perm.inverse(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_identity() {
        let id = Permutation::id(5);
        assert!(id.is_id());
        assert_eq!(id.__len__(), 5);
        for i in 0..5 {
            assert_eq!(id.__call__(i), i);
        }
    }

    #[test]
    fn test_permutation_cycle() {
        // (0 1 2) cycle: 0->1, 1->2, 2->0
        let perm = Permutation::new(vec![1, 2, 0]).unwrap();
        assert!(!perm.is_id());
        assert_eq!(perm.__call__(0), 1);
        assert_eq!(perm.__call__(1), 2);
        assert_eq!(perm.__call__(2), 0);
    }

    #[test]
    fn test_permutation_sign() {
        // Identity has sign 1
        assert_eq!(Permutation::id(3).sign(), 1);

        // Single transposition (0 1) has sign -1
        let trans = Permutation::new(vec![1, 0]).unwrap();
        assert_eq!(trans.sign(), -1);

        // 3-cycle (0 1 2) has sign 1 (even)
        let cycle3 = Permutation::new(vec![1, 2, 0]).unwrap();
        assert_eq!(cycle3.sign(), 1);
    }

    #[test]
    fn test_permutation_composition() {
        let p1 = Permutation::new(vec![1, 0, 2]).unwrap(); // swap 0,1
        let p2 = Permutation::new(vec![0, 2, 1]).unwrap(); // swap 1,2

        // (p1 * p2)(i) = p1(p2(i))
        let composed = p1.__mul__(&p2).unwrap();
        assert_eq!(composed.__call__(0), 1); // p2(0)=0, p1(0)=1
        assert_eq!(composed.__call__(1), 2); // p2(1)=2, p1(2)=2
        assert_eq!(composed.__call__(2), 0); // p2(2)=1, p1(1)=0
    }

    #[test]
    fn test_permutation_cformat() {
        let id = Permutation::id(3);
        assert_eq!(id.cformat(None), r"\text{id}");

        let trans = Permutation::new(vec![1, 0, 2]).unwrap();
        assert_eq!(trans.cformat(None), "(1 2)");

        let cycle3 = Permutation::new(vec![1, 2, 0]).unwrap();
        assert_eq!(cycle3.cformat(None), "(1 2 3)");
    }

    #[test]
    fn test_rowcol_identity() {
        let id = RowColPermutation::id(3);
        assert!(id.is_id());
        assert_eq!(id.__len__(), 3);
        assert_eq!(id.__call__(1, 2), (1, 2));
    }

    #[test]
    fn test_rowcol_transpose() {
        let mt = RowColPermutation::matrix_transpose(3);
        // For a 3x3 matrix, transpose swaps (i,j) with (2-i, 2-j)
        // Actually this reverses indices: (0,0)->(2,2), (0,1)->(2,1), etc.
        assert_eq!(mt.__call__(0, 0), (2, 2));
        assert_eq!(mt.__call__(0, 1), (2, 1));
        assert_eq!(mt.__call__(1, 1), (1, 1));
    }
}
