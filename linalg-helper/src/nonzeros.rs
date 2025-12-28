//! Sparse non-zero storage for adjacency matrices using compact bitlists.
use crate::bitlist::BitList;
use crate::permutation::{Permutation, RowColPermutation};

#[derive(Clone, Debug)]
pub struct Nonzeros {
    rows: usize,
    cols: usize,
    bits: BitList,
}

impl Nonzeros {
    pub fn empty(rows: usize, cols: usize) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("Nonzeros::empty: rows * cols overflowed");
        Self {
            rows,
            cols,
            bits: BitList::zeros(len),
        }
    }

    /// Build from a shape and predicate returning whether (r, c) is non-zero.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> bool,
    {
        let mut nz = Self::empty(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                if f(r, c) {
                    nz.set(r, c, true);
                }
            }
        }
        nz
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        if row < self.rows && col < self.cols {
            Some(row * self.cols + col)
        } else {
            None
        }
    }

    pub fn contains(&self, row: usize, col: usize) -> bool {
        let i = self
            .idx(row, col)
            .expect("Nonzeros::contains: index out of bounds");
        self.bits.get(i)
    }

    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        let i = self
            .idx(row, col)
            .expect("Nonzeros::set: index out of bounds");
        self.bits.set(i, value);
    }

    /// Permute the matrix using separate row and column permutations.
    /// row_perm/col_perm map old index -> new index.
    pub fn permute(&self, row_perm: &Permutation, col_perm: &Permutation) -> Self {
        assert_eq!(
            row_perm.len(),
            self.rows,
            "Nonzeros::permute: row_perm length {} does not match rows {}",
            row_perm.len(),
            self.rows
        );
        assert_eq!(
            col_perm.len(),
            self.cols,
            "Nonzeros::permute: col_perm length {} does not match cols {}",
            col_perm.len(),
            self.cols
        );
        let mut out = Self::empty(row_perm.len(), col_perm.len());
        for (r_old, &r_new) in row_perm.iter().enumerate() {
            for (c_old, &c_new) in col_perm.iter().enumerate() {
                if self.contains(r_old, c_old) {
                    out.set(r_new, c_new, true);
                }
            }
        }
        out
    }

    /// Permute the matrix using a RowColPermutation.
    pub fn permute_rowcol(&self, perm: &RowColPermutation) -> Self {
        self.permute(perm.row(), perm.col())
    }

    /// Permute the matrix using the inverse of the given permutations.
    pub fn permute_inv(&self, row_perm: &Permutation, col_perm: &Permutation) -> Self {
        self.permute(&row_perm.inverse(), &col_perm.inverse())
    }

    /// Permute the matrix using the inverse of the given RowColPermutation.
    pub fn permute_rowcol_inv(&self, perm: &RowColPermutation) -> Self {
        self.permute_rowcol(&perm.inverse())
    }

    pub fn to_vec(&self) -> Vec<(usize, usize)> {
        let mut entries = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.contains(r, c) {
                    entries.push((r, c));
                }
            }
        }
        entries
    }

    pub fn count(&self) -> usize {
        self.bits.count_ones()
    }
}
