//! Sparse non-zero storage for adjacency matrices using compact bitlists.
use crate::bitlist::BitList;

#[derive(Clone, Debug)]
pub struct Nonzeros {
    rows: usize,
    cols: usize,
    bits: BitList,
}

impl Nonzeros {
    pub fn empty(rows: usize, cols: usize) -> Self {
        let len = rows.saturating_mul(cols);
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
        self.idx(row, col)
            .map(|i| self.bits.get(i))
            .unwrap_or(false)
    }

    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        if let Some(i) = self.idx(row, col) {
            self.bits.set(i, value);
        }
    }

    pub fn permute(&self, row_perm: &[usize], col_perm: &[usize]) -> Self {
        // row_perm/col_perm map old index -> new index
        let mut out = Self::empty(row_perm.len(), col_perm.len());
        for r_old in 0..self.rows {
            let r_new = *row_perm.get(r_old).unwrap_or(&r_old);
            for c_old in 0..self.cols {
                let c_new = *col_perm.get(c_old).unwrap_or(&c_old);
                if self.contains(r_old, c_old) {
                    out.set(r_new, c_new, true);
                }
            }
        }
        out
    }

    pub fn permute_inv(&self, row_perm: &[usize], col_perm: &[usize]) -> Self {
        let mut inv_row = vec![0; row_perm.len()];
        let mut inv_col = vec![0; col_perm.len()];
        for (i, &r) in row_perm.iter().enumerate() {
            inv_row[r] = i;
        }
        for (i, &c) in col_perm.iter().enumerate() {
            inv_col[c] = i;
        }
        self.permute(&inv_row, &inv_col)
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
