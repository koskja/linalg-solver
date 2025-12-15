/// Adjacency matrix representation (row-major, true = nonzero)
#[derive(Clone)]
pub struct AdjacencyMatrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<bool>,
}

impl AdjacencyMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![false; rows * cols],
        }
    }

    pub fn from_vec(matrix: Vec<Vec<bool>>) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        let mut data = vec![false; rows * cols];
        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[i * cols + j] = val;
            }
        }
        Self { rows, cols, data }
    }

    pub fn get(&self, row: usize, col: usize) -> bool {
        self.data[row * self.cols + col]
    }

    /// Get all columns connected to a given row
    pub fn row_neighbors(&self, row: usize) -> Vec<usize> {
        (0..self.cols).filter(|&c| self.get(row, c)).collect()
    }

    /// Get all rows connected to a given column
    pub fn col_neighbors(&self, col: usize) -> Vec<usize> {
        (0..self.rows).filter(|&r| self.get(r, col)).collect()
    }

    /// Set a value at the given position
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        self.data[row * self.cols + col] = value;
    }

    /// Extract a submatrix with the given rows and columns
    /// The resulting matrix has dimensions (row_indices.len(), col_indices.len())
    pub fn submatrix(&self, row_indices: &[usize], col_indices: &[usize]) -> Self {
        let new_rows = row_indices.len();
        let new_cols = col_indices.len();
        let mut data = vec![false; new_rows * new_cols];

        for (new_r, &old_r) in row_indices.iter().enumerate() {
            for (new_c, &old_c) in col_indices.iter().enumerate() {
                data[new_r * new_cols + new_c] = self.get(old_r, old_c);
            }
        }

        Self {
            rows: new_rows,
            cols: new_cols,
            data,
        }
    }

    /// Create a new matrix with rows swapped
    pub fn with_swapped_rows(&self, r1: usize, r2: usize) -> Self {
        let mut result = self.clone();
        for c in 0..self.cols {
            let tmp = result.get(r1, c);
            result.set(r1, c, result.get(r2, c));
            result.set(r2, c, tmp);
        }
        result
    }

    /// Create a new matrix after adding row `src` (scaled) to row `dst` to eliminate (dst, pivot_col)
    /// In terms of sparsity: dst[c] becomes non-zero if src[c] OR dst[c] is non-zero,
    /// EXCEPT at pivot_col where it becomes zero (that's the point of the elimination)
    pub fn with_add_row(&self, src: usize, dst: usize, pivot_col: usize) -> Self {
        let mut result = self.clone();
        for c in 0..self.cols {
            if c == pivot_col {
                // The elimination zeros out this entry
                result.set(dst, c, false);
            } else {
                // Entry is non-zero if either src or dst had a non-zero
                // (in symbolic terms, we're adding two potentially non-zero values)
                let new_val = self.get(src, c) || self.get(dst, c);
                result.set(dst, c, new_val);
            }
        }
        result
    }

    /// Count non-zero entries in a row
    pub fn row_nnz(&self, row: usize) -> usize {
        (0..self.cols).filter(|&c| self.get(row, c)).count()
    }

    /// Count non-zero entries in a column
    pub fn col_nnz(&self, col: usize) -> usize {
        (0..self.rows).filter(|&r| self.get(r, col)).count()
    }

    /// Count total non-zero entries
    pub fn total_nnz(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }
}

/// Represents a matching in a bipartite graph
#[derive(Clone)]
pub struct Matching {
    /// row_to_col[r] = Some(c) means row r is matched to column c
    pub row_to_col: Vec<Option<usize>>,
    /// col_to_row[c] = Some(r) means column c is matched to row r
    pub col_to_row: Vec<Option<usize>>,
}

impl Matching {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            row_to_col: vec![None; rows],
            col_to_row: vec![None; cols],
        }
    }

    pub fn match_pair(&mut self, row: usize, col: usize) {
        self.row_to_col[row] = Some(col);
        self.col_to_row[col] = Some(row);
    }

    pub fn size(&self) -> usize {
        self.row_to_col.iter().filter(|x| x.is_some()).count()
    }
}
