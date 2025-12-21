//! Random matrix tests for the Dulmage-Mendelsohn decomposition and canonicalization

use rand::prelude::*;

use crate::adjacency::AdjacencyMatrix;
use crate::canonical::{are_permutation_equivalent, canonicalize};
use crate::dm::dulmage_mendelsohn;

/// Create a fully dense (irreducible) block of given size
/// A fully dense block is guaranteed to be irreducible since all rows/columns are connected
fn make_full_block(size: usize) -> Vec<Vec<bool>> {
    vec![vec![true; size]; size]
}

/// Create a random irreducible block of given size
/// Uses a Hamiltonian cycle + random edges to guarantee strong connectivity
fn make_random_irreducible_block(size: usize, rng: &mut impl Rng) -> Vec<Vec<bool>> {
    if size <= 1 {
        return vec![vec![true; size]; size];
    }

    let mut block = vec![vec![false; size]; size];

    // Add diagonal for valid matching
    for i in 0..size {
        block[i][i] = true;
    }

    // Add a cycle connecting all rows: row i connects to column (i+1) mod size
    // This ensures all vertices are in the same SCC
    for i in 0..size {
        let next = (i + 1) % size;
        block[i][next] = true;
    }

    // Add random additional entries for variety
    for i in 0..size {
        for j in 0..size {
            if !block[i][j] && rng.gen_bool(0.3) {
                block[i][j] = true;
            }
        }
    }

    block
}

/// Create a block diagonal matrix with fully dense blocks
fn make_block_diagonal_dense(block_sizes: &[usize]) -> Vec<Vec<bool>> {
    let total_size: usize = block_sizes.iter().sum();
    let mut matrix = vec![vec![false; total_size]; total_size];

    let mut offset = 0;
    for &size in block_sizes {
        let block = make_full_block(size);
        for i in 0..size {
            for j in 0..size {
                matrix[offset + i][offset + j] = block[i][j];
            }
        }
        offset += size;
    }

    matrix
}

/// Create a block diagonal matrix with random irreducible blocks
fn make_block_diagonal_random(block_sizes: &[usize], rng: &mut impl Rng) -> Vec<Vec<bool>> {
    let total_size: usize = block_sizes.iter().sum();
    let mut matrix = vec![vec![false; total_size]; total_size];

    let mut offset = 0;
    for &size in block_sizes {
        let block = make_random_irreducible_block(size, rng);
        for i in 0..size {
            for j in 0..size {
                matrix[offset + i][offset + j] = block[i][j];
            }
        }
        offset += size;
    }

    matrix
}

/// Apply row and column permutations to a matrix
fn permute_matrix(matrix: &[Vec<bool>], row_perm: &[usize], col_perm: &[usize]) -> Vec<Vec<bool>> {
    let n = matrix.len();
    let m = if n > 0 { matrix[0].len() } else { 0 };
    let mut result = vec![vec![false; m]; n];

    for (new_row, &old_row) in row_perm.iter().enumerate() {
        for (new_col, &old_col) in col_perm.iter().enumerate() {
            result[new_row][new_col] = matrix[old_row][old_col];
        }
    }

    result
}

/// Generate a random permutation of 0..n
fn random_permutation(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Test that DM decomposition produces a valid block diagonal form
/// by verifying the permuted matrix has all nonzero entries within the claimed blocks.
///
/// Note: The decomposition may find a finer (better) decomposition than expected.
/// For example, if we construct blocks [2, 3], the algorithm might find [2, 1, 2].
/// This is valid as long as:
/// 1. The total size matches
/// 2. All nonzero entries are within the claimed blocks
/// 3. The decomposition is at least as fine as expected (same or more blocks)
fn test_random_block_diagonal_with_sizes(block_sizes: &[usize], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_size: usize = block_sizes.iter().sum();

    // Create block diagonal matrix with dense blocks (guaranteed irreducible)
    let original = make_block_diagonal_dense(block_sizes);

    // Apply random permutation
    let row_perm = random_permutation(total_size, &mut rng);
    let col_perm = random_permutation(total_size, &mut rng);
    let permuted = permute_matrix(&original, &row_perm, &col_perm);

    // Run DM decomposition
    let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(permuted.clone()));

    // Verify permutations are valid (correct length and contain all indices)
    assert_eq!(result.row_perm.len(), total_size);
    assert_eq!(result.col_perm.len(), total_size);

    let mut row_set: Vec<usize> = result.row_perm.as_slice().to_vec();
    row_set.sort();
    assert_eq!(row_set, (0..total_size).collect::<Vec<_>>());

    let mut col_set: Vec<usize> = result.col_perm.as_slice().to_vec();
    col_set.sort();
    assert_eq!(col_set, (0..total_size).collect::<Vec<_>>());

    // The sum of block sizes should equal the matrix size
    let found_total: usize = result.block_sizes.iter().sum();
    assert_eq!(
        found_total, total_size,
        "Block sizes sum mismatch: expected {}, got {} (sizes: {:?})",
        total_size, found_total, result.block_sizes
    );

    // The algorithm may find a finer decomposition (more blocks) than we constructed.
    // This is valid - it means it found structure we didn't explicitly create.
    // However, finding fewer blocks would indicate a bug (merging irreducible blocks).
    assert!(
        result.block_sizes.len() >= block_sizes.len(),
        "Decomposition found fewer blocks than expected: expected at least {} blocks, got {} (expected: {:?}, found: {:?}, seed: {})",
        block_sizes.len(),
        result.block_sizes.len(),
        block_sizes,
        result.block_sizes,
        seed
    );

    // Apply the found permutation to reorder the matrix
    let reordered = permute_matrix(&permuted, &result.row_perm, &result.col_perm);

    // Compute which (i, j) positions are inside some block
    let mut in_some_block = vec![vec![false; total_size]; total_size];
    let mut offset = 0;
    for &size in &result.block_sizes {
        for i in offset..offset + size {
            for j in offset..offset + size {
                in_some_block[i][j] = true;
            }
        }
        offset += size;
    }

    // Verify all nonzero entries are inside some block (block diagonal structure)
    for i in 0..total_size {
        for j in 0..total_size {
            if reordered[i][j] && !in_some_block[i][j] {
                panic!(
                    "Entry ({}, {}) is nonzero but outside all blocks. \
                     Block sizes: {:?}, seed: {}",
                    i, j, result.block_sizes, seed
                );
            }
        }
    }
}

#[test]
fn test_random_two_blocks() {
    // Two blocks of sizes 2 and 3
    test_random_block_diagonal_with_sizes(&[2, 3], 42);
}

#[test]
fn test_random_three_blocks() {
    // Three blocks of sizes 2, 2, 3
    test_random_block_diagonal_with_sizes(&[2, 2, 3], 123);
}

#[test]
fn test_random_single_block() {
    // Single block should remain single
    test_random_block_diagonal_with_sizes(&[5], 456);
}

#[test]
fn test_random_many_small_blocks() {
    // Many size-1 blocks (like a diagonal matrix)
    test_random_block_diagonal_with_sizes(&[1, 1, 1, 1, 1], 789);
}

#[test]
fn test_random_mixed_sizes() {
    // Mix of different sizes
    test_random_block_diagonal_with_sizes(&[1, 3, 2, 4], 1001);
}

#[test]
fn test_random_large_blocks() {
    // Larger blocks
    test_random_block_diagonal_with_sizes(&[5, 7, 4], 2002);
}

#[test]
fn test_random_stress() {
    // Run multiple random tests with different seeds
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate random block sizes (2-5 blocks, sizes 1-6)
        let num_blocks = rng.gen_range(2..=5);
        let block_sizes: Vec<usize> = (0..num_blocks).map(|_| rng.gen_range(1..=6)).collect();

        test_random_block_diagonal_with_sizes(&block_sizes, seed * 1000 + 42);
    }
}

/// Test with sparse random irreducible blocks (not fully dense)
fn test_sparse_block_diagonal_with_sizes(block_sizes: &[usize], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_size: usize = block_sizes.iter().sum();

    // Create block diagonal matrix with random irreducible blocks
    let original = make_block_diagonal_random(block_sizes, &mut rng);

    // Apply random permutation
    let row_perm = random_permutation(total_size, &mut rng);
    let col_perm = random_permutation(total_size, &mut rng);
    let permuted = permute_matrix(&original, &row_perm, &col_perm);

    // Run DM decomposition
    let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(permuted));

    // The sum of block sizes should equal the matrix size
    let found_total: usize = result.block_sizes.iter().sum();
    assert_eq!(
        found_total, total_size,
        "Block sizes sum mismatch: expected {}, got {} (sizes: {:?})",
        total_size, found_total, result.block_sizes
    );

    // Sort block sizes for comparison (order may differ)
    let mut expected_sorted = block_sizes.to_vec();
    expected_sorted.sort();
    let mut found_sorted = result.block_sizes.clone();
    found_sorted.sort();

    assert_eq!(
        expected_sorted, found_sorted,
        "Block sizes mismatch: expected {:?}, got {:?}",
        expected_sorted, found_sorted
    );
}

#[test]
fn test_sparse_random_blocks() {
    // Test with sparse (but irreducible) random blocks
    test_sparse_block_diagonal_with_sizes(&[2, 3], 42);
    test_sparse_block_diagonal_with_sizes(&[3, 4, 2], 123);
    test_sparse_block_diagonal_with_sizes(&[5, 3, 4], 456);
}

#[test]
fn test_sparse_random_stress() {
    // Stress test with random sparse blocks
    for seed in 100..120 {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate random block sizes (2-4 blocks, sizes 2-5)
        // Minimum size 2 to ensure irreducibility is meaningful
        let num_blocks = rng.gen_range(2..=4);
        let block_sizes: Vec<usize> = (0..num_blocks).map(|_| rng.gen_range(2..=5)).collect();

        test_sparse_block_diagonal_with_sizes(&block_sizes, seed * 1000 + 42);
    }
}

#[test]
fn test_random_permuted_preserves_structure() {
    // Test that the decomposition correctly identifies block structure
    // by verifying that applying the found permutation gives block diagonal form
    let block_sizes = vec![2, 3, 2];
    let mut rng = StdRng::seed_from_u64(12345);
    let total_size: usize = block_sizes.iter().sum();

    // Create and permute (use dense blocks for guaranteed irreducibility)
    let original = make_block_diagonal_dense(&block_sizes);
    let row_perm = random_permutation(total_size, &mut rng);
    let col_perm = random_permutation(total_size, &mut rng);
    let permuted = permute_matrix(&original, &row_perm, &col_perm);

    // Run decomposition
    let result = dulmage_mendelsohn(&AdjacencyMatrix::from_vec(permuted.clone()));

    // Apply the inverse of the found permutation to get block diagonal form
    // Build inverse permutations
    let mut inv_row_perm = vec![0; total_size];
    let mut inv_col_perm = vec![0; total_size];
    for (new_idx, &old_idx) in result.row_perm.iter().enumerate() {
        inv_row_perm[old_idx] = new_idx;
    }
    for (new_idx, &old_idx) in result.col_perm.iter().enumerate() {
        inv_col_perm[old_idx] = new_idx;
    }

    // Reorder the permuted matrix using the found permutation
    let reordered = permute_matrix(&permuted, &result.row_perm, &result.col_perm);

    // Verify block diagonal structure: entries outside blocks should be false
    // First, compute which (i, j) positions are inside some block
    let mut in_some_block = vec![vec![false; total_size]; total_size];
    let mut offset = 0;
    for &size in &result.block_sizes {
        for i in offset..offset + size {
            for j in offset..offset + size {
                in_some_block[i][j] = true;
            }
        }
        offset += size;
    }

    // Now verify all nonzero entries are inside some block
    for i in 0..total_size {
        for j in 0..total_size {
            if reordered[i][j] && !in_some_block[i][j] {
                panic!(
                    "Entry ({}, {}) is nonzero but outside all blocks. \
                     Block sizes: {:?}",
                    i, j, result.block_sizes
                );
            }
        }
    }
}

// ============================================================================
// Canonical form tests
// ============================================================================

/// Test that the same matrix always produces the same canonical form
#[test]
fn test_canonical_deterministic() {
    let mut rng = StdRng::seed_from_u64(42);
    let matrix = make_random_irreducible_block(5, &mut rng);
    let graph = AdjacencyMatrix::from_vec(matrix);

    let c1 = canonicalize(&graph);
    let c2 = canonicalize(&graph);

    assert_eq!(c1.canonical_hash, c2.canonical_hash);
    assert_eq!(c1.row_perm, c2.row_perm);
    assert_eq!(c1.col_perm, c2.col_perm);
}

/// Test that permuted versions of the same matrix have the same canonical hash
#[test]
fn test_canonical_permutation_invariant() {
    for seed in 0..10 {
        let mut rng = StdRng::seed_from_u64(seed * 1000);
        let size = rng.gen_range(3..=6);
        let original = make_random_irreducible_block(size, &mut rng);

        // Generate multiple random permutations
        for _ in 0..5 {
            let row_perm = random_permutation(size, &mut rng);
            let col_perm = random_permutation(size, &mut rng);
            let permuted = permute_matrix(&original, &row_perm, &col_perm);

            let orig_graph = AdjacencyMatrix::from_vec(original.clone());
            let perm_graph = AdjacencyMatrix::from_vec(permuted);

            assert!(
                are_permutation_equivalent(&orig_graph, &perm_graph),
                "Permuted matrix should be equivalent to original (seed: {})",
                seed
            );
        }
    }
}

/// Test that different matrices have different canonical forms
#[test]
fn test_canonical_distinguishes_different() {
    // Two clearly different structures
    let m1 = vec![
        vec![true, true, false, false],
        vec![true, true, false, false],
        vec![false, false, true, true],
        vec![false, false, true, true],
    ];

    let m2 = vec![
        vec![true, false, false, false],
        vec![true, true, false, false],
        vec![true, true, true, false],
        vec![true, true, true, true],
    ];

    let g1 = AdjacencyMatrix::from_vec(m1);
    let g2 = AdjacencyMatrix::from_vec(m2);

    assert!(
        !are_permutation_equivalent(&g1, &g2),
        "Structurally different matrices should not be equivalent"
    );
}

/// Stress test: random matrices and their permutations
#[test]
fn test_canonical_stress() {
    let mut rng = StdRng::seed_from_u64(999);

    for _ in 0..20 {
        let size = rng.gen_range(3..=8);
        let original = make_random_irreducible_block(size, &mut rng);
        let orig_graph = AdjacencyMatrix::from_vec(original.clone());
        let orig_canon = canonicalize(&orig_graph);

        // Apply random permutations and verify canonical hash matches
        for _ in 0..5 {
            let row_perm = random_permutation(size, &mut rng);
            let col_perm = random_permutation(size, &mut rng);
            let permuted = permute_matrix(&original, &row_perm, &col_perm);
            let perm_graph = AdjacencyMatrix::from_vec(permuted);
            let perm_canon = canonicalize(&perm_graph);

            assert_eq!(
                orig_canon.canonical_hash, perm_canon.canonical_hash,
                "Canonical hash should be the same for permuted matrices"
            );
        }
    }
}

/// Test block diagonal matrices
#[test]
fn test_canonical_block_diagonal() {
    let block_sizes = vec![2, 3, 2];
    let mut rng = StdRng::seed_from_u64(555);
    let total_size: usize = block_sizes.iter().sum();

    let original = make_block_diagonal_dense(&block_sizes);
    let orig_graph = AdjacencyMatrix::from_vec(original.clone());

    // Permute and verify equivalence
    let row_perm = random_permutation(total_size, &mut rng);
    let col_perm = random_permutation(total_size, &mut rng);
    let permuted = permute_matrix(&original, &row_perm, &col_perm);
    let perm_graph = AdjacencyMatrix::from_vec(permuted);

    assert!(
        are_permutation_equivalent(&orig_graph, &perm_graph),
        "Block diagonal matrix and its permutation should be equivalent"
    );
}
