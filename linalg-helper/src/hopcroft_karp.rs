//! Hopcroft-Karp Maximum Bipartite Matching Algorithm
//!
//! Finds a maximum matching in a bipartite graph in O(E * sqrt(V)) time.

use std::collections::VecDeque;

use crate::adjacency::{AdjacencyMatrix, Matching};

const INF: usize = usize::MAX;

/// Hopcroft-Karp algorithm for maximum bipartite matching
/// Returns a maximum matching in O(E * sqrt(V)) time
pub fn hopcroft_karp(graph: &AdjacencyMatrix) -> Matching {
    let rows = graph.rows;
    let cols = graph.cols;
    let mut matching = Matching::new(rows, cols);

    // dist[r] = distance of row r from unmatched rows in BFS
    // dist[rows] represents the "nil" vertex distance
    let mut dist = vec![0usize; rows + 1];

    while bfs_hk(graph, &matching, &mut dist) {
        for r in 0..rows {
            if matching.row_to_col[r].is_none() {
                dfs_hk(graph, &mut matching, &mut dist, r);
            }
        }
    }

    matching
}

/// BFS phase of Hopcroft-Karp: builds layers of alternating paths
fn bfs_hk(graph: &AdjacencyMatrix, matching: &Matching, dist: &mut [usize]) -> bool {
    let rows = graph.rows;
    let mut queue = VecDeque::new();

    // Initialize distances
    for r in 0..rows {
        if matching.row_to_col[r].is_none() {
            dist[r] = 0;
            queue.push_back(r);
        } else {
            dist[r] = INF;
        }
    }
    dist[rows] = INF; // nil vertex

    while let Some(r) = queue.pop_front() {
        if dist[r] < dist[rows] {
            for c in graph.row_neighbors(r) {
                // Get the row matched to this column (or "nil" = rows)
                let matched_row = matching.col_to_row[c].unwrap_or(rows);
                if dist[matched_row] == INF {
                    dist[matched_row] = dist[r] + 1;
                    if matched_row != rows {
                        queue.push_back(matched_row);
                    }
                }
            }
        }
    }

    dist[rows] != INF
}

/// DFS phase of Hopcroft-Karp: finds augmenting paths along BFS layers
fn dfs_hk(graph: &AdjacencyMatrix, matching: &mut Matching, dist: &mut [usize], r: usize) -> bool {
    let rows = graph.rows;
    if r == rows {
        return true; // reached nil vertex
    }

    for c in graph.row_neighbors(r) {
        let matched_row = matching.col_to_row[c].unwrap_or(rows);
        if dist[matched_row] == dist[r] + 1 && dfs_hk(graph, matching, dist, matched_row) {
            matching.match_pair(r, c);
            return true;
        }
    }

    dist[r] = INF; // remove from consideration
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopcroft_karp_simple() {
        // Simple 2x2 identity matrix pattern
        let graph = AdjacencyMatrix::from_vec(vec![vec![true, false], vec![false, true]]);
        let matching = hopcroft_karp(&graph);
        assert_eq!(matching.size(), 2);
    }

    #[test]
    fn test_hopcroft_karp_full() {
        // Full 2x2 matrix
        let graph = AdjacencyMatrix::from_vec(vec![vec![true, true], vec![true, true]]);
        let matching = hopcroft_karp(&graph);
        assert_eq!(matching.size(), 2);
    }
}
