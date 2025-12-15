//! Tarjan's Algorithm for Strongly Connected Components
//!
//! Finds all SCCs in a directed graph in O(V + E) time.
//! Returns SCCs in reverse topological order (sinks first).

struct TarjanState {
    index: usize,
    indices: Vec<Option<usize>>,
    lowlinks: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    sccs: Vec<Vec<usize>>,
}

/// Tarjan's algorithm for finding strongly connected components
/// Returns SCCs in reverse topological order (sinks first)
pub fn tarjan_scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut state = TarjanState {
        index: 0,
        indices: vec![None; n],
        lowlinks: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };

    for v in 0..n {
        if state.indices[v].is_none() {
            strongconnect(adj, &mut state, v);
        }
    }

    state.sccs
}

fn strongconnect(adj: &[Vec<usize>], state: &mut TarjanState, v: usize) {
    state.indices[v] = Some(state.index);
    state.lowlinks[v] = state.index;
    state.index += 1;
    state.stack.push(v);
    state.on_stack[v] = true;

    for &w in &adj[v] {
        if state.indices[w].is_none() {
            strongconnect(adj, state, w);
            state.lowlinks[v] = state.lowlinks[v].min(state.lowlinks[w]);
        } else if state.on_stack[w] {
            state.lowlinks[v] = state.lowlinks[v].min(state.indices[w].unwrap());
        }
    }

    // If v is a root node, pop the stack and generate an SCC
    if state.lowlinks[v] == state.indices[v].unwrap() {
        let mut scc = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack[w] = false;
            scc.push(w);
            if w == v {
                break;
            }
        }
        state.sccs.push(scc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tarjan_simple() {
        // Simple chain: 0 -> 1 -> 2
        let adj = vec![vec![1], vec![2], vec![]];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 3); // Each vertex is its own SCC
    }

    #[test]
    fn test_tarjan_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let adj = vec![vec![1], vec![2], vec![0]];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 1); // All in one SCC
        assert_eq!(sccs[0].len(), 3);
    }
}
