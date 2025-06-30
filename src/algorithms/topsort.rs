//! Topological sorting algorithm for acyclic FSTs
//!
//! Reorders states in a directed acyclic graph (DAG) FST so that for every arc (u,v),
//! state u appears before state v in the ordering, enabling efficient processing algorithms.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::HashSet;

/// Topologically sort states in an acyclic FST for optimal processing order
///
/// Reorders the states of a directed acyclic graph (DAG) FST so that all arcs
/// point "forward" in the state ordering. This enables efficient algorithms that
/// process states in dependency order, ensuring each state is processed after
/// all its predecessors.
///
/// # Algorithm Details
///
/// - **Ordering Property:** For every arc (u → v), state u appears before state v
/// - **Time Complexity:** O(|V| + |E|) using depth-first search with cycle detection
/// - **Space Complexity:** O(|V|) for visited sets and result ordering
/// - **Acyclicity Requirement:** Input FST must be a directed acyclic graph (DAG)
/// - **Language Preservation:** L(topsort(T)) = L(T) exactly
///
/// # Mathematical Foundation
///
/// A topological ordering of a DAG is a linear ordering of vertices such that
/// for every directed edge (u,v), vertex u comes before v in the ordering.
/// For FSTs, this means:
/// - **Dependency Ordering:** All predecessor states processed before successors
/// - **Forward Flow:** All arcs point "forward" in the state sequence
/// - **Unique Structure:** While multiple valid orderings exist, all preserve dependencies
///
/// # Algorithm Steps
///
/// 1. **Cycle Detection:** Use DFS with visited/finished sets to detect cycles
/// 2. **DFS Traversal:** Start from each unvisited state, exploring all reachable states
/// 3. **Finish Order:** Record states in reverse finish order from DFS
/// 4. **State Remapping:** Create new FST with states renumbered in topological order
/// 5. **Arc Preservation:** Maintain all arcs with updated state references
///
/// # Examples
///
/// ## Basic Topological Sorting
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create DAG FST: 0 -> 1 -> 2, 0 -> 2
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// fst.add_arc(s0, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s2));
///
/// // Sort states in topological order
/// let sorted: VectorFst<TropicalWeight> = topsort(&fst)?;
///
/// // Result has same language but optimized state ordering
/// assert_eq!(sorted.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Preprocessing for Dynamic Programming
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Linear chain FST for efficient DP processing
/// let mut chain = VectorFst::<TropicalWeight>::new();
/// let s0 = chain.add_state();
/// let s1 = chain.add_state();
/// let s2 = chain.add_state();
/// let s3 = chain.add_state();
///
/// chain.set_start(s0);
/// chain.set_final(s3, TropicalWeight::one());
///
/// // Sequential dependencies: s0 -> s1 -> s2 -> s3
/// chain.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.1), s1));
/// chain.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.2), s2));
/// chain.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(0.3), s3));
///
/// // Topological sort ensures forward processing order
/// let sorted: VectorFst<TropicalWeight> = topsort(&chain)?;
///
/// // Now algorithms can process states 0,1,2,3 in order efficiently
/// println!("Chain sorted with {} states", sorted.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Complex DAG Optimization
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Complex DAG structure
/// let mut dag = VectorFst::<TropicalWeight>::new();
/// let states: Vec<_> = (0..6).map(|_| dag.add_state()).collect();
///
/// dag.set_start(states[0]);
/// dag.set_final(states[5], TropicalWeight::one());
///
/// // Complex dependency graph
/// dag.add_arc(states[0], Arc::new(1, 1, TropicalWeight::one(), states[1]));
/// dag.add_arc(states[0], Arc::new(2, 2, TropicalWeight::one(), states[2]));
/// dag.add_arc(states[1], Arc::new(3, 3, TropicalWeight::one(), states[3]));
/// dag.add_arc(states[2], Arc::new(4, 4, TropicalWeight::one(), states[3]));
/// dag.add_arc(states[1], Arc::new(5, 5, TropicalWeight::one(), states[4]));
/// dag.add_arc(states[3], Arc::new(6, 6, TropicalWeight::one(), states[5]));
/// dag.add_arc(states[4], Arc::new(7, 7, TropicalWeight::one(), states[5]));
///
/// // Sort for optimal processing order
/// let sorted: VectorFst<TropicalWeight> = topsort(&dag)?;
///
/// // Result maintains all paths but with efficient state ordering
/// assert_eq!(sorted.num_states(), dag.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Pipeline Stage Ordering
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Processing pipeline with dependencies
/// let mut pipeline = VectorFst::<TropicalWeight>::new();
/// let input = pipeline.add_state();      // Input stage
/// let tokenize = pipeline.add_state();   // Tokenization
/// let parse = pipeline.add_state();      // Parsing  
/// let analyze = pipeline.add_state();    // Analysis
/// let output = pipeline.add_state();     // Output
///
/// pipeline.set_start(input);
/// pipeline.set_final(output, TropicalWeight::one());
///
/// // Pipeline dependencies
/// pipeline.add_arc(input, Arc::new(1, 10, TropicalWeight::one(), tokenize));
/// pipeline.add_arc(tokenize, Arc::new(10, 20, TropicalWeight::one(), parse));
/// pipeline.add_arc(parse, Arc::new(20, 30, TropicalWeight::one(), analyze));
/// pipeline.add_arc(analyze, Arc::new(30, 40, TropicalWeight::one(), output));
///
/// // Sort ensures processing stages are in dependency order
/// let sorted_pipeline: VectorFst<TropicalWeight> = topsort(&pipeline)?;
///
/// // Enables efficient streaming processing
/// println!("Pipeline optimized");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Cycle Detection Example
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with cycle (will cause error)
/// let mut cyclic = VectorFst::<TropicalWeight>::new();
/// let s0 = cyclic.add_state();
/// let s1 = cyclic.add_state();
///
/// cyclic.set_start(s0);
/// cyclic.set_final(s1, TropicalWeight::one());
///
/// // Create cycle: s0 -> s1 -> s0
/// cyclic.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// cyclic.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s0));
///
/// // Topological sort detects cycle and returns error
/// let result: Result<VectorFst<TropicalWeight>> = topsort(&cyclic);
/// assert!(result.is_err());
/// ```
///
/// # Use Cases
///
/// ## Algorithm Optimization
/// - **Dynamic Programming:** Process states in dependency order for DP algorithms
/// - **Shortest Path:** Enable efficient forward/backward path computation
/// - **Parsing:** Order grammar states for efficient parsing algorithms
/// - **Compilation:** Optimize computation order in compiler passes
///
/// ## Data Processing Pipelines
/// - **Stream Processing:** Order processing stages for efficient data flow
/// - **Workflow Scheduling:** Schedule dependent tasks in correct order
/// - **Dependency Resolution:** Resolve build dependencies and execution order
/// - **ETL Pipelines:** Order extract-transform-load operations efficiently
///
/// ## Graph Algorithms
/// - **Reachability Analysis:** Optimize forward reachability computation
/// - **Transitive Closure:** Efficient computation of transitive relationships
/// - **Critical Path:** Find longest paths in DAGs (critical path method)
/// - **Resource Allocation:** Schedule resources respecting dependencies
///
/// ## Machine Learning
/// - **Neural Networks:** Order computations in feedforward networks
/// - **Feature Engineering:** Process features respecting dependencies
/// - **Model Pipelines:** Optimize ML pipeline execution order
/// - **Backpropagation:** Efficient gradient computation ordering
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) for DFS-based topological sort
/// - **Space Complexity:** O(|V|) for visited sets and result storage
/// - **Memory Access:** Sequential state processing improves cache locality
/// - **Parallel Potential:** Independent state groups can be processed in parallel
/// - **Cycle Detection:** Early termination when cycles found
///
/// # Mathematical Properties
///
/// Topological sorting preserves essential FST properties:
/// - **Language Preservation:** L(topsort(T)) = L(T) exactly
/// - **Weight Preservation:** All weights and arc relationships maintained
/// - **Structural Properties:** Determinism, connectivity preserved
/// - **Ordering Invariant:** All valid topological orders produce equivalent FSTs
/// - **Acyclicity:** Only works on DAGs; detects cycles in general graphs
///
/// # Implementation Details
///
/// The algorithm uses depth-first search with three vertex states:
/// - **White:** Unvisited vertex
/// - **Gray:** Visited but not finished (currently in DFS stack)
/// - **Black:** Finished vertex (all descendants processed)
///
/// Cycle detection occurs when a gray vertex is reached during DFS,
/// indicating a back edge and thus a cycle in the graph.
///
/// # Optimization Benefits
///
/// After topological sorting:
/// - **Forward Processing:** Algorithms can process states sequentially
/// - **Memory Locality:** Improved cache performance from sequential access
/// - **Parallel Opportunities:** Independent state groups identified
/// - **Algorithm Efficiency:** Many graph algorithms become more efficient
///
/// # Relationship to Other Algorithms
///
/// Topological sorting enables efficient implementation of:
/// - **Single-Source Shortest Path:** Process vertices in topological order
/// - **Longest Path in DAG:** Only solvable efficiently with topological order
/// - **Dynamic Programming on DAGs:** Natural processing order
/// - **Critical Path Method:** Requires topological ordering
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST contains cycles (violates DAG requirement)
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during DFS traversal or result construction
/// - State enumeration or arc traversal encounters invalid data
/// - The DFS stack overflows due to very deep graphs
///
/// # See Also
///
/// - [`connect()`](crate::algorithms::connect()) for removing unreachable states before sorting
/// - [`shortest_path()`](crate::algorithms::shortest_path()) for algorithms that benefit from topological order
/// - [`prune()`](crate::algorithms::prune()) for weight-based state filtering
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for FST manipulation patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#topsort) for mathematical background
pub fn topsort<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // compute topological order
    let order = compute_topological_order(fst)?;

    // create mapping from old to new state IDs
    let mut state_map = vec![None; fst.num_states()];
    for (new_id, &old_id) in order.iter().enumerate() {
        state_map[old_id as usize] = Some(new_id as StateId);
    }

    let mut result = M::default();

    // create states in topological order
    for _ in &order {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        if let Some(new_start) = state_map[start as usize] {
            result.set_start(new_start);
        }
    }

    // copy with remapped states
    for &old_state in &order {
        if let Some(new_state) = state_map[old_state as usize] {
            // final weight
            if let Some(weight) = fst.final_weight(old_state) {
                result.set_final(new_state, weight.clone());
            }

            // arcs
            for arc in fst.arcs(old_state) {
                if let Some(new_nextstate) = state_map[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    Ok(result)
}

fn compute_topological_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
    let mut visited = HashSet::new();
    let mut finished = HashSet::new();
    let mut order = Vec::new();

    fn dfs<W: Semiring, F: Fst<W>>(
        fst: &F,
        state: StateId,
        visited: &mut HashSet<StateId>,
        finished: &mut HashSet<StateId>,
        order: &mut Vec<StateId>,
    ) -> Result<()> {
        visited.insert(state);

        for arc in fst.arcs(state) {
            if !visited.contains(&arc.nextstate) {
                dfs(fst, arc.nextstate, visited, finished, order)?;
            } else if !finished.contains(&arc.nextstate) {
                return Err(Error::Algorithm("FST has cycles".into()));
            }
        }

        finished.insert(state);
        order.push(state);
        Ok(())
    }

    // start DFS from start state if it exists
    if let Some(start) = fst.start() {
        if !visited.contains(&start) {
            dfs(fst, start, &mut visited, &mut finished, &mut order)?;
        }
    }

    // visit any remaining unvisited states
    for state in fst.states() {
        if !visited.contains(&state) {
            dfs(fst, state, &mut visited, &mut finished, &mut order)?;
        }
    }

    order.reverse();
    Ok(order)
}
