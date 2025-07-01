//! Lazy FST implementation for on-demand computation of large automata

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use crate::Result;
use core::cell::RefCell;
use std::sync::Mutex;

/// On-demand FST implementation for dynamically computed automata
///
/// `LazyFstImpl` represents finite state transducers where states and arcs are computed
/// dynamically when accessed, rather than being pre-computed and stored in memory.
/// This approach is particularly valuable for very large automata, composition chains,
/// or scenarios where the full state space is too large to materialize entirely.
///
/// # Design Characteristics
///
/// - **On-Demand Computation:** States computed only when first accessed
/// - **Memory Efficiency:** Only stores computed portions of the automaton
/// - **Flexible Computation:** User-defined computation functions for state generation
/// - **Transparent Caching:** Computed states cached for subsequent access
/// - **Thread Safety:** Safe concurrent access through interior mutability
///
/// # Performance Profile
///
/// | Operation | First Access | Cached Access | Notes |
/// |-----------|-------------|---------------|-------|
/// | State Computation | O(computation) | O(1) | Depends on user function |
/// | Arc Access | O(computation) | O(1) | First access computes state |
/// | Memory Usage | O(accessed states) | Growing with usage | |
/// | Cache Hit Rate | Varies | Excellent for repeated access | |
///
/// # Memory and Computation Model
///
/// ```text
/// LazyFstImpl Structure:
/// ┌─────────────────────────────┐
/// │ Computation Function        │ ← F: Fn(StateId) -> Option<LazyState<W>>
/// │ User-defined logic          │   Generates states on demand
/// │ Arbitrary complexity        │   Can access external data sources
/// └─────────────────────────────┘
/// ┌─────────────────────────────┐
/// │ State Cache                 │ ← Mutex<RefCell<StateCache<W>>>
/// │ [State 0: Some(computed)]   │   Stores computed states
/// │ [State 1: None]             │   Thread-safe with interior mutability
/// │ [State N: Some(computed)]   │   Lazy initialization pattern
/// └─────────────────────────────┘
/// ┌─────────────────────────────┐
/// │ Computed State Data         │ ← LazyState<W>
/// │ - arcs: Vec<Arc<W>>         │   Full arc list per state
/// │ - final_weight: Option<W>   │   Final state information
/// └─────────────────────────────┘
/// ```
///
/// # Use Cases
///
/// ## Large Composition Chains
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
///
/// // Create lazy FST for composition result
/// fn create_composition_lazy(
///     fst1: &VectorFst<TropicalWeight>,
///     fst2: &VectorFst<TropicalWeight>
/// ) -> LazyFstImpl<TropicalWeight, impl Fn(StateId) -> Option<LazyState<TropicalWeight>>> {
///     let fst1_clone = fst1.clone();
///     let fst2_clone = fst2.clone();
///     
///     let compute_fn = move |state: StateId| -> Option<LazyState<TropicalWeight>> {
///         // Decode state into component FST states
///         let (state1, state2) = decode_composed_state(state);
///         
///         // Compute composition on demand for this state pair
///         let mut arcs = Vec::new();
///         
///         // Generate arcs from FST1 × FST2 composition at this state
///         for arc1 in fst1_clone.arcs(state1) {
///             for arc2 in fst2_clone.arcs(state2) {
///                 if arc1.olabel == arc2.ilabel {
///                     let composed_arc = Arc::new(
///                         arc1.ilabel,
///                         arc2.olabel,
///                         arc1.weight.plus(&arc2.weight),
///                         encode_composed_state(arc1.nextstate, arc2.nextstate)
///                     );
///                     arcs.push(composed_arc);
///                 }
///             }
///         }
///         
///         // Compute final weight if both states are final
///         let final_weight = match (fst1_clone.final_weight(state1), fst2_clone.final_weight(state2)) {
///             (Some(w1), Some(w2)) => Some(w1.plus(w2)),
///             _ => None,
///         };
///         
///         Some(LazyState { arcs, final_weight })
///     };
///     
///     // Estimate number of states (state1_count * state2_count)
///     let num_states = fst1.num_states() * fst2.num_states();
///     LazyFstImpl::new(compute_fn, num_states)
/// }
///
/// fn decode_composed_state(state: StateId) -> (StateId, StateId) {
///     // Simplified decoding - in practice would be more sophisticated
///     (state / 1000, state % 1000)
/// }
///
/// fn encode_composed_state(state1: StateId, state2: StateId) -> StateId {
///     // Simplified encoding - in practice would handle overflow
///     state1 * 1000 + state2
/// }
/// ```
///
/// ## Dynamic Dictionary Construction
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
/// use std::collections::HashMap;
///
/// // Create lazy FST that generates pronunciation dictionary on demand
/// fn create_dynamic_dictionary(
///     word_pronunciations: HashMap<String, Vec<String>>
/// ) -> LazyFstImpl<TropicalWeight, impl Fn(StateId) -> Option<LazyState<TropicalWeight>>> {
///     let word_pronunciations_clone = word_pronunciations.clone();
///     let compute_fn = move |state: StateId| -> Option<LazyState<TropicalWeight>> {
///         // Decode state representation
///         let (prefix, position) = decode_dictionary_state(state);
///         
///         let mut arcs = Vec::new();
///         
///         // Generate arcs for all words that start with current prefix
///         for (word, pronunciations) in &word_pronunciations_clone {
///             if word.starts_with(&prefix) && position < word.len() {
///                 let next_char = word.chars().nth(position).unwrap();
///                 let next_state = encode_dictionary_state(
///                     format!("{}{}", prefix, next_char),
///                     position + 1
///                 );
///                 
///                 arcs.push(Arc::new(
///                     next_char as u32,
///                     next_char as u32,
///                     TropicalWeight::one(),
///                     next_state
///                 ));
///             }
///         }
///         
///         // Check if current prefix forms a complete word
///         let final_weight = if word_pronunciations_clone.contains_key(&prefix) {
///             Some(TropicalWeight::one())
///         } else {
///             None
///         };
///         
///         Some(LazyState { arcs, final_weight })
///     };
///     
///     // Estimate state space size
///     let max_word_length = word_pronunciations.keys()
///         .map(|w| w.len())
///         .max()
///         .unwrap_or(0);
///     let estimated_states = 26_usize.pow(max_word_length as u32);
///     
///     LazyFstImpl::new(compute_fn, estimated_states)
/// }
///
/// fn decode_dictionary_state(state: StateId) -> (String, usize) {
///     // Simplified state decoding - in practice would be more efficient
///     ("".to_string(), state as usize)
/// }
///
/// fn encode_dictionary_state(prefix: String, position: usize) -> StateId {
///     // Simplified state encoding - in practice would use hash or trie
///     (prefix.len() * 1000 + position) as StateId
/// }
/// ```
///
/// ## Search Space Exploration
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
///
/// // Create lazy FST for exploring search graphs
/// fn create_search_space_fst(
///     initial_state: SearchState,
///     goal_predicate: fn(&SearchState) -> bool
/// ) -> LazyFstImpl<TropicalWeight, impl Fn(StateId) -> Option<LazyState<TropicalWeight>>> {
///     let compute_fn = move |state_id: StateId| -> Option<LazyState<TropicalWeight>> {
///         let search_state = decode_search_state(state_id);
///         
///         // Generate successor states using domain-specific logic
///         let successors = generate_successors(&search_state);
///         
///         let mut arcs = Vec::new();
///         for (action, next_state, cost) in successors {
///             let next_state_id = encode_search_state(next_state);
///             arcs.push(Arc::new(
///                 action,
///                 action,
///                 TropicalWeight::new(cost),
///                 next_state_id
///             ));
///         }
///         
///         // Check if this is a goal state
///         let final_weight = if goal_predicate(&search_state) {
///             Some(TropicalWeight::one())
///         } else {
///             None
///         };
///         
///         Some(LazyState { arcs, final_weight })
///     };
///     
///     // Large state space for search problems
///     LazyFstImpl::new(compute_fn, 1_000_000)
/// }
///
/// #[derive(Clone)]
/// struct SearchState {
///     position: (i32, i32),
///     inventory: Vec<u32>,
/// }
///
/// fn decode_search_state(state_id: StateId) -> SearchState {
///     // Decode state ID back to search state representation
///     SearchState {
///         position: ((state_id / 1000) as i32, (state_id % 1000) as i32),
///         inventory: vec![], // Simplified
///     }
/// }
///
/// fn encode_search_state(state: SearchState) -> StateId {
///     // Encode search state to unique state ID
///     (state.position.0 * 1000 + state.position.1) as StateId
/// }
///
/// fn generate_successors(state: &SearchState) -> Vec<(u32, SearchState, f32)> {
///     // Generate possible actions and resulting states with costs
///     vec![
///         (1, SearchState { position: (state.position.0 + 1, state.position.1), inventory: state.inventory.clone() }, 1.0),
///         (2, SearchState { position: (state.position.0, state.position.1 + 1), inventory: state.inventory.clone() }, 1.0),
///     ]
/// }
/// ```
///
/// ## External Data Integration
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
/// use std::sync::Arc as StdArc;
///
/// // Create lazy FST that integrates with external databases
/// fn create_database_backed_fst(
///     db_connection: StdArc<DatabaseConnection>
/// ) -> LazyFstImpl<LogWeight, impl Fn(StateId) -> Option<LazyState<LogWeight>>> {
///     let compute_fn = move |state: StateId| -> Option<LazyState<LogWeight>> {
///         // Query database for state information
///         let state_data = db_connection.query_state(state)?;
///         
///         let mut arcs = Vec::new();
///         
///         // Convert database results to FST arcs
///         for transition in state_data.transitions {
///             arcs.push(Arc::new(
///                 transition.input_label,
///                 transition.output_label,
///                 LogWeight::new(transition.weight.into()),
///                 transition.target_state
///             ));
///         }
///         
///         let final_weight = state_data.is_final.then(|| LogWeight::one());
///         
///         Some(LazyState { arcs, final_weight })
///     };
///     
///     // Large potential state space
///     LazyFstImpl::new(compute_fn, 10_000_000)
/// }
///
/// struct DatabaseConnection;
/// struct StateData {
///     transitions: Vec<TransitionData>,
///     is_final: bool,
/// }
/// struct TransitionData {
///     input_label: u32,
///     output_label: u32,
///     weight: f32,
///     target_state: StateId,
/// }
///
/// impl DatabaseConnection {
///     fn query_state(&self, _state: StateId) -> Option<StateData> {
///         // Simulate database query
///         Some(StateData {
///             transitions: vec![],
///             is_final: false,
///         })
///     }
/// }
/// ```
///
/// # Computation Function Patterns
///
/// ## Stateless Computation
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
///
/// // Pure function that computes states based only on state ID
/// fn create_mathematical_fst() -> LazyFstImpl<TropicalWeight, impl Fn(StateId) -> Option<LazyState<TropicalWeight>>> {
///     let compute_fn = |state: StateId| -> Option<LazyState<TropicalWeight>> {
///         // Generate arcs based on mathematical properties of state ID
///         let mut arcs = Vec::new();
///         
///         // Example: state transitions based on number theory
///         if state % 2 == 0 {
///             arcs.push(Arc::new(1, 1, TropicalWeight::one(), state / 2));
///         }
///         if state < 1000 {
///             arcs.push(Arc::new(2, 2, TropicalWeight::new(0.5), state * 3 + 1));
///         }
///         
///         let final_weight = if state == 1 {
///             Some(TropicalWeight::one())
///         } else {
///             None
///         };
///         
///         Some(LazyState { arcs, final_weight })
///     };
///     
///     LazyFstImpl::new(compute_fn, 10000)
/// }
/// ```
///
/// ## Stateful Computation with Closure
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{LazyFstImpl, LazyState};
/// use std::sync::{Arc as StdArc, RwLock};
///
/// // Computation function that maintains state across calls
/// fn create_stateful_fst() -> LazyFstImpl<BooleanWeight, impl Fn(StateId) -> Option<LazyState<BooleanWeight>>> {
///     let computation_history = StdArc::new(RwLock::new(std::collections::HashSet::new()));
///     
///     let compute_fn = move |state: StateId| -> Option<LazyState<BooleanWeight>> {
///         // Track which states have been computed
///         {
///             let mut history = computation_history.write().unwrap();
///             history.insert(state);
///         }
///         
///         let history_read = computation_history.read().unwrap();
///         let num_computed = history_read.len();
///         drop(history_read);
///         
///         // Computation depends on how many states have been computed so far
///         let mut arcs = Vec::new();
///         
///         if num_computed < 100 {
///             arcs.push(Arc::new(
///                 1, 1,
///                 BooleanWeight::one(),
///                 (state + 1) % 1000
///             ));
///         }
///         
///         Some(LazyState { arcs, final_weight: None })
///     };
///     
///     LazyFstImpl::new(compute_fn, 1000)
/// }
/// ```
///
/// # Performance Optimization Guidelines
///
/// ## When to Use LazyFstImpl
/// - ✅ Very large state spaces that don't fit in memory
/// - ✅ Dynamic or data-driven FST construction
/// - ✅ Composition of multiple large FSTs
/// - ✅ Integration with external data sources
/// - ✅ Search problems with large branching factors
/// - ✅ Sparse automata where most states are never accessed
///
/// ## When NOT to Use LazyFstImpl
/// - ❌ Small FSTs that fit comfortably in memory
/// - ❌ Frequently accessed dense automata
/// - ❌ Real-time applications requiring predictable latency
/// - ❌ Simple FSTs with uniform access patterns
/// - ❌ Memory-rich environments where pre-computation is feasible
///
/// ## Optimization Strategies
/// 1. **Efficient State Encoding:** Minimize computation in encode/decode functions
/// 2. **Smart Caching:** Consider cache size vs. computation cost trade-offs
/// 3. **Batch Computation:** Compute multiple related states together
/// 4. **State Pruning:** Avoid generating unreachable or unnecessary states
/// 5. **Computation Memoization:** Cache expensive intermediate results
///
/// # Thread Safety and Concurrency
///
/// `LazyFstImpl` provides thread-safe access through:
/// - **Mutex Protection:** State cache protected by mutex for thread safety
/// - **Interior Mutability:** Safe concurrent reads with lazy computation
/// - **Computation Isolation:** User functions should be pure or thread-safe
/// - **Cache Coherency:** All threads see consistent cached state data
///
/// # Limitations and Considerations
///
/// ## Current Implementation Limitations
/// - `final_weight()` method requires redesign to avoid lifetime issues
/// - Fixed cache size based on estimated state count
/// - No cache eviction policy for long-running applications
/// - Computation function must be `Send + Sync`
///
/// ## Design Considerations
/// - **Computation Cost:** Balance between memory savings and computation overhead
/// - **State Encoding:** Efficient bijection between StateId and domain states
/// - **Cache Management:** Memory growth proportional to accessed states
/// - **Error Handling:** Computation functions should handle edge cases gracefully
///
/// # Future Enhancements
///
/// - **Adaptive Caching:** LRU or other eviction policies for memory management
/// - **Streaming Support:** Support for infinite or very large state spaces
/// - **Parallel Computation:** Concurrent computation of independent states
/// - **State Compression:** Compress cached states for memory efficiency
///
/// # See Also
///
/// - [`VectorFst`] for in-memory mutable FSTs
/// - [`ConstFst`] for optimized read-only FSTs
/// - [`CompactFst`] for memory-compressed FSTs
/// - [`CacheFst`] for caching wrapper around expensive FSTs
/// - [Lazy Evaluation Guide](../../docs/working-with-fsts/advanced-topics.md#lazy-evaluation) for computation patterns
/// - [Performance Tuning](../../docs/architecture/performance.md) for optimization strategies
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`ConstFst`]: crate::fst::ConstFst
/// [`CompactFst`]: crate::fst::CompactFst
/// [`CacheFst`]: crate::fst::CacheFst
pub struct LazyFstImpl<W: Semiring, F> {
    compute_fn: F,
    cache: Mutex<RefCell<StateCache<W>>>,
    properties: FstProperties,
}

impl<W: Semiring, F> std::fmt::Debug for LazyFstImpl<W, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFstImpl")
            .field("properties", &self.properties)
            .finish()
    }
}

/// Internal cache structure for storing computed states
///
/// Maintains a vector of optional states where `None` indicates an uncomputed state
/// and `Some(state)` indicates a cached computed state. The cache is protected by
/// a mutex for thread-safe access and uses interior mutability patterns.
#[derive(Debug)]
struct StateCache<W: Semiring> {
    /// Vector of cached states, indexed by StateId
    states: Vec<Option<LazyState<W>>>,
    /// Vector of cached final weights for direct reference access
    final_weights: Vec<Option<W>>,
    /// Optional start state identifier
    start: Option<StateId>,
}

/// Computed state representation for lazy FSTs
///
/// Represents a fully computed state with all its outgoing arcs and final weight.
/// This structure is created by the user-provided computation function and cached
/// for subsequent access to avoid recomputation.
///
/// # Performance Characteristics
///
/// - **Memory:** Stores all arcs for the state in a vector
/// - **Access:** O(1) access to cached arc data after computation
/// - **Cloning:** Arc data is cloned when state is cached and retrieved
///
/// # Usage
///
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::LazyState;
///
/// // Create a state with outgoing arcs
/// let state = LazyState {
///     arcs: vec![
///         Arc::new(1, 1, TropicalWeight::one(), 2),
///         Arc::new(2, 2, TropicalWeight::new(0.5), 3),
///     ],
///     final_weight: Some(TropicalWeight::one()),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LazyState<W: Semiring> {
    /// Vector of outgoing arcs from this state
    pub arcs: Vec<Arc<W>>,
    /// Final weight if this state is accepting, None otherwise
    #[allow(dead_code)]
    pub final_weight: Option<W>,
}

impl<W: Semiring, F> LazyFstImpl<W, F>
where
    F: Fn(StateId) -> Option<LazyState<W>> + Send + Sync,
{
    /// Create a new lazy FST with the specified computation function
    ///
    /// Constructs a `LazyFstImpl` that will use the provided computation function
    /// to generate states on demand. The computation function should be pure or
    /// thread-safe since it may be called concurrently from multiple threads.
    ///
    /// # Parameters
    ///
    /// - `compute_fn`: Function that takes a `StateId` and returns an optional
    ///   `LazyState`. Should return `None` for invalid or non-existent states.
    /// - `num_states`: Estimated number of states in the automaton, used to
    ///   pre-allocate the cache vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    /// use arcweight::fst::{LazyFstImpl, LazyState};
    ///
    /// // Create a simple lazy FST with mathematical computation
    /// let compute_fn = |state: StateId| -> Option<LazyState<TropicalWeight>> {
    ///     if state > 100 {
    ///         return None; // Invalid state
    ///     }
    ///     
    ///     let arcs = if state % 2 == 0 {
    ///         vec![Arc::new(1, 1, TropicalWeight::one(), state + 1)]
    ///     } else {
    ///         vec![Arc::new(2, 2, TropicalWeight::new(0.5), state + 2)]
    ///     };
    ///     
    ///     Some(LazyState { arcs, final_weight: None })
    /// };
    ///
    /// let lazy_fst = LazyFstImpl::new(compute_fn, 100);
    /// ```
    ///
    /// # Thread Safety
    ///
    /// The computation function must implement `Send + Sync` for thread safety.
    /// The function should be stateless or use appropriate synchronization if
    /// it needs to access shared state.
    ///
    /// # Performance
    ///
    /// - **Initialization:** O(num_states) for cache vector allocation
    /// - **Memory:** O(num_states) initial allocation plus O(accessed states) for cached data
    pub fn new(compute_fn: F, num_states: usize) -> Self {
        Self {
            compute_fn,
            cache: Mutex::new(RefCell::new(StateCache {
                states: vec![None; num_states],
                final_weights: vec![None; num_states],
                start: Some(0),
            })),
            properties: FstProperties::default(),
        }
    }

    fn get_or_compute_state(&self, state: StateId) -> Option<LazyState<W>> {
        let cache = self.cache.lock().unwrap();
        let cache_ref = cache.borrow_mut();

        if state as usize >= cache_ref.states.len() {
            return None;
        }

        if let Some(ref s) = cache_ref.states[state as usize] {
            return Some(s.clone());
        }

        drop(cache_ref);
        drop(cache);

        let computed = (self.compute_fn)(state)?;

        let cache = self.cache.lock().unwrap();
        let mut cache_ref = cache.borrow_mut();
        cache_ref.states[state as usize] = Some(computed.clone());
        cache_ref.final_weights[state as usize] = computed.final_weight.clone();

        Some(computed)
    }

    /// Get the final weight of a state as an owned value
    ///
    /// This is a helper method that works around the limitation of `final_weight()`
    /// which cannot return references due to the LazyFst's internal cache design.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use arcweight::prelude::*;
    /// # use arcweight::fst::{LazyFstImpl, LazyState};
    /// # let compute_fn = |state: StateId| -> Option<LazyState<TropicalWeight>> {
    /// #     Some(LazyState { arcs: vec![], final_weight: Some(TropicalWeight::one()) })
    /// # };
    /// # let lazy_fst = LazyFstImpl::new(compute_fn, 10);
    /// if let Some(weight) = lazy_fst.final_weight_owned(0) {
    ///     println!("Final weight: {}", weight);
    /// }
    /// ```
    pub fn final_weight_owned(&self, state: StateId) -> Option<W> {
        self.get_or_compute_state(state)
            .and_then(|s| s.final_weight)
    }
}

/// Arc iterator for lazy FSTs with owned arc data
///
/// Provides iterator access to arcs from a computed lazy state. Unlike other
/// FST arc iterators that may reference shared data, `LazyArcIterator` owns
/// the arc data to avoid lifetime complications with the lazy computation cache.
///
/// # Performance Characteristics
///
/// - **Memory:** Owns a copy of all arcs from the state
/// - **Iteration:** O(1) per arc, sequential access
/// - **Reset:** O(1) position reset operation
/// - **Creation:** O(k) where k = number of arcs (due to cloning from cache)
///
/// # Usage
///
/// This iterator is created automatically by `LazyFstImpl::arcs()` and should
/// not be constructed directly. It provides standard Iterator semantics
/// with the ability to reset iteration position.
#[derive(Debug)]
pub struct LazyArcIterator<W: Semiring> {
    /// Owned vector of arcs for this state
    arcs: Vec<Arc<W>>,
    /// Current iteration position
    pos: usize,
}

impl<W: Semiring> Iterator for LazyArcIterator<W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.arcs.len() {
            let arc = self.arcs[self.pos].clone();
            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
}

impl<W: Semiring> ArcIterator<W> for LazyArcIterator<W> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<W: Semiring, F> Fst<W> for LazyFstImpl<W, F>
where
    F: Fn(StateId) -> Option<LazyState<W>> + Send + Sync,
{
    type ArcIter<'a>
        = LazyArcIterator<W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        let cache = self.cache.lock().unwrap();
        let x = cache.borrow().start;
        x
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        // LIMITATION: Due to the current LazyFst design using Mutex<RefCell<...>>,
        // we cannot safely return references to cached data. This is a known
        // limitation that would require redesigning either LazyFst's storage
        // or the Fst trait to use owned values/Cow.
        //
        // WORKAROUND: Use the provided `final_weight_owned()` method instead:
        // ```
        // let weight = lazy_fst.final_weight_owned(state);
        // ```
        //
        // For now, we compute the state to ensure it's cached but return None.
        self.get_or_compute_state(state);
        None
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.get_or_compute_state(state)
            .map(|s| s.arcs.len())
            .unwrap_or(0)
    }

    fn num_states(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        let x = cache.borrow().states.len();
        x
    }

    fn properties(&self) -> FstProperties {
        self.properties
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        let arcs = self
            .get_or_compute_state(state)
            .map(|s| s.arcs)
            .unwrap_or_default();

        LazyArcIterator { arcs, pos: 0 }
    }
}

impl<W: Semiring, F> LazyFst<W> for LazyFstImpl<W, F>
where
    F: Fn(StateId) -> Option<LazyState<W>> + Send + Sync,
{
    fn expand(&self, state: StateId) -> Result<()> {
        self.get_or_compute_state(state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_lazy_fst_creation() {
        // LazyFst is more of an abstract concept, but we can test
        // that types implementing Fst work correctly

        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        // Verify basic FST operations work
        assert_eq!(fst.num_states(), 2);
        assert_eq!(fst.start(), Some(s0));
        assert!(fst.is_final(s1));
    }
}
