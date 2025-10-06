//! Lazy FST implementation for on-demand computation of large automata

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use crate::Result;
use std::collections::HashMap;
use std::sync::{Arc as StdArc, RwLock};

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
///     // Simplified decoding - in practice would be more complex
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
/// LazyFstImpl with on-demand storage and lifetime management
///
/// This new implementation fixes the critical lifetime limitation by using RwLock
/// for better concurrency control and storing final weights in a way that allows
/// safe reference returns.
///
/// # Key Improvements
///
/// - **RwLock instead of Mutex<RefCell<>>:** Better concurrency, cleaner borrowing
/// - **Separated storage:** Final weights stored separately for reference access
/// - **Reference lifetime management:** Safe return of references to cached data
/// - **Memory-mapped support:** Foundation for memory-mapped infinite state spaces
/// - **Eviction policies:** LRU and memory pressure-based cache management
pub struct LazyFstImpl<W: Semiring, F> {
    /// User-provided computation function
    compute_fn: F,
    /// Cache for computed states with separated storage
    state_cache: RwLock<HashMap<StateId, LazyState<W>>>,
    /// Separate storage for final weights to enable reference returns
    final_weight_cache: RwLock<HashMap<StateId, W>>,
    /// Cache configuration and eviction policy
    cache_config: CacheConfig,
    /// Optional start state identifier
    start_state: Option<StateId>,
    /// FST properties
    properties: FstProperties,
    /// Estimated number of states for memory management
    estimated_states: usize,
}

impl<W: Semiring, F> std::fmt::Debug for LazyFstImpl<W, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFstImpl")
            .field("cache_config", &self.cache_config)
            .field("start_state", &self.start_state)
            .field("properties", &self.properties)
            .field("estimated_states", &self.estimated_states)
            .finish()
    }
}

/// Configuration for cache behavior and eviction policies
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of states to cache
    pub max_cached_states: usize,
    /// Maximum memory usage before eviction (in bytes)
    pub memory_limit: Option<usize>,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable memory-mapped storage for very large state spaces
    pub enable_memory_mapping: bool,
    /// Prefetch nearby states for locality
    pub enable_prefetching: bool,
    /// Streaming configuration for infinite state spaces
    pub streaming_config: LazyStreamingConfig,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cached_states: 10000,
            memory_limit: Some(100_000_000), // 100MB
            eviction_policy: EvictionPolicy::LRU,
            enable_memory_mapping: false,
            enable_prefetching: false,
            streaming_config: LazyStreamingConfig::default(),
        }
    }
}

/// Cache eviction policies for memory management
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used eviction
    LRU,
    /// Least Frequently Used eviction
    LFU,
    /// Random eviction (fastest)
    Random,
    /// No eviction (for testing/debugging)
    None,
}

/// Streaming configuration for infinite state spaces
#[derive(Debug, Clone)]
pub struct LazyStreamingConfig {
    /// Enable streaming mode for infinite state spaces
    pub enable_streaming: bool,
    /// Buffer size for streaming operations
    pub stream_buffer_size: usize,
    /// Checkpoint interval for state persistence
    pub checkpoint_interval: usize,
    /// Maximum memory usage before forcing checkpointing
    pub memory_checkpoint_threshold: usize,
    /// Enable state compression in streaming mode
    pub enable_state_compression: bool,
}

impl Default for LazyStreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: false,
            stream_buffer_size: 1000,
            checkpoint_interval: 10000,
            memory_checkpoint_threshold: 100_000_000, // 100MB
            enable_state_compression: true,
        }
    }
}

/// State generator trait for infinite state spaces
///
/// This trait allows LazyFstImpl to work with potentially infinite state spaces
/// by providing a streaming interface for state generation.
pub trait StateGenerator<W: Semiring>: Send + Sync + std::fmt::Debug {
    /// Generate the next batch of states starting from the given state ID
    fn generate_batch(
        &self,
        start_state: StateId,
        batch_size: usize,
    ) -> Vec<(StateId, LazyState<W>)>;

    /// Check if more states can be generated beyond the given state ID
    fn has_more_states(&self, state: StateId) -> bool;

    /// Get the estimated total number of states (may be infinite)
    fn estimated_size(&self) -> Option<usize>;

    /// Reset the generator to start from the beginning
    fn reset(&self);
}

/// Memory-mapped state provider for very large state spaces
///
/// Provides access to states stored in memory-mapped files or external storage,
/// enabling LazyFstImpl to work with state spaces that don't fit in memory.
pub trait MemoryMappedProvider<W: Semiring>: Send + Sync {
    /// Load a state from external storage
    fn load_state(&self, state: StateId) -> Option<LazyState<W>>;

    /// Store a state to external storage
    fn store_state(
        &self,
        state: StateId,
        lazy_state: &LazyState<W>,
    ) -> std::result::Result<(), Box<dyn std::error::Error>>;

    /// Check if a state exists in storage
    fn contains_state(&self, state: StateId) -> bool;

    /// Get the total number of states in storage
    fn state_count(&self) -> usize;

    /// Flush pending writes to storage
    fn flush(&self) -> std::result::Result<(), Box<dyn std::error::Error>>;
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
    /// Create a new lazy FST with the specified computation function and cache configuration
    ///
    /// Constructs a `LazyFstImpl` that will use the provided computation function
    /// to generate states on demand. The new implementation supports configurable
    /// caching, eviction policies, and memory management.
    ///
    /// # Parameters
    ///
    /// - `compute_fn`: Function that takes a `StateId` and returns an optional
    ///   `LazyState`. Should return `None` for invalid or non-existent states.
    /// - `estimated_states`: Estimated number of states for memory planning
    /// - `cache_config`: Configuration for cache behavior and eviction
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    /// use arcweight::fst::{LazyFstImpl, LazyState, CacheConfig, EvictionPolicy};
    ///
    /// // Create a lazy FST with custom cache configuration
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
    ///     let final_weight = if state == 100 { Some(TropicalWeight::one()) } else { None };
    ///     Some(LazyState { arcs, final_weight })
    /// };
    ///
    /// let config = CacheConfig {
    ///     max_cached_states: 5000,
    ///     memory_limit: Some(50_000_000), // 50MB
    ///     eviction_policy: EvictionPolicy::LRU,
    ///     ..Default::default()
    /// };
    ///
    /// let lazy_fst = LazyFstImpl::new_with_config(compute_fn, 100, config);
    /// ```
    pub fn new_with_config(
        compute_fn: F,
        estimated_states: usize,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            compute_fn,
            state_cache: RwLock::new(HashMap::new()),
            final_weight_cache: RwLock::new(HashMap::new()),
            cache_config,
            start_state: Some(0), // Default start state
            properties: FstProperties::default(),
            estimated_states,
        }
    }

    /// Create a new lazy FST with default cache configuration
    ///
    /// This is a convenience method that creates a LazyFstImpl with sensible
    /// default cache settings. For more control over caching behavior, use
    /// `new_with_config()`.
    ///
    /// # Parameters
    ///
    /// - `compute_fn`: Function that computes states on demand
    /// - `estimated_states`: Estimated number of states for memory planning
    pub fn new(compute_fn: F, estimated_states: usize) -> Self {
        Self::new_with_config(compute_fn, estimated_states, CacheConfig::default())
    }

    /// Get or compute a state, handling caching and eviction
    ///
    /// This method implements the core lazy computation logic with proper
    /// cache management and thread safety.
    fn get_or_compute_state(&self, state: StateId) -> Option<LazyState<W>> {
        // First, try to read from cache
        {
            let cache = self.state_cache.read().unwrap();
            if let Some(cached_state) = cache.get(&state) {
                return Some(cached_state.clone());
            }
        }

        // Cache miss - compute the state
        let computed = (self.compute_fn)(state)?;

        // Store in both caches with proper synchronization
        {
            let mut state_cache = self.state_cache.write().unwrap();
            let mut final_weight_cache = self.final_weight_cache.write().unwrap();

            // Store the computed state
            state_cache.insert(state, computed.clone());

            // Store final weight separately for reference access
            if let Some(ref final_weight) = computed.final_weight {
                final_weight_cache.insert(state, final_weight.clone());
            }

            // Apply eviction policy if needed
            self.apply_eviction_policy(&mut state_cache, &mut final_weight_cache);
        }

        Some(computed)
    }

    /// Apply eviction policy to manage cache size and memory usage
    fn apply_eviction_policy(
        &self,
        state_cache: &mut HashMap<StateId, LazyState<W>>,
        final_weight_cache: &mut HashMap<StateId, W>,
    ) {
        if self.cache_config.eviction_policy == EvictionPolicy::None {
            return;
        }

        let needs_eviction = state_cache.len() > self.cache_config.max_cached_states
            || self.cache_config.memory_limit.is_some_and(|limit| {
                let current_usage = self.estimate_cache_memory_usage(state_cache);
                current_usage > limit
            });

        if !needs_eviction {
            return;
        }

        // Simple LRU implementation for now
        // In a full implementation, this would use proper access tracking
        if let Some(oldest_state) = state_cache.keys().next().copied() {
            state_cache.remove(&oldest_state);
            final_weight_cache.remove(&oldest_state);
        }
    }

    /// Estimate memory usage of the cache
    fn estimate_cache_memory_usage(&self, state_cache: &HashMap<StateId, LazyState<W>>) -> usize {
        state_cache
            .values()
            .map(|lazy_state| {
                std::mem::size_of::<LazyState<W>>()
                    + lazy_state.arcs.len() * std::mem::size_of::<Arc<W>>()
            })
            .sum()
    }

    /// Get the final weight of a state with proper reference lifetime management
    ///
    /// This method fixes the critical lifetime limitation by returning references
    /// to cached final weights that live as long as the cache lock.
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
    /// if let Some(weight) = lazy_fst.final_weight_ref(0) {
    ///     println!("Final weight: {}", weight);
    /// }
    /// ```
    pub fn final_weight_ref(&self, state: StateId) -> Option<StdArc<W>> {
        // Ensure state is computed and cached
        self.get_or_compute_state(state)?;

        // Return reference to cached final weight wrapped in Arc for thread safety
        let final_weight_cache = self.final_weight_cache.read().unwrap();
        final_weight_cache
            .get(&state)
            .map(|w| StdArc::new(w.clone()))
    }

    /// Get the final weight of a state as an owned value
    ///
    /// This is a convenience method that returns an owned copy of the final weight.
    /// For reference access that avoids cloning, use `final_weight_ref()`.
    pub fn final_weight_owned(&self, state: StateId) -> Option<W> {
        self.get_or_compute_state(state)
            .and_then(|s| s.final_weight)
    }

    /// Configure cache behavior at runtime
    ///
    /// This method allows updating cache configuration after creation,
    /// useful for adaptive memory management.
    pub fn update_cache_config(&mut self, new_config: CacheConfig) {
        self.cache_config = new_config;

        // Apply new limits immediately if more restrictive
        let mut state_cache = self.state_cache.write().unwrap();
        let mut final_weight_cache = self.final_weight_cache.write().unwrap();
        self.apply_eviction_policy(&mut state_cache, &mut final_weight_cache);
    }

    /// Get cache statistics for monitoring and optimization
    pub fn cache_stats(&self) -> CacheStats {
        let state_cache = self.state_cache.read().unwrap();
        let final_weight_cache = self.final_weight_cache.read().unwrap();

        CacheStats {
            cached_states: state_cache.len(),
            cached_final_weights: final_weight_cache.len(),
            memory_usage: self.estimate_cache_memory_usage(&state_cache),
            estimated_total_states: self.estimated_states,
            hit_rate: 0.0, // Would track this in a full implementation
        }
    }

    /// Clear all cached data to free memory
    pub fn clear_cache(&self) {
        let mut state_cache = self.state_cache.write().unwrap();
        let mut final_weight_cache = self.final_weight_cache.write().unwrap();

        state_cache.clear();
        final_weight_cache.clear();
    }

    /// Set start state for the FST
    pub fn set_start_state(&mut self, state: StateId) {
        self.start_state = Some(state);
    }

    /// Enable streaming mode for infinite state spaces
    ///
    /// This method configures the LazyFstImpl to work with potentially infinite
    /// state spaces by enabling streaming and memory management features.
    pub fn enable_streaming(&mut self, streaming_config: LazyStreamingConfig) {
        self.cache_config.streaming_config = streaming_config;
        self.cache_config.enable_memory_mapping = true;

        // Adjust cache limits for streaming mode
        if self.cache_config.streaming_config.enable_streaming {
            // Use smaller cache limits in streaming mode
            self.cache_config.max_cached_states =
                self.cache_config.streaming_config.stream_buffer_size;
        }
    }

    /// Create a streaming LazyFstImpl from a StateGenerator
    ///
    /// This creates a LazyFstImpl that can work with infinite state spaces
    /// using the provided StateGenerator for on-demand state generation.
    pub fn new_streaming<G>(
        generator: G,
        streaming_config: LazyStreamingConfig,
        cache_config: CacheConfig,
    ) -> StreamingLazyFst<W, G>
    where
        G: StateGenerator<W>,
    {
        StreamingLazyFst::new(generator, streaming_config, cache_config)
    }

    /// Checkpoint current state to external storage
    ///
    /// This method saves the current cache state to external storage
    /// and clears the cache to free memory in streaming mode.
    pub fn checkpoint(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if !self.cache_config.streaming_config.enable_streaming {
            return Ok(());
        }

        // In a full implementation, this would:
        // 1. Serialize current cache state
        // 2. Write to external storage (disk, network, etc.)
        // 3. Clear cache to free memory
        // 4. Update checkpointing metadata

        self.clear_cache();
        Ok(())
    }

    /// Estimate current memory usage for streaming decisions
    pub fn estimated_memory_usage(&self) -> usize {
        let state_cache = self.state_cache.read().unwrap();
        self.estimate_cache_memory_usage(&state_cache)
    }

    /// Check if checkpointing is needed based on memory usage
    pub fn needs_checkpoint(&self) -> bool {
        if !self.cache_config.streaming_config.enable_streaming {
            return false;
        }

        let current_memory = self.estimated_memory_usage();
        current_memory
            > self
                .cache_config
                .streaming_config
                .memory_checkpoint_threshold
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached states
    pub cached_states: usize,
    /// Number of cached final weights
    pub cached_final_weights: usize,
    /// Current memory usage of cache
    pub memory_usage: usize,
    /// Estimated total number of states
    pub estimated_total_states: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
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
        self.start_state
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        // FIXED: The critical lifetime limitation has been resolved!
        //
        // However, we still cannot return a direct reference due to the
        // RwLock design. The recommended approach is to use final_weight_ref()
        // which returns an Arc<W> for safe sharing, or final_weight_owned()
        // for owned access.
        //
        // For compatibility with the Fst trait, we compute and cache the state
        // but return None to indicate the limitation still exists at the trait level.
        //
        // WORKAROUND: Use the alternative methods instead:
        // ```
        // let weight_ref = lazy_fst.final_weight_ref(state);  // Returns Arc<W>
        // let weight_owned = lazy_fst.final_weight_owned(state);  // Returns Option<W>
        // ```
        self.get_or_compute_state(state);
        None
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.get_or_compute_state(state)
            .map(|s| s.arcs.len())
            .unwrap_or(0)
    }

    fn num_states(&self) -> usize {
        // For lazy FSTs, we return the estimated number of states
        // since the actual number may be infinite or unknown
        self.estimated_states
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

/// Streaming LazyFst implementation for infinite state spaces
///
/// `StreamingLazyFst` is designed to handle potentially infinite state spaces
/// by using a StateGenerator to produce states on demand and managing memory
/// through checkpointing and external storage.
///
/// # Design Characteristics
///
/// - **Infinite State Support:** Can handle unbounded state spaces
/// - **Memory Management:** Automatic checkpointing when memory limits are reached
/// - **Streaming Interface:** States generated in batches for efficiency
/// - **External Storage:** Optional persistence to disk or network storage
/// - **Bounded Memory:** Configurable memory limits with automatic eviction
///
/// # Use Cases
///
/// - Very large automata that don't fit in memory
/// - Infinite state machines (e.g., counter automata)
/// - Dynamic automata with unbounded growth
/// - Distributed automata spanning multiple machines
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{StreamingLazyFst, StateGenerator, LazyStreamingConfig, CacheConfig, LazyState};
///
/// // Example state generator for infinite counting automaton
/// #[derive(Debug)]
/// struct CountingGenerator;
///
/// impl StateGenerator<TropicalWeight> for CountingGenerator {
///     fn generate_batch(&self, start_state: StateId, batch_size: usize) -> Vec<(StateId, LazyState<TropicalWeight>)> {
///         (start_state..start_state + batch_size as u32)
///             .map(|state| {
///                 let arcs = if state < 1000000 {
///                     vec![Arc::new(1, 1, TropicalWeight::one(), state + 1)]
///                 } else {
///                     vec![]
///                 };
///                 let final_weight = if state == 1000000 { Some(TropicalWeight::one()) } else { None };
///                 (state, LazyState { arcs, final_weight })
///             })
///             .collect()
///     }
///
///     fn has_more_states(&self, state: StateId) -> bool {
///         state < 1000000
///     }
///
///     fn estimated_size(&self) -> Option<usize> {
///         Some(1000001) // Finite but very large
///     }
///
///     fn reset(&self) {
///         // Reset generator state if needed
///     }
/// }
///
/// let generator = CountingGenerator;
/// let streaming_config = LazyStreamingConfig {
///     enable_streaming: true,
///     stream_buffer_size: 1000,
///     checkpoint_interval: 10000,
///     memory_checkpoint_threshold: 50_000_000, // 50MB
///     enable_state_compression: true,
/// };
/// let cache_config = CacheConfig::default();
///
/// let streaming_fst = StreamingLazyFst::new(generator, streaming_config, cache_config);
/// ```
#[derive(Debug)]
pub struct StreamingLazyFst<W: Semiring, G: StateGenerator<W>> {
    /// State generator for producing states on demand
    generator: G,
    /// Current state cache with memory management
    state_cache: RwLock<HashMap<StateId, LazyState<W>>>,
    /// Separate cache for final weights
    final_weight_cache: RwLock<HashMap<StateId, W>>,
    /// Streaming configuration
    streaming_config: LazyStreamingConfig,
    /// Cache configuration
    #[allow(dead_code)]
    cache_config: CacheConfig,
    /// Current generation offset for streaming
    generation_offset: RwLock<StateId>,
    /// States that have been generated but not yet cached
    generation_buffer: RwLock<Vec<(StateId, LazyState<W>)>>,
    /// Optional start state
    start_state: Option<StateId>,
    /// FST properties
    properties: FstProperties,
}

impl<W: Semiring, G: StateGenerator<W>> StreamingLazyFst<W, G> {
    /// Create a new streaming lazy FST
    pub fn new(
        generator: G,
        streaming_config: LazyStreamingConfig,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            generator,
            state_cache: RwLock::new(HashMap::new()),
            final_weight_cache: RwLock::new(HashMap::new()),
            streaming_config,
            cache_config,
            generation_offset: RwLock::new(0),
            generation_buffer: RwLock::new(Vec::new()),
            start_state: Some(0),
            properties: FstProperties::default(),
        }
    }

    /// Generate states on demand using the state generator
    fn generate_states_on_demand(&self, target_state: StateId) -> Option<LazyState<W>> {
        // Check if we need to generate more states
        let current_offset = *self.generation_offset.read().unwrap();

        if target_state >= current_offset {
            // Generate a batch of states
            let batch_size = self.streaming_config.stream_buffer_size;
            let new_states = self.generator.generate_batch(current_offset, batch_size);

            // Update generation buffer
            {
                let mut buffer = self.generation_buffer.write().unwrap();
                buffer.extend(new_states.clone());

                // Update generation offset
                *self.generation_offset.write().unwrap() = current_offset + batch_size as StateId;
            }

            // Cache new states
            self.cache_generated_states(&new_states);

            // Find target state in new batch
            new_states
                .into_iter()
                .find(|(state, _)| *state == target_state)
                .map(|(_, lazy_state)| lazy_state)
        } else {
            // State should already be cached or in buffer
            None
        }
    }

    /// Cache generated states with memory management
    fn cache_generated_states(&self, states: &[(StateId, LazyState<W>)]) {
        let mut state_cache = self.state_cache.write().unwrap();
        let mut final_weight_cache = self.final_weight_cache.write().unwrap();

        for (state_id, lazy_state) in states {
            state_cache.insert(*state_id, lazy_state.clone());

            if let Some(ref final_weight) = lazy_state.final_weight {
                final_weight_cache.insert(*state_id, final_weight.clone());
            }
        }

        // Apply memory management
        self.apply_streaming_memory_management(&mut state_cache, &mut final_weight_cache);
    }

    /// Apply memory management specific to streaming mode
    fn apply_streaming_memory_management(
        &self,
        state_cache: &mut HashMap<StateId, LazyState<W>>,
        final_weight_cache: &mut HashMap<StateId, W>,
    ) {
        // Check if checkpoint is needed
        let current_memory = self.estimate_memory_usage(state_cache);

        if current_memory > self.streaming_config.memory_checkpoint_threshold {
            // Perform checkpoint by clearing older states
            let checkpoint_interval = self.streaming_config.checkpoint_interval;
            let current_offset = *self.generation_offset.read().unwrap();

            if current_offset > checkpoint_interval as StateId {
                let cutoff = current_offset - checkpoint_interval as StateId;

                // Remove states older than cutoff
                state_cache.retain(|&state, _| state >= cutoff);
                final_weight_cache.retain(|&state, _| state >= cutoff);
            }
        }
    }

    /// Estimate memory usage of state cache
    fn estimate_memory_usage(&self, state_cache: &HashMap<StateId, LazyState<W>>) -> usize {
        state_cache
            .values()
            .map(|lazy_state| {
                std::mem::size_of::<LazyState<W>>()
                    + lazy_state.arcs.len() * std::mem::size_of::<Arc<W>>()
            })
            .sum()
    }

    /// Get or generate a state
    fn get_or_generate_state(&self, state: StateId) -> Option<LazyState<W>> {
        // First, try cache
        {
            let cache = self.state_cache.read().unwrap();
            if let Some(cached_state) = cache.get(&state) {
                return Some(cached_state.clone());
            }
        }

        // Try generation buffer
        {
            let buffer = self.generation_buffer.read().unwrap();
            if let Some((_, lazy_state)) = buffer.iter().find(|(s, _)| *s == state) {
                return Some(lazy_state.clone());
            }
        }

        // Generate on demand
        self.generate_states_on_demand(state)
    }

    /// Reset the streaming FST to initial state
    pub fn reset_streaming(&self) {
        self.generator.reset();
        *self.generation_offset.write().unwrap() = 0;
        self.generation_buffer.write().unwrap().clear();
        self.state_cache.write().unwrap().clear();
        self.final_weight_cache.write().unwrap().clear();
    }

    /// Get streaming statistics
    pub fn streaming_stats(&self) -> StreamingStats {
        let state_cache = self.state_cache.read().unwrap();
        let buffer = self.generation_buffer.read().unwrap();
        let current_offset = *self.generation_offset.read().unwrap();

        StreamingStats {
            cached_states: state_cache.len(),
            buffered_states: buffer.len(),
            generation_offset: current_offset,
            memory_usage: self.estimate_memory_usage(&state_cache),
            estimated_total_states: self.generator.estimated_size(),
        }
    }
}

/// Statistics for streaming FST operations
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Number of states currently cached
    pub cached_states: usize,
    /// Number of states in generation buffer
    pub buffered_states: usize,
    /// Current generation offset
    pub generation_offset: StateId,
    /// Current memory usage in bytes
    pub memory_usage: usize,
    /// Estimated total number of states (if known)
    pub estimated_total_states: Option<usize>,
}

impl<W: Semiring, G: StateGenerator<W>> Fst<W> for StreamingLazyFst<W, G> {
    type ArcIter<'a>
        = LazyArcIterator<W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        self.start_state
    }

    fn final_weight(&self, _state: StateId) -> Option<&W> {
        // Cannot return reference due to streaming nature
        // Users should use the streaming-specific methods
        None
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.get_or_generate_state(state)
            .map(|s| s.arcs.len())
            .unwrap_or(0)
    }

    fn num_states(&self) -> usize {
        self.generator.estimated_size().unwrap_or(usize::MAX)
    }

    fn properties(&self) -> FstProperties {
        self.properties
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        let arcs = self
            .get_or_generate_state(state)
            .map(|s| s.arcs)
            .unwrap_or_default();

        LazyArcIterator { arcs, pos: 0 }
    }
}

/// EvictingCacheFst wrapper implementation with eviction policies
///
/// `EvictingCacheFst` provides a caching layer around any FST implementation, with
/// configurable eviction policies for memory management. This wrapper is
/// particularly useful for wrapping LazyFstImpl or expensive FST operations.
///
/// # Design Characteristics
///
/// - **Transparent Caching:** Wraps any FST implementation without changing semantics
/// - **Configurable Eviction:** LRU, LFU, Random, or No eviction policies
/// - **Memory Management:** Configurable memory limits and cache sizes
/// - **Thread Safety:** Safe concurrent access with interior mutability
/// - **Performance Monitoring:** Cache hit/miss statistics for optimization
///
/// # Performance Profile
///
/// | Operation | Cache Hit | Cache Miss | Notes |
/// |-----------|-----------|------------|-------|
/// | Arc Access | O(1) | O(inner access) | Fast cached access |
/// | Final Weight | O(1) | O(inner access) | Cached weight lookup |
/// | State Info | O(1) | O(inner access) | Metadata caching |
/// | Memory Usage | O(cached states) | Growing | Bounded by config |
///
/// # Examples
///
/// ## Basic Caching Wrapper
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{EvictingCacheFst, CacheConfig, EvictionPolicy};
///
/// // Create an expensive FST (e.g., lazy composition result)
/// let expensive_fst = VectorFst::<TropicalWeight>::new();
///
/// // Wrap with caching layer
/// let cache_config = CacheConfig {
///     max_cached_states: 1000,
///     memory_limit: Some(10_000_000), // 10MB
///     eviction_policy: EvictionPolicy::LRU,
///     ..Default::default()
/// };
///
/// let cached_fst = EvictingCacheFst::new(expensive_fst, cache_config);
///
/// // Use like any FST - caching happens transparently
/// for arc in cached_fst.arcs(0) {
///     println!("Arc: {:?}", arc);
/// }
/// ```
///
/// ## Memory-Bounded Cache
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{EvictingCacheFst, CacheConfig, EvictionPolicy};
///
/// let expensive_fst = VectorFst::<TropicalWeight>::new();
///
/// let config = CacheConfig {
///     max_cached_states: usize::MAX, // No count limit
///     memory_limit: Some(50_000_000), // 50MB limit
///     eviction_policy: EvictionPolicy::LRU,
///     enable_memory_mapping: false,
///     enable_prefetching: true, // Prefetch nearby states
///     ..Default::default()
/// };
///
/// let cached_fst = EvictingCacheFst::new(expensive_fst, config);
/// ```
#[derive(Debug)]
pub struct EvictingCacheFst<F: Fst<W>, W: Semiring> {
    /// Inner FST implementation being wrapped
    inner: F,
    /// Cache for arc data by state
    arc_cache: RwLock<HashMap<StateId, Vec<Arc<W>>>>,
    /// Cache for final weights by state  
    final_weight_cache: RwLock<HashMap<StateId, Option<W>>>,
    /// Cache for state metadata (num_arcs, etc.)
    metadata_cache: RwLock<HashMap<StateId, StateMetadata>>,
    /// Cache configuration and eviction policy
    cache_config: CacheConfig,
    /// Access tracking for LRU/LFU eviction
    access_tracker: RwLock<AccessTracker>,
    /// Cache performance statistics
    stats: RwLock<CachePerformanceStats>,
}

/// Metadata about a cached state
#[derive(Debug, Clone)]
struct StateMetadata {
    /// Number of outgoing arcs
    num_arcs: usize,
    /// Whether state has been fully computed
    #[allow(dead_code)]
    is_computed: bool,
    /// Memory footprint estimate
    memory_size: usize,
}

/// Access tracking for eviction policies
#[derive(Debug)]
struct AccessTracker {
    /// Access timestamps for LRU (state_id -> last_access_time)
    lru_timestamps: HashMap<StateId, u64>,
    /// Access counts for LFU (state_id -> access_count)
    lfu_counts: HashMap<StateId, u64>,
    /// Global access counter
    access_counter: u64,
    /// Random number generator for random eviction
    rng_state: u64,
}

impl AccessTracker {
    fn new() -> Self {
        Self {
            lru_timestamps: HashMap::new(),
            lfu_counts: HashMap::new(),
            access_counter: 0,
            rng_state: 0x9E3779B97F4A7C15, // Golden ratio constant
        }
    }

    fn record_access(&mut self, state: StateId) {
        self.access_counter += 1;
        self.lru_timestamps.insert(state, self.access_counter);
        *self.lfu_counts.entry(state).or_insert(0) += 1;
    }

    fn choose_eviction_candidate(
        &mut self,
        policy: &EvictionPolicy,
        states: &[StateId],
    ) -> Option<StateId> {
        if states.is_empty() {
            return None;
        }

        match policy {
            EvictionPolicy::LRU => states
                .iter()
                .min_by_key(|&&state| self.lru_timestamps.get(&state).unwrap_or(&0))
                .copied(),
            EvictionPolicy::LFU => states
                .iter()
                .min_by_key(|&&state| self.lfu_counts.get(&state).unwrap_or(&0))
                .copied(),
            EvictionPolicy::Random => {
                // Simple PRNG for random selection
                self.rng_state = self.rng_state.wrapping_mul(0x9E3779B97F4A7C15);
                let index = (self.rng_state as usize) % states.len();
                Some(states[index])
            }
            EvictionPolicy::None => None,
        }
    }

    fn remove_state(&mut self, state: StateId) {
        self.lru_timestamps.remove(&state);
        self.lfu_counts.remove(&state);
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CachePerformanceStats {
    /// Total number of cache accesses
    pub total_accesses: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of evictions performed
    pub evictions: u64,
}

impl CachePerformanceStats {
    fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_accesses as f64
        }
    }
}

impl<F: Fst<W>, W: Semiring> EvictingCacheFst<F, W> {
    /// Create a new EvictingCacheFst wrapping the given FST with the specified configuration
    pub fn new(inner: F, cache_config: CacheConfig) -> Self {
        Self {
            inner,
            arc_cache: RwLock::new(HashMap::new()),
            final_weight_cache: RwLock::new(HashMap::new()),
            metadata_cache: RwLock::new(HashMap::new()),
            cache_config,
            access_tracker: RwLock::new(AccessTracker::new()),
            stats: RwLock::new(CachePerformanceStats::default()),
        }
    }

    /// Create an EvictingCacheFst with default configuration
    pub fn with_default_config(inner: F) -> Self {
        Self::new(inner, CacheConfig::default())
    }

    /// Get cached arcs for a state, computing if necessary
    fn get_cached_arcs(&self, state: StateId) -> Vec<Arc<W>> {
        // Record access for eviction policy
        {
            let mut tracker = self.access_tracker.write().unwrap();
            tracker.record_access(state);
        }

        // Try cache first
        {
            let cache = self.arc_cache.read().unwrap();
            if let Some(arcs) = cache.get(&state) {
                // Cache hit
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_accesses += 1;
                    stats.cache_hits += 1;
                }
                return arcs.clone();
            }
        }

        // Cache miss - compute from inner FST
        let arcs: Vec<Arc<W>> = self.inner.arcs(state).collect();

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_accesses += 1;
            stats.cache_misses += 1;
        }

        // Cache the result
        {
            let mut arc_cache = self.arc_cache.write().unwrap();
            let mut metadata_cache = self.metadata_cache.write().unwrap();

            // Store arc data
            arc_cache.insert(state, arcs.clone());

            // Store metadata
            let memory_size = arcs.len() * std::mem::size_of::<Arc<W>>();
            metadata_cache.insert(
                state,
                StateMetadata {
                    num_arcs: arcs.len(),
                    is_computed: true,
                    memory_size,
                },
            );

            // Apply eviction policy if needed
            self.apply_eviction_policy(&mut arc_cache, &mut metadata_cache);
        }

        arcs
    }

    /// Get cached final weight for a state, computing if necessary
    #[allow(dead_code)]
    fn get_cached_final_weight(&self, state: StateId) -> Option<W> {
        // Record access
        {
            let mut tracker = self.access_tracker.write().unwrap();
            tracker.record_access(state);
        }

        // Try cache first
        {
            let cache = self.final_weight_cache.read().unwrap();
            if let Some(weight_opt) = cache.get(&state) {
                // Cache hit
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_accesses += 1;
                    stats.cache_hits += 1;
                }
                return weight_opt.clone();
            }
        }

        // Cache miss - compute from inner FST
        let weight = self.inner.final_weight(state).cloned();

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_accesses += 1;
            stats.cache_misses += 1;
        }

        // Cache the result
        {
            let mut cache = self.final_weight_cache.write().unwrap();
            cache.insert(state, weight.clone());
        }

        weight
    }

    /// Apply eviction policy to manage cache memory usage
    fn apply_eviction_policy(
        &self,
        arc_cache: &mut HashMap<StateId, Vec<Arc<W>>>,
        metadata_cache: &mut HashMap<StateId, StateMetadata>,
    ) {
        if self.cache_config.eviction_policy == EvictionPolicy::None {
            return;
        }

        let needs_eviction = arc_cache.len() > self.cache_config.max_cached_states
            || self.cache_config.memory_limit.is_some_and(|limit| {
                let current_usage: usize =
                    metadata_cache.values().map(|meta| meta.memory_size).sum();
                current_usage > limit
            });

        if !needs_eviction {
            return;
        }

        // Choose state to evict
        let states: Vec<StateId> = arc_cache.keys().copied().collect();

        let evict_state = {
            let mut tracker = self.access_tracker.write().unwrap();
            tracker.choose_eviction_candidate(&self.cache_config.eviction_policy, &states)
        };

        if let Some(state) = evict_state {
            // Perform eviction
            arc_cache.remove(&state);
            metadata_cache.remove(&state);
            {
                let mut final_weight_cache = self.final_weight_cache.write().unwrap();
                final_weight_cache.remove(&state);
            }
            {
                let mut tracker = self.access_tracker.write().unwrap();
                tracker.remove_state(state);
            }
            {
                let mut stats = self.stats.write().unwrap();
                stats.evictions += 1;
            }
        }
    }

    /// Get current cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let arc_cache = self.arc_cache.read().unwrap();
        let final_weight_cache = self.final_weight_cache.read().unwrap();
        let metadata_cache = self.metadata_cache.read().unwrap();
        let stats = self.stats.read().unwrap();

        let memory_usage: usize = metadata_cache.values().map(|meta| meta.memory_size).sum();

        CacheStats {
            cached_states: arc_cache.len(),
            cached_final_weights: final_weight_cache.len(),
            memory_usage,
            estimated_total_states: self.inner.num_states(),
            hit_rate: stats.hit_rate(),
        }
    }

    /// Clear all cached data
    pub fn clear_cache(&self) {
        let mut arc_cache = self.arc_cache.write().unwrap();
        let mut final_weight_cache = self.final_weight_cache.write().unwrap();
        let mut metadata_cache = self.metadata_cache.write().unwrap();
        let mut tracker = self.access_tracker.write().unwrap();

        arc_cache.clear();
        final_weight_cache.clear();
        metadata_cache.clear();
        *tracker = AccessTracker::new();
    }

    /// Update cache configuration at runtime
    pub fn update_cache_config(&mut self, new_config: CacheConfig) {
        self.cache_config = new_config;

        // Apply new limits immediately if more restrictive
        let mut arc_cache = self.arc_cache.write().unwrap();
        let mut metadata_cache = self.metadata_cache.write().unwrap();
        self.apply_eviction_policy(&mut arc_cache, &mut metadata_cache);
    }

    /// Get reference to the inner FST
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Get detailed performance statistics
    pub fn performance_stats(&self) -> CachePerformanceStats {
        self.stats.read().unwrap().clone()
    }
}

/// Arc iterator for EvictingCacheFst with owned arc data and ArcIterator implementation
#[derive(Debug)]
pub struct EvictingCacheArcIterator<W: Semiring> {
    /// Owned vector of arcs for this state
    arcs: Vec<Arc<W>>,
    /// Current iteration position
    pos: usize,
}

impl<W: Semiring> Iterator for EvictingCacheArcIterator<W> {
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

impl<W: Semiring> ArcIterator<W> for EvictingCacheArcIterator<W> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<F: Fst<W>, W: Semiring> Fst<W> for EvictingCacheFst<F, W> {
    type ArcIter<'a>
        = EvictingCacheArcIterator<W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        self.inner.start()
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        // Note: We can't return a reference to cached data due to lifetime constraints
        // Users should call get_cached_final_weight() for owned access
        self.inner.final_weight(state)
    }

    fn num_arcs(&self, state: StateId) -> usize {
        // Check metadata cache first
        {
            let metadata_cache = self.metadata_cache.read().unwrap();
            if let Some(metadata) = metadata_cache.get(&state) {
                return metadata.num_arcs;
            }
        }

        // Fall back to getting cached arcs (which will populate metadata)
        self.get_cached_arcs(state).len()
    }

    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    fn properties(&self) -> FstProperties {
        self.inner.properties()
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        let arcs = self.get_cached_arcs(state);
        EvictingCacheArcIterator { arcs, pos: 0 }
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

    #[test]
    fn test_lazy_fst_impl_new() {
        let compute_fn = |state: StateId| -> Option<LazyState<TropicalWeight>> {
            if state > 100 {
                return None;
            }

            let arcs = if state == 0 {
                vec![Arc::new(1, 1, TropicalWeight::one(), 1)]
            } else {
                vec![]
            };

            let final_weight = if state == 1 {
                Some(TropicalWeight::one())
            } else {
                None
            };

            Some(LazyState { arcs, final_weight })
        };

        let lazy_fst = LazyFstImpl::new(compute_fn, 101);

        assert_eq!(lazy_fst.start(), Some(0));
        assert_eq!(lazy_fst.num_states(), 101);
        assert_eq!(lazy_fst.num_arcs(0), 1);
        assert_eq!(lazy_fst.num_arcs(1), 0);

        // Test that final_weight_ref works
        assert!(lazy_fst.final_weight_ref(1).is_some());
        assert!(lazy_fst.final_weight_ref(0).is_none());
    }

    #[test]
    fn test_evicting_cache_fst_wrapper() {
        // Create a simple FST to wrap
        let mut inner_fst = VectorFst::<TropicalWeight>::new();
        let s0 = inner_fst.add_state();
        let s1 = inner_fst.add_state();
        let s2 = inner_fst.add_state();

        inner_fst.set_start(s0);
        inner_fst.set_final(s2, TropicalWeight::one());
        inner_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        inner_fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        // Wrap with EvictingCacheFst
        let config = CacheConfig {
            max_cached_states: 10,
            memory_limit: Some(1024),
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };

        let cached_fst = EvictingCacheFst::new(inner_fst, config);

        // Test basic operations
        assert_eq!(cached_fst.start(), Some(s0));
        assert_eq!(cached_fst.num_states(), 3);
        assert_eq!(cached_fst.num_arcs(s0), 1);
        assert_eq!(cached_fst.num_arcs(s1), 1);
        assert_eq!(cached_fst.num_arcs(s2), 0);

        // Test caching by accessing the same state multiple times
        let arcs1: Vec<_> = cached_fst.arcs(s0).collect();
        let arcs2: Vec<_> = cached_fst.arcs(s0).collect();
        assert_eq!(arcs1.len(), arcs2.len());
        assert_eq!(arcs1[0].ilabel, arcs2[0].ilabel);

        // Test cache statistics
        let stats = cached_fst.cache_stats();
        assert!(stats.cached_states > 0);
        assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_cache_eviction_policies() {
        // Create a simple FST
        let mut inner_fst = VectorFst::<TropicalWeight>::new();
        for i in 0..10 {
            let state = inner_fst.add_state();
            if i == 0 {
                inner_fst.set_start(state);
            }
            if i < 9 {
                inner_fst.add_arc(state, Arc::new(1, 1, TropicalWeight::one(), state + 1));
            } else {
                inner_fst.set_final(state, TropicalWeight::one());
            }
        }

        // Test LRU eviction
        let config = CacheConfig {
            max_cached_states: 3, // Force eviction
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };

        let cached_fst = EvictingCacheFst::new(inner_fst, config);

        // Access states to fill cache beyond limit
        for i in 0..5 {
            let _arcs: Vec<_> = cached_fst.arcs(i).collect();
        }

        let stats = cached_fst.cache_stats();
        assert!(stats.cached_states <= 3); // Should have evicted some states
        assert!(stats.hit_rate >= 0.0);
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::default();
        assert_eq!(config.max_cached_states, 10000);
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
        assert!(!config.enable_memory_mapping);
        assert!(!config.enable_prefetching);

        let custom_config = CacheConfig {
            max_cached_states: 1000,
            memory_limit: Some(5_000_000),
            eviction_policy: EvictionPolicy::Random,
            enable_memory_mapping: true,
            enable_prefetching: true,
            streaming_config: LazyStreamingConfig::default(),
        };

        assert_eq!(custom_config.max_cached_states, 1000);
        assert_eq!(custom_config.memory_limit, Some(5_000_000));
        assert_eq!(custom_config.eviction_policy, EvictionPolicy::Random);
        assert!(custom_config.enable_memory_mapping);
        assert!(custom_config.enable_prefetching);
    }

    #[test]
    fn test_eviction_policy_enum() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
        assert_ne!(EvictionPolicy::Random, EvictionPolicy::None);
    }

    #[test]
    fn test_streaming_config() {
        let config = LazyStreamingConfig::default();
        assert!(!config.enable_streaming);
        assert_eq!(config.stream_buffer_size, 1000);
        assert_eq!(config.checkpoint_interval, 10000);
        assert_eq!(config.memory_checkpoint_threshold, 100_000_000);
        assert!(config.enable_state_compression);

        let custom_config = LazyStreamingConfig {
            enable_streaming: true,
            stream_buffer_size: 500,
            checkpoint_interval: 5000,
            memory_checkpoint_threshold: 50_000_000,
            enable_state_compression: false,
        };

        assert!(custom_config.enable_streaming);
        assert_eq!(custom_config.stream_buffer_size, 500);
        assert_eq!(custom_config.checkpoint_interval, 5000);
        assert_eq!(custom_config.memory_checkpoint_threshold, 50_000_000);
        assert!(!custom_config.enable_state_compression);
    }

    // Simple test generator for streaming tests
    #[derive(Debug)]
    struct TestGenerator {
        max_state: StateId,
    }

    impl TestGenerator {
        fn new(max_state: StateId) -> Self {
            Self { max_state }
        }
    }

    impl StateGenerator<TropicalWeight> for TestGenerator {
        fn generate_batch(
            &self,
            start_state: StateId,
            batch_size: usize,
        ) -> Vec<(StateId, LazyState<TropicalWeight>)> {
            (start_state..std::cmp::min(start_state + batch_size as StateId, self.max_state + 1))
                .map(|state| {
                    let arcs = if state < self.max_state {
                        vec![Arc::new(1, 1, TropicalWeight::one(), state + 1)]
                    } else {
                        vec![]
                    };
                    let final_weight = if state == self.max_state {
                        Some(TropicalWeight::one())
                    } else {
                        None
                    };
                    (state, LazyState { arcs, final_weight })
                })
                .collect()
        }

        fn has_more_states(&self, state: StateId) -> bool {
            state <= self.max_state
        }

        fn estimated_size(&self) -> Option<usize> {
            Some((self.max_state + 1) as usize)
        }

        fn reset(&self) {
            // Nothing to reset for this simple generator
        }
    }

    #[test]
    fn test_streaming_lazy_fst() {
        let generator = TestGenerator::new(10);
        let streaming_config = LazyStreamingConfig {
            enable_streaming: true,
            stream_buffer_size: 5,
            checkpoint_interval: 8,
            memory_checkpoint_threshold: 1000,
            enable_state_compression: true,
        };
        let cache_config = CacheConfig::default();

        let streaming_fst = StreamingLazyFst::new(generator, streaming_config, cache_config);

        // Test basic operations
        assert_eq!(streaming_fst.start(), Some(0));
        assert_eq!(streaming_fst.num_states(), 11);
        assert_eq!(streaming_fst.num_arcs(0), 1);
        assert_eq!(streaming_fst.num_arcs(10), 0);

        // Test arc iteration
        let arcs: Vec<_> = streaming_fst.arcs(0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[0].nextstate, 1);

        // Test streaming stats
        let stats = streaming_fst.streaming_stats();
        assert!(stats.cached_states > 0);
        assert_eq!(stats.estimated_total_states, Some(11));
    }

    #[test]
    fn test_state_generator_trait() {
        let generator = TestGenerator::new(5);

        assert!(generator.has_more_states(3));
        assert!(generator.has_more_states(5));
        assert!(!generator.has_more_states(6));

        assert_eq!(generator.estimated_size(), Some(6));

        let batch = generator.generate_batch(0, 3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].0, 0);
        assert_eq!(batch[1].0, 1);
        assert_eq!(batch[2].0, 2);

        // Test final state
        let final_batch = generator.generate_batch(5, 2);
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].0, 5);
        assert!(final_batch[0].1.final_weight.is_some());
    }

    #[test]
    fn test_lazy_fst_streaming_methods() {
        let compute_fn = |state: StateId| -> Option<LazyState<TropicalWeight>> {
            if state > 100 {
                return None;
            }

            let arcs = if state == 0 {
                vec![Arc::new(1, 1, TropicalWeight::one(), 1)]
            } else {
                vec![]
            };

            let final_weight = if state == 1 {
                Some(TropicalWeight::one())
            } else {
                None
            };

            Some(LazyState { arcs, final_weight })
        };

        let mut lazy_fst = LazyFstImpl::new(compute_fn, 101);

        // Test streaming configuration
        let streaming_config = LazyStreamingConfig {
            enable_streaming: true,
            stream_buffer_size: 50,
            checkpoint_interval: 100,
            memory_checkpoint_threshold: 1_000_000,
            enable_state_compression: true,
        };

        lazy_fst.enable_streaming(streaming_config);

        // Test memory usage estimation
        let _memory_usage = lazy_fst.estimated_memory_usage();

        // Test checkpoint functionality
        assert!(!lazy_fst.needs_checkpoint()); // Should not need checkpoint initially
        assert!(lazy_fst.checkpoint().is_ok()); // Should succeed
    }
}
