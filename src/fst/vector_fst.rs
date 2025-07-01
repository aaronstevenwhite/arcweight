//! Vector-based FST implementation

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::{compute_properties, FstProperties};
use crate::semiring::Semiring;
use core::slice;

/// State data in vector FST
#[derive(Debug, Clone)]
struct VectorState<W: Semiring> {
    /// Final weight (None if not final)
    final_weight: Option<W>,
    /// Outgoing arcs
    arcs: Vec<Arc<W>>,
}

impl<W: Semiring> Default for VectorState<W> {
    fn default() -> Self {
        Self {
            final_weight: None,
            arcs: Vec::new(),
        }
    }
}

/// Vector-based mutable FST implementation optimized for construction and modification
///
/// `VectorFst` is the primary mutable FST implementation in ArcWeight, storing states
/// and arcs in dynamically resizable vectors. This provides excellent performance for
/// FST construction, modification, and random access to states and arcs.
///
/// # Design Characteristics
///
/// - **Mutability:** Full support for adding/removing states and arcs
/// - **Memory Layout:** Contiguous vector storage for cache-friendly access
/// - **Random Access:** O(1) access to any state or arc by index
/// - **Dynamic Growth:** Automatically resizes as states and arcs are added
/// - **Type Safety:** Parameterized by semiring type for compile-time weight verification
///
/// # Performance Profile
///
/// | Operation | Time Complexity | Notes |
/// |-----------|----------------|-------|
/// | Add State | O(1) amortized | May require vector reallocation |
/// | Add Arc | O(1) amortized | Appends to state's arc vector |
/// | State Access | O(1) | Direct vector indexing |
/// | Arc Iteration | O(k) | k = number of arcs from state |
/// | Property Computation | O(V + E) | Full graph traversal |
///
/// # Memory Characteristics
///
/// - **State Storage:** `Vec<VectorState<W>>` with 24 bytes overhead per state
/// - **Arc Storage:** `Vec<Arc<W>>` per state, ~32 bytes per arc
/// - **Growth Strategy:** Exponential growth for amortized O(1) insertion
/// - **Memory Efficiency:** Suitable for FSTs up to millions of states
///
/// # Use Cases
///
/// ## FST Construction
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build a simple word acceptor for "hello"
/// fn build_word_acceptor(word: &str) -> VectorFst<BooleanWeight> {
///     let mut fst = VectorFst::new();
///     
///     // Create state chain
///     let mut states = Vec::new();
///     for _ in 0..=word.len() {
///         states.push(fst.add_state());
///     }
///     
///     fst.set_start(states[0]);
///     fst.set_final(states[word.len()], BooleanWeight::one());
///     
///     // Add character transitions
///     for (i, ch) in word.chars().enumerate() {
///         fst.add_arc(states[i], Arc::new(
///             ch as u32, ch as u32, BooleanWeight::one(), states[i + 1]
///         ));
///     }
///     
///     fst
/// }
///
/// let hello_fst = build_word_acceptor("hello");
/// assert_eq!(hello_fst.num_states(), 6);
/// ```
///
/// ## Weighted Transduction
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build pronunciation dictionary: orthography -> phonemes
/// fn build_pronunciation_entry(
///     orthography: &str,
///     phonemes: &str,
///     frequency: f32
/// ) -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     
///     let mut current = fst.add_state();
///     fst.set_start(current);
///     
///     // Input: orthographic characters
///     for ch in orthography.chars() {
///         let next = fst.add_state();
///         fst.add_arc(current, Arc::new(
///             ch as u32, 0, // Input char, epsilon output
///             TropicalWeight::new(-frequency.ln()), // Negative log frequency
///             next
///         ));
///         current = next;
///     }
///     
///     // Output: phonemic sequence
///     for ph in phonemes.chars() {
///         let next = fst.add_state();
///         fst.add_arc(current, Arc::new(
///             0, ph as u32, // Epsilon input, phoneme output
///             TropicalWeight::one(),
///             next
///         ));
///         current = next;
///     }
///     
///     fst.set_final(current, TropicalWeight::one());
///     fst
/// }
/// ```
///
/// ## Dynamic FST Modification
/// ```rust
/// use arcweight::prelude::*;
///
/// // Incrementally build vocabulary FST
/// fn build_vocabulary_incrementally(words: &[&str]) -> VectorFst<BooleanWeight> {
///     let mut fst = VectorFst::new();
///     let root = fst.add_state();
///     fst.set_start(root);
///     
///     for word in words {
///         // Add word to existing trie structure
///         add_word_to_trie(&mut fst, root, word);
///     }
///     
///     fst
/// }
///
/// fn add_word_to_trie(fst: &mut VectorFst<BooleanWeight>, mut current: u32, word: &str) {
///     for ch in word.chars() {
///         // Find existing arc or create new path
///         let label = ch as u32;
///         
///         if let Some(next) = find_arc_target(fst, current, label) {
///             current = next;
///         } else {
///             let next = fst.add_state();
///             fst.add_arc(current, Arc::new(label, label, BooleanWeight::one(), next));
///             current = next;
///         }
///     }
///     fst.set_final(current, BooleanWeight::one());
/// }
///
/// fn find_arc_target(fst: &VectorFst<BooleanWeight>, state: u32, label: u32) -> Option<u32> {
///     fst.arcs(state).find(|arc| arc.ilabel == label).map(|arc| arc.nextstate)
/// }
/// ```
///
/// # Optimization Considerations
///
/// ## Memory Pre-allocation
/// ```rust
/// use arcweight::prelude::*;
///
/// // Pre-allocate for known size to avoid reallocations
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// fst.reserve_states(1000);  // Reserve space for 1000 states
///
/// let state = fst.add_state();
/// fst.reserve_arcs(state, 50);  // Reserve space for 50 arcs from this state
/// ```
///
/// ## Batch Operations
/// ```rust
/// use arcweight::prelude::*;
///
/// // Batch state creation for better performance
/// fn build_linear_chain(length: usize) -> VectorFst<BooleanWeight> {
///     let mut fst = VectorFst::new();
///     fst.reserve_states(length + 1);
///     
///     // Create all states at once
///     let states: Vec<_> = (0..=length).map(|_| fst.add_state()).collect();
///     
///     fst.set_start(states[0]);
///     fst.set_final(states[length], BooleanWeight::one());
///     
///     // Add transitions
///     for i in 0..length {
///         fst.add_arc(states[i], Arc::new(
///             (i + 1) as u32, (i + 1) as u32, BooleanWeight::one(), states[i + 1]
///         ));
///     }
///     
///     fst
/// }
/// ```
///
/// # Thread Safety
///
/// `VectorFst` is `Send + Sync` when the semiring type is `Send + Sync`, enabling:
/// - **Parallel Construction:** Build different FSTs in parallel threads
/// - **Read-Only Sharing:** Share immutable references across threads
/// - **Algorithm Parallelization:** Use in parallel FST algorithms
///
/// Note: Mutation requires exclusive access (`&mut self`), preventing data races.
///
/// # See Also
///
/// - `ConstFst` for read-only memory-optimized FSTs
/// - `CacheFst` for lazy evaluation and large FSTs
/// - [Quick Start Guide](../../docs/quick-start.md) for construction examples
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for manipulation patterns
#[derive(Debug, Clone)]
pub struct VectorFst<W: Semiring> {
    states: Vec<VectorState<W>>,
    start: Option<StateId>,
    properties: FstProperties,
}

impl<W: Semiring> VectorFst<W> {
    /// Create a new empty FST
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let fst = VectorFst::<TropicalWeight>::new();
    /// assert_eq!(fst.num_states(), 0);
    /// assert!(fst.start().is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            start: None,
            properties: FstProperties::default(),
        }
    }

    /// Create with capacity
    ///
    /// Pre-allocates space for the specified number of states to avoid
    /// reallocations during FST construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::with_capacity(100);
    ///
    /// // Add many states efficiently
    /// for _ in 0..100 {
    ///     fst.add_state();
    /// }
    ///
    /// assert_eq!(fst.num_states(), 100);
    /// ```
    pub fn with_capacity(states: usize) -> Self {
        Self {
            states: Vec::with_capacity(states),
            start: None,
            properties: FstProperties::default(),
        }
    }

    /// Compute and cache properties
    pub fn compute_properties(&mut self) {
        self.properties = compute_properties(self);
    }
}

impl<W: Semiring> Default for VectorFst<W> {
    fn default() -> Self {
        Self::new()
    }
}

/// Arc iterator for VectorFst
#[derive(Debug)]
pub struct VectorArcIterator<'a, W: Semiring> {
    arcs: slice::Iter<'a, Arc<W>>,
}

impl<W: Semiring> Iterator for VectorArcIterator<'_, W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        self.arcs.next().cloned()
    }
}

impl<W: Semiring> ArcIterator<W> for VectorArcIterator<'_, W> {}

impl<W: Semiring> Fst<W> for VectorFst<W> {
    type ArcIter<'a>
        = VectorArcIterator<'a, W>
    where
        W: 'a;

    fn start(&self) -> Option<StateId> {
        self.start
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        self.states
            .get(state as usize)
            .and_then(|s| s.final_weight.as_ref())
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.states
            .get(state as usize)
            .map(|s| s.arcs.len())
            .unwrap_or(0)
    }

    fn num_states(&self) -> usize {
        self.states.len()
    }

    fn properties(&self) -> FstProperties {
        // If properties are not computed, compute them
        if self.properties.known.is_empty() {
            compute_properties(self)
        } else {
            self.properties
        }
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        let arcs = self
            .states
            .get(state as usize)
            .map(|s| s.arcs.iter())
            .unwrap_or_else(|| [].iter());
        VectorArcIterator { arcs }
    }
}

impl<W: Semiring> MutableFst<W> for VectorFst<W> {
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    ///
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    ///
    /// assert_eq!(s0, 0);
    /// assert_eq!(s1, 1);
    /// assert_eq!(fst.num_states(), 2);
    /// ```
    fn add_state(&mut self) -> StateId {
        let id = self.states.len() as StateId;
        self.states.push(VectorState::default());
        self.properties.invalidate_all();
        id
    }

    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    ///
    /// // Add arc: input=1, output=1, weight=0.5, target=s1
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    ///
    /// assert_eq!(fst.num_arcs(s0), 1);
    /// ```
    fn add_arc(&mut self, state: StateId, arc: Arc<W>) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.arcs.push(arc);
            self.properties.invalidate_all();
        }
    }

    fn set_start(&mut self, state: StateId) {
        self.start = Some(state);
        self.properties.invalidate_all();
    }

    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    ///
    /// // Make s1 a final state with weight 0.8
    /// fst.set_final(s1, TropicalWeight::new(0.8));
    ///
    /// assert!(fst.is_final(s1));
    /// assert_eq!(fst.final_weight(s1), Some(&TropicalWeight::new(0.8)));
    /// ```
    fn set_final(&mut self, state: StateId, weight: W) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.final_weight = if <W as num_traits::Zero>::is_zero(&weight) {
                None
            } else {
                Some(weight)
            };
            self.properties.invalidate_all();
        }
    }

    fn delete_arcs(&mut self, state: StateId) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.arcs.clear();
            self.properties.invalidate_all();
        }
    }

    fn delete_arc(&mut self, state: StateId, arc_idx: usize) {
        if let Some(s) = self.states.get_mut(state as usize) {
            if arc_idx < s.arcs.len() {
                s.arcs.remove(arc_idx);
                self.properties.invalidate_all();
            }
        }
    }

    fn reserve_states(&mut self, n: usize) {
        self.states.reserve(n);
    }

    fn reserve_arcs(&mut self, state: StateId, n: usize) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.arcs.reserve(n);
        }
    }

    fn clear(&mut self) {
        self.states.clear();
        self.start = None;
        self.properties = FstProperties::default();
    }
}

impl<W: Semiring> ExpandedFst<W> for VectorFst<W> {
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>] {
        self.states
            .get(state as usize)
            .map(|s| s.arcs.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::TropicalWeight;
    use num_traits::One;

    #[test]
    fn test_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        assert_eq!(fst.num_states(), 0);
        assert!(fst.is_empty());
        assert_eq!(fst.start(), None);
        assert_eq!(fst.num_arcs_total(), 0);
    }

    #[test]
    fn test_add_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();

        let s0 = fst.add_state();
        let s1 = fst.add_state();

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(fst.num_states(), 2);

        // FST is considered empty until start state is set
        fst.set_start(s0);
        assert!(!fst.is_empty());
    }

    #[test]
    fn test_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();

        assert_eq!(fst.start(), None);

        fst.set_start(s0);
        assert_eq!(fst.start(), Some(s0));
    }

    #[test]
    fn test_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        assert!(!fst.is_final(s0));
        assert!(!fst.is_final(s1));

        fst.set_final(s1, TropicalWeight::new(2.5));
        assert!(!fst.is_final(s0));
        assert!(fst.is_final(s1));
        assert_eq!(*fst.final_weight(s1).unwrap().value(), 2.5);

        fst.remove_final(s1);
        assert!(!fst.is_final(s1));
    }

    #[test]
    fn test_add_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        assert_eq!(fst.num_arcs(s0), 0);
        assert_eq!(fst.num_arcs(s1), 0);

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.5), s1));
        fst.add_arc(s0, Arc::new(3, 4, TropicalWeight::new(2.0), s1));

        assert_eq!(fst.num_arcs(s0), 2);
        assert_eq!(fst.num_arcs(s1), 0);
        assert_eq!(fst.num_arcs_total(), 2);
    }

    #[test]
    fn test_arc_iteration() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.5), s1));
        fst.add_arc(s0, Arc::new(3, 4, TropicalWeight::new(2.0), s1));

        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 2);

        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[0].olabel, 2);
        assert_eq!(*arcs[0].weight.value(), 1.5);
        assert_eq!(arcs[0].nextstate, s1);

        assert_eq!(arcs[1].ilabel, 3);
        assert_eq!(arcs[1].olabel, 4);
        assert_eq!(*arcs[1].weight.value(), 2.0);
        assert_eq!(arcs[1].nextstate, s1);
    }

    #[test]
    fn test_reserve_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        fst.reserve_states(100);

        // Add states and verify they are added efficiently
        for i in 0..100 {
            let state = fst.add_state();
            assert_eq!(state, i);
        }

        assert_eq!(fst.num_states(), 100);
    }

    #[test]
    fn test_reserve_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.reserve_arcs(s0, 50);

        // Add arcs and verify they are added efficiently
        for i in 0..50 {
            fst.add_arc(
                s0,
                Arc::new(i, i, TropicalWeight::new(i as f32), s1),
            );
        }

        assert_eq!(fst.num_arcs(s0), 50);
    }

    #[test]
    fn test_clear() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.5), s1));

        assert!(!fst.is_empty());
        assert_eq!(fst.num_states(), 2);

        fst.clear();

        assert!(fst.is_empty());
        assert_eq!(fst.num_states(), 0);
        assert_eq!(fst.start(), None);
    }

    #[test]
    fn test_state_iteration() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        let states: Vec<_> = fst.states().collect();
        assert_eq!(states, vec![s0, s1, s2]);
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_fst_state_consistency_property(num_states in 1..100usize) {
                let mut fst = VectorFst::<TropicalWeight>::new();

                for _ in 0..num_states {
                    fst.add_state();
                }

                assert_eq!(fst.num_states(), num_states);

                // all states should be valid
                for state in fst.states() {
                    assert!(state < num_states as StateId);
                }

                // Arc counts should be consistent
                let total_arcs = fst.num_arcs_total();
                let sum_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
                assert_eq!(total_arcs, sum_arcs);
            }
        }
    }

}
