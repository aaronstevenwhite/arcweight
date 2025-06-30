//! Core FST traits

use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use crate::Result;
use core::fmt::Debug;

/// State identifier type
pub type StateId = u32;

/// Label type for input/output symbols
pub type Label = u32;

/// Special state ID for non-existent state
pub const NO_STATE_ID: StateId = u32::MAX;

/// Special label for epsilon transitions
pub const NO_LABEL: Label = 0;

/// Core trait for all FST types providing read-only access
///
/// The `Fst` trait defines the fundamental interface for weighted finite state transducers.
/// All FST implementations must provide these operations for querying structure and weights.
/// This trait is implemented by all FST types in ArcWeight, including [`VectorFst`], [`ConstFst`],
/// and [`CacheFst`].
///
/// # Implementation Guidelines
///
/// When implementing this trait:
/// - Ensure `start()` returns `None` for empty FSTs
/// - `final_weight()` should return `None` for non-final states
/// - Arc iterators must be valid for the lifetime of the FST
/// - States are numbered from 0 to `num_states() - 1`
/// - Label 0 is reserved for epsilon transitions
///
/// # Thread Safety
///
/// All FST implementations are required to be `Send + Sync`, making them safe to share
/// between threads for read-only operations.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn analyze_fst<W: Semiring>(fst: &impl Fst<W>) {
///     if let Some(start) = fst.start() {
///         println!("FST has {} states", fst.num_states());
///         println!("Start state: {}", start);
///         
///         // Iterate over arcs from start state
///         for arc in fst.arcs(start) {
///             println!("Arc: {} -> {} / {} : {}",
///                      arc.ilabel, arc.olabel, arc.weight, arc.nextstate);
///         }
///         
///         // Check if start state is final
///         if let Some(weight) = fst.final_weight(start) {
///             println!("Start state is final with weight: {}", weight);
///         }
///     } else {
///         println!("Empty FST");
///     }
/// }
/// ```
///
/// # See Also
///
/// - [`MutableFst`] for FSTs that can be modified
/// - [`ExpandedFst`] for FSTs with all arcs in memory
/// - [Core Concepts](../../docs/core-concepts/fsts.md) for FST theory
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for FST manipulation
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`ConstFst`]: crate::fst::ConstFst
/// [`CacheFst`]: crate::fst::CacheFst
pub trait Fst<W: Semiring>: Debug + Send + Sync {
    /// Arc iterator type
    type ArcIter<'a>: ArcIterator<W>
    where
        Self: 'a;

    /// Get the start state
    fn start(&self) -> Option<StateId>;

    /// Get the final weight of a state
    fn final_weight(&self, state: StateId) -> Option<&W>;

    /// Check if a state is final
    fn is_final(&self, state: StateId) -> bool {
        self.final_weight(state).is_some()
    }

    /// Get the number of arcs from a state
    fn num_arcs(&self, state: StateId) -> usize;

    /// Get the number of states
    fn num_states(&self) -> usize;

    /// Get properties of the FST
    fn properties(&self) -> FstProperties;

    /// Create an iterator over arcs from a state
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_>;

    /// Iterate over all states
    fn states(&self) -> impl Iterator<Item = StateId> {
        0..self.num_states() as StateId
    }

    /// Get the total number of arcs
    fn num_arcs_total(&self) -> usize {
        self.states().map(|s| self.num_arcs(s)).sum()
    }

    /// Check if FST is empty
    fn is_empty(&self) -> bool {
        self.start().is_none() || self.num_states() == 0
    }
}

/// Trait for FSTs that can be modified
///
/// The `MutableFst` trait extends [`Fst`] with operations for modifying FST structure.
/// This includes adding states and arcs, setting start and final states, and managing
/// the FST's topology. All mutable FST types implement this trait.
///
/// # Implementation Guidelines
///
/// When implementing this trait:
/// - `add_state()` should return a unique state ID
/// - State IDs should be consecutive starting from 0
/// - `set_final()` with `W::zero()` removes final status
/// - `clear()` should reset the FST to an empty state
/// - Memory reservations are hints for optimization
///
/// # Performance Notes
///
/// - Use `reserve_states()` and `reserve_arcs()` when the final size is known
/// - Batch operations when possible to reduce memory reallocations
/// - `clear()` is more efficient than creating a new FST
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn build_simple_fst() -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     
///     // Add states
///     let s0 = fst.add_state();
///     let s1 = fst.add_state();
///     
///     // Set start state
///     fst.set_start(s0);
///     
///     // Add an arc: 'a' -> 'b' with weight 1.5
///     fst.add_arc(s0, Arc::new(
///         'a' as u32,
///         'b' as u32,
///         TropicalWeight::new(1.5),
///         s1
///     ));
///     
///     // Make s1 final with weight 0.0
///     fst.set_final(s1, TropicalWeight::one());
///     
///     fst
/// }
/// ```
///
/// ## Efficient Construction
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn build_large_fst(words: &[&str]) -> VectorFst<BooleanWeight> {
///     let mut fst = VectorFst::new();
///     
///     // Reserve space for better performance
///     fst.reserve_states(words.len() * 10); // Estimate
///     
///     let start = fst.add_state();
///     fst.set_start(start);
///     
///     for word in words {
///         let mut current = start;
///         for ch in word.chars() {
///             let next = fst.add_state();
///             fst.add_arc(current, Arc::new(
///                 ch as u32, ch as u32, BooleanWeight::one(), next
///             ));
///             current = next;
///         }
///         fst.set_final(current, BooleanWeight::one());
///     }
///     
///     fst
/// }
/// ```
///
/// # See Also
///
/// - [`Fst`] for read-only operations
/// - [`VectorFst`] for the main mutable implementation
/// - [Quick Start Guide](../../docs/quick-start.md) for construction examples
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for building patterns
///
/// [`VectorFst`]: crate::fst::VectorFst
pub trait MutableFst<W: Semiring>: Fst<W> {
    /// Add a new state
    fn add_state(&mut self) -> StateId;

    /// Add an arc
    fn add_arc(&mut self, state: StateId, arc: Arc<W>);

    /// Set the start state
    fn set_start(&mut self, state: StateId);

    /// Set final weight for a state
    fn set_final(&mut self, state: StateId, weight: W);

    /// Remove final weight from a state
    fn remove_final(&mut self, state: StateId) {
        self.set_final(state, W::zero());
    }

    /// Delete all arcs from a state
    fn delete_arcs(&mut self, state: StateId);

    /// Delete a single arc
    fn delete_arc(&mut self, state: StateId, arc_idx: usize);

    /// Reserve space for states
    fn reserve_states(&mut self, n: usize);

    /// Reserve space for arcs from a state
    fn reserve_arcs(&mut self, state: StateId, n: usize);

    /// Clear the FST
    fn clear(&mut self);
}

/// Trait for FSTs with all states and arcs in memory
///
/// The `ExpandedFst` trait is implemented by FST types that store all their arcs
/// in memory, allowing direct access to arc slices. This enables more efficient
/// algorithms that can work with arc arrays directly rather than iterators.
///
/// # Performance Benefits
///
/// - Direct memory access to arcs (no iterator overhead)
/// - Better cache locality for arc traversal
/// - Enables vectorized operations on arc arrays
/// - More efficient for algorithms that need multiple passes over arcs
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn count_epsilon_arcs<W: Semiring>(fst: &impl ExpandedFst<W>) -> usize {
///     let mut count = 0;
///     for state in fst.states() {
///         let arcs = fst.arcs_slice(state);
///         count += arcs.iter()
///             .filter(|arc| arc.ilabel == 0)  // Epsilon arcs
///             .count();
///     }
///     count
/// }
/// ```
///
/// # See Also
///
/// - [`VectorFst`] implements this trait
/// - [`ConstFst`] implements this trait
/// - [`CacheFst`] does NOT implement this (arcs computed on-demand)
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`ConstFst`]: crate::fst::ConstFst
/// [`CacheFst`]: crate::fst::CacheFst
pub trait ExpandedFst<W: Semiring>: Fst<W> {
    /// Get a slice of arcs from a state
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>];
}

/// Trait for FSTs computed on-demand
pub trait LazyFst<W: Semiring>: Fst<W> {
    /// Expand a state (compute its arcs)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - State computation fails due to invalid state data
    /// - Memory allocation fails during arc expansion
    /// - The underlying FST computation encounters an error
    fn expand(&self, state: StateId) -> Result<()>;
}
