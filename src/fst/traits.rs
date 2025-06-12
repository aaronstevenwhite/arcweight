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

/// Core trait for all FST types
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

/// Trait for FSTs with all states in memory
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
