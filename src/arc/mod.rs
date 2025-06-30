//! Arc types and traits for weighted FSTs
//!
//! This module provides the fundamental arc representation used throughout
//! the library. Arcs represent weighted transitions between states in a
//! finite state transducer (FST).
//!
//! ## Overview
//!
//! An [`Arc`] consists of four components:
//! - **Input label** (`ilabel`) - Symbol consumed from input tape
//! - **Output label** (`olabel`) - Symbol produced on output tape
//! - **Weight** - Cost or probability of taking the transition
//! - **Next state** - Target state ID
//!
//! ## Arc Types
//!
//! ### Regular Arcs
//! Standard transitions with explicit input/output labels:
//! ```
//! use arcweight::prelude::*;
//!
//! // Transducer arc: transforms input 1 to output 2
//! let arc = Arc::new(1, 2, TropicalWeight::new(0.5), 3);
//! ```
//!
//! ### Acceptor Arcs
//! When input and output labels match (common in acceptors):
//! ```
//! use arcweight::prelude::*;
//!
//! // Acceptor arc: consumes and produces symbol 1
//! let arc = Arc::new(1, 1, TropicalWeight::one(), 2);
//! ```
//!
//! ### Epsilon Arcs
//! Transitions that don't consume or produce symbols:
//! ```
//! use arcweight::prelude::*;
//!
//! // Epsilon arc: no input/output symbols
//! let arc = Arc::epsilon(TropicalWeight::new(0.1), 1);
//! assert!(arc.is_epsilon());
//! ```
//!
//! ## Arc Iterators
//!
//! The [`ArcIterator`] trait provides a common interface for iterating
//! over arcs from a state. Implementations can be found in FST types:
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let fst = VectorFst::<TropicalWeight>::new();
//! // FST types provide arcᵢter() methods returning ArcIterator
//! ```
//!
//! ## Usage in FST Construction
//!
//! Arcs are the building blocks for FST construction:
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//!
//! // Add arc from s0 to s1
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
//! ```

use crate::fst::{Label, StateId};
use crate::semiring::Semiring;
use core::fmt;

/// Arc in a weighted FST
///
/// Represents a weighted transition between states in an FST.
/// Each arc has input and output labels, a weight, and a target state.
///
/// # Examples
///
/// Creating different types of arcs:
///
/// ```
/// use arcweight::prelude::*;
///
/// // Regular arc with different input/output labels
/// let arc1 = Arc::new(1, 2, TropicalWeight::new(0.5), 3);
///
/// // Acceptor arc (same input/output)
/// let arc2 = Arc::new(1, 1, TropicalWeight::one(), 2);
///
/// // Epsilon arc (input=0, output=0)
/// let epsilon = Arc::epsilon(TropicalWeight::new(0.1), 1);
///
/// assert_eq!(arc1.ilabel, 1);
/// assert_eq!(arc1.olabel, 2);
/// assert!(epsilon.is_epsilon());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Arc<W: Semiring> {
    /// Input label
    pub ilabel: Label,
    /// Output label
    pub olabel: Label,
    /// Weight
    pub weight: W,
    /// Next state
    pub nextstate: StateId,
}

impl<W: Semiring> Arc<W> {
    /// Create a new arc
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let arc = Arc::new(1, 2, TropicalWeight::new(0.5), 3);
    ///
    /// assert_eq!(arc.ilabel, 1);
    /// assert_eq!(arc.olabel, 2);
    /// assert_eq!(arc.weight, TropicalWeight::new(0.5));
    /// assert_eq!(arc.nextstate, 3);
    /// ```
    pub fn new(ilabel: Label, olabel: Label, weight: W, nextstate: StateId) -> Self {
        Self {
            ilabel,
            olabel,
            weight,
            nextstate,
        }
    }

    /// Create an epsilon arc
    ///
    /// Epsilon arcs have input and output labels of 0, representing
    /// transitions that don't consume or produce symbols.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let epsilon = Arc::epsilon(TropicalWeight::one(), 2);
    ///
    /// assert_eq!(epsilon.ilabel, 0);
    /// assert_eq!(epsilon.olabel, 0);
    /// assert!(epsilon.is_epsilon());
    /// assert!(epsilon.is_epsilon_input());
    /// assert!(epsilon.is_epsilon_output());
    /// ```
    pub fn epsilon(weight: W, nextstate: StateId) -> Self {
        Self::new(0, 0, weight, nextstate)
    }

    /// Check if arc has epsilon input
    pub fn is_epsilon_input(&self) -> bool {
        self.ilabel == 0
    }

    /// Check if arc has epsilon output
    pub fn is_epsilon_output(&self) -> bool {
        self.olabel == 0
    }

    /// Check if arc is fully epsilon
    pub fn is_epsilon(&self) -> bool {
        self.is_epsilon_input() && self.is_epsilon_output()
    }
}

impl<W: Semiring> fmt::Display for Arc<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{} -> {}",
            self.ilabel, self.olabel, self.weight, self.nextstate
        )
    }
}

/// Iterator over arcs from a state
///
/// This trait provides a uniform interface for iterating over arcs
/// leaving a particular state in an FST. Different FST implementations
/// provide their own concrete iterator types implementing this trait.
///
/// ## Usage
///
/// Arc iterators are typically obtained from FST types:
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let state = fst.add_state();
/// fst.add_arc(state, Arc::new(1, 2, TropicalWeight::new(0.5), state));
///
/// // Get iterator over arcs from state
/// for arc in fst.arcs(state) {
///     println!("Arc: {} -> {}", arc.ilabel, arc.olabel);
/// }
/// ```
///
/// ## Implementation Notes
///
/// - Iterators may cache arcs internally for efficiency
/// - The `reset()` method allows reusing an iterator without reallocation
/// - Some implementations may lazily compute arcs during iteration
pub trait ArcIterator<W: Semiring>: Iterator<Item = Arc<W>> {
    /// Reset the iterator to the beginning
    ///
    /// This allows reusing the iterator without creating a new one,
    /// which can be more efficient for repeated iterations.
    ///
    /// The default implementation does nothing, which is suitable
    /// for iterators that don't maintain internal state.
    fn reset(&mut self) {
        // default implementation does nothing
    }
}
