//! Arc types and traits

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

/// Iterator over arcs
pub trait ArcIterator<W: Semiring>: Iterator<Item = Arc<W>> {
    /// Reset the iterator
    fn reset(&mut self) {
        // default implementation does nothing
    }
}
