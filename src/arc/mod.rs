//! Arc types and traits

use crate::fst::{Label, StateId};
use crate::semiring::Semiring;
use core::fmt;

/// Arc in a weighted FST
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
    pub fn new(ilabel: Label, olabel: Label, weight: W, nextstate: StateId) -> Self {
        Self {
            ilabel,
            olabel,
            weight,
            nextstate,
        }
    }

    /// Create an epsilon arc
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
