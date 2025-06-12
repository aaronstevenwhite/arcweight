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

/// Vector-based FST implementation
/// 
/// A mutable FST implementation that stores states and arcs in vectors,
/// providing efficient random access and modification operations.
/// 
/// # Examples
/// 
/// Creating a simple acceptor FST:
/// 
/// ```
/// use arcweight::prelude::*;
/// 
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// 
/// // Add states
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// 
/// // Set start and final states
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// 
/// // Add arcs (label, label, weight, target_state)
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
/// 
/// // The FST now accepts the sequence [1, 2] with total weight 0.8
/// assert_eq!(fst.num_states(), 3);
/// assert_eq!(fst.num_arcs(s0), 1);
/// ```
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
