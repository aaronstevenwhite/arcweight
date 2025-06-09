//! Vector-based FST implementation

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::semiring::Semiring;
use crate::properties::{FstProperties, compute_properties};
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
#[derive(Debug, Clone)]
pub struct VectorFst<W: Semiring> {
    states: Vec<VectorState<W>>,
    start: Option<StateId>,
    properties: FstProperties,
}

impl<W: Semiring> VectorFst<W> {
    /// Create a new empty FST
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            start: None,
            properties: FstProperties::default(),
        }
    }
    
    /// Create with capacity
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

impl<'a, W: Semiring> Iterator for VectorArcIterator<'a, W> {
    type Item = Arc<W>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.arcs.next().cloned()
    }
}

impl<'a, W: Semiring> ArcIterator<W> for VectorArcIterator<'a, W> {}

impl<W: Semiring> Fst<W> for VectorFst<W> {
    type ArcIter<'a> = VectorArcIterator<'a, W> where W: 'a;
    
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
        self.properties
    }
    
    fn arcs<'a>(&'a self, state: StateId) -> Self::ArcIter<'a> {
        let arcs = self.states
            .get(state as usize)
            .map(|s| s.arcs.iter())
            .unwrap_or_else(|| [].iter());
        VectorArcIterator { arcs }
    }
}

impl<W: Semiring> MutableFst<W> for VectorFst<W> {
    fn add_state(&mut self) -> StateId {
        let id = self.states.len() as StateId;
        self.states.push(VectorState::default());
        self.properties.invalidate_all();
        id
    }
    
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
    
    fn set_final(&mut self, state: StateId, weight: W) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.final_weight = if <W as num_traits::Zero>::is_zero(&weight) { None } else { Some(weight) };
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