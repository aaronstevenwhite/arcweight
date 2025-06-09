//! Lazy FST implementation

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::semiring::Semiring;
use crate::properties::FstProperties;
use crate::Result;
use core::cell::RefCell;
use std::sync::Mutex;

/// Lazy FST that computes states on demand
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

#[derive(Debug)]
struct StateCache<W: Semiring> {
    states: Vec<Option<LazyState<W>>>,
    start: Option<StateId>,
}

#[derive(Debug, Clone)]
pub struct LazyState<W: Semiring> {
    arcs: Vec<Arc<W>>,
    #[allow(dead_code)]
    final_weight: Option<W>,
}

impl<W: Semiring, F> LazyFstImpl<W, F>
where
    F: Fn(StateId) -> Option<LazyState<W>> + Send + Sync,
{
    /// Create a new lazy FST
    pub fn new(compute_fn: F, num_states: usize) -> Self {
        Self {
            compute_fn,
            cache: Mutex::new(RefCell::new(StateCache {
                states: vec![None; num_states],
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
        
        Some(computed)
    }
}

#[derive(Debug)]
pub struct LazyArcIterator<W: Semiring> {
    arcs: Vec<Arc<W>>,
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
    type ArcIter<'a> = LazyArcIterator<W> where Self: 'a;
    
    fn start(&self) -> Option<StateId> {
        let cache = self.cache.lock().unwrap();
        let x = cache.borrow().start;
        x
    }
    
    fn final_weight(&self, _state: StateId) -> Option<&W> {
        // this needs redesign to avoid lifetime issues
        unimplemented!("LazyFst final_weight needs redesign")
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
    
    fn arcs<'a>(&'a self, state: StateId) -> Self::ArcIter<'a> {
        let arcs = self.get_or_compute_state(state)
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