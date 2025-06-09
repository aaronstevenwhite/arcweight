//! Cache FST wrapper

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::semiring::Semiring;
use crate::properties::FstProperties;
use std::sync::{Arc as SyncArc, RwLock};
use std::collections::HashMap;

/// Cache wrapper for expensive FST operations
#[derive(Debug)]
pub struct CacheFst<W: Semiring, F: Fst<W>> {
    fst: SyncArc<F>,
    arc_cache: RwLock<HashMap<StateId, Vec<Arc<W>>>>,
    weight_cache: RwLock<HashMap<StateId, Option<W>>>,
    _phantom: core::marker::PhantomData<W>,
}

impl<W: Semiring, F: Fst<W>> CacheFst<W, F> {
    /// Create a new cache FST
    pub fn new(fst: F) -> Self {
        Self {
            fst: SyncArc::new(fst),
            arc_cache: RwLock::new(HashMap::new()),
            weight_cache: RwLock::new(HashMap::new()),
            _phantom: core::marker::PhantomData,
        }
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) {
        self.arc_cache.write().unwrap().clear();
        self.weight_cache.write().unwrap().clear();
    }
}

#[derive(Debug)]
pub struct CacheArcIterator<W: Semiring> {
    arcs: Vec<Arc<W>>,
    pos: usize,
}

impl<W: Semiring> Iterator for CacheArcIterator<W> {
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

impl<W: Semiring> ArcIterator<W> for CacheArcIterator<W> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<W: Semiring, F: Fst<W>> Fst<W> for CacheFst<W, F> {
    type ArcIter<'a> = CacheArcIterator<W> where Self: 'a;
    
    fn start(&self) -> Option<StateId> {
        self.fst.start()
    }
    
    fn final_weight(&self, state: StateId) -> Option<&W> {
        // cache lookup and storage would need redesign
        self.fst.final_weight(state)
    }
    
    fn num_arcs(&self, state: StateId) -> usize {
        // check cache first
        if let Ok(cache) = self.arc_cache.read() {
            if let Some(arcs) = cache.get(&state) {
                return arcs.len();
            }
        }
        
        self.fst.num_arcs(state)
    }
    
    fn num_states(&self) -> usize {
        self.fst.num_states()
    }
    
    fn properties(&self) -> FstProperties {
        self.fst.properties()
    }
    
    fn arcs<'a>(&'a self, state: StateId) -> Self::ArcIter<'a> {
        // check cache
        let arcs = {
            let cache = self.arc_cache.read().unwrap();
            cache.get(&state).cloned()
        };
        
        let arcs = match arcs {
            Some(arcs) => arcs,
            None => {
                // compute and cache
                let computed: Vec<_> = self.fst.arcs(state).collect();
                let mut cache = self.arc_cache.write().unwrap();
                cache.insert(state, computed.clone());
                computed
            }
        };
        
        CacheArcIterator { arcs, pos: 0 }
    }
}