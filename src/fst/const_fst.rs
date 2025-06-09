//! Constant (immutable) FST implementation

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::semiring::Semiring;
use crate::properties::FstProperties;
use crate::Result;
use core::slice;

/// Immutable FST optimized for space and lookup speed
#[derive(Debug, Clone)]
pub struct ConstFst<W: Semiring> {
    states: Box<[ConstState<W>]>,
    arcs: Box<[Arc<W>]>,
    start: Option<StateId>,
    properties: FstProperties,
}

#[derive(Debug, Clone)]
struct ConstState<W: Semiring> {
    final_weight: Option<W>,
    arcs_start: u32,
    num_arcs: u32,
}

impl<W: Semiring> ConstFst<W> {
    /// Create from a vector FST
    pub fn from_fst<F: Fst<W>>(fst: &F) -> Result<Self> {
        let mut states = Vec::with_capacity(fst.num_states());
        let mut arcs = Vec::new();
        
        for state in fst.states() {
            let arcs_start = arcs.len() as u32;
            let state_arcs: Vec<_> = fst.arcs(state).collect();
            let num_arcs = state_arcs.len() as u32;
            
            arcs.extend(state_arcs);
            
            states.push(ConstState {
                final_weight: fst.final_weight(state).cloned(),
                arcs_start,
                num_arcs,
            });
        }
        
        Ok(Self {
            states: states.into_boxed_slice(),
            arcs: arcs.into_boxed_slice(),
            start: fst.start(),
            properties: fst.properties(),
        })
    }
}

/// Arc iterator for ConstFst
#[derive(Debug)]
pub struct ConstArcIterator<'a, W: Semiring> {
    arcs: slice::Iter<'a, Arc<W>>,
}

impl<'a, W: Semiring> ArcIterator<W> for ConstArcIterator<'a, W> {
    fn reset(&mut self) {
        unimplemented!("ConstArcIterator reset not implemented")
    }
}

impl<'a, W: Semiring> Iterator for ConstArcIterator<'a, W> {
    type Item = Arc<W>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.arcs.next().cloned()
    }
}

impl<W: Semiring> Fst<W> for ConstFst<W> {
    type ArcIter<'a> = ConstArcIterator<'a, W> where W: 'a;
    
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
            .map(|s| s.num_arcs as usize)
            .unwrap_or(0)
    }
    
    fn num_states(&self) -> usize {
        self.states.len()
    }
    
    fn properties(&self) -> FstProperties {
        self.properties
    }
    
    fn arcs<'a>(&'a self, state: StateId) -> Self::ArcIter<'a> {
        if let Some(s) = self.states.get(state as usize) {
            let start = s.arcs_start as usize;
            let end = start + s.num_arcs as usize;
            ConstArcIterator {
                arcs: self.arcs[start..end].iter(),
            }
        } else {
            ConstArcIterator {
                arcs: [].iter(),
            }
        }
    }
}

impl<W: Semiring> ExpandedFst<W> for ConstFst<W> {
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>] {
        if let Some(s) = self.states.get(state as usize) {
            let start = s.arcs_start as usize;
            let end = start + s.num_arcs as usize;
            &self.arcs[start..end]
        } else {
            &[]
        }
    }
}