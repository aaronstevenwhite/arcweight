//! Compact FST implementation

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::semiring::Semiring;
use crate::properties::FstProperties;
use core::marker::PhantomData;
use core::fmt::Debug;

/// Compact FST with compressed arc representation
#[derive(Debug, Clone)]
pub struct CompactFst<W: Semiring, C: Compactor<W>> {
    states: Vec<CompactState>,
    data: Vec<C::Element>,
    start: Option<StateId>,
    properties: FstProperties,
    _phantom: PhantomData<(W, C)>,
}

#[derive(Debug, Clone)]
struct CompactState {
    #[allow(dead_code)]
    final_weight_idx: Option<u32>,
    arcs_start: u32,
    num_arcs: u32,
}

/// Trait for arc compaction strategies
pub trait Compactor<W: Semiring>: Debug + Send + Sync + 'static {
    /// Compacted element type
    type Element: Clone + Debug + Send + Sync;
    
    /// Compact an arc
    fn compact(arc: &Arc<W>) -> Self::Element;
    
    /// Expand a compacted arc
    fn expand(element: &Self::Element) -> Arc<W>;
    
    /// Compact a weight
    fn compact_weight(weight: &W) -> Self::Element;
    
    /// Expand a compacted weight
    fn expand_weight(element: &Self::Element) -> W;
}

/// Default compactor for fixed-size alphabets
#[derive(Debug)]
pub struct DefaultCompactor<W: Semiring> {
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Compactor<W> for DefaultCompactor<W> {
    type Element = CompactElement<W>;
    
    fn compact(arc: &Arc<W>) -> Self::Element {
        CompactElement::Arc {
            ilabel: arc.ilabel,
            olabel: arc.olabel,
            weight: arc.weight.clone(),
            nextstate: arc.nextstate,
        }
    }
    
    fn expand(element: &Self::Element) -> Arc<W> {
        match element {
            CompactElement::Arc { ilabel, olabel, weight, nextstate } => {
                Arc::new(*ilabel, *olabel, weight.clone(), *nextstate)
            }
            _ => panic!("Expected arc element"),
        }
    }
    
    fn compact_weight(weight: &W) -> Self::Element {
        CompactElement::Weight(weight.clone())
    }
    
    fn expand_weight(element: &Self::Element) -> W {
        match element {
            CompactElement::Weight(w) => w.clone(),
            _ => panic!("Expected weight element"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CompactElement<W: Semiring> {
    Arc {
        ilabel: Label,
        olabel: Label,
        weight: W,
        nextstate: StateId,
    },
    Weight(W),
}

impl<W: Semiring, C: Compactor<W>> Default for CompactFst<W, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Semiring, C: Compactor<W>> CompactFst<W, C> {
    /// Create a new compact FST
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            data: Vec::new(),
            start: None,
            properties: FstProperties::default(),
            _phantom: PhantomData,
        }
    }
}

/// Arc iterator for CompactFst
#[derive(Debug)]
pub struct CompactArcIterator<'a, W: Semiring, C: Compactor<W>> {
    data: &'a [C::Element],
    pos: usize,
    end: usize,
    _phantom: PhantomData<W>,
}

impl<W: Semiring, C: Compactor<W>> ArcIterator<W> for CompactArcIterator<'_, W, C> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<W: Semiring, C: Compactor<W>> Iterator for CompactArcIterator<'_, W, C> {
    type Item = Arc<W>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let arc = C::expand(&self.data[self.pos]);
            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
}

impl<W: Semiring, C: Compactor<W>> Fst<W> for CompactFst<W, C> {
    type ArcIter<'a> = CompactArcIterator<'a, W, C> where W: 'a, C: 'a;
    
    fn start(&self) -> Option<StateId> {
        self.start
    }
    
    fn final_weight(&self, _state: StateId) -> Option<&W> {
        // this would need redesign to avoid returning references
        unimplemented!("CompactFst final_weight needs redesign")
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
    
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        if let Some(s) = self.states.get(state as usize) {
            let start = s.arcs_start as usize;
            let end = start + s.num_arcs as usize;
            CompactArcIterator {
                data: &self.data,
                pos: start,
                end,
                _phantom: PhantomData,
            }
        } else {
            CompactArcIterator {
                data: &self.data,
                pos: 0,
                end: 0,
                _phantom: PhantomData,
            }
        }
    }
}