//! Arc encoding utilities

use crate::arc::Arc;
use crate::fst::Label;
use crate::semiring::Semiring;
use std::collections::HashMap;

/// Encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeType {
    /// Encode labels and weights
    EncodeLabelsAndWeights,
    /// Encode weights only
    EncodeWeightsOnly,
    /// Encode labels only
    EncodeLabelsOnly,
}

/// Arc encoder/decoder
#[derive(Debug, Clone)]
pub struct EncodeMapper<W: Semiring + Eq + std::hash::Hash> {
    encode_type: EncodeType,
    label_map: HashMap<(Label, Label), Label>,
    weight_map: HashMap<W, u32>,
    next_label: Label,
    next_weight: u32,
}

impl<W: Semiring + Eq + std::hash::Hash> EncodeMapper<W> {
    /// Create a new encoder
    pub fn new(encode_type: EncodeType) -> Self {
        Self {
            encode_type,
            label_map: HashMap::new(),
            weight_map: HashMap::new(),
            next_label: 1,
            next_weight: 0,
        }
    }
    
    /// Encode an arc
    pub fn encode(&mut self, arc: &Arc<W>) -> Arc<W> {
        match self.encode_type {
            EncodeType::EncodeLabelsAndWeights => {
                let label = self.encode_labels(arc.ilabel, arc.olabel);
                let weight_id = self.encode_weight(&arc.weight);
                Arc::new(label, weight_id, W::one(), arc.nextstate)
            }
            EncodeType::EncodeWeightsOnly => {
                let weight_id = self.encode_weight(&arc.weight);
                Arc::new(arc.ilabel, weight_id, W::one(), arc.nextstate)
            }
            EncodeType::EncodeLabelsOnly => {
                let label = self.encode_labels(arc.ilabel, arc.olabel);
                Arc::new(label, 0, arc.weight.clone(), arc.nextstate)
            }
        }
    }
    
    fn encode_labels(&mut self, ilabel: Label, olabel: Label) -> Label {
        let key = (ilabel, olabel);
        if let Some(&label) = self.label_map.get(&key) {
            label
        } else {
            let label = self.next_label;
            self.label_map.insert(key, label);
            self.next_label += 1;
            label
        }
    }
    
    fn encode_weight(&mut self, weight: &W) -> u32 {
        if let Some(&id) = self.weight_map.get(weight) {
            id
        } else {
            let id = self.next_weight;
            self.weight_map.insert(weight.clone(), id);
            self.next_weight += 1;
            id
        }
    }
}