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
    /// Alias for EncodeLabelsAndWeights
    LabelsAndWeights,
    /// Alias for EncodeWeightsOnly  
    Weights,
    /// Alias for EncodeLabelsOnly
    Labels,
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

    /// Get encoding type
    pub fn encode_type(&self) -> EncodeType {
        self.encode_type
    }

    /// Get number of encoded elements
    pub fn size(&self) -> usize {
        self.label_map.len() + self.weight_map.len()
    }

    /// Encode an arc
    pub fn encode(&mut self, arc: &Arc<W>) -> Arc<W> {
        match self.encode_type {
            EncodeType::EncodeLabelsAndWeights | EncodeType::LabelsAndWeights => {
                let label = self.encode_labels(arc.ilabel, arc.olabel);
                let weight_id = self.encode_weight(&arc.weight);
                Arc::new(label, weight_id, W::one(), arc.nextstate)
            }
            EncodeType::EncodeWeightsOnly | EncodeType::Weights => {
                let weight_id = self.encode_weight(&arc.weight);
                Arc::new(arc.ilabel, weight_id, W::one(), arc.nextstate)
            }
            EncodeType::EncodeLabelsOnly | EncodeType::Labels => {
                let label = self.encode_labels(arc.ilabel, arc.olabel);
                Arc::new(label, 0, arc.weight.clone(), arc.nextstate)
            }
        }
    }

    /// Decode an arc (placeholder - actual decoding would require reverse mappings)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The decoding operation is not yet implemented
    /// - The encoded arc contains invalid or unmapped labels/weights
    /// - The reverse mapping tables are not available
    pub fn decode(&self, _arc: &Arc<W>) -> Result<Arc<W>, &'static str> {
        Err("Decoding not implemented - would require reverse mappings")
    }

    /// Get input symbols (placeholder - returns empty symbol table)
    pub fn input_symbols(&self) -> crate::utils::SymbolTable {
        crate::utils::SymbolTable::new()
    }

    /// Get output symbols (placeholder - returns empty symbol table)
    pub fn output_symbols(&self) -> crate::utils::SymbolTable {
        crate::utils::SymbolTable::new()
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
