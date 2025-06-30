//! Arc encoding utilities for memory-efficient FST representation
//!
//! This module provides arc encoding functionality to reduce memory footprint
//! of FSTs by mapping frequent label pairs and weights to compact integer IDs.
//! This is particularly useful for large FSTs with many repeated patterns.
//!
//! ## Overview
//!
//! Arc encoding transforms FST arcs by replacing their components with compact
//! integer representations:
//! - **Label encoding:** Maps (input, output) label pairs to single integers
//! - **Weight encoding:** Maps unique weights to integer IDs
//! - **Combined encoding:** Encodes both labels and weights
//!
//! ## Encoding Strategies
//!
//! ### Label Encoding
//! Useful when FSTs have many repeated label pairs:
//! ```
//! use arcweight::utils::{EncodeMapper, EncodeType};
//! use arcweight::prelude::*;
//!
//! let mut encoder = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);
//!
//! // Original arc with labels (97, 98) - 'a' -> 'b'
//! let arc = Arc::new(97, 98, TropicalWeight::new(0.5), 1);
//!
//! // Encoded arc has combined label, original weight
//! let encoded = encoder.encode(&arc);
//! assert_eq!(encoded.ilabel, 1); // Combined label ID
//! assert_eq!(encoded.olabel, 0); // Always 0 for label encoding
//! assert_eq!(encoded.weight, TropicalWeight::new(0.5)); // Weight unchanged
//! ```
//!
//! ### Weight Encoding
//! Useful for FSTs with limited weight vocabulary:
//! ```
//! use arcweight::utils::{EncodeMapper, EncodeType};
//! use arcweight::prelude::*;
//!
//! let mut encoder = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeWeightsOnly);
//!
//! // Arcs with same weight
//! let arc1 = Arc::new(1, 2, TropicalWeight::new(0.5), 1);
//! let arc2 = Arc::new(3, 4, TropicalWeight::new(0.5), 2);
//!
//! let enc1 = encoder.encode(&arc1);
//! let enc2 = encoder.encode(&arc2);
//!
//! // Same weight gets same ID
//! assert_eq!(enc1.olabel, enc2.olabel);
//! assert_eq!(enc1.weight, TropicalWeight::one()); // Weight replaced by one
//! ```
//!
//! ## Use Cases
//!
//! 1. **Compact FST Storage:** Reduce memory usage for large FSTs
//! 2. **FST Compression:** Prepare FSTs for efficient serialization
//! 3. **Cache Optimization:** Improve cache locality with smaller arcs
//! 4. **Network Transfer:** Reduce bandwidth for distributed FST processing

use crate::arc::Arc;
use crate::fst::Label;
use crate::semiring::Semiring;
use std::collections::HashMap;

/// Arc encoding strategies
///
/// Determines which components of arcs are encoded for compression.
///
/// ## Variants
///
/// - `EncodeLabelsAndWeights`: Encode both label pairs and weights (maximum compression)
/// - `EncodeWeightsOnly`: Encode only weights, preserve original labels
/// - `EncodeLabelsOnly`: Encode only label pairs, preserve original weights
///
/// The aliased variants (`LabelsAndWeights`, `Weights`, `Labels`) are provided
/// for convenience and backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeType {
    /// Encode both labels and weights for maximum compression
    EncodeLabelsAndWeights,
    /// Encode weights only, preserving original labels
    EncodeWeightsOnly,
    /// Encode labels only, preserving original weights
    EncodeLabelsOnly,
    /// Convenience alias for EncodeLabelsAndWeights
    LabelsAndWeights,
    /// Convenience alias for EncodeWeightsOnly  
    Weights,
    /// Convenience alias for EncodeLabelsOnly
    Labels,
}

/// Arc encoder for FST compression
///
/// Maps arc components (labels and/or weights) to compact integer representations
/// to reduce memory usage. The encoding is deterministic - identical inputs always
/// produce identical outputs.
///
/// ## Type Parameters
///
/// - `W`: Semiring type for arc weights (must be hashable for weight encoding)
///
/// ## Examples
///
/// ### Basic Usage
/// ```
/// use arcweight::utils::{EncodeMapper, EncodeType};
/// use arcweight::prelude::*;
///
/// // Create encoder for label compression
/// let mut encoder = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);
///
/// // Encode multiple arcs
/// let arc1 = Arc::new(1, 2, TropicalWeight::new(0.5), 10);
/// let arc2 = Arc::new(1, 2, TropicalWeight::new(0.3), 20);
/// let arc3 = Arc::new(3, 4, TropicalWeight::new(0.5), 30);
///
/// let enc1 = encoder.encode(&arc1);
/// let enc2 = encoder.encode(&arc2);
/// let enc3 = encoder.encode(&arc3);
///
/// // Same label pair gets same encoding
/// assert_eq!(enc1.ilabel, enc2.ilabel);
/// // Different label pair gets different encoding
/// assert_ne!(enc1.ilabel, enc3.ilabel);
/// ```
///
/// ### Encoding FST for Storage
/// ```
/// use arcweight::utils::{EncodeMapper, EncodeType};
/// use arcweight::prelude::*;
///
/// fn compress_fst(fst: &VectorFst<TropicalWeight>) -> (VectorFst<TropicalWeight>, EncodeMapper<TropicalWeight>) {
///     let mut encoder = EncodeMapper::new(EncodeType::EncodeLabelsAndWeights);
///     let mut compressed = VectorFst::new();
///     
///     // Copy states
///     for _ in 0..fst.num_states() {
///         compressed.add_state();
///     }
///     
///     if let Some(start) = fst.start() {
///         compressed.set_start(start);
///     }
///     
///     // Encode arcs
///     for state in fst.states() {
///         for arc in fst.arcs(state) {
///             compressed.add_arc(state, encoder.encode(&arc));
///         }
///         
///         if let Some(weight) = fst.final_weight(state) {
///             compressed.set_final(state, weight.clone());
///         }
///     }
///     
///     (compressed, encoder)
/// }
/// ```
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
