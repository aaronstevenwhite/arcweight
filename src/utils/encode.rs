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
    // Forward mappings
    label_map: HashMap<(Label, Label), Label>,
    weight_map: HashMap<W, u32>,
    // Reverse mappings for decoding
    reverse_label_map: HashMap<Label, (Label, Label)>,
    reverse_weight_map: HashMap<u32, W>,
    // For weight-only encoding: map input_label -> original_output_label
    original_olabel_map: HashMap<Label, Label>,
    // Symbol tables for tracking labels
    input_symbols: crate::utils::SymbolTable,
    output_symbols: crate::utils::SymbolTable,
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
            reverse_label_map: HashMap::new(),
            reverse_weight_map: HashMap::new(),
            original_olabel_map: HashMap::new(),
            input_symbols: crate::utils::SymbolTable::new(),
            output_symbols: crate::utils::SymbolTable::new(),
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
                // Store original output label for decoding
                self.original_olabel_map.insert(arc.ilabel, arc.olabel);
                Arc::new(arc.ilabel, weight_id, W::one(), arc.nextstate)
            }
            EncodeType::EncodeLabelsOnly | EncodeType::Labels => {
                let label = self.encode_labels(arc.ilabel, arc.olabel);
                Arc::new(label, 0, arc.weight.clone(), arc.nextstate)
            }
        }
    }

    /// Decode an arc back to its original form
    ///
    /// Reverses the encoding transformation by looking up original labels and weights
    /// from the reverse mapping tables. The decoding strategy depends on the encoding
    /// type used during encoding.
    ///
    /// # Decoding Strategies
    ///
    /// - **Label Decoding:** Recovers original (input, output) label pairs from combined ID
    /// - **Weight Decoding:** Recovers original weight from weight ID stored in output label
    /// - **Combined Decoding:** Recovers both labels and weights from encoded representation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::utils::{EncodeMapper, EncodeType};
    /// use arcweight::prelude::*;
    ///
    /// let mut encoder = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);
    ///
    /// // Original arc
    /// let arc = Arc::new(97, 98, TropicalWeight::new(0.5), 1);
    ///
    /// // Encode and decode
    /// let encoded = encoder.encode(&arc);
    /// let decoded = encoder.decode(&encoded)?;
    ///
    /// // Should recover original arc
    /// assert_eq!(decoded.ilabel, arc.ilabel);
    /// assert_eq!(decoded.olabel, arc.olabel);
    /// assert_eq!(decoded.weight, arc.weight);
    /// assert_eq!(decoded.nextstate, arc.nextstate);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The encoded arc contains invalid or unmapped labels/weights
    /// - The reverse mapping tables don't contain the required entries
    /// - The encoding type doesn't match the arc structure
    pub fn decode(&self, arc: &Arc<W>) -> Result<Arc<W>, &'static str> {
        match self.encode_type {
            EncodeType::EncodeLabelsAndWeights | EncodeType::LabelsAndWeights => {
                // Decode both labels and weights
                let (ilabel, olabel) = self
                    .reverse_label_map
                    .get(&arc.ilabel)
                    .ok_or("Label not found in reverse mapping")?;
                let weight = self
                    .reverse_weight_map
                    .get(&arc.olabel)
                    .ok_or("Weight not found in reverse mapping")?;
                Ok(Arc::new(*ilabel, *olabel, weight.clone(), arc.nextstate))
            }
            EncodeType::EncodeWeightsOnly | EncodeType::Weights => {
                // Decode only weights, recover original labels
                let weight = self
                    .reverse_weight_map
                    .get(&arc.olabel)
                    .ok_or("Weight not found in reverse mapping")?;
                let original_olabel = self
                    .original_olabel_map
                    .get(&arc.ilabel)
                    .ok_or("Original output label not found")?;
                Ok(Arc::new(
                    arc.ilabel,
                    *original_olabel,
                    weight.clone(),
                    arc.nextstate,
                ))
            }
            EncodeType::EncodeLabelsOnly | EncodeType::Labels => {
                // Decode only labels, weight is preserved
                let (ilabel, olabel) = self
                    .reverse_label_map
                    .get(&arc.ilabel)
                    .ok_or("Label not found in reverse mapping")?;
                Ok(Arc::new(
                    *ilabel,
                    *olabel,
                    arc.weight.clone(),
                    arc.nextstate,
                ))
            }
        }
    }

    /// Get input symbols table
    ///
    /// Returns the symbol table tracking all input labels that have been
    /// encoded. This can be used to understand the mapping between original
    /// and encoded labels.
    pub fn input_symbols(&self) -> &crate::utils::SymbolTable {
        &self.input_symbols
    }

    /// Get output symbols table
    ///
    /// Returns the symbol table tracking all output labels that have been
    /// encoded. This can be used to understand the mapping between original
    /// and encoded labels.
    pub fn output_symbols(&self) -> &crate::utils::SymbolTable {
        &self.output_symbols
    }

    fn encode_labels(&mut self, ilabel: Label, olabel: Label) -> Label {
        let key = (ilabel, olabel);
        if let Some(&label) = self.label_map.get(&key) {
            label
        } else {
            let label = self.next_label;
            self.label_map.insert(key, label);
            self.reverse_label_map.insert(label, key);

            // Track labels in symbol tables
            self.input_symbols.add_symbol(&format!("i{ilabel}"));
            self.output_symbols.add_symbol(&format!("o{olabel}"));

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
            self.reverse_weight_map.insert(id, weight.clone());
            self.next_weight += 1;
            id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::Arc;
    use crate::semiring::TropicalWeight;
    use num_traits::One;

    #[test]
    fn test_encode_mapper_creation() {
        let mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        assert_eq!(mapper.encode_type(), EncodeType::Labels);
        assert_eq!(mapper.size(), 0);
    }

    #[test]
    fn test_encode_mapper_encode_arc() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let encoded_arc = mapper.encode(&arc);

        // Encoding should produce a valid arc
        assert_eq!(encoded_arc.nextstate, arc.nextstate);
        assert_eq!(encoded_arc.weight, arc.weight);

        // Labels might be encoded differently
        if mapper.encode_type() == EncodeType::Labels {
            // Labels could be mapped to different values (u32 is always >= 0)
            assert!(encoded_arc.ilabel < u32::MAX);
            assert!(encoded_arc.olabel < u32::MAX);
        } else {
            assert_eq!(encoded_arc.ilabel, arc.ilabel);
            assert_eq!(encoded_arc.olabel, arc.olabel);
        }
    }

    #[test]
    fn test_encode_mapper_decode_arc() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        let original_arc = Arc::new(10, 20, TropicalWeight::new(5.0), 30);
        let encoded_arc = mapper.encode(&original_arc);

        // Decode should recover the original arc
        let decoded_arc = mapper.decode(&encoded_arc).unwrap();
        assert_eq!(decoded_arc.ilabel, original_arc.ilabel);
        assert_eq!(decoded_arc.olabel, original_arc.olabel);
        assert_eq!(decoded_arc.weight, original_arc.weight);
        assert_eq!(decoded_arc.nextstate, original_arc.nextstate);
    }

    #[test]
    fn test_encode_mapper_encode_types() {
        let label_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);
        let weights_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Weights);
        let both_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);

        assert_eq!(label_mapper.encode_type(), EncodeType::Labels);
        assert_eq!(weights_mapper.encode_type(), EncodeType::Weights);
        assert_eq!(both_mapper.encode_type(), EncodeType::LabelsAndWeights);
    }

    #[test]
    fn test_encode_mapper_consistency() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(3.0), 5);
        let arc3 = Arc::new(1, 2, TropicalWeight::new(4.0), 4);

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);
        let encoded3 = mapper.encode(&arc3);

        // Same labels and weights should get same encoding
        assert_eq!(encoded1.ilabel, encoded2.ilabel);
        assert_eq!(encoded1.olabel, encoded2.olabel);

        // Different weights should get different encoding
        assert_ne!(encoded1.olabel, encoded3.olabel);
    }

    #[test]
    fn test_encode_mapper_size() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        assert_eq!(mapper.size(), 0);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        mapper.encode(&arc1);
        assert_eq!(mapper.size(), 1);

        let arc2 = Arc::new(3, 4, TropicalWeight::new(5.0), 6);
        mapper.encode(&arc2);
        assert_eq!(mapper.size(), 2);

        // Same label pair should not increase size
        let arc3 = Arc::new(1, 2, TropicalWeight::new(7.0), 8);
        mapper.encode(&arc3);
        assert_eq!(mapper.size(), 2);
    }

    #[test]
    fn test_encode_mapper_weights() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Weights);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(5, 6, TropicalWeight::new(3.0), 7);
        let arc3 = Arc::new(8, 9, TropicalWeight::new(4.0), 10);

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);
        let encoded3 = mapper.encode(&arc3);

        // Original labels should be preserved
        assert_eq!(encoded1.ilabel, arc1.ilabel);
        assert_eq!(encoded2.ilabel, arc2.ilabel);
        assert_eq!(encoded3.ilabel, arc3.ilabel);

        // Same weight should get same encoding
        assert_eq!(encoded1.olabel, encoded2.olabel);
        // Different weight should get different encoding
        assert_ne!(encoded1.olabel, encoded3.olabel);
    }

    #[test]
    fn test_encode_mapper_symbol_tables() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        // Initially empty except for epsilon
        assert_eq!(mapper.input_symbols().size(), 1); // Only epsilon
        assert_eq!(mapper.output_symbols().size(), 1); // Only epsilon

        // After encoding, should track labels
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        mapper.encode(&arc);

        // Should have added symbols for the labels
        assert!(mapper.input_symbols().size() > 1);
        assert!(mapper.output_symbols().size() > 1);
    }

    #[test]
    fn test_encode_decode_labels_only() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(5.0), 6);
        let arc3 = Arc::new(3, 4, TropicalWeight::new(3.0), 7);

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);
        let encoded3 = mapper.encode(&arc3);

        // Same labels should get same encoding
        assert_eq!(encoded1.ilabel, encoded2.ilabel);
        assert_eq!(encoded1.olabel, encoded2.olabel);

        // Different labels should get different encoding
        assert_ne!(encoded1.ilabel, encoded3.ilabel);

        // Weights should be preserved
        assert_eq!(encoded1.weight, arc1.weight);
        assert_eq!(encoded2.weight, arc2.weight);

        // Decode should recover original arcs
        let decoded1 = mapper.decode(&encoded1).unwrap();
        let decoded2 = mapper.decode(&encoded2).unwrap();
        let decoded3 = mapper.decode(&encoded3).unwrap();

        assert_eq!(decoded1.ilabel, arc1.ilabel);
        assert_eq!(decoded1.olabel, arc1.olabel);
        assert_eq!(decoded1.weight, arc1.weight);
        assert_eq!(decoded1.nextstate, arc1.nextstate);

        assert_eq!(decoded2.ilabel, arc2.ilabel);
        assert_eq!(decoded2.olabel, arc2.olabel);
        assert_eq!(decoded2.weight, arc2.weight);
        assert_eq!(decoded2.nextstate, arc2.nextstate);

        assert_eq!(decoded3.ilabel, arc3.ilabel);
        assert_eq!(decoded3.olabel, arc3.olabel);
        assert_eq!(decoded3.weight, arc3.weight);
        assert_eq!(decoded3.nextstate, arc3.nextstate);
    }

    #[test]
    fn test_encode_decode_weights_only() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeWeightsOnly);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(5, 6, TropicalWeight::new(3.0), 7);
        let arc3 = Arc::new(1, 2, TropicalWeight::new(8.0), 9);

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);
        let encoded3 = mapper.encode(&arc3);

        // Same weights should get same encoding
        assert_eq!(encoded1.olabel, encoded2.olabel);

        // Different weights should get different encoding
        assert_ne!(encoded1.olabel, encoded3.olabel);

        // Labels should be preserved
        assert_eq!(encoded1.ilabel, arc1.ilabel);
        assert_eq!(encoded2.ilabel, arc2.ilabel);
        assert_eq!(encoded3.ilabel, arc3.ilabel);

        // Encoded weight should be one
        assert_eq!(encoded1.weight, TropicalWeight::one());
        assert_eq!(encoded2.weight, TropicalWeight::one());
        assert_eq!(encoded3.weight, TropicalWeight::one());

        // Decode should recover original arcs
        let decoded1 = mapper.decode(&encoded1).unwrap();
        let decoded2 = mapper.decode(&encoded2).unwrap();
        let decoded3 = mapper.decode(&encoded3).unwrap();

        assert_eq!(decoded1.ilabel, arc1.ilabel);
        assert_eq!(decoded1.olabel, arc1.olabel);
        assert_eq!(decoded1.weight, arc1.weight);
        assert_eq!(decoded1.nextstate, arc1.nextstate);

        assert_eq!(decoded2.ilabel, arc2.ilabel);
        assert_eq!(decoded2.olabel, arc2.olabel);
        assert_eq!(decoded2.weight, arc2.weight);
        assert_eq!(decoded2.nextstate, arc2.nextstate);

        assert_eq!(decoded3.ilabel, arc3.ilabel);
        assert_eq!(decoded3.olabel, arc3.olabel);
        assert_eq!(decoded3.weight, arc3.weight);
        assert_eq!(decoded3.nextstate, arc3.nextstate);
    }

    #[test]
    fn test_encode_decode_labels_and_weights() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsAndWeights);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(3.0), 5);
        let arc3 = Arc::new(1, 2, TropicalWeight::new(6.0), 7);
        let arc4 = Arc::new(8, 9, TropicalWeight::new(3.0), 10);

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);
        let encoded3 = mapper.encode(&arc3);
        let encoded4 = mapper.encode(&arc4);

        // Same labels and weights should get same encoding
        assert_eq!(encoded1.ilabel, encoded2.ilabel);
        assert_eq!(encoded1.olabel, encoded2.olabel);

        // Different weights should get different encoding
        assert_ne!(encoded1.olabel, encoded3.olabel);

        // Different labels should get different encoding
        assert_ne!(encoded1.ilabel, encoded4.ilabel);

        // Encoded weight should be one
        assert_eq!(encoded1.weight, TropicalWeight::one());
        assert_eq!(encoded2.weight, TropicalWeight::one());
        assert_eq!(encoded3.weight, TropicalWeight::one());
        assert_eq!(encoded4.weight, TropicalWeight::one());

        // Decode should recover original arcs
        let decoded1 = mapper.decode(&encoded1).unwrap();
        let decoded2 = mapper.decode(&encoded2).unwrap();
        let decoded3 = mapper.decode(&encoded3).unwrap();
        let decoded4 = mapper.decode(&encoded4).unwrap();

        assert_eq!(decoded1.ilabel, arc1.ilabel);
        assert_eq!(decoded1.olabel, arc1.olabel);
        assert_eq!(decoded1.weight, arc1.weight);
        assert_eq!(decoded1.nextstate, arc1.nextstate);

        assert_eq!(decoded2.ilabel, arc2.ilabel);
        assert_eq!(decoded2.olabel, arc2.olabel);
        assert_eq!(decoded2.weight, arc2.weight);
        assert_eq!(decoded2.nextstate, arc2.nextstate);

        assert_eq!(decoded3.ilabel, arc3.ilabel);
        assert_eq!(decoded3.olabel, arc3.olabel);
        assert_eq!(decoded3.weight, arc3.weight);
        assert_eq!(decoded3.nextstate, arc3.nextstate);

        assert_eq!(decoded4.ilabel, arc4.ilabel);
        assert_eq!(decoded4.olabel, arc4.olabel);
        assert_eq!(decoded4.weight, arc4.weight);
        assert_eq!(decoded4.nextstate, arc4.nextstate);
    }

    #[test]
    fn test_encode_decode_error_cases() {
        let mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);

        // Try to decode arc with invalid label
        let invalid_arc = Arc::new(999, 0, TropicalWeight::new(1.0), 1);
        let result = mapper.decode(&invalid_arc);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Label not found in reverse mapping");

        let weight_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeWeightsOnly);

        // Try to decode arc with invalid weight ID
        let invalid_weight_arc = Arc::new(1, 999, TropicalWeight::one(), 1);
        let result = weight_mapper.decode(&invalid_weight_arc);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Weight not found in reverse mapping");
    }

    #[test]
    fn test_encode_alias_types() {
        // Test that alias types work the same as full names
        let labels_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);
        let encode_labels_mapper =
            EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);

        assert_eq!(labels_mapper.encode_type(), EncodeType::Labels);
        assert_eq!(
            encode_labels_mapper.encode_type(),
            EncodeType::EncodeLabelsOnly
        );

        let weights_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Weights);
        let encode_weights_mapper =
            EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeWeightsOnly);

        assert_eq!(weights_mapper.encode_type(), EncodeType::Weights);
        assert_eq!(
            encode_weights_mapper.encode_type(),
            EncodeType::EncodeWeightsOnly
        );

        let both_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);
        let encode_both_mapper =
            EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsAndWeights);

        assert_eq!(both_mapper.encode_type(), EncodeType::LabelsAndWeights);
        assert_eq!(
            encode_both_mapper.encode_type(),
            EncodeType::EncodeLabelsAndWeights
        );
    }

    #[test]
    fn test_roundtrip_encoding_large_labels() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsOnly);

        // Test with large label values
        let arc = Arc::new(u32::MAX - 1, u32::MAX, TropicalWeight::new(42.0), 100);
        let encoded = mapper.encode(&arc);
        let decoded = mapper.decode(&encoded).unwrap();

        assert_eq!(decoded.ilabel, arc.ilabel);
        assert_eq!(decoded.olabel, arc.olabel);
        assert_eq!(decoded.weight, arc.weight);
        assert_eq!(decoded.nextstate, arc.nextstate);
    }

    #[test]
    fn test_multiple_encoding_sessions() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::EncodeLabelsAndWeights);

        // First session
        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let encoded1 = mapper.encode(&arc1);
        let decoded1 = mapper.decode(&encoded1).unwrap();
        assert_eq!(decoded1.ilabel, arc1.ilabel);

        // Second session with different arc
        let arc2 = Arc::new(5, 6, TropicalWeight::new(7.0), 8);
        let encoded2 = mapper.encode(&arc2);
        let decoded2 = mapper.decode(&encoded2).unwrap();
        assert_eq!(decoded2.ilabel, arc2.ilabel);

        // Third session reusing labels from first
        let arc3 = Arc::new(1, 2, TropicalWeight::new(9.0), 10);
        let encoded3 = mapper.encode(&arc3);
        let decoded3 = mapper.decode(&encoded3).unwrap();
        assert_eq!(decoded3.ilabel, arc3.ilabel);

        // Labels should be reused but weights should be different
        assert_eq!(encoded1.ilabel, encoded3.ilabel); // Same labels
        assert_ne!(encoded1.olabel, encoded3.olabel); // Different weights
    }
}
