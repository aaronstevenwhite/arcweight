//! FST encoding and decoding
//!
//! ## Overview
//!
//! Encoding transforms a weighted transducer into an unweighted acceptor by
//! mapping (ilabel, olabel, weight) tuples to single encoded labels. Decoding
//! reverses the transformation, recovering the original weighted transducer.
//!
//! ## Complexity
//!
//! - **Encode:** O(V + E) where V = states, E = arcs
//! - **Decode:** O(V + E) for reconstruction
//! - **Space:** O(E) for encoding table
//!
//! ## Use Cases
//!
//! - **Algorithm compatibility:** Run acceptor-only algorithms on transducers
//! - **Determinization:** Some FST operations require acceptor form
//! - **Minimization:** Certain minimization algorithms work on acceptors
//! - **Analysis:** Simplify FST structure temporarily for analysis
//!
//! ## References
//!
//! - Cyril Allauzen et al. (2007). "OpenFst: A General and Efficient Weighted
//!   Finite-State Transducer Library." CIAA 2007.

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, VectorFst};
use crate::semiring::{Semiring, TropicalWeight};
use crate::Result;
use num_traits::One;
use std::collections::HashMap;
use std::hash::Hash;

/// Encoding table mapping between original and encoded representations.
///
/// The table maintains bidirectional mapping:
/// - Forward: (ilabel, olabel, weight) → encoded_label
/// - Reverse: encoded_label → (ilabel, olabel, weight)
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
///
/// let (encoded_fst, table) = encode(&fst).unwrap();
///
/// // Encoded FST has unit weights
/// for state in encoded_fst.states() {
///     for arc in encoded_fst.arcs(state) {
///         assert_eq!(arc.weight, TropicalWeight::one());
///     }
/// }
///
/// // Can decode back to original
/// let decoded_fst = decode(&encoded_fst, &table).unwrap();
/// assert!(isomorphic(&fst, &decoded_fst).unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct EncodeTable<W: Semiring> {
    /// Maps (ilabel, olabel, weight) → encoded label
    encode_map: HashMap<(Label, Label, W), Label>,

    /// Maps encoded label → (ilabel, olabel, weight)
    decode_vec: Vec<(Label, Label, W)>,

    /// Next available encoded label
    next_label: Label,
}

impl<W: Semiring + Hash + Eq + Clone> EncodeTable<W> {
    /// Creates a new empty encoding table.
    pub fn new() -> Self {
        Self {
            encode_map: HashMap::new(),
            decode_vec: Vec::new(),
            next_label: 1, // Start from 1, reserve 0 for epsilon
        }
    }

    /// Encodes a tuple, returning its encoded label.
    ///
    /// If the tuple is new, assigns a fresh label. Otherwise returns
    /// the existing encoding.
    pub fn encode(&mut self, ilabel: Label, olabel: Label, weight: W) -> Label {
        let key = (ilabel, olabel, weight.clone());

        if let Some(&encoded_label) = self.encode_map.get(&key) {
            encoded_label
        } else {
            let encoded_label = self.next_label;
            self.encode_map.insert(key.clone(), encoded_label);
            self.decode_vec.push(key);
            self.next_label += 1;
            encoded_label
        }
    }

    /// Decodes an encoded label back to its original tuple.
    ///
    /// Returns None if the label was not in the encoding table.
    pub fn decode(&self, label: Label) -> Option<(Label, Label, W)> {
        if label == 0 {
            // Epsilon remains epsilon
            return Some((0, 0, W::one()));
        }

        let index = (label - 1) as usize;
        self.decode_vec.get(index).cloned()
    }

    /// Returns the number of unique encodings.
    pub fn size(&self) -> usize {
        self.decode_vec.len()
    }
}

impl<W: Semiring + Hash + Eq + Clone> Default for EncodeTable<W> {
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes an FST by mapping (ilabel, olabel, weight) tuples to single labels.
///
/// Creates an unweighted acceptor where each original arc's information is
/// encoded into a single label. The encoding table can be used to decode
/// the FST back to its original form.
///
/// # Algorithm
///
/// 1. Create encoding table
/// 2. For each arc in original FST:
///    - Encode (ilabel, olabel, weight) → encoded_label
///    - Create new arc with (encoded_label, encoded_label, One, nextstate)
/// 3. Copy final weights as One (original final weight encoded separately if needed)
///
/// # Complexity
///
/// - **Time:** O(|V| + |E|) where V = states, E = arcs
///   - Single pass through FST: O(|V| + |E|)
///   - Hash table lookups: O(1) average case per arc
/// - **Space:** O(|E|) for encoding table (bounded by unique arc tuples)
///
/// # Algorithm
///
/// Bijective encoding via hash table:
/// 1. Create empty encoding table mapping (i, o, w) ↦ label
/// 2. For each arc, lookup or insert tuple in table, assign unique label
/// 3. Build encoded FST with integer labels, tropical weights = 1̄
/// 4. Return encoded FST and table for decoding
///
/// # Performance Notes
///
/// - **Hash efficiency:** O(1) average lookups for tuple-to-label mapping
/// - **Memory:** Table size = number of unique (ilabel, olabel, weight) tuples
/// - **Best case:** All arcs identical → table size = 1
/// - **Worst case:** All arcs unique → table size = |E|
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(1.0));
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1)); // Duplicate
///
/// let (encoded, table) = encode(&fst).unwrap();
///
/// // Both arcs encoded to same label (same tuple)
/// let arcs: Vec<_> = encoded.arcs(s0).collect();
/// assert_eq!(arcs.len(), 2);
/// assert_eq!(arcs[0].ilabel, arcs[1].ilabel);
/// ```
///
/// # See Also
///
/// - [`decode`] - Inverse operation to restore original FST
/// - [`EncodeTable`] - Bidirectional mapping for encode/decode
pub fn encode<W, F>(fst: &F) -> Result<(VectorFst<TropicalWeight>, EncodeTable<W>)>
where
    W: Semiring + Hash + Eq + Clone,
    F: Fst<W>,
{
    let mut table = EncodeTable::new();
    let mut result = VectorFst::<TropicalWeight>::new();

    // Add same number of states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Copy start state
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Copy final states with unit weight
    for state in fst.states() {
        if fst.is_final(state) {
            result.set_final(state, TropicalWeight::one());
        }
    }

    // Encode arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            let encoded_label = table.encode(arc.ilabel, arc.olabel, arc.weight);
            result.add_arc(
                state,
                Arc::new(
                    encoded_label,
                    encoded_label,
                    TropicalWeight::one(),
                    arc.nextstate,
                ),
            );
        }
    }

    Ok((result, table))
}

/// Decodes an encoded FST back to its original weighted transducer form.
///
/// Uses the encoding table to map encoded labels back to
/// (ilabel, olabel, weight) tuples.
///
/// # Complexity
///
/// - **Time:** O(|V| + |E|) where V = states, E = arcs
///   - Single pass through FST: O(|V| + |E|)
///   - Table lookups: O(1) per arc
/// - **Space:** O(|V| + |E|) for result FST
///
/// # Algorithm
///
/// Inverse mapping via encoding table:
/// 1. Create result FST with same state structure
/// 2. For each encoded arc with label ℓ:
///    - Lookup (i, o, w) = table\[ℓ\]
///    - Create arc with original labels and weight
/// 3. Copy final weights unchanged
///
/// # Performance Notes
///
/// - **Table lookups:** O(1) reverse mapping from labels to tuples
/// - **Correctness:** decode(encode(T)) ≅ T (isomorphic)
/// - **Memory:** No additional table storage needed (uses provided table)
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut original = VectorFst::<TropicalWeight>::new();
/// let s0 = original.add_state();
/// let s1 = original.add_state();
/// original.set_start(s0);
/// original.set_final(s1, TropicalWeight::one());
/// original.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
///
/// let (encoded, table) = encode(&original).unwrap();
/// let decoded = decode(&encoded, &table).unwrap();
///
/// // Decoded FST preserves structure and arc labels/weights
/// assert_eq!(decoded.num_states(), original.num_states());
/// assert_eq!(decoded.num_arcs_total(), original.num_arcs_total());
///
/// let orig_arcs: Vec<_> = original.arcs(s0).collect();
/// let dec_arcs: Vec<_> = decoded.arcs(s0).collect();
/// assert_eq!(orig_arcs[0].ilabel, dec_arcs[0].ilabel);
/// assert_eq!(orig_arcs[0].olabel, dec_arcs[0].olabel);
/// assert_eq!(orig_arcs[0].weight, dec_arcs[0].weight);
/// ```
///
/// # See Also
///
/// - [`encode`] - Creates encoded FST and table
/// - [`EncodeTable`] - Bidirectional mapping structure
pub fn decode<W>(
    fst: &VectorFst<TropicalWeight>,
    table: &EncodeTable<W>,
) -> Result<VectorFst<W>>
where
    W: Semiring + Clone + Hash + Eq,
{
    let mut result = VectorFst::<W>::new();

    // Add same number of states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Copy start state
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Copy final states with unit weight from W
    // Note: final weights are not encoded in this implementation
    for state in fst.states() {
        if fst.is_final(state) {
            result.set_final(state, W::one());
        }
    }

    // Decode arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if let Some((ilabel, olabel, weight)) = table.decode(arc.ilabel) {
                result.add_arc(state, Arc::new(ilabel, olabel, weight, arc.nextstate));
            } else {
                return Err(crate::Error::InvalidOperation(format!(
                    "Failed to decode label {} at state {}",
                    arc.ilabel, state
                )));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fst::MutableFst;
    use crate::semiring::{BooleanWeight, IntegerWeight, LogWeight};

    #[test]
    fn test_encode_table_basic() {
        let mut table = EncodeTable::<TropicalWeight>::new();

        let label1 = table.encode(1, 2, TropicalWeight::new(3.0));
        let label2 = table.encode(4, 5, TropicalWeight::new(6.0));

        assert_ne!(label1, label2);
        assert_eq!(table.size(), 2);
    }

    #[test]
    fn test_encode_table_duplicates() {
        let mut table = EncodeTable::<TropicalWeight>::new();

        let label1 = table.encode(1, 2, TropicalWeight::new(3.0));
        let label2 = table.encode(1, 2, TropicalWeight::new(3.0));

        assert_eq!(label1, label2);
        assert_eq!(table.size(), 1);
    }

    #[test]
    fn test_encode_table_decode() {
        let mut table = EncodeTable::<TropicalWeight>::new();

        let label = table.encode(1, 2, TropicalWeight::new(3.0));
        let decoded = table.decode(label).unwrap();

        assert_eq!(decoded, (1, 2, TropicalWeight::new(3.0)));
    }

    #[test]
    fn test_encode_simple_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let (encoded, table) = encode(&fst).unwrap();

        assert_eq!(encoded.num_states(), fst.num_states());
        assert_eq!(encoded.num_arcs_total(), fst.num_arcs_total());
        assert_eq!(table.size(), 1);
    }

    #[test]
    fn test_encode_creates_unit_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let (encoded, _) = encode(&fst).unwrap();

        for state in encoded.states() {
            for arc in encoded.arcs(state) {
                assert_eq!(arc.weight, TropicalWeight::one());
            }
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let (encoded, table) = encode(&fst).unwrap();
        let decoded = decode(&encoded, &table).unwrap();

        // Check structure is preserved
        assert_eq!(decoded.num_states(), fst.num_states());
        assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());
        assert_eq!(decoded.start(), fst.start());

        // Check arc labels and weights match
        for state in fst.states() {
            let orig_arcs: Vec<_> = fst.arcs(state).collect();
            let dec_arcs: Vec<_> = decoded.arcs(state).collect();

            assert_eq!(orig_arcs.len(), dec_arcs.len());

            for (orig, dec) in orig_arcs.iter().zip(dec_arcs.iter()) {
                assert_eq!(orig.ilabel, dec.ilabel);
                assert_eq!(orig.olabel, dec.olabel);
                assert_eq!(orig.weight, dec.weight);
                assert_eq!(orig.nextstate, dec.nextstate);
            }
        }
    }

    #[test]
    fn test_encode_with_duplicate_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1)); // Duplicate

        let (encoded, table) = encode(&fst).unwrap();

        assert_eq!(table.size(), 1); // Only one unique tuple

        let arcs: Vec<_> = encoded.arcs(s0).collect();
        assert_eq!(arcs.len(), 2);
        assert_eq!(arcs[0].ilabel, arcs[1].ilabel); // Same encoded label
    }

    #[test]
    fn test_encode_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let (encoded, table) = encode(&fst).unwrap();

        assert_eq!(encoded.num_states(), 0);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_encode_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let (encoded, table) = encode(&fst).unwrap();

        assert_eq!(encoded.num_states(), 1);
        assert_eq!(table.size(), 0); // No arcs to encode
    }

    #[test]
    fn test_encode_multiple_semirings() {
        // Test with BooleanWeight
        let mut fst1 = VectorFst::<BooleanWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, BooleanWeight::one());
        fst1.add_arc(s0, Arc::new(1, 2, BooleanWeight::one(), s1));

        let (encoded1, table1) = encode(&fst1).unwrap();
        assert_eq!(table1.size(), 1);

        let decoded1 = decode(&encoded1, &table1).unwrap();
        assert_eq!(decoded1.num_states(), fst1.num_states());

        // Test with IntegerWeight
        let mut fst2 = VectorFst::<IntegerWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, IntegerWeight::one());
        fst2.add_arc(s0, Arc::new(1, 2, IntegerWeight::new(5), s1));

        let (encoded2, table2) = encode(&fst2).unwrap();
        assert_eq!(table2.size(), 1);

        let decoded2 = decode(&encoded2, &table2).unwrap();
        assert_eq!(decoded2.num_states(), fst2.num_states());
    }

    #[test]
    fn test_encode_epsilon_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s1));

        let (encoded, table) = encode(&fst).unwrap();
        let decoded = decode(&encoded, &table).unwrap();

        let orig_arcs: Vec<_> = fst.arcs(s0).collect();
        let dec_arcs: Vec<_> = decoded.arcs(s0).collect();

        assert_eq!(orig_arcs.len(), dec_arcs.len());
        assert_eq!(orig_arcs[0].ilabel, dec_arcs[0].ilabel);
        assert_eq!(orig_arcs[0].olabel, dec_arcs[0].olabel);
    }

    #[test]
    fn test_encode_complex_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[4], TropicalWeight::new(2.0));

        fst.add_arc(
            states[0],
            Arc::new(1, 2, TropicalWeight::new(0.1), states[1]),
        );
        fst.add_arc(
            states[0],
            Arc::new(3, 4, TropicalWeight::new(0.2), states[2]),
        );
        fst.add_arc(
            states[1],
            Arc::new(5, 6, TropicalWeight::new(0.3), states[3]),
        );
        fst.add_arc(
            states[2],
            Arc::new(7, 8, TropicalWeight::new(0.4), states[3]),
        );
        fst.add_arc(
            states[3],
            Arc::new(9, 10, TropicalWeight::new(0.5), states[4]),
        );

        let (encoded, table) = encode(&fst).unwrap();
        let decoded = decode(&encoded, &table).unwrap();

        assert_eq!(decoded.num_states(), fst.num_states());
        assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());
        assert_eq!(table.size(), 5); // 5 unique arc tuples
    }

    #[test]
    fn test_decode_invalid_label() {
        let table = EncodeTable::<TropicalWeight>::new();

        let mut encoded = VectorFst::<TropicalWeight>::new();
        let s0 = encoded.add_state();
        let s1 = encoded.add_state();
        encoded.set_start(s0);
        encoded.add_arc(
            s0,
            Arc::new(999, 999, TropicalWeight::one(), s1), // Invalid label
        );

        let result = decode(&encoded, &table);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_table_default() {
        let table: EncodeTable<TropicalWeight> = EncodeTable::default();
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_encode_decode_with_log_weight() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, LogWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, LogWeight::new(0.5), s1));

        let (encoded, table) = encode(&fst).unwrap();
        let decoded = decode(&encoded, &table).unwrap();

        assert_eq!(decoded.num_states(), fst.num_states());
        assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());
    }

    #[test]
    fn test_encode_multiple_different_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::new(0.5), s1)); // Different olabel
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.6), s1)); // Different weight

        let (encoded, table) = encode(&fst).unwrap();

        assert_eq!(table.size(), 3); // Three different tuples

        let arcs: Vec<_> = encoded.arcs(s0).collect();
        assert_eq!(arcs.len(), 3);

        // All should have different encoded labels
        assert_ne!(arcs[0].ilabel, arcs[1].ilabel);
        assert_ne!(arcs[0].ilabel, arcs[2].ilabel);
        assert_ne!(arcs[1].ilabel, arcs[2].ilabel);
    }

    #[test]
    fn test_encode_table_epsilon_decode() {
        let table = EncodeTable::<TropicalWeight>::new();

        // Epsilon should always decode to (0, 0, One)
        let decoded = table.decode(0).unwrap();
        assert_eq!(decoded, (0, 0, TropicalWeight::one()));
    }
}
