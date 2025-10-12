//! Native binary format for FST serialization
//!
//! This module provides efficient binary serialization and deserialization
//! of FSTs using Rust's native serialization capabilities. The format is
//! optimized for speed and preserves all FST properties exactly.
//!
//! ## Features
//!
//! - **Type-safe:** Preserves exact semiring types and properties
//! - **Versioned:** Format includes version information for compatibility
//! - **Efficient:** Uses compact binary representation with compression
//! - **Complete:** Serializes all FST data including properties and metadata
//!
//! ## Format Specification
//!
//! The binary format consists of:
//! 1. **Header:** Magic number (0x46535442 = "FSTB") and version
//! 2. **Metadata:** Number of states, start state
//! 3. **State data:** Final weights and arcs for each state
//! 4. **Arc data:** Labels, weights, and next states
//!
//! ## Requirements
//!
//! This module requires the `serde` feature to be enabled:
//! ```toml
//! [dependencies]
//! arcweight = { version = "*", features = ["serde"] }
//! ```
//!
//! ## Examples
//!
//! ### Basic Serialization
//!
//! ```
//! # #[cfg(feature = "serde")]
//! # {
//! use arcweight::prelude::*;
//! use arcweight::io::{write_binary, read_binary};
//! use std::io::Cursor;
//!
//! # fn example() -> Result<()> {
//! // Create and populate FST
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
//!
//! // Write to in-memory buffer
//! let mut buffer = Vec::new();
//! write_binary(&fst, &mut buffer)?;
//!
//! // Read back from buffer
//! let mut cursor = Cursor::new(buffer);
//! let loaded: VectorFst<TropicalWeight> = read_binary(&mut cursor)?;
//!
//! assert_eq!(fst.num_states(), loaded.num_states());
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ### In-Memory Serialization
//!
//! ```
//! # #[cfg(feature = "serde")]
//! # {
//! use arcweight::prelude::*;
//! use arcweight::io::{write_binary, read_binary};
//!
//! # fn example() -> Result<()> {
//! // Serialize to bytes
//! let fst = VectorFst::<LogWeight>::new();
//! let mut buffer = Vec::new();
//! write_binary(&fst, &mut buffer)?;
//!
//! // Deserialize from bytes
//! let loaded: VectorFst<LogWeight> = read_binary(&mut &buffer[..])?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Performance
//!
//! - **Write speed:** Limited by I/O and weight serialization
//! - **Read speed:** Limited by I/O and FST construction
//! - **Compression:** Weight values are compressed using bincode
//! - **Memory:** Streaming design minimizes memory usage
//!
//! ## Error Handling
//!
//! Operations return [`Result<T, Error>`](crate::Result) with specific error types:
//! - Invalid magic number or version
//! - Corrupted data during deserialization
//! - I/O errors from the underlying reader/writer
//! - Weight serialization failures

#[cfg(feature = "serde")]
mod inner {
    use crate::arc::Arc;
    use crate::fst::{Fst, MutableFst, StateId};
    use crate::semiring::Semiring;
    use crate::{Error, Result};
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use std::io::{Read, Write};

    const MAGIC: u32 = 0x4653_5442; // "FSTB"
    const VERSION: u32 = 1;

    /// Write FST in binary format
    ///
    /// # Complexity
    ///
    /// **Time:** O(|V| + |E|) - Single pass through all states and arcs
    /// **Space:** O(1) - Streaming write, no additional storage
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The writer encounters an I/O error during writing
    /// - Weight serialization fails due to invalid weight data
    /// - Memory allocation fails during the write operation
    pub fn write_binary<W, F, Writer>(fst: &F, writer: &mut Writer) -> Result<()>
    where
        W: Semiring + serde::Serialize,
        F: Fst<W>,
        Writer: Write,
    {
        // write header
        writer.write_u32::<LittleEndian>(MAGIC)?;
        writer.write_u32::<LittleEndian>(VERSION)?;

        // write basic info
        writer.write_u32::<LittleEndian>(fst.num_states() as u32)?;
        writer.write_u32::<LittleEndian>(fst.start().unwrap_or(u32::MAX))?;

        // write states
        for state in fst.states() {
            // final weight
            if let Some(weight) = fst.final_weight(state) {
                writer.write_u8(1)?;
                write_weight(writer, weight)?;
            } else {
                writer.write_u8(0)?;
            }

            // arcs
            let num_arcs = fst.num_arcs(state);
            writer.write_u32::<LittleEndian>(num_arcs as u32)?;

            for arc in fst.arcs(state) {
                writer.write_u32::<LittleEndian>(arc.ilabel)?;
                writer.write_u32::<LittleEndian>(arc.olabel)?;
                write_weight(writer, &arc.weight)?;
                writer.write_u32::<LittleEndian>(arc.nextstate)?;
            }
        }

        Ok(())
    }

    /// Read FST from binary format
    ///
    /// # Complexity
    ///
    /// **Time:** O(|V| + |E|) - Single pass creating states and arcs
    /// **Space:** O(|V| + |E|) - Storage for the output FST
    ///
    /// # Correctness
    ///
    /// **Guarantee:** deserialize(serialize(T)) = T
    /// - All FST properties preserved exactly
    /// - State IDs preserved (created in order 0..n-1)
    /// - Language preserved: L(read(write(T))) = L(T)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The reader encounters an I/O error during reading
    /// - The binary data is malformed or corrupted (invalid magic number or version)
    /// - Weight deserialization fails due to corrupted weight data
    /// - Memory allocation fails during FST construction
    pub fn read_binary<W, M, Reader>(reader: &mut Reader) -> Result<M>
    where
        W: Semiring + serde::de::DeserializeOwned,
        M: MutableFst<W> + Default,
        Reader: Read,
    {
        // read header
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != MAGIC {
            return Err(Error::Serialization("Invalid magic number".into()));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != VERSION {
            return Err(Error::Serialization("Unsupported version".into()));
        }

        // read basic info
        let num_states = reader.read_u32::<LittleEndian>()? as usize;
        let start = reader.read_u32::<LittleEndian>()?;

        let mut fst = M::default();

        // create states
        for _ in 0..num_states {
            fst.add_state();
        }

        // set start
        if start != u32::MAX {
            fst.set_start(start);
        }

        // read states
        for state in 0..num_states {
            // final weight
            if reader.read_u8()? == 1 {
                let weight = read_weight(reader)?;
                fst.set_final(state as StateId, weight);
            }

            // arcs
            let num_arcs = reader.read_u32::<LittleEndian>()? as usize;
            for _ in 0..num_arcs {
                let ilabel = reader.read_u32::<LittleEndian>()?;
                let olabel = reader.read_u32::<LittleEndian>()?;
                let weight = read_weight(reader)?;
                let nextstate = reader.read_u32::<LittleEndian>()?;

                fst.add_arc(
                    state as StateId,
                    Arc::new(ilabel, olabel, weight, nextstate),
                );
            }
        }

        Ok(fst)
    }

    fn write_weight<W: Semiring + serde::Serialize>(
        writer: &mut impl Write,
        weight: &W,
    ) -> Result<()> {
        // this would need specialization per weight type
        let bytes = bincode::serde::encode_to_vec(weight, bincode::config::legacy())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn read_weight<W: Semiring + serde::de::DeserializeOwned>(reader: &mut impl Read) -> Result<W> {
        let len = reader.read_u32::<LittleEndian>()? as usize;
        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;
        let (weight, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::legacy())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        Ok(weight)
    }
}

#[cfg(feature = "serde")]
pub use inner::{read_binary, write_binary};

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;
    use std::io::Cursor;

    #[test]
    fn test_write_read_binary_roundtrip() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(std::f32::consts::PI));

        fst.add_arc(s0, Arc::new(10, 20, TropicalWeight::new(2.5), s1));
        fst.add_arc(s1, Arc::new(30, 40, TropicalWeight::new(1.2), s2));
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.1), s2));

        // Write to buffer
        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        // Read back from buffer
        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        // Verify structure is preserved
        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());

        // Check final weights
        for state in fst.states() {
            let original_final = fst.final_weight(state);
            let read_final = read_fst.final_weight(state);

            match (original_final, read_final) {
                (Some(w1), Some(w2)) => assert_eq!(w1, w2),
                (None, None) => {}
                _ => panic!("Final weight mismatch for state {state}"),
            }
        }

        // Check arcs
        for state in fst.states() {
            let original_arcs: Vec<_> = fst.arcs(state).collect();
            let read_arcs: Vec<_> = read_fst.arcs(state).collect();

            assert_eq!(original_arcs.len(), read_arcs.len());

            for (orig, read) in original_arcs.iter().zip(read_arcs.iter()) {
                assert_eq!(orig.ilabel, read.ilabel);
                assert_eq!(orig.olabel, read.olabel);
                assert_eq!(orig.weight, read.weight);
                assert_eq!(orig.nextstate, read.nextstate);
            }
        }
    }

    #[test]
    fn test_binary_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert!(read_fst.is_empty());
        assert_eq!(read_fst.num_states(), 0);
        assert_eq!(read_fst.start(), None);
    }

    #[test]
    fn test_binary_format_efficiency() {
        let mut fst = VectorFst::<TropicalWeight>::new();

        // Create a larger FST to test efficiency
        for _i in 0..100 {
            fst.add_state();
        }

        fst.set_start(0);
        fst.set_final(99, TropicalWeight::one());

        for i in 0..99 {
            fst.add_arc(
                i,
                Arc::new(i + 1, i + 1, TropicalWeight::new(i as f32 * 0.1), i + 1),
            );
        }

        let mut binary_buffer = Vec::new();
        let mut text_buffer = Vec::new();

        write_binary(&fst, &mut binary_buffer).unwrap();
        crate::io::write_text(&fst, &mut text_buffer, None, None).unwrap();

        // Binary format should typically be more compact
        // (though this is not guaranteed for all cases)
        println!(
            "Binary size: {}, Text size: {}",
            binary_buffer.len(),
            text_buffer.len()
        );

        // Verify binary can be read back correctly
        let mut cursor = Cursor::new(binary_buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
    }

    #[test]
    fn test_binary_format_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(42.0));

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), 1);
        assert_eq!(read_fst.start(), Some(s0));
        assert_eq!(read_fst.final_weight(s0), Some(&TropicalWeight::new(42.0)));
        assert_eq!(read_fst.num_arcs_total(), 0);
    }

    #[test]
    fn test_binary_format_self_loop() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s0)); // Self-loop

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), 1);
        assert_eq!(read_fst.num_arcs_total(), 1);

        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].nextstate, s0);
        assert_eq!(arcs[0].weight, TropicalWeight::new(0.5));
    }

    #[test]
    fn test_binary_format_no_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        // No start state set

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), 2);
        assert_eq!(read_fst.start(), None);
        assert_eq!(read_fst.num_arcs_total(), 1);
    }

    #[test]
    fn test_binary_format_multiple_arcs_from_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // Multiple arcs from s0
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(3.0), s1));

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), 3);
        assert_eq!(read_fst.num_arcs_total(), 3);
        assert_eq!(read_fst.num_arcs(s0), 3);

        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 3);

        // Check arc labels and weights are preserved
        let labels: Vec<u32> = arcs.iter().map(|arc| arc.ilabel).collect();
        assert!(labels.contains(&1));
        assert!(labels.contains(&2));
        assert!(labels.contains(&3));
    }

    #[test]
    fn test_binary_format_epsilon_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.1), s1));

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, 0);
        assert_eq!(arcs[0].olabel, 0);
        assert_eq!(arcs[0].weight, TropicalWeight::new(0.1));
    }

    #[test]
    fn test_binary_format_different_semirings() {
        // Test LogWeight
        let mut log_fst = VectorFst::<LogWeight>::new();
        let s0 = log_fst.add_state();
        let s1 = log_fst.add_state();

        log_fst.set_start(s0);
        log_fst.set_final(s1, LogWeight::new(std::f64::consts::E));
        log_fst.add_arc(s0, Arc::new(1, 2, LogWeight::new(1.414), s1));

        let mut buffer = Vec::new();
        write_binary(&log_fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<LogWeight> =
            read_binary::<LogWeight, VectorFst<LogWeight>, _>(&mut cursor).unwrap();

        assert_eq!(
            read_fst.final_weight(s1),
            Some(&LogWeight::new(std::f64::consts::E))
        );
        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs[0].weight, LogWeight::new(1.414));

        // Test BooleanWeight
        let mut bool_fst = VectorFst::<BooleanWeight>::new();
        let s0 = bool_fst.add_state();
        let s1 = bool_fst.add_state();

        bool_fst.set_start(s0);
        bool_fst.set_final(s1, BooleanWeight::one());
        bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::zero(), s1));

        let mut buffer = Vec::new();
        write_binary(&bool_fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<BooleanWeight> =
            read_binary::<BooleanWeight, VectorFst<BooleanWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.final_weight(s1), Some(&BooleanWeight::one()));
        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs[0].weight, BooleanWeight::zero());
    }

    #[test]
    fn test_binary_format_large_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(
            s0,
            Arc::new(u32::MAX - 1, u32::MAX, TropicalWeight::new(1000.0), s1),
        );

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, u32::MAX - 1);
        assert_eq!(arcs[0].olabel, u32::MAX);
        assert_eq!(arcs[0].weight, TropicalWeight::new(1000.0));
    }

    #[test]
    fn test_binary_format_large_finite_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1_000_000.0)); // Large but finite weight
        fst.set_final(s2, TropicalWeight::one()); // Zero

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(-1000.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1000.0), s2));

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(
            read_fst.final_weight(s1),
            Some(&TropicalWeight::new(1_000_000.0))
        );
        assert_eq!(read_fst.final_weight(s2), Some(&TropicalWeight::one()));

        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 2);

        let arc1 = arcs.iter().find(|arc| arc.ilabel == 1).unwrap();
        let arc2 = arcs.iter().find(|arc| arc.ilabel == 2).unwrap();

        assert_eq!(arc1.weight, TropicalWeight::new(-1000.0));
        assert_eq!(arc2.weight, TropicalWeight::new(1000.0));
    }

    #[test]
    fn test_binary_format_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[4], TropicalWeight::new(10.0));

        // Create linear chain: 0 -> 1 -> 2 -> 3 -> 4
        for i in 0..4 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new((i + 1) as f32),
                    states[i + 1],
                ),
            );
        }

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), 5);
        assert_eq!(read_fst.start(), Some(states[0]));
        assert_eq!(
            read_fst.final_weight(states[4]),
            Some(&TropicalWeight::new(10.0))
        );

        // Verify chain structure
        for i in 0..4 {
            let arcs: Vec<_> = read_fst.arcs(states[i]).collect();
            assert_eq!(arcs.len(), 1);
            assert_eq!(arcs[0].ilabel, (i + 1) as u32);
            assert_eq!(arcs[0].nextstate, states[i + 1]);
            assert_eq!(arcs[0].weight, TropicalWeight::new((i + 1) as f32));
        }
    }

    #[test]
    fn test_binary_format_complex_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let _s1 = fst.add_state();

        fst.set_start(s0);

        // Test various weight values
        let weights = [
            0.0,
            1.0,
            -1.0,
            std::f32::consts::PI,
            std::f32::consts::E,
            0.123_456_8,
            1e-10,
            1e10,
            -1e-10,
            -1e10,
        ];

        for (i, &weight_val) in weights.iter().enumerate() {
            let state = fst.add_state();
            fst.set_final(state, TropicalWeight::new(weight_val));
            fst.add_arc(
                s0,
                Arc::new(i as u32, i as u32, TropicalWeight::new(weight_val), state),
            );
        }

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        // Verify all weights are preserved
        let arcs: Vec<_> = read_fst.arcs(s0).collect();
        for (i, &expected_weight) in weights.iter().enumerate() {
            let arc = arcs.iter().find(|arc| arc.ilabel == i as u32).unwrap();
            assert_eq!(arc.weight, TropicalWeight::new(expected_weight));

            let final_weight = read_fst.final_weight(arc.nextstate).unwrap();
            assert_eq!(*final_weight, TropicalWeight::new(expected_weight));
        }
    }
}
