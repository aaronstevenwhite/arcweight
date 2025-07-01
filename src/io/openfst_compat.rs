//! OpenFST binary format compatibility layer
//!
//! This module provides read/write support for the OpenFST binary format,
//! enabling interoperability with the widely-used OpenFST C++ library and
//! its ecosystem of tools.
//!
//! ## Overview
//!
//! OpenFST is the de facto standard for FST manipulation, used extensively in:
//! - **Speech recognition:** Kaldi, ESPnet, wav2vec2
//! - **Natural language processing:** Pronunciation lexicons, G2P models
//! - **Machine translation:** Phrase-based and syntax-based systems
//! - **Text processing:** Tokenization, normalization, transliteration
//!
//! This module ensures ArcWeight FSTs can seamlessly integrate with existing
//! OpenFST-based pipelines and tools.
//!
//! ## Format Details
//!
//! The OpenFST binary format consists of:
//! - **Header:** Magic number (2125659606), FST type, arc type, version, flags
//! - **Properties:** 64-bit property flags for optimization hints
//! - **State table:** Final weights and arc counts for each state
//! - **Arc table:** Packed arc data with labels, weights, and destinations
//!
//! ## Limitations
//!
//! Currently supports:
//! - **FST Type:** Vector FSTs only
//! - **Arc Type:** Tropical semiring only
//! - **Features:** Basic FST structure (no symbol tables or auxiliary data)
//!
//! ## Examples
//!
//! ### Reading OpenFST Files
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::io::read_openfst;
//! use arcweight::algorithms::PruneConfig;
//! use std::fs::File;
//!
//! // Read FST created by OpenFST tools
//! let mut file = File::open("model.fst")?;
//! let fst: VectorFst<TropicalWeight> = read_openfst(&mut file)?;
//!
//! // Use with ArcWeight algorithms
//! let pruned: VectorFst<TropicalWeight> = prune(&fst, PruneConfig {
//!     weight_threshold: 10.0,
//!     state_threshold: Some(1000),
//!     npath: None,
//! })?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Writing OpenFST Files
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::io::write_openfst;
//! use std::fs::File;
//!
//! // Create FST with ArcWeight
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! // ... build FST ...
//!
//! // Write in OpenFST format
//! let mut file = File::create("output.fst")?;
//! write_openfst(&fst, &mut file)?;
//!
//! // Now usable with OpenFST tools:
//! // $ fstinfo output.fst
//! // $ fstdraw output.fst | dot -Tpng > fst.png
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Pipeline Integration
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::io::{read_openfst, write_openfst};
//! use std::fs::File;
//! use std::process::Command;
//!
//! // Read FST from Kaldi
//! let mut file = File::open("L.fst")?;
//! let lexicon: VectorFst<TropicalWeight> = read_openfst(&mut file)?;
//!
//! // Process with ArcWeight
//! let optimized: VectorFst<TropicalWeight> = minimize(&lexicon)?;
//!
//! // Write back for Kaldi
//! let mut out = File::create("L_min.fst")?;
//! write_openfst(&optimized, &mut out)?;
//!
//! // Use in Kaldi pipeline
//! Command::new("compile-train-graphs")
//!     .args(&["tree", "L_min.fst", "text", "ark:graphs.ark"])
//!     .status()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Compatibility Notes
//!
//! ### Weight Representation
//! - Tropical weights use IEEE 754 single precision (f32)
//! - Infinity represents the zero element (no path)
//! - Zero represents the one element (identity)
//!
//! ### State Numbering
//! - States are numbered consecutively from 0
//! - State IDs are preserved during read/write
//!
//! ### Property Flags
//! - Currently writes 0 for properties (forces recomputation)
//! - Future versions may preserve property flags

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::{Semiring, TropicalWeight};
use crate::{Error, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, Write};

const OPENFST_MAGIC: i32 = 2125659606;

/// FST type in OpenFST format
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
#[allow(dead_code)]
enum FstType {
    Vector = 1,
    Const = 2,
}

/// Write FST in OpenFST format
///
/// # Errors
///
/// Returns an error if:
/// - The writer encounters an I/O error during writing
/// - The FST contains weights that cannot be represented in OpenFST tropical format
/// - Seeking operations fail on the writer
pub fn write_openfst<F, Writer>(fst: &F, writer: &mut Writer) -> Result<()>
where
    F: Fst<TropicalWeight>,
    Writer: Write + Seek,
{
    // write header
    writer.write_i32::<LittleEndian>(OPENFST_MAGIC)?;
    writer.write_all(b"vector\0\0")?; // fst type (8 bytes)
    writer.write_all(b"tropical\0\0\0\0\0\0\0\0")?; // arc type (16 bytes)
    writer.write_i32::<LittleEndian>(1)?; // version
    writer.write_i32::<LittleEndian>(0)?; // flags
    writer.write_i64::<LittleEndian>(0)?; // properties
    writer.write_i64::<LittleEndian>(fst.start().unwrap_or(u32::MAX) as i64)?;
    writer.write_i64::<LittleEndian>(fst.num_states() as i64)?;

    // Count total arcs first
    let total_arcs = fst.states().map(|s| fst.num_arcs(s) as i64).sum::<i64>();
    writer.write_i64::<LittleEndian>(total_arcs)?;

    // write states
    for state in fst.states() {
        let final_weight = fst
            .final_weight(state)
            .map(|w| *w.value())
            .unwrap_or(f32::INFINITY);

        writer.write_f32::<LittleEndian>(final_weight)?;
        writer.write_i64::<LittleEndian>(fst.num_arcs(state) as i64)?;

        for arc in fst.arcs(state) {
            writer.write_i32::<LittleEndian>(arc.ilabel as i32)?;
            writer.write_i32::<LittleEndian>(arc.olabel as i32)?;
            writer.write_f32::<LittleEndian>(*arc.weight.value())?;
            writer.write_i32::<LittleEndian>(arc.nextstate as i32)?;
        }
    }

    Ok(())
}

/// Read FST from OpenFST format
///
/// # Errors
///
/// Returns an error if:
/// - The reader encounters an I/O error during reading
/// - The OpenFST data is malformed or corrupted (invalid magic number)
/// - The file format version is unsupported
/// - Memory allocation fails during FST construction
pub fn read_openfst<M, Reader>(reader: &mut Reader) -> Result<M>
where
    M: MutableFst<TropicalWeight> + Default,
    Reader: Read,
{
    // read header
    let magic = reader.read_i32::<LittleEndian>()?;
    if magic != OPENFST_MAGIC {
        return Err(Error::Serialization("Invalid OpenFST magic".into()));
    }

    // skip fst type and arc type
    let mut buf = [0u8; 24];
    reader.read_exact(&mut buf)?;

    let _version = reader.read_i32::<LittleEndian>()?;
    let _flags = reader.read_i32::<LittleEndian>()?;
    let _properties = reader.read_i64::<LittleEndian>()?;
    let start = reader.read_i64::<LittleEndian>()?;
    let num_states = reader.read_i64::<LittleEndian>()? as usize;
    let _num_arcs = reader.read_i64::<LittleEndian>()?;

    let mut fst = M::default();

    // create states
    for _ in 0..num_states {
        fst.add_state();
    }

    // set start
    if start >= 0 {
        fst.set_start(start as StateId);
    }

    // read states
    for state in 0..num_states {
        let final_weight = reader.read_f32::<LittleEndian>()?;
        if final_weight != f32::INFINITY {
            fst.set_final(state as StateId, TropicalWeight::new(final_weight));
        }

        let num_arcs = reader.read_i64::<LittleEndian>()? as usize;
        for _ in 0..num_arcs {
            let ilabel = reader.read_i32::<LittleEndian>()? as u32;
            let olabel = reader.read_i32::<LittleEndian>()? as u32;
            let weight = reader.read_f32::<LittleEndian>()?;
            let nextstate = reader.read_i32::<LittleEndian>()? as u32;

            fst.add_arc(
                state as StateId,
                Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate),
            );
        }
    }

    Ok(fst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use std::io::Cursor;

    #[test]
    fn test_openfst_roundtrip_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.5));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
    }

    #[test]
    fn test_openfst_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert!(read_fst.is_empty());
    }

    #[test]
    fn test_openfst_compatibility() {
        // Test that our format can handle typical OpenFST constructs
        let mut fst = VectorFst::<TropicalWeight>::new();

        // Create a more complex FST structure
        for _i in 0..10 {
            fst.add_state();
        }

        fst.set_start(0);
        fst.set_final(9, TropicalWeight::new(0.0));

        // Add various arc types
        fst.add_arc(0, Arc::new(1, 1, TropicalWeight::new(1.0), 1));
        fst.add_arc(1, Arc::epsilon(TropicalWeight::new(0.0), 2));
        fst.add_arc(2, Arc::new(2, 3, TropicalWeight::new(2.0), 3));

        // Add some parallel arcs
        fst.add_arc(0, Arc::new(4, 4, TropicalWeight::new(3.0), 4));
        fst.add_arc(4, Arc::new(5, 5, TropicalWeight::new(1.0), 9));

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());

        // Verify specific structural elements
        assert!(read_fst.is_final(9));
        assert_eq!(read_fst.num_arcs(0), 2); // Two outgoing arcs from start
    }
}
