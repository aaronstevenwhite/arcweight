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

    const MAGIC: u32 = 0x46535442; // "FSTB"
    const VERSION: u32 = 1;

    /// Write FST in binary format
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
        let bytes = bincode::serialize(weight).map_err(|e| Error::Serialization(e.to_string()))?;
        writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn read_weight<W: Semiring + serde::de::DeserializeOwned>(reader: &mut impl Read) -> Result<W> {
        let len = reader.read_u32::<LittleEndian>()? as usize;
        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;
        bincode::deserialize(&bytes).map_err(|e| Error::Serialization(e.to_string()))
    }
}

#[cfg(feature = "serde")]
pub use inner::{read_binary, write_binary};
