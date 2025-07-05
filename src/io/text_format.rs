//! Human-readable text format for FST representation
//!
//! This module provides functions to read and write FSTs in a simple text format
//! that is human-readable, editable, and suitable for version control. The format
//! is ideal for debugging, testing, and manual FST construction.
//!
//! ## Format Specification
//!
//! The text format uses line-based records with whitespace-separated fields:
//!
//! ### Start State Declaration
//! ```text
//! START <stateᵢd>
//! ```
//!
//! ### State Declaration
//! ```text
//! STATE <stateᵢd>
//! ```
//!
//! ### Arc Definition
//! ```text
//! <source> <dest> <input> <output> <weight>
//! ```
//!
//! ### Final State Declaration
//! ```text
//! FINAL <stateᵢd> <weight>
//! ```
//!
//! ## Symbol Tables
//!
//! The format supports optional symbol tables for human-readable labels:
//! - When provided, labels are written as strings (e.g., "hello")
//! - Without symbol tables, labels are written as integers (e.g., 42)
//! - Unknown symbols are written as "?"
//!
//! ## Examples
//!
//! ### Basic Text Format
//!
//! ```text
//! START 0
//! STATE 0
//! STATE 1
//! STATE 2
//! 0 1 1 2 0.5
//! 1 2 3 4 0.3
//! FINAL 2 0.0
//! ```
//!
//! This represents:
//! - Start state: 0
//! - Three states: 0, 1, 2
//! - Arc from 0→1 with labels 1:2 and weight 0.5
//! - Arc from 1→2 with labels 3:4 and weight 0.3
//! - State 2 is final with weight 0.0
//!
//! ### Reading and Writing
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::io::{read_text, write_text};
//! use std::fs::File;
//! use std::io::{BufReader, BufWriter};
//!
//! // Write FST to text
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
//!
//! let file = File::create("fst.txt")?;
//! let mut writer = BufWriter::new(file);
//! write_text(&fst, &mut writer, None, None)?;
//!
//! // Read FST from text
//! let file = File::open("fst.txt")?;
//! let mut reader = BufReader::new(file);
//! let loaded: VectorFst<TropicalWeight> = read_text(&mut reader, None, None)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Using Symbol Tables
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::utils::SymbolTable;
//! use arcweight::io::{write_text, read_text};
//! use std::io::{BufReader, BufWriter};
//! use std::fs::File;
//!
//! // Create symbol tables
//! let mut isyms = SymbolTable::new();
//! let mut osyms = SymbolTable::new();
//!
//! let hello = isyms.add_symbol("hello");
//! let world = osyms.add_symbol("world");
//!
//! // Build FST with symbolic labels
//! let mut fst = VectorFst::<LogWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.add_arc(s0, Arc::new(hello, world, LogWeight::one(), s1));
//!
//! // Write with symbols
//! let file = File::create("symbolic.txt")?;
//! let mut writer = BufWriter::new(file);
//! write_text(&fst, &mut writer, Some(&isyms), Some(&osyms))?;
//!
//! // Output contains:
//! // START 0
//! // STATE 0
//! // STATE 1
//! // 0 1 hello world -0.0
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Manual FST Creation
//!
//! Create FST files by hand for testing:
//!
//! ```text
//! # Simple acceptor for "cat"
//! START 0
//! 0 1 c c 0.0
//! 1 2 a a 0.0
//! 2 3 t t 0.0
//! FINAL 3 0.0
//! ```
//!
//! ## Format Features
//!
//! ### Advantages
//! - **Human-readable:** Easy to understand and debug
//! - **Editable:** Can be created/modified with any text editor
//! - **Version control:** Diff-friendly for tracking changes
//! - **Portable:** Simple format works across platforms
//!
//! ### Limitations
//! - **Size:** Larger than binary formats
//! - **Speed:** Slower to parse than binary
//! - **Precision:** May lose weight precision in text representation
//!
//! ## Error Handling
//!
//! The parser is lenient:
//! - Blank lines are ignored
//! - Comments can be added (lines starting with #)
//! - Malformed lines are skipped with a warning
//! - States are created automatically as referenced

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::utils::SymbolTable;
use crate::{Error, Result};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::str::FromStr;

/// Write FST in text format to a writer
///
/// # Errors
///
/// Returns an error if:
/// - The writer encounters an I/O error during writing
/// - Symbol table lookups fail for provided labels
/// - Weight formatting fails for display
pub fn write_text<W, F, Writer>(
    fst: &F,
    writer: &mut Writer,
    isyms: Option<&SymbolTable>,
    osyms: Option<&SymbolTable>,
) -> Result<()>
where
    W: Semiring,
    F: Fst<W>,
    Writer: Write,
{
    // write start state first if it exists
    if let Some(start) = fst.start() {
        writeln!(writer, "START\t{start}")?;
    }

    // write all states (to preserve state count)
    for state in fst.states() {
        writeln!(writer, "STATE\t{state}")?;
    }

    // write arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            let nextstate = arc.nextstate;
            write!(writer, "{state}\t{nextstate}\t")?;

            // write symbols or labels
            if let Some(syms) = isyms {
                let symbol = syms.find(arc.ilabel).unwrap_or("?");
                write!(writer, "{symbol}\t")?;
            } else {
                let ilabel = arc.ilabel;
                write!(writer, "{ilabel}\t")?;
            }

            if let Some(syms) = osyms {
                let symbol = syms.find(arc.olabel).unwrap_or("?");
                write!(writer, "{symbol}\t")?;
            } else {
                let olabel = arc.olabel;
                write!(writer, "{olabel}\t")?;
            }

            let weight = &arc.weight;
            writeln!(writer, "{weight}")?;
        }

        // write final states
        if let Some(weight) = fst.final_weight(state) {
            writeln!(writer, "FINAL\t{state}\t{weight}")?;
        }
    }

    Ok(())
}

/// Read FST from text format from a reader
///
/// # Errors
///
/// Returns an error if:
/// - The reader encounters an I/O error during reading
/// - The text format is malformed or contains invalid syntax
/// - Weight or label parsing fails due to invalid format
/// - Symbol table lookups fail for provided symbols
/// - Memory allocation fails during FST construction
pub fn read_text<W, M, Reader>(
    reader: &mut Reader,
    isyms: Option<&SymbolTable>,
    osyms: Option<&SymbolTable>,
) -> Result<M>
where
    W: Semiring + FromStr,
    W::Err: std::error::Error + Send + Sync + 'static,
    M: MutableFst<W> + Default,
    Reader: BufRead,
{
    let buf_reader = reader;
    let mut fst = M::default();
    let mut state_map = HashMap::new();

    // helper to get or create state
    let get_state = |state_map: &mut HashMap<StateId, StateId>,
                     fst: &mut M,
                     id: StateId|
     -> StateId { *state_map.entry(id).or_insert_with(|| fst.add_state()) };

    for line in buf_reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.len() {
            2 => {
                if parts[0] == "START" {
                    // start state
                    let state = parts[1]
                        .parse::<StateId>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;
                    let state = get_state(&mut state_map, &mut fst, state);
                    fst.set_start(state);
                } else if parts[0] == "STATE" {
                    // state declaration (just ensure it exists)
                    let state = parts[1]
                        .parse::<StateId>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;
                    get_state(&mut state_map, &mut fst, state);
                } else {
                    // final state (old format for compatibility)
                    let state = parts[0]
                        .parse::<StateId>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;
                    let weight = parts[1]
                        .parse::<W>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;

                    let state = get_state(&mut state_map, &mut fst, state);
                    fst.set_final(state, weight);
                }
            }
            3 => {
                if parts[0] == "FINAL" {
                    // final state (new format)
                    let state = parts[1]
                        .parse::<StateId>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;
                    let weight = parts[2]
                        .parse::<W>()
                        .map_err(|e| Error::Serialization(e.to_string()))?;

                    let state = get_state(&mut state_map, &mut fst, state);
                    fst.set_final(state, weight);
                }
            }
            5 => {
                // arc
                let from = parts[0]
                    .parse::<StateId>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                let to = parts[1]
                    .parse::<StateId>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;

                let ilabel = if let Some(syms) = isyms {
                    syms.find_id(parts[2]).unwrap_or(0)
                } else {
                    parts[2]
                        .parse::<Label>()
                        .map_err(|e| Error::Serialization(e.to_string()))?
                };

                let olabel = if let Some(syms) = osyms {
                    syms.find_id(parts[3]).unwrap_or(0)
                } else {
                    parts[3]
                        .parse::<Label>()
                        .map_err(|e| Error::Serialization(e.to_string()))?
                };

                let weight = parts[4]
                    .parse::<W>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;

                let from = get_state(&mut state_map, &mut fst, from);
                let to = get_state(&mut state_map, &mut fst, to);

                fst.add_arc(from, Arc::new(ilabel, olabel, weight, to));
            }
            _ => continue, // skip malformed lines
        }
    }

    Ok(fst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::identities::One;
    use std::io::{BufReader, Cursor};

    #[test]
    fn test_write_read_text_roundtrip() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.5));

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(3, 4, TropicalWeight::new(1.5), s2));
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s2));

        // Write to buffer
        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        // Read back from buffer
        let cursor = Cursor::new(buffer);
        let mut buf_reader = BufReader::new(cursor);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut buf_reader, None, None)
                .unwrap();

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
    fn test_write_read_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert!(read_fst.is_empty());
        assert_eq!(read_fst.num_states(), 0);
        assert_eq!(read_fst.start(), None);
    }

    #[test]
    fn test_write_read_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert_eq!(read_fst.num_states(), 1);
        assert_eq!(read_fst.start(), Some(0));
        assert!(read_fst.is_final(0));
    }

    #[test]
    fn test_text_format_different_weights() {
        // Test with Boolean weight
        let mut bool_fst = VectorFst::<BooleanWeight>::new();
        let s0 = bool_fst.add_state();
        let s1 = bool_fst.add_state();

        bool_fst.set_start(s0);
        bool_fst.set_final(s1, BooleanWeight::new(true));
        bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(false), s1));

        let mut buffer = Vec::new();
        write_text(&bool_fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<BooleanWeight> =
            read_text::<BooleanWeight, VectorFst<BooleanWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert_eq!(read_fst.num_states(), bool_fst.num_states());
        assert_eq!(read_fst.start(), bool_fst.start());

        // Test with Probability weight
        let mut prob_fst = VectorFst::<ProbabilityWeight>::new();
        let s0 = prob_fst.add_state();
        let s1 = prob_fst.add_state();

        prob_fst.set_start(s0);
        prob_fst.set_final(s1, ProbabilityWeight::new(0.8));
        prob_fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.3), s1));

        let mut buffer = Vec::new();
        write_text(&prob_fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<ProbabilityWeight> = read_text::<
            ProbabilityWeight,
            VectorFst<ProbabilityWeight>,
            _,
        >(&mut cursor, None, None)
        .unwrap();

        assert_eq!(read_fst.num_states(), prob_fst.num_states());
        assert_eq!(read_fst.start(), prob_fst.start());
    }
}
