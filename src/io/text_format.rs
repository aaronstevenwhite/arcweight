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
        writeln!(writer, "START\t{}", start)?;
    }

    // write all states (to preserve state count)
    for state in fst.states() {
        writeln!(writer, "STATE\t{}", state)?;
    }

    // write arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            write!(writer, "{}\t{}\t", state, arc.nextstate)?;

            // write symbols or labels
            if let Some(syms) = isyms {
                write!(writer, "{}\t", syms.find(arc.ilabel).unwrap_or("?"))?;
            } else {
                write!(writer, "{}\t", arc.ilabel)?;
            }

            if let Some(syms) = osyms {
                write!(writer, "{}\t", syms.find(arc.olabel).unwrap_or("?"))?;
            } else {
                write!(writer, "{}\t", arc.olabel)?;
            }

            writeln!(writer, "{}", arc.weight)?;
        }

        // write final states
        if let Some(weight) = fst.final_weight(state) {
            writeln!(writer, "FINAL\t{}\t{}", state, weight)?;
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
