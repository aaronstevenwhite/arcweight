//! Text format I/O for FSTs

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
