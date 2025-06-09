//! Text format I/O for FSTs

use crate::fst::{Fst, MutableFst, StateId, Label};
use crate::semiring::Semiring;
use crate::arc::Arc;
use crate::utils::SymbolTable;
use crate::{Result, Error};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

/// Write FST in text format
pub fn write_text<W, F, P>(
    fst: &F,
    path: P,
    isyms: Option<&SymbolTable>,
    osyms: Option<&SymbolTable>,
) -> Result<()>
where
    W: Semiring,
    F: Fst<W>,
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;
    
    // write arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            write!(
                file,
                "{}\t{}\t",
                state,
                arc.nextstate
            )?;
            
            // write symbols or labels
            if let Some(syms) = isyms {
                write!(file, "{}\t", syms.find(arc.ilabel).unwrap_or("?"))?;
            } else {
                write!(file, "{}\t", arc.ilabel)?;
            }
            
            if let Some(syms) = osyms {
                write!(file, "{}\t", syms.find(arc.olabel).unwrap_or("?"))?;
            } else {
                write!(file, "{}\t", arc.olabel)?;
            }
            
            writeln!(file, "{}", arc.weight)?;
        }
        
        // write final states
        if let Some(weight) = fst.final_weight(state) {
            writeln!(file, "{}\t{}", state, weight)?;
        }
    }
    
    Ok(())
}

/// Read FST from text format
pub fn read_text<W, M, P>(
    path: P,
    isyms: Option<&SymbolTable>,
    osyms: Option<&SymbolTable>,
) -> Result<M>
where
    W: Semiring + FromStr,
    W::Err: std::error::Error + Send + Sync + 'static,
    M: MutableFst<W> + Default,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut fst = M::default();
    let mut state_map = HashMap::new();
    let mut start_set = false;
    
    // helper to get or create state
    let get_state = |state_map: &mut HashMap<StateId, StateId>, 
                     fst: &mut M, 
                     id: StateId| -> StateId {
        *state_map.entry(id).or_insert_with(|| fst.add_state())
    };
    
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        
        match parts.len() {
            2 => {
                // final state
                let state = parts[0].parse::<StateId>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                let weight = parts[1].parse::<W>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                
                let state = get_state(&mut state_map, &mut fst, state);
                fst.set_final(state, weight);
            }
            5 => {
                // arc
                let from = parts[0].parse::<StateId>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                let to = parts[1].parse::<StateId>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                
                let ilabel = if let Some(syms) = isyms {
                    syms.find_id(parts[2]).unwrap_or(0)
                } else {
                    parts[2].parse::<Label>()
                        .map_err(|e| Error::Serialization(e.to_string()))?
                };
                
                let olabel = if let Some(syms) = osyms {
                    syms.find_id(parts[3]).unwrap_or(0)
                } else {
                    parts[3].parse::<Label>()
                        .map_err(|e| Error::Serialization(e.to_string()))?
                };
                
                let weight = parts[4].parse::<W>()
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                
                let from = get_state(&mut state_map, &mut fst, from);
                let to = get_state(&mut state_map, &mut fst, to);
                
                if !start_set {
                    fst.set_start(from);
                    start_set = true;
                }
                
                fst.add_arc(from, Arc::new(ilabel, olabel, weight, to));
            }
            _ => continue, // skip malformed lines
        }
    }
    
    Ok(fst)
}