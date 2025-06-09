//! Binary format I/O for FSTs

use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::arc::Arc;
use crate::{Result, Error};
use std::io::{Read, Write};
use std::path::Path;
use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

const MAGIC: u32 = 0x46535442; // "FSTB"
const VERSION: u32 = 1;

/// Write FST in binary format
pub fn write_binary<W, F, P>(fst: &F, path: P) -> Result<()>
where
    W: Semiring + serde::Serialize,
    F: Fst<W>,
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;
    
    // write header
    file.write_u32::<LittleEndian>(MAGIC)?;
    file.write_u32::<LittleEndian>(VERSION)?;
    
    // write basic info
    file.write_u32::<LittleEndian>(fst.num_states() as u32)?;
    file.write_u32::<LittleEndian>(fst.start().unwrap_or(u32::MAX))?;
    
    // write states
    for state in fst.states() {
        // final weight
        if let Some(weight) = fst.final_weight(state) {
            file.write_u8(1)?;
            write_weight(&mut file, weight)?;
        } else {
            file.write_u8(0)?;
        }
        
        // arcs
        let num_arcs = fst.num_arcs(state);
        file.write_u32::<LittleEndian>(num_arcs as u32)?;
        
        for arc in fst.arcs(state) {
            file.write_u32::<LittleEndian>(arc.ilabel)?;
            file.write_u32::<LittleEndian>(arc.olabel)?;
            write_weight(&mut file, &arc.weight)?;
            file.write_u32::<LittleEndian>(arc.nextstate)?;
        }
    }
    
    Ok(())
}

/// Read FST from binary format
pub fn read_binary<W, M, P>(path: P) -> Result<M>
where
    W: Semiring + serde::de::DeserializeOwned,
    M: MutableFst<W> + Default,
    P: AsRef<Path>,
{
    let mut file = File::open(path)?;
    
    // read header
    let magic = file.read_u32::<LittleEndian>()?;
    if magic != MAGIC {
        return Err(Error::Serialization("Invalid magic number".into()));
    }
    
    let version = file.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(Error::Serialization("Unsupported version".into()));
    }
    
    // read basic info
    let num_states = file.read_u32::<LittleEndian>()? as usize;
    let start = file.read_u32::<LittleEndian>()?;
    
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
        if file.read_u8()? == 1 {
            let weight = read_weight(&mut file)?;
            fst.set_final(state as StateId, weight);
        }
        
        // arcs
        let num_arcs = file.read_u32::<LittleEndian>()? as usize;
        for _ in 0..num_arcs {
            let ilabel = file.read_u32::<LittleEndian>()?;
            let olabel = file.read_u32::<LittleEndian>()?;
            let weight = read_weight(&mut file)?;
            let nextstate = file.read_u32::<LittleEndian>()?;
            
            fst.add_arc(state as StateId, Arc::new(ilabel, olabel, weight, nextstate));
        }
    }
    
    Ok(fst)
}

fn write_weight<W: Semiring + serde::Serialize>(writer: &mut impl Write, weight: &W) -> Result<()> {
    // this would need specialization per weight type
    let bytes = bincode::serialize(weight)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
    writer.write_all(&bytes)?;
    Ok(())
}

fn read_weight<W: Semiring + serde::de::DeserializeOwned>(reader: &mut impl Read) -> Result<W> {
    let len = reader.read_u32::<LittleEndian>()? as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    bincode::deserialize(&bytes)
        .map_err(|e| Error::Serialization(e.to_string()))
}