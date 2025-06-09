//! OpenFST format compatibility

use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::{Semiring, TropicalWeight};
use crate::arc::Arc;
use crate::{Result, Error};
use std::io::{Read, Write, Seek};
use std::path::Path;
use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

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
pub fn write_openfst<F, P>(fst: &F, path: P) -> Result<()>
where
    F: Fst<TropicalWeight>,
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;
    
    // write header
    file.write_i32::<LittleEndian>(OPENFST_MAGIC)?;
    file.write_all(b"vector\0\0")?; // fst type
    file.write_all(b"tropical\0\0\0\0\0\0\0\0")?; // arc type
    file.write_i32::<LittleEndian>(1)?; // version
    file.write_i32::<LittleEndian>(0)?; // flags
    file.write_i64::<LittleEndian>(0)?; // properties
    file.write_i64::<LittleEndian>(fst.start().unwrap_or(u32::MAX) as i64)?;
    file.write_i64::<LittleEndian>(fst.num_states() as i64)?;
    file.write_i64::<LittleEndian>(0)?; // num arcs (filled later)
    
    let mut total_arcs = 0i64;
    
    // write states
    for state in fst.states() {
        let final_weight = fst.final_weight(state)
            .map(|w| *w.value())
            .unwrap_or(f32::INFINITY);
        
        file.write_f32::<LittleEndian>(final_weight)?;
        file.write_i64::<LittleEndian>(fst.num_arcs(state) as i64)?;
        
        for arc in fst.arcs(state) {
            file.write_i32::<LittleEndian>(arc.ilabel as i32)?;
            file.write_i32::<LittleEndian>(arc.olabel as i32)?;
            file.write_f32::<LittleEndian>(*arc.weight.value())?;
            file.write_i32::<LittleEndian>(arc.nextstate as i32)?;
            total_arcs += 1;
        }
    }
    
    // update arc count
    file.seek(std::io::SeekFrom::Start(48))?;
    file.write_i64::<LittleEndian>(total_arcs)?;
    
    Ok(())
}

/// Read FST from OpenFST format
pub fn read_openfst<M, P>(path: P) -> Result<M>
where
    M: MutableFst<TropicalWeight> + Default,
    P: AsRef<Path>,
{
    let mut file = File::open(path)?;
    
    // read header
    let magic = file.read_i32::<LittleEndian>()?;
    if magic != OPENFST_MAGIC {
        return Err(Error::Serialization("Invalid OpenFST magic".into()));
    }
    
    // skip fst type and arc type
    let mut buf = [0u8; 24];
    file.read_exact(&mut buf)?;
    
    let _version = file.read_i32::<LittleEndian>()?;
    let _flags = file.read_i32::<LittleEndian>()?;
    let _properties = file.read_i64::<LittleEndian>()?;
    let start = file.read_i64::<LittleEndian>()?;
    let num_states = file.read_i64::<LittleEndian>()? as usize;
    let _num_arcs = file.read_i64::<LittleEndian>()?;
    
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
        let final_weight = file.read_f32::<LittleEndian>()?;
        if final_weight != f32::INFINITY {
            fst.set_final(state as StateId, TropicalWeight::new(final_weight));
        }
        
        let num_arcs = file.read_i64::<LittleEndian>()? as usize;
        for _ in 0..num_arcs {
            let ilabel = file.read_i32::<LittleEndian>()? as u32;
            let olabel = file.read_i32::<LittleEndian>()? as u32;
            let weight = file.read_f32::<LittleEndian>()?;
            let nextstate = file.read_i32::<LittleEndian>()? as u32;
            
            fst.add_arc(state as StateId, Arc::new(
                ilabel,
                olabel,
                TropicalWeight::new(weight),
                nextstate
            ));
        }
    }
    
    Ok(fst)
}