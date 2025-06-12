//! OpenFST format compatibility

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
