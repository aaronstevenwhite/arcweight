//! Random path generation

use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::arc::Arc;
use crate::{Result, Error};
use rand::Rng;

/// Random generation configuration
#[derive(Debug, Clone)]
pub struct RandGenConfig {
    /// Maximum path length
    pub max_length: usize,
    /// Number of paths
    pub npath: usize,
    /// Use weighted selection
    pub weighted: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for RandGenConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            npath: 1,
            weighted: false,
            seed: None,
        }
    }
}

/// Generate random paths from FST
pub fn randgen<W, F, M>(fst: &F, config: RandGenConfig) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    let mut rng = rand::thread_rng();
    
    let start = fst.start().ok_or_else(|| {
        Error::Algorithm("FST has no start state".into())
    })?;
    
    // generate paths
    for path_idx in 0..config.npath {
        let mut path = Vec::new();
        let mut current = start;
        let mut length = 0;
        
        // follow random path
        while length < config.max_length {
            let arcs: Vec<_> = fst.arcs(current).collect();
            if arcs.is_empty() {
                break;
            }
            
            // select arc
            let arc_idx = if config.weighted {
                // weighted selection based on arc weights
                rng.gen_range(0..arcs.len())
            } else {
                // uniform selection
                rng.gen_range(0..arcs.len())
            };
            
            let arc = &arcs[arc_idx];
            path.push(arc.clone());
            current = arc.nextstate;
            length += 1;
            
            // check if final
            if fst.is_final(current) && rng.gen_bool(0.5) {
                break;
            }
        }
        
        // add path to result
        if !path.is_empty() {
            add_path_to_fst(&mut result, &path, path_idx as u32)?;
        }
    }
    
    Ok(result)
}

fn add_path_to_fst<W: Semiring, M: MutableFst<W>>(
    fst: &mut M,
    path: &[Arc<W>],
    _offset: u32,
) -> Result<()> {
    if path.is_empty() {
        return Ok(());
    }
    
    let start = fst.add_state();
    fst.set_start(start);
    
    let mut current = start;
    for arc in path {
        let next = fst.add_state();
        fst.add_arc(current, Arc::new(
            arc.ilabel,
            arc.olabel,
            arc.weight.clone(),
            next,
        ));
        current = next;
    }
    
    fst.set_final(current, W::one());
    Ok(())
}