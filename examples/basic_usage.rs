//! Basic usage examples

use arcweight::prelude::*;

fn main() -> arcweight::Result<()> {
    // create a simple acceptor
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    
    // set start and final states
    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::one());
    
    // add arcs (transitions)
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.5), s1));
    fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(2.5), s2));
    fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(3.5), s3));
    
    // find shortest path
    let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst)?;
    
    println!("Original FST has {} states", fst.num_states());
    println!("Shortest path FST has {} states", shortest.num_states());
    
    // print the shortest path
    if let Some(start) = shortest.start() {
        let mut current = start;
        let mut path_weight = TropicalWeight::one();
        
        print!("Shortest path: {}", current);
        
        while !shortest.is_final(current) {
            for arc in shortest.arcs(current) {
                print!(" --{}/{}:{}-> {}", 
                    arc.ilabel, arc.olabel, arc.weight, arc.nextstate);
                path_weight = path_weight.times(&arc.weight);
                current = arc.nextstate;
                break; // shortest path has only one arc per state
            }
        }
        
        if let Some(final_weight) = shortest.final_weight(current) {
            path_weight = path_weight.times(final_weight);
        }
        
        println!("\nTotal weight: {}", path_weight);
    }
    
    Ok(())
}