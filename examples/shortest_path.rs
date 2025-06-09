//! Shortest path examples

use arcweight::prelude::*;

fn create_lattice() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    
    // create a lattice structure
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    let s4 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s4, TropicalWeight::one());
    
    // multiple paths with different weights
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
    
    fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(2.0), s3));
    fst.add_arc(s1, Arc::new(4, 4, TropicalWeight::new(1.0), s4));
    
    fst.add_arc(s2, Arc::new(5, 5, TropicalWeight::new(1.0), s3));
    fst.add_arc(s2, Arc::new(6, 6, TropicalWeight::new(3.0), s4));
    
    fst.add_arc(s3, Arc::new(7, 7, TropicalWeight::new(1.0), s4));
    
    fst
}

fn main() -> arcweight::Result<()> {
    let lattice = create_lattice();
    
    println!("Lattice FST: {} states, {} arcs", 
        lattice.num_states(), lattice.num_arcs_total());
    
    // find single shortest path
    let shortest: VectorFst<TropicalWeight> = shortest_path_single(&lattice)?;
    println!("\nSingle shortest path:");
    print_path(&shortest);
    
    // find k-shortest paths
    let config = ShortestPathConfig {
        nshortest: 3,
        unique: true,
        ..Default::default()
    };
    
    let k_shortest: VectorFst<TropicalWeight> = shortest_path(&lattice, config)?;
    println!("\n3-shortest paths FST: {} states, {} arcs", 
        k_shortest.num_states(), k_shortest.num_arcs_total());
    
    Ok(())
}

fn print_path<W: Semiring>(fst: &impl Fst<W>) {
    if let Some(start) = fst.start() {
        let mut visited = vec![false; fst.num_states()];
        print_path_from(fst, start, &mut visited, Vec::new(), W::one());
    }
}

fn print_path_from<W: Semiring>(
    fst: &impl Fst<W>,
    state: StateId,
    visited: &mut [bool],
    mut path: Vec<Label>,
    weight: W,
) {
    if visited[state as usize] {
        return;
    }
    visited[state as usize] = true;
    
    if fst.is_final(state) {
        let final_weight = fst.final_weight(state).unwrap();
        let total = weight.times(final_weight);
        print!("Path: ");
        for label in &path {
            print!("{} ", label);
        }
        println!("(weight: {})", total);
    }
    
    for arc in fst.arcs(state) {
        let mut new_path = path.clone();
        new_path.push(arc.ilabel);
        print_path_from(
            fst, 
            arc.nextstate, 
            visited, 
            new_path, 
            weight.times(&arc.weight)
        );
    }
    
    visited[state as usize] = false;
}