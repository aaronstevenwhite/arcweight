//! FST composition example

use arcweight::prelude::*;

fn create_input_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    
    // "ab" -> "AB"
    fst.add_arc(s0, Arc::new(1, 10, TropicalWeight::new(0.0), s1)); // a -> A
    fst.add_arc(s1, Arc::new(2, 11, TropicalWeight::new(0.0), s2)); // b -> B
    
    fst
}

fn create_transform_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    
    let s0 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s0, TropicalWeight::one());
    
    // uppercase to doubled
    fst.add_arc(s0, Arc::new(10, 100, TropicalWeight::new(1.0), s0)); // A -> AA
    fst.add_arc(s0, Arc::new(11, 101, TropicalWeight::new(1.0), s0)); // B -> BB
    
    fst
}

fn main() -> arcweight::Result<()> {
    let input = create_input_fst();
    let transform = create_transform_fst();
    
    // compose: "ab" -> "AB" -> "AABB"
    let composed: VectorFst<TropicalWeight> = compose_default(&input, &transform)?;
    
    println!("Input FST: {} states, {} arcs", 
        input.num_states(), input.num_arcs_total());
    println!("Transform FST: {} states, {} arcs", 
        transform.num_states(), transform.num_arcs_total());
    println!("Composed FST: {} states, {} arcs", 
        composed.num_states(), composed.num_arcs_total());
    
    // find and print the path
    let path: VectorFst<TropicalWeight> = shortest_path_single(&composed)?;
    
    if let Some(start) = path.start() {
        let mut current = start;
        print!("Path: ");
        
        while !path.is_final(current) {
            for arc in path.arcs(current) {
                print!("{}/{} ", arc.ilabel, arc.olabel);
                current = arc.nextstate;
                break;
            }
        }
        println!();
    }
    
    Ok(())
}