//! Speech recognition example using WFSTs

use arcweight::prelude::*;
use std::collections::HashMap;

/// Simple acoustic model FST
fn create_acoustic_model() -> VectorFst<ProbabilityWeight> {
    let mut fst = VectorFst::new();
    
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s3, ProbabilityWeight::one());
    
    // acoustic observations to phonemes with probabilities
    // observation 1 -> phoneme /h/ with high probability
    fst.add_arc(s0, Arc::new(1, 100, ProbabilityWeight::new(0.9), s1));
    // observation 1 -> phoneme /k/ with low probability  
    fst.add_arc(s0, Arc::new(1, 101, ProbabilityWeight::new(0.1), s1));
    
    // observation 2 -> phoneme /e/
    fst.add_arc(s1, Arc::new(2, 102, ProbabilityWeight::new(0.8), s2));
    // observation 2 -> phoneme /a/
    fst.add_arc(s1, Arc::new(2, 103, ProbabilityWeight::new(0.2), s2));
    
    // observation 3 -> phoneme /l/
    fst.add_arc(s2, Arc::new(3, 104, ProbabilityWeight::new(0.85), s3));
    // observation 3 -> phoneme /r/
    fst.add_arc(s2, Arc::new(3, 105, ProbabilityWeight::new(0.15), s3));
    
    fst
}

/// Pronunciation lexicon FST
fn create_lexicon() -> VectorFst<ProbabilityWeight> {
    let mut fst = VectorFst::new();
    
    let s0 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s0, ProbabilityWeight::one());
    
    // "hello" pronunciation: /h/ /e/ /l/
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    
    fst.add_arc(s0, Arc::new(100, 200, ProbabilityWeight::one(), s1)); // /h/ -> h
    fst.add_arc(s1, Arc::new(102, 0, ProbabilityWeight::one(), s2));   // /e/ -> ε
    fst.add_arc(s2, Arc::new(104, 0, ProbabilityWeight::one(), s3));   // /l/ -> ε
    fst.add_arc(s3, Arc::epsilon(ProbabilityWeight::one(), s0));       // "hello" complete
    
    // "call" pronunciation: /k/ /a/ /l/
    let s4 = fst.add_state();
    let s5 = fst.add_state();
    let s6 = fst.add_state();
    
    fst.add_arc(s0, Arc::new(101, 201, ProbabilityWeight::one(), s4)); // /k/ -> c
    fst.add_arc(s4, Arc::new(103, 0, ProbabilityWeight::one(), s5));   // /a/ -> ε
    fst.add_arc(s5, Arc::new(104, 0, ProbabilityWeight::one(), s6));   // /l/ -> ε
    fst.add_arc(s6, Arc::epsilon(ProbabilityWeight::one(), s0));       // "call" complete
    
    fst
}

/// Simple language model FST
fn create_language_model() -> VectorFst<ProbabilityWeight> {
    let mut fst = VectorFst::new();
    
    let s0 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s0, ProbabilityWeight::one());
    
    // "hello" is more likely than "call"
    fst.add_arc(s0, Arc::new(200, 200, ProbabilityWeight::new(0.7), s0)); // hello
    fst.add_arc(s0, Arc::new(201, 201, ProbabilityWeight::new(0.3), s0)); // call
    
    fst
}

fn main() -> arcweight::Result<()> {
    println!("Building speech recognition components...");
    
    let acoustic = create_acoustic_model();
    let lexicon = create_lexicon();
    let language = create_language_model();
    
    println!("Acoustic model: {} states", acoustic.num_states());
    println!("Lexicon: {} states", lexicon.num_states());
    println!("Language model: {} states", language.num_states());
    
    // compose A ∘ L ∘ G
    println!("\nComposing models...");
    let al: VectorFst<ProbabilityWeight> = compose_default(&acoustic, &lexicon)?;
    let alg: VectorFst<ProbabilityWeight> = compose_default(&al, &language)?;
    
    println!("Composed A∘L: {} states", al.num_states());
    println!("Composed A∘L∘G: {} states", alg.num_states());
    
    // find best path (most likely word sequence)
    println!("\nFinding best hypothesis...");
    
    // convert to tropical for shortest path
    let tropical_alg: VectorFst<TropicalWeight> = weight_convert(&alg, |w: &ProbabilityWeight| {
        TropicalWeight::new(-(*w.value() as f32).ln())
    })?;
    
    let best: VectorFst<TropicalWeight> = shortest_path_single(&tropical_alg)?;
    
    // extract and print the recognized word
    let word_map: HashMap<Label, &str> = HashMap::from([
        (200, "hello"),
        (201, "call"),
    ]);
    
    if let Some(start) = best.start() {
        let mut current = start;
        print!("Recognized: ");
        
        while !best.is_final(current) {
            for arc in best.arcs(current) {
                if arc.olabel != 0 {
                    if let Some(word) = word_map.get(&arc.olabel) {
                        print!("{} ", word);
                    }
                }
                current = arc.nextstate;
                break;
            }
        }
        println!();
    }
    
    Ok(())
}