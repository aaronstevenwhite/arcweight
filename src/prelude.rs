//! Prelude module for convenient imports

pub use num_traits::{One, Zero};

pub use crate::{
    // core types
    arc::{Arc, ArcIterator},
    fst::{Fst, MutableFst, ExpandedFst, StateId, Label, NO_STATE_ID, NO_LABEL},
    
    // fst implementations
    fst::{VectorFst, ConstFst, CompactFst},
    
    // semirings
    semiring::{
        Semiring, DivisibleSemiring, StarSemiring, NaturallyOrderedSemiring,
        TropicalWeight, ProbabilityWeight, BooleanWeight, LogWeight,
    },
    
    // algorithms
    algorithms::{
        compose, compose_default,
        concat,
        connect,
        determinize,
        minimize,
        reverse,
        shortest_path, shortest_path_single, ShortestPathConfig,
        closure, closure_plus,
        union,
        project_input, project_output,
        prune,
        remove_epsilons,
        topsort,
        weight_convert,
    },
    
    // utilities
    utils::{SymbolTable},
    
    // i/o
    io::{read_text, write_text, read_binary, write_binary, read_openfst, write_openfst},
    
    // error handling
    Error, Result,
};