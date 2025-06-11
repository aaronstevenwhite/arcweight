//! Prelude module for convenient imports

pub use num_traits::{One, Zero};

pub use crate::{
    // algorithms
    algorithms::{
        closure, closure_plus, compose, compose_default, concat, connect, determinize, minimize,
        project_input, project_output, prune, remove_epsilons, reverse, shortest_path,
        shortest_path_single, topsort, union, weight_convert, ShortestPathConfig,
    },

    // core types
    arc::{Arc, ArcIterator},
    // fst implementations
    fst::{CompactFst, ConstFst, VectorFst},

    fst::{ExpandedFst, Fst, Label, MutableFst, StateId, NO_LABEL, NO_STATE_ID},

    // i/o
    io::{read_binary, read_openfst, read_text, write_binary, write_openfst, write_text},

    // semirings
    semiring::{
        BooleanWeight, DivisibleSemiring, LogWeight, NaturallyOrderedSemiring, ProbabilityWeight,
        Semiring, StarSemiring, TropicalWeight,
    },

    // utilities
    utils::SymbolTable,

    // error handling
    Error,
    Result,
};
