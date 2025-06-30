//! FST properties computation and tracking
//!
//! This module provides a comprehensive system for analyzing and tracking properties
//! of weighted finite state transducers (FSTs). Properties are structural and algorithmic
//! characteristics that enable optimization and guide algorithm selection.
//!
//! ## Overview
//!
//! FST properties describe characteristics like determinism, connectivity, and weight
//! patterns. These properties are crucial for:
//! - **Algorithm Selection:** Choose optimal algorithms based on FST structure
//! - **Performance Optimization:** Skip unnecessary operations for certain properties
//! - **Correctness Verification:** Ensure FSTs meet required constraints
//! - **Memory Optimization:** Use specialized representations for specific properties
//!
//! ## Core Components
//!
//! - [`PropertyFlags`] - Bitflags representing individual FST properties
//! - [`FstProperties`] - Container tracking known and set properties
//! - [`compute_properties()`] - Analyze FST structure to determine properties
//!
//! ## Property Categories
//!
//! ### Structural Properties
//!
//! - **Acceptor/Transducer:** Whether input equals output labels
//! - **String:** Linear path vs branching structure
//! - **Connectivity:** Accessibility and coaccessibility of states
//!
//! ### Epsilon Properties
//!
//! - **No Epsilons:** Absence of epsilon transitions
//! - **Input/Output Epsilons:** Location of epsilon transitions
//!
//! ### Determinism Properties
//!
//! - **Input Deterministic:** At most one arc per input label from each state
//! - **Output Deterministic:** At most one arc per output label from each state
//! - **Functional:** Single output sequence per input sequence
//!
//! ### Topological Properties
//!
//! - **Acyclic:** No cycles in state graph
//! - **Top Sorted:** States ordered topologically
//! - **Arc Sorted:** Arcs sorted by label from each state
//!
//! ### Weight Properties
//!
//! - **Weighted/Unweighted:** Presence of non-trivial weights
//! - **Path Weights:** Distribution and characteristics of path weights
//!
//! ## Usage Examples
//!
//! ### Basic Property Analysis
//!
//! ```
//! use arcweight::prelude::*;
//! use arcweight::properties::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//!
//! let props = compute_properties(&fst);
//!
//! // Check individual properties
//! if props.has_property(PropertyFlags::ACCEPTOR) {
//!     println!("FST is an acceptor");
//! }
//!
//! if props.has_property(PropertyFlags::ACYCLIC) {
//!     println!("FST has no cycles");
//! }
//! ```
//!
//! ### Property-Based Optimization
//!
//! ```
//! use arcweight::prelude::*;
//! use arcweight::properties::*;
//!
//! fn optimize_fst<W: StarSemiring + DivisibleSemiring + std::hash::Hash + Eq + Ord>(
//!     fst: &VectorFst<W>
//! ) -> Result<VectorFst<W>> {
//!     let props = compute_properties(fst);
//!     
//!     // Skip epsilon removal if no epsilons present
//!     let fst_no_eps: VectorFst<W> = if props.has_property(PropertyFlags::NO_EPSILONS) {
//!         fst.clone()
//!     } else {
//!         remove_epsilons(fst)?
//!     };
//!     
//!     // Use acceptor-specific algorithms when possible
//!     if props.has_property(PropertyFlags::ACCEPTOR) {
//!         // Acceptor-specific minimization is more efficient
//!         minimize(&fst_no_eps)
//!     } else {
//!         // General transducer minimization
//!         minimize(&fst_no_eps)
//!     }
//! }
//! ```
//!
//! ### Property Preservation
//!
//! ```
//! use arcweight::prelude::*;
//! use arcweight::properties::*;
//!
//! fn verify_operation_preserves_determinism<W: Semiring>(
//!     input: &impl Fst<W>,
//!     output: &impl Fst<W>
//! ) -> bool {
//!     let input_props = compute_properties(input);
//!     let output_props = compute_properties(output);
//!     
//!     // Check if determinism is preserved
//!     if input_props.has_property(PropertyFlags::INPUT_DETERMINISTIC) {
//!         output_props.has_property(PropertyFlags::INPUT_DETERMINISTIC)
//!     } else {
//!         true // No determinism to preserve
//!     }
//! }
//! ```
//!
//! ## Performance Considerations
//!
//! - **Computation Cost:** O(|V| + |E|) for full property analysis
//! - **Caching:** Properties should be computed once and cached
//! - **Incremental Updates:** Some FST types track property changes incrementally
//! - **Lazy Computation:** Properties can be computed on-demand for large FSTs
//!
//! ## Implementation Notes
//!
//! Properties are implemented using bitflags for efficient storage and manipulation.
//! The system distinguishes between:
//! - **Known properties:** Properties that have been computed
//! - **Set properties:** Properties that are true for the FST
//!
//! This allows algorithms to distinguish between "property is false" and
//! "property hasn't been computed yet".

mod traits;

pub use traits::{compute_properties, FstProperties, PropertyFlags};
