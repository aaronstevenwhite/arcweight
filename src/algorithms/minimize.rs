//! Minimization algorithm

use crate::algorithms::{connect, determinize, reverse};
use crate::fst::{Fst, MutableFst};
use crate::semiring::DivisibleSemiring;
use crate::Result;
use core::hash::Hash;

/// Minimize a deterministic FST to canonical minimal form
///
/// Reduces the FST to the minimum number of states while preserving the accepted
/// language. Uses Brzozowski's algorithm which is guaranteed to produce the unique
/// minimal FST for any regular language.
///
/// # Algorithm Details
///
/// - **Algorithm:** Brzozowski's minimization (1962)
/// - **Steps:** reverse → determinize → reverse → determinize → connect
/// - **Time Complexity:** O(2ⁿ) worst case due to determinization steps
/// - **Space Complexity:** O(2ⁿ) for subset construction
/// - **Optimality:** Produces the unique minimal automaton
///
/// # Prerequisites
///
/// - **Input FST:** Should ideally be deterministic (non-deterministic FSTs are determinized)
/// - **Semiring:** Must be [`DivisibleSemiring`] for weight normalization during determinization
/// - **Connected:** Works best on accessible and coaccessible FSTs
///
/// # Implementation Notes
///
/// The algorithm works by exploiting the fact that determinizing the reverse of an FST
/// merges states that have identical suffixes, while determinizing again merges states
/// with identical prefixes. This process results in the canonical minimal form.
///
/// # Examples
///
/// ## Basic Minimization
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create an FST with redundant states that accept "ab"
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state(); // Redundant state
/// let s4 = fst.add_state(); // Redundant state
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.set_final(s4, TropicalWeight::one()); // Same language as s2
///
/// // Create redundant paths
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
///
/// // Redundant path with same language
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s3));
/// fst.add_arc(s3, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s4));
///
/// // Minimize merges equivalent states
/// let minimized: VectorFst<TropicalWeight> = minimize(&fst)?;
///
/// println!("Original: {} states, Minimized: {} states",
///          fst.num_states(), minimized.num_states());
/// assert!(minimized.num_states() <= fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Optimization Pipeline
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Demonstrates a typical FST optimization pipeline with separate functions
/// // Note: No single semiring in ArcWeight implements all required traits simultaneously
///
/// // For DivisibleSemiring operations (determinization, minimization)
/// fn optimize_divisible_fst<W: DivisibleSemiring + std::hash::Hash + Eq + Ord>(
///     fst: &VectorFst<W>
/// ) -> Result<VectorFst<W>> {
///     let connected: VectorFst<W> = connect(fst)?;
///     // Note: remove_epsilons requires StarSemiring, skipped for DivisibleSemiring-only
///     let deterministic: VectorFst<W> = determinize(&connected)?;
///     minimize(&deterministic)
/// }
///
/// // For StarSemiring operations (closure, epsilon handling)
/// fn optimize_star_fst<W: StarSemiring + std::hash::Hash + Eq + Ord>(
///     fst: &VectorFst<W>
/// ) -> Result<VectorFst<W>> {
///     let connected: VectorFst<W> = connect(fst)?;
///     let no_eps: VectorFst<W> = remove_epsilons(&connected)?;
///     // Note: Cannot minimize star semirings without divisibility
///     Ok(no_eps)
/// }
///
/// // Example with TropicalWeight (DivisibleSemiring + Hash + Eq + Ord)
/// let mut tropical_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = tropical_fst.add_state();
/// let s1 = tropical_fst.add_state();
/// tropical_fst.set_start(s0);
/// tropical_fst.set_final(s1, TropicalWeight::one());
/// tropical_fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
///
/// let optimized_tropical = optimize_divisible_fst(&tropical_fst).unwrap();
/// assert!(optimized_tropical.num_states() > 0);
///
/// // Example with BooleanWeight (StarSemiring + Hash + Eq + Ord)
/// let mut boolean_fst = VectorFst::<BooleanWeight>::new();
/// let s0 = boolean_fst.add_state();
/// let s1 = boolean_fst.add_state();
/// boolean_fst.set_start(s0);
/// boolean_fst.set_final(s1, BooleanWeight::one());
/// boolean_fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
///
/// let optimized_boolean = optimize_star_fst(&boolean_fst).unwrap();
/// assert!(optimized_boolean.num_states() > 0);
/// ```
///
/// ## Dictionary Optimization
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Simple dictionary FST minimization
/// let mut dict = VectorFst::<TropicalWeight>::new();
/// let s0 = dict.add_state();
/// let s1 = dict.add_state();
/// dict.set_start(s0);
/// dict.set_final(s1, TropicalWeight::one());
/// dict.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
///
/// // Minimize the dictionary FST
/// let minimized: VectorFst<TropicalWeight> = minimize(&dict).unwrap();
/// println!("Original: {} states, Minimized: {} states",
///          dict.num_states(), minimized.num_states());
/// ```
///
/// # Performance Considerations
///
/// - **Deterministic Input:** Much faster when input is already deterministic
/// - **Size Reduction:** Effectiveness depends on redundancy in original FST
/// - **Memory Usage:** Can temporarily create large intermediate FSTs during determinization
/// - **Alternative:** Consider direct minimization algorithms for special cases
///
/// # When to Use
///
/// - **After construction:** Minimize hand-built or programmatically generated FSTs
/// - **After union/concatenation:** These operations often create redundant states
/// - **Storage optimization:** Before serializing FSTs for deployment
/// - **Composition preprocessing:** Smaller FSTs compose more efficiently
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or has no start state
/// - Memory allocation fails during any intermediate step
/// - The semiring doesn't support required division operations
/// - Reversal, determinization, or connection operations fail
/// - Intermediate FSTs become too large to process
///
/// # See Also
///
/// - [`determinize()`] for resolving nondeterminism before minimization
/// - [`connect()`] for removing unreachable states as preprocessing
/// - [`reverse()`] for the reversal operation used internally
/// - [Working with FSTs - Minimization](../../docs/working-with-fsts/optimization-operations.md#minimization) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#minimization) for theoretical background
pub fn minimize<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring + Hash + Eq + Ord,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // brzozowski's algorithm:
    // 1. reverse
    // 2. determinize
    // 3. reverse
    // 4. determinize

    let rev1: M = reverse(fst)?;
    let det1: M = determinize(&rev1)?;
    let rev2: M = reverse(&det1)?;
    let det2: M = determinize(&rev2)?;

    // ensure connected
    connect(&det2)
}
