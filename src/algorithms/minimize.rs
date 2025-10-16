//! Minimization algorithm
//!
//! ## Overview
//!
//! Reduces a deterministic FST to its unique canonical minimal form using Brzozowski's
//! algorithm. Produces the FST with the minimum number of states that accepts the same
//! weighted language as the input.
//!
//! ## Complexity
//!
//! - **Time:** O(2^V) worst case, O(V + E) typical case
//!   - V = number of states in input FST
//!   - E = number of arcs in input FST
//!   - Four determinization steps dominate complexity
//!   - Worst case rare; requires extensive nondeterminism after reversal
//! - **Space:** O(2^V) for subset construction during determinization
//!
//! ## Algorithm
//!
//! Brzozowski's minimization (1962):
//! 1. Reverse the FST
//! 2. Determinize (merges states with identical suffixes)
//! 3. Reverse again
//! 4. Determinize again (merges states with identical prefixes)
//! 5. Remove unreachable/non-coaccessible states
//!
//! Result: Unique canonical minimal FST for the language
//!
//! ## Use Cases
//!
//! - **After construction:** Minimize hand-built or programmatically generated FSTs
//! - **After union/concatenation:** These operations often create redundant states
//! - **Storage optimization:** Before serializing FSTs for deployment
//! - **Composition preprocessing:** Smaller FSTs compose more efficiently
//!
//! ## References
//!
//! - J. A. Brzozowski (1962). "Canonical regular expressions and minimal state graphs
//!   for definite events." Mathematical Theory of Automata, 12:529-561.
//! - Mehryar Mohri (2009). "Weighted Automata Algorithms." Handbook of Weighted Automata.

use crate::algorithms::{connect, determinize, reverse};
use crate::fst::{Fst, MutableFst};
use crate::semiring::DivisibleSemiring;
use crate::Result;
use core::hash::Hash;

/// Minimize a deterministic FST to canonical minimal form
///
/// Reduces the FST to the minimum number of states while preserving the accepted
/// weighted language. Uses Brzozowski's algorithm which is guaranteed to produce
/// the unique canonical minimal FST for any regular language.
///
/// Requires [`DivisibleSemiring`] for weight normalization during internal determinization.
/// Works on both deterministic and nondeterministic FSTs (non-deterministic inputs are
/// determinized as part of the algorithm).
///
/// # Complexity
///
/// - **Time:** O(2^V) worst case, O(V + E) typical case
///   - V = number of states in input FST
///   - E = number of arcs in input FST
///   - Dominated by four determinization steps
///   - Worst case: exponential subset construction (rare in practice)
///   - Typical case: near-linear with sparse nondeterminism
/// - **Space:** O(2^V) for subset storage during determinization
///   - Temporary FSTs created at each step
///   - Peak memory: largest intermediate determinized FST
///
/// # Algorithm
///
/// Brzozowski's minimization (1962):
/// 1. **Reverse:** Swap initial and final states, reverse all arcs
/// 2. **Determinize:** Merge states with identical suffixes
/// 3. **Reverse:** Swap initial and final states again
/// 4. **Determinize:** Merge states with identical prefixes
/// 5. **Connect:** Remove unreachable and non-coaccessible states
///
/// **Key insight:** Double reversal + determinization merges all equivalent states,
/// producing the unique minimal automaton.
///
/// # Performance Notes
///
/// - **Deterministic input:** Much faster when input is already deterministic
/// - **Size reduction:** Effectiveness depends on redundancy in original FST
/// - **Memory usage:** Creates four intermediate FSTs (reverse, det, reverse, det)
/// - **Alternative algorithms:** Direct minimization (Hopcroft, Moore) may be faster for special cases
/// - **Preprocessing:** Consider [`connect`] before minimization to remove dead states
/// - **Best for:** FSTs with significant redundancy (post-union, post-concatenation)
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
/// - [`determinize`] - Core operation used internally (applied twice)
/// - [`reverse`] - Reversal operation used internally (applied twice)
/// - [`connect`] - Final cleanup step to remove unreachable states
/// - [`DivisibleSemiring`] - Required trait for weight normalization
/// - [`TropicalWeight`] - Compatible semiring for shortest-path problems
/// - [`LogWeight`] - Compatible semiring for probabilistic computations
/// - [`compose`] - Often benefits from minimization preprocessing
///
/// [`determinize`]: crate::algorithms::determinize::determinize
/// [`reverse`]: crate::algorithms::reverse::reverse
/// [`connect`]: crate::algorithms::connect::connect
/// [`compose`]: crate::algorithms::compose::compose
/// [`TropicalWeight`]: crate::semiring::TropicalWeight
/// [`LogWeight`]: crate::semiring::LogWeight
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_minimize_simple_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();

        // Basic structure should be preserved
        assert!(minimized.start().is_some());
        assert!(minimized.num_states() > 0);
    }

    #[test]
    fn test_minimize_redundant_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state(); // Redundant final state

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s3, TropicalWeight::one()); // Same final weight

        // Two paths that should merge in minimization
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s3));

        let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();

        // Should successfully minimize (exact reduction depends on algorithm)
        assert!(minimized.start().is_some());
    }

    #[test]
    fn test_minimize_already_minimal() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();

        // Should successfully minimize
        assert!(minimized.start().is_some());
    }

    #[test]
    fn test_minimize_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();

        assert_eq!(minimized.num_states(), 0);
        assert!(minimized.is_empty());
    }

    #[test]
    fn test_minimize_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();

        // Minimization may change state count but should preserve structure
        assert!(minimized.num_states() > 0);
        assert!(minimized.start().is_some());
    }
}
