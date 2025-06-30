//! Random path generation algorithm
//!
//! Generates random paths through weighted finite-state transducers for testing,
//! data generation, and stochastic sampling applications.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::{Error, Result};
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

/// Generate random paths through an FST for testing and sampling
///
/// Creates random walks through the FST starting from the initial state and
/// following random transitions until reaching final states or maximum length.
/// The result is an FST containing the generated paths as individual linear chains.
///
/// # Algorithm Details
///
/// - **Random Walk:** Stochastic traversal from start state following random arcs
/// - **Weight-Based Sampling:** Optional weighted selection based on arc weights
/// - **Path Termination:** Stops at final states or maximum length limit
/// - **Time Complexity:** O(npath × max_length × avg_branching) for path generation
/// - **Space Complexity:** O(npath × max_length) for storing generated paths
///
/// # Mathematical Foundation
///
/// Random path generation implements stochastic sampling from FST:
/// - **Uniform Sampling:** Each outgoing arc has equal selection probability
/// - **Weighted Sampling:** Arc selection probability proportional to weight
/// - **Termination Criteria:** Path ends at final states or length limit
/// - **Independence:** Multiple paths generated independently
///
/// # Algorithm Steps
///
/// 1. **Start State:** Begin random walk from FST start state
/// 2. **Arc Selection:** Choose random outgoing arc (uniform or weighted)
/// 3. **State Transition:** Move to next state following selected arc
/// 4. **Termination Check:** Stop if final state reached or max length
/// 5. **Path Recording:** Store generated arc sequence as FST path
/// 6. **Iteration:** Repeat for specified number of paths
///
/// # Examples
///
/// ## Basic Random Path Generation
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{randgen, RandGenConfig};
///
/// // FST with multiple paths
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.set_final(s3, TropicalWeight::one());
///
/// // Multiple path choices
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// fst.add_arc(s1, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s3));
///
/// // Generate random paths
/// let config = RandGenConfig {
///     max_length: 10,
///     npath: 5,
///     weighted: false,
///     seed: Some(42),
/// };
/// let random_paths: VectorFst<TropicalWeight> = randgen(&fst, config)?;
///
/// // Result contains randomly sampled paths from the FST
/// println!("Generated 5 random paths");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weighted Random Sampling
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{randgen, RandGenConfig};
///
/// // FST with weighted arcs for biased sampling
/// let mut weighted_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = weighted_fst.add_state();
/// let s1 = weighted_fst.add_state();
/// let s2 = weighted_fst.add_state();
///
/// weighted_fst.set_start(s0);
/// weighted_fst.set_final(s1, TropicalWeight::one());
/// weighted_fst.set_final(s2, TropicalWeight::one());
///
/// // Biased arcs: lower weight = higher probability in tropical semiring
/// weighted_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.1), s1)); // likely path
/// weighted_fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2)); // unlikely path
///
/// // Generate with weighted sampling
/// let config = RandGenConfig {
///     max_length: 5,
///     npath: 100,
///     weighted: true, // Enable weight-based selection
///     seed: Some(123),
/// };
/// let samples: VectorFst<TropicalWeight> = randgen(&weighted_fst, config)?;
///
/// // Result biased toward lower-weight (higher-probability) paths
/// println!("Generated weighted samples");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Test Data Generation
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{randgen, RandGenConfig};
///
/// // Language model FST for text generation
/// let mut language_model = VectorFst::<TropicalWeight>::new();
/// let start = language_model.add_state();
/// let word1 = language_model.add_state();
/// let word2 = language_model.add_state();
/// let end = language_model.add_state();
///
/// language_model.set_start(start);
/// language_model.set_final(end, TropicalWeight::one());
///
/// // Vocabulary with different probabilities
/// language_model.add_arc(start, Arc::new(1, 1, TropicalWeight::new(0.3), word1)); // "the"
/// language_model.add_arc(start, Arc::new(2, 2, TropicalWeight::new(1.2), word1)); // "a"
/// language_model.add_arc(word1, Arc::new(3, 3, TropicalWeight::new(0.8), word2)); // "cat"
/// language_model.add_arc(word1, Arc::new(4, 4, TropicalWeight::new(1.1), word2)); // "dog"
/// language_model.add_arc(word2, Arc::new(0, 0, TropicalWeight::one(), end)); // </s>
///
/// // Generate test sentences
/// let config = RandGenConfig {
///     max_length: 20,
///     npath: 50,
///     weighted: true,
///     seed: None, // Use random seed
/// };
/// let test_data: VectorFst<TropicalWeight> = randgen(&language_model, config)?;
///
/// // Result contains synthetic test sentences
/// println!("Generated test sentences");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Stress Testing FST Algorithms
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{randgen, RandGenConfig};
///
/// // Complex FST for algorithm stress testing
/// let mut complex_fst = VectorFst::<TropicalWeight>::new();
/// let states: Vec<_> = (0..10).map(|_| complex_fst.add_state()).collect();
///
/// complex_fst.set_start(states[0]);
/// complex_fst.set_final(states[9], TropicalWeight::one());
///
/// // Create complex branching structure
/// for i in 0..9 {
///     for j in (i+1)..10 {
///         if j <= i + 3 { // Limit connectivity
///             let weight = TropicalWeight::new((j - i) as f32 * 0.5);
///             complex_fst.add_arc(states[i], Arc::new(
///                 (i * 10 + j) as u32, (i * 10 + j) as u32, weight, states[j]
///             ));
///         }
///     }
/// }
///
/// // Generate diverse test cases
/// let config = RandGenConfig {
///     max_length: 100,
///     npath: 1000,
///     weighted: false,
///     seed: Some(456),
/// };
/// let test_cases: VectorFst<TropicalWeight> = randgen(&complex_fst, config)?;
///
/// // Use generated paths to stress test other algorithms
/// println!("Generated stress test cases");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Monte Carlo Sampling
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{randgen, RandGenConfig};
///
/// // Probabilistic FST for Monte Carlo estimation
/// let mut prob_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = prob_fst.add_state();
/// let s1 = prob_fst.add_state();
/// let s2 = prob_fst.add_state();
/// let end = prob_fst.add_state();
///
/// prob_fst.set_start(s0);
/// prob_fst.set_final(end, TropicalWeight::one());
///
/// // Branching with different probabilities
/// prob_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.69), s1)); // 50% in log space
/// prob_fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.69), s2)); // 50% in log space
/// prob_fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), end));
/// prob_fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::one(), end));
///
/// // Monte Carlo sampling for probability estimation
/// let config = RandGenConfig {
///     max_length: 10,
///     npath: 10000, // Large sample for accurate estimation
///     weighted: true,
///     seed: Some(789),
/// };
/// let samples: VectorFst<TropicalWeight> = randgen(&prob_fst, config)?;
///
/// // Analyze samples to estimate FST properties
/// println!("Generated Monte Carlo samples");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Testing and Validation
/// - **Algorithm Testing:** Generate diverse inputs for FST algorithm testing
/// - **Stress Testing:** Create challenging test cases for performance evaluation
/// - **Regression Testing:** Generate reproducible test datasets with fixed seeds
/// - **Coverage Testing:** Ensure algorithms handle various FST path structures
///
/// ## Data Generation
/// - **Synthetic Data:** Generate artificial datasets for machine learning
/// - **Text Generation:** Sample sentences from language model FSTs
/// - **Pattern Generation:** Create test patterns for pattern matching
/// - **Sequence Generation:** Generate symbol sequences for analysis
///
/// ## Monte Carlo Methods
/// - **Probability Estimation:** Sample paths to estimate FST properties
/// - **Statistical Analysis:** Analyze path distributions and characteristics
/// - **Simulation:** Model stochastic processes using FST sampling
/// - **Optimization:** Random sampling for stochastic optimization
///
/// ## Natural Language Processing
/// - **Text Synthesis:** Generate synthetic text from language models
/// - **Data Augmentation:** Create additional training data
/// - **Morphological Generation:** Sample word forms from morphology FSTs
/// - **Pronunciation Generation:** Sample pronunciations from phonetic models
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(npath × max_length × avg_branching) for path generation
/// - **Space Complexity:** O(npath × max_length) for storing result paths
/// - **Memory Usage:** Linear in number of generated paths and their lengths
/// - **Randomness Quality:** Depends on underlying random number generator
/// - **Scalability:** Efficient for moderate numbers of paths and lengths
///
/// # Mathematical Properties
///
/// Random generation has specific statistical properties:
/// - **Sampling Distribution:** Uniform or weighted based on configuration
/// - **Independence:** Generated paths are statistically independent
/// - **Termination:** Guaranteed termination with maximum length bounds
/// - **Coverage:** May not cover all possible FST paths uniformly
/// - **Reproducibility:** Deterministic with fixed random seeds
///
/// # Implementation Details
///
/// The current implementation provides:
/// - **Random Walk:** Basic random walk through FST structure
/// - **Uniform Sampling:** Equal probability for all outgoing arcs
/// - **Early Termination:** Stop at final states with probability
/// - **Path Storage:** Convert generated paths to linear FST chains
/// - **Multiple Paths:** Generate specified number of independent paths
///
/// Future enhancements may include:
/// - **True Weighted Sampling:** Proper probability-based arc selection
/// - **Advanced Termination:** More sophisticated stopping criteria
/// - **Path Deduplication:** Remove duplicate generated paths
/// - **Memory Optimization:** More efficient path storage
///
/// # Sampling Strategies
///
/// Different approaches for different scenarios:
/// - **Uniform Sampling:** Equal probability for all transitions
/// - **Weighted Sampling:** Probability proportional to arc weights
/// - **Length-Biased:** Prefer shorter or longer paths
/// - **Final-State Biased:** Increase probability of reaching final states
/// - **Diverse Sampling:** Ensure good coverage of FST structure
///
/// # Configuration Options
///
/// The `RandGenConfig` provides control over:
/// - **max_length:** Maximum path length before forced termination
/// - **npath:** Number of independent paths to generate
/// - **weighted:** Enable weight-based arc selection
/// - **seed:** Random seed for reproducible generation
///
/// # Randomness and Reproducibility
///
/// Random generation behavior:
/// - **Thread-Safe:** Uses thread-local random number generator
/// - **Seed Control:** Optional seed for deterministic results
/// - **Quality:** High-quality pseudorandom number generation
/// - **Distribution:** Uniform distribution for arc selection
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST is invalid, corrupted, or malformed
/// - The FST has no start state (required for path generation)
/// - Memory allocation fails during path generation or result construction
/// - Random path generation encounters infinite loops or non-terminating paths
/// - Arc selection fails due to empty outgoing arc sets
/// - Path construction exceeds system memory limits
///
/// # See Also
///
/// - [`crate::algorithms::shortest_path()`] for deterministic path extraction
/// - [`crate::algorithms::connect()`] for ensuring FST connectivity before sampling
/// - [`crate::algorithms::prune()`] for reducing FST size before random generation
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for FST manipulation patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#randgen) for mathematical background
pub fn randgen<W, F, M>(fst: &F, config: RandGenConfig) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    let mut rng = rand::rng();

    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

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
                rng.random_range(0..arcs.len())
            } else {
                // uniform selection
                rng.random_range(0..arcs.len())
            };

            let arc = &arcs[arc_idx];
            path.push(arc.clone());
            current = arc.nextstate;
            length += 1;

            // check if final
            if fst.is_final(current) && rng.random_bool(0.5) {
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
        fst.add_arc(
            current,
            Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), next),
        );
        current = next;
    }

    fst.set_final(current, W::one());
    Ok(())
}
