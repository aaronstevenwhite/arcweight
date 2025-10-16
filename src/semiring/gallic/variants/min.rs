//! Min Gallic semiring variant

use crate::fst::Label;
use crate::semiring::gallic::variant::GallicVariant;
use crate::semiring::{Semiring, SemiringProperties};

/// Min Gallic semiring variant
///
/// The Min Gallic variant implements optimization-based selection: the addition
/// operation selects the label sequence associated with the minimum weight. This
/// provides clean path-selection semantics for shortest-path algorithms.
///
/// # Mathematical Definition
///
/// For label sequences s₁, s₂ and weights w₁, w₂:
///
/// **Addition (⊕):**
/// ```text
/// (s₁, w₁) ⊕ (s₂, w₂) = {
///     if w₁ < w₂  { (s₁, w₁) }
///     else if w₂ < w₁ { (s₂, w₂) }
///     else { (s₁, w₁) }  // Tie-breaking: choose first
/// }
/// ```
///
/// **Multiplication (⊗):**
/// ```text
/// (s₁, w₁) ⊗ (s₂, w₂) = (s₁ · s₂, w₁ ⊗ w₂)
/// ```
///
/// Where:
/// - `·` denotes label sequence concatenation
/// - Weight comparison uses the underlying semiring's ordering
/// - Tie-breaking is deterministic (first operand wins)
///
/// # Properties
///
/// - **Path semiring**: Selects one of the operands
/// - **Deterministic**: Consistent tie-breaking
/// - **Optimization-friendly**: Natural for shortest-path algorithms
/// - **Commutative** (with consistent tie-breaking)
///
/// # Use Cases
///
/// ## Shortest Path with Label Tracking
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
///
/// type ShortestPathWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// // Path 1: labels [1, 2], cost 5.0
/// let path1 = ShortestPathWeight::from_labels(&[1, 2], TropicalWeight::new(5.0));
///
/// // Path 2: labels [3, 4], cost 3.0
/// let path2 = ShortestPathWeight::from_labels(&[3, 4], TropicalWeight::new(3.0));
///
/// // Selection: chooses path with minimum cost
/// let best_path = path1.plus(&path2);
/// assert_eq!(best_path.labels(), &[3, 4]); // Path 2 selected (lower cost)
/// assert_eq!(*best_path.weight().value(), 3.0);
/// ```
///
/// ## Viterbi Decoding with State Sequence
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
/// use arcweight::semiring::LogWeight;
///
/// type ViterbiWeight = GallicWeight<LogWeight, MinGallic>;
///
/// // State sequence 1: [S0, S1, S2], log-prob -2.5
/// let seq1 = ViterbiWeight::from_labels(&[0, 1, 2], LogWeight::from_probability(0.082));
///
/// // State sequence 2: [S0, S2, S3], log-prob -1.8
/// let seq2 = ViterbiWeight::from_labels(&[0, 2, 3], LogWeight::from_probability(0.165));
///
/// // Viterbi: select most likely state sequence
/// let best_seq = seq1.plus(&seq2);
/// assert_eq!(best_seq.labels(), &[0, 2, 3]); // Higher probability sequence
/// ```
///
/// ## Optimal Translation with Alignment
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type TranslationWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// // Translation 1: word IDs [100, 101, 102], score 2.5
/// let trans1 = TranslationWeight::from_labels(&[100, 101, 102], TropicalWeight::new(2.5));
///
/// // Translation 2: word IDs [103, 104], score 1.8 (better!)
/// let trans2 = TranslationWeight::from_labels(&[103, 104], TropicalWeight::new(1.8));
///
/// let best_translation = trans1.plus(&trans2);
/// assert_eq!(best_translation.labels(), &[103, 104]);
/// assert_eq!(*best_translation.weight().value(), 1.8);
/// ```
///
/// ## Path Concatenation
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type PathWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// // Segment 1: labels [1, 2], cost 1.0
/// let seg1 = PathWeight::from_labels(&[1, 2], TropicalWeight::new(1.0));
///
/// // Segment 2: labels [3, 4], cost 2.0
/// let seg2 = PathWeight::from_labels(&[3, 4], TropicalWeight::new(2.0));
///
/// // Concatenation: labels join, costs add
/// let full_path = seg1.times(&seg2);
/// assert_eq!(full_path.labels(), &[1, 2, 3, 4]);
/// assert_eq!(*full_path.weight().value(), 3.0);
/// ```
///
/// # Algorithm Complexity
///
/// - **Plus operation**: O(1) - weight comparison only
/// - **Times operation**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: O(selected label count)
///
/// # Tie-Breaking
///
/// When weights are equal (w₁ == w₂), MinGallic chooses the first operand (s₁, w₁).
/// This provides deterministic behavior essential for reproducible results.
///
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type MinWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// let path1 = MinWeight::from_labels(&[1, 2], TropicalWeight::new(5.0));
/// let path2 = MinWeight::from_labels(&[3, 4], TropicalWeight::new(5.0)); // Same weight!
///
/// let result = path1.plus(&path2);
/// assert_eq!(result.labels(), &[1, 2]); // First operand wins tie
/// ```
///
/// # Comparison with Other Variants
///
/// - **LeftGallic**: Computes LCP, MinGallic selects one path entirely
/// - **RestrictGallic**: Requires matching labels, MinGallic allows any labels
/// - **UnionGallic**: Keeps both alternatives, MinGallic chooses one
///
/// # Theoretical Background
///
/// MinGallic provides path-selection semantics compatible with shortest-path
/// algorithms. The path property (selecting one operand) ensures that the
/// semiring maintains optimization characteristics while tracking label sequences.
///
/// See: Mohri (2002) - "Semiring Frameworks and Algorithms for Shortest-Distance
/// Problems"
///
/// # See Also
///
/// - [`LeftGallic`](super::LeftGallic) for prefix-based combination
/// - [`RestrictGallic`](super::RestrictGallic) for functional constraints
/// - [`UnionGallic`](super::UnionGallic) for keeping all alternatives
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinGallic;

impl GallicVariant for MinGallic {
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W) {
        // Compare weights and select the minimum
        match weight1.partial_cmp(weight2) {
            Some(std::cmp::Ordering::Less) => {
                // w1 < w2: choose first
                (labels1.to_vec(), weight1.clone())
            }
            Some(std::cmp::Ordering::Greater) => {
                // w2 < w1: choose second
                (labels2.to_vec(), weight2.clone())
            }
            Some(std::cmp::Ordering::Equal) | None => {
                // Tie or incomparable: choose first (deterministic tie-breaking)
                (labels1.to_vec(), weight1.clone())
            }
        }
    }

    fn variant_name() -> &'static str {
        "Min"
    }

    fn properties<W: Semiring>() -> SemiringProperties {
        let weight_props = W::properties();

        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            // Min operation is commutative for weights with total order
            commutative: weight_props.commutative,
            // Min is idempotent: min(a, a) = a
            idempotent: true,
            // Path property: selects one of the operands
            path: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{ProbabilityWeight, TropicalWeight};

    #[test]
    fn test_min_gallic_plus_first_smaller() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![4, 5, 6];
        let w1 = TropicalWeight::new(1.0); // Smaller
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2, 3]); // First selected
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_min_gallic_plus_second_smaller() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![4, 5, 6];
        let w1 = TropicalWeight::new(5.0);
        let w2 = TropicalWeight::new(2.0); // Smaller

        let (result_labels, result_weight) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![4, 5, 6]); // Second selected
        assert_eq!(*result_weight.value(), 2.0);
    }

    #[test]
    fn test_min_gallic_plus_equal_weights() {
        let labels1 = vec![1, 2];
        let labels2 = vec![3, 4];
        let w1 = TropicalWeight::new(5.0);
        let w2 = TropicalWeight::new(5.0); // Equal

        let (result_labels, result_weight) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        // Tie-breaking: choose first
        assert_eq!(result_labels, vec![1, 2]);
        assert_eq!(*result_weight.value(), 5.0);
    }

    #[test]
    fn test_min_gallic_plus_with_probability() {
        let labels1 = vec![10, 20];
        let labels2 = vec![30, 40];
        let w1 = ProbabilityWeight::new(0.8); // Higher probability
        let w2 = ProbabilityWeight::new(0.3);

        let (result_labels, _result_weight) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        // For probability semiring, "smaller" weight is less preferred
        // ProbabilityWeight uses standard comparison (0.3 < 0.8)
        assert_eq!(result_labels, vec![30, 40]); // Smaller value selected
    }

    #[test]
    fn test_min_gallic_plus_empty_labels() {
        let labels1: Vec<Label> = vec![];
        let labels2 = vec![1, 2];
        let w1 = TropicalWeight::new(1.0); // Smaller
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_min_gallic_idempotent() {
        let labels = vec![5, 6, 7];
        let w = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = MinGallic::plus(&labels, &w, &labels, &w);

        // min(a, a) = a (idempotent)
        assert_eq!(result_labels, vec![5, 6, 7]);
        assert_eq!(*result_weight.value(), 2.0);
    }

    #[test]
    fn test_min_gallic_variant_name() {
        assert_eq!(MinGallic::variant_name(), "Min");
    }

    #[test]
    fn test_min_gallic_properties() {
        let props = MinGallic::properties::<TropicalWeight>();

        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative); // Tropical is commutative
        assert!(props.idempotent); // Min is idempotent
        assert!(props.path); // Selects one operand
    }

    #[test]
    fn test_min_gallic_path_property() {
        // Demonstrate path property: result is one of the inputs
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![4, 5, 6];
        let w1 = TropicalWeight::new(2.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, _) = MinGallic::plus(&labels1, &w1, &labels2, &w2);

        // Result must be either labels1 or labels2
        assert!(result_labels == labels1 || result_labels == labels2);
    }

    #[test]
    fn test_min_gallic_deterministic_tie_breaking() {
        let labels1 = vec![1];
        let labels2 = vec![2];
        let w = TropicalWeight::new(5.0);

        // Multiple calls with same input should give same result
        let (result1, _) = MinGallic::plus(&labels1, &w, &labels2, &w);
        let (result2, _) = MinGallic::plus(&labels1, &w, &labels2, &w);
        let (result3, _) = MinGallic::plus(&labels1, &w, &labels2, &w);

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        assert_eq!(result1, vec![1]); // Always chooses first
    }

    #[test]
    fn test_min_gallic_not_functional() {
        assert!(!MinGallic::is_functional());
    }
}
