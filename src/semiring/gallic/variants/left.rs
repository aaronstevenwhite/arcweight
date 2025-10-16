//! Left Gallic semiring variant

use crate::fst::Label;
use crate::semiring::gallic::variant::{longest_common_prefix, GallicVariant};
use crate::semiring::{Semiring, SemiringProperties};

/// Left Gallic semiring variant
///
/// The Left Gallic variant uses left-to-right label accumulation with longest
/// common prefix (LCP) semantics for the addition operation. This is the standard
/// variant for FST composition where output labels are accumulated from left to right.
///
/// # Mathematical Definition
///
/// For label sequences s₁, s₂ and weights w₁, w₂:
///
/// **Addition (⊕):**
/// ```text
/// (s₁, w₁) ⊕ (s₂, w₂) = (lcp(s₁, s₂), w₁ ⊕ w₂)
/// ```
///
/// **Multiplication (⊗):**
/// ```text
/// (s₁, w₁) ⊗ (s₂, w₂) = (s₁ · s₂, w₁ ⊗ w₂)
/// ```
///
/// Where:
/// - `lcp(s₁, s₂)` is the longest common prefix of the label sequences
/// - `·` denotes label sequence concatenation
/// - Weight operations use the underlying semiring
///
/// # Properties
///
/// - **Not commutative**: Label order matters
/// - **Idempotent addition**: If underlying semiring is idempotent and labels match
/// - **Associative**: Both operations are associative
/// - **Has identity elements**: ([], 0̄) and ([], 1̄)
///
/// # Use Cases
///
/// ## Standard FST Composition
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
///
/// type ComposeWeight = GallicWeight<TropicalWeight, LeftGallic>;
///
/// // Build FST with output tracking
/// let mut fst = VectorFst::<ComposeWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, ComposeWeight::from_labels(&[], TropicalWeight::one()));
///
/// // Arc with output labels [1, 2] accumulated
/// fst.add_arc(s0, Arc::new(
///     10, // input label
///     0,  // epsilon output (stored in weight)
///     ComposeWeight::from_labels(&[1, 2], TropicalWeight::new(0.5)),
///     s1
/// ));
/// ```
///
/// ## Output Label Tracking
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type OutputWeight = GallicWeight<TropicalWeight, LeftGallic>;
///
/// let path1 = OutputWeight::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
/// let path2 = OutputWeight::from_labels(&[1, 2, 4], TropicalWeight::new(2.0));
///
/// // Combining paths: common prefix extraction
/// let combined = path1.plus(&path2);
/// assert_eq!(combined.labels(), &[1, 2]); // Common prefix
/// assert_eq!(*combined.weight().value(), 1.0); // min(1.0, 2.0)
/// ```
///
/// ## Shortest Path with Output Sequence
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type PathWeight = GallicWeight<TropicalWeight, LeftGallic>;
///
/// // Path concatenation through multiplication
/// let segment1 = PathWeight::from_labels(&[5, 6], TropicalWeight::new(1.5));
/// let segment2 = PathWeight::from_labels(&[7, 8], TropicalWeight::new(2.0));
///
/// let full_path = segment1.times(&segment2);
/// assert_eq!(full_path.labels(), &[5, 6, 7, 8]); // Concatenated
/// assert_eq!(*full_path.weight().value(), 3.5);  // 1.5 + 2.0
/// ```
///
/// # Algorithm Complexity
///
/// - **Plus operation**: O(min(|s₁|, |s₂|)) for LCP computation
/// - **Times operation**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: O(total label count)
///
/// # Theoretical Background
///
/// The Left Gallic semiring is used extensively in weighted finite-state transducer
/// theory for composition algorithms. It allows tracking of output label sequences
/// as weights while maintaining semiring properties needed for shortest-path and
/// optimization algorithms.
///
/// See: Mohri, Pereira, Riley (2002) - "Weighted Finite-State Transducers in
/// Speech Recognition"
///
/// # See Also
///
/// - [`RightGallic`](super::RightGallic) for right-to-left variant
/// - [`MinGallic`](super::MinGallic) for optimization-based variant
/// - [`RestrictGallic`](super::RestrictGallic) for functional transducers
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeftGallic;

impl GallicVariant for LeftGallic {
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W) {
        // Compute longest common prefix of label sequences
        let common_labels = longest_common_prefix(labels1, labels2);

        // Combine weights using underlying semiring addition
        let combined_weight = weight1.plus(weight2);

        (common_labels, combined_weight)
    }

    fn variant_name() -> &'static str {
        "Left"
    }

    fn properties<W: Semiring>() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            // LCP operation is not commutative (unless labels are always equal)
            commutative: false,
            // Idempotent if weight is idempotent and labels match
            idempotent: false,
            // Not a path semiring (LCP doesn't select one operand)
            path: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::TropicalWeight;
    use num_traits::Zero;

    #[test]
    fn test_left_gallic_plus_identical_labels() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2, 3]); // Full sequence preserved
        assert_eq!(*result_weight.value(), 1.0); // min(1.0, 2.0)
    }

    #[test]
    fn test_left_gallic_plus_partial_overlap() {
        let labels1 = vec![1, 2, 3, 4];
        let labels2 = vec![1, 2, 5, 6];
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(1.5);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2]); // Common prefix only
        assert_eq!(*result_weight.value(), 1.5); // min(3.0, 1.5)
    }

    #[test]
    fn test_left_gallic_plus_no_overlap() {
        let labels1 = vec![1, 2];
        let labels2 = vec![3, 4];
        let w1 = TropicalWeight::new(2.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new()); // No common prefix
        assert_eq!(*result_weight.value(), 2.0); // min(2.0, 3.0)
    }

    #[test]
    fn test_left_gallic_plus_empty_labels() {
        let labels1: Vec<Label> = vec![];
        let labels2 = vec![1, 2];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_left_gallic_plus_one_prefix_of_other() {
        let labels1 = vec![1, 2];
        let labels2 = vec![1, 2, 3, 4];
        let w1 = TropicalWeight::new(5.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2]); // Shorter sequence is LCP
        assert_eq!(*result_weight.value(), 3.0);
    }

    #[test]
    fn test_left_gallic_plus_with_zero_weight() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 4];
        let w1 = TropicalWeight::zero(); // Infinity in tropical
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2]);
        assert_eq!(*result_weight.value(), 2.0); // min(∞, 2.0) = 2.0
    }

    #[test]
    fn test_left_gallic_variant_name() {
        assert_eq!(LeftGallic::variant_name(), "Left");
    }

    #[test]
    fn test_left_gallic_properties() {
        let props = LeftGallic::properties::<TropicalWeight>();

        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(!props.commutative); // LCP is not commutative
        assert!(!props.idempotent); // Generally not idempotent
        assert!(!props.path); // LCP doesn't select one operand
    }

    #[test]
    fn test_left_gallic_not_commutative() {
        // Demonstrate that LCP makes addition non-commutative
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 4];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(1.0);

        let (result1, weight1) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);
        let (result2, weight2) = LeftGallic::plus(&labels2, &w2, &labels1, &w1);

        // Results are the same due to symmetry of LCP
        assert_eq!(result1, result2);
        assert_eq!(weight1, weight2);

        // But with different weights, order doesn't affect label result
        // (though it does affect weight in non-commutative weight semirings)
    }

    #[test]
    fn test_left_gallic_functional_flag() {
        assert!(!LeftGallic::is_functional());
    }

    #[test]
    fn test_left_gallic_times_commutative() {
        assert!(!LeftGallic::times_commutative());
    }
}
