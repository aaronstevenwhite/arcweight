//! Right Gallic semiring variant

use crate::fst::Label;
use crate::semiring::gallic::variant::{longest_common_suffix, GallicVariant};
use crate::semiring::{Semiring, SemiringProperties};

/// Right Gallic semiring variant
///
/// The Right Gallic variant uses right-to-left label accumulation with longest
/// common suffix (LCS) semantics for the addition operation. This is the mirror
/// of LeftGallic, useful for reverse composition and right-to-left processing.
///
/// # Mathematical Definition
///
/// For label sequences s₁, s₂ and weights w₁, w₂:
///
/// **Addition (⊕):**
/// ```text
/// (s₁, w₁) ⊕ (s₂, w₂) = (lcs(s₁, s₂), w₁ ⊕ w₂)
/// ```
///
/// **Multiplication (⊗):**
/// ```text
/// (s₁, w₁) ⊗ (s₂, w₂) = (s₁ · s₂, w₁ ⊗ w₂)
/// ```
///
/// Where:
/// - `lcs(s₁, s₂)` is the longest common suffix of the label sequences
/// - `·` denotes label sequence concatenation
/// - Weight operations use the underlying semiring
///
/// # Properties
///
/// - **Not commutative**: Label order matters
/// - **Associative**: Both operations are associative
/// - **Has identity elements**: ([], 0̄) and ([], 1̄)
/// - **Mirror of LeftGallic**: Symmetric behavior for reversed sequences
///
/// # Use Cases
///
/// ## Reverse FST Composition
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, RightGallic};
///
/// type ReverseWeight = GallicWeight<TropicalWeight, RightGallic>;
///
/// // Track labels from right to left
/// let path1 = ReverseWeight::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
/// let path2 = ReverseWeight::from_labels(&[5, 2, 3], TropicalWeight::new(2.0));
///
/// // Combining extracts common suffix
/// let combined = path1.plus(&path2);
/// assert_eq!(combined.labels(), &[2, 3]); // Common suffix
/// assert_eq!(*combined.weight().value(), 1.0); // min(1.0, 2.0)
/// ```
///
/// ## Right-to-Left Processing
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, RightGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type RightWeight = GallicWeight<TropicalWeight, RightGallic>;
///
/// // Useful for algorithms that process FSTs in reverse
/// let segment1 = RightWeight::from_labels(&[1, 2], TropicalWeight::new(0.5));
/// let segment2 = RightWeight::from_labels(&[3, 4], TropicalWeight::new(1.0));
///
/// let concatenated = segment1.times(&segment2);
/// assert_eq!(concatenated.labels(), &[1, 2, 3, 4]);
/// assert_eq!(*concatenated.weight().value(), 1.5);
/// ```
///
/// # Algorithm Complexity
///
/// - **Plus operation**: O(min(|s₁|, |s₂|)) for LCS computation
/// - **Times operation**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: O(total label count)
///
/// # Relationship to LeftGallic
///
/// RightGallic is the natural dual of LeftGallic:
/// - LeftGallic extracts longest common *prefix*
/// - RightGallic extracts longest common *suffix*
/// - For reversed sequences: `RightGallic(rev(s₁), w₁) = rev(LeftGallic(s₁, w₁))`
///
/// # See Also
///
/// - [`LeftGallic`](super::LeftGallic) for left-to-right variant
/// - [`MinGallic`](super::MinGallic) for optimization-based variant
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RightGallic;

impl GallicVariant for RightGallic {
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W) {
        // Compute longest common suffix of label sequences
        let common_labels = longest_common_suffix(labels1, labels2);

        // Combine weights using underlying semiring addition
        let combined_weight = weight1.plus(weight2);

        (common_labels, combined_weight)
    }

    fn variant_name() -> &'static str {
        "Right"
    }

    fn properties<W: Semiring>() -> SemiringProperties {
        let _weight_props = W::properties();

        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            // LCS operation is not commutative
            commutative: false,
            idempotent: false,
            path: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::TropicalWeight;

    #[test]
    fn test_right_gallic_plus_identical_labels() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RightGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2, 3]);
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_right_gallic_plus_partial_overlap() {
        let labels1 = vec![1, 2, 3, 4];
        let labels2 = vec![5, 6, 3, 4];
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(1.5);

        let (result_labels, result_weight) = RightGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![3, 4]); // Common suffix only
        assert_eq!(*result_weight.value(), 1.5);
    }

    #[test]
    fn test_right_gallic_plus_no_overlap() {
        let labels1 = vec![1, 2];
        let labels2 = vec![3, 4];
        let w1 = TropicalWeight::new(2.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = RightGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 2.0);
    }

    #[test]
    fn test_right_gallic_plus_empty_labels() {
        let labels1: Vec<Label> = vec![];
        let labels2 = vec![1, 2];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RightGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_right_gallic_suffix_extraction() {
        // Demonstrate suffix extraction vs prefix (LeftGallic)
        let labels1 = vec![1, 2, 3, 4];
        let labels2 = vec![5, 6, 3, 4];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(1.0);

        let (result_labels, _) = RightGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![3, 4]); // Suffix, not prefix [1, 2]
    }

    #[test]
    fn test_right_gallic_variant_name() {
        assert_eq!(RightGallic::variant_name(), "Right");
    }

    #[test]
    fn test_right_gallic_properties() {
        let props = RightGallic::properties::<TropicalWeight>();

        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(!props.commutative);
        assert!(!props.idempotent);
        assert!(!props.path);
    }

    #[test]
    fn test_right_gallic_not_functional() {
        assert!(!RightGallic::is_functional());
    }
}
