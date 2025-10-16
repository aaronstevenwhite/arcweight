//! Union Gallic semiring variant

use crate::fst::Label;
use crate::semiring::gallic::variant::GallicVariant;
use crate::semiring::{Semiring, SemiringProperties};

/// Union Gallic semiring variant
///
/// The Union Gallic variant is the most general form, implementing union semantics
/// for the addition operation. When label sequences differ, it uses longest common
/// prefix logic similar to LeftGallic. This provides maximum flexibility for
/// general-purpose FST operations.
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
/// - `lcp(s₁, s₂)` is the longest common prefix
/// - `·` denotes label sequence concatenation
/// - Weight operations use the underlying semiring
///
/// # Properties
///
/// - **General**: Most flexible variant, no special restrictions
/// - **Union semantics**: Combines alternatives naturally
/// - **Compatible**: Works with any underlying weight semiring
/// - **Default choice**: Suitable when variant behavior is not critical
///
/// # Use Cases
///
/// ## General FST Composition
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
///
/// // UnionGallic is the default variant
/// type GeneralWeight = GallicWeight<TropicalWeight>; // UnionGallic implicit
///
/// let path1 = GeneralWeight::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
/// let path2 = GeneralWeight::from_labels(&[1, 2, 4], TropicalWeight::new(2.0));
///
/// let combined = path1.plus(&path2);
/// assert_eq!(combined.labels(), &[1, 2]); // Common prefix
/// assert_eq!(*combined.weight().value(), 1.0); // min(1.0, 2.0)
/// ```
///
/// ## Non-Deterministic FST Operations
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// type NonDetWeight = GallicWeight<TropicalWeight, UnionGallic>;
///
/// // Multiple paths with different outputs - all valid
/// let path1 = NonDetWeight::from_labels(&[5, 6], TropicalWeight::new(2.0));
/// let path2 = NonDetWeight::from_labels(&[7, 8], TropicalWeight::new(3.0));
///
/// // Union naturally combines alternatives
/// let alternatives = path1.plus(&path2);
/// // Result extracts common structure
/// ```
///
/// ## Flexible Label Tracking
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
/// use arcweight::semiring::{ProbabilityWeight, Semiring};
///
/// type FlexWeight = GallicWeight<ProbabilityWeight, UnionGallic>;
///
/// // Flexible combination without strict constraints
/// let seq1 = FlexWeight::from_labels(&[1, 2, 3, 4], ProbabilityWeight::new(0.6));
/// let seq2 = FlexWeight::from_labels(&[1, 2, 5, 6], ProbabilityWeight::new(0.3));
///
/// let combined = seq1.plus(&seq2);
/// assert_eq!(combined.labels(), &[1, 2]); // Extracts common prefix
/// assert!((*combined.weight().value() - 0.9).abs() < 1e-10); // 0.6 + 0.3
/// ```
///
/// ## Default Variant for Type Aliases
/// ```rust
/// use arcweight::semiring::gallic::GallicWeight;
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// // UnionGallic is the default when variant not specified
/// type SimpleGallic = GallicWeight<TropicalWeight>;
/// // Equivalent to: GallicWeight<TropicalWeight, UnionGallic>
///
/// let weight = SimpleGallic::from_labels(&[10, 20], TropicalWeight::new(5.0));
/// ```
///
/// # Algorithm Complexity
///
/// - **Plus operation**: O(min(|s₁|, |s₂|)) for LCP computation
/// - **Times operation**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: O(total label count)
///
/// # Comparison with Other Variants
///
/// | Variant        | Plus Operation       | Restriction      |
/// |----------------|----------------------|------------------|
/// | **Union**      | LCP, combine weights | None (general)   |
/// | **Left**       | LCP, combine weights | None             |
/// | **Right**      | LCS, combine weights | None             |
/// | **Restrict**   | Requires match       | Functional only  |
/// | **Min**        | Select minimum       | Path selection   |
///
/// UnionGallic is functionally equivalent to LeftGallic in this implementation,
/// but semantically represents the most general union operation.
///
/// # When to Use UnionGallic
///
/// Choose UnionGallic when:
/// - You need maximum flexibility
/// - No special constraints are required
/// - You're prototyping and unsure which variant to use
/// - Your FSTs are non-deterministic or multi-path
///
/// Consider alternatives when:
/// - **RestrictGallic**: Need functional/deterministic property
/// - **MinGallic**: Need path selection semantics
/// - **Left/RightGallic**: Explicit directionality required
///
/// # Theoretical Background
///
/// The Union Gallic semiring provides general semiring semantics for weighted
/// transducers without imposing additional constraints. It's the natural choice
/// for composition algorithms that handle arbitrary FST structures.
///
/// # See Also
///
/// - [`LeftGallic`](super::LeftGallic) for explicit left-oriented processing
/// - [`MinGallic`](super::MinGallic) for optimization-based selection
/// - [`RestrictGallic`](super::RestrictGallic) for functional constraints
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnionGallic;

impl GallicVariant for UnionGallic {
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W) {
        // Union semantics: compute longest common prefix
        // (This is equivalent to LeftGallic in our implementation)
        let common_labels = labels1
            .iter()
            .zip(labels2.iter())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| *a)
            .collect();

        // Combine weights using underlying semiring addition
        let combined_weight = weight1.plus(weight2);

        (common_labels, combined_weight)
    }

    fn variant_name() -> &'static str {
        "Union"
    }

    fn properties<W: Semiring>() -> SemiringProperties {
        let _weight_props = W::properties();

        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            // Union/LCP operation is generally not commutative
            commutative: false,
            // Generally not idempotent
            idempotent: false,
            // Not a path semiring (combines rather than selects)
            path: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{ProbabilityWeight, TropicalWeight};

    #[test]
    fn test_union_gallic_plus_identical_labels() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2, 3]);
        assert_eq!(*result_weight.value(), 1.0); // min(1.0, 2.0)
    }

    #[test]
    fn test_union_gallic_plus_partial_overlap() {
        let labels1 = vec![1, 2, 3, 4];
        let labels2 = vec![1, 2, 5, 6];
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(1.5);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2]); // Common prefix
        assert_eq!(*result_weight.value(), 1.5);
    }

    #[test]
    fn test_union_gallic_plus_no_overlap() {
        let labels1 = vec![1, 2];
        let labels2 = vec![3, 4];
        let w1 = TropicalWeight::new(2.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 2.0);
    }

    #[test]
    fn test_union_gallic_plus_empty_labels() {
        let labels1: Vec<Label> = vec![];
        let labels2 = vec![1, 2];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 1.0);
    }

    #[test]
    fn test_union_gallic_plus_with_probability() {
        let labels1 = vec![5, 6, 7];
        let labels2 = vec![5, 6, 8];
        let w1 = ProbabilityWeight::new(0.3);
        let w2 = ProbabilityWeight::new(0.5);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![5, 6]);
        assert_eq!(*result_weight.value(), 0.8); // 0.3 + 0.5
    }

    #[test]
    fn test_union_gallic_general_flexibility() {
        // Demonstrate that UnionGallic handles diverse scenarios
        let scenarios = vec![
            (vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]), // Identical
            (vec![1, 2, 3], vec![1, 2, 4], vec![1, 2]),    // Partial
            (vec![1, 2], vec![3, 4], vec![]),              // No overlap
            (vec![1], vec![1, 2, 3], vec![1]),             // Prefix
        ];

        for (labels1, labels2, expected) in scenarios {
            let w1 = TropicalWeight::new(1.0);
            let w2 = TropicalWeight::new(2.0);

            let (result_labels, _) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);
            assert_eq!(result_labels, expected);
        }
    }

    #[test]
    fn test_union_gallic_variant_name() {
        assert_eq!(UnionGallic::variant_name(), "Union");
    }

    #[test]
    fn test_union_gallic_properties() {
        let props = UnionGallic::properties::<TropicalWeight>();

        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(!props.commutative); // LCP is not commutative
        assert!(!props.idempotent);
        assert!(!props.path);
    }

    #[test]
    fn test_union_gallic_not_functional() {
        assert!(!UnionGallic::is_functional());
    }

    #[test]
    fn test_union_gallic_similarity_to_left() {
        // UnionGallic is equivalent to LeftGallic in behavior (LCP + weight combine)
        let labels1 = vec![10, 20, 30];
        let labels2 = vec![10, 20, 40];
        let w1 = TropicalWeight::new(5.0);
        let w2 = TropicalWeight::new(3.0);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        // Same behavior as LeftGallic
        assert_eq!(result_labels, vec![10, 20]);
        assert_eq!(*result_weight.value(), 3.0);
    }

    #[test]
    fn test_union_gallic_default_use_case() {
        // Demonstrate typical usage as default variant
        let labels1 = vec![1, 2];
        let labels2 = vec![1, 3];
        let w1 = TropicalWeight::new(2.0);
        let w2 = TropicalWeight::new(4.0);

        let (result_labels, result_weight) = UnionGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1]); // Common prefix
        assert_eq!(*result_weight.value(), 2.0); // min(2.0, 4.0)
    }
}
