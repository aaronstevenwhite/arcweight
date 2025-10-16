//! Restricted Gallic semiring variant

use crate::fst::Label;
use crate::semiring::gallic::variant::GallicVariant;
use crate::semiring::{Semiring, SemiringProperties};

/// Restricted Gallic semiring variant
///
/// The Restricted Gallic variant enforces functional transducer properties by
/// requiring label sequences to match exactly in addition operations. This is
/// crucial for algorithms that assume deterministic or functional transducers.
///
/// # Mathematical Definition
///
/// For label sequences s₁, s₂ and weights w₁, w₂:
///
/// **Addition (⊕):**
/// ```text
/// (s₁, w₁) ⊕ (s₂, w₂) = {
///     if s₁ == s₂ { (s₁, w₁ ⊕ w₂) }
///     else { ([], 0̄) }  // Returns zero weight
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
/// - `0̄` is the zero element of the underlying weight semiring
/// - Mismatched label sequences produce a zero weight result
///
/// # Properties
///
/// - **Functional**: Only allows identical label sequences to combine
/// - **Strict**: Returns zero for mismatched sequences
/// - **Commutative addition**: When labels match, result is symmetric
/// - **Deterministic**: Enforces determinism at the semiring level
///
/// # Use Cases
///
/// ## Functional Transducer Enforcement
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, RestrictGallic};
///
/// type FunctionalWeight = GallicWeight<TropicalWeight, RestrictGallic>;
///
/// // Valid: identical label sequences
/// let path1 = FunctionalWeight::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
/// let path2 = FunctionalWeight::from_labels(&[1, 2, 3], TropicalWeight::new(2.0));
/// let combined = path1.plus(&path2);
/// assert_eq!(combined.labels(), &[1, 2, 3]);
/// assert_eq!(*combined.weight().value(), 1.0); // min(1.0, 2.0)
///
/// // Invalid: different label sequences → zero weight
/// let path3 = FunctionalWeight::from_labels(&[1, 2, 4], TropicalWeight::new(1.5));
/// let invalid = path1.plus(&path3);
/// assert!(invalid.weight().is_zero()); // Zero weight indicates error
/// ```
///
/// ## Determinization with Output Constraints
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, RestrictGallic};
/// use arcweight::semiring::TropicalWeight;
///
/// type DetWeight = GallicWeight<TropicalWeight, RestrictGallic>;
///
/// // Determinization requires functional property
/// // RestrictGallic enforces this at the semiring level
/// let det_path = DetWeight::from_labels(&[5, 6], TropicalWeight::new(2.0));
/// // Algorithm can safely assume functional property
/// ```
///
/// ## Composition with Determinism Check
/// ```rust,ignore
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, RestrictGallic};
///
/// type CheckWeight = GallicWeight<TropicalWeight, RestrictGallic>;
///
/// // Composition will signal errors via zero weights when
/// // non-deterministic paths are encountered
/// let result = compose(&fst1, &fst2)?;
/// // Check for zero weights to detect violations
/// ```
///
/// # Algorithm Complexity
///
/// - **Plus operation**: O(min(|s₁|, |s₂|)) for equality check
/// - **Times operation**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: O(total label count)
///
/// # Error Handling
///
/// RestrictGallic uses the semiring's zero element to signal constraint violations:
/// - **Matched sequences**: Normal semiring addition
/// - **Mismatched sequences**: Returns `([], W::zero())`
/// - **Detection**: Check `weight.is_zero()` to detect violations
///
/// This approach maintains semiring properties while signaling errors mathematically.
///
/// # Theoretical Background
///
/// The Restricted Gallic semiring is essential for algorithms that require
/// functional transducers (single output for each input). By enforcing this
/// property at the semiring level, algorithms can safely assume determinism
/// without explicit checking at each step.
///
/// See: Allauzen, Mohri (2004) - "Efficient Algorithms for Testing the Twins
/// Property"
///
/// # See Also
///
/// - [`LeftGallic`](super::LeftGallic) for unrestricted left variant
/// - [`MinGallic`](super::MinGallic) for optimization without restriction
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RestrictGallic;

impl GallicVariant for RestrictGallic {
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W) {
        // Check if label sequences are identical
        if labels1 == labels2 {
            // Sequences match: combine weights normally
            let combined_weight = weight1.plus(weight2);
            (labels1.to_vec(), combined_weight)
        } else {
            // Sequences don't match: return zero weight
            // This signals a violation of the functional property
            (Vec::new(), W::zero())
        }
    }

    fn variant_name() -> &'static str {
        "Restrict"
    }

    fn properties<W: Semiring>() -> SemiringProperties {
        let weight_props = W::properties();

        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            // Commutative when labels match (symmetric condition)
            commutative: weight_props.commutative,
            // Idempotent when weight is idempotent and labels match
            idempotent: weight_props.idempotent,
            // Path property inherited from weight when labels match
            path: weight_props.path,
        }
    }

    fn is_functional() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{ProbabilityWeight, TropicalWeight};
    use num_traits::Zero;

    #[test]
    fn test_restrict_gallic_plus_identical_labels() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![1, 2, 3]);
        assert_eq!(*result_weight.value(), 1.0); // min(1.0, 2.0)
        assert!(!Semiring::is_zero(&result_weight));
    }

    #[test]
    fn test_restrict_gallic_plus_different_labels() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 4];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert!(Semiring::is_zero(&result_weight)); // Signals violation
    }

    #[test]
    fn test_restrict_gallic_plus_empty_vs_nonempty() {
        let labels1: Vec<Label> = vec![];
        let labels2 = vec![1, 2];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert!(Semiring::is_zero(&result_weight));
    }

    #[test]
    fn test_restrict_gallic_plus_both_empty() {
        let labels1: Vec<Label> = vec![];
        let labels2: Vec<Label> = vec![];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert_eq!(*result_weight.value(), 1.0);
        assert!(!Semiring::is_zero(&result_weight));
    }

    #[test]
    fn test_restrict_gallic_plus_different_lengths() {
        let labels1 = vec![1, 2];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(1.0);
        let w2 = TropicalWeight::new(2.0);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, Vec::<Label>::new());
        assert!(Semiring::is_zero(&result_weight));
    }

    #[test]
    fn test_restrict_gallic_plus_probability_weights() {
        let labels1 = vec![5, 6, 7];
        let labels2 = vec![5, 6, 7];
        let w1 = ProbabilityWeight::new(0.3);
        let w2 = ProbabilityWeight::new(0.5);

        let (result_labels, result_weight) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);

        assert_eq!(result_labels, vec![5, 6, 7]);
        assert_eq!(*result_weight.value(), 0.8); // 0.3 + 0.5
        assert!(!Semiring::is_zero(&result_weight));
    }

    #[test]
    fn test_restrict_gallic_enforces_functional() {
        // Demonstrate functional property enforcement
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let labels3 = vec![1, 2, 4]; // Different!
        let w = TropicalWeight::new(1.0);

        // Valid combination
        let (_, weight_valid) = RestrictGallic::plus(&labels1, &w, &labels2, &w);
        assert!(!Semiring::is_zero(&weight_valid));

        // Invalid combination
        let (_, weight_invalid) = RestrictGallic::plus(&labels1, &w, &labels3, &w);
        assert!(Semiring::is_zero(&weight_invalid));
    }

    #[test]
    fn test_restrict_gallic_variant_name() {
        assert_eq!(RestrictGallic::variant_name(), "Restrict");
    }

    #[test]
    fn test_restrict_gallic_properties() {
        let props = RestrictGallic::properties::<TropicalWeight>();

        assert!(props.left_semiring);
        assert!(props.right_semiring);
        // Tropical is commutative, so RestrictGallic inherits it
        assert!(props.commutative);
        // Tropical is idempotent
        assert!(props.idempotent);
    }

    #[test]
    fn test_restrict_gallic_is_functional() {
        assert!(RestrictGallic::is_functional());
    }

    #[test]
    fn test_restrict_gallic_commutative_when_matched() {
        let labels1 = vec![1, 2, 3];
        let labels2 = vec![1, 2, 3];
        let w1 = TropicalWeight::new(5.0);
        let w2 = TropicalWeight::new(3.0);

        let (result1, weight1) = RestrictGallic::plus(&labels1, &w1, &labels2, &w2);
        let (result2, weight2) = RestrictGallic::plus(&labels2, &w2, &labels1, &w1);

        assert_eq!(result1, result2);
        assert_eq!(weight1, weight2);
    }

    #[test]
    fn test_restrict_gallic_zero_detection() {
        let labels1 = vec![1];
        let labels2 = vec![2];
        let w = TropicalWeight::new(1.0);

        let (labels, weight) = RestrictGallic::plus(&labels1, &w, &labels2, &w);

        // Zero weight is the signal for violation
        assert!(labels.is_empty());
        assert!(Semiring::is_zero(&weight));
        assert_eq!(weight, TropicalWeight::zero());
    }
}
