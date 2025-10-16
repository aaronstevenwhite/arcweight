//! Gallic variant trait defining behavior for different Gallic semiring types

use crate::fst::Label;
use crate::semiring::{Semiring, SemiringProperties};
use core::fmt::Debug;

/// Trait defining the behavior of a Gallic semiring variant
///
/// This trait enables type-level selection of Gallic semiring behavior through
/// zero-cost abstractions. Each variant implements different semantics for the
/// addition (⊕) operation while sharing the same multiplication (⊗) logic.
///
/// # Design Philosophy
///
/// The GallicVariant trait leverages Rust's type system to provide:
/// - **Compile-time dispatch**: No runtime cost for variant selection
/// - **Type safety**: Impossible to mix incompatible variants
/// - **Modularity**: Each variant encapsulates its own behavior
/// - **Extensibility**: New variants can be added by implementing this trait
///
/// # Semiring Operations
///
/// Variants define how to combine two `(label_sequence, weight)` pairs:
/// - **Plus (⊕)**: Variant-specific combination logic
/// - **Times (⊗)**: Always concatenates labels and multiplies weights
///
/// # Examples
///
/// ```rust
/// use arcweight::semiring::gallic::{GallicVariant, LeftGallic};
/// use arcweight::semiring::TropicalWeight;
/// use arcweight::fst::Label;
///
/// // Example with LeftGallic (longest common prefix)
/// let labels1 = vec![1, 2, 3, 4];
/// let labels2 = vec![1, 2, 5, 6];
/// let w1 = TropicalWeight::new(1.0);
/// let w2 = TropicalWeight::new(2.0);
///
/// let (result_labels, result_weight) = LeftGallic::plus(&labels1, &w1, &labels2, &w2);
/// // Result: labels=[1,2] (common prefix), weight=min(1.0, 2.0)=1.0
/// ```
///
/// # Implementing Custom Variants
///
/// ```rust,ignore
/// use arcweight::semiring::gallic::GallicVariant;
/// use arcweight::semiring::{Semiring, SemiringProperties};
/// use arcweight::fst::Label;
///
/// #[derive(Clone, Debug)]
/// pub struct CustomGallic;
///
/// impl GallicVariant for CustomGallic {
///     fn plus<W: Semiring>(
///         labels1: &[Label],
///         weight1: &W,
///         labels2: &[Label],
///         weight2: &W,
///     ) -> (Vec<Label>, W) {
///         // Custom combination logic
///         (labels1.to_vec(), weight1.plus(weight2))
///     }
///
///     fn variant_name() -> &'static str {
///         "Custom"
///     }
///
///     fn properties<W: Semiring>() -> SemiringProperties {
///         SemiringProperties::default()
///     }
/// }
/// ```
pub trait GallicVariant: Clone + Debug + Send + Sync + 'static {
    /// Combine two (label sequence, weight) pairs
    ///
    /// This method defines the semiring addition (⊕) operation for this variant.
    /// Different variants implement different combination strategies:
    ///
    /// - **LeftGallic**: Longest common prefix of labels
    /// - **RestrictGallic**: Requires identical label sequences
    /// - **MinGallic**: Selects labels of minimum weight
    /// - **UnionGallic**: General union logic
    ///
    /// # Arguments
    ///
    /// * `labels1` - First label sequence
    /// * `weight1` - First weight
    /// * `labels2` - Second label sequence
    /// * `weight2` - Second weight
    ///
    /// # Returns
    ///
    /// A tuple of `(combined_labels, combined_weight)`
    ///
    /// # Mathematical Property
    ///
    /// This operation should satisfy semiring axioms when combined with `times`.
    fn plus<W: Semiring>(
        labels1: &[Label],
        weight1: &W,
        labels2: &[Label],
        weight2: &W,
    ) -> (Vec<Label>, W);

    /// Name of this variant for display and debugging
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicVariant, LeftGallic};
    ///
    /// assert_eq!(LeftGallic::variant_name(), "Left");
    /// ```
    fn variant_name() -> &'static str;

    /// Semiring properties for this variant
    ///
    /// Returns properties that characterize the mathematical structure:
    /// - `commutative`: Whether addition is commutative
    /// - `idempotent`: Whether a ⊕ a = a
    /// - `path`: Whether addition selects one operand
    ///
    /// Properties may depend on the underlying weight semiring.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The underlying weight semiring type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicVariant, MinGallic};
    /// use arcweight::semiring::TropicalWeight;
    ///
    /// let props = MinGallic::properties::<TropicalWeight>();
    /// assert!(props.path); // MinGallic has path property
    /// ```
    fn properties<W: Semiring>() -> SemiringProperties;

    /// Whether multiplication (label concatenation) is commutative for this variant
    ///
    /// # Returns
    ///
    /// Always returns `false` since label concatenation is not commutative:
    /// `[1, 2] · [3, 4] ≠ [3, 4] · [1, 2]`
    fn times_commutative() -> bool {
        false
    }

    /// Whether this variant enforces functional transducer properties
    ///
    /// Functional variants (like RestrictGallic) require label sequences to match
    /// exactly in plus operations, useful for deterministic transducers.
    ///
    /// # Returns
    ///
    /// `true` for RestrictGallic, `false` for others
    fn is_functional() -> bool {
        false
    }
}

/// Helper function to compute longest common prefix of two label sequences
///
/// # Time Complexity
///
/// O(min(|labels1|, |labels2|))
///
/// # Examples
///
/// ```rust
/// # use arcweight::semiring::gallic::longest_common_prefix;
/// let labels1 = vec![1, 2, 3, 4];
/// let labels2 = vec![1, 2, 5, 6];
/// let lcp = longest_common_prefix(&labels1, &labels2);
/// assert_eq!(lcp, vec![1, 2]);
/// ```
pub fn longest_common_prefix(labels1: &[Label], labels2: &[Label]) -> Vec<Label> {
    labels1
        .iter()
        .zip(labels2.iter())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| *a)
        .collect()
}

/// Helper function to compute longest common suffix of two label sequences
///
/// # Time Complexity
///
/// O(min(|labels1|, |labels2|))
///
/// # Examples
///
/// ```rust
/// # use arcweight::semiring::gallic::longest_common_suffix;
/// let labels1 = vec![1, 2, 3, 4];
/// let labels2 = vec![5, 6, 3, 4];
/// let lcs = longest_common_suffix(&labels1, &labels2);
/// assert_eq!(lcs, vec![3, 4]);
/// ```
pub fn longest_common_suffix(labels1: &[Label], labels2: &[Label]) -> Vec<Label> {
    let len1 = labels1.len();
    let len2 = labels2.len();
    let min_len = len1.min(len2);

    let mut common_len = 0;
    for i in 0..min_len {
        if labels1[len1 - 1 - i] == labels2[len2 - 1 - i] {
            common_len += 1;
        } else {
            break;
        }
    }

    labels1[len1 - common_len..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_common_prefix() {
        // Identical sequences
        assert_eq!(
            longest_common_prefix(&[1, 2, 3], &[1, 2, 3]),
            vec![1, 2, 3]
        );

        // Partial overlap
        assert_eq!(
            longest_common_prefix(&[1, 2, 3, 4], &[1, 2, 5, 6]),
            vec![1, 2]
        );

        // No overlap
        assert_eq!(longest_common_prefix(&[1, 2], &[3, 4]), vec![]);

        // One empty
        assert_eq!(longest_common_prefix(&[], &[1, 2]), vec![]);
        assert_eq!(longest_common_prefix(&[1, 2], &[]), vec![]);

        // Both empty
        assert_eq!(longest_common_prefix(&[], &[]), vec![]);

        // One is prefix of other
        assert_eq!(longest_common_prefix(&[1, 2], &[1, 2, 3, 4]), vec![1, 2]);
    }

    #[test]
    fn test_longest_common_suffix() {
        // Identical sequences
        assert_eq!(
            longest_common_suffix(&[1, 2, 3], &[1, 2, 3]),
            vec![1, 2, 3]
        );

        // Partial overlap
        assert_eq!(
            longest_common_suffix(&[1, 2, 3, 4], &[5, 6, 3, 4]),
            vec![3, 4]
        );

        // No overlap
        assert_eq!(longest_common_suffix(&[1, 2], &[3, 4]), vec![]);

        // One empty
        assert_eq!(longest_common_suffix(&[], &[1, 2]), vec![]);
        assert_eq!(longest_common_suffix(&[1, 2], &[]), vec![]);

        // Both empty
        assert_eq!(longest_common_suffix(&[], &[]), vec![]);

        // One is suffix of other
        assert_eq!(longest_common_suffix(&[3, 4], &[1, 2, 3, 4]), vec![3, 4]);
    }
}
