//! Semiring trait implementations for GallicWeight

use super::variant::GallicVariant;
use super::weight::GallicWeight;
use crate::fst::Label;
use crate::semiring::{DivisibleSemiring, Semiring, SemiringProperties};

// === Semiring Trait ===

impl<W, V> Semiring for GallicWeight<W, V>
where
    W: Semiring,
    V: GallicVariant,
    W::Value: Clone + Send + Sync,
{
    type Value = (Vec<Label>, W::Value);

    fn new(value: Self::Value) -> Self {
        Self::new(value.0, W::new(value.1))
    }

    fn value(&self) -> &Self::Value {
        // Note: This requires storing the tuple, but for efficiency we compute on demand
        // This is a limitation - we'd need to cache this value similar to ProductWeight
        // For now, we'll use a simple implementation that works but may allocate
        // In a production implementation, consider using OnceLock like ProductWeight
        unimplemented!("GallicWeight::value() requires caching - use labels() and weight() instead")
    }

    fn plus(&self, other: &Self) -> Self {
        // Handle zero (additive identity) specially to preserve semiring axioms
        if Semiring::is_zero(&self.weight) {
            return other.clone();
        }
        if Semiring::is_zero(&other.weight) {
            return self.clone();
        }

        let (labels, weight) = V::plus(&self.labels, &self.weight, &other.labels, &other.weight);
        Self::new(labels, weight)
    }

    fn times(&self, other: &Self) -> Self {
        // Concatenate label sequences
        let mut labels = self.labels.clone();
        labels.extend_from_slice(&other.labels);

        // Multiply weights
        let weight = self.weight.times(&other.weight);

        Self::new(labels, weight)
    }

    fn plus_assign(&mut self, other: &Self) {
        // Handle zero (additive identity) specially
        if Semiring::is_zero(&self.weight) {
            self.labels = other.labels.clone();
            self.weight = other.weight.clone();
            return;
        }
        if Semiring::is_zero(&other.weight) {
            return; // No change
        }

        let (labels, weight) = V::plus(&self.labels, &self.weight, &other.labels, &other.weight);
        self.labels = labels;
        self.weight = weight;
    }

    fn times_assign(&mut self, other: &Self) {
        self.labels.extend_from_slice(&other.labels);
        self.weight.times_assign(&other.weight);
    }

    fn properties() -> SemiringProperties {
        V::properties::<W>()
    }

    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.labels == other.labels && self.weight.approx_eq(&other.weight, epsilon)
    }
}

// === DivisibleSemiring ===

impl<W, V> DivisibleSemiring for GallicWeight<W, V>
where
    W: DivisibleSemiring,
    V: GallicVariant,
    W::Value: Clone + Send + Sync,
{
    fn divide(&self, other: &Self) -> Option<Self> {
        // Check if other's labels are a suffix of self's labels
        if self.labels.len() >= other.labels.len() {
            let start_idx = self.labels.len() - other.labels.len();
            if &self.labels[start_idx..] == other.labels.as_slice() {
                // Labels are compatible: remove suffix
                let result_labels = self.labels[..start_idx].to_vec();
                // Divide weights
                if let Some(result_weight) = self.weight.divide(&other.weight) {
                    return Some(Self::new(result_labels, result_weight));
                }
            }
        }
        // Division not possible
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::gallic::variants::{LeftGallic, MinGallic, RestrictGallic, UnionGallic};
    use crate::semiring::{ProbabilityWeight, TropicalWeight};
    use num_traits::{One, Zero};

    #[test]
    fn test_gallic_zero_one() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let zero = GW::zero();
        let one = GW::one();

        assert!(zero.labels().is_empty());
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_zero(&zero.weight));

        assert!(one.labels().is_empty());
        assert!(Semiring::is_one(&one.weight));
    }

    #[test]
    fn test_gallic_plus_left_variant() {
        type GW = GallicWeight<TropicalWeight, LeftGallic>;

        let w1 = GW::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
        let w2 = GW::from_labels(&[1, 2, 4], TropicalWeight::new(2.0));

        let result = w1.plus(&w2);
        assert_eq!(result.labels(), &[1, 2]); // LCP
        assert_eq!(*result.weight().value(), 1.0); // min(1.0, 2.0)
    }

    #[test]
    fn test_gallic_plus_min_variant() {
        type GW = GallicWeight<TropicalWeight, MinGallic>;

        let w1 = GW::from_labels(&[1, 2, 3], TropicalWeight::new(5.0));
        let w2 = GW::from_labels(&[4, 5, 6], TropicalWeight::new(3.0));

        let result = w1.plus(&w2);
        assert_eq!(result.labels(), &[4, 5, 6]); // Min weight labels
        assert_eq!(*result.weight().value(), 3.0);
    }

    #[test]
    fn test_gallic_plus_restrict_variant() {
        type GW = GallicWeight<TropicalWeight, RestrictGallic>;

        // Matching labels
        let w1 = GW::from_labels(&[1, 2, 3], TropicalWeight::new(1.0));
        let w2 = GW::from_labels(&[1, 2, 3], TropicalWeight::new(2.0));
        let result = w1.plus(&w2);
        assert_eq!(result.labels(), &[1, 2, 3]);
        assert_eq!(*result.weight().value(), 1.0);

        // Non-matching labels
        let w3 = GW::from_labels(&[1, 2, 4], TropicalWeight::new(1.0));
        let result_invalid = w1.plus(&w3);
        assert!(Semiring::is_zero(result_invalid.weight())); // Signals violation
    }

    #[test]
    fn test_gallic_times() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let w1 = GW::from_labels(&[1, 2], TropicalWeight::new(1.0));
        let w2 = GW::from_labels(&[3, 4], TropicalWeight::new(2.0));

        let result = w1.times(&w2);
        assert_eq!(result.labels(), &[1, 2, 3, 4]); // Concatenated
        assert_eq!(*result.weight().value(), 3.0); // 1.0 + 2.0 (tropical)
    }

    #[test]
    fn test_gallic_operator_overloads() {
        type GW = GallicWeight<TropicalWeight, MinGallic>;

        let w1 = GW::from_labels(&[1, 2], TropicalWeight::new(5.0));
        let w2 = GW::from_labels(&[3, 4], TropicalWeight::new(3.0));

        // Test + operator (min selection)
        let sum = w1.clone() + w2.clone();
        assert_eq!(sum.labels(), &[3, 4]);

        // Test * operator (concatenation)
        let product = w1 * w2;
        assert_eq!(product.labels(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_gallic_identity_laws() {
        type GW = GallicWeight<TropicalWeight, LeftGallic>;

        let w = GW::from_labels(&[1, 2, 3], TropicalWeight::new(2.0));
        let zero = GW::zero();
        let one = GW::one();

        // Additive identity
        let w_plus_zero = w.clone().plus(&zero);
        assert_eq!(w_plus_zero.labels(), w.labels());
        // Weight: min(2.0, ∞) = 2.0

        // Multiplicative identity
        let w_times_one = w.clone().times(&one);
        assert_eq!(w_times_one.labels(), w.labels());
        assert_eq!(w_times_one.weight(), w.weight());

        // Annihilation by zero
        let w_times_zero = w.times(&zero);
        assert!(Semiring::is_zero(&w_times_zero.weight));
    }

    #[test]
    fn test_gallic_associativity() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let a = GW::from_labels(&[1], TropicalWeight::new(1.0));
        let b = GW::from_labels(&[2], TropicalWeight::new(2.0));
        let c = GW::from_labels(&[3], TropicalWeight::new(3.0));

        // Multiplication associativity
        let left = a.clone().times(&b.clone()).times(&c.clone());
        let right = a.times(&b.times(&c));
        assert_eq!(left.labels(), right.labels());
        assert_eq!(left.weight(), right.weight());
    }

    #[test]
    fn test_gallic_division() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let w1 = GW::from_labels(&[1, 2, 3, 4], TropicalWeight::new(7.0));
        let w2 = GW::from_labels(&[3, 4], TropicalWeight::new(2.0));

        // w1 / w2 should give ([1, 2], 5.0)
        let result = w1.divide(&w2).unwrap();
        assert_eq!(result.labels(), &[1, 2]);
        assert_eq!(*result.weight().value(), 5.0); // 7.0 - 2.0 in tropical

        // Division with incompatible labels
        let w3 = GW::from_labels(&[5, 6], TropicalWeight::new(1.0));
        assert!(w1.divide(&w3).is_none());
    }

    #[test]
    fn test_gallic_properties() {
        // LeftGallic properties
        let props_left = GallicWeight::<TropicalWeight, LeftGallic>::properties();
        assert!(props_left.left_semiring);
        assert!(props_left.right_semiring);
        assert!(!props_left.commutative);

        // MinGallic properties
        let props_min = GallicWeight::<TropicalWeight, MinGallic>::properties();
        assert!(props_min.path); // Min has path property
        assert!(props_min.idempotent);

        // RestrictGallic properties
        let props_restrict = GallicWeight::<TropicalWeight, RestrictGallic>::properties();
        assert!(props_restrict.commutative); // Tropical is commutative
    }

    #[test]
    fn test_gallic_with_probability_weight() {
        type GW = GallicWeight<ProbabilityWeight, MinGallic>;

        let w1 = GW::from_labels(&[1, 2], ProbabilityWeight::new(0.8));
        let w2 = GW::from_labels(&[3, 4], ProbabilityWeight::new(0.3));

        // MinGallic with probability: selects lower numerical value
        let result = w1.plus(&w2);
        assert_eq!(result.labels(), &[3, 4]);
        assert_eq!(*result.weight().value(), 0.3);
    }

    #[test]
    fn test_gallic_approx_eq() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let w1 = GW::from_labels(&[1, 2], TropicalWeight::new(1.0001));
        let w2 = GW::from_labels(&[1, 2], TropicalWeight::new(1.0));

        assert!(w1.approx_eq(&w2, 0.001));
        assert!(!w1.approx_eq(&w2, 0.00001));

        // Different labels - not equal even with tolerance
        let w3 = GW::from_labels(&[1, 3], TropicalWeight::new(1.0));
        assert!(!w1.approx_eq(&w3, 0.01));
    }

    #[test]
    fn test_gallic_plus_assign() {
        type GW = GallicWeight<TropicalWeight, LeftGallic>;

        let mut w1 = GW::from_labels(&[1, 2, 3], TropicalWeight::new(5.0));
        let w2 = GW::from_labels(&[1, 2, 4], TropicalWeight::new(3.0));

        w1.plus_assign(&w2);
        assert_eq!(w1.labels(), &[1, 2]);
        assert_eq!(*w1.weight().value(), 3.0);
    }

    #[test]
    fn test_gallic_times_assign() {
        type GW = GallicWeight<TropicalWeight, UnionGallic>;

        let mut w1 = GW::from_labels(&[1, 2], TropicalWeight::new(2.0));
        let w2 = GW::from_labels(&[3, 4], TropicalWeight::new(3.0));

        w1.times_assign(&w2);
        assert_eq!(w1.labels(), &[1, 2, 3, 4]);
        assert_eq!(*w1.weight().value(), 5.0);
    }
}
