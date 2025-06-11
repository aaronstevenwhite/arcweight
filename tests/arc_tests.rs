//! Comprehensive tests for arc module

use arcweight::fst::NO_LABEL;
use arcweight::prelude::*;
use proptest::prelude::*;

#[cfg(test)]
mod arc_basic_tests {
    use super::*;

    #[test]
    fn test_arc_creation() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);

        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
        assert_eq!(*arc.weight.value(), 3.0);
        assert_eq!(arc.nextstate, 4);
    }

    #[test]
    fn test_epsilon_arc() {
        let arc = Arc::epsilon(TropicalWeight::new(1.5), 5);

        assert_eq!(arc.ilabel, NO_LABEL);
        assert_eq!(arc.olabel, NO_LABEL);
        assert_eq!(*arc.weight.value(), 1.5);
        assert_eq!(arc.nextstate, 5);
        assert!(arc.is_epsilon());
    }

    #[test]
    fn test_epsilon_checks() {
        let epsilon_arc = Arc::epsilon(TropicalWeight::new(1.0), 1);
        let regular_arc = Arc::new(1, 2, TropicalWeight::new(1.0), 1);
        let epsilon_input = Arc::new(0, 2, TropicalWeight::new(1.0), 1);
        let epsilon_output = Arc::new(1, 0, TropicalWeight::new(1.0), 1);

        // Full epsilon
        assert!(epsilon_arc.is_epsilon());
        assert!(epsilon_arc.is_epsilon_input());
        assert!(epsilon_arc.is_epsilon_output());

        // Regular arc
        assert!(!regular_arc.is_epsilon());
        assert!(!regular_arc.is_epsilon_input());
        assert!(!regular_arc.is_epsilon_output());

        // Epsilon input only
        assert!(!epsilon_input.is_epsilon());
        assert!(epsilon_input.is_epsilon_input());
        assert!(!epsilon_input.is_epsilon_output());

        // Epsilon output only
        assert!(!epsilon_output.is_epsilon());
        assert!(!epsilon_output.is_epsilon_input());
        assert!(epsilon_output.is_epsilon_output());
    }

    #[test]
    fn test_arc_display() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let display_str = format!("{}", arc);

        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_arc_equality() {
        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc3 = Arc::new(1, 2, TropicalWeight::new(3.1), 4);
        let arc4 = Arc::new(1, 3, TropicalWeight::new(3.0), 4);

        assert_eq!(arc1, arc2);
        assert_ne!(arc1, arc3);
        assert_ne!(arc1, arc4);
    }

    #[test]
    fn test_arc_clone() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc_clone = arc.clone();

        assert_eq!(arc, arc_clone);
        assert_eq!(arc.ilabel, arc_clone.ilabel);
        assert_eq!(arc.olabel, arc_clone.olabel);
        assert_eq!(arc.weight, arc_clone.weight);
        assert_eq!(arc.nextstate, arc_clone.nextstate);
    }

    #[test]
    fn test_arc_debug() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let debug_str = format!("{:?}", arc);

        assert!(debug_str.contains("Arc"));
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
        assert!(debug_str.contains("4"));
    }

    #[test]
    fn test_arc_hash() {
        use std::collections::HashMap;

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc3 = Arc::new(2, 3, TropicalWeight::new(4.0), 5);

        let mut map = HashMap::new();
        map.insert(arc1.clone(), "value1");
        map.insert(arc3, "value3");

        // arc2 should have same hash as arc1
        assert_eq!(map.get(&arc2), Some(&"value1"));
    }
}

#[cfg(test)]
mod arc_iterator_tests {
    use super::*;

    struct TestArcIterator {
        arcs: Vec<Arc<TropicalWeight>>,
        index: usize,
    }

    impl TestArcIterator {
        fn new(arcs: Vec<Arc<TropicalWeight>>) -> Self {
            Self { arcs, index: 0 }
        }
    }

    impl Iterator for TestArcIterator {
        type Item = Arc<TropicalWeight>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.index < self.arcs.len() {
                let arc = self.arcs[self.index].clone();
                self.index += 1;
                Some(arc)
            } else {
                None
            }
        }
    }

    impl ArcIterator<TropicalWeight> for TestArcIterator {
        fn reset(&mut self) {
            self.index = 0;
        }
    }

    #[test]
    fn test_arc_iterator_basic() {
        let arcs = vec![
            Arc::new(1, 1, TropicalWeight::new(1.0), 1),
            Arc::new(2, 2, TropicalWeight::new(2.0), 2),
            Arc::new(3, 3, TropicalWeight::new(3.0), 3),
        ];

        let iter = TestArcIterator::new(arcs.clone());

        let collected: Vec<_> = iter.collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], arcs[0]);
        assert_eq!(collected[1], arcs[1]);
        assert_eq!(collected[2], arcs[2]);
    }

    #[test]
    fn test_arc_iterator_reset() {
        let arcs = vec![
            Arc::new(1, 1, TropicalWeight::new(1.0), 1),
            Arc::new(2, 2, TropicalWeight::new(2.0), 2),
        ];

        let mut iter = TestArcIterator::new(arcs.clone());

        // Consume first arc
        let first = iter.next().unwrap();
        assert_eq!(first, arcs[0]);

        // Reset and get first arc again
        iter.reset();
        let first_again = iter.next().unwrap();
        assert_eq!(first_again, arcs[0]);
    }

    #[test]
    fn test_arc_iterator_empty() {
        let mut iter = TestArcIterator::new(vec![]);

        assert_eq!(iter.next(), None);

        iter.reset();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_arc_iterator_single() {
        let arc = Arc::new(42, 24, TropicalWeight::new(1.5), 10);
        let mut iter = TestArcIterator::new(vec![arc.clone()]);

        assert_eq!(iter.next(), Some(arc.clone()));
        assert_eq!(iter.next(), None);

        iter.reset();
        assert_eq!(iter.next(), Some(arc.clone()));
        assert_eq!(iter.next(), None);
    }
}

#[cfg(test)]
mod arc_with_different_weights_tests {
    use super::*;

    #[test]
    fn test_arc_with_probability_weight() {
        let arc = Arc::new(1, 2, ProbabilityWeight::new(0.5), 3);

        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
        assert_eq!(*arc.weight.value(), 0.5);
        assert_eq!(arc.nextstate, 3);
    }

    #[test]
    fn test_arc_with_boolean_weight() {
        let arc = Arc::new(5, 6, BooleanWeight::new(true), 7);

        assert_eq!(arc.ilabel, 5);
        assert_eq!(arc.olabel, 6);
        assert!(*arc.weight.value());
        assert_eq!(arc.nextstate, 7);
    }

    #[test]
    fn test_epsilon_with_different_weights() {
        let tropical_eps = Arc::epsilon(TropicalWeight::new(2.5), 1);
        let prob_eps = Arc::epsilon(ProbabilityWeight::new(0.3), 2);
        let bool_eps = Arc::epsilon(BooleanWeight::new(false), 3);

        assert!(tropical_eps.is_epsilon());
        assert!(prob_eps.is_epsilon());
        assert!(bool_eps.is_epsilon());

        assert_eq!(*tropical_eps.weight.value(), 2.5);
        assert_eq!(*prob_eps.weight.value(), 0.3);
        assert!(!(*bool_eps.weight.value()));
    }
}

// Property-based tests
proptest! {
    #[test]
    fn arc_creation_preserves_fields(
        ilabel: u32,
        olabel: u32,
        weight: f32,
        nextstate: u32,
    ) {
        let arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);

        prop_assert_eq!(arc.ilabel, ilabel);
        prop_assert_eq!(arc.olabel, olabel);
        prop_assert_eq!(*arc.weight.value(), weight);
        prop_assert_eq!(arc.nextstate, nextstate);
    }

    #[test]
    fn epsilon_arc_always_epsilon(weight: f32, nextstate: u32) {
        let arc = Arc::epsilon(TropicalWeight::new(weight), nextstate);

        prop_assert!(arc.is_epsilon());
        prop_assert!(arc.is_epsilon_input());
        prop_assert!(arc.is_epsilon_output());
        prop_assert_eq!(arc.ilabel, NO_LABEL);
        prop_assert_eq!(arc.olabel, NO_LABEL);
        prop_assert_eq!(*arc.weight.value(), weight);
        prop_assert_eq!(arc.nextstate, nextstate);
    }

    #[test]
    fn arc_epsilon_checks_correct(ilabel: u32, olabel: u32, weight: f32, nextstate: u32) {
        let arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);

        let expected_epsilon_input = ilabel == NO_LABEL;
        let expected_epsilon_output = olabel == NO_LABEL;
        let expected_epsilon = expected_epsilon_input && expected_epsilon_output;

        prop_assert_eq!(arc.is_epsilon_input(), expected_epsilon_input);
        prop_assert_eq!(arc.is_epsilon_output(), expected_epsilon_output);
        prop_assert_eq!(arc.is_epsilon(), expected_epsilon);
    }

    #[test]
    fn arc_equality_reflexive(ilabel: u32, olabel: u32, weight: f32, nextstate: u32) {
        let arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);
        let arc_clone = arc.clone();

        prop_assert_eq!(arc, arc_clone);
    }

    #[test]
    fn arc_display_contains_fields(ilabel: u32, olabel: u32, weight: f32, nextstate: u32) {
        let arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);
        let display_str = format!("{}", arc);

        prop_assert!(display_str.contains(&ilabel.to_string()));
        prop_assert!(display_str.contains(&olabel.to_string()));
        prop_assert!(display_str.contains(&nextstate.to_string()));
        // Weight display might be formatted differently, so just check it's not empty
        prop_assert!(!display_str.is_empty());
    }
}
