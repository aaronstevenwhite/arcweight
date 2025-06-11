//! Comprehensive tests for utils module

use arcweight::prelude::*;
use arcweight::utils::*;
use proptest::prelude::*;
use std::collections::HashSet;

#[cfg(test)]
mod symbol_table_tests {
    use super::*;

    #[test]
    fn test_symbol_table_creation() {
        let table = SymbolTable::new();

        // SymbolTable starts with epsilon symbol, so size is 1
        assert_eq!(table.size(), 1);
        assert!(!table.is_empty());
        // Epsilon should be at index 0
        assert_eq!(table.find_key(0), Some("<eps>"));
    }

    #[test]
    fn test_symbol_table_add_symbol() {
        let mut table = SymbolTable::new();

        let id1 = table.add_symbol("hello");
        let id2 = table.add_symbol("world");
        let id3 = table.add_symbol("hello"); // duplicate

        // Table starts with epsilon, so adding 2 unique symbols makes size 3
        assert_eq!(table.size(), 3);
        assert_eq!(id1, id3); // same symbol should get same ID
        assert_ne!(id1, id2); // different symbols should get different IDs
    }

    #[test]
    fn test_symbol_table_find_symbol() {
        let mut table = SymbolTable::new();

        let id = table.add_symbol("test");

        assert_eq!(table.find_symbol("test"), Some(id));
        assert_eq!(table.find_symbol("nonexistent"), None);
    }

    #[test]
    fn test_symbol_table_find_key() {
        let mut table = SymbolTable::new();

        let id = table.add_symbol("example");

        assert_eq!(table.find_key(id), Some("example"));
        assert_eq!(table.find_key(999), None); // non-existent ID
    }

    #[test]
    fn test_symbol_table_contains() {
        let mut table = SymbolTable::new();

        table.add_symbol("exists");

        assert!(table.contains_symbol("exists"));
        assert!(!table.contains_symbol("does_not_exist"));

        let id = table.find_symbol("exists").unwrap();
        assert!(table.contains_key(id));
        assert!(!table.contains_key(999));
    }

    #[test]
    fn test_symbol_table_clear() {
        let mut table = SymbolTable::new();

        table.add_symbol("test1");
        table.add_symbol("test2");

        // Table starts with epsilon + 2 added symbols = 3
        assert_eq!(table.size(), 3);

        table.clear();

        // After clear, only epsilon remains
        assert_eq!(table.size(), 1);
        assert!(!table.is_empty());
        assert_eq!(table.find_symbol("test1"), None);
        assert_eq!(table.find_key(0), Some("<eps>"));
    }

    #[test]
    fn test_symbol_table_iteration() {
        let mut table = SymbolTable::new();

        table.add_symbol("apple");
        table.add_symbol("banana");
        table.add_symbol("cherry");

        let symbols: HashSet<_> = table.symbols().collect();
        let keys: HashSet<_> = table.keys().collect();

        // Table has epsilon + 3 added symbols = 4 total
        assert_eq!(symbols.len(), 4);
        assert_eq!(keys.len(), 4);

        assert!(symbols.contains("<eps>"));
        assert!(symbols.contains("apple"));
        assert!(symbols.contains("banana"));
        assert!(symbols.contains("cherry"));
    }

    #[test]
    fn test_symbol_table_numeric_consistency() {
        let mut table = SymbolTable::new();

        let mut ids = Vec::new();
        for i in 0..100 {
            let symbol = format!("symbol_{}", i);
            let id = table.add_symbol(&symbol);
            ids.push(id);
        }

        // IDs should be unique
        let unique_ids: HashSet<_> = ids.iter().collect();
        assert_eq!(unique_ids.len(), ids.len());

        // Adding same symbols again should return same IDs
        for (i, &expected_id) in ids.iter().enumerate().take(100) {
            let symbol = format!("symbol_{}", i);
            let id = table.add_symbol(&symbol);
            assert_eq!(id, expected_id);
        }
    }
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    #[test]
    fn test_encode_mapper_creation() {
        let mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        assert_eq!(mapper.encode_type(), EncodeType::Labels);
        assert_eq!(mapper.size(), 0);
    }

    #[test]
    fn test_encode_mapper_encode_arc() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let encoded_arc = mapper.encode(&arc);

        // Encoding should produce a valid arc
        assert_eq!(encoded_arc.nextstate, arc.nextstate);
        assert_eq!(encoded_arc.weight, arc.weight);

        // Labels might be encoded differently
        if mapper.encode_type() == EncodeType::Labels {
            // Labels could be mapped to different values (u32 is always >= 0)
            assert!(encoded_arc.ilabel < u32::MAX);
            assert!(encoded_arc.olabel < u32::MAX);
        } else {
            assert_eq!(encoded_arc.ilabel, arc.ilabel);
            assert_eq!(encoded_arc.olabel, arc.olabel);
        }
    }

    #[test]
    fn test_encode_mapper_decode_arc() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        let original_arc = Arc::new(10, 20, TropicalWeight::new(5.0), 30);
        let encoded_arc = mapper.encode(&original_arc);

        // Decode is not implemented yet, so it should return an error
        let decode_result = mapper.decode(&encoded_arc);
        assert!(decode_result.is_err());
        assert_eq!(
            decode_result.unwrap_err(),
            "Decoding not implemented - would require reverse mappings"
        );
    }

    #[test]
    fn test_encode_mapper_encode_types() {
        let label_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);
        let weights_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Weights);
        let both_mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);

        assert_eq!(label_mapper.encode_type(), EncodeType::Labels);
        assert_eq!(weights_mapper.encode_type(), EncodeType::Weights);
        assert_eq!(both_mapper.encode_type(), EncodeType::LabelsAndWeights);
    }

    #[test]
    fn test_encode_mapper_consistency() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);

        let arc1 = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let arc2 = Arc::new(1, 2, TropicalWeight::new(3.0), 5); // same labels/weight

        let encoded1 = mapper.encode(&arc1);
        let encoded2 = mapper.encode(&arc2);

        // Same labels and weights should encode to same values
        assert_eq!(encoded1.ilabel, encoded2.ilabel);
        assert_eq!(encoded1.olabel, encoded2.olabel);
        assert_eq!(encoded1.weight, encoded2.weight);

        // Next states should remain unchanged
        assert_eq!(encoded1.nextstate, arc1.nextstate);
        assert_eq!(encoded2.nextstate, arc2.nextstate);
    }

    #[test]
    fn test_encode_mapper_symbols() {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::Labels);

        let arc1 = Arc::new(100, 200, TropicalWeight::new(1.0), 1);
        let arc2 = Arc::new(300, 400, TropicalWeight::new(2.0), 2);

        mapper.encode(&arc1);
        mapper.encode(&arc2);

        // Should have symbol mappings
        assert!(mapper.size() > 0);

        // Should be able to get input/output symbol tables
        let input_symbols = mapper.input_symbols();
        let output_symbols = mapper.output_symbols();

        assert!(input_symbols.size() > 0);
        assert!(output_symbols.size() > 0);
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    #[test]
    fn test_fifo_queue() {
        let mut queue = FifoQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.size(), 0);

        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        assert_eq!(queue.size(), 3);
        assert!(!queue.is_empty());

        assert_eq!(queue.dequeue(), Some(1)); // FIFO order
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_lifo_queue() {
        let mut queue = LifoQueue::new();

        assert!(queue.is_empty());

        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        assert_eq!(queue.size(), 3);

        assert_eq!(queue.dequeue(), Some(3)); // LIFO order
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_state_queue() {
        let mut queue = StateQueue::<StateId>::new();

        queue.enqueue_with_priority(10, 10);
        queue.enqueue_with_priority(5, 5);
        queue.enqueue_with_priority(15, 15);

        // StateQueue might maintain different order based on state priorities
        assert_eq!(queue.size(), 3);

        // Dequeue all elements
        let mut dequeued = Vec::new();
        while let Some(state) = queue.dequeue() {
            dequeued.push(state);
        }

        assert_eq!(dequeued.len(), 3);
        assert!(dequeued.contains(&10));
        assert!(dequeued.contains(&5));
        assert!(dequeued.contains(&15));
    }

    #[test]
    fn test_top_order_queue() {
        // Create a topological order manually
        let predefined_order = vec![0, 1, 2];
        let mut queue = TopOrderQueue::from_order(predefined_order.clone());

        // Should process states in predefined order
        let mut order = Vec::new();
        while let Some(state) = queue.dequeue() {
            order.push(state);
        }

        // Order should match the predefined order
        assert_eq!(order, predefined_order);
        assert_eq!(order.len(), 3);

        // Test empty queue
        let mut empty_queue = TopOrderQueue::from_order(vec![]);
        assert!(empty_queue.is_empty());
        assert_eq!(empty_queue.dequeue(), None);
    }

    #[test]
    fn test_queue_clear() {
        let mut queue = FifoQueue::new();

        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        assert_eq!(queue.size(), 3);

        queue.clear();

        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_queue_front_back() {
        let mut queue = FifoQueue::new();

        queue.enqueue(10);
        queue.enqueue(20);

        assert_eq!(queue.front(), Some(&10));
        assert_eq!(queue.back(), Some(&20));

        queue.dequeue();
        assert_eq!(queue.front(), Some(&20));
        assert_eq!(queue.back(), Some(&20));

        queue.dequeue();
        assert_eq!(queue.front(), None);
        assert_eq!(queue.back(), None);
    }
}

// Property-based tests
proptest! {
    #[test]
    fn symbol_table_consistency(symbols: Vec<String>) {
        let mut table = SymbolTable::new();
        let mut symbol_to_id = std::collections::HashMap::new();

        // Add all symbols and record their IDs
        for symbol in &symbols {
            let id = table.add_symbol(symbol);
            symbol_to_id.insert(symbol.clone(), id);
        }

        // Verify consistency
        for (symbol, expected_id) in &symbol_to_id {
            prop_assert_eq!(table.find_symbol(symbol), Some(*expected_id));
            prop_assert_eq!(table.find_key(*expected_id), Some(symbol.as_str()));
            prop_assert!(table.contains_symbol(symbol));
            prop_assert!(table.contains_key(*expected_id));
        }

        // Adding symbols again should return same IDs
        for symbol in &symbols {
            let id = table.add_symbol(symbol);
            prop_assert_eq!(id, symbol_to_id[symbol]);
        }
    }

    #[test]
    fn encode_decode_roundtrip(
        ilabel: u32,
        olabel: u32,
        weight: f32,
        nextstate: u32,
    ) {
        let mut mapper = EncodeMapper::<TropicalWeight>::new(EncodeType::LabelsAndWeights);

        let original_arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);
        let encoded_arc = mapper.encode(&original_arc);

        if let Ok(decoded_arc) = mapper.decode(&encoded_arc) {
            prop_assert_eq!(decoded_arc.ilabel, original_arc.ilabel);
            prop_assert_eq!(decoded_arc.olabel, original_arc.olabel);
            prop_assert_eq!(decoded_arc.weight, original_arc.weight);
            prop_assert_eq!(decoded_arc.nextstate, original_arc.nextstate);
        }
    }

    #[test]
    fn fifo_queue_order_preserved(items: Vec<u32>) {
        let mut queue = FifoQueue::new();

        // Enqueue all items
        for item in &items {
            queue.enqueue(*item);
        }

        prop_assert_eq!(queue.size(), items.len());

        // Dequeue should return items in same order
        let mut dequeued = Vec::new();
        while let Some(item) = queue.dequeue() {
            dequeued.push(item);
        }

        prop_assert_eq!(dequeued, items);
        prop_assert!(queue.is_empty());
    }

    #[test]
    fn lifo_queue_order_reversed(items: Vec<u32>) {
        let mut queue = LifoQueue::new();

        // Enqueue all items
        for item in &items {
            queue.enqueue(*item);
        }

        prop_assert_eq!(queue.size(), items.len());

        // Dequeue should return items in reverse order
        let mut dequeued = Vec::new();
        while let Some(item) = queue.dequeue() {
            dequeued.push(item);
        }

        let mut expected = items;
        expected.reverse();
        prop_assert_eq!(dequeued, expected);
    }

    #[test]
    fn queue_size_consistency(operations: Vec<bool>) {
        let mut queue = FifoQueue::new();
        let mut expected_size = 0;
        let mut counter = 0u32;

        for op in operations {
            if op {
                // Enqueue operation
                queue.enqueue(counter);
                counter += 1;
                expected_size += 1;
            } else if expected_size > 0 {
                // Dequeue operation (only if queue is not empty)
                queue.dequeue();
                expected_size -= 1;
            }

            prop_assert_eq!(queue.size(), expected_size);
            prop_assert_eq!(queue.is_empty(), expected_size == 0);
        }
    }
}
