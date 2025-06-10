//! Queue implementations for FST algorithms

use crate::fst::StateId;
use std::collections::{VecDeque, BinaryHeap};
use core::cmp::Ordering;

/// Queue trait for state exploration
pub trait Queue {
    /// Enqueue a state
    fn enqueue(&mut self, state: StateId);
    
    /// Dequeue a state
    fn dequeue(&mut self) -> Option<StateId>;
    
    /// Check if empty
    fn is_empty(&self) -> bool;
    
    /// Clear the queue
    fn clear(&mut self);
}

/// FIFO queue
#[derive(Debug, Clone, Default)]
pub struct FifoQueue {
    queue: VecDeque<StateId>,
}

impl FifoQueue {
    /// Create a new FIFO queue
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Get queue size
    pub fn size(&self) -> usize {
        self.queue.len()
    }
    
    /// Get front element
    pub fn front(&self) -> Option<&StateId> {
        self.queue.front()
    }
    
    /// Get back element
    pub fn back(&self) -> Option<&StateId> {
        self.queue.back()
    }
}

impl Queue for FifoQueue {
    fn enqueue(&mut self, state: StateId) {
        self.queue.push_back(state);
    }
    
    fn dequeue(&mut self) -> Option<StateId> {
        self.queue.pop_front()
    }
    
    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    fn clear(&mut self) {
        self.queue.clear();
    }
}

/// LIFO queue (stack)
#[derive(Debug, Clone, Default)]
pub struct LifoQueue {
    stack: Vec<StateId>,
}

impl LifoQueue {
    /// Create a new LIFO queue
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Get stack size
    pub fn size(&self) -> usize {
        self.stack.len()
    }
}

impl Queue for LifoQueue {
    fn enqueue(&mut self, state: StateId) {
        self.stack.push(state);
    }
    
    fn dequeue(&mut self) -> Option<StateId> {
        self.stack.pop()
    }
    
    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
    
    fn clear(&mut self) {
        self.stack.clear();
    }
}

/// State queue with priorities
#[derive(Debug, Clone)]
pub struct StateQueue<P: Ord> {
    heap: BinaryHeap<StateWithPriority<P>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct StateWithPriority<P: Ord> {
    state: StateId,
    priority: P,
}

impl<P: Ord> Ord for StateWithPriority<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl<P: Ord> PartialOrd for StateWithPriority<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Ord> Default for StateQueue<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Ord> StateQueue<P> {
    /// Create a new state queue
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
    
    /// Enqueue with priority
    pub fn enqueue_with_priority(&mut self, state: StateId, priority: P) {
        self.heap.push(StateWithPriority { state, priority });
    }
    
    /// Get size of queue
    pub fn size(&self) -> usize {
        self.heap.len()
    }
}

impl<P: Ord> Queue for StateQueue<P> {
    fn enqueue(&mut self, _state: StateId) {
        panic!("Use enqueue_with_priority for StateQueue");
    }
    
    fn dequeue(&mut self) -> Option<StateId> {
        self.heap.pop().map(|s| s.state)
    }
    
    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    
    fn clear(&mut self) {
        self.heap.clear();
    }
}

/// Topological order queue
#[derive(Debug, Clone)]
pub struct TopOrderQueue {
    order: Vec<StateId>,
    pos: usize,
}

impl TopOrderQueue {
    /// Create from topological order
    pub fn from_order(order: Vec<StateId>) -> Self {
        Self { order, pos: 0 }
    }
    
    /// Create a new empty topological order queue
    pub fn new<W: crate::semiring::Semiring, F: crate::fst::Fst<W>>(_fst: &F) -> Self {
        Self { order: Vec::new(), pos: 0 }
    }
}

impl Queue for TopOrderQueue {
    fn enqueue(&mut self, _state: StateId) {
        panic!("TopOrderQueue is read-only");
    }
    
    fn dequeue(&mut self) -> Option<StateId> {
        if self.pos < self.order.len() {
            let state = self.order[self.pos];
            self.pos += 1;
            Some(state)
        } else {
            None
        }
    }
    
    fn is_empty(&self) -> bool {
        self.pos >= self.order.len()
    }
    
    fn clear(&mut self) {
        self.pos = self.order.len();
    }
}