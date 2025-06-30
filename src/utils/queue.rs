//! Queue implementations for FST state exploration algorithms
//!
//! This module provides various queue types used in FST algorithms for managing
//! the order of state exploration. Different queue types implement different
//! traversal strategies, affecting algorithm behavior and performance.
//!
//! ## Queue Types
//!
//! ### [`FifoQueue`] - First-In-First-Out (Breadth-First)
//! - **Traversal:** Level-by-level exploration
//! - **Properties:** Finds shortest paths, explores nearby states first
//! - **Use cases:** Shortest path algorithms, reachability analysis
//!
//! ### [`LifoQueue`] - Last-In-First-Out (Depth-First)
//! - **Traversal:** Deep exploration before backtracking
//! - **Properties:** Memory efficient, explores one path completely
//! - **Use cases:** Cycle detection, path enumeration
//!
//! ### [`StateQueue`] - Priority-Based
//! - **Traversal:** States processed by priority/cost
//! - **Properties:** Optimal for weighted problems
//! - **Use cases:** Dijkstra's algorithm, A* search
//!
//! ### [`TopOrderQueue`] - Topological Order
//! - **Traversal:** Respects state dependencies
//! - **Properties:** Efficient for acyclic FSTs
//! - **Use cases:** Dynamic programming, weight pushing
//!
//! ## Examples
//!
//! ### Basic Queue Usage
//! ```
//! use arcweight::utils::{Queue, FifoQueue, LifoQueue};
//! use arcweight::fst::StateId;
//!
//! // FIFO for breadth-first traversal
//! let mut fifo = FifoQueue::new();
//! fifo.enqueue(0);
//! fifo.enqueue(1);
//! assert_eq!(fifo.dequeue(), Some(0)); // First in, first out
//!
//! // LIFO for depth-first traversal
//! let mut lifo = LifoQueue::new();
//! lifo.enqueue(0);
//! lifo.enqueue(1);
//! assert_eq!(lifo.dequeue(), Some(1)); // Last in, first out
//! ```
//!
//! ### Generic Algorithm with Queue Trait
//! ```
//! use arcweight::prelude::*;
//! use arcweight::utils::{Queue, FifoQueue};
//!
//! fn explore_states<Q: Queue>(
//!     fst: &impl Fst<TropicalWeight>,
//!     mut queue: Q
//! ) -> Vec<StateId> {
//!     let mut visited = Vec::new();
//!     let mut seen = std::collections::HashSet::new();
//!     
//!     if let Some(start) = fst.start() {
//!         queue.enqueue(start);
//!         seen.insert(start);
//!     }
//!     
//!     while let Some(state) = queue.dequeue() {
//!         visited.push(state);
//!         
//!         for arc in fst.arcs(state) {
//!             if seen.insert(arc.nextstate) {
//!                 queue.enqueue(arc.nextstate);
//!             }
//!         }
//!     }
//!     
//!     visited
//! }
//! ```
//!
//! ### Priority Queue for Shortest Path
//! ```
//! use arcweight::utils::{StateQueue, Queue};
//! use arcweight::prelude::*;
//! use std::cmp::Ordering;
//!
//! // Custom wrapper to make f32 orderable
//! #[derive(Copy, Clone, PartialEq)]
//! struct OrderedFloat(f32);
//!
//! impl Eq for OrderedFloat {}
//!
//! impl PartialOrd for OrderedFloat {
//!     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//!         self.0.partial_cmp(&other.0)
//!     }
//! }
//!
//! impl Ord for OrderedFloat {
//!     fn cmp(&self, other: &Self) -> Ordering {
//!         self.partial_cmp(other).unwrap_or(Ordering::Equal)
//!     }
//! }
//!
//! // Find shortest path using priority queue
//! fn dijkstra_distance(
//!     fst: &impl Fst<TropicalWeight>,
//!     source: StateId
//! ) -> Vec<Option<f32>> {
//!     let n = fst.num_states();
//!     let mut dist = vec![None; n];
//!     let mut queue = StateQueue::new();
//!     
//!     dist[source as usize] = Some(0.0);
//!     queue.enqueue_with_priority(source, OrderedFloat(0.0));
//!     
//!     while let Some(state) = queue.dequeue() {
//!         let d = dist[state as usize].unwrap();
//!         
//!         for arc in fst.arcs(state) {
//!             let next = arc.nextstate as usize;
//!             let new_dist = d + arc.weight.value();
//!             
//!             if dist[next].map_or(true, |old| new_dist < old) {
//!                 dist[next] = Some(new_dist);
//!                 queue.enqueue_with_priority(arc.nextstate, OrderedFloat(new_dist));
//!             }
//!         }
//!     }
//!     
//!     dist
//! }
//! ```

use crate::fst::StateId;
use core::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Common interface for state exploration queues
///
/// This trait provides a uniform interface for different queue implementations,
/// allowing algorithms to be generic over the traversal strategy.
///
/// ## Implementation Requirements
///
/// - `enqueue`: Add state to queue (order depends on implementation)
/// - `dequeue`: Remove and return next state (order depends on implementation)
/// - `is_empty`: Check if queue has no states
/// - `clear`: Remove all states from queue
///
/// ## Design Notes
///
/// The trait uses `StateId` (u32) for efficiency. For priority queues that need
/// additional data, use the specific type's methods like `enqueue_with_priority`.
pub trait Queue {
    /// Add a state to the queue
    ///
    /// The position where the state is added depends on the queue implementation.
    fn enqueue(&mut self, state: StateId);

    /// Remove and return the next state
    ///
    /// Returns `None` if the queue is empty. The state returned depends on
    /// the queue's ordering strategy.
    fn dequeue(&mut self) -> Option<StateId>;

    /// Check if the queue is empty
    fn is_empty(&self) -> bool;

    /// Remove all states from the queue
    fn clear(&mut self);
}

/// First-In-First-Out queue for breadth-first traversal
///
/// States are processed in the order they were discovered, exploring
/// the FST level by level. This is optimal for finding shortest paths
/// in unweighted FSTs.
///
/// ## Performance
///
/// - `enqueue`: O(1) amortized
/// - `dequeue`: O(1) amortized
/// - Space: O(n) where n is the number of enqueued states
///
/// ## Example
///
/// ```
/// use arcweight::utils::{Queue, FifoQueue};
///
/// let mut queue = FifoQueue::new();
/// queue.enqueue(0);
/// queue.enqueue(1);
/// queue.enqueue(2);
///
/// assert_eq!(queue.dequeue(), Some(0));
/// assert_eq!(queue.dequeue(), Some(1));
/// assert_eq!(queue.size(), 1);
/// ```
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

/// Last-In-First-Out queue (stack) for depth-first traversal
///
/// States are processed in reverse order of discovery, exploring
/// one path completely before backtracking. This is memory efficient
/// and useful for finding any path or detecting cycles.
///
/// ## Performance
///
/// - `enqueue`: O(1) amortized
/// - `dequeue`: O(1)
/// - Space: O(n) where n is the maximum depth
///
/// ## Example
///
/// ```
/// use arcweight::utils::{Queue, LifoQueue};
///
/// let mut stack = LifoQueue::new();
/// stack.enqueue(0);
/// stack.enqueue(1);
/// stack.enqueue(2);
///
/// // States come out in reverse order
/// assert_eq!(stack.dequeue(), Some(2));
/// assert_eq!(stack.dequeue(), Some(1));
/// assert_eq!(stack.size(), 1);
/// ```
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

/// Priority queue for best-first state exploration
///
/// States are processed in order of their priority values, making this
/// ideal for algorithms like Dijkstra's shortest path or A* search.
/// Lower priority values are processed first (using `std::cmp::Reverse`).
///
/// ## Type Parameters
///
/// - `P`: Priority type (must implement `Ord`)
///
/// ## Performance
///
/// - `enqueue_with_priority`: O(log n)
/// - `dequeue`: O(log n)
/// - Space: O(n) where n is the number of enqueued states
///
/// ## Example
///
/// ```
/// use arcweight::utils::{StateQueue, Queue};
/// use std::cmp::Reverse;
///
/// let mut pq = StateQueue::new();
///
/// // Add states with priorities (lower values = higher priority)
/// pq.enqueue_with_priority(0, Reverse(10));
/// pq.enqueue_with_priority(1, Reverse(5));
/// pq.enqueue_with_priority(2, Reverse(15));
///
/// // States dequeued by priority
/// assert_eq!(pq.dequeue(), Some(1)); // Priority 5
/// assert_eq!(pq.dequeue(), Some(0)); // Priority 10
/// assert_eq!(pq.dequeue(), Some(2)); // Priority 15
/// ```
///
/// ## Usage in Algorithms
///
/// ```
/// use arcweight::utils::{StateQueue, Queue};
/// use arcweight::prelude::*;
/// use std::cmp::Ordering;
///
/// // Custom wrapper to make f32 orderable
/// #[derive(Copy, Clone, PartialEq)]
/// struct OrderedFloat(f32);
///
/// impl Eq for OrderedFloat {}
///
/// impl PartialOrd for OrderedFloat {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         self.0.partial_cmp(&other.0)
///     }
/// }
///
/// impl Ord for OrderedFloat {
///     fn cmp(&self, other: &Self) -> Ordering {
///         self.partial_cmp(other).unwrap_or(Ordering::Equal)
///     }
/// }
///
/// // A* search with heuristic
/// fn astar_search(
///     fst: &impl Fst<TropicalWeight>,
///     start: StateId,
///     goal: StateId,
///     heuristic: impl Fn(StateId) -> f32
/// ) -> Option<f32> {
///     let mut queue = StateQueue::new();
///     let mut g_score = vec![f32::INFINITY; fst.num_states()];
///     
///     g_score[start as usize] = 0.0;
///     let f_score = heuristic(start);
///     queue.enqueue_with_priority(start, OrderedFloat(f_score));
///     
///     while let Some(current) = queue.dequeue() {
///         if current == goal {
///             return Some(g_score[goal as usize]);
///         }
///         
///         for arc in fst.arcs(current) {
///             let tentative_g = g_score[current as usize] + arc.weight.value();
///             let next = arc.nextstate as usize;
///             
///             if tentative_g < g_score[next] {
///                 g_score[next] = tentative_g;
///                 let f = tentative_g + heuristic(arc.nextstate);
///                 queue.enqueue_with_priority(arc.nextstate, OrderedFloat(f));
///             }
///         }
///     }
///     
///     None
/// }
/// ```
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

/// Queue that processes states in topological order
///
/// For acyclic FSTs, this queue visits states in an order that respects
/// dependencies - a state is only visited after all its predecessors.
/// This enables efficient dynamic programming algorithms.
///
/// ## Requirements
///
/// The FST must be acyclic and topologically sorted. Use the
/// [`topsort()`](crate::algorithms::topsort) algorithm first if needed.
///
/// ## Performance
///
/// - `dequeue`: O(1)
/// - Space: O(n) where n is the number of states
///
/// ## Example
///
/// ```
/// use arcweight::utils::{TopOrderQueue, Queue};
/// use arcweight::prelude::*;
///
/// // Create a simple acyclic FST
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
///
/// // Get topological order (would use topsort() in practice)
/// let order = vec![s0, s1, s2];
/// let mut queue = TopOrderQueue::from_order(order);
///
/// // States visited in topological order
/// assert_eq!(queue.dequeue(), Some(s0));
/// assert_eq!(queue.dequeue(), Some(s1));
/// assert_eq!(queue.dequeue(), Some(s2));
/// ```
///
/// ## Usage in Dynamic Programming
///
/// ```
/// use arcweight::utils::{TopOrderQueue, Queue};
/// use arcweight::prelude::*;
///
/// // Single-source shortest distance on acyclic FST
/// fn acyclic_shortest_distance(
///     fst: &impl Fst<TropicalWeight>,
///     order: Vec<StateId>
/// ) -> Vec<TropicalWeight> {
///     let n = fst.num_states();
///     let mut dist = vec![TropicalWeight::zero(); n];
///     
///     if let Some(start) = fst.start() {
///         dist[start as usize] = TropicalWeight::one();
///     }
///     
///     let mut queue = TopOrderQueue::from_order(order);
///     
///     while let Some(state) = queue.dequeue() {
///         let d = dist[state as usize].clone();
///         
///         for arc in fst.arcs(state) {
///             let next = arc.nextstate as usize;
///             let new_dist = d.clone() * arc.weight.clone();
///             dist[next] = dist[next].clone() + new_dist;
///         }
///     }
///     
///     dist
/// }
/// ```
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
        Self {
            order: Vec::new(),
            pos: 0,
        }
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
