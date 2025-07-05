//! Min/Max semiring implementations for optimization and bottleneck problems

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;

/// Min weight for bottleneck optimization and capacity-constrained problems
///
/// The Min semiring provides a mathematical framework for solving bottleneck optimization
/// problems where the goal is to maximize the minimum capacity along a path. This semiring
/// uses minimum for path selection and maximum for path composition, making it ideal for
/// network flow problems, resource allocation, and constraint satisfaction.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers ℝ with ±∞
/// - **Addition (⊕):** `min(a, b)` - selects path with better bottleneck
/// - **Multiplication (⊗):** `max(a, b)` - determines bottleneck capacity
/// - **Zero (0̄):** `+∞` - represents no path/infinite cost
/// - **One (1̄):** `-∞` - represents unlimited capacity
///
/// # Use Cases
///
/// ## Network Bottleneck Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Network link capacities
/// let link1_capacity = MinWeight::new(100.0);  // 100 Mbps
/// let link2_capacity = MinWeight::new(50.0);   // 50 Mbps bottleneck
/// let link3_capacity = MinWeight::new(200.0);  // 200 Mbps
///
/// // Path capacity using MinWeight semiring operations
/// let path_capacity = link1_capacity
///     .times(&link2_capacity)
///     .times(&link3_capacity);  // max(100, max(50, 200)) = max(100, 200) = 200
///
/// // Note: MinWeight semiring operations don't directly model physical bottlenecks
/// // The semiring provides mathematical structure for optimization algorithms
/// ```
///
/// ## Resource Allocation Optimization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Available resources at different stages
/// let stage1_resource = MinWeight::new(80.0);   // 80 units available
/// let stage2_resource = MinWeight::new(120.0);  // 120 units available
/// let stage3_resource = MinWeight::new(60.0);   // 60 units available
///
/// // Alternative resource allocation strategies
/// let strategy1 = MinWeight::new(70.0);  // Strategy 1 capacity
/// let strategy2 = MinWeight::new(90.0);  // Strategy 2 capacity
///
/// // Choose better strategy (higher capacity)
/// let best_strategy = strategy1.plus(&strategy2);  // min(70, 90) = 70
/// ```
///
/// ## Quality Assurance and Testing
/// ```rust
/// use arcweight::prelude::*;
///
/// // Test failure rates (lower is better)
/// let unit_test_failures = MinWeight::new(0.02);      // 2% failure rate
/// let integration_failures = MinWeight::new(0.05);    // 5% failure rate
/// let system_test_failures = MinWeight::new(0.01);    // 1% failure rate
///
/// // Overall quality is worst (maximum) failure rate
/// let overall_quality = unit_test_failures
///     .times(&integration_failures)
///     .times(&system_test_failures);  // max(0.02, max(0.05, 0.01)) = 0.05
/// ```
///
/// ## Supply Chain Optimization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Supplier reliability scores (higher is better, but we want minimum guaranteed)
/// let supplier1_reliability = MinWeight::new(0.95);  // 95% reliability
/// let supplier2_reliability = MinWeight::new(0.88);  // 88% reliability
/// let supplier3_reliability = MinWeight::new(0.92);  // 92% reliability
///
/// // Alternative supply routes
/// let route1 = supplier1_reliability.times(&supplier2_reliability);  // max(0.95, 0.88) = 0.95
/// let route2 = supplier2_reliability.times(&supplier3_reliability);  // max(0.88, 0.92) = 0.92
///
/// // Choose more reliable route
/// let best_route = route1.plus(&route2);  // min(0.95, 0.92) = 0.92
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let weight1 = MinWeight::new(10.0);
/// let weight2 = MinWeight::new(20.0);
///
/// // Addition selects minimum (better bottleneck)
/// let sum = weight1 + weight2;  // min(10.0, 20.0) = 10.0
/// assert_eq!(sum, MinWeight::new(10.0));
///
/// // Multiplication takes maximum (bottleneck composition)
/// let product = weight1 * weight2;  // max(10.0, 20.0) = 20.0
/// assert_eq!(product, MinWeight::new(20.0));
///
/// // Identity elements
/// assert_eq!(MinWeight::zero(), MinWeight::INFINITY);     // No path
/// assert_eq!(MinWeight::one(), MinWeight::NEG_INFINITY);  // Unlimited capacity
/// ```
///
/// # Advanced Applications
///
/// ## Critical Path Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Task durations (negative values for maximization problems)
/// let task_a_duration = MinWeight::new(-5.0);  // 5 time units
/// let task_b_duration = MinWeight::new(-3.0);  // 3 time units
/// let task_c_duration = MinWeight::new(-7.0);  // 7 time units
///
/// // Parallel paths (choose longer path for critical analysis)
/// let path1 = task_a_duration.times(&task_b_duration);  // max(-5, -3) = -3
/// let path2 = task_c_duration;                          // -7
///
/// let critical_path = path1.plus(&path2);               // min(-3, -7) = -7
/// ```
///
/// ## Fault Tolerance Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // System component reliabilities (1.0 = perfect, 0.0 = always fails)
/// let component1 = MinWeight::new(0.99);  // 99% reliable
/// let component2 = MinWeight::new(0.95);  // 95% reliable
/// let component3 = MinWeight::new(0.98);  // 98% reliable
///
/// // Series system: reliability is minimum of components
/// let series_system = component1
///     .times(&component2)
///     .times(&component3);  // max(0.99, max(0.95, 0.98)) = 0.99
///
/// // Note: For actual series reliability, product would be used
/// // This demonstrates the semiring operations, not physical modeling
/// ```
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Both min and max operations are O(1)
/// - **Memory:** 4 bytes per weight (single f32)
/// - **Comparison:** Fast floating-point comparison
/// - **Numerical:** Standard IEEE 754 precision and edge case handling
/// - **Optimization:** Highly optimizable by compilers
///
/// # Mathematical Properties
///
/// The Min semiring exhibits important algebraic properties:
/// - **Idempotent:** `min(a, a) = a` and `max(a, a) = a`
/// - **Commutative:** Operations are symmetric in arguments
/// - **Associative:** Enables efficient parallel computation
/// - **Distributive:** Follows semiring distributivity laws
/// - **Absorptive:** Demonstrates lattice properties
///
/// # Integration with FST Algorithms
///
/// Min weights work with FST algorithms for specialized optimization:
/// - **Shortest Path:** Finds paths with maximum bottleneck capacity
/// - **Composition:** Combines capacity-constrained models
/// - **Determinization:** Maintains optimization properties
/// - **Minimization:** Preserves bottleneck characteristics
///
/// # See Also
///
/// - [`MaxWeight`] for dual maximization problems
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for standard shortest-path optimization
/// - [Core Concepts - Custom Semirings](../../docs/core-concepts/semirings.md#custom-semirings) for mathematical background
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinWeight(OrderedFloat<f32>);

impl MinWeight {
    /// Positive infinity (zero element)
    pub const INFINITY: Self = Self(OrderedFloat(f32::INFINITY));

    /// Negative infinity (one element)
    pub const NEG_INFINITY: Self = Self(OrderedFloat(f32::NEG_INFINITY));

    /// Create a new min weight
    pub fn new(value: f32) -> Self {
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for MinWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "∞")
            } else {
                write!(f, "-∞")
            }
        } else {
            let value = self.0;
        write!(f, "{value}")
        }
    }
}

impl Zero for MinWeight {
    fn zero() -> Self {
        Self::INFINITY
    }

    fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_positive()
    }
}

impl One for MinWeight {
    fn one() -> Self {
        Self::NEG_INFINITY
    }
}

impl Add for MinWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl Mul for MinWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.max(rhs.0))
    }
}

impl Semiring for MinWeight {
    type Value = f32;

    fn new(value: Self::Value) -> Self {
        Self::new(value)
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }

    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            commutative: true,
            idempotent: true,
            path: true,
        }
    }
}

impl NaturallyOrderedSemiring for MinWeight {}

/// Max weight for maximization optimization and capacity problems
///
/// The Max semiring provides a mathematical framework for solving maximization
/// optimization problems where the goal is to find paths with maximum capacity or
/// minimum bottleneck. This semiring uses maximum for path selection and minimum
/// for path composition, making it the dual of the Min semiring and ideal for
/// throughput optimization, resource maximization, and reliability analysis.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers ℝ with ±∞
/// - **Addition (⊕):** `max(a, b)` - selects path with better capacity
/// - **Multiplication (⊗):** `min(a, b)` - determines limiting bottleneck
/// - **Zero (0̄):** `-∞` - represents no path/infinite negative cost
/// - **One (1̄):** `+∞` - represents unlimited capacity
///
/// # Use Cases
///
/// ## Network Throughput Optimization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Network link capacities (higher is better)
/// let link1_capacity = MaxWeight::new(100.0);  // 100 Mbps
/// let link2_capacity = MaxWeight::new(200.0);  // 200 Mbps
/// let link3_capacity = MaxWeight::new(50.0);   // 50 Mbps bottleneck
///
/// // Path capacity is limited by bottleneck (minimum capacity)
/// let path_capacity = link1_capacity
///     .times(&link2_capacity)
///     .times(&link3_capacity);  // min(100, min(200, 50)) = min(100, 50) = 50
///
/// // Choose between alternative paths (maximum throughput)
/// let path1 = MaxWeight::new(50.0);   // Path 1 throughput
/// let path2 = MaxWeight::new(75.0);   // Path 2 throughput
/// let best_path = path1.plus(&path2); // max(50, 75) = 75
/// ```
///
/// ## Resource Allocation and Scheduling
/// ```rust
/// use arcweight::prelude::*;
///
/// // Available computational resources
/// let cpu_capacity = MaxWeight::new(80.0);    // 80% CPU available
/// let memory_capacity = MaxWeight::new(90.0); // 90% memory available
/// let disk_capacity = MaxWeight::new(60.0);   // 60% disk available
///
/// // System capacity limited by most constrained resource
/// let system_capacity = cpu_capacity
///     .times(&memory_capacity)
///     .times(&disk_capacity);  // min(80, min(90, 60)) = 60
///
/// // Choose between scheduling strategies
/// let strategy1 = MaxWeight::new(60.0);  // Conservative approach
/// let strategy2 = MaxWeight::new(85.0);  // Aggressive approach
/// let best_strategy = strategy1.plus(&strategy2);  // max(60, 85) = 85
/// ```
///
/// ## Signal Processing and Quality Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Signal quality measurements (higher is better)
/// let signal_strength = MaxWeight::new(0.8);     // 80% signal strength
/// let noise_rejection = MaxWeight::new(0.9);     // 90% noise filtering
/// let channel_quality = MaxWeight::new(0.7);     // 70% channel quality
///
/// // Overall quality limited by weakest component
/// let system_quality = signal_strength
///     .times(&noise_rejection)
///     .times(&channel_quality);  // min(0.8, min(0.9, 0.7)) = 0.7
///
/// // Compare alternative signal paths
/// let primary_path = MaxWeight::new(0.7);
/// let backup_path = MaxWeight::new(0.6);
/// let best_signal = primary_path.plus(&backup_path);  // max(0.7, 0.6) = 0.7
/// ```
///
/// ## Financial Portfolio Optimization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Investment returns and risk constraints
/// let asset1_return = MaxWeight::new(0.12);      // 12% expected return
/// let asset2_return = MaxWeight::new(0.08);      // 8% expected return
/// let risk_constraint = MaxWeight::new(0.05);    // 5% maximum risk
///
/// // Portfolio return limited by risk constraints
/// let constrained_return1 = asset1_return.times(&risk_constraint);  // min(0.12, 0.05) = 0.05
/// let constrained_return2 = asset2_return.times(&risk_constraint);  // min(0.08, 0.05) = 0.05
///
/// // Choose best available return under constraints
/// let optimal_return = constrained_return1.plus(&constrained_return2);  // max(0.05, 0.05) = 0.05
/// ```
///
/// ## Manufacturing Quality Control
/// ```rust
/// use arcweight::prelude::*;
///
/// // Quality scores from different production stages
/// let material_quality = MaxWeight::new(0.95);   // 95% material grade
/// let process_quality = MaxWeight::new(0.88);    // 88% process quality
/// let testing_quality = MaxWeight::new(0.92);    // 92% testing score
///
/// // Final product quality limited by weakest stage
/// let product_quality = material_quality
///     .times(&process_quality)
///     .times(&testing_quality);  // min(0.95, min(0.88, 0.92)) = 0.88
///
/// // Compare production lines
/// let line1_quality = MaxWeight::new(0.88);
/// let line2_quality = MaxWeight::new(0.91);
/// let best_line = line1_quality.plus(&line2_quality);  // max(0.88, 0.91) = 0.91
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let weight1 = MaxWeight::new(10.0);
/// let weight2 = MaxWeight::new(20.0);
///
/// // Addition selects maximum (better capacity)
/// let sum = weight1 + weight2;  // max(10.0, 20.0) = 20.0
/// assert_eq!(sum, MaxWeight::new(20.0));
///
/// // Multiplication takes minimum (bottleneck constraint)
/// let product = weight1 * weight2;  // min(10.0, 20.0) = 10.0
/// assert_eq!(product, MaxWeight::new(10.0));
///
/// // Identity elements
/// assert_eq!(MaxWeight::zero(), MaxWeight::NEG_INFINITY);   // No path
/// assert_eq!(MaxWeight::one(), MaxWeight::INFINITY);       // Unlimited capacity
/// ```
///
/// # Advanced Applications
///
/// ## Load Balancing and Distribution
/// ```rust
/// use arcweight::prelude::*;
///
/// // Server capacities (requests per second)
/// let server1_capacity = MaxWeight::new(1000.0);  // 1000 RPS
/// let server2_capacity = MaxWeight::new(1500.0);  // 1500 RPS
/// let server3_capacity = MaxWeight::new(800.0);   // 800 RPS
///
/// // Load balancer chooses server with highest available capacity
/// let available_capacity = server1_capacity
///     .plus(&server2_capacity)
///     .plus(&server3_capacity);  // max(1000, max(1500, 800)) = 1500
///
/// // Chain multiple load balancers (capacity limited by weakest link)
/// let lb1_capacity = MaxWeight::new(1500.0);
/// let lb2_capacity = MaxWeight::new(1200.0);
/// let chain_capacity = lb1_capacity.times(&lb2_capacity);  // min(1500, 1200) = 1200
/// ```
///
/// ## Security and Access Control
/// ```rust
/// use arcweight::prelude::*;
///
/// // Security clearance levels (higher values = higher clearance)
/// let user_clearance = MaxWeight::new(3.0);      // Level 3 clearance
/// let resource_requirement = MaxWeight::new(2.0); // Level 2 required
/// let system_constraint = MaxWeight::new(4.0);    // Level 4 system max
///
/// // Access granted if user clearance meets all requirements
/// let effective_clearance = user_clearance
///     .times(&resource_requirement)
///     .times(&system_constraint);  // min(3, min(2, 4)) = 2
///
/// // Choose between access methods (highest security level available)
/// let method1_security = MaxWeight::new(2.0);
/// let method2_security = MaxWeight::new(3.0);
/// let best_security = method1_security.plus(&method2_security);  // max(2, 3) = 3
/// ```
///
/// ## Reliability and Fault Tolerance
/// ```rust
/// use arcweight::prelude::*;
///
/// // Component reliability scores (1.0 = perfect, 0.0 = always fails)
/// let component1 = MaxWeight::new(0.99);  // 99% reliable
/// let component2 = MaxWeight::new(0.95);  // 95% reliable
/// let component3 = MaxWeight::new(0.98);  // 98% reliable
///
/// // System reliability limited by least reliable component
/// let system_reliability = component1
///     .times(&component2)
///     .times(&component3);  // min(0.99, min(0.95, 0.98)) = 0.95
///
/// // Redundant systems (choose most reliable)
/// let primary_system = MaxWeight::new(0.95);
/// let backup_system = MaxWeight::new(0.92);
/// let overall_reliability = primary_system.plus(&backup_system);  // max(0.95, 0.92) = 0.95
/// ```
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Both max and min operations are O(1)
/// - **Memory:** 4 bytes per weight (single f32)
/// - **Comparison:** Fast floating-point comparison
/// - **Numerical:** Standard IEEE 754 precision and edge case handling
/// - **Optimization:** Highly optimizable by compilers
///
/// # Mathematical Properties
///
/// The Max semiring exhibits important algebraic properties:
/// - **Idempotent:** `max(a, a) = a` and `min(a, a) = a`
/// - **Commutative:** Operations are symmetric in arguments
/// - **Associative:** Enables efficient parallel computation
/// - **Distributive:** Follows semiring distributivity laws
/// - **Absorptive:** Demonstrates lattice properties
/// - **Dual:** Exact dual of the Min semiring under negation
///
/// # Integration with FST Algorithms
///
/// Max weights work with FST algorithms for maximization optimization:
/// - **Shortest Path:** Finds paths with maximum total capacity
/// - **Composition:** Combines throughput-optimized models
/// - **Determinization:** Maintains maximization properties
/// - **Minimization:** Preserves capacity characteristics
///
/// # See Also
///
/// - [`MinWeight`] for dual minimization problems
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for standard shortest-path optimization
/// - [Core Concepts - Custom Semirings](../../docs/core-concepts/semirings.md#custom-semirings) for mathematical background
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxWeight(OrderedFloat<f32>);

impl MaxWeight {
    /// Negative infinity (zero element)
    pub const NEG_INFINITY: Self = Self(OrderedFloat(f32::NEG_INFINITY));

    /// Positive infinity (one element)
    pub const INFINITY: Self = Self(OrderedFloat(f32::INFINITY));

    /// Create a new max weight
    pub fn new(value: f32) -> Self {
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for MaxWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "∞")
            } else {
                write!(f, "-∞")
            }
        } else {
            let value = self.0;
        write!(f, "{value}")
        }
    }
}

impl Zero for MaxWeight {
    fn zero() -> Self {
        Self::NEG_INFINITY
    }

    fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_negative()
    }
}

impl One for MaxWeight {
    fn one() -> Self {
        Self::INFINITY
    }
}

impl Add for MaxWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.max(rhs.0))
    }
}

impl Mul for MaxWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl Semiring for MaxWeight {
    type Value = f32;

    fn new(value: Self::Value) -> Self {
        Self::new(value)
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }

    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            commutative: true,
            idempotent: true,
            path: true,
        }
    }
}

impl NaturallyOrderedSemiring for MaxWeight {}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_min_weight_creation() {
        let w = MinWeight::new(5.0);
        assert_eq!(*w.value(), 5.0);
    }

    #[test]
    fn test_min_zero_one() {
        let zero = MinWeight::zero();
        let one = MinWeight::one();

        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert!(zero.value().is_infinite());
        assert!(one.value().is_infinite() && one.value().is_sign_negative());
    }

    #[test]
    fn test_min_operations() {
        let w1 = MinWeight::new(5.0);
        let w2 = MinWeight::new(3.0);

        let result = w1.plus(&w2);
        assert_eq!(*result.value(), 3.0); // min operation

        let result = w1.times(&w2);
        assert_eq!(*result.value(), 5.0); // max operation
    }

    #[test]
    fn test_min_display() {
        let w = MinWeight::new(5.0);
        let zero = MinWeight::zero();
        let one = MinWeight::one();

        assert_eq!(format!("{w}"), "5");
        assert_eq!(format!("{zero}"), "∞");
        assert_eq!(format!("{one}"), "-∞");
    }

    #[test]
    fn test_min_properties() {
        let props = MinWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(props.idempotent);
        assert!(props.path);
    }

    #[test]
    fn test_min_identity_laws() {
        let w = MinWeight::new(5.0);
        let zero = MinWeight::zero();
        let one = MinWeight::one();

        // Additive identity
        assert_eq!(w + zero, w);
        assert_eq!(zero + w, w);

        // Multiplicative identity
        assert_eq!(w * one, w);
        assert_eq!(one * w, w);

        // Annihilation by zero
        assert!(Semiring::is_zero(&(w * zero)));
        assert!(Semiring::is_zero(&(zero * w)));
    }

    #[test]
    fn test_max_weight_creation() {
        let w = MaxWeight::new(5.0);
        assert_eq!(*w.value(), 5.0);
    }

    #[test]
    fn test_max_zero_one() {
        let zero = MaxWeight::zero();
        let one = MaxWeight::one();

        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert!(zero.value().is_infinite() && zero.value().is_sign_negative());
        assert!(one.value().is_infinite());
    }

    #[test]
    fn test_max_operations() {
        let w1 = MaxWeight::new(5.0);
        let w2 = MaxWeight::new(3.0);

        let result = w1.plus(&w2);
        assert_eq!(*result.value(), 5.0); // max operation

        let result = w1.times(&w2);
        assert_eq!(*result.value(), 3.0); // min operation
    }

    #[test]
    fn test_max_display() {
        let w = MaxWeight::new(5.0);
        let zero = MaxWeight::zero();
        let one = MaxWeight::one();

        assert_eq!(format!("{w}"), "5");
        assert_eq!(format!("{zero}"), "-∞");
        assert_eq!(format!("{one}"), "∞");
    }

    #[test]
    fn test_max_properties() {
        let props = MaxWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(props.idempotent);
        assert!(props.path);
    }

    #[test]
    fn test_max_identity_laws() {
        let w = MaxWeight::new(5.0);
        let zero = MaxWeight::zero();
        let one = MaxWeight::one();

        // Additive identity
        assert_eq!(w + zero, w);
        assert_eq!(zero + w, w);

        // Multiplicative identity
        assert_eq!(w * one, w);
        assert_eq!(one * w, w);

        // Annihilation by zero
        assert!(Semiring::is_zero(&(w * zero)));
        assert!(Semiring::is_zero(&(zero * w)));
    }

    #[test]
    fn test_minmax_semiring_axioms() {
        // Test Min semiring
        let a = MinWeight::new(2.0);
        let b = MinWeight::new(3.0);
        let c = MinWeight::new(4.0);

        // Associativity
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));

        // Commutativity
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);

        // Distributivity
        assert_eq!((a + b) * c, (a * c) + (b * c));

        // Test Max semiring
        let a = MaxWeight::new(2.0);
        let b = MaxWeight::new(3.0);
        let c = MaxWeight::new(4.0);

        // Associativity
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));

        // Commutativity
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);

        // Distributivity
        assert_eq!((a + b) * c, (a * c) + (b * c));
    }
}
