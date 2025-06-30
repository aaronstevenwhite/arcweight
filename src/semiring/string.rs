//! String semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// String semiring for sequence analysis and common subsequence computation
///
/// The String semiring provides a mathematical framework for string operations
/// where addition computes the longest common prefix (LCP) and multiplication
/// performs string concatenation. This semiring is essential for pattern matching,
/// sequence alignment, string processing, and automata-based text analysis.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Finite sequences over an alphabet Σ* (plus special zero element)
/// - **Addition (⊕):** `lcp(a, b)` - longest common prefix of two strings
/// - **Multiplication (⊗):** `a · b` - string concatenation
/// - **Zero (0̄):** Special marker (represented as `[0xFF]`) - impossible/rejected string
/// - **One (1̄):** Empty string `ε` - identity for concatenation
///
/// # Key Properties
///
/// - **Non-Commutative:** String concatenation is order-dependent (`ab ≠ ba`)
/// - **Idempotent:** LCP operation is idempotent (`lcp(a, a) = a`)
/// - **Left/Right Semiring:** Satisfies distributivity laws
/// - **Path Tracking:** Can track actual string sequences through FST paths
///
/// # Use Cases
///
/// ## String Pattern Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Find common prefix patterns
/// let pattern1 = StringWeight::from_string("programming");
/// let pattern2 = StringWeight::from_string("program");
/// let pattern3 = StringWeight::from_string("progress");
///
/// // Longest common prefix across alternatives
/// let common = pattern1.plus(&pattern2).plus(&pattern3);
/// assert_eq!(common.to_string().unwrap(), "progr");
/// ```
///
/// ## Sequence Concatenation
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build sequences through concatenation
/// let prefix = StringWeight::from_string("pre");
/// let root = StringWeight::from_string("process");
/// let suffix = StringWeight::from_string("ing");
///
/// // Sequential combination
/// let compound = prefix.times(&root).times(&suffix);
/// assert_eq!(compound.to_string().unwrap(), "preprocessing");
/// ```
///
/// ## Automata-Based Text Processing
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build weighted FST for text transformation
/// let mut fst = VectorFst::<StringWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, StringWeight::one());
///
/// // Transform "cat" -> "cats"
/// fst.add_arc(s0, Arc::new(
///     'c' as u32,
///     'c' as u32,
///     StringWeight::from_string("c"),
///     s1
/// ));
///
/// fst.add_arc(s1, Arc::new(
///     'a' as u32,
///     'a' as u32,
///     StringWeight::from_string("a"),
///     s1
/// ));
///
/// // Add suffix transformation
/// fst.add_arc(s1, Arc::new(
///     't' as u32,
///     0,  // epsilon output
///     StringWeight::from_string("ts"),  // Plural transformation
///     s2
/// ));
/// ```
///
/// ## Morphological Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Decompose words into morphological components
/// let stem = StringWeight::from_string("walk");
/// let suffix1 = StringWeight::from_string("ing");   // Progressive
/// let suffix2 = StringWeight::from_string("ed");    // Past tense
///
/// // Generate inflected forms
/// let walking = stem.clone().times(&suffix1);
/// let walked = stem.times(&suffix2);
///
/// assert_eq!(walking.to_string().unwrap(), "walking");
/// assert_eq!(walked.to_string().unwrap(), "walked");
///
/// // Find common stem (using addition for LCP)
/// let common_stem = walking.plus(&walked);
/// assert_eq!(common_stem.to_string().unwrap(), "walk");
/// ```
///
/// ## Phonological Rule Application
/// ```rust
/// use arcweight::prelude::*;
///
/// // Model phonological processes
/// let base_form = StringWeight::from_string("cat");
/// let plural_rule = StringWeight::from_string("s");
/// let liaison_rule = StringWeight::from_string("z");  // Voicing in context
///
/// // Apply phonological rules
/// let surface_form = base_form.times(&plural_rule);
/// let phonetic_form = base_form.times(&liaison_rule);
///
/// // Find common phonetic base
/// let common_base = surface_form.plus(&phonetic_form);
/// assert_eq!(common_base.to_string().unwrap(), "cat");
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let string1 = StringWeight::from_string("hello");
/// let string2 = StringWeight::from_string("help");
///
/// // Addition computes longest common prefix
/// let lcp = string1.clone() + string2.clone();  // "hel"
/// assert_eq!(lcp.to_string().unwrap(), "hel");
///
/// // Multiplication concatenates strings
/// let concat = string1 * string2;  // "hellohelp"
/// assert_eq!(concat.to_string().unwrap(), "hellohelp");
///
/// // Identity elements
/// assert_eq!(StringWeight::zero().as_bytes(), &[0xFF]);  // Special zero marker
/// assert_eq!(StringWeight::one().as_bytes(), &[]);       // Empty string
/// ```
///
/// # Advanced Applications
///
/// ## Edit Distance with String Tracking
/// ```rust
/// use arcweight::prelude::*;
///
/// // Track actual edit operations as strings
/// let insertion = StringWeight::from_string("i:");     // Insert operation
/// let deletion = StringWeight::from_string("d:");      // Delete operation
/// let substitution = StringWeight::from_string("s:");  // Substitute operation
///
/// // Build edit sequence
/// let edit_sequence = insertion
///     .times(&substitution)
///     .times(&deletion);
///
/// assert_eq!(edit_sequence.to_string().unwrap(), "i:s:d:");
/// ```
///
/// ## Longest Common Subsequence
/// ```rust
/// use arcweight::prelude::*;
///
/// // Find common subsequences using LCP
/// let seq1 = StringWeight::from_string("ABCDGH");
/// let seq2 = StringWeight::from_string("AEDFHR");
/// let seq3 = StringWeight::from_string("ABXDGY");
///
/// // Common prefix across all sequences
/// let common = seq1.plus(&seq2).plus(&seq3);
/// assert_eq!(common.to_string().unwrap(), "A");  // Common starting character
/// ```
///
/// ## DNA/RNA Sequence Analysis
/// ```rust
/// use arcweight::prelude::*;
///
/// // Genetic sequence analysis
/// let dna1 = StringWeight::from_string("ATCGATCG");
/// let dna2 = StringWeight::from_string("ATCGTTCG");
/// let dna3 = StringWeight::from_string("ATCGAACG");
///
/// // Find conserved regions (common prefix)
/// let conserved = dna1.plus(&dna2).plus(&dna3);
/// assert_eq!(conserved.to_string().unwrap(), "ATCG");
///
/// // Model sequence concatenation (gene assembly)
/// let gene_segment1 = StringWeight::from_string("ATCG");
/// let gene_segment2 = StringWeight::from_string("GCTA");
/// let assembled_gene = gene_segment1.times(&gene_segment2);
/// assert_eq!(assembled_gene.to_string().unwrap(), "ATCGGCTA");
/// ```
///
/// ## Compiler and Parser Applications
/// ```rust
/// use arcweight::prelude::*;
///
/// // Track syntax patterns
/// let keyword = StringWeight::from_string("if");
/// let condition = StringWeight::from_string("(x > 0)");
/// let block = StringWeight::from_string(" { ... }");
///
/// // Build syntax tree representations
/// let conditional = keyword
///     .times(&condition)
///     .times(&block);
///
/// assert_eq!(conditional.to_string().unwrap(), "if(x > 0) { ... }");
/// ```
///
/// # Byte-Level Operations
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Work with arbitrary byte sequences
/// let bytes1 = StringWeight::from_bytes(vec![0x41, 0x42, 0x43]);  // "ABC"
/// let bytes2 = StringWeight::from_bytes(vec![0x41, 0x42, 0x44]);  // "ABD"
///
/// // LCP works on byte level
/// let common_bytes = bytes1.plus(&bytes2);
/// assert_eq!(common_bytes.as_bytes(), &[0x41, 0x42]);  // "AB"
///
/// // Concatenation preserves byte sequences
/// let combined = bytes1.times(&bytes2);
/// assert_eq!(combined.as_bytes(), &[0x41, 0x42, 0x43, 0x41, 0x42, 0x44]);
/// ```
///
/// # Performance Characteristics
///
/// - **LCP Computation:** O(min(|a|, |b|)) where |a|, |b| are string lengths
/// - **Concatenation:** O(|a| + |b|) with memory allocation for result
/// - **Memory:** Linear in total string length plus Vec overhead
/// - **Comparison:** Lexicographic ordering, O(min(|a|, |b|)) average case
/// - **Storage:** UTF-8 compatible, supports arbitrary byte sequences
///
/// # UTF-8 and Encoding Considerations
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Unicode support through UTF-8
/// let unicode_str = StringWeight::from_string("Hello 世界");
/// assert!(unicode_str.to_string().is_ok());
///
/// // Handle potential encoding errors
/// let invalid_utf8 = StringWeight::from_bytes(vec![0xFF, 0xFE]);
/// assert!(invalid_utf8.to_string().is_err());
///
/// // Graceful fallback for display
/// println!("{}", invalid_utf8);  // Shows byte representation
/// ```
///
/// # Integration with FST Algorithms
///
/// String weights provide unique capabilities in FST algorithms:
/// - **Shortest Path:** Finds paths with specific string properties
/// - **Composition:** Combines string transformations
/// - **Determinization:** Maintains string output while removing nondeterminism
/// - **String-to-String Translation:** Direct implementation of string transducers
///
/// # Mathematical Properties
///
/// The String semiring exhibits important properties:
/// - **Associative:** Both LCP and concatenation are associative
/// - **Non-Commutative:** Order matters in concatenation
/// - **Idempotent Addition:** `lcp(s, s) = s` for any string s
/// - **Identity Elements:** Empty string for multiplication, special marker for addition
/// - **Distributive:** Left and right distributivity hold
///
/// # See Also
///
/// - [Core Concepts - String Semiring](../../docs/core-concepts/semirings.md#string-semiring) for mathematical background
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for optimization-based string processing
/// - [`compose()`](crate::algorithms::compose) for string-to-string transduction
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StringWeight(Vec<u8>);

impl StringWeight {
    /// Empty string (one element)
    pub const EMPTY: Self = Self(Vec::new());

    /// Create from bytes
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Create from string slice
    pub fn from_string(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }

    /// Convert to string
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal byte sequence is not valid UTF-8
    pub fn to_string(&self) -> Result<String, core::str::Utf8Error> {
        core::str::from_utf8(&self.0).map(|s| s.to_string())
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Longest common prefix
    fn lcp(&self, other: &Self) -> Self {
        let len = self.0.len().min(other.0.len());
        let mut i = 0;
        while i < len && self.0[i] == other.0[i] {
            i += 1;
        }
        Self(self.0[..i].to_vec())
    }
}

impl fmt::Display for StringWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_string() {
            Ok(s) => write!(f, "\"{}\"", s),
            Err(_) => write!(f, "{:?}", self.0),
        }
    }
}

impl Zero for StringWeight {
    fn zero() -> Self {
        // special marker for zero (infinity)
        Self(vec![0xFF])
    }

    fn is_zero(&self) -> bool {
        self.0 == vec![0xFF]
    }
}

impl One for StringWeight {
    fn one() -> Self {
        Self::EMPTY
    }
}

impl Add for StringWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) {
            rhs
        } else if <Self as num_traits::Zero>::is_zero(&rhs) {
            self
        } else {
            self.lcp(&rhs)
        }
    }
}

impl Mul for StringWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) || <Self as num_traits::Zero>::is_zero(&rhs) {
            Self::zero()
        } else {
            let mut result = self.0;
            result.extend_from_slice(&rhs.0);
            Self(result)
        }
    }
}

impl Semiring for StringWeight {
    type Value = Vec<u8>;

    fn new(value: Self::Value) -> Self {
        Self(value)
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }

    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            commutative: false,
            idempotent: true,
            path: false,
        }
    }
}
