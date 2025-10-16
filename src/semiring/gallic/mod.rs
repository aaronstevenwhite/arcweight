//! Gallic semiring: combining labels with weights for advanced FST composition
//!
//! The Gallic semiring provides a sophisticated algebraic structure that pairs
//! label sequences with semiring weights. This enables advanced finite-state
//! transducer algorithms, particularly composition operations that need to track
//! output labels as part of the weight structure.
//!
//! # Overview
//!
//! A Gallic weight `(s, w)` combines:
//! - **s**: A sequence of labels (`Vec<Label>`)
//! - **w**: A weight from an arbitrary semiring W
//!
//! The Gallic semiring is parameterized by a **variant** that determines how
//! two weights are combined during addition (⊕). Multiplication (⊗) always
//! concatenates label sequences and multiplies weights.
//!
//! # Architecture
//!
//! This implementation leverages Rust's type system for beautiful, modular design:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │       GallicVariant Trait               │
//! │  (Defines combination behavior)         │
//! └─────────────────────────────────────────┘
//!              ▲
//!              │ implements
//!      ┌───────┼───────┐
//!      │       │       │
//! ┌────────┐ ┌────────┐ ┌────────┐
//! │ Left   │ │ Right  │ │  Min   │ ...
//! │ Gallic │ │ Gallic │ │ Gallic │
//! └────────┘ └────────┘ └────────┘
//!      │       │       │
//!      └───────┼───────┘
//!              ▼
//!   ┌────────────────────────┐
//!   │  GallicWeight<W, V>    │
//!   │  - labels: Vec<Label>  │
//!   │  - weight: W           │
//!   │  - variant: PhantomData<V> │
//!   └────────────────────────┘
//! ```
//!
//! # Variants
//!
//! ## [`LeftGallic`]
//! Uses longest common prefix (LCP) for addition:
//! ```text
//! (s₁, w₁) ⊕ (s₂, w₂) = (lcp(s₁, s₂), w₁ ⊕ w₂)
//! ```
//!
//! ## [`RightGallic`]
//! Uses longest common suffix (LCS) for addition:
//! ```text
//! (s₁, w₁) ⊕ (s₂, w₂) = (lcs(s₁, s₂), w₁ ⊕ w₂)
//! ```
//!
//! ## [`RestrictGallic`]
//! Enforces functional transducer property (labels must match):
//! ```text
//! (s₁, w₁) ⊕ (s₂, w₂) = if s₁ == s₂ { (s₁, w₁ ⊕ w₂) } else { zero }
//! ```
//!
//! ## [`MinGallic`]
//! Selects label sequence with minimum weight:
//! ```text
//! (s₁, w₁) ⊕ (s₂, w₂) = if w₁ ≤ w₂ { (s₁, w₁) } else { (s₂, w₂) }
//! ```
//!
//! ## [`UnionGallic`]
//! General union semantics (default):
//! ```text
//! (s₁, w₁) ⊕ (s₂, w₂) = (lcp(s₁, s₂), w₁ ⊕ w₂)
//! ```
//!
//! # Examples
//!
//! ## Basic Usage with LeftGallic
//!
//! ```rust
//! use arcweight::prelude::*;
//! use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
//!
//! // Create Gallic weights with label sequences
//! let path1 = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
//!     &[1, 2, 3],
//!     TropicalWeight::new(1.0)
//! );
//! let path2 = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
//!     &[1, 2, 4],
//!     TropicalWeight::new(2.0)
//! );
//!
//! // Addition uses LCP
//! let combined = path1 + path2;
//! assert_eq!(combined.labels(), &[1, 2]); // Common prefix
//! assert_eq!(*combined.weight().value(), 1.0); // min(1.0, 2.0)
//! ```
//!
//! ## Path Selection with MinGallic
//!
//! ```rust
//! use arcweight::semiring::gallic::{GallicWeight, MinGallic};
//! use arcweight::semiring::{Semiring, TropicalWeight};
//!
//! type PathWeight = GallicWeight<TropicalWeight, MinGallic>;
//!
//! let path1 = PathWeight::from_labels(&[1, 2], TropicalWeight::new(5.0));
//! let path2 = PathWeight::from_labels(&[3, 4], TropicalWeight::new(3.0));
//!
//! // MinGallic selects minimum weight path
//! let shortest = path1 + path2;
//! assert_eq!(shortest.labels(), &[3, 4]); // Path with cost 3.0
//! assert_eq!(*shortest.weight().value(), 3.0);
//! ```
//!
//! ## Label Concatenation
//!
//! ```rust
//! use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
//! use arcweight::semiring::{Semiring, TropicalWeight};
//!
//! let seg1 = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
//!     &[1, 2],
//!     TropicalWeight::new(1.0)
//! );
//! let seg2 = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
//!     &[3, 4],
//!     TropicalWeight::new(2.0)
//! );
//!
//! // Multiplication concatenates labels
//! let full_path = seg1 * seg2;
//! assert_eq!(full_path.labels(), &[1, 2, 3, 4]);
//! assert_eq!(*full_path.weight().value(), 3.0);
//! ```
//!
//! ## Functional Transducer Enforcement
//!
//! ```rust
//! use arcweight::semiring::gallic::{GallicWeight, RestrictGallic};
//! use arcweight::semiring::TropicalWeight;
//! use arcweight::semiring::Semiring;
//!
//! type FunctionalWeight = GallicWeight<TropicalWeight, RestrictGallic>;
//!
//! let w1 = FunctionalWeight::from_labels(&[1, 2], TropicalWeight::new(1.0));
//! let w2 = FunctionalWeight::from_labels(&[1, 2], TropicalWeight::new(2.0));
//! let w3 = FunctionalWeight::from_labels(&[3, 4], TropicalWeight::new(1.0));
//!
//! // Valid: identical labels
//! let valid = w1.clone() + w2;
//! assert!(!valid.weight().is_zero());
//!
//! // Invalid: different labels → zero weight
//! let invalid = w1 + w3;
//! assert!(invalid.weight().is_zero()); // Signals violation
//! ```
//!
//! ## Converting Between Variants
//!
//! ```rust
//! use arcweight::semiring::gallic::{GallicWeight, LeftGallic, MinGallic};
//! use arcweight::semiring::TropicalWeight;
//!
//! let left_weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
//!     &[1, 2, 3],
//!     TropicalWeight::new(5.0)
//! );
//!
//! // Convert to MinGallic variant
//! let min_weight: GallicWeight<TropicalWeight, MinGallic> =
//!     left_weight.into_variant();
//! ```
//!
//! # Type Aliases
//!
//! Common variant combinations for convenience:
//!
//! ```rust
//! use arcweight::semiring::gallic::{GallicWeight, LeftGallic, MinGallic};
//! use arcweight::semiring::TropicalWeight;
//!
//! // Explicit type aliases
//! type LeftGallicWeight<W> = GallicWeight<W, LeftGallic>;
//! type MinGallicWeight<W> = GallicWeight<W, MinGallic>;
//!
//! // UnionGallic is the default
//! type DefaultGallic<W> = GallicWeight<W>; // UnionGallic implicit
//! ```
//!
//! # Use Cases
//!
//! ## FST Composition with Output Tracking
//! Track output labels as weights during composition operations
//!
//! ## Shortest Path with Label Sequences
//! Find optimal paths while preserving the label sequences
//!
//! ## Viterbi Decoding
//! Track state sequences alongside probabilities
//!
//! ## Machine Translation
//! Track word alignments with translation scores
//!
//! ## Speech Recognition
//! Maintain phone sequences with acoustic scores
//!
//! # Performance
//!
//! - **Plus**: Variant-dependent (O(1) for MinGallic, O(n) for LeftGallic)
//! - **Times**: O(|s₁| + |s₂|) for label concatenation
//! - **Memory**: Linear in total label count
//! - **Zero-cost variant selection**: Compile-time dispatch via PhantomData
//!
//! # Design Principles
//!
//! This implementation demonstrates:
//! 1. **Type-level programming**: Variants as zero-sized types
//! 2. **Trait-based modularity**: `GallicVariant` trait for extensibility
//! 3. **Zero-cost abstractions**: No runtime overhead for variant selection
//! 4. **Compile-time safety**: Type system prevents variant mixing
//! 5. **Idiomatic Rust**: Leverages generics, traits, and PhantomData
//!
//! # See Also
//!
//! - [`GallicVariant`] - Trait for implementing custom variants
//! - [`GallicWeight`] - Main weight type
//! - Individual variant types: [`LeftGallic`], [`MinGallic`], etc.
//!
//! # References
//!
//! - Mohri, M. (2002). "Semiring Frameworks and Algorithms for Shortest-Distance Problems"
//! - Allauzen, C., & Mohri, M. (2009). "Linear-Space Composition of Finite-State Transducers"
//! - Cognetta, M., & Allauzen, C. (2025). "Tutorial: φ-Transductions in OpenFst via the Gallic Semiring"

pub mod variant;
pub mod variants;
mod weight;
mod weight_semiring;

pub use variant::{longest_common_prefix, longest_common_suffix, GallicVariant};
pub use variants::{LeftGallic, MinGallic, RestrictGallic, RightGallic, UnionGallic};
pub use weight::GallicWeight;

// Type aliases for common combinations
/// Left Gallic weight with specified base semiring
pub type LeftGallicWeight<W> = GallicWeight<W, LeftGallic>;

/// Right Gallic weight with specified base semiring
pub type RightGallicWeight<W> = GallicWeight<W, RightGallic>;

/// Restricted Gallic weight with specified base semiring
pub type RestrictGallicWeight<W> = GallicWeight<W, RestrictGallic>;

/// Min Gallic weight with specified base semiring
pub type MinGallicWeight<W> = GallicWeight<W, MinGallic>;

/// Standard Gallic weight (UnionGallic variant)
pub type StandardGallicWeight<W> = GallicWeight<W, UnionGallic>;
