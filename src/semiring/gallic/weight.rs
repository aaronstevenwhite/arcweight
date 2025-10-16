//! Gallic weight implementation combining labels with semiring weights

use super::variant::GallicVariant;
use super::variants::UnionGallic;
use crate::fst::Label;
use crate::semiring::Semiring;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Gallic weight combining a label sequence with a semiring weight
///
/// `GallicWeight` is a sophisticated semiring that pairs a sequence of labels
/// (FST arc labels) with an arbitrary semiring weight. This enables advanced
/// FST algorithms that need to track output labels as part of the weight structure,
/// particularly in composition operations.
///
/// # Type Parameters
///
/// - `W`: The underlying semiring weight type
/// - `V`: The Gallic variant determining addition behavior (defaults to `UnionGallic`)
///
/// # Mathematical Structure
///
/// A Gallic weight `(s, w)` consists of:
/// - `s`: A sequence of labels (`Vec<Label>`)
/// - `w`: A weight from semiring W
///
/// Operations depend on the variant `V` (see [`GallicVariant`]):
/// - **Addition (⊕)**: Variant-specific label combination + weight addition
/// - **Multiplication (⊗)**: Label concatenation + weight multiplication
///
/// # Design Philosophy
///
/// This implementation demonstrates:
/// 1. **Type-level variant selection**: Zero-cost abstraction via `PhantomData<V>`
/// 2. **Modular behavior**: Each variant implements `GallicVariant` trait
/// 3. **Full semiring compliance**: All axioms satisfied
/// 4. **Ergonomic API**: Rich set of constructors and accessors
/// 5. **Idiomatic Rust**: Leverages type system for compile-time guarantees
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
///
/// // Create with LeftGallic variant
/// let weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
///     &[1, 2, 3],
///     TropicalWeight::new(2.5)
/// );
///
/// assert_eq!(weight.labels(), &[1, 2, 3]);
/// assert_eq!(*weight.weight().value(), 2.5);
/// ```
///
/// ## Addition (Variant-Dependent)
///
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// type PathWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// let path1 = PathWeight::from_labels(&[1, 2], TropicalWeight::new(5.0));
/// let path2 = PathWeight::from_labels(&[3, 4], TropicalWeight::new(3.0));
///
/// // MinGallic selects minimum weight path
/// let best = path1 + path2;
/// assert_eq!(best.labels(), &[3, 4]); // Lower cost path
/// assert_eq!(*best.weight().value(), 3.0);
/// ```
///
/// ## Multiplication (Concatenation)
///
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// let seg1 = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
///     &[1, 2],
///     TropicalWeight::new(1.0)
/// );
/// let seg2 = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
///     &[3, 4],
///     TropicalWeight::new(2.0)
/// );
///
/// let concatenated = seg1 * seg2;
/// assert_eq!(concatenated.labels(), &[1, 2, 3, 4]); // Concatenated
/// assert_eq!(*concatenated.weight().value(), 3.0);  // 1.0 + 2.0
/// ```
///
/// ## Using Different Variants
///
/// ```rust
/// use arcweight::semiring::gallic::{GallicWeight, LeftGallic, MinGallic, RestrictGallic};
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// // Left: Uses longest common prefix
/// type LeftWeight = GallicWeight<TropicalWeight, LeftGallic>;
///
/// // Min: Selects minimum weight
/// type MinWeight = GallicWeight<TropicalWeight, MinGallic>;
///
/// // Restrict: Requires identical labels
/// type RestrictWeight = GallicWeight<TropicalWeight, RestrictGallic>;
///
/// // Different semantics for same operations
/// let labels1 = vec![1, 2, 3];
/// let labels2 = vec![1, 2, 4];
/// let w = TropicalWeight::new(1.0);
///
/// let left1 = LeftWeight::from_labels(&labels1, w.clone());
/// let left2 = LeftWeight::from_labels(&labels2, w.clone());
/// assert_eq!((left1 + left2).labels(), &[1, 2]); // Common prefix
///
/// let min1 = MinWeight::from_labels(&labels1, TropicalWeight::new(5.0));
/// let min2 = MinWeight::from_labels(&labels2, TropicalWeight::new(3.0));
/// assert_eq!((min1 + min2).labels(), &[1, 2, 4]); // Min weight labels
/// ```
///
/// # Type Aliases
///
/// Common variant combinations:
///
/// ```rust
/// use arcweight::semiring::gallic::{
///     GallicWeight, LeftGallic, RightGallic, MinGallic, RestrictGallic
/// };
/// use arcweight::semiring::{Semiring, TropicalWeight};
///
/// type LeftGallicWeight<W> = GallicWeight<W, LeftGallic>;
/// type RightGallicWeight<W> = GallicWeight<W, RightGallic>;
/// type MinGallicWeight<W> = GallicWeight<W, MinGallic>;
/// type RestrictGallicWeight<W> = GallicWeight<W, RestrictGallic>;
///
/// // Default variant is UnionGallic
/// type SimpleGallic<W> = GallicWeight<W>; // UnionGallic implicit
/// ```
///
/// # Thread Safety
///
/// `GallicWeight` is `Send + Sync` when the underlying weight `W` is `Send + Sync`,
/// making it safe for concurrent FST operations.
///
/// # Performance Considerations
///
/// - **Label storage**: Uses `Vec<Label>` for flexibility
/// - **Clone cost**: O(label_count + weight_clone)
/// - **Plus cost**: Depends on variant (O(1) for MinGallic, O(n) for LeftGallic)
/// - **Times cost**: O(|s₁| + |s₂|) for concatenation
/// - **Memory**: Linear in label sequence length
///
/// # See Also
///
/// - [`GallicVariant`] trait for implementing custom variants
/// - [`LeftGallic`](super::LeftGallic), [`MinGallic`](super::MinGallic), etc. for variant types
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GallicWeight<W: Semiring, V: GallicVariant = UnionGallic> {
    /// Sequence of labels accumulated through FST paths
    pub(crate) labels: Vec<Label>,

    /// Semiring weight (cost, probability, etc.)
    pub(crate) weight: W,

    /// Zero-sized type parameter for variant selection
    /// This enables compile-time dispatch with zero runtime cost
    #[cfg_attr(feature = "serde", serde(skip))]
    _variant: PhantomData<V>,
}

// === Constructors ===

impl<W: Semiring, V: GallicVariant> GallicWeight<W, V> {
    /// Create a new Gallic weight from labels and weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, LeftGallic>::new(
    ///     vec![1, 2, 3],
    ///     TropicalWeight::new(5.0)
    /// );
    /// ```
    pub fn new(labels: Vec<Label>, weight: W) -> Self {
        Self {
            labels,
            weight,
            _variant: PhantomData,
        }
    }

    /// Create from a label slice and weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
    ///     &[1, 2, 3],
    ///     TropicalWeight::new(2.0)
    /// );
    /// ```
    pub fn from_labels(labels: &[Label], weight: W) -> Self {
        Self::new(labels.to_vec(), weight)
    }

    /// Create from a single label and weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, MinGallic>::from_label(
    ///     42,
    ///     TropicalWeight::new(1.5)
    /// );
    /// assert_eq!(weight.labels(), &[42]);
    /// ```
    pub fn from_label(label: Label, weight: W) -> Self {
        Self::new(vec![label], weight)
    }

    /// Create with empty label sequence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, LeftGallic>::empty(
    ///     TropicalWeight::new(3.0)
    /// );
    /// assert!(weight.labels().is_empty());
    /// ```
    pub fn empty(weight: W) -> Self {
        Self::new(Vec::new(), weight)
    }

    /// Get the label sequence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
    ///     &[10, 20, 30],
    ///     TropicalWeight::new(1.0)
    /// );
    /// assert_eq!(weight.labels(), &[10, 20, 30]);
    /// ```
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Get the weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
    ///     &[1, 2],
    ///     TropicalWeight::new(5.5)
    /// );
    /// assert_eq!(*weight.weight().value(), 5.5);
    /// ```
    pub fn weight(&self) -> &W {
        &self.weight
    }

    /// Decompose into labels and weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let gallic_weight = GallicWeight::<TropicalWeight, MinGallic>::from_labels(
    ///     &[1, 2, 3],
    ///     TropicalWeight::new(2.0)
    /// );
    /// let (labels, weight) = gallic_weight.into_parts();
    /// assert_eq!(labels, vec![1, 2, 3]);
    /// assert_eq!(*weight.value(), 2.0);
    /// ```
    pub fn into_parts(self) -> (Vec<Label>, W) {
        (self.labels, self.weight)
    }

    /// Get a cloned copy of the label sequence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
    ///     &[5, 6],
    ///     TropicalWeight::new(1.0)
    /// );
    /// let labels_copy = weight.clone_labels();
    /// assert_eq!(labels_copy, vec![5, 6]);
    /// ```
    pub fn clone_labels(&self) -> Vec<Label> {
        self.labels.clone()
    }

    /// Get a cloned copy of the weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
    ///     &[1],
    ///     TropicalWeight::new(3.0)
    /// );
    /// let weight_copy = weight.clone_weight();
    /// assert_eq!(*weight_copy.value(), 3.0);
    /// ```
    pub fn clone_weight(&self) -> W {
        self.weight.clone()
    }

    /// Convert to a different Gallic variant
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic, MinGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let left_weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
    ///     &[1, 2, 3],
    ///     TropicalWeight::new(2.0)
    /// );
    ///
    /// // Convert to MinGallic variant
    /// let min_weight: GallicWeight<TropicalWeight, MinGallic> =
    ///     left_weight.into_variant();
    /// assert_eq!(min_weight.labels(), &[1, 2, 3]);
    /// ```
    pub fn into_variant<V2: GallicVariant>(self) -> GallicWeight<W, V2> {
        GallicWeight::new(self.labels, self.weight)
    }

    /// Map a function over the labels
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, UnionGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, UnionGallic>::from_labels(
    ///     &[1, 2, 3],
    ///     TropicalWeight::new(1.0)
    /// );
    ///
    /// let doubled = weight.map_labels(|labels| {
    ///     labels.iter().map(|&l| l * 2).collect()
    /// });
    /// assert_eq!(doubled.labels(), &[2, 4, 6]);
    /// ```
    pub fn map_labels<F>(self, f: F) -> Self
    where
        F: FnOnce(Vec<Label>) -> Vec<Label>,
    {
        Self::new(f(self.labels), self.weight)
    }

    /// Map a function over the weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, LeftGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, LeftGallic>::from_labels(
    ///     &[1, 2],
    ///     TropicalWeight::new(2.0)
    /// );
    ///
    /// let doubled_weight = weight.map_weight(|w| w.times(&TropicalWeight::new(2.0)));
    /// assert_eq!(*doubled_weight.weight().value(), 4.0);
    /// ```
    pub fn map_weight<F>(self, f: F) -> Self
    where
        F: FnOnce(W) -> W,
    {
        Self::new(self.labels, f(self.weight))
    }

    /// Map functions over both labels and weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::semiring::gallic::{GallicWeight, MinGallic};
    /// use arcweight::semiring::{Semiring, TropicalWeight};
    ///
    /// let weight = GallicWeight::<TropicalWeight, MinGallic>::from_labels(
    ///     &[1, 2],
    ///     TropicalWeight::new(3.0)
    /// );
    ///
    /// let modified = weight.map_both(
    ///     |labels| labels.into_iter().rev().collect(),
    ///     |w| w.times(&TropicalWeight::new(2.0))
    /// );
    /// assert_eq!(modified.labels(), &[2, 1]);
    /// assert_eq!(*modified.weight().value(), 5.0);
    /// ```
    pub fn map_both<F1, F2>(self, f_labels: F1, f_weight: F2) -> Self
    where
        F1: FnOnce(Vec<Label>) -> Vec<Label>,
        F2: FnOnce(W) -> W,
    {
        Self::new(f_labels(self.labels), f_weight(self.weight))
    }
}

// === Display ===

impl<W: Semiring, V: GallicVariant> fmt::Display for GallicWeight<W, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {})", self.labels, self.weight)
    }
}

// === Equality and Ordering ===

impl<W: Semiring, V: GallicVariant> PartialEq for GallicWeight<W, V> {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels && self.weight == other.weight
    }
}

impl<W: Semiring, V: GallicVariant> Eq for GallicWeight<W, V> {}

impl<W: Semiring, V: GallicVariant> PartialOrd for GallicWeight<W, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Lexicographic ordering: first by labels, then by weight
        match self.labels.partial_cmp(&other.labels) {
            Some(std::cmp::Ordering::Equal) => self.weight.partial_cmp(&other.weight),
            other => other,
        }
    }
}

impl<W, V> Ord for GallicWeight<W, V>
where
    W: Semiring + Ord,
    V: GallicVariant,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.labels.cmp(&other.labels) {
            std::cmp::Ordering::Equal => self.weight.cmp(&other.weight),
            other => other,
        }
    }
}

impl<W, V> std::hash::Hash for GallicWeight<W, V>
where
    W: Semiring + std::hash::Hash,
    V: GallicVariant,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.labels.hash(state);
        self.weight.hash(state);
    }
}

// === Zero and One ===

impl<W: Semiring, V: GallicVariant> Zero for GallicWeight<W, V> {
    fn zero() -> Self {
        Self::new(Vec::new(), W::zero())
    }

    fn is_zero(&self) -> bool {
        Semiring::is_zero(&self.weight)
    }
}

impl<W: Semiring, V: GallicVariant> One for GallicWeight<W, V> {
    fn one() -> Self {
        Self::new(Vec::new(), W::one())
    }
}

// === Operator Overloads ===

impl<W: Semiring, V: GallicVariant> Add for GallicWeight<W, V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let (labels, weight) = V::plus(&self.labels, &self.weight, &rhs.labels, &rhs.weight);
        Self::new(labels, weight)
    }
}

impl<W: Semiring, V: GallicVariant> Mul for GallicWeight<W, V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Concatenate label sequences
        let mut labels = self.labels;
        labels.extend_from_slice(&rhs.labels);

        // Multiply weights
        let weight = self.weight.times(&rhs.weight);

        Self::new(labels, weight)
    }
}

// Continued in next message due to length...
