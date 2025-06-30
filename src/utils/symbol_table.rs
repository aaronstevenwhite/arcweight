//! Symbol table for mapping labels to strings

use crate::fst::Label;
use std::collections::HashMap;

/// Symbol table for mapping between human-readable strings and numeric FST labels
///
/// `SymbolTable` provides bidirectional mapping between symbolic names (strings) and
/// the numeric labels used internally by FSTs. This is essential for human-readable
/// FST construction, debugging, serialization, and text processing applications.
///
/// # Design Principles
///
/// - **Bijective Mapping:** Each symbol maps to exactly one label and vice versa
/// - **Epsilon Handling:** Label 0 is reserved for epsilon (`<eps>`) transitions
/// - **Efficient Lookup:** O(1) access in both directions using Vec and HashMap
/// - **Incremental Construction:** Symbols can be added dynamically as needed
/// - **Memory Efficiency:** Stores symbols compactly with minimal overhead
///
/// # Core Concepts
///
/// ## Label 0 (Epsilon)
/// Label 0 is universally reserved for epsilon transitions in FST theory.
/// The symbol table automatically includes `"<eps>"` at label 0.
///
/// ## Symbol Assignment
/// New symbols are assigned consecutive labels starting from 1, ensuring
/// no collisions and maintaining deterministic ordering.
///
/// # Use Cases
///
/// ## Text Processing Applications
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build vocabulary for text analysis
/// let mut vocab = SymbolTable::new();
///
/// // Add linguistic symbols
/// let word_boundary = vocab.add_symbol("<wb>");
/// let sentence_end = vocab.add_symbol("</s>");
/// let unknown_word = vocab.add_symbol("<unk>");
///
/// // Add word tokens
/// let words = ["the", "quick", "brown", "fox"];
/// let wordᵢds: Vec<_> = words.iter()
///     .map(|&word| vocab.add_symbol(word))
///     .collect();
///
/// // Use in FST construction for language modeling
/// let mut lm_fst = VectorFst::<LogWeight>::new();
/// // ... build language model using symbolic labels
/// ```
///
/// ## FST Construction with Meaningful Labels
/// ```rust
/// use arcweight::prelude::*;
///
/// // Phonetic transcription FST
/// let mut phonemes = SymbolTable::new();
/// let mut graphemes = SymbolTable::new();
///
/// // Add phonetic symbols
/// let p_ae = phonemes.add_symbol("ae");  // /æ/ in "cat"
/// let p_t = phonemes.add_symbol("t");    // /t/ sound
///
/// // Add graphemic symbols  
/// let g_a = graphemes.add_symbol("a");
/// let g_t = graphemes.add_symbol("t");
///
/// // Build pronunciation FST: "at" -> /æt/
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // Grapheme 'a' -> phoneme /æ/
/// fst.add_arc(s0, Arc::new(g_a, p_ae, TropicalWeight::one(), s1));
/// // Grapheme 't' -> phoneme /t/
/// fst.add_arc(s1, Arc::new(g_t, p_t, TropicalWeight::one(), s2));
/// ```
///
/// ## Symbol Table Synchronization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Ensure consistent symbol mappings across FSTs
/// fn build_translation_pipeline() -> std::result::Result<(VectorFst<TropicalWeight>, SymbolTable, SymbolTable), Box<dyn std::error::Error>> {
///     let mut source_vocab = SymbolTable::new();
///     let mut target_vocab = SymbolTable::new();
///     
///     // Build parallel vocabularies
///     let source_words = ["hello", "world", "good", "morning"];
///     let target_words = ["hola", "mundo", "bueno", "mañana"];
///     
///     let sourceᵢds: Vec<_> = source_words.iter()
///         .map(|&word| source_vocab.add_symbol(word))
///         .collect();
///     
///     let targetᵢds: Vec<_> = target_words.iter()
///         .map(|&word| target_vocab.add_symbol(word))
///         .collect();
///     
///     // Build translation FST with aligned vocabularies
///     let mut translation_fst = VectorFst::new();
///     let s0 = translation_fst.add_state();
///     translation_fst.set_start(s0);
///     translation_fst.set_final(s0, TropicalWeight::one());
///     
///     // Add translation arcs
///     for (sourceᵢd, targetᵢd) in sourceᵢds.iter().zip(targetᵢds.iter()) {
///         translation_fst.add_arc(s0, Arc::new(
///             *sourceᵢd, *targetᵢd,
///             TropicalWeight::one(),
///             s0
///         ));
///     }
///     
///     Ok((translation_fst, source_vocab, target_vocab))
/// }
/// ```
///
/// ## Debugging and Visualization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create human-readable FST descriptions
/// fn print_fst_arcs(
///     fst: &VectorFst<TropicalWeight>,
///     input_syms: &SymbolTable,
///     output_syms: &SymbolTable
/// ) {
///     for state in fst.states() {
///         for arc in fst.arcs(state) {
///             let input_sym = input_syms.find(arc.ilabel).unwrap_or("<unknown>");
///             let output_sym = output_syms.find(arc.olabel).unwrap_or("<unknown>");
///             
///             println!("State {} --{}:{}-- State {}",
///                      state, input_sym, output_sym, arc.nextstate);
///         }
///     }
/// }
/// ```
///
/// # Performance Characteristics
///
/// | Operation | Time Complexity | Space Complexity |
/// |-----------|----------------|------------------|
/// | `add_symbol` | O(1) amortized | O(n) total for n symbols |
/// | `find` (by label) | O(1) | O(1) |
/// | `find_id` (by string) | O(1) average | O(1) |
/// | `size` | O(1) | O(1) |
///
/// # Memory Layout
///
/// - **String Storage:** `Vec<String>` for label → symbol mapping
/// - **Reverse Lookup:** `HashMap<String, Label>` for symbol → label mapping
/// - **Memory Overhead:** ~40 bytes per symbol (string + HashMap entry)
/// - **Cache Efficiency:** Sequential access is cache-friendly for label lookups
///
/// # Thread Safety
///
/// `SymbolTable` is not `Sync` by default due to internal HashMap usage.
/// For concurrent access, wrap in appropriate synchronization primitives:
///
/// ```rust
/// use std::sync::{Arc, RwLock};
/// use arcweight::prelude::*;
///
/// let shared_symbols = Arc::new(RwLock::new(SymbolTable::new()));
///
/// // Read access
/// let symbols = shared_symbols.read().unwrap();
/// let label = symbols.find_id("example");
/// drop(symbols);
///
/// // Write access  
/// let mut symbols = shared_symbols.write().unwrap();
/// let new_label = symbols.add_symbol("new_word");
/// ```
///
/// # Integration with FST I/O
///
/// Symbol tables are essential for FST serialization formats:
/// - **Text Format:** Human-readable FST descriptions
/// - **Binary Format:** Compact symbol encoding with symbol table headers
/// - **OpenFST Compatibility:** Standard symbol table format support
///
/// # See Also
///
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for usage patterns
/// - [`Arc`](crate::arc::Arc) for the arc type that uses symbolic labels
/// - [I/O module](crate::io) for serialization with symbol tables
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    symbols: Vec<String>,
    symbol_map: HashMap<String, Label>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    ///
    /// The table starts with epsilon (`<eps>`) as symbol 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let syms = SymbolTable::new();
    /// assert_eq!(syms.len(), 1); // contains epsilon
    /// assert_eq!(syms.find(0), Some("<eps>"));
    /// ```
    pub fn new() -> Self {
        let mut table = Self::default();
        // epsilon is always symbol 0
        table.add_symbol("<eps>");
        table
    }

    /// Add a symbol
    ///
    /// Returns the numeric ID for the symbol. If the symbol already exists,
    /// returns its existing ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut syms = SymbolTable::new();
    ///
    /// let first_id = syms.add_symbol("hello");
    /// let second_id = syms.add_symbol("hello"); // same symbol
    ///
    /// assert_eq!(first_id, second_id); // returns same ID
    /// assert_eq!(syms.len(), 2); // epsilon + hello
    /// ```
    pub fn add_symbol(&mut self, symbol: &str) -> Label {
        if let Some(&id) = self.symbol_map.get(symbol) {
            id
        } else {
            let id = self.symbols.len() as Label;
            self.symbols.push(symbol.to_string());
            self.symbol_map.insert(symbol.to_string(), id);
            id
        }
    }

    /// Find a symbol by ID
    pub fn find(&self, id: Label) -> Option<&str> {
        self.symbols.get(id as usize).map(|s| s.as_str())
    }

    /// Find ID by symbol
    pub fn find_id(&self, symbol: &str) -> Option<Label> {
        self.symbol_map.get(symbol).copied()
    }

    /// Number of symbols
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Number of symbols (alias for len)
    pub fn size(&self) -> usize {
        self.len()
    }

    /// Find ID by symbol (alias for find_id)
    pub fn find_symbol(&self, symbol: &str) -> Option<Label> {
        self.find_id(symbol)
    }

    /// Find symbol by ID (alias for find)
    pub fn find_key(&self, id: Label) -> Option<&str> {
        self.find(id)
    }

    /// Check if contains symbol
    pub fn contains_symbol(&self, symbol: &str) -> bool {
        self.symbol_map.contains_key(symbol)
    }

    /// Check if contains key (ID)
    pub fn contains_key(&self, id: Label) -> bool {
        (id as usize) < self.symbols.len()
    }

    /// Clear all symbols
    pub fn clear(&mut self) {
        self.symbols.clear();
        self.symbol_map.clear();
        // Re-add epsilon
        self.add_symbol("<eps>");
    }

    /// Get all symbols
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.symbols.iter().map(|s| s.as_str())
    }

    /// Get all keys (IDs)
    pub fn keys(&self) -> impl Iterator<Item = Label> {
        0..self.symbols.len() as Label
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}
