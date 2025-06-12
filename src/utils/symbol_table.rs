//! Symbol table for mapping labels to strings

use crate::fst::Label;
use std::collections::HashMap;

/// Symbol table for FST labels
/// 
/// Maps between human-readable symbols (strings) and numeric labels used internally
/// by FSTs. Essential for text I/O and human-readable FST representations.
/// 
/// The symbol table automatically includes epsilon ("<eps>") as label 0.
/// 
/// # Examples
/// 
/// ```
/// use arcweight::prelude::*;
/// 
/// let mut syms = SymbolTable::new();
/// 
/// // Add symbols and get their numeric IDs
/// let hello_id = syms.add_symbol("hello");
/// let world_id = syms.add_symbol("world");
/// let eos_id = syms.add_symbol("</s>");
/// 
/// // Lookup by ID
/// assert_eq!(syms.find(hello_id), Some("hello"));
/// assert_eq!(syms.find(0), Some("<eps>")); // epsilon is always 0
/// 
/// // Lookup by symbol
/// assert_eq!(syms.find_id("world"), Some(world_id));
/// assert_eq!(syms.find_id("missing"), None);
/// 
/// // Use in FST construction
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new(hello_id, world_id, TropicalWeight::one(), s1));
/// 
/// assert_eq!(fst.num_arcs(s0), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    symbols: Vec<String>,
    symbol_map: HashMap<String, Label>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    /// 
    /// The table starts with epsilon ("<eps>") as symbol 0.
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
