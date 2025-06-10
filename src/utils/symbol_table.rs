//! Symbol table for mapping labels to strings

use std::collections::HashMap;
use crate::fst::Label;

/// Symbol table for FST labels
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    symbols: Vec<String>,
    symbol_map: HashMap<String, Label>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    pub fn new() -> Self {
        let mut table = Self::default();
        // epsilon is always symbol 0
        table.add_symbol("<eps>");
        table
    }
    
    /// Add a symbol
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