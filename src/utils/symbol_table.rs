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
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}