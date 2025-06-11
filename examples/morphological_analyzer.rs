//! Finite State Morphology Example
//!
//! This example demonstrates finite state morphological analysis following the
//! foundational work of Lauri Karttunen and subsequent developments in computational
//! morphology. It shows:
//!
//! 1. Two-level morphology principles (Koskenniemi & Karttunen)
//! 2. Lexicon construction and morphotactics
//! 3. Classic examples: Finnish morphology, English derivation, agglutination
//! 4. Morphophonological alternations and surface realization
//! 5. Bidirectional morphological processing (analysis ↔ generation)
//!
//! This implementation follows the theoretical framework established in:
//! - Karttunen, L. (1993). Finite-state lexicon compiler
//! - Karttunen, L. & Beesley, K. (2001). Finite State Morphology
//! - Koskenniemi, K. (1983). Two-level morphology
//! - Karttunen, L. (1994). Constructing lexical transducers
//!
//! Usage:
//! ```bash
//! cargo run --example morphological_analyzer
//! ```

use arcweight::prelude::*;
use std::collections::HashMap;

/// Morphological categories following Karttunen's taxonomy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)] // Some variants for API completeness
enum MorphCategory {
    // Major lexical categories
    Noun, Verb, Adjective, Adverb,
    
    // Grammatical features
    Singular, Plural, Dual,
    Nominative, Genitive, Partitive, Accusative, Ablative, Allative,
    Elative, Illative, Inessive, Adessive, Essive, Translative,
    
    // Verbal features
    Present, Past, Conditional, Imperative,
    FirstPerson, SecondPerson, ThirdPerson,
    Active, Passive,
    
    // Derivational features
    Agent, Diminutive, Augmentative, Abstract,
    Causative, Frequentative,
    
    // Morpheme types
    Root, Stem, Suffix, Prefix,
    InflectionalSuffix, DerivationalSuffix,
}

impl MorphCategory {
    fn to_tag(&self) -> &'static str {
        match self {
            MorphCategory::Noun => "N",
            MorphCategory::Verb => "V", 
            MorphCategory::Adjective => "A",
            MorphCategory::Adverb => "Adv",
            
            MorphCategory::Singular => "Sg",
            MorphCategory::Plural => "Pl",
            MorphCategory::Dual => "Du",
            
            MorphCategory::Nominative => "Nom",
            MorphCategory::Genitive => "Gen", 
            MorphCategory::Partitive => "Part",
            MorphCategory::Accusative => "Acc",
            MorphCategory::Ablative => "Abl",
            MorphCategory::Allative => "All",
            MorphCategory::Elative => "Ela",
            MorphCategory::Illative => "Ill",
            MorphCategory::Inessive => "Ine",
            MorphCategory::Adessive => "Ade",
            MorphCategory::Essive => "Ess",
            MorphCategory::Translative => "Tra",
            
            MorphCategory::Present => "Pres",
            MorphCategory::Past => "Past",
            MorphCategory::Conditional => "Cond",
            MorphCategory::Imperative => "Imp",
            
            MorphCategory::FirstPerson => "1",
            MorphCategory::SecondPerson => "2", 
            MorphCategory::ThirdPerson => "3",
            
            MorphCategory::Active => "Act",
            MorphCategory::Passive => "Pass",
            
            MorphCategory::Agent => "Ag",
            MorphCategory::Diminutive => "Dim",
            MorphCategory::Augmentative => "Aug",
            MorphCategory::Abstract => "Abstr",
            MorphCategory::Causative => "Caus",
            MorphCategory::Frequentative => "Freq",
            
            MorphCategory::Root => "Root",
            MorphCategory::Stem => "Stem",
            MorphCategory::Suffix => "Suff",
            MorphCategory::Prefix => "Pref",
            MorphCategory::InflectionalSuffix => "ISuff",
            MorphCategory::DerivationalSuffix => "DSuff",
        }
    }
}

/// Morphological analysis result
#[derive(Debug, Clone)]
struct MorphAnalysis {
    _surface_form: String,
    lexical_form: String,
    morphemes: Vec<String>,
    categories: Vec<MorphCategory>,
    gloss: String,
}

/// Finite state morphological lexicon following Karttunen's approach
struct FiniteStateLexicon {
    // Lexical entries organized by category
    noun_stems: HashMap<String, Vec<MorphCategory>>,
    verb_stems: HashMap<String, Vec<MorphCategory>>,
    adjective_stems: HashMap<String, Vec<MorphCategory>>,
    
    // Affixes with their morphological properties
    noun_suffixes: HashMap<String, Vec<MorphCategory>>,
    verb_suffixes: HashMap<String, Vec<MorphCategory>>,
    derivational_suffixes: HashMap<String, Vec<MorphCategory>>,
    
    // Morphophonological alternations
    phonological_rules: Vec<(String, String, String)>, // (context, input, output)
}

impl FiniteStateLexicon {
    fn new() -> Self {
        let mut lexicon = FiniteStateLexicon {
            noun_stems: HashMap::new(),
            verb_stems: HashMap::new(),
            adjective_stems: HashMap::new(),
            noun_suffixes: HashMap::new(),
            verb_suffixes: HashMap::new(),
            derivational_suffixes: HashMap::new(),
            phonological_rules: Vec::new(),
        };
        
        lexicon.initialize_finnish_examples();
        lexicon.initialize_english_examples();
        lexicon.initialize_phonological_rules();
        
        lexicon
    }
    
    /// Finnish morphology examples (classic Karttunen cases)
    fn initialize_finnish_examples(&mut self) {
        // Finnish noun stems (Karttunen's classic examples)
        self.noun_stems.insert("kala".to_string(), vec![MorphCategory::Noun]); // fish
        self.noun_stems.insert("talo".to_string(), vec![MorphCategory::Noun]); // house
        self.noun_stems.insert("lintu".to_string(), vec![MorphCategory::Noun]); // bird
        self.noun_stems.insert("katu".to_string(), vec![MorphCategory::Noun]); // street
        self.noun_stems.insert("kirja".to_string(), vec![MorphCategory::Noun]); // book
        self.noun_stems.insert("mies".to_string(), vec![MorphCategory::Noun]); // man
        self.noun_stems.insert("nainen".to_string(), vec![MorphCategory::Noun]); // woman
        
        // Finnish verb stems
        self.verb_stems.insert("luke".to_string(), vec![MorphCategory::Verb]); // read
        self.verb_stems.insert("kirjoita".to_string(), vec![MorphCategory::Verb]); // write  
        self.verb_stems.insert("juosta".to_string(), vec![MorphCategory::Verb]); // run
        self.verb_stems.insert("puhu".to_string(), vec![MorphCategory::Verb]); // speak
        self.verb_stems.insert("tule".to_string(), vec![MorphCategory::Verb]); // come
        
        // Finnish case suffixes (simplified)
        self.noun_suffixes.insert("".to_string(), vec![MorphCategory::Nominative, MorphCategory::Singular]);
        self.noun_suffixes.insert("n".to_string(), vec![MorphCategory::Genitive, MorphCategory::Singular]);
        self.noun_suffixes.insert("a".to_string(), vec![MorphCategory::Partitive, MorphCategory::Singular]);
        self.noun_suffixes.insert("ä".to_string(), vec![MorphCategory::Partitive, MorphCategory::Singular]);
        self.noun_suffixes.insert("ssa".to_string(), vec![MorphCategory::Inessive, MorphCategory::Singular]);
        self.noun_suffixes.insert("ssä".to_string(), vec![MorphCategory::Inessive, MorphCategory::Singular]);
        self.noun_suffixes.insert("sta".to_string(), vec![MorphCategory::Elative, MorphCategory::Singular]);
        self.noun_suffixes.insert("stä".to_string(), vec![MorphCategory::Elative, MorphCategory::Singular]);
        self.noun_suffixes.insert("an".to_string(), vec![MorphCategory::Illative, MorphCategory::Singular]);
        self.noun_suffixes.insert("än".to_string(), vec![MorphCategory::Illative, MorphCategory::Singular]);
        
        // Plural markers
        self.noun_suffixes.insert("t".to_string(), vec![MorphCategory::Nominative, MorphCategory::Plural]);
        self.noun_suffixes.insert("ien".to_string(), vec![MorphCategory::Genitive, MorphCategory::Plural]);
        self.noun_suffixes.insert("ia".to_string(), vec![MorphCategory::Partitive, MorphCategory::Plural]);
        self.noun_suffixes.insert("iä".to_string(), vec![MorphCategory::Partitive, MorphCategory::Plural]);
        
        // Finnish verb suffixes
        self.verb_suffixes.insert("n".to_string(), vec![MorphCategory::Present, MorphCategory::FirstPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("t".to_string(), vec![MorphCategory::Present, MorphCategory::SecondPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("".to_string(), vec![MorphCategory::Present, MorphCategory::ThirdPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("mme".to_string(), vec![MorphCategory::Present, MorphCategory::FirstPerson, MorphCategory::Plural]);
        self.verb_suffixes.insert("tte".to_string(), vec![MorphCategory::Present, MorphCategory::SecondPerson, MorphCategory::Plural]);
        self.verb_suffixes.insert("vat".to_string(), vec![MorphCategory::Present, MorphCategory::ThirdPerson, MorphCategory::Plural]);
        self.verb_suffixes.insert("vät".to_string(), vec![MorphCategory::Present, MorphCategory::ThirdPerson, MorphCategory::Plural]);
        
        // Past tense
        self.verb_suffixes.insert("in".to_string(), vec![MorphCategory::Past, MorphCategory::FirstPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("it".to_string(), vec![MorphCategory::Past, MorphCategory::SecondPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("i".to_string(), vec![MorphCategory::Past, MorphCategory::ThirdPerson, MorphCategory::Singular]);
    }
    
    /// English derivational morphology examples
    fn initialize_english_examples(&mut self) {
        // English stems
        self.noun_stems.insert("cat".to_string(), vec![MorphCategory::Noun]);
        self.noun_stems.insert("dog".to_string(), vec![MorphCategory::Noun]); 
        self.noun_stems.insert("book".to_string(), vec![MorphCategory::Noun]);
        self.noun_stems.insert("house".to_string(), vec![MorphCategory::Noun]);
        
        self.verb_stems.insert("walk".to_string(), vec![MorphCategory::Verb]);
        self.verb_stems.insert("work".to_string(), vec![MorphCategory::Verb]);
        self.verb_stems.insert("teach".to_string(), vec![MorphCategory::Verb]);
        self.verb_stems.insert("write".to_string(), vec![MorphCategory::Verb]);
        
        self.adjective_stems.insert("happy".to_string(), vec![MorphCategory::Adjective]);
        self.adjective_stems.insert("quick".to_string(), vec![MorphCategory::Adjective]);
        self.adjective_stems.insert("kind".to_string(), vec![MorphCategory::Adjective]);
        
        // English inflectional suffixes
        self.noun_suffixes.insert("s".to_string(), vec![MorphCategory::Plural]);
        self.verb_suffixes.insert("s".to_string(), vec![MorphCategory::Present, MorphCategory::ThirdPerson, MorphCategory::Singular]);
        self.verb_suffixes.insert("ed".to_string(), vec![MorphCategory::Past]);
        self.verb_suffixes.insert("ing".to_string(), vec![MorphCategory::Present]); // simplified
        
        // English derivational suffixes (Karttunen's framework)
        self.derivational_suffixes.insert("er".to_string(), vec![MorphCategory::Agent, MorphCategory::Noun]);
        self.derivational_suffixes.insert("ness".to_string(), vec![MorphCategory::Abstract, MorphCategory::Noun]);
        self.derivational_suffixes.insert("ly".to_string(), vec![MorphCategory::Adverb]);
        self.derivational_suffixes.insert("able".to_string(), vec![MorphCategory::Adjective]);
        self.derivational_suffixes.insert("tion".to_string(), vec![MorphCategory::Abstract, MorphCategory::Noun]);
        self.derivational_suffixes.insert("ment".to_string(), vec![MorphCategory::Abstract, MorphCategory::Noun]);
    }
    
    /// Morphophonological rules (Karttunen's two-level approach)
    fn initialize_phonological_rules(&mut self) {
        // Finnish vowel harmony rules
        self.phonological_rules.push(("back".to_string(), "ä".to_string(), "a".to_string()));
        self.phonological_rules.push(("back".to_string(), "ö".to_string(), "o".to_string()));
        
        // English morphophonological alternations
        self.phonological_rules.push(("_y".to_string(), "y".to_string(), "i".to_string())); // happy → happiness
        self.phonological_rules.push(("_e".to_string(), "e".to_string(), "".to_string())); // write → writer
        self.phonological_rules.push(("double".to_string(), "p".to_string(), "pp".to_string())); // stop → stopping
        
        // Finnish consonant gradation (simplified)
        self.phonological_rules.push(("weak".to_string(), "k".to_string(), "".to_string())); // katu → kadun
        self.phonological_rules.push(("weak".to_string(), "p".to_string(), "v".to_string())); // kapa → kavan
        self.phonological_rules.push(("weak".to_string(), "t".to_string(), "d".to_string())); // katu → kadun
    }
    
    /// Analyze a surface form following Karttunen's approach
    fn analyze(&self, surface_form: &str) -> Vec<MorphAnalysis> {
        let mut analyses = Vec::new();
        
        // Try noun analysis
        analyses.extend(self.analyze_as_noun(surface_form));
        
        // Try verb analysis  
        analyses.extend(self.analyze_as_verb(surface_form));
        
        // Try derivational analysis
        analyses.extend(self.analyze_derivational(surface_form));
        
        analyses
    }
    
    fn analyze_as_noun(&self, surface_form: &str) -> Vec<MorphAnalysis> {
        let mut analyses = Vec::new();
        
        for (stem, stem_cats) in &self.noun_stems {
            for (suffix, suffix_cats) in &self.noun_suffixes {
                let expected_form = format!("{}{}", stem, suffix);
                if expected_form == surface_form {
                    let mut categories = stem_cats.clone();
                    categories.extend(suffix_cats.clone());
                    
                    let morphemes = if suffix.is_empty() {
                        vec![stem.clone()]
                    } else {
                        vec![stem.clone(), suffix.clone()]
                    };
                    
                    analyses.push(MorphAnalysis {
                        _surface_form: surface_form.to_string(),
                        lexical_form: format!("{}+{}", stem, suffix),
                        morphemes,
                        categories,
                        gloss: self.generate_gloss(stem, suffix_cats),
                    });
                }
            }
        }
        
        analyses
    }
    
    fn analyze_as_verb(&self, surface_form: &str) -> Vec<MorphAnalysis> {
        let mut analyses = Vec::new();
        
        for (stem, stem_cats) in &self.verb_stems {
            for (suffix, suffix_cats) in &self.verb_suffixes {
                let expected_form = format!("{}{}", stem, suffix);
                if expected_form == surface_form || self.check_with_phonology(stem, suffix, surface_form) {
                    let mut categories = stem_cats.clone();
                    categories.extend(suffix_cats.clone());
                    
                    let morphemes = if suffix.is_empty() {
                        vec![stem.clone()]
                    } else {
                        vec![stem.clone(), suffix.clone()]
                    };
                    
                    analyses.push(MorphAnalysis {
                        _surface_form: surface_form.to_string(),
                        lexical_form: format!("{}+{}", stem, suffix),
                        morphemes,
                        categories,
                        gloss: self.generate_gloss(stem, suffix_cats),
                    });
                }
            }
        }
        
        analyses
    }
    
    fn analyze_derivational(&self, surface_form: &str) -> Vec<MorphAnalysis> {
        let mut analyses = Vec::new();
        
        // Check all stem types for derivational suffixes
        let all_stems: Vec<(&String, &Vec<MorphCategory>)> = self.noun_stems.iter()
            .chain(self.verb_stems.iter())
            .chain(self.adjective_stems.iter())
            .collect();
        
        for (stem, stem_cats) in all_stems {
            for (suffix, suffix_cats) in &self.derivational_suffixes {
                let expected_form = format!("{}{}", stem, suffix);
                if expected_form == surface_form || self.check_with_phonology(stem, suffix, surface_form) {
                    let mut categories = stem_cats.clone();
                    categories.extend(suffix_cats.clone());
                    
                    analyses.push(MorphAnalysis {
                        _surface_form: surface_form.to_string(),
                        lexical_form: format!("{}+{}", stem, suffix),
                        morphemes: vec![stem.clone(), suffix.clone()],
                        categories,
                        gloss: self.generate_gloss(stem, suffix_cats),
                    });
                }
            }
        }
        
        analyses
    }
    
    fn check_with_phonology(&self, stem: &str, suffix: &str, surface_form: &str) -> bool {
        // Simplified phonological checking
        // In practice, this would apply two-level rules
        
        // Check for y → i rule (happy + ness → happiness)
        if stem.ends_with('y') && !suffix.is_empty() {
            let modified_stem = format!("{}{}", &stem[..stem.len()-1], "i");
            let expected = format!("{}{}", modified_stem, suffix);
            if expected == surface_form {
                return true;
            }
        }
        
        // Check for e-deletion (write + er → writer)
        if stem.ends_with('e') && !suffix.is_empty() {
            let modified_stem = &stem[..stem.len()-1];
            let expected = format!("{}{}", modified_stem, suffix);
            if expected == surface_form {
                return true;
            }
        }
        
        false
    }
    
    fn generate_gloss(&self, stem: &str, suffix_cats: &[MorphCategory]) -> String {
        let mut gloss = stem.to_string();
        
        for cat in suffix_cats {
            match cat {
                MorphCategory::Plural => gloss.push_str(".PL"),
                MorphCategory::Past => gloss.push_str(".PAST"),
                MorphCategory::Present => gloss.push_str(".PRES"),
                MorphCategory::Genitive => gloss.push_str(".GEN"),
                MorphCategory::Partitive => gloss.push_str(".PART"),
                MorphCategory::Inessive => gloss.push_str(".INESS"),
                MorphCategory::Agent => gloss.push_str(".AGENT"),
                MorphCategory::Abstract => gloss.push_str(".ABSTR"),
                _ => {}
            }
        }
        
        gloss
    }
    
    /// Generate surface forms from lexical representation (following Karttunen)
    fn generate(&self, lexical_form: &str) -> Vec<String> {
        let mut results = Vec::new();
        
        if let Some(plus_pos) = lexical_form.find('+') {
            let stem = &lexical_form[..plus_pos];
            let suffix = &lexical_form[plus_pos + 1..];
            
            // Direct concatenation
            results.push(format!("{}{}", stem, suffix));
            
            // Apply phonological rules
            results.extend(self.apply_phonological_rules(stem, suffix));
        }
        
        results
    }
    
    fn apply_phonological_rules(&self, stem: &str, suffix: &str) -> Vec<String> {
        let mut results = Vec::new();
        
        // Apply y → i rule
        if stem.ends_with('y') && !suffix.is_empty() {
            let modified_stem = format!("{}{}", &stem[..stem.len()-1], "i");
            results.push(format!("{}{}", modified_stem, suffix));
        }
        
        // Apply e-deletion
        if stem.ends_with('e') && !suffix.is_empty() {
            let modified_stem = &stem[..stem.len()-1];
            results.push(format!("{}{}", modified_stem, suffix));
        }
        
        results
    }
}

/// Build a simple FST for morphotactics (following Karttunen's LEXC approach)
fn build_morphotactic_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let noun_stem = fst.add_state();
    let noun_inflected = fst.add_state();
    let final_state = fst.add_state();
    
    fst.set_start(start);
    fst.set_final(final_state, TropicalWeight::one());
    
    // Simplified morphotactic FST structure
    // Start → NounStem → NounInflected → Final
    
    // Add noun stem "cat"
    for ch in "cat".chars() {
        let next = fst.add_state();
        fst.add_arc(start, Arc::new(
            ch as u32,
            ch as u32,
            TropicalWeight::one(),
            next
        ));
    }
    
    // Add morpheme boundary
    fst.add_arc(noun_stem, Arc::new(
        '+' as u32,
        0, // epsilon output
        TropicalWeight::one(),
        noun_inflected
    ));
    
    // Add plural suffix "s"
    fst.add_arc(noun_inflected, Arc::new(
        's' as u32,
        's' as u32,
        TropicalWeight::one(),
        final_state
    ));
    
    fst
}

fn main() -> Result<()> {
    println!("Finite State Morphology");
    println!("======================");
    println!("Based on Karttunen's finite state morphology framework");
    println!("Following two-level morphology and LEXC principles\n");
    
    // Initialize the finite state lexicon
    let lexicon = FiniteStateLexicon::new();
    
    // Build demonstration FST
    let _morphotactic_fst = build_morphotactic_fst();
    
    println!("Lexicon initialized with:");
    println!("  {} noun stems", lexicon.noun_stems.len());
    println!("  {} verb stems", lexicon.verb_stems.len());
    println!("  {} adjective stems", lexicon.adjective_stems.len());
    println!("  {} noun suffixes", lexicon.noun_suffixes.len());
    println!("  {} verb suffixes", lexicon.verb_suffixes.len());
    println!("  {} derivational suffixes", lexicon.derivational_suffixes.len());
    println!("  {} phonological rules", lexicon.phonological_rules.len());
    
    // Example 1: Finnish morphological analysis (Karttunen's classic examples)
    println!("\n1. Finnish Morphological Analysis");
    println!("---------------------------------");
    println!("Classic examples from Karttunen's work on Finnish:");
    
    let finnish_examples = vec![
        "kala",     // fish.NOM.SG
        "kalan",    // fish.GEN.SG  
        "kalaa",    // fish.PART.SG
        "kalassa",  // fish.INESS.SG
        "kalasta",  // fish.ELAT.SG
        "talossa",  // house.INESS.SG
        "talosta",  // house.ELAT.SG
        "kirjassa", // book.INESS.SG
        "linnut",   // bird.NOM.PL
    ];
    
    for word in finnish_examples {
        println!("\nAnalyzing '{}':", word);
        let analyses = lexicon.analyze(word);
        if analyses.is_empty() {
            println!("  No analysis found (not in simplified lexicon)");
        } else {
            for (i, analysis) in analyses.iter().enumerate() {
                let tags: Vec<&str> = analysis.categories.iter()
                    .map(|c| c.to_tag())
                    .collect();
                println!("  Analysis {}: {} [{}]", 
                         i + 1, 
                         analysis.lexical_form,
                         tags.join("+"));
                println!("    Gloss: {}", analysis.gloss);
                println!("    Morphemes: {}", analysis.morphemes.join(" + "));
            }
        }
    }
    
    // Example 2: English derivational morphology
    println!("\n2. English Derivational Morphology");
    println!("----------------------------------");
    println!("Following Karttunen's approach to English derivation:");
    
    let english_examples = vec![
        "cats",      // cat+s
        "dogs",      // dog+s  
        "worked",    // work+ed
        "walking",   // walk+ing
        "worker",    // work+er
        "teacher",   // teach+er
        "writer",    // write+er (with e-deletion)
        "happiness", // happy+ness (with y→i)
        "quickly",   // quick+ly
        "kindness",  // kind+ness
    ];
    
    for word in english_examples {
        println!("\nAnalyzing '{}':", word);
        let analyses = lexicon.analyze(word);
        if analyses.is_empty() {
            println!("  No analysis found");
        } else {
            for (i, analysis) in analyses.iter().enumerate() {
                let tags: Vec<&str> = analysis.categories.iter()
                    .map(|c| c.to_tag())
                    .collect();
                println!("  Analysis {}: {} [{}]", 
                         i + 1, 
                         analysis.lexical_form,
                         tags.join("+"));
                println!("    Gloss: {}", analysis.gloss);
                println!("    Morphemes: {}", analysis.morphemes.join(" + "));
            }
        }
    }
    
    // Example 3: Generation (lexical → surface)
    println!("\n3. Morphological Generation");
    println!("---------------------------");
    println!("Generating surface forms from lexical representations:");
    
    let lexical_forms = vec![
        "cat+s",
        "work+er", 
        "happy+ness",
        "write+er",
        "walk+ed",
        "teach+er",
    ];
    
    for lexical_form in lexical_forms {
        println!("\nGenerating '{}':", lexical_form);
        let surface_forms = lexicon.generate(lexical_form);
        for (i, surface) in surface_forms.iter().enumerate() {
            println!("  Form {}: {}", i + 1, surface);
        }
    }
    
    // Example 4: Morphophonological alternations
    println!("\n4. Morphophonological Alternations");
    println!("----------------------------------");
    println!("Two-level rules in action (following Koskenniemi/Karttunen):");
    
    let alternation_examples = vec![
        ("happy + ness", "happiness", "y → i / _+C"),
        ("write + er", "writer", "e → ∅ / _+V"),
        ("stop + ing", "stopping", "p → pp / V_+V (simplified)"),
        ("city + s", "cities", "y → ie / _+s"),
    ];
    
    for (input, output, rule) in alternation_examples {
        println!("  {} → {} ({})", input, output, rule);
    }
    
    // Example 5: Theoretical framework
    println!("\n5. Theoretical Framework");
    println!("------------------------");
    println!("Karttunen's contributions to finite state morphology:");
    println!("  • LEXC: Lexicon compiler for morphotactics");
    println!("  • TWOLC: Two-level rule compiler");
    println!("  • Intersection of finite automata");
    println!("  • Composition of finite state transducers");
    println!("  • Constraint-based morphophonology");
    
    println!("\nKey principles:");
    println!("  • Lexicon = regular language over alphabet of morphemes");
    println!("  • Morphophonology = regular relation (FST)");
    println!("  • Surface forms = intersection of lexicon and phonology");
    println!("  • Bidirectional processing through FST inversion");
    println!("  • Compositional architecture: Lexicon ∘ Phonology");
    
    // Example 6: Applications
    println!("\n6. Applications in Computational Linguistics");
    println!("--------------------------------------------");
    println!("Karttunen's framework enables:");
    println!("  • Large-scale morphological analyzers");
    println!("  • Spell checkers with morphological awareness");
    println!("  • Machine translation for morphologically rich languages");
    println!("  • Information retrieval with morphological normalization");
    println!("  • Text generation with correct morphological forms");
    println!("  • Corpus linguistics and morphological annotation");
    
    println!("\nHistorical impact:");
    println!("  • Xerox finite state tools (1990s)");
    println!("  • HFST: Helsinki Finite State Technology");
    println!("  • Foma: open-source implementation");
    println!("  • Integration into major NLP pipelines");
    println!("  • Foundation for modern morphological processing");
    
    Ok(())
}