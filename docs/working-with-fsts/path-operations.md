# Path Operations

**Find optimal solutions and rankings in your FSTs**

*Shortest path • N-best • Pruning*

Path operations help you extract meaningful results from FSTs. Whether you need the single best answer or multiple alternatives, these operations are essential for practical applications.

## Shortest Path

The shortest path operation finds the path through an FST with the minimum total weight. This is fundamental for applications like spell correction, speech recognition, and machine translation.

### When to Use Shortest Path

- Finding the best correction for a misspelled word
- Selecting the most likely translation
- Choosing optimal pronunciations
- Any "best answer" problem

### Basic Shortest Path

```rust,ignore
use arcweight::prelude::*;

fn find_best_path(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Find single shortest path
    let shortest = shortest_path(fst)?;
    
    // The result is a linear FST containing only the shortest path
    println!("Shortest path has {} states", shortest.num_states());
    
    Ok(shortest)
}
```text

### Extracting Path Information

```rust,ignore
fn extract_shortest_path(fst: &VectorFst<TropicalWeight>) -> Result<(Vec<Label>, Weight)> {
    let shortest = shortest_path(fst)?;
    
    let mut path = Vec::new();
    let mut total_weight = TropicalWeight::one();
    
    // Traverse the linear FST
    let mut state = shortest.start().unwrap();
    
    while !shortest.is_final(state) {
        // Get the single outgoing arc
        let arc = shortest.arcs(state).next().unwrap();
        
        path.push(arc.ilabel());
        total_weight = total_weight.times(&arc.weight());
        state = arc.nextstate();
    }
    
    // Add final weight
    if let Some(final_weight) = shortest.final_weight(state) {
        total_weight = total_weight.times(&final_weight);
    }
    
    Ok((path, total_weight))
}
```text

### Real-World Example: Spell Correction

```rust,ignore
fn spell_correct(
    word: &str,
    dictionary: &VectorFst<TropicalWeight>
) -> Result<String> {
    // Build error model (edit distance FST)
    let input_fst = build_string_fst(word)?;
    let error_model = build_edit_distance_fst(max_edits: 2)?;
    
    // Find words within edit distance
    let candidates = compose(&input_fst, &error_model)?;
    let valid_words = compose(&candidates, dictionary)?;
    
    // Get best correction
    let best = shortest_path(&valid_words)?;
    
    // Extract the corrected word
    extract_output_string(&best)
}
```text

## N-Best Paths

Sometimes you need multiple solutions, not just the best one. N-best paths gives you the top N paths ranked by weight.

### When to Use N-Best

- Providing multiple suggestions (autocomplete)
- Showing alternative translations
- Ranking search results
- Beam search in NLP

### Basic N-Best

```rust,ignore
fn find_top_paths(
    fst: &VectorFst<TropicalWeight>,
    n: usize
) -> Result<VectorFst<TropicalWeight>> {
    // Get top N paths
    let nbest = shortest_path_with_config(
        fst,
        ShortestPathConfig::new()
            .with_nshortest(n)
            .with_unique(true) // Remove duplicate strings
    )?;
    
    Ok(nbest)
}
```text

### Extracting Multiple Paths

```rust,ignore
fn extract_nbest_paths(
    fst: &VectorFst<TropicalWeight>,
    n: usize
) -> Result<Vec<(Vec<Label>, Weight)>> {
    let nbest_fst = shortest_path_with_config(
        fst,
        ShortestPathConfig::new().with_nshortest(n)
    )?;
    
    let mut paths = Vec::new();
    
    // The n-best FST has multiple paths from start to final states
    extract_all_paths(&nbest_fst, &mut paths)?;
    
    // Sort by weight (best first)
    paths.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    Ok(paths)
}
```text

### Real-World Example: Autocomplete

```rust,ignore
struct AutocompleteEngine {
    prefix_fst: VectorFst<TropicalWeight>,
    vocabulary: VectorFst<TropicalWeight>,
}

impl AutocompleteEngine {
    fn suggest(&self, prefix: &str, count: usize) -> Result<Vec<String>> {
        // Build FST for the prefix
        let prefix_fst = build_prefix_fst(prefix)?;
        
        // Find words starting with prefix
        let matches = compose(&prefix_fst, &self.vocabulary)?;
        
        // Get top suggestions by frequency
        let top_matches = shortest_path_with_config(
            &matches,
            ShortestPathConfig::new()
                .with_nshortest(count)
                .with_unique(true)
        )?;
        
        // Extract suggestions
        extract_suggestions(&top_matches)
    }
}
```text

### Unique vs Non-Unique Paths

```rust,ignore
fn demonstrate_unique_paths() -> Result<()> {
    let fst = /* FST with multiple paths producing same output */;
    
    // Without unique - may get duplicate strings
    let all_paths = shortest_path_with_config(
        &fst,
        ShortestPathConfig::new()
            .with_nshortest(10)
            .with_unique(false)
    )?;
    
    // With unique - each output string appears once
    let unique_paths = shortest_path_with_config(
        &fst,
        ShortestPathConfig::new()
            .with_nshortest(10)
            .with_unique(true)
    )?;
    
    Ok(())
}
```text

## Pruning

Pruning removes paths with weights above a threshold, keeping only the most likely paths. This is essential for managing FST size and focusing on relevant solutions.

### When to Use Pruning

- FST has too many unlikely paths
- Need to limit search space
- Implementing beam search
- Memory constraints

### Weight-Based Pruning

```rust,ignore
fn prune_by_weight(
    fst: &VectorFst<TropicalWeight>,
    threshold: f32
) -> Result<VectorFst<TropicalWeight>> {
    // Remove paths with weight > threshold
    let pruned = prune(fst, TropicalWeight::new(threshold))?;
    
    // Clean up disconnected states
    let connected = connect(&pruned)?;
    
    println!("Pruning: {} → {} states", 
        fst.num_states(), 
        connected.num_states());
    
    Ok(connected)
}
```text

### Relative Pruning

```rust,ignore
fn prune_relative(
    fst: &VectorFst<TropicalWeight>,
    delta: f32
) -> Result<VectorFst<TropicalWeight>> {
    // Find shortest path weight
    let shortest = shortest_path(fst)?;
    let best_weight = get_total_weight(&shortest)?;
    
    // Prune paths worse than best + delta
    let threshold = best_weight.value() + delta;
    prune(fst, TropicalWeight::new(threshold))
}
```text

### Pruning in Beam Search

```rust,ignore
struct BeamSearchDecoder {
    beam_width: f32,
}

impl BeamSearchDecoder {
    fn decode_step(
        &self,
        current: &VectorFst<TropicalWeight>,
        observation: &VectorFst<TropicalWeight>
    ) -> Result<VectorFst<TropicalWeight>> {
        // Expand paths with new observation
        let expanded = compose(current, observation)?;
        
        // Keep only paths within beam
        prune_relative(&expanded, self.beam_width)
    }
}
```text

## Advanced Path Operations

### Randomized Path Selection

```rust,ignore
use arcweight::algorithms::RandGen;

fn sample_random_path(
    fst: &VectorFst<TropicalWeight>,
    seed: u64
) -> Result<Vec<Label>> {
    let mut rng = RandGen::new(seed);
    
    // Sample path according to weights
    let path = rng.generate(fst)?;
    
    Ok(path)
}
```text

### Path Enumeration

```rust,ignore
fn enumerate_all_paths(
    fst: &VectorFst<TropicalWeight>,
    max_length: usize
) -> Result<Vec<Vec<Label>>> {
    let mut paths = Vec::new();
    let mut stack = vec![(fst.start().unwrap(), Vec::new())];
    
    while let Some((state, path)) = stack.pop() {
        if path.len() > max_length {
            continue;
        }
        
        if fst.is_final(state) {
            paths.push(path.clone());
        }
        
        for arc in fst.arcs(state) {
            let mut new_path = path.clone();
            new_path.push(arc.ilabel());
            stack.push((arc.nextstate(), new_path));
        }
    }
    
    Ok(paths)
}
```text

### Distance-Based Operations

```rust,ignore
fn find_paths_within_distance(
    fst: &VectorFst<TropicalWeight>,
    target: &str,
    max_distance: f32
) -> Result<VectorFst<TropicalWeight>> {
    // Build target FST
    let target_fst = build_string_fst(target)?;
    
    // Compute distances
    let distances = compute_distances(fst, &target_fst)?;
    
    // Keep only close paths
    prune(&distances, TropicalWeight::new(max_distance))
}
```text

## Practical Applications

### Application: Multi-Stage Search

```rust,ignore
fn multi_stage_search(
    query: &str,
    stages: Vec<VectorFst<TropicalWeight>>
) -> Result<Vec<String>> {
    let mut current = build_string_fst(query)?;
    
    // Apply each stage and keep top candidates
    for (i, stage) in stages.iter().enumerate() {
        current = compose(&current, stage)?;
        
        // Prune after each stage to control size
        if i < stages.len() - 1 {
            current = prune_relative(&current, 5.0)?;
        }
    }
    
    // Get final results
    let results = shortest_path_with_config(
        &current,
        ShortestPathConfig::new().with_nshortest(10)
    )?;
    
    extract_strings(&results)
}
```text

### Application: Confidence Scoring

```rust,ignore
fn get_confidence_scores(
    fst: &VectorFst<TropicalWeight>,
    n: usize
) -> Result<Vec<(String, f32)>> {
    // Get n-best paths
    let nbest = shortest_path_with_config(
        fst,
        ShortestPathConfig::new().with_nshortest(n)
    )?;
    
    // Extract paths with weights
    let paths = extract_weighted_paths(&nbest)?;
    
    // Convert weights to probabilities
    let total = paths.iter()
        .map(|(_, w)| (-w.value()).exp())
        .sum::<f32>();
    
    let scores = paths.into_iter()
        .map(|(s, w)| (s, (-w.value()).exp() / total))
        .collect();
    
    Ok(scores)
}
```text

## Performance Considerations

### Shortest Path Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Dijkstra | O(E + V log V) | O(V) |
| Bellman-Ford | O(VE) | O(V) |
| N-shortest | O(N(E + V log V)) | O(NV) |

### Optimization Tips

1. **Pre-process FSTs**: Determinize and minimize before path operations
2. **Use pruning early**: Don't wait until FST is huge
3. **Consider approximations**: Sometimes top-10 is as good as top-100
4. **Cache results**: Path operations can be expensive

## Common Patterns

### Pattern: Fallback Search

```rust,ignore
fn search_with_fallback(
    query: &str,
    strict_matcher: &VectorFst<TropicalWeight>,
    fuzzy_matcher: &VectorFst<TropicalWeight>
) -> Result<String> {
    // Try exact match first
    let exact = compose(&build_string_fst(query)?, strict_matcher)?;
    if let Ok(result) = shortest_path(&exact) {
        if result.num_states() > 1 {
            return extract_string(&result);
        }
    }
    
    // Fall back to fuzzy match
    let fuzzy = compose(&build_string_fst(query)?, fuzzy_matcher)?;
    let best = shortest_path(&fuzzy)?;
    extract_string(&best)
}
```text

### Pattern: Incremental Results

```rust,ignore
fn incremental_search(
    prefix: &str,
    fst: &VectorFst<TropicalWeight>
) -> impl Iterator<Item = String> {
    let results = Arc::new(Mutex::new(Vec::new()));
    let results_clone = results.clone();
    
    // Start search in background
    thread::spawn(move || {
        let matches = compose(&build_prefix_fst(prefix).unwrap(), fst).unwrap();
        let paths = enumerate_paths_lazy(&matches);
        
        for path in paths {
            results_clone.lock().unwrap().push(path);
        }
    });
    
    // Return iterator that yields results as they're found
    IncrementalIterator::new(results)
}
```text

## Next Steps

Now that you can find solutions in FSTs:

1. **Analyze FST structure** → [Structural Operations](structural-operations.md)
2. **Learn advanced techniques** → [Advanced Topics](advanced-topics.md)
3. **See complete examples** → [Examples Gallery](../examples/)

---

**Ready to analyze FST structure?** Continue to [Structural Operations](structural-operations.md)