# Advanced Topics

**Master production-ready FST techniques**

*Performance • Scalability • Best practices*

This section covers advanced FST techniques for production systems, including performance optimization, memory management, and specialized operations.

## Closure Operations

Closure operations (Kleene star and plus) allow FSTs to accept repeated applications of patterns.

### Kleene Star (Zero or More)

```rust,ignore
use arcweight::prelude::*;

fn build_word_breaker() -> Result<VectorFst<TropicalWeight>> {
    // Match any sequence of letters
    let letter = build_character_class_fst(&['a'..='z'])?;
    
    // Zero or more letters
    let word = closure(&letter, ClosureType::Star)?;
    
    Ok(word)
}
```text

### Kleene Plus (One or More)

```rust,ignore
fn build_number_matcher() -> Result<VectorFst<TropicalWeight>> {
    // Match digits
    let digit = build_character_class_fst(&['0'..='9'])?;
    
    // One or more digits
    let number = closure(&digit, ClosureType::Plus)?;
    
    Ok(number)
}
```text

### Practical Example: Pattern Repetition

```rust,ignore
fn build_pattern_matcher() -> Result<VectorFst<TropicalWeight>> {
    // Components
    let word = build_word_pattern()?;
    let space = build_string_fst(" ")?;
    let punctuation = build_character_class_fst(&['.', '!', '?'])?;
    
    // Word + space (repeated)
    let word_space = concat(&word, &space)?;
    let words = closure(&word_space, ClosureType::Star)?;
    
    // Sentence = words + word + punctuation
    let sentence = concat(&words, &word)?;
    let sentence = concat(&sentence, &punctuation)?;
    
    Ok(sentence)
}
```text

## Performance Guidelines

### Operation Ordering

The order of operations significantly impacts performance:

```rust,ignore
fn optimal_pipeline(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // 1. Remove useless states first
    let connected = connect(fst)?;
    
    // 2. Simplify structure
    let no_eps = rm_epsilon(&connected)?;
    
    // 3. Determinize (can be expensive)
    let det = determinize(&no_eps)?;
    
    // 4. Minimize last
    let min = minimize(&det)?;
    
    Ok(min)
}
```text

### When to Optimize

```rust,ignore
struct OptimizationStrategy {
    size_threshold: usize,
    time_budget: Duration,
}

impl OptimizationStrategy {
    fn should_optimize(&self, fst: &VectorFst<TropicalWeight>) -> bool {
        // Consider FST size
        if fst.num_states() < self.size_threshold {
            return false;
        }
        
        // Consider properties
        let props = fst.properties();
        
        // Already optimal?
        if props.contains(FstProperties::DETERMINISTIC) && 
           props.contains(FstProperties::MINIMAL) {
            return false;
        }
        
        true
    }
    
    fn optimize_with_budget(
        &self, 
        fst: &VectorFst<TropicalWeight>
    ) -> Result<VectorFst<TropicalWeight>> {
        let start = Instant::now();
        let mut result = fst.clone();
        
        // Quick optimizations first
        if start.elapsed() < self.time_budget {
            result = connect(&result)?;
        }
        
        // More expensive operations if time allows
        if start.elapsed() < self.time_budget / 2 {
            result = rm_epsilon(&result)?;
        }
        
        if start.elapsed() < self.time_budget / 4 {
            if let Ok(det) = determinize_with_timeout(&result, self.time_budget - start.elapsed()) {
                result = det;
            }
        }
        
        Ok(result)
    }
}
```text

### Lazy Evaluation

For very large FSTs, use lazy evaluation:

```rust,ignore
use arcweight::fst::{ComposeFst, LazyFst};

fn lazy_pipeline(
    stages: Vec<Arc<dyn Fst<TropicalWeight>>>
) -> Arc<dyn Fst<TropicalWeight>> {
    let mut result = stages[0].clone();
    
    for stage in &stages[1..] {
        result = Arc::new(ComposeFst::new(result, stage.clone()));
    }
    
    result
}
```text

## Memory Management

### Large FST Handling

```rust,ignore
struct LargeFstHandler {
    memory_limit: usize,
    disk_cache_path: PathBuf,
}

impl LargeFstHandler {
    fn process_large_fst(
        &self,
        fst: &VectorFst<TropicalWeight>
    ) -> Result<()> {
        let size_estimate = estimate_memory_usage(fst);
        
        if size_estimate > self.memory_limit {
            // Use memory-mapped files
            self.process_with_mmap(fst)?;
        } else {
            // Process in memory
            self.process_in_memory(fst)?;
        }
        
        Ok(())
    }
    
    fn process_with_mmap(&self, fst: &VectorFst<TropicalWeight>) -> Result<()> {
        // Save to disk
        let temp_file = self.disk_cache_path.join("temp.fst");
        fst.write(&temp_file)?;
        
        // Memory-map the file
        let mmap_fst = MmapFst::read(&temp_file)?;
        
        // Process using memory-mapped FST
        self.process_chunks(&mmap_fst)?;
        
        Ok(())
    }
}
```text

### Streaming Operations

```rust,ignore
fn stream_compose<F>(
    fst1: &VectorFst<TropicalWeight>,
    fst2: &VectorFst<TropicalWeight>,
    mut process_fn: F
) -> Result<()> 
where
    F: FnMut(&Arc<TropicalWeight>) -> Result<()>
{
    // Create lazy composition
    let composed = ComposeFst::new(fst1, fst2);
    
    // Process states on-demand
    let mut queue = vec![composed.start().unwrap()];
    let mut visited = HashSet::new();
    
    while let Some(state) = queue.pop() {
        if !visited.insert(state) {
            continue;
        }
        
        // Process arcs from this state
        for arc in composed.arcs(state) {
            process_fn(&arc)?;
            queue.push(arc.nextstate());
        }
    }
    
    Ok(())
}
```text

### Memory Pooling

```rust,ignore
struct FstPool {
    pool: Vec<VectorFst<TropicalWeight>>,
    max_size: usize,
}

impl FstPool {
    fn acquire(&mut self) -> VectorFst<TropicalWeight> {
        self.pool.pop().unwrap_or_else(|| VectorFst::new())
    }
    
    fn release(&mut self, mut fst: VectorFst<TropicalWeight>) {
        if self.pool.len() < self.max_size {
            fst.clear();
            self.pool.push(fst);
        }
    }
}
```text

## Specialized Techniques

### Weight Pushing

Redistribute weights for better numerical stability:

```rust,ignore
fn optimize_weights(
    fst: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Push weights toward initial state
    let pushed = push(fst, PushType::Initial)?;
    
    // Helps with shortest path and prevents underflow
    Ok(pushed)
}
```text

### Epsilon Sequencing

Handle epsilon transitions efficiently:

```rust,ignore
fn epsilon_sequence_optimize(
    fst: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Sort states topologically
    let sorted = topsort(fst)?;
    
    // Process epsilon transitions in order
    let mut result = fst.clone();
    for state in sorted {
        collapse_epsilon_sequences(&mut result, state)?;
    }
    
    Ok(result)
}
```text

### Incremental Determinization

```rust,ignore
struct IncrementalDeterminizer {
    partial_fst: VectorFst<TropicalWeight>,
    determinized_states: HashSet<StateId>,
}

impl IncrementalDeterminizer {
    fn determinize_path(&mut self, path: &[Label]) -> Result<()> {
        let mut current_state = self.partial_fst.start().unwrap();
        
        for label in path {
            if !self.determinized_states.contains(&current_state) {
                self.determinize_state(current_state)?;
                self.determinized_states.insert(current_state);
            }
            
            // Follow determinized path
            current_state = self.follow_label(current_state, *label)?;
        }
        
        Ok(())
    }
}
```text

## Production Patterns

### Pattern: Multi-threaded Processing

```rust,ignore
use rayon::prelude::*;

fn parallel_fst_processing(
    inputs: Vec<String>,
    fst: Arc<VectorFst<TropicalWeight>>
) -> Vec<Result<String>> {
    inputs.par_iter()
        .map(|input| {
            let input_fst = build_string_fst(input)?;
            let result = compose(&input_fst, &fst)?;
            let shortest = shortest_path(&result)?;
            extract_output_string(&shortest)
        })
        .collect()
}
```text

### Pattern: Caching and Memoization

```rust,ignore
struct FstCache {
    cache: LruCache<String, Arc<VectorFst<TropicalWeight>>>,
    build_fn: Box<dyn Fn(&str) -> Result<VectorFst<TropicalWeight>>>,
}

impl FstCache {
    fn get_or_build(&mut self, key: &str) -> Result<Arc<VectorFst<TropicalWeight>>> {
        if let Some(fst) = self.cache.get(key) {
            return Ok(fst.clone());
        }
        
        let fst = Arc::new((self.build_fn)(key)?);
        self.cache.put(key.to_string(), fst.clone());
        
        Ok(fst)
    }
}
```text

### Pattern: Graceful Degradation

```rust,ignore
struct RobustProcessor {
    primary_fst: VectorFst<TropicalWeight>,
    fallback_fst: VectorFst<TropicalWeight>,
    timeout: Duration,
}

impl RobustProcessor {
    fn process(&self, input: &str) -> Result<String> {
        // Try primary FST with timeout
        let result = timeout(self.timeout, || {
            self.process_with_fst(input, &self.primary_fst)
        });
        
        match result {
            Ok(Ok(output)) => Ok(output),
            _ => {
                // Fall back to simpler FST
                warn!("Primary FST timed out, using fallback");
                self.process_with_fst(input, &self.fallback_fst)
            }
        }
    }
}
```text

## Debugging and Profiling

### FST Visualization

```rust,ignore
fn debug_fst(fst: &VectorFst<TropicalWeight>, name: &str) -> Result<()> {
    // Generate DOT format
    let dot = fst_to_dot(fst)?;
    
    // Write to file
    let path = format!("{}.dot", name);
    fs::write(&path, dot)?;
    
    // Convert to image if graphviz available
    if let Ok(_) = Command::new("dot")
        .args(&["-Tpng", &path, "-o", &format!("{}.png", name)])
        .output() 
    {
        println!("Visualization saved to {}.png", name);
    }
    
    Ok(())
}
```text

### Performance Profiling

```rust,ignore
struct FstProfiler {
    timings: HashMap<String, Duration>,
}

impl FstProfiler {
    fn profile<F, T>(&mut self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>
    {
        let start = Instant::now();
        let result = f()?;
        let duration = start.elapsed();
        
        self.timings.insert(name.to_string(), duration);
        
        Ok(result)
    }
    
    fn report(&self) {
        println!("FST Operation Timings:");
        for (name, duration) in &self.timings {
            println!("  {}: {:?}", name, duration);
        }
    }
}
```text

## Best Practices Summary

### 1. Design for Scale

```rust,ignore
// Good: Lazy evaluation for large FSTs
let large_composition = LazyComposeFst::new(fst1, fst2);

// Bad: Eager evaluation of huge FST
let huge_fst = compose(&million_state_fst1, &million_state_fst2)?;
```text

### 2. Handle Failures Gracefully

```rust,ignore
// Good: Timeout and fallback
match timeout(Duration::from_secs(5), || expensive_operation(fst)) {
    Ok(result) => result,
    Err(_) => simple_fallback(fst),
}

// Bad: Unbounded operation
let result = expensive_operation(fst)?; // Might hang
```text

### 3. Monitor Resource Usage

```rust,ignore
// Good: Check before operations
if estimate_composition_size(fst1, fst2) > MAX_FST_SIZE {
    return Err("Composition would be too large");
}

// Bad: Blind composition
let result = compose(fst1, fst2)?; // Might OOM
```text

## Next Steps

You've mastered advanced FST techniques! Here's where to go next:

1. **Apply these patterns** → [Examples Gallery](../examples/)
2. **Understand the theory** → [Core Concepts](../core-concepts/)
3. **Check the API** → [API Reference](../api-reference.md)

---

**Ready to build something amazing?** Explore the [Examples Gallery](../examples/) for complete applications using these techniques.