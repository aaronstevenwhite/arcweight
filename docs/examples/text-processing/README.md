# Text Processing

This section presents fundamental string algorithms implemented using finite state transducers, including string metrics, sequence alignment, and error correction.

## Examples

### [Edit Distance](edit_distance.md)
Implements the Levenshtein distance algorithm using weighted finite state transducers. The implementation supports configurable operation costs for insertion, deletion, and substitution.

```bash
cargo run --example edit_distance
```

### [String Alignment](string_alignment.md)  
Computes optimal alignments between strings using path extraction from weighted transducers. The system handles multiple equally optimal alignments.

```bash
cargo run --example string_alignment
```

### [Spell Checking](spell_checking.md)
Demonstrates spell checking through transducer composition, combining error models with lexicon constraints for candidate generation and ranking.

```bash
cargo run --example spell_checking
```

## Organization

The examples progress from basic string metrics through alignment algorithms to practical applications in spell checking. Each builds upon concepts from the previous examples.

---

[← Examples Overview](../README.md)