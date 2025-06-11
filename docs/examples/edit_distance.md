# Edit Distance Example

This example demonstrates how to use Finite State Transducers (FSTs) to find words within a specified edit distance of a target word. It's particularly useful for spell checking and fuzzy string matching applications.

## Overview

The example implements a spell checker that can find corrections for misspelled words by:
1. Building an FST that accepts all strings within edit distance k of a target word
2. Creating a dictionary FST from a list of valid words
3. Composing these FSTs to find matching words
4. Finding the shortest paths to get the best matches

## Edit Operations

The edit distance FST supports three types of edits, each with a cost of 1:
- **Substitution**: Replace one character with another
- **Insertion**: Add a new character
- **Deletion**: Remove a character

## Implementation Details

### Edit Distance FST Construction

The `build_edit_distance_fst` function creates an FST that accepts all strings within edit distance k of the target word. The FST is constructed as a grid where:
- Each state represents a position in the target word and an edit distance
- Transitions represent possible edits (match, substitution, insertion, deletion)
- Final states are those that reach the end of the word within the edit distance limit

### Dictionary FST Construction

The `build_dictionary_fst` function creates a trie-based FST from a list of words where:
- Each state represents a prefix of one or more words
- Transitions represent characters in the words
- Final states mark the end of valid words

### Path Extraction

The `extract_paths` function uses depth-first search to find all paths from the start state to final states. Each path represents a valid word in the FST, and its weight represents the total cost (edit distance) of that word.

## Usage

```bash
cargo run --example edit_distance
```

The example will find corrections for the word "helo" within edit distance 2, using a small dictionary of words. The output will show the corrections along with their edit distances.

### Example Output

```
Finding corrections for 'helo':
  hello (edit distance: 1)
  help (edit distance: 1)
  held (edit distance: 1)
```

## Customization

You can modify the example to:
1. Use a different dictionary of words
2. Change the target word to check
3. Adjust the maximum edit distance
4. Modify the number of corrections to return

## Performance Considerations

- The edit distance FST grows quadratically with the length of the target word and the maximum edit distance
- The dictionary FST's size is linear in the total number of characters in all words
- Composition and shortest path operations are the most computationally intensive parts

## Related Examples

- [FST Composition](../fst_composition.md) - Shows how to compose FSTs for text transformation
- [Shortest Path](../shortest_path.md) - Demonstrates finding the best paths in an FST 