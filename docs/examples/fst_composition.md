# FST Composition Example

This example demonstrates how to compose two Finite State Transducers (FSTs) to create a more complex transducer. It's particularly useful for text transformation and normalization tasks.

## Overview

The example shows how to:
1. Build an FST from an input string
2. Create a rules FST that implements transformation rules
3. Compose these FSTs to apply the transformations
4. Find the shortest path in the composed FST
5. Extract and display the results

## Transformation Rules

The rules FST implements character-level transformations specified in the format:
```
input_char->output_char
```

For example, the rule `h->H` transforms lowercase 'h' to uppercase 'H'.

## Implementation Details

### Input FST Construction

The `build_input_fst` function creates an FST that accepts a single input string:
- Each state represents a position in the input string
- Transitions represent characters in the input
- The final state marks the end of the input string

### Rules FST Construction

The `build_rules_fst` function creates an FST that implements transformation rules:
- A single state with self-loops for each rule
- Each arc represents a character transformation
- The state is both initial and final to allow multiple transformations

### Output Extraction

The `extract_output` function follows the path in the FST to extract the transformed string:
- Starts from the initial state
- Follows arcs until reaching a final state
- Collects output labels to build the result string

## Usage

```bash
cargo run --example fst_composition -- "input_string" "transformation_rules"
```

### Example

```bash
cargo run --example fst_composition -- "hello" "h->H e->E l->L o->O"
```

This will transform "hello" to "HELLO" using the provided rules.

### Example Output

```
Input string: hello
Transformation rules: h->H e->E l->L o->O

Input FST:
Number of states: 6
Total number of arcs: 5
Start state: 0
Final states: [5]

Rules FST:
Number of states: 1
Total number of arcs: 4
Start state: 0
Final states: [0]

Composed FST:
Number of states: 6
Start state: Some(0)
Final states: [5]

Shortest Path FST:
Number of states: 6
Start state: Some(0)
Final states: [5]

Result: HELLO
```

## Customization

You can modify the example to:
1. Use different input strings
2. Define custom transformation rules
3. Add more complex transformation patterns
4. Modify the output format

## Performance Considerations

- The input FST's size is linear in the length of the input string
- The rules FST's size is linear in the number of rules
- Composition and shortest path operations are the most computationally intensive parts

## Related Examples

- [Edit Distance](../edit_distance.md) - Shows how to use FSTs for spell checking
- [Shortest Path](../shortest_path.md) - Demonstrates finding the best paths in an FST 