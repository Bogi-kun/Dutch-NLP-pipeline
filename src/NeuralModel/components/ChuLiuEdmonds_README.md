# Chu-Liu-Edmonds Algorithm Implementation

## Overview

The Chu-Liu-Edmonds algorithm finds the **maximum spanning arborescence** (directed tree) in a weighted directed graph. In dependency parsing, it solves the problem:

**Given:**
- A sequence of tokens (words)
- A ROOT token at position 0
- Edge weights representing dependency scores (score of token i having token j as parent)

**Find:**
- The optimal head (parent) for each token such that:
  1. Every token (except ROOT) has exactly one incoming edge
  2. The graph forms a tree rooted at ROOT
  3. The total weight is maximized
  4. No cycles exist

## Why It's Important

Standard greedy approaches (each token picks the best-scoring parent independently) often create cycles or violate tree constraints. The Chu-Liu-Edmonds algorithm guarantees a valid tree while maximizing the total score.

### Example: Why Greedy Fails

```
Tokens: ROOT, "dog", "barks", "loudly"

Greedy per-token:
- Token 1 (dog) → picks parent 2 (barks) with score 8.0
- Token 2 (barks) → picks parent 1 (dog) with score 7.0  ← CYCLE! 
- Token 3 (loudly) → picks parent 2 (barks) with score 6.0

Chu-Liu-Edmonds:
- Detects cycle: 1→2→1
- Contracts it
- Resolves optimally: ROOT→barks→dog, ROOT→loudly
```

## Algorithm Steps

### Step 1: Initialize
For each non-root token, select the incoming edge with the highest weight.

```python
for i in range(1, seq_len):
    heads[i] = argmax(scores[i, :])  # Best parent for token i
```

### Step 2: Cycle Detection
Check if the current assignment forms a valid tree (no cycles).

```python
cycle_nodes = find_cycle(heads)  # DFS to detect cycles
if no cycles:
    return heads  # Valid solution found
```

### Step 3: Cycle Contraction
Contract any cycles into a single super-node and recursively solve on the smaller graph.

```python
# Create mapping: which nodes form the cycle
cycle_set = {nodes in the cycle}

# Create new score matrix
new_scores[i, j] = modified score for new graph
new_seq_len = seq_len - (len(cycle) - 1)

# Recursively solve
new_heads = resolve_cycles(new_heads, new_scores, new_seq_len)
```

### Step 4: Cycle Expansion
Expand contracted cycles back into the original graph.

```python
# Restore edges within the cycle
# Map new head assignments back to original indices
```

## Score Matrix Format

```
Shape: [batch_size, seq_len, seq_len]

scores[b, i, j] = score of token i having token j as parent

Example for "The cat sat":
          ROOT  The   cat   sat
ROOT  [  -∞    -∞    -∞    -∞  ]
The   [ 5.2   -∞    2.1   0.8  ]  ← Token 1 (The) prefers ROOT (score 5.2)
cat   [ 3.8   1.5   -∞    8.2  ]  ← Token 2 (cat) prefers sat (score 8.2)
sat   [ 9.1   0.3   2.5   -∞   ]  ← Token 3 (sat) prefers ROOT (score 9.1)
```

**Note:** ROOT token always has -∞ scores (it's the root, doesn't take parents)

## Python Implementation

### Basic Usage

```python
from src.NeuralModel import ChuLiuEdmonds

# Initialize
cle = ChuLiuEdmonds(use_cuda=True)

# Run on batch of scores
scores = torch.tensor([...])  # [batch, seq_len, seq_len]
heads = cle(scores)  # [batch, seq_len]

# heads[b, i] = index of parent token for token i in sentence b
# heads[b, 0] = 0 (ROOT points to itself)
```

### Integration with NLP Pipeline

```python
# In NLP_pipeline.py:
self.cle_decoder = ChuLiuEdmonds(use_cuda=torch.cuda.is_available())

# During decoding:
heads_optimal = self.cle_decoder(arc_scores)  # arc_scores: [1, seq_len, seq_len]

# Get relation labels using optimal heads
for i, head_idx in enumerate(heads_optimal[0]):
    rel_logits = rel_scores[0, i, head_idx, :]
    rel_id = argmax(rel_logits)
    predicted_rel = reverse_dep[rel_id]
```

## Time Complexity

- **Average case:** O(seq_len²)
- **Worst case:** O(seq_len³) when many cycles need contraction
- **In practice:** Usually O(seq_len²) as cycles are rare

## Key Properties

✅ **Guarantees valid tree structure**
✅ **Maximizes total edge weight**
✅ **Handles non-projective parses**
✅ **Single root (no forests)**
✅ **Every token (except root) has exactly one parent**

## Related Algorithms

- **Edmonds' Algorithm** (1967): Original directed version
- **Chu-Liu-Edmonds**: Named after both authors
- **Tarjan's Algorithm**: Alternative implementation with better constants
- **Greedy** (baseline): Fast but creates cycles/invalid trees

## References

- Chu, Y. J., & Liu, T. H. (1965). "On the Shortest Arborescence of a Directed Graph"
- Edmonds, J. (1967). "Optimum Branchings"
- McDonald et al. (2005). "Non-projective dependency parsing using spanning tree algorithms"

