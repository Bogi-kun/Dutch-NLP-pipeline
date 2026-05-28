import torch
import numpy as np
from typing import List, Optional


class ChuLiuEdmonds:
    """
    Chu-Liu-Edmonds algorithm for finding maximum spanning arborescence.

    Used in dependency parsing to find the best non-projective parse tree.
    Given a weighted directed graph where nodes represent tokens and edge
    weights represent dependency scores, this finds the optimal head assignment
    for each token such that:
    - Every token (except ROOT) has exactly one incoming edge
    - The graph forms a tree rooted at ROOT
    - The total weight is maximized
    """

    def __init__(self, use_cuda: bool = False):
        """
        Args:
            use_cuda: Whether to use CUDA for computation
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Find optimal heads using Chu-Liu-Edmonds algorithm.

        Args:
            scores: [batch_size, seq_len, seq_len]
                   Dependency scores where scores[b, i, j] = score for token i having head j
                   Row index (i) = dependent, Column index (j) = head

        Returns:
            heads: [batch_size, seq_len]
                   Head index for each token. heads[b, 0] = 0 (ROOT points to itself)
        """
        batch_size, seq_len, _ = scores.shape
        heads_batch = []

        for b in range(batch_size):
            # Extract score matrix for this sentence
            score_matrix = scores[b]  # [seq_len, seq_len]
            heads = self._chu_liu_edmonds_single(score_matrix, seq_len)
            heads_batch.append(heads)

        return torch.stack(heads_batch, dim=0)

    def _chu_liu_edmonds_single(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Chu-Liu-Edmonds for a single sentence.

        Args:
            scores: [seq_len, seq_len] score matrix
            seq_len: sequence length

        Returns:
            heads: [seq_len] head indices, where heads[i] is the head of token i
        """
        device = scores.device

        # Create score matrix on CPU for processing (easier to handle indexing)
        scores_np = scores.detach().cpu().numpy().astype(np.float64)

        # Set ROOT (token 0) to always point to itself
        # This ensures a single root
        scores_np[0, :] = -np.inf
        scores_np[0, 0] = 0.0

        # Initialize: each token points to best scoring parent
        heads = np.zeros(seq_len, dtype=np.int32)
        heads[0] = 0  # ROOT points to itself

        # For each non-root token, select edge with max score
        for i in range(1, seq_len):
            best_head = np.argmax(scores_np[i, :])  # Best incoming edge for token i
            heads[i] = best_head

        # Check for cycles and resolve them
        heads = self._resolve_cycles(heads, scores_np, seq_len)

        # Ensure ROOT still points to itself after resolution
        heads[0] = 0

        return torch.from_numpy(heads).to(device).long()

    def _resolve_cycles(self, heads: np.ndarray, scores: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Detect and resolve cycles using contraction and recursion.

        Args:
            heads: Current head assignments
            scores: Score matrix
            seq_len: Sequence length

        Returns:
            heads: Valid head assignments (acyclic, tree structure)
        """
        # Ensure ROOT is always heads[0] = 0
        heads[0] = 0
        
        # Detect cycles
        cycle_nodes = self._find_cycle(heads)
        
        if cycle_nodes is None or len(cycle_nodes) == 0:
            # No cycles - ensure single tree connected to ROOT
            heads = self._ensure_single_tree(heads, scores, seq_len)
            return heads

        # Contract cycle into single super-node
        cycle_id = min(cycle_nodes)  # Representative of the cycle

        # Create mapping: old token index -> new index
        cycle_set = set(cycle_nodes)
        new_to_old = []
        old_to_new = {}

        for i in range(seq_len):
            if i not in cycle_set:
                old_to_new[i] = len(new_to_old)
                new_to_old.append(i)
            elif i == cycle_id:
                old_to_new[i] = len(new_to_old)
                new_to_old.append(i)

        new_seq_len = len(new_to_old)

        # Create new score matrix with contracted cycle
        new_scores = np.full((new_seq_len, new_seq_len), -np.inf, dtype=np.float32)

        for i in range(new_seq_len):
            for j in range(new_seq_len):
                old_i = new_to_old[i]
                old_j = new_to_old[j]

                if i == j:
                    new_scores[i, j] = -np.inf  # No self-loops
                    continue

                # If j is the cycle representative
                if old_j == cycle_id:
                    # Incoming edge to cycle: take minimum edge weight in cycle
                    # minus the edge being replaced
                    best_score = -np.inf

                    if old_i in cycle_set:
                        # Both in cycle: find best way to point into cycle
                        for cycle_node in cycle_nodes:
                            if cycle_node != old_i:
                                score = scores[old_i, cycle_node]
                                # Subtract edge that would be replaced in cycle
                                for cn in cycle_nodes:
                                    if heads[cn] == cycle_node:
                                        score -= scores[cn, heads[cn]]
                                        break
                                best_score = max(best_score, score)
                    else:
                        # External node to cycle
                        for cycle_node in cycle_nodes:
                            score = scores[old_i, cycle_node]
                            # Subtract edge being replaced in cycle
                            for cn in cycle_nodes:
                                if heads[cn] == cycle_node:
                                    score -= scores[cn, heads[cn]]
                                    break
                            best_score = max(best_score, score)

                    new_scores[i, j] = best_score if best_score > -np.inf else -np.inf

                elif old_i in cycle_set and old_j not in cycle_set:
                    # From cycle to external: use best internal edge
                    best_score = -np.inf
                    for cycle_node in cycle_nodes:
                        score = scores[cycle_node, old_j]
                        best_score = max(best_score, score)
                    new_scores[i, j] = best_score

                elif old_i not in cycle_set and old_j not in cycle_set:
                    # Both external: keep original score
                    new_scores[i, j] = scores[old_i, old_j]

                elif old_i not in cycle_set and old_j in cycle_set:
                    # External to cycle: take best incoming
                    best_score = -np.inf
                    for cycle_node in cycle_nodes:
                        score = scores[old_i, cycle_node]
                        best_score = max(best_score, score)
                    new_scores[i, j] = best_score

        # Recursively solve on contracted graph
        new_heads = np.arange(new_seq_len, dtype=np.int32)
        new_heads[0] = 0

        for i in range(1, new_seq_len):
            best_head = np.argmax(new_scores[i, :])
            if new_scores[i, best_head] == -np.inf:
                best_head = i  # Fallback
            new_heads[i] = best_head

        # Resolve cycles in contracted graph
        new_heads = self._resolve_cycles(new_heads, new_scores, new_seq_len)

        # Expand back to original graph
        heads = np.arange(seq_len, dtype=np.int32)
        heads[0] = 0

        for new_i, new_head_idx in enumerate(new_heads):
            old_i = new_to_old[new_i]
            old_head = new_to_old[new_head_idx]

            if old_i == cycle_id:
                # Cycle super-node: restore edges within cycle
                for cycle_node in cycle_nodes:
                    heads[cycle_node] = heads[cycle_node]  # Keep original
            else:
                heads[old_i] = old_head

        # Final validation: ensure all tokens connect to ROOT
        heads = self._ensure_single_tree(heads, scores, seq_len)

        return heads

    def _ensure_single_tree(self, heads: np.ndarray, scores: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Ensure all tokens eventually connect to ROOT (token 0).
        Fix any disconnected components by redirecting to ROOT.

        Args:
            heads: Current head assignments
            scores: Score matrix
            seq_len: Sequence length

        Returns:
            heads: Valid head assignments forming single tree rooted at ROOT
        """
        heads = heads.copy()
        heads[0] = 0  # ROOT always points to itself
        
        # For each token, verify it connects to ROOT
        for start_token in range(1, seq_len):
            visited = set()
            current = start_token
            steps = 0
            
            # Follow parent pointers
            while current != 0 and current not in visited:
                visited.add(current)
                current = heads[current]
                steps += 1
                
                if steps > seq_len:
                    # Infinite loop detected - disconnected from ROOT
                    # Redirect to ROOT
                    heads[start_token] = 0
                    break
            
            # If we didn't reach ROOT, it's disconnected
            if current != 0 and current in visited:
                # Cycle detected that doesn't include ROOT
                # Redirect to ROOT
                heads[start_token] = 0
        
        # Double-check: ensure no token (except ROOT) points to itself
        for i in range(1, seq_len):
            if heads[i] == i:
                heads[i] = 0  # Self-loop: redirect to ROOT
        
        return heads

    def _find_cycle(self, heads: np.ndarray) -> Optional[List[int]]:
        """
        Find a cycle in the head assignments.

        Args:
            heads: Head assignment array

        Returns:
            List of node indices in cycle, or None if no cycle
        """
        seq_len = len(heads)
        visited = [False] * seq_len
        rec_stack = [False] * seq_len

        def dfs(node, path):
            visited[node] = True
            rec_stack[node] = True
            path.append(node)

            head = heads[node]
            if head != 0:  # Not ROOT
                if not visited[head]:
                    result = dfs(head, path[:])
                    if result:
                        return result
                elif rec_stack[head]:
                    # Found cycle
                    cycle_start_idx = path.index(head)
                    return path[cycle_start_idx:] + [head]

            rec_stack[node] = False
            return None

        for i in range(1, seq_len):  # Skip ROOT
            if not visited[i]:
                path = []
                cycle = dfs(i, path)
                if cycle:
                    return cycle[:-1]  # Remove duplicate

        return None


# ===== TESTING =====
if __name__ == "__main__":
    # Simple test
    cle = ChuLiuEdmonds()

    # Create sample scores [batch=1, seq_len=4, seq_len=4]
    # Token 0 = ROOT, 1=noun, 2=verb, 3=punct
    scores = torch.tensor([
        [
            [-100., -100., -100., -100.],  # ROOT to self/others (shouldn't be selected)
            [5.0, -100., 10.0, 1.0],        # token 1 (noun) prefers head 2 (verb)
            [8.0, 2.0, -100., 1.0],         # token 2 (verb) prefers head 0 (ROOT)
            [6.0, 1.0, 1.0, -100.],         # token 3 (punct) prefers head 0 (ROOT)
        ]
    ], dtype=torch.float32)

    heads = cle(scores)
    print(f"Predicted heads: {heads}")
    print(f"Expected: token 0->0 (ROOT), token 1->2, token 2->0, token 3->0")

