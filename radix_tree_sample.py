def longest_common_prefix(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return tuple(a[:i])


class RadixNode:
    def __init__(self):
        self.children = {}   # tuple(tokens) -> RadixNode
        self.is_end = False

        # Placeholder for KV cache (to be filled later)
        # Example: {"k": tensor, "v": tensor}
        self.kv_cache = None

class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, tokens):
        """
        tokens: list[int]
        """
        node = self.root

        while True:
            for edge, child in list(node.children.items()):
                prefix = longest_common_prefix(tokens, edge)

                if not prefix:
                    continue

                # Case 1: edge fully matches
                if prefix == edge:
                    tokens = tokens[len(edge):]
                    node = child
                    if not tokens:
                        node.is_end = True
                        return
                    break

                # Case 2: partial match → split edge
                remaining_edge = edge[len(prefix):]
                remaining_tokens = tokens[len(prefix):]

                mid = RadixNode()

                # Old child becomes child of mid
                mid.children[remaining_edge] = child

                # Replace old edge
                node.children.pop(edge)
                node.children[prefix] = mid

                if remaining_tokens:
                    new_child = RadixNode()
                    new_child.is_end = True
                    mid.children[tuple(remaining_tokens)] = new_child
                else:
                    mid.is_end = True

                return

            else:
                # No matching edge → create new edge
                new_child = RadixNode()
                new_child.is_end = True
                node.children[tuple(tokens)] = new_child
                return

import torch

def build_radix_tree_from_batch(token_batch):
    """
    token_batch: torch.LongTensor of shape (B, T)
    """
    tree = RadixTree()

    for i in range(token_batch.size(0)):
        tokens = token_batch[i].tolist()
        tree.insert(tokens)

    return tree

# Batch of tokenized prompts
tokens = torch.tensor([
    [1, 2, 3, 4],
    [1, 2, 3, 9],
    [1, 2, 8, 9],
    [5, 6, 7, 9]
])

tree = build_radix_tree_from_batch(tokens)

# print the tree
def print_tree(node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    print(prefix + connector + ("(end)" if node.is_end else ""))
    prefix += "    " if is_last else "│   "
    child_count = len(node.children)
    for i, (edge, child) in enumerate(node.children.items()):
        edge_str = ' '.join(map(str, edge))
        print(prefix + ("└── " if i == child_count - 1 else "├── ") + edge_str)
        print_tree(child, prefix + ("    " if i == child_count - 1 else "│   "), i == child_count - 1)

print_tree(tree.root)