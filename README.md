# GPU Prefix Sum

- Uses **Blelloch's Algorithm** (therefore, this is exclusive scan)
- Stable version
- Not limited by 2048 items (initial restriction due to max threads per block in current GPUs)
- Not limited by input sizes that are powers of 2 (initial restriction due to inherent binary tree-approach of algorithm)
- Can still improve performance by avoiding bank conflicts as per this [paper](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf). 
