# GPU Prefix Sum

- Uses **Blelloch's Algorithm** (exclusive scan)
- Not limited by 2048 items (a former restriction on the initial implementation of the algorithm due to the maximum threads that can run in a thread block on current GPUs)
- Not limited by input sizes that are powers of 2 (a former restriction due to inherent binary tree-approach of the algorithm)
- Free of shared memory bank conflicts using the index padding method in this [paper](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf). 
