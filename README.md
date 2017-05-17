# Blelloch Sum Scan

- Stable version
- Not limited by 2048 items (initial restriction due to max threads per block in current GPUs)
- Not limited by input sizes that are powers of 2 (initial restriction due to basic nature of algorithm)
- Currently can handle up to 67 108 864 (2^26) unsigned ints (problem why I cannot go above this is still unknown)
- Can still improve performance by avoiding bank conflicts as per this [paper](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf). 
