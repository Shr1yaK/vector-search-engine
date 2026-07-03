| index | params | recall@k | mean ms | p95 ms | QPS | build s | mem MB |
|---|---|---|---|---|---|---|---|
| brute_force | — | 1.000 | 0.324 | 0.405 | 3084 | 0.00 | 0.72 |
| ivf | nlist=141, nprobe=1 | 0.707 | 0.015 | 0.019 | 65428 | 0.90 | 0.89 |
| ivf | nlist=141, nprobe=4 | 0.960 | 0.030 | 0.039 | 32802 | 0.90 | 0.89 |
| ivf | nlist=141, nprobe=8 | 0.979 | 0.081 | 0.199 | 12283 | 0.90 | 0.89 |
| ivf | nlist=141, nprobe=16 | 0.982 | 0.137 | 0.238 | 7282 | 0.90 | 0.89 |
| ivf | nlist=141, nprobe=32 | 0.981 | 0.262 | 0.349 | 3817 | 0.90 | 0.89 |
| hnsw | M=16, ef_construction=200, ef_search=10 | 0.980 | 0.414 | 0.519 | 2413 | 92.33 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=20 | 0.983 | 0.593 | 0.788 | 1687 | 92.33 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=50 | 0.981 | 0.944 | 1.230 | 1059 | 92.33 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=100 | 0.980 | 1.474 | 1.948 | 678 | 92.33 | 6.20 |
