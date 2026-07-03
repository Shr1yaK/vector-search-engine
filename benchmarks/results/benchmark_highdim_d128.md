| index | params | recall@k | mean ms | p95 ms | QPS | build s | mem MB |
|---|---|---|---|---|---|---|---|
| brute_force | — | 1.000 | 0.732 | 0.791 | 1366 | 0.00 | 10.24 |
| ivf | nlist=141, nprobe=1 | 0.882 | 0.029 | 0.036 | 35075 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=4 | 1.000 | 0.055 | 0.071 | 18191 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=8 | 1.000 | 0.100 | 0.135 | 9999 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=16 | 1.000 | 0.229 | 0.281 | 4358 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=32 | 1.000 | 0.444 | 0.517 | 2251 | 0.20 | 10.47 |
| hnsw | M=16, ef_construction=200, ef_search=10 | 0.986 | 0.241 | 0.480 | 4157 | 49.05 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=20 | 0.996 | 0.264 | 0.307 | 3793 | 49.05 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=50 | 1.000 | 0.346 | 0.418 | 2890 | 49.05 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=100 | 1.000 | 0.428 | 0.511 | 2336 | 49.05 | 15.61 |
