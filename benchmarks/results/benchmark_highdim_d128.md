| index | params | recall@k | mean ms | p95 ms | QPS | build s | mem MB |
|---|---|---|---|---|---|---|---|
| brute_force | — | 1.000 | 0.737 | 0.808 | 1357 | 0.00 | 10.24 |
| ivf | nlist=141, nprobe=1 | 0.882 | 0.027 | 0.034 | 36837 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=4 | 1.000 | 0.055 | 0.071 | 18187 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=8 | 1.000 | 0.106 | 0.143 | 9476 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=16 | 1.000 | 0.233 | 0.294 | 4295 | 0.20 | 10.47 |
| ivf | nlist=141, nprobe=32 | 1.000 | 0.464 | 0.541 | 2157 | 0.20 | 10.47 |
| hnsw | M=16, ef_construction=200, ef_search=10 | 0.986 | 0.192 | 0.241 | 5201 | 48.40 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=20 | 0.996 | 0.234 | 0.272 | 4281 | 48.40 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=50 | 1.000 | 0.301 | 0.350 | 3321 | 48.40 | 15.61 |
| hnsw | M=16, ef_construction=200, ef_search=100 | 1.000 | 0.391 | 0.464 | 2555 | 48.40 | 15.61 |
| ivfpq | m=8, ksub=256, nprobe=16, rerank=0 | 0.363 | 0.642 | 0.684 | 1557 | 3.90 | 0.36 |
| ivfpq | m=8, ksub=256, nprobe=16, rerank=100 | 0.943 | 0.662 | 0.704 | 1510 | 3.90 | 0.36 |
