| index | params | recall@k | mean ms | p95 ms | QPS | build s | mem MB |
|---|---|---|---|---|---|---|---|
| brute_force | — | 1.000 | 0.271 | 0.326 | 3685 | 0.00 | 0.72 |
| ivf | nlist=141, nprobe=1 | 0.707 | 0.016 | 0.019 | 64172 | 0.82 | 0.89 |
| ivf | nlist=141, nprobe=4 | 0.960 | 0.030 | 0.040 | 33538 | 0.82 | 0.89 |
| ivf | nlist=141, nprobe=8 | 0.979 | 0.047 | 0.063 | 21173 | 0.82 | 0.89 |
| ivf | nlist=141, nprobe=16 | 0.982 | 0.081 | 0.107 | 12298 | 0.82 | 0.89 |
| ivf | nlist=141, nprobe=32 | 0.981 | 0.144 | 0.183 | 6927 | 0.82 | 0.89 |
| hnsw | M=16, ef_construction=200, ef_search=10 | 0.980 | 0.159 | 0.198 | 6293 | 35.64 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=20 | 0.983 | 0.218 | 0.271 | 4594 | 35.64 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=50 | 0.981 | 0.369 | 0.470 | 2709 | 35.64 | 6.20 |
| hnsw | M=16, ef_construction=200, ef_search=100 | 0.980 | 0.574 | 0.722 | 1741 | 35.64 | 6.20 |
| ivfpq | m=3, ksub=256, nprobe=16, rerank=0 | 0.789 | 0.293 | 0.318 | 3418 | 2.24 | 0.07 |
| ivfpq | m=3, ksub=256, nprobe=16, rerank=100 | 0.979 | 0.303 | 0.328 | 3298 | 2.24 | 0.07 |
