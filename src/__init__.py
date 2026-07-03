"""vecsearch — a vector search engine built from scratch.

Modules:
    vectors         data loading, feature normalization, distance functions
    brute_force     exact k-NN (ground truth for every approximate index)
    kmeans          Lloyd's k-means, implemented from scratch (used by IVF)
    ivf_index       inverted-file index: cluster once, probe a few cells
    hnsw_index      hierarchical navigable small world graph index
    benchmark       recall@k / latency / build-time / memory comparisons
    mood_translator natural-language mood -> audio-feature query vector
"""

__version__ = "0.1.0"
