# Convenience targets. `make help` lists them.

.PHONY: help install data native test app bench bench-highdim bench-native clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	python -m pip install -r requirements.txt

data:  ## Download the Spotify dataset (~20 MB) into data/
	python scripts/fetch_data.py

native:  ## Compile the C++ hot-path extension in place
	python cpp/build.py

test:  ## Run the pytest suite
	pytest -q

app:  ## Launch the Streamlit app
	streamlit run src/app.py

bench:  ## Benchmark on the d=9 Spotify data
	python -m src.benchmark --n 20000 --queries 300 --tag n20k

bench-highdim:  ## Benchmark on synthetic d=128 embeddings
	python benchmarks/bench_highdim.py --n 20000 --d 128

bench-native:  ## A/B the C++ hot-path vs numpy
	python benchmarks/bench_native.py

clean:  ## Remove build artifacts and caches
	rm -rf **/__pycache__ .pytest_cache cpp/build
	rm -f vecsearch_native*.so
