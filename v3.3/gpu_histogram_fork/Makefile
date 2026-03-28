# gpu_histogram_fork — Build & Test Makefile
# ============================================================
# Usage:  make build        — compile CUDA shared library
#         make test         — run full test suite
#         make bench        — quick 4h benchmark
#         make gpu-info     — verify GPU is visible
# ============================================================

# Build the CUDA shared library
build:
	mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON && make -j$$(nproc)

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run only GPU tests
test-gpu:
	pytest tests/ -v --tb=short -m gpu

# Run only CPU tests (no GPU needed)
test-cpu:
	pytest tests/ -v --tb=short -m "not gpu"

# Run benchmarks (single timeframe profile)
bench:
	python benchmark/bench_kernel_speed.py --profile 4h

# Run benchmarks across all major timeframes
bench-all:
	python benchmark/bench_kernel_speed.py --profile 1w
	python benchmark/bench_kernel_speed.py --profile 1d
	python benchmark/bench_kernel_speed.py --profile 4h

# End-to-end training benchmark
bench-e2e:
	python benchmark/bench_end_to_end.py --rows 17520 --features 100000

# Generate synthetic test data
generate-data:
	python benchmark/generate_test_data.py --profile 4h --output data/

# Remove all build artifacts
clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache data/

# Install Python package in editable/dev mode
install-dev:
	pip install -e .

# Build pip wheel for distribution
wheel:
	python -m build --wheel

# Print GPU name, VRAM, and compute capability
gpu-info:
	python -c "import cupy; d=cupy.cuda.Device(); print(f'GPU: {d.attributes[\"name\"]}, VRAM: {d.mem_info[1]/1e9:.1f}GB, CC: {d.compute_capability}')"

.PHONY: build test test-gpu test-cpu bench bench-all bench-e2e generate-data clean install-dev wheel gpu-info
