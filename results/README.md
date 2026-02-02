# Results Directory

This directory stores experimental results from ALSE benchmarks.

## Structure

```
results/
├── benchmark_*.json    # Benchmark results
├── figures/           # Generated figures
└── logs/             # Training logs
```

## Running Experiments

To generate results:

```bash
python -m alse.evaluation.benchmark --vocab-sizes 128 512 2048
```

Results will be saved as JSON files with timestamps.
