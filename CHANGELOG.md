# Changelog

All notable changes to the ALSE project will be documented in this file.

## [3.4.0] - 2026-02-02

### Added
- ALSE V3.3 core architecture with VQ-VAE and soft segmentation
- Boundary predictor for learning segmentation points
- Adaptive soft segmentation with curriculum learning
- Vector quantizer with straight-through estimator
- Segment prior LM for perplexity evaluation
- Large-scale LM (50M params) for fair comparison with BPE
- Deterministic amortizer for production inference
- Comprehensive evaluation metrics including BPB
- BPE baseline tokenizer for comparison
- WikiText-2 and GLUE SST-2 data loaders
- Benchmarking framework for vocab sizes 128/512/2048
- Publication-quality figure generation

### Results
- 62% better BPB than BPE at vocab=128
- 24% better BPB than BPE at vocab=512
- 22% better BPB than BPE at vocab=2048
- 70% better BPB with 50M param LM (PATH C)
- 60% better BPB in distillation (PATH B)

### Documentation
- Comprehensive README with usage examples
- Contributing guidelines
- MIT License
- Basic usage examples
- Experiment configurations

## [3.3.0] - 2025-12-15

### Changed
- Improved boundary predictor architecture
- Enhanced curriculum learning schedule
- Optimized VQ loss weighting

## [3.2.0] - 2025-11-20

### Added
- Initial soft segmentation implementation
- Basic VQ-VAE architecture

## [3.0.0] - 2025-10-01

### Added
- Initial project structure
- Core model architecture sketch
