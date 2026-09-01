# Changelog

All notable changes to CortexC are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-01

Initial public release of CortexC, a CPU-only machine-learning engine in C17.

### Added

- C17 project foundation with a portable Makefile.
- CSV dataset loading and validation (malformed rows, invalid labels, non-numeric fields, finite-value checks).
- `Dataset`/`Sample` structures with dynamic growth, train/test splitting (seeded, reproducible), memory estimation, and safe cleanup.
- Min-max normalization `Scaler` (fit/transform) with constant-feature and train-only-fit safeguards.
- Single-layer perceptron: initialization, forward pass, perceptron learning rule, per-epoch error reporting, and early stopping.
- Accuracy and confusion-matrix evaluation (TP/TN/FP/FN) with input validation.
- Binary model persistence (magic identifier, format version, feature-count and finiteness validation) with clean rejection of corrupted/truncated/incompatible files.
- Single, dataset, and batch prediction.
- Batch prediction with optional CSV output (`feature...,actual,predicted`) and automatic header/label-column detection.
- CLI: `train`, `predict`, `predict-batch`, `info`, `help`, `version`, `--help`, `--version`.
- GTK3 GUI (dataset load, train, model load/save, predict) as an optional build target.
- Comprehensive test suite (dataset, preprocessing, perceptron, evaluation, model I/O, batch, integration).
- Synthetic dataset generator (`tools/generate_dataset.c`).
- CPU benchmark tool (`tools/benchmark.c`).
- Documentation: README, architecture, API reference, project audit.

### Changed

- Standardized diagnostic messages to `ERROR:` / `WARNING:` / `INFO:` prefixes.
- Consolidated batch CSV parsing into the prediction module.
- Replaced `strncpy` bounded copies with `snprintf` (avoids truncation warnings and non-termination bugs).
- Hardened training/loading against non-finite floating-point values.

### Fixed

- Batch prediction silently producing zero rows when input rows included a label column.
- Batch prediction lacking the `--output` option.
- `model_load()` rejecting legitimate models that reported 0 epochs.
- Compiler warnings exposed under strict warning flags.

### Security / Safety

- No GPU, CUDA, OpenCL, TensorFlow, PyTorch, or scikit-learn usage; CPU-only by design.
- Never trusts CSV files, model files, CLI arguments, or GUI input; validates all external inputs.

[0.1.0]: https://example.invalid/cortexc/compare/v0.0.0...v0.1.0
