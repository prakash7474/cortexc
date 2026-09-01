# CortexC — CPU-Only Machine Learning Engine

**CortexC** is a lightweight, beginner-friendly machine-learning application written entirely in **C17**. It implements a single-layer perceptron from scratch (no external ML libraries) that predicts whether a student **PASS** (1) or **FAIL** (0) based on study habits and performance metrics.

> **This project is CPU-only.** All computation runs on the CPU using system RAM. No GPU, CUDA, OpenCL, TensorFlow, PyTorch, or scikit-learn.

## Features

- **CPU-only ML** — scalar C17 computation, single-threaded
- **Single-layer perceptron** — classic perceptron learning rule with early stopping
- **CSV datasets** — robust loading and validation (malformed rows, invalid labels, non-numeric fields, finite-value checks)
- **Preprocessing** — min-max normalization (`Scaler`) with train-only fit to prevent data leakage
- **Training** — seeded, reproducible train/test split; per-epoch error reporting
- **Prediction** — single, dataset, and batch prediction
- **Batch prediction** — CSV in → CSV out (`feature...,actual,predicted`)
- **Model persistence** — compact binary save/load with magic + version + feature validation
- **Evaluation** — accuracy and confusion matrix (TP/TN/FP/FN)
- **GUI** — optional GTK3 desktop interface
- **Testing** — 70-test suite covering every module + integration tests
- **Benchmarks** — CPU and RAM measurement tool

## Architecture

CortexC is layered: a CLI (or GUI) dispatches to application logic backed by small, focused modules.

```
+----------------+       +----------------+
|      CLI       |       |   GUI (GTK3)   |
+----------------+       +----------------+
        |                       |
        +-----------+-----------+
                    |
                 APPLICATION
        +-----------+-----------+---------+
        |           |           |         |
     Dataset   Preprocessing  Perceptron Evaluation
        |           |           |
        +-----------+-----------+------+----+
                                      | Model I/O
                                      | Prediction
```

| Module | Purpose |
|--------|---------|
| `dataset` | CSV parsing, `Sample`/`Dataset`, train/test split, memory estimation |
| `preprocessing` | Min-max normalization and constant-feature handling |
| `perceptron` | Forward pass and perceptron learning rule |
| `evaluation` | Accuracy and confusion matrix |
| `model_io` | Binary model save/load with validation |
| `prediction` | Single, dataset, and batch prediction |

See [`docs/architecture.md`](docs/architecture.md) for full diagrams and data flow, and [`docs/api.md`](docs/api.md) for the public function reference.

## Build

### Dependencies

The core engine requires only a **C17 compiler** and **GNU Make**.

| Platform | Compiler / Tooling | GUI (optional) |
|----------|--------------------|----------------|
| **Linux** | GCC ≥ 9 or Clang ≥ 10, GNU Make ≥ 4.0 | `sudo apt install libgtk-3-dev` (Debian/Ubuntu) |
| **macOS** | Xcode CLT (Clang), GNU Make | `brew install gtk+3` |
| **Windows** | GCC via MinGW/MSYS2, GNU Make | MSYS2: `pacman -S mingw-w64-x86_64-gtk3` |

No admin privileges are required as long as the toolchain is installed for your user.

### Build commands

```bash
make            # optimized build (build/cortexc)
make debug      # -g -O0, adds ASan/UBSan when the toolchain supports them
make release    # -O2
make test       # build and run the 70-test suite
make test-debug # run tests under AddressSanitizer + UBSan
make bench      # CPU benchmark (100k samples x 3 features)
make gui        # build the GTK GUI (requires GTK 3)
make clean      # remove artifacts
```

## CLI

```
cortexc <command> [options]

Commands:
  train <dataset.csv>                Train a model and save it
  predict <model.bin> <f1> ...       Predict using a saved model
  predict-batch <model.bin> <csv>    Batch predict from a CSV file
                                     [--output <results.csv>]
  info <model.bin>                   Display model information
  help                               Show this help
  version                            Show version information

Options:
  -h, --help     Show help
  -v, --version  Show version information
```

### Examples

```bash
# Train on the sample dataset (saves models/student_model.bin)
cortexc train data/students.csv

# Inspect a model
cortexc info models/student_model.bin

# Single prediction (features: study_hours, attendance, assignments_completed)
cortexc predict models/student_model.bin 7 85 8

# Batch prediction to stdout
cortexc predict-batch models/student_model.bin data/students.csv

# Batch prediction to a CSV file
cortexc predict-batch models/student_model.bin data/students.csv --output results.csv
```

The batch output CSV uses the input header feature names and a label column:

```
study_hours,attendance,assignments_completed,actual,predicted
1,50,2,0,0
7,85,8,1,1
```

`results.csv` is git-ignored (see `.gitignore`).

## GUI

To use the GUI, build it and run it:

```bash
make gui
./build/cortexc-gui
```

The GUI has four tabs:

1. **Dataset** — enter a CSV path and click **Load**; the file/sample/feature/label summary is shown.
2. **Train Model** — click **Train Perceptron** to split, normalize, train, and evaluate; results appear in the output view.
3. **Model** — **Load** or **Save** a `.bin` model file.
4. **Predict** — enter comma-separated features and click **Predict** to get PASS/FAIL.

**Limitation:** the GUI is single-threaded, so training blocks the interface until it completes. This is intentional for simplicity. No worker threads are used.

## Testing

```bash
make test
```

The suite covers:

- **Dataset:** valid/empty/malformed/header-only CSVs, missing files, non-numeric fields, incorrect column counts, invalid labels, cleanup/double-free/NULL safety, 80/20 split, split-preserves-data, reproducibility, memory estimation, large datasets (5k rows).
- **Preprocessing:** min-max bounds, constant features (zero denominator), scaler free/NULL, train-only fit (no leakage), feature-count mismatch.
- **Perceptron:** initialization, invalid params, weight/bias updates, forward pass, convergence, early stopping.
- **Evaluation:** accuracy, perfect/zero scores, TP/TN/FP/FN, invalid-label rejection.
- **Model I/O:** save, save/load round-trip equality, non-existent file, corrupted magic, zero features, unsupported version, truncated file, NULL safety.
- **Batch:** output generation, correct row counts, label/no-label inputs, empty/missing files, NULL safety.
- **Integration:** `train → save → exit → load → predict → compare` (identical predictions, exact weights/bias/scaler), version mismatch, corrupted magic, truncated file.

Run sanitizer-backed tests (if the compiler provides ASan/UBSan):

```bash
make test-debug
```

On a platform with Valgrind, instead:

```bash
make test
valgrind --leak-check=full --error-exitcode=1 ./build/test_dataset
```

## Performance

CortexC is CPU-only, single-threaded, and memory-light. This benchmark (GCC 16, x86-64, `make bench` default = 100k samples × 3 features) shows typical timings:

| Operation | Time |
|-----------|------|
| Dataset load (100k) | ≈ 56 ms |
| Preprocessing | ≈ 1 ms |
| Training (768 epochs) | ≈ 221 ms |
| Prediction (20k samples) | ≈ 0 ms |
| Model save | ≈ 1 ms |
| Model load | ≈ 0 ms |

> Exact numbers vary with CPU, compiler, and dataset size.

**RAM** grows approximately with `samples × features × sizeof(float)` plus metadata and allocation overhead:

| Samples | Features | Estimated RAM |
|---------|----------|---------------|
| 1,000 | 3 | ≈ 32 KB |
| 10,000 | 3 | ≈ 0.40 MB |
| 100,000 | 3 | ≈ 3.5 MB |
| 1,000,000 | 3 | ≈ 31 MB |

To generate large datasets for experiments:

```bash
make generate
./build/generate_dataset 100000 3 data/big.csv 42
```

The benchmark and generator are single-threaded and CPU-only.

## Limitations

- **Binary classifier only:** the perceptron predicts PASS/FAIL (0/1). Multi-class output is out of scope.
- **Linearly separable data only:** a single-layer perceptron cannot learn non-linearly separable functions (e.g. XOR).
- **No regularization, momentum, or learning-rate schedules.**
- **CSV input only** (no JSON/XML/database).
- **Single-threaded execution** (no concurrency).
- **Home-hosted, local application:** there is deliberately no cloud, Docker, Kubernetes, database, or web server — CortexC is a local desktop tool.
- **Model format** stores binary floats in host endianness; intended for same-platform use.

## License

**Not yet selected.** See [`LICENSE`](LICENSE).

## Version

`0.1.0` — see [`CHANGELOG.md`](CHANGELOG.md).
