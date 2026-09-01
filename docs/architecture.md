# CortexC — System Architecture

## Overview

CortexC is a lightweight, CPU-only machine-learning engine written in C17. It implements a single-layer perceptron trained by the classic perceptron learning rule, with min-max normalization, CSV dataset handling, model persistence, a CLI, and an optional GTK GUI.

## System Architecture

```
                     USER
                       |
             +---------+--------+
             |                  |
           CLI                GUI (GTK, optional)
             |                  |
             +---------+--------+
                       |
                 APPLICATION
                  LAYER
                       |
      +----------------+-----------------+
      |                |                 |
   Dataset        Preprocessing       Perceptron
      |                |                 |
      +--------+-------+-----------------+------+
                       |                        |
                    Training                 Evaluation
                       |                        |
                   Prediction              Model I/O
                       |                        |
                   Batch I/O                    |
             +---------+----------------+-------+-----+
                       |                          |
                    SYSTEM RAM                   SSD/HDD
```

## Module Architecture

| Module | Header | Responsibility |
|--------|--------|----------------|
| `config` | `config.h` | Global constants (`CORTEXC_VERSION`, defaults, limits). |
| `utils` | `utils.h` | Shared utilities (currently a placeholder translation unit). |
| `dataset` | `dataset.h` | CSV parsing, `Sample`/`Dataset` structures, train/test split, memory estimate, safe cleanup. |
| `preprocessing` | `preprocessing.h` | Min-max normalization (`Scaler`), train-only fit, constant-feature handling. |
| `perceptron` | `perceptron.h` | Single-layer perceptron: init, forward pass, training rule, early stopping. |
| `evaluation` | `evaluation.h` | Accuracy and confusion matrix (TP/TN/FP/FN). |
| `model_io` | `model_io.h` | Binary model persistence (save/load), magic+version validation, scaler round-trip. |
| `prediction` | `prediction.h` | Single, dataset and batch prediction; batch CSV parsing with optional output. |
| `main` | `src/main.c` | CLI dispatch (`train`, `predict`, `predict-batch`, `info`, `help`, `version`). |
| `main_gui` | `src/main_gui.c` | GTK3 GUI (dataset load, train, model load/save, predict). |

## Data Flow

```
1. User supplies a CSV (CLI: `train <csv>`; GUI: "Load Dataset")
        |
2. dataset_load_csv() parses the file into a Dataset (features + label)
        |
3. dataset_split() splits into train/test sets (shuffled, seeded)
        |
4. scaler_fit() learns min/max from the TRAINING set only (no leakage)
        |
5. scaler_transform() normalizes both train and test sets
        |
6. perceptron_train() runs the perceptron learning rule on train set
        |
7. evaluate() computes accuracy + confusion matrix on the test set
        |
8. model_save() writes model + scaler to a binary .bin file
```

## Training Flow

```
perceptron_init(features, lr, epochs)        -> zeroed weights/bias
perceptron_train(p, train_set)
    for each epoch:
        for each sample:
            prediction = step(bias + sum(weight[i]*feature[i]))
            error      = actual - prediction
            if error != 0:
                weight[i] += lr * error * feature[i]
                bias      += lr * error
        early-stop when 0 errors
    store p->epochs
```

## Prediction Flow

```
model_load(path)  -> Perceptron + Scaler
normalize(raw) = (raw - scaler.min) / (scaler.max - scaler.min)   # constant feature -> 0
prediction      = perceptron_forward(model, normalized)            # step(sum)
```

## Model Persistence Flow

```
model_save(path, model, scaler)
    header: magic="CORTEXC", version=1, feature_count, lr, epochs, bias
    write:  scaler.min[], scaler.max[], model.weights[]

model_load(path)
    read + validate header (magic, version, feature_count, finiteness)
    allocate perceptron + scaler
    read scaler.min[], scaler.max[], model.weights[]
    reject corrupted/truncated/version-mismatched files cleanly
```

## GUI Flow

1. Dataset tab: enter a CSV path, click Load → `dataset_load_csv()` → info label updated.
2. Training tab: click Train → split, fit scaler, normalize, train, evaluate; output printed to a text view; model stored in memory.
3. Model tab: Load/Save a model to/from a `.bin` file via `model_load()`/`model_save()`.
4. Predict tab: enter comma-separated features, click Predict → normalize + `perceptron_predict()` → "PASS"/"FAIL".

**Threading:** the GUI is single-threaded. Training blocks the UI (a known, documented limitation). No worker threads are used.

## CPU / RAM / Storage Architecture

- **CPU:** all matrix-free perceptron arithmetic uses the scalar CPU. No SIMD/GPU required.
- **RAM:** datasets and models are held entirely in system memory. RAM grows approximately with `samples × features × sizeof(float)` plus `sizeof(Sample)` per sample and allocation overhead (see `dataset_estimate_memory()`). Example measurements:

| Samples | Features | Estimated RAM |
|---------|----------|---------------|
| 1,000 | 3 | ≈ 32 KB |
| 10,000 | 3 | ≈ 0.40 MB |
| 100,000 | 3 | ≈ 3.5 MB |
| 1,000,000 | 3 | ≈ 31 MB |

- **Storage:** the executable, sample CSVs, and model files (`models/*.bin`) live on disk. Models are small (header + a few float arrays).

## CPU-Only Verification

The project intentionally contains **no** GPU-specific code or ML frameworks. There are no references to:

- CUDA, cuDNN, OpenCL, HIP, Vulkan compute
- TensorFlow, PyTorch, JAX, scikit-learn, ONNX
- Any third-party ML library

A search of the source tree for `cuda|opencl|tensorflow|pytorch|scikit|cublas|nvidia|gpu` yields no matches in the engine code. All computation is single-threaded C17 scalar math. The GUI is the only optional external dependency (GTK3), and only for display.

## Build System

```
Makefile
+-- make             -> optimized build (-O2)
+-- make debug       -> -g -O0 (+ sanitizers when the toolchain supports them)
+-- make release     -> -O2
+-- make test        -> build + run the 70-test suite
+-- make test-debug  -> run tests under AddressSanitizer + UBSan
+-- make bench       -> run the CPU benchmark (100k x 3)
+-- make gui         -> build the GTK GUI (requires GTK 3)
+-- make package     -> create a release layout in ./bin/
+-- make clean       -> remove artifacts

Compiler flags: -std=c17 -Wall -Wextra -Wpedantic -Wshadow -Wwrite-strings \
                -Wpointer-arith -Wcast-align -Wformat=2
```

### Sanitizers

AddressSanitizer + UndefinedBehaviorSanitizer are used automatically for `make debug`/`make test-debug` when the compiler supports them (`-fsanitize=address,undefined`). On toolchains without ASan libs the Makefile reports the limitation. Equivalent manual invocation:

```bash
# Devuan/Ubuntu (gcc)
make clean && CC=gcc make CFLAGS="-std=c17 -Wall -Wextra -Wpedantic -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined"
# Valgrind
make test
valgrind --leak-check=full ./build/test_dataset
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure ISO C17 | Educational, portable, no runtime dependencies. |
| CPU-only | Runs anywhere; no GPU/cloud/network. |
| Min-max normalization | Simple, explainable, fast. |
| Single-layer perceptron | Simplest learnable model; beginner-friendly. |
| Binary model format | Compact, fast save/load. |
| Optional GTK GUI | Desktop convenience without mandatory dependency. |
