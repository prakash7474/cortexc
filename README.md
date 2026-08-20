# C-AI — CPU-Only Machine Learning in Pure C

## Project Purpose

C-AI is a beginner-friendly machine learning application written entirely in C. It predicts whether a student will PASS or FAIL based on study habits and performance metrics.

**This project does not use GPU acceleration.** All machine learning computation is performed on the CPU using system RAM.

## Problem Statement

Given a student's:
- **Study Hours** — number of hours studied
- **Attendance** — attendance percentage
- **Assignments Completed** — number of assignments finished

The application predicts:
- **0 = FAIL**
- **1 = PASS**

## Initial ML Algorithm

The initial machine learning algorithm is a **single-layer perceptron**, implemented from scratch without any external ML libraries.

## Requirements

### Hardware
- Modern x64 CPU
- 4 GB RAM minimum (8 GB recommended)
- 500 MB storage
- **No GPU required**

### Software
- C17 compiler (GCC ≥ 9 or Clang ≥ 10)
- GNU Make ≥ 4.0
- Git

## Project Structure

```
c-ai/
├── src/                    # Source files (.c)
│   ├── main.c              # Entry point
│   ├── dataset.c           # CSV parsing (Phase 2)
│   ├── preprocessing.c     # Normalization (Phase 3)
│   ├── perceptron.c        # ML algorithm (Phase 4)
│   ├── training.c          # Training pipeline (Phase 5)
│   ├── evaluation.c        # Metrics (Phase 6)
│   ├── model_io.c          # Save/load models (Phase 7)
│   ├── prediction.c        # Predictions (Phase 7)
│   ├── config.c            # Configuration (Phase 2)
│   └── utils.c             # Utilities (Phase 2)
│
├── include/                # Header files (.h)
│   ├── dataset.h
│   ├── preprocessing.h
│   ├── perceptron.h
│   ├── training.h
│   ├── evaluation.h
│   ├── model_io.h
│   ├── prediction.h
│   ├── config.h
│   └── utils.h
│
├── tests/                  # Test files
│   ├── test_dataset.c
│   ├── test_preprocessing.c
│   ├── test_perceptron.c
│   └── test_model.c
│
├── data/                   # Dataset files (CSV)
├── models/                 # Saved model files
├── gui/                    # GUI files (GTK, future)
├── docs/                   # Documentation
│   ├── requirements.md
│   ├── architecture.md
│   └── system-requirements.md
│
├── build/                  # Build output directory
├── Makefile                # Build system
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Build Instructions

```bash
# Build the project
make

# Build with debug symbols
make debug

# Build optimized
make release

# Clean build artifacts
make clean

# Build and run
make run
```

## Run Instructions

```bash
# After building
./c-ai

# Or use the make target
make run
```

## Current Development Phase

**Phase 1: Project Foundation** (Current)

- [x] Project requirements defined
- [x] System requirements defined
- [x] Architecture documented
- [x] Directory structure created
- [x] Initial C17 program created
- [x] Makefile created
- [x] Git repository initialized
- [x] .gitignore configured
- [x] README documentation

## Future Phases

| Phase | Description |
|-------|-------------|
| Phase 2 | CSV parsing, dataset management, configuration |
| Phase 3 | Data preprocessing, normalization |
| Phase 4 | Single-layer perceptron implementation |
| Phase 5 | Training pipeline |
| Phase 6 | Evaluation metrics |
| Phase 7 | Model serialization, prediction |
| Phase 8 | CLI interface, GUI with GTK |

## License

This project is for educational purposes.

## Author

C-AI Project — Learning C and Machine Learning
