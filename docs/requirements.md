# C-AI — Project Requirements

## Project Overview

- **Project Name:** C-AI
- **Language:** C17
- **Type:** CPU-only machine learning application
- **Purpose:** Predict student PASS/FAIL status using a single-layer perceptron

## Problem Statement

Build a machine learning application that predicts whether a student will PASS or FAIL based on three input features. The application must be written entirely in C, run on CPU only, and use no external ML libraries.

## Input Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Study Hours | Double | Number of hours studied |
| 2 | Attendance | Double | Attendance percentage |
| 3 | Assignments Completed | Double | Number of assignments completed |

## Output

| Value | Meaning |
|-------|---------|
| 0 | FAIL |
| 1 | PASS |

## Functional Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| FR1 | Load datasets from CSV | Parse CSV files into in-memory dataset structures |
| FR2 | Validate datasets | Check for missing values, non-numeric data, invalid labels |
| FR3 | Store datasets in RAM | Hold loaded data in memory for computation |
| FR4 | Split training/testing data | Divide dataset into train and test sets |
| FR5 | Normalize numerical features | Apply min-max normalization to input features |
| FR6 | Train a single-layer perceptron | Implement perceptron learning rule from scratch |
| FR7 | Make predictions | Predict PASS/FAIL for new input data |
| FR8 | Calculate evaluation metrics | Compute accuracy, precision, recall, F1-score, confusion matrix |
| FR9 | Save trained models to SSD/HDD | Serialize model weights and parameters to binary files |
| FR10 | Load saved models | Deserialize model from binary files |
| FR11 | Batch prediction | Predict on multiple samples from a CSV file |
| FR12 | CLI interface | Command-line menu-driven interface |
| FR13 | GUI interface | Graphical user interface (planned for later phase) |

## Non-Functional Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR1 | C17 implementation | Code must compile with C17 standard |
| NFR2 | From-scratch ML | ML algorithm implemented without external ML libraries |
| NFR3 | No GPU acceleration | GPU must not be used for any computation |
| NFR4 | CPU-only computation | All ML calculations performed by CPU |
| NFR5 | RAM for computation | System memory used for all runtime data |
| NFR6 | SSD/HDD for storage | Datasets and model files stored on disk |
| NFR7 | Offline operation | Application works without internet connection |
| NFR8 | Safe memory management | Proper allocation/deallocation, no memory leaks |
| NFR9 | Input validation | All user and file inputs validated before use |
| NFR10 | Cross-platform design | Code designed to work on Windows, Linux, and macOS |

## Scope

### In Scope (Phase 1)
- Project structure and documentation
- Build system (Makefile)
- Git repository setup
- Initial C17 program skeleton

### In Scope (Future Phases)
- CSV parsing and dataset management
- Data preprocessing and normalization
- Single-layer perceptron implementation
- Training pipeline
- Evaluation metrics
- Model serialization
- Prediction system
- CLI interface
- GUI interface (GTK)

### Out of Scope
- GPU acceleration (CUDA, cuDNN, TensorRT)
- External ML libraries
- Multi-layer neural networks (Phase 1)
- Multi-class classification (Phase 1)
- Real-time or streaming data
- Web interface
- Cloud deployment

## Limitations

- Binary classification only (PASS/FAIL) in initial implementation
- Single-layer perceptron cannot learn non-linearly separable functions
- Limited to 3 input features initially
- No cross-validation or advanced training techniques in Phase 1
- CSV format only (no JSON, XML, or database input)
- Single-threaded execution

## Future Features (Phase 8+)

- Multi-algorithm support (adaboost, SVM, etc.)
- Multi-class classification
- Cross-validation
- Learning rate scheduling
- Advanced evaluation (ROC curves, AUC)
- GUI with GTK
- Batch processing mode
- Configuration file support
- Data visualization

## Success Criteria

1. Project compiles cleanly with `-Wall -Wextra -Wpedantic`
2. Initial program runs and displays expected output
3. No memory leaks (verified with AddressSanitizer/Valgrind)
4. Clean git history with meaningful commits
5. Complete documentation covering requirements, architecture, and system needs
6. All placeholder files in place for future phases
7. Build system works on target platforms
