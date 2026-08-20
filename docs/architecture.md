# C-AI — System Architecture

## High-Level Architecture

```
                    USER
                      |
              +-------+-------+
              |               |
             CLI             GUI
              |               |
              +-------+-------+
                      |
               APPLICATION
                    LAYER
                      |
       +--------------+--------------+
       |              |              |
    Dataset      Preprocessing      Model
       |              |              |
       +--------------+--------------+
                      |
                   Training
                      |
                 Perceptron
                      |
                 Evaluation
                      |
                 Prediction
                      |
                 Model I/O
                      |
                    SSD/HDD
```

## Module Responsibilities

### config
- **Purpose:** Global configuration and constants
- **Responsibilities:**
  - Define project version
  - Set default training parameters (learning rate, epochs, etc.)
  - Define size limits (max features, max line length)
  - Store training configuration structures

### utils
- **Purpose:** Shared utility functions
- **Responsibilities:**
  - Safe memory allocation (malloc wrappers with error handling)
  - Error/warning/info logging with file:line info
  - String manipulation (trim, validation, duplication)
  - Common helper functions used across modules

### dataset
- **Purpose:** Data loading and management
- **Responsibilities:**
  - Parse CSV files into in-memory structures
  - Validate data (check for missing values, correct types)
  - Store datasets in RAM for computation
  - Provide dataset summary and memory usage info
  - Free dataset memory safely

### preprocessing
- **Purpose:** Data normalization
- **Responsibilities:**
  - Implement min-max normalization
  - Fit scaler to training data only (prevent data leakage)
  - Transform datasets using fitted scaler
  - Inverse transform for display purposes

### perceptron
- **Purpose:** Single-layer perceptron algorithm
- **Responsibilities:**
  - Initialize weights and bias
  - Implement forward pass (weighted sum + step activation)
  - Implement perceptron learning rule
  - Track training history per epoch
  - Predict single and batch predictions

### training
- **Purpose:** Training pipeline orchestration
- **Responsibilities:**
  - Load dataset and split into train/test
  - Coordinate normalization and training
  - Manage training context and state
  - Provide high-level training API

### evaluation
- **Purpose:** Model evaluation metrics
- **Responsibilities:**
  - Compute accuracy, precision, recall, F1-score
  - Build confusion matrix
  - Print formatted evaluation results

### model_io
- **Purpose:** Model persistence
- **Responsibilities:**
  - Serialize model to binary format
  - Deserialize model from binary files
  - Validate model files (magic number, version, integrity)
  - Store scaler parameters alongside model weights

### prediction
- **Purpose:** Prediction orchestration
- **Responsibilities:**
  - Single prediction with normalization
  - Batch prediction from CSV files
  - Write prediction results to CSV

### gui (Future)
- **Purpose:** Graphical user interface
- **Responsibilities:**
  - Provide visual interface using GTK
  - Display training progress and results
  - Allow interactive prediction
  - Visualize data and model performance

## Data Flow

```
1. User loads CSV file
         |
2. Dataset module parses CSV into Dataset struct
         |
3. Training module splits into train/test sets
         |
4. Preprocessing module fits scaler on training data
         |
5. Preprocessing module transforms both train and test data
         |
6. Perceptron module trains on normalized training data
         |
7. Evaluation module computes metrics on test data
         |
8. Model I/O module saves trained model to disk
         |
9. Prediction module can load model and predict on new data
```

## Memory Management

### Ownership Model
- Each module is responsible for freeing its own allocated memory
- `Dataset` is allocated by `dataset_load()`, freed by `dataset_free()`
- `Scaler` is allocated by `scaler_create()`, freed by `scaler_free()`
- `Perceptron` is allocated by `perceptron_create()`, freed by `perceptron_free()`
- `TrainContext` owns and frees all intermediate data

### Safety Rules
- All `malloc` calls go through `xmalloc()` which aborts on failure
- Error paths free all allocated memory before returning
- No memory leaks (verified with AddressSanitizer/Valgrind)

## Build System

```
Makefile
+-- make          -> Build c-ai.exe (only main.c in Phase 1)
+-- make debug    -> Build with -g -O0 (debug symbols)
+-- make release  -> Build with -O2 (optimized)
+-- make clean    -> Remove build artifacts
```

### Compiler Flags
- **Standard:** `-std=c17`
- **Warnings:** `-Wall -Wextra -Wpedantic`
- **Debug:** `-g -O0`
- **Release:** `-O2`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure C, no ML libraries | Educational: expose every algorithmic detail |
| CPU-only, offline | No GPU/network dependencies; runs anywhere |
| Min-max normalization | Simple to explain and implement |
| Binary model format | Compact and portable |
| Single-layer perceptron | Simplest neural network; good for learning |
| Menu-driven CLI | Easy to use and extend |
| Placeholder files in Phase 1 | Clean foundation for incremental development |

## Future Phases

| Phase | Features |
|-------|----------|
| Phase 2 | CSV parsing, dataset management |
| Phase 3 | Data preprocessing, normalization |
| Phase 4 | Single-layer perceptron |
| Phase 5 | Training pipeline |
| Phase 6 | Evaluation metrics |
| Phase 7 | Model serialization, prediction |
| Phase 8 | CLI interface, GUI with GTK |
