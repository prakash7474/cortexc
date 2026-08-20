# C-AI — System Requirements

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| CPU | Modern x64 processor |
| RAM | 4 GB |
| Storage | 500 MB free space |
| GPU | **Not required** |

### Recommended

| Component | Requirement |
|-----------|-------------|
| CPU | 4+ CPU cores |
| RAM | 8 GB |
| Storage | 1 GB+ free space |
| GPU | **Not required** |

> **Important:** This project does NOT use GPU acceleration.
> All machine learning computations are performed on the CPU using system RAM.

## Software Requirements

### Required

| Software | Version | Purpose |
|----------|---------|---------|
| C compiler | GCC ≥ 9 or Clang ≥ 10 | C17 support required |
| GNU Make | ≥ 4.0 | Build automation |
| Git | Any recent version | Version control |

### Development Tools (Recommended)

| Tool | Purpose |
|------|---------|
| GDB | Debugging |
| AddressSanitizer | Memory error detection (built into GCC/Clang) |
| Valgrind | Memory leak detection (Linux/macOS) |

### Future Requirements

| Software | Purpose | Phase |
|----------|---------|-------|
| GTK | GUI development | Phase 8 |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows | Primary target | Build with MSYS2/MinGW or WSL |
| Linux | Supported | Native GCC/Clang |
| macOS | Supported | Native Clang |

## Resource Usage

### CPU
- All ML calculations are performed by the CPU
- Single-threaded execution in Phase 1
- No SIMD or vectorization required (but may be added later)

### RAM
- Used for:
  - Loading datasets into memory
  - Storing model parameters (weights, bias)
  - Intermediate calculations during training
  - Feature normalization buffers
- Estimated usage: Dataset size × (features + labels) × 8 bytes

### Storage (SSD/HDD)
- Used for:
  - Storing CSV dataset files
  - Saving trained model files (binary format)
  - Application source code and binaries
- Model files: ~1 KB per model (3 features)

## GPU Statement

**This project does not use GPU acceleration.**

The following technologies are explicitly excluded:
- CUDA
- cuDNN
- TensorRT
- OpenCL
- Any GPU-based computing framework

All machine learning computation is performed using:
- CPU instructions
- System RAM for data storage
- SSD/HDD for file persistence

## Network Requirements

- **None.** The application works completely offline.
- No network calls, API requests, or internet connectivity required.
