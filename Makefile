# C-AI Makefile
# CPU-Only Machine Learning in Pure C
# Phase 3 - Perceptron Training Pipeline

CC       = gcc
INCLUDES = -Iinclude

# Standard flags (C17 + strict warnings)
STD_CFLAGS = -std=c17 -Wall -Wextra -Wpedantic

# Debug flags
DEBUG_CFLAGS = $(STD_CFLAGS) -g -O0

# Release flags
RELEASE_CFLAGS = $(STD_CFLAGS) -O2

# Default CFLAGS (normal build)
CFLAGS = $(STD_CFLAGS)

# Source files for the main binary
SRCS = src/main.c \
       src/dataset.c \
       src/preprocessing.c \
       src/perceptron.c \
       src/prediction.c \
       src/evaluation.c \
       src/config.c \
       src/utils.c

# Test source files
TEST_DATASET_SRCS    = tests/test_dataset.c src/dataset.c src/config.c src/utils.c
TEST_PREPROC_SRCS    = tests/test_preprocessing.c src/dataset.c src/preprocessing.c src/config.c src/utils.c
TEST_PERCEPTRON_SRCS = tests/test_perceptron.c src/dataset.c src/preprocessing.c src/perceptron.c src/prediction.c src/evaluation.c src/config.c src/utils.c

# Output directory and targets
BUILDDIR           = build
TARGET             = $(BUILDDIR)/c-ai
TEST_DATASET       = $(BUILDDIR)/test_dataset
TEST_PREPROC       = $(BUILDDIR)/test_preprocessing
TEST_PERCEPTRON    = $(BUILDDIR)/test_perceptron

# --- Default target ---
all: $(TARGET)

# --- Ensure build directory exists ---
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Link main binary ---
$(TARGET): $(SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# --- Test targets ---
tests: $(TEST_DATASET) $(TEST_PREPROC) $(TEST_PERCEPTRON)
	@echo ""
	@echo "=== Running Dataset Tests ==="
	./$(TEST_DATASET)
	@echo ""
	@echo "=== Running Preprocessing Tests ==="
	./$(TEST_PREPROC)
	@echo ""
	@echo "=== Running Perceptron Tests ==="
	./$(TEST_PERCEPTRON)

$(TEST_DATASET): $(TEST_DATASET_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_DATASET_SRCS) -o $(TEST_DATASET)

$(TEST_PREPROC): $(TEST_PREPROC_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_PREPROC_SRCS) -o $(TEST_PREPROC)

$(TEST_PERCEPTRON): $(TEST_PERCEPTRON_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_PERCEPTRON_SRCS) -o $(TEST_PERCEPTRON)

# --- Debug build ---
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean $(TARGET)
	@echo "Debug build complete"

# --- Release build ---
release: CFLAGS = $(RELEASE_CFLAGS)
release: clean $(TARGET)
	@echo "Release build complete"

# --- Clean ---
clean:
	rm -f $(TARGET) $(TARGET).exe \
	      $(TEST_DATASET) $(TEST_DATASET).exe \
	      $(TEST_PREPROC) $(TEST_PREPROC).exe \
	      $(TEST_PERCEPTRON) $(TEST_PERCEPTRON).exe \
	      src/*.o
	@echo "Clean complete"

# --- Run ---
run: $(TARGET)
	./$(TARGET)

# --- Help ---
help:
	@echo "C-AI Build System (Phase 3)"
	@echo "==========================="
	@echo "  make          - Build the C-AI binary"
	@echo "  make debug    - Build with debug symbols (-g -O0)"
	@echo "  make release  - Build optimized (-O2)"
	@echo "  make tests    - Build and run all tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Build and run"
	@echo "  make help     - Show this help"

.PHONY: all debug release clean run tests help
