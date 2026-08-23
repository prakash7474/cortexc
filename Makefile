# C-AI Makefile
# CPU-Only Machine Learning in Pure C
# Phase 4 - CLI + GUI Application

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

# Source files for the main CLI binary
SRCS = src/main.c \
       src/dataset.c \
       src/preprocessing.c \
       src/perceptron.c \
       src/prediction.c \
       src/evaluation.c \
       src/model_io.c \
       src/config.c \
       src/utils.c

# GUI source files (requires GTK 3)
GUI_SRCS = src/main_gui.c \
           src/dataset.c \
           src/preprocessing.c \
           src/perceptron.c \
           src/prediction.c \
           src/evaluation.c \
           src/model_io.c \
           src/config.c \
           src/utils.c

# GTK flags (pkg-config)
GTK_CFLAGS  = $(shell pkg-config --cflags gtk+-3.0 2>/dev/null)
GTK_LDFLAGS = $(shell pkg-config --libs gtk+-3.0 2>/dev/null)

# Test source files
TEST_DATASET_SRCS    = tests/test_dataset.c src/dataset.c src/config.c src/utils.c
TEST_PREPROC_SRCS    = tests/test_preprocessing.c src/dataset.c src/preprocessing.c src/config.c src/utils.c
TEST_PERCEPTRON_SRCS = tests/test_perceptron.c src/dataset.c src/preprocessing.c src/perceptron.c src/prediction.c src/evaluation.c src/config.c src/utils.c
TEST_MODEL_SRCS      = tests/test_model.c src/model_io.c src/perceptron.c src/preprocessing.c src/config.c src/utils.c

# Output directory and targets
BUILDDIR           = build
TARGET             = $(BUILDDIR)/cortexc
GUI_TARGET         = $(BUILDDIR)/cortexc-gui
TEST_DATASET       = $(BUILDDIR)/test_dataset
TEST_PREPROC       = $(BUILDDIR)/test_preprocessing
TEST_PERCEPTRON    = $(BUILDDIR)/test_perceptron
TEST_MODEL         = $(BUILDDIR)/test_model

# Default model directory
MODELS_DIR = models

# --- Default target ---
all: $(TARGET)
	@mkdir -p $(MODELS_DIR)

# --- Ensure build directory exists ---
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Link CLI binary ---
$(TARGET): $(SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# --- Link GUI binary (requires GTK 3) ---
gui: $(GUI_SRCS) | $(BUILDDIR)
	@if pkg-config --exists gtk+-3.0 2>/dev/null; then \
		$(CC) $(CFLAGS) $(GTK_CFLAGS) $(INCLUDES) $(GUI_SRCS) -o $(GUI_TARGET) $(GTK_LDFLAGS); \
		echo "GUI build complete: $(GUI_TARGET)"; \
	else \
		echo "Error: GTK 3 development libraries not found."; \
		echo "Install with: sudo apt install libgtk-3-dev (Debian/Ubuntu)"; \
		echo "              brew install gtk+3 (macOS)"; \
		exit 1; \
	fi

# --- Test targets ---
tests: $(TEST_DATASET) $(TEST_PREPROC) $(TEST_PERCEPTRON) $(TEST_MODEL)
	@echo ""
	@echo "=== Running Dataset Tests ==="
	./$(TEST_DATASET)
	@echo ""
	@echo "=== Running Preprocessing Tests ==="
	./$(TEST_PREPROC)
	@echo ""
	@echo "=== Running Perceptron Tests ==="
	./$(TEST_PERCEPTRON)
	@echo ""
	@echo "=== Running Model I/O Tests ==="
	./$(TEST_MODEL)

$(TEST_DATASET): $(TEST_DATASET_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_DATASET_SRCS) -o $(TEST_DATASET)

$(TEST_PREPROC): $(TEST_PREPROC_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_PREPROC_SRCS) -o $(TEST_PREPROC)

$(TEST_PERCEPTRON): $(TEST_PERCEPTRON_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_PERCEPTRON_SRCS) -o $(TEST_PERCEPTRON)

$(TEST_MODEL): $(TEST_MODEL_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_MODEL_SRCS) -o $(TEST_MODEL)

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
	      $(GUI_TARGET) $(GUI_TARGET).exe \
	      $(TEST_DATASET) $(TEST_DATASET).exe \
	      $(TEST_PREPROC) $(TEST_PREPROC).exe \
	      $(TEST_PERCEPTRON) $(TEST_PERCEPTRON).exe \
	      $(TEST_MODEL) $(TEST_MODEL).exe \
	      src/*.o
	@echo "Clean complete"

# --- Run CLI ---
run: $(TARGET)
	./$(TARGET)

# --- Run with train command ---
train: $(TARGET)
	./$(TARGET) train data/students.csv

# --- Help ---
help:
	@echo "CortexC Build System (Phase 4)"
	@echo "=============================="
	@echo "  make          - Build the CLI binary"
	@echo "  make gui      - Build the GTK GUI (requires GTK 3)"
	@echo "  make debug    - Build with debug symbols (-g -O0)"
	@echo "  make release  - Build optimized (-O2)"
	@echo "  make tests    - Build and run all tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Build and run CLI"
	@echo "  make train    - Build and train on students.csv"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "CLI Usage:"
	@echo "  cortexc train <dataset.csv>"
	@echo "  cortexc predict <model.bin> <f1> <f2> <f3>"
	@echo "  cortexc info <model.bin>"

.PHONY: all debug release clean run train tests help gui
