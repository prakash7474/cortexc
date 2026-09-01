# =========================================================================
#  CortexC Makefile
#  CPU-Only Machine Learning Engine in Pure C17
#  Phase 5 - Testing + Optimization + Documentation + Deployment
# =========================================================================

# --- Compiler (override with: make CC=clang) -----------------------------
CC ?= gcc

# --- Directories --------------------------------------------------------
BUILDDIR   = build
MODELLDIR  = models
TESTDIR    = tests
TOOLSDIR   = tools
BINDIR     = bin

# --- Compiler flags -----------------------------------------------------
STD_CFLAGS    = -std=c17
WARN_CFLAGS   = -Wall -Wextra -Wpedantic -Wshadow -Wwrite-strings \
                -Wpointer-arith -Wcast-align -Wformat=2
INCLUDES      = -Iinclude

# Optimization flags
OPT_RELEASE   = -O2
OPT_DEBUG     = -g -O0

# Sanitizer support (address + undefined behavior). Only enabled for
# debug builds (or when SANITIZE=1 is passed to make).
SANITIZE_CFLAGS  = -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZE_LDFLAGS = -fsanitize=address,undefined

# Detect whether the compiler supports sanitizers (used by make debug).
SAN_SUPPORTED := $(shell printf 'int main(void){return 0;}' | $(CC) $(STD_CFLAGS) $(SANITIZE_CFLAGS) -x c - -o /dev/null 2>/dev/null && echo yes || echo no)

# --- Build-mode selection ----------------------------------------------
# Default CFLAGS used by the "test" targets.
OPT = $(OPT_RELEASE)
SAN_CFLAGS =
SAN_LDFLAGS =

ifeq ($(SANITIZE),1)
  OPT = $(OPT_DEBUG)
  SAN_CFLAGS = $(SANITIZE_CFLAGS)
  SAN_LDFLAGS = $(SANITIZE_LDFLAGS)
endif

CFLAGS    = $(STD_CFLAGS) $(WARN_CFLAGS) $(OPT) $(SAN_CFLAGS)
LDFLAGS   = $(SAN_LDFLAGS)

# --- Source sets --------------------------------------------------------
CORE_SRCS = src/main.c \
            src/dataset.c \
            src/preprocessing.c \
            src/perceptron.c \
            src/prediction.c \
            src/evaluation.c \
            src/model_io.c \
            src/config.c \
            src/utils.c

GUI_SRCS  = src/main_gui.c \
            src/dataset.c \
            src/preprocessing.c \
            src/perceptron.c \
            src/prediction.c \
            src/evaluation.c \
            src/model_io.c \
            src/config.c \
            src/utils.c

LIB_SRCS  = src/dataset.c \
            src/preprocessing.c \
            src/perceptron.c \
            src/prediction.c \
            src/evaluation.c \
            src/model_io.c \
            src/config.c \
            src/utils.c

# --- Targets ------------------------------------------------------------
TARGET                 = $(BUILDDIR)/cortexc
GUI_TARGET             = $(BUILDDIR)/cortexc-gui
BENCH_TARGET           = $(BUILDDIR)/benchmark

TEST_DATASET       = $(BUILDDIR)/test_dataset
TEST_PREPROC       = $(BUILDDIR)/test_preprocessing
TEST_PERCEPTRON    = $(BUILDDIR)/test_perceptron
TEST_MODEL         = $(BUILDDIR)/test_model
TEST_EVAL          = $(BUILDDIR)/test_evaluation
TEST_BATCH         = $(BUILDDIR)/test_batch
TEST_INTEGRATION   = $(BUILDDIR)/test_integration

ALL_TESTS = $(TEST_DATASET) $(TEST_PREPROC) $(TEST_PERCEPTRON) \
            $(TEST_MODEL) $(TEST_EVAL) $(TEST_BATCH) $(TEST_INTEGRATION)

# GTK flags (only used if GTK 3 is available).
GTK_CFLAGS  = $(shell pkg-config --cflags gtk+-3.0 2>/dev/null)
GTK_LDFLAGS = $(shell pkg-config --libs gtk+-3.0 2>/dev/null)

# =========================================================================
#  Default target
# =========================================================================
.PHONY: all debug release test test-debug bench benchmark gui clean \
        run train package help

all: $(TARGET)
	@mkdir -p $(MODELLDIR)
	@echo "CortexC built: $(TARGET)"

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- CLI binary ---------------------------------------------------------
$(TARGET): $(CORE_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CORE_SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# --- Debug build (with sanitizers where supported) ----------------------
debug: CFLAGS = $(STD_CFLAGS) $(WARN_CFLAGS) $(OPT_DEBUG) $(SANITIZE_CFLAGS)
debug: LDFLAGS = $(SANITIZE_LDFLAGS)
debug: clean
	$(MAKE) $(TARGET)
	@echo "Debug build complete: $(TARGET)"
	@if [ "$(SAN_SUPPORTED)" = "yes" ]; then \
		echo "Sanitizers (ASan+UBSan) enabled for debug build."; \
	else \
		echo "WARNING: AddressSanitizer is not supported by $(CC); debug build is symbol-only."; \
	fi

# --- Release build ------------------------------------------------------
release: CFLAGS = $(STD_CFLAGS) $(WARN_CFLAGS) $(OPT_RELEASE)
release: clean
	$(MAKE) $(TARGET)
	@echo "Release build complete: $(TARGET)"

# --- Tests --------------------------------------------------------------
test: $(ALL_TESTS)
	@echo ""
	@echo "========================================="
	@echo "  CortexC Test Suite"
	@echo "========================================="
	@echo "=== Dataset ===";       ./$(TEST_DATASET)
	@echo "=== Preprocessing ==="; ./$(TEST_PREPROC)
	@echo "=== Perceptron ===";    ./$(TEST_PERCEPTRON)
	@echo "=== Evaluation ===";    ./$(TEST_EVAL)
	@echo "=== Model I/O ===";     ./$(TEST_MODEL)
	@echo "=== Batch Prediction ==="; ./$(TEST_BATCH)
	@echo "=== Integration ===";   ./$(TEST_INTEGRATION)

# Run tests with sanitizers enabled (needs compiler support).
test-debug: CFLAGS = $(STD_CFLAGS) $(WARN_CFLAGS) $(OPT_DEBUG) $(SANITIZE_CFLAGS)
test-debug: LDFLAGS = $(SANITIZE_LDFLAGS)
test-debug: clean $(ALL_TESTS)
	@echo "Running tests with AddressSanitizer + UBSan..."
	@echo "=== Dataset ===";       ./$(TEST_DATASET)
	@echo "=== Preprocessing ==="; ./$(TEST_PREPROC)
	@echo "=== Perceptron ===";    ./$(TEST_PERCEPTRON)
	@echo "=== Evaluation ===";    ./$(TEST_EVAL)
	@echo "=== Model I/O ===";     ./$(TEST_MODEL)
	@echo "=== Batch Prediction ==="; ./$(TEST_BATCH)
	@echo "=== Integration ===";   ./$(TEST_INTEGRATION)

$(TEST_DATASET): tests/test_dataset.c src/dataset.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_DATASET) $(LDFLAGS)

$(TEST_PREPROC): tests/test_preprocessing.c src/dataset.c src/preprocessing.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_PREPROC) $(LDFLAGS)

$(TEST_PERCEPTRON): tests/test_perceptron.c src/dataset.c src/preprocessing.c src/perceptron.c src/prediction.c src/evaluation.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_PERCEPTRON) $(LDFLAGS)

$(TEST_MODEL): tests/test_model.c src/model_io.c src/perceptron.c src/preprocessing.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_MODEL) $(LDFLAGS)

$(TEST_EVAL): tests/test_evaluation.c src/evaluation.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_EVAL) $(LDFLAGS)

$(TEST_BATCH): tests/test_batch.c src/prediction.c src/perceptron.c src/preprocessing.c src/model_io.c src/dataset.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_BATCH) $(LDFLAGS)

$(TEST_INTEGRATION): tests/test_integration.c src/dataset.c src/preprocessing.c src/perceptron.c src/prediction.c src/evaluation.c src/model_io.c src/config.c src/utils.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(TEST_INTEGRATION) $(LDFLAGS)

# --- Benchmark ----------------------------------------------------------
bench: bench-make
bench-make: $(BENCH_TARGET)
	@echo ""
	@echo "=== CortexC Benchmark (100000 samples x 3 features) ==="
	./$(BENCH_TARGET) 100000 3

$(BENCH_TARGET): tools/benchmark.c $(LIB_SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(BENCH_TARGET) $(LDFLAGS)

# --- Dataset generator --------------------------------------------------
$(BUILDDIR)/generate_dataset: tools/generate_dataset.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $(BUILDDIR)/generate_dataset $(LDFLAGS)

generate: $(BUILDDIR)/generate_dataset
	@echo "Usage: ./$(BUILDDIR)/generate_dataset <samples> <features> <out.csv> [seed]"

# --- GUI (requires GTK 3) -----------------------------------------------
gui: $(GUI_SRCS) | $(BUILDDIR)
	@if pkg-config --exists gtk+-3.0 2>/dev/null; then \
		$(CC) $(CFLAGS) $(GTK_CFLAGS) $(INCLUDES) $(GUI_SRCS) -o $(GUI_TARGET) $(GTK_LDFLAGS) $(LDFLAGS); \
		echo "GUI build complete: $(GUI_TARGET)"; \
	else \
		echo "ERROR: GTK 3 development libraries not found."; \
		echo "Install with: sudo apt install libgtk-3-dev (Debian/Ubuntu)"; \
		echo "              brew install gtk+3 (macOS)"; \
		echo "              MSYS2: pacman -S mingw-w64-x86_64-gtk3"; \
		exit 1; \
	fi

# --- Packaging ----------------------------------------------------------
package: all
	mkdir -p $(BINDIR)
	cp $(TARGET) $(BINDIR)/ 2>/dev/null || true
	mkdir -p $(MODELLDIR) data docs
	@echo "CortexC packaged in ./$(BINDIR)/"
	@echo "  $(shell ls $(BINDIR))   <- executable"

# --- Run helpers --------------------------------------------------------
run: $(TARGET)
	./$(TARGET)

train: $(TARGET)
	./$(TARGET) train data/students.csv

# --- Clean --------------------------------------------------------------
clean:
	rm -f $(BUILDDIR)/cortexc $(BUILDDIR)/cortexc.exe \
	      $(BUILDDIR)/cortexc-gui $(BUILDDIR)/cortexc-gui.exe \
	      $(TEST_DATASET) $(TEST_DATASET).exe \
	      $(TEST_PREPROC) $(TEST_PREPROC).exe \
	      $(TEST_PERCEPTRON) $(TEST_PERCEPTRON).exe \
	      $(TEST_MODEL) $(TEST_MODEL).exe \
	      $(TEST_EVAL) $(TEST_EVAL).exe \
	      $(TEST_BATCH) $(TEST_BATCH).exe \
	      $(TEST_INTEGRATION) $(TEST_INTEGRATION).exe \
	      $(BENCH_TARGET) $(BENCH_TARGET).exe \
	      $(BUILDDIR)/generate_dataset $(BUILDDIR)/generate_dataset.exe \
	      src/*.o
	rm -f $(BINDIR)/cortexc $(BINDIR)/cortexc.exe
	rm -rf $(BUILDDIR)/benchmark_data.csv $(BUILDDIR)/benchmark_model.bin
	@echo "Clean complete"

# --- Help ---------------------------------------------------------------
help:
	@echo "CortexC Build System (Phase 5)"
	@echo "=============================="
	@echo "  make            Build the CLI binary (optimized)"
	@echo "  make debug      Build with -g -O0 (+ sanitizers if supported)"
	@echo "  make release    Build optimized (-O2)"
	@echo "  make test       Build and run the full test suite"
	@echo "  make test-debug Run tests with AddressSanitizer + UBSan"
	@echo "  make bench      Run the CPU benchmark (100k samples)"
	@echo "  make gui        Build the GTK GUI (requires GTK 3)"
	@echo "  make run        Build and run CLI"
	@echo "  make train      Build and train on data/students.csv"
	@echo "  make package    Create a release layout in ./bin/"
	@echo "  make generate   Build the synthetic dataset generator"
	@echo "  make clean      Remove build artifacts"
	@echo "  make help       Show this help"
	@echo ""
	@echo "CLI Commands:"
	@echo "  cortexc train <dataset.csv>"
	@echo "  cortexc predict <model.bin> <f1> <f2> ... <fn>"
	@echo "  cortexc predict-batch <model.bin> <csv> [--output <out.csv>]"
	@echo "  cortexc info <model.bin>"
	@echo "  cortexc --version | --help"
