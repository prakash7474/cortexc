# C-AI Makefile
# CPU-Only Machine Learning in Pure C
# Phase 1 - Initial Build System

CC       = gcc
CFLAGS   = -std=c17 -Wall -Wextra -Wpedantic
INCLUDES = -Iinclude

# Debug build
DEBUG_CFLAGS = -std=c17 -Wall -Wextra -Wpedantic -g -O0

# Release build
RELEASE_CFLAGS = -std=c17 -Wall -Wextra -Wpedantic -O2

# Source files (Phase 1: only main.c)
SRCS = src/main.c

# Output directory and target
BUILDDIR = build
TARGET   = $(BUILDDIR)/c-ai

# --- Default target ---
all: $(TARGET)

# --- Ensure build directory exists ---
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Link main binary ---
$(TARGET): $(SRCS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

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
	rm -f $(TARGET) $(TARGET).exe src/*.o
	@echo "Clean complete"

# --- Run ---
run: $(TARGET)
	./$(TARGET)

# --- Help ---
help:
	@echo "C-AI Build System"
	@echo "================="
	@echo "  make          - Build the C-AI binary"
	@echo "  make debug    - Build with debug symbols (-g -O0)"
	@echo "  make release  - Build optimized (-O2)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Build and run"
	@echo "  make help     - Show this help"

.PHONY: all debug release clean run help
