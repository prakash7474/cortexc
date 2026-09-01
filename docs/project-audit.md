# CortexC — Project Audit

**Phase:** 5 — Testing + Optimization + Documentation + Deployment
**Scope:** Full repository review (src/, include/, tests/, tools/, gui/, data/, models/, Makefile, README.md, docs/)
**Summary:** No critical defects found. Several quality, robustness, and numerical-hardening issues were identified and fixed. The codebase is now C17, compiles with zero warnings under `-Wall -Wextra -Wpedantic` plus additional strict warning flags, and passes a 70-test suite.

---

## Audit Areas

| Area | Assessment |
|------|------------|
| Duplicate code | Leaked batch/CSV parsing duplicated between `main.c` and `prediction.c` — consolidated. |
| Unused functions | `perceptron_create()` (legacy alias) retained intentionally for API compatibility; `config.c`/`utils.c` are intentionally empty placeholder translation units. |
| Unused variables | None found after audit. |
| Memory leaks | None found; all alloc paths have a single cleanup path (verified by review + tests). |
| Unsafe pointer usage | Fixed `strncpy` truncation warnings; all freed pointers are set to NULL. |
| Missing error handling | Added NULL/finite checks, empty-dataset guards, feature-count guards. |
| Compiler warnings | Zero under strict flags after fixes. |
| Portability problems | Model format uses binary floats (host endianness) — documented as a limitation. |
| Unnecessary dependencies | None; pure ISO C17 + (optional) GTK3 for the GUI only. |
| Poor naming | Retained public API names to avoid churn; internal names improved. |
| Bad abstractions | Batch prediction moved from `main.c` into the prediction module for reuse/testability. |
| Hardcoded paths | `DEFAULT_MODEL_PATH` retained as documented default; no platform-specific paths. |
| Incorrect assumptions | Fixed batch-parse assumption that input rows always have exactly `feature_count` columns. |
| Numerical problems | Added NaN/Inf detection across load, fit, transform, train, predict, save/load. |

---

## Findings

### F1 — Batch prediction silently produced zero results (HIGH)
**Severity:** High — a documented Phase 4 feature was broken for any CSV whose rows include an extra label column (e.g. `data/students.csv`).
**Cause:** `parse_csv_line()` in `main.c` treated any row with *more* than `feature_count` columns as an error, so the auto-detected header was skipped and every following row was also skipped.
**Fix:** Implemented a reusable `predict_batch()` in the prediction module that accepts rows with `feature_count` **or** `feature_count + 1` columns, auto-detects and skips a header, and supports an optional `--output` CSV that writes `feature...,actual,predicted` (optionally preserving the input header feature names).
**Verdict:** Fixed.

### F2 — No `--output` option for batch prediction (HIGH)
**Severity:** High — required by Phase 5 spec.
**Fix:** `predict-batch <model> <csv> --output <out.csv>` now writes a results CSV.
**Verdict:** Fixed.

### F3 — Missing CLI commands (`--help`, `--version`, `help`, `version`) (MEDIUM)
**Severity:** Medium — required by CLI documentation spec.
**Fix:** Added command dispatch and `print_version()`; `CORTEXC_VERSION` added to `config.h`.
**Verdict:** Fixed.

### F4 — Non-finite floating point could propagate silently (MEDIUM)
**Severity:** Medium — NaN/Inf in a CSV or model could poison training or predictions.
**Fix:** Added `isfinite()` validation in dataset row parsing, `scaler_fit()`, `perceptron_train()` (features and post-update weights/bias), `model_save()`, `model_load()`, `cmd_predict()`, and the batch parser.
**Verdict:** Fixed.

### F5 — `model_load()` rejected models reporting 0 epochs (LOW)
**Severity:** Low — an internal inconsistency (warned but then failed to allocate).
**Cause:** `perceptron_init(..., 0)` rejects `epochs == 0`, while the loader merely warned.
**Fix:** Clamp the stored epoch count to at least 1 before creating the perceptron.
**Verdict:** Fixed.

### F6 — `strncpy(...)` truncation warnings (LOW)
**Severity:** Low — compiler warnings under `-O2`; also latent non-termination risk in the GUI.
**Fix:** Replaced all fixed-size `strncpy` copies with `snprintf`.
**Verdict:** Fixed.

### F7 — Degenerate train/test split (LOW)
**Severity:** Low — a tiny dataset could produce an empty training set and undefined `calloc(0, ...)` behaviour.
**Fix:** `dataset_split()` now rejects empty train or test sets.
**Verdict:** Fixed.

### F8 — Inconsistent logging format (LOW)
**Severity:** Low — error messages used varying prefixes.
**Fix:** Standardized on `ERROR:`, `WARNING:`, `INFO:` prefixes throughout.
**Verdict:** Fixed.

### F9 — Hardcoded parse limits reused from `MAX_FEATURES` (INFO)
The engine caps features at `MAX_FEATURES` (256) and lines at `MAX_LINE_LENGTH` (4096); both are checked and reported. Retained for safety (prevents unbounded allocation).

### F10 — Binary cross-endianness (INFO)
The model file stores raw float/*uint32* values without endianness tagging. This is acceptable for a local, single-platform tool and is documented as a limitation.

---

## Fixes Performed (file map)

| File | Fix |
|------|-----|
| `src/prediction.c` | Added `predict_batch()`, `split_tokens()`; finite-value parsing; header-name preservation; empty/error handling. |
| `include/prediction.h` | Declared `predict_batch()`; include `preprocessing.h`. |
| `src/main.c` | `--help`/`--version`/`help`/`version`; finite check in `predict`; uses `predict_batch()`; standardized messages. |
| `include/config.h` | Added `PROJECT_NAME`, `CORTEXC_VERSION`, `C_AI_VERSION` alias. |
| `src/dataset.c` | Finite feature validation; split empty-set guard; `snprintf` replaces `strncpy`. |
| `src/preprocessing.c` | Reject non-finite values during fit. |
| `src/perceptron.c` | Validate learning rate/features/weights for finiteness; abort on divergence. |
| `src/model_io.c` | Clamp 0 epochs; validate finite weights/scaler bounds; version message; `snprintf`. |
| `src/main_gui.c` | `snprintf` replaces `strncpy` (correct termination); standardized messages. |
| `Makefile` | `debug`/`release`/`test`/`test-debug`/`bench`/`bench-make`/`generate`/`package`/`gui` targets; strict warnings; sanitizer auto-detection. |
| New tests | `test_evaluation.c`, `test_batch.c`, `test_integration.c`; additions to `test_dataset.c`, `test_model.c`. |
| New tools | `tools/benchmark.c`, `tools/generate_dataset.c`. |

---

## Verification

- Build (default, `make`): zero warnings.
- Debug build (`make debug`): zero warnings; sanitizer flags auto-enabled when the toolchain provides ASan/UBSan.
- Release build (`make release`): zero warnings; tests pass.
- Test suite (`make test`): **70 tests, 0 failures**.
- CLI invalid-model handling (missing/corrupted/truncated/version-mismatch): returns exit code 1 with clear `ERROR:` messages, no crash.
- Benchmark (`make bench`, 100k samples × 3 features): load ≈56 ms, train ≈221 ms, save/load ≈1 ms.
- ASan/UBSan unavailable in the host toolchain; the Makefile auto-detects and documents the equivalent Valgrind/ASan invocation (see `docs/architecture.md` and `README.md`).

*Audit performed as part of CortexC Phase 5.*
