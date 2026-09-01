# CortexC — Public API Reference

All symbols below are C17 and reside in header files under `include/`. Unless noted, a function that returns an error code returns `0` on success and `-1` on failure, printing a message prefixed with `ERROR:` to `stderr`.

Memory ownership is explicit: the caller of any `*_new`/`*_load`/`*_split` function owns the returned object and must release it with the matching `*_free` function.

---

## Dataset module — `dataset.h`

### `Dataset *dataset_create(void)`
- **Returns:** pointer to an empty `Dataset`, or `NULL` on allocation failure.
- **Ownership:** caller frees with `dataset_free()`.
- **Errors:** allocation failure.

### `Dataset *dataset_load_csv(const char *filepath)`
- **Parameters:** `filepath` — path to a CSV file. First line must be a header `f1,...,fn,label`; following rows must contain `n` numeric features and an integer label (0 or 1).
- **Returns:** pointer to a populated `Dataset`, or `NULL` on any error (missing file, malformed data, invalid label, non-finite value, too many features, out of memory).
- **Ownership:** caller frees with `dataset_free()`.
- **Errors:** file not found; empty file; empty header; too few/too many columns; invalid number; invalid label; non-finite value; allocation failure.

### `void dataset_print_info(const Dataset *dataset)`
- **Returns:** `void`. Prints dataset summary (file, samples, features, labels, estimated memory) to stdout.
- **Errors:** `NULL` is handled (prints "(null)").

### `size_t dataset_estimate_memory(const Dataset *dataset)`
- **Returns:** approximate bytes used by the dataset, or `0` if `NULL`.
- **Notes:** includes `Sample` array, feature arrays, label storage, and struct overhead.

### `int dataset_split(const Dataset *original, Dataset **train_out, Dataset **test_out, float train_ratio, unsigned int seed)`
- **Parameters:** `original` (not modified), `train_out`/`test_out` (output pointers, must be non-NULL), `train_ratio` (exclusive in `(0,1)`), `seed` (reproducibility).
- **Returns:** `0` on success, `-1` on error.
- **Ownership:** on success the caller owns both outputs and must free each with `dataset_free()`.
- **Errors:** NULL args; out-of-range ratio; empty set; zero-size train/test split; allocation failure.

### `void dataset_free(Dataset **dataset)`
- **Returns:** `void`. Frees all samples, feature arrays, and the struct. Sets the pointer to `NULL`. Safe on `NULL`.

---

## Preprocessing module — `preprocessing.h`

### `int scaler_fit(Scaler *scaler, const struct Dataset *dataset)`
- **Parameters:** `scaler` (output; caller pre-zeroes or leaks nothing), `dataset` (training data only — do not fit on test data to avoid leakage).
- **Returns:** `0` on success, `-1` on error.
- **Ownership:** on success `scaler` owns two allocated arrays; caller frees with `scaler_free()`.
- **Errors:** NULL args; empty dataset; non-finite feature values; allocation failure.

### `int scaler_transform(const Scaler *scaler, struct Dataset *dataset)`
- **Parameters:** `scaler` (fitted), `dataset` (mutated in place).
- **Returns:** `0` on success, `-1` on error.
- **Ownership:** no new memory.
- **Errors:** NULL args; feature-count mismatch.

### `void scaler_free(Scaler *scaler)`
- **Returns:** `void`. Frees `min_values`/`max_values`, zeroes `num_features`. Safe on `NULL`.

---

## Perceptron module — `perceptron.h`

### `Perceptron *perceptron_init(size_t feature_count, float learning_rate, size_t epochs)`
- **Parameters:** `feature_count` (>0), `learning_rate` (exclusive in `(0,1]`), `epochs` (>0).
- **Returns:** pointer to a zeroed-weight perceptron, or `NULL` on invalid parameters/allocation failure.
- **Ownership:** caller frees with `perceptron_free()`.

### `Perceptron *perceptron_create(size_t feature_count, float learning_rate, unsigned int seed)`
- **Notes:** legacy alias for `perceptron_init` (kept for API compatibility; seed ignored). Returns as above.

### `int perceptron_forward(const Perceptron *p, const float *features)`
- **Returns:** `1` if weighted sum ≥ 0, else `0`. Returns `0` if `NULL` inputs.
- **Ownership:** none.

### `int perceptron_predict(const Perceptron *p, const float *features)`
- **Notes:** alias for `perceptron_forward()`.

### `int perceptron_train(Perceptron *p, const struct Dataset *dataset)`
- **Parameters:** `p` (mutated), `dataset` (normalized training data).
- **Returns:** number of epochs completed (early-stops when 0 errors), or `-1` on error.
- **Ownership:** none beyond `p`; `p->epochs` is set to the completed count.
- **Errors:** NULL args; feature-count mismatch; empty dataset; non-finite features; divergence (weights/bias became non-finite).

### `void perceptron_free(Perceptron **p)`
- **Returns:** `void`. Frees weights and struct; sets pointer to `NULL`. Safe on `NULL`.

---

## Prediction module — `prediction.h`

### `int predict_sample(const Perceptron *p, const float *features, int *prediction)`
- **Returns:** `0` on success, `-1` on error. On success `*prediction` is `0` or `1`.
- **Errors:** NULL args; zero features; NULL weights.

### `int *predict_dataset(const Perceptron *p, const Dataset *dataset)`
- **Returns:** heap array of predictions (one per sample), or `NULL` on error.
- **Ownership:** caller frees with `free()`.
- **Errors:** NULL args; feature-count mismatch; empty dataset; allocation failure.

### `int predict_batch(const Perceptron *p, const Scaler *scaler, const char *csv_path, const char *out_csv_path)`
- **Parameters:** `p`, `scaler` (fitted), `csv_path` (input CSV), `out_csv_path` (optional; `NULL` to skip writing a file).
- **Returns:** number of rows successfully predicted, or `-1` on a fatal error.
- **Behavior:** accepts rows with `feature_count` or `feature_count + 1` columns (the extra column is the actual label and is copied to output); skips a header row; optionally writes `feature...,actual,predicted`.
- **Errors:** NULL args; feature-count mismatch; missing scaler arrays; missing input file; cannot open output file.

---

## Evaluation module — `evaluation.h`

### `double calculate_accuracy(const int *predictions, const int *actuals, size_t count)`
- **Returns:** accuracy as a percentage `[0.0, 100.0]`, or `-1.0` if any arg is `NULL`/`count == 0`.

### `int calculate_confusion_matrix(const int *predictions, const int *actuals, size_t count, ConfusionMatrix *matrix)`
- **Parameters:** all non-NULL, labels must be `0` or `1`, `count > 0`.
- **Returns:** `0` on success, `-1` on error (NULL args, invalid label, `count == 0`).
- **Ownership:** none (fills `matrix`).

### `void print_confusion_matrix(const ConfusionMatrix *matrix)`
- **Returns:** `void`. Prints the TP/TN/FP/FN table. Safe on `NULL`.

---

## Model I/O module — `model_io.h`

### `int model_save(const char *filepath, const Perceptron *model, const Scaler *scaler)`
- **Returns:** `0` on success, `-1` on error.
- **File format:** header (magic `CORTEXC`, format version, feature count, learning rate, epochs, bias) then three `float` arrays: scaler min, scaler max, weights.
- **Errors:** NULL args; NULL weights; zero features; scaler mismatch; non-finite values; cannot open file; write errors.

### `int model_load(const char *filepath, Perceptron **model_out, Scaler *scaler_out)`
- **Parameters:** `filepath`, `model_out` (output), `scaler_out` (output; zeroed on failure).
- **Returns:** `0` on success, `-1` on error. On success the caller owns `*model_out` and `*scaler_out` (free with `perceptron_free()`/`scaler_free()`). On failure outputs are left safe (`model_out == NULL`, empty scaler).
- **Errors:** NULL args; missing file; truncated header; bad magic; unsupported version; zero/too-many features; invalid learning rate/bias; non-finite weights/bounds; read failures.

### `int model_load_info(const char *filepath, ModelInfo *info)`
- **Returns:** `0` on success, `-1` on error. Loads a model and records the source path in `info->loaded_from`. Caller frees with `model_info_free()`.

### `void model_info_free(ModelInfo *info)`
- **Returns:** `void`. Frees the embedded perceptron and scaler. Safe on a zeroed struct or `NULL`.

### `void model_print_info(const ModelInfo *info)`
- **Returns:** `void`. Prints algorithm, version, features, learning rate, epochs, bias, weights, and source path.

---

## Constants (config.h)

| Macro | Value | Meaning |
|-------|-------|---------|
| `CORTEXC_VERSION` / `C_AI_VERSION` | `"0.1.0"` | Engine version string. |
| `DEFAULT_LEARNING_RATE` | `0.1f` | Default training step size. |
| `DEFAULT_MAX_EPOCHS` | `1000` | Default epoch cap. |
| `DEFAULT_TRAIN_RATIO` | `0.8f` | Default train split fraction. |
| `DEFAULT_RANDOM_SEED` | `42` | Reproducibility seed. |
| `MAX_LINE_LENGTH` | `4096` | Max CSV line length. |
| `MAX_FEATURES` | `256` | Max features per sample. |
| `MAX_FILENAME` | `256` | Max path length. |
