/**
 * model_io.h - Model persistence (save/load)
 *
 * Saves and loads trained perceptron models to/from binary files.
 * Includes preprocessing scaler statistics so prediction inputs
 * can be normalized consistently.
 *
 * C17 standard
 */

#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <stddef.h>
#include <stdint.h>

#include "perceptron.h"
#include "preprocessing.h"

/* Magic identifier written at the start of every model file */
#define MODEL_MAGIC         "CORTEXC"
#define MODEL_MAGIC_LEN     7

/* Current file format version */
#define MODEL_FORMAT_VERSION 1

/**
 * ModelFileHeader - On-disk header written before model data.
 *
 * magic:          7-byte magic string "CORTEXC"
 * format_version: uint32_t, must match MODEL_FORMAT_VERSION on load
 * feature_count:  uint32_t, number of input features
 * learning_rate:  float, training learning rate
 * epochs:         uint32_t, epochs actually trained
 * bias:           float, perceptron bias
 */
typedef struct {
    char     magic[MODEL_MAGIC_LEN];
    uint32_t format_version;
    uint32_t feature_count;
    float    learning_rate;
    uint32_t epochs;
    float    bias;
} ModelFileHeader;

/**
 * ModelInfo - In-memory representation of a loaded model file.
 *
 * perceptron:     loaded perceptron (caller owns after model_load)
 * scaler:         loaded scaler (caller owns after model_load)
 * loaded_from:    path the model was loaded from (for display)
 */
typedef struct {
    Perceptron  *perceptron;
    Scaler       scaler;
    char         loaded_from[256];
} ModelInfo;

/**
 * model_save - Save a trained perceptron and scaler to a binary file.
 *
 * The file format is:
 *   - ModelFileHeader (fixed-size header)
 *   - weight_count floats (scaler min values)
 *   - weight_count floats (scaler max values)
 *   - weight_count floats (perceptron weights)
 *
 * Returns 0 on success, -1 on error.
 */
int model_save(const char *filepath, const Perceptron *model,
               const Scaler *scaler);

/**
 * model_load - Load a perceptron and scaler from a binary file.
 *
 * Validates magic identifier, format version, feature count, and
 * file read success. Rejects corrupted or incompatible files.
 *
 * On success, *model_out and *scaler_out are allocated and must be freed
 * by the caller using perceptron_free() and scaler_free().
 *
 * Returns 0 on success, -1 on error.
 */
int model_load(const char *filepath, Perceptron **model_out,
               Scaler *scaler_out);

/**
 * model_load_info - Load a model and populate a ModelInfo structure.
 *
 * Convenience wrapper around model_load that also records the source path.
 *
 * Returns 0 on success, -1 on error.
 */
int model_load_info(const char *filepath, ModelInfo *info);

/**
 * model_info_free - Release resources owned by a ModelInfo.
 *
 * Frees the perceptron and scaler. Safe to call on a zeroed struct.
 */
void model_info_free(ModelInfo *info);

/**
 * model_print_info - Display model information to stdout.
 *
 * Shows algorithm, version, features, learning rate, epochs,
 * bias, and weight values.
 */
void model_print_info(const ModelInfo *info);

#endif /* MODEL_IO_H */
