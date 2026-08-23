/**
 * model_io.c - Model persistence (save/load)
 *
 * Binary file format (little-endian assumed on target platform):
 *   1. ModelFileHeader  (magic, version, feature_count, lr, epochs, bias)
 *   2. float[feature_count]  scaler min values
 *   3. float[feature_count]  scaler max values
 *   4. float[feature_count]  perceptron weights
 *
 * All file operations are checked. Corrupted files are rejected
 * with descriptive error messages.
 *
 * C17 standard
 */

#include "model_io.h"
#include "perceptron.h"
#include "preprocessing.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Private helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * write_floats - Write an array of floats to a file.
 * Returns 0 on success, -1 on write error.
 */
static int write_floats(FILE *fp, const float *data, size_t count)
{
    size_t written = fwrite(data, sizeof(float), count, fp);
    if (written != count) {
        fprintf(stderr, "Error: Failed to write float array (wrote %zu of %zu).\n",
                written, count);
        return -1;
    }
    return 0;
}

/**
 * read_floats - Read an array of floats from a file.
 * Returns 0 on success, -1 on read error.
 */
static int read_floats(FILE *fp, float *data, size_t count)
{
    size_t read_count = fread(data, sizeof(float), count, fp);
    if (read_count != count) {
        fprintf(stderr, "Error: Failed to read float array (read %zu of %zu).\n",
                read_count, count);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int model_save(const char *filepath, const Perceptron *model,
               const Scaler *scaler)
{
    if (!filepath || !model || !scaler) {
        fprintf(stderr, "Error: NULL argument to model_save.\n");
        return -1;
    }

    if (!model->weights) {
        fprintf(stderr, "Error: Model has NULL weights.\n");
        return -1;
    }

    if (model->feature_count == 0) {
        fprintf(stderr, "Error: Model has zero features.\n");
        return -1;
    }

    if (scaler->num_features != model->feature_count) {
        fprintf(stderr, "Error: Scaler feature count (%zu) does not match model (%zu).\n",
                scaler->num_features, model->feature_count);
        return -1;
    }

    if (!scaler->min_values || !scaler->max_values) {
        fprintf(stderr, "Error: Scaler has NULL min/max arrays.\n");
        return -1;
    }

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open '%s' for writing.\n", filepath);
        return -1;
    }

    /* Build header */
    ModelFileHeader header;
    memset(&header, 0, sizeof(header));

    memcpy(header.magic, MODEL_MAGIC, MODEL_MAGIC_LEN);
    header.format_version = MODEL_FORMAT_VERSION;
    header.feature_count  = (uint32_t)model->feature_count;
    header.learning_rate  = model->learning_rate;
    header.epochs         = (uint32_t)model->epochs;
    header.bias           = model->bias;

    /* Write header */
    size_t written = fwrite(&header, sizeof(header), 1, fp);
    if (written != 1) {
        fprintf(stderr, "Error: Failed to write model header.\n");
        fclose(fp);
        return -1;
    }

    size_t fc = model->feature_count;

    /* Write scaler min values */
    if (write_floats(fp, scaler->min_values, fc) != 0) {
        fclose(fp);
        return -1;
    }

    /* Write scaler max values */
    if (write_floats(fp, scaler->max_values, fc) != 0) {
        fclose(fp);
        return -1;
    }

    /* Write perceptron weights */
    if (write_floats(fp, model->weights, fc) != 0) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int model_load(const char *filepath, Perceptron **model_out,
               Scaler *scaler_out)
{
    if (!filepath || !model_out || !scaler_out) {
        fprintf(stderr, "Error: NULL argument to model_load.\n");
        return -1;
    }

    /* Initialize outputs to safe state */
    *model_out = NULL;
    memset(scaler_out, 0, sizeof(Scaler));

    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open model file '%s'.\n", filepath);
        return -1;
    }

    /* Read header */
    ModelFileHeader header;
    size_t read_count = fread(&header, sizeof(header), 1, fp);
    if (read_count != 1) {
        fprintf(stderr, "Error: Failed to read model header from '%s'.\n", filepath);
        fclose(fp);
        return -1;
    }

    /* Validate magic identifier */
    if (memcmp(header.magic, MODEL_MAGIC, MODEL_MAGIC_LEN) != 0) {
        fprintf(stderr, "Error: Invalid magic identifier in '%s'. "
                "Expected '%s'.\n", filepath, MODEL_MAGIC);
        fclose(fp);
        return -1;
    }

    /* Validate format version */
    if (header.format_version != MODEL_FORMAT_VERSION) {
        fprintf(stderr, "Error: Unsupported format version %u in '%s'. "
                "Expected %u.\n", header.format_version, filepath,
                MODEL_FORMAT_VERSION);
        fclose(fp);
        return -1;
    }

    /* Validate feature count */
    if (header.feature_count == 0) {
        fprintf(stderr, "Error: Model file has zero features.\n");
        fclose(fp);
        return -1;
    }

    if (header.feature_count > MAX_FEATURES) {
        fprintf(stderr, "Error: Model file has too many features (%u). "
                "Maximum is %d.\n", header.feature_count, MAX_FEATURES);
        fclose(fp);
        return -1;
    }

    size_t fc = (size_t)header.feature_count;

    /* Validate numeric values */
    if (header.learning_rate <= 0.0f || header.learning_rate > 1.0f) {
        fprintf(stderr, "Error: Invalid learning rate %f in model file.\n",
                header.learning_rate);
        fclose(fp);
        return -1;
    }

    if (header.epochs == 0) {
        fprintf(stderr, "Warning: Model file reports 0 epochs trained.\n");
    }

    /* Create perceptron with loaded parameters */
    Perceptron *model = perceptron_init(fc, header.learning_rate,
                                        (size_t)header.epochs);
    if (!model) {
        fprintf(stderr, "Error: Failed to create perceptron from model file.\n");
        fclose(fp);
        return -1;
    }

    model->bias   = header.bias;
    model->epochs = (size_t)header.epochs;

    /* Read scaler min values */
    scaler_out->min_values = malloc(fc * sizeof(float));
    scaler_out->max_values = malloc(fc * sizeof(float));
    if (!scaler_out->min_values || !scaler_out->max_values) {
        fprintf(stderr, "Error: Memory allocation failed for scaler arrays.\n");
        free(scaler_out->min_values);
        free(scaler_out->max_values);
        scaler_out->min_values = NULL;
        scaler_out->max_values = NULL;
        perceptron_free(&model);
        fclose(fp);
        return -1;
    }
    scaler_out->num_features = fc;

    if (read_floats(fp, scaler_out->min_values, fc) != 0) {
        scaler_free(scaler_out);
        perceptron_free(&model);
        fclose(fp);
        return -1;
    }

    /* Read scaler max values */
    if (read_floats(fp, scaler_out->max_values, fc) != 0) {
        scaler_free(scaler_out);
        perceptron_free(&model);
        fclose(fp);
        return -1;
    }

    /* Read perceptron weights */
    if (read_floats(fp, model->weights, fc) != 0) {
        scaler_free(scaler_out);
        perceptron_free(&model);
        fclose(fp);
        return -1;
    }

    /* Verify we read exactly the expected number of weights */
    /* Check if there is unexpected trailing data */
    float probe;
    if (fread(&probe, sizeof(float), 1, fp) == 1) {
        fprintf(stderr, "Warning: Unexpected trailing data in model file '%s'.\n",
                filepath);
    }

    fclose(fp);

    *model_out = model;
    return 0;
}

int model_load_info(const char *filepath, ModelInfo *info)
{
    if (!filepath || !info) {
        fprintf(stderr, "Error: NULL argument to model_load_info.\n");
        return -1;
    }

    memset(info, 0, sizeof(ModelInfo));

    if (model_load(filepath, &info->perceptron, &info->scaler) != 0) {
        return -1;
    }

    strncpy(info->loaded_from, filepath, sizeof(info->loaded_from) - 1);
    info->loaded_from[sizeof(info->loaded_from) - 1] = '\0';

    return 0;
}

void model_info_free(ModelInfo *info)
{
    if (!info) {
        return;
    }

    perceptron_free(&info->perceptron);
    scaler_free(&info->scaler);
    info->loaded_from[0] = '\0';
}

void model_print_info(const ModelInfo *info)
{
    if (!info || !info->perceptron) {
        printf("Model Information\n");
        printf("-------------------------\n");
        printf("No model loaded.\n");
        return;
    }

    const Perceptron *p = info->perceptron;

    printf("Model Information\n");
    printf("-------------------------\n");
    printf("Algorithm     : Single-Layer Perceptron\n");
    printf("Version       : %d\n", MODEL_FORMAT_VERSION);
    printf("Features      : %zu\n", p->feature_count);
    printf("Learning Rate : %.3f\n", p->learning_rate);
    printf("Epochs        : %zu\n", p->epochs);
    printf("Bias          : %.6f\n", p->bias);
    printf("Weights       : ");
    for (size_t i = 0; i < p->feature_count; i++) {
        if (i > 0) printf(", ");
        printf("%.6f", p->weights[i]);
    }
    printf("\n");

    if (info->loaded_from[0] != '\0') {
        printf("Source        : %s\n", info->loaded_from);
    }
}
