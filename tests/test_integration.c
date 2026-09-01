/**
 * test_integration.c - End-to-end integration and model-format tests
 *
 * Exercises the full pipeline:
 *   train -> save -> exit -> load -> predict -> compare results
 *
 * Also verifies model-format version rejection, truncation handling,
 * and memory-safety clean paths.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"
#include "model_io.h"
#include "config.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_BEGIN(name) \
    do { printf("  TEST: %-44s ", name); } while (0)

#define TEST_PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while (0)

#define TEST_FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while (0)

static const char *MODEL_PATH = "build/test_integration_model.bin";

/* Normalize a single row with a given scaler. */
static void normalize(const float *raw, float *out, const Scaler *s, size_t n)
{
    for (size_t f = 0; f < n; f++) {
        float range = s->max_values[f] - s->min_values[f];
        out[f] = (range < 1e-9f) ? 0.0f : (raw[f] - s->min_values[f]) / range;
    }
}

/* ------------------------------------------------------------------ */
/*  Integration: train -> save -> load -> predict -> compare            */
/* ------------------------------------------------------------------ */

static void test_integration_roundtrip(void)
{
    TEST_BEGIN("Integration: train->save->load->predict round-trip");

    /* Load dataset */
    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) { TEST_FAIL("dataset_load_csv failed"); return; }

    /* Split */
    Dataset *train = NULL, *test = NULL;
    if (dataset_split(ds, &train, &test, DEFAULT_TRAIN_RATIO,
                      DEFAULT_RANDOM_SEED) != 0) {
        TEST_FAIL("dataset_split failed"); dataset_free(&ds); return;
    }

    /* Fit + transform */
    Scaler scaler = {0};
    if (scaler_fit(&scaler, train) != 0) {
        TEST_FAIL("scaler_fit failed"); dataset_free(&train); dataset_free(&test);
        dataset_free(&ds); return;
    }
    scaler_transform(&scaler, train);
    scaler_transform(&scaler, test);

    /* Train */
    Perceptron *model = perceptron_init(train->num_features,
                                        DEFAULT_LEARNING_RATE,
                                        (size_t)DEFAULT_MAX_EPOCHS);
    if (!model) { TEST_FAIL("perceptron_init failed"); goto cleanup; }

    int epochs = perceptron_train(model, train);
    if (epochs < 1) { TEST_FAIL("perceptron_train failed"); goto cleanup; }

    /* Save */
    if (model_save(MODEL_PATH, model, &scaler) != 0) {
        TEST_FAIL("model_save failed"); goto cleanup;
    }

    /* Load */
    Perceptron *loaded = NULL;
    Scaler loaded_scaler = {0};
    if (model_load(MODEL_PATH, &loaded, &loaded_scaler) != 0) {
        TEST_FAIL("model_load failed"); goto cleanup;
    }

    /* Compare weights, bias, scaler (float32 round-trip is exact). */
    if (loaded->feature_count != model->feature_count) {
        TEST_FAIL("feature_count mismatch"); goto cleanup_loaded;
    }
    if (fabsf(loaded->bias - model->bias) > 0.0f) {
        TEST_FAIL("bias mismatch after round-trip"); goto cleanup_loaded;
    }
    if (fabsf(loaded->learning_rate - model->learning_rate) > 0.0f) {
        TEST_FAIL("learning_rate mismatch after round-trip"); goto cleanup_loaded;
    }
    for (size_t f = 0; f < model->feature_count; f++) {
        if (fabsf(loaded->weights[f] - model->weights[f]) > 0.0f) {
            TEST_FAIL("weight mismatch after round-trip"); goto cleanup_loaded;
        }
    }
    for (size_t f = 0; f < scaler.num_features; f++) {
        if (fabsf(loaded_scaler.min_values[f] - scaler.min_values[f]) > 0.0f ||
            fabsf(loaded_scaler.max_values[f] - scaler.max_values[f]) > 0.0f) {
            TEST_FAIL("scaler mismatch after round-trip"); goto cleanup_loaded;
        }
    }

    /* Predict on every original row with both models; results must match. */
    size_t nf = ds->num_features;
    float *raw = malloc(nf * sizeof(float));
    float *norm1 = malloc(nf * sizeof(float));
    float *norm2 = malloc(nf * sizeof(float));
    if (!raw || !norm1 || !norm2) {
        TEST_FAIL("allocation failed"); free(raw); free(norm1); free(norm2);
        goto cleanup_loaded;
    }

    for (size_t i = 0; i < ds->num_samples; i++) {
        for (size_t f = 0; f < nf; f++) raw[f] = ds->samples[i].features[f];

        normalize(raw, norm1, &scaler, nf);
        normalize(raw, norm2, &loaded_scaler, nf);

        int p1 = perceptron_predict(model, norm1);
        int p2 = perceptron_predict(loaded, norm2);
        if (p1 != p2) {
            TEST_FAIL("Prediction mismatch between original and loaded model");
            free(raw); free(norm1); free(norm2);
            goto cleanup_loaded;
        }
    }

    free(raw); free(norm1); free(norm2);
    perceptron_free(&loaded);
    scaler_free(&loaded_scaler);

    /* Cleanup successfully; free original model set. */
    perceptron_free(&model);
    scaler_free(&scaler);
    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);

    TEST_PASS();
    return;

cleanup_loaded:
    perceptron_free(&loaded);
    scaler_free(&loaded_scaler);
cleanup:
    perceptron_free(&model);
    scaler_free(&scaler);
    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);
}

/* ------------------------------------------------------------------ */
/*  Version mismatch                                                    */
/* ------------------------------------------------------------------ */

static int write_header_with_version(const char *path, uint32_t version,
                                     uint32_t feature_count)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    ModelFileHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, MODEL_MAGIC, MODEL_MAGIC_LEN);
    header.format_version = version;
    header.feature_count  = feature_count;
    header.learning_rate  = 0.1f;
    header.epochs         = 10;
    header.bias           = 0.0f;
    fwrite(&header, sizeof(header), 1, fp);
    fclose(fp);
    return 0;
}

static void test_version_mismatch(void)
{
    TEST_BEGIN("Model format version mismatch rejected");

    const char *path = "build/test_bad_version.bin";
    if (write_header_with_version(path, MODEL_FORMAT_VERSION + 1, 3) != 0) {
        TEST_FAIL("could not write fixture"); return;
    }

    Perceptron *loaded = NULL;
    Scaler scaler = {0};

    /* The load must fail cleanly and leave outputs in a safe state. */
    if (model_load(path, &loaded, &scaler) != -1) {
        TEST_FAIL("model_load should reject unsupported version");
        perceptron_free(&loaded);
        scaler_free(&scaler);
        remove(path);
        return;
    }
    if (loaded != NULL) {
        TEST_FAIL("model_out must be NULL after failed load");
        perceptron_free(&loaded);
        scaler_free(&scaler);
        remove(path);
        return;
    }

    /* scaler_out must be a zeroed/empty scaler (no dangling pointers). */
    if (scaler.min_values != NULL || scaler.max_values != NULL ||
        scaler.num_features != 0) {
        TEST_FAIL("scaler_out must be empty after failed load");
        scaler_free(&scaler);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Corrupted magic                                                    */
/* ------------------------------------------------------------------ */

static void test_corrupted_magic(void)
{
    TEST_BEGIN("Corrupted magic identifier rejected");

    const char *path = "build/test_bad_magic2.bin";
    FILE *fp = fopen(path, "wb");
    if (!fp) { TEST_FAIL("could not create fixture"); return; }
    fwrite("XXXXXXXX", 1, 8, fp);
    fclose(fp);

    Perceptron *loaded = NULL;
    Scaler scaler = {0};
    if (model_load(path, &loaded, &scaler) != -1) {
        TEST_FAIL("model_load should reject bad magic");
        perceptron_free(&loaded); scaler_free(&scaler); remove(path); return;
    }
    if (loaded != NULL) { TEST_FAIL("model_out must be NULL"); }

    scaler_free(&scaler);
    remove(path);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Truncated file                                                     */
/* ------------------------------------------------------------------ */

static void test_truncated_file(void)
{
    TEST_BEGIN("Truncated model file rejected");

    const char *path = "build/test_truncated.bin";

    /* Write a valid header claiming 3 features, but no weight data. */
    FILE *fp = fopen(path, "wb");
    if (!fp) { TEST_FAIL("could not create fixture"); return; }

    ModelFileHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, MODEL_MAGIC, MODEL_MAGIC_LEN);
    header.format_version = MODEL_FORMAT_VERSION;
    header.feature_count  = 3;
    header.learning_rate  = 0.1f;
    header.epochs         = 10;
    header.bias           = 0.0f;
    fwrite(&header, sizeof(header), 1, fp);
    fclose(fp);

    Perceptron *loaded = NULL;
    Scaler scaler = {0};
    if (model_load(path, &loaded, &scaler) != -1) {
        TEST_FAIL("model_load should reject truncated file");
        perceptron_free(&loaded); scaler_free(&scaler); remove(path); return;
    }
    if (loaded != NULL) { TEST_FAIL("model_out must be NULL"); }

    scaler_free(&scaler);
    remove(path);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Empty / NULL safety                                                */
/* ------------------------------------------------------------------ */

static void test_null_args(void)
{
    TEST_BEGIN("model_load NULL args rejected");
    Perceptron *p = NULL;
    Scaler s = {0};
    if (model_load(NULL, &p, &s) != -1) { TEST_FAIL("NULL filepath"); return; }
    if (model_load("x", NULL, &s) != -1) { TEST_FAIL("NULL model_out"); return; }
    if (model_load("x", &p, NULL) != -1) { TEST_FAIL("NULL scaler_out"); return; }
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Integration Tests (Phase 5)\n");
    printf("========================================\n\n");

    test_integration_roundtrip();
    test_version_mismatch();
    test_corrupted_magic();
    test_truncated_file();
    test_null_args();

    /* Remove the round-trip model artifact. */
    remove(MODEL_PATH);

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
