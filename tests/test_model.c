/**
 * test_model.c - Tests for model I/O module
 *
 * Tests:
 * 1. model_save with valid model and scaler
 * 2. model_load round-trip (save -> load -> verify)
 * 3. model_load with non-existent file
 * 4. model_load with corrupted magic identifier
 * 5. model_load_info convenience wrapper
 * 6. model_info_free safety
 * 7. model_print_info display
 * 8. Null argument safety for all functions
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model_io.h"
#include "perceptron.h"
#include "preprocessing.h"
#include "config.h"

/* ------------------------------------------------------------------ */
/*  Test helpers                                                       */
/* ------------------------------------------------------------------ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_BEGIN(name) \
    do { printf("  TEST: %-40s ", name); } while (0)

#define TEST_PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while (0)

#define TEST_FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while (0)

/** Path for temporary test model files */
static const char *TEST_MODEL_PATH = "build/test_model.bin";

/* ------------------------------------------------------------------ */
/*  Helper: create a trained model and scaler for testing              */
/* ------------------------------------------------------------------ */

static int create_test_model(Perceptron **model_out, Scaler *scaler_out)
{
    /* Create a perceptron */
    Perceptron *p = perceptron_init(3, 0.1f, 50);
    if (!p) return -1;

    /* Set known weights and bias */
    p->weights[0] = 1.234f;
    p->weights[1] = -0.567f;
    p->weights[2] = 0.891f;
    p->bias = -0.42f;
    p->epochs = 25;

    /* Create a scaler with known min/max */
    scaler_out->min_values = malloc(3 * sizeof(float));
    scaler_out->max_values = malloc(3 * sizeof(float));
    if (!scaler_out->min_values || !scaler_out->max_values) {
        perceptron_free(&p);
        free(scaler_out->min_values);
        free(scaler_out->max_values);
        return -1;
    }

    scaler_out->min_values[0] = 1.0f;
    scaler_out->min_values[1] = 20.0f;
    scaler_out->min_values[2] = 3.0f;

    scaler_out->max_values[0] = 10.0f;
    scaler_out->max_values[1] = 100.0f;
    scaler_out->max_values[2] = 10.0f;

    scaler_out->num_features = 3;

    *model_out = p;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Test 1: Save model                                                 */
/* ------------------------------------------------------------------ */

static void test_model_save(void)
{
    TEST_BEGIN("model_save with valid data");

    Perceptron *model = NULL;
    Scaler scaler = {0};
    if (create_test_model(&model, &scaler) != 0) {
        TEST_FAIL("Failed to create test model");
        return;
    }

    if (model_save(TEST_MODEL_PATH, model, &scaler) != 0) {
        TEST_FAIL("model_save returned error");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Verify file was created */
    FILE *fp = fopen(TEST_MODEL_PATH, "rb");
    if (!fp) {
        TEST_FAIL("Model file was not created");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }
    fclose(fp);

    perceptron_free(&model);
    scaler_free(&scaler);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 2: Save and load round-trip                                   */
/* ------------------------------------------------------------------ */

static void test_model_save_load_roundtrip(void)
{
    TEST_BEGIN("model_save/load round-trip");

    Perceptron *model = NULL;
    Scaler scaler = {0};
    if (create_test_model(&model, &scaler) != 0) {
        TEST_FAIL("Failed to create test model");
        return;
    }

    /* Save */
    if (model_save(TEST_MODEL_PATH, model, &scaler) != 0) {
        TEST_FAIL("model_save failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Load */
    Perceptron *loaded = NULL;
    Scaler loaded_scaler = {0};
    if (model_load(TEST_MODEL_PATH, &loaded, &loaded_scaler) != 0) {
        TEST_FAIL("model_load failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Verify perceptron fields */
    if (loaded->feature_count != model->feature_count) {
        TEST_FAIL("feature_count mismatch");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    if (fabsf(loaded->learning_rate - model->learning_rate) > 1e-6f) {
        TEST_FAIL("learning_rate mismatch");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    if (loaded->epochs != model->epochs) {
        TEST_FAIL("epochs mismatch");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    if (fabsf(loaded->bias - model->bias) > 1e-6f) {
        TEST_FAIL("bias mismatch");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    /* Verify weights */
    for (size_t i = 0; i < model->feature_count; i++) {
        if (fabsf(loaded->weights[i] - model->weights[i]) > 1e-6f) {
            TEST_FAIL("weight mismatch");
            perceptron_free(&model);
            scaler_free(&scaler);
            perceptron_free(&loaded);
            scaler_free(&loaded_scaler);
            return;
        }
    }

    /* Verify scaler */
    if (loaded_scaler.num_features != scaler.num_features) {
        TEST_FAIL("scaler num_features mismatch");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    for (size_t i = 0; i < scaler.num_features; i++) {
        if (fabsf(loaded_scaler.min_values[i] - scaler.min_values[i]) > 1e-6f) {
            TEST_FAIL("scaler min mismatch");
            perceptron_free(&model);
            scaler_free(&scaler);
            perceptron_free(&loaded);
            scaler_free(&loaded_scaler);
            return;
        }
        if (fabsf(loaded_scaler.max_values[i] - scaler.max_values[i]) > 1e-6f) {
            TEST_FAIL("scaler max mismatch");
            perceptron_free(&model);
            scaler_free(&scaler);
            perceptron_free(&loaded);
            scaler_free(&loaded_scaler);
            return;
        }
    }

    /* Verify loaded model produces same predictions */
    float test_features[] = {5.0f, 50.0f, 6.0f};
    int orig_pred = perceptron_forward(model, test_features);
    int load_pred = perceptron_forward(loaded, test_features);

    if (orig_pred != load_pred) {
        TEST_FAIL("Prediction mismatch after round-trip");
        perceptron_free(&model);
        scaler_free(&scaler);
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    perceptron_free(&model);
    scaler_free(&scaler);
    perceptron_free(&loaded);
    scaler_free(&loaded_scaler);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 3: Load from non-existent file                                */
/* ------------------------------------------------------------------ */

static void test_model_load_nonexistent(void)
{
    TEST_BEGIN("model_load with non-existent file");

    Perceptron *loaded = NULL;
    Scaler loaded_scaler = {0};
    if (model_load("build/does_not_exist.bin", &loaded, &loaded_scaler) == 0) {
        TEST_FAIL("Should have returned error for non-existent file");
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    if (loaded != NULL) {
        TEST_FAIL("Model should be NULL on failure");
        return;
    }

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 4: Load with corrupted magic                                  */
/* ------------------------------------------------------------------ */

static void test_model_load_corrupted_magic(void)
{
    TEST_BEGIN("model_load with corrupted magic");

    /* Write a file with wrong magic */
    const char *bad_path = "build/test_bad_magic.bin";
    FILE *fp = fopen(bad_path, "wb");
    if (!fp) {
        TEST_FAIL("Failed to create bad magic file");
        return;
    }

    /* Write wrong magic + rest of header */
    char bad_magic[MODEL_MAGIC_LEN];
    memset(bad_magic, 'X', MODEL_MAGIC_LEN);
    fwrite(bad_magic, 1, MODEL_MAGIC_LEN, fp);

    /* Fill rest of header with zeros */
    uint32_t zero32 = 0;
    float zero_float = 0.0f;
    fwrite(&zero32, sizeof(uint32_t), 1, fp);  /* format_version */
    fwrite(&zero32, sizeof(uint32_t), 1, fp);  /* feature_count */
    fwrite(&zero_float, sizeof(float), 1, fp); /* learning_rate */
    fwrite(&zero32, sizeof(uint32_t), 1, fp);  /* epochs */
    fwrite(&zero_float, sizeof(float), 1, fp); /* bias */
    fclose(fp);

    Perceptron *loaded = NULL;
    Scaler loaded_scaler = {0};
    if (model_load(bad_path, &loaded, &loaded_scaler) == 0) {
        TEST_FAIL("Should have returned error for corrupted magic");
        perceptron_free(&loaded);
        scaler_free(&loaded_scaler);
        return;
    }

    if (loaded != NULL) {
        TEST_FAIL("Model should be NULL on failure");
        return;
    }

    /* Clean up temp file */
    remove(bad_path);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 5: model_load_info convenience wrapper                        */
/* ------------------------------------------------------------------ */

static void test_model_load_info(void)
{
    TEST_BEGIN("model_load_info wrapper");

    Perceptron *model = NULL;
    Scaler scaler = {0};
    if (create_test_model(&model, &scaler) != 0) {
        TEST_FAIL("Failed to create test model");
        return;
    }

    /* Save */
    if (model_save(TEST_MODEL_PATH, model, &scaler) != 0) {
        TEST_FAIL("model_save failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Load with info wrapper */
    ModelInfo info = {0};
    if (model_load_info(TEST_MODEL_PATH, &info) != 0) {
        TEST_FAIL("model_load_info failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Verify loaded_from path */
    if (strcmp(info.loaded_from, TEST_MODEL_PATH) != 0) {
        TEST_FAIL("loaded_from path mismatch");
        model_info_free(&info);
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Verify perceptron is loaded */
    if (!info.perceptron) {
        TEST_FAIL("perceptron is NULL in ModelInfo");
        model_info_free(&info);
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    if (info.perceptron->feature_count != model->feature_count) {
        TEST_FAIL("feature_count mismatch in ModelInfo");
        model_info_free(&info);
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    model_info_free(&info);
    perceptron_free(&model);
    scaler_free(&scaler);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 6: model_info_free safety                                     */
/* ------------------------------------------------------------------ */

static void test_model_info_free_safety(void)
{
    TEST_BEGIN("model_info_free safety");

    /* Should not crash on zeroed struct */
    ModelInfo info = {0};
    model_info_free(&info);

    /* Should not crash on NULL */
    model_info_free(NULL);

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 7: model_print_info display                                   */
/* ------------------------------------------------------------------ */

static void test_model_print_info(void)
{
    TEST_BEGIN("model_print_info display");

    Perceptron *model = NULL;
    Scaler scaler = {0};
    if (create_test_model(&model, &scaler) != 0) {
        TEST_FAIL("Failed to create test model");
        return;
    }

    if (model_save(TEST_MODEL_PATH, model, &scaler) != 0) {
        TEST_FAIL("model_save failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    ModelInfo info = {0};
    if (model_load_info(TEST_MODEL_PATH, &info) != 0) {
        TEST_FAIL("model_load_info failed");
        perceptron_free(&model);
        scaler_free(&scaler);
        return;
    }

    /* Print info (visual verification) */
    printf("\n");
    model_print_info(&info);
    printf("\n");

    /* Print empty model info */
    ModelInfo empty = {0};
    model_print_info(&empty);

    model_info_free(&info);
    perceptron_free(&model);
    scaler_free(&scaler);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 8: Null argument safety                                       */
/* ------------------------------------------------------------------ */

static void test_null_arguments(void)
{
    TEST_BEGIN("Null argument safety");

    Perceptron *p = NULL;
    Scaler s = {0};

    /* model_save with NULL args */
    if (model_save(NULL, p, &s) != -1) {
        TEST_FAIL("model_save should fail with NULL filepath");
        return;
    }

    if (model_save(TEST_MODEL_PATH, NULL, &s) != -1) {
        TEST_FAIL("model_save should fail with NULL model");
        return;
    }

    /* model_load with NULL args */
    if (model_load(NULL, &p, &s) != -1) {
        TEST_FAIL("model_load should fail with NULL filepath");
        return;
    }

    /* model_load_info with NULL args */
    ModelInfo info = {0};
    if (model_load_info(NULL, &info) != -1) {
        TEST_FAIL("model_load_info should fail with NULL filepath");
        return;
    }

    /* model_print_info with NULL (should not crash) */
    model_print_info(NULL);

    /* model_info_free with NULL (should not crash) */
    model_info_free(NULL);

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Model I/O Module Tests\n");
    printf("========================================\n\n");

    test_model_save();
    test_model_save_load_roundtrip();
    test_model_load_nonexistent();
    test_model_load_corrupted_magic();
    test_model_load_info();
    test_model_info_free_safety();
    test_model_print_info();
    test_null_arguments();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
