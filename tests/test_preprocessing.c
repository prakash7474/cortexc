/**
 * test_preprocessing.c - Tests for preprocessing module
 *
 * Tests Min-Max normalization, scaler fit/transform, constant features,
 * and data leakage prevention.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "dataset.h"
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

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < 1e-6f)

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_basic_normalization(void)
{
    TEST_BEGIN("Basic Min-Max normalization");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    Scaler scaler = {0};

    if (scaler_fit(&scaler, ds) != 0) {
        TEST_FAIL("scaler_fit failed");
        dataset_free(&ds);
        return;
    }

    if (scaler.num_features != ds->num_features) {
        TEST_FAIL("Feature count mismatch");
        scaler_free(&scaler);
        dataset_free(&ds);
        return;
    }

    if (scaler_transform(&scaler, ds) != 0) {
        TEST_FAIL("scaler_transform failed");
        scaler_free(&scaler);
        dataset_free(&ds);
        return;
    }

    /* After normalization, min should map to 0.0 and max should map to 1.0 */
    /* Check that all values are in [0, 1] range */
    for (size_t i = 0; i < ds->num_samples; i++) {
        for (size_t f = 0; f < ds->num_features; f++) {
            float v = ds->samples[i].features[f];
            if (v < -0.001f || v > 1.001f) {
                TEST_FAIL("Normalized value out of [0,1] range");
                scaler_free(&scaler);
                dataset_free(&ds);
                return;
            }
        }
    }

    scaler_free(&scaler);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_normalization_bounds(void)
{
    TEST_BEGIN("Normalization min=0, max=1");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    Scaler scaler = {0};
    scaler_fit(&scaler, ds);

    /* Find which sample has the min for feature 0 */
    scaler_transform(&scaler, ds);

    /* Find the sample that had the min value - it should be ~0.0 */
    int found_min = 0, found_max = 0;
    for (size_t i = 0; i < ds->num_samples; i++) {
        if (FLOAT_EQ(ds->samples[i].features[0], 0.0f)) found_min = 1;
        if (FLOAT_EQ(ds->samples[i].features[0], 1.0f)) found_max = 1;
    }

    scaler_free(&scaler);
    dataset_free(&ds);

    if (!found_min) {
        TEST_FAIL("No sample maps to 0.0 for feature 0");
        return;
    }
    if (!found_max) {
        TEST_FAIL("No sample maps to 1.0 for feature 0");
        return;
    }

    TEST_PASS();
}

static void test_constant_feature(void)
{
    TEST_BEGIN("Constant feature handling (no division by zero)");

    /* Create a custom CSV with a constant feature */
    char path[] = "test_constant.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "feat1,feat2,label\n");
        fprintf(f, "5.0,1.0,0\n");
        fprintf(f, "5.0,2.0,1\n");
        fprintf(f, "5.0,3.0,0\n");
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    remove(path);

    if (!ds) {
        TEST_FAIL("Failed to load custom dataset");
        return;
    }

    Scaler scaler = {0};
    if (scaler_fit(&scaler, ds) != 0) {
        TEST_FAIL("scaler_fit failed");
        dataset_free(&ds);
        return;
    }

    /* feat1 is constant (all 5.0), so min == max */
    if (fabsf(scaler.min_values[0] - scaler.max_values[0]) > 1e-9f) {
        TEST_FAIL("Expected min == max for constant feature");
        scaler_free(&scaler);
        dataset_free(&ds);
        return;
    }

    if (scaler_transform(&scaler, ds) != 0) {
        TEST_FAIL("scaler_transform failed (div by zero?)");
        scaler_free(&scaler);
        dataset_free(&ds);
        return;
    }

    /* Constant feature should be 0.0 after normalization */
    for (size_t i = 0; i < ds->num_samples; i++) {
        if (!FLOAT_EQ(ds->samples[i].features[0], 0.0f)) {
            TEST_FAIL("Constant feature should map to 0.0");
            scaler_free(&scaler);
            dataset_free(&ds);
            return;
        }
    }

    scaler_free(&scaler);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_scaler_free(void)
{
    TEST_BEGIN("Scaler cleanup");

    Scaler scaler = {0};
    scaler.min_values = malloc(3 * sizeof(float));
    scaler.max_values = malloc(3 * sizeof(float));
    scaler.num_features = 3;

    scaler_free(&scaler);

    if (scaler.min_values != NULL) {
        TEST_FAIL("min_values not freed");
        return;
    }
    if (scaler.max_values != NULL) {
        TEST_FAIL("max_values not freed");
        return;
    }
    if (scaler.num_features != 0) {
        TEST_FAIL("num_features not zeroed");
        return;
    }

    TEST_PASS();
}

static void test_scaler_free_null(void)
{
    TEST_BEGIN("Scaler free NULL safety");

    /* Should not crash */
    scaler_free(NULL);

    TEST_PASS();
}

static void test_data_leakage_prevention(void)
{
    TEST_BEGIN("Data leakage prevention (train-only fit)");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    Dataset *train = NULL;
    Dataset *test = NULL;

    if (dataset_split(ds, &train, &test, 0.8f, 42) != 0) {
        TEST_FAIL("Split failed");
        dataset_free(&ds);
        return;
    }

    /* Fit scaler ONLY on training data */
    Scaler scaler = {0};
    if (scaler_fit(&scaler, train) != 0) {
        TEST_FAIL("scaler_fit failed");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    /* Apply same scaler to both */
    scaler_transform(&scaler, train);
    scaler_transform(&scaler, test);

    /* Test data should still be in valid range */
    for (size_t i = 0; i < test->num_samples; i++) {
        for (size_t f = 0; f < test->num_features; f++) {
            float v = test->samples[i].features[f];
            if (v < -0.5f || v > 1.5f) {
                TEST_FAIL("Test data out of reasonable range after transform");
                scaler_free(&scaler);
                dataset_free(&train);
                dataset_free(&test);
                dataset_free(&ds);
                return;
            }
        }
    }

    scaler_free(&scaler);
    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_scaler_feature_mismatch(void)
{
    TEST_BEGIN("Scaler/dataset feature count mismatch");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    Scaler scaler = {0};
    scaler_fit(&scaler, ds);

    /* Tamper with feature count to simulate mismatch */
    size_t orig = scaler.num_features;
    scaler.num_features = 999;

    if (scaler_transform(&scaler, ds) == 0) {
        TEST_FAIL("Should fail on feature count mismatch");
        scaler.num_features = orig;
        scaler_free(&scaler);
        dataset_free(&ds);
        return;
    }

    scaler.num_features = orig;
    scaler_free(&scaler);
    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Preprocessing Module Tests\n");
    printf("========================================\n\n");

    test_basic_normalization();
    test_normalization_bounds();
    test_constant_feature();
    test_scaler_free();
    test_scaler_free_null();
    test_data_leakage_prevention();
    test_scaler_feature_mismatch();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
