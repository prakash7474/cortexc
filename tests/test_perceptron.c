/**
 * test_perceptron.c - Tests for perceptron module
 *
 * Tests creation, forward pass, prediction, training, convergence,
 * and memory management of the single-layer perceptron.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "perceptron.h"
#include "dataset.h"
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

/**
 * create_simple_dataset - Create a small dataset for testing.
 * Linearly separable: label = 1 if feature[0] > 5 else 0
 */
static Dataset *create_simple_dataset(void)
{
    Dataset *ds = calloc(1, sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_features = 1;
    ds->num_samples = 6;
    ds->capacity = 6;
    ds->label_count = 2;
    strcpy(ds->source_file, "test_simple");

    ds->samples = calloc(6, sizeof(Sample));
    if (!ds->samples) { free(ds); return NULL; }

    /* Values: 1, 3, 5, 7, 9, 11 -> labels: 0, 0, 0, 1, 1, 1 */
    float values[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    int labels[] = {0, 0, 0, 1, 1, 1};

    for (int i = 0; i < 6; i++) {
        ds->samples[i].features = malloc(sizeof(float));
        if (!ds->samples[i].features) {
            /* Cleanup on failure */
            for (int j = 0; j < i; j++) free(ds->samples[j].features);
            free(ds->samples);
            free(ds);
            return NULL;
        }
        ds->samples[i].features[0] = values[i];
        ds->samples[i].label = labels[i];
    }

    return ds;
}

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_perceptron_create(void)
{
    TEST_BEGIN("Perceptron creation");

    Perceptron *p = perceptron_create(3, 0.1f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    if (p->num_features != 3) {
        TEST_FAIL("num_features != 3");
        perceptron_free(&p);
        return;
    }

    if (fabsf(p->learning_rate - 0.1f) > 1e-6f) {
        TEST_FAIL("learning_rate != 0.1");
        perceptron_free(&p);
        return;
    }

    if (fabsf(p->bias) > 1e-6f) {
        TEST_FAIL("bias should start at 0.0");
        perceptron_free(&p);
        return;
    }

    if (!p->weights) {
        TEST_FAIL("weights is NULL");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

static void test_perceptron_create_invalid(void)
{
    TEST_BEGIN("Perceptron creation (invalid params)");

    /* Zero features */
    Perceptron *p1 = perceptron_create(0, 0.1f, 42);
    if (p1 != NULL) {
        TEST_FAIL("Should fail with 0 features");
        perceptron_free(&p1);
        return;
    }

    /* Zero learning rate */
    Perceptron *p2 = perceptron_create(3, 0.0f, 42);
    if (p2 != NULL) {
        TEST_FAIL("Should fail with lr=0");
        perceptron_free(&p2);
        return;
    }

    /* Negative learning rate */
    Perceptron *p3 = perceptron_create(3, -0.5f, 42);
    if (p3 != NULL) {
        TEST_FAIL("Should fail with negative lr");
        perceptron_free(&p3);
        return;
    }

    TEST_PASS();
}

static void test_perceptron_forward(void)
{
    TEST_BEGIN("Forward pass");

    Perceptron *p = perceptron_create(2, 0.1f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    /* Manually set weights and bias for deterministic test */
    p->weights[0] = 1.0f;
    p->weights[1] = 1.0f;
    p->bias = -1.5f;

    /* sum = 1*1 + 1*1 + (-1.5) = 0.5 >= 0 -> 1 */
    float features1[] = {1.0f, 1.0f};
    int out1 = perceptron_forward(p, features1);
    if (out1 != 1) {
        TEST_FAIL("Expected output 1 for [1,1]");
        perceptron_free(&p);
        return;
    }

    /* sum = 1*0.1 + 1*0.1 + (-1.5) = -1.3 < 0 -> 0 */
    float features2[] = {0.1f, 0.1f};
    int out2 = perceptron_forward(p, features2);
    if (out2 != 0) {
        TEST_FAIL("Expected output 0 for [0.1, 0.1]");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

static void test_perceptron_predict(void)
{
    TEST_BEGIN("Batch prediction");

    Dataset *ds = create_simple_dataset();
    if (!ds) {
        TEST_FAIL("Failed to create test dataset");
        return;
    }

    Perceptron *p = perceptron_create(1, 0.1f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    /* Set weights to always predict 1 */
    p->weights[0] = 1.0f;
    p->bias = 10.0f;

    int predictions[6] = {0};
    if (perceptron_predict(p, ds, predictions) != 0) {
        TEST_FAIL("predict failed");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    for (int i = 0; i < 6; i++) {
        if (predictions[i] != 1) {
            TEST_FAIL("Expected all predictions to be 1");
            perceptron_free(&p);
            dataset_free(&ds);
            return;
        }
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_perceptron_train_converges(void)
{
    TEST_BEGIN("Training converges on linear data");

    Dataset *ds = create_simple_dataset();
    if (!ds) {
        TEST_FAIL("Failed to create test dataset");
        return;
    }

    Perceptron *p = perceptron_create(1, 0.1f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    int epochs = perceptron_train(p, ds, 100, 42);
    if (epochs < 1) {
        TEST_FAIL("Training did not complete any epoch");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    /* Verify convergence: all training samples should be correct */
    int predictions[6] = {0};
    perceptron_predict(p, ds, predictions);

    for (int i = 0; i < 6; i++) {
        if (predictions[i] != ds->samples[i].label) {
            TEST_FAIL("Training did not converge");
            perceptron_free(&p);
            dataset_free(&ds);
            return;
        }
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_perceptron_train_students(void)
{
    TEST_BEGIN("Training on students.csv dataset");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load students.csv");
        return;
    }

    Perceptron *p = perceptron_create(ds->num_features, 0.01f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    int epochs = perceptron_train(p, ds, 1000, 42);
    if (epochs < 1) {
        TEST_FAIL("Training did not complete any epoch");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    /* Check training accuracy */
    int correct = 0;
    for (size_t i = 0; i < ds->num_samples; i++) {
        int pred = perceptron_forward(p, ds->samples[i].features);
        if (pred == ds->samples[i].label) {
            correct++;
        }
    }

    float accuracy = (float)correct / (float)ds->num_samples;

    /* After normalization + training, should get reasonable accuracy */
    if (accuracy < 0.5f) {
        TEST_FAIL("Accuracy below 50%");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_perceptron_free(void)
{
    TEST_BEGIN("Perceptron cleanup");

    Perceptron *p = perceptron_create(3, 0.1f, 42);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    perceptron_free(&p);

    if (p != NULL) {
        TEST_FAIL("Pointer not set to NULL after free");
        return;
    }

    TEST_PASS();
}

static void test_perceptron_free_null(void)
{
    TEST_BEGIN("Perceptron free NULL safety");

    /* Should not crash */
    perceptron_free(NULL);

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Perceptron Module Tests\n");
    printf("========================================\n\n");

    test_perceptron_create();
    test_perceptron_create_invalid();
    test_perceptron_forward();
    test_perceptron_predict();
    test_perceptron_train_converges();
    test_perceptron_train_students();
    test_perceptron_free();
    test_perceptron_free_null();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
