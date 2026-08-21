/**
 * test_perceptron.c - Tests for Phase 3: perceptron, prediction, evaluation
 *
 * Tests:
 * 1. Model initialization
 * 2. Weight allocation
 * 3. Prediction (forward pass)
 * 4. Training
 * 5. Weight updates
 * 6. Bias updates
 * 7. Early stopping
 * 8. Accuracy calculation
 * 9. Confusion matrix
 * 10. Model cleanup
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"
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
/*  Test 1: Model initialization                                       */
/* ------------------------------------------------------------------ */

static void test_perceptron_init(void)
{
    TEST_BEGIN("Perceptron initialization");

    Perceptron *p = perceptron_init(3, 0.1f, 100);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    if (p->feature_count != 3) {
        TEST_FAIL("feature_count != 3");
        perceptron_free(&p);
        return;
    }

    if (fabsf(p->learning_rate - 0.1f) > 1e-6f) {
        TEST_FAIL("learning_rate != 0.1");
        perceptron_free(&p);
        return;
    }

    if (p->epochs != 100) {
        TEST_FAIL("epochs != 100");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

static void test_perceptron_init_invalid(void)
{
    TEST_BEGIN("Perceptron init (invalid params)");

    /* Zero features */
    Perceptron *p1 = perceptron_init(0, 0.1f, 100);
    if (p1 != NULL) {
        TEST_FAIL("Should fail with 0 features");
        perceptron_free(&p1);
        return;
    }

    /* Zero learning rate */
    Perceptron *p2 = perceptron_init(3, 0.0f, 100);
    if (p2 != NULL) {
        TEST_FAIL("Should fail with lr=0");
        perceptron_free(&p2);
        return;
    }

    /* Negative learning rate */
    Perceptron *p3 = perceptron_init(3, -0.5f, 100);
    if (p3 != NULL) {
        TEST_FAIL("Should fail with negative lr");
        perceptron_free(&p3);
        return;
    }

    /* Zero epochs */
    Perceptron *p4 = perceptron_init(3, 0.1f, 0);
    if (p4 != NULL) {
        TEST_FAIL("Should fail with 0 epochs");
        perceptron_free(&p4);
        return;
    }

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 2: Weight allocation                                          */
/* ------------------------------------------------------------------ */

static void test_weight_allocation(void)
{
    TEST_BEGIN("Weight allocation");

    Perceptron *p = perceptron_init(5, 0.1f, 100);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    if (!p->weights) {
        TEST_FAIL("weights is NULL");
        perceptron_free(&p);
        return;
    }

    /* Verify all weights start at 0 */
    for (size_t i = 0; i < p->feature_count; i++) {
        if (fabsf(p->weights[i]) > 1e-9f) {
            TEST_FAIL("Weight not initialized to 0");
            perceptron_free(&p);
            return;
        }
    }

    /* Verify bias starts at 0 */
    if (fabsf(p->bias) > 1e-9f) {
        TEST_FAIL("Bias not initialized to 0");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 3: Prediction (forward pass)                                  */
/* ------------------------------------------------------------------ */

static void test_perceptron_forward(void)
{
    TEST_BEGIN("Forward pass (prediction)");

    Perceptron *p = perceptron_init(2, 0.1f, 100);
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

    /* Test perceptron_predict alias */
    int pred = perceptron_predict(p, features1);
    if (pred != 1) {
        TEST_FAIL("perceptron_predict failed");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 4: Training                                                   */
/* ------------------------------------------------------------------ */

static void test_perceptron_train_converges(void)
{
    TEST_BEGIN("Training converges on linear data");

    Dataset *ds = create_simple_dataset();
    if (!ds) {
        TEST_FAIL("Failed to create test dataset");
        return;
    }

    Perceptron *p = perceptron_init(1, 0.1f, 100);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    int epochs = perceptron_train(p, ds);
    if (epochs < 1) {
        TEST_FAIL("Training did not complete any epoch");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    /* Verify convergence: all training samples should be correct */
    int predictions[6] = {0};
    for (int i = 0; i < 6; i++) {
        predictions[i] = perceptron_forward(p, ds->samples[i].features);
    }

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

    Perceptron *p = perceptron_init(ds->num_features, 0.01f, 1000);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    int epochs = perceptron_train(p, ds);
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

/* ------------------------------------------------------------------ */
/*  Test 5: Weight updates                                             */
/* ------------------------------------------------------------------ */

static void test_weight_updates(void)
{
    TEST_BEGIN("Weight updates during training");

    Perceptron *p = perceptron_init(1, 0.1f, 1);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    /* Record initial weight */
    float initial_weight = p->weights[0];

    /* Create a simple dataset: 1 sample, label=1, feature=2.0 */
    Dataset *ds = calloc(1, sizeof(Dataset));
    ds->num_features = 1;
    ds->num_samples = 1;
    ds->capacity = 1;
    ds->label_count = 2;
    strcpy(ds->source_file, "test");
    ds->samples = calloc(1, sizeof(Sample));
    ds->samples[0].features = malloc(sizeof(float));
    ds->samples[0].features[0] = 2.0f;
    ds->samples[0].label = 1;

    /* With all zeros, prediction = step(0) = 1, so error = 1 - 1 = 0 */
    /* No update happens. Let's set label=0 to force an update */
    ds->samples[0].label = 0;

    /* Now prediction=step(0)=1, error=0-1=-1 */
    /* weight += 0.1 * (-1) * 2.0 = -0.2 */
    perceptron_train(p, ds);

    if (fabsf(p->weights[0] - (initial_weight + 0.1f * (-1.0f) * 2.0f)) > 1e-6f) {
        TEST_FAIL("Weight not updated correctly");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 6: Bias updates                                               */
/* ------------------------------------------------------------------ */

static void test_bias_updates(void)
{
    TEST_BEGIN("Bias updates during training");

    Perceptron *p = perceptron_init(1, 0.1f, 1);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    float initial_bias = p->bias;

    /* Create dataset: 1 sample, feature=0 (so only bias matters) */
    Dataset *ds = calloc(1, sizeof(Dataset));
    ds->num_features = 1;
    ds->num_samples = 1;
    ds->capacity = 1;
    ds->label_count = 2;
    strcpy(ds->source_file, "test");
    ds->samples = calloc(1, sizeof(Sample));
    ds->samples[0].features = malloc(sizeof(float));
    ds->samples[0].features[0] = 0.0f;
    ds->samples[0].label = 0;

    /* prediction = step(0) = 1, error = 0 - 1 = -1 */
    /* bias += 0.1 * (-1) = -0.1 */
    perceptron_train(p, ds);

    if (fabsf(p->bias - (initial_bias + 0.1f * (-1.0f))) > 1e-6f) {
        TEST_FAIL("Bias not updated correctly");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 7: Early stopping                                             */
/* ------------------------------------------------------------------ */

static void test_early_stopping(void)
{
    TEST_BEGIN("Early stopping on linearly separable data");

    Dataset *ds = create_simple_dataset();
    if (!ds) {
        TEST_FAIL("Failed to create test dataset");
        return;
    }

    /* Set max epochs very high */
    Perceptron *p = perceptron_init(1, 0.1f, 10000);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    int epochs = perceptron_train(p, ds);

    /* Should converge well before 10000 epochs */
    if (epochs >= 10000) {
        TEST_FAIL("Did not stop early (expected convergence)");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    if (epochs < 1) {
        TEST_FAIL("No epochs completed");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    /* Verify all correct after early stop */
    for (size_t i = 0; i < ds->num_samples; i++) {
        int pred = perceptron_forward(p, ds->samples[i].features);
        if (pred != ds->samples[i].label) {
            TEST_FAIL("Early stop but not all correct");
            perceptron_free(&p);
            dataset_free(&ds);
            return;
        }
    }

    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 8: Accuracy calculation                                       */
/* ------------------------------------------------------------------ */

static void test_accuracy_calculation(void)
{
    TEST_BEGIN("Accuracy calculation");

    int predictions[] = {1, 0, 1, 1, 0};
    int actuals[]     = {1, 0, 0, 1, 1};

    /* 3 correct out of 5 = 60% */
    double accuracy = calculate_accuracy(predictions, actuals, 5);

    if (fabs(accuracy - 60.0) > 0.01) {
        TEST_FAIL("Accuracy calculation wrong");
        return;
    }

    /* Test perfect accuracy */
    int preds2[] = {1, 0, 1};
    int acts2[]  = {1, 0, 1};
    double acc2 = calculate_accuracy(preds2, acts2, 3);
    if (fabs(acc2 - 100.0) > 0.01) {
        TEST_FAIL("Perfect accuracy should be 100%");
        return;
    }

    /* Test zero accuracy */
    int preds3[] = {0, 1};
    int acts3[]  = {1, 0};
    double acc3 = calculate_accuracy(preds3, acts3, 2);
    if (fabs(acc3 - 0.0) > 0.01) {
        TEST_FAIL("Zero accuracy should be 0%");
        return;
    }

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 9: Confusion matrix                                           */
/* ------------------------------------------------------------------ */

static void test_confusion_matrix(void)
{
    TEST_BEGIN("Confusion matrix");

    /* predictions: 1, 0, 1, 1, 0
     * actuals:     1, 0, 0, 1, 1
     *
     * TP=2 (pred=1,act=1), TN=1 (pred=0,act=0)
     * FP=1 (pred=1,act=0), FN=1 (pred=0,act=1)
     */
    int predictions[] = {1, 0, 1, 1, 0};
    int actuals[]     = {1, 0, 0, 1, 1};

    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(predictions, actuals, 5, &cm) != 0) {
        TEST_FAIL("calculate_confusion_matrix failed");
        return;
    }

    if (cm.tp != 2) { TEST_FAIL("TP != 2"); return; }
    if (cm.tn != 1) { TEST_FAIL("TN != 1"); return; }
    if (cm.fp != 1) { TEST_FAIL("FP != 1"); return; }
    if (cm.fn != 1) { TEST_FAIL("FN != 1"); return; }

    /* Verify TP + TN + FP + FN = total */
    if (cm.tp + cm.tn + cm.fp + cm.fn != 5) {
        TEST_FAIL("Sum of matrix elements != total");
        return;
    }

    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Test 10: Model cleanup                                             */
/* ------------------------------------------------------------------ */

static void test_perceptron_free(void)
{
    TEST_BEGIN("Perceptron cleanup");

    Perceptron *p = perceptron_init(3, 0.1f, 100);
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
/*  Test: Prediction module                                            */
/* ------------------------------------------------------------------ */

static void test_predict_sample(void)
{
    TEST_BEGIN("predict_sample function");

    Perceptron *p = perceptron_init(2, 0.1f, 100);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        return;
    }

    p->weights[0] = 1.0f;
    p->weights[1] = 1.0f;
    p->bias = -1.5f;

    float features[] = {1.0f, 1.0f};
    int prediction = -1;

    if (predict_sample(p, features, &prediction) != 0) {
        TEST_FAIL("predict_sample returned error");
        perceptron_free(&p);
        return;
    }

    if (prediction != 1) {
        TEST_FAIL("Expected prediction 1");
        perceptron_free(&p);
        return;
    }

    perceptron_free(&p);
    TEST_PASS();
}

static void test_predict_dataset(void)
{
    TEST_BEGIN("predict_dataset function");

    Dataset *ds = create_simple_dataset();
    if (!ds) {
        TEST_FAIL("Failed to create test dataset");
        return;
    }

    Perceptron *p = perceptron_init(1, 0.1f, 100);
    if (!p) {
        TEST_FAIL("Failed to create perceptron");
        dataset_free(&ds);
        return;
    }

    /* Set weights to always predict 1 */
    p->weights[0] = 1.0f;
    p->bias = 10.0f;

    int *predictions = predict_dataset(p, ds);
    if (!predictions) {
        TEST_FAIL("predict_dataset returned NULL");
        perceptron_free(&p);
        dataset_free(&ds);
        return;
    }

    for (int i = 0; i < 6; i++) {
        if (predictions[i] != 1) {
            TEST_FAIL("Expected all predictions to be 1");
            free(predictions);
            perceptron_free(&p);
            dataset_free(&ds);
            return;
        }
    }

    free(predictions);
    perceptron_free(&p);
    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Perceptron Module Tests (Phase 3)\n");
    printf("========================================\n\n");

    test_perceptron_init();
    test_perceptron_init_invalid();
    test_weight_allocation();
    test_perceptron_forward();
    test_perceptron_train_converges();
    test_perceptron_train_students();
    test_weight_updates();
    test_bias_updates();
    test_early_stopping();
    test_accuracy_calculation();
    test_confusion_matrix();
    test_perceptron_free();
    test_perceptron_free_null();
    test_predict_sample();
    test_predict_dataset();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
