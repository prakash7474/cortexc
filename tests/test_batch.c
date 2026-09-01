/**
 * test_batch.c - Tests for the batch prediction function
 *
 * Verifies that predict_batch:
 *  - skips a header row,
 *  - handles rows with and without an actual-label column,
 *  - writes a CSV with feature,actual,predicted columns,
 *  - returns the correct number of rows processed,
 *  - rejects NULL/empty inputs safely.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "perceptron.h"
#include "preprocessing.h"
#include "prediction.h"
#include "config.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_BEGIN(name) \
    do { printf("  TEST: %-44s ", name); } while (0)

#define TEST_PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while (0)

#define TEST_FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while (0)

#define INPUT_CSV      "build/batch_input.csv"
#define OUTPUT_CSV     "build/batch_output.csv"

/*
 * Build a 2-feature model/scaler where predictions are deterministic:
 * weights = (1, 0), bias = -0.5, scaler maps [0,1] -> identity.
 * Then prediction is 1 iff feature_0 >= 0.5.
 */
static int make_model(Perceptron **p_out, Scaler *scaler_out)
{
    Perceptron *p = perceptron_init(2, 0.1f, 10);
    if (!p) return -1;
    p->weights[0] = 1.0f;
    p->weights[1] = 0.0f;
    p->bias = -0.5f;
    p->epochs = 3;

    scaler_out->min_values = malloc(2 * sizeof(float));
    scaler_out->max_values = malloc(2 * sizeof(float));
    if (!scaler_out->min_values || !scaler_out->max_values) {
        free(scaler_out->min_values); free(scaler_out->max_values);
        perceptron_free(&p);
        return -1;
    }
    scaler_out->min_values[0] = 0.0f;
    scaler_out->min_values[1] = 0.0f;
    scaler_out->max_values[0] = 1.0f;
    scaler_out->max_values[1] = 1.0f;
    scaler_out->num_features = 2;

    *p_out = p;
    return 0;
}

/* Write a CSV with a header and a label column. */
static int write_input(void)
{
    FILE *f = fopen(INPUT_CSV, "w");
    if (!f) return -1;
    fprintf(f, "a,b,label\n");
    fprintf(f, "0.2,5,0\n");
    fprintf(f, "0.8,9,1\n");
    fprintf(f, "1.2,3,1\n");
    fprintf(f, "\n"); /* blank line should be skipped */
    fclose(f);
    return 0;
}

/* Read the output CSV: count data rows, capture first label, last prediction. */
static int count_output_rows(const char *path, int *first_label, int *last_pred)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[128];
    int rows = 0;
    *first_label = -1;
    *last_pred = -1;
    /* Skip header */
    if (fgets(line, sizeof(line), f)) {
        (void)line;
    }
    while (fgets(line, sizeof(line), f)) {
        /* Trim newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue; /* trailing blank line */

        double f0, f1;
        int label, pred;
        if (sscanf(line, "%lf,%lf,%d,%d", &f0, &f1, &label, &pred) == 4) {
            if (rows == 0) *first_label = label;
            *last_pred = pred;
            rows++;
        }
    }
    fclose(f);
    return rows;
}

static void test_batch_with_label(void)
{
    TEST_BEGIN("predict_batch with actual-label column");
    if (write_input() != 0) { TEST_FAIL("could not write fixture"); return; }

    Perceptron *p = NULL;
    Scaler s = {0};
    if (make_model(&p, &s) != 0) { TEST_FAIL("could not build model"); return; }

    int n = predict_batch(p, &s, INPUT_CSV, OUTPUT_CSV);
    perceptron_free(&p);
    scaler_free(&s);

    if (n != 3) { TEST_FAIL("Expected 3 rows processed"); return; }

    int first_label = -1, last_pred = -1;
    int rows = count_output_rows(OUTPUT_CSV, &first_label, &last_pred);
    if (rows != 3) { TEST_FAIL("Expected 3 output rows"); return; }
    if (first_label != 0) { TEST_FAIL("First actual label should be 0"); return; }
    if (last_pred != 1) { TEST_FAIL("Last predicted label should be 1"); return; }

    /* Verify header present. */
    FILE *f = fopen(OUTPUT_CSV, "r");
    if (f) {
        char hdr[128];
        if (fgets(hdr, sizeof(hdr), f)) {
            if (strstr(hdr, "actual") == NULL || strstr(hdr, "predicted") == NULL) {
                TEST_FAIL("Output header missing actual/predicted");
                fclose(f);
                return;
            }
        }
        fclose(f);
    }

    TEST_PASS();
}

static void test_batch_no_label(void)
{
    TEST_BEGIN("predict_batch without label column");
    FILE *f = fopen(INPUT_CSV, "w");
    if (!f) { TEST_FAIL("could not write fixture"); return; }
    fprintf(f, "0.2,5\n0.8,9\n-3,1\n");
    fclose(f);

    Perceptron *p = NULL;
    Scaler s = {0};
    if (make_model(&p, &s) != 0) { TEST_FAIL("could not build model"); return; }

    int n = predict_batch(p, &s, INPUT_CSV, OUTPUT_CSV);
    perceptron_free(&p);
    scaler_free(&s);

    if (n != 3) { TEST_FAIL("Expected 3 rows processed"); return; }

    /* For no-label input the "actual" column should be "?". */
    FILE *g = fopen(OUTPUT_CSV, "r");
    if (!g) { TEST_FAIL("output missing"); return; }
    char line[128];
    int found_q = 0;
    if (fgets(line, sizeof(line), g)) (void)line; /* header */
    while (fgets(line, sizeof(line), g)) {
        if (strstr(line, ",?,")) found_q = 1;
    }
    fclose(g);
    if (!found_q) { TEST_FAIL("Expected '?' actual column for label-less input"); return; }

    TEST_PASS();
}

static void test_batch_empty_file(void)
{
    TEST_BEGIN("predict_batch on empty file returns 0");
    FILE *f = fopen(INPUT_CSV, "w");
    if (!f) { TEST_FAIL("could not write fixture"); return; }
    fclose(f);

    Perceptron *p = NULL;
    Scaler s = {0};
    if (make_model(&p, &s) != 0) { TEST_FAIL("could not build model"); return; }

    int n = predict_batch(p, &s, INPUT_CSV, OUTPUT_CSV);
    perceptron_free(&p);
    scaler_free(&s);

    if (n != 0) { TEST_FAIL("Expected 0 rows for empty file"); return; }
    TEST_PASS();
}

static void test_batch_missing_file(void)
{
    TEST_BEGIN("predict_batch on missing file fails");
    Perceptron *p = NULL;
    Scaler s = {0};
    if (make_model(&p, &s) != 0) { TEST_FAIL("could not build model"); return; }

    int n = predict_batch(p, &s, "build/does_not_exist.csv", OUTPUT_CSV);
    perceptron_free(&p);
    scaler_free(&s);

    if (n != -1) { TEST_FAIL("Expected -1 for missing file"); return; }
    TEST_PASS();
}

static void test_batch_null_args(void)
{
    TEST_BEGIN("predict_batch NULL args rejected");
    Perceptron *p = NULL;
    Scaler s = {0};
    if (make_model(&p, &s) != 0) { TEST_FAIL("could not build model"); return; }

    if (predict_batch(NULL, &s, INPUT_CSV, OUTPUT_CSV) != -1) { TEST_FAIL("NULL model"); }
    if (predict_batch(p, NULL, INPUT_CSV, OUTPUT_CSV) != -1) { TEST_FAIL("NULL scaler"); }
    if (predict_batch(p, &s, NULL, OUTPUT_CSV) != -1) { TEST_FAIL("NULL csv"); }

    perceptron_free(&p);
    scaler_free(&s);
    TEST_PASS();
}

int main(void)
{
    printf("========================================\n");
    printf("  Batch Prediction Module Tests\n");
    printf("========================================\n\n");

    test_batch_with_label();
    test_batch_no_label();
    test_batch_empty_file();
    test_batch_missing_file();
    test_batch_null_args();

    remove(INPUT_CSV);
    remove(OUTPUT_CSV);

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
