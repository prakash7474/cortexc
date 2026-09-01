/**
 * test_dataset.c - Tests for dataset module
 *
 * Tests CSV loading, validation, memory management, and train/test split.
 * Uses assertions for verification.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
 * write_temp_file - Create a temporary CSV file for testing.
 * Returns 0 on success, -1 on error.
 */
/* static char *write_temp_file(const char *content) - unused for now */

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_valid_csv_load(void)
{
    TEST_BEGIN("Valid CSV loading");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load valid CSV");
        return;
    }

    if (ds->num_samples != 10) {
        TEST_FAIL("Expected 10 samples");
        dataset_free(&ds);
        return;
    }

    if (ds->num_features != 3) {
        TEST_FAIL("Expected 3 features");
        dataset_free(&ds);
        return;
    }

    /* Verify first row values */
    if (ds->samples[0].features[0] != 1.0f) {
        TEST_FAIL("First sample study_hours != 1.0");
        dataset_free(&ds);
        return;
    }
    if (ds->samples[0].label != 0) {
        TEST_FAIL("First sample label != 0");
        dataset_free(&ds);
        return;
    }

    /* Verify last row */
    if (ds->samples[9].label != 1) {
        TEST_FAIL("Last sample label != 1");
        dataset_free(&ds);
        return;
    }

    dataset_free(&ds);
    TEST_PASS();
}

static void test_missing_file(void)
{
    TEST_BEGIN("Missing file handling");

    Dataset *ds = dataset_load_csv("nonexistent_file.csv");
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for missing file");
        dataset_free(&ds);
        return;
    }

    TEST_PASS();
}

static void test_empty_file(void)
{
    TEST_BEGIN("Empty file handling");

    char path[] = "test_empty.csv";
    FILE *f = fopen(path, "w");
    if (f) fclose(f);

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for empty file");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_header_only(void)
{
    TEST_BEGIN("Header-only file (no data rows)");

    char path[] = "test_header_only.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "study_hours,attendance,label\n");
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for header-only file");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_malformed_row(void)
{
    TEST_BEGIN("Malformed row handling");

    char path[] = "test_malformed.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "feat1,feat2,label\n");
        fprintf(f, "1.0,2.0,0\n");
        fprintf(f, "bad,data,here,extra\n"); /* wrong column count */
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for malformed row");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_invalid_label(void)
{
    TEST_BEGIN("Invalid label handling");

    char path[] = "test_bad_label.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "feat1,feat2,label\n");
        fprintf(f, "1.0,2.0,0\n");
        fprintf(f, "3.0,4.0,5\n"); /* label 5 is invalid */
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for invalid label");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_dataset_cleanup(void)
{
    TEST_BEGIN("Dataset cleanup (free)");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    dataset_free(&ds);

    if (ds != NULL) {
        TEST_FAIL("Pointer not set to NULL after free");
        return;
    }

    TEST_PASS();
}

static void test_double_free_safety(void)
{
    TEST_BEGIN("Double free safety");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    dataset_free(&ds);
    /* Calling free again should be safe (pointer is NULL) */
    dataset_free(&ds);

    TEST_PASS();
}

static void test_null_free(void)
{
    TEST_BEGIN("NULL free safety");

    /* Should not crash */
    dataset_free(NULL);

    TEST_PASS();
}

static void test_train_test_split(void)
{
    TEST_BEGIN("Train/test split (80/20)");

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

    if (train->num_samples != 8) {
        TEST_FAIL("Expected 8 training samples");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    if (test->num_samples != 2) {
        TEST_FAIL("Expected 2 testing samples");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    if (train->num_samples + test->num_samples != ds->num_samples) {
        TEST_FAIL("Split counts don't add up");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    /* Verify original is unchanged */
    if (ds->num_samples != 10) {
        TEST_FAIL("Original dataset was modified");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_split_preserves_data(void)
{
    TEST_BEGIN("Split preserves all original data");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    /* Sum all labels in original */
    int orig_sum = 0;
    for (size_t i = 0; i < ds->num_samples; i++) {
        orig_sum += ds->samples[i].label;
    }

    Dataset *train = NULL;
    Dataset *test = NULL;

    if (dataset_split(ds, &train, &test, 0.8f, 42) != 0) {
        TEST_FAIL("Split failed");
        dataset_free(&ds);
        return;
    }

    /* Sum labels in split */
    int split_sum = 0;
    for (size_t i = 0; i < train->num_samples; i++) {
        split_sum += train->samples[i].label;
    }
    for (size_t i = 0; i < test->num_samples; i++) {
        split_sum += test->samples[i].label;
    }

    if (split_sum != orig_sum) {
        TEST_FAIL("Label sum mismatch after split");
        dataset_free(&train);
        dataset_free(&test);
        dataset_free(&ds);
        return;
    }

    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);
    TEST_PASS();
}

static void test_reproducible_split(void)
{
    TEST_BEGIN("Split reproducibility with same seed");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    Dataset *train1 = NULL, *test1 = NULL;
    Dataset *train2 = NULL, *test2 = NULL;

    dataset_split(ds, &train1, &test1, 0.8f, 42);
    dataset_split(ds, &train2, &test2, 0.8f, 42);

    int match = 1;
    for (size_t i = 0; i < train1->num_samples; i++) {
        if (train1->samples[i].label != train2->samples[i].label) {
            match = 0;
            break;
        }
    }

    dataset_free(&train1);
    dataset_free(&test1);
    dataset_free(&train2);
    dataset_free(&test2);
    dataset_free(&ds);

    if (!match) {
        TEST_FAIL("Splits not reproducible with same seed");
        return;
    }

    TEST_PASS();
}

static void test_memory_estimation(void)
{
    TEST_BEGIN("RAM memory estimation");

    Dataset *ds = dataset_load_csv("data/students.csv");
    if (!ds) {
        TEST_FAIL("Failed to load dataset");
        return;
    }

    size_t est = dataset_estimate_memory(ds);

    /* Rough lower bound: at least num_samples * sizeof(float) * num_features */
    size_t min_expected = ds->num_samples * ds->num_features * sizeof(float);

    if (est < min_expected) {
        TEST_FAIL("Estimate too low");
        dataset_free(&ds);
        return;
    }

    /* Rough upper bound: should not be absurdly large */
    if (est > 1024 * 1024) { /* 1 MB */
        TEST_FAIL("Estimate unreasonably high");
        dataset_free(&ds);
        return;
    }

    dataset_free(&ds);
    TEST_PASS();
}

static void test_invalid_number(void)
{
    TEST_BEGIN("Non-numeric feature handling");

    char path[] = "test_bad_number.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "feat1,feat2,label\n");
        fprintf(f, "1.0,2.0,0\n");
        fprintf(f, "3.0,abc,1\n"); /* non-numeric feature */
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for non-numeric feature");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_incorrect_column_count(void)
{
    TEST_BEGIN("Incorrect column count handling");

    char path[] = "test_wrong_cols.csv";
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "feat1,feat2,label\n");
        fprintf(f, "1.0,2.0,0\n");
        fprintf(f, "3.0,4.0\n"); /* missing label column */
        fclose(f);
    }

    Dataset *ds = dataset_load_csv(path);
    if (ds != NULL) {
        TEST_FAIL("Should return NULL for wrong column count");
        dataset_free(&ds);
        remove(path);
        return;
    }

    remove(path);
    TEST_PASS();
}

static void test_large_dataset(void)
{
    TEST_BEGIN("Large dataset handling (5000 rows)");

    char path[] = "test_large.csv";
    FILE *f = fopen(path, "w");
    if (!f) {
        TEST_FAIL("Could not create large fixture");
        return;
    }
    fprintf(f, "feat1,feat2,feat3,label\n");
    for (int i = 0; i < 5000; i++) {
        fprintf(f, "%d,%d,%d,%d\n", i % 10, i % 50, i % 8, (i % 2));
    }
    fclose(f);

    Dataset *ds = dataset_load_csv(path);
    remove(path);

    if (!ds) {
        TEST_FAIL("Failed to load large dataset");
        return;
    }

    if (ds->num_samples != 5000) {
        TEST_FAIL("Expected 5000 samples");
        dataset_free(&ds);
        return;
    }
    if (ds->num_features != 3) {
        TEST_FAIL("Expected 3 features");
        dataset_free(&ds);
        return;
    }

    /* Memory estimate must scale with samples*features. */
    size_t min_expected = (size_t)ds->num_samples * ds->num_features * sizeof(float);
    if (dataset_estimate_memory(ds) < min_expected) {
        TEST_FAIL("Memory estimate too low for large dataset");
        dataset_free(&ds);
        return;
    }

    dataset_free(&ds);
    TEST_PASS();
}

static void test_create_free_cycle(void)
{
    TEST_BEGIN("Create and free cycle");

    Dataset *ds = dataset_create();
    if (!ds) {
        TEST_FAIL("Failed to create dataset");
        return;
    }

    if (ds->num_samples != 0) {
        TEST_FAIL("New dataset should have 0 samples");
        dataset_free(&ds);
        return;
    }

    dataset_free(&ds);
    TEST_PASS();
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================\n");
    printf("  Dataset Module Tests\n");
    printf("========================================\n\n");

    test_valid_csv_load();
    test_missing_file();
    test_empty_file();
    test_header_only();
    test_malformed_row();
    test_invalid_number();
    test_incorrect_column_count();
    test_invalid_label();
    test_dataset_cleanup();
    test_double_free_safety();
    test_null_free();
    test_train_test_split();
    test_split_preserves_data();
    test_reproducible_split();
    test_memory_estimation();
    test_large_dataset();
    test_create_free_cycle();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
