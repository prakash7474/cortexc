/**
 * test_evaluation.c - Tests for the evaluation module
 *
 * Covers accuracy and confusion matrix (TP/TN/FP/FN) computation,
 * including error/anomaly cases.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "evaluation.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_BEGIN(name) \
    do { printf("  TEST: %-44s ", name); } while (0)

#define TEST_PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while (0)

#define TEST_FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while (0)

/* --- Accuracy ------------------------------------------------------ */

static void test_accuracy_basic(void)
{
    TEST_BEGIN("Accuracy: correct/incorrect mix");
    int preds[] = {1, 0, 1, 1, 0};
    int acts[]  = {1, 0, 0, 1, 1};
    double a = calculate_accuracy(preds, acts, 5);
    if (fabs(a - 60.0) > 0.01) { TEST_FAIL("Expected 60.0%"); return; }
    TEST_PASS();
}

static void test_accuracy_perfect(void)
{
    TEST_BEGIN("Accuracy: perfect score");
    int preds[] = {1, 0, 1, 0};
    int acts[]  = {1, 0, 1, 0};
    double a = calculate_accuracy(preds, acts, 4);
    if (fabs(a - 100.0) > 0.01) { TEST_FAIL("Expected 100%"); return; }
    TEST_PASS();
}

static void test_accuracy_zero(void)
{
    TEST_BEGIN("Accuracy: zero score");
    int preds[] = {0, 1};
    int acts[]  = {1, 0};
    double a = calculate_accuracy(preds, acts, 2);
    if (fabs(a - 0.0) > 0.01) { TEST_FAIL("Expected 0%"); return; }
    TEST_PASS();
}

static void test_accuracy_null_count_zero(void)
{
    TEST_BEGIN("Accuracy: NULL/count-0 -> -1");
    int preds[] = {1};
    int acts[]  = {1};
    if (calculate_accuracy(NULL, acts, 1) != -1.0) { TEST_FAIL("NULL preds"); return; }
    if (calculate_accuracy(preds, NULL, 1) != -1.0) { TEST_FAIL("NULL actuals"); return; }
    if (calculate_accuracy(preds, acts, 0) != -1.0) { TEST_FAIL("count 0"); return; }
    TEST_PASS();
}

/* --- Confusion matrix --------------------------------------------- */

static void test_confusion_mix(void)
{
    TEST_BEGIN("Confusion matrix: mixed");
    int preds[] = {1, 0, 1, 1, 0};
    int acts[]  = {1, 0, 0, 1, 1};
    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(preds, acts, 5, &cm) != 0) {
        TEST_FAIL("returned error"); return;
    }
    if (cm.tp != 2) { TEST_FAIL("TP != 2"); return; }
    if (cm.tn != 1) { TEST_FAIL("TN != 1"); return; }
    if (cm.fp != 1) { TEST_FAIL("FP != 1"); return; }
    if (cm.fn != 1) { TEST_FAIL("FN != 1"); return; }
    if (cm.tp + cm.tn + cm.fp + cm.fn != 5) { TEST_FAIL("sum != 5"); return; }
    TEST_PASS();
}

static void test_confusion_all_positives(void)
{
    TEST_BEGIN("Confusion matrix: all positives");
    int preds[] = {1, 1, 1};
    int acts[]  = {1, 1, 1};
    ConfusionMatrix cm = {0};
    calculate_confusion_matrix(preds, acts, 3, &cm);
    if (cm.tp != 3 || cm.tn != 0 || cm.fp != 0 || cm.fn != 0) {
        TEST_FAIL("Expected TP=3"); return;
    }
    TEST_PASS();
}

static void test_confusion_invalid_prediction(void)
{
    TEST_BEGIN("Confusion matrix: invalid prediction rejected");
    int preds[] = {2, 0};
    int acts[]  = {1, 0};
    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(preds, acts, 2, &cm) != -1) {
        TEST_FAIL("Should reject non-binary prediction"); return;
    }
    TEST_PASS();
}

static void test_confusion_invalid_label(void)
{
    TEST_BEGIN("Confusion matrix: invalid label rejected");
    int preds[] = {1, 0};
    int acts[]  = {3, 0};
    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(preds, acts, 2, &cm) != -1) {
        TEST_FAIL("Should reject non-binary label"); return;
    }
    TEST_PASS();
}

static void test_confusion_null_args(void)
{
    TEST_BEGIN("Confusion matrix: NULL args rejected");
    ConfusionMatrix cm = {0};
    int preds[] = {1};
    int acts[]  = {1};
    if (calculate_confusion_matrix(NULL, acts, 1, &cm) != -1) { TEST_FAIL("NULL preds"); return; }
    if (calculate_confusion_matrix(preds, NULL, 1, &cm) != -1) { TEST_FAIL("NULL acts"); return; }
    if (calculate_confusion_matrix(preds, acts, 1, NULL) != -1) { TEST_FAIL("NULL matrix"); return; }
    if (calculate_confusion_matrix(preds, acts, 0, &cm) != -1) { TEST_FAIL("count 0"); return; }
    TEST_PASS();
}

/* --- print --------------------------------------------------------- */

static void test_print_confusion_null(void)
{
    TEST_BEGIN("print_confusion_matrix: NULL safe");
    print_confusion_matrix(NULL);
    TEST_PASS();
}

int main(void)
{
    printf("========================================\n");
    printf("  Evaluation Module Tests\n");
    printf("========================================\n\n");

    test_accuracy_basic();
    test_accuracy_perfect();
    test_accuracy_zero();
    test_accuracy_null_count_zero();
    test_confusion_mix();
    test_confusion_all_positives();
    test_confusion_invalid_prediction();
    test_confusion_invalid_label();
    test_confusion_null_args();
    test_print_confusion_null();

    printf("\n----------------------------------------\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("----------------------------------------\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
