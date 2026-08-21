/**
 * evaluation.c - Model evaluation metrics
 *
 * Computes accuracy and confusion matrix for binary classification.
 *
 * C17 standard
 */

#include "evaluation.h"

#include <stdio.h>
#include <stddef.h>

double calculate_accuracy(const int *predictions, const int *actuals,
                          size_t count)
{
    if (!predictions || !actuals || count == 0) {
        return -1.0;
    }

    int correct = 0;
    for (size_t i = 0; i < count; i++) {
        if (predictions[i] == actuals[i]) {
            correct++;
        }
    }

    return ((double)correct / (double)count) * 100.0;
}

int calculate_confusion_matrix(const int *predictions, const int *actuals,
                               size_t count, ConfusionMatrix *matrix)
{
    if (!predictions || !actuals || !matrix || count == 0) {
        fprintf(stderr, "Error: NULL argument to calculate_confusion_matrix.\n");
        return -1;
    }

    /* Initialize counts to zero */
    matrix->tp = 0;
    matrix->tn = 0;
    matrix->fp = 0;
    matrix->fn = 0;

    for (size_t i = 0; i < count; i++) {
        int pred = predictions[i];
        int actual = actuals[i];

        /* Validate labels */
        if (pred != 0 && pred != 1) {
            fprintf(stderr, "Error: Invalid prediction %d at index %zu. Expected 0 or 1.\n",
                    pred, i);
            return -1;
        }
        if (actual != 0 && actual != 1) {
            fprintf(stderr, "Error: Invalid label %d at index %zu. Expected 0 or 1.\n",
                    actual, i);
            return -1;
        }

        if (pred == 1 && actual == 1) {
            matrix->tp++;
        } else if (pred == 0 && actual == 0) {
            matrix->tn++;
        } else if (pred == 1 && actual == 0) {
            matrix->fp++;
        } else { /* pred == 0 && actual == 1 */
            matrix->fn++;
        }
    }

    return 0;
}

void print_confusion_matrix(const ConfusionMatrix *matrix)
{
    if (!matrix) {
        return;
    }

    printf("\n              Predicted\n");
    printf("              0       1\n");
    printf("\n");
    printf("Actual 0      %-8d%d\n", matrix->tn, matrix->fp);
    printf("Actual 1      %-8d%d\n", matrix->fn, matrix->tp);
}
