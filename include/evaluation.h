/**
 * evaluation.h - Model evaluation metrics
 *
 * Computes accuracy and confusion matrix for binary classification.
 * Uses TP (True Positive), TN (True Negative), FP (False Positive),
 * FN (False Negative).
 *
 * C17 standard
 */

#ifndef EVALUATION_H
#define EVALUATION_H

#include <stddef.h>

/**
 * ConfusionMatrix - Results of comparing predictions to actual labels.
 *
 * tp: True Positives   (predicted=1, actual=1)
 * tn: True Negatives   (predicted=0, actual=0)
 * fp: False Positives  (predicted=1, actual=0)
 * fn: False Negatives  (predicted=0, actual=1)
 */
typedef struct {
    int tp;
    int tn;
    int fp;
    int fn;
} ConfusionMatrix;

/**
 * calculate_accuracy - Compute classification accuracy.
 *
 * predictions:  array of predicted labels (0 or 1)
 * actuals:      array of actual labels (0 or 1)
 * count:        number of samples
 *
 * Returns accuracy as a percentage (0.0 to 100.0).
 * Returns -1.0 on error.
 */
double calculate_accuracy(const int *predictions, const int *actuals,
                          size_t count);

/**
 * calculate_confusion_matrix - Compute TP, TN, FP, FN counts.
 *
 * predictions:  array of predicted labels (0 or 1)
 * actuals:      array of actual labels (0 or 1)
 * count:        number of samples
 * matrix:       output confusion matrix (must not be NULL)
 *
 * Validates that all labels are 0 or 1.
 * Returns 0 on success, -1 on error.
 */
int calculate_confusion_matrix(const int *predictions, const int *actuals,
                               size_t count, ConfusionMatrix *matrix);

/**
 * print_confusion_matrix - Display the confusion matrix in a formatted table.
 *
 * Prints:
 *               Predicted
 *               0       1
 *   Actual 0   TN      FP
 *   Actual 1   FN      TP
 */
void print_confusion_matrix(const ConfusionMatrix *matrix);

#endif /* EVALUATION_H */
