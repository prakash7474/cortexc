/**
 * prediction.h - Prediction orchestration
 *
 * Provides high-level functions for single and batch prediction
 * using the perceptron model.
 *
 * C17 standard
 */

#ifndef PREDICTION_H
#define PREDICTION_H

#include "perceptron.h"
#include "dataset.h"

/**
 * predict_sample - Predict the label for a single sample.
 *
 * Uses the perceptron's forward pass to compute the prediction.
 * Returns 0 or 1 on success, -1 on error.
 */
int predict_sample(const Perceptron *p, const float *features, int *prediction);

/**
 * predict_dataset - Predict labels for all samples in a dataset.
 *
 * Allocates and fills an array of predictions (one per sample).
 * The caller must free the returned array with free().
 *
 * Returns pointer to predictions array on success, NULL on error.
 */
int *predict_dataset(const Perceptron *p, const Dataset *dataset);

#endif /* PREDICTION_H */
