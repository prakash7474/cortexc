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
#include "preprocessing.h"

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

/**
 * predict_batch - Batch-predict every row of a CSV file.
 *
 * Reads a CSV whose rows contain either `feature_count` numeric columns
 * (no label) or `feature_count + 1` numeric columns (the extra column is
 * treated as the actual label and emitted in the output). A non-numeric
 * first line (e.g. a header) is skipped automatically.
 *
 * If out_csv_path is non-NULL the results are also written to that file
 * with a header row of the form:
 *   feature1,...,featureN,actual,predicted
 * Otherwise the results are only printed to stdout.
 *
 * Returns the number of rows successfully predicted, or -1 on a fatal
 * error (model/scaler mismatch, missing file, allocation failure).
 */
int predict_batch(const Perceptron *p, const Scaler *scaler,
		  const char *csv_path, const char *out_csv_path);

#endif /* PREDICTION_H */
