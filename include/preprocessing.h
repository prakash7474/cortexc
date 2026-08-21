/**
 * preprocessing.h - Data preprocessing (normalization)
 *
 * Provides min-max normalization with fit/transform pattern.
 * The scaler is fitted on training data only to prevent data leakage.
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stddef.h>

#include "dataset.h"

/**
 * Scaler - Min-max normalization state
 *
 * min_values:    minimum value for each feature (fitted from training data)
 * max_values:    maximum value for each feature (fitted from training data)
 * num_features:  number of features to scale
 *
 * Fitted using scaler_fit() on training data only.
 * Applied to both training and testing data using scaler_transform().
 */
typedef struct {
    float  *min_values;
    float  *max_values;
    size_t  num_features;
} Scaler;

/**
 * scaler_fit - Compute min and max for each feature from a dataset.
 *
 * Only call this on the TRAINING dataset to prevent data leakage.
 * Returns 0 on success, -1 on error.
 */
int scaler_fit(Scaler *scaler, const struct Dataset *dataset);

/**
 * scaler_transform - Apply min-max normalization to a dataset in-place.
 *
 * Uses the min/max values computed by scaler_fit().
 * Handles constant features (max == min) by setting output to 0.0.
 * Returns 0 on success, -1 on error.
 */
int scaler_transform(const Scaler *scaler, struct Dataset *dataset);

/**
 * scaler_free - Safely release scaler memory.
 *
 * Handles NULL pointers gracefully.
 */
void scaler_free(Scaler *scaler);

#endif /* PREPROCESSING_H */
