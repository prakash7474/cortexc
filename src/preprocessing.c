/**
 * preprocessing.c - Min-max normalization
 *
 * Provides fit/transform pattern for data normalization.
 * The scaler is fitted ONLY on training data to prevent data leakage.
 *
 * C17 standard
 */

#include "preprocessing.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

int scaler_fit(Scaler *scaler, const struct Dataset *dataset)
{
    if (!scaler || !dataset) {
        fprintf(stderr, "Error: NULL argument to scaler_fit.\n");
        return -1;
    }

    if (dataset->num_samples == 0) {
        fprintf(stderr, "Error: Cannot fit scaler on empty dataset.\n");
        return -1;
    }

    size_t nf = dataset->num_features;

    /* Allocate min and max arrays */
    scaler->min_values = malloc(nf * sizeof(float));
    scaler->max_values = malloc(nf * sizeof(float));
    if (!scaler->min_values || !scaler->max_values) {
        fprintf(stderr, "Error: Memory allocation failed for scaler.\n");
        free(scaler->min_values);
        free(scaler->max_values);
        scaler->min_values = NULL;
        scaler->max_values = NULL;
        return -1;
    }

    scaler->num_features = nf;

    /* Initialize min to +infinity and max to -infinity */
    for (size_t f = 0; f < nf; f++) {
        scaler->min_values[f] = FLT_MAX;
        scaler->max_values[f] = -FLT_MAX;
    }

    /* Find min and max for each feature across training samples */
    for (size_t i = 0; i < dataset->num_samples; i++) {
        const float *features = dataset->samples[i].features;
        for (size_t f = 0; f < nf; f++) {
            if (features[f] < scaler->min_values[f]) {
                scaler->min_values[f] = features[f];
            }
            if (features[f] > scaler->max_values[f]) {
                scaler->max_values[f] = features[f];
            }
        }
    }

    return 0;
}

int scaler_transform(const Scaler *scaler, struct Dataset *dataset)
{
    if (!scaler || !dataset) {
        fprintf(stderr, "Error: NULL argument to scaler_transform.\n");
        return -1;
    }

    if (scaler->num_features != dataset->num_features) {
        fprintf(stderr, "Error: Feature count mismatch (scaler=%zu, dataset=%zu).\n",
                scaler->num_features, dataset->num_features);
        return -1;
    }

    for (size_t i = 0; i < dataset->num_samples; i++) {
        float *features = dataset->samples[i].features;
        for (size_t f = 0; f < scaler->num_features; f++) {
            float range = scaler->max_values[f] - scaler->min_values[f];

            /* Handle constant feature (max == min) to avoid division by zero */
            if (range < 1e-9f) {
                features[f] = 0.0f;
            } else {
                features[f] = (features[f] - scaler->min_values[f]) / range;
            }
        }
    }

    return 0;
}

void scaler_free(Scaler *scaler)
{
    if (!scaler) {
        return;
    }

    if (scaler->min_values) {
        free(scaler->min_values);
        scaler->min_values = NULL;
    }

    if (scaler->max_values) {
        free(scaler->max_values);
        scaler->max_values = NULL;
    }

    scaler->num_features = 0;
}
