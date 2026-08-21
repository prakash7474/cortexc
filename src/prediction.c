/**
 * prediction.c - Prediction orchestration
 *
 * Provides high-level functions for single and batch prediction
 * using the perceptron model.
 *
 * Does not duplicate perceptron mathematics; delegates to perceptron_forward().
 *
 * C17 standard
 */

#include "prediction.h"
#include "perceptron.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>

int predict_sample(const Perceptron *p, const float *features, int *prediction)
{
    if (!p || !features || !prediction) {
        fprintf(stderr, "Error: NULL argument to predict_sample.\n");
        return -1;
    }

    *prediction = perceptron_forward(p, features);
    return 0;
}

int *predict_dataset(const Perceptron *p, const Dataset *dataset)
{
    if (!p || !dataset) {
        fprintf(stderr, "Error: NULL argument to predict_dataset.\n");
        return NULL;
    }

    if (p->feature_count != dataset->num_features) {
        fprintf(stderr, "Error: Feature count mismatch in predict_dataset.\n");
        return NULL;
    }

    int *predictions = malloc(dataset->num_samples * sizeof(int));
    if (!predictions) {
        fprintf(stderr, "Error: Memory allocation failed for predictions.\n");
        return NULL;
    }

    for (size_t i = 0; i < dataset->num_samples; i++) {
        predictions[i] = perceptron_forward(p, dataset->samples[i].features);
    }

    return predictions;
}
