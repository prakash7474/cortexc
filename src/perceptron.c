/**
 * perceptron.c - Single-layer perceptron
 *
 * Implements the perceptron learning rule for binary classification.
 * Uses step activation: output = (weighted_sum >= 0) ? 1 : 0
 *
 * C17 standard
 */

#include "perceptron.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

Perceptron *perceptron_create(size_t num_features, float learning_rate,
                              unsigned int seed)
{
    if (num_features == 0) {
        fprintf(stderr, "Error: perceptron_create requires at least 1 feature.\n");
        return NULL;
    }

    if (learning_rate <= 0.0f || learning_rate > 1.0f) {
        fprintf(stderr, "Error: learning_rate must be in (0, 1].\n");
        return NULL;
    }

    Perceptron *p = calloc(1, sizeof(Perceptron));
    if (!p) {
        fprintf(stderr, "Error: Memory allocation failed for Perceptron.\n");
        return NULL;
    }

    p->weights = malloc(num_features * sizeof(float));
    if (!p->weights) {
        fprintf(stderr, "Error: Memory allocation failed for weights.\n");
        free(p);
        return NULL;
    }

    p->num_features = num_features;
    p->learning_rate = learning_rate;
    p->bias = 0.0f;

    /* Initialize weights to small random values in [-0.5, 0.5] */
    srand(seed);
    for (size_t i = 0; i < num_features; i++) {
        p->weights[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }

    return p;
}

int perceptron_forward(const Perceptron *p, const float *features)
{
    if (!p || !features) {
        return 0;
    }

    float sum = p->bias;
    for (size_t i = 0; i < p->num_features; i++) {
        sum += p->weights[i] * features[i];
    }

    /* Step activation: 1 if sum >= 0, else 0 */
    return (sum >= 0.0f) ? 1 : 0;
}

int perceptron_predict(const Perceptron *p, const struct Dataset *dataset,
                       int *predictions)
{
    if (!p || !dataset || !predictions) {
        fprintf(stderr, "Error: NULL argument to perceptron_predict.\n");
        return -1;
    }

    if (p->num_features != dataset->num_features) {
        fprintf(stderr, "Error: Feature count mismatch (perceptron=%zu, dataset=%zu).\n",
                p->num_features, dataset->num_features);
        return -1;
    }

    for (size_t i = 0; i < dataset->num_samples; i++) {
        predictions[i] = perceptron_forward(p, dataset->samples[i].features);
    }

    return 0;
}

int perceptron_train(Perceptron *p, const struct Dataset *dataset,
                     int max_epochs, unsigned int seed)
{
    if (!p || !dataset) {
        fprintf(stderr, "Error: NULL argument to perceptron_train.\n");
        return -1;
    }

    if (p->num_features != dataset->num_features) {
        fprintf(stderr, "Error: Feature count mismatch in training.\n");
        return -1;
    }

    if (max_epochs <= 0) {
        fprintf(stderr, "Error: max_epochs must be positive.\n");
        return -1;
    }

    size_t n = dataset->num_samples;

    /* Create index array for shuffling */
    size_t *indices = malloc(n * sizeof(size_t));
    if (!indices) {
        fprintf(stderr, "Error: Memory allocation failed for shuffle indices.\n");
        return -1;
    }

    int epochs_completed = 0;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        /* Shuffle indices (Fisher-Yates) */
        for (size_t i = 0; i < n; i++) {
            indices[i] = i;
        }
        srand(seed + (unsigned int)epoch);
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = (size_t)((double)rand() / (double)RAND_MAX * (double)(i + 1));
            if (j > i) j = i;
            size_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        int all_correct = 1;

        for (size_t idx = 0; idx < n; idx++) {
            size_t i = indices[idx];
            const float *features = dataset->samples[i].features;
            int target = dataset->samples[i].label;

            /* Forward pass */
            int output = perceptron_forward(p, features);

            /* Compute error */
            int error = target - output;

            if (error != 0) {
                all_correct = 0;

                /* Update weights: wi += lr * error * xi */
                for (size_t f = 0; f < p->num_features; f++) {
                    p->weights[f] += p->learning_rate * (float)error * features[f];
                }

                /* Update bias */
                p->bias += p->learning_rate * (float)error;
            }
        }

        epochs_completed = epoch + 1;

        /* Early stop if all samples classified correctly */
        if (all_correct) {
            break;
        }
    }

    free(indices);
    return epochs_completed;
}

void perceptron_free(Perceptron **p)
{
    if (!p || !(*p)) {
        return;
    }

    Perceptron *ptr = *p;

    if (ptr->weights) {
        free(ptr->weights);
        ptr->weights = NULL;
    }

    free(ptr);
    *p = NULL;
}
