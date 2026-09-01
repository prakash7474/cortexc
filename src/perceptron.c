/**
 * perceptron.c - Single-layer perceptron
 *
 * Implements the perceptron learning rule for binary classification.
 * Uses step activation: output = (weighted_sum >= 0) ? 1 : 0
 *
 * All weights and bias are initialized to 0.
 * Training prints per-epoch error counts and supports early stopping.
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

Perceptron *perceptron_init(size_t feature_count, float learning_rate,
                            size_t epochs)
{
    if (feature_count == 0) {
        fprintf(stderr, "ERROR: perceptron_init requires at least 1 feature.\n");
        return NULL;
    }

    if (learning_rate <= 0.0f || learning_rate > 1.0f ||
        !isfinite(learning_rate)) {
        fprintf(stderr, "ERROR: learning_rate must be in (0, 1].\n");
        return NULL;
    }

    if (epochs == 0) {
        fprintf(stderr, "ERROR: epochs must be > 0.\n");
        return NULL;
    }

    Perceptron *p = calloc(1, sizeof(Perceptron));
    if (!p) {
        fprintf(stderr, "ERROR: Memory allocation failed for Perceptron.\n");
        return NULL;
    }

    p->weights = calloc(feature_count, sizeof(float));
    if (!p->weights) {
        fprintf(stderr, "ERROR: Memory allocation failed for weights.\n");
        free(p);
        return NULL;
    }

    p->feature_count  = feature_count;
    p->learning_rate  = learning_rate;
    p->bias           = 0.0f;
    p->epochs         = epochs;

    /* Weights are initialized to 0 by calloc */

    return p;
}

Perceptron *perceptron_create(size_t feature_count, float learning_rate,
                              unsigned int seed)
{
    /* Legacy alias: ignore seed, use default max_epochs from config */
    (void)seed;
    return perceptron_init(feature_count, learning_rate, (size_t)1000);
}

int perceptron_forward(const Perceptron *p, const float *features)
{
    if (!p || !features) {
        return 0;
    }

    float sum = p->bias;
    for (size_t i = 0; i < p->feature_count; i++) {
        sum += p->weights[i] * features[i];
    }

    /* Step activation: 1 if sum >= 0, else 0 */
    return (sum >= 0.0f) ? 1 : 0;
}

int perceptron_predict(const Perceptron *p, const float *features)
{
    return perceptron_forward(p, features);
}

int perceptron_train(Perceptron *p, const struct Dataset *dataset)
{
    if (!p || !dataset) {
        fprintf(stderr, "ERROR: NULL argument to perceptron_train.\n");
        return -1;
    }

    if (p->feature_count != dataset->num_features) {
        fprintf(stderr, "ERROR: Feature count mismatch (perceptron=%zu, dataset=%zu).\n",
                p->feature_count, dataset->num_features);
        return -1;
    }

    size_t n = dataset->num_samples;
    size_t max_epochs = p->epochs;
    int epochs_completed = 0;

    if (n == 0) {
        fprintf(stderr, "ERROR: Cannot train on an empty dataset.\n");
        return -1;
    }

    for (size_t epoch = 0; epoch < max_epochs; epoch++) {
        int errors = 0;

        for (size_t i = 0; i < n; i++) {
            const float *features = dataset->samples[i].features;

            /* Reject non-finite features so NaN/Inf cannot corrupt training. */
            for (size_t f = 0; f < p->feature_count; f++) {
                if (!isfinite(features[f])) {
                    fprintf(stderr, "ERROR: Non-finite feature at sample %zu, "
                                    "feature %zu.\n", i, f);
                    return -1;
                }
            }

            int target = dataset->samples[i].label;

            /* Forward pass */
            int output = perceptron_forward(p, features);

            /* Compute error */
            int error = target - output;

            if (error != 0) {
                errors++;

                /* Update weights: wi += lr * error * xi */
                for (size_t f = 0; f < p->feature_count; f++) {
                    p->weights[f] += p->learning_rate * (float)error * features[f];
                }

                /* Update bias: bias += lr * error */
                p->bias += p->learning_rate * (float)error;

                /* Abort if training diverged into NaN/Inf. */
                if (!isfinite(p->bias)) {
                    fprintf(stderr, "ERROR: Training diverged (bias became "
                                    "non-finite) at epoch %zu.\n", epoch + 1);
                    return -1;
                }
                for (size_t f = 0; f < p->feature_count; f++) {
                    if (!isfinite(p->weights[f])) {
                        fprintf(stderr, "ERROR: Training diverged (weight %zu "
                                        "became non-finite) at epoch %zu.\n",
                                f, epoch + 1);
                        return -1;
                    }
                }
            }
        }

        epochs_completed = (int)(epoch + 1);

        /* Print epoch info */
        printf("Epoch %d   Errors: %d\n", epochs_completed, errors);

        /* Early stop if all samples classified correctly */
        if (errors == 0) {
            break;
        }
    }

    /* Store actual epochs trained */
    p->epochs = (size_t)epochs_completed;

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
