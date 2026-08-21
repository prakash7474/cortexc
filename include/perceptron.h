/**
 * perceptron.h - Single-layer perceptron
 *
 * Implements the perceptron learning rule for binary classification.
 * Uses a step activation function: output = (sum >= 0) ? 1 : 0
 *
 * C17 standard
 */

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stddef.h>

/* Forward declaration */
struct Dataset;

/**
 * Perceptron - Single-layer perceptron model
 *
 * weights:        dynamically allocated array of weights (one per feature)
 * bias:           bias term
 * num_features:   number of input features
 * learning_rate:  step size for weight updates (0 < lr <= 1)
 *
 * The perceptron computes:
 *   output = step(w1*x1 + w2*x2 + ... + wn*xn + bias)
 *   where step(z) = (z >= 0) ? 1 : 0
 *
 * Training uses the perceptron learning rule:
 *   error = target - output
 *   wi = wi + learning_rate * error * xi
 *   bias = bias + learning_rate * error
 */
typedef struct Perceptron {
    float  *weights;
    float   bias;
    size_t  num_features;
    float   learning_rate;
} Perceptron;

/**
 * perceptron_create - Allocate and initialize a new perceptron.
 *
 * num_features:   number of input features
 * learning_rate:  step size for weight updates
 * seed:           random seed for weight initialization
 *
 * Returns NULL on allocation failure.
 */
Perceptron *perceptron_create(size_t num_features, float learning_rate,
                              unsigned int seed);

/**
 * perceptron_forward - Compute the perceptron output for a single sample.
 *
 * Returns 0 or 1 (step activation).
 */
int perceptron_forward(const Perceptron *p, const float *features);

/**
 * perceptron_predict - Predict labels for all samples in a dataset.
 *
 * Writes predictions into the provided output array (must have
 * at least num_samples elements). Returns 0 on success, -1 on error.
 */
int perceptron_predict(const Perceptron *p, const struct Dataset *dataset,
                       int *predictions);

/**
 * perceptron_train - Train the perceptron on a dataset.
 *
 * dataset:      training data
 * max_epochs:   maximum number of training epochs
 * seed:         random seed for sample shuffling
 *
 * Returns the number of epochs completed. Stops early if
 * all samples are classified correctly (convergence).
 */
int perceptron_train(Perceptron *p, const struct Dataset *dataset,
                     int max_epochs, unsigned int seed);

/**
 * perceptron_free - Safely release all perceptron memory.
 *
 * Handles NULL pointers gracefully.
 */
void perceptron_free(Perceptron **p);

#endif /* PERCEPTRON_H */
