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
 * feature_count:  number of input features
 * learning_rate:  step size for weight updates (0 < lr <= 1)
 * epochs:         number of training epochs completed
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
    float   learning_rate;
    size_t  feature_count;
    size_t  epochs;
} Perceptron;

/**
 * perceptron_init - Allocate and initialize a new perceptron.
 *
 * All weights and bias are initialized to 0.
 *
 * feature_count:  number of input features (must be > 0)
 * learning_rate:  step size for weight updates (must be > 0)
 * epochs:         maximum number of training epochs (must be > 0)
 *
 * Returns NULL on allocation failure or invalid parameters.
 */
Perceptron *perceptron_init(size_t feature_count, float learning_rate,
                            size_t epochs);

/**
 * perceptron_create - Allocate and initialize a new perceptron (legacy alias).
 *
 * This is an alias for perceptron_init, kept for backward compatibility.
 * Random seed parameter is ignored (weights start at 0 per Phase 3 spec).
 */
Perceptron *perceptron_create(size_t feature_count, float learning_rate,
                              unsigned int seed);

/**
 * perceptron_forward - Compute the perceptron output for a single sample.
 *
 * Computes: step(bias + sum(weight[i] * feature[i]))
 * Returns 0 or 1 (step activation).
 */
int perceptron_forward(const Perceptron *p, const float *features);

/**
 * perceptron_predict - Predict label for a single sample.
 *
 * Alias for perceptron_forward. Returns 0 or 1.
 */
int perceptron_predict(const Perceptron *p, const float *features);

/**
 * perceptron_train - Train the perceptron on a dataset.
 *
 * Implements the classic perceptron learning rule:
 *   For each epoch:
 *     For each training sample:
 *       prediction = forward(features)
 *       error = actual - prediction
 *       weight[i] += learning_rate * error * feature[i]
 *       bias += learning_rate * error
 *     Count errors
 *     Print epoch info
 *     Stop early if errors == 0
 *
 * Stores the number of epochs completed in p->epochs.
 *
 * Returns the number of epochs completed. Stops early if
 * all samples are classified correctly (convergence).
 * Returns -1 on error.
 */
int perceptron_train(Perceptron *p, const struct Dataset *dataset);

/**
 * perceptron_free - Safely release all perceptron memory.
 *
 * Frees weights array and the perceptron struct itself.
 * Handles NULL pointers gracefully.
 * Sets the pointer to NULL after freeing.
 */
void perceptron_free(Perceptron **p);

#endif /* PERCEPTRON_H */
