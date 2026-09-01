/**
 * generate_dataset.c - Synthetic CSV dataset generator
 *
 * Generates a linearly separable dataset with the requested number of
 * samples and features. The label is 1 when the (deterministic) weighted
 * sum of the features exceeds a threshold, so it is learnable by a
 * single-layer perceptron.
 *
 * Usage:
 *   generate_dataset <samples> <features> <output.csv> [seed]
 *
 * Example:
 *   generate_dataset 100000 3 data/big.csv 42
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int g_seed;

static unsigned int next_rand(void)
{
    /* Simple deterministic LCG (only used for data generation). */
    g_seed = g_seed * 1103515245u + 12345u;
    return (g_seed / 65536u) % 32768u;
}

static double next_float(void)
{
    return (double)next_rand() / 32768.0; /* [0, 1) */
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "ERROR: Usage: %s <samples> <features> <output.csv> [seed]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    long samples = strtol(argv[1], NULL, 10);
    long features = strtol(argv[2], NULL, 10);
    const char *path = argv[3];
    unsigned int seed = (argc >= 5) ? (unsigned int)strtoul(argv[4], NULL, 10) : 42u;

    if (samples <= 0) {
        fprintf(stderr, "ERROR: samples must be > 0.\n");
        return EXIT_FAILURE;
    }
    if (features <= 0) {
        fprintf(stderr, "ERROR: features must be > 0.\n");
        return EXIT_FAILURE;
    }

    g_seed = seed;

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s' for writing.\n", path);
        return EXIT_FAILURE;
    }

    /* Header */
    for (long f = 0; f < features; f++) {
        fprintf(fp, "feature%ld", f + 1);
        if (f + 1 < features) fprintf(fp, ",");
    }
    fprintf(fp, ",label\n");

    /* Generate rows. The decision boundary is sum(f_i) > features/2. */
    double threshold = (double)features * 0.5;
    for (long i = 0; i < samples; i++) {
        double sum = 0.0;
        for (long f = 0; f < features; f++) {
            double v = next_float() * 10.0; /* features in [0,10) */
            sum += v;
            fprintf(fp, "%.6g", v);
            if (f + 1 < features) fprintf(fp, ",");
        }
        int label = (sum > threshold) ? 1 : 0;
        fprintf(fp, ",%d\n", label);
    }

    fclose(fp);

    printf("Generated %ld samples x %ld features -> %s (seed %u)\n",
           samples, features, path, seed);
    return EXIT_SUCCESS;
}
