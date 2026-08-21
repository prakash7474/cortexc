/**
 * main.c - C-AI Entry Point
 *
 * Phase 2: Demonstrates the dataset and preprocessing pipeline.
 * Loads a CSV dataset, prints info, splits into train/test,
 * normalizes features, and cleans up all memory.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "dataset.h"
#include "preprocessing.h"

int main(void)
{
    printf("========================================\n");
    printf("             CortexC\n");
    printf("      CPU-ONLY ML ENGINE\n");
    printf("========================================\n\n");

    /* --- Step 1: Load dataset --- */
    const char *csv_path = "data/students.csv";

    printf("Loading dataset:\n%s\n\n", csv_path);

    Dataset *dataset = dataset_load_csv(csv_path);
    if (!dataset) {
        fprintf(stderr, "Failed to load dataset. Exiting.\n");
        return EXIT_FAILURE;
    }

    dataset_print_info(dataset);

    /* --- Step 2: Train/test split --- */
    printf("\nSplitting dataset...\n");

    Dataset *train_set = NULL;
    Dataset *test_set = NULL;

    if (dataset_split(dataset, &train_set, &test_set,
                      DEFAULT_TRAIN_RATIO, DEFAULT_RANDOM_SEED) != 0) {
        fprintf(stderr, "Failed to split dataset. Exiting.\n");
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Training samples: %zu\n", train_set->num_samples);
    printf("Testing samples : %zu\n", test_set->num_samples);

    /* --- Step 3: Fit scaler on training data only --- */
    printf("\nFitting scaler using training data...\n");

    Scaler scaler = {0};

    if (scaler_fit(&scaler, train_set) != 0) {
        fprintf(stderr, "Failed to fit scaler. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 4: Transform both train and test data --- */
    if (scaler_transform(&scaler, train_set) != 0) {
        fprintf(stderr, "Failed to transform training data. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    if (scaler_transform(&scaler, test_set) != 0) {
        fprintf(stderr, "Failed to transform testing data. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Normalization complete.\n");

    /* --- Step 5: Confirmation --- */
    printf("\nPhase 2 dataset pipeline completed successfully.\n");

    /* --- Step 6: Free all memory --- */
    scaler_free(&scaler);
    dataset_free(&train_set);
    dataset_free(&test_set);
    dataset_free(&dataset);

    printf("Memory cleanup completed.\n");

    return EXIT_SUCCESS;
}
