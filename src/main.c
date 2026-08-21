/**
 * main.c - C-AI Entry Point
 *
 * Phase 3: Demonstrates the full dataset → preprocessing → perceptron pipeline.
 * Loads CSV data, normalizes features, trains a perceptron, and makes predictions.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"

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

    /* --- Step 5: Train perceptron --- */
    printf("\nTraining perceptron...\n");

    Perceptron *model = perceptron_create(train_set->num_features,
                                          DEFAULT_LEARNING_RATE,
                                          DEFAULT_RANDOM_SEED);
    if (!model) {
        fprintf(stderr, "Failed to create perceptron. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    int epochs = perceptron_train(model, train_set, DEFAULT_MAX_EPOCHS,
                                  DEFAULT_RANDOM_SEED);
    if (epochs < 0) {
        fprintf(stderr, "Training failed. Exiting.\n");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Training complete: %d epochs\n", epochs);

    /* --- Step 6: Evaluate on training data --- */
    printf("\nTraining accuracy:\n");

    {
        int correct = 0;
        for (size_t i = 0; i < train_set->num_samples; i++) {
            int pred = perceptron_forward(model, train_set->samples[i].features);
            if (pred == train_set->samples[i].label) {
                correct++;
            }
        }
        printf("  Train: %d / %zu (%.1f%%)\n", correct, train_set->num_samples,
               100.0f * (float)correct / (float)train_set->num_samples);
    }

    {
        int correct = 0;
        for (size_t i = 0; i < test_set->num_samples; i++) {
            int pred = perceptron_forward(model, test_set->samples[i].features);
            if (pred == test_set->samples[i].label) {
                correct++;
            }
        }
        printf("  Test : %d / %zu (%.1f%%)\n", correct, test_set->num_samples,
               100.0f * (float)correct / (float)test_set->num_samples);
    }

    /* --- Step 7: Print predictions --- */
    printf("\nPredictions on test data:\n");
    for (size_t i = 0; i < test_set->num_samples; i++) {
        int pred = perceptron_forward(model, test_set->samples[i].features);
        int actual = test_set->samples[i].label;
        printf("  Sample %zu: predicted=%d, actual=%d %s\n",
               i + 1, pred, actual, (pred == actual) ? "OK" : "MISS");
    }

    /* --- Step 8: Cleanup --- */
    printf("\nPhase 3 pipeline completed successfully.\n");

    perceptron_free(&model);
    scaler_free(&scaler);
    dataset_free(&train_set);
    dataset_free(&test_set);
    dataset_free(&dataset);

    printf("Memory cleanup completed.\n");

    return EXIT_SUCCESS;
}
