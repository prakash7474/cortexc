/**
 * main.c - C-AI Entry Point
 *
 * Phase 3: Demonstrates the full dataset -> preprocessing -> perceptron ->
 * prediction -> evaluation pipeline.
 *
 * Loads CSV data, normalizes features, trains a perceptron,
 * makes predictions, calculates accuracy, and prints confusion matrix.
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"

int main(void)
{
    printf("========================================\n");
    printf("             CortexC\n");
    printf("      CPU-ONLY ML ENGINE\n");
    printf("========================================\n\n");

    /* --- Step 1: Load CSV --- */
    const char *csv_path = "data/students.csv";

    printf("Dataset:\n%s\n\n", csv_path);

    Dataset *dataset = dataset_load_csv(csv_path);
    if (!dataset) {
        fprintf(stderr, "Failed to load dataset. Exiting.\n");
        return EXIT_FAILURE;
    }

    printf("Features:\n%zu\n\n", dataset->num_features);

    /* --- Step 2: Split dataset --- */
    Dataset *train_set = NULL;
    Dataset *test_set = NULL;

    if (dataset_split(dataset, &train_set, &test_set,
                      DEFAULT_TRAIN_RATIO, DEFAULT_RANDOM_SEED) != 0) {
        fprintf(stderr, "Failed to split dataset. Exiting.\n");
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Training samples:\n%zu\n", train_set->num_samples);
    printf("Testing samples:\n%zu\n\n", test_set->num_samples);

    /* --- Step 3: Fit scaler on training data only --- */
    Scaler scaler = {0};

    if (scaler_fit(&scaler, train_set) != 0) {
        fprintf(stderr, "Failed to fit scaler. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 4: Transform training data --- */
    if (scaler_transform(&scaler, train_set) != 0) {
        fprintf(stderr, "Failed to transform training data. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 5: Transform testing data --- */
    if (scaler_transform(&scaler, test_set) != 0) {
        fprintf(stderr, "Failed to transform testing data. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 6: Initialize perceptron --- */
    printf("----------------------------------------\n");
    printf("PERCEPTRON TRAINING\n");
    printf("----------------------------------------\n\n");

    printf("Learning rate: %.3f\n", DEFAULT_LEARNING_RATE);
    printf("Epochs: %d\n\n", DEFAULT_MAX_EPOCHS);

    Perceptron *model = perceptron_init(train_set->num_features,
                                        DEFAULT_LEARNING_RATE,
                                        (size_t)DEFAULT_MAX_EPOCHS);
    if (!model) {
        fprintf(stderr, "Failed to create perceptron. Exiting.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 7: Train perceptron --- */
    int epochs = perceptron_train(model, train_set);
    if (epochs < 0) {
        fprintf(stderr, "Training failed. Exiting.\n");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("\nTraining completed.\n\n");

    /* --- Step 8: Predict test samples --- */
    int *predictions = predict_dataset(model, test_set);
    if (!predictions) {
        fprintf(stderr, "Prediction failed. Exiting.\n");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* --- Step 9: Calculate accuracy --- */
    int *actuals = malloc(test_set->num_samples * sizeof(int));
    if (!actuals) {
        fprintf(stderr, "Memory allocation failed. Exiting.\n");
        free(predictions);
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < test_set->num_samples; i++) {
        actuals[i] = test_set->samples[i].label;
    }

    double accuracy = calculate_accuracy(predictions, actuals,
                                         test_set->num_samples);

    printf("----------------------------------------\n");
    printf("EVALUATION\n");
    printf("----------------------------------------\n\n");

    printf("Accuracy: %.2f%%\n", accuracy);

    /* --- Step 10: Print confusion matrix --- */
    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(predictions, actuals,
                                   test_set->num_samples, &cm) == 0) {
        print_confusion_matrix(&cm);
    }

    printf("\n");

    /* --- Step 11: Free model --- */
    perceptron_free(&model);

    /* --- Step 12: Free datasets --- */
    scaler_free(&scaler);
    dataset_free(&train_set);
    dataset_free(&test_set);
    dataset_free(&dataset);

    free(predictions);
    free(actuals);

    /* --- Step 13: Exit cleanly --- */
    printf("Phase 3 pipeline completed successfully.\n");
    printf("Memory cleanup completed.\n");

    return EXIT_SUCCESS;
}
