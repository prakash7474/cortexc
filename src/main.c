/**
 * main.c - CortexC Entry Point
 *
 * CLI application with train, predict, predict-batch, and info commands.
 *
 * Usage:
 *   cortexc train <dataset.csv>              Train and save model
 *   cortexc predict <model.bin> <f1> ...     Predict using a saved model
 *   cortexc predict-batch <model.bin> <csv>  Batch predict from CSV file
 *   cortexc info <model.bin>                 Display model information
 *   cortexc                                  Show help
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"
#include "model_io.h"

/* Default model save path */
#define DEFAULT_MODEL_PATH "models/student_model.bin"

/* ------------------------------------------------------------------ */
/*  Help / Usage / Version                                             */
/* ------------------------------------------------------------------ */

static void print_version(void)
{
    printf("%s version %s\n", PROJECT_NAME, CORTEXC_VERSION);
    printf("%s\n", PROJECT_DESCRIPTION);
    printf("Single-layer perceptron\n");
}

static void print_usage(void)
{
    printf("========================================\n");
    printf("             %s\n", PROJECT_NAME);
    printf("      %s\n", PROJECT_DESCRIPTION);
    printf("========================================\n\n");
    printf("Usage:\n");
    printf("  cortexc <command> [options]\n\n");
    printf("Commands:\n");
    printf("  train <dataset.csv>                Train a model and save it\n");
    printf("  predict <model.bin> <f1> ...       Predict using a saved model\n");
    printf("  predict-batch <model.bin> <csv>    Batch predict from CSV file\n");
    printf("                                     [--output <results.csv>]\n");
    printf("  info <model.bin>                   Display model information\n");
    printf("  help                               Show this help\n");
    printf("  version                            Show version information\n\n");
    printf("Options:\n");
    printf("  -h, --help    Show this help\n");
    printf("  -v, --version Show version information\n\n");
    printf("Examples:\n");
    printf("  %s train data/students.csv\n", PROJECT_NAME);
    printf("  %s predict models/student_model.bin 7 85 8\n", PROJECT_NAME);
    printf("  %s predict-batch models/student_model.bin data/students.csv\n", PROJECT_NAME);
    printf("  %s predict-batch models/student_model.bin data/students.csv --output results.csv\n", PROJECT_NAME);
    printf("  %s info models/student_model.bin\n", PROJECT_NAME);
}

/* ------------------------------------------------------------------ */
/*  Command: train                                                     */
/* ------------------------------------------------------------------ */

static int cmd_train(const char *csv_path)
{
    printf("========================================\n");
    printf("  TRAINING PIPELINE\n");
    printf("========================================\n\n");

    /* Step 1: Load CSV */
    printf("Loading dataset: %s\n\n", csv_path);

    Dataset *dataset = dataset_load_csv(csv_path);
    if (!dataset) {
        fprintf(stderr, "ERROR: Failed to load dataset '%s'.\n", csv_path);
        return EXIT_FAILURE;
    }

    dataset_print_info(dataset);
    printf("\n");

    /* Step 2: Split dataset */
    Dataset *train_set = NULL;
    Dataset *test_set = NULL;

    if (dataset_split(dataset, &train_set, &test_set,
                      DEFAULT_TRAIN_RATIO, DEFAULT_RANDOM_SEED) != 0) {
        fprintf(stderr, "ERROR: Failed to split dataset.\n");
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Training samples: %zu\n", train_set->num_samples);
    printf("Testing samples:  %zu\n\n", test_set->num_samples);

    /* Step 3: Fit scaler on training data only */
    Scaler scaler = {0};

    if (scaler_fit(&scaler, train_set) != 0) {
        fprintf(stderr, "ERROR: Failed to fit scaler.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* Step 4: Transform training data */
    if (scaler_transform(&scaler, train_set) != 0) {
        fprintf(stderr, "ERROR: Failed to transform training data.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* Step 5: Transform testing data */
    if (scaler_transform(&scaler, test_set) != 0) {
        fprintf(stderr, "ERROR: Failed to transform testing data.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* Step 6: Initialize perceptron */
    printf("----------------------------------------\n");
    printf("PERCEPTRON TRAINING\n");
    printf("----------------------------------------\n\n");

    printf("Learning rate: %.3f\n", DEFAULT_LEARNING_RATE);
    printf("Max epochs:    %d\n\n", DEFAULT_MAX_EPOCHS);

    Perceptron *model = perceptron_init(train_set->num_features,
                                        DEFAULT_LEARNING_RATE,
                                        (size_t)DEFAULT_MAX_EPOCHS);
    if (!model) {
        fprintf(stderr, "ERROR: Failed to create perceptron.\n");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    /* Step 7: Train perceptron */
    int epochs = perceptron_train(model, train_set);
    if (epochs < 0) {
        fprintf(stderr, "ERROR: Training failed.\n");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("\nTraining completed in %d epochs.\n\n", epochs);

    /* Step 8: Evaluate on test set */
    int *predictions = predict_dataset(model, test_set);
    if (!predictions) {
        fprintf(stderr, "ERROR: Prediction failed.\n");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    int *actuals = malloc(test_set->num_samples * sizeof(int));
    if (!actuals) {
        fprintf(stderr, "ERROR: Memory allocation failed.\n");
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

    ConfusionMatrix cm = {0};
    if (calculate_confusion_matrix(predictions, actuals,
                                   test_set->num_samples, &cm) == 0) {
        print_confusion_matrix(&cm);
    }
    printf("\n");

    /* Step 9: Save model */
    printf("----------------------------------------\n");
    printf("SAVING MODEL\n");
    printf("----------------------------------------\n\n");

    if (model_save(DEFAULT_MODEL_PATH, model, &scaler) != 0) {
        fprintf(stderr, "ERROR: Failed to save model to '%s'.\n",
                DEFAULT_MODEL_PATH);
        free(predictions);
        free(actuals);
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    printf("Model saved to: %s\n\n", DEFAULT_MODEL_PATH);

    /* Step 10: Cleanup */
    free(predictions);
    free(actuals);
    perceptron_free(&model);
    scaler_free(&scaler);
    dataset_free(&train_set);
    dataset_free(&test_set);
    dataset_free(&dataset);

    printf("Training pipeline completed successfully.\n");
    return EXIT_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Command: predict                                                   */
/* ------------------------------------------------------------------ */

static int cmd_predict(const char *model_path, int argc, const char **argv)
{
    printf("========================================\n");
    printf("  PREDICTION\n");
    printf("========================================\n\n");

    /* Load model */
    ModelInfo info = {0};
    if (model_load_info(model_path, &info) != 0) {
        fprintf(stderr, "ERROR: Failed to load model from '%s'.\n", model_path);
        return EXIT_FAILURE;
    }

    printf("Loaded model: %s\n", model_path);
    printf("Features expected: %zu\n\n", info.perceptron->feature_count);

    /* Check we have the right number of features */
    size_t expected = info.perceptron->feature_count;
    size_t provided = (size_t)argc;

    if (provided != expected) {
        fprintf(stderr, "ERROR: Expected %zu feature values, got %zu.\n",
                expected, provided);
        fprintf(stderr, "Usage: cortexc predict <model.bin> <f1> <f2> ... <f%zu>\n",
                expected);
        model_info_free(&info);
        return EXIT_FAILURE;
    }

    /* Parse raw input features */
    float *raw_features = malloc(expected * sizeof(float));
    if (!raw_features) {
        fprintf(stderr, "ERROR: Memory allocation failed.\n");
        model_info_free(&info);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < expected; i++) {
        char *endptr = NULL;
        float val = strtof(argv[i], &endptr);
        if (endptr == argv[i] || *endptr != '\0' || !isfinite(val)) {
            fprintf(stderr, "ERROR: Invalid feature value '%s' at position %zu.\n",
                    argv[i], i + 1);
            free(raw_features);
            model_info_free(&info);
            return EXIT_FAILURE;
        }
        raw_features[i] = val;
    }

    /* Normalize using model's stored scaler */
    float *norm_features = malloc(expected * sizeof(float));
    if (!norm_features) {
        fprintf(stderr, "ERROR: Memory allocation failed.\n");
        free(raw_features);
        model_info_free(&info);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < expected; i++) {
        float range = info.scaler.max_values[i] - info.scaler.min_values[i];
        if (range < 1e-9f) {
            norm_features[i] = 0.0f;
        } else {
            norm_features[i] = (raw_features[i] - info.scaler.min_values[i]) / range;
        }
    }

    /* Predict */
    int prediction = perceptron_predict(info.perceptron, norm_features);

    printf("Input features: ");
    for (size_t i = 0; i < expected; i++) {
        if (i > 0) printf(", ");
        printf("%.2f", raw_features[i]);
    }
    printf("\n");

    printf("Prediction:     %s\n\n", prediction ? "PASS" : "FAIL");

    /* Cleanup */
    free(raw_features);
    free(norm_features);
    model_info_free(&info);

    return EXIT_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Command: predict-batch                                             */
/* ------------------------------------------------------------------ */

static int cmd_predict_batch(const char *model_path, const char *csv_path,
                             const char *out_csv_path)
{
    printf("========================================\n");
    printf("  BATCH PREDICTION\n");
    printf("========================================\n\n");

    /* Load model */
    ModelInfo info = {0};
    if (model_load_info(model_path, &info) != 0) {
        fprintf(stderr, "ERROR: Failed to load model from '%s'.\n", model_path);
        return EXIT_FAILURE;
    }

    printf("Loaded model: %s\n", model_path);
    printf("Input file:   %s\n", csv_path);

    int n = predict_batch(info.perceptron, &info.scaler,
                          csv_path, out_csv_path);
    if (n < 0) {
        model_info_free(&info);
        return EXIT_FAILURE;
    }

    model_info_free(&info);
    return EXIT_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Command: info                                                      */
/* ------------------------------------------------------------------ */

static int cmd_info(const char *model_path)
{
    ModelInfo info = {0};
    if (model_load_info(model_path, &info) != 0) {
        fprintf(stderr, "ERROR: Failed to load model from '%s'.\n", model_path);
        return EXIT_FAILURE;
    }

    model_print_info(&info);
    model_info_free(&info);

    return EXIT_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    /* No arguments: show help */
    if (argc < 2) {
        print_usage();
        return EXIT_SUCCESS;
    }

    const char *command = argv[1];

    /* help / version flags */
    if (strcmp(command, "--help") == 0 || strcmp(command, "-h") == 0 ||
        strcmp(command, "help") == 0) {
        print_usage();
        return EXIT_SUCCESS;
    }

    if (strcmp(command, "--version") == 0 || strcmp(command, "-v") == 0 ||
        strcmp(command, "version") == 0) {
        print_version();
        return EXIT_SUCCESS;
    }

    /* train command */
    if (strcmp(command, "train") == 0) {
        if (argc < 3) {
            fprintf(stderr, "ERROR: 'train' requires a dataset path.\n");
            fprintf(stderr, "Usage: cortexc train <dataset.csv>\n");
            return EXIT_FAILURE;
        }
        return cmd_train(argv[2]);
    }

    /* predict command */
    if (strcmp(command, "predict") == 0) {
        if (argc < 4) {
            fprintf(stderr, "ERROR: 'predict' requires a model path and feature values.\n");
            fprintf(stderr, "Usage: cortexc predict <model.bin> <f1> <f2> ... <fn>\n");
            return EXIT_FAILURE;
        }
        return cmd_predict(argv[2], argc - 3, (const char **)&argv[3]);
    }

    /* predict-batch command */
    if (strcmp(command, "predict-batch") == 0) {
        if (argc < 4) {
            fprintf(stderr, "ERROR: 'predict-batch' requires a model path and a CSV file.\n");
            fprintf(stderr, "Usage: cortexc predict-batch <model.bin> <input.csv>\n");
            fprintf(stderr, "                         [--output <results.csv>]\n");
            return EXIT_FAILURE;
        }

        /* Optional --output flag: search argv[4..] for "--output" */
        const char *out_csv_path = NULL;
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: '--output' requires a file path.\n");
                    return EXIT_FAILURE;
                }
                out_csv_path = argv[i + 1];
                break;
            }
        }

        return cmd_predict_batch(argv[2], argv[3], out_csv_path);
    }

    /* info command */
    if (strcmp(command, "info") == 0) {
        if (argc < 3) {
            fprintf(stderr, "ERROR: 'info' requires a model path.\n");
            fprintf(stderr, "Usage: cortexc info <model.bin>\n");
            return EXIT_FAILURE;
        }
        return cmd_info(argv[2]);
    }

    /* Unknown command */
    fprintf(stderr, "ERROR: Unknown command '%s'.\n\n", command);
    print_usage();
    return EXIT_FAILURE;
}
