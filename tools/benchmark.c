/**
 * benchmark.c - CPU and RAM benchmark for CortexC
 *
 * Generates a synthetic CSV dataset, then measures wall-clock time for:
 *   - dataset loading
 *   - min/max preprocessing (fit + transform)
 *   - perceptron training
 *   - prediction
 *   - model save
 *   - model load
 *
 * Results are printed in milliseconds using clock() (single-threaded,
 * CPU-only). Also reports estimated RAM usage.
 *
 * Usage:
 *   benchmark [samples] [features]
 *
 * Example:
 *   benchmark 100000 3
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"
#include "model_io.h"
#include "config.h"

static double now_ms(void)
{
    return ((double)clock() / (double)CLOCKS_PER_SEC) * 1000.0;
}

int main(int argc, char *argv[])
{
    long samples = (argc >= 2) ? strtol(argv[1], NULL, 10) : 100000;
    long features = (argc >= 3) ? strtol(argv[2], NULL, 10) : 3;

    if (samples <= 0) { fprintf(stderr, "ERROR: samples must be > 0.\n"); return 1; }
    if (features <= 0) { fprintf(stderr, "ERROR: features must be > 0.\n"); return 1; }
    if (features > MAX_FEATURES) { fprintf(stderr, "ERROR: features exceed %d.\n", MAX_FEATURES); return 1; }

    const char *csv_path = "build/benchmark_data.csv";
    const char *model_path = "build/benchmark_model.bin";

    /* ---- Build a synthetic (separable) dataset on disk ---- */
    FILE *gen = fopen(csv_path, "w");
    if (!gen) { fprintf(stderr, "ERROR: Cannot write %s.\n", csv_path); return 1; }
    for (long f = 0; f < features; f++) {
        fprintf(gen, "f%ld", f + 1);
        if (f + 1 < features) fprintf(gen, ",");
    }
    fprintf(gen, ",label\n");
    double threshold = (double)features * 0.5;
    for (long i = 0; i < samples; i++) {
        double sum = 0.0;
        for (long f = 0; f < features; f++) {
            double v = ((double)rand() / (double)RAND_MAX) * 10.0;
            sum += v;
            fprintf(gen, "%.6g", v);
            if (f + 1 < features) fprintf(gen, ",");
        }
        fprintf(gen, ",%d\n", (sum > threshold) ? 1 : 0);
    }
    fclose(gen);

    printf("========================================\n");
    printf("  CortexC CPU Benchmark\n");
    printf("========================================\n");
    printf("Samples : %ld\n", samples);
    printf("Features: %ld\n\n", features);

    /* ---- 1. Dataset loading ---- */
    double t0 = now_ms();
    Dataset *ds = dataset_load_csv(csv_path);
    double t_load = now_ms() - t0;
    if (!ds) { fprintf(stderr, "ERROR: Failed to load dataset.\n"); return 1; }
    printf("Dataset load    : %10.2f ms\n", t_load);
    printf("  Estimated RAM : %zu bytes (%.2f MiB)\n",
           dataset_estimate_memory(ds),
           (double)dataset_estimate_memory(ds) / (1024.0 * 1024.0));

    /* ---- 2. Preprocessing (fit + transform on a train split) ---- */
    Dataset *train = NULL, *test = NULL;
    if (dataset_split(ds, &train, &test, DEFAULT_TRAIN_RATIO, DEFAULT_RANDOM_SEED) != 0) {
        fprintf(stderr, "ERROR: Split failed.\n");
        dataset_free(&ds);
        return 1;
    }

    double t1 = now_ms();
    Scaler scaler = {0};
    scaler_fit(&scaler, train);
    scaler_transform(&scaler, train);
    scaler_transform(&scaler, test);
    double t_pre = now_ms() - t1;
    printf("Preprocessing   : %10.2f ms\n", t_pre);

    /* ---- 3. Training ---- */
    Perceptron *model = perceptron_init(train->num_features, DEFAULT_LEARNING_RATE,
                                        (size_t)DEFAULT_MAX_EPOCHS);
    if (!model) { fprintf(stderr, "ERROR: Failed to init perceptron.\n"); return 1; }

    double t2 = now_ms();
    int epochs = perceptron_train(model, train);
    double t_train = now_ms() - t2;
    printf("Training        : %10.2f ms  (%d epochs)\n", t_train, epochs);

    /* ---- 4. Prediction ---- */
    double t3 = now_ms();
    int *preds = predict_dataset(model, test);
    double t_predict = now_ms() - t3;
    if (preds) free(preds);
    printf("Prediction      : %10.2f ms  (over %zu test samples)\n",
           t_predict, test->num_samples);

    /* ---- 5. Model save ---- */
    double t4 = now_ms();
    int save_ok = model_save(model_path, model, &scaler);
    double t_save = now_ms() - t4;
    printf("Model save      : %10.2f ms  (%s)\n", t_save,
           save_ok == 0 ? "ok" : "FAILED");

    /* ---- 6. Model load ---- */
    Perceptron *loaded = NULL;
    Scaler loaded_scaler = {0};
    double t5 = now_ms();
    int load_ok = model_load(model_path, &loaded, &loaded_scaler);
    double t_load_model = now_ms() - t5;
    printf("Model load      : %10.2f ms  (%s)\n", t_load_model,
           load_ok == 0 ? "ok" : "FAILED");

    printf("\nTotal (core ops): %.2f ms\n", t_load + t_pre + t_train + t_predict + t_save + t_load_model);

    if (loaded) perceptron_free(&loaded);
    scaler_free(&loaded_scaler);
    perceptron_free(&model);
    scaler_free(&scaler);
    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&ds);

    remove(csv_path);
    remove(model_path);

    return 0;
}
