/**
 * dataset.h - Dataset loading and management
 *
 * Loads CSV files into dynamically allocated Dataset structures.
 * Supports dynamic resizing, train/test split, memory estimation,
 * and safe cleanup.
 */

#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

/**
 * Sample - A single data point
 *
 * features: dynamically allocated array of float values
 * label:    integer class label (0 = FAIL, 1 = PASS)
 */
typedef struct Sample {
    float *features;
    int    label;
} Sample;

/**
 * Dataset - A collection of samples
 *
 * samples:         dynamically allocated array of Sample
 * num_samples:     number of loaded samples
 * num_features:    number of features per sample (excluding label)
 * capacity:        current allocated capacity (for dynamic growth)
 * label_count:     number of distinct labels (0 and 1 in binary case)
 * source_file:     path of the CSV file this dataset was loaded from
 */
typedef struct Dataset {
    Sample *samples;
    size_t  num_samples;
    size_t  num_features;
    size_t  capacity;
    int     label_count;
    char    source_file[256];
} Dataset;

/**
 * dataset_create - Create an empty dataset.
 * Returns NULL on allocation failure.
 */
Dataset *dataset_create(void);

/**
 * dataset_load_csv - Load a CSV file into a dataset.
 *
 * The first line must be a header: feature1,feature2,...,label
 * All subsequent rows must contain the correct number of numeric columns.
 *
 * Returns NULL on any error (file not found, malformed data, allocation failure).
 * Error messages are printed to stderr.
 */
Dataset *dataset_load_csv(const char *filepath);

/**
 * dataset_print_info - Display dataset information.
 *
 * Prints: file path, sample count, feature count, approximate memory usage.
 */
void dataset_print_info(const Dataset *dataset);

/**
 * dataset_estimate_memory - Estimate memory usage of a dataset in bytes.
 *
 * This is an approximate estimate that includes:
 * - sizeof(Sample) for each sample
 * - sizeof(float) * num_features for feature arrays
 * - sizeof(int) for labels
 * - struct Dataset overhead
 */
size_t dataset_estimate_memory(const Dataset *dataset);

/**
 * dataset_split - Split dataset into training and testing sets.
 *
 * train_ratio: fraction of samples for training (e.g., 0.8 for 80%)
 * seed:        random seed for reproducibility
 *
 * Returns 0 on success, -1 on error.
 * Original dataset is NOT modified.
 */
int dataset_split(const Dataset *original,
                  Dataset **train_out,
                  Dataset **test_out,
                  float train_ratio,
                  unsigned int seed);

/**
 * dataset_free - Safely release all memory owned by a dataset.
 *
 * Handles NULL pointers gracefully.
 * Sets the pointer to NULL after freeing to prevent double-free.
 */
void dataset_free(Dataset **dataset);

#endif /* DATASET_H */
