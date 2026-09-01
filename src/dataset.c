/**
 * dataset.c - CSV dataset loading and management
 *
 * Loads CSV files into dynamically allocated Dataset structures.
 * Supports dynamic resizing, train/test split, memory estimation,
 * and safe cleanup.
 *
 * C17 standard
 */

#include "dataset.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Initial capacity for dynamic growth */
#define INITIAL_CAPACITY 16

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * trim_whitespace - Remove leading and trailing whitespace from a string.
 * Returns pointer into the original string (no allocation).
 */
static char *trim_whitespace(char *str)
{
    while (isspace((unsigned char)*str)) {
        str++;
    }
    if (*str == '\0') {
        return str;
    }
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }
    end[1] = '\0';
    return str;
}

/**
 * grow_dataset - Double the capacity of a dataset's sample array.
 * Returns 0 on success, -1 on allocation failure.
 */
static int grow_dataset(Dataset *ds)
{
    size_t new_cap = ds->capacity * 2;
    if (new_cap < INITIAL_CAPACITY) {
        new_cap = INITIAL_CAPACITY;
    }

    Sample *new_samples = realloc(ds->samples, new_cap * sizeof(Sample));
    if (!new_samples) {
        return -1;
    }

    /* Zero-initialize new slots */
    for (size_t i = ds->capacity; i < new_cap; i++) {
        new_samples[i].features = NULL;
        new_samples[i].label = 0;
    }

    ds->samples = new_samples;
    ds->capacity = new_cap;
    return 0;
}

/**
 * parse_row - Parse a single CSV row into features and label.
 *
 * Expects: "val1,val2,...,label"
 * Returns 0 on success, -1 on parse error.
 */
static int parse_row(const char *line, size_t expected_features,
                     float *out_features, int *out_label)
{
    char buffer[MAX_LINE_LENGTH];
    snprintf(buffer, sizeof(buffer), "%s", line);

    /* Remove trailing newline/carriage return */
    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
        buffer[--len] = '\0';
    }

    /* Count commas to verify column count */
    size_t comma_count = 0;
    for (size_t i = 0; buffer[i] != '\0'; i++) {
        if (buffer[i] == ',') {
            comma_count++;
        }
    }

    /* Expected: num_features commas for features + 1 for label = num_features + 1 */
    if (comma_count != expected_features) {
        return -1;
    }

    /* Parse feature values */
    char *token = strtok(buffer, ",");
    for (size_t i = 0; i < expected_features; i++) {
        if (!token) {
            return -1;
        }
        char *trimmed = trim_whitespace(token);
        char *endptr = NULL;
        float val = strtof(trimmed, &endptr);
        if (endptr == trimmed || *endptr != '\0' || !isfinite(val)) {
            return -1;
        }
        out_features[i] = val;
        token = strtok(NULL, ",");
    }

    /* Parse label (last column, must be an integer) */
    if (!token) {
        return -1;
    }
    {
        char *trimmed = trim_whitespace(token);
        char *endptr = NULL;
        long label = strtol(trimmed, &endptr, 10);
        if (endptr == trimmed || *endptr != '\0') {
            return -1;
        }
        *out_label = (int)label;
    }

    return 0;
}

/**
 * count_columns - Count the number of commas in a line + 1.
 * Used to determine the number of columns from a header line.
 */
static size_t count_columns(const char *line)
{
    size_t count = 1;
    for (size_t i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') {
            count++;
        }
    }
    return count;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

Dataset *dataset_create(void)
{
    Dataset *ds = calloc(1, sizeof(Dataset));
    if (!ds) {
        fprintf(stderr, "ERROR: Failed to allocate Dataset structure.\n");
        return NULL;
    }

    ds->samples = calloc(INITIAL_CAPACITY, sizeof(Sample));
    if (!ds->samples) {
        fprintf(stderr, "ERROR: Failed to allocate initial sample array.\n");
        free(ds);
        return NULL;
    }

    ds->num_samples = 0;
    ds->num_features = 0;
    ds->capacity = INITIAL_CAPACITY;
    ds->label_count = 0;
    ds->source_file[0] = '\0';

    return ds;
}

Dataset *dataset_load_csv(const char *filepath)
{
    if (!filepath) {
        fprintf(stderr, "ERROR: NULL file path.\n");
        return NULL;
    }

    FILE *file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "ERROR: Cannot open file '%s'.\n", filepath);
        return NULL;
    }

    Dataset *ds = dataset_create();
    if (!ds) {
        fclose(file);
        return NULL;
    }

    snprintf(ds->source_file, sizeof(ds->source_file), "%s", filepath);

    char line[MAX_LINE_LENGTH];

    /* --- Read and validate header line --- */
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "ERROR: File '%s' is empty.\n", filepath);
        dataset_free(&ds);
        fclose(file);
        return NULL;
    }

    /* Trim trailing newline from header */
    size_t hlen = strlen(line);
    while (hlen > 0 && (line[hlen - 1] == '\n' || line[hlen - 1] == '\r')) {
        line[--hlen] = '\0';
    }

    if (hlen == 0) {
        fprintf(stderr, "ERROR: File '%s' has an empty header.\n", filepath);
        dataset_free(&ds);
        fclose(file);
        return NULL;
    }

    size_t total_columns = count_columns(line);
    if (total_columns < 2) {
        fprintf(stderr, "ERROR: Header must have at least 1 feature + 1 label column.\n");
        dataset_free(&ds);
        fclose(file);
        return NULL;
    }

    /* Last column is the label; rest are features */
    ds->num_features = total_columns - 1;

    if (ds->num_features > MAX_FEATURES) {
        fprintf(stderr, "ERROR: Too many features (%zu). Maximum is %d.\n",
                ds->num_features, MAX_FEATURES);
        dataset_free(&ds);
        fclose(file);
        return NULL;
    }

    /* --- Read data rows --- */
    size_t row_num = 1; /* header is row 0 */
    int has_pass = 0, has_fail = 0;

    while (fgets(line, sizeof(line), file)) {
        row_num++;

        /* Skip empty lines */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }

        /* Grow if needed */
        if (ds->num_samples >= ds->capacity) {
            if (grow_dataset(ds) != 0) {
                fprintf(stderr, "ERROR: Memory allocation failed at row %zu.\n", row_num);
                dataset_free(&ds);
                fclose(file);
                return NULL;
            }
        }

        Sample *s = &ds->samples[ds->num_samples];

        /* Allocate feature array */
        s->features = malloc(ds->num_features * sizeof(float));
        if (!s->features) {
            fprintf(stderr, "ERROR: Memory allocation failed for features at row %zu.\n",
                    row_num);
            dataset_free(&ds);
            fclose(file);
            return NULL;
        }

        /* Parse the row */
        if (parse_row(line, ds->num_features, s->features, &s->label) != 0) {
            fprintf(stderr, "ERROR: Malformed data at row %zu in '%s'.\n",
                    row_num, filepath);
            free(s->features);
            s->features = NULL;
            dataset_free(&ds);
            fclose(file);
            return NULL;
        }

        /* Validate label is 0 or 1 */
        if (s->label != 0 && s->label != 1) {
            fprintf(stderr, "ERROR: Invalid label %d at row %zu. Expected 0 or 1.\n",
                    s->label, row_num);
            free(s->features);
            s->features = NULL;
            dataset_free(&ds);
            fclose(file);
            return NULL;
        }

        if (s->label == 0) has_fail = 1;
        if (s->label == 1) has_pass = 1;

        ds->num_samples++;
    }

    fclose(file);

    /* Validate we got some data */
    if (ds->num_samples == 0) {
        fprintf(stderr, "ERROR: No data rows found in '%s'.\n", filepath);
        dataset_free(&ds);
        return NULL;
    }

    /* Count distinct labels */
    ds->label_count = 0;
    if (has_fail) ds->label_count++;
    if (has_pass) ds->label_count++;

    return ds;
}

void dataset_print_info(const Dataset *dataset)
{
    if (!dataset) {
        printf("Dataset: (null)\n");
        return;
    }

    printf("Dataset loaded successfully.\n\n");
    printf("File    : %s\n", dataset->source_file);
    printf("Samples : %zu\n", dataset->num_samples);
    printf("Features: %zu\n", dataset->num_features);
    printf("Labels  : %d\n", dataset->label_count);
    printf("Memory  : approximately %zu bytes\n", dataset_estimate_memory(dataset));
}

size_t dataset_estimate_memory(const Dataset *dataset)
{
    if (!dataset) {
        return 0;
    }

    /* Base struct overhead */
    size_t total = sizeof(Dataset);

    /* Sample array */
    total += dataset->capacity * sizeof(Sample);

    /* Feature arrays and labels */
    for (size_t i = 0; i < dataset->num_samples; i++) {
        if (dataset->samples[i].features) {
            total += dataset->num_features * sizeof(float);
        }
        total += sizeof(int); /* label */
    }

    return total;
}

int dataset_split(const Dataset *original,
                  Dataset **train_out,
                  Dataset **test_out,
                  float train_ratio,
                  unsigned int seed)
{
    if (!original || !train_out || !test_out) {
        fprintf(stderr, "ERROR: NULL argument to dataset_split.\n");
        return -1;
    }

    if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
        fprintf(stderr, "ERROR: train_ratio must be between 0 and 1 (exclusive).\n");
        return -1;
    }

    if (original->num_samples == 0) {
        fprintf(stderr, "ERROR: Cannot split an empty dataset.\n");
        return -1;
    }

    size_t total = original->num_samples;
    size_t train_count = (size_t)((float)total * train_ratio);
    size_t test_count = total - train_count;

    /* Guard against empty train/test splits. */
    if (train_count == 0) {
        fprintf(stderr, "ERROR: Train ratio yields an empty training set "
                        "(%zu samples).\n", total);
        return -1;
    }
    if (test_count == 0) {
        fprintf(stderr, "ERROR: Train ratio yields an empty testing set "
                        "(%zu samples).\n", total);
        return -1;
    }

    /* Create index array [0, 1, 2, ..., total-1] */
    size_t *indices = malloc(total * sizeof(size_t));
    if (!indices) {
        fprintf(stderr, "ERROR: Memory allocation failed for shuffle indices.\n");
        return -1;
    }
    for (size_t i = 0; i < total; i++) {
        indices[i] = i;
    }

    /* Fisher-Yates shuffle with reproducible seed */
    srand(seed);
    for (size_t i = total - 1; i > 0; i--) {
        size_t j = (size_t)((double)rand() / (double)RAND_MAX * (double)(i + 1));
        if (j > i) j = i; /* safety clamp */
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    /* Create training dataset */
    *train_out = calloc(1, sizeof(Dataset));
    if (!*train_out) {
        fprintf(stderr, "ERROR: Memory allocation failed for training dataset.\n");
        free(indices);
        return -1;
    }

    (*train_out)->samples = calloc(train_count, sizeof(Sample));
    if (!(*train_out)->samples) {
        fprintf(stderr, "ERROR: Memory allocation failed for training samples.\n");
        free(*train_out);
        *train_out = NULL;
        free(indices);
        return -1;
    }
    (*train_out)->num_features = original->num_features;
    (*train_out)->capacity = train_count;
    (*train_out)->label_count = original->label_count;
    snprintf((*train_out)->source_file, sizeof((*train_out)->source_file),
             "%s", original->source_file);

    /* Create testing dataset */
    *test_out = calloc(1, sizeof(Dataset));
    if (!*test_out) {
        fprintf(stderr, "ERROR: Memory allocation failed for testing dataset.\n");
        dataset_free(train_out);
        free(indices);
        return -1;
    }

    (*test_out)->samples = calloc(test_count, sizeof(Sample));
    if (!(*test_out)->samples) {
        fprintf(stderr, "ERROR: Memory allocation failed for testing samples.\n");
        dataset_free(train_out);
        dataset_free(test_out);
        free(indices);
        return -1;
    }
    (*test_out)->num_features = original->num_features;
    (*test_out)->capacity = test_count;
    (*test_out)->label_count = original->label_count;
    snprintf((*test_out)->source_file, sizeof((*test_out)->source_file),
             "%s", original->source_file);

    /* Copy samples into train and test sets */
    size_t ti = 0, vi = 0;
    for (size_t i = 0; i < total; i++) {
        size_t idx = indices[i];
        const Sample *src = &original->samples[idx];

        /* Decide: first train_count go to training, rest to testing */
        Sample *dest;
        if (i < train_count) {
            dest = &(*train_out)->samples[ti++];
        } else {
            dest = &(*test_out)->samples[vi++];
        }

        /* Deep copy features */
        dest->features = malloc(original->num_features * sizeof(float));
        if (!dest->features) {
            fprintf(stderr, "ERROR: Memory allocation failed during split.\n");
            dataset_free(train_out);
            dataset_free(test_out);
            free(indices);
            return -1;
        }
        memcpy(dest->features, src->features,
               original->num_features * sizeof(float));
        dest->label = src->label;
    }

    (*train_out)->num_samples = train_count;
    (*test_out)->num_samples = test_count;

    free(indices);
    return 0;
}

void dataset_free(Dataset **dataset)
{
    if (!dataset || !(*dataset)) {
        return;
    }

    Dataset *ds = *dataset;

    if (ds->samples) {
        for (size_t i = 0; i < ds->num_samples; i++) {
            if (ds->samples[i].features) {
                free(ds->samples[i].features);
                ds->samples[i].features = NULL;
            }
        }
        free(ds->samples);
        ds->samples = NULL;
    }

    free(ds);
    *dataset = NULL;
}
