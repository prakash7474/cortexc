/**
 * prediction.c - Prediction orchestration
 *
 * Provides high-level functions for single and batch prediction
 * using the perceptron model.
 *
 * Does not duplicate perceptron mathematics; delegates to perceptron_forward().
 *
 * C17 standard
 */

#include "prediction.h"
#include "perceptron.h"
#include "dataset.h"
#include "preprocessing.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int predict_sample(const Perceptron *p, const float *features, int *prediction)
{
    if (!p || !features || !prediction) {
        fprintf(stderr, "ERROR: NULL argument to predict_sample.\n");
        return -1;
    }

    if (p->feature_count == 0) {
        fprintf(stderr, "ERROR: Perceptron has zero features.\n");
        return -1;
    }

    if (p->weights == NULL) {
        fprintf(stderr, "ERROR: Perceptron has NULL weights.\n");
        return -1;
    }

    *prediction = perceptron_forward(p, features);
    return 0;
}

int *predict_dataset(const Perceptron *p, const Dataset *dataset)
{
    if (!p || !dataset) {
        fprintf(stderr, "ERROR: NULL argument to predict_dataset.\n");
        return NULL;
    }

    if (p->feature_count != dataset->num_features) {
        fprintf(stderr, "ERROR: Feature count mismatch in predict_dataset.\n");
        return NULL;
    }

    if (dataset->num_samples == 0) {
        fprintf(stderr, "ERROR: Cannot predict on an empty dataset.\n");
        return NULL;
    }

    int *predictions = malloc(dataset->num_samples * sizeof(int));
    if (!predictions) {
        fprintf(stderr, "ERROR: Memory allocation failed for predictions.\n");
        return NULL;
    }

    for (size_t i = 0; i < dataset->num_samples; i++) {
        predictions[i] = perceptron_forward(p, dataset->samples[i].features);
    }

    return predictions;
}

/* ------------------------------------------------------------------ */
/*  Batch prediction                                                   */
/* ------------------------------------------------------------------ */

/* Skip leading whitespace. */
static const char *skip_ws(const char *s)
{
    while (*s == ' ' || *s == '\t' || *s == '\r') {
        s++;
    }
    return s;
}

/**
 * Tokenize a CSV line into the first max_tokens whitespace-trimmed
 * comma-separated fields. Returns the number of non-empty fields found,
 * or -1 if a field is not a valid finite number.
 *
 * If out_label is non-NULL it is set to 1 when the line contains more
 * than expect fields (i.e. a trailing label column is present).
 */
static int parse_line(const char *line, float *out, size_t max_tokens,
                      int expect, int *out_has_label)
{
    size_t count = 0;
    const char *p = skip_ws(line);

    if (*p == '\0' || *p == '\n' || *p == '\r') {
        return 0;
    }

    if (out_has_label) {
        *out_has_label = 0;
    }

    /* Bounded per-field scan buffer (values are floats, so small). */
    while (*p != '\0' && *p != '\n' && *p != '\r') {
        /* Skip field into a temp buffer. */
        const char *start = p;
        while (*p != '\0' && *p != ',' && *p != '\n' && *p != '\r') {
            p++;
        }
        const char *field_end = p;

        /* Copy field (trimmed) into a small scratch buffer. */
        char scratch[64];
        size_t len = (size_t)(field_end - start);
        /* Trim trailing whitespace. */
        while (len > 0 && (start[len - 1] == ' ' || start[len - 1] == '\t')) {
            len--;
        }
        /* Trim leading whitespace. */
        while (len > 0 && (*start == ' ' || *start == '\t')) {
            start++;
            len--;
        }

        if (len == 0) {
            /* Empty field: skip but keep counting? Treat as malformed. */
            if (*p == ',') {
                p++;
                p = skip_ws(p);
                continue;
            }
            break;
        }

        if (len >= sizeof(scratch)) {
            /* Field too long to be a meaningful float. */
            if (*p == ',') {
                p++;
                p = skip_ws(p);
                continue;
            }
            break;
        }

        memcpy(scratch, start, len);
        scratch[len] = '\0';

        char *endptr = NULL;
        float val = strtof(scratch, &endptr);
        if (endptr == scratch || *endptr != '\0' || !isfinite(val)) {
            return -1;
        }

        if (count < max_tokens) {
            out[count] = val;
        }
        count++;

        if (*p == ',') {
            p++;
            p = skip_ws(p);
        }
    }

    if (out_has_label) {
        *out_has_label = (count > (size_t)expect);
    }

    return (int)count;
}

/**
 * Split a line into up to max_fields whitespace-trimmed comma-separated
 * tokens (as raw strings). Records how many were found. Used to preserve
 * the input CSV's header column names for the output file.
 */
static int split_tokens(const char *line, char (*out)[64], size_t max_fields)
{
    size_t count = 0;
    const char *p = skip_ws(line);

    if (*p == '\0' || *p == '\n' || *p == '\r') {
        return 0;
    }

    while (*p != '\0' && *p != '\n' && *p != '\r' && count < max_fields) {
        const char *start = p;
        while (*p != '\0' && *p != ',' && *p != '\n' && *p != '\r') {
            p++;
        }
        const char *end = p;

        size_t len = (size_t)(end - start);
        while (len > 0 && (start[len - 1] == ' ' || start[len - 1] == '\t')) {
            len--;
        }
        while (len > 0 && (*start == ' ' || *start == '\t')) {
            start++;
            len--;
        }
        if (len >= 64) len = 63;
        memcpy(out[count], start, len);
        out[count][len] = '\0';
        count++;

        if (*p == ',') {
            p++;
            p = skip_ws(p);
        }
    }

    return (int)count;
}

int predict_batch(const Perceptron *p, const Scaler *scaler,
                  const char *csv_path, const char *out_csv_path)
{
    if (!p || !scaler || !csv_path) {
        fprintf(stderr, "ERROR: NULL argument to predict_batch.\n");
        return -1;
    }

    if (p->feature_count == 0) {
        fprintf(stderr, "ERROR: Perceptron has zero features.\n");
        return -1;
    }

    if (p->feature_count > MAX_FEATURES) {
        fprintf(stderr, "ERROR: Feature count %zu exceeds maximum %d.\n",
                p->feature_count, MAX_FEATURES);
        return -1;
    }

    if (scaler->num_features != p->feature_count) {
        fprintf(stderr, "ERROR: Scaler feature count %zu does not match model %zu.\n",
                scaler->num_features, p->feature_count);
        return -1;
    }

    if (!scaler->min_values || !scaler->max_values) {
        fprintf(stderr, "ERROR: Scaler has NULL min/max arrays.\n");
        return -1;
    }

    size_t expected = p->feature_count;

    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s'.\n", csv_path);
        return -1;
    }

    FILE *outfp = NULL;
    if (out_csv_path) {
        outfp = fopen(out_csv_path, "w");
        if (!outfp) {
            fprintf(stderr, "ERROR: Cannot open output file '%s' for writing.\n",
                    out_csv_path);
            fclose(fp);
            return -1;
        }
    }

    /* Determine header: attempt to parse first line. */
    char line[MAX_LINE_LENGTH];
    float first[MAX_FEATURES];
    int first_has_label = 0;
    int first_n;

    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "WARNING: Input file '%s' is empty.\n", csv_path);
        fclose(fp);
        if (outfp) fclose(outfp);
        return 0;
    }

    /* Token width to inspect: expected features plus optional label. */
    size_t inspect = expected + 1;

    first_n = parse_line(line, first, inspect, (int)expected, &first_has_label);
    int has_header = 0;
    if (first_n < 0) {
        /* Not fully numeric -> header row; skip it. */
        has_header = 1;
    } else if (!first_has_label && first_n != (int)expected) {
        /* Numeric but wrong column count -> treat as header and skip. */
        has_header = 1;
    } else if (first_has_label && first_n != (int)(expected + 1)) {
        has_header = 1;
    }

    /* Preserve the input header names (if any) for the output CSV. */
    char feat_names[MAX_FEATURES][64];
    int have_names = 0;
    if (has_header) {
        int ntok = split_tokens(line, feat_names, inspect);
        have_names = ((size_t)ntok >= expected);
    }

    /* Emit output CSV header. */
    if (outfp) {
        for (size_t i = 0; i < expected; i++) {
            if (have_names) {
                fprintf(outfp, "%s,", feat_names[i]);
            } else {
                fprintf(outfp, "feature%zu,", i + 1);
            }
        }
        fprintf(outfp, "actual,predicted\n");
    }

    printf("Row   ");
    for (size_t i = 0; i < expected; i++) {
        printf("  %12s", "Feature");
    }
    printf("  %8s  %10s\n", "Actual", "Predicted");
    printf("----- ");
    for (size_t i = 0; i < expected; i++) {
        printf("  ------------");
    }
    printf("  %8s  %10s\n", "-----", "---------");

    size_t row_num = 0;
    size_t processed = 0;
    int pass_count = 0;
    int fail_count = 0;

    /* A helper macro to process an already-parsed row. */
#define EMIT_ROW(feat, nvals, has_lbl, lbl) do {                              \
        float norm[MAX_FEATURES];                                             \
        for (size_t f = 0; f < expected; f++) {                               \
            float range = scaler->max_values[f] - scaler->min_values[f];      \
            norm[f] = (range < 1e-9f) ? 0.0f                                  \
                      : ((feat)[f] - scaler->min_values[f]) / range;          \
        }                                                                     \
        int pred = perceptron_forward(p, norm);                               \
        if (pred) pass_count++; else fail_count++;                            \
        processed++;                                                          \
        row_num++;                                                            \
        printf("%-5zu", row_num);                                             \
        for (size_t f = 0; f < expected; f++) {                               \
            printf("  %12.2f", (feat)[f]);                                    \
        }                                                                     \
        printf("  %8s  %10s\n", (has_lbl) ? ((lbl) ? "1" : "0") : "-",         \
               pred ? "PASS" : "FAIL");                                       \
        if (outfp) {                                                          \
            for (size_t f = 0; f < expected; f++) {                           \
                fprintf(outfp, "%.6g,", (feat)[f]);                            \
            }                                                                 \
            fprintf(outfp, "%s,%d\n",                                         \
                    (has_lbl) ? ((lbl) ? "1" : "0") : "?", pred);             \
        }                                                                     \
    } while (0)

    if (!has_header) {
        int lbl = first_has_label ? (first[expected] != 0.0f) : 0;
        EMIT_ROW(first, first_n, first_has_label, lbl);
    }

    while (fgets(line, sizeof(line), fp)) {
        float feat[MAX_FEATURES];
        int has_lbl = 0;
        int n = parse_line(line, feat, inspect, (int)expected, &has_lbl);

        if (n <= 0) {
            continue; /* blank / blank line */
        }
        if (has_lbl) {
            if (n != (int)(expected + 1)) {
                fprintf(stderr, "WARNING: Skipping row %zu (expected %zu or %zu columns, got %d).\n",
                        row_num + 1, expected, expected + 1, n);
                continue;
            }
        } else {
            if (n != (int)expected) {
                fprintf(stderr, "WARNING: Skipping row %zu (expected %zu columns, got %d).\n",
                        row_num + 1, expected, n);
                continue;
            }
        }

        int lbl = has_lbl ? (feat[expected] != 0.0f) : 0;
        EMIT_ROW(feat, n, has_lbl, lbl);
    }

#undef EMIT_ROW

    fclose(fp);
    if (outfp) {
        fprintf(outfp, "\n");
        fclose(outfp);
    }

    printf("\n----------------------------------------\n");
    printf("Rows processed: %zu  |  PASS: %d  |  FAIL: %d\n",
           row_num, pass_count, fail_count);
    printf("----------------------------------------\n");

    if (out_csv_path) {
        printf("Results written to: %s\n", out_csv_path);
    }

    return (int)processed;
}
