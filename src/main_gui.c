/**
 * main_gui.c - CortexC GTK GUI
 *
 * Phase 4: Graphical user interface for the CortexC ML engine.
 * Provides tabs for dataset loading, model training, model management,
 * and prediction.
 *
 * Build: make gui  (requires GTK 3 development libraries)
 *
 * C17 standard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gtk/gtk.h>

#include "config.h"
#include "dataset.h"
#include "preprocessing.h"
#include "perceptron.h"
#include "prediction.h"
#include "evaluation.h"
#include "model_io.h"

/* ------------------------------------------------------------------ */
/*  Application state                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Main window */
    GtkWidget *window;

    /* Notebook (tab container) */
    GtkWidget *notebook;

    /* Dataset tab widgets */
    GtkWidget *dataset_path_entry;
    GtkWidget *dataset_info_label;
    GtkWidget *dataset_load_button;

    /* Training tab widgets */
    GtkWidget *train_button;
    GtkWidget *train_status_label;
    GtkWidget *train_output_textview;

    /* Model tab widgets */
    GtkWidget *model_path_entry;
    GtkWidget *model_info_label;
    GtkWidget *model_load_button;
    GtkWidget *model_save_button;

    /* Predict tab widgets */
    GtkWidget *predict_features_entry;
    GtkWidget *predict_button;
    GtkWidget *predict_result_label;

    /* State */
    Dataset    *current_dataset;
    Perceptron *current_model;
    Scaler      current_scaler;
    char        current_model_path[256];
    char        current_csv_path[256];
} AppData;

/* ------------------------------------------------------------------ */
/*  Utility: append text to a GtkTextView                               */
/* ------------------------------------------------------------------ */

static void textview_append(GtkTextView *textview, const char *text)
{
    GtkTextBuffer *buffer = gtk_text_view_get_buffer(textview);
    GtkTextIter end;
    gtk_text_buffer_get_end_iter(buffer, &end);
    gtk_text_buffer_insert(buffer, &end, text, -1);

    /* Auto-scroll to bottom */
    gtk_text_buffer_get_end_iter(buffer, &end);
    GtkTextMark *mark = gtk_text_buffer_create_mark(buffer, NULL, &end, FALSE);
    gtk_text_view_scroll_mark_onscreen(textview, mark);
}

static void textview_clear(GtkTextView *textview)
{
    GtkTextBuffer *buffer = gtk_text_view_get_buffer(textview);
    gtk_text_buffer_set_text(buffer, "", -1);
}

/* ------------------------------------------------------------------ */
/*  Dataset tab callbacks                                               */
/* ------------------------------------------------------------------ */

static void on_dataset_load(GtkWidget *button, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)button;

    const char *path = gtk_entry_get_text(GTK_ENTRY(app->dataset_path_entry));
    if (!path || path[0] == '\0') {
        gtk_label_set_text(GTK_LABEL(app->dataset_info_label),
                           "Error: Please enter a dataset path.");
        return;
    }

    /* Free previous dataset if any */
    if (app->current_dataset) {
        dataset_free(&app->current_dataset);
        app->current_dataset = NULL;
    }

    app->current_dataset = dataset_load_csv(path);
    if (!app->current_dataset) {
        gtk_label_set_text(GTK_LABEL(app->dataset_info_label),
                           "Error: Failed to load dataset.");
        return;
    }

    strncpy(app->current_csv_path, path, sizeof(app->current_csv_path) - 1);

    /* Build info string */
    char info[512];
    snprintf(info, sizeof(info),
             "Dataset loaded successfully!\n"
             "File:    %s\n"
             "Samples: %zu\n"
             "Features: %zu\n"
             "Labels:  %d",
             app->current_dataset->source_file,
             app->current_dataset->num_samples,
             app->current_dataset->num_features,
             app->current_dataset->label_count);

    gtk_label_set_text(GTK_LABEL(app->dataset_info_label), info);
}

/* ------------------------------------------------------------------ */
/*  Training tab callbacks                                              */
/* ------------------------------------------------------------------ */

static void on_train_clicked(GtkWidget *button, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)button;

    if (!app->current_dataset) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Load a dataset first.");
        return;
    }

    /* Clear previous output */
    textview_clear(GTK_TEXT_VIEW(app->train_output_textview));

    gtk_label_set_text(GTK_LABEL(app->train_status_label), "Training...");
    while (gtk_events_pending()) {
        gtk_main_iteration();
    }

    /* Split dataset */
    Dataset *train_set = NULL;
    Dataset *test_set = NULL;

    if (dataset_split(app->current_dataset, &train_set, &test_set,
                      DEFAULT_TRAIN_RATIO, DEFAULT_RANDOM_SEED) != 0) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Failed to split dataset.");
        return;
    }

    textview_append(GTK_TEXT_VIEW(app->train_output_textview),
                    "Dataset split complete.\n");
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Training samples: %zu\nTesting samples:  %zu\n\n",
                 train_set->num_samples, test_set->num_samples);
        textview_append(GTK_TEXT_VIEW(app->train_output_textview), buf);
    }

    /* Fit scaler */
    Scaler scaler = {0};
    if (scaler_fit(&scaler, train_set) != 0) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Failed to fit scaler.");
        dataset_free(&train_set);
        dataset_free(&test_set);
        return;
    }

    if (scaler_transform(&scaler, train_set) != 0 ||
        scaler_transform(&scaler, test_set) != 0) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Failed to normalize data.");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        return;
    }

    textview_append(GTK_TEXT_VIEW(app->train_output_textview),
                    "Data normalized.\n\n");

    /* Initialize perceptron */
    Perceptron *model = perceptron_init(train_set->num_features,
                                        DEFAULT_LEARNING_RATE,
                                        (size_t)DEFAULT_MAX_EPOCHS);
    if (!model) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Failed to create perceptron.");
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        return;
    }

    /* Train */
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Learning rate: %.3f\nMax epochs: %d\n\n",
                 DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS);
        textview_append(GTK_TEXT_VIEW(app->train_output_textview), buf);
    }

    /* Note: perceptron_train prints to stdout; we capture basic info here */
    int epochs = perceptron_train(model, train_set);
    if (epochs < 0) {
        gtk_label_set_text(GTK_LABEL(app->train_status_label),
                           "Error: Training failed.");
        perceptron_free(&model);
        scaler_free(&scaler);
        dataset_free(&train_set);
        dataset_free(&test_set);
        return;
    }

    {
        char buf[64];
        snprintf(buf, sizeof(buf), "\nTraining completed in %d epochs.\n\n", epochs);
        textview_append(GTK_TEXT_VIEW(app->train_output_textview), buf);
    }

    /* Evaluate */
    int *predictions = predict_dataset(model, test_set);
    if (predictions) {
        int *actuals = malloc(test_set->num_samples * sizeof(int));
        if (actuals) {
            for (size_t i = 0; i < test_set->num_samples; i++) {
                actuals[i] = test_set->samples[i].label;
            }

            double accuracy = calculate_accuracy(predictions, actuals,
                                                 test_set->num_samples);
            {
                char buf[128];
                snprintf(buf, sizeof(buf), "Accuracy: %.2f%%\n", accuracy);
                textview_append(GTK_TEXT_VIEW(app->train_output_textview), buf);
            }

            ConfusionMatrix cm = {0};
            if (calculate_confusion_matrix(predictions, actuals,
                                           test_set->num_samples, &cm) == 0) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "\n              Predicted\n"
                         "              0       1\n\n"
                         "Actual 0      %-8d%d\n"
                         "Actual 1      %-8d%d\n",
                         cm.tn, cm.fp, cm.fn, cm.tp);
                textview_append(GTK_TEXT_VIEW(app->train_output_textview), buf);
            }

            free(actuals);
        }
        free(predictions);
    }

    /* Store model in app state */
    if (app->current_model) {
        perceptron_free(&app->current_model);
    }
    scaler_free(&app->current_scaler);

    app->current_model = model;
    app->current_scaler = scaler;

    gtk_label_set_text(GTK_LABEL(app->train_status_label),
                       "Training complete! Model ready to save.");

    dataset_free(&train_set);
    dataset_free(&test_set);
}

/* ------------------------------------------------------------------ */
/*  Model tab callbacks                                                 */
/* ------------------------------------------------------------------ */

static void on_model_load(GtkWidget *button, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)button;

    const char *path = gtk_entry_get_text(GTK_ENTRY(app->model_path_entry));
    if (!path || path[0] == '\0') {
        gtk_label_set_text(GTK_LABEL(app->model_info_label),
                           "Error: Please enter a model path.");
        return;
    }

    /* Free previous model */
    if (app->current_model) {
        perceptron_free(&app->current_model);
    }
    scaler_free(&app->current_scaler);

    if (model_load(path, &app->current_model, &app->current_scaler) != 0) {
        gtk_label_set_text(GTK_LABEL(app->model_info_label),
                           "Error: Failed to load model.");
        return;
    }

    strncpy(app->current_model_path, path, sizeof(app->current_model_path) - 1);

    /* Build info string */
    char info[512];
    snprintf(info, sizeof(info),
             "Model loaded!\n"
             "File:     %s\n"
             "Features: %zu\n"
             "LR:       %.3f\n"
             "Epochs:   %zu\n"
             "Bias:     %.6f",
             path,
             app->current_model->feature_count,
             app->current_model->learning_rate,
             app->current_model->epochs,
             app->current_model->bias);

    gtk_label_set_text(GTK_LABEL(app->model_info_label), info);
}

static void on_model_save(GtkWidget *button, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)button;

    if (!app->current_model) {
        gtk_label_set_text(GTK_LABEL(app->model_info_label),
                           "Error: No model to save. Train or load one first.");
        return;
    }

    const char *path = gtk_entry_get_text(GTK_ENTRY(app->model_path_entry));
    if (!path || path[0] == '\0') {
        path = DEFAULT_MODEL_PATH;
    }

    if (model_save(path, app->current_model, &app->current_scaler) != 0) {
        gtk_label_set_text(GTK_LABEL(app->model_info_label),
                           "Error: Failed to save model.");
        return;
    }

    strncpy(app->current_model_path, path, sizeof(app->current_model_path) - 1);

    char msg[256];
    snprintf(msg, sizeof(msg), "Model saved to: %s", path);
    gtk_label_set_text(GTK_LABEL(app->model_info_label), msg);
}

/* ------------------------------------------------------------------ */
/*  Predict tab callbacks                                               */
/* ------------------------------------------------------------------ */

static void on_predict_clicked(GtkWidget *button, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)button;

    if (!app->current_model) {
        gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                           "Error: No model loaded. Load or train one first.");
        return;
    }

    const char *input = gtk_entry_get_text(GTK_ENTRY(app->predict_features_entry));
    if (!input || input[0] == '\0') {
        gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                           "Error: Enter feature values separated by commas.");
        return;
    }

    size_t expected = app->current_model->feature_count;

    /* Parse comma-separated features */
    float *raw = malloc(expected * sizeof(float));
    if (!raw) {
        gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                           "Error: Memory allocation failed.");
        return;
    }

    char *buffer = strdup(input);
    if (!buffer) {
        free(raw);
        gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                           "Error: Memory allocation failed.");
        return;
    }

    size_t count = 0;
    char *token = strtok(buffer, ",");
    while (token && count < expected) {
        char *endptr = NULL;
        raw[count] = strtof(token, &endptr);
        if (endptr == token) {
            free(buffer);
            free(raw);
            gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                               "Error: Invalid feature value.");
            return;
        }
        count++;
        token = strtok(NULL, ",");
    }

    free(buffer);

    if (count != expected) {
        free(raw);
        char msg[128];
        snprintf(msg, sizeof(msg), "Error: Expected %zu features, got %zu.",
                 expected, count);
        gtk_label_set_text(GTK_LABEL(app->predict_result_label), msg);
        return;
    }

    /* Normalize using model's scaler */
    float *norm = malloc(expected * sizeof(float));
    if (!norm) {
        free(raw);
        gtk_label_set_text(GTK_LABEL(app->predict_result_label),
                           "Error: Memory allocation failed.");
        return;
    }

    for (size_t i = 0; i < expected; i++) {
        float range = app->current_scaler.max_values[i] - app->current_scaler.min_values[i];
        if (range < 1e-9f) {
            norm[i] = 0.0f;
        } else {
            norm[i] = (raw[i] - app->current_scaler.min_values[i]) / range;
        }
    }

    /* Predict */
    int result = perceptron_predict(app->current_model, norm);

    char msg[256];
    snprintf(msg, sizeof(msg), "Prediction: %s", result ? "PASS" : "FAIL");
    gtk_label_set_text(GTK_LABEL(app->predict_result_label), msg);

    free(raw);
    free(norm);
}

/* ------------------------------------------------------------------ */
/*  Window destroy                                                     */
/* ------------------------------------------------------------------ */

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
    AppData *app = (AppData *)user_data;
    (void)widget;

    if (app->current_dataset) {
        dataset_free(&app->current_dataset);
    }
    if (app->current_model) {
        perceptron_free(&app->current_model);
    }
    scaler_free(&app->current_scaler);

    gtk_main_quit();
}

/* ------------------------------------------------------------------ */
/*  Build GUI                                                          */
/* ------------------------------------------------------------------ */

static void build_gui(AppData *app)
{
    app->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(app->window), "CortexC - CPU-Only ML Engine");
    gtk_window_set_default_size(GTK_WINDOW(app->window), 700, 500);
    g_signal_connect(app->window, "destroy", G_CALLBACK(on_window_destroy), app);

    app->notebook = gtk_notebook_new();
    gtk_container_add(GTK_CONTAINER(app->window), app->notebook);

    /* ---- Tab 1: Dataset ---- */
    {
        GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(vbox), 12);

        GtkWidget *label = gtk_label_new("Load Dataset");
        gtk_notebook_append_page(GTK_NOTEBOOK(app->notebook), vbox, label);

        /* Path entry row */
        GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

        GtkWidget *path_label = gtk_label_new("CSV Path:");
        gtk_box_pack_start(GTK_BOX(hbox), path_label, FALSE, FALSE, 0);

        app->dataset_path_entry = gtk_entry_new();
        gtk_entry_set_text(GTK_ENTRY(app->dataset_path_entry), "data/students.csv");
        gtk_entry_set_width_chars(GTK_ENTRY(app->dataset_path_entry), 40);
        gtk_box_pack_start(GTK_BOX(hbox), app->dataset_path_entry, TRUE, TRUE, 0);

        app->dataset_load_button = gtk_button_new_with_label("Load");
        g_signal_connect(app->dataset_load_button, "clicked",
                         G_CALLBACK(on_dataset_load), app);
        gtk_box_pack_start(GTK_BOX(hbox), app->dataset_load_button, FALSE, FALSE, 0);

        /* Info display */
        app->dataset_info_label = gtk_label_new("No dataset loaded.");
        gtk_label_set_xalign(GTK_LABEL(app->dataset_info_label), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(app->dataset_info_label), TRUE);
        gtk_box_pack_start(GTK_BOX(vbox), app->dataset_info_label, TRUE, TRUE, 0);
    }

    /* ---- Tab 2: Training ---- */
    {
        GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(vbox), 12);

        GtkWidget *label = gtk_label_new("Train Model");
        gtk_notebook_append_page(GTK_NOTEBOOK(app->notebook), vbox, label);

        /* Train button */
        app->train_button = gtk_button_new_with_label("Train Perceptron");
        g_signal_connect(app->train_button, "clicked",
                         G_CALLBACK(on_train_clicked), app);
        gtk_box_pack_start(GTK_BOX(vbox), app->train_button, FALSE, FALSE, 0);

        /* Status label */
        app->train_status_label = gtk_label_new("Load a dataset, then click Train.");
        gtk_label_set_xalign(GTK_LABEL(app->train_status_label), 0.0f);
        gtk_box_pack_start(GTK_BOX(vbox), app->train_status_label, FALSE, FALSE, 0);

        /* Output text view with scroll */
        GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                       GTK_POLICY_AUTOMATIC,
                                       GTK_POLICY_AUTOMATIC);
        gtk_box_pack_start(GTK_BOX(vbox), scroll, TRUE, TRUE, 0);

        app->train_output_textview = gtk_text_view_new();
        gtk_text_view_set_editable(GTK_TEXT_VIEW(app->train_output_textview), FALSE);
        gtk_text_view_set_monospace(GTK_TEXT_VIEW(app->train_output_textview), TRUE);
        gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(app->train_output_textview),
                                    GTK_WRAP_WORD_CHAR);
        gtk_container_add(GTK_CONTAINER(scroll), app->train_output_textview);
    }

    /* ---- Tab 3: Model ---- */
    {
        GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(vbox), 12);

        GtkWidget *label = gtk_label_new("Model");
        gtk_notebook_append_page(GTK_NOTEBOOK(app->notebook), vbox, label);

        /* Path entry row */
        GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

        GtkWidget *path_label = gtk_label_new("Model Path:");
        gtk_box_pack_start(GTK_BOX(hbox), path_label, FALSE, FALSE, 0);

        app->model_path_entry = gtk_entry_new();
        gtk_entry_set_text(GTK_ENTRY(app->model_path_entry),
                           DEFAULT_MODEL_PATH);
        gtk_entry_set_width_chars(GTK_ENTRY(app->model_path_entry), 40);
        gtk_box_pack_start(GTK_BOX(hbox), app->model_path_entry, TRUE, TRUE, 0);

        /* Load and Save buttons */
        app->model_load_button = gtk_button_new_with_label("Load");
        g_signal_connect(app->model_load_button, "clicked",
                         G_CALLBACK(on_model_load), app);
        gtk_box_pack_start(GTK_BOX(hbox), app->model_load_button, FALSE, FALSE, 0);

        app->model_save_button = gtk_button_new_with_label("Save");
        g_signal_connect(app->model_save_button, "clicked",
                         G_CALLBACK(on_model_save), app);
        gtk_box_pack_start(GTK_BOX(hbox), app->model_save_button, FALSE, FALSE, 0);

        /* Info display */
        app->model_info_label = gtk_label_new("No model loaded.");
        gtk_label_set_xalign(GTK_LABEL(app->model_info_label), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(app->model_info_label), TRUE);
        gtk_box_pack_start(GTK_BOX(vbox), app->model_info_label, TRUE, TRUE, 0);
    }

    /* ---- Tab 4: Predict ---- */
    {
        GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(vbox), 12);

        GtkWidget *label = gtk_label_new("Predict");
        gtk_notebook_append_page(GTK_NOTEBOOK(app->notebook), vbox, label);

        /* Features entry */
        GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

        GtkWidget *feat_label = gtk_label_new("Features (comma-separated):");
        gtk_box_pack_start(GTK_BOX(hbox), feat_label, FALSE, FALSE, 0);

        app->predict_features_entry = gtk_entry_new();
        gtk_entry_set_width_chars(GTK_ENTRY(app->predict_features_entry), 30);
        gtk_box_pack_start(GTK_BOX(hbox), app->predict_features_entry, TRUE, TRUE, 0);

        app->predict_button = gtk_button_new_with_label("Predict");
        g_signal_connect(app->predict_button, "clicked",
                         G_CALLBACK(on_predict_clicked), app);
        gtk_box_pack_start(GTK_BOX(hbox), app->predict_button, FALSE, FALSE, 0);

        /* Result display */
        app->predict_result_label = gtk_label_new("Load a model, then enter features.");
        gtk_label_set_xalign(GTK_LABEL(app->predict_result_label), 0.0f);
        gtk_box_pack_start(GTK_BOX(vbox), app->predict_result_label, TRUE, TRUE, 0);
    }
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    gtk_init(&argc, &argv);

    AppData app;
    memset(&app, 0, sizeof(app));

    build_gui(&app);

    gtk_widget_show_all(app.window);
    gtk_main();

    return EXIT_SUCCESS;
}
