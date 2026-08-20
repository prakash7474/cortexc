/**
 * config.h - Global configuration for C-AI
 *
 * Defines constants, default values, and configuration structures.
 * Will be fully implemented in Phase 2+.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

/* Project version */
#define C_AI_VERSION "0.1.0"

/* Default training parameters */
#define DEFAULT_LEARNING_RATE  0.1
#define DEFAULT_MAX_EPOCHS     1000
#define DEFAULT_TRAIN_RATIO    0.8
#define DEFAULT_RANDOM_SEED    42

/* Limits */
#define MAX_LINE_LENGTH        4096
#define MAX_FEATURES           256
#define MAX_FILENAME           256

/* Numeric tolerance for floating-point comparison */
#define EPSILON  1e-9

/* TODO: Implement TrainConfig and TrainStatus in Phase 2 */

#endif /* CONFIG_H */
