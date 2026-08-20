/**
 * main.c - C-AI Entry Point
 *
 * This is the initial program for the C-AI project.
 * It prints a banner showing project info and confirms
 * the application is ready for future development phases.
 *
 * C17 standard
 */

#include <stdio.h>

#define C_AI_VERSION "0.1.0"

int main(void)
{
    printf("========================================\n");
    printf("              C-AI\n");
    printf("     CPU-ONLY MACHINE LEARNING\n");
    printf("========================================\n");
    printf("\n");
    printf("Version: %s\n", C_AI_VERSION);
    printf("Execution: CPU\n");
    printf("GPU: Disabled\n");
    printf("RAM: System Memory\n");
    printf("Storage: SSD/HDD\n");
    printf("\n");
    printf("Project initialized successfully.\n");

    return 0;
}
