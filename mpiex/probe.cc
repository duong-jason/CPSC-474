#include "mpi.h"
#include <ctime>
#include <iostream>

#define MAX_NUMBERS 100

int main(int argc, char *argv[]) {
    int rank, size;
    int numbers[MAX_NUMBERS], number_amount = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        srand(time(NULL));
        // Sends up to MAX_NUMBERS of integers to process one
        number_amount = (rand() / (float) RAND_MAX) * MAX_NUMBERS;
        // Send the amount of integers to process one
        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("0 sent %d numbers to 1\n", number_amount);
    } else if (rank == 1) {
        MPI_Status status;
        // Probe for an incoming message from process zero
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        // After receiving the message, check the status to determine how many numbers were actually received
        MPI_Get_count(&status, MPI_INT, &number_amount);
        // Allocate a buffer to hold the incoming numbers
        int* number_buf = (int*) malloc(sizeof(int) * number_amount);
        // Now receive the message with the allocated buffer
        MPI_Recv(number_buf, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        // Print off the amount of numbers, and also print additional information in the status object
        printf("1 dynamically received %d numbers from 0.\n", number_amount);
        free(number_buf);
    }

    MPI_Finalize();
    return 0;
}
