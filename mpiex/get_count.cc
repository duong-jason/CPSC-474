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

    if (rank == 1) {
        // Pick a random amount of integers \in [0, 100] to send to process one
        srand(time(NULL));
        // Sends up to MAX_NUMBERS of integers to process one
        number_amount = (rand() / (float) RAND_MAX) * MAX_NUMBERS;
        // Send the amount of integers to process one
        MPI_Send(numbers, number_amount, MPI_INT, 3, 0, MPI_COMM_WORLD);
        printf("0 sent %d numbers to 1\n", number_amount);
    } else if (rank == 3) {
        MPI_Status status;
        // Receive at most MAX_NUMBERS from process zero
        MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        // After receiving the message, check the status to determine how many numbers were actually received
        MPI_Get_count(&status, MPI_INT, &number_amount);
        // Print off the amount of numbers, and also print additional information in the status object
        printf("3 received %d numbers from 0.\nMessage source = %d, tag = %d\n", number_amount, status.MPI_SOURCE, status.MPI_TAG);
    }

    MPI_Finalize();
    return 0;
}
