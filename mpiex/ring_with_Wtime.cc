#include "mpi.h"
#include <stdio.h>
#include <iostream>

int main(int argc, char *argv[]) {
    // int rank, size, token;
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int token[3] = {0, 0, 0};

    double start_time, stop_time;
    MPI_Barrier(MPI_COMM_WORLD);

    start_time = MPI_Wtime();

    if (rank != 0) {
        MPI_Recv(token, 3, MPI_INT, (rank-1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d,%d,%d from process %d\n", rank, *token, *(token+1), *(token+2), (rank-1) % size);
    } else {
        token[0] = token[1] = token[2] = -1;
    }

    MPI_Send(token, 3, MPI_INT, (rank+1) % size, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Recv(token, 3, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d,%d,%d from process %d\n", rank, *token, *(token+1), *(token+2), size-1);
    }
    
    stop_time = MPI_Wtime();

    if (rank == 0)
        std::cout << "Execution time for rank=" << rank << " is " << stop_time-start_time << std::endl;

    MPI_Finalize();
    return 0;
}
