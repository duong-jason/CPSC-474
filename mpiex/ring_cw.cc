#include "mpi.h"
#include <stdio.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, token[3] = {0, 0, 0};
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        MPI_Recv(token, 3, MPI_INT, (rank-1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d,%d,%d from process %d\n", rank, *token, *(token+1), *(token+2), (rank-1) % size);
    } else {
        token[0] = token[1] = token[2] = -1;
    }

    MPI_Send(token, 3, MPI_INT, (rank+1) % size, 0, MPI_COMM_WORLD); // last process -> (size-1) + 1 % size = 0

    if (rank == 0) {
        MPI_Recv(token, 3, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d,%d,%d from process %d\n", rank, *token, *(token+1), *(token+2), size-1);
    }
    
    MPI_Finalize();
    return 0;
}
