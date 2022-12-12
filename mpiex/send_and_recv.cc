#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, number = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // std::cout << "I am rank " << rank << " of " << size << " processes" << std::endl;

    if (rank == 0) {
        number = -10;
        printf("10\n");
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("20\n");
    }
    else if (rank == 1) {
        printf("11\n");
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("21\n");
    }

    MPI_Finalize();
    return 0;
}
