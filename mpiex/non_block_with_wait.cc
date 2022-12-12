#include "mpi.h"
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, number = 0;

    MPI_Request ireq;
    MPI_Status istatus;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        number = 10;
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Isend(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &ireq);
        std::cout << "Process " << rank << " sent number=" << number << " to process 1" << std::endl;
    } else if (rank == 1) {
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Irecv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &ireq);
        MPI_Wait(&ireq, &istatus); // force waiting until receive is complete
        std::cout << "Process " << rank << " received number=" << number << " from process 0" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
