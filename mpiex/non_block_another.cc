#include "mpi.h"
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, number = 0, recv_number = 0;

    MPI_Request ireq1, ireq2;
    MPI_Status istatus;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Process " << rank << " has number=" << number << std::endl;

    if (rank == 0) {
        number = -10;
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Isend(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &ireq1);
        MPI_Isend(&number, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &ireq2);
        std::cout << "Process " << rank << " sent number=" << number << " to process 1" << std::endl;
    } else if (rank == 1) {
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Isend(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &ireq1);
        MPI_Isend(&number, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &ireq2);
        MPI_Irecv(&recv_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &ireq1);
        MPI_Wait(&ireq1, &istatus); // force waiting until receive is complete
        std::cout << "Process " << rank << " received number=" << number << " from process 0" << std::endl;
    } else { // rank >= 2
        std::cout << "At rank " << rank << " number=" << number << " from process 0" << std::endl;
        MPI_Irecv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &ireq1);
        MPI_Irecv(&recv_number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &ireq2);
        std::cout << "Process " << rank << " received number=" << number << " from process 0" << std::endl;
        std::cout << "Process " << rank << " received number=" << number << " from process 1" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
