#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, number = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "I am rank " << rank << " of " << size << " processes" << std::endl;
    std::cout << "Process " << rank << " has number=" << number << std::endl;

    if (rank == 1) {
        number = -10;
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Send(&number, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
        std::cout << "Process " << rank << " sent number=" << number << " to process 3" << std::endl;
    }
    else if (rank == 3) {
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Recv(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " received number=" << number << " from process 1" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
