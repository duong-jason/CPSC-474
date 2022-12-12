#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size, number = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, stop_time;
    MPI_Barrier(MPI_COMM_WORLD);
    
    start_time = MPI_Wtime();

    if (rank == 0) {
        number = -10;
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process " << rank << " sent number=" << number << " to process 1" << std::endl;
    }
    else if (rank == 1) {
        std::cout << "At rank " << rank << " number=" << number << std::endl;
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " received number=" << number << " from process 0" << std::endl;
    }

    stop_time = MPI_Wtime();
    std::cout << "Execution time for rank=" << rank << " is " << stop_time-start_time << std::endl;

    MPI_Finalize();
    return 0;
}
