#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int A[3] = {0, 0, 0};
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "I am rank " << rank << " of " << size << " processes" << std::endl;
    // std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1] << "," << A[2] << ")" << std::endl;

    if (rank == 0) {
        A[0] = A[1] = A[2] = -10;
        std::cout << "At rank " << rank << " array A[]=(" << A[0] << "," << A[1] << "," << A[2] << ")" << std::endl;
        MPI_Send(A, 3, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process " << rank << " sent array A[]=(" << A[0] << "," << A[1] << "," << A[2] << ")" << " to process 1" << std::endl;
    }
    else if (rank == 1) {
        std::cout << "At rank " << rank << " array A[]=(" << A[0] << "," << A[1] << "," << A[2] << ")" << std::endl;
        MPI_Recv(A, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " received array A[]=(" << A[0] << "," << A[1] << "," << A[2] << ")"  << " to process 0" << std::endl;
   }

    MPI_Finalize();
    return 0;
}
