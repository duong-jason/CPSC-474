#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, size;
    int A[5] = {0, 0, 0, 0, 0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // std::cout << "I am rank " << rank << " of " << size << " processes \n";   

    // std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1]
    //           << "," << A[2] <<  "," << A[3] << "," << A[4] << ")" << std::endl;

    if (rank == 0) {
        for (int i = 0; i < 5; i++) A[i] = i;
    }

    // std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1]
    //           << "," << A[2] <<  "," << A[3] << "," << A[4] << ")" << std::endl;

    MPI_Bcast(A, 3, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1]
              << "," << A[2] <<  "," << A[3] << "," << A[4] << ")" << std::endl;

    MPI_Finalize();
    return 0;
}
