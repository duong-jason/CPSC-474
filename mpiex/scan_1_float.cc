/*
 * Each process has one float (computed based on its rank),
 * then apply MPI-Scan to obtain the partial products of these floats
 * based on the process rank, and displays the reduced value at each process.
 * Let A[] be the send buffer with the two floats: 
 *  A[0] = rank*2 + 0.5
 *
 * Receive buffer is array B[]
 *
 * Scan command: send_data=A[], rcvd_data=B[], send_count=1, type = MPI_FLOAT, op=MPI_PROD
 */

#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float A[1]= {rank+1};
	float B[20] = {0.0, 0.0};

	std::cout << "Process " << rank << " has array A[]=(" << A[0] << "), and array B[]=(" << B[0] << ")\n" << std::endl;

    for (int i = 0; i < 20; i++) B[i] = 0.0;

	// std::cout << "Process " << rank << " has array A[]=(" << A[0]
    //           << "), and array B[]=("  << B[0] <<  "," << B[1] << "," << B[2] << "," << B[3] << "," << B[4] << ")\n" << std::endl;

	MPI_Scan(A, B, 1, MPI_FLOAT, MPI_PROD, MPI_COMM_WORLD);

	std::cout << "After partial parallel reducing, process " << rank << " has array B[]=(" << B[0]
              <<  "," << B[1] << "," << B[2] << "," << B[3] << "," << B[4] << ")\n" << std::endl;

	MPI_Finalize();
	return 0;
}
