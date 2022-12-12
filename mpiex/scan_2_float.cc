/*
 * Each process has two floats (computed based on its rank),
 * then apply MPI-Scan to obtain the partial products of these floats
 * based on the process rank, and displays the reduced value at each process.
 * Let A[] be the send buffer with the two floats: 
 *  A[0] = rank*2 + 0.5
 *  A[1] = rank*3 - 0.5
 *
 * Receive buffer is array B[]
 *
 * Scan command: send_data=A[], rcvd_data=B[], send_count=2, type = MPI_FLOAT, op=MPI_SUM
 */

#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float A[2]= {rank+1, rank+2};
	float B[20] = {0.0, 0.0};

	std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1] << "), and array B[]=(" << B[0] << ")\n" << std::endl;

    for (int i = 0; i < 20; i++) B[i] = 0.0;

	// std::cout << "Process " << rank << " has array A[]=(" << A[0] <<  "," << A[1]
    //           << "), and array B[]=("  << B[0] <<  "," << B[1] << ","
    //           << B[2] << ")\n" << std::endl;

	MPI_Scan(A, B, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	std::cout << "After partial parallel reducing, process " << rank
              << " has array B[]=(" << B[0] << "," << B[1] << ")\n" << std::endl;

	MPI_Finalize();
	return 0;
}
