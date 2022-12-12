/*
 * Each process has 2 floats (computed based on its rank)
 * Then, process of rank 3 reduces these floats by multiplying,
 * one from each other process and displays the collected reduced value
 *
 * Let A[] be the send buffer with one floats:
 *  A[0] = rank*2 + 0.5 
 *
 * Let B[] be the recv buffer
 *
 * Reduced command: send_data=A[], send_count=1, type=MPI_FLOAT, recv_data=B[], root=3
 */

#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float A[2]= {2*rank + 0.5, 3*rank - 0.5};
	float B[20] = {0.0, 0.0};

	std::cout << "Process " << rank << " has array A[]=(" << A[0] << "), and array B[]=(" << B[0] << ")\n" << std::endl;

	if (rank == 3)
		for (int i = 0; i < 20; i++) B[i] = 0.0;

	std::cout << "Process " << rank << " has array A[]=(" << A[0] <<  ","
              << A[1] << "), and array B[]=("  << B[0] <<  "," << B[1] << "," << B[2] << ")\n" << std::endl;

	MPI_Reduce(A, B, 2, MPI_FLOAT, MPI_PROD, 3, MPI_COMM_WORLD);

	std::cout << "After reducing, process " << rank << " has array B[]=(" << B[0]
              << "," << B[1] << "," << B[2] << ")\n" << std::endl;

	MPI_Finalize();
	return 0;
}
