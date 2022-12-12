/*
 * Each process has 2 floats (computed based on its rank)
 * Then, process of rank 3 collects these two floats from each other process and displays the collected values
 *
 * Let A[] be the send buffer with two floats:
 *  A[0] = rank*2 + 0.5 
 *  A[1] = rank*3 - 0.5
 *
 * Let B[] be the recv buffer
 *
 * Gather command: send_data=A, send_count=2, type=MPI_FLOAT, recv_count=2, recv_data=B, root=3
 */

#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// std::cout << "I am rank " << rank << " of " << size << " processe \n";    

	float A[3]= {2*rank+0.5, 3*rank-0.5, 4*rank-0.5};
	float B[20] = {0.0,0.0};

	std::cout << "Process " << rank << " has array A[]=(" << A[0] << "," << A[1] << ")\n" << std::endl;

	if (rank == 4) 
	    for (int i = 0; i < 20; i++) B[i] = 0.0;

	std::cout << "Process " << rank << " has array A[]=(" << A[0] << ","
              << A[1] << "), and array B[]=(" << B[0] <<  "," << B[1]
              << "," << B[2] << "," << B[3] <<  "," << B[4] << "," << B[5] << ")\n" << std::endl;

	MPI_Gather(A, 2, MPI_FLOAT, B, 3, MPI_FLOAT, 2, MPI_COMM_WORLD);

	std::cout << "After gathering, process " << rank << " has array B[]=(" << std::endl;
	for (int i = 0; i < 15; ++i) {
		if (i % (rank+1) == 0){
			std::cout << "  ";
		}
		std::cout << B[i] << ",";
	}
	printf("\n");

	MPI_Finalize();
	return 0;
}

