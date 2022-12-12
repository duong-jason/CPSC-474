#include "mpi.h"
#include <iostream>

int main(int argc, char *argv[]) {
	int size, rank;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size); // size is obtained
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank is obtained

	std::cout << "Hello, world! from " << rank << " out of " << size << " processes" << std::endl;
	MPI_Finalize();
	return 0;
}
