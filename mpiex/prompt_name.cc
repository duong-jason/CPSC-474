#include "mpi.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    std::string myname;
    MPI_Init(&argc, &argv);

    std::cout << "Input your name, no spaces" << std::endl;
    std::cin >> myname;
    std::cout << "Hello, world! " << myname << std::endl;

    MPI_Finalize();
    return 0;
}
