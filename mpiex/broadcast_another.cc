#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int n = 100;    
    int a[100];    
    int i, number = -1;    
    int rank;    
    int size; 
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);    
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if ( rank == 0 ) 
        for ( i = 1; i <= n; i++ )     a[i-1] = i;    
    else {        
       // display the current content of a[0] 
       printf("Process %d, a[0] = %d\n", rank, a[0]);    
     }    
     MPI_Bcast(&a, n, MPI_INT, 0, MPI_COMM_WORLD); 
     if (rank !=0) {        
         // a[0] will contain now the value broadcast by process 0 
         printf("After broadcast, at process %d, a[0] = %d, a[1] = %d \n", rank, a[0], a[1]);
     }        
     MPI_Finalize();    
     return 0;
}
