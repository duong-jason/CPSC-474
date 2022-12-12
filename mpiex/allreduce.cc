#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])   {    
    int n = 10 ;
    int a[10];  
    int b[2];  
    int i;    
    int rank;    
    int size;        
    MPI_Init(&argc,&argv);     
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);    
    MPI_Comm_size(MPI_COMM_WORLD,&size);  
    b[0]=b[1]=rank;
    for (i=0; i< n; i++) a[i] = 0;
    printf(" Process of rank %d has array a=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d] and b[]=[%d,%d] \n", rank, a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],b[0],b[1]);  
    MPI_Allreduce(b, a, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  
     printf(" Process of rank %d has array a=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d] and b[]=[%d,%d] \n", rank, a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],b[0],b[1]);  
     MPI_Finalize();    
     return 0;
}

