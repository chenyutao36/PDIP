
#include "stdlib.h"
#include <stdbool.h>
#include "mex.h"

void Block_Fill(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha){
       
    size_t i,j;
    size_t s;
    for (j=0;j<n;j++){
        s = idn*ldG + idm + j*ldG;
        for (i=0;i<m;i++){
            G[s+i] = alpha * Gi[j*m+i];
        }
    }
       
}

void Block_Fill_Trans(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha){
       
    size_t i,j;
    size_t s;
    for (j=0;j<m;j++){
        s = idn*ldG + idm + j*ldG;
        for (i=0;i<n;i++){
            G[s+i] = Gi[i*m+j];
        }
    }      
}

void Block_Access(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha){
       
    size_t i,j;
    size_t s;
    for (j=0;j<n;j++){
        s = idn*ldG + idm + j*ldG;
        for (i=0;i<m;i++){
            Gi[j*m+i]=G[s+i];
        }
    }
       
}

bool vec_bigger(size_t n, double *a, double *b){
    int i;
    size_t sum=0;
    bool flag = false;
    for(i=0;i<n;i++){
        if (a[i]>=b[i])
            sum += 1;
    }
    if (sum==n)
        flag = true;
    return flag;
}

void set_zeros(size_t n, double *a){
    int i;
    for (i=0;i<n;i++)
        a[i] = 0;
}

void regularization(size_t n, double *A, double reg){
    int i;
    for (i=0;i<n;i++)
        A[i*n+i] += reg;
}

void print_matrix(size_t m, size_t n, double *A, size_t ldA)
{
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++)
            mexPrintf("%4.2e ",A[j*ldA+i]);
        mexPrintf("\n");
    }
}

void print_vector(size_t n, double *a)
{
    int i;
    for(i=0;i<n;i++)      
        mexPrintf("%4.2e ",a[i]);
    mexPrintf("\n");
}

void print_vector_int(size_t n, int *a)
{
    int i;
    for(i=0;i<n;i++)      
        mexPrintf("%d ",a[i]);
    mexPrintf("\n");
}