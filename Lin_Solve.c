#include "mex.h"
#include "string.h"

#include "blas.h"
#include "lapack.h"

void Block_Fill(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG){
       
    size_t i,j;
    size_t s;
    for (j=0;j<n;j++){
        s = idn*ldG + idm + j*ldG;
        for (i=0;i<m;i++){
            G[s+i] = Gi[j*m+i];
        }
    }
       
}

// void Block_Fill_Trans(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG){
//        
//     size_t i,j;
//     size_t s;
//     for (j=0;j<m;j++){
//         s = idn*ldG + idm + j*ldG;
//         for (i=0;i<n;i++){
//             G[s+i] = Gi[i*m+j];
//         }
//     }      
// }

void Block_Access(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG){
       
    size_t i,j;
    size_t s;
    for (j=0;j<n;j++){
        s = idn*ldG + idm + j*ldG;
        for (i=0;i<m;i++){
            Gi[j*m+i]=G[s+i];
        }
    }
       
}


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{    
    double *A = mxGetPr(prhs[0]);
    double *B = mxGetPr(prhs[1]);
    double *phi = mxGetPr(prhs[2]);
    mwSize nx = mxGetScalar(prhs[3]);
    mwSize nu = mxGetScalar(prhs[4]);
    mwSize N = mxGetScalar(prhs[5]);   
    double *beta = mxGetPr(prhs[6]);
    
    mwSize nz = nx+nu;
    mwSize neq = (N+1)*nx; 
        
    plhs[0] = mxCreateDoubleMatrix(neq,1,mxREAL);
    double *sol=mxGetPr(plhs[0]);
    
    memcpy(sol,beta,neq*sizeof(double));
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    /* Allocate Memory */
    
    double *LY = mxMalloc((2*nx*nx*N+nx*nx)*sizeof(double));
        
    double *phi_i = mxMalloc(nz*nz*sizeof(double));
    double *phi_N = mxMalloc(nx*nx*sizeof(double));
            
    double *D = mxCalloc(nx*nz,sizeof(double));
    for (i=0;i<nx;i++)
        D[i*nx+i]= -1.0; // this is D_1
    
    double *DN = mxCalloc(nx*nx,sizeof(double));
    for (i=0;i<nx;i++)
        DN[i*nx+i]= -1.0; // this is D_N
    
    double *V = mxCalloc(nx*nz*(N+1),sizeof(double));    
    for (i=0;i<nx;i++)
        V[i*nx+i]= 1.0; // this is Z_{-1}
        
    double *W = mxCalloc(nx*nz*(N-1),sizeof(double));
       
    /* Compute V and W*/
    
     // facotr phi_0
    Block_Access(nz, nz, phi_i, phi, 0, 0, N*nz+nx);
    dpotrf(UPLO, &nz, phi_i, &nz, &INFO);
    
    // solve V_{-1}L_0^T = Z{-1}
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi_i, &nz, V, &nx);
        
    memcpy(V+nx*nz, A, nx*nx*sizeof(double));
    memcpy(V+nx*nz+nx*nx, B, nx*nu*sizeof(double));
    
    // solve V_0 L_0^T = Z_0
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi_i, &nz, V+nx*nz, &nx);
    
    for (i=1; i<N; i++){
        Block_Access(nz, nz, phi_i, phi, i*nz, i*nz, N*nz+nx);
        // facotr phi_i
        dpotrf(UPLO, &nz, phi_i, &nz, &INFO);
        
        memcpy(V+(i+1)*nx*nz, A+i*nx*nx, nx*nx*sizeof(double));
        memcpy(V+(i+1)*nx*nz+nx*nx, B+i*nx*nu, nx*nu*sizeof(double));

        // solve V_i L_i^T = Z_i
        dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi_i, &nz, V+(i+1)*nx*nz, &nx);
             
        // solve W_i L_i^T = D_i
        memcpy(W+(i-1)*nx*nz, D, nx*nz*sizeof(double));
        dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi_i, &nz, W+(i-1)*nx*nz, &nx);
    }
    Block_Access(nx, nx, phi_N, phi, N*nz, N*nz, N*nz+nx);
    dpotrf(UPLO, &nx, phi_N, &nx, &INFO);
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nx, &one_d, phi_N, &nx, DN, &nx);
    
    /* Compute Y, LY and forward solve*/
    
    // Y00
    dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, V, &nx, V, &nx, &zero_d, LY, &nx);
    
    // L00
    dpotrf(UPLO, &nx, LY, &nx, &INFO);
    
    // x0
    dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY, &nx, sol, &nx, &INFO);
    
    // Y01
    dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, V, &nx, V+nx*nz, &nx, &zero_d, LY+nx*nx, &nx);
    
    // L10^T
    dtrtrs(UPLO, noTRANS, DIAG, &nx, &nx, LY, &nx, LY+nx*nx, &nx, &INFO);
    
    for (i=1;i<N;i++){
        
        // Y(i,i)
        dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, V+i*nx*nz, &nx, V+i*nx*nz, &nx, &zero_d, LY+2*i*nx*nx, &nx);
        dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, W+(i-1)*nx*nz, &nx, W+(i-1)*nx*nz, &nx, &one_d, LY+2*i*nx*nx, &nx);
        
        // L(i,i)
        dgemm(TRANS, noTRANS, &nx, &nx, &nx, &minus_one_d, LY+(2*i-1)*nx*nx, &nx, LY+(2*i-1)*nx*nx, &nx, &one_d, LY+2*i*nx*nx, &nx);
        dpotrf(UPLO, &nx, LY+2*i*nx*nx, &nx, &INFO);
                
        // x(i)
        dgemv(TRANS, &nx, &nx, &minus_one_d, LY+(2*i-1)*nx*nx, &nx, sol+(i-1)*nx, &one_i, &one_d, sol+i*nx, &one_i);
        dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY+2*i*nx*nx, &nx, sol+i*nx, &nx, &INFO);
        
        // Y(i,i+1)
        dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, W+(i-1)*nz*nx, &nx, V+(i+1)*nz*nx, &nx, &zero_d, LY+(2*i+1)*nx*nx, &nx);
                
        // L(i+1,i)^T        
        dtrtrs(UPLO, noTRANS, DIAG, &nx, &nx, LY+2*i*nx*nx, &nx, LY+(2*i+1)*nx*nx, &nx, &INFO);
       
    }
    
    // Y(N,N)
    dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, V+N*nx*nz, &nx, V+N*nx*nz, &nx, &zero_d, LY+2*N*nx*nx, &nx);
    dgemm(noTRANS, TRANS, &nx, &nx, &nx, &one_d, DN, &nx, DN, &nx, &one_d, LY+2*N*nx*nx, &nx);
    
    // L(N,N)
    dgemm(TRANS, noTRANS, &nx, &nx, &nx, &minus_one_d, LY+(2*N-1)*nx*nx, &nx, LY+(2*N-1)*nx*nx, &nx, &one_d, LY+2*N*nx*nx, &nx);
    dpotrf(UPLO, &nx, LY+2*N*nx*nx, &nx, &INFO);
    
    // x(N)
    dgemv(TRANS, &nx, &nx, &minus_one_d, LY+(2*N-1)*nx*nx, &nx, sol+(N-1)*nx, &one_i, &one_d, sol+N*nx, &one_i);
    dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY+2*N*nx*nx, &nx, sol+N*nx, &nx, &INFO);
    
    /*Backward solve*/
    dtrtrs(UPLO, TRANS, DIAG, &nx, &one_i, LY+2*N*nx*nx, &nx, sol+N*nx, &nx, &INFO);
    for(i=N-1;i>-1;i--){
        dgemv(noTRANS, &nx, &nx, &minus_one_d, LY+(2*i+1)*nx*nx, &nx, sol+(i+1)*nx, &one_i, &one_d, sol+i*nx, &one_i);
        dtrtrs(UPLO, TRANS, DIAG, &nx, &one_i, LY+2*i*nx*nx, &nx, sol+i*nx, &nx, &INFO);
    }
}