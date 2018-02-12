#include "mex.h"
#include "stdlib.h"
#include "string.h"
#include "pdip_common.h"
#include "common.h"
#include "funcs.h"
#include "blas.h"
#include "lapack.h"

/* Functions */
void compute_phi(double *Q, double *S, double *R, double *C, double *s, double *mu, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nz = dim->nz;
    size_t N =dim->N;
    
    double *Ci = work->Ci;
    double *hat_s_inv = work->hat_s_inv;
    double *hat_mu = work->hat_mu;
    double *tmp1 = work->tmp1;
    double *tmp2 = work->tmp2;
    
    double *CN = work->CN;
    double *hat_sN_inv = work->hat_sN_inv;
    double *hat_muN = work->hat_muN;
    double *tmp1N = work->tmp1N;
    double *tmp2N = work->tmp2N;
    
    double *phi = work->phi;
    double *phi_N = work->phi_N;
    
    double reg = work->reg;
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    for (i=0;i<N;i++){
        Block_Fill(nx,nx,Q+i*nx*nx,phi+i*nz*nz,0,0,nz);
        Block_Fill(nx,nu,S+i*nx*nu,phi+i*nz*nz,0,nx,nz);
        Block_Fill_Trans(nx,nu,S+i*nx*nu,phi+i*nz*nz,nx,0,nz);
        Block_Fill(nu,nu,R+i*nu*nu,phi+i*nz*nz,nx,nx,nz);
        
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN);
        for (j=0;j<nc;j++){
            hat_s_inv[j*nc+j] = 1/s[i*nc+j];
            hat_mu[j*nc+j] = mu[i*nc+j];
        }
        
        dgemm(TRANS, noTRANS, &nz, &nc, &nc, &one_d, Ci, &nc, hat_s_inv, &nc, &zero_d, tmp1, &nz);
        dgemm(noTRANS, noTRANS, &nz, &nc, &nc, &one_d, tmp1, &nz, hat_mu, &nc, &zero_d, tmp2, &nz);
        dgemm(noTRANS, noTRANS, &nz, &nz, &nc, &one_d, tmp2, &nz, Ci, &nc, &one_d, phi+i*nz*nz, &nz);
        
        regularization(nz, phi+i*nz*nz, reg);
               
        dpotrf(UPLO, &nz, phi+i*nz*nz, &nz, &INFO);
        
        if (INFO<0){
            mexPrintf("the %d-th argument had an illegal value\n in %d-th block of phi", -1*INFO, i);
            mexErrMsgTxt("Error occured when factorizing phi");
        }
        if (INFO>0){
            mexPrintf("the leading minor of order %d is not positive definite in %d-th block of phi, and the factorization could not be completed\n", INFO, i);
            mexErrMsgTxt("Error occured when factorizing phi");
        }
    }
    memcpy(phi_N, Q+N*nx*nx, nx*nx*sizeof(double));
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);
    for (j=0;j<ncN;j++){
        hat_sN_inv[j*ncN+j] = 1/s[N*nc+j];
        hat_muN[j*ncN+j] = mu[N*nc+j];
    }
    
    dgemm(TRANS, noTRANS, &nx, &ncN, &ncN, &one_d, CN, &ncN, hat_sN_inv, &ncN, &zero_d, tmp1N, &nx);
    dgemm(noTRANS, noTRANS, &nx, &ncN, &ncN, &one_d, tmp1N, &nx, hat_muN, &ncN, &zero_d, tmp2N, &nx);
    dgemm(noTRANS, noTRANS, &nx, &nx, &ncN, &one_d, tmp2N, &nx, CN, &ncN, &one_d, phi_N, &nx);
    
    regularization(nx, phi_N, reg);
    
    dpotrf(UPLO, &nx, phi_N, &nx, &INFO);
    if (INFO<0){
        mexPrintf("the %d-th argument had an illegal value in %d-th block of phi\n", -1*INFO,N);
        mexErrMsgTxt("Error occured when factorizing phi_N");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite in %d-th block of phi, and the factorization could not be completed\n", INFO, N);
        mexErrMsgTxt("Error occured when factorizing phi_N");
    }
}

void compute_LY(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nz = dim->nz;
    size_t N =dim->N;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *phi = work->phi;
    double *phi_N = work->phi_N;
    
    double *D = work->D;
    for (i=0;i<nx;i++)
        D[i*nx+i]= -1.0; // this is D_1    
    double *DN = work->DN;
    for (i=0;i<nx;i++)
        DN[i*nx+i]= -1.0; // this is D_N    
    double *V = work->V;
    for (i=0;i<nx;i++)
        V[i*nx+i]= 1.0; // this is Z_{-1}        
    double *W = work->W;    
    double *LY = work->LY;
    
    // solve V_{-1}L_0^T = Z{-1}
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi, &nz, V, &nx);
        
    memcpy(V+nx*nz, A, nx*nx*sizeof(double));
    memcpy(V+nx*nz+nx*nx, B, nx*nu*sizeof(double));
    
    // solve V_0 L_0^T = Z_0
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi, &nz, V+nx*nz, &nx);
    
    for (i=1; i<N; i++){        
        memcpy(V+(i+1)*nx*nz, A+i*nx*nx, nx*nx*sizeof(double));
        memcpy(V+(i+1)*nx*nz+nx*nx, B+i*nx*nu, nx*nu*sizeof(double));

        // solve V_i L_i^T = Z_i
        dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi+i*nz*nz, &nz, V+(i+1)*nx*nz, &nx);
             
        // solve W_i L_i^T = D_i
        memcpy(W+(i-1)*nx*nz, D, nx*nz*sizeof(double));
        dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi+i*nz*nz, &nz, W+(i-1)*nx*nz, &nx);
    }
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nx, &one_d, phi_N, &nx, DN, &nx);
    
    /* Compute Y and LY*/
    
    // Y00
    dgemm(noTRANS, TRANS, &nx, &nx, &nz, &one_d, V, &nx, V, &nx, &zero_d, LY, &nx);
    
    // L00
    dpotrf(UPLO, &nx, LY, &nx, &INFO);    
    if (INFO<0){
        mexPrintf("the %d-th argument had an illegal value\n", -1*INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite, and the factorization could not be completed\n", INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
        
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
        if (INFO<0){
            mexPrintf("the %d-th argument had an illegal value\n", -1*INFO);
            mexErrMsgTxt("Error occured when factorizing Y");
        }
        if (INFO>0){
            mexPrintf("the leading minor of order %d is not positive definite, and the factorization could not be completed\n", INFO);
            mexErrMsgTxt("Error occured when factorizing Y");
        }
                        
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
    if (INFO<0){
        mexPrintf("the %d-th argument had an illegal value\n", -1*INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite, and the factorization could not be completed\n", INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
}

void lin_solve(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nz = dim->nz;
    size_t N =dim->N;
    
    double *LY = work->LY;
    double *sol = work->dlambda;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    /*Forward solve*/
    dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY, &nx, sol, &nx, &INFO);    
    for (i=1;i<N;i++){
        dgemv(TRANS, &nx, &nx, &minus_one_d, LY+(2*i-1)*nx*nx, &nx, sol+(i-1)*nx, &one_i, &one_d, sol+i*nx, &one_i);
        dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY+2*i*nx*nx, &nx, sol+i*nx, &nx, &INFO);
    }
    dgemv(TRANS, &nx, &nx, &minus_one_d, LY+(2*N-1)*nx*nx, &nx, sol+(N-1)*nx, &one_i, &one_d, sol+N*nx, &one_i);
    dtrtrs(UPLO, noTRANS, DIAG, &nx, &one_i, LY+2*N*nx*nx, &nx, sol+N*nx, &nx, &INFO);
    
    /*Backward solve*/
    dtrtrs(UPLO, TRANS, DIAG, &nx, &one_i, LY+2*N*nx*nx, &nx, sol+N*nx, &nx, &INFO);
    for(i=N-1;i>-1;i--){
        dgemv(noTRANS, &nx, &nx, &minus_one_d, LY+(2*i+1)*nx*nx, &nx, sol+(i+1)*nx, &one_i, &one_d, sol+i*nx, &one_i);
        dtrtrs(UPLO, TRANS, DIAG, &nx, &one_i, LY+2*i*nx*nx, &nx, sol+i*nx, &nx, &INFO);
    }
}

void compute_rC(double *Q, double *S, double *R, double *A, double *B, double *C,
        double *g, double *w, double *lambda, double *mu, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nz = dim->nz;
    size_t N =dim->N;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Ci = work->Ci;
    double *CN = work->CN;
    double *rc = work->rC;
    
    memcpy(rc,g,(N*nz+nx)*sizeof(double));
    Block_Access(nc,nz,Ci,C,0,0,N*nc+ncN);
    
    dgemv(noTRANS, &nx, &nx, &one_d, Q, &nx, w, &one_i, &one_d, rc, &one_i);
    dgemv(noTRANS, &nx, &nu, &one_d, S, &nx, w+nx, &one_i, &one_d, rc, &one_i);
    dgemv(TRANS, &nx, &nx, &one_d, A, &nx, lambda+nx, &one_i, &one_d, rc, &one_i);
    dgemv(TRANS, &nc, &nx, &one_d, Ci, &nc, mu, &one_i, &one_d, rc, &one_i);
    daxpy(&nx,&one_d,lambda,&one_i,rc,&one_i);
    
    dgemv(TRANS, &nx, &nu, &one_d, S, &nx, w, &one_i, &one_d, rc+nx, &one_i);
    dgemv(noTRANS, &nu, &nu, &one_d, R, &nu, w+nx, &one_i, &one_d, rc+nx, &one_i);
    dgemv(TRANS, &nx, &nu, &one_d, B, &nx, lambda+nx, &one_i, &one_d, rc+nx, &one_i);
    dgemv(TRANS, &nc, &nu, &one_d, Ci+nc*nx, &nc, mu, &one_i, &one_d, rc+nx, &one_i);
    
    for (i=1;i<N;i++){
        
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN);
        
        dgemv(noTRANS, &nx, &nx, &one_d, Q+i*nx*nx, &nx, w+i*nz, &one_i, &one_d, rc+i*nz, &one_i);
        dgemv(noTRANS, &nx, &nu, &one_d, S+i*nx*nu, &nx, w+i*nz+nx, &one_i, &one_d, rc+i*nz, &one_i);
        dgemv(TRANS, &nx, &nx, &one_d, A+i*nx*nx, &nx, lambda+(i+1)*nx, &one_i, &one_d, rc+i*nz, &one_i);
        dgemv(TRANS, &nc, &nx, &one_d, Ci, &nc, mu+i*nc, &one_i, &one_d, rc+i*nz, &one_i);
        daxpy(&nx,&minus_one_d,lambda+i*nx,&one_i,rc+i*nz,&one_i);
        
        dgemv(TRANS, &nx, &nu, &one_d, S+i*nx*nu, &nx, w+i*nz, &one_i, &one_d, rc+i*nz+nx, &one_i);
        dgemv(noTRANS, &nu, &nu, &one_d, R+i*nu*nu, &nu, w+i*nz+nx, &one_i, &one_d, rc+i*nz+nx, &one_i);
        dgemv(TRANS, &nx, &nu, &one_d, B+i*nx*nu, &nx, lambda+(i+1)*nx, &one_i, &one_d, rc+i*nz+nx, &one_i);
        dgemv(TRANS, &nc, &nu, &one_d, Ci+nc*nx, &nc, mu+i*nc, &one_i, &one_d, rc+i*nz+nx, &one_i);
    }
        
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);
    dgemv(noTRANS, &nx, &nx, &one_d, Q+N*nx*nx, &nx, w+N*nz, &one_i, &one_d, rc+N*nz, &one_i);
    dgemv(TRANS, &ncN, &nx, &one_d, CN, &ncN, mu+N*nc, &one_i, &one_d, rc+N*nz, &one_i);
    daxpy(&nx,&minus_one_d,lambda+N*nx,&one_i,rc+N*nz,&one_i);
    
}

void compute_rE(double *A, double *B, double *w, double *b, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t neq = dim->neq;
    
    double *rE = work->rE;
    
    int i;    
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    memcpy(rE, b, neq*sizeof(double));
    for (i=1;i<nx;i++)
        rE[i] += w[i]; 
 
    for (i=1;i<N+1;i++){
        dgemv(noTRANS, &nx, &nx, &one_d, A+(i-1)*nx*nx, &nx, w+(i-1)*nz, &one_i, &one_d, rE+i*nx, &one_i);
        dgemv(noTRANS, &nx, &nu, &one_d, B+(i-1)*nx*nu, &nx, w+(i-1)*nz+nx, &one_i, &one_d, rE+i*nx, &one_i);
        daxpy(&nx, &minus_one_d, w+i*nz, &one_i, rE+i*nx, &one_i);
    }
}

void compute_rI(double *C, double *c, double *w, double *mu, double *s, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    double *rI = work->rI;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Ci = work->Ci;
    double *CN = work->CN;
    
    memcpy(rI, c, nineq*sizeof(double));
    daxpy(&nineq,&one_d,s,&one_i,rI,&one_i);
 
    for (i=0;i<N;i++){
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN);
        dgemv(noTRANS, &nc, &nz, &one_d, Ci, &nc, w+i*nz, &one_i, &one_d, rI+i*nc, &one_i);
    }
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);
    dgemv(noTRANS, &ncN, &nx, &one_d, CN, &ncN, w+N*nz, &one_i, &one_d, rI+N*nc, &one_i);
  
}

void compute_rs(double *mu, double *s, pdip_dims *dim, pdip_workspace *work)
{
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    
    double sigma = work->sigma;
    double t = work->t;
    double *dmu = work->dmu;
    double *ds = work->ds;
    
    double *rs = work->rs;
    
    int i,j;
           
    for (i=0;i<N;i++){
        for(j=0;j<nc;j++)
            rs[i*nc+j] = s[i*nc+j]*mu[i*nc+j] - sigma*t + ds[i*nc+j]*dmu[i*nc+j];
    }
    for(j=0;j<ncN;j++)
        rs[N*nc+j] = s[N*nc+j]*mu[N*nc+j] - sigma*t + ds[N*nc+j]*dmu[N*nc+j];
    
}

void compute_rd(double *C, double *c, double *mu, double *s, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    double *Ci = work->Ci;
    double *hat_s_inv = work->hat_s_inv;
    double *hat_mu = work->hat_mu;
    double *tmp = work->tmp1;   
    double *CN = work->CN;
    double *hat_sN_inv = work->hat_sN_inv;
    double *hat_muN = work->hat_muN;
    double *tmpN = work->tmp1N;    
    double *e = work->e;
    
    double *rI = work->rI;
    double *rs = work->rs;
    double *rC = work->rC;
    double *dmu = work->dmu;
    double *ds = work->ds;
    double *rd = work->rd;
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
        
    memcpy(rd, rC, (N*nz+nx)*sizeof(double));
    memcpy(e, rs, nineq*sizeof(double));
    
    for (i=0;i<N;i++){
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN); 
        for(j=0;j<nc;j++){
            hat_s_inv[j*nc+j]=1/s[i*nc+j];
            hat_mu[j*nc+j]=mu[i*nc+j];
        }
        dgemm(TRANS, noTRANS, &nz, &nc, &nc, &minus_one_d, Ci, &nc, hat_s_inv, &nc, &zero_d, tmp, &nz);        
        dgemv(noTRANS, &nc, &nc, &minus_one_d, hat_mu, &nc, rI+i*nc, &one_i, &one_d, e+i*nc, &one_i);       
        dgemv(noTRANS, &nz, &nc, &one_d, tmp, &nz, e+i*nc, &one_i, &one_d, rd+i*nz, &one_i);        
    }
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);  
    for(j=0;j<ncN;j++){
        hat_sN_inv[j*ncN+j]=1/s[N*nc+j];
        hat_muN[j*ncN+j]=mu[N*nc+j];
    }
    dgemm(TRANS, noTRANS, &nx, &ncN, &ncN, &minus_one_d, CN, &ncN, hat_sN_inv, &ncN, &zero_d, tmpN, &nx);        
    dgemv(noTRANS, &ncN, &ncN, &minus_one_d, hat_muN, &ncN, rI+N*nc, &one_i, &one_d, e+N*nc, &one_i);       
    dgemv(noTRANS, &nx, &ncN, &one_d, tmpN, &nx, e+N*nc, &one_i, &one_d, rd+N*nz, &one_i);        
    
}

void compute_beta(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t neq = dim->neq;
    
    double *rE = work->rE;
    double *rd = work->rd;
    double *phi = work->phi;
    double *phi_N = work->phi_N;
    double *beta = work->dlambda;
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Z = work->D;
    for (i=0;i<nx;i++)
        Z[i*nx+i] = 1.0;
    
    /* solve for p */
    for (i=0;i<N;i++){
        dpotrs(UPLO, &nz, &one_i, phi+i*nz*nz, &nz, rd+i*nz, &nz, &INFO);
    }
    dpotrs(UPLO, &nx, &one_i, phi_N, &nx, rd+N*nz, &nx, &INFO);
    
    /* Compute beta */
    memcpy(beta, rE, neq*sizeof(double));
       
    dgemv(noTRANS, &nx, &nz, &minus_one_d, Z, &nx, rd, &one_i, &one_d, beta, &one_i); 
    
    for (i=1;i<N;i++){        
        
        dgemv(noTRANS, &nx, &nx, &minus_one_d, A+(i-1)*nx*nx, &nx, rd+(i-1)*nz, &one_i, &one_d, beta+i*nx, &one_i); 
        dgemv(noTRANS, &nx, &nu, &minus_one_d, B+(i-1)*nx*nu, &nx, rd+(i-1)*nz+nx, &one_i, &one_d, beta+i*nx, &one_i); 
        
        dgemv(noTRANS, &nx, &nz, &one_d, Z, &nx, rd+i*nz, &one_i, &one_d, beta+i*nx, &one_i); 
    }     
    dgemv(noTRANS, &nx, &nx, &minus_one_d, A+(N-1)*nx*nx, &nx, rd+(N-1)*nz, &one_i, &one_d, beta+N*nx, &one_i); 
    dgemv(noTRANS, &nx, &nu, &minus_one_d, B+(N-1)*nx*nu, &nx, rd+(N-1)*nz+nx, &one_i, &one_d, beta+N*nx, &one_i); 
    daxpy(&nx, &one_d, rd+N*nz, &one_i, beta+N*nx, &one_i);
        
}

void recover_dw(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    
    double *rd = work->rd;
    double *phi = work->phi;
    double *phi_N = work->phi_N;
    double *dlambda = work->dlambda;
    double *dw = work->dw;
    
    int i;  
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    memcpy(dw, dlambda, nx*sizeof(double));
    
    dgemv(TRANS, &nx, &nx, &minus_one_d, A, &nx, dlambda+nx, &one_i, &minus_one_d, dw, &one_i); 
    dgemv(TRANS, &nx, &nu, &minus_one_d, B, &nx, dlambda+nx, &one_i, &zero_d, dw+nx, &one_i);
    dpotrs(UPLO, &nz, &one_i, phi, &nz, dw, &nz, &INFO);
    
    for (i=1;i<N;i++){
        memcpy(dw+i*nz, dlambda+i*nx, nx*sizeof(double));
        dgemv(TRANS, &nx, &nx, &minus_one_d, A+i*nx*nx, &nx, dlambda+(i+1)*nx, &one_i, &one_d, dw+i*nz, &one_i); 
        dgemv(TRANS, &nx, &nu, &minus_one_d, B+i*nx*nu, &nx, dlambda+(i+1)*nx, &one_i, &zero_d, dw+i*nz+nx, &one_i);
        dpotrs(UPLO, &nz, &one_i, phi+i*nz*nz, &nz, dw+i*nz, &nz, &INFO);
    }
    memcpy(dw+N*nz, dlambda+N*nx, nx*sizeof(double));
    dpotrs(UPLO, &nx, &one_i, phi_N, &nx, dw+N*nz, &nx, &INFO);   
    
    daxpy(&nw, &minus_one_d, rd, &one_i, dw, &one_i);
}

void recover_dmu(double *C, double *mu, double *s, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    double *rI = work->rI;
    double *rs = work->rs;
    double *dw = work->dw;
    double *dmu = work->dmu;
       
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Ci = work->Ci;
    double *CN = work->CN;
        
    memcpy(dmu, rI, nineq*sizeof(double));
    
    for(i=0;i<N;i++){
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN);
        dgemv(noTRANS, &nc, &nz, &one_d, Ci, &nc, dw+i*nz, &one_i, &one_d, dmu+i*nc, &one_i);              
        for (j=0;j<nc;j++){
            dmu[i*nc+j] *= 1/s[i*nc+j] * mu[i*nc+j] ;
            dmu[i*nc+j] -= 1/s[i*nc+j] * rs[i*nc+j];
        }
    }
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);
    dgemv(noTRANS, &ncN, &nx, &one_d, CN, &ncN, dw+N*nz, &one_i, &one_d, dmu+N*nc, &one_i);
    for (j=0;j<ncN;j++){
        dmu[N*nc+j] *= 1/s[N*nc+j] * mu[N*nc+j] ;
        dmu[N*nc+j] -= 1/s[N*nc+j] * rs[N*nc+j];
    }
    
}

void recover_ds(double *C, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    double *rI = work->rI;
    double *dw = work->dw;
    double *ds = work->ds;
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Ci = work->Ci;
    double *CN = work->CN;
    
    memcpy(ds, rI, nineq*sizeof(double));
    
    for(i=0;i<N;i++){
        Block_Access(nc,nz,Ci,C,i*nc,i*nz,N*nc+ncN);
        dgemv(noTRANS, &nc, &nz, &minus_one_d, Ci, &nc, dw+i*nz, &one_i, &minus_one_d, ds+i*nc, &one_i);
    }
    Block_Access(ncN,nx,CN,C,N*nc,N*nz,N*nc+ncN);
    dgemv(noTRANS, &ncN, &nx, &minus_one_d, CN, &ncN, dw+N*nz, &one_i, &minus_one_d, ds+N*nc, &one_i);
    
}

void compute_fval(double *Q, double *S, double *R, double *g, double *w, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    
    double *fval = work->fval;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, half = 0.5;
    mwSize one_i = 1;
       
    memcpy(fval,g,(N*nz+nx)*sizeof(double));
        
    for (i=0;i<N;i++){
        dgemv(noTRANS, &nx, &nx, &half, Q+i*nx*nx, &nx, w+i*nz, &one_i, &one_d, fval+i*nz, &one_i);
        dgemv(noTRANS, &nx, &nu, &half, S+i*nx*nu, &nx, w+i*nz+nx, &one_i, &one_d, fval+i*nz, &one_i);
        
        dgemv(TRANS, &nx, &nu, &half, S+i*nx*nu, &nx, w+i*nz, &one_i, &one_d, fval+i*nz+nx, &one_i);
        dgemv(noTRANS, &nu, &nu, &half, R+i*nu*nu, &nu, w+i*nz+nx, &one_i, &one_d, fval+i*nz+nx, &one_i);
    }
            
    dgemv(noTRANS, &nx, &nx, &half, Q+N*nx*nx, &nx, w+N*nz, &one_i, &one_d, fval+N*nz, &one_i);   
    
}