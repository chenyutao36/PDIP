#include "mex.h"
#include "stdlib.h"
#include "string.h"
#include "blockIP_common.h"
#include "common.h"
#include "blockIP_funcs.h"
#include "blas.h"
#include "lapack.h"

/* Functions */
void compute_phi(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t nz = dim->nz;
    size_t N =dim->N;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *H = work->H;
    double *HN = work->HN;
    double *mu = work->mu;
    double *s = work->s;
    
    double *hat_s_inv = work->hat_s_inv;
    double *hat_mu = work->hat_mu;
    double *tmp1 = work->tmp1;
    double *tmp2 = work->tmp2;
    
    double *hat_sN_inv = work->hat_sN_inv;
    double *hat_muN = work->hat_muN;
    double *tmp1N = work->tmp1N;
    double *tmp2N = work->tmp2N;
    
    double *phi = work->phi;
    double *phi_N = work->phi_N;
          
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    memcpy(phi, H, nz*nz*N*sizeof(double));
    
    for (i=0;i<N;i++){

        for (j=0;j<row_C;j++){
            hat_s_inv[j*row_C+j] = 1/s[i*row_C+j];
            hat_mu[j*row_C+j] = mu[i*row_C+j];
        }
        
        dgemm(TRANS, noTRANS, &nz, &row_C, &row_C, &one_d, Cz+i*row_C*nz, &row_C, hat_s_inv, &row_C, &zero_d, tmp1, &nz);
        dgemm(noTRANS, noTRANS, &nz, &row_C, &row_C, &one_d, tmp1, &nz, hat_mu, &row_C, &zero_d, tmp2, &nz);
        dgemm(noTRANS, noTRANS, &nz, &nz, &row_C, &one_d, tmp2, &nz, Cz+i*row_C*nz, &row_C, &one_d, phi+i*nz*nz, &nz);
                       
        dpotrf(UPLO, &nz, phi+i*nz*nz, &nz, &INFO);
        
        if (INFO<0){
            mexPrintf("the %d-th argument had an illegal value\n in phi_%d", -1*INFO, i);
            mexErrMsgTxt("Error occured when factorizing phi");
        }
        if (INFO>0){
            mexPrintf("the leading minor of order %d is not positive definite in phi_%d\n", INFO, i);
            mexErrMsgTxt("Error occured when factorizing phi");
        }
    }
    memcpy(phi_N, HN, nx*nx*sizeof(double));
    
    for (j=0;j<row_CN;j++){
        hat_sN_inv[j*row_CN+j] = 1/s[N*row_C+j];
        hat_muN[j*row_CN+j] = mu[N*row_C+j];
    }
    
    dgemm(TRANS, noTRANS, &nx, &row_CN, &row_CN, &one_d, CzN, &row_CN, hat_sN_inv, &row_CN, &zero_d, tmp1N, &nx);
    dgemm(noTRANS, noTRANS, &nx, &row_CN, &row_CN, &one_d, tmp1N, &nx, hat_muN, &row_CN, &zero_d, tmp2N, &nx);
    dgemm(noTRANS, noTRANS, &nx, &nx, &row_CN, &one_d, tmp2N, &nx, CzN, &row_CN, &one_d, phi_N, &nx);
          
    dpotrf(UPLO, &nx, phi_N, &nx, &INFO);
    
    if (INFO<0){
        mexPrintf("the %d-th argument had an illegal value in phi_N\n", -1*INFO, N);
        mexErrMsgTxt("Error occured when factorizing phi");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite in phi_N\n", INFO, N);
        mexErrMsgTxt("Error occured when factorizing phi");
    }
}

void compute_LY(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
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
    double *DN = work->DN;      
    double *V = work->V;         
    double *W = work->W;    
    double *LY = work->LY;
    
    for(i=0;i<nx;i++)
        DN[i*nx+i]= -1.0; 
    
    // solve V_{-1}L_0^T = Z{-1}
    for(i=0;i<nx;i++)
        V[i*nx+i]= 1.0; 
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi, &nz, V, &nx);
        
     // solve V_0 L_0^T = Z_0
    memcpy(V+nx*nz, A, nx*nx*sizeof(double));
    memcpy(V+nx*nz+nx*nx, B, nx*nu*sizeof(double));      
    dtrsm(SIDE, UPLO, TRANS, DIAG, &nx, &nz, &one_d, phi, &nz, V+nx*nz, &nx);
    
    for (i=1; i<N; i++){  
        // solve V_i L_i^T = Z_i
        memcpy(V+(i+1)*nx*nz, A+i*nx*nx, nx*nx*sizeof(double));
        memcpy(V+(i+1)*nx*nz+nx*nx, B+i*nx*nu, nx*nu*sizeof(double));        
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
        mexPrintf("the %d-th argument had an illegal value in Y_00\n", -1*INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite in Y_00\n", INFO);
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
            mexPrintf("the %d-th argument had an illegal value in Y_%d%d\n", -1*INFO, i, i);
            mexErrMsgTxt("Error occured when factorizing Y");
        }
        if (INFO>0){
            mexPrintf("the leading minor of order %d is not positive definite in Y_%d%d\n", INFO, i, i);
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
        mexPrintf("the %d-th argument had an illegal value in Y_NN\n", -1*INFO);
        mexErrMsgTxt("Error occured when factorizing Y");
    }
    if (INFO>0){
        mexPrintf("the leading minor of order %d is not positive definite in Y_NN\n", INFO);
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

void compute_rC(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t N =dim->N;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *rc = work->rC;
    double *H = work->H;
    double *HN = work->HN;
    double *g = work->g;
    double *w = work->w;
    double *lambda = work->lambda;
    double *mu = work->mu;
    
    memcpy(rc, g, nw*sizeof(double));
            
    dgemv(noTRANS, &nz, &nz, &one_d, H, &nz, w, &one_i, &one_d, rc, &one_i);
    dgemv(TRANS, &row_C, &nz, &one_d, Cz, &row_C, mu, &one_i, &one_d, rc, &one_i);
    
    dgemv(TRANS, &nx, &nx, &one_d, A, &nx, lambda+nx, &one_i, &one_d, rc, &one_i);    
    daxpy(&nx, &one_d, lambda, &one_i,rc, &one_i);
    
    dgemv(TRANS, &nx, &nu, &one_d, B, &nx, lambda+nx, &one_i, &one_d, rc+nx, &one_i);
    
    
    for (i=1;i<N;i++){
        
        dgemv(noTRANS, &nz, &nz, &one_d, H+i*nz*nz, &nz, w+i*nz, &one_i, &one_d, rc+i*nz, &one_i);
        dgemv(TRANS, &row_C, &nz, &one_d, Cz+i*row_C*nz, &row_C, mu+i*row_C, &one_i, &one_d, rc+i*nz, &one_i);
        
        dgemv(TRANS, &nx, &nx, &one_d, A+i*nx*nx, &nx, lambda+(i+1)*nx, &one_i, &one_d, rc+i*nz, &one_i);        
        daxpy(&nx,&minus_one_d,lambda+i*nx,&one_i,rc+i*nz,&one_i);
        
        dgemv(TRANS, &nx, &nu, &one_d, B+i*nx*nu, &nx, lambda+(i+1)*nx, &one_i, &one_d, rc+i*nz+nx, &one_i);
    }
        
    dgemv(noTRANS, &nx, &nx, &one_d, HN, &nx, w+N*nz, &one_i, &one_d, rc+N*nz, &one_i);
    dgemv(TRANS, &row_CN, &nx, &one_d, CzN, &row_CN, mu+N*row_C, &one_i, &one_d, rc+N*nz, &one_i);
    daxpy(&nx, &minus_one_d, lambda+N*nx, &one_i, rc+N*nz, &one_i);
    
}

void compute_rE(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t neq = dim->neq;
    
    double *b = work->b;
    double *w = work->w;
    double *rE = work->rE;
    
    int i;    
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    
    memcpy(rE, b, neq*sizeof(double));
    for (i=0;i<nx;i++)
        rE[i] += w[i]; 
 
    for (i=1;i<N+1;i++){
        dgemv(noTRANS, &nx, &nx, &one_d, A+(i-1)*nx*nx, &nx, w+(i-1)*nz, &one_i, &one_d, rE+i*nx, &one_i);
        dgemv(noTRANS, &nx, &nu, &one_d, B+(i-1)*nx*nu, &nx, w+(i-1)*nz+nx, &one_i, &one_d, rE+i*nx, &one_i);
        daxpy(&nx, &minus_one_d, w+i*nz, &one_i, rE+i*nx, &one_i);
    }
}

void compute_rI(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double *w = work->w;
    double *s = work->s;
    double *mu = work->mu;
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *c = work->c;
    double *rI = work->rI;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
        
    memcpy(rI, c, nineq*sizeof(double));
    daxpy(&nineq, &one_d, s, &one_i, rI, &one_i);
 
    for (i=0;i<N;i++){
        dgemv(noTRANS, &row_C, &nz, &one_d, Cz+i*row_C*nz, &row_C, w+i*nz, &one_i, &one_d, rI+i*row_C, &one_i);
    }
    dgemv(noTRANS, &row_CN, &nx, &one_d, CzN, &row_CN, w+N*nz, &one_i, &one_d, rI+N*row_C, &one_i);
  
}

void compute_rs(pdip_dims *dim, pdip_workspace *work)
{
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N = dim->N;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double sigma = work->sigma;
    double t = work->t;
    double *dmu = work->dmu;
    double *ds = work->ds;
    
    double *s = work->s;
    double *mu = work->mu;
    double *rs = work->rs;
    
    int i,j;
           
    for (i=0;i<N;i++){
        for(j=0;j<row_C;j++)
            rs[i*row_C+j] = s[i*row_C+j]*mu[i*row_C+j] - sigma*t + ds[i*row_C+j]*dmu[i*row_C+j];
    }
    for(j=0;j<row_CN;j++)
        rs[N*row_C+j] = s[N*row_C+j]*mu[N*row_C+j] - sigma*t + ds[N*row_C+j]*dmu[N*row_C+j];
    
}

void compute_rd(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double *s = work->s;
    double *mu = work->mu;
    
    double *hat_s_inv = work->hat_s_inv;
    double *hat_mu = work->hat_mu;
    double *tmp = work->tmp1;
    double *hat_sN_inv = work->hat_sN_inv;
    double *hat_muN = work->hat_muN;
    double *tmpN = work->tmp1N;    
    double *e = work->e;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
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
        
    memcpy(rd, rC, nw*sizeof(double));
    memcpy(e, rs, nineq*sizeof(double));
    
    for (i=0;i<N;i++){
        for(j=0;j<row_C;j++){
            hat_s_inv[j*row_C+j]=1/s[i*row_C+j];
            hat_mu[j*row_C+j]=mu[i*row_C+j];
        }
        dgemm(TRANS, noTRANS, &nz, &row_C, &row_C, &minus_one_d, Cz+i*row_C*nz, &row_C, hat_s_inv, &row_C, &zero_d, tmp, &nz);        
        dgemv(noTRANS, &row_C, &row_C, &minus_one_d, hat_mu, &row_C, rI+i*row_C, &one_i, &one_d, e+i*row_C, &one_i);       
        dgemv(noTRANS, &nz, &row_C, &one_d, tmp, &nz, e+i*row_C, &one_i, &one_d, rd+i*nz, &one_i);        
    }
    for(j=0;j<row_CN;j++){
        hat_sN_inv[j*row_CN+j]=1/s[N*row_C+j];
        hat_muN[j*row_CN+j]=mu[N*row_C+j];
    }
    dgemm(TRANS, noTRANS, &nx, &row_CN, &row_CN, &minus_one_d, CzN, &row_CN, hat_sN_inv, &row_CN, &zero_d, tmpN, &nx);        
    dgemv(noTRANS, &row_CN, &row_CN, &minus_one_d, hat_muN, &row_CN, rI+N*row_C, &one_i, &one_d, e+N*row_C, &one_i);       
    dgemv(noTRANS, &nx, &row_CN, &one_d, tmpN, &nx, e+N*row_C, &one_i, &one_d, rd+N*nz, &one_i);        
    
}

void compute_beta(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
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
    
    double *D = work->D;
    
    /* solve for p */
    for (i=0;i<N;i++){
        dpotrs(UPLO, &nz, &one_i, phi+i*nz*nz, &nz, rd+i*nz, &nz, &INFO);
    }
    dpotrs(UPLO, &nx, &one_i, phi_N, &nx, rd+N*nz, &nx, &INFO);
    
    /* Compute beta */
    memcpy(beta, rE, neq*sizeof(double));
       
    dgemv(noTRANS, &nx, &nz, &one_d, D, &nx, rd, &one_i, &one_d, beta, &one_i); 
    
    for (i=1;i<N;i++){        
        
        dgemv(noTRANS, &nx, &nx, &minus_one_d, A+(i-1)*nx*nx, &nx, rd+(i-1)*nz, &one_i, &one_d, beta+i*nx, &one_i); 
        dgemv(noTRANS, &nx, &nu, &minus_one_d, B+(i-1)*nx*nu, &nx, rd+(i-1)*nz+nx, &one_i, &one_d, beta+i*nx, &one_i); 
        
        dgemv(noTRANS, &nx, &nz, &minus_one_d, D, &nx, rd+i*nz, &one_i, &one_d, beta+i*nx, &one_i); 
    }     
    dgemv(noTRANS, &nx, &nx, &minus_one_d, A+(N-1)*nx*nx, &nx, rd+(N-1)*nz, &one_i, &one_d, beta+N*nx, &one_i); 
    dgemv(noTRANS, &nx, &nu, &minus_one_d, B+(N-1)*nx*nu, &nx, rd+(N-1)*nz+nx, &one_i, &one_d, beta+N*nx, &one_i); 
    daxpy(&nx, &one_d, rd+N*nz, &one_i, beta+N*nx, &one_i);
        
}

void recover_dw(double *A, double *B, pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
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

void recover_dmu(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double *s = work->s;
    double *mu = work->mu;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *rI = work->rI;
    double *rs = work->rs;
    double *dw = work->dw;
    double *dmu = work->dmu;
       
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
   
    memcpy(dmu, rI, nineq*sizeof(double));
    
    for(i=0;i<N;i++){
        dgemv(noTRANS, &row_C, &nz, &one_d, Cz+i*row_C*nz, &row_C, dw+i*nz, &one_i, &one_d, dmu+i*row_C, &one_i);              
        for (j=0;j<row_C;j++){
            dmu[i*row_C+j] *= 1/s[i*row_C+j] * mu[i*row_C+j] ;
            dmu[i*row_C+j] -= 1/s[i*row_C+j] * rs[i*row_C+j];
        }
    }
    dgemv(noTRANS, &row_CN, &nx, &one_d, CzN, &row_CN, dw+N*nz, &one_i, &one_d, dmu+N*row_C, &one_i);
    for (j=0;j<row_CN;j++){
        dmu[N*row_C+j] *= 1/s[N*row_C+j] * mu[N*row_C+j] ;
        dmu[N*row_C+j] -= 1/s[N*row_C+j] * rs[N*row_C+j];
    }
    
}

void recover_ds(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *rI = work->rI;
    double *dw = work->dw;
    double *ds = work->ds;
    
    int i,j;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
       
    memcpy(ds, rI, nineq*sizeof(double));
    
    for(i=0;i<N;i++){
        dgemv(noTRANS, &row_C, &nz, &minus_one_d, Cz+i*row_C*nz, &row_C, dw+i*nz, &one_i, &minus_one_d, ds+i*row_C, &one_i);
    }
    dgemv(noTRANS, &row_CN, &nx, &minus_one_d, CzN, &row_CN, dw+N*nz, &one_i, &minus_one_d, ds+N*row_C, &one_i);
    
}

void compute_fval(pdip_dims *dim, pdip_workspace *work)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t N =dim->N;
    size_t nz = dim->nz;
    
    double *H = work->H;
    double *HN = work->HN;
    double *g = work->g;
    double *w = work->w;
    
    double *fval = work->fval;
    
    int i;
    mwSize INFO;
    char *UPLO="L", *TRANS="T", *noTRANS="N", *DIAG="N", *SIDE="R";
    double one_d = 1.0, zero_d=0.0, half = 0.5;
    mwSize one_i = 1;
       
    memcpy(fval,g,(N*nz+nx)*sizeof(double));
        
    for (i=0;i<N;i++){        
        dgemv(noTRANS, &nz, &nz, &half, H+i*nz*nz, &nz, w+i*nz, &one_i, &one_d, fval+i*nz, &one_i);
    }
            
    dgemv(noTRANS, &nx, &nx, &half, HN, &nx, w+N*nz, &one_i, &one_d, fval+N*nz, &one_i);   
    
}

void recover_sol(pdip_dims *dim, pdip_workspace *work, 
        double *dz, double *dxN, double *lambda, double *mu, double *muN, double *mu_u)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nz = dim->nz;
    size_t N =dim->N;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t neq = dim->neq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
    int *nbu_idx = dim->nbu_idx;
    
    int i,j;
    
    memcpy(dz, work->w, N*nz*sizeof(double));
    memcpy(dxN, work->w+N*nz, nx*sizeof(double));
    memcpy(lambda, work->lambda, neq*sizeof(double));
    
    for (i=0;i<N;i++){
        for(j=0;j<nc;j++)
            mu[i*nc+j] = work->mu[i*row_C+j] - work->mu[i*row_C+nc+nbu+j];
        for(j=0;j<nbu;j++)
            mu_u[i*nu+nbu_idx[j]] = work->mu[i*row_C+nc+j] - work->mu[i*row_C+nc+nbu+nc+j];
    }
    
    for(j=0;j<ncN;j++)
        mu[N*nc+j] = work->mu[N*row_C+j] - work->mu[N*row_C+ncN+j];
    
}