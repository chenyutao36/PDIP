#include "mex.h"
#include "string.h"
#include "funcs.h"
#include "blas.h"
#include "math.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{    
    double *Q = mxGetPr(prhs[0]);
    double *S = mxGetPr(prhs[1]);
    double *R = mxGetPr(prhs[2]);
    double *A = mxGetPr(prhs[3]);
    double *B = mxGetPr(prhs[4]);
    double *C = mxGetPr(prhs[5]);
    double *g = mxGetPr(prhs[6]);
    double *b = mxGetPr(prhs[7]);
    double *c = mxGetPr(prhs[8]);  
    mwSize nx = mxGetScalar(prhs[9]);
    mwSize nu = mxGetScalar(prhs[10]);
    mwSize nc = mxGetScalar(prhs[11]);
    mwSize ncN = mxGetScalar(prhs[12]);
    mwSize N = mxGetScalar(prhs[13]);
    
    mwSize nz = nx+nu;
    mwSize nw = N*nz+nx;
    mwSize neq = (N+1)*nx; 
    mwSize nineq = N*nc+ncN; 
    
    int i;
    
    plhs[0] = mxCreateDoubleMatrix(nw,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(neq,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nineq,1,mxREAL);
    plhs[3] = mxCreateDoubleMatrix(nineq,1,mxREAL);
    double *w = mxGetPr(plhs[0]);
    double *lambda = mxGetPr(plhs[1]);
    double *mu = mxGetPr(plhs[2]);
    double *s = mxGetPr(plhs[3]);    
    for (i=0;i<nineq;i++){
        s[i] = 10;
        mu[i] = 10;
    }
               
    /* Allocate Memory */        
    double *phi = mxMalloc(nz*nz*N*sizeof(double));
    double *phi_N = mxMalloc(nx*nx*sizeof(double));
    double *LY = mxMalloc((2*nx*nx*N+nx*nx)*sizeof(double));
    
    double *rC = mxMalloc(nw*sizeof(double));
    double *rE = mxMalloc(neq*sizeof(double));
    double *rI = mxMalloc(nineq*sizeof(double));
    double *rs = mxMalloc(nineq*sizeof(double));
    double *rd = mxMalloc(nw*sizeof(double));
        
    double *dw = mxMalloc(nw*sizeof(double));
    double *dlambda = mxMalloc(neq*sizeof(double));   
    double *dmu = mxMalloc(nineq*sizeof(double));
    double *ds = mxMalloc(nineq*sizeof(double));
    
    double *s_new = mxMalloc(nineq*sizeof(double));
    double *mu_new = mxMalloc(nineq*sizeof(double));
    
    /* Define constants*/    
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    double sigma, t, t_aff, tau=0.8, sca;
    double alpha, alpha_aff, alpha_pri_tau, alpha_dual_tau;
    int it_max = 100;
    double measure = 1E+4;
    double tol = 1E-8;
    int it=0;
        
    /* Start loop*/
    while (it<it_max && measure>tol){    
        sigma=0; t=0;      
        set_zeros(nineq, dmu);
        set_zeros(nineq, ds);

        /* Compute residuals */
        compute_rC(Q, S, R, A, B, C, g, w, lambda, mu, nx, nu, nc, ncN, N, rC);  
        compute_rE(A, B, w, b, nx, nu, N, rE);    
        compute_rI(C, c, w, mu, s, nx, nu, nc, ncN, N, rI);     
        compute_rs(mu, s, dmu, ds, sigma, t, nx, nu, nc, ncN, N, rs);
        compute_rd(C, c, mu, s, rI, rC, rs, dmu, ds, nx, nu, nc, ncN, N, rd); 

        /* Compute phi and its Cholesky factor L, stored in phi */
        compute_phi(Q, S, R, C, s, mu, phi, phi_N, nx, nu, nc, ncN, N);  

        /* Compute beta*/
        /* on exit, rd is the solution of phi_k^{-1} r_d^k */
        compute_beta(A, B, rE, rd, phi, phi_N, nx, nu, N, dlambda);     

        /* Compute Y and its Cholesky factor LY, stored in LY */
        compute_LY(phi, phi_N, A, B, LY, nx, nu, N);

        /* Solve the normal equation */
        lin_solve(LY, dlambda, nx, nu, N); 

        /* Recover primal solution */
        recover_dw(A, B, rd, phi, phi_N, dlambda, nx, nu, N, dw);    
        recover_dmu(C, mu, s, rI, dw, rs, nx, nu, nc, ncN, N, dmu);    
        recover_ds(C, rI, dw, nx, nu, nc, ncN, N, ds);

        /* Choose parameters*/
        t = ddot(&nineq, s, &one_i, mu, &one_i)/nineq;
        alpha_aff = 1;
        memcpy(s_new, s, nineq*sizeof(double));
        memcpy(mu_new, mu, nineq*sizeof(double));
        daxpy(&nineq, &alpha_aff, ds, &one_i, s_new, &one_i);
        daxpy(&nineq, &alpha_aff, dmu, &one_i, mu_new, &one_i);
        t_aff = ddot(&nineq, s_new, &one_i, mu_new, &one_i);
        while (t_aff<=0 && alpha_aff>1E-12){
            alpha_aff *= 0.95;
            memcpy(s_new, s, nineq*sizeof(double));
            memcpy(mu_new, mu, nineq*sizeof(double));
            daxpy(&nineq, &alpha_aff, ds, &one_i, s_new, &one_i);
            daxpy(&nineq, &alpha_aff, dmu, &one_i, mu_new, &one_i);
            t_aff = ddot(&nineq, s_new, &one_i, mu_new, &one_i);
        }
        t_aff = t_aff/nineq;
        sigma = pow((t_aff/t),3);
        
        /* Corrector */
        compute_rs(mu, s, dmu, ds, sigma, t, nx, nu, nc, ncN, N, rs);
        compute_rd(C, c, mu, s, rI, rC, rs, dmu, ds, nx, nu, nc, ncN, N, rd);
        compute_beta(A, B, rE, rd, phi, phi_N, nx, nu, N, dlambda);   
        lin_solve(LY, dlambda, nx, nu, N); 
        recover_dw(A, B, rd, phi, phi_N, dlambda, nx, nu, N, dw);    
        recover_dmu(C, mu, s, rI, dw, rs, nx, nu, nc, ncN, N, dmu);    
        recover_ds(C, rI, dw, nx, nu, nc, ncN, N, ds);

        /* Step length selection*/
        sca = 1-tau;
        
        alpha_pri_tau=1;
        memcpy(s_new, s, nineq*sizeof(double));
        memcpy(mu_new, s, nineq*sizeof(double));
        daxpy(&nineq, &alpha_pri_tau, ds, &one_i, s_new, &one_i);
        dscal(&nineq, &sca, mu_new, &one_i);
        while (!vec_bigger(nineq, s_new, mu_new) && alpha_pri_tau>1E-12){
            alpha_pri_tau *= 0.95;
            memcpy(s_new, s, nineq*sizeof(double));
            memcpy(mu_new, s, nineq*sizeof(double));
            daxpy(&nineq, &alpha_pri_tau, ds, &one_i, s_new, &one_i);
            dscal(&nineq, &sca, mu_new, &one_i);
        }

        alpha_dual_tau=1;
        memcpy(s_new, mu, nineq*sizeof(double));
        memcpy(mu_new, mu, nineq*sizeof(double));
        daxpy(&nineq, &alpha_dual_tau, dmu, &one_i, s_new, &one_i);
        dscal(&nineq, &sca, mu_new, &one_i);
        while (!vec_bigger(nineq, s_new, mu_new) && alpha_dual_tau>1E-12){
            alpha_dual_tau *= 0.95;
            memcpy(s_new, mu, nineq*sizeof(double));
            memcpy(mu_new, mu, nineq*sizeof(double));
            daxpy(&nineq, &alpha_dual_tau, dmu, &one_i, s_new, &one_i);
            dscal(&nineq, &sca, mu_new, &one_i);
        }

        alpha = MIN(alpha_pri_tau, alpha_dual_tau);
        
        /* Update solution */
        daxpy(&nw, &alpha, dw, &one_i, w, &one_i);
        daxpy(&neq, &alpha, dlambda, &one_i, lambda, &one_i);
        daxpy(&nineq, &alpha, dmu, &one_i, mu, &one_i);
        daxpy(&nineq, &alpha, ds, &one_i, s, &one_i);

        /* Measure Optimality*/
        compute_rC(Q, S, R, A, B, C, g, w, lambda, mu, nx, nu, nc, ncN, N, rC);  
        compute_rE(A, B, w, b, nx, nu, N, rE);    
        compute_rI(C, c, w, mu, s, nx, nu, nc, ncN, N, rI);     

        measure = ddot(&nw, rC, &one_i, rC, &one_i) + ddot(&neq, rE, &one_i, rE, &one_i)
                  + ddot(&nineq, rI, &one_i, rI, &one_i) + ddot(&nineq, s, &one_i, mu, &one_i);
                     
        it++;
        
        tau = exp(-0.2/it);
        
    }
    
    plhs[4] = mxCreateDoubleScalar(measure);
    plhs[5] = mxCreateDoubleScalar((double) it);
        
    /* Free memory */
    mxFree(phi);
    mxFree(phi_N);
    
    mxFree(LY);
    
    mxFree(rC);
    mxFree(rE);
    mxFree(rI);
    mxFree(rs);
    mxFree(rd);
    
    mxFree(dw);
    mxFree(dlambda);
    mxFree(dmu);
    mxFree(ds);
    
    mxFree(s_new);
    mxFree(mu_new);
    
}