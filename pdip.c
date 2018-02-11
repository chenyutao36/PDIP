#include "mex.h"
#include "string.h"
#include "pdip_common.h"
#include "funcs.h"
#include "common.h"
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
    pdip_dims dim;
    dim.nx = nx;
    dim.nu = nu;
    dim.nc = nc;
    dim.ncN = ncN;
    dim.N = N;
    dim.nz = nz;
    dim.nw = nw;
    dim.neq = neq;
    dim.nineq = nineq;
     
    int size = pdip_calculate_workspace_size(&dim);
    void *work = mxMalloc(size);
    pdip_workspace *workspace = (pdip_workspace *) pdip_cast_workspace(&dim, work);
    pdip_init_workspace(&dim, workspace);
    
    double *s_new = workspace->s_new;
    double *mu_new = workspace->mu_new;
    double *dw = workspace->dw;
    double *dlambda = workspace->dlambda;
    double *ds = workspace->ds;
    double *dmu = workspace->dmu;
    
    /* Define constants*/    
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    double t_aff, sca;
    double alpha, alpha_aff, alpha_pri_tau, alpha_dual_tau;
    int it_max = 100;
    double measure = 1E+4;
    double tol = 1E-4;
    int it=0;
        
    /* Start loop*/
    while (it<it_max && measure>tol){    
        workspace->sigma=0; 
        workspace->t=0;      
        set_zeros(nineq, workspace->dmu);
        set_zeros(nineq, workspace->ds);
        
        /* Predictor */

        /* Compute residuals */
        compute_rC(Q, S, R, A, B, C, g, w, lambda, mu, &dim, workspace);  
        compute_rE(A, B, w, b, &dim, workspace);    
        compute_rI(C, c, w, mu, s, &dim, workspace);     
        compute_rs(mu, s, &dim, workspace);
        compute_rd(C, c, mu, s, &dim, workspace); 

        /* Compute phi and its Cholesky factor L, stored in phi */
        compute_phi(Q, S, R, C, s, mu, &dim, workspace);  

        /* Compute beta*/
        /* on exit, rd is the solution of phi_k^{-1} r_d^k */
        compute_beta(A, B, &dim, workspace);     

        /* Compute Y and its Cholesky factor LY, stored in LY */
        compute_LY(A, B, &dim, workspace);

        /* Solve the normal equation */
        lin_solve(&dim, workspace); 

        /* Recover primal solution */
        recover_dw(A, B, &dim, workspace);    
        recover_dmu(C, mu, s, &dim, workspace);    
        recover_ds(C, &dim, workspace);

        /* Compute centering parameter sigma*/
        workspace->t = ddot(&nineq, s, &one_i, mu, &one_i)/nineq;
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
        workspace->sigma = pow((t_aff/workspace->t),3);
        
        /* Corrector */
        compute_rs(mu, s, &dim, workspace);
        compute_rd(C, c, mu, s, &dim, workspace); 
        compute_beta(A, B, &dim, workspace); 
        lin_solve(&dim, workspace);
        recover_dw(A, B, &dim, workspace);    
        recover_dmu(C, mu, s, &dim, workspace);    
        recover_ds(C, &dim, workspace);

        /* Step length selection*/
        sca = 1-workspace->tau;
        
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
        compute_rC(Q, S, R, A, B, C, g, w, lambda, mu, &dim, workspace);  
        compute_rE(A, B, w, b, &dim, workspace);    
        compute_rI(C, c, w, mu, s, &dim, workspace);      

        measure = ddot(&nw, workspace->rC, &one_i, workspace->rC, &one_i) + ddot(&neq, workspace->rE, &one_i, workspace->rE, &one_i)
                  + ddot(&nineq, workspace->rI, &one_i, workspace->rI, &one_i) + ddot(&nineq, s, &one_i, mu, &one_i);
                     
        it++;
        
        workspace->tau = exp(-0.1/it);
        
    }
    
    /* Compute 0.5*Hw+g and then the optimal value*/
    compute_fval(Q, S, R, g, w, &dim, workspace);
    double fval = ddot(&nw, w, &one_i, workspace->fval, &one_i); 
    
    plhs[4] = mxCreateDoubleScalar(measure);
    plhs[5] = mxCreateDoubleScalar((double) it);
    plhs[6] = mxCreateDoubleScalar(fval);
        
    /* Free memory */
    mxFree(work);    
}