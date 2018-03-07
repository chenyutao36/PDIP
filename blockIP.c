#include "mex.h"
#include "string.h"
#include "blockIP_common.h"
#include "blockIP_funcs.h"
#include "common.h"
#include "blas.h"
#include "math.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{    
    
    double *Q = mxGetPr( mxGetField(prhs[0], 0, "Q") );
    double *S = mxGetPr( mxGetField(prhs[0], 0, "S") );
    double *R = mxGetPr( mxGetField(prhs[0], 0, "R") );
    double *A = mxGetPr( mxGetField(prhs[0], 0, "A") );
    double *B = mxGetPr( mxGetField(prhs[0], 0, "B") );
    double *Cx = mxGetPr( mxGetField(prhs[0], 0, "Cx") );
    double *CN = mxGetPr( mxGetField(prhs[0], 0, "CN") );
    double *Cu = mxGetPr( mxGetField(prhs[0], 0, "Cu") );
    double *gx = mxGetPr( mxGetField(prhs[0], 0, "gx") );
    double *gu = mxGetPr( mxGetField(prhs[0], 0, "gu") );
    double *a = mxGetPr( mxGetField(prhs[0], 0, "a") );
    double *ds0 = mxGetPr( mxGetField(prhs[0], 0, "ds0") );
    double *uc = mxGetPr( mxGetField(prhs[0], 0, "uc") );
    double *lc = mxGetPr( mxGetField(prhs[0], 0, "lc") );
    double *ub_du = mxGetPr( mxGetField(prhs[0], 0, "ub_du") );
    double *lb_du = mxGetPr( mxGetField(prhs[0], 0, "lb_du") );
    
    size_t nx = mxGetScalar( mxGetField(prhs[1], 0, "nx") );
    size_t nu = mxGetScalar( mxGetField(prhs[1], 0, "nu") );
    size_t nc = mxGetScalar( mxGetField(prhs[1], 0, "nc") );
    size_t ncN = mxGetScalar( mxGetField(prhs[1], 0, "ncN") );
    size_t nbu = mxGetScalar( mxGetField(prhs[1], 0, "nbu") );
    double *nbu_idx = mxGetPr( mxGetField(prhs[1], 0, "nbu_idx") );
    size_t N = mxGetScalar( mxGetField(prhs[1], 0, "N") );
    
    int it_max = mxGetScalar( mxGetField(prhs[2], 0, "it_max") );
    double tol = mxGetScalar( mxGetField(prhs[2], 0, "tol") );
    int prt = mxGetScalar( mxGetField(prhs[2], 0, "print_level") );
    double reg = mxGetScalar( mxGetField(prhs[2], 0, "reg") );
    
    size_t nz = nx+nu;
    size_t nw = N*nz+nx;
    size_t neq = (N+1)*nx; 
    size_t nineq = N*2*(nc+nbu)+ncN*2; 
    
    int i;
    
    double *dx = mxGetPr( mxGetField(prhs[0], 0, "dx") );
    double *du = mxGetPr( mxGetField(prhs[0], 0, "du") );
    double *lambda = mxGetPr( mxGetField(prhs[0], 0, "lambda_new") );
    double *mu = mxGetPr( mxGetField(prhs[0], 0, "mu_new") );
    double *muN = mxGetPr( mxGetField(prhs[0], 0, "muN_new") );
    double *mu_u = mxGetPr( mxGetField(prhs[0], 0, "mu_u_new") );
         
    /* Allocate Memory */ 
    int size_dim = pdip_calculate_dim_size(nbu);
    void *dim_mem = mxMalloc(size_dim);
    pdip_dims *dim = (pdip_dims *) pdip_cast_dim(nbu, dim_mem);
    pdip_init_dim(dim, nx, nu, nc, ncN, nbu, N, nbu_idx, nz, nw, neq, nineq);  
     
    int size = pdip_calculate_workspace_size(dim);
    void *work = mxMalloc(size);
    pdip_workspace *workspace = (pdip_workspace *) pdip_cast_workspace(dim, work);
    pdip_init_workspace(dim, workspace, Cx, Cu, CN, uc, lc, ub_du, lb_du,
        Q, S, R, A, B, gx, gu, ds0, a, reg);
            
    /* Define constants*/    
    double one_d = 1.0, zero_d=0.0, minus_one_d = -1.0;
    mwSize one_i = 1;
    double t_aff, sca;
    double alpha, alpha_aff, alpha_pri_tau, alpha_dual_tau;
    double dual_res, eq_res, ineq_res, comp_res, fval;
    double measure = fabs(tol)*2;
    int it=0;
    
    if (prt==2){
        mexPrintf("%-10s","It:");
        mexPrintf("%-20s","Dual Res");
        mexPrintf("%-20s","Eq Res");
        mexPrintf("%-20s","Ineq Res");
        mexPrintf("%-20s","Compl Res");
        mexPrintf("%-20s\n","Fval");
    }
        
    /* Start loop*/
    while (it<it_max && measure>tol){    
        workspace->sigma=0; 
        workspace->t=0;      
        set_zeros(nineq, workspace->dmu);
        set_zeros(nineq, workspace->ds);
                      
        /* Predictor */

        /* Compute residuals */
        compute_rC(A, B, dim, workspace);  
        compute_rE(A, B, dim, workspace);    
        compute_rI(dim, workspace);     
        compute_rs(dim, workspace);
        compute_rd(dim, workspace); 

        /* Compute phi and its Cholesky factor L, stored in phi */
        compute_phi(dim, workspace);  

        /* Compute beta*/
        /* on exit, rd is the solution of phi^{-1} r_d */
        compute_beta(A, B, dim, workspace);     

        /* Compute Y and its Cholesky factor LY, stored in LY */
        compute_LY(A, B, dim, workspace);

        /* Solve the normal equation */
        lin_solve(dim, workspace); 

        /* Recover primal solution */
        recover_dw(A, B, dim, workspace);    
        recover_dmu(dim, workspace);    
        recover_ds(dim, workspace);

        /* Compute centering parameter sigma*/
               
        workspace->t = ddot(&nineq, workspace->s, &one_i, workspace->mu, &one_i)/nineq;
        alpha_aff = 0.995;
        memcpy(workspace->s_new, workspace->s, nineq*sizeof(double));
        memcpy(workspace->mu_new, workspace->mu, nineq*sizeof(double));
        daxpy(&nineq, &alpha_aff, workspace->ds, &one_i, workspace->s_new, &one_i);
        daxpy(&nineq, &alpha_aff, workspace->dmu, &one_i, workspace->mu_new, &one_i);
        t_aff = ddot(&nineq, workspace->s_new, &one_i, workspace->mu_new, &one_i);
        while (t_aff<=0 && alpha_aff>1E-8){
            alpha_aff *= 0.9;
            memcpy(workspace->s_new, workspace->s, nineq*sizeof(double));
            memcpy(workspace->mu_new, workspace->mu, nineq*sizeof(double));
            daxpy(&nineq, &alpha_aff, workspace->ds, &one_i, workspace->s_new, &one_i);
            daxpy(&nineq, &alpha_aff, workspace->dmu, &one_i, workspace->mu_new, &one_i);
            t_aff = ddot(&nineq, workspace->s_new, &one_i, workspace->mu_new, &one_i);
        }
        t_aff = t_aff/nineq;
        workspace->sigma = pow((t_aff/workspace->t),3);
        
        /* Corrector */
        compute_rs(dim, workspace);
        compute_rd(dim, workspace); 
        compute_beta(A, B, dim, workspace); 
        lin_solve(dim, workspace);
        recover_dw(A, B, dim, workspace);    
        recover_dmu(dim, workspace);    
        recover_ds(dim, workspace);

        /* Step length selection*/
        sca = 1-workspace->tau;
        
        alpha_pri_tau=0.995;
        memcpy(workspace->s_new, workspace->s, nineq*sizeof(double));
        memcpy(workspace->mu_new, workspace->s, nineq*sizeof(double));
        daxpy(&nineq, &alpha_pri_tau, workspace->ds, &one_i, workspace->s_new, &one_i);
        dscal(&nineq, &sca, workspace->mu_new, &one_i);
        while (!vec_bigger(nineq, workspace->s_new, workspace->mu_new) && alpha_pri_tau>1E-8){
            alpha_pri_tau *= 0.95;
            memcpy(workspace->s_new, workspace->s, nineq*sizeof(double));
            memcpy(workspace->mu_new, workspace->s, nineq*sizeof(double));
            daxpy(&nineq, &alpha_pri_tau, workspace->ds, &one_i, workspace->s_new, &one_i);
            dscal(&nineq, &sca, workspace->mu_new, &one_i);
        }

        alpha_dual_tau=0.995;
        memcpy(workspace->s_new, workspace->mu, nineq*sizeof(double));
        memcpy(workspace->mu_new, workspace->mu, nineq*sizeof(double));
        daxpy(&nineq, &alpha_dual_tau, workspace->dmu, &one_i, workspace->s_new, &one_i);
        dscal(&nineq, &sca, workspace->mu_new, &one_i);
        while (!vec_bigger(nineq, workspace->s_new, workspace->mu_new) && alpha_dual_tau>1E-8){
            alpha_dual_tau *= 0.95;
            memcpy(workspace->s_new, workspace->mu, nineq*sizeof(double));
            memcpy(workspace->mu_new, workspace->mu, nineq*sizeof(double));
            daxpy(&nineq, &alpha_dual_tau, workspace->dmu, &one_i, workspace->s_new, &one_i);
            dscal(&nineq, &sca, workspace->mu_new, &one_i);
        }

        alpha = MIN(alpha_pri_tau, alpha_dual_tau);
        
        /* Update solution */
        
        daxpy(&nw, &alpha, workspace->dw, &one_i, workspace->w, &one_i);
        daxpy(&neq, &alpha, workspace->dlambda, &one_i, workspace->lambda, &one_i);
        daxpy(&nineq, &alpha, workspace->dmu, &one_i, workspace->mu, &one_i);
        daxpy(&nineq, &alpha, workspace->ds, &one_i, workspace->s, &one_i);

        /* Measure Optimality*/
        compute_rC(A, B, dim, workspace);  
        compute_rE(A, B, dim, workspace);    
        compute_rI(dim, workspace);      
        
        dual_res = ddot(&nw, workspace->rC, &one_i, workspace->rC, &one_i);
        eq_res = ddot(&neq, workspace->rE, &one_i, workspace->rE, &one_i);
        ineq_res = ddot(&nineq, workspace->rI, &one_i, workspace->rI, &one_i);
        comp_res = ddot(&nineq, workspace->s, &one_i, workspace->mu, &one_i);
        
        measure = dual_res + eq_res + ineq_res + comp_res;
        
        /* Compute 0.5*Hw+g and then the optimal value*/
        compute_fval(dim, workspace);
        fval = ddot(&nw, workspace->w, &one_i, workspace->fval, &one_i); 
                     
        it++;
        
        if (prt==2){
            mexPrintf("%-10d",it);
            mexPrintf("%-20.3e",dual_res);
            mexPrintf("%-20.3e",eq_res);
            mexPrintf("%-20.3e",ineq_res);
            mexPrintf("%-20.3e",comp_res);
            mexPrintf("%-20.3e\n",fval);
        }
        
        workspace->tau = exp(-0.3/it);
        
    }
    if (prt==1){
        mexPrintf("No. of It: %d,   KKT: %6.3e\n", it, measure);
    }
        
    recover_sol(dim, workspace, dx, du, lambda, mu, muN, mu_u);
        
    /* Free memory */
    dim = NULL;
    workspace = NULL;
    mxFree(dim_mem);    
    mxFree(work); 
    
}