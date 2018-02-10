#include "mex.h"
#include <stdlib.h>
#include <assert.h>
#include "pdip_common.h"
#include "common.h"

/* Structs and memory*/
int pdip_calculate_workspace_size(pdip_dims *dim)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N = dim->N;
    
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t neq = dim->neq;
    size_t nineq = dim->nineq;

    int size = sizeof(pdip_workspace);
     
    size += nz*nz*N*sizeof(double); // phi
    size += nx*nx*sizeof(double); // phi_N
    size += (2*nx*nx*N+nx*nx)*sizeof(double); // LY
    size += 2*nw*sizeof(double); // rC, rd
    size += neq*sizeof(double); // rE
    size += 2*nineq*sizeof(double); // rI, rs
    size += nw*sizeof(double); // dw
    size += neq*sizeof(double); // dlambda
    size += 2*nineq*sizeof(double); // dmu, ds
    size += 2*nineq*sizeof(double); // s_new, mu_new
    size += nw*sizeof(double); // fval
    size += nc*nz*sizeof(double); // Ci
    size += 2*nc*nc*sizeof(double); // hat_s_inv, hat_mu
    size += 2*nz*nc*sizeof(double); // tmp1, tmp2
    size += ncN*nx*sizeof(double); // CN
    size += 2*ncN*ncN*sizeof(double); // hat_sN_inv, hat_muN
    size += 2*nx*ncN*sizeof(double); // tmp1N, tmp2N       
    size += nineq*sizeof(double); // e       
    size += nx*nz*sizeof(double); // D       
    size += nx*nx*sizeof(double); // DN  
    size += nx*nz*(N+1)*sizeof(double); // V
    size += nx*nz*(N-1)*sizeof(double); // W
                
    return size;
}

void *pdip_cast_workspace(pdip_dims *dim, void *raw_memory)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N = dim->N;
    
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t neq = dim->neq;
    size_t nineq = dim->nineq;
    
    char *c_ptr = (char *)raw_memory;
    
    pdip_workspace *workspace = (pdip_workspace *) c_ptr;
    c_ptr += sizeof(pdip_workspace);
    
    workspace->phi = (double *)c_ptr;
    c_ptr += nz*nz*N*sizeof(double);
    
    workspace->phi_N = (double *)c_ptr;
    c_ptr += nx*nx*sizeof(double);
    
    workspace->LY = (double *)c_ptr;
    c_ptr += (2*nx*nx*N+nx*nx)*sizeof(double);
    
    workspace->rC = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->rd = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->rE = (double *)c_ptr;
    c_ptr += neq*sizeof(double);
    
    workspace->rI = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->rs = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->dw = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->dlambda = (double *)c_ptr;
    c_ptr += neq*sizeof(double);
    
    workspace->dmu = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->ds = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->s_new = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->mu_new = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->fval = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->Ci = (double *)c_ptr;
    c_ptr += nc*nz*sizeof(double);
    
    workspace->hat_s_inv = (double *)c_ptr;
    c_ptr += nc*nc*sizeof(double);
    
    workspace->hat_mu = (double *)c_ptr;
    c_ptr += nc*nc*sizeof(double);
    
    workspace->tmp1 = (double *)c_ptr;
    c_ptr += nz*nc*sizeof(double);
    
    workspace->tmp2 = (double *)c_ptr;
    c_ptr += nz*nc*sizeof(double);
    
    workspace->CN = (double *)c_ptr;
    c_ptr += ncN*nx*sizeof(double);
    
    workspace->hat_sN_inv = (double *)c_ptr;
    c_ptr += ncN*ncN*sizeof(double);
    
    workspace->hat_muN = (double *)c_ptr;
    c_ptr += ncN*ncN*sizeof(double);
    
    workspace->tmp1N = (double *)c_ptr;
    c_ptr += nx*ncN*sizeof(double);
    
    workspace->tmp2N = (double *)c_ptr;
    c_ptr += nx*ncN*sizeof(double);
    
    workspace->e = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->D = (double *)c_ptr;
    c_ptr += nx*nz*sizeof(double);
    
    workspace->DN = (double *)c_ptr;
    c_ptr += nx*nx*sizeof(double);
    
    workspace->V = (double *)c_ptr;
    c_ptr += nx*nz*(N+1)*sizeof(double);
    
    workspace->W = (double *)c_ptr;
    c_ptr += nx*nz*(N-1)*sizeof(double);
       
//     assert((char*)raw_memory + pdip_calculate_workspace_size(dim) >= c_ptr);
    mexPrintf("size calculated: %d\n", pdip_calculate_workspace_size(dim));
    mexPrintf("pointer moved: %d\n", c_ptr - (char*)raw_memory);

    return (void *)workspace;

}

void *pdip_init_workspace(pdip_dims *dim, pdip_workspace *work){
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t N = dim->N;
    size_t nz = dim->nz;
    
    set_zeros(nc*nc, work->hat_s_inv);
    set_zeros(ncN*ncN, work->hat_sN_inv);
    set_zeros(nc*nc, work->hat_mu);
    set_zeros(ncN*ncN, work->hat_muN);
    set_zeros(nx*nz, work->D);
    set_zeros(nx*nx, work->DN);
    set_zeros(nx*nz*(N+1), work->V);
    set_zeros(nx*nz*(N-1), work->W);
    
    work->tau = 0.8;
}


