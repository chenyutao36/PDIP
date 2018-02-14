#include "mex.h"
#include <stdlib.h>
#include "string.h"
#include "blockIP_common.h"
#include "common.h"

/* Structs and memory*/
int pdip_calculate_dim_size(size_t nbu)
{    
    int size = sizeof(pdip_dims);
    
    size += nbu*sizeof(int); // nbu_idx
    
    return size;
}

void *pdip_cast_dim(size_t nbu, void *raw_memory)
{
    
    char *c_ptr = (char *)raw_memory;
    
    pdip_dims *dim = (pdip_dims *) c_ptr;
    c_ptr += sizeof(pdip_dims);
    
    dim->nbu_idx = (int *)c_ptr;
    c_ptr += nbu*sizeof(int);

    return (void *)dim;
}

void pdip_init_dim(pdip_dims *dim, size_t nx, size_t nu, size_t nc, size_t ncN, size_t nbu,
        size_t N, double *nbu_idx, size_t nz, size_t nw, size_t neq, size_t nineq)
{
    dim->nx = nx;
    dim->nu = nu;
    dim->nc = nc;
    dim->ncN = ncN;
    dim->nbu = nbu;
    dim->N = N;
    dim->nz = nz;
    dim->nw = nw;
    dim->neq = neq;
    dim->nineq = nineq;
    
    int i;
    for (i=0;i<nbu;i++)
        dim->nbu_idx[i] = (int) nbu_idx[i]-1;
    
//     print_vector_int(nbu, dim->nbu_idx);
}



int pdip_calculate_workspace_size(pdip_dims *dim)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N = dim->N;
    
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t neq = dim->neq;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;

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
    size += 2*row_C*row_C*sizeof(double); // hat_s_inv, hat_mu
    size += 2*nz*row_C*sizeof(double); // tmp1, tmp2
    size += 2*row_CN*row_CN*sizeof(double); // hat_sN_inv, hat_muN
    size += 2*nx*row_CN*sizeof(double); // tmp1N, tmp2N       
    size += nineq*sizeof(double); // e       
    size += nx*nz*sizeof(double); // D       
    size += nx*nx*sizeof(double); // DN  
    size += nx*nz*(N+1)*sizeof(double); // V
    size += nx*nz*(N-1)*sizeof(double); // W
    
    size += nbu*nu*sizeof(double); // Cu
    size += nbu*nx*sizeof(double); // Cx
    size += row_C*nz*N*sizeof(double); // Cz
    size += row_CN*nx*sizeof(double); // CzN
    size += (row_C*N + row_CN)*sizeof(double); // c
    
    size += nz*nz*N*sizeof(double); // H
    size += nx*nx*sizeof(double); // HN
    size += nw*sizeof(double); // g
    size += neq*sizeof(double); // b
    
    size += nw*sizeof(double); // w
    size += neq*sizeof(double); // lambda
    size += nineq*sizeof(double); // mu
    size += nineq*sizeof(double); // s
                    
    return size;
}

void *pdip_cast_workspace(pdip_dims *dim, void *raw_memory)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    size_t N = dim->N;
    
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t neq = dim->neq;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
    
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
        
    workspace->hat_s_inv = (double *)c_ptr;
    c_ptr += row_C*row_C*sizeof(double);
    
    workspace->hat_mu = (double *)c_ptr;
    c_ptr += row_C*row_C*sizeof(double);
    
    workspace->tmp1 = (double *)c_ptr;
    c_ptr += nz*row_C*sizeof(double);
    
    workspace->tmp2 = (double *)c_ptr;
    c_ptr += nz*row_C*sizeof(double);
        
    workspace->hat_sN_inv = (double *)c_ptr;
    c_ptr += row_CN*row_CN*sizeof(double);
    
    workspace->hat_muN = (double *)c_ptr;
    c_ptr += row_CN*row_CN*sizeof(double);
    
    workspace->tmp1N = (double *)c_ptr;
    c_ptr += nx*row_CN*sizeof(double);
    
    workspace->tmp2N = (double *)c_ptr;
    c_ptr += nx*row_CN*sizeof(double);
    
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
    
    workspace->Cx = (double *)c_ptr;
    c_ptr += nbu*nx*sizeof(double);
    
    workspace->Cu = (double *)c_ptr;
    c_ptr += nbu*nu*sizeof(double);
    
    workspace->Cz = (double *)c_ptr;
    c_ptr += row_C*nz*N*sizeof(double);
    
    workspace->CzN = (double *)c_ptr;
    c_ptr += row_CN*nx*sizeof(double);
    
    workspace->c = (double *)c_ptr;
    c_ptr += (row_C*N + row_CN)*sizeof(double);
    
    workspace->H = (double *)c_ptr;
    c_ptr += nz*nz*N*sizeof(double);
    
    workspace->HN = (double *)c_ptr;
    c_ptr += nx*nx*sizeof(double);
    
    workspace->g = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->b = (double *)c_ptr;
    c_ptr += neq*sizeof(double);
    
    workspace->w = (double *)c_ptr;
    c_ptr += nw*sizeof(double);
    
    workspace->lambda = (double *)c_ptr;
    c_ptr += neq*sizeof(double);
    
    workspace->mu = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
    
    workspace->s = (double *)c_ptr;
    c_ptr += nineq*sizeof(double);
           
//     mexPrintf("size calculated: %d\n", pdip_calculate_workspace_size(dim));
//     mexPrintf("pointer moved: %d\n", c_ptr - (char*)raw_memory);

    return (void *)workspace;

}

void pdip_init_workspace(pdip_dims *dim, pdip_workspace *work, 
        double *Cx, double *Cu, double *CxN, double *uc, double *lc, double *ub_du, double *lb_du,
        double *Q, double *S, double *R, double *A, double *B, double *gx, double *gu, double *ds0, double *a, double reg)
{
    size_t nx = dim->nx;
    size_t nu = dim->nu;
    size_t nc = dim->nc;
    size_t ncN = dim->ncN;
    size_t nbu = dim->nbu;
    int *nbu_idx = dim->nbu_idx;
    size_t N = dim->N;
    size_t nz = dim->nz;
    size_t nw = dim->nw;
    size_t neq = dim->neq;
    size_t nineq = dim->nineq;
    
    size_t row_C = 2*(nbu+nc);
    size_t row_CN = 2*ncN;
        
    set_zeros(nw, work->w);
    set_zeros(neq, work->lambda);
    
    int i,j;
    for (i=0;i<nineq;i++){
        work->s[i] = 10;
        work->mu[i] = 10;
    }
    
    set_zeros(row_C*row_C, work->hat_s_inv);
    set_zeros(row_CN*row_CN, work->hat_sN_inv);
    set_zeros(row_C*row_C, work->hat_mu);
    set_zeros(row_CN*row_CN, work->hat_muN);
    set_zeros(nx*nz, work->D);
    set_zeros(nx*nx, work->DN);
    set_zeros(nx*nz*(N+1), work->V);
    set_zeros(nx*nz*(N-1), work->W);
    
    set_zeros(nbu*nu, work->Cu);
    set_zeros(nbu*nx, work->Cx);
    
    work->tau = 0.9;
    
    double *Cz = work->Cz;
    double *CzN = work->CzN;
    double *c = work->c;
    double *H = work->H;
    double *HN = work->HN;
    double *g = work->g;
    double *b = work->b;
    
    double *D = work->D;
    double *DN = work->DN;
    
    for (i=0;i<nx;i++){
        D[i*nx+i]= -1.0; // this is D_1
    }
    
    for (i=0;i<nbu;i++)
        work->Cu[nbu_idx[i]*nbu+i] = 1.0;
        
    memcpy(b, ds0, nx*sizeof(double));
    
    for(i=0;i<N;i++){
        
        // Cz
        Block_Fill(nc,nx,Cx+i*nc*nx,Cz,0,i*nz,row_C, 1.0);
        Block_Fill(nc,nu,Cu+i*nc*nu,Cz,0,i*nz+nx,row_C, 1.0);
        Block_Fill(nbu,nx,work->Cx,Cz,nc,i*nz,row_C, 0.0);
        Block_Fill(nbu,nu,work->Cu,Cz,nc,i*nz+nx,row_C, 1.0);
                
        Block_Fill(nc,nx,Cx+i*nc*nx,Cz,nc+nbu,i*nz,row_C, -1.0);
        Block_Fill(nc,nu,Cu+i*nc*nu,Cz,nc+nbu,i*nz+nx,row_C, -1.0);
        Block_Fill(nbu,nx,work->Cx,Cz,nc+nbu+nc,i*nz,row_C, 0.0);
        Block_Fill(nbu,nu,work->Cu,Cz,nc+nbu+nc,i*nz+nx,row_C, -1.0);
        
        // c
        for (j=0;j<nc;j++){
            c[i*row_C+j] = -1*uc[i*nc+j];
            c[i*row_C+nc+nbu+j] = lc[i*nc+j];
        }
        for (j=0;j<nbu;j++){
            c[i*row_C+nc+j] = -1*ub_du[i*nu+nbu_idx[j]];
            c[i*row_C+nc+nbu+nc+j] = lb_du[i*nu+nbu_idx[j]];
        }
        
        // H
        Block_Fill(nx,nx,Q+i*nx*nx,H,0,i*nz,nz,1);
        Block_Fill(nx,nu,S+i*nx*nu,H,0,i*nz+nx,nz,1);
        Block_Fill_Trans(nx,nu,S+i*nx*nu,H,nx,i*nz,nz,1);
        Block_Fill(nu,nu,R+i*nu*nu,H,nx,i*nz+nx,nz,1);
        
        regularization(nz, H+i*nz*nz, reg);
        
        // g
        memcpy(g+i*nz, gx+i*nx, nx*sizeof(double));
        memcpy(g+i*nz+nx, gu+i*nu, nu*sizeof(double));
        
        // b
        memcpy(b+(i+1)*nx, a+i*nx, nx*sizeof(double));
                        
    }
        
    Block_Fill(ncN,nx,CxN,CzN,0,0,row_CN, 1.0);
    Block_Fill(ncN,nx,CxN,CzN,ncN,0,row_CN, -1.0);
    
    for (j=0;j<ncN;j++){
        c[N*row_C+j] = -uc[N*nc+j];
        c[N*row_C+ncN+j] = lc[N*nc+j];
    }
    
    memcpy(HN, Q+N*nx*nx, nx*nx*sizeof(double));
    regularization(nx, HN, reg);
    
    memcpy(g+N*nz, gx+N*nx, nx*sizeof(double));
    
}


