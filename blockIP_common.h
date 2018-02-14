/* Structs and memory*/

#ifndef BLOCKIP_COMMON_H_
#define BLOCKIP_COMMON_H_

#include "stdlib.h"

typedef struct{
    size_t nx;
    size_t nu;
    size_t nc;
    size_t ncN;
    size_t nbu;   
    size_t N;    
    size_t nz;
    size_t nw;
    size_t neq;
    size_t nineq;
    
    int *nbu_idx;
}pdip_dims;

typedef struct{
    double tau;
    double sigma;
    double t;   
    
    double *phi;
    double *phi_N;
    double *LY;    
    double *rC;
    double *rE;
    double *rI;
    double *rs;
    double *rd;       
    double *dw;
    double *dlambda;   
    double *dmu;
    double *ds;  
    double *s_new;
    double *mu_new;    
    double *fval;
    
    double *hat_s_inv;
    double *hat_mu;
    double *tmp1;
    double *tmp2;     
    
    double *hat_sN_inv;
    double *hat_muN;
    double *tmp1N;
    double *tmp2N;
    
    double *e;
    
    double *D;    
    double *DN;    
    double *V; 
    double *W;
    
    double *Cu;
    double *Cx;
    double *Cz;
    double *CzN;
    double *c;
    
    double *H;
    double *HN;
    double *g;
    double *b;
    
    double *w;
    double *lambda;
    double *mu;
    double *s;
}pdip_workspace;

/* for dim */
int pdip_calculate_dim_size(size_t nbu);

void *pdip_cast_dim(size_t nbu, void *raw_memory);

void pdip_init_dim(pdip_dims *dim, size_t nx, size_t nu, size_t nc, size_t ncN, size_t nbu,
        size_t N, double *nbu_idx, size_t nz, size_t nw, size_t neq, size_t nineq);

/* for workspace */
int pdip_calculate_workspace_size(pdip_dims *dim);

void *pdip_cast_workspace(pdip_dims *dim, void *raw_memory);

void pdip_init_workspace(pdip_dims *dim, pdip_workspace *work, 
        double *Cx, double *Cu, double *CxN, double *uc, double *lc, double *ub_du, double *lb_du,
        double *Q, double *S, double *R, double *A, double *B, double *gx, double *gu, double *ds0, double *a, double reg);

#endif
