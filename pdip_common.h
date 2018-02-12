/* Structs and memory*/

#ifndef PDIP_COMMON_H_
#define PDIP_COMMON_H_

#include "stdlib.h"

typedef struct{
    size_t nx;
    size_t nu;
    size_t nc;
    size_t ncN;
    size_t N;
    
    size_t nz;
    size_t nw;
    size_t neq;
    size_t nineq;
}pdip_dims;

typedef struct{
    double tau;
    double sigma;
    double t;
    
    double reg;
    
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
    
    double *Ci;
    double *hat_s_inv;
    double *hat_mu;
    double *tmp1;
    double *tmp2;     
    
    double *CN;
    double *hat_sN_inv;
    double *hat_muN;
    double *tmp1N;
    double *tmp2N;
    
    double *e;
    
    double *D;    
    double *DN;    
    double *V; 
    double *W;
}pdip_workspace;

int pdip_calculate_workspace_size(pdip_dims *dim);

void *pdip_cast_workspace(pdip_dims *dim, void *raw_memory);

void *pdip_init_workspace(pdip_dims *dim, pdip_workspace *work);

#endif
