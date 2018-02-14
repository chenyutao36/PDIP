#ifndef BLOCKIP_FUNCS_H_
#define BLOCKIP_FUNCS_H_

#include "blockIP_common.h"

/* Functions */

void compute_phi(pdip_dims *dim, pdip_workspace *work);

void compute_LY(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void lin_solve(pdip_dims *dim, pdip_workspace *work);

void compute_rC(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void compute_rE(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void compute_rI(pdip_dims *dim, pdip_workspace *work);

void compute_rs(pdip_dims *dim, pdip_workspace *work);

void compute_rd(pdip_dims *dim, pdip_workspace *work);

void compute_beta(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void recover_dw(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void recover_dmu(pdip_dims *dim, pdip_workspace *work);

void recover_ds(pdip_dims *dim, pdip_workspace *work);

void compute_fval(pdip_dims *dim, pdip_workspace *work);

void recover_sol(pdip_dims *dim, pdip_workspace *work, 
        double *dz, double *xN, double *lambda, double *mu, double *muN, double *mu_u);

#endif