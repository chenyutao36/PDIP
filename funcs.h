#ifndef FUNCS_H_
#define FUNCS_H_

#include "pdip_common.h"

/* Functions */
void compute_phi(double *Q, double *S, double *R, double *C, double *s, double *mu, pdip_dims *dim, pdip_workspace *work);

void compute_LY(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void lin_solve(pdip_dims *dim, pdip_workspace *work);

void compute_rC(double *Q, double *S, double *R, double *A, double *B, double *C,
                double *g, double *w, double *lambda, double *mu, pdip_dims *dim, pdip_workspace *work);

void compute_rE(double *A, double *B, double *w, double *b, pdip_dims *dim, pdip_workspace *work);

void compute_rI(double *C, double *c, double *w, double *mu, double *s, pdip_dims *dim, pdip_workspace *work);

void compute_rs(double *mu, double *s, pdip_dims *dim, pdip_workspace *work);

void compute_rd(double *C, double *c, double *mu, double *s, pdip_dims *dim, pdip_workspace *work);

void compute_beta(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void recover_dw(double *A, double *B, pdip_dims *dim, pdip_workspace *work);

void recover_dmu(double *C, double *mu, double *s, pdip_dims *dim, pdip_workspace *work);

void recover_ds(double *C, pdip_dims *dim, pdip_workspace *work);

void compute_fval(double *Q, double *S, double *R, double *g, double *w, pdip_dims *dim, pdip_workspace *work);

#endif