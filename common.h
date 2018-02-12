#ifndef COMMON_H_
#define COMMON_H_

#include "stdlib.h"
#include <stdbool.h>

void Block_Fill(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

void Block_Fill_Trans(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

void Block_Access(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

bool vec_bigger(size_t n, double *a, double *b);

void set_zeros(size_t n, double *a);

void regularization(size_t n, double *A, double reg);

#endif