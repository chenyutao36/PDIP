#ifndef COMMON_H_
#define COMMON_H_

#include "stdlib.h"
#include <stdbool.h>

void Block_Fill(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha);

void Block_Fill_Trans(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha);

void Block_Access(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG, double alpha);

bool vec_bigger(size_t n, double *a, double *b);

void set_zeros(size_t n, double *a);

void regularization(size_t n, double *A, double reg);

void print_matrix(size_t m, size_t n, double *A, size_t ldA);

void print_vector(size_t n, double *a);

void print_vector_int(size_t n, int *a);

#endif