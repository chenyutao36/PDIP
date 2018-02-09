
void Block_Fill(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

void Block_Fill_Trans(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

void Block_Access(size_t m, size_t n, double *Gi, double *G, size_t idm, size_t idn, size_t ldG);

bool vec_bigger(size_t n, double *a, double *b);

void set_zeros(size_t n, double *a);

void compute_phi(double *Q, double *S, double *R, double *C, double *s, double *mu, double *phi, double *phi_N,
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N);

void compute_LY(double *phi, double *phi_N, double *A, double *B, double *LY,
        size_t nx, size_t nu, size_t N);

void lin_solve(double *LY, double *sol, size_t nx, size_t nu, size_t N);

void compute_rC(double *Q, double *S, double *R, double *A, double *B, double *C,
        double *g, double *w, double *lambda, double *mu,
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *rc);

void compute_rE(double *A, double *B, double *w, double *b,
        size_t nx, size_t nu, size_t N, double *rE);

void compute_rI(double *C, double *c, double *w, double *mu, double *s,
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *rI);

void compute_rs(double *mu, double *s, double *dmu, double *ds, double sigma, double t, 
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *rs);

void compute_rd(double *C, double *c, double *mu, double *s, double *rI, double *rC, double *rs,
        double *dmu, double *ds, size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *rd);

void compute_beta(double *A, double *B, double *rE, double *rd, double *phi, double *phi_N, 
        size_t nx, size_t nu, size_t N, double *beta);

void recover_dw(double *A, double *B, double *rd, double *phi, double *phi_N, double *dlambda, 
        size_t nx, size_t nu, size_t N, double *dw);

void recover_dmu(double *C, double *mu, double *s, double *rI, double *dw, double *rs,
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *dmu);

void recover_ds(double *C, double *rI, double *dw,
        size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *ds);

// void compute_rd_corrector(double *C, double *c, double *mu, double *s, double *rI, double *rC,
//         double *dmu, double *ds, double sigma, double t,
//         size_t nx, size_t nu, size_t nc, size_t ncN, size_t N, double *rd);