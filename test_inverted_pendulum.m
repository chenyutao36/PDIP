clear;close all;clc;

load myData;

mem.lc(1) = -2;
mem.uc(1) = 2;

N=settings.N;
nx=settings.nx;
nu=settings.nu;
nc=settings.nc;
ncN=settings.ncN;

nbu=1;
nbu_idx=1;
Cu = zeros(nbu,nu);
for j=1:nbu
    Cu(j,nbu_idx(j))=1;
end

H=[];
g=[];
B=zeros(settings.neq,settings.nw);   B(1:nx,1:nx)=eye(nx);
b=mem.ds0;
C=[];
c=[];
for i=1:N
    Hi=[mem.Q_h(:,(i-1)*nx+1:i*nx), mem.S(:,(i-1)*nu+1:i*nu);
        (mem.S(:,(i-1)*nu+1:i*nu))', mem.R(:,(i-1)*nu+1:i*nu)];
    H=blkdiag(H,Hi);
    
    g=[g;mem.gx(:,i);mem.gu(:,i)];
    
    B(i*nx+1:(i+1)*nx,(i-1)*(nx+nu)+1:i*(nx+nu)+nx)=[mem.A_sens(:,(i-1)*nx+1:i*nx), mem.B_sens(:,(i-1)*nu+1:i*nu), -eye(nx)];
    b=[b;mem.a(:,i)];
        
    Ci=[mem.Cx(:,(i-1)*nx+1:i*nx), mem.Cu(:,(i-1)*nu+1:i*nu);zeros(nbu,nx), Cu];
    C=blkdiag(C,[Ci;-Ci]);
    ub_du = mem.ub_du((i-1)*nu+1:i*nu);
    lb_du = mem.lb_du((i-1)*nu+1:i*nu);
    c=[c;-mem.uc((i-1)*nc+1:i*nc);-ub_du(nbu_idx);mem.lc((i-1)*nc+1:i*nc);lb_du(nbu_idx)];    
end
H=blkdiag(H,mem.Q_h(:,N*nx+1:(N+1)*nx));
g=[g;mem.gx(:,N+1)];
C=blkdiag(C,[mem.CxN;-mem.CxN]);
c=[c;-mem.uc(N*nc+1:N*nc+ncN);mem.lc(N*nc+1:N*nc+ncN)];

tic;
[w,lambda,mu,s,info] = pdip_multistage(H,g,B,b,C,c);
toc;

tau = 0.9;
tic;
[w_mex, lambda_mex, mu_mex, s_mex, OM, IT] = pdip(mem.Q_h,mem.S, mem.R, mem.A_sens, mem.B_sens, C, g,b,c,4,1,4,2,40);
toc;

% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter');
% [x,fval,exitflag,output,multipliers] = quadprog(H,g,C,-c,B,-b,[],[],[],options);