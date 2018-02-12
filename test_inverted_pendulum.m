clear;close all;clc;

% load myData_TiltHex;
load myData_IP;

N=settings.N;
nx=settings.nx;
nu=settings.nu;
nbu=settings.nbu;
nc=settings.nc;
ncN=settings.ncN;

neq = (N+1)*nx;
nw = N*(nx+nu)+nx;

nbu_idx=settings.nbu_idx;
Cu = zeros(nbu,nu);
for j=1:nbu
    Cu(j,nbu_idx(j))=1;
end

H=[];
g=[];
B=zeros(neq,nw);   B(1:nx,1:nx)=eye(nx);
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
    
    if nbu>0
        ub_du = mem.ub_du((i-1)*nu+1:i*nu);
        lb_du = mem.lb_du((i-1)*nu+1:i*nu);
        c=[c;-mem.uc((i-1)*nc+1:i*nc);-ub_du(nbu_idx);mem.lc((i-1)*nc+1:i*nc);lb_du(nbu_idx)];  
    else
        c=[c;-mem.uc((i-1)*nc+1:i*nc);mem.lc((i-1)*nc+1:i*nc)]; 
    end
end
H=blkdiag(H,mem.Q_h(:,N*nx+1:(N+1)*nx));
g=[g;mem.gx(:,N+1)];
C=blkdiag(C,[mem.CxN;-mem.CxN]);
c=[c;-mem.uc(N*nc+1:N*nc+ncN);mem.lc(N*nc+1:N*nc+ncN)];

for i=1:nw
    if H(i,i)==0
        H(i,i) = 1e-12;
    end
end

%%
% tic;
% [w,lambda,mu,s,info] = pdip_multistage(H,g,B,b,C,c);
% toc;
%%
% tic;
% [w,lambda,mu,s,info] = pdip_general(H,g,B,b,C,c);
% toc;

%%
settings.nc=(settings.nc+nbu)*2;
settings.ncN=settings.ncN*2;

opts.it_max=50;
opts.tol = 1e-8;
opts.print_level = 1;
opts.reg = 0;
tic;
[w_mex, lambda_mex, mu_mex, s_mex, OM, IT, fval] = pdip(mem, C,g, b, c, settings, opts);
toc;

%%
% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter');
% tic;
% [x,fval,exitflag,output,multipliers] = quadprog(H,g,C,-c,B,-b,[],[],[],options);
% toc;