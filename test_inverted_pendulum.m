clear;close all;clc;

load myData;

N=settings.N;
nx=settings.nx;
nu=settings.nu;
nc=settings.nc;
ncN=settings.ncN;

H=[];
g=[];
B=zeros(settings.neq,settings.nw);   B(1:nx,1:nx)=eye(nx);
b=mem.ds0;
Cx=[];
Cu=zeros(N*nu,settings.nw);
cux=[];
clx=[];
cuu=[];
clu=[];
for i=1:N
    Hi=[mem.Q_h(:,(i-1)*nx+1:i*nx), mem.S(:,(i-1)*nu+1:i*nu);
        (mem.S(:,(i-1)*nu+1:i*nu))', mem.R(:,(i-1)*nu+1:i*nu)];
    H=blkdiag(H,Hi);
    
    g=[g;mem.gx(:,i);mem.gu(:,i)];
    
    B(i*nx+1:(i+1)*nx,(i-1)*(nx+nu)+1:i*(nx+nu)+nx)=[mem.A_sens(:,(i-1)*nx+1:i*nx), mem.B_sens(:,(i-1)*nu+1:i*nu), -eye(nx)];
    b=[b;mem.a(:,i)];
    
    Ci=[mem.Cx(:,(i-1)*nx+1:i*nx), mem.Cu(:,(i-1)*nu+1:i*nu)];
    Cx=blkdiag(Cx,Ci);
    cux=[cux;mem.uc((i-1)*nc+1:i*nc)];
    clx=[clx;mem.lc((i-1)*nc+1:i*nc)];
    
    Cu((i-1)*nu+1:i*nu,(i-1)*(nx+nu)+nx+1:i*(nx+nu))=eye(nu);
    cuu=[cuu;mem.ub_du((i-1)*nu+1:i*nu)];
    clu=[clu;mem.lb_du((i-1)*nu+1:i*nu)];
end
H=blkdiag(H,mem.Q_h(:,N*nx+1:(N+1)*nx));
g=[g;mem.gx(:,N+1)];
Cx=blkdiag(Cx,mem.CxN);
cux=[cux;mem.uc(N*nc+1:N*nc+ncN)];
clx=[clx;mem.lc(N*nc+1:N*nc+ncN)];

C=[Cx(nu+1:end,:);Cu;-Cx(nu+1:end,:);-Cu];
c=[-cux(nu+1:end);-cuu;clx(nu+1:end);clu];

tic;
[w,lambda,mu,s,info] = pdip(H,g,B,b,C,c);
toc;

% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter');
% [x,fval,exitflag,output,multipliers] = quadprog(H,g,C,-c,B,-b,[],[],[],options);