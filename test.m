
clear;close all;clc;

%%

H=[1 -1;-1 2]; g=[-2;-6];

C=[1 1;-1 2;2 1; -1 0;0 -1]; c=[-2;-2;-3; 0; 0];

tic;
[w,lambda,mu,s,info] = pdip(H,g,[],[],C,c);
toc;
% 
% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter');
% 
% tic;
% [x,fval,exitflag,output,multipliers] = quadprog(H,g,C,-c,[],[],[],[],[],options);
% toc;
%%

% v = sparse([1,-.25,0,0,0,0,0,-.25]);
% H = gallery('circul',v); H=full(H);
% g = (-4:3)';
% C = ones(1,8);c = 2;
% 
% [w,lambda,mu,s,info] = pdip(H,g,[],[],C,c);
% 
% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter');
% [x,fval,exitflag,output,multipliers] = quadprog(H,g,C,-c,[],[],[],[],[],options);

%%
