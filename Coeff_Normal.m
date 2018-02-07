function [Y,LY] = Coeff_Normal(A,B,phi,nx,nu,N)

    Y = zeros((N+1)*nx, (N+1)*nx);
    LY = zeros((N+1)*nx, (N+1)*nx);
%     LY2 = zeros((N+1)*nx, (N+1)*nx);
    
    nz = nx+nu;
    
    L = struct;
    V = struct;
    W = struct;
    
    %% Compute V and W
    C_m1 = [eye(nx),zeros(nx,nu)];
    phi_0 = phi(1:nz,1:nz);
    L.(['l',num2str(0)]) = chol(phi_0,'lower');
    V.(['vm',num2str(1)]) = C_m1/(L.l0)';
    A_0 = A(:,1:nx);
    B_0 = B(:,1:nu);
    C_0 = [A_0,B_0];
    V.(['v',num2str(0)]) = C_0/(L.l0)';
    
    D = [-eye(nx),zeros(nx,nu)];    
    for i=1:N-1
        phi_k = phi(i*nz+1:(i+1)*nz,i*nz+1:(i+1)*nz);
        L.(['l',num2str(i)]) = chol(phi_k,'lower');
        A_k = A(:,i*nx+1:(i+1)*nx);
        B_k = B(:,i*nu+1:(i+1)*nu);
        C_k = [A_k,B_k];       
        V.(['v',num2str(i)]) = C_k/(L.(['l',num2str(i)]))';
        W.(['w',num2str(i)]) = D/(L.(['l',num2str(i)]))';      
    end
    phi_k = phi(N*nz+1:(N+1)*nz-nu,N*nz+1:(N+1)*nz-nu);
    L.(['l',num2str(N)]) = chol(phi_k,'lower');
    
    D = -eye(nx);
    W.(['w',num2str(N)]) = D/(L.(['l',num2str(N)]))'; 
    
    %% Compute Y
    Y(1:nx, 1:nx) = V.vm1*(V.vm1)';
    Y(1:nx, nx+1:2*nx) = V.vm1*(V.v0)';
%     Y(nx+1:2*nx, 1:nx) = V.v0*(V.vm1)';
    for i=0:N-2
        Y((i+1)*nx+1:(i+2)*nx, (i+1)*nx+1:(i+2)*nx) = V.(['v',num2str(i)])*(V.(['v',num2str(i)]))'...
                                                      + W.(['w',num2str(i+1)])*(W.(['w',num2str(i+1)]))';  
        
        Y((i+1)*nx+1:(i+2)*nx, (i+2)*nx+1:(i+3)*nx) = W.(['w',num2str(i+1)])*(V.(['v',num2str(i+1)]))';
        
%         Y((i+2)*nx+1:(i+3)*nx, (i+1)*nx+1:(i+2)*nx) = (Y((i+1)*nx+1:(i+2)*nx, (i+2)*nx+1:(i+3)*nx))';
    end
    Y(N*nx+1:(N+1)*nx, N*nx+1:(N+1)*nx) = V.(['v',num2str(N-1)])*(V.(['v',num2str(N-1)]))'...
                                          + W.(['w',num2str(N)])*(W.(['w',num2str(N)]))';

                                      
    %% Compute LY
    LY(1:nx,1:nx) = chol(Y(1:nx, 1:nx),'lower');
    LY(nx+1:2*nx,1:nx) = (Y(1:nx, nx+1:2*nx))'/ (LY(1:nx,1:nx))';
    for i=1:N-1
        LY(i*nx+1:(i+1)*nx,i*nx+1:(i+1)*nx) = chol( Y(i*nx+1:(i+1)*nx, i*nx+1:(i+1)*nx) - LY(i*nx+1:(i+1)*nx,(i-1)*nx+1:i*nx)*(LY(i*nx+1:(i+1)*nx,(i-1)*nx+1:i*nx))', 'lower' );
        LY((i+1)*nx+1:(i+2)*nx,i*nx+1:(i+1)*nx) = (Y(i*nx+1:(i+1)*nx, (i+1)*nx+1:(i+2)*nx))' / (LY(i*nx+1:(i+1)*nx,i*nx+1:(i+1)*nx))';
    end
    LY(N*nx+1:(N+1)*nx,N*nx+1:(N+1)*nx) = chol( Y(N*nx+1:(N+1)*nx, N*nx+1:(N+1)*nx) - LY(N*nx+1:(N+1)*nx,(N-1)*nx+1:N*nx)*(LY(N*nx+1:(N+1)*nx,(N-1)*nx+1:N*nx))', 'lower' );
    
    %%
%     LY2(1:nx,1:nx) = V.vm1(:,1:nx);
%     LY2(nx+1:2*nx,1:nx) = V.v0(:,1:nx);
%     for i=1:N-1
%         LY2(i*nx+1:(i+1)*nx,i*nx+1:(i+1)*nx) = W.(['w',num2str(i)])(:,1:nx);
%         LY2((i+1)*nx+1:(i+2)*nx,i*nx+1:(i+1)*nx) = V.(['v',num2str(i)])(:,1:nx);
%     end
%     LY2(N*nx+1:(N+1)*nx,N*nx+1:(N+1)*nx) = W.(['w',num2str(N)]);
end

