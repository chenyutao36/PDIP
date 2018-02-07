function [sol] = BackSolve(LY,beta,nx,N)

    sol=zeros((N+1)*nx,1);
    x = zeros((N+1)*nx,1);
    opts.LT=true;
    
    x(1:nx) = linsolve(LY(1:nx,1:nx), beta(1:nx), opts);
    for i=1:N
        x(i*nx+1:(i+1)*nx) = linsolve(LY(i*nx+1:(i+1)*nx,i*nx+1:(i+1)*nx),...
                                beta(i*nx+1:(i+1)*nx)-LY(i*nx+1:(i+1)*nx,(i-1)*nx+1:i*nx)*x((i-1)*nx+1:i*nx), opts);
    end

    opts.LT = false;
    opts.UT = true;
    sol(N*nx+1:end) = linsolve((LY(N*nx+1:end,N*nx+1:end))', x(N*nx+1:end), opts);
    for i=N-1:-1:0
        sol(i*nx+1:(i+1)*nx) = linsolve((LY(i*nx+1:(i+1)*nx,i*nx+1:(i+1)*nx))',...
                                x(i*nx+1:(i+1)*nx)-(LY((i+1)*nx+1:(i+2)*nx,i*nx+1:(i+1)*nx))'*sol((i+1)*nx+1:(i+2)*nx), opts);
    end
end

