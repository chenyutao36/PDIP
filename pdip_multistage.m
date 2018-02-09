function [w,lambda,mu,s,info] = pdip_multistage(H,g,B,b,C,c)

    load myData;

    nw = size(H,1);
    nE = size(B,1);
    nI = size(C,1);

    w = zeros(nw,1);
    lambda=zeros(nE,1);
    mu = 10*ones(nI,1);
    s = 10*ones(nI,1);
    e = ones(nI,1);
    
    opts.SYM = true;
    
    maxIT = 100;
    TOL = 1e-8;
    k=0;
    tau = 0.8;
    OM=1e8;
    while k<maxIT && OM>TOL

        L=diag(mu);
        S=diag(s);
        iS=diag(1./s);
        
        rI = c+ C*w + s;
        rS = S*L*e;       
        rC = H*w+g+B'*lambda+C'*mu;
        rE = b + B*w;
        rd = -C'*iS*(rS-L*rI)+rC;
                                      
        phi = H + C'*iS*L*C;
        beta = rE-B/phi*rd;           
        Y = B/phi*B';
                  
        dlambda_aff = linsolve(Y, beta, opts);
        dw_aff = linsolve(phi,-rd-B'*dlambda_aff);        
        dmu_aff = iS*L*(rI+C*dw_aff)-iS*rS;        
        ds_aff = -C*dw_aff-rI;
       
        t = s'*mu/nI;
        
        alpha_aff = 1;
        while (s+alpha_aff*ds_aff)'*(mu+alpha_aff*dmu_aff)<=0
            alpha_aff = alpha_aff*0.95;
        end
        
        t_aff = (s+alpha_aff*ds_aff)'*(mu+alpha_aff*dmu_aff)/nI;
        sigma = (t_aff/t)^3;
                
        dL_aff = diag(dmu_aff);
        dS_aff = diag(ds_aff);   
        rS = S*L*e - sigma*t*e + dL_aff*dS_aff*e;
        
        rd = -C'*iS*(rS-L*rI)+rC;
        beta = rE-B/phi*rd;
                              
        dlambda = linsolve(Y, beta, opts);               
        dw = linsolve(phi,-rd-B'*dlambda);       
        dmu = iS*L*(rI+C*dw)-iS*rS;
        ds = -C*dw-rI;
        

        alpha_pri_tau=1;
        while sum(s+alpha_pri_tau*ds>=(1-tau)*s)<nI
            alpha_pri_tau = alpha_pri_tau*0.95;
        end
        alpha_dual_tau=1;
        while sum(mu+alpha_dual_tau*dmu>=(1-tau)*mu)<nI
            alpha_dual_tau = alpha_dual_tau*0.95;
        end
                      
        alpha = min(alpha_pri_tau, alpha_dual_tau); 
        
        w = w+alpha*dw;
        lambda = lambda+alpha*dlambda;
        mu = mu+alpha*dmu;
        s = s+alpha*ds;
              
        k=k+1;
              
        OM = norm(H*w+g+B'*lambda+C'*mu)^2 + norm(c+C*w+s)^2 + s'*mu + norm(b+B*w)^2;
                
        tau = exp(-.2/k);
                    
    end
    
    info.optimality = OM;
    info.numIT = k;
    info.fval = 0.5*w'*H*w+g'*w;
    
end

