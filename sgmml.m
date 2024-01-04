
function model = sgmml( X, Y, optmParameter)
   %% optimization parameters
    lambda1          = optmParameter.lambda1; % missing labels
    lambda2          = optmParameter.lambda2; % regularization of W
    lambda3          = optmParameter.lambda3; % regularization of C
    lambda4          = optmParameter.lambda4; % regularization of graph laplacian
    lambda5          = optmParameter.lambda5; % 
    lambda6          = optmParameter.lambda6;     %lambda6 new parameter
    eta              = optmParameter.eta;
    maxIter          = optmParameter.maxIter;
    rho              = optmParameter.rho;

    num_dim   = size(X,2);
    num_class = size(Y,2);
    num_inst =  size(X,1);
    
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    YTX = Y'*X;

    W_k   = (X'*X + rho*eye(num_dim)) \ (X'*Y);%zeros(num_dim,num_label)
    %C_k = eye(num_class,num_class); %ones(num_class,num_class); %eye(num_class,num_class);
    %M_k = ones(num_inst,num_inst);
    C_k = zeros(num_class);
    M_k = rand(num_inst) .* (ones(num_inst) - eye(num_inst));
   
    iter = 1; oldloss = 0;
  
    tinyeps = 10^-4;
    while iter <= maxIter
       Dc = diag(1 ./(2 * vecnorm(C_k')+tinyeps));
       %https://in.mathworks.com/matlabcentral/answers/429543-warning-matrix-is-close-to-singular-or-badly-scaled-results-may-be-inaccurate-rcond-2-202823e
       %C_k = (YTY + lambda1* YTY + 2* lambda2 * Dc) \ (Y' * X * W_k + lambda1 * YTY); 
       C_k = pinv(YTY + lambda1* YTY + 2* lambda2 * Dc) * (Y' * X * W_k + lambda1 * YTY);       
       Dm = diag(1 ./(2 * vecnorm(M_k')+tinyeps));
       %M_k = lambda5* (2 * lambda6 * Dm + lambda5* X * W_k * W_k' * X') \ (X * W_k * W_k' * X');
       M_k = lambda5* pinv(2 * lambda6 * Dm + lambda5* X * W_k * W_k' * X') * (X * W_k * W_k' * X');
       Q= eye(num_inst) - M_k;
       Dw = diag(1 ./(2 * vecnorm(W_k')+tinyeps));
       L = diag(sum(C_k,2)) - C_k;
       
       delW = X' * (X*W_k - Y*C_k) + lambda3 * 2 * Dw * W_k + lambda4 * W_k * (L + L') + lambda5 * X' * Q * Q' * X * W_k;
       
       W_k = W_k - eta * delW; 
       
%        LS = X*W_k - Y*C_k;
%        DiscriminantLoss = trace(LS'* LS);
%        LS = Y*C_k - Y;
%        CorrelationLoss  = trace(LS'*LS);
%        CorrelationLoss2 = trace(W_k*L*W_k');
%        sparesW    = sum(sqrt(sum(abs(W_k).^2,2)));
%        sparesC    = sum(sqrt(sum(abs(C_k).^2,2)));
%        sparesM    = sum(sqrt(sum(abs(M_k).^2,2)));
%        last =  sum(sum((X*W_k - M_k*X*W_k).^2,1));
%        totalloss = 0.5 *DiscriminantLoss +0.5 * lambda1*CorrelationLoss + lambda3*sparesW + lambda2*sparesC+lambda6* sparesM+ lambda4*CorrelationLoss2+ lambda5*last;
%        loss(iter,1) = totalloss;
%        if abs((oldloss - totalloss)/oldloss) <= 0.00001
%            break;          
%        elseif totalloss <=0 
%            break;
%        else
%            oldloss = totalloss;
%        end
       iter=iter+1;
    end
    model.W = W_k;
    model.C = C_k;
    model.M = M_k;
    %model.loss = loss;
%    plot(loss)
    model.optmParameter = optmParameter;
end


