function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Regulalization of Cost function
%Note: grad should have the same dimensions as theta
%
%Let first compute the h(x) matric which is:-
h = [];          % the empty hypothesis matrix
h = sigmoid((X*theta));     % the solved hypothesis matric with values of theta for logical regression.
cost = -y.*log(h)-(1-y).*log(1-h);      % the deviation from actual running sample
cost_reg = (lambda/(2*m))*(sum(theta(:,1).^2)-theta(1,1).^2);
total_cost = cost +cost_reg;
result = sum(total_cost(:,1));  %summing up the results
J = (result)/(m);         %returning the present value of J for given value of theta


%GRADIENTDESCENT Performs gradient descent to learn theta

    for iter = 1:size(theta)
        summation = [];
        grad_reg = 0;
        summation = (h-y) .*X(:,iter);
        if iter == 1
            grad_reg = 0;                   %ignoring j=0 as j starts from 1 till n 
        else
            grad_reg = (lambda/m)*theta(iter);
        endif
        grad_val = (sum(summation(:,1))/m);
        grad(iter)= grad_val+grad_reg;   %this is the gradient value not the descent value of theta.
    end




% =============================================================

end
