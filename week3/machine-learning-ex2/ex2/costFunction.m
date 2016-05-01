function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
%Let first compute the h(x) matric which is:-
h = [];          % the empty hypothesis matrix
h = sigmoid((X*theta));     % the solved hypothesis matric with values of theta for logical regression.
cost = y.*log(h)+(1-y).*log(1-h);      % the deviation from actual running sample
result = sum(cost(:,1));  %summing up the results
J = -(result)/(m);         %returning the present value of J for given value of theta


%GRADIENTDESCENT Performs gradient descent to learn theta

    for iter = 1:size(theta)
        summation = [];
        summation = (h-y) .*X(:,iter)
        grad(iter)= (sum(summation(:,1))/m);   %this is the gradient value not the descent value of theta.
    end
    
% =============================================================

end
