function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%Let first compute the h(x) matric which is:-
h = [];          % the empty hypothesis matrix
h = X*theta;     % the solved hypothesis matric with both theta0 and theta1
diff = h-y;      % the deviation from actual running sample
square = diff.^2;% doing the square 
result = sum(square(:,1));  %summing up the results
cost = (result)/(2*m);
cost_reg = (lambda/(2*m))*(sum(theta(:,1).^2)-theta(1,1).^2);
J = cost + cost_reg; 

% =========================================================================

grad = (X'*(h-y))/m;
temp = theta ;
temp(1) = 0;
grad = grad + (lambda/m).*temp;
grad = grad(:);

end
