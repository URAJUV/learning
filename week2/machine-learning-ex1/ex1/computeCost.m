function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%Let first compute the h(x) matric which is:-
h = [];          % the empty hypothesis matrix
h = X*theta;     % the solved hypothesis matric with both theta0 and theta1
diff = h-y;      % the deviation from actual running sample
square = diff.^2;% doing the square 
result = sum(square(:,1));  %summing up the results
J = (result)/(2*m);         %returning the present value of J for given value of theta0 and theta1



% =========================================================================

end
