function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
        %calculate  Z^2 i.e Z superscript 2 i.e hidden layer values
        a1 = X ;
        Z_2 =X*Theta1';
        %use the sigmoid function to limit it between 0 and 1
        a_2 = 1 ./ (1 + e.^-Z_2);
        %Add the bais to hidden layer
        % Add ones to the X data matrix
        a_2 = [ones(m, 1) a_2];
        %calculating the values of z^3
        Z_3 = a_2*Theta2';
        %use the sigmoid function to limit it between 0 and 1
        a_3 = 1 ./ (1 + e.^-Z_3);
        
        %lets compute the inner function without summantion part it will be a m*k matrix
        %the below part is tricky this make y as y_matrix using identity matrix
        %based on the position given by y output .i.e make y_matix 1 for output... corresponding to particular y which is an identity matix.
        y_eye = eye(num_labels);
        y_mat = y_eye(y,:);
        resolved = -(y_mat.*log(a_3))-((1-y_mat).*log(1-a_3));
        row_sum = sum(resolved');
        col_sum = sum(row_sum);
        J_unreg = col_sum/m;
        %calculating the regularization parts
        % calculate the square of each element ,removing bias units and then double sum ..
        ...Theta1(:,2:end) gives us all the rows of Theta1, but leaves out the first column.
        ... (:) turns the resulting matrix into a vector containing all those elements.

        theta1_sum = sum (Theta1(:,2:end)(:).^2);
        theta2_sum = sum (Theta2(:,2:end)(:).^2);
        J = J_unreg + (lambda/(2*m))*(theta1_sum+theta2_sum);
        
        %start calculating the backpropogation algorithm.
        delta_3 = a_3  - y_mat;
        %below one is tricky indeed
        delta_2 = (delta_3 *Theta2(:,2:end)) .*sigmoidGradient(Z_2);
        Theta1(:,1) = 0;
        Theta2(:,1) = 0;
        delta1_grad = (delta_2' * X)/m +(lambda/m).*Theta1;
        size(delta1_grad);
        delta2_grad =  (delta_3' * a_2)/m + (lambda/m).*Theta2;
        size(delta2_grad);
        Theta1_grad = delta1_grad;
        Theta2_grad = delta2_grad;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
