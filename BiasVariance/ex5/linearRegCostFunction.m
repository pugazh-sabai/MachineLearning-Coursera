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

A = zeros (m,1);            % Initialize to zero
A = (X*theta - y);          % Set the A vector to measure the error
J=(1/(2*m)*sum(A.*A)) + ...
    (lambda/(2*m))*sum(theta(2:end).^2);      % Compute the cost function

% For loop implementation
% for j=1:size(theta,1)
%     grad(:,j) = (1/m) * X'*(X*theta - y);
%     if j > 1
%         grad(:,j) = grad(:,j) + lambda/m * theta; 
%     end
% end
  

% Vectorized Implementation
grad = (1/m) * X'*(X*repmat(theta, 1, size(theta,2)) ...
    - repmat(y,1, size(theta,2)));
grad(2:end) = grad (2:end) + lambda/m * theta(2:end);











% =========================================================================

grad = grad(:);

end
