function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
SigmaList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
predictionError = zeros(size(Clist,2), size(SigmaList,2));
for i=1:size(Clist,2)
	for j=1:size(SigmaList,2)
		model = svmTrain(X, y, Clist(i), @(x1, x2) gaussianKernel(x1, x2, SigmaList(j))); 
		predictions = svmPredict(model, Xval);
		predictionError(i,j) = mean(double(predictions ~= yval));
	end
end
		
[minError, I] = min(predictionError(:));
q = floor(I/size(SigmaList,2)) + 1;
i = q; j = I - (q-1)*size(SigmaList, 2);
C = Clist(j); sigma = SigmaList(i);


% =========================================================================

end
