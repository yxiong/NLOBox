%   Author: Ying Xiong.
%   Created: Apr 28, 2014.

rng(0);

x = randn(100, 1);
y = randn(100, 1);
costFcn = @(a)NLOCurveToCost(a, @NLOTest1DCurve, x, y);
CheckGradient(costFcn, 4);

fprintf('Passed.\n');
